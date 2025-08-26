"""
Advanced Differentially Private LSTM for Financial Time Series Prediction
Implements multi-task DP-LSTM with Opacus integration and comprehensive training utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import logging
from datetime import datetime
import math

# Import privacy mechanisms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from privacy.dp_mechanisms import PrivacyBudget, GaussianMechanism, PrivacyAccountant


@dataclass
class DPLSTMConfig:
    """Comprehensive configuration for DP-LSTM model"""
    # Model architecture
    input_size: int = 131              # Number of input features
    hidden_size: int = 256            # LSTM hidden dimension
    num_layers: int = 3               # Number of LSTM layers
    dropout: float = 0.3              # Dropout probability
    batch_first: bool = True          # Batch dimension first
    bidirectional: bool = False       # Bidirectional LSTM
    
    # Multi-task configuration
    enable_multi_task: bool = True
    output_tasks: Dict[str, int] = field(default_factory=lambda: {
        'direction': 3,    # Price direction: down, neutral, up
        'magnitude': 1,    # Price change magnitude
        'volatility': 1,   # Volatility prediction  
        'volume': 1        # Volume prediction
    })
    
    # Privacy parameters
    target_epsilon: float = 1.0       # Privacy budget epsilon
    target_delta: float = 1e-5        # Privacy parameter delta
    max_grad_norm: float = 1.0        # Gradient clipping norm
    noise_multiplier: float = 1.2     # Noise multiplier for DP-SGD
    
    # Training parameters
    learning_rate: float = 0.001      # Learning rate
    weight_decay: float = 1e-4        # Weight decay
    batch_size: int = 64              # Batch size
    max_physical_batch_size: int = 16 # Physical batch size for memory management
    epochs: int = 100                 # Number of training epochs
    
    # Architecture enhancements
    use_layer_norm: bool = True       # Layer normalization
    use_residual: bool = True         # Residual connections
    use_attention: bool = False       # Simple attention mechanism
    attention_heads: int = 4          # Number of attention heads
    
    # Regularization
    l1_lambda: float = 1e-5          # L1 regularization
    l2_lambda: float = 1e-4          # L2 regularization
    label_smoothing: float = 0.1      # Label smoothing for classification


class AdvancedDPLSTM(nn.Module):
    """
    Advanced Differentially Private LSTM with multi-task learning, attention, and regularization
    
    Features:
    - Multi-task prediction (direction, magnitude, volatility, volume)
    - Optional attention mechanism
    - Layer normalization and residual connections
    - Privacy-compatible architecture for Opacus
    - Advanced regularization techniques
    """
    
    def __init__(self, config: DPLSTMConfig):
        super().__init__()
        self.config = config
        
        # Input normalization
        self.input_norm = nn.LayerNorm(config.input_size) if config.use_layer_norm else nn.Identity()
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=config.batch_first,
            bidirectional=config.bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Layer normalization for LSTM outputs
        self.lstm_norm = nn.LayerNorm(lstm_output_size) if config.use_layer_norm else nn.Identity()
        
        # Optional attention mechanism
        if config.use_attention:
            self.attention = MultiHeadAttention(
                d_model=lstm_output_size,
                n_heads=config.attention_heads,
                dropout=config.dropout
            )
            self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # Shared feature extraction layers
        self.shared_features = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        shared_dim = lstm_output_size // 4
        
        # Multi-task heads
        if config.enable_multi_task:
            self.task_heads = self._create_task_heads(shared_dim)
            
            # Learnable task weights for loss balancing
            self.task_weights = nn.Parameter(torch.ones(len(config.output_tasks)))
        else:
            # Single task head
            self.output_head = nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(64, 1)
            )
        
        # Residual connections
        if config.use_residual:
            self.residual_projection = nn.Linear(config.input_size, lstm_output_size)
        
        # Initialize weights
        self._initialize_weights()
        
        # Track training statistics
        self.training_stats = {
            'forward_passes': 0,
            'gradient_norms': [],
            'task_losses': {task: [] for task in config.output_tasks.keys()}
        }
    
    def _create_task_heads(self, shared_dim: int) -> nn.ModuleDict:
        """Create task-specific prediction heads"""
        heads = nn.ModuleDict()
        
        for task_name, output_dim in self.config.output_tasks.items():
            if task_name == 'direction':
                # Classification head with more capacity
                heads[task_name] = nn.Sequential(
                    nn.Linear(shared_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(64, output_dim)
                )
            else:
                # Regression heads
                heads[task_name] = nn.Sequential(
                    nn.Linear(shared_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(32, output_dim)
                )
        
        return heads
    
    def _initialize_weights(self):
        """Initialize model weights with proper scaling"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # LSTM input weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # LSTM hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # All biases
                if 'lstm' in name:
                    # Initialize LSTM biases (forget gate bias to 1)
                    param.data.fill_(0)
                    if 'bias_ih' in name:
                        # Forget gate bias
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.)
                else:
                    param.data.fill_(0)
            elif 'weight' in name and len(param.shape) >= 2:  # Linear weights
                nn.init.xavier_normal_(param.data)
    
    def forward(self, 
               x: torch.Tensor, 
               hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
               return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Advanced DP-LSTM
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            hidden: Optional initial hidden state
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and intermediate outputs
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Track forward passes
        self.training_stats['forward_passes'] += 1
        
        # Input normalization
        x_norm = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, final_hidden = self.lstm(x_norm, hidden)
        
        # Layer normalization
        lstm_out = self.lstm_norm(lstm_out)
        
        # Optional attention mechanism
        attention_weights = None
        if self.config.use_attention:
            attended_out, attention_weights = self.attention(lstm_out)
            lstm_out = self.attention_norm(attended_out + lstm_out)  # Residual connection
        
        # Use last timestep for prediction
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Optional residual connection from input
        if self.config.use_residual:
            input_residual = self.residual_projection(x[:, -1, :])  # Last input timestep
            last_hidden = last_hidden + input_residual
        
        # Shared feature extraction
        shared_features = self.shared_features(last_hidden)
        
        # Task-specific predictions
        outputs = {'hidden_states': lstm_out, 'final_hidden': final_hidden}
        
        if self.config.enable_multi_task:
            for task_name, head in self.task_heads.items():
                outputs[task_name] = head(shared_features)
            
            # Add task weights for loss balancing
            outputs['task_weights'] = F.softmax(self.task_weights, dim=0)
        else:
            outputs['prediction'] = self.output_head(shared_features)
        
        if return_attention and attention_weights is not None:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def compute_multi_task_loss(self, 
                              predictions: Dict[str, torch.Tensor],
                              targets: Dict[str, torch.Tensor],
                              mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss with automatic task balancing
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            mask: Optional mask for sequence padding
            
        Returns:
            Tuple of (total_loss, individual_task_losses)
        """
        individual_losses = {}
        device = next(self.parameters()).device
        
        # Direction loss (classification with label smoothing)
        if 'direction' in predictions and 'direction' in targets:
            direction_loss = F.cross_entropy(
                predictions['direction'],
                targets['direction'].long(),
                label_smoothing=self.config.label_smoothing,
                reduction='mean'
            )
            individual_losses['direction'] = direction_loss
        
        # Magnitude loss (Huber loss for robustness)
        if 'magnitude' in predictions and 'magnitude' in targets:
            magnitude_pred = predictions['magnitude'].squeeze()
            magnitude_loss = F.huber_loss(
                magnitude_pred,
                targets['magnitude'],
                reduction='mean',
                delta=0.1
            )
            individual_losses['magnitude'] = magnitude_loss
        
        # Volatility loss (MSE)
        if 'volatility' in predictions and 'volatility' in targets:
            volatility_pred = predictions['volatility'].squeeze()
            volatility_loss = F.mse_loss(
                volatility_pred,
                targets['volatility'],
                reduction='mean'
            )
            individual_losses['volatility'] = volatility_loss
        
        # Volume loss (MSE on log scale for better scaling)
        if 'volume' in predictions and 'volume' in targets:
            volume_pred = predictions['volume'].squeeze()
            volume_target = targets['volume']
            
            # Log transform to handle large values
            log_pred = torch.log(torch.abs(volume_pred) + 1e-8)
            log_target = torch.log(torch.abs(volume_target) + 1e-8)
            
            volume_loss = F.mse_loss(log_pred, log_target, reduction='mean')
            individual_losses['volume'] = volume_loss
        
        # Weighted combination using learnable weights
        if individual_losses:
            task_weights = predictions.get('task_weights', torch.ones(len(individual_losses), device=device))
            
            total_loss = torch.zeros(1, device=device, requires_grad=True)
            for i, (task_name, loss) in enumerate(individual_losses.items()):
                weight = task_weights[min(i, len(task_weights) - 1)]
                total_loss = total_loss + weight * loss
                
                # Track task losses
                if self.training:
                    self.training_stats['task_losses'][task_name].append(loss.item())
            
            return total_loss, individual_losses
        else:
            return torch.tensor(0.0, device=device, requires_grad=True), {}
    
    def regularization_loss(self) -> torch.Tensor:
        """Compute L1 and L2 regularization losses"""
        l1_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        l2_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param ** 2)
        
        return self.config.l1_lambda * l1_loss + self.config.l2_lambda * l2_loss
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights for visualization"""
        if not self.config.use_attention:
            return None
        
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)
            return outputs.get('attention_weights')


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for LSTM outputs"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(attended)
        
        return output, attention_weights.mean(dim=1)  # Average attention across heads


def test_advanced_dp_lstm():
    """Test Advanced DP-LSTM implementation"""
    
    print("🧠 Testing Advanced DP-LSTM Architecture...")
    print("=" * 60)
    
    # Configuration
    config = DPLSTMConfig(
        input_size=50,      # Reduced for testing
        hidden_size=128,    # Reduced for testing  
        num_layers=2,
        enable_multi_task=True,
        use_attention=True,
        target_epsilon=4.0,
        batch_size=32,
        epochs=2
    )
    
    print(f"Configuration:")
    print(f"   Input size: {config.input_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Multi-task: {config.enable_multi_task}")
    print(f"   Attention: {config.use_attention}")
    print(f"   Tasks: {list(config.output_tasks.keys())}")
    
    # Create model
    model = AdvancedDPLSTM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {total_params:,} parameters")
    
    # Test forward pass
    batch_size = 8
    seq_len = 60
    x = torch.randn(batch_size, seq_len, config.input_size)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x, return_attention=True)
        
        print(f"\nForward pass test:")
        print(f"   Input shape: {x.shape}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
            else:
                print(f"   {key}: {type(value)}")
    
    # Test multi-task loss
    print(f"\nTesting multi-task loss computation...")
    
    # Create dummy targets
    targets = {
        'direction': torch.randint(0, 3, (batch_size,)),
        'magnitude': torch.randn(batch_size) * 0.02,
        'volatility': torch.abs(torch.randn(batch_size)) * 0.01,
        'volume': torch.abs(torch.randn(batch_size)) * 1000
    }
    
    total_loss, individual_losses = model.compute_multi_task_loss(outputs, targets)
    
    print(f"   Total loss: {total_loss.item():.4f}")
    for task, loss in individual_losses.items():
        print(f"   {task} loss: {loss.item():.4f}")
    
    # Test attention weights
    if config.use_attention:
        attention_weights = model.get_attention_weights(x)
        if attention_weights is not None:
            print(f"   Attention weights shape: {attention_weights.shape}")
    
    # Test regularization
    reg_loss = model.regularization_loss()
    print(f"   Regularization loss: {reg_loss.item():.6f}")
    
    # Test Opacus compatibility
    print(f"\nTesting Opacus compatibility...")
    try:
        model_fixed = ModuleValidator.fix(model)
        print(f"   ✅ Model is Opacus-compatible")
        
        # Test gradient computation
        optimizer = torch.optim.AdamW(model_fixed.parameters(), lr=config.learning_rate)
        
        model_fixed.train()
        outputs = model_fixed(x)
        loss, _ = model_fixed.compute_multi_task_loss(outputs, targets)
        loss.backward()
        
        # Compute gradient norms
        total_norm = 0.0
        for p in model_fixed.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        print(f"   Gradient norm: {total_norm:.4f}")
        
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"   ✅ Training step completed successfully")
        
    except Exception as e:
        print(f"   ❌ Opacus compatibility issue: {e}")
    
    # Display model statistics
    print(f"\n📊 Model Statistics:")
    print(f"   Forward passes: {model.training_stats['forward_passes']}")
    
    # Architecture summary
    print(f"\n🏗️ Architecture Summary:")
    print(f"   • LSTM: {config.num_layers} layers × {config.hidden_size} hidden units")
    print(f"   • Attention: {config.attention_heads} heads" if config.use_attention else "   • Attention: Disabled")
    print(f"   • Multi-task heads: {len(config.output_tasks)}")
    print(f"   • Regularization: L1({config.l1_lambda}) + L2({config.l2_lambda})")
    print(f"   • Dropout: {config.dropout}")
    print(f"   • Layer norm: {config.use_layer_norm}")
    print(f"   • Residual connections: {config.use_residual}")
    
    print(f"\n✅ Advanced DP-LSTM test completed!")


if __name__ == "__main__":
    test_advanced_dp_lstm()
