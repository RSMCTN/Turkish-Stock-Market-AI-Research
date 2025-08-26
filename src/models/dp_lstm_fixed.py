"""
Differentially Private LSTM for BIST Trading Signals (Fixed for Opacus)
Implements DP-LSTM with Opacus integration and adaptive privacy calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime

# Import our privacy mechanisms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privacy.dp_mechanisms import PrivacyBudget, NoiseCalibration, PrivacyAccountant


@dataclass
class DPLSTMConfig:
    """Configuration for DP-LSTM model"""
    # Model architecture
    input_size: int = 131  # Number of features (from feature engineering)
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    
    # Multi-task outputs
    output_tasks: Dict[str, int] = None  # Will be set in __post_init__
    
    # Differential Privacy parameters  
    noise_multiplier: float = 1.2  # Noise multiplier for DP-SGD
    max_grad_norm: float = 1.0     # Gradient clipping norm
    delta: float = 1e-5            # Privacy parameter delta
    target_epsilon: float = 1.0    # Target privacy parameter epsilon
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    max_physical_batch_size: int = 16  # For memory management
    epochs: int = 100
    
    # Adaptive privacy parameters
    enable_adaptive_privacy: bool = True
    privacy_calibration: NoiseCalibration = None
    
    def __post_init__(self):
        if self.output_tasks is None:
            self.output_tasks = {
                'direction': 3,  # [down, neutral, up]
                'magnitude': 1,  # Continuous value
                'volatility': 1, # Volatility prediction
                'volume': 1      # Volume prediction
            }
        
        if self.privacy_calibration is None:
            self.privacy_calibration = NoiseCalibration()


class DPLSTMModel(nn.Module):
    """
    Differentially Private LSTM for multi-task financial prediction
    
    Predicts:
    - Price direction (classification)
    - Price magnitude (regression) 
    - Volatility (regression)
    - Volume (regression)
    """
    
    def __init__(self, config: DPLSTMConfig):
        super(DPLSTMModel, self).__init__()
        self.config = config
        
        # Input normalization layer
        self.input_norm = nn.LayerNorm(config.input_size)
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        shared_features_size = lstm_output_size // 4
        
        # Multi-task heads
        self.task_heads = nn.ModuleDict()
        
        # Direction prediction head (classification)
        self.task_heads['direction'] = nn.Sequential(
            nn.Linear(shared_features_size, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.output_tasks['direction'])
        )
        
        # Magnitude prediction head (regression)  
        self.task_heads['magnitude'] = nn.Sequential(
            nn.Linear(shared_features_size, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, config.output_tasks['magnitude'])
        )
        
        # Volatility prediction head (regression)
        self.task_heads['volatility'] = nn.Sequential(
            nn.Linear(shared_features_size, 32),
            nn.ReLU(), 
            nn.Dropout(config.dropout),
            nn.Linear(32, config.output_tasks['volatility'])
        )
        
        # Volume prediction head (regression)
        self.task_heads['volume'] = nn.Sequential(
            nn.Linear(shared_features_size, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout), 
            nn.Linear(32, config.output_tasks['volume'])
        )
        
        # Task-specific loss weights (learnable)
        self.task_weights = nn.Parameter(torch.ones(len(config.output_tasks)))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # Linear/LSTM weights
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of DP-LSTM
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            hidden: Optional hidden state tuple
            
        Returns:
            Tuple of (task_outputs_dict, final_hidden_state)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, hidden_state = self.lstm(x, hidden)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Shared feature extraction
        shared_features = self.feature_extractor(last_output)
        
        # Multi-task predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)
        
        return outputs, hidden_state


def compute_multi_task_loss(predictions: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor],
                          task_weights: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute multi-task loss with learnable task weighting
    
    Args:
        predictions: Model predictions for each task
        targets: Ground truth targets for each task
        task_weights: Learnable task weights
        mask: Optional mask for sequence padding
        
    Returns:
        Tuple of (total_loss, individual_losses)
    """
    individual_losses = {}
    
    # Direction loss (cross-entropy)
    if 'direction' in predictions and 'direction' in targets:
        direction_loss = F.cross_entropy(
            predictions['direction'], 
            targets['direction'].long(),
            reduction='mean'
        )
        individual_losses['direction'] = direction_loss
    
    # Magnitude loss (MSE)
    if 'magnitude' in predictions and 'magnitude' in targets:
        magnitude_loss = F.mse_loss(
            predictions['magnitude'].squeeze(),
            targets['magnitude'],
            reduction='mean'
        )
        individual_losses['magnitude'] = magnitude_loss
    
    # Volatility loss (MSE)  
    if 'volatility' in predictions and 'volatility' in targets:
        volatility_loss = F.mse_loss(
            predictions['volatility'].squeeze(),
            targets['volatility'],
            reduction='mean'
        )
        individual_losses['volatility'] = volatility_loss
    
    # Volume loss (MSE)
    if 'volume' in predictions and 'volume' in targets:
        volume_loss = F.mse_loss(
            predictions['volume'].squeeze(),
            targets['volume'],
            reduction='mean'
        )
        individual_losses['volume'] = volume_loss
    
    # Weighted combination of losses
    total_loss = torch.tensor(0.0, device=task_weights.device, requires_grad=True)
    task_names = list(individual_losses.keys())
    
    # Create a new tensor for accumulation
    loss_sum = torch.zeros_like(total_loss)
    
    for i, (task_name, loss) in enumerate(individual_losses.items()):
        # Use softmax for positive weights that sum to 1
        weight = F.softmax(task_weights, dim=0)[i]
        loss_sum = loss_sum + weight * loss
    
    return loss_sum, individual_losses


class DPLSTMTrainer:
    """
    Trainer for DP-LSTM with privacy accounting and adaptive calibration
    """
    
    def __init__(self, 
                 model: DPLSTMModel,
                 config: DPLSTMConfig,
                 device: torch.device = None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Make model compatible with Opacus
        self.model = ModuleValidator.fix(self.model)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine()
        
        # Privacy accounting
        self.privacy_accountant = PrivacyAccountant(
            target_epsilon=config.target_epsilon,
            target_delta=config.delta
        )
        
        self.logger = logging.getLogger(__name__)
        
    def make_private(self, train_loader: DataLoader):
        """
        Make model, optimizer and data loader private using Opacus
        
        Args:
            train_loader: Training data loader
        """
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            epochs=self.config.epochs,
            target_epsilon=self.config.target_epsilon,
            target_delta=self.config.delta,
            max_grad_norm=self.config.max_grad_norm,
        )
        
        # Get the noise multiplier used by Opacus
        self.actual_noise_multiplier = self.optimizer.noise_multiplier
        
        self.logger.info(f"Model made private with:")
        self.logger.info(f"  Target ε: {self.config.target_epsilon}")
        self.logger.info(f"  Target δ: {self.config.delta}")
        self.logger.info(f"  Noise multiplier: {self.actual_noise_multiplier:.4f}")
        self.logger.info(f"  Max grad norm: {self.config.max_grad_norm}")
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   validation_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Train one epoch with differential privacy
        
        Args:
            train_loader: Training data loader
            validation_loader: Optional validation data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.config.output_tasks.keys()}
        num_batches = 0
        
        # Use BatchMemoryManager to handle varying batch sizes
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=self.config.max_physical_batch_size,
            optimizer=self.optimizer
        ) as memory_safe_data_loader:
            
            for batch_idx, batch in enumerate(memory_safe_data_loader):
                if len(batch) == 2:
                    inputs, targets = batch
                    # Handle case where targets might be a dict or tensor
                    if not isinstance(targets, dict):
                        # Convert to multi-task format if needed
                        targets = {'direction': targets}
                else:
                    inputs = batch[0]
                    targets = {task: batch[i+1] for i, task in enumerate(self.config.output_tasks.keys())}
                
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                predictions, _ = self.model(inputs)
                
                # Compute loss (external function, not model method)
                loss, individual_losses = compute_multi_task_loss(
                    predictions, 
                    targets, 
                    self.model.task_weights
                )
                
                # Backward pass (Opacus handles gradient clipping and noise)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Accumulate metrics
                total_loss += loss.item()
                for task, task_loss in individual_losses.items():
                    task_losses[task] += task_loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0 and batch_idx > 0:
                    self.logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Average metrics
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            **{f'{task}_loss': loss / num_batches for task, loss in task_losses.items()}
        }
        
        # Validation metrics
        if validation_loader is not None:
            val_metrics = self.validate(validation_loader)
            avg_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
        
        return avg_metrics
    
    def validate(self, validation_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model (no privacy constraints during validation)
        
        Args:
            validation_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.config.output_tasks.keys()}
        num_batches = 0
        
        with torch.no_grad():
            for batch in validation_loader:
                if len(batch) == 2:
                    inputs, targets = batch
                    if not isinstance(targets, dict):
                        targets = {'direction': targets}
                else:
                    inputs = batch[0]
                    targets = {task: batch[i+1] for i, task in enumerate(self.config.output_tasks.keys())}
                
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                predictions, _ = self.model(inputs)
                loss, individual_losses = compute_multi_task_loss(
                    predictions, 
                    targets, 
                    self.model.task_weights
                )
                
                total_loss += loss.item()
                for task, task_loss in individual_losses.items():
                    task_losses[task] += task_loss.item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            **{f'{task}_loss': loss / num_batches for task, loss in task_losses.items()}
        }
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get privacy budget spent during training"""
        if hasattr(self.privacy_engine, 'get_epsilon'):
            epsilon = self.privacy_engine.get_epsilon(self.config.delta)
            return epsilon, self.config.delta
        else:
            return 0.0, 0.0


def create_sample_data(batch_size: int = 64, 
                      sequence_length: int = 60,
                      input_size: int = 131) -> Tuple[DataLoader, DataLoader]:
    """Create sample data loaders for testing"""
    
    # Generate synthetic financial time series data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Training data
    train_x = torch.randn(batch_size * 5, sequence_length, input_size)  # Reduced for testing
    train_direction = torch.randint(0, 3, (batch_size * 5,))  # 0=down, 1=neutral, 2=up
    train_magnitude = torch.randn(batch_size * 5) * 0.02  # Price change magnitude
    train_volatility = torch.abs(torch.randn(batch_size * 5)) * 0.01  # Volatility
    train_volume = torch.abs(torch.randn(batch_size * 5)) * 1000  # Volume
    
    # Validation data
    val_x = torch.randn(batch_size * 1, sequence_length, input_size)  # Reduced for testing
    val_direction = torch.randint(0, 3, (batch_size * 1,))
    val_magnitude = torch.randn(batch_size * 1) * 0.02
    val_volatility = torch.abs(torch.randn(batch_size * 1)) * 0.01
    val_volume = torch.abs(torch.randn(batch_size * 1)) * 1000
    
    # Create datasets
    train_dataset = TensorDataset(train_x, train_direction, train_magnitude, train_volatility, train_volume)
    val_dataset = TensorDataset(val_x, val_direction, val_magnitude, val_volatility, val_volume)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def test_dp_lstm():
    """Test function for DP-LSTM"""
    
    print("🤖 Testing DP-LSTM Architecture (Fixed)...")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = DPLSTMConfig(
        input_size=131,
        hidden_size=64,  # Smaller for testing
        num_layers=1,
        target_epsilon=8.0,  # Higher epsilon for testing
        batch_size=32,
        max_physical_batch_size=16,
        epochs=2  # Quick test
    )
    
    print(f"Configuration:")
    print(f"   Input size: {config.input_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Target ε: {config.target_epsilon}")
    print(f"   Target δ: {config.delta}")
    
    # Create model
    model = DPLSTMModel(config)
    trainer = DPLSTMTrainer(model, config)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create sample data
    train_loader, val_loader = create_sample_data(
        batch_size=config.batch_size,
        input_size=config.input_size
    )
    
    print(f"Data created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Test forward pass before making private
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(4, 60, config.input_size)
        outputs, hidden = model(sample_input)
        
        print(f"\nForward pass test:")
        for task, output in outputs.items():
            print(f"   {task}: {output.shape}")
    
    # Make model private
    trainer.make_private(train_loader)
    print("✅ Model made private with Opacus")
    
    # Train for a few epochs
    print(f"\nTraining for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        metrics = trainer.train_epoch(train_loader, val_loader)
        epsilon, delta = trainer.get_privacy_spent()
        
        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"   Train loss: {metrics['total_loss']:.4f}")
        print(f"   Val loss: {metrics['val_total_loss']:.4f}")
        print(f"   Privacy spent: ε = {epsilon:.4f}, δ = {delta:.2e}")
    
    print(f"\n✅ DP-LSTM test completed!")


if __name__ == "__main__":
    test_dp_lstm()
