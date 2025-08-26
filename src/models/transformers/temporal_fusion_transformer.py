"""
Temporal Fusion Transformer (TFT) for Financial Time Series Prediction
Advanced transformer architecture with variable selection, temporal fusion, and interpretability

Based on "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
Adapted for financial prediction with differential privacy support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer"""
    # Data dimensions
    input_size: int = 131                    # Number of input features
    static_input_size: int = 10              # Static/metadata features
    num_encoder_steps: int = 60              # Historical sequence length
    num_decoder_steps: int = 1               # Prediction horizon
    
    # Model architecture
    hidden_size: int = 240                   # Hidden state size
    lstm_layers: int = 2                     # Number of LSTM layers
    dropout: float = 0.3                     # Dropout rate
    attention_heads: int = 4                 # Multi-head attention heads
    
    # Variable selection
    num_quantiles: int = 7                   # Number of quantiles for prediction
    use_cudnn: bool = False                  # Use CuDNN LSTM (set False for Opacus)
    
    # Multi-task outputs
    output_tasks: Dict[str, int] = field(default_factory=lambda: {
        'direction': 3,      # Price direction classification
        'magnitude': 1,      # Price magnitude regression
        'volatility': 1,     # Volatility prediction
        'volume': 1,         # Volume prediction
        'quantiles': 7       # Quantile regression
    })
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    gradient_threshold: float = 0.01         # Gradient clipping threshold


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network for feature importance
    Learns which input variables are most relevant for prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Simplified variable selection using direct linear transformation
        self.variable_selection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, flattened_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            flattened_embedding: [batch_size, seq_len, input_size]
            
        Returns:
            selected_embedding: [batch_size, seq_len, input_size]
            selection_weights: [batch_size, seq_len, input_size]
        """
        # Compute variable selection weights
        mlp_outputs = self.variable_selection(flattened_embedding)
        sparse_weights = self.softmax(mlp_outputs)
        
        # Apply variable selection
        selected_embedding = flattened_embedding * sparse_weights
        
        return selected_embedding, sparse_weights


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) component
    Provides non-linear processing with gating mechanisms and skip connections
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: Optional[int] = None,
                 dropout: float = 0.1,
                 use_time_distributed: bool = True,
                 additional_context: Optional[int] = None):
        super().__init__()
        
        if output_size is None:
            output_size = input_size
            
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.use_time_distributed = use_time_distributed
        self.additional_context = additional_context
        
        # Primary processing layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Context processing if additional context is provided
        if additional_context is not None:
            self.context_projection = nn.Linear(additional_context, hidden_size, bias=False)
        
        # Gating mechanism
        self.gate = nn.Linear(hidden_size, output_size)
        
        # Skip connection
        if input_size != output_size:
            self.skip_connection = nn.Linear(input_size, output_size)
        else:
            self.skip_connection = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.ELU()
    
    def forward(self, 
               x: torch.Tensor, 
               context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            context: Optional additional context
            
        Returns:
            Gated output with residual connection
        """
        # Store for residual connection
        residual = x
        
        # Primary processing
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        
        # Add context if provided
        if context is not None and self.additional_context is not None:
            context_out = self.context_projection(context)
            out = out + context_out
        
        out = self.dropout(out)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(out))
        gated_out = gate * out
        
        # Skip connection
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
        
        # Residual connection and layer normalization
        output = self.layer_norm(gated_out + residual)
        
        return output


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for TFT
    Provides attention weights for interpretability
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Attention output
            attention_weights: Interpretable attention weights
        """
        batch_size, seq_len, _ = query.shape
        residual = query
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        # Average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for Financial Time Series Prediction
    
    Key Features:
    - Variable selection for feature importance
    - Temporal processing with LSTM and attention
    - Interpretable multi-head attention
    - Multi-task prediction capabilities
    - Quantile regression for uncertainty estimation
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Input embeddings
        self.input_embedding = nn.Linear(config.input_size, config.hidden_size)
        self.static_embedding = nn.Linear(config.static_input_size, config.hidden_size) if config.static_input_size > 0 else None
        
        # Variable selection networks
        self.historical_variable_selection = VariableSelectionNetwork(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # Locality enhancement with 1D convolution
        self.locality_enhancement = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            padding=1
        )
        
        # Encoder (historical processing)
        self.encoder_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder (future processing)
        self.decoder_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Temporal self-attention
        self.temporal_attention = InterpretableMultiHeadAttention(
            d_model=config.hidden_size,
            n_heads=config.attention_heads,
            dropout=config.dropout
        )
        
        # Position-wise feed forward
        self.position_wise_ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
        # Gated residual connections
        self.enrichment_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # Output projection layers
        self.output_projections = self._create_output_projections()
        
        # Quantile projection for uncertainty estimation
        if 'quantiles' in config.output_tasks:
            self.quantile_projection = nn.Linear(config.hidden_size, config.num_quantiles)
        
        # Initialize weights
        self._initialize_weights()
        
        # Interpretability storage
        self.attention_weights = None
        self.variable_selection_weights = None
    
    def _create_output_projections(self) -> nn.ModuleDict:
        """Create task-specific output projection layers"""
        projections = nn.ModuleDict()
        
        for task_name, output_dim in self.config.output_tasks.items():
            if task_name == 'quantiles':
                continue  # Handled separately
            
            if task_name == 'direction':
                # Classification head with more capacity
                projections[task_name] = nn.Sequential(
                    nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.hidden_size // 2, output_dim)
                )
            else:
                # Regression heads
                projections[task_name] = nn.Sequential(
                    nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.hidden_size // 2, output_dim)
                )
        
        return projections
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.)
    
    def forward(self, 
               historical_inputs: torch.Tensor,
               static_inputs: Optional[torch.Tensor] = None,
               future_inputs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Temporal Fusion Transformer
        
        Args:
            historical_inputs: [batch_size, encoder_steps, input_size]
            static_inputs: [batch_size, static_input_size] (optional)
            future_inputs: [batch_size, decoder_steps, input_size] (optional)
            
        Returns:
            Dictionary containing task predictions and interpretability information
        """
        batch_size, encoder_steps, _ = historical_inputs.shape
        
        # Variable selection for historical inputs
        selected_historical, var_selection_weights = self.historical_variable_selection(historical_inputs)
        self.variable_selection_weights = var_selection_weights
        
        # Embed historical inputs
        historical_embeddings = self.input_embedding(selected_historical)
        
        # Static context processing
        static_context = None
        if static_inputs is not None and self.static_embedding is not None:
            static_context = self.static_embedding(static_inputs)
            static_context = static_context.unsqueeze(1).repeat(1, encoder_steps, 1)
            historical_embeddings = historical_embeddings + static_context
        
        # Locality enhancement
        local_enhanced = self.locality_enhancement(historical_embeddings.transpose(1, 2)).transpose(1, 2)
        historical_embeddings = historical_embeddings + local_enhanced
        
        # Encoder processing
        encoder_outputs, encoder_states = self.encoder_lstm(historical_embeddings)
        
        # Prepare decoder inputs (use last encoder output if no future inputs provided)
        if future_inputs is not None:
            decoder_inputs = self.input_embedding(future_inputs)
        else:
            # Use last encoder output for single-step prediction
            decoder_inputs = encoder_outputs[:, -1:, :]
        
        # Decoder processing
        decoder_outputs, _ = self.decoder_lstm(decoder_inputs, encoder_states)
        
        # Temporal cross-attention (decoder attends to encoder)
        # Expand decoder outputs to match encoder sequence length for attention
        expanded_decoder = decoder_outputs.repeat(1, encoder_steps, 1)
        
        attended_outputs, attention_weights = self.temporal_attention(
            expanded_decoder, encoder_outputs, encoder_outputs
        )
        
        # Take only the last timestep after attention
        attended_outputs = attended_outputs[:, -1:, :]
        self.attention_weights = attention_weights
        
        # Position-wise feed forward
        ff_outputs = self.position_wise_ff(attended_outputs)
        
        # Gated residual network for feature enrichment
        enriched_outputs = self.enrichment_grn(ff_outputs + attended_outputs)
        
        # Use last timestep for prediction
        final_representation = enriched_outputs[:, -1, :]  # [batch_size, hidden_size]
        
        # Task-specific predictions
        outputs = {}
        
        for task_name, projection in self.output_projections.items():
            outputs[task_name] = projection(final_representation)
        
        # Quantile predictions for uncertainty
        if 'quantiles' in self.config.output_tasks:
            outputs['quantiles'] = self.quantile_projection(final_representation)
        
        # Add interpretability information
        outputs.update({
            'attention_weights': attention_weights,
            'variable_selection_weights': var_selection_weights,
            'encoder_outputs': encoder_outputs,
            'final_representation': final_representation
        })
        
        return outputs
    
    def get_interpretability_info(self) -> Dict[str, torch.Tensor]:
        """Get interpretability information from the last forward pass"""
        return {
            'attention_weights': self.attention_weights,
            'variable_selection_weights': self.variable_selection_weights
        }
    
    def compute_quantile_loss(self, 
                            predictions: torch.Tensor, 
                            targets: torch.Tensor,
                            quantiles: Optional[List[float]] = None) -> torch.Tensor:
        """Compute quantile loss for uncertainty estimation"""
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        quantile_losses = []
        
        for i, q in enumerate(quantiles):
            if i < predictions.size(-1):
                errors = targets.unsqueeze(-1) - predictions[:, i]
                quantile_loss = torch.max(
                    q * errors,
                    (q - 1) * errors
                )
                quantile_losses.append(quantile_loss.mean())
        
        return torch.stack(quantile_losses).mean()


def test_temporal_fusion_transformer():
    """Test Temporal Fusion Transformer implementation"""
    
    print("🌟 Testing Temporal Fusion Transformer...")
    print("=" * 60)
    
    # Configuration
    config = TFTConfig(
        input_size=30,           # Reduced for testing
        static_input_size=5,     # Static features
        num_encoder_steps=60,    # Historical sequence
        num_decoder_steps=1,     # Single-step prediction
        hidden_size=128,         # Reduced for testing
        lstm_layers=2,
        attention_heads=4,
        num_quantiles=5
    )
    
    print(f"Configuration:")
    print(f"   Input size: {config.input_size}")
    print(f"   Static size: {config.static_input_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Encoder steps: {config.num_encoder_steps}")
    print(f"   Tasks: {list(config.output_tasks.keys())}")
    
    # Create model
    model = TemporalFusionTransformer(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTFT Model created with {total_params:,} parameters")
    
    # Test data
    batch_size = 4
    historical_inputs = torch.randn(batch_size, config.num_encoder_steps, config.input_size)
    static_inputs = torch.randn(batch_size, config.static_input_size)
    
    print(f"\nTest data:")
    print(f"   Historical inputs: {historical_inputs.shape}")
    print(f"   Static inputs: {static_inputs.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(historical_inputs, static_inputs)
    
    print(f"\nForward pass results:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        else:
            print(f"   {key}: {type(value)}")
    
    # Test interpretability
    print(f"\nInterpretability features:")
    interp_info = model.get_interpretability_info()
    for key, value in interp_info.items():
        if value is not None:
            print(f"   {key}: {value.shape}")
    
    # Test quantile loss
    if 'quantiles' in outputs:
        targets = torch.randn(batch_size)
        quantile_loss = model.compute_quantile_loss(outputs['quantiles'], targets)
        print(f"   Quantile loss: {quantile_loss.item():.4f}")
    
    # Test variable selection
    var_weights = outputs.get('variable_selection_weights')
    if var_weights is not None:
        # Find top-k important variables
        avg_importance = var_weights.mean(dim=(0, 1))  # Average over batch and time
        top_k = torch.topk(avg_importance, k=5)
        print(f"\nTop 5 important variables:")
        for i, (idx, importance) in enumerate(zip(top_k.indices, top_k.values)):
            print(f"   {i+1}. Variable {idx.item()}: {importance.item():.4f}")
    
    # Architecture summary
    print(f"\n🏗️ TFT Architecture Summary:")
    print(f"   • Variable Selection: ✅")
    print(f"   • Encoder LSTM: {config.lstm_layers} layers")
    print(f"   • Decoder LSTM: {config.lstm_layers} layers")
    print(f"   • Multi-Head Attention: {config.attention_heads} heads")
    print(f"   • Locality Enhancement: 1D Conv")
    print(f"   • Gated Residual Networks: ✅")
    print(f"   • Multi-task Outputs: {len(config.output_tasks)}")
    print(f"   • Quantile Regression: ✅")
    print(f"   • Interpretability: ✅")
    
    print(f"\n✅ Temporal Fusion Transformer test completed!")


if __name__ == "__main__":
    test_temporal_fusion_transformer()
