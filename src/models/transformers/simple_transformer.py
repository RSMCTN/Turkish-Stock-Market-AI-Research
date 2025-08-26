"""
Simple Transformer Implementation for Financial Time Series Prediction
Clean, interpretable transformer architecture with multi-head attention and positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field


@dataclass
class SimpleTransformerConfig:
    """Configuration for Simple Transformer"""
    # Data parameters
    input_size: int = 131                    # Number of input features
    d_model: int = 256                       # Model dimension
    max_seq_len: int = 60                   # Maximum sequence length
    
    # Architecture parameters
    n_heads: int = 8                        # Number of attention heads
    n_layers: int = 6                       # Number of encoder layers
    d_ff: int = 1024                        # Feed-forward dimension
    dropout: float = 0.1                    # Dropout rate
    
    # Output configuration
    output_tasks: Dict[str, int] = field(default_factory=lambda: {
        'direction': 3,      # Price direction: down, neutral, up
        'magnitude': 1,      # Price change magnitude
        'volatility': 1,     # Volatility prediction
        'volume': 1          # Volume prediction
    })
    
    # Training parameters
    learning_rate: float = 0.001
    warmup_steps: int = 4000                # Learning rate warmup steps
    label_smoothing: float = 0.1            # Label smoothing for classification
    
    # Regularization
    weight_decay: float = 1e-4
    grad_clip: float = 1.0


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for Transformer
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        # Register as buffer (not trainable)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings [seq_len, batch_size, d_model]
            
        Returns:
            Positionally encoded embeddings
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism
    
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # For attention visualization
        self.attention_weights = None
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention
        
        Args:
            Q: Query matrix [batch_size, n_heads, seq_len, d_k]
            K: Key matrix [batch_size, n_heads, seq_len, d_k]
            V: Value matrix [batch_size, n_heads, seq_len, d_k]
            mask: Optional attention mask
            
        Returns:
            attention_output: Attended values
            attention_weights: Attention weights for interpretability
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, 
               query: torch.Tensor, 
               key: torch.Tensor, 
               value: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Multi-head attention output
            attention_weights: Average attention weights across heads
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for multi-head attention if provided
        if mask is not None:
            # Expand mask to [batch_size, n_heads, seq_len, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            mask = mask.repeat(1, self.n_heads, seq_len, 1)
        
        # Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        # Store attention weights for visualization (average across heads)
        self.attention_weights = attention_weights.mean(dim=1)
        
        return output, self.attention_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Feed-forward output [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Components:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    3. Residual connections and layer normalization
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise feed forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
               x: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Encoder layer output
            attention_weights: Self-attention weights
        """
        # Multi-head self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Position-wise feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class FinancialTransformer(nn.Module):
    """
    Simple Financial Transformer for Multi-task Time Series Prediction
    
    Architecture:
    1. Input embedding and positional encoding
    2. Stack of Transformer encoder layers
    3. Multi-task prediction heads
    4. Interpretability features
    """
    
    def __init__(self, config: SimpleTransformerConfig):
        super().__init__()
        self.config = config
        
        # Input projection to model dimension
        self.input_projection = nn.Linear(config.input_size, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model, 
            config.max_seq_len, 
            config.dropout
        )
        
        # Stack of transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            ) for _ in range(config.n_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Multi-task prediction heads
        self.task_heads = self._create_task_heads()
        
        # Global average pooling for sequence aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._initialize_weights()
        
        # Store attention weights for interpretability
        self.attention_weights_per_layer = []
    
    def _create_task_heads(self) -> nn.ModuleDict:
        """Create task-specific prediction heads"""
        heads = nn.ModuleDict()
        
        for task_name, output_dim in self.config.output_tasks.items():
            if task_name == 'direction':
                # Classification head
                heads[task_name] = nn.Sequential(
                    nn.Linear(self.config.d_model, self.config.d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.d_model // 2, output_dim)
                )
            else:
                # Regression heads
                heads[task_name] = nn.Sequential(
                    nn.Linear(self.config.d_model, self.config.d_model // 4),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.d_model // 4, output_dim)
                )
        
        return heads
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def create_padding_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Create padding mask for variable length sequences
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            lengths: Actual sequence lengths [batch_size]
            
        Returns:
            Padding mask [batch_size, seq_len] or None
        """
        if lengths is None:
            return None
        
        batch_size, seq_len, _ = x.shape
        mask = torch.arange(seq_len, device=x.device).expand(
            batch_size, seq_len
        ) < lengths.unsqueeze(1)
        
        return mask
    
    def forward(self, 
               x: torch.Tensor, 
               lengths: Optional[torch.Tensor] = None,
               return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Financial Transformer
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            lengths: Optional sequence lengths for masking
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing task predictions and optional attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Create padding mask
        mask = self.create_padding_mask(x, lengths)
        
        # Input projection and positional encoding
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)         # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)         # [batch_size, seq_len, d_model]
        
        # Pass through transformer encoder layers
        self.attention_weights_per_layer = []
        
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, mask)
            if return_attention:
                self.attention_weights_per_layer.append(attention_weights)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Sequence aggregation: use last timestep (or global average pooling)
        if mask is not None:
            # Use last valid timestep for each sequence
            last_indices = lengths - 1
            sequence_output = x[torch.arange(batch_size), last_indices]
        else:
            # Use last timestep
            sequence_output = x[:, -1, :]
        
        # Alternative: Global average pooling
        # pooled_output = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        # Multi-task predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(sequence_output)
        
        # Add interpretability information
        if return_attention:
            outputs['attention_weights'] = self.attention_weights_per_layer
            outputs['final_representation'] = sequence_output
        
        return outputs
    
    def get_attention_rollout(self) -> Optional[torch.Tensor]:
        """
        Compute attention rollout for interpretability
        Combines attention weights from all layers
        """
        if not self.attention_weights_per_layer:
            return None
        
        # Start with identity matrix
        rollout = torch.eye(self.attention_weights_per_layer[0].size(-1))
        rollout = rollout.unsqueeze(0).repeat(
            self.attention_weights_per_layer[0].size(0), 1, 1
        ).to(self.attention_weights_per_layer[0].device)
        
        # Multiply attention matrices from all layers
        for attention_weights in self.attention_weights_per_layer:
            rollout = torch.bmm(attention_weights, rollout)
        
        return rollout


def test_simple_transformer():
    """Test Simple Financial Transformer implementation"""
    
    print("🤖 Testing Simple Financial Transformer...")
    print("=" * 60)
    
    # Configuration
    config = SimpleTransformerConfig(
        input_size=25,      # Reduced for testing
        d_model=128,        # Reduced for testing
        n_heads=4,
        n_layers=3,         # Reduced for testing
        d_ff=256,           # Reduced for testing
        max_seq_len=60
    )
    
    print(f"Configuration:")
    print(f"   Input size: {config.input_size}")
    print(f"   Model dimension: {config.d_model}")
    print(f"   Attention heads: {config.n_heads}")
    print(f"   Encoder layers: {config.n_layers}")
    print(f"   Tasks: {list(config.output_tasks.keys())}")
    
    # Create model
    model = FinancialTransformer(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTransformer Model created with {total_params:,} parameters")
    
    # Test data
    batch_size = 4
    seq_len = 60
    x = torch.randn(batch_size, seq_len, config.input_size)
    lengths = torch.tensor([60, 45, 50, 55])  # Variable lengths
    
    print(f"\nTest data:")
    print(f"   Input shape: {x.shape}")
    print(f"   Sequence lengths: {lengths.tolist()}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x, lengths, return_attention=True)
    
    print(f"\nForward pass results:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"   {key}: {len(value)} layers of attention weights")
            if value:
                print(f"     - Each layer: {value[0].shape}")
        else:
            print(f"   {key}: {type(value)}")
    
    # Test attention rollout
    attention_rollout = model.get_attention_rollout()
    if attention_rollout is not None:
        print(f"   Attention rollout: {attention_rollout.shape}")
    
    # Test without attention return
    print(f"\nTesting without attention weights...")
    with torch.no_grad():
        simple_outputs = model(x, lengths, return_attention=False)
    
    print(f"   Simple outputs: {list(simple_outputs.keys())}")
    
    # Performance analysis
    print(f"\n📊 Model Analysis:")
    
    # Parameter breakdown
    embedding_params = sum(p.numel() for p in model.input_projection.parameters())
    encoder_params = sum(p.numel() for p in model.encoder_layers.parameters())
    head_params = sum(p.numel() for p in model.task_heads.parameters())
    
    print(f"   • Input projection: {embedding_params:,} parameters")
    print(f"   • Transformer encoder: {encoder_params:,} parameters")
    print(f"   • Task heads: {head_params:,} parameters")
    
    # Attention analysis
    if 'attention_weights' in outputs:
        avg_attention = torch.stack(outputs['attention_weights']).mean(dim=0)
        attention_entropy = -(avg_attention * torch.log(avg_attention + 1e-9)).sum(dim=-1)
        
        print(f"   • Average attention entropy: {attention_entropy.mean().item():.4f}")
        print(f"   • Attention diversity: {attention_entropy.std().item():.4f}")
    
    # Architecture summary
    print(f"\n🏗️ Simple Transformer Architecture:")
    print(f"   • Input Embedding: {config.input_size} → {config.d_model}")
    print(f"   • Positional Encoding: Sinusoidal")
    print(f"   • Encoder Layers: {config.n_layers} × Transformer Block")
    print(f"   • Multi-Head Attention: {config.n_heads} heads")
    print(f"   • Feed-Forward: {config.d_model} → {config.d_ff} → {config.d_model}")
    print(f"   • Multi-task Heads: {len(config.output_tasks)}")
    print(f"   • Interpretability: Attention weights + rollout")
    
    print(f"\n✅ Simple Financial Transformer test completed!")


if __name__ == "__main__":
    test_simple_transformer()
