"""
Simplified Differentially Private LSTM for BIST Trading Signals
Implements DP-LSTM with Opacus integration - Simplified for compatibility
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


@dataclass 
class DPLSTMConfig:
    """Configuration for DP-LSTM model"""
    # Model architecture
    input_size: int = 131
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    
    # Output configuration - simplified to single task
    num_classes: int = 3  # Price direction: down, neutral, up
    
    # Differential Privacy parameters
    target_epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    max_physical_batch_size: int = 16
    epochs: int = 10


class SimpleDPLSTM(nn.Module):
    """
    Simplified DP-LSTM for price direction prediction
    Single task to avoid Opacus compatibility issues
    """
    
    def __init__(self, config: DPLSTMConfig):
        super(SimpleDPLSTM, self).__init__()
        self.config = config
        
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(config.input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 4, config.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            
        Returns:
            Logits for price direction classification [batch_size, num_classes]
        """
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]
        
        # Use last timestep
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(last_hidden)  # [batch_size, num_classes]
        
        return logits


class DPLSTMTrainer:
    """Trainer for Simple DP-LSTM"""
    
    def __init__(self, model: SimpleDPLSTM, config: DPLSTMConfig, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Make compatible with Opacus
        self.model = ModuleValidator.fix(self.model)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Privacy engine
        self.privacy_engine = PrivacyEngine()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger = logging.getLogger(__name__)
        
        # Flags
        self.is_private = False
        
    def make_private(self, train_loader: DataLoader):
        """Make model private using Opacus"""
        
        self.model, self.optimizer, self.private_train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer, 
            data_loader=train_loader,
            epochs=self.config.epochs,
            target_epsilon=self.config.target_epsilon,
            target_delta=self.config.delta,
            max_grad_norm=self.config.max_grad_norm,
        )
        
        self.noise_multiplier = self.optimizer.noise_multiplier
        self.is_private = True
        
        self.logger.info(f"Model made private:")
        self.logger.info(f"  Target ε: {self.config.target_epsilon}")
        self.logger.info(f"  Target δ: {self.config.delta}")
        self.logger.info(f"  Noise multiplier: {self.noise_multiplier:.4f}")
        
        return self.private_train_loader
    
    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        if self.is_private:
            # Use BatchMemoryManager for private training
            with BatchMemoryManager(
                data_loader=data_loader,
                max_physical_batch_size=self.config.max_physical_batch_size,
                optimizer=self.optimizer
            ) as memory_safe_data_loader:
                
                for batch_idx, (inputs, targets) in enumerate(memory_safe_data_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device).long()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Metrics
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0 and batch_idx > 0:
                        self.logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        else:
            # Regular training
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).long()
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': 100. * correct / total,
            'num_samples': total
        }
    
    def validate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).long()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': 100. * correct / total,
            'num_samples': total
        }
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get privacy budget spent"""
        if self.is_private:
            epsilon = self.privacy_engine.get_epsilon(self.config.delta)
            return epsilon, self.config.delta
        else:
            return 0.0, 0.0


def create_sample_data(config: DPLSTMConfig) -> Tuple[DataLoader, DataLoader]:
    """Create sample financial time series data"""
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    sequence_length = 60
    num_train_samples = 320  # Small dataset for testing
    num_val_samples = 64
    
    # Generate synthetic features (like technical indicators)
    def generate_price_features(n_samples):
        # Simulate price movements and technical indicators
        prices = np.cumsum(np.random.normal(0, 0.01, n_samples)) + 100
        
        features = []
        for i in range(sequence_length, n_samples):
            # Price-based features
            price_window = prices[i-sequence_length:i]
            
            # Moving averages
            ma_5 = np.mean(price_window[-5:])
            ma_20 = np.mean(price_window[-20:]) if len(price_window) >= 20 else ma_5
            ma_50 = np.mean(price_window[-50:]) if len(price_window) >= 50 else ma_20
            
            # Price ratios
            price_to_ma5 = prices[i-1] / ma_5
            price_to_ma20 = prices[i-1] / ma_20
            
            # Volatility
            volatility = np.std(price_window[-10:]) if len(price_window) >= 10 else 0.01
            
            # Returns
            returns = np.diff(price_window[-20:]) if len(price_window) >= 20 else [0]
            avg_return = np.mean(returns)
            
            # Create feature vector (simplified to key indicators)
            feature_vector = [
                prices[i-1], ma_5, ma_20, ma_50, price_to_ma5, price_to_ma20,
                volatility, avg_return
            ]
            
            # Pad to config.input_size with noise
            while len(feature_vector) < config.input_size:
                feature_vector.append(np.random.normal(0, 0.1))
            
            features.append(feature_vector[:config.input_size])
        
        return np.array(features), prices[sequence_length:]
    
    # Generate training data
    train_features, train_prices = generate_price_features(num_train_samples + sequence_length)
    
    # Create sequences and labels
    train_X = []
    train_y = []
    
    for i in range(len(train_features) - sequence_length):
        train_X.append(train_features[i:i+sequence_length])
        
        # Price direction label (simplified)
        current_price = train_prices[i + sequence_length - 1]
        next_price = train_prices[i + sequence_length] if i + sequence_length < len(train_prices) else current_price
        
        price_change = (next_price - current_price) / current_price
        
        if price_change < -0.002:  # Down
            label = 0
        elif price_change > 0.002:   # Up
            label = 2
        else:                        # Neutral
            label = 1
            
        train_y.append(label)
    
    # Generate validation data
    val_features, val_prices = generate_price_features(num_val_samples + sequence_length)
    
    val_X = []
    val_y = []
    
    for i in range(len(val_features) - sequence_length):
        val_X.append(val_features[i:i+sequence_length])
        
        current_price = val_prices[i + sequence_length - 1]
        next_price = val_prices[i + sequence_length] if i + sequence_length < len(val_prices) else current_price
        
        price_change = (next_price - current_price) / current_price
        
        if price_change < -0.002:
            label = 0
        elif price_change > 0.002:
            label = 2
        else:
            label = 1
            
        val_y.append(label)
    
    # Convert to tensors
    train_X = torch.FloatTensor(train_X)
    train_y = torch.LongTensor(train_y)
    val_X = torch.FloatTensor(val_X)
    val_y = torch.LongTensor(val_y)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader


def test_dp_lstm():
    """Test Simplified DP-LSTM"""
    
    print("🤖 Testing Simplified DP-LSTM...")
    print("=" * 60)
    
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = DPLSTMConfig(
        input_size=131,
        hidden_size=32,  # Small for testing
        num_layers=1,
        target_epsilon=8.0,
        batch_size=16,
        epochs=3
    )
    
    print(f"Configuration:")
    print(f"   Input size: {config.input_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Target ε: {config.target_epsilon}")
    print(f"   Classes: {config.num_classes}")
    
    # Create model
    model = SimpleDPLSTM(config)
    trainer = DPLSTMTrainer(model, config)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data
    train_loader, val_loader = create_sample_data(config)
    print(f"Data created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(4, 60, config.input_size)
        output = model(sample_input)
        print(f"Forward pass test: {sample_input.shape} -> {output.shape}")
    
    # Make model private
    private_train_loader = trainer.make_private(train_loader)
    print("✅ Model made private with Opacus")
    
    # Training loop
    print(f"\nTraining for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Train
        train_metrics = trainer.train_epoch(private_train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Privacy spent
        epsilon, delta = trainer.get_privacy_spent()
        
        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"   Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
        print(f"   Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%") 
        print(f"   Privacy: ε={epsilon:.4f}, δ={delta:.2e}")
    
    print(f"\n✅ Simplified DP-LSTM test completed!")


if __name__ == "__main__":
    test_dp_lstm()
