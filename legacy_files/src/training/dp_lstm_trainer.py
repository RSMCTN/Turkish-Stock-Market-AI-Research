"""
DP-LSTM Training System

Differential Privacy inspired LSTM neural network training system for BIST stock prediction.

This module implements the core academic research methodology:
"Diferansiyel Gizlilikten Esinlenen LSTM ile Finansal Haberleri ve Değerleri 
Kullanarak İsabet Oranı Yüksek Hisse Senedi Tahmini"

Key Features:
- Privacy-preserving training with gradient clipping
- Adaptive noise injection during training
- BIST-specific feature engineering
- Academic validation metrics
- Model persistence and deployment
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DifferentialPrivacyMechanism:
    """
    Differential Privacy mechanism for LSTM training
    Implements gradient clipping and calibrated noise injection
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Differential privacy parameter
        self.sensitivity = sensitivity  # L2 sensitivity of gradients
        self.sigma = self._calculate_noise_scale()
        
    def _calculate_noise_scale(self) -> float:
        """Calculate noise scale for Gaussian mechanism"""
        # Gaussian mechanism: σ = sqrt(2 * log(1.25/δ)) * Δ / ε
        return np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
    
    def add_noise_to_gradients(self, model: nn.Module) -> None:
        """Add calibrated Gaussian noise to model gradients"""
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(0, self.sigma, size=param.grad.shape)
                param.grad += noise.to(param.device)
    
    def clip_gradients(self, model: nn.Module, max_norm: float = 1.0) -> float:
        """Clip gradients to bound sensitivity"""
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return total_norm.item()

class BISTDataPreprocessor:
    """
    BIST-specific data preprocessing for DP-LSTM training
    """
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.volume_scaler = StandardScaler()
        self.feature_scalers = {}
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators as features"""
        data = df.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Volatility features
        data['volatility'] = data['returns'].rolling(window=20).std()
        data['high_low_pct'] = (data['high'] - data['low']) / data['close']
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Technical indicators
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        bb_window = 20
        data['bb_middle'] = data['close'].rolling(window=bb_window).mean()
        bb_std = data['close'].rolling(window=bb_window).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        return data
    
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        # Select features for training
        feature_columns = [
            'close', 'volume', 'returns', 'volatility', 
            'sma_5', 'sma_20', 'rsi', 'macd', 'bb_position', 
            'volume_ratio', 'high_low_pct'
        ]
        
        # Remove NaN values
        data = data.dropna()
        
        # Scale features
        features = []
        for col in feature_columns:
            if col in ['close']:
                scaled = self.price_scaler.fit_transform(data[col].values.reshape(-1, 1))
            elif col in ['volume']:
                scaled = self.volume_scaler.fit_transform(data[col].values.reshape(-1, 1))
            else:
                scaler = StandardScaler()
                self.feature_scalers[col] = scaler
                scaled = scaler.fit_transform(data[col].values.reshape(-1, 1))
            features.append(scaled.flatten())
        
        features = np.column_stack(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features) - self.prediction_horizon + 1):
            X.append(features[i-self.sequence_length:i])
            y.append(features[i + self.prediction_horizon - 1, 0])  # Predict close price
        
        return np.array(X), np.array(y)

class DPLSTMModel(nn.Module):
    """
    Differential Privacy inspired LSTM model for stock prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, privacy_noise_scale: float = 0.1):
        super(DPLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.privacy_noise_scale = privacy_noise_scale
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Additional dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x, add_privacy_noise: bool = False):
        """Forward pass with optional privacy noise"""
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use output from last time step
        out = lstm_out[:, -1, :]
        
        # Add privacy noise during training if specified
        if add_privacy_noise and self.training:
            noise = torch.normal(0, self.privacy_noise_scale, size=out.shape).to(x.device)
            out = out + noise
        
        # Feed forward layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_privacy_loss(self) -> float:
        """Calculate privacy loss for current model state"""
        total_params = sum(p.numel() for p in self.parameters())
        privacy_loss = total_params * (self.privacy_noise_scale ** 2)
        return privacy_loss

class DPLSTMTrainer:
    """
    Main trainer class for DP-LSTM model
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.preprocessor = BISTDataPreprocessor(
            sequence_length=config.get('sequence_length', 60),
            prediction_horizon=config.get('prediction_horizon', 1)
        )
        
        self.dp_mechanism = DifferentialPrivacyMechanism(
            epsilon=config.get('epsilon', 1.0),
            delta=config.get('delta', 1e-5),
            sensitivity=config.get('sensitivity', 1.0)
        )
        
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.privacy_losses = []
        
    def prepare_data(self, data_path: str, train_split: float = 0.8, validation_split: float = 0.1):
        """Load and prepare BIST data for training"""
        logger.info("Preparing BIST data for DP-LSTM training...")
        
        # Load data (assuming CSV with OHLCV columns)
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.db'):
            # Handle SQLite database
            import sqlite3
            conn = sqlite3.connect(data_path)
            df = pd.read_sql_query("SELECT * FROM historical_data ORDER BY date", conn)
            conn.close()
        
        # Create technical features
        df = self.preprocessor.create_technical_features(df)
        
        # Prepare sequences
        X, y = self.preprocessor.prepare_sequences(df)
        
        # Split data
        n_total = len(X)
        n_train = int(n_total * train_split)
        n_val = int(n_total * validation_split)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 32), 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.get('batch_size', 32), 
            shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.get('batch_size', 32), 
            shuffle=False
        )
        
        return X.shape[2]  # Return number of features
    
    def initialize_model(self, input_size: int):
        """Initialize DP-LSTM model"""
        logger.info("Initializing DP-LSTM model...")
        
        self.model = DPLSTMModel(
            input_size=input_size,
            hidden_size=self.config.get('hidden_size', 128),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.2),
            privacy_noise_scale=self.config.get('privacy_noise_scale', 0.1)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5, verbose=True
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, epoch: int) -> float:
        """Train model for one epoch with differential privacy"""
        self.model.train()
        total_loss = 0.0
        total_privacy_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with privacy noise
            output = self.model(data, add_privacy_noise=True)
            output = output.squeeze()
            
            # Calculate loss
            loss = self.criterion(output, target)
            
            # Add privacy regularization
            privacy_loss = self.model.get_privacy_loss()
            total_loss_with_privacy = loss + self.config.get('privacy_lambda', 0.01) * privacy_loss
            
            # Backward pass
            total_loss_with_privacy.backward()
            
            # Apply differential privacy (gradient clipping + noise)
            grad_norm = self.dp_mechanism.clip_gradients(
                self.model, 
                max_norm=self.config.get('grad_clip_norm', 1.0)
            )
            
            self.dp_mechanism.add_noise_to_gradients(self.model)
            
            # Update parameters
            self.optimizer.step()
            
            total_loss += loss.item()
            total_privacy_loss += privacy_loss
            
            # Log batch progress
            if batch_idx % 50 == 0:
                logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}, '
                    f'Loss: {loss.item():.6f}, '
                    f'Privacy Loss: {privacy_loss:.6f}, '
                    f'Grad Norm: {grad_norm:.4f}'
                )
        
        avg_loss = total_loss / len(self.train_loader)
        avg_privacy_loss = total_privacy_loss / len(self.train_loader)
        
        return avg_loss, avg_privacy_loss
    
    def validate_epoch(self) -> float:
        """Validate model performance"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass without privacy noise
                output = self.model(data, add_privacy_noise=False)
                output = output.squeeze()
                
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs: int = 100, save_path: str = 'models/dp_lstm_best.pth'):
        """Main training loop"""
        logger.info(f"Starting DP-LSTM training for {num_epochs} epochs...")
        logger.info(f"Privacy parameters - ε: {self.dp_mechanism.epsilon}, δ: {self.dp_mechanism.delta}")
        
        best_val_loss = float('inf')
        patience = self.config.get('early_stopping_patience', 15)
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss, privacy_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.privacy_losses.append(privacy_loss)
            
            # Log epoch results
            logger.info(
                f'Epoch {epoch:3d} | '
                f'Train Loss: {train_loss:.6f} | '
                f'Val Loss: {val_loss:.6f} | '
                f'Privacy Loss: {privacy_loss:.6f} | '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.8f}'
            )
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'privacy_loss': privacy_loss,
                    'config': self.config,
                    'preprocessor_state': {
                        'price_scaler': self.preprocessor.price_scaler,
                        'volume_scaler': self.preprocessor.volume_scaler,
                        'feature_scalers': self.preprocessor.feature_scalers
                    }
                }, save_path)
                
                logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        return best_val_loss
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate model on test set with academic metrics"""
        logger.info("Evaluating DP-LSTM model...")
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data, add_privacy_noise=False)
                output = output.squeeze()
                
                predictions.extend(output.cpu().numpy())
                actuals.extend(target.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform predictions (from normalized to actual prices)
        predictions_prices = self.preprocessor.price_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        actuals_prices = self.preprocessor.price_scaler.inverse_transform(
            actuals.reshape(-1, 1)
        ).flatten()
        
        # Calculate academic metrics
        mae = mean_absolute_error(actuals_prices, predictions_prices)
        mse = mean_squared_error(actuals_prices, predictions_prices)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals_prices - predictions_prices) / actuals_prices)) * 100
        
        # Directional accuracy
        actual_directions = np.sign(np.diff(actuals_prices))
        pred_directions = np.sign(np.diff(predictions_prices))
        directional_accuracy = np.mean(actual_directions == pred_directions) * 100
        
        # Correlation
        correlation = np.corrcoef(actuals_prices, predictions_prices)[0, 1]
        
        # R-squared
        ss_res = np.sum((actuals_prices - predictions_prices) ** 2)
        ss_tot = np.sum((actuals_prices - np.mean(actuals_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Privacy metrics
        final_privacy_loss = self.model.get_privacy_loss()
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Correlation': correlation,
            'R_Squared': r_squared,
            'Privacy_Loss': final_privacy_loss,
            'Epsilon': self.dp_mechanism.epsilon,
            'Delta': self.dp_mechanism.delta
        }
        
        logger.info("Academic Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics, predictions_prices, actuals_prices
    
    def save_training_history(self, save_path: str = 'results/training_history.json'):
        """Save training history and metrics"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'privacy_losses': self.privacy_losses,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to {save_path}")
    
    def plot_training_curves(self, save_path: str = 'results/training_curves.png'):
        """Plot training and validation curves"""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Privacy loss curve
        plt.subplot(1, 3, 2)
        plt.plot(self.privacy_losses, label='Privacy Loss', color='green')
        plt.title('Privacy Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Privacy Loss')
        plt.legend()
        plt.grid(True)
        
        # Combined view
        plt.subplot(1, 3, 3)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Val Loss', alpha=0.7)
        plt.plot(np.array(self.privacy_losses) * 100, label='Privacy Loss (×100)', alpha=0.7)
        plt.title('Combined Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {save_path}")

# Training configuration template
DEFAULT_CONFIG = {
    # Data parameters
    'sequence_length': 60,
    'prediction_horizon': 1,
    
    # Model parameters
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'privacy_noise_scale': 0.1,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'grad_clip_norm': 1.0,
    'privacy_lambda': 0.01,
    'early_stopping_patience': 15,
    
    # Privacy parameters
    'epsilon': 1.0,
    'delta': 1e-5,
    'sensitivity': 1.0
}

if __name__ == "__main__":
    # Example training script
    logger.info("DP-LSTM Training System - Academic Research Implementation")
    logger.info("=" * 80)
    
    # Configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'batch_size': 64,
        'learning_rate': 0.0005,
        'hidden_size': 256,
        'num_layers': 3,
        'epsilon': 0.8  # Stronger privacy
    })
    
    # Initialize trainer
    trainer = DPLSTMTrainer(config)
    
    # Prepare data (example path - adjust as needed)
    data_path = "data/bist_historical_data.csv"
    if os.path.exists(data_path):
        input_size = trainer.prepare_data(data_path)
        
        # Initialize and train model
        trainer.initialize_model(input_size)
        best_val_loss = trainer.train(num_epochs=100)
        
        # Evaluate model
        metrics, predictions, actuals = trainer.evaluate_model()
        
        # Save results
        trainer.save_training_history()
        trainer.plot_training_curves()
        
        logger.info("DP-LSTM training completed successfully!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
    else:
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Please prepare BIST historical data for training")
