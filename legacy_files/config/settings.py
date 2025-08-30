"""
📊 MAMUT_R600 Configuration Settings
=====================================
Centralized configuration for BIST DP-LSTM Trading System
"""

import os
from typing import List, Dict, Any
from pathlib import Path
from pydantic import BaseSettings, Field

class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="mamut_r600", env="DB_NAME")
    username: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class ModelSettings(BaseSettings):
    """DP-LSTM model configuration"""
    # Time series parameters
    sequence_length: int = 10  # 10-day rolling window
    prediction_horizon: int = 1  # Next day prediction
    
    # LSTM architecture
    lstm_units: List[int] = [64, 32]
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    
    # Training parameters
    train_split: float = 0.85
    validation_split: float = 0.15
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    
    # Differential Privacy
    dp_noise_multiplier: float = 0.1
    dp_l2_norm_clip: float = 1.0

class FeatureSettings(BaseSettings):
    """Technical indicators and feature engineering"""
    # Time frames for multi-timeframe analysis
    timeframes: List[str] = ["1m", "5m", "15m", "60m", "1d"]
    
    # Technical indicator families
    trend_periods: List[int] = [9, 21, 50, 200]  # EMA periods
    momentum_periods: List[int] = [7, 14, 21]    # RSI, Stoch periods
    volatility_periods: List[int] = [14, 20]     # ATR, BB periods
    
    # Feature selection thresholds
    min_information_coefficient: float = 0.05
    max_correlation_threshold: float = 0.9
    max_vif_threshold: float = 10.0
    max_features: int = 25

class SentimentSettings(BaseSettings):
    """VADER sentiment analysis configuration"""
    # VADER configuration
    vader_lexicon_path: str = "data/vader_lexicon_turkish.txt"
    
    # News processing
    max_news_per_day: int = 100
    news_sources_weights: Dict[str, float] = {
        "AA": 1.0,      # Anadolu Ajansı
        "KAP": 1.2,     # KAP (higher weight)
        "Reuters": 1.1,  # Reuters
        "Bloomberg": 1.1 # Bloomberg
    }
    
    # Sentiment features
    sentiment_window: int = 10  # days
    compound_weight: float = 0.6
    polarity_weights: Dict[str, float] = {
        "positive": 0.25,
        "neutral": 0.15,
        "negative": 0.25
    }

class TradingSettings(BaseSettings):
    """Trading and signal generation configuration"""
    # Signal thresholds
    buy_threshold: float = 0.62   # Probability threshold for BUY
    sell_threshold: float = 0.38  # Probability threshold for SELL
    
    # Risk management
    max_position_size: float = 0.05  # 5% of portfolio
    stop_loss_atr_multiple: float = 1.2
    take_profit_atr_multiple: float = 2.0
    max_concurrent_positions: int = 10
    
    # Commission and slippage
    commission_rate: float = 0.001  # 0.1% commission
    slippage_bps: float = 5.0       # 5 basis points slippage

class APISettings(BaseSettings):
    """FastAPI service configuration"""
    title: str = "MAMUT_R600 BIST DP-LSTM API"
    version: str = "0.1.0"
    description: str = "BIST Trading Signals API with Differential Privacy"
    
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # Rate limiting
    rate_limit: str = "100/minute"
    
    # Authentication (if needed)
    secret_key: str = Field(default="", env="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

class Settings(BaseSettings):
    """Main application settings"""
    # Project metadata
    project_name: str = "MAMUT_R600"
    version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = data_dir / "models" 
    logs_dir: Path = base_dir / "logs"
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    model: ModelSettings = ModelSettings()
    features: FeatureSettings = FeatureSettings()
    sentiment: SentimentSettings = SentimentSettings()
    trading: TradingSettings = TradingSettings()
    api: APISettings = APISettings()
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
