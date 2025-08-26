"""
MAMUT_R600 Configuration Settings
"""
from typing import Optional
import os

class Settings:
    """Application settings"""
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/bist_trading")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # MatriksIQ API
    MATRIKS_API_KEY: str = os.getenv("MATRIKS_API_KEY", "")
    MATRIKS_BASE_URL: str = os.getenv("MATRIKS_BASE_URL", "https://api.matriks.com.tr/v1")
    
    # Model Settings
    DP_EPSILON: float = float(os.getenv("DP_EPSILON", "1.0"))
    DP_DELTA: float = float(os.getenv("DP_DELTA", "1e-5"))
    
    # Trading
    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
    COMMISSION_RATE: float = float(os.getenv("COMMISSION_RATE", "0.001"))
    
    def __repr__(self):
        return f"Settings(PAPER_TRADING={self.PAPER_TRADING})"

# Global settings instance
settings = Settings()

if __name__ == "__main__":
    print("✅ Settings loaded successfully!")
    print(f"   Paper Trading: {settings.PAPER_TRADING}")
    print(f"   Commission Rate: {settings.COMMISSION_RATE}")
    print(f"   DP Epsilon: {settings.DP_EPSILON}")
