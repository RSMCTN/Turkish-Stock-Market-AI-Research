"""
SQLAlchemy Database Schemas for BIST Trading System
"""

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, Text, 
    Index, ForeignKey, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()


class MarketData(Base):
    """Market OHLCV data optimized for time-series queries"""
    __tablename__ = 'market_data'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core fields
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0)
    
    # Metadata
    timeframe = Column(String(5), nullable=False, default='1d')  # 1m, 5m, 15m, 1h, 1d
    source = Column(String(20), nullable=False, default='matriks')
    
    # Calculated fields (can be populated later)
    vwap = Column(Float, nullable=True)  # Volume Weighted Average Price
    trades_count = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Composite indexes for performance
    __table_args__ = (
        # Primary query patterns
        Index('ix_symbol_timestamp_timeframe', 'symbol', 'timestamp', 'timeframe'),
        Index('ix_timestamp_symbol', 'timestamp', 'symbol'),
        Index('ix_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        
        # Unique constraint to prevent duplicates
        UniqueConstraint('symbol', 'timestamp', 'timeframe', 'source', name='uq_market_data'),
        
        # Performance indexes
        Index('ix_created_at', 'created_at'),
        Index('ix_volume', 'volume'),  # For volume analysis
    )
    
    def __repr__(self):
        return (f"<MarketData({self.symbol}, {self.timestamp}, "
                f"C:{self.close}, V:{self.volume}, TF:{self.timeframe})>")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe,
            'source': self.source,
            'vwap': self.vwap,
            'trades_count': self.trades_count,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Symbol(Base):
    """BIST symbols metadata"""
    __tablename__ = 'symbols'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core fields
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    
    # Classification
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    
    # Financial data
    market_cap = Column(Float, nullable=True)
    currency = Column(String(3), nullable=False, default='TRY')
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    listing_date = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    source = Column(String(20), nullable=False, default='matriks')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    market_data = relationship("MarketData", backref="symbol_info", 
                              primaryjoin="Symbol.symbol == foreign(MarketData.symbol)")
    
    # Indexes
    __table_args__ = (
        Index('ix_symbol_active', 'symbol', 'is_active'),
        Index('ix_sector', 'sector'),
        Index('ix_market_cap', 'market_cap'),
    )
    
    def __repr__(self):
        return f"<Symbol({self.symbol}, {self.name}, Active:{self.is_active})>"


class NewsData(Base):
    """News articles with sentiment analysis for market impact"""
    __tablename__ = 'news_data'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Content
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)  # AI-generated summary
    
    # Source information
    source = Column(String(100), nullable=False, index=True)
    url = Column(String(1000), nullable=True, unique=True)
    author = Column(String(200), nullable=True)
    
    # Timing
    published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Related symbols (comma-separated or JSON)
    symbols = Column(String(500), nullable=True)  # e.g., "AKBNK,GARAN,ISCTR"
    
    # Sentiment scores (VADER Turkish)
    compound_score = Column(Float, nullable=True)  # Overall sentiment [-1, 1]
    positive_score = Column(Float, nullable=True)  # [0, 1]
    negative_score = Column(Float, nullable=True)  # [0, 1]
    neutral_score = Column(Float, nullable=True)   # [0, 1]
    
    # Advanced sentiment features
    confidence_score = Column(Float, nullable=True)  # Model confidence [0, 1]
    emotion_label = Column(String(50), nullable=True)  # fear, greed, neutral, etc.
    
    # Content classification
    category = Column(String(50), nullable=True)  # earnings, merger, policy, etc.
    importance_score = Column(Float, nullable=True)  # [0, 1] - market impact
    
    # Processing status
    is_processed = Column(Boolean, nullable=False, default=False)
    processing_version = Column(String(10), nullable=True)  # Track model versions
    
    # Language detection
    language = Column(String(5), nullable=False, default='tr')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_published_symbols', 'published_at', 'symbols'),
        Index('ix_source_published', 'source', 'published_at'),
        Index('ix_sentiment_compound', 'compound_score'),
        Index('ix_importance', 'importance_score'),
        Index('ix_processed', 'is_processed'),
        Index('ix_category_published', 'category', 'published_at'),
        
        # Text search indexes (if using PostgreSQL)
        # Index('ix_title_gin', 'title', postgresql_using='gin', postgresql_ops={'title': 'gin_trgm_ops'}),
    )
    
    def __repr__(self):
        return (f"<NewsData({self.id}, {self.source}, "
                f"Sentiment:{self.compound_score}, Symbols:{self.symbols})>")
    
    def get_symbols_list(self) -> list:
        """Get symbols as a list"""
        if not self.symbols:
            return []
        return [s.strip().upper() for s in self.symbols.split(',') if s.strip()]
    
    def set_symbols_list(self, symbols: list):
        """Set symbols from a list"""
        if symbols:
            self.symbols = ','.join([s.upper() for s in symbols if s])
        else:
            self.symbols = None


class TradingSignal(Base):
    """Generated trading signals from ML models"""
    __tablename__ = 'trading_signals'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Signal identification
    symbol = Column(String(10), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)  # dp_lstm, sentiment_arma, etc.
    model_version = Column(String(20), nullable=False)
    
    # Signal details
    signal_type = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)  # [0, 1]
    strength = Column(Float, nullable=False)   # Signal strength [0, 1]
    
    # Price predictions
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    # Market context
    current_price = Column(Float, nullable=False)
    volume_context = Column(String(20), nullable=True)  # high, normal, low
    
    # Timing
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    valid_until = Column(DateTime(timezone=True), nullable=True)
    
    # Signal metadata
    features_hash = Column(String(64), nullable=True)  # For deduplication
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Performance tracking (filled after execution)
    actual_return = Column(Float, nullable=True)
    hit_target = Column(Boolean, nullable=True)
    hit_stop_loss = Column(Boolean, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('ix_symbol_generated', 'symbol', 'generated_at'),
        Index('ix_model_generated', 'model_name', 'generated_at'),
        Index('ix_signal_active', 'signal_type', 'is_active'),
        Index('ix_confidence', 'confidence'),
        UniqueConstraint('symbol', 'model_name', 'features_hash', 'generated_at',
                        name='uq_trading_signal'),
    )
    
    def __repr__(self):
        return (f"<TradingSignal({self.symbol}, {self.signal_type}, "
                f"Conf:{self.confidence:.2f}, Model:{self.model_name})>")


class ModelPerformance(Base):
    """Track ML model performance metrics"""
    __tablename__ = 'model_performance'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Model identification
    model_name = Column(String(50), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    
    # Performance period
    evaluation_date = Column(DateTime(timezone=True), nullable=False, index=True)
    period_days = Column(Integer, nullable=False, default=30)
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)  # Direction accuracy
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Financial metrics
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    total_return = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    
    # Sample size
    total_signals = Column(Integer, nullable=False, default=0)
    profitable_signals = Column(Integer, nullable=False, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('ix_model_evaluation', 'model_name', 'evaluation_date'),
        UniqueConstraint('model_name', 'model_version', 'evaluation_date',
                        name='uq_model_performance'),
    )
    
    def __repr__(self):
        return (f"<ModelPerformance({self.model_name}, "
                f"Acc:{self.accuracy:.3f}, Sharpe:{self.sharpe_ratio:.2f})>")


# Database utility functions
def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """Drop all tables (use with caution)"""
    Base.metadata.drop_all(engine)


if __name__ == "__main__":
    # Test schema creation
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory SQLite for testing
    engine = create_engine('sqlite:///:memory:', echo=True)
    create_tables(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Test data
    symbol = Symbol(
        symbol='AKBNK',
        name='Akbank T.A.Ş.',
        sector='Banking',
        currency='TRY',
        is_active=True
    )
    
    market_data = MarketData(
        symbol='AKBNK',
        timestamp=datetime.utcnow(),
        open=10.5,
        high=10.8,
        low=10.2,
        close=10.6,
        volume=1000000,
        timeframe='1d'
    )
    
    news = NewsData(
        title='Akbank Q3 results exceed expectations',
        content='Akbank reported strong quarterly results...',
        source='bloomberg',
        published_at=datetime.utcnow(),
        symbols='AKBNK',
        compound_score=0.8
    )
    
    # Add test data
    session.add_all([symbol, market_data, news])
    session.commit()
    
    # Query test
    results = session.query(MarketData).filter_by(symbol='AKBNK').all()
    print(f"Found {len(results)} market data records")
    
    session.close()
    print("Schema test completed successfully!")
