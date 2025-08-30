"""
Test MatriksIQ Integration and Database Schemas
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, AsyncMock, patch

# Import our modules
from src.data.collectors.matriks_collector import MatriksCollector, test_connection
from src.data.storage.schemas import (
    Base, MarketData, Symbol, NewsData, TradingSignal, 
    create_tables, drop_tables
)
from src.config.settings import settings


class TestMatriksCollector:
    """Test MatriksCollector functionality"""
    
    @pytest.fixture
    def mock_api_key(self):
        return "test_api_key_12345"
    
    @pytest.fixture
    def collector(self, mock_api_key):
        return MatriksCollector(api_key=mock_api_key)
    
    @pytest.mark.asyncio
    async def test_collector_initialization(self, collector):
        """Test collector can be initialized"""
        assert collector.api_key == "test_api_key_12345"
        assert collector.base_url == "https://api.matriks.com.tr/v1"
        assert collector.session is None
        
    @pytest.mark.asyncio
    async def test_context_manager(self, collector):
        """Test async context manager functionality"""
        async with collector as c:
            assert c.session is not None
            assert c.session.closed is False
        
        # Session should be closed after context exit
        assert c.session.closed is True
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, collector):
        """Test rate limiting functionality"""
        with patch.object(collector, 'session') as mock_session:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"test": "data"})
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            # Make multiple requests quickly
            start_time = asyncio.get_event_loop().time()
            
            await collector._rate_limited_request("GET", "http://test.com")
            await collector._rate_limited_request("GET", "http://test.com")
            
            end_time = asyncio.get_event_loop().time()
            
            # Should take at least min_request_interval between requests
            assert end_time - start_time >= collector.min_request_interval
    
    @pytest.mark.asyncio
    async def test_get_symbols_success(self, collector):
        """Test successful symbol fetching"""
        mock_symbols_response = {
            "symbols": [
                {
                    "symbol": "AKBNK",
                    "name": "Akbank T.A.S.",
                    "sector": "Banking",
                    "market_cap": 50000000000,
                    "type": "equity",
                    "currency": "TRY",
                    "active": True
                },
                {
                    "symbol": "GARAN",
                    "name": "Turkiye Garanti Bankasi A.S.",
                    "sector": "Banking", 
                    "market_cap": 45000000000,
                    "type": "equity",
                    "currency": "TRY",
                    "active": True
                }
            ]
        }
        
        with patch.object(collector, '_rate_limited_request', 
                         return_value=mock_symbols_response):
            symbols = await collector.get_symbols()
            
            assert len(symbols) == 2
            assert symbols[0]['symbol'] == 'AKBNK'
            assert symbols[1]['symbol'] == 'GARAN'
            assert all(s['type'] == 'equity' for s in symbols)
    
    @pytest.mark.asyncio
    async def test_get_historical_data_success(self, collector):
        """Test successful historical data fetching"""
        mock_historical_response = {
            "data": [
                {
                    "timestamp": "2024-01-01T09:30:00Z",
                    "open": 10.5,
                    "high": 10.8,
                    "low": 10.2,
                    "close": 10.6,
                    "volume": 1000000
                },
                {
                    "timestamp": "2024-01-02T09:30:00Z",
                    "open": 10.6,
                    "high": 10.9,
                    "low": 10.4,
                    "close": 10.7,
                    "volume": 1200000
                }
            ]
        }
        
        with patch.object(collector, '_rate_limited_request',
                         return_value=mock_historical_response):
            df = await collector.get_historical_data("AKBNK", "1d")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
            assert df.index.name == 'timestamp'
            assert df.iloc[0]['close'] == 10.6
    
    @pytest.mark.asyncio
    async def test_get_real_time_quote_success(self, collector):
        """Test successful real-time quote fetching"""
        mock_quote_response = {
            "data": {
                "last": 10.75,
                "bid": 10.70,
                "ask": 10.80,
                "volume": 500000,
                "change": 0.15,
                "change_percent": 1.41,
                "high": 10.85,
                "low": 10.65,
                "open": 10.70
            }
        }
        
        with patch.object(collector, '_rate_limited_request',
                         return_value=mock_quote_response):
            quote = await collector.get_real_time_quote("AKBNK")
            
            assert quote['symbol'] == 'AKBNK'
            assert quote['last_price'] == 10.75
            assert quote['bid'] == 10.70
            assert quote['ask'] == 10.80
            assert quote['change_percent'] == 1.41
    
    @pytest.mark.asyncio
    async def test_health_check(self, collector):
        """Test health check functionality"""
        with patch.object(collector, '_rate_limited_request',
                         return_value={"status": "ok"}):
            health = await collector.health_check()
            assert health is True
        
        # Test fallback to symbols check
        with patch.object(collector, '_rate_limited_request', return_value=None):
            with patch.object(collector, 'get_symbols', return_value=[{"symbol": "TEST"}]):
                health = await collector.health_check()
                assert health is True


class TestDatabaseSchemas:
    """Test database schemas functionality"""
    
    @pytest.fixture
    def test_engine(self):
        """Create in-memory SQLite engine for testing"""
        engine = create_engine('sqlite:///:memory:', echo=False)
        create_tables(engine)
        return engine
    
    @pytest.fixture
    def session(self, test_engine):
        """Create database session"""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        yield session
        session.close()
    
    def test_symbol_creation(self, session):
        """Test Symbol model creation"""
        symbol = Symbol(
            symbol='AKBNK',
            name='Akbank T.A.Ş.',
            sector='Banking',
            market_cap=50000000000,
            currency='TRY',
            is_active=True
        )
        
        session.add(symbol)
        session.commit()
        
        # Query back
        result = session.query(Symbol).filter_by(symbol='AKBNK').first()
        assert result is not None
        assert result.name == 'Akbank T.A.Ş.'
        assert result.sector == 'Banking'
        assert result.is_active is True
    
    def test_market_data_creation(self, session):
        """Test MarketData model creation"""
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
        
        session.add(market_data)
        session.commit()
        
        # Query back
        result = session.query(MarketData).filter_by(symbol='AKBNK').first()
        assert result is not None
        assert result.close == 10.6
        assert result.volume == 1000000
        assert result.timeframe == '1d'
    
    def test_news_data_creation(self, session):
        """Test NewsData model creation"""
        news = NewsData(
            title='Akbank Q3 results exceed expectations',
            content='Akbank reported strong quarterly results...',
            source='bloomberg',
            published_at=datetime.utcnow(),
            symbols='AKBNK,GARAN',
            compound_score=0.8,
            positive_score=0.6,
            negative_score=0.1,
            neutral_score=0.3
        )
        
        session.add(news)
        session.commit()
        
        # Query back
        result = session.query(NewsData).first()
        assert result is not None
        assert result.compound_score == 0.8
        assert result.get_symbols_list() == ['AKBNK', 'GARAN']
    
    def test_trading_signal_creation(self, session):
        """Test TradingSignal model creation"""
        signal = TradingSignal(
            symbol='AKBNK',
            model_name='dp_lstm',
            model_version='1.0',
            signal_type='BUY',
            confidence=0.85,
            strength=0.7,
            target_price=11.50,
            stop_loss=10.20,
            current_price=10.75
        )
        
        session.add(signal)
        session.commit()
        
        # Query back
        result = session.query(TradingSignal).filter_by(symbol='AKBNK').first()
        assert result is not None
        assert result.signal_type == 'BUY'
        assert result.confidence == 0.85
        assert result.target_price == 11.50
    
    def test_market_data_indexes(self, session):
        """Test that indexes work properly"""
        # Add multiple data points
        for i in range(10):
            md = MarketData(
                symbol='AKBNK',
                timestamp=datetime.utcnow() + timedelta(days=i),
                open=10.0 + i * 0.1,
                high=10.2 + i * 0.1,
                low=9.8 + i * 0.1,
                close=10.1 + i * 0.1,
                volume=1000000 + i * 10000,
                timeframe='1d'
            )
            session.add(md)
        
        session.commit()
        
        # Query with index usage
        results = session.query(MarketData).filter_by(symbol='AKBNK').order_by(MarketData.timestamp).all()
        assert len(results) == 10
        
        # Test date range query
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=5)
        
        range_results = session.query(MarketData).filter(
            MarketData.symbol == 'AKBNK',
            MarketData.timestamp.between(start_date, end_date)
        ).all()
        
        assert len(range_results) <= 10  # Should work efficiently with indexes


# Integration tests
class TestIntegration:
    """Test integration between components"""
    
    @pytest.mark.asyncio
    async def test_full_data_pipeline_mock(self):
        """Test complete data pipeline with mocked API"""
        # Create in-memory database
        engine = create_engine('sqlite:///:memory:')
        create_tables(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Mock MatriksCollector
            collector = MatriksCollector("test_key")
            
            # Mock API responses
            mock_symbols = [{"symbol": "AKBNK", "name": "Akbank", "type": "equity", "active": True}]
            mock_historical = {
                "data": [{
                    "timestamp": "2024-01-01T10:00:00Z",
                    "open": 10.5, "high": 10.8, "low": 10.2, "close": 10.6, "volume": 1000000
                }]
            }
            
            with patch.object(collector, '_rate_limited_request') as mock_request:
                mock_request.side_effect = [mock_symbols, mock_historical]
                
                async with collector:
                    # Get symbols and store
                    symbols = await collector.get_symbols()
                    for sym_data in symbols:
                        symbol = Symbol(
                            symbol=sym_data['symbol'],
                            name=sym_data['name'],
                            is_active=sym_data['active']
                        )
                        session.add(symbol)
                    
                    # Get historical data and store
                    if symbols:
                        df = await collector.get_historical_data(symbols[0]['symbol'])
                        if not df.empty:
                            for timestamp, row in df.iterrows():
                                market_data = MarketData(
                                    symbol=symbols[0]['symbol'],
                                    timestamp=timestamp,
                                    open=row['open'],
                                    high=row['high'],
                                    low=row['low'],
                                    close=row['close'],
                                    volume=row['volume'],
                                    timeframe='1d'
                                )
                                session.add(market_data)
                    
                    session.commit()
                    
                    # Verify data was stored
                    symbol_count = session.query(Symbol).count()
                    data_count = session.query(MarketData).count()
                    
                    assert symbol_count == 1
                    assert data_count == 1
                    
        finally:
            session.close()


# Run tests if script is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
