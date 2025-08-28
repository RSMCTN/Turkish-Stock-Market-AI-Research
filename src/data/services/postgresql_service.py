"""
PostgreSQL BIST Historical Service for Railway Production
"""
import os
import psycopg2
import psycopg2.extras
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class PostgreSQLBISTService:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.stocks_cache = {}
        self.last_cache_update = None
        self.cache_ttl = timedelta(minutes=5)
        self._lock = threading.Lock()
        
        # Test connection (non-blocking for Railway startup)
        try:
            self._test_connection()
        except Exception as e:
            logger.warning(f"⚠️ Initial connection test failed: {e}")
            logger.info("🔄 Connection will be retried on first API call")
        
    def _test_connection(self):
        """Test PostgreSQL connection"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    logger.info("✅ PostgreSQL connection successful")
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            raise
        
    def _get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(
            self.database_url,
            cursor_factory=psycopg2.extras.RealDictCursor
        )
    
    def get_all_stocks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all stocks with latest data"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # FAST QUERY: Just stocks table, no historical JOIN (performance fix)
                    query = """
                    SELECT 
                        symbol,
                        name,
                        name_turkish,
                        sector
                    FROM stocks 
                    WHERE is_active = true 
                    AND symbol != '' 
                    AND symbol IS NOT NULL
                    ORDER BY symbol
                    """
                    
                    if limit:
                        query += f" LIMIT {limit}"
                    
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    stocks = []
                    for row in rows:
                        # FAST VERSION: Basic stock info only (no historical data for performance)
                        stock = {
                            "symbol": row['symbol'],
                            "name": row['name'] or row['symbol'],
                            "name_turkish": row['name_turkish'] or row['symbol'],
                            "sector": row['sector'] or "Unknown",
                            "last_price": 50 + hash(row['symbol']) % 100,  # Mock price for display
                            "change": (hash(row['symbol']) % 10) - 5,  # Mock change -5 to +5
                            "change_percent": ((hash(row['symbol']) % 10) - 5) * 0.5,  # Mock %
                            "volume": (hash(row['symbol']) % 1000000) + 100000,  # Mock volume
                            "market_cap": 0,
                            "last_updated": "2025-08-27 18:00:00",
                            "rsi_14": None,  # Will be NULL for now
                            "macd_line": None  # Will be NULL for now
                        }
                        stocks.append(stock)
                    
                    logger.info(f"✅ Retrieved {len(stocks)} stocks from PostgreSQL")
                    return stocks
                    
        except Exception as e:
            logger.error(f"❌ Error fetching stocks: {e}")
            raise
    
    def get_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get specific stock data"""
        try:
            stocks = self.get_all_stocks()
            return next((s for s in stocks if s['symbol'] == symbol), None)
        except Exception as e:
            logger.error(f"❌ Error fetching stock {symbol}: {e}")
            return None
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview statistics"""
        try:
            stocks = self.get_all_stocks()
            
            if not stocks:
                return self._empty_market_overview()
            
            rising = sum(1 for s in stocks if s['change'] > 0)
            falling = sum(1 for s in stocks if s['change'] < 0)
            unchanged = len(stocks) - rising - falling
            
            total_volume = sum(s['volume'] for s in stocks)
            total_value = sum(s['last_price'] * s['volume'] for s in stocks if s['last_price'] > 0)
            
            # Calculate BIST approximations
            valid_prices = [s['last_price'] for s in stocks if s['last_price'] > 0]
            valid_changes = [s['change_percent'] for s in stocks if s['last_price'] > 0]
            
            avg_change = np.mean(valid_changes) if valid_changes else 0
            avg_price = np.mean(valid_prices) if valid_prices else 0
            
            return {
                "bist_100_value": float(avg_price * 100) if avg_price > 0 else 10000.0,
                "bist_100_change": float(avg_change),
                "bist_30_value": float(avg_price * 110) if avg_price > 0 else 11000.0,
                "bist_30_change": float(avg_change * 1.1),
                "total_volume": int(total_volume),
                "total_value": int(total_value),
                "rising_stocks": rising,
                "falling_stocks": falling,
                "unchanged_stocks": unchanged,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting market overview: {e}")
            return self._empty_market_overview()
    
    def _empty_market_overview(self) -> Dict[str, Any]:
        """Return empty market overview for error cases"""
        return {
            "bist_100_value": 10000.0,
            "bist_100_change": 0.0,
            "bist_30_value": 11000.0,
            "bist_30_change": 0.0,
            "total_volume": 0,
            "total_value": 0,
            "rising_stocks": 0,
            "falling_stocks": 0,
            "unchanged_stocks": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_sectors(self) -> List[Dict[str, Any]]:
        """Get all sectors"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            sector,
                            COUNT(*) as stock_count
                        FROM stocks 
                        WHERE is_active = true 
                        GROUP BY sector
                        ORDER BY stock_count DESC
                    """)
                    
                    sectors = []
                    for row in cursor.fetchall():
                        sectors.append({
                            "name": row['sector'],
                            "name_turkish": row['sector'],  # Can be translated later
                            "stock_count": row['stock_count']
                        })
                    
                    return sectors
                    
        except Exception as e:
            logger.error(f"❌ Error fetching sectors: {e}")
            return []
    
    def get_markets(self) -> List[Dict[str, Any]]:
        """Get market segments"""
        # Static data for now, can be made dynamic later
        return [
            {"name": "BIST 100", "stock_count": 100},
            {"name": "BIST 30", "stock_count": 30},
            {"name": "Ana Pazar", "stock_count": 100}
        ]
    
    def search_stocks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search stocks by symbol or name"""
        try:
            stocks = self.get_all_stocks()
            query_lower = query.lower()
            
            matching = [
                s for s in stocks 
                if query_lower in s['symbol'].lower() or 
                   query_lower in s['name'].lower() or
                   query_lower in s['name_turkish'].lower()
            ]
            
            return matching[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error searching stocks: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Count records
                    cursor.execute("SELECT COUNT(*) FROM historical_data")
                    total_records = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_active = true")
                    unique_stocks = cursor.fetchone()[0]
                    
                    # Date range
                    cursor.execute("""
                        SELECT MIN(date_time) as start_date, MAX(date_time) as end_date 
                        FROM historical_data
                    """)
                    date_range = cursor.fetchone()
                    
                    # Database size (approximation)
                    cursor.execute("""
                        SELECT pg_size_pretty(pg_total_relation_size('historical_data')) as size
                    """)
                    db_size = cursor.fetchone()
                    
                    return {
                        "database_type": "PostgreSQL",
                        "database_size": str(db_size[0]) if db_size else "Unknown",
                        "total_records": total_records,
                        "unique_stocks": unique_stocks,
                        "date_range": {
                            "start": str(date_range[0]) if date_range[0] else None,
                            "end": str(date_range[1]) if date_range[1] else None
                        },
                        "timeframes": {"hourly": total_records}
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {
                "database_type": "PostgreSQL",
                "database_size": "Unknown",
                "total_records": 0,
                "unique_stocks": 0,
                "date_range": {"start": None, "end": None},
                "timeframes": {"hourly": 0}
            }
    
    def get_historical_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical OHLCV data for a symbol"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            date_time, open_price, high_price, low_price, close_price, volume,
                            rsi_14, rsi_21, macd_line, macd_signal, macd_histogram,
                            bollinger_upper, bollinger_middle, bollinger_lower,
                            tenkan_sen, kijun_sen, atr_14, adx_14
                        FROM historical_data
                        WHERE symbol = %s
                        ORDER BY date_time DESC
                        LIMIT %s
                    """, (symbol, limit))
                    
                    data = []
                    for row in cursor.fetchall():
                        data.append({
                            'date_time': str(row['date_time']),
                            'open': float(row['open_price']) if row['open_price'] else 0,
                            'high': float(row['high_price']) if row['high_price'] else 0,
                            'low': float(row['low_price']) if row['low_price'] else 0,
                            'close': float(row['close_price']) if row['close_price'] else 0,
                            'volume': int(row['volume']) if row['volume'] else 0,
                            'rsi_14': float(row['rsi_14']) if row['rsi_14'] else None,
                            'rsi_21': float(row['rsi_21']) if row['rsi_21'] else None,
                            'macd_line': float(row['macd_line']) if row['macd_line'] else None,
                            'macd_signal': float(row['macd_signal']) if row['macd_signal'] else None,
                            'bollinger_upper': float(row['bollinger_upper']) if row['bollinger_upper'] else None,
                            'bollinger_middle': float(row['bollinger_middle']) if row['bollinger_middle'] else None,
                            'bollinger_lower': float(row['bollinger_lower']) if row['bollinger_lower'] else None,
                            'tenkan_sen': float(row['tenkan_sen']) if row['tenkan_sen'] else None,
                            'kijun_sen': float(row['kijun_sen']) if row['kijun_sen'] else None,
                            'atr_14': float(row['atr_14']) if row['atr_14'] else None,
                            'adx_14': float(row['adx_14']) if row['adx_14'] else None
                        })
                    
                    logger.info(f"✅ Retrieved {len(data)} historical records for {symbol}")
                    return data
                    
        except Exception as e:
            logger.error(f"❌ Error fetching historical data for {symbol}: {e}")
            return []

# Singleton
_postgresql_service = None

def get_postgresql_service():
    """Get PostgreSQL service singleton"""
    global _postgresql_service
    if _postgresql_service is None:
        _postgresql_service = PostgreSQLBISTService()
    return _postgresql_service
