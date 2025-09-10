"""
BIST Historical Database Service
===============================
SQLite-based historical data service for BIST stocks
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class BISTHistoricalStock:
    """Historical stock data structure"""
    symbol: str
    name: str = ""
    name_turkish: str = ""
    sector: str = ""
    last_price: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    market_cap: float = 0.0
    last_updated: str = ""
    
    # Technical indicators (latest values)
    rsi_14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr_14: Optional[float] = None
    adx_14: Optional[float] = None

@dataclass
class HistoricalDataPoint:
    """Single historical data point"""
    date_time: str
    timeframe: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    
    # Technical indicators
    rsi_14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr_14: Optional[float] = None
    adx_14: Optional[float] = None

@dataclass
class MarketOverview:
    """Market overview with BIST indices"""
    bist_100_value: float
    bist_100_change: float
    bist_30_value: float
    bist_30_change: float
    total_volume: int
    total_value: int
    rising_stocks: int
    falling_stocks: int
    unchanged_stocks: int
    last_updated: datetime

class BISTHistoricalService:
    """Service for accessing historical BIST data from SQLite database"""
    
    def __init__(self, db_path: str = "enhanced_bist_data.db"):
        self.db_path = Path(db_path)
        self.stocks_cache = {}
        self.last_cache_update = None
        self.cache_ttl = timedelta(minutes=5)  # 5 dakika cache TTL
        self._lock = threading.Lock()
        
        # Database bağlantı kontrolü
        if not self.db_path.exists():
            logger.error(f"Database not found: {self.db_path}")
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        logger.info(f"✅ BISTHistoricalService initialized with database: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with optimizations"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Named access to columns
        # Performance optimizations
        conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
        conn.execute("PRAGMA temp_store = memory")
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
        return conn
    
    def _refresh_stocks_cache(self):
        """Refresh stocks cache with latest data"""
        with self._lock:
            if (self.last_cache_update and 
                datetime.now() - self.last_cache_update < self.cache_ttl):
                return
            
            logger.info("🔄 Refreshing stocks cache...")
            
            with self._get_connection() as conn:
                # Get latest data for each stock
                query = """
                WITH latest_data AS (
                    SELECT 
                        h.symbol,
                        h.close as close_price,
                        h.volume,
                        h.date || ' ' || COALESCE(h.time, '00:00') as date_time,
                        h.rsi_14,
                        h.macd_26_12 as macd_line,
                        h.macd_trigger_9 as macd_signal,
                        h.bol_upper_20_2 as bollinger_upper,
                        h.bol_middle_20_2 as bollinger_middle,
                        h.bol_lower_20_2 as bollinger_lower,
                        h.atr_14,
                        h.adx_14,
                        ROW_NUMBER() OVER (PARTITION BY h.symbol ORDER BY h.date DESC, h.time DESC) as rn
                    FROM enhanced_stock_data h
                ),
                previous_close AS (
                    SELECT 
                        h.symbol,
                        h.close as prev_close,
                        ROW_NUMBER() OVER (PARTITION BY h.symbol ORDER BY h.date DESC, h.time DESC) as rn
                    FROM enhanced_stock_data h
                    WHERE h.date < (
                        SELECT MAX(date) FROM enhanced_stock_data h2 WHERE h2.symbol = h.symbol
                    )
                )
                SELECT 
                    l.symbol,
                    l.symbol as name,
                    l.symbol as name_turkish,
                    'Technology' as sector,
                    l.close_price,
                    l.volume,
                    l.date_time,
                    l.rsi_14,
                    l.macd_line,
                    l.macd_signal,
                    l.bollinger_upper,
                    l.bollinger_middle,
                    l.bollinger_lower,
                    l.atr_14,
                    l.adx_14,
                    p.prev_close,
                    0 as market_cap
                FROM latest_data l
                LEFT JOIN previous_close p ON l.symbol = p.symbol AND p.rn = 1
                WHERE l.rn = 1
                ORDER BY l.symbol
                """
                
                stocks_data = {}
                for row in conn.execute(query):
                    # Calculate change
                    current_price = row['close_price'] or 0
                    prev_price = row['prev_close'] or current_price
                    change = current_price - prev_price
                    change_percent = (change / prev_price * 100) if prev_price > 0 else 0
                    
                    stock = BISTHistoricalStock(
                        symbol=row['symbol'],
                        name=row['name'] or '',
                        name_turkish=row['name_turkish'] or '',
                        sector=row['sector'] or '',
                        last_price=current_price,
                        change=change,
                        change_percent=change_percent,
                        volume=row['volume'] or 0,
                        market_cap=row['market_cap'] or 0,
                        last_updated=row['date_time'] or '',
                        rsi_14=row['rsi_14'],
                        macd_line=row['macd_line'],
                        macd_signal=row['macd_signal'],
                        bollinger_upper=row['bollinger_upper'],
                        bollinger_middle=row['bollinger_middle'],
                        bollinger_lower=row['bollinger_lower'],
                        atr_14=row['atr_14'],
                        adx_14=row['adx_14']
                    )
                    
                    stocks_data[row['symbol']] = stock
                
                self.stocks_cache = stocks_data
                self.last_cache_update = datetime.now()
                logger.info(f"✅ Stocks cache updated: {len(stocks_data)} stocks")
    
    def get_all_stocks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all stocks with latest prices and indicators"""
        self._refresh_stocks_cache()
        
        stocks = list(self.stocks_cache.values())
        if limit:
            stocks = stocks[:limit]
        
        # Convert to dict format for API compatibility
        return [asdict(stock) for stock in stocks]
    
    def get_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get single stock data"""
        self._refresh_stocks_cache()
        
        stock = self.stocks_cache.get(symbol.upper())
        return asdict(stock) if stock else None
    
    def get_historical_data(self, symbol: str, timeframe: str = "hourly", 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical OHLCV data for a stock"""
        # Map timeframe names 
        timeframe_map = {
            "60min": "60m",
            "hourly": "60m", 
            "daily": "günlük",
            "30min": "30m"
        }
        
        mapped_timeframe = timeframe_map.get(timeframe, timeframe)
        
        with self._get_connection() as conn:
            query = """
            SELECT 
                date || ' ' || COALESCE(time, '00:00:00') as date_time,
                timeframe, 
                open as open_price, 
                high as high_price, 
                low as low_price, 
                close as close_price, 
                volume, 
                rsi_14, 
                macd_26_12 as macd_line, 
                macd_trigger_9 as macd_signal,
                bol_upper_20_2 as bollinger_upper, 
                bol_middle_20_2 as bollinger_middle, 
                bol_lower_20_2 as bollinger_lower,
                atr_14, 
                adx_14
            FROM enhanced_stock_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY date DESC, time DESC
            LIMIT ?
            """
            
            data_points = []
            for row in conn.execute(query, (symbol.upper(), mapped_timeframe, limit)):
                point = HistoricalDataPoint(
                    date_time=row['date_time'],
                    timeframe=row['timeframe'],
                    open_price=row['open_price'] or 0,
                    high_price=row['high_price'] or 0,
                    low_price=row['low_price'] or 0,
                    close_price=row['close_price'] or 0,
                    volume=row['volume'] or 0,
                    rsi_14=row['rsi_14'],
                    macd_line=row['macd_line'],
                    macd_signal=row['macd_signal'],
                    bollinger_upper=row['bollinger_upper'],
                    bollinger_middle=row['bollinger_middle'],
                    bollinger_lower=row['bollinger_lower'],
                    atr_14=row['atr_14'],
                    adx_14=row['adx_14']
                )
                data_points.append(asdict(point))
        
        return data_points
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Calculate market overview from current data"""
        self._refresh_stocks_cache()
        
        if not self.stocks_cache:
            return {
                "bist_100_value": 0,
                "bist_100_change": 0,
                "bist_30_value": 0,
                "bist_30_change": 0,
                "total_volume": 0,
                "total_value": 0,
                "rising_stocks": 0,
                "falling_stocks": 0,
                "unchanged_stocks": 0,
                "last_updated": datetime.now().isoformat()
            }
        
        stocks = list(self.stocks_cache.values())
        
        # Market calculations
        rising = sum(1 for s in stocks if s.change > 0)
        falling = sum(1 for s in stocks if s.change < 0)
        unchanged = len(stocks) - rising - falling
        
        total_volume = sum(s.volume for s in stocks)
        total_value = sum(s.last_price * s.volume for s in stocks if s.last_price and s.volume)
        
        # BIST100 approximation (using average of top stocks)
        top_stocks = sorted(stocks, key=lambda x: x.market_cap or 0, reverse=True)[:30]
        bist_100_value = np.mean([s.last_price for s in top_stocks[:100] if s.last_price > 0]) * 100
        bist_100_change = np.mean([s.change_percent for s in top_stocks[:100]])
        
        bist_30_value = np.mean([s.last_price for s in top_stocks[:30] if s.last_price > 0]) * 100
        bist_30_change = np.mean([s.change_percent for s in top_stocks[:30]])
        
        overview = MarketOverview(
            bist_100_value=float(bist_100_value) if not np.isnan(bist_100_value) else 10000.0,
            bist_100_change=float(bist_100_change) if not np.isnan(bist_100_change) else 0.0,
            bist_30_value=float(bist_30_value) if not np.isnan(bist_30_value) else 11000.0,
            bist_30_change=float(bist_30_change) if not np.isnan(bist_30_change) else 0.0,
            total_volume=int(total_volume),
            total_value=int(total_value),
            rising_stocks=rising,
            falling_stocks=falling,
            unchanged_stocks=unchanged,
            last_updated=datetime.now()
        )
        
        return asdict(overview)
    
    def get_sectors(self) -> List[Dict[str, Any]]:
        """Get sector information with stock counts"""
        self._refresh_stocks_cache()
        
        sectors = {}
        for stock in self.stocks_cache.values():
            sector_name = stock.sector or "Diğer"
            if sector_name not in sectors:
                sectors[sector_name] = {
                    "name": sector_name,
                    "name_turkish": sector_name,
                    "stock_count": 0,
                    "total_market_cap": 0.0,
                    "avg_change": 0.0
                }
            
            sectors[sector_name]["stock_count"] += 1
            sectors[sector_name]["total_market_cap"] += stock.market_cap or 0
        
        # Calculate average changes
        for sector_name, sector_data in sectors.items():
            sector_stocks = [s for s in self.stocks_cache.values() if (s.sector or "Diğer") == sector_name]
            avg_change = np.mean([s.change_percent for s in sector_stocks if s.change_percent is not None])
            sectors[sector_name]["avg_change"] = float(avg_change) if not np.isnan(avg_change) else 0.0
        
        return list(sectors.values())
    
    def get_markets(self) -> List[Dict[str, Any]]:
        """Get market information (BIST30, BIST50, BIST100 etc.)"""
        self._refresh_stocks_cache()
        
        stock_count = len(self.stocks_cache)
        
        # Approximation based on market cap
        markets = [
            {"name": "BIST 100", "stock_count": min(100, stock_count)},
            {"name": "BIST 30", "stock_count": min(30, stock_count)},
            {"name": "BIST 50", "stock_count": min(50, stock_count)},
            {"name": "Yıldız Pazar", "stock_count": stock_count},
            {"name": "Ana Pazar", "stock_count": stock_count},
        ]
        
        return markets
    
    def search_stocks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search stocks by symbol or name"""
        self._refresh_stocks_cache()
        
        query_lower = query.lower()
        matching_stocks = []
        
        for stock in self.stocks_cache.values():
            if (query_lower in stock.symbol.lower() or 
                query_lower in stock.name.lower() or 
                query_lower in stock.name_turkish.lower()):
                matching_stocks.append(asdict(stock))
            
            if len(matching_stocks) >= limit:
                break
        
        return matching_stocks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self._get_connection() as conn:
            stats = {
                "database_path": str(self.db_path),
                "database_size_mb": self.db_path.stat().st_size / (1024*1024),
                "total_records": 0,
                "unique_stocks": 0,
                "date_range": {"start": None, "end": None},
                "timeframes": {}
            }
            
            # Total records
            total = conn.execute("SELECT COUNT(*) FROM enhanced_stock_data").fetchone()[0]
            stats["total_records"] = total
            
            # Unique stocks
            unique = conn.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data").fetchone()[0]
            stats["unique_stocks"] = unique
            
            # Date range
            date_range = conn.execute("""
                SELECT MIN(date) as start_date, MAX(date) as end_date 
                FROM enhanced_stock_data
            """).fetchone()
            stats["date_range"] = {
                "start": date_range[0],
                "end": date_range[1]
            }
            
            # Timeframe breakdown
            timeframes = conn.execute("""
                SELECT timeframe, COUNT(*) as count 
                FROM enhanced_stock_data 
                GROUP BY timeframe
            """).fetchall()
            
            for tf, count in timeframes:
                stats["timeframes"][tf] = count
        
        return stats

# Singleton instance
_historical_service_instance = None

def get_historical_service() -> BISTHistoricalService:
    """Get singleton historical service instance"""
    global _historical_service_instance
    if _historical_service_instance is None:
        _historical_service_instance = BISTHistoricalService()
    return _historical_service_instance
