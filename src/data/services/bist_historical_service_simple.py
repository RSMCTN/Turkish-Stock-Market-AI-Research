"""
Simplified BIST Historical Service - Database working version
"""
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict
import threading
import logging

logger = logging.getLogger(__name__)

class SimpleBISTService:
    def __init__(self, db_path: str = "data/bist_stocks.db"):
        import os
        # Railway environment check - multiple path options
        possible_paths = [
            db_path,  # Original path
            f"/app/{db_path}",  # Railway container path
            os.path.join(os.getcwd(), db_path),  # Current working directory
            "data/bist_historical.db",  # Alternative database
            f"/app/data/bist_historical.db"  # Railway alternative
        ]
        
        self.db_path = None
        for path in possible_paths:
            if Path(path).exists():
                self.db_path = Path(path)
                logger.info(f"✅ Database found at: {path}")
                break
        
        if self.db_path is None:
            logger.error(f"❌ Database not found in any of: {possible_paths}")
            self.db_path = Path(db_path)  # Use default for fallback
        
        self.stocks_cache = {}
        self.last_cache_update = None
        self.cache_ttl = timedelta(minutes=5)
        self._lock = threading.Lock()
        
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_all_stocks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            query = """
            WITH latest_data AS (
                SELECT 
                    h.symbol,
                    h.close_price,
                    h.volume,
                    h.date_time,
                    h.rsi_14,
                    h.macd_line,
                    ROW_NUMBER() OVER (PARTITION BY h.symbol ORDER BY h.date_time DESC) as rn
                FROM historical_data h
            ),
            prev_data AS (
                SELECT 
                    h.symbol,
                    h.close_price as prev_close,
                    ROW_NUMBER() OVER (PARTITION BY h.symbol ORDER BY h.date_time DESC) as rn
                FROM historical_data h
                WHERE h.date_time < (
                    SELECT MAX(date_time) FROM historical_data h2 WHERE h2.symbol = h.symbol
                )
            )
            SELECT 
                l.symbol,
                l.close_price,
                l.volume,
                l.date_time,
                l.rsi_14,
                l.macd_line,
                p.prev_close
            FROM latest_data l
            LEFT JOIN prev_data p ON l.symbol = p.symbol AND p.rn = 1
            WHERE l.rn = 1
            ORDER BY l.symbol
            """
            
            stocks = []
            for row in conn.execute(query):
                current_price = row['close_price'] or 0
                prev_price = row['prev_close'] or current_price
                change = current_price - prev_price
                change_percent = (change / prev_price * 100) if prev_price > 0 else 0
                
                stock = {
                    "symbol": row['symbol'],
                    "name": row['symbol'],  # Use symbol as name for now
                    "name_turkish": row['symbol'],
                    "sector": "Unknown",
                    "last_price": float(current_price),
                    "change": float(change),
                    "change_percent": float(change_percent),
                    "volume": int(row['volume'] or 0),
                    "market_cap": 0,
                    "last_updated": row['date_time'] or '',
                    "rsi_14": float(row['rsi_14']) if row['rsi_14'] else None,
                    "macd_line": float(row['macd_line']) if row['macd_line'] else None
                }
                stocks.append(stock)
                
                if limit and len(stocks) >= limit:
                    break
            
            return stocks
    
    def get_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        stocks = self.get_all_stocks()
        return next((s for s in stocks if s['symbol'] == symbol), None)
    
    def get_market_overview(self) -> Dict[str, Any]:
        stocks = self.get_all_stocks()
        
        if not stocks:
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
        
        rising = sum(1 for s in stocks if s['change'] > 0)
        falling = sum(1 for s in stocks if s['change'] < 0)
        unchanged = len(stocks) - rising - falling
        
        total_volume = sum(s['volume'] for s in stocks)
        total_value = sum(s['last_price'] * s['volume'] for s in stocks)
        
        # Calculate BIST approximations
        avg_change = np.mean([s['change_percent'] for s in stocks])
        avg_price = np.mean([s['last_price'] for s in stocks if s['last_price'] > 0])
        
        return {
            "bist_100_value": float(avg_price * 100) if not np.isnan(avg_price) else 10000.0,
            "bist_100_change": float(avg_change) if not np.isnan(avg_change) else 0.0,
            "bist_30_value": float(avg_price * 110) if not np.isnan(avg_price) else 11000.0,
            "bist_30_change": float(avg_change * 1.1) if not np.isnan(avg_change) else 0.0,
            "total_volume": int(total_volume),
            "total_value": int(total_value),
            "rising_stocks": rising,
            "falling_stocks": falling,
            "unchanged_stocks": unchanged,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_sectors(self) -> List[Dict[str, Any]]:
        return [
            {"name": "Technology", "name_turkish": "Teknoloji", "stock_count": 20},
            {"name": "Banking", "name_turkish": "Bankacılık", "stock_count": 15},
            {"name": "Unknown", "name_turkish": "Diğer", "stock_count": 65}
        ]
    
    def get_markets(self) -> List[Dict[str, Any]]:
        return [
            {"name": "BIST 100", "stock_count": 100},
            {"name": "BIST 30", "stock_count": 30},
            {"name": "Ana Pazar", "stock_count": 100}
        ]
    
    def search_stocks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        stocks = self.get_all_stocks()
        query_lower = query.lower()
        
        matching = [
            s for s in stocks 
            if query_lower in s['symbol'].lower()
        ]
        
        return matching[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM historical_data").fetchone()[0]
            unique = conn.execute("SELECT COUNT(DISTINCT symbol) FROM historical_data").fetchone()[0]
            
            date_range = conn.execute("""
                SELECT MIN(date_time) as start_date, MAX(date_time) as end_date 
                FROM historical_data
            """).fetchone()
            
            return {
                "database_path": str(self.db_path),
                "database_size_mb": self.db_path.stat().st_size / (1024*1024),
                "total_records": total,
                "unique_stocks": unique,
                "date_range": {
                    "start": date_range[0],
                    "end": date_range[1]
                },
                "timeframes": {"hourly": total}  # All data is hourly for now
            }

# Singleton
_service = None

def get_simple_service():
    global _service
    if _service is None:
        _service = SimpleBISTService()
    return _service
