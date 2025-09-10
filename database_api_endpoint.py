"""
Direct PostgreSQL Database API Endpoint
Fixes Railway API missing symbols issue (100/218 symbols missing)
"""

from fastapi import APIRouter, HTTPException
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get direct PostgreSQL connection"""
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="Database URL not configured")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

@router.get("/api/db/stock/{symbol}")
async def get_stock_price_from_db(symbol: str):
    """
    Get stock price directly from PostgreSQL database
    Solves Railway API missing symbols issue (100/218 coverage)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get latest price from enhanced_stock_data for the symbol
        query = """
        SELECT symbol, close, date, time, timeframe, volume
        FROM enhanced_stock_data 
        WHERE symbol = %s 
        ORDER BY date DESC, time DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (symbol.upper(),))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"Symbol {symbol} not found in database"
            )
        
        return {
            "success": True,
            "symbol": result['symbol'],
            "price": float(result['close']),
            "date": result['date'].isoformat() if result['date'] else None,
            "time": result['time'],
            "timeframe": result['timeframe'],
            "volume": result['volume'],
            "source": "PostgreSQL Database",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database query failed for {symbol}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Database query failed: {str(e)}"
        )

@router.get("/api/db/symbols")
async def get_all_symbols_from_db():
    """
    Get all available symbols from PostgreSQL database
    Shows complete symbol coverage (218 symbols vs Railway API's 100)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get all unique symbols with their latest prices
        query = """
        SELECT DISTINCT 
            symbol,
            (SELECT close FROM enhanced_stock_data esd2 
             WHERE esd2.symbol = esd.symbol 
             ORDER BY date DESC, time DESC LIMIT 1) as latest_price,
            (SELECT date FROM enhanced_stock_data esd3 
             WHERE esd3.symbol = esd.symbol 
             ORDER BY date DESC, time DESC LIMIT 1) as latest_date
        FROM enhanced_stock_data esd
        ORDER BY symbol
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        symbols = []
        for row in results:
            symbols.append({
                "symbol": row['symbol'],
                "latest_price": float(row['latest_price']) if row['latest_price'] else None,
                "latest_date": row['latest_date'].isoformat() if row['latest_date'] else None
            })
        
        return {
            "success": True,
            "total_symbols": len(symbols),
            "symbols": symbols,
            "source": "PostgreSQL Database",
            "note": f"Complete coverage: {len(symbols)} symbols vs Railway API's 100",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get all symbols: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get symbols: {str(e)}"
        )

@router.get("/api/db/compare-coverage")
async def compare_api_coverage():
    """
    Compare Railway API coverage vs PostgreSQL database coverage
    Identifies missing symbols
    """
    try:
        # Get symbols from database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT symbol FROM enhanced_stock_data ORDER BY symbol")
        db_symbols = set(row[0] for row in cursor.fetchall())
        
        cursor.close()
        conn.close()
        
        # Note: Railway API symbols would need to be fetched separately
        # This is a placeholder for the comparison
        
        return {
            "success": True,
            "database_symbols": len(db_symbols),
            "railway_api_symbols": 100,  # Known from investigation
            "missing_from_api": len(db_symbols) - 100,
            "coverage_percentage": (100 / len(db_symbols)) * 100,
            "database_symbols_list": sorted(list(db_symbols)),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Coverage comparison failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Coverage comparison failed: {str(e)}"
        )
