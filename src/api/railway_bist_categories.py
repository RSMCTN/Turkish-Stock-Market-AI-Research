#!/usr/bin/env python3
"""
Railway API - BIST Categories Endpoints
BIST_30, BIST_50, BIST_100 categorized stock data endpoints
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class RailwayBISTCategoriesAPI:
    def __init__(self):
        self.database_url = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"
        
    def get_connection(self):
        return psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
        
    async def get_bist_categories(self):
        """Get all BIST categories with stock counts"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    category,
                    COUNT(*) as stock_count,
                    AVG(priority) as avg_priority
                FROM stock_categories 
                GROUP BY category 
                ORDER BY AVG(priority)
            """)
            
            categories = [dict(row) for row in cursor.fetchall()]
            
            return {
                "success": True,
                "data": {
                    "categories": categories,
                    "total_categories": len(categories),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Categories fetch error: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            cursor.close()
            conn.close()
            
    async def get_stocks_by_category(self, category: str, limit: int = 100):
        """Get stocks by BIST category (BIST_30, BIST_50, BIST_100)"""
        if category not in ['BIST_30', 'BIST_50', 'BIST_100']:
            raise HTTPException(status_code=400, detail="Invalid category. Must be BIST_30, BIST_50, or BIST_100")
            
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get stocks with their latest data
            # Use simplified working query
            cursor.execute("""
                SELECT DISTINCT
                    sm.symbol,
                    sm.name,
                    sm.sector,
                    sm.is_active,
                    sm.api_available,
                    sc.category,
                    sc.priority,
                    
                    -- Get latest price safely with subquery
                    COALESCE(
                        (SELECT close FROM enhanced_stock_data esd 
                         WHERE esd.symbol = sm.symbol 
                         ORDER BY date DESC LIMIT 1), 
                        (SELECT close_price FROM historical_data hd 
                         WHERE hd.symbol = sm.symbol 
                         ORDER BY date_time DESC LIMIT 1),
                        0
                    ) as latest_price_enhanced,
                    
                    -- Get latest date
                    COALESCE(
                        (SELECT date FROM enhanced_stock_data esd 
                         WHERE esd.symbol = sm.symbol 
                         ORDER BY date DESC LIMIT 1), 
                        (SELECT date_time FROM historical_data hd 
                         WHERE hd.symbol = sm.symbol 
                         ORDER BY date_time DESC LIMIT 1)
                    ) as latest_date_enhanced,
                    
                    -- Record counts
                    (SELECT COUNT(*) FROM enhanced_stock_data esd 
                     WHERE esd.symbol = sm.symbol) as enhanced_records,
                    (SELECT COUNT(*) FROM historical_data hd 
                     WHERE hd.symbol = sm.symbol) as historical_records,
                    
                    -- Technical indicators
                    (SELECT rsi_14 FROM enhanced_stock_data esd 
                     WHERE esd.symbol = sm.symbol 
                     ORDER BY date DESC LIMIT 1) as rsi_14,
                    (SELECT macd_26_12 FROM enhanced_stock_data esd 
                     WHERE esd.symbol = sm.symbol 
                     ORDER BY date DESC LIMIT 1) as macd_26_12,
                    (SELECT bol_upper_20_2 FROM enhanced_stock_data esd 
                     WHERE esd.symbol = sm.symbol 
                     ORDER BY date DESC LIMIT 1) as bol_upper_20_2,
                    (SELECT bol_middle_20_2 FROM enhanced_stock_data esd 
                     WHERE esd.symbol = sm.symbol 
                     ORDER BY date DESC LIMIT 1) as bol_middle_20_2,
                    (SELECT bol_lower_20_2 FROM enhanced_stock_data esd 
                     WHERE esd.symbol = sm.symbol 
                     ORDER BY date DESC LIMIT 1) as bol_lower_20_2,
                    (SELECT atr_14 FROM enhanced_stock_data esd 
                     WHERE esd.symbol = sm.symbol 
                     ORDER BY date DESC LIMIT 1) as atr_14,
                    (SELECT adx_14 FROM enhanced_stock_data esd 
                     WHERE esd.symbol = sm.symbol 
                     ORDER BY date DESC LIMIT 1) as adx_14
                    
                FROM stocks_meta sm
                JOIN stock_categories sc ON sm.symbol = sc.symbol
                WHERE sc.category = %s
                AND sm.is_active = TRUE
                ORDER BY COALESCE(sc.priority, 999), sm.symbol
                LIMIT %s
            """, (category, limit))
            
            stocks = [dict(row) for row in cursor.fetchall()]
            
            # Debug logging
            logger.info(f"🔍 Raw query returned {len(stocks)} rows for {category}")
            if stocks:
                unique_symbols = set(stock['symbol'] for stock in stocks)
                logger.info(f"🔍 Unique symbols found: {len(unique_symbols)}")
                logger.info(f"🔍 First 5 symbols: {list(unique_symbols)[:5]}")
            
            # Process stock data
            processed_stocks = []
            for stock in stocks:
                # Get latest price and date
                latest_price = stock.get('latest_price_enhanced') or 0
                latest_date = stock.get('latest_date_enhanced')
                
                processed_stock = {
                    "symbol": stock['symbol'],
                    "name": stock['name'] or f"{stock['symbol']} Company",
                    "sector": stock['sector'] or "Unknown",
                    "category": stock['category'],
                    "priority": stock['priority'],
                    "latest_price": float(latest_price) if latest_price else 0,
                    "latest_date": latest_date.isoformat() if latest_date else None,
                    "data_sources": {
                        "enhanced_available": bool(stock.get('enhanced_records', 0) > 0),
                        "historical_available": bool(stock.get('historical_records', 0) > 0),
                        "enhanced_records": stock.get('enhanced_records', 0),
                        "historical_records": stock.get('historical_records', 0),
                        "api_available": stock.get('api_available', False)
                    },
                    "technical_indicators": {
                        "rsi_14": float(stock['rsi_14']) if stock.get('rsi_14') else None,
                        "macd": float(stock['macd_26_12']) if stock.get('macd_26_12') else None,
                        "bollinger_upper": float(stock['bol_upper_20_2']) if stock.get('bol_upper_20_2') else None,
                        "bollinger_middle": float(stock['bol_middle_20_2']) if stock.get('bol_middle_20_2') else None,
                        "bollinger_lower": float(stock['bol_lower_20_2']) if stock.get('bol_lower_20_2') else None,
                        "atr_14": float(stock['atr_14']) if stock.get('atr_14') else None,
                        "adx_14": float(stock['adx_14']) if stock.get('adx_14') else None
                    }
                }
                processed_stocks.append(processed_stock)
                
            return {
                "success": True,
                "data": {
                    "category": category,
                    "stocks": processed_stocks,
                    "total_stocks": len(processed_stocks),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Stocks by category error: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            cursor.close()
            conn.close()
            
    async def get_stock_historical_data(self, symbol: str, timeframe: str = "60min", limit: int = 500):
        """Get historical data for specific stock with Railway database integration"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if symbol exists in our categories
            cursor.execute("""
                SELECT sc.category, sc.priority, sm.name
                FROM stocks_meta sm
                JOIN stock_categories sc ON sm.symbol = sc.symbol
                WHERE sm.symbol = %s
                LIMIT 1
            """, (symbol,))
            
            stock_info = cursor.fetchone()
            if not stock_info:
                raise HTTPException(status_code=404, detail=f"Stock {symbol} not found in BIST categories")
                
            # Get historical data based on timeframe
            if timeframe in ['60min', 'hourly']:
                # Use historical_data table (intraday data)
                cursor.execute("""
                    SELECT 
                        date_time as timestamp,
                        open_price as open,
                        high_price as high,
                        low_price as low,
                        close_price as close,
                        volume,
                        rsi_14,
                        rsi_21,
                        macd_line,
                        macd_signal,
                        macd_histogram,
                        bollinger_upper,
                        bollinger_middle,
                        bollinger_lower,
                        tenkan_sen,
                        kijun_sen,
                        senkou_span_a,
                        senkou_span_b,
                        chikou_span,
                        atr_14,
                        adx_14
                    FROM historical_data 
                    WHERE symbol = %s 
                    ORDER BY date_time DESC 
                    LIMIT %s
                """, (symbol, limit))
                
            else:
                # Use enhanced_stock_data table (daily+ data)  
                cursor.execute("""
                    SELECT 
                        CONCAT(date, ' ', COALESCE(time, '00:00:00')) as timestamp,
                        open,
                        high, 
                        low,
                        close,
                        volume,
                        timeframe,
                        rsi_14,
                        macd_26_12 as macd_line,
                        macd_trigger_9 as macd_signal,
                        bol_upper_20_2 as bollinger_upper,
                        bol_middle_20_2 as bollinger_middle,
                        bol_lower_20_2 as bollinger_lower,
                        atr_14,
                        adx_14,
                        NULL as tenkan_sen,
                        NULL as kijun_sen,
                        NULL as senkou_span_a,
                        NULL as senkou_span_b,
                        NULL as chikou_span,
                        NULL as rsi_21,
                        NULL as macd_histogram
                    FROM enhanced_stock_data 
                    WHERE symbol = %s 
                    AND (timeframe = %s OR timeframe IS NULL)
                    ORDER BY date DESC, time DESC
                    LIMIT %s
                """, (symbol, timeframe, limit))
                
            historical_data = [dict(row) for row in cursor.fetchall()]
            
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "stock_info": dict(stock_info),
                    "timeframe": timeframe,
                    "historical_data": historical_data,
                    "total_records": len(historical_data),
                    "data_source": "historical_data" if timeframe in ['60min', 'hourly'] else "enhanced_stock_data",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            cursor.close()
            conn.close()
            
    async def get_bist_summary_stats(self):
        """Get summary statistics for all BIST categories"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    sc.category,
                    COUNT(DISTINCT sm.symbol) as total_stocks,
                    COUNT(DISTINCT CASE WHEN esd.symbol IS NOT NULL THEN sm.symbol END) as enhanced_data_count,
                    COUNT(DISTINCT CASE WHEN hd.symbol IS NOT NULL THEN sm.symbol END) as historical_data_count,
                    COUNT(DISTINCT CASE WHEN sm.api_available = TRUE THEN sm.symbol END) as api_available_count
                FROM stock_categories sc
                JOIN stocks_meta sm ON sc.symbol = sm.symbol
                LEFT JOIN (SELECT DISTINCT symbol FROM enhanced_stock_data) esd ON sm.symbol = esd.symbol  
                LEFT JOIN (SELECT DISTINCT symbol FROM historical_data) hd ON sm.symbol = hd.symbol
                WHERE sm.is_active = TRUE
                GROUP BY sc.category
                ORDER BY AVG(sc.priority)
            """)
            
            category_stats = [dict(row) for row in cursor.fetchall()]
            
            # Overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT sm.symbol) as total_unique_stocks,
                    COUNT(DISTINCT esd.symbol) as total_enhanced_symbols,
                    COUNT(DISTINCT hd.symbol) as total_historical_symbols,
                    SUM(CASE WHEN sm.api_available = TRUE THEN 1 ELSE 0 END) as total_api_available
                FROM stocks_meta sm
                LEFT JOIN (SELECT DISTINCT symbol FROM enhanced_stock_data) esd ON sm.symbol = esd.symbol
                LEFT JOIN (SELECT DISTINCT symbol FROM historical_data) hd ON sm.symbol = hd.symbol
                WHERE sm.is_active = TRUE
            """)
            
            overall_stats = dict(cursor.fetchone())
            
            return {
                "success": True,
                "data": {
                    "category_stats": category_stats,
                    "overall_stats": overall_stats,
                    "data_readiness": {
                        "bist_30_ready": next((c['enhanced_data_count'] + c['historical_data_count'] for c in category_stats if c['category'] == 'BIST_30'), 0) >= 30,
                        "bist_50_ready": next((c['enhanced_data_count'] + c['historical_data_count'] for c in category_stats if c['category'] == 'BIST_50'), 0) >= 50,
                        "bist_100_ready": next((c['enhanced_data_count'] + c['historical_data_count'] for c in category_stats if c['category'] == 'BIST_100'), 0) >= 100
                    },
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Summary stats error: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            cursor.close()
            conn.close()

# FastAPI endpoints
def setup_bist_categories_routes(app: FastAPI):
    api = RailwayBISTCategoriesAPI()
    
    @app.get("/api/bist/categories")
    async def get_bist_categories():
        """Get all BIST categories"""
        return await api.get_bist_categories()
        
    @app.get("/api/bist/stocks/{category}")
    async def get_stocks_by_category(
        category: str,
        limit: int = Query(default=100, ge=1, le=500, description="Number of stocks to return")
    ):
        """Get stocks by BIST category"""
        return await api.get_stocks_by_category(category, limit)
        
    @app.get("/api/bist/historical/{symbol}")
    async def get_stock_historical(
        symbol: str,
        timeframe: str = Query(default="60min", description="Data timeframe"),
        limit: int = Query(default=500, ge=1, le=2000, description="Number of records")
    ):
        """Get historical data for specific stock"""
        return await api.get_stock_historical_data(symbol, timeframe, limit)
        
    @app.get("/api/bist/summary")
    async def get_bist_summary():
        """Get BIST categories summary statistics"""
        return await api.get_bist_summary_stats()

if __name__ == "__main__":
    # Test the API locally
    import asyncio
    
    async def test_api():
        api = RailwayBISTCategoriesAPI()
        
        print("🧪 Testing BIST Categories API...")
        
        # Test categories
        categories = await api.get_bist_categories()
        print(f"📊 Categories: {categories}")
        
        # Test BIST_30 stocks
        bist30 = await api.get_stocks_by_category("BIST_30", 30)
        print(f"📊 BIST_30 stocks: {len(bist30['data']['stocks'])}")
        
        # Test historical data
        if bist30['data']['stocks']:
            symbol = bist30['data']['stocks'][0]['symbol']
            historical = await api.get_stock_historical_data(symbol, "60min", 100)
            print(f"📈 Historical data for {symbol}: {len(historical['data']['historical_data'])} records")
            
        # Test summary
        summary = await api.get_bist_summary_stats()
        print(f"📋 Summary: {summary}")
        
    asyncio.run(test_api())
