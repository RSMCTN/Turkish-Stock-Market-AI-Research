#!/usr/bin/env python3
"""
AI Price Integration Service
Ensures AI models use current Profit.com sync'd prices
"""

import logging
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

# Railway Database URL
DATABASE_URL = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"

class AIPriceIntegration:
    """
    🤖 AI Price Integration Service
    
    Provides current prices for AI models and analysis
    - Uses Profit.com sync'd prices when available
    - Falls back to historical data
    - Updates AI predictions with current market data
    """
    
    def __init__(self):
        self.database_url = DATABASE_URL
        logger.info("🤖 AIPriceIntegration initialized")
    
    async def get_current_price_for_ai(self, symbol: str) -> Dict:
        """
        Get current price for AI analysis with context
        """
        try:
            conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            cursor = conn.cursor()
            
            # Get current price with historical context
            cursor.execute("""
                SELECT 
                    -- Current price (Profit.com sync or historical)
                    COALESCE(
                        (SELECT current_price FROM current_prices WHERE symbol = %s),
                        (SELECT close FROM enhanced_stock_data 
                         WHERE symbol = %s ORDER BY date DESC LIMIT 1),
                        (SELECT close_price FROM historical_data 
                         WHERE symbol = %s ORDER BY date_time DESC LIMIT 1)
                    ) as current_price,
                    
                    -- Source info
                    CASE 
                        WHEN EXISTS (SELECT 1 FROM current_prices WHERE symbol = %s) 
                        THEN 'profit_sync'
                        ELSE 'historical'
                    END as price_source,
                    
                    -- Last updated
                    COALESCE(
                        (SELECT last_updated FROM current_prices WHERE symbol = %s),
                        CURRENT_TIMESTAMP
                    ) as last_updated,
                    
                    -- Previous day price for change calculation
                    COALESCE(
                        (SELECT close FROM enhanced_stock_data 
                         WHERE symbol = %s AND date < CURRENT_DATE 
                         ORDER BY date DESC LIMIT 1),
                        (SELECT close_price FROM historical_data 
                         WHERE symbol = %s AND date_time < CURRENT_DATE 
                         ORDER BY date_time DESC LIMIT 1)
                    ) as previous_close
            """, (symbol, symbol, symbol, symbol, symbol, symbol, symbol))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                current_price = float(result['current_price']) if result['current_price'] else 0
                previous_close = float(result['previous_close']) if result['previous_close'] else current_price
                
                # Calculate change
                change = current_price - previous_close
                change_percent = (change / previous_close * 100) if previous_close > 0 else 0
                
                return {
                    "symbol": symbol.upper(),
                    "current_price": current_price,
                    "previous_close": previous_close,
                    "change": change,
                    "change_percent": change_percent,
                    "price_source": result['price_source'],
                    "is_live": result['price_source'] == 'profit_sync',
                    "last_updated": result['last_updated'].isoformat() if result['last_updated'] else None,
                    "ai_ready": True
                }
            else:
                logger.warning(f"⚠️ No price data found for {symbol}")
                return {
                    "symbol": symbol.upper(),
                    "current_price": 0,
                    "ai_ready": False,
                    "error": "No price data available"
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting AI price for {symbol}: {e}")
            return {
                "symbol": symbol.upper(),
                "ai_ready": False,
                "error": str(e)
            }
    
    async def get_portfolio_current_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple symbols (for AI portfolio analysis)
        """
        try:
            conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            cursor = conn.cursor()
            
            # Batch query for efficiency
            placeholders = ','.join(['%s'] * len(symbols))
            
            cursor.execute(f"""
                SELECT DISTINCT
                    sm.symbol,
                    -- Current price
                    COALESCE(
                        cp.current_price,
                        (SELECT close FROM enhanced_stock_data esd 
                         WHERE esd.symbol = sm.symbol ORDER BY date DESC LIMIT 1),
                        (SELECT close_price FROM historical_data hd 
                         WHERE hd.symbol = sm.symbol ORDER BY date_time DESC LIMIT 1),
                        0
                    ) as current_price,
                    
                    -- Price source
                    CASE 
                        WHEN cp.current_price IS NOT NULL THEN 'profit_sync'
                        ELSE 'historical'
                    END as price_source,
                    
                    cp.last_updated
                    
                FROM stocks_meta sm
                LEFT JOIN current_prices cp ON sm.symbol = cp.symbol
                WHERE sm.symbol IN ({placeholders})
            """, symbols)
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Process results
            portfolio_prices = {}
            for row in results:
                symbol = row['symbol']
                portfolio_prices[symbol] = {
                    "current_price": float(row['current_price']) if row['current_price'] else 0,
                    "price_source": row['price_source'],
                    "is_live": row['price_source'] == 'profit_sync',
                    "last_updated": row['last_updated'].isoformat() if row['last_updated'] else None
                }
            
            logger.info(f"📊 Retrieved prices for {len(portfolio_prices)} symbols for AI analysis")
            return portfolio_prices
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio prices: {e}")
            return {}
    
    async def get_market_sentiment_price_data(self, symbols: List[str]) -> Dict:
        """
        Get price data optimized for sentiment analysis
        Includes recent price movements and volatility
        """
        try:
            conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            cursor = conn.cursor()
            
            sentiment_data = {}
            
            for symbol in symbols:
                # Get current + recent historical prices
                cursor.execute("""
                    SELECT 
                        -- Current price
                        COALESCE(
                            (SELECT current_price FROM current_prices WHERE symbol = %s),
                            (SELECT close FROM enhanced_stock_data 
                             WHERE symbol = %s ORDER BY date DESC LIMIT 1)
                        ) as current_price,
                        
                        -- Source
                        CASE 
                            WHEN EXISTS (SELECT 1 FROM current_prices WHERE symbol = %s) 
                            THEN 'profit_sync'
                            ELSE 'historical'
                        END as source,
                        
                        -- Recent prices for volatility calculation
                        ARRAY(
                            SELECT close FROM enhanced_stock_data 
                            WHERE symbol = %s 
                            ORDER BY date DESC LIMIT 5
                        ) as recent_closes
                """, (symbol, symbol, symbol, symbol))
                
                result = cursor.fetchone()
                
                if result and result['current_price']:
                    current_price = float(result['current_price'])
                    recent_closes = [float(p) for p in result['recent_closes'] if p] if result['recent_closes'] else [current_price]
                    
                    # Calculate volatility
                    if len(recent_closes) > 1:
                        avg_price = sum(recent_closes) / len(recent_closes)
                        volatility = (max(recent_closes) - min(recent_closes)) / avg_price * 100
                    else:
                        volatility = 0
                    
                    sentiment_data[symbol] = {
                        "current_price": current_price,
                        "is_live": result['source'] == 'profit_sync',
                        "recent_volatility": round(volatility, 2),
                        "price_trend": "up" if len(recent_closes) > 1 and current_price > recent_closes[1] else "down",
                        "confidence": 0.9 if result['source'] == 'profit_sync' else 0.7
                    }
            
            cursor.close()
            conn.close()
            
            logger.info(f"🎭 Prepared sentiment price data for {len(sentiment_data)} symbols")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment price data: {e}")
            return {}

# Global instance
ai_price = AIPriceIntegration()

async def main():
    """Test AI price integration"""
    test_symbols = ["AKBNK", "BRSAN", "THYAO"]
    
    print("🤖 Testing AI Price Integration...")
    
    # Test individual price
    akbnk_data = await ai_price.get_current_price_for_ai("AKBNK")
    print(f"\n📊 AKBNK AI Data:")
    print(f"  Price: ₺{akbnk_data.get('current_price', 0)}")
    print(f"  Live: {akbnk_data.get('is_live', False)}")
    print(f"  Change: {akbnk_data.get('change_percent', 0):.2f}%")
    
    # Test portfolio prices
    portfolio = await ai_price.get_portfolio_current_prices(test_symbols)
    print(f"\n💼 Portfolio Prices:")
    for symbol, data in portfolio.items():
        print(f"  {symbol}: ₺{data['current_price']} ({data['price_source']})")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
