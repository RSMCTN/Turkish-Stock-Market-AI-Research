#!/usr/bin/env python3
"""
Updated Price Endpoint for Search Bar
Serves current prices from sync'd database
"""

from fastapi import HTTPException
import psycopg2
import psycopg2.extras
import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

# Railway Database URL
DATABASE_URL = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"

async def get_updated_prices_for_stocks(stocks: List[Dict]) -> List[Dict]:
    """
    Update stock list with current prices from sync'd database
    """
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        cursor = conn.cursor()
        
        # Get symbols from stocks
        symbols = [stock['symbol'] for stock in stocks]
        
        if not symbols:
            return stocks
        
        # Get current prices
        placeholders = ','.join(['%s'] * len(symbols))
        cursor.execute(f"""
            SELECT symbol, current_price, last_updated
            FROM current_prices
            WHERE symbol IN ({placeholders})
            AND last_updated > NOW() - INTERVAL '1 day'
        """, symbols)
        
        current_prices = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Create price lookup dict
        price_lookup = {}
        for row in current_prices:
            price_lookup[row['symbol']] = {
                'price': float(row['current_price']),
                'updated': row['last_updated']
            }
        
        # Update stocks with current prices
        updated_stocks = []
        for stock in stocks:
            symbol = stock['symbol']
            updated_stock = stock.copy()
            
            if symbol in price_lookup:
                # Use current price from Profit.com sync
                current_data = price_lookup[symbol]
                updated_stock['latest_price'] = current_data['price']
                updated_stock['price_source'] = 'profit_sync'
                updated_stock['price_updated'] = current_data['updated'].isoformat()
                
                logger.info(f"✅ Updated {symbol}: ₺{current_data['price']}")
            else:
                # Keep original price but mark as historical
                updated_stock['price_source'] = 'historical'
                logger.warning(f"⚠️ No sync price for {symbol}, using historical: ₺{stock.get('latest_price', 0)}")
            
            updated_stocks.append(updated_stock)
        
        logger.info(f"📊 Updated {len(price_lookup)}/{len(stocks)} stock prices")
        return updated_stocks
        
    except Exception as e:
        logger.error(f"❌ Error getting updated prices: {e}")
        # Return original stocks if error
        return stocks

if __name__ == "__main__":
    # Test the price update function
    test_stocks = [
        {"symbol": "AKBNK", "latest_price": 100.0},
        {"symbol": "BRSAN", "latest_price": 500.0},
        {"symbol": "THYAO", "latest_price": 300.0}
    ]
    
    import asyncio
    
    async def test():
        updated = await get_updated_prices_for_stocks(test_stocks)
        print("🧪 Test Results:")
        for stock in updated:
            print(f"  {stock['symbol']}: ₺{stock['latest_price']} ({stock.get('price_source', 'unknown')})")
    
    asyncio.run(test())
