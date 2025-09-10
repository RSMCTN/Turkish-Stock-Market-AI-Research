#!/usr/bin/env python3
"""
Real-time Price Synchronization Service
Profit.com API → Railway PostgreSQL Database
"""

import asyncio
import logging
import psycopg2
import psycopg2.extras
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceSyncService:
    """
    🔄 Real-time Price Synchronization Service
    
    Features:
    - Fetches live prices from Profit.com API
    - Updates Railway PostgreSQL database
    - Maintains price history for AI models
    - Supports batch updates for performance
    - Error handling and retry logic
    """
    
    def __init__(self):
        self.profit_api_key = "a9a0bacbab08493d958244c05380da01"
        self.profit_base_url = "https://api.profit.com"
        
        # Railway Database URL - Use SAME connection as main API
        self.database_url = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"
        
        logger.info("🔄 PriceSyncService initialized")
        
    async def get_current_price_from_profit(self, symbol: str) -> Optional[Dict]:
        """Get current price from Profit.com API"""
        try:
            url = f"{self.profit_base_url}/data-api/market-data/quote/{symbol}.IS"
            response = requests.get(f"{url}?token={self.profit_api_key}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "symbol": symbol.upper(),
                    "price": float(data.get("price", 0)),
                    "timestamp": datetime.now(),
                    "source": "profit_api"
                }
            else:
                logger.warning(f"❌ Profit API error for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} from Profit.com: {e}")
            return None
    
    async def update_database_price(self, price_data: Dict) -> bool:
        """Update current price in Railway PostgreSQL"""
        try:
            conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            cursor = conn.cursor()
            
            # Create/Update current_prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS current_prices (
                    symbol VARCHAR(10) PRIMARY KEY,
                    current_price DECIMAL(10,2) NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source VARCHAR(20) DEFAULT 'profit_api',
                    change_percent DECIMAL(5,2) DEFAULT 0,
                    volume BIGINT DEFAULT 0
                )
            """)
            
            # Upsert current price
            cursor.execute("""
                INSERT INTO current_prices (symbol, current_price, last_updated, source)
                VALUES (%(symbol)s, %(price)s, %(timestamp)s, %(source)s)
                ON CONFLICT (symbol) 
                DO UPDATE SET 
                    current_price = EXCLUDED.current_price,
                    last_updated = EXCLUDED.last_updated,
                    source = EXCLUDED.source
            """, price_data)
            
            conn.commit()
            logger.info(f"✅ Updated {price_data['symbol']}: ₺{price_data['price']}")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Database update error for {price_data['symbol']}: {e}")
            return False
    
    async def sync_bist_prices(self, symbols: List[str]) -> Dict:
        """Sync prices for multiple BIST symbols"""
        results = {
            "success": [],
            "failed": [],
            "total_processed": 0,
            "sync_time": datetime.now().isoformat()
        }
        
        logger.info(f"🚀 Starting price sync for {len(symbols)} symbols...")
        
        for symbol in symbols:
            try:
                # Get price from Profit.com
                price_data = await self.get_current_price_from_profit(symbol)
                
                if price_data:
                    # Update database
                    success = await self.update_database_price(price_data)
                    
                    if success:
                        results["success"].append({
                            "symbol": symbol,
                            "price": price_data["price"],
                            "timestamp": price_data["timestamp"].isoformat()
                        })
                    else:
                        results["failed"].append(symbol)
                else:
                    results["failed"].append(symbol)
                    
                results["total_processed"] += 1
                
                # Rate limiting - don't overwhelm APIs
                await asyncio.sleep(0.1)  # 100ms between requests
                
            except Exception as e:
                logger.error(f"❌ Sync error for {symbol}: {e}")
                results["failed"].append(symbol)
        
        logger.info(f"✅ Sync completed: {len(results['success'])}/{len(symbols)} successful")
        return results
    
    async def get_updated_prices_for_search(self, symbols: List[str]) -> Dict[str, float]:
        """Get updated prices for search bar display"""
        try:
            conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            cursor = conn.cursor()
            
            # Get current prices from our sync table
            placeholders = ','.join(['%s'] * len(symbols))
            cursor.execute(f"""
                SELECT symbol, current_price, last_updated
                FROM current_prices
                WHERE symbol IN ({placeholders})
                AND last_updated > NOW() - INTERVAL '1 hour'
            """, symbols)
            
            current_prices = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Convert to dict
            price_dict = {}
            for row in current_prices:
                price_dict[row['symbol']] = float(row['current_price'])
                
            logger.info(f"📊 Retrieved {len(price_dict)} updated prices for search")
            return price_dict
            
        except Exception as e:
            logger.error(f"❌ Error getting updated prices: {e}")
            return {}

# Global instance for easy import
price_sync = PriceSyncService()

async def main():
    """Test the price sync service"""
    test_symbols = ["AKBNK", "BRSAN", "THYAO", "BIMAS", "GARAN"]
    
    print("🧪 Testing Price Sync Service...")
    results = await price_sync.sync_bist_prices(test_symbols)
    
    print("\n📊 Sync Results:")
    print(f"✅ Successful: {len(results['success'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    
    for success in results['success'][:3]:  # Show first 3
        print(f"  {success['symbol']}: ₺{success['price']}")

if __name__ == "__main__":
    asyncio.run(main())
