"""
MatriksIQ API Integration for BIST Market Data
"""

import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass


@dataclass
class MarketDataPoint:
    """Single market data point"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1d"


class MatriksCollector:
    """MatriksIQ API client for BIST data"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.matriks.com.tr/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.rate_limit_semaphore = asyncio.Semaphore(10)  # 10 concurrent requests
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    async def __aenter__(self):
        """Async context manager entry"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'MAMUT_R600/1.0 (BIST Trading System)'
        }
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=connector
        )
        
        self.logger.info("MatriksCollector initialized successfully")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.logger.info("MatriksCollector session closed")
    
    async def _rate_limited_request(self, method: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make rate-limited HTTP request"""
        async with self.rate_limit_semaphore:
            # Enforce minimum time between requests
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            try:
                async with getattr(self.session, method.lower())(url, **kwargs) as response:
                    self.last_request_time = asyncio.get_event_loop().time()
                    
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        self.logger.warning(f"Rate limited, waiting 60s...")
                        await asyncio.sleep(60)
                        return await self._rate_limited_request(method, url, **kwargs)
                    elif response.status == 401:
                        self.logger.error("Authentication failed - check API key")
                        return None
                    else:
                        self.logger.error(f"API Error {response.status}: {await response.text()}")
                        return None
                        
            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout for {url}")
                return None
            except Exception as e:
                self.logger.error(f"Request failed for {url}: {str(e)}")
                return None
    
    async def get_symbols(self) -> List[Dict[str, Any]]:
        """Get all BIST symbols with metadata"""
        self.logger.info("Fetching BIST symbols...")
        
        url = f"{self.base_url}/symbols"
        data = await self._rate_limited_request("GET", url, params={"market": "BIST"})
        
        if not data:
            self.logger.error("Failed to fetch symbols")
            return []
        
        symbols = []
        
        # Process symbol data (assuming API response format)
        for item in data.get('symbols', data.get('data', [])):
            try:
                symbol_info = {
                    'symbol': item.get('symbol', item.get('code', '')).upper(),
                    'name': item.get('name', item.get('long_name', '')),
                    'sector': item.get('sector', item.get('industry', '')),
                    'market_cap': item.get('market_cap', item.get('marketCap', 0)),
                    'type': item.get('type', 'equity'),
                    'currency': item.get('currency', 'TRY'),
                    'active': item.get('active', item.get('is_active', True))
                }
                
                # Filter only equities and active symbols
                if (symbol_info['type'].lower() == 'equity' and 
                    symbol_info['active'] and 
                    symbol_info['symbol']):
                    symbols.append(symbol_info)
                    
            except Exception as e:
                self.logger.warning(f"Error processing symbol {item}: {str(e)}")
                continue
        
        self.logger.info(f"Fetched {len(symbols)} BIST symbols")
        return symbols
    
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1d",
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a symbol"""
        
        self.logger.info(f"Fetching historical data for {symbol} ({period})")
        
        # Prepare parameters
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': limit
        }
        
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        url = f"{self.base_url}/historical"
        data = await self._rate_limited_request("GET", url, params=params)
        
        if not data:
            self.logger.warning(f"No data received for {symbol}")
            return pd.DataFrame()
        
        try:
            # Process the response data
            records = data.get('data', data.get('bars', []))
            
            if not records:
                self.logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for record in records:
                try:
                    row = {
                        'timestamp': pd.to_datetime(record.get('timestamp', record.get('date', record.get('time')))),
                        'open': float(record.get('open', record.get('o', 0))),
                        'high': float(record.get('high', record.get('h', 0))),
                        'low': float(record.get('low', record.get('l', 0))),
                        'close': float(record.get('close', record.get('c', 0))),
                        'volume': float(record.get('volume', record.get('v', 0)))
                    }
                    df_data.append(row)
                except (ValueError, TypeError, KeyError) as e:
                    self.logger.warning(f"Error processing record for {symbol}: {record} - {str(e)}")
                    continue
            
            if not df_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Data validation
            df = df[(df > 0).all(axis=1)]  # Remove invalid rows
            
            self.logger.info(f"Fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol"""
        
        self.logger.debug(f"Fetching real-time quote for {symbol}")
        
        url = f"{self.base_url}/quote/{symbol.upper()}"
        data = await self._rate_limited_request("GET", url)
        
        if not data:
            return {}
        
        try:
            # Process quote data
            quote = data.get('data', data)
            
            result = {
                'symbol': symbol.upper(),
                'timestamp': datetime.utcnow(),
                'last_price': float(quote.get('last', quote.get('price', 0))),
                'bid': float(quote.get('bid', 0)),
                'ask': float(quote.get('ask', 0)),
                'volume': float(quote.get('volume', 0)),
                'change': float(quote.get('change', 0)),
                'change_percent': float(quote.get('change_percent', quote.get('changePercent', 0))),
                'high': float(quote.get('high', 0)),
                'low': float(quote.get('low', 0)),
                'open': float(quote.get('open', 0))
            }
            
            return result
            
        except (ValueError, TypeError, KeyError) as e:
            self.logger.error(f"Error processing quote for {symbol}: {str(e)}")
            return {}
    
    async def health_check(self) -> bool:
        """Check if API is accessible"""
        try:
            url = f"{self.base_url}/health"
            data = await self._rate_limited_request("GET", url)
            
            if data is None:
                # Try with a simple endpoint
                symbols = await self.get_symbols()
                return len(symbols) > 0
            
            return data.get('status') == 'ok'
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False


# Utility functions
def create_matriks_collector(api_key: str) -> MatriksCollector:
    """Factory function to create MatriksCollector"""
    if not api_key:
        raise ValueError("MatriksIQ API key is required")
    
    return MatriksCollector(api_key=api_key)


if __name__ == "__main__":
    # Basic test
    async def main():
        import os
        api_key = os.getenv("MATRIKS_API_KEY", "test_key")
        
        print("🔍 Testing MatriksCollector...")
        collector = MatriksCollector(api_key)
        
        print("✅ MatriksCollector created successfully!")
        print(f"   API Key: {api_key[:10]}...")
        print(f"   Base URL: {collector.base_url}")
        print(f"   Rate limit: {collector.min_request_interval}s")
        
        print("\n📊 MatriksCollector test completed!")
    
    asyncio.run(main())
