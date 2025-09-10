#!/usr/bin/env python3
"""
Profit.com PRO API Collector for BIST Stock Data
Real-time and historical Turkish stock market data
"""

import asyncio
import aiohttp
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
import json
import time

class ProfitCollector:
    """
    Profit.com PRO API collector for BIST stock data
    
    Features:
    - Real-time quotes for Turkish stocks (.IS symbols)
    - Historical data (daily, intraday)
    - Stock fundamentals and company information
    - Market overview and sector data
    - Rate limiting and error handling
    """
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.profit.com"
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting (30 API calls per second max)
        self.rate_limit = 25  # Conservative limit
        self.last_request_time = 0
        self.request_interval = 1.0 / self.rate_limit
        
        # Session for connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache for symbol mappings
        self.symbol_cache: Dict[str, Dict] = {}
        self.cache_expiry = 3600  # 1 hour cache
        self.last_cache_update = 0
        
        self.logger.info(f"🏦 ProfitCollector initialized with API token")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'MAMUT_R600/1.0 BIST Trading System'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _rate_limit_check(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            sleep_time = self.request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make authenticated request to Profit.com API
        
        Args:
            endpoint: API endpoint (e.g., '/data-api/reference/stocks')
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        await self._rate_limit_check()
        
        if not self.session:
            raise RuntimeError("ProfitCollector not initialized as context manager")
        
        # Add authentication token to parameters
        if params is None:
            params = {}
        params['token'] = self.api_token
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                self.logger.debug(f"API Request: {endpoint} - Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 401:
                    raise ValueError("Invalid API token - authentication failed")
                elif response.status == 403:
                    error_data = await response.json()
                    raise ValueError(f"API access forbidden: {error_data.get('message', 'Unknown error')}")
                elif response.status == 429:
                    raise RuntimeError("Rate limit exceeded - please wait")
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"API request failed: {response.status} - {error_text[:200]}")
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error for {endpoint}: {str(e)}")
            raise RuntimeError(f"Network error: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Check if Profit.com API is accessible and token is valid
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            data = await self._make_request('/data-api/reference/stocks', {'limit': 1})
            
            if isinstance(data, dict) and 'data' in data:
                self.logger.info("✅ Profit.com API health check passed")
                return True
            else:
                self.logger.warning("⚠️ Profit.com API health check failed - unexpected response")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Profit.com API health check failed: {str(e)}")
            return False
    
    async def get_turkish_stocks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all Turkish stocks from BIST (Borsa Istanbul)
        
        Args:
            limit: Maximum number of stocks to return
            
        Returns:
            List of Turkish stock data
        """
        try:
            data = await self._make_request('/data-api/reference/stocks', {
                'country': 'Turkey',
                'limit': limit
            })
            
            if 'data' in data:
                stocks = data['data']
                self.logger.info(f"📊 Retrieved {len(stocks)} Turkish stocks from Profit.com")
                
                # Process and standardize stock data
                processed_stocks = []
                for stock in stocks:
                    processed_stock = {
                        'symbol': stock.get('symbol'),
                        'ticker': stock.get('ticker'),  # Includes .IS suffix
                        'name': stock.get('name'),
                        'type': stock.get('type'),
                        'currency': stock.get('currency', 'TRY'),
                        'exchange': stock.get('exchange', 'IS'),
                        'country': stock.get('country', 'Turkey')
                    }
                    processed_stocks.append(processed_stock)
                
                return processed_stocks
            else:
                self.logger.warning("No Turkish stock data found in API response")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching Turkish stocks: {str(e)}")
            return []
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a Turkish stock
        
        Args:
            symbol: Stock symbol (e.g., 'GARAN' or 'GARAN.IS')
            
        Returns:
            Real-time quote data
        """
        # Ensure symbol has .IS suffix for Turkish stocks
        if not symbol.endswith('.IS'):
            ticker = f"{symbol}.IS"
        else:
            ticker = symbol
        
        try:
            data = await self._make_request(f'/data-api/market-data/quote/{ticker}')
            
            if data and 'price' in data:
                quote = {
                    'symbol': data.get('symbol', symbol),
                    'ticker': data.get('ticker', ticker),
                    'name': data.get('name', ''),
                    'last_price': data.get('price', 0.0),
                    'previous_close': data.get('previous_close', 0.0),
                    'change': data.get('daily_price_change', 0.0),
                    'change_percent': data.get('daily_percentage_change', 0.0),
                    'volume': data.get('volume', 0),
                    'timestamp': data.get('timestamp', int(datetime.now().timestamp())),
                    'currency': 'TRY',
                    'market_state': data.get('market_state', 'unknown')
                }
                
                self.logger.debug(f"📊 Quote for {ticker}: {quote['last_price']} TRY")
                return quote
            else:
                self.logger.warning(f"No quote data found for {ticker}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error fetching quote for {ticker}: {str(e)}")
            return {}
    
    async def get_historical_daily(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get historical daily data for a Turkish stock
        
        Args:
            symbol: Stock symbol (e.g., 'GARAN' or 'GARAN.IS')
            days: Number of days of historical data
            
        Returns:
            List of daily OHLCV data
        """
        # Ensure symbol has .IS suffix for Turkish stocks
        if not symbol.endswith('.IS'):
            ticker = f"{symbol}.IS"
        else:
            ticker = symbol
        
        try:
            # Calculate date range (approximate)
            end_date = int(datetime.now().timestamp())
            start_date = end_date - (days * 24 * 60 * 60)  # days ago
            
            data = await self._make_request(f'/data-api/market-data/historical/daily/{ticker}', {
                'start_date': start_date,
                'end_date': end_date
            })
            
            if 'data' in data and data['data']:
                historical_data = []
                for record in data['data']:
                    ohlcv = {
                        'date': record.get('date', ''),
                        'open': record.get('open', 0.0),
                        'high': record.get('high', 0.0),
                        'low': record.get('low', 0.0),
                        'close': record.get('close', 0.0),
                        'volume': record.get('volume', 0),
                        'adjusted_close': record.get('adjusted_close', record.get('close', 0.0))
                    }
                    historical_data.append(ohlcv)
                
                self.logger.info(f"📈 Retrieved {len(historical_data)} days of historical data for {ticker}")
                return historical_data
            else:
                self.logger.warning(f"No historical data found for {ticker}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
            return []
    
    async def get_stock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data for a Turkish stock
        
        Args:
            symbol: Stock symbol (e.g., 'GARAN' or 'GARAN.IS')
            
        Returns:
            Fundamental data (company info, financials, etc.)
        """
        # Ensure symbol has .IS suffix for Turkish stocks
        if not symbol.endswith('.IS'):
            ticker = f"{symbol}.IS"
        else:
            ticker = symbol
        
        try:
            data = await self._make_request(f'/data-api/fundamentals/stocks/general/{ticker}')
            
            if data:
                fundamentals = {
                    'symbol': data.get('Symbol', symbol),
                    'name': data.get('Name', ''),
                    'sector': data.get('Sector', ''),
                    'industry': data.get('Industry', ''),
                    'market_cap': data.get('MarketCapitalization', 0),
                    'pe_ratio': data.get('PERatio', 0),
                    'eps': data.get('EPS', 0),
                    'dividend_yield': data.get('DividendYield', 0),
                    'book_value': data.get('BookValue', 0),
                    'country': data.get('Country', 'Turkey'),
                    'exchange': data.get('Exchange', 'IS'),
                    'currency': data.get('Currency', 'TRY')
                }
                
                self.logger.debug(f"📋 Fundamentals for {ticker}: {fundamentals['name']}")
                return fundamentals
            else:
                self.logger.warning(f"No fundamental data found for {ticker}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
            return {}
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get real-time quotes for multiple Turkish stocks
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to quote data
        """
        quotes = {}
        
        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Fetch quotes concurrently within batch
            tasks = [self.get_real_time_quote(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, dict) and result:
                    quotes[symbol] = result
                elif isinstance(result, Exception):
                    self.logger.warning(f"Error fetching quote for {symbol}: {result}")
                    quotes[symbol] = {}
                else:
                    quotes[symbol] = {}
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.5)
        
        self.logger.info(f"📊 Retrieved quotes for {len([q for q in quotes.values() if q])} out of {len(symbols)} symbols")
        return quotes

# Convenience function for quick testing
async def test_profit_collector():
    """Test function to verify Profit.com API integration"""
    
    PROFIT_API_KEY = "a9a0bacbab08493d958244c05380da01"
    
    async with ProfitCollector(PROFIT_API_KEY) as collector:
        print("🧪 TESTING PROFIT.COM COLLECTOR...")
        print("=" * 50)
        
        # Health check
        if await collector.health_check():
            print("✅ API connection successful")
        else:
            print("❌ API connection failed")
            return
        
        # Get Turkish stocks
        print("\n📊 FETCHING TURKISH STOCKS...")
        stocks = await collector.get_turkish_stocks(limit=10)
        print(f"Found {len(stocks)} Turkish stocks")
        
        if stocks:
            # Show sample stocks
            for stock in stocks[:3]:
                print(f"   📈 {stock['ticker']}: {stock['name']}")
        
        # Test real-time quotes
        print("\n💰 TESTING REAL-TIME QUOTES...")
        test_symbols = ['GARAN', 'AKBNK', 'THYAO']
        
        for symbol in test_symbols:
            quote = await collector.get_real_time_quote(symbol)
            if quote:
                print(f"   💵 {symbol}: {quote['last_price']} TRY ({quote['change_percent']:+.2f}%)")
            else:
                print(f"   ❌ {symbol}: No data")
        
        # Test historical data
        print("\n📈 TESTING HISTORICAL DATA...")
        historical = await collector.get_historical_daily('GARAN', days=5)
        if historical:
            print(f"   📊 GARAN: {len(historical)} days of data")
            if historical:
                latest = historical[-1]
                print(f"   📅 Latest: {latest['date']} - Close: {latest['close']}")
        
        print("\n✅ PROFIT.COM COLLECTOR TEST COMPLETED!")

if __name__ == "__main__":
    asyncio.run(test_profit_collector())
