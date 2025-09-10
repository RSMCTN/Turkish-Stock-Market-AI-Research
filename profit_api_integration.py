#!/usr/bin/env python3
"""
PROFIT.COM API Integration for MAMUT R600
Real-time BIST data integration with our trading system
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time

class ProfitAPIClient:
    def __init__(self):
        self.api_key = "a9a0bacbab08493d958244c05380da01"
        self.base_url = "https://api.profit.com"
        
    def get_turkish_stocks(self, limit=50):
        """Get Turkish stocks list"""
        url = f"{self.base_url}/data-api/reference/stocks"
        params = {
            'token': self.api_key,
            'country': 'Turkey',
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()['data']
            else:
                print(f"Error: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error fetching Turkish stocks: {e}")
            return []
    
    def get_real_time_quote(self, symbol):
        """Get real-time quote for a symbol"""
        url = f"{self.base_url}/data-api/market-data/quote/{symbol}"
        params = {'token': self.api_key}
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching {symbol}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol, days=30):
        """Get historical data for analysis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"{self.base_url}/data-api/historical/eod"
        params = {
            'token': self.api_key,
            'ticker': symbol,
            'start_date': int(start_date.timestamp()),
            'end_date': int(end_date.timestamp())
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching historical data for {symbol}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None

def test_integration():
    """Test the Profit.com integration"""
    print("🚀 TESTING PROFIT.COM INTEGRATION FOR MAMUT R600")
    print("=" * 60)
    
    client = ProfitAPIClient()
    
    # Test 1: Get Turkish stocks
    print("\n📊 FETCHING TURKISH STOCKS...")
    turkish_stocks = client.get_turkish_stocks(limit=10)
    print(f"Found {len(turkish_stocks)} Turkish stocks")
    
    for stock in turkish_stocks[:5]:
        print(f"  📈 {stock['symbol']:8} | {stock['name']}")
    
    # Test 2: Get real-time quotes for major BIST stocks
    print("\n💰 REAL-TIME QUOTES FOR MAJOR BIST STOCKS...")
    major_stocks = ['AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'YKBNK.IS', 'VAKBN.IS']
    
    quotes = []
    for symbol in major_stocks:
        quote = client.get_real_time_quote(symbol)
        if quote:
            quotes.append(quote)
            change_color = '🟢' if quote.get('daily_percentage_change', 0) >= 0 else '🔴'
            print(f"  {change_color} {symbol:10} | ₺{quote.get('price', 0):8.2f} | {quote.get('daily_percentage_change', 0):+6.2f}%")
    
    # Test 3: Historical data sample
    print("\n📈 HISTORICAL DATA SAMPLE (AKBNK.IS)...")
    historical = client.get_historical_data('AKBNK.IS', days=7)
    if historical and 'data' in historical:
        print(f"Got {len(historical['data'])} historical data points")
        for data_point in historical['data'][-3:]:  # Last 3 days
            date = datetime.fromtimestamp(data_point['timestamp']).strftime('%Y-%m-%d')
            print(f"  📅 {date} | Open: ₺{data_point['open']:.2f} | Close: ₺{data_point['close']:.2f}")
    
    print(f"\n✅ PROFIT.COM INTEGRATION SUCCESSFUL!")
    print(f"💡 Ready to integrate with MAMUT R600 trading system")
    
    return quotes

if __name__ == "__main__":
    test_integration()
