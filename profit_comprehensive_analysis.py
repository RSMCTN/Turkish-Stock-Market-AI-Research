#!/usr/bin/env python3
"""
Profit.com API Kapsamlı Veri Analizi
Comprehensive analysis of all available Profit.com API data fields
"""

import requests
import json
import pandas as pd
from datetime import datetime
import time

class ProfitAPIAnalyzer:
    def __init__(self):
        self.api_key = "a9a0bacbab08493d958244c05380da01"
        self.base_url = "https://api.profit.com"
        
    def test_reference_stocks_endpoint(self):
        """Test the reference stocks endpoint for available fields"""
        print("🔍 REFERENCE STOCKS ENDPOINT TESTİ")
        print("=" * 50)
        
        url = f"{self.base_url}/data-api/reference/stocks"
        params = {
            'token': self.api_key,
            'country': 'Turkey',
            'limit': 5  # Get 5 stocks to analyze structure
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            print(f"📡 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Data structure:")
                print(f"📊 Response keys: {list(data.keys())}")
                
                if 'data' in data and len(data['data']) > 0:
                    first_stock = data['data'][0]
                    print(f"\n📋 REFERENCE DATA FIELDS ({len(first_stock)} alanı):")
                    print("-" * 60)
                    
                    for i, (key, value) in enumerate(first_stock.items(), 1):
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        print(f"{i:2d}. {key:20} : {value_str}")
                    
                    return data['data']
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"📄 Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"💥 Exception: {e}")
            
        return None
    
    def test_market_data_endpoint(self, symbol="AKBNK.IS"):
        """Test market data endpoint for live data fields"""
        print(f"\n🔍 MARKET DATA ENDPOINT TESTİ - {symbol}")
        print("=" * 50)
        
        url = f"{self.base_url}/data-api/market-data/quote/{symbol}"
        params = {'token': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            print(f"📡 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Market data retrieved")
                
                print(f"\n📋 MARKET DATA FIELDS ({len(data)} alanı):")
                print("-" * 60)
                
                for i, (key, value) in enumerate(data.items(), 1):
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    print(f"{i:2d}. {key:20} : {value_str}")
                
                # Check for volume, sector, index info
                important_fields = ['volume', 'sector', 'market', 'index', 'group', 'industry']
                print(f"\n🎯 ÖNEMLI ALANLAR KONTROLÜ:")
                print("-" * 30)
                
                for field in important_fields:
                    if field in data:
                        print(f"✅ {field:15} : {data[field]}")
                    else:
                        # Check case variations
                        found = False
                        for key in data.keys():
                            if field.lower() in key.lower():
                                print(f"✅ {key:15} : {data[key]} (contains '{field}')")
                                found = True
                                break
                        if not found:
                            print(f"❌ {field:15} : Not found")
                
                return data
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"📄 Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"💥 Exception: {e}")
            
        return None
    
    def get_comprehensive_stock_list(self):
        """Get comprehensive list of Turkish stocks"""
        print(f"\n🔍 COMPREHENSIVE STOCK LIST")
        print("=" * 50)
        
        url = f"{self.base_url}/data-api/reference/stocks"
        params = {
            'token': self.api_key,
            'country': 'Turkey',
            'limit': 100  # Get more stocks
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()['data']
                print(f"✅ Retrieved {len(data)} Turkish stocks")
                
                # Analyze sectors and markets
                sectors = set()
                markets = set()
                currencies = set()
                
                for stock in data:
                    if 'sector' in stock:
                        sectors.add(stock['sector'])
                    if 'market' in stock:
                        markets.add(stock['market'])
                    if 'currency' in stock:
                        currencies.add(stock['currency'])
                
                print(f"\n📊 FOUND SECTORS ({len(sectors)}):")
                for i, sector in enumerate(sorted(sectors), 1):
                    print(f"   {i}. {sector}")
                
                print(f"\n📊 FOUND MARKETS ({len(markets)}):")
                for i, market in enumerate(sorted(markets), 1):
                    print(f"   {i}. {market}")
                    
                print(f"\n💰 CURRENCIES ({len(currencies)}):")
                for currency in sorted(currencies):
                    print(f"   • {currency}")
                
                return data
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"💥 Exception: {e}")
            
        return None
    
    def analyze_multiple_stocks(self, count=5):
        """Analyze multiple stocks to see variation in available data"""
        print(f"\n🔍 MULTIPLE STOCKS ANALYSIS ({count} stocks)")
        print("=" * 60)
        
        # Get stock list first
        stocks = self.get_comprehensive_stock_list()
        if not stocks:
            return None
            
        # Test first few stocks
        test_stocks = stocks[:count]
        all_fields = set()
        stock_data = {}
        
        for i, stock in enumerate(test_stocks, 1):
            symbol = stock.get('ticker', stock.get('symbol', ''))
            if not symbol.endswith('.IS'):
                symbol += '.IS'
                
            print(f"\n📊 {i}/{count}: Testing {symbol} ({stock.get('name', 'Unknown')})")
            
            market_data = self.test_market_data_endpoint(symbol)
            if market_data:
                stock_data[symbol] = {
                    'reference': stock,
                    'market_data': market_data
                }
                all_fields.update(market_data.keys())
            
            time.sleep(0.5)  # Rate limiting
        
        print(f"\n✅ ANALYSIS COMPLETE")
        print(f"📊 Total unique fields across all stocks: {len(all_fields)}")
        
        print(f"\n📋 ALL UNIQUE FIELDS:")
        print("-" * 40)
        for i, field in enumerate(sorted(all_fields), 1):
            print(f"{i:2d}. {field}")
        
        return stock_data, all_fields
    
    def create_search_data(self):
        """Create searchable stock data structure"""
        print(f"\n🔍 CREATING SEARCHABLE STOCK DATA")
        print("=" * 50)
        
        stocks = self.get_comprehensive_stock_list()
        if not stocks:
            return None
        
        search_data = []
        for stock in stocks:
            # Create searchable structure
            search_item = {
                'symbol': stock.get('symbol', ''),
                'ticker': stock.get('ticker', ''),
                'name': stock.get('name', ''),
                'sector': stock.get('sector', 'Unknown'),
                'market': stock.get('market', 'Unknown'),
                'currency': stock.get('currency', 'TRY'),
                'search_text': f"{stock.get('symbol', '')} {stock.get('name', '')} {stock.get('sector', '')}".lower()
            }
            search_data.append(search_item)
        
        print(f"✅ Created searchable data for {len(search_data)} stocks")
        
        # Save to JSON for dashboard use
        with open('profit_search_data.json', 'w', encoding='utf-8') as f:
            json.dump(search_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved to: profit_search_data.json")
        
        # Show sample
        print(f"\n📋 SAMPLE SEARCH DATA:")
        print("-" * 40)
        for i, item in enumerate(search_data[:3]):
            print(f"{i+1}. {item['symbol']:10} | {item['name'][:30]:30} | {item['sector']:15}")
        
        return search_data

    def run_comprehensive_analysis(self):
        """Run complete analysis"""
        print("🚀 PROFIT.COM API COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Test reference endpoint
        reference_data = self.test_reference_stocks_endpoint()
        
        # Step 2: Test market data endpoint 
        market_data = self.test_market_data_endpoint("GARAN.IS")
        
        # Step 3: Analyze multiple stocks
        multi_data, all_fields = self.analyze_multiple_stocks(5)
        
        # Step 4: Create search data
        search_data = self.create_search_data()
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("=" * 30)
        print(f"✅ Reference data fields analyzed")
        print(f"✅ Market data fields analyzed") 
        print(f"✅ Multi-stock comparison done")
        print(f"✅ Search data structure created")
        print(f"📊 Total API fields discovered: {len(all_fields) if all_fields else 'N/A'}")
        
        return {
            'reference': reference_data,
            'market': market_data,
            'multi': multi_data,
            'search': search_data,
            'all_fields': all_fields
        }

if __name__ == "__main__":
    analyzer = ProfitAPIAnalyzer()
    results = analyzer.run_comprehensive_analysis()
