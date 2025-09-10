#!/usr/bin/env python3
"""
EXHAUSTIVE Profit.com API Fetch
Get EVERY SINGLE Turkish stock - no duplicates, no missing
"""

import requests
import json
import time
from datetime import datetime
from collections import defaultdict

class ExhaustiveAPIFetcher:
    def __init__(self):
        self.api_key = "a9a0bacbab08493d958244c05380da01"
        self.base_url = "https://api.profit.com"
    
    def fetch_all_stocks_exhaustive(self):
        """Fetch ALL stocks with maximum possible limit and pagination"""
        print("🔥 EXHAUSTIVE API FETCH - TÜM HİSSELER")
        print("=" * 50)
        
        all_stocks = []
        seen_symbols = set()  # Track duplicates
        
        # Try different approaches to get ALL stocks
        approaches = [
            {"limit": 1000, "name": "Max Limit 1000"},
            {"limit": 500, "name": "Large Batch 500"},
            {"limit": 200, "name": "Medium Batch 200"},
        ]
        
        for approach in approaches:
            print(f"\n🎯 Approach: {approach['name']}")
            print("-" * 30)
            
            page = 0
            limit = approach["limit"]
            approach_stocks = []
            
            while True:
                offset = page * limit
                print(f"📄 Page {page + 1}: offset={offset}, limit={limit}")
                
                url = f"{self.base_url}/data-api/reference/stocks"
                params = {
                    'token': self.api_key,
                    'country': 'Turkey',
                    'limit': limit,
                    'offset': offset
                }
                
                try:
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code != 200:
                        print(f"❌ Status: {response.status_code}")
                        break
                    
                    data = response.json()
                    stocks_batch = data.get('data', [])
                    total_in_api = data.get('total', 0)
                    
                    print(f"   📊 Received: {len(stocks_batch)} stocks")
                    print(f"   📊 API Total: {total_in_api}")
                    
                    if not stocks_batch:
                        print("   ✅ No more stocks")
                        break
                    
                    approach_stocks.extend(stocks_batch)
                    
                    # Check if we've reached the end
                    if len(stocks_batch) < limit:
                        print("   ✅ Last page reached")
                        break
                    
                    page += 1
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    break
            
            print(f"🎯 {approach['name']}: {len(approach_stocks)} stocks fetched")
            
            # Add unique stocks only
            new_unique = 0
            for stock in approach_stocks:
                symbol = stock.get('symbol', '')
                if symbol and symbol not in seen_symbols:
                    all_stocks.append(stock)
                    seen_symbols.add(symbol)
                    new_unique += 1
            
            print(f"✅ New unique stocks added: {new_unique}")
            print(f"📊 Total unique so far: {len(all_stocks)}")
        
        print(f"\n🎊 EXHAUSTIVE FETCH COMPLETE!")
        print(f"📊 Total unique Turkish stocks: {len(all_stocks)}")
        
        return all_stocks
    
    def create_clean_database(self, stocks):
        """Create clean database without duplicates"""
        print(f"\n🧹 CLEANING DATABASE")
        print("=" * 20)
        
        clean_stocks = []
        symbol_tracking = defaultdict(list)
        
        # Group by symbol to handle duplicates
        for stock in stocks:
            symbol = stock.get('symbol', '').upper()
            if symbol:
                symbol_tracking[symbol].append(stock)
        
        print(f"📊 Unique symbols found: {len(symbol_tracking)}")
        
        # For each symbol, keep the best version
        for symbol, stock_versions in symbol_tracking.items():
            if len(stock_versions) == 1:
                # Single version - keep it
                best_stock = stock_versions[0]
            else:
                # Multiple versions - choose best one
                print(f"🔄 {symbol}: {len(stock_versions)} versions - choosing best")
                
                # Prefer the one with most complete data
                best_stock = max(stock_versions, key=lambda s: len(str(s.get('name', ''))))
            
            # Create clean stock entry
            clean_stock = {
                "symbol": symbol,
                "ticker": best_stock.get('ticker', f"{symbol}.IS"),
                "name": best_stock.get('name', symbol),
                "type": best_stock.get('type', 'Common Stock'),
                "currency": best_stock.get('currency', 'TRY'),
                "country": best_stock.get('country', 'Turkey'),
                "exchange": best_stock.get('exchange', 'IS'),
                "market": "BIST",
                "sector": "Unknown",  # Will be mapped later
                "region": "turkey",
                "search_text": self.create_search_text(symbol, best_stock.get('name', symbol))
            }
            
            clean_stocks.append(clean_stock)
        
        print(f"✅ Clean database created: {len(clean_stocks)} unique stocks")
        return clean_stocks
    
    def create_search_text(self, symbol, name):
        """Create comprehensive search text"""
        components = [
            symbol.lower(),
            name.lower(),
            'turkey',
            'bist'
        ]
        
        # Add name words
        name_words = name.lower().split()
        components.extend(name_words)
        
        # Add symbol substrings
        if len(symbol) >= 3:
            for i in range(1, len(symbol)):
                components.append(symbol[:i].lower())
                components.append(symbol[i:].lower())
        
        return ' '.join(set(components))
    
    def save_exhaustive_database(self, stocks):
        """Save exhaustive clean database"""
        print(f"\n💾 SAVING EXHAUSTIVE DATABASE")
        print("=" * 30)
        
        database = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_stocks": len(stocks),
                "source": "Profit.com API - Exhaustive Fetch",
                "version": "3.0-exhaustive-clean",
                "description": "Complete Turkish stocks with duplicates removed",
                "fetch_method": "Multiple approaches with deduplication"
            },
            "all_turkish_stocks": stocks
        }
        
        filename = 'exhaustive_turkish_stocks_clean.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(database, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved: {filename}")
        print(f"📊 Stocks: {len(stocks)}")
        
        # Show sample
        print(f"\n📋 SAMPLE (first 20):")
        print("-" * 40)
        for i, stock in enumerate(stocks[:20], 1):
            symbol = stock['symbol']
            name = stock['name'][:30]
            print(f"{i:2d}. {symbol:8} | {name}")
        
        if len(stocks) > 20:
            print(f"... ve {len(stocks) - 20} tane daha!")
        
        return filename
    
    def run_exhaustive_fetch(self):
        """Run complete exhaustive fetch"""
        print("🚀 EXHAUSTIVE TURKISH STOCKS FETCHER")
        print("=" * 60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Objective: Get EVERY Turkish stock, no duplicates")
        print()
        
        # Step 1: Exhaustive fetch
        all_stocks = self.fetch_all_stocks_exhaustive()
        
        if not all_stocks:
            print("❌ No stocks fetched!")
            return None
        
        # Step 2: Clean and deduplicate
        clean_stocks = self.create_clean_database(all_stocks)
        
        # Step 3: Save
        filename = self.save_exhaustive_database(clean_stocks)
        
        print(f"\n🎊 EXHAUSTIVE FETCH SUCCESS!")
        print(f"📁 File: {filename}")
        print(f"📊 Unique stocks: {len(clean_stocks)}")
        
        return clean_stocks

if __name__ == "__main__":
    fetcher = ExhaustiveAPIFetcher()
    result = fetcher.run_exhaustive_fetch()
