#!/usr/bin/env python3
"""
Fetch ALL Turkish Stocks from Profit.com API
Get complete Turkish stocks list (600+)
"""

import requests
import json
import time
from datetime import datetime

class CompleteTurkishStocksFetcher:
    def __init__(self):
        self.api_key = "a9a0bacbab08493d958244c05380da01"
        self.base_url = "https://api.profit.com"
    
    def fetch_all_turkish_stocks(self):
        """Fetch ALL Turkish stocks with pagination"""
        print("🔍 FETCHING ALL TURKISH STOCKS")
        print("=" * 40)
        
        all_stocks = []
        page = 0
        limit = 100
        total_fetched = 0
        
        while True:
            print(f"📄 Fetching page {page + 1} (limit: {limit}, offset: {page * limit})")
            
            url = f"{self.base_url}/data-api/reference/stocks"
            params = {
                'token': self.api_key,
                'country': 'Turkey',
                'limit': limit,
                'offset': page * limit
            }
            
            try:
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code != 200:
                    print(f"❌ API Error: {response.status_code}")
                    break
                
                data = response.json()
                stocks_batch = data.get('data', [])
                total_in_api = data.get('total', 0)
                
                print(f"✅ Received {len(stocks_batch)} stocks in this batch")
                print(f"📊 Total in API: {total_in_api}")
                
                if not stocks_batch:
                    print("✅ No more stocks to fetch")
                    break
                
                all_stocks.extend(stocks_batch)
                total_fetched += len(stocks_batch)
                
                print(f"📈 Total fetched so far: {total_fetched}")
                
                # If we got less than limit, we're done
                if len(stocks_batch) < limit:
                    print("✅ Reached end of data")
                    break
                
                # If we have the total count and we've fetched enough
                if total_in_api > 0 and total_fetched >= total_in_api:
                    print("✅ Fetched all available stocks")
                    break
                
                page += 1
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error on page {page + 1}: {e}")
                break
        
        print(f"\n🎉 FETCH COMPLETE!")
        print(f"📊 Total Turkish stocks fetched: {len(all_stocks)}")
        
        return all_stocks
    
    def create_searchable_database(self, stocks):
        """Create searchable database from all stocks"""
        print(f"\n🔧 CREATING SEARCHABLE DATABASE")
        print("=" * 35)
        
        searchable_stocks = []
        
        for stock in stocks:
            # Extract available fields
            symbol = stock.get('symbol', stock.get('ticker', ''))
            name = stock.get('name', '')
            currency = stock.get('currency', 'TRY')
            exchange = stock.get('exchange', 'IS')
            
            # Create ticker
            ticker = stock.get('ticker', f"{symbol}.{exchange}")
            if not ticker.endswith('.IS'):
                ticker = f"{symbol}.IS"
            
            # Create searchable entry
            searchable_entry = {
                "symbol": symbol,
                "ticker": ticker,
                "name": name,
                "sector": "Unknown",  # Will be filled later
                "market": "BIST",
                "currency": currency,
                "region": "turkey",
                "search_text": f"{symbol} {name} turkey".lower(),
                # Keep original data for reference
                "original_data": stock
            }
            
            searchable_stocks.append(searchable_entry)
        
        print(f"✅ Created {len(searchable_stocks)} searchable entries")
        
        return searchable_stocks
    
    def save_complete_database(self, stocks):
        """Save complete database"""
        print(f"\n💾 SAVING COMPLETE DATABASE")
        print("=" * 30)
        
        # Create complete database structure
        database = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_stocks": len(stocks),
                "source": "Profit.com API - Complete Fetch",
                "version": "2.0-complete"
            },
            "all_turkish_stocks": stocks
        }
        
        # Save to JSON
        filename = 'complete_turkish_stocks.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(database, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved to: {filename}")
        
        # Show sample
        print(f"\n📋 SAMPLE STOCKS:")
        print("-" * 50)
        
        for i, stock in enumerate(stocks[:20], 1):
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:30]
            print(f"{i:2d}. {symbol:8} | {name:30}")
        
        if len(stocks) > 20:
            print(f"... ve {len(stocks) - 20} tane daha!")
        
        return filename
    
    def run_complete_fetch(self):
        """Run complete fetch process"""
        print("🚀 COMPLETE TURKISH STOCKS FETCHER")
        print("=" * 50)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Fetch all stocks
        all_stocks = self.fetch_all_turkish_stocks()
        
        if not all_stocks:
            print("❌ No stocks fetched!")
            return None
        
        # Step 2: Create searchable database  
        searchable_stocks = self.create_searchable_database(all_stocks)
        
        # Step 3: Save to file
        filename = self.save_complete_database(searchable_stocks)
        
        print(f"\n🎊 SUCCESS!")
        print(f"📊 Total Turkish stocks: {len(all_stocks)}")
        print(f"📁 Saved to: {filename}")
        
        return searchable_stocks

if __name__ == "__main__":
    fetcher = CompleteTurkishStocksFetcher()
    result = fetcher.run_complete_fetch()
