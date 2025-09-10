#!/usr/bin/env python3
"""
Add missing stocks to the database
Found stocks via individual quote endpoint: ZOREN, ONRYT, GLCVY etc.
"""

import json
import requests
from datetime import datetime

PROFIT_API_KEY = "a9a0bacbab08493d958244c05380da01"

def get_stock_details(symbol):
    """Get detailed stock info from individual quote endpoint"""
    ticker = f"{symbol}.IS"
    url = f"https://api.profit.com/data-api/market-data/quote/{ticker}"
    
    try:
        resp = requests.get(url, params={'token': PROFIT_API_KEY}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def add_missing_stocks_to_database():
    """Add missing stocks found via individual queries to the main database"""
    
    print("🔧 MISSING STOCKS DATABASE UPDATE")
    print("=" * 35)
    
    # Load current database
    try:
        with open('complete_turkish_stocks.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        current_stocks = data['all_turkish_stocks']
        print(f"📊 Current stocks: {len(current_stocks)}")
    except:
        print("❌ No current database found")
        return
    
    # Get current symbols
    current_symbols = set()
    for stock in current_stocks:
        symbol = stock.get('symbol', '').upper().strip()
        if symbol:
            current_symbols.add(symbol)
    
    print(f"📊 Current unique symbols: {len(current_symbols)}")
    
    # Missing stocks that we found working
    found_missing = [
        'ZOREN', 'ONRYT', 'GLCVY', 'THYAO', 'CCOLA', 'MGROS', 'KOZAL', 'XU030'
    ]
    
    added_stocks = []
    
    for symbol in found_missing:
        if symbol in current_symbols:
            print(f"✅ {symbol} already in database")
            continue
            
        print(f"🔍 Fetching details for {symbol}...")
        stock_data = get_stock_details(symbol)
        
        if stock_data:
            # Create standardized stock entry
            stock_entry = {
                "symbol": symbol,
                "ticker": f"{symbol}.IS", 
                "name": stock_data.get('name', symbol),
                "sector": "Unknown",  
                "market": "BIST",
                "currency": stock_data.get('currency', 'TRY'),
                "region": "turkey",
                "search_text": f"{symbol} {stock_data.get('name', '')} turkey".lower(),
                "original_data": stock_data
            }
            
            added_stocks.append(stock_entry)
            current_symbols.add(symbol)
            
            price = stock_data.get('price', 'N/A')
            name = stock_data.get('name', 'N/A')[:30]
            print(f"  ✅ {symbol}: {price} - {name}")
        else:
            print(f"  ❌ {symbol}: Failed to fetch")
    
    if added_stocks:
        # Add new stocks to database
        updated_stocks = current_stocks + added_stocks
        
        # Update database
        updated_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_stocks": len(updated_stocks),
                "source": "Profit.com API - Complete + Missing Stocks",
                "version": "2.1-with-missing",
                "added_missing": len(added_stocks)
            },
            "all_turkish_stocks": updated_stocks
        }
        
        # Save updated database
        filename = 'complete_turkish_stocks_with_missing.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Database updated!")
        print(f"📊 Added stocks: {len(added_stocks)}")
        print(f"📊 Total stocks: {len(updated_stocks)}")
        print(f"📊 Unique symbols: {len(current_symbols)}")
        print(f"💾 Saved to: {filename}")
        
        # Show added stocks
        print(f"\n📋 ADDED STOCKS:")
        for stock in added_stocks:
            print(f"  {stock['symbol']:8} | {stock['name'][:30]:30}")
            
        return filename
    else:
        print("⚠️  No new stocks added")
        return None

def update_global_dashboard(database_file):
    """Update global dashboard with new stocks"""
    if not database_file:
        return
        
    print(f"\n🌍 UPDATING GLOBAL DASHBOARD")
    print("=" * 28)
    
    try:
        # Load updated database
        with open(database_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        turkish_stocks = data['all_turkish_stocks']
        
        # Load current global dashboard data
        global_file = "global-dashboard/public/global_stocks_data.json"
        try:
            with open(global_file, 'r', encoding='utf-8') as f:
                global_data = json.load(f)
        except:
            global_data = []
        
        # Remove old Turkish stocks
        non_turkish_stocks = [s for s in global_data 
                             if s.get('market', '').lower() != 'turkey']
        
        print(f"📊 Non-Turkish stocks: {len(non_turkish_stocks)}")
        
        # Add updated Turkish stocks
        for stock in turkish_stocks:
            stock_for_global = {
                "symbol": stock.get('symbol', ''),
                "name": stock.get('name', ''),
                "market": "Turkey",
                "search_text": f"{stock.get('symbol', '')} {stock.get('name', '')} Turkey BIST".lower()
            }
            non_turkish_stocks.append(stock_for_global)
        
        # Save updated global data
        with open(global_file, 'w', encoding='utf-8') as f:
            json.dump(non_turkish_stocks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Global dashboard updated!")
        print(f"📊 Total global stocks: {len(non_turkish_stocks)}")
        print(f"📊 Turkish stocks: {len(turkish_stocks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Global dashboard update failed: {e}")
        return False

def main():
    """Main function"""
    print("🎯 ADD MISSING STOCKS TO DATABASE")
    print("=" * 38)
    
    # Add missing stocks
    database_file = add_missing_stocks_to_database()
    
    if database_file:
        # Update global dashboard
        update_global_dashboard(database_file)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Missing stocks added to database")
        print(f"✅ Global dashboard updated")
        print(f"🚀 Dashboard ready with missing stocks!")
    else:
        print(f"\n⚠️  No updates made")

if __name__ == "__main__":
    main()
