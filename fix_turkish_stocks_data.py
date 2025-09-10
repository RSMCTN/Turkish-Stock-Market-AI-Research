#!/usr/bin/env python3
"""
Fix Turkish Stocks Data Loss
Restore full Turkish stocks database and merge with global data
"""

import json
from datetime import datetime

def fix_turkish_stocks_data():
    """Fix the Turkish stocks data loss and create proper global database"""
    
    print("🔧 TURKISH STOCKS DATA FIX")
    print("=" * 40)
    
    # Load original profit search data (100 Turkish stocks)
    try:
        with open('profit_search_data.json', 'r', encoding='utf-8') as f:
            original_turkish = json.load(f)
        print(f"✅ Original Turkish data loaded: {len(original_turkish)} stocks")
    except Exception as e:
        print(f"❌ Error loading original data: {e}")
        return
    
    # Prepare Turkish stocks for global database
    turkish_stocks = []
    for stock in original_turkish:
        turkish_stock = {
            "symbol": stock.get("symbol", ""),
            "name": stock.get("name", ""),
            "sector": stock.get("sector", "Unknown"),
            "market": stock.get("market", "BIST"),
            "currency": stock.get("currency", "TRY"),
            "ticker": stock.get("ticker", f"{stock.get('symbol', '')}.IS"),
            "region": "turkey"
        }
        turkish_stocks.append(turkish_stock)
    
    print(f"✅ Prepared {len(turkish_stocks)} Turkish stocks for global database")
    
    # Other global markets (keep existing)
    usa_stocks = [
        {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "AAPL", "region": "usa"},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "MSFT", "region": "usa"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "GOOGL", "region": "usa"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "E-commerce", "market": "NASDAQ", "currency": "USD", "ticker": "AMZN", "region": "usa"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive", "market": "NASDAQ", "currency": "USD", "ticker": "TSLA", "region": "usa"},
        {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "META", "region": "usa"},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "NVDA", "region": "usa"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Banking", "market": "NYSE", "currency": "USD", "ticker": "JPM", "region": "usa"},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "market": "NYSE", "currency": "USD", "ticker": "JNJ", "region": "usa"},
        {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services", "market": "NYSE", "currency": "USD", "ticker": "V", "region": "usa"},
    ]
    
    europe_stocks = [
        {"symbol": "ASML", "name": "ASML Holding N.V.", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "ASML", "region": "europe"},
        {"symbol": "NESN", "name": "Nestlé S.A.", "sector": "Consumer Goods", "market": "SIX", "currency": "CHF", "ticker": "NESN.SW", "region": "europe"},
        {"symbol": "SAP", "name": "SAP SE", "sector": "Technology", "market": "XETRA", "currency": "EUR", "ticker": "SAP.DE", "region": "europe"},
        {"symbol": "SHELL", "name": "Shell plc", "sector": "Energy", "market": "LSE", "currency": "GBP", "ticker": "SHEL.L", "region": "europe"},
    ]
    
    asia_stocks = [
        {"symbol": "TSM", "name": "Taiwan Semiconductor Manufacturing Company Limited", "sector": "Technology", "market": "NYSE", "currency": "USD", "ticker": "TSM", "region": "asia"},
        {"symbol": "BABA", "name": "Alibaba Group Holding Limited", "sector": "E-commerce", "market": "NYSE", "currency": "USD", "ticker": "BABA", "region": "asia"},
        {"symbol": "TM", "name": "Toyota Motor Corporation", "sector": "Automotive", "market": "NYSE", "currency": "USD", "ticker": "TM", "region": "asia"},
    ]
    
    # Create comprehensive database
    global_stocks = {
        "turkey": turkish_stocks,
        "usa": usa_stocks, 
        "europe": europe_stocks,
        "asia": asia_stocks
    }
    
    # Create searchable format
    all_stocks = []
    for region, stocks in global_stocks.items():
        for stock in stocks:
            stock_entry = {
                **stock,
                "search_text": f"{stock['symbol']} {stock['name']} {stock['sector']} {region}".lower()
            }
            all_stocks.append(stock_entry)
    
    # Create final output
    output_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "total_stocks": len(all_stocks),
            "regions": list(global_stocks.keys()),
            "version": "1.1-fixed",
            "turkish_stocks_restored": len(turkish_stocks)
        },
        "stocks_by_region": global_stocks,
        "all_stocks": all_stocks
    }
    
    # Save fixed database
    with open('global_stocks_data_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ FIXED DATABASE CREATED!")
    print(f"📊 Total stocks: {len(all_stocks)}")
    print(f"🇹🇷 Turkish stocks: {len(turkish_stocks)} (RESTORED!)")
    print(f"🇺🇸 USA stocks: {len(usa_stocks)}")
    print(f"🇪🇺 Europe stocks: {len(europe_stocks)}")
    print(f"🌏 Asia stocks: {len(asia_stocks)}")
    
    # Show some Turkish samples
    print(f"\n📋 TURKISH STOCKS SAMPLE:")
    for i, stock in enumerate(turkish_stocks[:10], 1):
        print(f"   {i:2d}. {stock['symbol']:6} | {stock['name'][:35]:35} | {stock['sector'][:15]}")
    print(f"   ... ve {len(turkish_stocks)-10} tane daha!")
    
    return output_data

if __name__ == "__main__":
    fix_turkish_stocks_data()
