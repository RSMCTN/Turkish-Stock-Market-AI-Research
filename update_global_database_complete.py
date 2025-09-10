#!/usr/bin/env python3
"""
Update Global Database with Complete Turkish Stocks (700)
Merge 700 Turkish stocks with global markets
"""

import json
from datetime import datetime

def update_global_database_complete():
    """Update global database with complete 700 Turkish stocks"""
    
    print("🔄 UPDATING GLOBAL DATABASE WITH 700 TURKISH STOCKS")
    print("=" * 60)
    
    # Load complete Turkish stocks
    try:
        with open('complete_turkish_stocks.json', 'r', encoding='utf-8') as f:
            turkish_data = json.load(f)
        
        turkish_stocks = turkish_data['all_turkish_stocks']
        print(f"✅ Loaded {len(turkish_stocks)} Turkish stocks")
        
    except Exception as e:
        print(f"❌ Error loading Turkish stocks: {e}")
        return
    
    # Global markets (keep existing)
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
        {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Healthcare", "market": "NYSE", "currency": "USD", "ticker": "UNH", "region": "usa"},
        {"symbol": "HD", "name": "The Home Depot Inc.", "sector": "Retail", "market": "NYSE", "currency": "USD", "ticker": "HD", "region": "usa"}
    ]
    
    europe_stocks = [
        {"symbol": "ASML", "name": "ASML Holding N.V.", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "ASML", "region": "europe"},
        {"symbol": "NESN", "name": "Nestlé S.A.", "sector": "Consumer Goods", "market": "SIX", "currency": "CHF", "ticker": "NESN.SW", "region": "europe"},
        {"symbol": "NOVN", "name": "Novartis AG", "sector": "Healthcare", "market": "SIX", "currency": "CHF", "ticker": "NOVN.SW", "region": "europe"},
        {"symbol": "SAP", "name": "SAP SE", "sector": "Technology", "market": "XETRA", "currency": "EUR", "ticker": "SAP.DE", "region": "europe"},
        {"symbol": "SHELL", "name": "Shell plc", "sector": "Energy", "market": "LSE", "currency": "GBP", "ticker": "SHEL.L", "region": "europe"},
        {"symbol": "ASTRAZENECA", "name": "AstraZeneca PLC", "sector": "Healthcare", "market": "LSE", "currency": "GBP", "ticker": "AZN.L", "region": "europe"}
    ]
    
    asia_stocks = [
        {"symbol": "TSM", "name": "Taiwan Semiconductor Manufacturing Company Limited", "sector": "Technology", "market": "NYSE", "currency": "USD", "ticker": "TSM", "region": "asia"},
        {"symbol": "TCEHY", "name": "Tencent Holdings Limited", "sector": "Technology", "market": "OTC", "currency": "USD", "ticker": "TCEHY", "region": "asia"},
        {"symbol": "BABA", "name": "Alibaba Group Holding Limited", "sector": "E-commerce", "market": "NYSE", "currency": "USD", "ticker": "BABA", "region": "asia"},
        {"symbol": "TM", "name": "Toyota Motor Corporation", "sector": "Automotive", "market": "NYSE", "currency": "USD", "ticker": "TM", "region": "asia"},
        {"symbol": "SONY", "name": "Sony Group Corporation", "sector": "Technology", "market": "NYSE", "currency": "USD", "ticker": "SONY", "region": "asia"}
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
                "search_text": f"{stock['symbol']} {stock['name']} {stock.get('sector', '')} {region}".lower()
            }
            all_stocks.append(stock_entry)
    
    # Create final output
    output_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "total_stocks": len(all_stocks),
            "regions": list(global_stocks.keys()),
            "version": "2.0-complete",
            "turkish_stocks_complete": len(turkish_stocks),
            "breakdown": {
                "turkey": len(turkish_stocks),
                "usa": len(usa_stocks),
                "europe": len(europe_stocks), 
                "asia": len(asia_stocks)
            }
        },
        "stocks_by_region": global_stocks,
        "all_stocks": all_stocks
    }
    
    # Save updated database
    filename = 'global_stocks_data_complete.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎊 UPDATED GLOBAL DATABASE!")
    print(f"📊 Total stocks: {len(all_stocks)}")
    print(f"🇹🇷 Turkish stocks: {len(turkish_stocks)} (COMPLETE!)")
    print(f"🇺🇸 USA stocks: {len(usa_stocks)}")
    print(f"🇪🇺 Europe stocks: {len(europe_stocks)}")
    print(f"🌏 Asia stocks: {len(asia_stocks)}")
    print(f"📁 Saved to: {filename}")
    
    return output_data

if __name__ == "__main__":
    update_global_database_complete()
