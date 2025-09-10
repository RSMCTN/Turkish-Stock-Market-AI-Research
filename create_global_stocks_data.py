#!/usr/bin/env python3
"""
Global Markets Stock Database Creator
Create comprehensive stock database for all markets
"""

import json
from datetime import datetime

def create_global_stocks_database():
    """Create comprehensive global stock database"""
    
    global_stocks = {
        "turkey": [
            {"symbol": "AKBNK", "name": "Akbank T.A.Ş.", "sector": "Bankacılık", "market": "BIST", "currency": "TRY", "ticker": "AKBNK.IS"},
            {"symbol": "GARAN", "name": "Türkiye Garanti Bankası A.Ş.", "sector": "Bankacılık", "market": "BIST", "currency": "TRY", "ticker": "GARAN.IS"},
            {"symbol": "ISCTR", "name": "Türkiye İş Bankası A.Ş.", "sector": "Bankacılık", "market": "BIST", "currency": "TRY", "ticker": "ISCTR.IS"},
            {"symbol": "YKBNK", "name": "Yapı ve Kredi Bankası A.Ş.", "sector": "Bankacılık", "market": "BIST", "currency": "TRY", "ticker": "YKBNK.IS"},
            {"symbol": "VAKBN", "name": "Türkiye Vakıflar Bankası T.A.O.", "sector": "Bankacılık", "market": "BIST", "currency": "TRY", "ticker": "VAKBN.IS"},
            {"symbol": "TUPRS", "name": "Tüpraş-Türkiye Petrol Rafinerileri A.Ş.", "sector": "Petrol & Kimya", "market": "BIST", "currency": "TRY", "ticker": "TUPRS.IS"},
            {"symbol": "THYAO", "name": "Türk Hava Yolları A.O.", "sector": "Taşımacılık", "market": "BIST", "currency": "TRY", "ticker": "THYAO.IS"},
            {"symbol": "TCELL", "name": "Turkcell İletişim Hizmetleri A.Ş.", "sector": "Telekomünikasyon", "market": "BIST", "currency": "TRY", "ticker": "TCELL.IS"},
            {"symbol": "ASELS", "name": "Aselsan Elektronik Sanayi ve Ticaret A.Ş.", "sector": "Teknoloji", "market": "BIST", "currency": "TRY", "ticker": "ASELS.IS"},
            {"symbol": "KCHOL", "name": "Koç Holding A.Ş.", "sector": "Holding", "market": "BIST", "currency": "TRY", "ticker": "KCHOL.IS"},
            {"symbol": "SAHOL", "name": "Sabancı Holding A.Ş.", "sector": "Holding", "market": "BIST", "currency": "TRY", "ticker": "SAHOL.IS"},
            {"symbol": "ARCLK", "name": "Arçelik A.Ş.", "sector": "Ev Aletleri", "market": "BIST", "currency": "TRY", "ticker": "ARCLK.IS"},
            {"symbol": "BIMAS", "name": "BİM Birleşik Mağazalar A.Ş.", "sector": "Perakende", "market": "BIST", "currency": "TRY", "ticker": "BIMAS.IS"},
            {"symbol": "EREGL", "name": "Ereğli Demir ve Çelik Fabrikaları T.A.Ş.", "sector": "Metal", "market": "BIST", "currency": "TRY", "ticker": "EREGL.IS"},
            {"symbol": "KOZAL", "name": "Koza Altın İşletmeleri A.Ş.", "sector": "Madencilik", "market": "BIST", "currency": "TRY", "ticker": "KOZAL.IS"}
        ],
        
        "usa": [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "AAPL"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "MSFT"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "GOOGL"},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "E-commerce", "market": "NASDAQ", "currency": "USD", "ticker": "AMZN"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive", "market": "NASDAQ", "currency": "USD", "ticker": "TSLA"},
            {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "META"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "NVDA"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Banking", "market": "NYSE", "currency": "USD", "ticker": "JPM"},
            {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "market": "NYSE", "currency": "USD", "ticker": "JNJ"},
            {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services", "market": "NYSE", "currency": "USD", "ticker": "V"},
            {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Healthcare", "market": "NYSE", "currency": "USD", "ticker": "UNH"},
            {"symbol": "HD", "name": "The Home Depot Inc.", "sector": "Retail", "market": "NYSE", "currency": "USD", "ticker": "HD"}
        ],
        
        "europe": [
            {"symbol": "ASML", "name": "ASML Holding N.V.", "sector": "Technology", "market": "NASDAQ", "currency": "USD", "ticker": "ASML"},
            {"symbol": "NESN", "name": "Nestlé S.A.", "sector": "Consumer Goods", "market": "SIX", "currency": "CHF", "ticker": "NESN.SW"},
            {"symbol": "NOVN", "name": "Novartis AG", "sector": "Healthcare", "market": "SIX", "currency": "CHF", "ticker": "NOVN.SW"},
            {"symbol": "SAP", "name": "SAP SE", "sector": "Technology", "market": "XETRA", "currency": "EUR", "ticker": "SAP.DE"},
            {"symbol": "SHELL", "name": "Shell plc", "sector": "Energy", "market": "LSE", "currency": "GBP", "ticker": "SHEL.L"},
            {"symbol": "ASTRAZENECA", "name": "AstraZeneca PLC", "sector": "Healthcare", "market": "LSE", "currency": "GBP", "ticker": "AZN.L"},
            {"symbol": "LVMH", "name": "LVMH Moët Hennessy Louis Vuitton", "sector": "Luxury", "market": "EURONEXT", "currency": "EUR", "ticker": "MC.PA"},
            {"symbol": "TOTALENERGIES", "name": "TotalEnergies SE", "sector": "Energy", "market": "EURONEXT", "currency": "EUR", "ticker": "TTE.PA"}
        ],
        
        "asia": [
            {"symbol": "TSM", "name": "Taiwan Semiconductor Manufacturing Company Limited", "sector": "Technology", "market": "NYSE", "currency": "USD", "ticker": "TSM"},
            {"symbol": "TCEHY", "name": "Tencent Holdings Limited", "sector": "Technology", "market": "OTC", "currency": "USD", "ticker": "TCEHY"},
            {"symbol": "BABA", "name": "Alibaba Group Holding Limited", "sector": "E-commerce", "market": "NYSE", "currency": "USD", "ticker": "BABA"},
            {"symbol": "TM", "name": "Toyota Motor Corporation", "sector": "Automotive", "market": "NYSE", "currency": "USD", "ticker": "TM"},
            {"symbol": "SONY", "name": "Sony Group Corporation", "sector": "Technology", "market": "NYSE", "currency": "USD", "ticker": "SONY"},
            {"symbol": "NVO", "name": "Novo Nordisk A/S", "sector": "Healthcare", "market": "NYSE", "currency": "USD", "ticker": "NVO"}
        ]
    }
    
    # Create searchable format
    all_stocks = []
    for market_region, stocks in global_stocks.items():
        for stock in stocks:
            stock_entry = {
                **stock,
                "region": market_region,
                "search_text": f"{stock['symbol']} {stock['name']} {stock['sector']} {market_region}".lower()
            }
            all_stocks.append(stock_entry)
    
    # Save to JSON
    output_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "total_stocks": len(all_stocks),
            "regions": list(global_stocks.keys()),
            "version": "1.0"
        },
        "stocks_by_region": global_stocks,
        "all_stocks": all_stocks
    }
    
    with open('global_stocks_data.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Global stocks database created!")
    print(f"📊 Total stocks: {len(all_stocks)}")
    print(f"🌍 Regions: {', '.join(global_stocks.keys())}")
    
    # Show sample by region
    for region, stocks in global_stocks.items():
        print(f"\n📍 {region.upper()} ({len(stocks)} stocks):")
        for i, stock in enumerate(stocks[:3], 1):
            print(f"   {i}. {stock['symbol']:6} | {stock['name'][:30]:30} | {stock['currency']}")
        if len(stocks) > 3:
            print(f"   ... and {len(stocks)-3} more")
    
    return output_data

if __name__ == "__main__":
    print("🌍 GLOBAL STOCKS DATABASE CREATOR")
    print("=" * 40)
    create_global_stocks_database()
