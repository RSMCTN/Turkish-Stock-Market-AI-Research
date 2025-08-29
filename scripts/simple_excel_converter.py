#!/usr/bin/env python3
"""Simple working Excel converter"""

import pandas as pd
import json

def simple_excel_converter():
    excel_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/basestock2808.xlsx"
    
    print(f"📖 Reading Excel file...")
    df = pd.read_excel(excel_path)
    print(f"✅ Loaded {len(df)} rows")
    
    stocks = []
    sectors = {}
    
    print(f"🔄 Processing rows...")
    processed = 0
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f"   Processing row {i}...")
            
        try:
            symbol = str(row.get('SEMBOL', '')).strip()
            if not symbol or symbol == 'nan':
                continue
                
            # Basic stock data
            stock = {
                "symbol": symbol,
                "name": str(row.get('ACKL', '')).strip(),
                "name_turkish": str(row.get('ACKL', '')).strip(),
                "sector": str(row.get('SEKTOR', '')).strip(),
                "sector_turkish": str(row.get('SEKTOR', '')).strip(),
                "last_price": float(row.get('SON', 0)) if pd.notna(row.get('SON')) else 0,
                "change": float(row.get('FARK', 0)) if pd.notna(row.get('FARK')) else 0,
                "change_percent": float(row.get('%FARK', 0)) if pd.notna(row.get('%FARK')) else 0,
                "volume": int(row.get('T.HACİM', 0)) if pd.notna(row.get('T.HACİM')) else 0,
                "market_cap": float(row.get('PIY.DEG', 0)) if pd.notna(row.get('PIY.DEG')) else 0,
                "pe_ratio": float(row.get('F/K', 0)) if pd.notna(row.get('F/K')) else 0,
                "pb_ratio": float(row.get('PD/DD', 0)) if pd.notna(row.get('PD/DD')) else 0,
                # BIST markets based on XU indices
                "bist_markets": [],
                "xu030_member": bool(row.get('XU030 DAKI AG.', 0)) if pd.notna(row.get('XU030 DAKI AG.')) else False,
                "xu050_member": bool(row.get('XU050 DEKI AG.', 0)) if pd.notna(row.get('XU050 DEKI AG.')) else False,
                "xu100_member": bool(row.get('XU100 DEKI AG.', 0)) if pd.notna(row.get('XU100 DEKI AG.')) else False,
            }
            
            # Determine BIST markets
            markets = []
            if stock["xu030_member"]:
                markets.append("bist_30")
            if stock["xu100_member"]:
                markets.append("bist_100")
            if stock["xu050_member"]:
                markets.append("bist_50")
            if stock["volume"] > 0:
                markets.append("bist_all")
                
            stock["bist_markets"] = markets
            
            # Update sectors
            sector = stock["sector"]
            if sector and sector not in sectors:
                sectors[sector] = {
                    "name": sector,
                    "name_turkish": sector,
                    "companies": []
                }
            if sector:
                sectors[sector]["companies"].append({
                    "symbol": symbol,
                    "name": stock["name_turkish"]
                })
            
            stocks.append(stock)
            processed += 1
            
        except Exception as e:
            print(f"⚠️ Error processing row {i}: {e}")
    
    print(f"✅ Processed {processed} stocks")
    
    # Save to JSON
    output = {
        "stocks": stocks,
        "sectors": sectors,
        "metadata": {
            "total_stocks": len(stocks),
            "total_sectors": len(sectors),
            "bist_30_count": len([s for s in stocks if "bist_30" in s["bist_markets"]]),
            "bist_100_count": len([s for s in stocks if "bist_100" in s["bist_markets"]])
        }
    }
    
    json_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/working_bist_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved to: {json_path}")
    print(f"📊 Summary: {len(stocks)} stocks, {len(sectors)} sectors")
    print(f"   BIST 30: {output['metadata']['bist_30_count']}")
    print(f"   BIST 100: {output['metadata']['bist_100_count']}")
    
    # Sample data
    sample_stocks = ['GARAN', 'ASTOR', 'THYAO', 'TUPRS', 'BRSAN']
    print(f"\n🎯 SAMPLE DATA:")
    for symbol in sample_stocks:
        stock = next((s for s in stocks if s["symbol"] == symbol), None)
        if stock:
            print(f"   {symbol}: Price: {stock['last_price']}, Markets: {stock['bist_markets']}")
    
    return stocks, sectors

if __name__ == "__main__":
    simple_excel_converter()
