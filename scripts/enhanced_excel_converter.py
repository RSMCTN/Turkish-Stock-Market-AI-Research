#!/usr/bin/env python3
"""Enhanced Excel to data converter for comprehensive BIST stock data"""

import pandas as pd
import json
import os
from pathlib import Path

def enhanced_excel_converter():
    """Convert basestock2808.xlsx to comprehensive format with all fields"""
    
    excel_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/basestock2808.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"❌ Excel file not found: {excel_path}")
        return
    
    try:
        # Read Excel file
        print(f"📖 Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        
        print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Enhanced stock data with all available fields
        enhanced_stocks = []
        sector_mapping = {}
        
        for _, row in df.iterrows():
            try:
                # Try different symbol column names (handle None values)
                borsasembolu = row.get('BORSASEMBOLU', '') or ''
                sembol = row.get('SEMBOL', '') or ''
                symbol_col = row.get('SYMBOL', '') or ''
                
                symbol = str(borsasembolu or sembol or symbol_col).strip()
                if not symbol or symbol == 'nan':
                    continue
                
                # Extract comprehensive data using actual Excel column names
                stock_data = {
                    # Basic Info
                    "symbol": symbol,
                    "name": row.get('ACKL', '').strip(),
                    "name_turkish": row.get('ACKL', '').strip(),
                    
                    # Sector & Classification  
                    "sector": row.get('SEKTOR', '').strip(),
                    "sector_turkish": row.get('SEKTOR', '').strip(),
                    
                    # Market Classification (from Excel data)
                    "bist_markets": [],  # Will be populated based on criteria
                    "market_segment": "Main",  # Default
                    
                    # Price Data
                    "last_price": float(row.get('SON', 0)) if pd.notna(row.get('SON', 0)) else 0,
                    "change": float(row.get('FARK', 0)) if pd.notna(row.get('FARK', 0)) else 0,
                    "change_percent": float(row.get('%FARK', 0)) if pd.notna(row.get('%FARK', 0)) else 0,
                    
                    # Volume & Trading
                    "volume": int(row.get('T.HACİM', 0)) if pd.notna(row.get('T.HACİM', 0)) else 0,
                    "volume_tl": float(row.get('G.HACIM', 0)) if pd.notna(row.get('G.HACIM', 0)) else 0,
                    
                    # Market Data
                    "market_cap": float(row.get('PIY.DEG', 0)) if pd.notna(row.get('PIY.DEG', 0)) else 0,
                    "shares_outstanding": int(row.get('OD.SER', 0)) if pd.notna(row.get('OD.SER', 0)) else 0,
                    
                    # Price Ranges
                    "week_52_high": float(row.get('52HAFTALIK.YUKSEK', 0)) if pd.notna(row.get('52HAFTALIK.YUKSEK', 0)) else 0,
                    "week_52_low": float(row.get('52HAFTALIK.DUSUK', 0)) if pd.notna(row.get('52HAFTALIK.DUSUK', 0)) else 0,
                    "daily_high": float(row.get('YÜKSEK', 0)) if pd.notna(row.get('YÜKSEK', 0)) else 0,
                    "daily_low": float(row.get('DÜŞÜK', 0)) if pd.notna(row.get('DÜŞÜK', 0)) else 0,
                    
                    # Financial Ratios
                    "pe_ratio": float(row.get('F/K', 0)) if pd.notna(row.get('F/K', 0)) else 0,
                    "pb_ratio": float(row.get('PD/DD', 0)) if pd.notna(row.get('PD/DD', 0)) else 0,
                    
                    # Additional Financial Data
                    "net_debt": float(row.get('NETBORC', 0)) if pd.notna(row.get('NETBORC', 0)) else 0,
                    
                    # Performance Metrics (calculated from available data)
                    "week_performance": float(row.get('BU.HAFTA.FARK%', 0)) if pd.notna(row.get('BU.HAFTA.FARK%', 0)) else 0,
                    "month_performance": float(row.get('BU.AY.FARK%', 0)) if pd.notna(row.get('BU.AY.FARK%', 0)) else 0,
                    "month3_performance": float(row.get('30GUNLUK.F%', 0)) if pd.notna(row.get('30GUNLUK.F%', 0)) else 0,
                    "month6_performance": float(row.get('BU.YIL.FARK%', 0)) if pd.notna(row.get('BU.YIL.FARK%', 0)) else 0,
                    "year_performance": float(row.get('BU.YIL.FARK%', 0)) if pd.notna(row.get('BU.YIL.FARK%', 0)) else 0,
                    
                    # BIST Index Memberships (from Excel)
                    "xu030_member": row.get('XU030 DAKI AG.', 0) > 0 if pd.notna(row.get('XU030 DAKI AG.', 0)) else False,
                    "xu050_member": row.get('XU050 DEKI AG.', 0) > 0 if pd.notna(row.get('XU050 DEKI AG.', 0)) else False,
                    "xu100_member": row.get('XU100 DEKI AG.', 0) > 0 if pd.notna(row.get('XU100 DEKI AG.', 0)) else False,
                }
                
                # Determine BIST market classifications based on actual Excel data
                bist_markets = []
                
                # Use Excel data for accurate BIST classifications
                if stock_data.get("xu030_member", False):
                    bist_markets.append("bist_30")
                
                if stock_data.get("xu100_member", False):
                    bist_markets.append("bist_100")
                    
                if stock_data.get("xu050_member", False):
                    bist_markets.append("bist_50")
                
                # All traded stocks are in BIST All
                if stock_data["volume"] > 0:
                    bist_markets.append("bist_all")
                
                # If not classified, use fallback criteria
                if not bist_markets:
                    if stock_data["market_cap"] > 10000:  # >10B TL market cap
                        bist_markets.append("bist_all")
                
                stock_data["bist_markets"] = bist_markets
                
                # Update sector mapping
                if stock_data["sector"]:
                    if stock_data["sector"] not in sector_mapping:
                        sector_mapping[stock_data["sector"]] = {
                            "name": stock_data["sector"],
                            "name_turkish": stock_data["sector"],
                            "companies": []
                        }
                    sector_mapping[stock_data["sector"]]["companies"].append({
                        "symbol": symbol,
                        "name": stock_data["name_turkish"]
                    })
                
                enhanced_stocks.append(stock_data)
                
            except Exception as e:
                if 'symbol' in locals():
                    print(f"⚠️ Error processing row for {symbol}: {e}")
                else:
                    print(f"⚠️ Error processing row (symbol not set): {e}")
                continue
        
        # Save enhanced CSV
        csv_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/enhanced_bist_data.csv"
        enhanced_df = pd.DataFrame(enhanced_stocks)
        enhanced_df.to_csv(csv_path, index=False)
        print(f"✅ Enhanced CSV saved: {csv_path}")
        
        # Save enhanced JSON
        json_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/enhanced_bist_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "stocks": enhanced_stocks,
                "sectors": sector_mapping,
                "metadata": {
                    "total_stocks": len(enhanced_stocks),
                    "total_sectors": len(sector_mapping),
                    "bist_30_count": len([s for s in enhanced_stocks if "bist_30" in s["bist_markets"]]),
                    "bist_100_count": len([s for s in enhanced_stocks if "bist_100" in s["bist_markets"]]),
                    "source_file": "basestock2808.xlsx",
                    "generated_at": pd.Timestamp.now().isoformat()
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Enhanced JSON saved: {json_path}")
        
        # Generate statistics only if we have data
        if enhanced_stocks:
            print(f"\n📊 ENHANCED DATA SUMMARY:")
            print(f"   Total Stocks: {len(enhanced_stocks)}")
            print(f"   Total Sectors: {len(sector_mapping)}")
            print(f"   BIST 30 Stocks: {len([s for s in enhanced_stocks if 'bist_30' in s['bist_markets']])}")
            print(f"   BIST 100 Stocks: {len([s for s in enhanced_stocks if 'bist_100' in s['bist_markets']])}")
            if len(enhanced_stocks) > 0 and 'market_cap' in enhanced_df.columns:
                print(f"   Average Market Cap: {enhanced_df['market_cap'].mean():.0f}M TL")
                print(f"   Average Volume: {enhanced_df['volume'].mean():.0f}")
            
            # Sample data for frontend
            print(f"\n🎯 SAMPLE ENHANCED STOCK DATA:")
            sample_stocks = ['GARAN', 'ASTOR', 'THYAO', 'TUPRS', 'BRSAN']
            for symbol in sample_stocks:
                stock = next((s for s in enhanced_stocks if s["symbol"] == symbol), None)
                if stock:
                    print(f"   {symbol}: Market Cap: {stock['market_cap']:.0f}M, Markets: {stock['bist_markets']}, PE: {stock['pe_ratio']}")
        else:
            print(f"\n⚠️ NO STOCKS PROCESSED - Check Excel column names and data")
        
        return enhanced_stocks, sector_mapping
        
    except Exception as e:
        print(f"❌ Error processing Excel file: {e}")
        return None, None

if __name__ == "__main__":
    enhanced_excel_converter()
