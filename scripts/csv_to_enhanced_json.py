#!/usr/bin/env python3
"""
Convert BIST CSV data to Enhanced JSON format
Extract all available financial metrics from the rich CSV dataset
"""

import pandas as pd
import json
from datetime import datetime
import sys

def convert_csv_to_enhanced_json():
    """Convert CSV to enhanced JSON with all available metrics"""
    
    csv_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/bist_real_data.csv"
    output_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/trading-dashboard/public/data/enhanced_bist_data.json"
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded CSV with {len(df)} stocks and {len(df.columns)} columns")
        
        # Print available columns for mapping
        print("\n📋 Available CSV columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Enhanced stock data structure
        enhanced_stocks = []
        
        for _, row in df.iterrows():
            try:
                # Safe float conversion function
                def safe_float(value, default=0.0):
                    try:
                        if pd.isna(value) or value == '':
                            return default
                        return float(str(value).replace(',', '.'))
                    except:
                        return default
                
                # Safe int conversion function  
                def safe_int(value, default=0):
                    try:
                        if pd.isna(value) or value == '':
                            return default
                        return int(float(str(value).replace(',', '.')))
                    except:
                        return default
                
                # Basic Info
                symbol = str(row['SEMBOL']).strip()
                name = str(row['ACKL']).strip() if pd.notna(row['ACKL']) else symbol
                sector = str(row['SEKTOR']).strip() if pd.notna(row['SEKTOR']) else 'DIGER'
                
                # Prices & Changes
                last_price = safe_float(row['SON'])
                change_percent = safe_float(row['%FARK'])
                change = safe_float(row['FARK'])
                high = safe_float(row['YÜKSEK'])
                low = safe_float(row['DÜŞÜK'])
                
                # Volume Data
                volume = safe_int(row['T.ADET'])
                volume_tl = safe_float(row['T.HACİM'])
                avg_volume_7d = safe_float(row['7GUN.ORT.HACIM'])
                avg_volume_30d = safe_float(row['30GUN.ORT.HACIM'])
                avg_volume_52w = safe_float(row['52HAFTA.ORT.HACIM'])
                avg_volume_year = safe_float(row['BUYIL.ORT.HACIM'])
                
                # Performance Metrics
                perf_7d = safe_float(row['7GUNLUK.F%'])
                perf_30d = safe_float(row['30GUNLUK.F%'])
                perf_year = safe_float(row['BU.YIL.FARK%'])
                perf_week = safe_float(row['BU.HAFTA.FARK%'])
                perf_month = safe_float(row['BU.AY.FARK%'])
                perf_5y = safe_float(row['5YILLIKGETIRI%'])
                
                # Price Ranges
                high_7d = safe_float(row['7GUNLUK.YUKSEK'])
                low_7d = safe_float(row['7GUNLUK.DUSUK'])
                high_30d = safe_float(row['30GUNLUK.YUKSEK'])
                low_30d = safe_float(row['30GUNLUK.DUSUK'])
                high_year = safe_float(row['BU.YIL.YÜK.'])
                low_year = safe_float(row['BU.YIL.DÜŞ.'])
                high_52w = safe_float(row['52HAFTALIK.YUKSEK'])
                low_52w = safe_float(row['52HAFTALIK.DUSUK'])
                high_5y = safe_float(row['5YIL.YUK.'])
                low_5y = safe_float(row['5YIL.DUS.'])
                
                # Financial Ratios
                pe = safe_float(row['F/K'])
                pb = safe_float(row['PD/DD'])
                market_cap = safe_float(row['PIY.DEG'])
                market_cap_usd = safe_float(row['PIY.DEG.($)'])
                market_cap_eur = safe_float(row['PIY.DEG.(E)'])
                net_income = safe_float(row['N.KAR'])
                net_debt = safe_float(row['NETBORC'])
                
                # Foreign Ownership
                foreign_ownership = safe_float(row['YABANCI PAYI %'])
                foreign_weekly_change = safe_float(row['YABANCI PAYI HAFTALIK DEĞ. %'])
                foreign_monthly_change = safe_float(row['YABANCI PAYI AYLIK DEĞ. %'])
                foreign_yearly_change = safe_float(row['YABANCI PAYI YILLIK DEĞ. %'])
                
                # Index Weights
                xu030_weight = safe_float(row['XU030 DAKI AG.'])
                xu050_weight = safe_float(row['XU050 DEKI AG.'])
                xu100_weight = safe_float(row['XU100 DEKI AG.'])
                xutum_weight = safe_float(row['XUTUM DEKI AG.'])
                
                # Other Metrics
                public_float = safe_float(row['HALK.ACK'])
                institution_ownership = safe_float(row['IHRYUZDE'])
                
                # Create enhanced stock record
                enhanced_stock = {
                    # Basic Info
                    "symbol": symbol,
                    "name": name,
                    "sector": sector,
                    "market": str(row['BORSA']).strip() if pd.notna(row['BORSA']) else 'UNKNOWN',
                    
                    # Prices & Changes
                    "lastPrice": last_price,
                    "change": change,
                    "changePercent": change_percent,
                    "high": high,
                    "low": low,
                    
                    # Volume Data
                    "volume": volume,
                    "volumeTL": volume_tl,
                    "avgVolume7d": avg_volume_7d,
                    "avgVolume30d": avg_volume_30d,
                    "avgVolume52w": avg_volume_52w,
                    "avgVolumeYear": avg_volume_year,
                    
                    # Performance Metrics
                    "perf7d": perf_7d,
                    "perf30d": perf_30d,
                    "perfYear": perf_year,
                    "perfWeek": perf_week,
                    "perfMonth": perf_month,
                    "perf5y": perf_5y,
                    
                    # Price Ranges
                    "high7d": high_7d,
                    "low7d": low_7d,
                    "high30d": high_30d,
                    "low30d": low_30d,
                    "highYear": high_year,
                    "lowYear": low_year,
                    "high52w": high_52w,
                    "low52w": low_52w,
                    "high5y": high_5y,
                    "low5y": low_5y,
                    
                    # Financial Ratios
                    "pe": pe,
                    "pb": pb,
                    "marketCap": market_cap,
                    "marketCapUSD": market_cap_usd,
                    "marketCapEUR": market_cap_eur,
                    "netIncome": net_income,
                    "netDebt": net_debt,
                    
                    # Foreign Ownership
                    "foreignOwnership": foreign_ownership,
                    "foreignWeeklyChange": foreign_weekly_change,
                    "foreignMonthlyChange": foreign_monthly_change,
                    "foreignYearlyChange": foreign_yearly_change,
                    
                    # Index Weights
                    "xu030Weight": xu030_weight,
                    "xu050Weight": xu050_weight,
                    "xu100Weight": xu100_weight,
                    "xutumWeight": xutum_weight,
                    
                    # Other
                    "publicFloat": public_float,
                    "institutionOwnership": institution_ownership
                }
                
                enhanced_stocks.append(enhanced_stock)
                
            except Exception as e:
                print(f"⚠️ Error processing {row['SEMBOL']}: {e}")
                continue
        
        # Create final JSON structure
        enhanced_data = {
            "metadata": {
                "updated_at": datetime.now().isoformat(),
                "total_stocks": len(enhanced_stocks),
                "data_source": "bist_real_data.csv",
                "columns_extracted": len(df.columns),
                "description": "Enhanced BIST data with all available financial metrics"
            },
            "stocks": enhanced_stocks
        }
        
        # Write to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Enhanced JSON created successfully!")
        print(f"📄 Output: {output_path}")
        print(f"📊 Stocks: {len(enhanced_stocks)}")
        print(f"💾 File size: {len(json.dumps(enhanced_data)) / 1024 / 1024:.2f} MB")
        
        # Sample data preview
        if enhanced_stocks:
            sample = enhanced_stocks[0]
            print(f"\n📋 Sample stock data for {sample['symbol']}:")
            print(f"   Name: {sample['name']}")
            print(f"   Sector: {sample['sector']}")
            print(f"   Price: ₺{sample['lastPrice']}")
            print(f"   Change: {sample['changePercent']:+.2f}%")
            print(f"   P/E: {sample['pe']}")
            print(f"   Foreign: {sample['foreignOwnership']:.1f}%")
            print(f"   Volume: {sample['volume']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = convert_csv_to_enhanced_json()
    sys.exit(0 if success else 1)
