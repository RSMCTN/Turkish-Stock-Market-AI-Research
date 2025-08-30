#!/usr/bin/env python3
"""
Analyze historical Excel files to understand their structure
"""

import pandas as pd
import os
from datetime import datetime

def analyze_historical_excel():
    """Analyze a few sample historical Excel files"""
    
    data_dir = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ"
    
    # Sample files to analyze
    sample_files = [
        "GARAN_60Dk.xlsx",
        "GARAN_Günlük.xlsx", 
        "ASTOR_60Dk.xlsx",
        "BRSAN_Günlük.xlsx"
    ]
    
    for filename in sample_files:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filename}")
            continue
            
        print(f"\n📊 ANALYZING: {filename}")
        print("=" * 60)
        
        try:
            # Read Excel file
            df = pd.read_excel(filepath)
            
            print(f"📋 Rows: {len(df)}")
            print(f"📋 Columns: {len(df.columns)}")
            
            # Show column names
            print(f"📋 Column names:")
            for i, col in enumerate(df.columns, 1):
                print(f"   {i:2d}. {col}")
            
            # Show first few rows
            print(f"\n📋 First 3 rows:")
            print(df.head(3).to_string())
            
            # Show data types
            print(f"\n📋 Data types:")
            for col in df.columns:
                print(f"   {col}: {df[col].dtype}")
            
            # Show date range if there's a date column
            date_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'tarih', 'zaman', 'time'])]
            if date_columns:
                date_col = date_columns[0]
                print(f"\n📋 Date range ({date_col}):")
                print(f"   From: {df[date_col].min()}")
                print(f"   To: {df[date_col].max()}")
                print(f"   Total periods: {len(df)}")
            
            # Check for OHLCV pattern
            price_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['open', 'high', 'low', 'close', 'volume', 'açılış', 'yüksek', 'düşük', 'kapanış', 'hacim']):
                    price_columns.append(col)
            
            if price_columns:
                print(f"\n📋 Price/Volume columns found:")
                for col in price_columns:
                    print(f"   - {col}")
                    
                # Show sample price data
                sample_prices = df[price_columns].head(3)
                print(f"\n📋 Sample price data:")
                print(sample_prices.to_string())
            
            print(f"\n" + "="*60)
            
        except Exception as e:
            print(f"❌ Error analyzing {filename}: {e}")

if __name__ == "__main__":
    analyze_historical_excel()
