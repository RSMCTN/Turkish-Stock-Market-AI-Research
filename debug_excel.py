#!/usr/bin/env python3
"""
🔍 DEBUG EXCEL - Analyze New_excell_Graph_C_D files
"""

import pandas as pd
from pathlib import Path
import psycopg2
import os

def debug_excel_file(file_path):
    """Debug single Excel file"""
    print(f"\n🔍 DEBUGGING: {file_path.name}")
    
    try:
        # Read Excel
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show first few rows
        print("\n📊 First 3 rows:")
        print(df.head(3))
        
        # Check date column
        if not df.empty:
            # Try to find date column
            date_cols = [col for col in df.columns if 'date' in col.lower() or any(word in col.lower() for word in ['tarih', 'Date', 'DATE'])]
            if not date_cols:
                date_cols = [df.columns[0]]  # First column
            
            print(f"\n📅 Date column: {date_cols[0]}")
            print(f"📅 Sample dates: {df[date_cols[0]].head(5).tolist()}")
            
            # Try parsing dates
            try:
                # DD.MM.YYYY format
                parsed_dates = pd.to_datetime(df[date_cols[0]], format='%d.%m.%Y', errors='coerce')
                valid_dates = parsed_dates.dropna()
                print(f"✅ DD.MM.YYYY parsing: {len(valid_dates)}/{len(df)} success")
                if len(valid_dates) > 0:
                    print(f"📅 Parsed date range: {valid_dates.min()} → {valid_dates.max()}")
            except Exception as e:
                print(f"❌ DD.MM.YYYY parsing failed: {e}")
                
            try:
                # Auto parsing
                parsed_dates = pd.to_datetime(df[date_cols[0]], dayfirst=True, errors='coerce')
                valid_dates = parsed_dates.dropna()
                print(f"✅ Auto parsing (dayfirst=True): {len(valid_dates)}/{len(df)} success")
            except Exception as e:
                print(f"❌ Auto parsing failed: {e}")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def check_database_constraints():
    """Check database for existing data"""
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Check if any CANTE records exist
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data WHERE symbol = 'CANTE'")
        cante_count = cursor.fetchone()[0]
        print(f"\n🔍 CANTE existing records: {cante_count}")
        
        # Check recent records
        cursor.execute("SELECT symbol, COUNT(*) FROM enhanced_stock_data GROUP BY symbol ORDER BY COUNT(*) DESC LIMIT 10")
        top_symbols = cursor.fetchall()
        print(f"\n📊 Top symbols in database:")
        for symbol, count in top_symbols:
            print(f"  {symbol}: {count:,} records")
        
        # Check date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM enhanced_stock_data")
        date_range = cursor.fetchone()
        print(f"\n📅 Database date range: {date_range[0]} → {date_range[1]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database check error: {e}")

def main():
    print("🔍 EXCEL DEBUG ANALYSIS")
    
    # Check database first
    check_database_constraints()
    
    # Find Excel files
    excel_dir = Path("data/New_excell_Graph_C_D")
    excel_files = list(excel_dir.glob("*.xlsx"))[:3]  # First 3 files
    
    print(f"\n📁 Analyzing first 3 out of {len(list(excel_dir.glob('*.xlsx')))} files:")
    
    for file_path in excel_files:
        debug_excel_file(file_path)

if __name__ == "__main__":
    main()
