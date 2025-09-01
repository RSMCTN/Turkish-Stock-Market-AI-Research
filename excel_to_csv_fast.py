#!/usr/bin/env python3
"""
⚡ Excel → CSV FAST - Başarılı script'le uyumlu format
"""

import pandas as pd
from pathlib import Path
import os
import gzip

def excel_to_csv_fast():
    """New Excel files → CSV format compatible with successful csv_to_postgresql.py"""
    print("⚡ New Excel → CSV Fast Conversion")
    
    # New Excel directory
    excel_dir = Path("data/New_excell_Graph_C_D")
    excel_files = list(excel_dir.glob("*.xlsx"))
    
    print(f"📁 Found {len(excel_files)} Excel files")
    
    if not excel_files:
        print("❌ No Excel files found!")
        return
    
    # Output CSV
    output_csv = "new_excel_data.csv"
    
    all_records = []
    processed_count = 0
    
    # CSV header (compatible with csv_to_postgresql.py)
    csv_header = [
        'symbol', 'date', 'time', 'timeframe', 'open', 'high', 'low', 'close', 'volume',
        'rsi_14', 'macd_26_12', 'macd_trigger_9', 'bol_upper_20_2', 
        'bol_middle_20_2', 'bol_lower_20_2', 'atr_14', 'adx_14'
    ]
    
    print("🔄 Processing Excel files...")
    
    for file_path in excel_files[:20]:  # First 20 files for speed test
        try:
            # Extract symbol and timeframe
            parts = file_path.stem.split('_')
            symbol = parts[0].upper()
            timeframe_raw = parts[1] if len(parts) > 1 else 'unknown'
            
            timeframe_map = {
                '30Dk': '30min',
                '60Dk': '60min',
                'Günlük': 'daily'
            }
            timeframe = timeframe_map.get(timeframe_raw, timeframe_raw)
            
            # Read Excel - only OHLCV columns for speed
            df = pd.read_excel(
                file_path,
                engine='openpyxl',
                usecols=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                nrows=1000  # Limit to 1000 rows per file for testing
            )
            
            if df.empty:
                continue
            
            # Clean columns
            df.columns = df.columns.str.strip()
            
            # Parse dates
            df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
            df = df.dropna(subset=['Date'])
            
            if df.empty:
                continue
            
            # Create records
            for _, row in df.iterrows():
                record = [
                    symbol,  # symbol
                    row['Date'].strftime('%Y-%m-%d'),  # date
                    row['Time'] if 'Time' in df.columns and pd.notna(row['Time']) else '',  # time
                    timeframe,  # timeframe
                    float(row['Open']) if pd.notna(row['Open']) and row['Open'] != 0 else '',  # open
                    float(row['High']) if pd.notna(row['High']) and row['High'] != 0 else '',  # high
                    float(row['Low']) if pd.notna(row['Low']) and row['Low'] != 0 else '',  # low
                    float(row['Close']) if pd.notna(row['Close']) and row['Close'] != 0 else '',  # close
                    float(row['Volume']) if pd.notna(row['Volume']) and row['Volume'] != 0 else '',  # volume
                    '',  # rsi_14 (empty for now)
                    '',  # macd_26_12
                    '',  # macd_trigger_9
                    '',  # bol_upper_20_2
                    '',  # bol_middle_20_2
                    '',  # bol_lower_20_2
                    '',  # atr_14
                    ''   # adx_14
                ]
                all_records.append(record)
            
            processed_count += 1
            print(f"  ✅ {symbol}-{timeframe}: {len(df)} records")
            
        except Exception as e:
            print(f"  ❌ {file_path.name}: {e}")
    
    if not all_records:
        print("❌ No records to write!")
        return
    
    # Write CSV
    print(f"📝 Writing {len(all_records):,} records to CSV...")
    
    # Create DataFrame and save
    df_output = pd.DataFrame(all_records, columns=csv_header)
    df_output.to_csv(output_csv, index=False)
    
    # Create gzipped version (Railway compatible)
    with open(output_csv, 'rb') as f_in:
        with gzip.open(f"{output_csv}.gz", 'wb') as f_out:
            f_out.writelines(f_in)
    
    file_size = os.path.getsize(output_csv) / 1024 / 1024
    gz_size = os.path.getsize(f"{output_csv}.gz") / 1024 / 1024
    
    print(f"✅ CSV Created:")
    print(f"  📄 {output_csv}: {file_size:.1f} MB")
    print(f"  📦 {output_csv}.gz: {gz_size:.1f} MB")
    print(f"  📊 Total records: {len(all_records):,}")
    print(f"  📁 Files processed: {processed_count}")
    
    # Show sample
    print(f"\n🧪 Sample records:")
    for i, record in enumerate(all_records[:3]):
        print(f"  {i+1}: {record[:8]}")

if __name__ == "__main__":
    excel_to_csv_fast()
