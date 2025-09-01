#!/usr/bin/env python3
"""
🚀 Excel Batch → CSV - Offline processing for Railway migration
"""

import pandas as pd
from pathlib import Path
import os
import time

def process_excel_batch():
    """Convert Excel files to CSV format compatible with csv_to_postgresql.py"""
    
    print("🚀 Excel Batch → CSV Processor")
    print("=" * 50)
    
    # Input directory
    excel_dir = Path("data/New_excell_Graph_C_D")
    
    if not excel_dir.exists():
        print(f"❌ Directory not found: {excel_dir}")
        return
    
    excel_files = list(excel_dir.glob("*.xlsx"))
    print(f"📁 Found {len(excel_files)} Excel files")
    
    if not excel_files:
        print("❌ No Excel files found!")
        return
    
    # Output file
    output_csv = "new_excel_batch.csv"
    
    # CSV columns (compatible with csv_to_postgresql.py)
    csv_columns = [
        'symbol', 'date', 'time', 'timeframe', 'open', 'high', 'low', 'close', 'volume',
        'rsi_14', 'macd_26_12', 'macd_trigger_9', 'bol_upper_20_2', 
        'bol_middle_20_2', 'bol_lower_20_2', 'atr_14', 'adx_14'
    ]
    
    all_records = []
    successful_files = 0
    failed_files = 0
    
    print("🔄 Processing files...")
    start_time = time.time()
    
    # Process files in batches
    for i, file_path in enumerate(excel_files, 1):
        try:
            print(f"[{i:3d}/{len(excel_files)}] {file_path.name}", end=" ")
            
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
            
            # Read Excel with basic columns
            try:
                df = pd.read_excel(
                    file_path, 
                    engine='openpyxl',
                    usecols=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                )
            except:
                # Fallback - read all columns and select
                df = pd.read_excel(file_path, engine='openpyxl')
                available_cols = [col for col in ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]
                df = df[available_cols]
            
            if df.empty:
                print("⚠️ Empty")
                failed_files += 1
                continue
            
            # Strip column names
            df.columns = df.columns.str.strip()
            
            # Parse dates
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
            else:
                print("❌ No Date column")
                failed_files += 1
                continue
            
            # Remove null dates
            df = df.dropna(subset=['Date'])
            
            if df.empty:
                print("⚠️ No valid dates")
                failed_files += 1
                continue
            
            # Convert to records
            file_records = 0
            for _, row in df.iterrows():
                try:
                    # Basic validation
                    if pd.isna([row.get('Open'), row.get('High'), row.get('Low'), row.get('Close')]).all():
                        continue
                    
                    record = [
                        symbol,  # symbol
                        row['Date'].strftime('%Y-%m-%d'),  # date
                        str(row.get('Time', '')) if pd.notna(row.get('Time')) else '',  # time
                        timeframe,  # timeframe
                        float(row.get('Open', 0)) if pd.notna(row.get('Open')) and row.get('Open', 0) != 0 else '',  # open
                        float(row.get('High', 0)) if pd.notna(row.get('High')) and row.get('High', 0) != 0 else '',  # high
                        float(row.get('Low', 0)) if pd.notna(row.get('Low')) and row.get('Low', 0) != 0 else '',  # low
                        float(row.get('Close', 0)) if pd.notna(row.get('Close')) and row.get('Close', 0) != 0 else '',  # close
                        float(row.get('Volume', 0)) if pd.notna(row.get('Volume')) and row.get('Volume', 0) != 0 else '',  # volume
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
                    file_records += 1
                    
                except Exception as row_error:
                    continue
            
            print(f"✅ {file_records:,} records")
            successful_files += 1
            
            # Progress update every 50 files
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed * 60
                print(f"📊 Progress: {i}/{len(excel_files)} files, {rate:.1f} files/min, {len(all_records):,} records")
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            failed_files += 1
    
    # Write CSV
    if all_records:
        print(f"\n📝 Writing {len(all_records):,} records to CSV...")
        
        df_output = pd.DataFrame(all_records, columns=csv_columns)
        df_output.to_csv(output_csv, index=False)
        
        file_size = os.path.getsize(output_csv) / 1024 / 1024
        
        print(f"✅ CSV File Created:")
        print(f"  📄 {output_csv}")
        print(f"  📊 Size: {file_size:.1f} MB") 
        print(f"  📈 Records: {len(all_records):,}")
        print(f"  ✅ Successful files: {successful_files}")
        print(f"  ❌ Failed files: {failed_files}")
        
        # Show sample
        print(f"\n🧪 Sample records:")
        for i in range(min(3, len(all_records))):
            sample = all_records[i]
            print(f"  {i+1}: {sample[0]} | {sample[1]} {sample[2]} | {sample[3]} | OHLC: {sample[4]}-{sample[7]}")
        
        elapsed = time.time() - start_time
        print(f"\n🎉 COMPLETED in {elapsed:.1f}s")
        print(f"📋 Next step: Use csv_to_postgresql.py to upload this CSV")
        
    else:
        print("❌ No records created!")

if __name__ == "__main__":
    process_excel_batch()
