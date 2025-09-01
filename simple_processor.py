#!/usr/bin/env python3
"""
🎯 SIMPLE PROCESSOR - One file at a time with immediate commits
"""

import os
import pandas as pd
import psycopg2
from pathlib import Path
import time

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')

def get_processed_symbols():
    """Get symbols that already have significant data"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Get symbols with more than 100 records (indicates full processing)
        cursor.execute('SELECT symbol FROM enhanced_stock_data GROUP BY symbol HAVING COUNT(*) > 100')
        processed = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        return processed
        
    except Exception as e:
        print(f"❌ Database check error: {e}")
        return []

def extract_symbol_and_timeframe(filename: str) -> tuple:
    parts = filename.replace('.xlsx', '').split('_')
    if len(parts) >= 2:
        symbol = parts[0].upper()
        timeframe_raw = parts[1]
        
        timeframe_map = {
            '30Dk': '30min',
            '60Dk': '60min', 
            'Günlük': 'daily',
        }
        
        timeframe = timeframe_map.get(timeframe_raw, timeframe_raw)
        return symbol, timeframe
    else:
        return parts[0].upper(), 'unknown'

def process_single_file(file_path: Path):
    """Process single file completely"""
    try:
        symbol, timeframe = extract_symbol_and_timeframe(file_path.name)
        print(f"📊 Processing {symbol}-{timeframe} ({file_path.name})")
        
        # Read Excel - only essential columns
        df = pd.read_excel(
            file_path, 
            engine='openpyxl',
            usecols=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        if df.empty:
            print(f"  ⚠️ Empty file")
            return 0
        
        # Clean and process
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        
        if 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'].astype(str),
                errors='coerce'
            )
        else:
            df['datetime'] = df['Date']
            
        df = df.dropna(subset=['datetime'])
        
        if df.empty:
            print(f"  ⚠️ No valid dates")
            return 0
        
        # Create records
        records = []
        for _, row in df.iterrows():
            # Skip rows with no OHLC data
            if pd.isna([row['Open'], row['High'], row['Low'], row['Close']]).all():
                continue
                
            record = {
                'symbol': symbol,
                'date': row['datetime'].date(),
                'time': row['datetime'].time() if 'Time' in df.columns else None,
                'timeframe': timeframe,
                'open': float(row['Open']) if pd.notna(row['Open']) and row['Open'] != 0 else None,
                'high': float(row['High']) if pd.notna(row['High']) and row['High'] != 0 else None,
                'low': float(row['Low']) if pd.notna(row['Low']) and row['Low'] != 0 else None,
                'close': float(row['Close']) if pd.notna(row['Close']) and row['Close'] != 0 else None,
                'volume': float(row['Volume']) if pd.notna(row['Volume']) and row['Volume'] != 0 else None,
            }
            records.append(record)
        
        if not records:
            print(f"  ⚠️ No valid records")
            return 0
        
        # Insert to database
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        insert_sql = """
        INSERT INTO enhanced_stock_data 
        (symbol, date, time, timeframe, open, high, low, close, volume, created_at)
        VALUES (%(symbol)s, %(date)s, %(time)s, %(timeframe)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, CURRENT_TIMESTAMP)
        ON CONFLICT (symbol, date, time, timeframe) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume
        """
        
        # Insert in batches of 1000
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            cursor.executemany(insert_sql, batch)
            total_inserted += cursor.rowcount
            
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"  ✅ {total_inserted:,} records inserted")
        return total_inserted
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0

def main():
    print("🎯 SIMPLE PROCESSOR - Starting...")
    
    # Get processed symbols
    processed_symbols = get_processed_symbols()
    print(f"🔍 Already processed symbols ({len(processed_symbols)}): {processed_symbols[:10]}...")
    
    # Get files to process
    excel_dir = Path("data/New_excell_Graph_C_D")
    all_files = list(excel_dir.glob("*.xlsx"))
    
    # Filter remaining files
    remaining_files = []
    for file_path in all_files:
        symbol, _ = extract_symbol_and_timeframe(file_path.name)
        if symbol not in processed_symbols:
            remaining_files.append(file_path)
    
    print(f"📁 Total: {len(all_files)}, Remaining: {len(remaining_files)}")
    
    if not remaining_files:
        print("🎉 All files processed!")
        return
    
    # Process files one by one
    start_time = time.time()
    total_records = 0
    
    for i, file_path in enumerate(remaining_files, 1):
        print(f"\n[{i:3d}/{len(remaining_files)}]", end=" ")
        
        try:
            records_added = process_single_file(file_path)
            total_records += records_added
            
            # Progress every 10 files
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed * 60
                print(f"\n📊 Progress: {i}/{len(remaining_files)} files, {total_records:,} records, {rate:.1f} files/min")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Stopped by user after {i-1} files")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n🎉 Completed: {total_records:,} records from {i} files in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
