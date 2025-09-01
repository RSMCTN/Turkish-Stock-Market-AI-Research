#!/usr/bin/env python3
"""
⚡ FAST EXCEL PROCESSOR - Optimized for speed
"""

import os
import pandas as pd
import psycopg2
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')

def get_processed_symbols():
    """Get already processed symbols to skip them"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT symbol 
            FROM enhanced_stock_data 
            WHERE symbol IN (
                SELECT UPPER(SPLIT_PART(REPLACE(filename, '.xlsx', ''), '_', 1))
                FROM (VALUES 
                    ('CRDFA_60Dk.xlsx'),('CRFSA_60Dk.xlsx'),('CONSE_30Dk.xlsx'),
                    ('EDATA_30Dk.xlsx'),('CIMSA_60Dk.xlsx')
                ) AS files(filename)
            )
        ''')
        
        processed = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return processed
        
    except Exception as e:
        logger.warning(f"Could not check processed symbols: {e}")
        return ['CRDFA', 'CRFSA', 'CONSE', 'EDATA', 'CIMSA']  # Known processed

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

def process_excel_fast(file_path: Path):
    """Fast Excel processing with minimal overhead"""
    try:
        symbol, timeframe = extract_symbol_and_timeframe(file_path.name)
        
        # Read only required columns for speed
        df = pd.read_excel(
            file_path, 
            engine='openpyxl',
            usecols=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        if df.empty:
            return symbol, 0, "Empty"
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Fast date parsing
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        
        if df.empty:
            return symbol, 0, "No dates"
        
        # Create datetime
        if 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'].astype(str),
                errors='coerce'
            )
        else:
            df['datetime'] = df['Date']
        
        df = df.dropna(subset=['datetime'])
        
        if df.empty:
            return symbol, 0, "No datetimes"
        
        # Fast record creation - vectorized
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
        
        return symbol, len(records), "success", records
        
    except Exception as e:
        return symbol, 0, f"Error: {str(e)}"

def batch_insert_fast(conn, records, batch_size=5000):
    """Fast batch insert"""
    if not records:
        return 0
    
    cursor = conn.cursor()
    
    try:
        # Simple insert without all indicators for now - just OHLC data
        insert_sql = """
        INSERT INTO enhanced_stock_data 
        (symbol, date, time, timeframe, open, high, low, close, volume, created_at)
        VALUES (%(symbol)s, %(date)s, %(time)s, %(timeframe)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, CURRENT_TIMESTAMP)
        ON CONFLICT (symbol, date, time, timeframe) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
        """
        
        # Process in batches
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            cursor.executemany(insert_sql, batch)
            total_inserted += cursor.rowcount
            
        conn.commit()
        return total_inserted
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()

def main():
    logger.info("⚡ FAST EXCEL PROCESSOR - Starting...")
    
    # Get processed symbols
    processed_symbols = get_processed_symbols()
    logger.info(f"🔍 Already processed symbols: {processed_symbols}")
    
    # Get all Excel files
    excel_dir = Path("data/New_excell_Graph_C_D")
    all_files = list(excel_dir.glob("*.xlsx"))
    
    # Filter out processed files
    remaining_files = []
    for file_path in all_files:
        symbol, _ = extract_symbol_and_timeframe(file_path.name)
        if symbol not in processed_symbols:
            remaining_files.append(file_path)
    
    logger.info(f"📁 Total files: {len(all_files)}, Remaining: {len(remaining_files)}")
    
    if not remaining_files:
        logger.info("🎉 All files already processed!")
        return
    
    # Process in small batches of 10 files
    batch_size = 10
    start_time = time.time()
    total_records = 0
    
    for batch_start in range(0, len(remaining_files), batch_size):
        batch_files = remaining_files[batch_start:batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(remaining_files) + batch_size - 1) // batch_size
        
        logger.info(f"📦 Batch {batch_num}/{total_batches}: Processing {len(batch_files)} files")
        
        conn = psycopg2.connect(DATABASE_URL)
        batch_records = []
        
        for file_path in batch_files:
            try:
                result = process_excel_fast(file_path)
                symbol = result[0]
                count = result[1]
                status = result[2] if len(result) > 2 else "success"
                
                if len(result) > 3 and result[3]:
                    batch_records.extend(result[3])
                    logger.info(f"  ✅ {symbol}: {count:,} records")
                else:
                    logger.warning(f"  ⚠️ {symbol}: {status}")
                    
            except Exception as e:
                logger.error(f"  ❌ {file_path.name}: {e}")
        
        # Insert batch
        if batch_records:
            try:
                inserted = batch_insert_fast(conn, batch_records)
                total_records += inserted
                logger.info(f"💾 Batch {batch_num}: {inserted:,} records inserted")
            except Exception as e:
                logger.error(f"❌ Database error: {e}")
        
        conn.close()
        
        # Progress
        elapsed = time.time() - start_time
        files_done = min(batch_start + batch_size, len(remaining_files))
        logger.info(f"📊 Progress: {files_done}/{len(remaining_files)} files, {total_records:,} records, {elapsed:.1f}s")
        
        if batch_num % 5 == 0:  # Every 5 batches, brief pause
            time.sleep(2)
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 COMPLETED: {total_records:,} records from {len(remaining_files)} files in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
