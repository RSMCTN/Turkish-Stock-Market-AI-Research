#!/usr/bin/env python3
"""
Quick C-D Excel processor
"""

import os
import pandas as pd
import psycopg2
from pathlib import Path
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')

def connect_db():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return None

def extract_symbol_timeframe(filename):
    parts = filename.replace('.xlsx', '').split('_')
    if len(parts) >= 2:
        symbol = parts[0].upper()
        timeframe_raw = parts[1]
        
        timeframe_map = {
            '30Dk': '30min',
            '60Dk': '60min', 
            'Günlük': 'daily',
            '20Dk': '20min'
        }
        
        timeframe = timeframe_map.get(timeframe_raw, timeframe_raw)
        return symbol, timeframe
    return parts[0].upper(), 'unknown'

def process_file_quick(file_path, conn):
    symbol, timeframe = extract_symbol_timeframe(file_path.name)
    
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        if df.empty:
            logger.warning(f"Empty file: {symbol}")
            return 0
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Basic column mapping - only essential columns
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'date' in col_lower:
                column_map[col] = 'date'
            elif 'time' in col_lower:
                column_map[col] = 'time'  
            elif col_lower == 'open':
                column_map[col] = 'open'
            elif col_lower == 'high':
                column_map[col] = 'high'
            elif col_lower == 'low':
                column_map[col] = 'low'
            elif col_lower == 'close':
                column_map[col] = 'close'
            elif col_lower == 'volume':
                column_map[col] = 'volume'
        
        # Rename columns
        df.rename(columns=column_map, inplace=True)
        
        # Check required columns
        if 'date' not in df.columns:
            logger.error(f"No date column in {symbol}")
            return 0
        
        # Process dates
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        df = df.dropna(subset=['date'])
        
        if df.empty:
            logger.warning(f"No valid dates in {symbol}")
            return 0
        
        # Prepare records for insertion
        cursor = conn.cursor()
        inserted = 0
        
        for _, row in df.iterrows():
            try:
                # Simple insert with just OHLCV data
                cursor.execute("""
                    INSERT INTO enhanced_stock_data 
                    (symbol, date, time, timeframe, open, high, low, close, volume, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (symbol, date, time, timeframe) DO NOTHING
                """, (
                    symbol,
                    row['date'].date(),
                    row.get('time'),
                    timeframe,
                    row.get('open'),
                    row.get('high'), 
                    row.get('low'),
                    row.get('close'),
                    row.get('volume')
                ))
                inserted += cursor.rowcount
                
            except Exception as e:
                logger.warning(f"Row insert error: {e}")
                continue
        
        conn.commit()
        logger.info(f"✅ {symbol}-{timeframe}: {inserted} records inserted")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ {symbol} processing error: {e}")
        return 0

def main():
    conn = connect_db()
    if not conn:
        return
    
    # Process first 10 files to test
    excel_dir = Path("data/New_excell_Graph_C_D")
    excel_files = list(excel_dir.glob("*.xlsx"))[:10]
    
    logger.info(f"Processing {len(excel_files)} test files...")
    
    total_records = 0
    start_time = time.time()
    
    for i, file_path in enumerate(excel_files, 1):
        logger.info(f"[{i}/{len(excel_files)}] Processing: {file_path.name}")
        inserted = process_file_quick(file_path, conn)
        total_records += inserted
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 COMPLETED: {total_records:,} records in {elapsed:.1f} seconds")
    
    conn.close()

if __name__ == "__main__":
    main()