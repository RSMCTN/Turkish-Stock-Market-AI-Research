#!/usr/bin/env python3
"""
FIXED C-D Excel processor - Whitespace issue resolved
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

def safe_float(value):
    """Convert value to float safely"""
    if pd.isna(value) or value == 0:
        return None
    try:
        return float(value)
    except:
        return None

def process_file_fixed(file_path, conn):
    symbol, timeframe = extract_symbol_timeframe(file_path.name)
    
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        if df.empty:
            logger.warning(f"Empty file: {symbol}")
            return 0
        
        # 🎯 FIXED: Clean ALL column names - strip whitespace
        original_cols = list(df.columns)
        df.columns = [col.strip() for col in df.columns]
        
        logger.info(f"📋 Cleaned columns for {symbol}: {len(df.columns)} cols")
        
        # Enhanced column mapping with CLEANED names
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Basic OHLCV
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
            # Technical Indicators - FIXED with clean names
            elif col_lower == 'rsi (14)':
                column_map[col] = 'rsi_14'
            elif col_lower == 'macd (26,12)':
                column_map[col] = 'macd_26_12'
            elif col_lower == 'trigger (9)':
                column_map[col] = 'macd_trigger_9'
            elif col_lower == 'bol u (20,2)':
                column_map[col] = 'bol_upper_20_2'
            elif col_lower == 'bol m (20,2)':
                column_map[col] = 'bol_middle_20_2'
            elif col_lower == 'bol d (20,2)':
                column_map[col] = 'bol_lower_20_2'
            elif col_lower == 'atr (14)':
                column_map[col] = 'atr_14'
            elif col_lower == 'adx (14)':
                column_map[col] = 'adx_14'
        
        # Rename columns
        df.rename(columns=column_map, inplace=True)
        
        logger.info(f"🔄 Mapped {len(column_map)} columns for {symbol}")
        
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
        
        # Prepare records for batch insertion
        cursor = conn.cursor()
        records = []
        
        for _, row in df.iterrows():
            try:
                record = {
                    'symbol': symbol,
                    'date': row['date'].date(),
                    'time': row.get('time') if 'time' in df.columns else None,
                    'timeframe': timeframe,
                    'open': safe_float(row.get('open')),
                    'high': safe_float(row.get('high')), 
                    'low': safe_float(row.get('low')),
                    'close': safe_float(row.get('close')),
                    'volume': safe_float(row.get('volume')),
                    'rsi_14': safe_float(row.get('rsi_14')),
                    'macd_26_12': safe_float(row.get('macd_26_12')),
                    'macd_trigger_9': safe_float(row.get('macd_trigger_9')),
                    'bol_upper_20_2': safe_float(row.get('bol_upper_20_2')),
                    'bol_middle_20_2': safe_float(row.get('bol_middle_20_2')),
                    'bol_lower_20_2': safe_float(row.get('bol_lower_20_2')),
                    'atr_14': safe_float(row.get('atr_14')),
                    'adx_14': safe_float(row.get('adx_14'))
                }
                
                # Only add records with valid OHLC data
                ohlc_values = [record['open'], record['high'], record['low'], record['close']]
                if any(val is not None for val in ohlc_values):
                    records.append(record)
                    
            except Exception as e:
                logger.warning(f"Row processing error: {e}")
                continue
        
        # Batch insert
        if not records:
            logger.warning(f"No valid records for {symbol}")
            return 0
        
        insert_sql = """
        INSERT INTO enhanced_stock_data 
        (symbol, date, time, timeframe, open, high, low, close, volume,
         rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2, bol_middle_20_2, bol_lower_20_2,
         atr_14, adx_14, created_at)
        VALUES (%(symbol)s, %(date)s, %(time)s, %(timeframe)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s,
                %(rsi_14)s, %(macd_26_12)s, %(macd_trigger_9)s, %(bol_upper_20_2)s, %(bol_middle_20_2)s, %(bol_lower_20_2)s,
                %(atr_14)s, %(adx_14)s, CURRENT_TIMESTAMP)
        ON CONFLICT (symbol, date, time, timeframe) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            rsi_14 = EXCLUDED.rsi_14,
            macd_26_12 = EXCLUDED.macd_26_12,
            macd_trigger_9 = EXCLUDED.macd_trigger_9,
            bol_upper_20_2 = EXCLUDED.bol_upper_20_2,
            bol_middle_20_2 = EXCLUDED.bol_middle_20_2,
            bol_lower_20_2 = EXCLUDED.bol_lower_20_2,
            atr_14 = EXCLUDED.atr_14,
            adx_14 = EXCLUDED.adx_14
        """
        
        cursor.executemany(insert_sql, records)
        inserted = cursor.rowcount
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
    
    # Process first 5 files to test
    excel_dir = Path("data/New_excell_Graph_C_D")
    excel_files = list(excel_dir.glob("*.xlsx"))[:5]
    
    logger.info(f"Processing {len(excel_files)} test files...")
    
    total_records = 0
    start_time = time.time()
    
    for i, file_path in enumerate(excel_files, 1):
        logger.info(f"[{i}/{len(excel_files)}] Processing: {file_path.name}")
        inserted = process_file_fixed(file_path, conn)
        total_records += inserted
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 COMPLETED: {total_records:,} records in {elapsed:.1f} seconds")
    
    conn.close()

if __name__ == "__main__":
    main()