#!/usr/bin/env python3
"""
🚀 BATCH EXCEL PROCESSOR - Process all 188 Excel files
"""

import os
import pandas as pd
import psycopg2
from pathlib import Path
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')

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

def process_single_excel(file_path_str: str):
    """Process single Excel file - designed for multiprocessing"""
    try:
        file_path = Path(file_path_str)
        symbol, timeframe = extract_symbol_and_timeframe(file_path.name)
        
        # Read Excel
        df = pd.read_excel(file_path, engine='openpyxl')
        if df.empty:
            return symbol, 0, f"Empty file"
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Find volume column - prefer 'Volume' over 'VOLUME'
        volume_col = None
        if 'Volume' in df.columns:
            volume_col = 'Volume'
        elif 'VOLUME' in df.columns:
            volume_col = 'VOLUME'
        
        # Column mapping
        column_mapping = {}
        for col in df.columns:
            col_clean = col.lower().strip()
            
            if 'date' in col_clean:
                column_mapping[col] = 'date'
            elif 'time' in col_clean:
                column_mapping[col] = 'time'
            elif col_clean == 'open':
                column_mapping[col] = 'open'
            elif col_clean == 'high':
                column_mapping[col] = 'high'
            elif col_clean == 'low':
                column_mapping[col] = 'low'
            elif col_clean == 'close':
                column_mapping[col] = 'close'
            elif col == volume_col:
                column_mapping[col] = 'volume'
            elif 'rsi (14)' in col_clean:
                column_mapping[col] = 'rsi_14'
            elif 'macd (26,12)' in col_clean:
                column_mapping[col] = 'macd_26_12'
            elif 'trigger (9)' in col_clean:
                column_mapping[col] = 'macd_trigger_9'
            elif 'bol u (20,2)' in col_clean and '.1' not in col_clean:
                column_mapping[col] = 'bol_upper_20_2'
            elif 'bol m (20,2)' in col_clean and '.1' not in col_clean:
                column_mapping[col] = 'bol_middle_20_2'
            elif 'bol d (20,2)' in col_clean and '.1' not in col_clean:
                column_mapping[col] = 'bol_lower_20_2'
            elif 'atr (14)' in col_clean:
                column_mapping[col] = 'atr_14'
            elif 'adx (14)' in col_clean:
                column_mapping[col] = 'adx_14'
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Process dates
        if 'date' not in df.columns:
            return symbol, 0, "No date column"
        
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        except:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        
        df = df.dropna(subset=['date'])
        if df.empty:
            return symbol, 0, "No valid dates"
        
        # Create datetime
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str),
                errors='coerce'
            )
        else:
            df['datetime'] = df['date']
        
        df = df.dropna(subset=['datetime'])
        if df.empty:
            return symbol, 0, "No valid datetimes"
        
        # Prepare records
        records = []
        for _, row in df.iterrows():
            try:
                def safe_float(value):
                    if pd.isna(value) or value == 0:
                        return None
                    try:
                        return float(value)
                    except:
                        return None
                
                record = {
                    'symbol': symbol,
                    'date': row['datetime'].date(),
                    'time': row['datetime'].time() if 'time' in df.columns else None,
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
                
                if any([record['open'], record['high'], record['low'], record['close']]):
                    records.append(record)
                    
            except Exception:
                continue
        
        return symbol, len(records), "success", records
        
    except Exception as e:
        return symbol, 0, f"Error: {str(e)}", []

def batch_insert_to_postgresql(records):
    """Insert records to PostgreSQL"""
    if not records:
        return 0
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    try:
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
        inserted_count = cursor.rowcount
        conn.commit()
        return inserted_count
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def main():
    logger.info("🚀 BATCH EXCEL PROCESSOR - Starting all 188 files...")
    
    excel_dir = Path("data/New_excell_Graph_C_D")
    excel_files = list(excel_dir.glob("*.xlsx"))
    
    logger.info(f"📁 Found {len(excel_files)} Excel files")
    
    start_time = time.time()
    total_records = 0
    total_processed = 0
    
    # Process files in batches of 20 to avoid memory issues
    batch_size = 20
    
    for batch_start in range(0, len(excel_files), batch_size):
        batch_files = excel_files[batch_start:batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(excel_files) + batch_size - 1) // batch_size
        
        logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        
        # Process current batch
        batch_records = []
        
        for file_path in batch_files:
            try:
                result = process_single_excel(str(file_path))
                symbol = result[0]
                count = result[1]
                status = result[2] if len(result) > 2 else "success"
                
                if len(result) > 3 and result[3]:
                    batch_records.extend(result[3])
                    logger.info(f"✅ {symbol}: {count:,} records")
                else:
                    logger.warning(f"⚠️ {symbol}: {status}")
                
                total_processed += 1
                
            except Exception as e:
                logger.error(f"❌ {file_path.name}: {e}")
        
        # Insert current batch to database
        if batch_records:
            try:
                inserted = batch_insert_to_postgresql(batch_records)
                total_records += inserted
                logger.info(f"💾 Batch {batch_num}: {inserted:,} kayıt database'e eklendi")
            except Exception as e:
                logger.error(f"❌ Database insert error for batch {batch_num}: {e}")
        
        # Progress update
        elapsed = time.time() - start_time
        logger.info(f"📊 Progress: {total_processed}/{len(excel_files)} files, {total_records:,} records, {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 BATCH COMPLETED: {total_records:,} records from {total_processed} files in {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()
