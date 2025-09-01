#!/usr/bin/env python3
"""
🎯 FINAL EXCEL PROCESSOR - Duplicate column mapping fixed
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

def connect_to_postgresql():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        logger.info("✅ PostgreSQL connected")
        return conn
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return None

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

def process_excel_file(file_path: Path, symbol: str, timeframe: str):
    try:
        logger.info(f"📊 Processing: {symbol} - {timeframe}")
        
        df = pd.read_excel(file_path, engine='openpyxl')
        
        if df.empty:
            logger.warning(f"⚠️ {symbol}: Empty file")
            return []
        
        # Strip whitespace from all column names
        df.columns = df.columns.str.strip()
        
        # 🎯 FIXED: Handle duplicate Volume columns - prefer first occurrence
        original_columns = list(df.columns)
        logger.info(f"🔍 Original columns count: {len(original_columns)}")
        
        # Find volume column - prefer 'Volume' over 'VOLUME'
        volume_col = None
        if 'Volume' in df.columns:
            volume_col = 'Volume'
        elif 'VOLUME' in df.columns:
            volume_col = 'VOLUME'
        
        # Enhanced column mapping - NO DUPLICATES!
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
            elif col == volume_col:  # Only map the preferred volume column
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
        
        logger.info(f"🔄 Column mapping ({len(column_mapping)} mappings): {column_mapping}")
        
        # Check for duplicate values in mapping
        mapped_values = list(column_mapping.values())
        duplicates = [v for v in set(mapped_values) if mapped_values.count(v) > 1]
        if duplicates:
            logger.error(f"❌ DUPLICATE MAPPINGS DETECTED: {duplicates}")
            return []
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Process dates with DD.MM.YYYY format
        if 'date' not in df.columns:
            logger.error(f"❌ {symbol}: No date column found")
            return []
        
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        except:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        
        # Remove invalid dates
        df = df.dropna(subset=['date'])
        
        if df.empty:
            logger.warning(f"⚠️ {symbol}: No valid dates")
            return []
        
        # Create datetime column
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str),
                errors='coerce'
            )
        else:
            df['datetime'] = df['date']
        
        df = df.dropna(subset=['datetime'])
        
        if df.empty:
            logger.warning(f"⚠️ {symbol}: No valid datetimes")
            return []
        
        # Prepare records
        records = []
        for idx, row in df.iterrows():
            try:
                # 🎯 FIXED: Safe column access with explicit None checks
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
                
                # Only add records with valid OHLC data
                if any([record['open'], record['high'], record['low'], record['close']]):
                    records.append(record)
                    
            except Exception as e:
                logger.warning(f"⚠️ Row {idx} processing error: {e}")
                continue
        
        logger.info(f"✅ {symbol}-{timeframe}: {len(records)} valid kayıt hazırlandı")
        return records
        
    except Exception as e:
        logger.error(f"❌ {symbol}: Processing error: {e}")
        return []

def batch_insert_to_postgresql(conn, records, batch_size=1000):
    if not records:
        return 0
    
    cursor = conn.cursor()
    inserted_count = 0
    
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
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            cursor.executemany(insert_sql, batch)
            inserted_count += cursor.rowcount
            
        conn.commit()
        logger.info(f"✅ Toplam {inserted_count} kayıt eklendi/güncellendi")
        return inserted_count
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Database insert error: {e}")
        return 0
    finally:
        cursor.close()

def main():
    logger.info("🎯 FINAL EXCEL PROCESSOR - Starting...")
    
    conn = connect_to_postgresql()
    if not conn:
        return
    
    # Test with first 10 files
    excel_dir = Path("data/New_excell_Graph_C_D")
    excel_files = list(excel_dir.glob("*.xlsx"))[:10]  # First 10 for testing
    
    logger.info(f"📁 Processing {len(excel_files)} Excel files (TEST MODE)")
    
    total_records = 0
    start_time = time.time()
    
    for i, file_path in enumerate(excel_files, 1):
        try:
            symbol, timeframe = extract_symbol_and_timeframe(file_path.name)
            logger.info(f"[{i}/{len(excel_files)}] Processing: {file_path.name}")
            
            records = process_excel_file(file_path, symbol, timeframe)
            
            if records:
                inserted = batch_insert_to_postgresql(conn, records)
                total_records += inserted
                logger.info(f"✅ {symbol}-{timeframe}: {inserted} kayıt eklendi")
            else:
                logger.warning(f"⚠️ {symbol}-{timeframe}: No records")
                
        except Exception as e:
            logger.error(f"❌ File error: {file_path.name} - {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 TEST COMPLETED: {total_records:,} records in {elapsed:.1f} seconds")
    
    conn.close()

if __name__ == "__main__":
    main()
