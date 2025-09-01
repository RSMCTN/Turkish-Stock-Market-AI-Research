#!/usr/bin/env python3
"""
🚀 NEW EXCEL PROCESSOR for New_excell_Graph_C_D
Specifically designed for DD.MM.YYYY format Excel files
"""

import os
import pandas as pd
import psycopg2
from pathlib import Path
import logging
from datetime import datetime
import re
from typing import List, Dict, Any
import time

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')

def connect_to_postgresql():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        logger.info("✅ PostgreSQL connected successfully")
        return conn
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return None

def extract_symbol_and_timeframe(filename: str) -> tuple:
    """Extract symbol and timeframe from filename"""
    try:
        # Pattern: SYMBOL_TIMEFRAME.xlsx
        parts = filename.replace('.xlsx', '').split('_')
        if len(parts) >= 2:
            symbol = parts[0].upper()
            timeframe_raw = parts[1]
            
            # Map Turkish timeframes to standard
            timeframe_map = {
                '30Dk': '30min',
                '60Dk': '60min', 
                'Günlük': 'daily',
                '30m': '30min',
                '60m': '60min',
                'daily': 'daily'
            }
            
            timeframe = timeframe_map.get(timeframe_raw, 'unknown')
            return symbol, timeframe
        else:
            return parts[0].upper(), 'unknown'
            
    except Exception as e:
        logger.warning(f"⚠️ Filename parse error: {filename} - {e}")
        return filename.replace('.xlsx', '').upper(), 'unknown'

def process_excel_file(file_path: Path, symbol: str, timeframe: str) -> List[Dict]:
    """Process single Excel file with DD.MM.YYYY format support"""
    try:
        logger.info(f"📊 Processing: {symbol} - {timeframe}")
        
        # Read Excel file
        df = pd.read_excel(file_path, engine='openpyxl')
        
        if df.empty:
            logger.warning(f"⚠️ {symbol}: Empty file")
            return []
        
        # Column mapping
        column_mapping = {
            'Date': 'date',
            'Time': 'time', 
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'RSI (14)': 'rsi_14',
            'MACD (26,12)': 'macd_26_12',
            'TRIGGER (9)': 'macd_trigger_9',
            'BOL U (20,2)': 'bol_upper_20_2',
            'BOL M (20,2)': 'bol_middle_20_2',
            'BOL D (20,2)': 'bol_lower_20_2',
            'ATR (14)': 'atr_14',
            'ADX (14)': 'adx_14'
        }
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Find date column
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
        if not date_cols:
            # Try first column as date
            date_cols = [df.columns[0]]
        
        date_col = date_cols[0]
        time_col = 'time' if 'time' in df.columns else None
        
        # Process dates with DD.MM.YYYY format
        try:
            # Force DD.MM.YYYY parsing
            df[date_col] = pd.to_datetime(df[date_col], format='%d.%m.%Y', errors='coerce')
        except:
            try:
                # Try with dayfirst=True
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            except:
                logger.error(f"❌ {symbol}: Date parsing failed completely")
                return []
        
        # Remove rows with invalid dates
        df = df.dropna(subset=[date_col])
        
        if df.empty:
            logger.warning(f"⚠️ {symbol}: No valid dates after parsing")
            return []
        
        # Create datetime column
        if time_col and time_col in df.columns:
            # Combine date and time
            df['datetime'] = pd.to_datetime(
                df[date_col].dt.strftime('%Y-%m-%d') + ' ' + df[time_col].astype(str),
                errors='coerce'
            )
        else:
            # Use date only
            df['datetime'] = df[date_col]
        
        # Remove rows with invalid datetime
        df = df.dropna(subset=['datetime'])
        
        if df.empty:
            logger.warning(f"⚠️ {symbol}: No valid datetimes")
            return []
        
        # Prepare records for database
        records = []
        for _, row in df.iterrows():
            try:
                record = {
                    'symbol': symbol,
                    'date': row['datetime'].date(),
                    'time': row['datetime'].time() if time_col else None,
                    'timeframe': timeframe,
                    'open': float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                    'high': float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                    'low': float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                    'close': float(row.get('close', 0)) if pd.notna(row.get('close')) else None,
                    'volume': float(row.get('volume', 0)) if pd.notna(row.get('volume')) else None,
                    'rsi_14': float(row.get('rsi_14')) if pd.notna(row.get('rsi_14')) else None,
                    'macd_26_12': float(row.get('macd_26_12')) if pd.notna(row.get('macd_26_12')) else None,
                    'macd_trigger_9': float(row.get('macd_trigger_9')) if pd.notna(row.get('macd_trigger_9')) else None,
                    'bol_upper_20_2': float(row.get('bol_upper_20_2')) if pd.notna(row.get('bol_upper_20_2')) else None,
                    'bol_middle_20_2': float(row.get('bol_middle_20_2')) if pd.notna(row.get('bol_middle_20_2')) else None,
                    'bol_lower_20_2': float(row.get('bol_lower_20_2')) if pd.notna(row.get('bol_lower_20_2')) else None,
                    'atr_14': float(row.get('atr_14')) if pd.notna(row.get('atr_14')) else None,
                    'adx_14': float(row.get('adx_14')) if pd.notna(row.get('adx_14')) else None
                }
                records.append(record)
                
            except Exception as e:
                logger.warning(f"⚠️ {symbol}: Row processing error: {e}")
                continue
        
        logger.info(f"✅ {symbol}-{timeframe}: {len(records)} kayıt hazırlandı")
        return records
        
    except Exception as e:
        logger.error(f"❌ {symbol}: Processing error: {e}")
        return []

def batch_insert_to_postgresql(conn, records: List[Dict], batch_size: int = 1000):
    """Batch insert records to PostgreSQL"""
    if not records:
        return 0
    
    cursor = conn.cursor()
    inserted_count = 0
    
    try:
        # Prepare SQL
        insert_sql = """
        INSERT INTO enhanced_stock_data 
        (symbol, date, time, timeframe, open, high, low, close, volume,
         rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2, bol_middle_20_2, bol_lower_20_2,
         atr_14, adx_14, created_at)
        VALUES (%(symbol)s, %(date)s, %(time)s, %(timeframe)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s,
                %(rsi_14)s, %(macd_26_12)s, %(macd_trigger_9)s, %(bol_upper_20_2)s, %(bol_middle_20_2)s, %(bol_lower_20_2)s,
                %(atr_14)s, %(adx_14)s, CURRENT_TIMESTAMP)
        ON CONFLICT (symbol, date, time, timeframe) DO NOTHING
        """
        
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            cursor.executemany(insert_sql, batch)
            inserted_count += cursor.rowcount
            
            if i % (batch_size * 10) == 0 and i > 0:
                logger.info(f"📊 Batch {i//batch_size + 1}: {inserted_count} kayıt eklendi")
        
        conn.commit()
        logger.info(f"✅ Toplam {inserted_count} kayıt eklendi")
        return inserted_count
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Database insert error: {e}")
        return 0
    finally:
        cursor.close()

def get_database_stats(conn):
    """Get current database statistics"""
    try:
        cursor = conn.cursor()
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
        total_records = cursor.fetchone()[0]
        
        # Unique symbols
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data")
        unique_symbols = cursor.fetchone()[0]
        
        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM enhanced_stock_data")
        date_range = cursor.fetchone()
        
        cursor.close()
        
        return {
            'total_records': total_records,
            'unique_symbols': unique_symbols,
            'date_range': {
                'start': date_range[0].strftime('%Y-%m-%d') if date_range[0] else 'N/A',
                'end': date_range[1].strftime('%Y-%m-%d') if date_range[1] else 'N/A'
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Stats query error: {e}")
        return {}

def main():
    """Main processing function"""
    logger.info("🚀 NEW EXCEL PROCESSOR - Starting...")
    
    # Connect to database
    conn = connect_to_postgresql()
    if not conn:
        return
    
    # Get initial stats
    initial_stats = get_database_stats(conn)
    logger.info(f"📊 Initial database: {initial_stats['total_records']:,} records, {initial_stats['unique_symbols']} symbols")
    
    # Find New Excel files
    excel_dir = Path("data/New_excell_Graph_C_D")
    if not excel_dir.exists():
        logger.error(f"❌ Directory not found: {excel_dir}")
        return
    
    # Get all Excel files
    excel_files = list(excel_dir.glob("*.xlsx"))
    logger.info(f"📁 Found {len(excel_files)} Excel files in {excel_dir}")
    
    if not excel_files:
        logger.warning("⚠️ No Excel files found!")
        return
    
    # Process files
    total_records_processed = 0
    total_files_processed = 0
    start_time = time.time()
    
    for i, file_path in enumerate(excel_files, 1):
        try:
            # Extract symbol and timeframe
            symbol, timeframe = extract_symbol_and_timeframe(file_path.name)
            
            logger.info(f"[{i}/{len(excel_files)}] Processing: {file_path.name}")
            
            # Process Excel file
            records = process_excel_file(file_path, symbol, timeframe)
            
            if records:
                # Insert to database
                inserted = batch_insert_to_postgresql(conn, records)
                total_records_processed += inserted
                total_files_processed += 1
                
                logger.info(f"✅ {symbol}-{timeframe}: {inserted} kayıt eklendi")
            else:
                logger.warning(f"⚠️ {symbol}-{timeframe}: Kayıt bulunamadı")
            
            # Progress update
            if i % 20 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / i) * (len(excel_files) - i)
                logger.info(f"📊 Progress: {i}/{len(excel_files)} ({i/len(excel_files)*100:.1f}%) - ETA: {eta/60:.1f} min")
                
        except Exception as e:
            logger.error(f"❌ File processing error: {file_path.name} - {e}")
            continue
    
    # Final stats
    final_stats = get_database_stats(conn)
    elapsed_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("🎉 NEW EXCEL PROCESSING COMPLETED!")
    logger.info(f"📊 Files processed: {total_files_processed}/{len(excel_files)}")
    logger.info(f"📈 Records added: {total_records_processed:,}")
    logger.info(f"⏱️ Time taken: {elapsed_time/60:.1f} minutes")
    logger.info(f"⚡ Average: {elapsed_time/len(excel_files):.1f} seconds/file")
    logger.info("")
    logger.info("📊 FINAL DATABASE STATS:")
    logger.info(f"📈 Total records: {final_stats['total_records']:,}")
    logger.info(f"🏢 Total symbols: {final_stats['unique_symbols']}")
    logger.info(f"📅 Date range: {final_stats['date_range']['start']} → {final_stats['date_range']['end']}")
    logger.info(f"➕ New records: {final_stats['total_records'] - initial_stats['total_records']:,}")
    
    conn.close()
    logger.info("🚀 Processing completed successfully!")

if __name__ == "__main__":
    main()
