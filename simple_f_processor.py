#!/usr/bin/env python3
"""
Simple F folder processor - individual INSERT approach due to Railway COPY issues
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

def process_f_files():
    """Process F folder files with individual INSERTs"""
    
    folder_path = Path("data/New_excel_Graph_F")
    excel_files = list(folder_path.glob("*.xlsx"))
    
    logger.info(f"📁 Found {len(excel_files)} F folder files")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        total_inserted = 0
        processed_files = 0
        
        for i, file_path in enumerate(excel_files, 1):
            try:
                # Extract symbol and timeframe
                parts = file_path.name.replace('.xlsx', '').split('_')
                symbol = parts[0].upper()
                timeframe_raw = parts[1]
                
                timeframe_map = {'30Dk': '30min', '60Dk': '60min', 'Günlük': 'daily', '20Dk': '20min'}
                timeframe = timeframe_map.get(timeframe_raw, timeframe_raw)
                
                logger.info(f"[{i}/{len(excel_files)}] Processing: {symbol}-{timeframe}")
                
                # Read Excel
                df = pd.read_excel(file_path, engine='openpyxl')
                
                if df.empty:
                    logger.warning(f"  ⚠️ Empty: {symbol}")
                    continue
                
                # Clean column names
                df.columns = [col.strip() for col in df.columns]
                
                # Convert dates
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
                df = df.dropna(subset=['Date'])
                
                if df.empty:
                    logger.warning(f"  ⚠️ No valid dates: {symbol}")
                    continue
                
                # Process each row
                inserted_count = 0
                
                for _, row in df.iterrows():
                    try:
                        cursor.execute('''
                            INSERT INTO enhanced_stock_data 
                            (symbol, date, time, timeframe, open, high, low, close, volume, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                            ON CONFLICT (symbol, date, time, timeframe) DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume
                        ''', (
                            symbol,
                            row['Date'].date(),
                            str(row.get('Time', '')) if pd.notna(row.get('Time')) else '',
                            timeframe,
                            float(row['Open']) if pd.notna(row.get('Open')) and row.get('Open') != 0 else None,
                            float(row['High']) if pd.notna(row.get('High')) and row.get('High') != 0 else None,
                            float(row['Low']) if pd.notna(row.get('Low')) and row.get('Low') != 0 else None,
                            float(row['Close']) if pd.notna(row.get('Close')) and row.get('Close') != 0 else None,
                            float(row['Volume']) if pd.notna(row.get('Volume')) and row.get('Volume') != 0 else None
                        ))
                        
                        if cursor.rowcount > 0:
                            inserted_count += 1
                            
                    except Exception as e:
                        continue
                
                # Commit every file
                conn.commit()
                
                total_inserted += inserted_count
                processed_files += 1
                
                logger.info(f"  ✅ {symbol}: {inserted_count} records inserted")
                
            except Exception as e:
                logger.error(f"  ❌ {file_path.name}: {e}")
        
        logger.info(f"🎉 F folder processing completed!")
        logger.info(f"📊 Files processed: {processed_files}")
        logger.info(f"📈 Records inserted: {total_inserted:,}")
        
        # Final stats
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data WHERE symbol LIKE 'F%'")
        f_symbols = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data WHERE symbol LIKE 'F%'")
        f_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
        total_records = cursor.fetchone()[0]
        
        logger.info(f"📊 F Symbols: {f_symbols}")
        logger.info(f"📈 F Records: {f_records:,}")
        logger.info(f"🎯 TOTAL DATABASE RECORDS: {total_records:,}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")

def main():
    start_time = time.time()
    
    logger.info("🚀 F folder processing with individual INSERTs...")
    process_f_files()
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 Completed in {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()