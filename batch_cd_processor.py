#!/usr/bin/env python3
"""
Batch C-D Excel to PostgreSQL processor via CSV
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

def excel_to_csv_batch(excel_dir, batch_size=10):
    """Convert Excel files to CSV in batches"""
    
    excel_files = list(Path(excel_dir).glob("*.xlsx"))
    logger.info(f"📁 Found {len(excel_files)} Excel files")
    
    # Process in batches
    for batch_num, start_idx in enumerate(range(0, len(excel_files), batch_size)):
        batch_files = excel_files[start_idx:start_idx + batch_size]
        
        logger.info(f"🔄 Processing batch {batch_num + 1}: {len(batch_files)} files")
        
        combined_df = []
        
        for file_path in batch_files:
            try:
                # Extract symbol and timeframe
                parts = file_path.name.replace('.xlsx', '').split('_')
                symbol = parts[0].upper()
                timeframe_raw = parts[1]
                
                timeframe_map = {'30Dk': '30min', '60Dk': '60min', 'Günlük': 'daily', '20Dk': '20min'}
                timeframe = timeframe_map.get(timeframe_raw, timeframe_raw)
                
                # Read Excel
                df = pd.read_excel(file_path, engine='openpyxl')
                
                if df.empty:
                    logger.warning(f"⚠️ Empty: {symbol}")
                    continue
                
                # Clean column names
                df.columns = [col.strip() for col in df.columns]
                
                # Add metadata
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                
                # Convert dates
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
                df = df.dropna(subset=['Date'])
                
                # Select essential columns
                essential_cols = ['symbol', 'Date', 'Time', 'timeframe', 'Open', 'High', 'Low', 'Close', 'Volume']
                available_cols = [col for col in essential_cols if col in df.columns]
                df_clean = df[available_cols].copy()
                
                if not df_clean.empty:
                    combined_df.append(df_clean)
                    logger.info(f"  ✅ {symbol}: {len(df_clean)} rows")
                
            except Exception as e:
                logger.error(f"  ❌ {file_path.name}: {e}")
        
        # Save batch to CSV
        if combined_df:
            batch_df = pd.concat(combined_df, ignore_index=True)
            csv_file = f"cd_batch_{batch_num + 1}.csv"
            batch_df.to_csv(csv_file, index=False)
            
            logger.info(f"💾 Saved batch {batch_num + 1}: {len(batch_df)} rows to {csv_file}")
            
            # Import to PostgreSQL
            import_csv_to_postgresql(csv_file)
            
            # Clean up CSV
            Path(csv_file).unlink()
            
        else:
            logger.warning(f"⚠️ Batch {batch_num + 1}: No valid data")

def import_csv_to_postgresql(csv_file):
    """Import CSV to PostgreSQL"""
    
    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        inserted = 0
        
        for _, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT INTO enhanced_stock_data 
                    (symbol, date, time, timeframe, open, high, low, close, volume, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (symbol, date, time, timeframe) DO NOTHING
                ''', (
                    row['symbol'],
                    row['Date'].date(),
                    str(row['Time']) if pd.notna(row['Time']) else None,
                    row['timeframe'],
                    float(row['Open']) if pd.notna(row['Open']) and row['Open'] != 0 else None,
                    float(row['High']) if pd.notna(row['High']) and row['High'] != 0 else None,
                    float(row['Low']) if pd.notna(row['Low']) and row['Low'] != 0 else None,
                    float(row['Close']) if pd.notna(row['Close']) and row['Close'] != 0 else None,
                    float(row['Volume']) if pd.notna(row['Volume']) and row['Volume'] != 0 else None
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"  📊 PostgreSQL: {inserted} records inserted from {csv_file}")
        return inserted
        
    except Exception as e:
        logger.error(f"  ❌ PostgreSQL import error: {e}")
        return 0

def main():
    start_time = time.time()
    
    logger.info("🚀 Starting batch C-D Excel processing...")
    
    # Process in small batches to avoid memory issues
    excel_to_csv_batch("data/New_excell_Graph_C_D", batch_size=5)
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 Processing completed in {elapsed:.1f} seconds")
    
    # Final status
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data WHERE symbol LIKE 'C%' OR symbol LIKE 'D%' OR symbol LIKE 'E%'")
        cd_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data")
        total_count = cursor.fetchone()[0]
        
        logger.info(f"📈 Final Status: {cd_count} C-D-E symbols, {total_count} total symbols")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Status check error: {e}")

if __name__ == "__main__":
    main()