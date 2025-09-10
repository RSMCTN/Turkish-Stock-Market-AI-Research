#!/usr/bin/env python3
"""
FAST C-D-E processor using CSV COPY method
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

def process_all_cde_files():
    """Process all C-D-E files at once to single CSV then COPY"""
    
    excel_dir = Path("data/New_excell_Graph_C_D")
    excel_files = list(excel_dir.glob("*.xlsx"))
    
    logger.info(f"📁 Found {len(excel_files)} Excel files in C-D-E folder")
    
    all_data = []
    processed_count = 0
    
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
            
            # Clean column names - REMOVE WHITESPACE
            df.columns = [col.strip() for col in df.columns]
            
            # Convert dates
            df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
            df = df.dropna(subset=['Date'])
            
            if df.empty:
                logger.warning(f"  ⚠️ No valid dates: {symbol}")
                continue
            
            # Prepare data for PostgreSQL
            records = []
            for _, row in df.iterrows():
                record = [
                    symbol,                                           # symbol
                    row['Date'].strftime('%Y-%m-%d'),                # date
                    str(row.get('Time', '')) if pd.notna(row.get('Time')) else '',  # time
                    timeframe,                                       # timeframe
                    float(row['Open']) if pd.notna(row.get('Open')) and row.get('Open') != 0 else None,
                    float(row['High']) if pd.notna(row.get('High')) and row.get('High') != 0 else None,
                    float(row['Low']) if pd.notna(row.get('Low')) and row.get('Low') != 0 else None,
                    float(row['Close']) if pd.notna(row.get('Close')) and row.get('Close') != 0 else None,
                    float(row['Volume']) if pd.notna(row.get('Volume')) and row.get('Volume') != 0 else None,
                    None, None, None, None, None, None, None, None   # Technical indicators (null for now)
                ]
                records.append(record)
            
            all_data.extend(records)
            processed_count += 1
            
            logger.info(f"  ✅ {symbol}: {len(records)} rows added")
            
        except Exception as e:
            logger.error(f"  ❌ {file_path.name}: {e}")
    
    logger.info(f"🎉 Processed {processed_count} files, {len(all_data):,} total records")
    
    if not all_data:
        logger.error("No data to import!")
        return
    
    # Create CSV
    csv_file = "cde_combined.csv"
    
    # CSV header matching PostgreSQL columns
    header = "symbol,date,time,timeframe,open,high,low,close,volume,rsi_14,macd_26_12,macd_trigger_9,bol_upper_20_2,bol_middle_20_2,bol_lower_20_2,atr_14,adx_14"
    
    with open(csv_file, 'w') as f:
        f.write(header + '\n')
        for record in all_data:
            # Convert None to empty string for CSV
            csv_record = [str(val) if val is not None else '' for val in record]
            f.write(','.join(csv_record) + '\n')
    
    logger.info(f"💾 Saved {len(all_data):,} records to {csv_file}")
    
    # COPY to PostgreSQL
    copy_to_postgresql(csv_file)
    
    # Cleanup
    Path(csv_file).unlink()
    logger.info(f"🧹 Cleaned up {csv_file}")

def copy_to_postgresql(csv_file):
    """Fast COPY to PostgreSQL"""
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        logger.info("⚡ Starting PostgreSQL COPY...")
        
        with open(csv_file, 'r') as f:
            # Skip header
            next(f)
            
            cursor.copy_expert(
                """
                COPY enhanced_stock_data 
                (symbol, date, time, timeframe, open, high, low, close, volume,
                 rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
                 bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
                FROM STDIN WITH CSV
                """, f
            )
        
        conn.commit()
        
        # Check results
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data WHERE symbol LIKE 'C%' OR symbol LIKE 'D%' OR symbol LIKE 'E%'")
        cde_symbols = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data WHERE symbol LIKE 'C%' OR symbol LIKE 'D%' OR symbol LIKE 'E%'")
        cde_records = cursor.fetchone()[0]
        
        logger.info(f"✅ COPY completed!")
        logger.info(f"📊 C-D-E Symbols: {cde_symbols}")
        logger.info(f"📈 C-D-E Records: {cde_records:,}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ COPY error: {e}")

def main():
    start_time = time.time()
    
    logger.info("🚀 Fast C-D-E processing with COPY method...")
    process_all_cde_files()
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 Completed in {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()