#!/usr/bin/env python3
"""
Super simple C-D processor - minimal approach
"""

import os
import pandas as pd
import psycopg2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')

def process_single_file(file_path):
    """Process single Excel file - super simple approach"""
    
    symbol = file_path.name.split('_')[0].upper()
    timeframe_raw = file_path.name.split('_')[1].replace('.xlsx', '')
    
    timeframe_map = {'30Dk': '30min', '60Dk': '60min', 'Günlük': 'daily', '20Dk': '20min'}
    timeframe = timeframe_map.get(timeframe_raw, timeframe_raw)
    
    try:
        # Read Excel
        df = pd.read_excel(file_path, engine='openpyxl')
        logger.info(f"📊 {symbol}: {len(df)} rows loaded")
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Check required columns
        required = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.error(f"❌ {symbol}: Missing columns {missing}")
            return 0
            
        # Convert dates
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        
        if df.empty:
            logger.warning(f"⚠️ {symbol}: No valid dates")
            return 0
            
        # Connect to DB
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        inserted = 0
        
        # Process each row individually
        for idx, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO enhanced_stock_data 
                    (symbol, date, time, timeframe, open, high, low, close, volume, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (symbol, date, time, timeframe) DO NOTHING
                """, (
                    symbol,
                    row['Date'].date(),
                    str(row['Time']) if pd.notna(row['Time']) else None,
                    timeframe,
                    float(row['Open']) if pd.notna(row['Open']) and row['Open'] != 0 else None,
                    float(row['High']) if pd.notna(row['High']) and row['High'] != 0 else None,
                    float(row['Low']) if pd.notna(row['Low']) and row['Low'] != 0 else None,
                    float(row['Close']) if pd.notna(row['Close']) and row['Close'] != 0 else None,
                    float(row['Volume']) if pd.notna(row['Volume']) and row['Volume'] != 0 else None
                ))
                if cursor.rowcount > 0:
                    inserted += 1
                    
            except Exception as e:
                logger.warning(f"Row {idx} error: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ {symbol}-{timeframe}: {inserted} records inserted")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ {symbol} error: {e}")
        return 0

def main():
    # Test with single file first
    test_file = Path("data/New_excell_Graph_C_D/CANTE_60Dk.xlsx")
    
    if test_file.exists():
        logger.info(f"Testing with: {test_file}")
        result = process_single_file(test_file)
        logger.info(f"Result: {result} records")
    else:
        logger.error(f"Test file not found: {test_file}")

if __name__ == "__main__":
    main()