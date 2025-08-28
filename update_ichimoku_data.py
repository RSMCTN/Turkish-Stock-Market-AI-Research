"""
Update Database with Ichimoku Cloud Data
=======================================
Populate Ichimoku columns from Excel files
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def clean_numeric_value(value):
    """Clean numeric values"""
    if pd.isna(value) or value == '' or value == 0:
        return None
    try:
        return float(value)
    except:
        return None

def parse_turkish_datetime(date_str, time_str="00:00"):
    """Parse Turkish formatted dates and times"""
    try:
        if isinstance(date_str, str) and '.' in date_str:
            day, month, year = date_str.split('.')
            date_obj = datetime(int(year), int(month), int(day))
        else:
            date_obj = pd.to_datetime(date_str, dayfirst=True)
        
        if time_str and time_str != "00:00":
            if ':' in str(time_str):
                hour, minute = str(time_str).split(':')
                date_obj = date_obj.replace(hour=int(hour), minute=int(minute))
        
        return date_obj.isoformat()
    except:
        return None

def update_ichimoku_for_symbol(symbol: str, db_path: str, excel_dir: str):
    """Update Ichimoku data for a single symbol"""
    excel_files = [
        (f"{symbol}_Günlük.xlsx", "daily"),
        (f"{symbol}_60Dk.xlsx", "hourly")
    ]
    
    conn = sqlite3.connect(db_path)
    updates_count = 0
    
    for excel_file, timeframe in excel_files:
        excel_path = Path(excel_dir) / excel_file
        
        if not excel_path.exists():
            logger.warning(f"Excel file not found: {excel_path}")
            continue
            
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            logger.info(f"Processing {symbol} {timeframe}: {len(df)} records")
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Parse datetime
                    date_str = str(row.get('Date', ''))
                    time_str = str(row.get('Time', '00:00'))
                    dt = parse_turkish_datetime(date_str, time_str)
                    
                    if not dt:
                        continue
                    
                    # Extract Ichimoku values
                    tenkan_sen = clean_numeric_value(row.get('Tenkan-sen'))
                    kijun_sen = clean_numeric_value(row.get('Kijun-sen'))
                    senkou_span_a = clean_numeric_value(row.get('Senkou Span A'))
                    senkou_span_b = clean_numeric_value(row.get('Senkou Span B'))
                    chikou_span = clean_numeric_value(row.get('Chikou Span'))
                    
                    # Update database record
                    cursor = conn.execute("""
                        UPDATE historical_data 
                        SET tenkan_sen = ?, kijun_sen = ?, senkou_span_a = ?, 
                            senkou_span_b = ?, chikou_span = ?
                        WHERE symbol = ? AND date_time = ? AND timeframe = ?
                    """, (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, 
                          chikou_span, symbol, dt, timeframe))
                    
                    if cursor.rowcount > 0:
                        updates_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing row for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing {excel_file}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    logger.info(f"✅ {symbol}: {updates_count} records updated with Ichimoku data")
    return updates_count

def main():
    """Main execution"""
    db_path = "data/bist_historical.db"
    excel_dir = "data/excell_MIQ"
    
    # Get list of symbols from Excel files
    excel_path = Path(excel_dir)
    symbols = set()
    
    for file in excel_path.glob("*_Günlük.xlsx"):
        symbol = file.stem.replace('_Günlük', '')
        symbols.add(symbol)
    
    logger.info(f"🚀 Starting Ichimoku data update for {len(symbols)} symbols")
    
    total_updates = 0
    processed = 0
    
    for symbol in sorted(symbols):
        logger.info(f"[{processed+1}/{len(symbols)}] Processing {symbol}...")
        updates = update_ichimoku_for_symbol(symbol, db_path, excel_dir)
        total_updates += updates
        processed += 1
        
        # Progress update every 10 symbols
        if processed % 10 == 0:
            logger.info(f"📊 Progress: {processed}/{len(symbols)} symbols, {total_updates:,} updates")
    
    logger.info(f"🎉 Ichimoku update completed!")
    logger.info(f"📊 Total updates: {total_updates:,}")
    logger.info(f"📈 Symbols processed: {processed}")

if __name__ == "__main__":
    main()
