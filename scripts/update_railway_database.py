#!/usr/bin/env python3
"""
Railway PostgreSQL Database Update Script
Loads BIST real data from CSV to PostgreSQL
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import sys
from datetime import datetime, timedelta
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Railway PostgreSQL connection
DATABASE_URL = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"

def connect_to_postgresql():
    """Connect to Railway PostgreSQL"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        logger.info("✅ Connected to Railway PostgreSQL")
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
        return None

def create_tables_if_not_exist(conn):
    """Create necessary tables"""
    cursor = conn.cursor()
    
    # Create stocks table
    stocks_table = """
    CREATE TABLE IF NOT EXISTS stocks (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) UNIQUE NOT NULL,
        name VARCHAR(255),
        name_turkish VARCHAR(255),
        sector VARCHAR(100),
        sector_turkish VARCHAR(100),
        market_cap BIGINT DEFAULT 0,
        last_price DECIMAL(10,4) DEFAULT 0,
        change_value DECIMAL(10,4) DEFAULT 0,
        change_percent DECIMAL(8,4) DEFAULT 0,
        volume BIGINT DEFAULT 0,
        high_52w DECIMAL(10,4) DEFAULT 0,
        low_52w DECIMAL(10,4) DEFAULT 0,
        is_active BOOLEAN DEFAULT TRUE,
        bist_markets TEXT[] DEFAULT '{}',
        market_segment VARCHAR(50),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    # Create sectors table  
    sectors_table = """
    CREATE TABLE IF NOT EXISTS sectors (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) UNIQUE NOT NULL,
        name_turkish VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    cursor.execute(stocks_table)
    cursor.execute(sectors_table)
    conn.commit()
    cursor.close()
    logger.info("✅ Tables created/verified")

def load_csv_data():
    """Load BIST real data from CSV"""
    csv_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/bist_real_data.csv"
    
    try:
        # Read CSV with Turkish headers
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded CSV: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load CSV: {e}")
        return None

def process_stock_data(df, conn):
    """Process and insert stock data"""
    cursor = conn.cursor()
    
    inserted = 0
    updated = 0
    
    for index, row in df.iterrows():
        try:
            symbol = row['SEMBOL'].strip()
            name = row['ACKL'].strip() if pd.notna(row['ACKL']) else ''
            sector = row['SEKTOR'].strip() if pd.notna(row['SEKTOR']) else 'Diğer'
            last_price = float(row['SON']) if pd.notna(row['SON']) else 0
            change_percent = float(row['%FARK']) if pd.notna(row['%FARK']) else 0
            volume = int(row['T.ADET']) if pd.notna(row['T.ADET']) else 0
            high_52w = float(row['52HAFTALIK.YUKSEK']) if pd.notna(row['52HAFTALIK.YUKSEK']) else 0
            low_52w = float(row['52HAFTALIK.DUSUK']) if pd.notna(row['52HAFTALIK.DUSUK']) else 0
            
            # Market classification
            bist_markets = []
            if pd.notna(row['XU030 DAKI AG.']) and row['XU030 DAKI AG.'] > 0:
                bist_markets.append('bist_30')
            if pd.notna(row['XU050 DEKI AG.']) and row['XU050 DEKI AG.'] > 0:
                bist_markets.append('bist_50')
            if pd.notna(row['XU100 DEKI AG.']) and row['XU100 DEKI AG.'] > 0:
                bist_markets.append('bist_100')
            
            # Insert or update stock
            upsert_query = """
            INSERT INTO stocks (
                symbol, name, name_turkish, sector, sector_turkish, 
                last_price, change_percent, volume, high_52w, low_52w,
                bist_markets, market_segment, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (symbol) 
            DO UPDATE SET
                name = EXCLUDED.name,
                name_turkish = EXCLUDED.name_turkish,
                sector = EXCLUDED.sector,
                sector_turkish = EXCLUDED.sector_turkish,
                last_price = EXCLUDED.last_price,
                change_percent = EXCLUDED.change_percent,
                volume = EXCLUDED.volume,
                high_52w = EXCLUDED.high_52w,
                low_52w = EXCLUDED.low_52w,
                bist_markets = EXCLUDED.bist_markets,
                market_segment = EXCLUDED.market_segment,
                updated_at = NOW()
            """
            
            cursor.execute(upsert_query, (
                symbol, name, name, sector, sector,
                last_price, change_percent, volume, high_52w, low_52w,
                bist_markets, 'BIST', 
            ))
            
            if cursor.rowcount == 1:
                inserted += 1
            else:
                updated += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            continue
    
    # Insert unique sectors
    cursor.execute("SELECT DISTINCT sector FROM stocks WHERE sector IS NOT NULL")
    sectors = cursor.fetchall()
    
    for sector_row in sectors:
        sector = sector_row[0]
        cursor.execute("""
            INSERT INTO sectors (name, name_turkish) 
            VALUES (%s, %s) 
            ON CONFLICT (name) DO NOTHING
        """, (sector, sector))
    
    conn.commit()
    cursor.close()
    
    logger.info(f"✅ Processed stocks: {inserted} inserted, {updated} updated")
    return inserted + updated

def main():
    """Main execution"""
    logger.info("🚀 Starting Railway PostgreSQL update...")
    
    # Connect to database
    conn = connect_to_postgresql()
    if not conn:
        sys.exit(1)
    
    try:
        # Create tables
        create_tables_if_not_exist(conn)
        
        # Load CSV data
        df = load_csv_data()
        if df is None:
            sys.exit(1)
        
        # Process stock data
        total_processed = process_stock_data(df, conn)
        
        # Verify data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stocks")
        total_stocks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sectors")
        total_sectors = cursor.fetchone()[0]
        
        cursor.close()
        
        logger.info(f"✅ Database update complete!")
        logger.info(f"📊 Total stocks: {total_stocks}")
        logger.info(f"🏷️ Total sectors: {total_sectors}")
        
        # Sample data check
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT symbol, name, last_price, sector FROM stocks WHERE symbol IN ('AKSEN', 'ASTOR', 'GARAN') ORDER BY symbol")
        samples = cursor.fetchall()
        
        logger.info("🎯 Sample data:")
        for stock in samples:
            logger.info(f"   {stock['symbol']}: {stock['last_price']} TL - {stock['sector']}")
        
        cursor.close()
        
    except Exception as e:
        logger.error(f"❌ Main execution error: {e}")
        sys.exit(1)
    
    finally:
        conn.close()
        logger.info("🔚 Database connection closed")

if __name__ == "__main__":
    main()
