"""
BIST Historical Data Migration Script - Simplified Version
========================================================
Excel → SQLite Migration Tool
"""

import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BISTMigrator:
    """Simplified BIST data migrator"""
    
    def __init__(self, db_path: str = "data/bist_historical.db"):
        self.db_path = db_path
        self.excel_dir = Path("data/excell_MIQ")
        self.stats = {
            'files_processed': 0,
            'records_inserted': 0,
            'errors': 0,
            'start_time': None,
            'failed_files': []
        }
    
    def init_database(self):
        """Initialize SQLite database with schema"""
        logger.info("🗄️ Initializing database...")
        
        schema_sql = """
        -- Stocks table
        CREATE TABLE IF NOT EXISTS stocks (
            symbol VARCHAR(10) PRIMARY KEY,
            name VARCHAR(100),
            name_turkish VARCHAR(100),
            sector VARCHAR(50),
            is_active BOOLEAN DEFAULT TRUE
        );
        
        -- Historical data table
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(10) NOT NULL,
            date_time DATETIME NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            open_price DECIMAL(10,4),
            high_price DECIMAL(10,4),
            low_price DECIMAL(10,4),
            close_price DECIMAL(10,4),
            volume BIGINT,
            rsi_14 DECIMAL(8,4),
            macd_line DECIMAL(10,6),
            macd_signal DECIMAL(10,6),
            bollinger_upper DECIMAL(10,4),
            bollinger_middle DECIMAL(10,4),
            bollinger_lower DECIMAL(10,4),
            atr_14 DECIMAL(10,6),
            adx_14 DECIMAL(8,4),
            
            UNIQUE(symbol, date_time, timeframe)
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_symbol_date ON historical_data(symbol, date_time DESC);
        CREATE INDEX IF NOT EXISTS idx_timeframe ON historical_data(timeframe);
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema_sql)
            conn.commit()
        
        logger.info("✅ Database schema ready")
    
    def parse_date(self, date_str):
        """Parse Turkish date format"""
        if pd.isna(date_str):
            return None
        try:
            if isinstance(date_str, str) and '.' in date_str:
                # Turkish format: dd.mm.yyyy
                day, month, year = date_str.split('.')
                return datetime(int(year), int(month), int(day))
            else:
                return pd.to_datetime(date_str, dayfirst=True)
        except:
            return None
    
    def clean_numeric(self, value):
        """Clean numeric values"""
        if pd.isna(value) or value == '' or value == 0:
            return None
        try:
            return float(value)
        except:
            return None
    
    def process_file(self, file_path: Path):
        """Process single Excel file"""
        try:
            # Extract symbol and timeframe
            filename = file_path.stem
            if '_Günlük' in filename:
                symbol = filename.replace('_Günlük', '')
                timeframe = 'daily'
            elif '_60Dk' in filename:
                symbol = filename.replace('_60Dk', '')
                timeframe = 'hourly'
            else:
                return False
            
            logger.info(f"📈 Processing {symbol} ({timeframe})...")
            
            # Read Excel
            df = pd.read_excel(file_path)
            logger.info(f"   📊 {len(df):,} records loaded")
            
            # Prepare data
            records = []
            for _, row in df.iterrows():
                date_time = self.parse_date(row.get('Date'))
                if date_time is None:
                    continue
                
                # Add time component for hourly data
                if timeframe == 'hourly' and 'Time' in row:
                    time_str = str(row['Time'])
                    if ':' in time_str:
                        try:
                            hour, minute = time_str.split(':')
                            date_time = date_time.replace(hour=int(hour), minute=int(minute))
                        except:
                            pass
                
                record = (
                    symbol,
                    date_time.isoformat(),
                    timeframe,
                    self.clean_numeric(row.get('Open')),
                    self.clean_numeric(row.get('High')),
                    self.clean_numeric(row.get('Low')),
                    self.clean_numeric(row.get('Close')),
                    self.clean_numeric(row.get('Volume')),
                    self.clean_numeric(row.get('RSI (14)')),
                    self.clean_numeric(row.get('MACD (26,12)')),
                    self.clean_numeric(row.get('TRIGGER (9)')),
                    self.clean_numeric(row.get('BOL U (20,2)')),
                    self.clean_numeric(row.get('BOL M (20,2)')),
                    self.clean_numeric(row.get('BOL D (20,2)')),
                    self.clean_numeric(row.get('ATR (14)')),
                    self.clean_numeric(row.get('ADX (14)'))
                )
                
                # Validate OHLC data
                if all(x is not None for x in record[3:7]):  # open, high, low, close
                    records.append(record)
            
            # Insert to database
            if records:
                with sqlite3.connect(self.db_path) as conn:
                    conn.executemany("""
                    INSERT OR REPLACE INTO historical_data 
                    (symbol, date_time, timeframe, open_price, high_price, low_price, 
                     close_price, volume, rsi_14, macd_line, macd_signal, bollinger_upper, 
                     bollinger_middle, bollinger_lower, atr_14, adx_14)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, records)
                    conn.commit()
                
                self.stats['records_inserted'] += len(records)
                logger.info(f"   ✅ {len(records):,} records inserted")
                return True
            
        except Exception as e:
            logger.error(f"   ❌ Error processing {file_path.name}: {e}")
            self.stats['failed_files'].append(str(file_path))
            return False
    
    def populate_stocks_table(self):
        """Populate stocks from basestock.xlsx"""
        try:
            basestock_path = self.excel_dir / "basestock.xlsx"
            if not basestock_path.exists():
                logger.warning("basestock.xlsx not found")
                return
            
            df = pd.read_excel(basestock_path)
            logger.info(f"📊 Loading {len(df)} stocks from basestock.xlsx")
            
            with sqlite3.connect(self.db_path) as conn:
                for _, row in df.iterrows():
                    try:
                        conn.execute("""
                        INSERT OR REPLACE INTO stocks (symbol, name, name_turkish, sector)
                        VALUES (?, ?, ?, ?)
                        """, (
                            row.get('symbol', ''),
                            row.get('name', ''),
                            row.get('name_turkish', ''),
                            row.get('sector', '')
                        ))
                    except Exception as e:
                        logger.warning(f"Stock insert error: {e}")
                conn.commit()
            
            logger.info("✅ Stocks table populated")
            
        except Exception as e:
            logger.error(f"❌ Stocks table error: {e}")
    
    def migrate_all(self, limit: int = None):
        """Main migration method"""
        self.stats['start_time'] = datetime.now()
        logger.info(f"🚀 BIST Migration Starting at {self.stats['start_time']}")
        
        # Initialize database
        self.init_database()
        self.populate_stocks_table()
        
        # Find Excel files
        excel_files = [f for f in self.excel_dir.glob("*.xlsx") if f.name != "basestock.xlsx"]
        
        if limit:
            excel_files = excel_files[:limit]
        
        logger.info(f"📁 Found {len(excel_files)} Excel files")
        
        # Process files
        for i, file_path in enumerate(excel_files, 1):
            logger.info(f"[{i}/{len(excel_files)}] Processing {file_path.name}")
            
            if self.process_file(file_path):
                self.stats['files_processed'] += 1
            else:
                self.stats['errors'] += 1
            
            # Progress update every 10 files
            if i % 10 == 0:
                elapsed = datetime.now() - self.stats['start_time']
                logger.info(f"📊 Progress: {i}/{len(excel_files)} files, {self.stats['records_inserted']:,} records, {elapsed}")
        
        # Final stats
        total_time = datetime.now() - self.stats['start_time']
        logger.info(f"🎉 MIGRATION COMPLETE!")
        logger.info(f"   📊 Files: {self.stats['files_processed']}/{len(excel_files)}")
        logger.info(f"   📈 Records: {self.stats['records_inserted']:,}")
        logger.info(f"   ⏱️ Time: {total_time}")
        logger.info(f"   🚀 Speed: {self.stats['records_inserted'] / total_time.total_seconds():.0f} rec/sec")
        
        if self.stats['failed_files']:
            logger.info(f"   ❌ Failed: {len(self.stats['failed_files'])} files")

def main():
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Run migration
    migrator = BISTMigrator()
    
    # Test with first 10 files, then full migration
    print("🧪 Testing with first 10 files...")
    migrator.migrate_all(limit=10)
    
    response = input("\n✅ Test successful! Proceed with full migration? (y/n): ")
    if response.lower() == 'y':
        print("\n🚀 Starting FULL migration...")
        migrator.stats = {'files_processed': 0, 'records_inserted': 0, 'errors': 0, 'start_time': None, 'failed_files': []}
        migrator.migrate_all()

if __name__ == "__main__":
    main()
