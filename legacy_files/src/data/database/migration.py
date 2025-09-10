"""
BIST Historical Data Migration Script
====================================
Excel → SQLite/PostgreSQL Migration Tool
"""

import sys
import os
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MigrationStats:
    """Track migration progress and statistics"""
    stocks_processed: int = 0
    records_inserted: int = 0
    errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    failed_files: List[str] = None
    
    def __post_init__(self):
        if self.failed_files is None:
            self.failed_files = []

class BISTDataMigrator:
    """Main migration class for BIST historical data"""
    
    def __init__(self, db_path: str = "bist_historical.db", excel_dir: str = "data/excell_MIQ"):
        self.db_path = db_path
        self.excel_dir = Path(excel_dir)
        self.stats = MigrationStats()
        
        # Excel column mapping to database fields
        self.column_mapping = {
            'Date': 'date_time',
            'Time': 'time_component', 
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'volume',
            'WClose': 'weighted_close',
            'RSI (14)': 'rsi_14',
            'Tenkan-sen': 'tenkan_sen',
            'Kijun-sen': 'kijun_sen',
            'Senkou Span A': 'senkou_span_a',
            'Senkou Span B': 'senkou_span_b',
            'Chikou Span': 'chikou_span',
            'MACD (26,12)': 'macd_line',
            'TRIGGER (9)': 'macd_signal',
            'BOL U (20,2)': 'bollinger_upper',
            'BOL M (20,2)': 'bollinger_middle', 
            'BOL D (20,2)': 'bollinger_lower',
            'ATR (14)': 'atr_14',
            'ADX (14)': 'adx_14'
        }
    
    def init_database(self):
        """Initialize database with schema"""
        logger.info("🗄️ Initializing database schema...")
        
        # Read schema file
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Execute schema
        with sqlite3.connect(self.db_path) as conn:
            # Split and execute each statement
            statements = schema_sql.split(';')
            for statement in statements:
                statement = statement.strip()
                if statement:
                    try:
                        conn.execute(statement)
                    except Exception as e:
                        if "already exists" not in str(e):
                            logger.error(f"Schema error: {e}")
                            logger.error(f"Statement: {statement[:100]}...")
            conn.commit()
        
        logger.info("✅ Database schema initialized successfully")
    
    def parse_turkish_datetime(self, date_str: str, time_str: str = "00:00") -> datetime:
        """Parse Turkish formatted dates and times"""
        try:
            # Handle different date formats
            if isinstance(date_str, str):
                # Convert Turkish date format dd.mm.yyyy to datetime
                if '.' in date_str:
                    day, month, year = date_str.split('.')
                    date_obj = datetime(int(year), int(month), int(day))
                else:
                    date_obj = pd.to_datetime(date_str, dayfirst=True)
            else:
                # Already datetime or similar
                date_obj = pd.to_datetime(date_str)
            
            # Add time component if provided
            if time_str and time_str != "00:00":
                if ':' in str(time_str):
                    hour, minute = str(time_str).split(':')
                    date_obj = date_obj.replace(hour=int(hour), minute=int(minute))
            
            return date_obj
            
        except Exception as e:
            logger.warning(f"Date parsing error for '{date_str}' '{time_str}': {e}")
            return datetime.now()
    
    def clean_numeric_value(self, value: Any) -> Optional[float]:
        """Clean and convert numeric values"""
        if pd.isna(value) or value == '' or value == 0:
            return None
        
        try:
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point
                cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                return float(cleaned) if cleaned else None
            return float(value)
        except:
            return None
    
    def process_excel_file(self, file_path: Path) -> Tuple[str, str, List[Dict]]:
        """Process single Excel file and return clean data"""
        try:
            # Extract symbol and timeframe from filename
            filename = file_path.stem  # without .xlsx
            if '_Günlük' in filename:
                symbol = filename.replace('_Günlük', '')
                timeframe = 'daily'
            elif '_60Dk' in filename:
                symbol = filename.replace('_60Dk', '')
                timeframe = 'hourly' 
            else:
                raise ValueError(f"Unknown timeframe in filename: {filename}")
            
            logger.info(f"📈 Processing {symbol} {timeframe} data...")
            
            # Read Excel file
            df = pd.read_excel(file_path)
            logger.info(f"   📊 Loaded {len(df):,} records")
            
            # Process each row
            processed_records = []
            for idx, row in df.iterrows():
                try:
                    # Parse datetime
                    date_str = str(row.get('Date', ''))
                    time_str = str(row.get('Time', '00:00'))
                    dt = self.parse_turkish_datetime(date_str, time_str)
                    
                    # Clean numeric data
                    record = {
                        'symbol': symbol,
                        'date_time': dt.isoformat(),
                        'timeframe': timeframe,
                        'open_price': self.clean_numeric_value(row.get('Open')),
                        'high_price': self.clean_numeric_value(row.get('High')),
                        'low_price': self.clean_numeric_value(row.get('Low')),
                        'close_price': self.clean_numeric_value(row.get('Close')),
                        'volume': self.clean_numeric_value(row.get('Volume')),
                        'weighted_close': self.clean_numeric_value(row.get('WClose')),
                        'rsi_14': self.clean_numeric_value(row.get('RSI (14)')),
                        'tenkan_sen': self.clean_numeric_value(row.get('Tenkan-sen')),
                        'kijun_sen': self.clean_numeric_value(row.get('Kijun-sen')),
                        'senkou_span_a': self.clean_numeric_value(row.get('Senkou Span A')),
                        'senkou_span_b': self.clean_numeric_value(row.get('Senkou Span B')),
                        'chikou_span': self.clean_numeric_value(row.get('Chikou Span')),
                        'macd_line': self.clean_numeric_value(row.get('MACD (26,12)')),
                        'macd_signal': self.clean_numeric_value(row.get('TRIGGER (9)')),
                        'bollinger_upper': self.clean_numeric_value(row.get('BOL U (20,2)')),
                        'bollinger_middle': self.clean_numeric_value(row.get('BOL M (20,2)')),
                        'bollinger_lower': self.clean_numeric_value(row.get('BOL D (20,2)')),
                        'atr_14': self.clean_numeric_value(row.get('ATR (14)')),
                        'adx_14': self.clean_numeric_value(row.get('ADX (14)')),
                    }
                    
                    # Validate required fields
                    if all(record[field] is not None for field in ['open_price', 'high_price', 'low_price', 'close_price']):
                        processed_records.append(record)
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Row {idx} error: {e}")
                    continue
            
            logger.info(f"   ✅ Processed {len(processed_records):,}/{len(df):,} valid records")
            return symbol, timeframe, processed_records
            
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path}: {e}")
            self.stats.failed_files.append(str(file_path))
            return "", "", []
    
    def insert_records_batch(self, records: List[Dict], batch_size: int = 1000):
        """Insert records in batches for better performance"""
        if not records:
            return
        
        insert_sql = """
        INSERT OR REPLACE INTO historical_data 
        (symbol, date_time, timeframe, open_price, high_price, low_price, close_price, 
         volume, weighted_close, rsi_14, tenkan_sen, kijun_sen, senkou_span_a, 
         senkou_span_b, chikou_span, macd_line, macd_signal, bollinger_upper, 
         bollinger_middle, bollinger_lower, atr_14, adx_14)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        with sqlite3.connect(self.db_path) as conn:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_data = []
                
                for record in batch:
                    batch_data.append((
                        record['symbol'], record['date_time'], record['timeframe'],
                        record['open_price'], record['high_price'], record['low_price'], record['close_price'],
                        record['volume'], record['weighted_close'], record['rsi_14'],
                        record['tenkan_sen'], record['kijun_sen'], record['senkou_span_a'],
                        record['senkou_span_b'], record['chikou_span'], record['macd_line'], record['macd_signal'],
                        record['bollinger_upper'], record['bollinger_middle'], record['bollinger_lower'],
                        record['atr_14'], record['adx_14']
                    ))
                
                conn.executemany(insert_sql, batch_data)
                self.stats.records_inserted += len(batch)
                
                if i % (batch_size * 10) == 0:
                    logger.info(f"   💾 Inserted {self.stats.records_inserted:,} records...")
            
            conn.commit()
    
    def populate_stocks_table(self):
        """Populate stocks master table from basestock.xlsx"""
        logger.info("👥 Populating stocks master table...")
        
        try:
            basestock_path = self.excel_dir / "basestock.xlsx"
            if not basestock_path.exists():
                logger.warning(f"basestock.xlsx not found at {basestock_path}")
                return
            
            df = pd.read_excel(basestock_path)
            logger.info(f"📊 Loaded {len(df)} stocks from basestock.xlsx")
            
            # Insert stocks data
            with sqlite3.connect(self.db_path) as conn:
                for _, row in df.iterrows():
                    try:
                        conn.execute("""
                        INSERT OR REPLACE INTO stocks 
                        (symbol, name, name_turkish, sector, sector_turkish, market_cap, market_segment, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row.get('symbol', ''),
                            row.get('name', ''),
                            row.get('name_turkish', ''),
                            row.get('sector', ''),
                            row.get('sector_turkish', ''),
                            self.clean_numeric_value(row.get('market_cap')),
                            row.get('market_segment', ''),
                            True
                        ))
                    except Exception as e:
                        logger.warning(f"Stock insert error for {row.get('symbol', 'Unknown')}: {e}")
                
                conn.commit()
            
            logger.info("✅ Stocks master table populated")
            
        except Exception as e:
            logger.error(f"❌ Failed to populate stocks table: {e}")
    
    def migrate_all_files(self, max_workers: int = 4):
        """Main migration method - process all Excel files"""
        self.stats.start_time = datetime.now()
        logger.info(f"🚀 Starting BIST data migration at {self.stats.start_time}")
        logger.info(f"📁 Excel directory: {self.excel_dir.absolute()}")
        logger.info(f"🗄️ Database path: {Path(self.db_path).absolute()}")
        
        # Initialize database
        self.init_database()
        
        # Populate stocks table
        self.populate_stocks_table()
        
        # Find all Excel files (excluding basestock)
        excel_files = [f for f in self.excel_dir.glob("*.xlsx") if f.name != "basestock.xlsx"]
        logger.info(f"📊 Found {len(excel_files)} Excel files to process")
        
        # Process files with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {executor.submit(self.process_excel_file, file): file for file in excel_files}
            
            # Process completed futures
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    symbol, timeframe, records = future.result()
                    if records:
                        # Insert records to database
                        self.insert_records_batch(records)
                        self.stats.stocks_processed += 1
                        logger.info(f"✅ {symbol} {timeframe}: {len(records):,} records inserted")
                    else:
                        self.stats.errors += 1
                        logger.error(f"❌ {file_path.name}: No valid records")
                        
                except Exception as e:
                    self.stats.errors += 1
                    logger.error(f"❌ {file_path.name}: {e}")
        
        # Migration complete
        self.stats.end_time = datetime.now()
        duration = self.stats.end_time - self.stats.start_time
        
        logger.info("🎉 MIGRATION COMPLETED!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   • Files processed: {self.stats.stocks_processed}")
        logger.info(f"   • Records inserted: {self.stats.records_inserted:,}")
        logger.info(f"   • Errors: {self.stats.errors}")
        logger.info(f"   • Duration: {duration}")
        logger.info(f"   • Speed: {self.stats.records_inserted / duration.total_seconds():.0f} records/second")
        
        if self.stats.failed_files:
            logger.info(f"❌ Failed files: {self.stats.failed_files}")
        
        # Save stats
        self.save_migration_stats()
    
    def save_migration_stats(self):
        """Save migration statistics to JSON file"""
        stats_dict = {
            'stocks_processed': self.stats.stocks_processed,
            'records_inserted': self.stats.records_inserted,
            'errors': self.stats.errors,
            'start_time': self.stats.start_time.isoformat() if self.stats.start_time else None,
            'end_time': self.stats.end_time.isoformat() if self.stats.end_time else None,
            'failed_files': self.stats.failed_files,
            'database_path': str(Path(self.db_path).absolute()),
            'excel_directory': str(self.excel_dir.absolute())
        }
        
        with open('migration_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        logger.info("📈 Migration statistics saved to migration_stats.json")

def main():
    """Main execution function"""
    # Configuration
    DB_PATH = "data/bist_historical.db"
    EXCEL_DIR = "data/excell_MIQ"
    MAX_WORKERS = 4  # Parallel processing threads
    
    # Create migrator instance
    migrator = BISTDataMigrator(DB_PATH, EXCEL_DIR)
    
    # Run migration
    try:
        migrator.migrate_all_files(max_workers=MAX_WORKERS)
    except KeyboardInterrupt:
        logger.info("🛑 Migration interrupted by user")
    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        raise

if __name__ == "__main__":
    main()
