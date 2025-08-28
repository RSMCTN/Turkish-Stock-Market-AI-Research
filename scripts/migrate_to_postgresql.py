#!/usr/bin/env python3
"""
BIST Database Migration: SQLite -> PostgreSQL
Migrates 704,691 historical records from local SQLite to Railway PostgreSQL
"""
import os
import sys
import sqlite3
import psycopg2
from psycopg2.extras import execute_batch
import logging
from pathlib import Path
from datetime import datetime
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Railway PostgreSQL connection URL
# Get from Railway environment or manual input
DATABASE_URL = os.getenv('DATABASE_URL') or input("PostgreSQL DATABASE_URL: ")

class DatabaseMigrator:
    def __init__(self, sqlite_path: str, postgres_url: str):
        self.sqlite_path = Path(sqlite_path)
        self.postgres_url = postgres_url
        self.batch_size = 1000  # Process in batches for large dataset
        
    def connect_sqlite(self):
        """Connect to SQLite database"""
        if not self.sqlite_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.sqlite_path}")
        
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn
    
    def connect_postgresql(self):
        """Connect to PostgreSQL database"""
        return psycopg2.connect(self.postgres_url)
    
    def create_postgresql_tables(self):
        """Create PostgreSQL tables with same schema as SQLite"""
        logger.info("🗄️ Creating PostgreSQL tables...")
        
        with self.connect_postgresql() as pg_conn:
            with pg_conn.cursor() as cursor:
                # Create stocks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stocks (
                        symbol VARCHAR(10) PRIMARY KEY,
                        name VARCHAR(100),
                        name_turkish VARCHAR(100),
                        sector VARCHAR(50),
                        is_active BOOLEAN DEFAULT TRUE
                    );
                """)
                
                # Create historical_data table with proper indexes
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS historical_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        date_time TIMESTAMP,
                        open_price DECIMAL(10,4),
                        high_price DECIMAL(10,4),
                        low_price DECIMAL(10,4),
                        close_price DECIMAL(10,4),
                        volume BIGINT,
                        rsi_14 DECIMAL(8,4),
                        rsi_21 DECIMAL(8,4),
                        macd_line DECIMAL(10,6),
                        macd_signal DECIMAL(10,6),
                        macd_histogram DECIMAL(10,6),
                        bollinger_upper DECIMAL(10,4),
                        bollinger_middle DECIMAL(10,4),
                        bollinger_lower DECIMAL(10,4),
                        tenkan_sen DECIMAL(10,4),
                        kijun_sen DECIMAL(10,4),
                        senkou_span_a DECIMAL(10,4),
                        senkou_span_b DECIMAL(10,4),
                        chikou_span DECIMAL(10,4),
                        atr_14 DECIMAL(10,6),
                        adx_14 DECIMAL(8,4),
                        FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                    );
                """)
                
                # Create indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_symbol ON historical_data(symbol);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_datetime ON historical_data(date_time);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_symbol_datetime ON historical_data(symbol, date_time);")
                
                pg_conn.commit()
                logger.info("✅ PostgreSQL tables created successfully")
    
    def get_sqlite_stats(self):
        """Get SQLite database statistics"""
        with self.connect_sqlite() as sqlite_conn:
            cursor = sqlite_conn.cursor()
            
            # Count records
            stocks_count = cursor.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
            historical_count = cursor.execute("SELECT COUNT(*) FROM historical_data").fetchone()[0]
            
            # Date range
            date_range = cursor.execute("""
                SELECT MIN(date_time) as start_date, MAX(date_time) as end_date 
                FROM historical_data
            """).fetchone()
            
            return {
                'stocks_count': stocks_count,
                'historical_count': historical_count,
                'date_range': {'start': date_range[0], 'end': date_range[1]}
            }
    
    def migrate_stocks(self):
        """Migrate stocks table"""
        logger.info("📈 Migrating stocks table...")
        
        with self.connect_sqlite() as sqlite_conn, self.connect_postgresql() as pg_conn:
            sqlite_cursor = sqlite_conn.cursor()
            pg_cursor = pg_conn.cursor()
            
            # Get all stocks from SQLite
            sqlite_cursor.execute("SELECT * FROM stocks")
            stocks = sqlite_cursor.fetchall()
            
            # Insert into PostgreSQL
            insert_query = """
                INSERT INTO stocks (symbol, name, name_turkish, sector, is_active)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                name_turkish = EXCLUDED.name_turkish,
                sector = EXCLUDED.sector,
                is_active = EXCLUDED.is_active;
            """
            
            stock_data = [(row['symbol'], row['name'], row['name_turkish'], row['sector'], row['is_active']) for row in stocks]
            execute_batch(pg_cursor, insert_query, stock_data)
            pg_conn.commit()
            
            logger.info(f"✅ Migrated {len(stocks)} stocks")
    
    def migrate_historical_data(self):
        """Migrate historical_data table in batches"""
        logger.info("📊 Migrating historical data (this will take several minutes)...")
        
        with self.connect_sqlite() as sqlite_conn, self.connect_postgresql() as pg_conn:
            sqlite_cursor = sqlite_conn.cursor()
            pg_cursor = pg_conn.cursor()
            
            # Get total count
            total_records = sqlite_cursor.execute("SELECT COUNT(*) FROM historical_data").fetchone()[0]
            logger.info(f"📈 Total records to migrate: {total_records:,}")
            
            # Migration in batches
            offset = 0
            migrated = 0
            start_time = time.time()
            
            insert_query = """
                INSERT INTO historical_data (
                    symbol, date_time, open_price, high_price, low_price, close_price, volume,
                    rsi_14, rsi_21, macd_line, macd_signal, macd_histogram,
                    bollinger_upper, bollinger_middle, bollinger_lower,
                    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span,
                    atr_14, adx_14
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
            """
            
            while offset < total_records:
                # Fetch batch from SQLite
                sqlite_cursor.execute(f"""
                    SELECT 
                        symbol, date_time, open_price, high_price, low_price, close_price, volume,
                        rsi_14, rsi_21, macd_line, macd_signal, macd_histogram,
                        bollinger_upper, bollinger_middle, bollinger_lower,
                        tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span,
                        atr_14, adx_14
                    FROM historical_data 
                    LIMIT {self.batch_size} OFFSET {offset}
                """)
                
                batch = sqlite_cursor.fetchall()
                if not batch:
                    break
                
                # Convert to tuple format for PostgreSQL
                batch_data = [tuple(row) for row in batch]
                
                # Insert batch into PostgreSQL
                execute_batch(pg_cursor, insert_query, batch_data, page_size=500)
                pg_conn.commit()
                
                migrated += len(batch)
                offset += self.batch_size
                
                # Progress report
                elapsed = time.time() - start_time
                progress = (migrated / total_records) * 100
                rate = migrated / elapsed if elapsed > 0 else 0
                
                logger.info(f"📊 Progress: {migrated:,}/{total_records:,} ({progress:.1f}%) | Rate: {rate:.0f} records/sec")
            
            total_time = time.time() - start_time
            logger.info(f"✅ Migration completed in {total_time:.1f}s | {migrated:,} records migrated")
    
    def verify_migration(self):
        """Verify migration success"""
        logger.info("🔍 Verifying migration...")
        
        with self.connect_sqlite() as sqlite_conn, self.connect_postgresql() as pg_conn:
            sqlite_cursor = sqlite_conn.cursor()
            pg_cursor = pg_conn.cursor()
            
            # Count comparison
            sqlite_stocks = sqlite_cursor.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
            sqlite_historical = sqlite_cursor.execute("SELECT COUNT(*) FROM historical_data").fetchone()[0]
            
            pg_cursor.execute("SELECT COUNT(*) FROM stocks")
            pg_stocks = pg_cursor.fetchone()[0]
            pg_cursor.execute("SELECT COUNT(*) FROM historical_data")
            pg_historical = pg_cursor.fetchone()[0]
            
            logger.info(f"📈 Stocks: SQLite={sqlite_stocks}, PostgreSQL={pg_stocks}")
            logger.info(f"📊 Historical: SQLite={sqlite_historical:,}, PostgreSQL={pg_historical:,}")
            
            if sqlite_stocks == pg_stocks and sqlite_historical == pg_historical:
                logger.info("✅ Migration verification PASSED")
                return True
            else:
                logger.error("❌ Migration verification FAILED")
                return False
    
    def run_migration(self):
        """Run complete migration process"""
        try:
            logger.info("🚀 Starting BIST Database Migration: SQLite → PostgreSQL")
            
            # Get SQLite stats
            stats = self.get_sqlite_stats()
            logger.info(f"📊 Source database: {stats['stocks_count']} stocks, {stats['historical_count']:,} historical records")
            logger.info(f"📅 Date range: {stats['date_range']['start']} → {stats['date_range']['end']}")
            
            # Create PostgreSQL tables
            self.create_postgresql_tables()
            
            # Migrate data
            self.migrate_stocks()
            self.migrate_historical_data()
            
            # Verify migration
            if self.verify_migration():
                logger.info("🎉 DATABASE MIGRATION COMPLETED SUCCESSFULLY!")
                return True
            else:
                logger.error("💥 Migration failed verification")
                return False
                
        except Exception as e:
            logger.error(f"💥 Migration failed: {str(e)}")
            raise

def main():
    """Main migration function"""
    sqlite_path = "data/bist_stocks.db"
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not provided")
        print("Get it from Railway Dashboard → PostgreSQL service → Connect")
        return False
    
    migrator = DatabaseMigrator(sqlite_path, DATABASE_URL)
    return migrator.run_migration()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
