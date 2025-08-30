#!/usr/bin/env python3
"""
BIST DATABASE PARTITIONING SYSTEM - MAMUT R600
==============================================
Optimize large dataset with smart partitioning and sharding
Handles 800K+ records efficiently with year-based and symbol-based partitions
"""

import sqlite3
import os
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BISTDatabasePartitioner:
    """
    Advanced database partitioning for BIST data
    Strategies: Year-based + Symbol-group-based partitions
    """
    
    def __init__(self, source_db: str = "enhanced_bist_data.db"):
        self.source_db = source_db
        self.partition_dir = Path("data/partitions")
        self.partition_dir.mkdir(parents=True, exist_ok=True)
        
        # Partition configuration
        self.partitions = {
            'by_year': True,
            'by_symbol_group': True,
            'by_timeframe': True,
            'max_records_per_partition': 100000
        }
        
        logger.info("🗄️ BIST Database Partitioner initialized")
    
    def analyze_source_database(self) -> Dict:
        """Analyze source database for partitioning strategy"""
        logger.info("📊 Analyzing source database...")
        
        conn = sqlite3.connect(self.source_db)
        
        analysis = {}
        
        # Total records
        total_records = conn.execute("SELECT COUNT(*) FROM enhanced_stock_data").fetchone()[0]
        analysis['total_records'] = total_records
        
        # Year distribution
        year_dist = conn.execute("""
            SELECT substr(date, 1, 4) as year, COUNT(*) as count
            FROM enhanced_stock_data 
            GROUP BY substr(date, 1, 4)
            ORDER BY year
        """).fetchall()
        analysis['year_distribution'] = dict(year_dist)
        
        # Symbol distribution
        symbol_dist = conn.execute("""
            SELECT symbol, COUNT(*) as count
            FROM enhanced_stock_data 
            GROUP BY symbol
            ORDER BY count DESC
        """).fetchall()
        analysis['symbol_distribution'] = dict(symbol_dist)
        
        # Timeframe distribution
        timeframe_dist = conn.execute("""
            SELECT timeframe, COUNT(*) as count
            FROM enhanced_stock_data 
            GROUP BY timeframe
            ORDER BY count DESC
        """).fetchall()
        analysis['timeframe_distribution'] = dict(timeframe_dist)
        
        # Database size
        db_size = os.path.getsize(self.source_db) / (1024 * 1024)  # MB
        analysis['database_size_mb'] = db_size
        
        conn.close()
        
        logger.info(f"📈 Analysis completed: {total_records:,} records, {db_size:.1f}MB")
        return analysis
    
    def create_partition_databases(self) -> List[str]:
        """Create optimized partition databases"""
        analysis = self.analyze_source_database()
        partition_files = []
        
        logger.info("🔧 Creating partition databases...")
        
        # Strategy 1: Year-based partitions
        if self.partitions['by_year']:
            for year, count in analysis['year_distribution'].items():
                if count > 0:
                    partition_file = self.partition_dir / f"bist_data_{year}.db"
                    self.create_year_partition(year, str(partition_file))
                    partition_files.append(str(partition_file))
                    logger.info(f"✅ Year {year} partition: {count:,} records → {partition_file.name}")
        
        # Strategy 2: Large symbol groups (for heavy symbols)
        large_symbols = [(symbol, count) for symbol, count in analysis['symbol_distribution'].items() 
                        if count > 20000]  # Symbols with >20K records
        
        if large_symbols:
            for symbol, count in large_symbols:
                partition_file = self.partition_dir / f"bist_symbol_{symbol.lower()}.db"
                self.create_symbol_partition(symbol, str(partition_file))
                partition_files.append(str(partition_file))
                logger.info(f"✅ Symbol {symbol} partition: {count:,} records → {partition_file.name}")
        
        # Strategy 3: Timeframe-based partitions for analysis
        if self.partitions['by_timeframe']:
            for timeframe, count in analysis['timeframe_distribution'].items():
                if count > 10000:  # Only for significant timeframes
                    partition_file = self.partition_dir / f"bist_timeframe_{timeframe}.db"
                    self.create_timeframe_partition(timeframe, str(partition_file))
                    partition_files.append(str(partition_file))
                    logger.info(f"✅ Timeframe {timeframe} partition: {count:,} records → {partition_file.name}")
        
        return partition_files
    
    def create_year_partition(self, year: str, partition_file: str):
        """Create year-based partition"""
        # Create partition database
        partition_conn = sqlite3.connect(partition_file)
        
        # Copy schema from source
        source_conn = sqlite3.connect(self.source_db)
        schema = source_conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='enhanced_stock_data'").fetchone()[0]
        partition_conn.execute(schema)
        
        # Copy data for specific year
        data = source_conn.execute("""
            SELECT * FROM enhanced_stock_data 
            WHERE substr(date, 1, 4) = ?
        """, (year,)).fetchall()
        
        # Get column names
        columns = [desc[0] for desc in source_conn.execute("SELECT * FROM enhanced_stock_data LIMIT 1").description]
        
        # Insert data into partition
        placeholders = ','.join(['?' for _ in columns])
        partition_conn.executemany(f"INSERT INTO enhanced_stock_data VALUES ({placeholders})", data)
        
        # Create optimized indexes for this partition
        self.create_partition_indexes(partition_conn, 'year')
        
        partition_conn.commit()
        partition_conn.close()
        source_conn.close()
    
    def create_symbol_partition(self, symbol: str, partition_file: str):
        """Create symbol-based partition"""
        partition_conn = sqlite3.connect(partition_file)
        source_conn = sqlite3.connect(self.source_db)
        
        # Copy schema
        schema = source_conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='enhanced_stock_data'").fetchone()[0]
        partition_conn.execute(schema)
        
        # Copy data for specific symbol
        data = source_conn.execute("SELECT * FROM enhanced_stock_data WHERE symbol = ?", (symbol,)).fetchall()
        columns = [desc[0] for desc in source_conn.execute("SELECT * FROM enhanced_stock_data LIMIT 1").description]
        placeholders = ','.join(['?' for _ in columns])
        partition_conn.executemany(f"INSERT INTO enhanced_stock_data VALUES ({placeholders})", data)
        
        # Create optimized indexes
        self.create_partition_indexes(partition_conn, 'symbol')
        
        partition_conn.commit()
        partition_conn.close()
        source_conn.close()
    
    def create_timeframe_partition(self, timeframe: str, partition_file: str):
        """Create timeframe-based partition"""
        partition_conn = sqlite3.connect(partition_file)
        source_conn = sqlite3.connect(self.source_db)
        
        # Copy schema
        schema = source_conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='enhanced_stock_data'").fetchone()[0]
        partition_conn.execute(schema)
        
        # Copy data for specific timeframe
        data = source_conn.execute("SELECT * FROM enhanced_stock_data WHERE timeframe = ?", (timeframe,)).fetchall()
        columns = [desc[0] for desc in source_conn.execute("SELECT * FROM enhanced_stock_data LIMIT 1").description]
        placeholders = ','.join(['?' for _ in columns])
        partition_conn.executemany(f"INSERT INTO enhanced_stock_data VALUES ({placeholders})", data)
        
        # Create optimized indexes
        self.create_partition_indexes(partition_conn, 'timeframe')
        
        partition_conn.commit()
        partition_conn.close()
        source_conn.close()
    
    def create_partition_indexes(self, conn: sqlite3.Connection, partition_type: str):
        """Create optimized indexes for each partition type"""
        if partition_type == 'year':
            indexes = [
                "CREATE INDEX idx_symbol_date ON enhanced_stock_data (symbol, date)",
                "CREATE INDEX idx_timeframe ON enhanced_stock_data (timeframe)",
                "CREATE INDEX idx_rsi_macd ON enhanced_stock_data (rsi_14, macd_26_12)",
            ]
        elif partition_type == 'symbol':
            indexes = [
                "CREATE INDEX idx_date_timeframe ON enhanced_stock_data (date, timeframe)",
                "CREATE INDEX idx_technical_indicators ON enhanced_stock_data (rsi_14, macd_26_12, atr_14)",
                "CREATE INDEX idx_ohlcv ON enhanced_stock_data (date, open, high, low, close, volume)",
            ]
        elif partition_type == 'timeframe':
            indexes = [
                "CREATE INDEX idx_symbol_date ON enhanced_stock_data (symbol, date)",
                "CREATE INDEX idx_close_volume ON enhanced_stock_data (close, volume)",
                "CREATE INDEX idx_bollinger ON enhanced_stock_data (bol_upper_20_2, bol_lower_20_2)",
            ]
        else:
            indexes = [
                "CREATE INDEX idx_basic ON enhanced_stock_data (symbol, date, timeframe)",
            ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.OperationalError as e:
                logger.warning(f"Index creation warning: {e}")
    
    def create_master_view(self, partition_files: List[str]) -> str:
        """Create a master view that combines all partitions"""
        master_db = self.partition_dir / "bist_master_view.db"
        
        conn = sqlite3.connect(str(master_db))
        
        # Create master table structure
        source_conn = sqlite3.connect(self.source_db)
        schema = source_conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='enhanced_stock_data'").fetchone()[0]
        conn.execute(schema.replace("enhanced_stock_data", "master_view"))
        source_conn.close()
        
        # Attach all partition databases
        for i, partition_file in enumerate(partition_files):
            conn.execute(f"ATTACH DATABASE '{partition_file}' AS partition_{i}")
        
        logger.info(f"✅ Master view created: {len(partition_files)} partitions attached")
        
        conn.close()
        return str(master_db)
    
    def create_partition_query_router(self) -> str:
        """Create a smart query router for partitions"""
        router_code = '''
class BISTPartitionRouter:
    """Smart query router for BIST partitioned data"""
    
    def __init__(self, partition_dir="data/partitions"):
        self.partition_dir = Path(partition_dir)
        self.partitions = {
            'year': list(self.partition_dir.glob("bist_data_*.db")),
            'symbol': list(self.partition_dir.glob("bist_symbol_*.db")),
            'timeframe': list(self.partition_dir.glob("bist_timeframe_*.db"))
        }
    
    def route_query(self, symbol=None, year=None, timeframe=None, query_type="select"):
        """Route query to optimal partition"""
        
        # Symbol-specific query
        if symbol:
            symbol_db = self.partition_dir / f"bist_symbol_{symbol.lower()}.db"
            if symbol_db.exists():
                return str(symbol_db)
        
        # Year-specific query
        if year:
            year_db = self.partition_dir / f"bist_data_{year}.db"
            if year_db.exists():
                return str(year_db)
        
        # Timeframe-specific query
        if timeframe:
            timeframe_db = self.partition_dir / f"bist_timeframe_{timeframe}.db"
            if timeframe_db.exists():
                return str(timeframe_db)
        
        # Default to master view
        return str(self.partition_dir / "bist_master_view.db")
    
    def execute_partitioned_query(self, query, params=None, **filters):
        """Execute query on optimal partition"""
        db_file = self.route_query(**filters)
        
        conn = sqlite3.connect(db_file)
        if params:
            result = conn.execute(query, params).fetchall()
        else:
            result = conn.execute(query).fetchall()
        conn.close()
        
        return result
        '''
        
        router_file = self.partition_dir / "partition_router.py"
        with open(router_file, 'w') as f:
            f.write("from pathlib import Path\nimport sqlite3\n\n" + router_code)
        
        logger.info(f"✅ Partition router created: {router_file}")
        return str(router_file)
    
    def generate_partition_report(self) -> Dict:
        """Generate comprehensive partition report"""
        report = {
            'total_partitions': 0,
            'total_size_mb': 0,
            'partitions': []
        }
        
        for partition_file in self.partition_dir.glob("*.db"):
            if partition_file.name == "bist_master_view.db":
                continue
                
            size_mb = partition_file.stat().st_size / (1024 * 1024)
            
            # Count records
            conn = sqlite3.connect(str(partition_file))
            try:
                record_count = conn.execute("SELECT COUNT(*) FROM enhanced_stock_data").fetchone()[0]
            except:
                record_count = 0
            conn.close()
            
            report['partitions'].append({
                'file': partition_file.name,
                'size_mb': size_mb,
                'records': record_count
            })
            
            report['total_size_mb'] += size_mb
        
        report['total_partitions'] = len(report['partitions'])
        
        return report

def main():
    """Main partitioning process"""
    partitioner = BISTDatabasePartitioner()
    
    # Analyze database
    analysis = partitioner.analyze_source_database()
    
    print(f"""
    📊 BIST DATABASE ANALYSIS
    ========================
    📈 Total records: {analysis['total_records']:,}
    💾 Database size: {analysis['database_size_mb']:.1f} MB
    📅 Years: {len(analysis['year_distribution'])}
    🏢 Symbols: {len(analysis['symbol_distribution'])}
    ⏰ Timeframes: {len(analysis['timeframe_distribution'])}
    """)
    
    # Ask user for partitioning strategy
    print("🚀 Partitioning Options:")
    print("1. Create all partitions (recommended)")
    print("2. Year-based partitions only")
    print("3. Symbol-based partitions only") 
    print("4. Analysis only")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        # Full partitioning
        partition_files = partitioner.create_partition_databases()
        master_db = partitioner.create_master_view(partition_files)
        router_file = partitioner.create_partition_query_router()
        
        report = partitioner.generate_partition_report()
        print(f"""
        🎉 PARTITIONING COMPLETE!
        =========================
        ✅ Partitions created: {report['total_partitions']}
        💾 Total partition size: {report['total_size_mb']:.1f} MB
        🗄️ Master view: {master_db}
        🔀 Query router: {router_file}
        
        📊 Partition Details:
        """)
        
        for partition in report['partitions']:
            print(f"  📁 {partition['file']}: {partition['records']:,} records ({partition['size_mb']:.1f}MB)")
    
    elif choice == "2":
        logger.info("Creating year-based partitions only...")
        for year in analysis['year_distribution'].keys():
            partition_file = partitioner.partition_dir / f"bist_data_{year}.db"
            partitioner.create_year_partition(year, str(partition_file))
    
    elif choice == "3":
        logger.info("Creating symbol-based partitions only...")
        large_symbols = [(symbol, count) for symbol, count in analysis['symbol_distribution'].items() 
                        if count > 20000]
        for symbol, count in large_symbols:
            partition_file = partitioner.partition_dir / f"bist_symbol_{symbol.lower()}.db"
            partitioner.create_symbol_partition(symbol, str(partition_file))
    
    elif choice == "4":
        print("📊 Analysis complete. No partitions created.")
    
    print("\n✅ Operation completed!")

if __name__ == "__main__":
    main()
