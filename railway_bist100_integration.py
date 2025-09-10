#!/usr/bin/env python3
"""
Railway PostgreSQL BIST_100 Integration
Production-ready integration with existing Railway database
2.1M+ historical data + real-time API feeds
"""

import psycopg2
import psycopg2.extras
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RailwayBIST100Integration:
    def __init__(self):
        # Railway PostgreSQL connection
        self.database_url = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"
        
        # Profit.com API
        self.api_key = "a9a0bacbab08493d958244c05380da01"
        self.base_url = "https://api.profit.com/data-api"
        self.session = requests.Session()
        
        # Test connection
        self._test_connection()
        
    def _test_connection(self):
        """Test Railway PostgreSQL connection"""
        try:
            conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()
            logger.info(f"✅ Railway PostgreSQL connected: {version['version'][:50]}...")
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Railway connection failed: {e}")
            raise
            
    def analyze_existing_database(self):
        """Analyze existing Railway database structure and data"""
        logger.info("🔍 Analyzing existing Railway database...")
        
        conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
        cursor = conn.cursor()
        
        try:
            # Get table information
            cursor.execute("""
                SELECT table_name, column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                ORDER BY table_name, ordinal_position
            """)
            
            schema_info = cursor.fetchall()
            tables = {}
            
            for row in schema_info:
                table_name = row['table_name']
                if table_name not in tables:
                    tables[table_name] = []
                tables[table_name].append(f"{row['column_name']} ({row['data_type']})")
            
            logger.info("📊 DATABASE SCHEMA ANALYSIS:")
            for table_name, columns in tables.items():
                logger.info(f"   📋 Table: {table_name}")
                for column in columns:
                    logger.info(f"      • {column}")
                logger.info("")
                
            # Analyze enhanced_stock_data table (main historical data)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(DISTINCT timeframe) as timeframes_count
                FROM enhanced_stock_data
            """)
            
            enhanced_stats = cursor.fetchone()
            
            # Analyze historical_data table
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(date_time) as earliest_date,
                    MAX(date_time) as latest_date
                FROM historical_data
            """)
            
            historical_stats = cursor.fetchone()
            
            logger.info("📈 ENHANCED_STOCK_DATA ANALYSIS:")
            logger.info(f"   📊 Total Records: {enhanced_stats['total_records']:,}")
            logger.info(f"   🏢 Unique Symbols: {enhanced_stats['unique_symbols']}")
            logger.info(f"   📅 Date Range: {enhanced_stats['earliest_date']} → {enhanced_stats['latest_date']}")
            logger.info(f"   ⏰ Timeframes: {enhanced_stats['timeframes_count']}")
            
            logger.info("📈 HISTORICAL_DATA ANALYSIS:")  
            logger.info(f"   📊 Total Records: {historical_stats['total_records']:,}")
            logger.info(f"   🏢 Unique Symbols: {historical_stats['unique_symbols']}")
            logger.info(f"   📅 Date Range: {historical_stats['earliest_date']} → {historical_stats['latest_date']}")
            
            # Get available timeframes from enhanced_stock_data
            cursor.execute("SELECT DISTINCT timeframe FROM enhanced_stock_data ORDER BY timeframe")
            timeframes = [row['timeframe'] for row in cursor.fetchall()]
            logger.info(f"   🕐 Available Timeframes: {', '.join(timeframes)}")
            
            # Get top symbols by data volume from both tables
            cursor.execute("""
                SELECT symbol, COUNT(*) as record_count
                FROM enhanced_stock_data 
                GROUP BY symbol 
                ORDER BY record_count DESC 
                LIMIT 10
            """)
            
            top_symbols_enhanced = cursor.fetchall()
            logger.info(f"   🔝 Top Symbols by Data Volume (Enhanced):")
            for symbol in top_symbols_enhanced:
                logger.info(f"      • {symbol['symbol']}: {symbol['record_count']:,} records")
                
            # Check stocks table
            cursor.execute("SELECT COUNT(*) as stock_count FROM stocks")
            stocks_count = cursor.fetchone()
            logger.info(f"📋 STOCKS TABLE: {stocks_count['stock_count']} stocks registered")
                
            return {
                'tables': tables,
                'enhanced_stats': dict(enhanced_stats),
                'historical_stats': dict(historical_stats),
                'timeframes': timeframes,
                'top_symbols': [dict(s) for s in top_symbols_enhanced],
                'stocks_count': stocks_count['stock_count']
            }
            
        finally:
            cursor.close()
            conn.close()
            
    def load_bist_indexes_and_validate(self):
        """Load BIST indexes and validate against Railway database"""
        logger.info("📊 Loading BIST indexes and validating against Railway...")
        
        # Load BIST index files
        indexes = {}
        try:
            df_100 = pd.read_excel("BIST_100.xlsx")
            indexes['BIST_100'] = set(df_100.iloc[:, 0].dropna().astype(str).str.upper())
            
            df_50 = pd.read_excel("BIST_50.xlsx")
            indexes['BIST_50'] = set(df_50.iloc[:, 0].dropna().astype(str).str.upper())
            
            df_30 = pd.read_excel("BIST_30.xlsx") 
            indexes['BIST_30'] = set(df_30.iloc[:, 0].dropna().astype(str).str.upper())
            
            logger.info(f"✅ Loaded indexes: BIST_100({len(indexes['BIST_100'])}), BIST_50({len(indexes['BIST_50'])}), BIST_30({len(indexes['BIST_30'])})")
            
        except Exception as e:
            logger.error(f"❌ Failed to load BIST index files: {e}")
            return None
            
        # Validate against Railway database
        conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
        cursor = conn.cursor()
        
        try:
            # Get symbols from both enhanced_stock_data and historical_data
            cursor.execute("SELECT DISTINCT symbol FROM enhanced_stock_data")
            enhanced_symbols = set([row['symbol'] for row in cursor.fetchall()])
            
            cursor.execute("SELECT DISTINCT symbol FROM historical_data")
            historical_symbols = set([row['symbol'] for row in cursor.fetchall()])
            
            railway_symbols = enhanced_symbols | historical_symbols
            
            logger.info(f"🗄️ Railway database symbols:")
            logger.info(f"   📊 Enhanced data: {len(enhanced_symbols)} symbols")
            logger.info(f"   📊 Historical data: {len(historical_symbols)} symbols") 
            logger.info(f"   📊 Total unique: {len(railway_symbols)} symbols")
            
            validation_results = {}
            
            for index_name, index_symbols in indexes.items():
                # Find matches with Railway database
                matched = index_symbols & railway_symbols
                missing_in_railway = index_symbols - railway_symbols
                
                # Test API availability for missing symbols
                api_available = []
                for symbol in missing_in_railway:
                    if self._test_api_availability(symbol):
                        api_available.append(symbol)
                        
                validation_results[index_name] = {
                    'total': len(index_symbols),
                    'in_railway': len(matched),
                    'missing_in_railway': len(missing_in_railway),
                    'api_available_for_missing': len(api_available),
                    'matched_symbols': matched,
                    'missing_symbols': missing_in_railway,
                    'api_available_symbols': api_available
                }
                
                railway_coverage = (len(matched) / len(index_symbols)) * 100
                total_coverage = ((len(matched) + len(api_available)) / len(index_symbols)) * 100
                
                logger.info(f"📊 {index_name} VALIDATION:")
                logger.info(f"   📋 Total: {len(index_symbols)} stocks")
                logger.info(f"   🗄️ In Railway: {len(matched)} ({railway_coverage:.1f}%)")
                logger.info(f"   ❌ Missing in Railway: {len(missing_in_railway)}")
                logger.info(f"   📡 API Available for Missing: {len(api_available)}")
                logger.info(f"   🎯 Total Coverage (Railway + API): {total_coverage:.1f}%")
                logger.info("")
                
            return validation_results
            
        finally:
            cursor.close()
            conn.close()
            
    def _test_api_availability(self, symbol: str) -> bool:
        """Test if symbol is available in Profit.com API"""
        try:
            ticker = f"{symbol}.IS"
            url = f"{self.base_url}/market-data/quote/{ticker}"
            response = self.session.get(url, params={'token': self.api_key}, timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def create_bist_categorization_tables(self):
        """Create BIST categorization tables in Railway database"""
        logger.info("🔧 Creating BIST categorization tables...")
        
        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor()
        
        try:
            # Create stocks_meta table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks_meta (
                    symbol VARCHAR(10) PRIMARY KEY,
                    name VARCHAR(200),
                    sector VARCHAR(100),
                    is_active BOOLEAN DEFAULT TRUE,
                    api_available BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Create stock_categories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_categories (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) REFERENCES stocks_meta(symbol),
                    category VARCHAR(20) NOT NULL,  -- BIST_100, BIST_50, BIST_30
                    priority INTEGER NOT NULL,      -- 1=BIST_30, 2=BIST_50, 3=BIST_100
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(symbol, category)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stocks_meta_symbol ON stocks_meta(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_categories_symbol ON stock_categories(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_categories_category ON stock_categories(category)")
            
            conn.commit()
            logger.info("✅ BIST categorization tables created successfully")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Failed to create tables: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
            
    def populate_bist_categories(self, validation_results):
        """Populate BIST category data into Railway database"""
        logger.info("📊 Populating BIST categories into Railway database...")
        
        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor()
        
        try:
            # Clear existing data
            cursor.execute("DELETE FROM stock_categories")
            cursor.execute("DELETE FROM stocks_meta")
            
            # Collect all unique symbols
            all_symbols = set()
            category_data = []
            
            for index_name, results in validation_results.items():
                priority = 1 if index_name == 'BIST_30' else 2 if index_name == 'BIST_50' else 3
                
                # Add symbols from Railway + API available
                symbols_to_add = results['matched_symbols'] | set(results['api_available_symbols'])
                
                for symbol in symbols_to_add:
                    all_symbols.add(symbol)
                    category_data.append((symbol, index_name, priority))
            
            # Insert stocks_meta
            stocks_meta_data = []
            for symbol in all_symbols:
                # Check if symbol has historical data in Railway (check both tables)
                cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data WHERE symbol = %s", (symbol,))
                has_enhanced = cursor.fetchone()[0] > 0
                
                cursor.execute("SELECT COUNT(*) FROM historical_data WHERE symbol = %s", (symbol,))
                has_historical = cursor.fetchone()[0] > 0
                
                has_data = has_enhanced or has_historical
                
                # Check API availability (we tested this during validation)
                api_available = False
                for results in validation_results.values():
                    if symbol in results['api_available_symbols']:
                        api_available = True
                        break
                
                stocks_meta_data.append((symbol, None, None, True, has_data or api_available))
            
            # Bulk insert stocks_meta
            cursor.executemany("""
                INSERT INTO stocks_meta (symbol, name, sector, is_active, api_available)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    api_available = EXCLUDED.api_available,
                    updated_at = NOW()
            """, stocks_meta_data)
            
            # Bulk insert categories
            cursor.executemany("""
                INSERT INTO stock_categories (symbol, category, priority)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol, category) DO NOTHING
            """, category_data)
            
            conn.commit()
            
            logger.info(f"✅ Populated {len(all_symbols)} stocks with {len(category_data)} category assignments")
            
            # Summary by category
            cursor.execute("""
                SELECT category, COUNT(*) as stock_count
                FROM stock_categories 
                GROUP BY category 
                ORDER BY category
            """)
            
            summary = cursor.fetchall()
            logger.info("📊 Category Summary:")
            for row in summary:
                logger.info(f"   • {row[0]}: {row[1]} stocks")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Failed to populate categories: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
            
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        logger.info("📋 Generating comprehensive analysis report...")
        
        conn = psycopg2.connect(self.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
        cursor = conn.cursor()
        
        try:
            # Get categorized stocks with their data availability
            cursor.execute("""
                SELECT 
                    sm.symbol,
                    sm.name,
                    sm.api_available,
                    sc.category,
                    sc.priority,
                    COALESCE(md_counts.record_count, 0) as historical_records,
                    COALESCE(md_latest.latest_date, NULL) as latest_data_date
                FROM stocks_meta sm
                JOIN stock_categories sc ON sm.symbol = sc.symbol
                LEFT JOIN (
                    SELECT symbol, COUNT(*) as record_count
                    FROM enhanced_stock_data 
                    GROUP BY symbol
                ) md_counts ON sm.symbol = md_counts.symbol
                LEFT JOIN (
                    SELECT symbol, MAX(date) as latest_date
                    FROM enhanced_stock_data
                    GROUP BY symbol  
                ) md_latest ON sm.symbol = md_latest.symbol
                ORDER BY sc.priority, sm.symbol
            """)
            
            results = cursor.fetchall()
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([dict(row) for row in results])
            
            # Export to Excel
            output_file = "railway_bist100_analysis.xlsx"
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Main data
                df.to_excel(writer, sheet_name='BIST_Stocks_Analysis', index=False)
                
                # Summary by category
                summary = df.groupby('category').agg({
                    'symbol': 'count',
                    'api_available': 'sum', 
                    'historical_records': ['sum', 'mean', 'max']
                }).round(2)
                
                summary.columns = ['Total_Stocks', 'API_Available', 'Total_Historical_Records', 'Avg_Records_Per_Stock', 'Max_Records_Per_Stock']
                summary.to_excel(writer, sheet_name='Category_Summary')
                
                # Data availability analysis
                availability = df.groupby(['category', 'api_available']).size().unstack(fill_value=0)
                availability.to_excel(writer, sheet_name='API_Availability')
            
            logger.info(f"✅ Analysis report saved: {output_file}")
            
            # Print summary
            logger.info("\n📊 FINAL SUMMARY:")
            for category in ['BIST_30', 'BIST_50', 'BIST_100']:
                category_data = df[df['category'] == category]
                total = len(category_data)
                api_available = category_data['api_available'].sum()
                has_historical = (category_data['historical_records'] > 0).sum()
                
                logger.info(f"   🎯 {category}:")
                logger.info(f"      📊 Total: {total} stocks")
                logger.info(f"      📡 API Available: {api_available} ({api_available/total*100:.1f}%)")
                logger.info(f"      🗄️ Has Historical: {has_historical} ({has_historical/total*100:.1f}%)")
                logger.info(f"      ✅ Ready for Trading: {min(api_available, has_historical)} stocks")
                logger.info("")
            
        finally:
            cursor.close()
            conn.close()
            
    def run_railway_integration(self):
        """Run complete Railway PostgreSQL integration"""
        logger.info("🚀 RAILWAY POSTGRESQL BIST_100 INTEGRATION")
        logger.info("=" * 80)
        
        # Analyze existing database
        db_analysis = self.analyze_existing_database()
        
        # Load and validate BIST indexes
        validation_results = self.load_bist_indexes_and_validate()
        if not validation_results:
            return False
            
        # Create categorization tables
        self.create_bist_categorization_tables()
        
        # Populate categories
        self.populate_bist_categories(validation_results)
        
        # Generate analysis report
        self.generate_analysis_report()
        
        logger.info("✅ RAILWAY INTEGRATION COMPLETED!")
        logger.info("🎯 Ready for Phase 2: Trading Dashboard with Railway data")
        logger.info("📊 2.1M+ historical records + real-time API integration")
        
        return True

def main():
    integrator = RailwayBIST100Integration()
    success = integrator.run_railway_integration()
    
    if success:
        print("\n🎯 RAILWAY INTEGRATION SUCCESSFUL!")
        print("🚀 Ready for production trading dashboard!")
    else:
        print("\n❌ Integration failed. Check logs above.")

if __name__ == "__main__":
    main()
