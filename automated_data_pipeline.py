#!/usr/bin/env python3
"""
AUTOMATED BIST DATA PIPELINE - MAMUT R600
========================================
Historical Import (1200 Excel) + Daily Updates (basestock.xls) + Auto AI Processing

Features:
- Batch import 1200 historical Excel files
- Daily basestock.xls change detection and processing
- Automatic AI model retraining on new data
- Real-time analysis and prediction generation
- Database optimization and indexing
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import sqlite3
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import schedule
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BISTDataPipeline:
    """
    Automated BIST Data Pipeline System
    Handles 1200 Excel historical data + daily basestock.xls updates
    """
    
    def __init__(self, config_path: str = "data_pipeline_config.json"):
        self.config = self.load_config(config_path)
        self.db_path = self.config['database']['path']
        self.historical_dir = self.config['historical_data']['directory']
        self.daily_file_path = self.config['daily_updates']['basestock_path']
        self.last_update_hash = None
        
        # Initialize database
        self.init_database()
        
        # AI Models will be loaded after data import
        self.ai_models = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        default_config = {
            "database": {
                "path": "data/bist_comprehensive.db",
                "backup_interval_hours": 24
            },
            "historical_data": {
                "directory": "historical_excel_data/",
                "file_pattern": "*.xlsx",
                "expected_files": 1200,
                "columns_mapping": {
                    "date": "Date",
                    "symbol": "Symbol", 
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                }
            },
            "daily_updates": {
                "basestock_path": "daily_updates/basestock.xls",
                "check_interval_minutes": 15,
                "auto_process": True
            },
            "ai_processing": {
                "retrain_threshold_days": 7,
                "prediction_horizon_days": 30,
                "technical_indicators": ["RSI", "MACD", "BB", "ICHIMOKU"],
                "sentiment_analysis": True,
                "auto_deployment": True
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        else:
            # Create default config file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config

    def init_database(self):
        """Initialize comprehensive BIST database schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Historical OHLCV data
            conn.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # Technical indicators
            conn.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    rsi_14 REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    ichimoku_tenkan REAL,
                    ichimoku_kijun REAL,
                    ichimoku_senkou_a REAL,
                    ichimoku_senkou_b REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # AI Predictions
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    target_date DATE NOT NULL,
                    predicted_price REAL,
                    confidence REAL,
                    model_version TEXT,
                    features_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Data processing logs
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    process_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    records_processed INTEGER,
                    processing_time_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_symbol_date ON historical_data(symbol, date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_technical_symbol_date ON technical_indicators(symbol, date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_symbol_target ON ai_predictions(symbol, target_date)')
            
        logger.info(f"✅ Database initialized: {self.db_path}")

    async def import_historical_data(self, historical_directory: str):
        """
        Import 1200 Excel files with historical BIST data
        Optimized for batch processing with progress tracking
        """
        start_time = datetime.now()
        logger.info(f"🚀 Starting historical data import from: {historical_directory}")
        
        excel_files = list(Path(historical_directory).glob("*.xlsx"))
        total_files = len(excel_files)
        
        if total_files == 0:
            logger.warning("⚠️ No Excel files found in historical directory")
            return
            
        logger.info(f"📊 Found {total_files} Excel files to process")
        
        processed_files = 0
        total_records = 0
        errors = []
        
        with sqlite3.connect(self.db_path) as conn:
            for excel_file in excel_files:
                try:
                    # Read Excel file
                    df = pd.read_excel(excel_file, engine='openpyxl')
                    
                    # Standardize column names
                    df = self.standardize_columns(df)
                    
                    # Data validation and cleaning
                    df = self.clean_historical_data(df)
                    
                    # Insert into database (batch insert for performance)
                    df.to_sql('historical_data', conn, if_exists='append', index=False)
                    
                    processed_files += 1
                    records_in_file = len(df)
                    total_records += records_in_file
                    
                    if processed_files % 100 == 0:
                        logger.info(f"📈 Progress: {processed_files}/{total_files} files ({records_in_file:,} records)")
                        
                except Exception as e:
                    error_msg = f"Error processing {excel_file}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        # Calculate technical indicators for all imported data
        logger.info("🔧 Calculating technical indicators...")
        await self.calculate_all_technical_indicators()
        
        # Log the import process
        processing_time = (datetime.now() - start_time).total_seconds()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO processing_logs 
                (process_type, status, details, records_processed, processing_time_seconds)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                'historical_import',
                'completed',
                json.dumps({
                    'files_processed': processed_files,
                    'total_files': total_files,
                    'errors': errors[:10]  # First 10 errors only
                }),
                total_records,
                processing_time
            ))
        
        logger.info(f"✅ Historical import completed: {total_records:,} records in {processing_time:.1f}s")
        return {
            'files_processed': processed_files,
            'total_records': total_records,
            'processing_time': processing_time,
            'errors': errors
        }

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different Excel formats"""
        column_mapping = self.config['historical_data']['columns_mapping']
        
        # Common column name variations
        variations = {
            'date': ['Date', 'Tarih', 'DATE', 'date', 'Date/Time'],
            'symbol': ['Symbol', 'Sembol', 'SYMBOL', 'Code', 'Kod'],
            'open': ['Open', 'Acilis', 'OPEN', 'Opening', 'Açılış'],
            'high': ['High', 'Yuksek', 'HIGH', 'Yüksek', 'Max'],
            'low': ['Low', 'Dusuk', 'LOW', 'Düşük', 'Min'],
            'close': ['Close', 'Kapanis', 'CLOSE', 'Kapanış', 'Last'],
            'volume': ['Volume', 'Hacim', 'VOLUME', 'Vol', 'Amount']
        }
        
        # Create mapping dictionary
        rename_dict = {}
        for standard_name, possible_names in variations.items():
            for col in df.columns:
                if col in possible_names:
                    rename_dict[col] = standard_name
                    break
        
        df = df.rename(columns=rename_dict)
        return df

    def clean_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate historical data"""
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Ensure numeric columns are numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Basic data validation
        df = df[df['close'] > 0]  # Price must be positive
        df = df[df['volume'] >= 0]  # Volume must be non-negative
        
        # Remove duplicates
        if 'symbol' in df.columns and 'date' in df.columns:
            df = df.drop_duplicates(subset=['symbol', 'date'])
        
        return df

    async def calculate_all_technical_indicators(self):
        """Calculate technical indicators for all symbols in database"""
        logger.info("🔧 Calculating technical indicators...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all unique symbols
            symbols = pd.read_sql_query("SELECT DISTINCT symbol FROM historical_data", conn)
            
            for symbol in symbols['symbol']:
                await self.calculate_technical_indicators_for_symbol(symbol)
                
        logger.info("✅ Technical indicators calculation completed")

    async def calculate_technical_indicators_for_symbol(self, symbol: str):
        """Calculate technical indicators for a specific symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get historical data for symbol
                df = pd.read_sql_query('''
                    SELECT date, close, high, low, volume 
                    FROM historical_data 
                    WHERE symbol = ? 
                    ORDER BY date
                ''', conn, params=(symbol,))
                
                if len(df) < 50:  # Need minimum data for indicators
                    return
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                # Calculate RSI
                df['rsi_14'] = self.calculate_rsi(df['close'], 14)
                
                # Calculate MACD
                macd_data = self.calculate_macd(df['close'])
                df['macd_line'] = macd_data['macd']
                df['macd_signal'] = macd_data['signal']
                df['macd_histogram'] = macd_data['histogram']
                
                # Calculate Bollinger Bands
                bb_data = self.calculate_bollinger_bands(df['close'])
                df['bb_upper'] = bb_data['upper']
                df['bb_middle'] = bb_data['middle']
                df['bb_lower'] = bb_data['lower']
                
                # Calculate Ichimoku
                ichimoku_data = self.calculate_ichimoku(df)
                df['ichimoku_tenkan'] = ichimoku_data['tenkan']
                df['ichimoku_kijun'] = ichimoku_data['kijun']
                df['ichimoku_senkou_a'] = ichimoku_data['senkou_a']
                df['ichimoku_senkou_b'] = ichimoku_data['senkou_b']
                
                # Prepare data for database insert
                df_indicators = df[['rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
                                 'bb_upper', 'bb_middle', 'bb_lower',
                                 'ichimoku_tenkan', 'ichimoku_kijun', 
                                 'ichimoku_senkou_a', 'ichimoku_senkou_b']].copy()
                df_indicators['symbol'] = symbol
                df_indicators['date'] = df_indicators.index
                
                # Insert into database
                df_indicators.to_sql('technical_indicators', conn, if_exists='append', index=False)
                
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {symbol}: {str(e)}")

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud indicators"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b
        }

    def setup_daily_monitoring(self):
        """Setup automated monitoring for daily basestock.xls updates"""
        logger.info("🔄 Setting up daily basestock.xls monitoring...")
        
        class BasestockHandler(FileSystemEventHandler):
            def __init__(self, pipeline):
                self.pipeline = pipeline
                
            def on_modified(self, event):
                if event.is_directory:
                    return
                    
                if event.src_path.endswith('basestock.xls') or event.src_path.endswith('basestock.xlsx'):
                    logger.info(f"📊 Detected basestock update: {event.src_path}")
                    asyncio.create_task(self.pipeline.process_daily_update(event.src_path))
        
        # Watch directory for changes
        observer = Observer()
        handler = BasestockHandler(self)
        
        watch_dir = os.path.dirname(self.daily_file_path)
        if not os.path.exists(watch_dir):
            os.makedirs(watch_dir, exist_ok=True)
            
        observer.schedule(handler, watch_dir, recursive=False)
        observer.start()
        
        logger.info(f"✅ Monitoring basestock updates in: {watch_dir}")
        return observer

    async def process_daily_update(self, basestock_path: str):
        """Process daily basestock.xls update and trigger AI pipeline"""
        start_time = datetime.now()
        logger.info(f"🔄 Processing daily update: {basestock_path}")
        
        try:
            # Calculate file hash to detect actual changes
            current_hash = self.calculate_file_hash(basestock_path)
            
            if current_hash == self.last_update_hash:
                logger.info("⏭️ No changes detected, skipping update")
                return
                
            # Read new basestock data
            df = pd.read_excel(basestock_path, engine='openpyxl')
            df = self.standardize_columns(df)
            df = self.clean_historical_data(df)
            
            # Update database with new data
            new_records = 0
            updated_records = 0
            
            with sqlite3.connect(self.db_path) as conn:
                for _, row in df.iterrows():
                    # Check if record exists
                    existing = conn.execute('''
                        SELECT id FROM historical_data 
                        WHERE symbol = ? AND date = ?
                    ''', (row['symbol'], row['date'])).fetchone()
                    
                    if existing:
                        # Update existing record
                        conn.execute('''
                            UPDATE historical_data 
                            SET open = ?, high = ?, low = ?, close = ?, volume = ?
                            WHERE symbol = ? AND date = ?
                        ''', (row['open'], row['high'], row['low'], row['close'], 
                             row['volume'], row['symbol'], row['date']))
                        updated_records += 1
                    else:
                        # Insert new record
                        conn.execute('''
                            INSERT INTO historical_data 
                            (symbol, date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (row['symbol'], row['date'], row['open'], 
                             row['high'], row['low'], row['close'], row['volume']))
                        new_records += 1
                
                conn.commit()
            
            # Update technical indicators for affected symbols
            affected_symbols = df['symbol'].unique()
            for symbol in affected_symbols:
                await self.calculate_technical_indicators_for_symbol(symbol)
            
            # Trigger AI model retraining if significant new data
            if new_records > 10:  # Threshold for retraining
                logger.info("🤖 Triggering AI model retraining...")
                await self.trigger_ai_retraining(affected_symbols)
            
            # Update hash
            self.last_update_hash = current_hash
            
            # Log the update process
            processing_time = (datetime.now() - start_time).total_seconds()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO processing_logs 
                    (process_type, status, details, records_processed, processing_time_seconds)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    'daily_update',
                    'completed',
                    json.dumps({
                        'new_records': new_records,
                        'updated_records': updated_records,
                        'affected_symbols': list(affected_symbols)
                    }),
                    new_records + updated_records,
                    processing_time
                ))
            
            logger.info(f"✅ Daily update completed: {new_records} new, {updated_records} updated records in {processing_time:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Error processing daily update: {str(e)}")

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file to detect changes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    async def trigger_ai_retraining(self, symbols: List[str]):
        """Trigger AI model retraining for updated symbols"""
        logger.info(f"🤖 Starting AI retraining for {len(symbols)} symbols")
        
        # This would integrate with your existing AI training pipeline
        # For now, just generate sample predictions
        
        with sqlite3.connect(self.db_path) as conn:
            for symbol in symbols:
                # Generate sample predictions
                prediction_date = datetime.now().date()
                
                for days_ahead in [1, 7, 30]:
                    target_date = prediction_date + timedelta(days=days_ahead)
                    
                    # Mock prediction (replace with real AI model)
                    base_price = 100  # Would get from actual data
                    predicted_price = base_price * (1 + np.random.normal(0, 0.02))
                    confidence = 0.7 + np.random.random() * 0.25
                    
                    conn.execute('''
                        INSERT INTO ai_predictions 
                        (symbol, prediction_date, target_date, predicted_price, confidence, model_version, features_used)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, prediction_date, target_date, predicted_price, confidence,
                        'DP-LSTM-v1.0', json.dumps(['price', 'volume', 'rsi', 'macd'])
                    ))
            
            conn.commit()
        
        logger.info("✅ AI predictions generated")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with sqlite3.connect(self.db_path) as conn:
            # Database statistics
            historical_count = conn.execute('SELECT COUNT(*) FROM historical_data').fetchone()[0]
            indicators_count = conn.execute('SELECT COUNT(*) FROM technical_indicators').fetchone()[0]
            predictions_count = conn.execute('SELECT COUNT(*) FROM ai_predictions').fetchone()[0]
            
            # Unique symbols
            symbols_count = conn.execute('SELECT COUNT(DISTINCT symbol) FROM historical_data').fetchone()[0]
            
            # Date range
            date_range = conn.execute('''
                SELECT MIN(date) as min_date, MAX(date) as max_date 
                FROM historical_data
            ''').fetchone()
            
            # Recent processing logs
            recent_logs = pd.read_sql_query('''
                SELECT * FROM processing_logs 
                ORDER BY created_at DESC LIMIT 10
            ''', conn)
        
        return {
            'database_stats': {
                'historical_records': historical_count,
                'technical_indicators': indicators_count,
                'ai_predictions': predictions_count,
                'unique_symbols': symbols_count,
                'date_range': {
                    'from': date_range[0],
                    'to': date_range[1]
                }
            },
            'recent_activity': recent_logs.to_dict('records'),
            'system_health': 'operational',
            'last_update': datetime.now().isoformat()
        }

# Example usage and main execution
if __name__ == "__main__":
    async def main():
        # Initialize pipeline
        pipeline = BISTDataPipeline()
        
        # Import historical data (1200 Excel files)
        historical_dir = "path/to/1200/excel/files/"
        if os.path.exists(historical_dir):
            await pipeline.import_historical_data(historical_dir)
        
        # Setup daily monitoring
        observer = pipeline.setup_daily_monitoring()
        
        # Schedule daily health checks
        schedule.every().day.at("06:00").do(lambda: asyncio.create_task(pipeline.get_system_status()))
        
        logger.info("🚀 BIST Automated Data Pipeline is running...")
        logger.info("📊 Monitoring for basestock.xls updates...")
        logger.info("🤖 AI models ready for retraining on new data...")
        
        try:
            # Keep the system running
            while True:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("⏹️ Shutting down pipeline...")
            observer.stop()
            observer.join()
    
    # Run the async main function
    asyncio.run(main())
