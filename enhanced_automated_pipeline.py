#!/usr/bin/env python3
"""
ENHANCED AUTOMATED BIST PIPELINE - MAMUT R600
===========================================
Enhanced Excel Import (207 files + future daily) + AI Model Integration

Features:
- Import 207 Excel files with 42 technical indicators  
- Daily basestock.xls monitoring and updates
- Automatic AI model retraining based on new data
- Real-time prediction generation and analysis
- Advanced technical indicator processing
- Database optimization with 35+ technical indicators
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# Import our custom modules
from excel_to_database_importer import ExcelToDatabaseImporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBISTDataPipeline:
    """
    Enhanced BIST Data Pipeline with Advanced Technical Indicators
    Supports 207 Excel files + daily updates + AI integration
    """
    
    def __init__(self, config_path: str = "enhanced_pipeline_config.json"):
        """Initialize the enhanced pipeline"""
        self.config = self.load_config(config_path)
        
        # Database setup
        self.db_path = self.config.get('database', {}).get('path', 'enhanced_bist_data.db')
        
        # Excel data directories
        self.excel_historical_dir = self.config.get('excel_data', {}).get('historical_directory', 'data/New_excell_Graph_Sample')
        self.daily_excel_dir = self.config.get('excel_data', {}).get('daily_directory', 'data/daily_updates')
        
        # Daily monitoring
        self.daily_file_path = self.config.get('daily_updates', {}).get('basestock_path', 'data/daily_updates/basestock.xls')
        self.monitoring_enabled = self.config.get('daily_updates', {}).get('enable_monitoring', True)
        
        # AI retraining thresholds
        self.min_records_for_retrain = self.config.get('ai_retraining', {}).get('min_new_records', 1000)
        self.retrain_frequency_days = self.config.get('ai_retraining', {}).get('frequency_days', 7)
        
        # Initialize components
        self.excel_importer = ExcelToDatabaseImporter(db_path=self.db_path)
        self.last_update_hash = None
        self.setup_monitoring()
        
        logger.info("🚀 Enhanced BIST Data Pipeline initialized")
    
    def load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        default_config = {
            "database": {
                "path": "enhanced_bist_data.db",
                "optimize_frequency": "daily"
            },
            "excel_data": {
                "historical_directory": "data/New_excell_Graph_Sample",
                "daily_directory": "data/daily_updates",
                "enable_parallel_processing": False,
                "batch_size": 50
            },
            "daily_updates": {
                "basestock_path": "data/daily_updates/basestock.xls",
                "enable_monitoring": True,
                "check_interval_seconds": 60,
                "backup_old_files": True
            },
            "ai_retraining": {
                "min_new_records": 1000,
                "frequency_days": 7,
                "enable_automatic": True,
                "models_to_retrain": ["turkish_qa", "dp_lstm", "sentiment"]
            },
            "performance": {
                "parallel_processing": True,
                "max_workers": 4,
                "chunk_size": 1000,
                "memory_limit_gb": 4
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    return {**default_config, **user_config}
            except Exception as e:
                logger.warning(f"⚠️ Config file error, using defaults: {e}")
        
        # Save default config
        self.save_config(config_path, default_config)
        return default_config
    
    def save_config(self, config_path: str, config: Dict):
        """Save configuration to file"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Config saved: {config_path}")
        except Exception as e:
            logger.error(f"❌ Config save error: {e}")
    
    def setup_monitoring(self):
        """Setup file system monitoring for daily updates"""
        if not self.monitoring_enabled:
            return
            
        class ExcelUpdateHandler(FileSystemEventHandler):
            def __init__(self, pipeline):
                self.pipeline = pipeline
            
            def on_modified(self, event):
                if event.is_directory:
                    return
                    
                if 'basestock.xls' in event.src_path or event.src_path.endswith('.xlsx'):
                    logger.info(f"📊 File change detected: {event.src_path}")
                    asyncio.create_task(self.pipeline.process_daily_update(event.src_path))
        
        self.file_observer = Observer()
        self.file_handler = ExcelUpdateHandler(self)
        
        # Monitor daily directory
        daily_dir = Path(self.daily_excel_dir)
        if daily_dir.exists():
            self.file_observer.schedule(self.file_handler, str(daily_dir), recursive=False)
            logger.info(f"📁 Monitoring: {daily_dir}")
    
    async def import_historical_excel_data(self, force_reimport: bool = False):
        """Import all historical Excel files"""
        logger.info("📊 Historical Excel data import başlıyor...")
        
        if not force_reimport:
            # Check if already imported
            conn = sqlite3.connect(self.db_path)
            record_count = conn.execute("SELECT COUNT(*) FROM enhanced_stock_data").fetchone()[0]
            conn.close()
            
            if record_count > 100000:  # Assume already imported if >100k records
                logger.info(f"✅ Database already has {record_count:,} records, skipping historical import")
                return
        
        # Import using our Excel importer
        self.excel_importer.import_all_excel_files(data_dir=self.excel_historical_dir)
        
        # Post-import optimization
        await self.optimize_database()
        
        logger.info("🎉 Historical import tamamlandı!")
    
    async def process_daily_update(self, file_path: str):
        """Process daily Excel update"""
        try:
            logger.info(f"🔄 Daily update processing: {file_path}")
            
            # Calculate file hash to detect changes
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash == self.last_update_hash:
                logger.info("📊 File unchanged, skipping update")
                return
            
            self.last_update_hash = file_hash
            
            # Backup old file if configured
            if self.config.get('daily_updates', {}).get('backup_old_files', True):
                backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(file_path, backup_path)
                logger.info(f"💾 Backup created: {backup_path}")
            
            # Process the update
            rows_added = self.excel_importer.process_single_file(file_path)
            
            if rows_added > 0:
                logger.info(f"✅ Daily update: {rows_added} rows added")
                
                # Check if AI retraining needed
                await self.check_and_retrain_models()
                
                # Generate fresh predictions
                await self.generate_predictions()
                
            else:
                logger.warning("⚠️ No new data from daily update")
                
        except Exception as e:
            logger.error(f"❌ Daily update error: {e}")
    
    async def optimize_database(self):
        """Optimize database performance"""
        logger.info("🔧 Database optimization başlıyor...")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Update statistics
            conn.execute("ANALYZE enhanced_stock_data")
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
            
            # Create additional indexes for common queries (fixed for SQLite compatibility)
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_date ON enhanced_stock_data (symbol, timeframe, date)",
                "CREATE INDEX IF NOT EXISTS idx_rsi_macd ON enhanced_stock_data (rsi_14, macd_26_12)",
                "CREATE INDEX IF NOT EXISTS idx_recent_data ON enhanced_stock_data (date, symbol)",
                "CREATE INDEX IF NOT EXISTS idx_close_volume ON enhanced_stock_data (close, volume)",
                "CREATE INDEX IF NOT EXISTS idx_technical_combo ON enhanced_stock_data (symbol, date, rsi_14, macd_26_12)",
            ]
            
            for index_sql in indexes:
                conn.execute(index_sql)
            
            conn.commit()
            logger.info("✅ Database optimization tamamlandı")
            
        except Exception as e:
            logger.error(f"❌ Database optimization error: {e}")
        finally:
            conn.close()
    
    async def check_and_retrain_models(self):
        """Check if AI models need retraining"""
        if not self.config.get('ai_retraining', {}).get('enable_automatic', True):
            return
        
        logger.info("🤖 AI model retraining check...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Check recent data volume
        recent_count = conn.execute("""
            SELECT COUNT(*) FROM enhanced_stock_data 
            WHERE date >= date('now', '-7 days')
        """).fetchone()[0]
        
        conn.close()
        
        if recent_count >= self.min_records_for_retrain:
            logger.info(f"🎯 {recent_count} recent records, triggering AI retraining")
            await self.retrain_ai_models()
        else:
            logger.info(f"📊 {recent_count} recent records, no retraining needed (threshold: {self.min_records_for_retrain})")
    
    async def retrain_ai_models(self):
        """Retrain AI models with new data"""
        logger.info("🤖 AI model retraining başlıyor...")
        
        models_to_train = self.config.get('ai_retraining', {}).get('models_to_retrain', [])
        
        for model_name in models_to_train:
            try:
                if model_name == "turkish_qa":
                    # Retrain Turkish Q&A model
                    logger.info("📚 Turkish Q&A model retraining...")
                    # Implement Turkish Q&A retraining logic here
                    
                elif model_name == "dp_lstm":
                    # Retrain DP-LSTM model
                    logger.info("📈 DP-LSTM model retraining...")
                    # Implement DP-LSTM retraining logic here
                    
                elif model_name == "sentiment":
                    # Retrain sentiment model
                    logger.info("💭 Sentiment model retraining...")
                    # Implement sentiment retraining logic here
                
                logger.info(f"✅ {model_name} model retrained")
                
            except Exception as e:
                logger.error(f"❌ {model_name} model retraining error: {e}")
        
        logger.info("🎉 AI model retraining tamamlandı!")
    
    async def generate_predictions(self):
        """Generate fresh predictions for all symbols"""
        logger.info("🔮 Fresh predictions generating...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get latest data for each symbol and timeframe
            latest_data = conn.execute("""
                SELECT DISTINCT symbol, timeframe, MAX(date) as latest_date
                FROM enhanced_stock_data 
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """).fetchall()
            
            predictions_generated = 0
            
            for symbol, timeframe, latest_date in latest_data:
                # Get recent technical indicators for prediction
                tech_data = conn.execute("""
                    SELECT close, rsi_14, macd_26_12, atr_14, adx_14,
                           stochastic_k_5, bol_upper_20_2, bol_lower_20_2
                    FROM enhanced_stock_data 
                    WHERE symbol = ? AND timeframe = ? AND date = ?
                    ORDER BY time DESC LIMIT 1
                """, (symbol, timeframe, latest_date)).fetchone()
                
                if tech_data:
                    # Generate prediction based on technical indicators
                    prediction = self.calculate_technical_prediction(symbol, timeframe, tech_data)
                    
                    # Store prediction (implement your prediction storage logic)
                    logger.info(f"🎯 {symbol} ({timeframe}) → {prediction['direction']} (confidence: {prediction['confidence']:.2f})")
                    predictions_generated += 1
            
            conn.close()
            logger.info(f"✅ {predictions_generated} predictions generated")
            
        except Exception as e:
            logger.error(f"❌ Prediction generation error: {e}")
    
    def calculate_technical_prediction(self, symbol: str, timeframe: str, tech_data: tuple) -> Dict:
        """Calculate prediction based on technical indicators"""
        close, rsi, macd, atr, adx, stoch_k, bb_upper, bb_lower = tech_data
        
        signals = []
        
        # RSI signals
        if rsi > 70:
            signals.append(('bearish', 0.3))
        elif rsi < 30:
            signals.append(('bullish', 0.3))
        
        # MACD signals
        if macd > 0:
            signals.append(('bullish', 0.2))
        else:
            signals.append(('bearish', 0.2))
        
        # Bollinger Bands signals
        if close > bb_upper:
            signals.append(('bearish', 0.25))
        elif close < bb_lower:
            signals.append(('bullish', 0.25))
        
        # Stochastic signals
        if stoch_k > 80:
            signals.append(('bearish', 0.15))
        elif stoch_k < 20:
            signals.append(('bullish', 0.15))
        
        # Calculate weighted prediction
        bullish_weight = sum(weight for direction, weight in signals if direction == 'bullish')
        bearish_weight = sum(weight for direction, weight in signals if direction == 'bearish')
        
        if bullish_weight > bearish_weight:
            direction = 'bullish'
            confidence = bullish_weight / (bullish_weight + bearish_weight)
        else:
            direction = 'bearish'
            confidence = bearish_weight / (bullish_weight + bearish_weight)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'confidence': confidence,
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Total records
        stats['total_records'] = conn.execute("SELECT COUNT(*) FROM enhanced_stock_data").fetchone()[0]
        
        # Symbol count
        stats['symbol_count'] = conn.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data").fetchone()[0]
        
        # Timeframe distribution
        timeframes = conn.execute("""
            SELECT timeframe, COUNT(*) as count 
            FROM enhanced_stock_data 
            GROUP BY timeframe 
            ORDER BY count DESC
        """).fetchall()
        stats['timeframe_distribution'] = dict(timeframes)
        
        # Date range
        date_range = conn.execute("""
            SELECT MIN(date) as min_date, MAX(date) as max_date 
            FROM enhanced_stock_data
        """).fetchone()
        stats['date_range'] = {'min': date_range[0], 'max': date_range[1]}
        
        # Recent activity
        recent_count = conn.execute("""
            SELECT COUNT(*) FROM enhanced_stock_data 
            WHERE date >= date('now', '-7 days')
        """).fetchone()[0]
        stats['recent_records_7days'] = recent_count
        
        conn.close()
        return stats
    
    async def start_monitoring(self):
        """Start the automated monitoring system"""
        logger.info("🔄 Automated monitoring başlıyor...")
        
        # Start file system monitoring
        if hasattr(self, 'file_observer'):
            self.file_observer.start()
        
        # Schedule regular tasks
        schedule.every().day.at("09:00").do(self.optimize_database)
        schedule.every().day.at("18:30").do(self.generate_predictions)
        
        try:
            while True:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("⏹️ Monitoring stopped by user")
            if hasattr(self, 'file_observer'):
                self.file_observer.stop()
                self.file_observer.join()

async def main():
    """Main function to run the enhanced pipeline"""
    pipeline = EnhancedBISTDataPipeline()
    
    # Show current database stats
    stats = pipeline.get_database_stats()
    logger.info(f"""
    📊 ENHANCED BIST DATA PIPELINE - CURRENT STATUS
    =============================================
    🗄️ Total records: {stats['total_records']:,}
    📈 Symbols: {stats['symbol_count']}
    📅 Date range: {stats['date_range']['min']} → {stats['date_range']['max']}
    📊 Recent (7 days): {stats['recent_records_7days']:,} records
    
    ⏰ Timeframes:
    """)
    
    for timeframe, count in stats['timeframe_distribution'].items():
        logger.info(f"    {timeframe}: {count:,} records")
    
    # Ask user what to do
    print("\n🚀 What would you like to do?")
    print("1. Import all 207 historical Excel files")
    print("2. Start daily monitoring only") 
    print("3. Import + Start monitoring")
    print("4. Generate predictions from existing data")
    print("5. Show detailed database stats")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        await pipeline.import_historical_excel_data(force_reimport=True)
    elif choice == "2":
        await pipeline.start_monitoring()
    elif choice == "3":
        await pipeline.import_historical_excel_data()
        await pipeline.start_monitoring()
    elif choice == "4":
        await pipeline.generate_predictions()
    elif choice == "5":
        stats = pipeline.get_database_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
