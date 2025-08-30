#!/usr/bin/env python3
"""
QUICK START - BIST Automated Data Pipeline
=========================================
Kullanımı kolay test scripti - 1200 Excel + daily basestock.xls için

Usage:
    python quick_start_pipeline.py --import-historical
    python quick_start_pipeline.py --monitor-daily
    python quick_start_pipeline.py --status
"""

import argparse
import os
import json
from pathlib import Path
import asyncio
from automated_data_pipeline import BISTDataPipeline

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'historical_excel_data',
        'daily_updates',
        'daily_updates/backups', 
        'data',
        'data/processing_cache',
        'logs',
        'logs/error_logs',
        'ai_models'
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def create_sample_config():
    """Create sample configuration with user prompts"""
    print("\n🔧 CONFIGURATION SETUP")
    print("=" * 50)
    
    # Get user inputs
    historical_dir = input("📁 1200 Excel dosyalarının klasör yolu (örn: /path/to/excel/files): ").strip()
    if not historical_dir:
        historical_dir = "historical_excel_data/"
    
    daily_file = input("📊 Günlük basestock.xls dosya yolu (örn: daily_updates/basestock.xls): ").strip()  
    if not daily_file:
        daily_file = "daily_updates/basestock.xls"
    
    auto_retrain = input("🤖 Otomatik AI model retraining? (y/n) [y]: ").strip().lower()
    auto_retrain = auto_retrain in ['', 'y', 'yes']
    
    parallel_workers = input("⚡ Paralel işlem sayısı (1-8) [4]: ").strip()
    try:
        parallel_workers = int(parallel_workers) if parallel_workers else 4
        parallel_workers = max(1, min(8, parallel_workers))
    except:
        parallel_workers = 4
    
    # Create config
    config = {
        "database": {
            "path": "data/bist_comprehensive.db",
            "backup_interval_hours": 24
        },
        "historical_data": {
            "directory": historical_dir,
            "file_pattern": "*.xlsx",
            "expected_files": 1200,
            "parallel_processing": True,
            "max_workers": parallel_workers,
            "columns_mapping": {
                "date": ["Date", "Tarih", "DATE", "date", "Date/Time"],
                "symbol": ["Symbol", "Sembol", "SYMBOL", "Code", "Kod"],
                "open": ["Open", "Acilis", "OPEN", "Opening", "Açılış"],
                "high": ["High", "Yuksek", "HIGH", "Yüksek", "Max"],
                "low": ["Low", "Dusuk", "LOW", "Düşük", "Min"],
                "close": ["Close", "Kapanis", "CLOSE", "Kapanış", "Last"],
                "volume": ["Volume", "Hacim", "VOLUME", "Vol", "Amount"]
            }
        },
        "daily_updates": {
            "basestock_path": daily_file,
            "check_interval_minutes": 5,
            "auto_process": True
        },
        "ai_processing": {
            "retrain_threshold_days": 7,
            "prediction_horizon_days": [1, 3, 7, 14, 30],
            "technical_indicators": ["RSI", "MACD", "BB", "ICHIMOKU"],
            "auto_deployment": auto_retrain
        },
        "monitoring": {
            "logging_level": "INFO",
            "log_file": "logs/data_pipeline.log"
        }
    }
    
    # Save config
    with open('data_pipeline_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuration saved: data_pipeline_config.json")
    return config

def check_prerequisites():
    """Check if system is ready"""
    print("\n🔍 SYSTEM CHECK")
    print("=" * 30)
    
    issues = []
    
    # Check Python packages
    try:
        import pandas
        print("✅ pandas installed")
    except ImportError:
        issues.append("pip install pandas")
    
    try:
        import openpyxl
        print("✅ openpyxl installed")
    except ImportError:
        issues.append("pip install openpyxl")
    
    try:
        import schedule
        print("✅ schedule installed")
    except ImportError:
        issues.append("pip install schedule")
    
    try:
        from watchdog.observers import Observer
        print("✅ watchdog installed")
    except ImportError:
        issues.append("pip install watchdog")
    
    # Check directories
    config_exists = os.path.exists('data_pipeline_config.json')
    if config_exists:
        print("✅ Configuration file exists")
        with open('data_pipeline_config.json', 'r') as f:
            config = json.load(f)
            
        historical_dir = config['historical_data']['directory']
        if os.path.exists(historical_dir):
            excel_files = list(Path(historical_dir).glob("*.xlsx"))
            print(f"✅ Historical directory exists: {len(excel_files)} Excel files")
        else:
            issues.append(f"Create historical data directory: {historical_dir}")
    else:
        issues.append("Run with --setup to create configuration")
    
    if issues:
        print(f"\n❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print(f"\n✅ System ready!")
    return True

async def import_historical_data():
    """Import 1200 Excel files"""
    print("\n🚀 HISTORICAL DATA IMPORT")
    print("=" * 40)
    
    if not check_prerequisites():
        return
    
    pipeline = BISTDataPipeline()
    
    config = pipeline.config
    historical_dir = config['historical_data']['directory']
    
    print(f"📁 Importing from: {historical_dir}")
    
    if not os.path.exists(historical_dir):
        print(f"❌ Directory not found: {historical_dir}")
        print("   Please put your 1200 Excel files in this directory first.")
        return
    
    excel_files = list(Path(historical_dir).glob("*.xlsx"))
    if len(excel_files) == 0:
        print(f"❌ No Excel files found in: {historical_dir}")
        print("   Please copy your 1200 Excel files to this directory.")
        return
    
    print(f"📊 Found {len(excel_files)} Excel files")
    
    confirm = input("🚀 Start import? This may take 15-30 minutes (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("⏹️ Import cancelled")
        return
    
    # Start import
    result = await pipeline.import_historical_data(historical_dir)
    
    if result:
        print(f"\n✅ IMPORT COMPLETED!")
        print(f"   📊 Files processed: {result['files_processed']}")
        print(f"   🗄️ Records imported: {result['total_records']:,}")
        print(f"   ⏱️ Processing time: {result['processing_time']:.1f} seconds")
        
        if result['errors']:
            print(f"   ⚠️ Errors: {len(result['errors'])}")
    else:
        print("❌ Import failed")

async def monitor_daily_updates():
    """Start monitoring for daily basestock.xls updates"""
    print("\n🔄 DAILY UPDATE MONITORING")
    print("=" * 40)
    
    if not check_prerequisites():
        return
        
    pipeline = BISTDataPipeline()
    
    # Setup monitoring
    observer = pipeline.setup_daily_monitoring()
    
    config = pipeline.config
    daily_file = config['daily_updates']['basestock_path']
    watch_dir = os.path.dirname(daily_file)
    
    print(f"👀 Monitoring directory: {watch_dir}")
    print(f"📊 Watching for: basestock.xls updates")
    print(f"🔄 Check interval: {config['daily_updates']['check_interval_minutes']} minutes")
    print("\n📝 Instructions:")
    print(f"   1. Copy your daily basestock.xls to: {daily_file}")
    print(f"   2. System will automatically detect and process updates")
    print(f"   3. Press Ctrl+C to stop monitoring")
    
    try:
        print(f"\n🚀 Monitoring started... (Press Ctrl+C to stop)")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Stopping monitoring...")
        observer.stop()
        observer.join()
        print(f"✅ Monitoring stopped")

def show_status():
    """Show system status"""
    print("\n📊 SYSTEM STATUS")
    print("=" * 30)
    
    if not check_prerequisites():
        return
    
    try:
        pipeline = BISTDataPipeline()
        status = pipeline.get_system_status()
        
        db_stats = status['database_stats']
        print(f"🗄️ Database Statistics:")
        print(f"   📈 Historical records: {db_stats['historical_records']:,}")
        print(f"   📊 Technical indicators: {db_stats['technical_indicators']:,}")
        print(f"   🤖 AI predictions: {db_stats['ai_predictions']:,}")
        print(f"   📑 Unique symbols: {db_stats['unique_symbols']}")
        print(f"   📅 Date range: {db_stats['date_range']['from']} → {db_stats['date_range']['to']}")
        
        print(f"\n📋 Recent Activity:")
        recent_logs = status['recent_activity']
        for log in recent_logs[-5:]:  # Last 5 activities
            print(f"   • {log['process_type']}: {log['status']} ({log['records_processed']} records)")
            
        print(f"\n✅ System Health: {status['system_health']}")
        print(f"🕐 Last update: {status['last_update']}")
        
    except Exception as e:
        print(f"❌ Error getting status: {str(e)}")

def create_sample_data():
    """Create sample Excel files for testing"""
    print("\n🧪 SAMPLE DATA CREATION")
    print("=" * 35)
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample historical data
    symbols = ['AKBNK', 'GARAN', 'THYAO', 'TUPRS', 'BRSAN']
    start_date = datetime(2020, 1, 1)
    
    os.makedirs('historical_excel_data', exist_ok=True)
    
    for month in range(12):  # 12 months of sample data
        month_start = start_date + timedelta(days=month*30)
        month_data = []
        
        for symbol in symbols:
            base_price = 50 + np.random.random() * 100
            
            for day in range(30):  # 30 days per month
                date = month_start + timedelta(days=day)
                if date.weekday() >= 5:  # Skip weekends
                    continue
                    
                # Generate OHLCV data
                open_price = base_price * (1 + np.random.normal(0, 0.02))
                close_price = open_price * (1 + np.random.normal(0, 0.02))
                high_price = max(open_price, close_price) * (1 + np.random.random() * 0.01)
                low_price = min(open_price, close_price) * (1 - np.random.random() * 0.01)
                volume = int(1000000 + np.random.random() * 5000000)
                
                month_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Symbol': symbol,
                    'Open': round(open_price, 2),
                    'High': round(high_price, 2),
                    'Low': round(low_price, 2),
                    'Close': round(close_price, 2),
                    'Volume': volume
                })
                
                base_price = close_price  # Use close as next base
        
        # Save month data
        df = pd.DataFrame(month_data)
        filename = f"historical_excel_data/sample_{2020}_month_{month+1:02d}.xlsx"
        df.to_excel(filename, index=False)
        print(f"✅ Created: {filename} ({len(df)} records)")
    
    # Create sample daily basestock.xls
    os.makedirs('daily_updates', exist_ok=True)
    
    daily_data = []
    for symbol in symbols:
        base_price = 50 + np.random.random() * 100
        open_price = base_price * (1 + np.random.normal(0, 0.01))
        close_price = open_price * (1 + np.random.normal(0, 0.02))
        high_price = max(open_price, close_price) * (1 + np.random.random() * 0.005)
        low_price = min(open_price, close_price) * (1 - np.random.random() * 0.005)
        volume = int(2000000 + np.random.random() * 8000000)
        
        daily_data.append({
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Symbol': symbol,
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    df_daily = pd.DataFrame(daily_data)
    df_daily.to_excel('daily_updates/basestock.xls', index=False)
    print(f"✅ Created: daily_updates/basestock.xls ({len(df_daily)} records)")
    
    print(f"\n🎯 Sample data created!")
    print(f"   📁 Historical: 12 Excel files (12 months of data)")
    print(f"   📊 Daily: basestock.xls (today's data)")
    print(f"   🚀 Ready to test the pipeline!")

def main():
    parser = argparse.ArgumentParser(
        description='BIST Automated Data Pipeline - Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python quick_start_pipeline.py --setup              # Initial setup
  python quick_start_pipeline.py --check              # Check prerequisites  
  python quick_start_pipeline.py --import-historical  # Import 1200 Excel files
  python quick_start_pipeline.py --monitor-daily      # Monitor basestock.xls
  python quick_start_pipeline.py --status            # Show system status
  python quick_start_pipeline.py --sample-data       # Create test data
        '''
    )
    
    parser.add_argument('--setup', action='store_true', help='Setup directories and configuration')
    parser.add_argument('--check', action='store_true', help='Check system prerequisites')
    parser.add_argument('--import-historical', action='store_true', help='Import historical Excel data')
    parser.add_argument('--monitor-daily', action='store_true', help='Monitor daily basestock updates')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--sample-data', action='store_true', help='Create sample test data')
    
    args = parser.parse_args()
    
    print("🚀 BIST AUTOMATED DATA PIPELINE - QUICK START")
    print("=" * 60)
    
    if args.setup:
        setup_directories()
        create_sample_config()
        print(f"\n✅ Setup completed!")
        print(f"📝 Next steps:")
        print(f"   1. Copy your 1200 Excel files to historical_excel_data/")
        print(f"   2. Run: python quick_start_pipeline.py --import-historical")
        print(f"   3. Run: python quick_start_pipeline.py --monitor-daily")
        
    elif args.check:
        check_prerequisites()
        
    elif args.import_historical:
        asyncio.run(import_historical_data())
        
    elif args.monitor_daily:
        asyncio.run(monitor_daily_updates())
        
    elif args.status:
        show_status()
        
    elif args.sample_data:
        create_sample_data()
        
    else:
        parser.print_help()
        print(f"\n💡 Quick start:")
        print(f"   python quick_start_pipeline.py --setup")

if __name__ == "__main__":
    main()
