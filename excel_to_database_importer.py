#!/usr/bin/env python3
"""
Enhanced Excel to Database Importer
MAMUT_R600 - Excel dosyalarını database'e aktarma
207 dosya için optimized bulk import
"""
import pandas as pd
import sqlite3
import os
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('excel_import.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExcelToDatabaseImporter:
    
    def __init__(self, db_path="enhanced_bist_data.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Database ve tabloları oluştur"""
        logger.info("🗄️ Database setup başlıyor...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Ana tabloyu oluştur
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS enhanced_stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT,
            timeframe TEXT NOT NULL,
            
            -- OHLCV Data
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            wclose REAL,
            
            -- Technical Indicators
            adx_14 REAL,
            atr_14 REAL,
            psar REAL,
            rsi_14 REAL,
            stochastic_k_5 REAL,
            stochastic_d_3 REAL,
            stoccci_20 REAL,
            stoccci_trigger_20 REAL,
            macd_26_12 REAL,
            macd_trigger_9 REAL,
            bol_upper_20_2 REAL,
            bol_middle_20_2 REAL,
            bol_lower_20_2 REAL,
            bol_upper_20_2_alt REAL,
            bol_middle_20_2_alt REAL,
            bol_lower_20_2_alt REAL,
            tenkan_sen REAL,
            kijun_sen REAL,
            senkou_span_a REAL,
            senkou_span_b REAL,
            chikou_span REAL,
            wma_50 REAL,
            jaw_13_8 REAL,
            teeth_8_5 REAL,
            lips_5_3 REAL,
            awesome_oscillator_5_7 REAL,
            acc_dist_oscillator_21_10 REAL,
            supersmooth_fr REAL,
            supersmooth_filt REAL,
            cs REAL,
            prev_at_14_1_mfi REAL,
            alpha_14_1_mfi REAL,
            signal REAL,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(symbol, date, time, timeframe)
        )
        """
        
        conn.execute(create_table_sql)
        
        # İndeksler
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON enhanced_stock_data (symbol, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timeframe ON enhanced_stock_data (timeframe)")
        
        conn.commit()
        conn.close()
        logger.info("✅ Database setup tamamlandı")
    
    def parse_filename(self, filename):
        """Dosya adından sembol ve timeframe çıkar"""
        # AKBNK_30Dk.xlsx -> symbol=AKBNK, timeframe=30m
        name_without_ext = filename.replace('.xlsx', '')
        parts = name_without_ext.split('_')
        
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe_raw = parts[1]
            
            # Timeframe'i normalize et
            if 'Dk' in timeframe_raw:
                timeframe = timeframe_raw.replace('Dk', 'm')  # 30Dk -> 30m
            elif 'Günlük' in timeframe_raw:
                timeframe = 'daily'
            else:
                timeframe = timeframe_raw.lower()
                
            return symbol, timeframe
        
        return None, None
    
    def normalize_column_names(self, df):
        """Sütun adlarını database'e uygun hale getir"""
        column_mapping = {
            'Date': 'date',
            'Time': 'time', 
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'WClose': 'wclose',
            'ADX (14)   ': 'adx_14',
            'ATR (14)   ': 'atr_14',
            'PSar(0.02,0.2)   ': 'psar',
            'RSI (14)   ': 'rsi_14',
            'StochasticFast %K (5)   ': 'stochastic_k_5',
            'StochasticFast %D (3)   ': 'stochastic_d_3',
            'StocCCI (20)   ': 'stoccci_20',
            'Trigger (20)   ': 'stoccci_trigger_20',
            'MACD (26,12)   ': 'macd_26_12',
            'TRIGGER (9)   ': 'macd_trigger_9',
            'BOL U (20,2)   ': 'bol_upper_20_2',
            'BOL M (20,2)   ': 'bol_middle_20_2',
            'BOL D (20,2)   ': 'bol_lower_20_2',
            'BOL U (20,2)   .1': 'bol_upper_20_2_alt',
            'BOL M (20,2)   .1': 'bol_middle_20_2_alt',
            'BOL D (20,2)   .1': 'bol_lower_20_2_alt',
            'Tenkan-sen   ': 'tenkan_sen',
            'Kijun-sen   ': 'kijun_sen',
            'Senkou Span A   ': 'senkou_span_a',
            'Senkou Span B   ': 'senkou_span_b',
            'Chikou Span   ': 'chikou_span',
            'WMA (50)   ': 'wma_50',
            'Jaw (13,8)   ': 'jaw_13_8',
            'Teeth (8,5)   ': 'teeth_8_5',
            'Lips (5,3)   ': 'lips_5_3',
            'AwesomeOscillatorV2 (5,7)   ': 'awesome_oscillator_5_7',
            'ACCDistOscillator (21,10)   ': 'acc_dist_oscillator_21_10',
            'SuperSmootherFr   ': 'supersmooth_fr',
            'SuperSmootherFilt   ': 'supersmooth_filt',
            'CS   ': 'cs',
            'PrevAT(14,1,MFI)   ': 'prev_at_14_1_mfi',
            'Alpha(14,1,MFI)   ': 'alpha_14_1_mfi',
            'Signal   ': 'signal',
            'VOLUME   ': 'volume'  # duplicate volume column
        }
        
        # Rename columns
        df_renamed = df.rename(columns=column_mapping)
        
        return df_renamed
    
    def process_single_file(self, file_path):
        """Tek Excel dosyasını işle"""
        try:
            filename = os.path.basename(file_path)
            symbol, timeframe = self.parse_filename(filename)
            
            if not symbol or not timeframe:
                logger.warning(f"❌ Dosya adı parse edilemedi: {filename}")
                return 0
                
            logger.info(f"📊 İşleniyor: {symbol} - {timeframe} ({filename})")
            
            # Excel dosyasını oku
            df = pd.read_excel(file_path)
            
            if df.empty:
                logger.warning(f"❌ Boş dosya: {filename}")
                return 0
            
            # Sütun adlarını normalize et
            df = self.normalize_column_names(df)
            
            # Symbol ve timeframe ekle
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # NaN değerleri temizle
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            
            # Tarih formatını düzenle
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.strftime('%Y-%m-%d')
            
            # Database'e yaz
            conn = sqlite3.connect(self.db_path, timeout=30)
            
            # Conflict durumunda ignore et
            df.to_sql('enhanced_stock_data', conn, if_exists='append', index=False, method='multi')
            
            conn.close()
            
            logger.info(f"✅ {symbol} - {timeframe}: {len(df)} satır eklendi")
            return len(df)
            
        except Exception as e:
            logger.error(f"❌ Hata: {file_path} - {str(e)}")
            return 0
    
    def import_all_excel_files(self, data_dir="data/New_excell_Graph_Sample"):
        """Tüm Excel dosyalarını import et"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"❌ Klasör bulunamadı: {data_path}")
            return
            
        excel_files = list(data_path.glob("*.xlsx"))
        logger.info(f"🔍 {len(excel_files)} Excel dosyası bulundu")
        
        if not excel_files:
            logger.warning("❌ Excel dosyası bulunamadı")
            return
            
        total_rows = 0
        successful_files = 0
        
        # Sequential processing (daha güvenli)
        for file_path in excel_files:
            rows_added = self.process_single_file(file_path)
            if rows_added > 0:
                total_rows += rows_added
                successful_files += 1
                
        logger.info(f"""
        🎉 İMPORT TAMAMLANDI!
        =====================================
        ✅ İşlenen dosyalar: {successful_files}/{len(excel_files)}
        📊 Toplam satır: {total_rows:,}
        🗄️ Database: {self.db_path}
        📈 Ortalama: {total_rows//successful_files if successful_files > 0 else 0} satır/dosya
        """)
        
        # Database istatistikleri
        self.show_database_stats()
    
    def show_database_stats(self):
        """Database istatistiklerini göster"""
        conn = sqlite3.connect(self.db_path)
        
        # Toplam kayıt sayısı
        total_records = conn.execute("SELECT COUNT(*) FROM enhanced_stock_data").fetchone()[0]
        
        # Sembol sayısı
        symbol_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data").fetchone()[0]
        
        # Timeframe dağılımı
        timeframes = conn.execute("""
            SELECT timeframe, COUNT(*) as count 
            FROM enhanced_stock_data 
            GROUP BY timeframe 
            ORDER BY count DESC
        """).fetchall()
        
        # Tarih aralığı
        date_range = conn.execute("""
            SELECT MIN(date) as min_date, MAX(date) as max_date 
            FROM enhanced_stock_data
        """).fetchone()
        
        conn.close()
        
        logger.info(f"""
        📊 DATABASE İSTATİSTİKLERİ
        =====================================
        🗄️ Toplam kayıt: {total_records:,}
        📈 Sembol sayısı: {symbol_count}
        📅 Tarih aralığı: {date_range[0]} - {date_range[1]}
        
        ⏰ Timeframe Dağılımı:
        """)
        
        for timeframe, count in timeframes:
            logger.info(f"  {timeframe}: {count:,} kayıt")

def main():
    """Ana fonksiyon"""
    logger.info("🚀 Excel to Database Import başlıyor...")
    
    importer = ExcelToDatabaseImporter()
    importer.import_all_excel_files()
    
    logger.info("🎯 Import işlemi tamamlandı!")

if __name__ == "__main__":
    main()
