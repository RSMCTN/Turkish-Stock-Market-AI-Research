#!/usr/bin/env python3
"""
353 Excel dosyasını toplu olarak SQLite'a import et
New_excell_Graph_Sample klasöründeki tüm .xlsx dosyalarını işle
"""

import pandas as pd
import sqlite3
import os
import glob
from datetime import datetime
from pathlib import Path
import time

def parse_timeframe_from_filename(filename):
    """Dosya adından timeframe çıkar"""
    if '_30Dk.' in filename:
        return '30m'
    elif '_60Dk.' in filename:
        return '60m'
    elif '_20Dk.' in filename:
        return '20m'
    elif '_Günlük.' in filename:
        return 'günlük'
    else:
        return 'unknown'

def parse_symbol_from_filename(filename):
    """Dosya adından sembol çıkar"""
    basename = os.path.basename(filename)
    # SYMBOL_TimeFrame.xlsx formatı
    return basename.split('_')[0]

def import_excel_file(file_path, conn, cursor):
    """Tek Excel dosyasını import et"""
    try:
        symbol = parse_symbol_from_filename(file_path)
        timeframe = parse_timeframe_from_filename(file_path)
        
        print(f"📊 İşleniyor: {symbol} - {timeframe}")
        
        # Excel'i oku
        df = pd.read_excel(file_path)
        
        # Kolon isimlerini normalize et (Türkçe karakterler vs)
        df.columns = df.columns.str.strip().str.lower()
        
        # Tarih kolonu bul
        date_cols = [col for col in df.columns if any(x in col for x in ['tarih', 'date', 'time'])]
        
        if not date_cols:
            print(f"⚠️ {symbol}: Tarih kolonu bulunamadı")
            return 0
        
        date_col = date_cols[0]
        
        # Temel kolonları map et
        column_mapping = {
            'açılış': 'open',
            'acilis': 'open', 
            'open': 'open',
            'yüksek': 'high',
            'yuksek': 'high',
            'high': 'high',
            'düşük': 'low', 
            'dusuk': 'low',
            'low': 'low',
            'kapanış': 'close',
            'kapanis': 'close',
            'close': 'close',
            'hacim': 'volume',
            'volume': 'volume'
        }
        
        # Teknik indikatör kolonları
        indicator_mapping = {
            'rsi': 'rsi_14',
            'rsi_14': 'rsi_14',
            'rsi14': 'rsi_14',
            'macd': 'macd_26_12',
            'macd_line': 'macd_26_12',
            'macd_signal': 'macd_trigger_9',
            'macd_trigger': 'macd_trigger_9',
            'bb_upper': 'bol_upper_20_2',
            'bb_middle': 'bol_middle_20_2', 
            'bb_lower': 'bol_lower_20_2',
            'bollinger_upper': 'bol_upper_20_2',
            'bollinger_middle': 'bol_middle_20_2',
            'bollinger_lower': 'bol_lower_20_2',
            'atr': 'atr_14',
            'atr_14': 'atr_14',
            'adx': 'adx_14',
            'adx_14': 'adx_14'
        }
        
        # Tüm mapping'i birleştir
        all_mapping = {**column_mapping, **indicator_mapping}
        
        # Kolonları rename et
        df_renamed = df.rename(columns=all_mapping)
        
        # Gerekli kolonları kontrol et
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_renamed.columns]
        
        if missing_cols:
            print(f"⚠️ {symbol}: Eksik kolonlar: {missing_cols}")
            return 0
        
        # Tarih ve zaman ayrıştır
        df_renamed['symbol'] = symbol
        df_renamed['timeframe'] = timeframe
        
        # Tarih işleme
        try:
            if timeframe == 'günlük':
                df_renamed['date'] = pd.to_datetime(df_renamed[date_col]).dt.date
                df_renamed['time'] = None
            else:
                # Intraday data - tarih ve saat ayrıştır
                datetime_col = pd.to_datetime(df_renamed[date_col])
                df_renamed['date'] = datetime_col.dt.date
                df_renamed['time'] = datetime_col.dt.time
        except Exception as e:
            print(f"⚠️ {symbol}: Tarih parse hatası: {e}")
            return 0
        
        # Volume default değer
        if 'volume' not in df_renamed.columns:
            df_renamed['volume'] = 0
        
        # Teknik indikatörleri kontrol et ve default değer ata
        indicator_cols = ['rsi_14', 'macd_26_12', 'macd_trigger_9', 
                         'bol_upper_20_2', 'bol_middle_20_2', 'bol_lower_20_2',
                         'atr_14', 'adx_14']
        
        for col in indicator_cols:
            if col not in df_renamed.columns:
                df_renamed[col] = None
        
        # Database'e insert
        insert_cols = ['symbol', 'date', 'time', 'timeframe', 'open', 'high', 'low', 'close', 'volume'] + indicator_cols
        
        inserted_count = 0
        for _, row in df_renamed.iterrows():
            try:
                values = [row.get(col) for col in insert_cols]
                
                cursor.execute(f"""
                    INSERT OR REPLACE INTO enhanced_stock_data 
                    ({', '.join(insert_cols)})
                    VALUES ({', '.join(['?' for _ in insert_cols])})
                """, values)
                
                inserted_count += 1
                
            except Exception as e:
                # Duplicate veya diğer hataları sessizce atla
                continue
        
        conn.commit()
        print(f"✅ {symbol}-{timeframe}: {inserted_count} kayıt eklendi")
        return inserted_count
        
    except Exception as e:
        print(f"❌ {file_path} işlenirken hata: {e}")
        return 0

def main():
    """353 Excel dosyasını toplu import et"""
    print("🚀 353 Excel Dosyası Toplu Import Başlıyor...")
    print("=" * 60)
    
    # Database bağlantısı
    db_path = "enhanced_bist_data.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Excel dosyalarını bul
    excel_dir = "data/New_excell_Graph_Sample/"
    excel_files = glob.glob(os.path.join(excel_dir, "*.xlsx"))
    
    print(f"📂 Bulunan Excel dosyası: {len(excel_files)}")
    
    total_imported = 0
    processed_symbols = set()
    errors = 0
    
    start_time = time.time()
    
    for i, excel_file in enumerate(excel_files, 1):
        symbol = parse_symbol_from_filename(excel_file)
        timeframe = parse_timeframe_from_filename(excel_file)
        
        print(f"\n[{i:3d}/{len(excel_files)}] Processing: {symbol}_{timeframe}")
        
        try:
            imported = import_excel_file(excel_file, conn, cursor)
            total_imported += imported
            processed_symbols.add(symbol)
            
            # Her 10 dosyada bir progress raporu
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(excel_files) - i) * avg_time
                print(f"📊 Progress: {i}/{len(excel_files)} ({i/len(excel_files)*100:.1f}%) - ETA: {remaining/60:.1f} min")
                
        except Exception as e:
            print(f"❌ {excel_file}: {e}")
            errors += 1
            continue
    
    # Final istatistikler
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🎉 TOPLU IMPORT TAMAMLANDI!")
    print(f"📊 Toplam işlenen dosya: {len(excel_files)}")
    print(f"✅ Başarılı import: {len(excel_files) - errors}")
    print(f"❌ Hata: {errors}")
    print(f"📈 Toplam kayıt: {total_imported:,}")
    print(f"🏢 Benzersiz sembol: {len(processed_symbols)}")
    print(f"⏱️ Süre: {elapsed_time/60:.1f} dakika")
    print(f"⚡ Ortalama: {elapsed_time/len(excel_files):.1f} saniye/dosya")
    
    # Database istatistikleri
    cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data")
    total_symbols = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT timeframe) FROM enhanced_stock_data")
    total_timeframes = cursor.fetchone()[0]
    
    print(f"\n📊 DATABASE DURUMU:")
    print(f"📈 Toplam kayıt: {total_records:,}")
    print(f"🏢 Toplam sembol: {total_symbols}")
    print(f"⏰ Toplam timeframe: {total_timeframes}")
    
    # Sample verification
    cursor.execute("""
        SELECT symbol, timeframe, COUNT(*) as records
        FROM enhanced_stock_data 
        GROUP BY symbol, timeframe 
        ORDER BY records DESC 
        LIMIT 10
    """)
    
    print(f"\n🔍 En Zengin Veri (İlk 10):")
    for row in cursor.fetchall():
        print(f"  {row[0]} ({row[1]}): {row[2]:,} kayıt")
    
    conn.close()
    print(f"\n🚀 Import başarıyla tamamlandı! Database: {db_path}")

if __name__ == "__main__":
    main()
