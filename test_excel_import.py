#!/usr/bin/env python3
"""
Excel Import Test - Küçük örnek ile test
MAMUT_R600 - Sadece 3 dosya ile test
"""
import sys
import os
from excel_to_database_importer import ExcelToDatabaseImporter
from pathlib import Path
import shutil

def test_with_sample_files():
    """Küçük bir sample ile test et"""
    
    # Test için geçici klasör
    test_dir = "data/test_sample"
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # Orijinal dosyalardan 3 tanesini kopyala
    original_dir = Path("data/New_excell_Graph_Sample")
    excel_files = list(original_dir.glob("*.xlsx"))
    
    if len(excel_files) < 3:
        print("❌ En az 3 Excel dosyası gerekli")
        return
    
    # Farklı timeframe'lerden birer örnek al
    sample_files = []
    
    # 30dk örneği
    for f in excel_files:
        if "_30Dk.xlsx" in f.name:
            sample_files.append(f)
            break
    
    # 60dk örneği  
    for f in excel_files:
        if "_60Dk.xlsx" in f.name:
            sample_files.append(f)
            break
    
    # Günlük örneği
    for f in excel_files:
        if "_Günlük.xlsx" in f.name:
            sample_files.append(f)
            break
    
    print(f"🧪 Test için seçilen dosyalar:")
    for i, f in enumerate(sample_files[:3], 1):
        target = Path(test_dir) / f.name
        shutil.copy2(f, target)
        print(f"  {i}. {f.name}")
    
    # Test database
    test_db = "test_enhanced_bist.db"
    
    # Eski test database'ini sil
    if os.path.exists(test_db):
        os.remove(test_db)
    
    # Test import
    print("\n🚀 Test import başlıyor...")
    importer = ExcelToDatabaseImporter(db_path=test_db)
    importer.import_all_excel_files(data_dir=test_dir)
    
    # Test sonuçlarını kontrol et
    import sqlite3
    conn = sqlite3.connect(test_db)
    
    results = conn.execute("""
        SELECT symbol, timeframe, COUNT(*) as records, 
               MIN(date) as first_date, MAX(date) as last_date
        FROM enhanced_stock_data 
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """).fetchall()
    
    print(f"\n📊 TEST SONUÇLARI:")
    print("="*60)
    
    for symbol, timeframe, records, first_date, last_date in results:
        print(f"📈 {symbol:8} | {timeframe:6} | {records:5,} kayıt | {first_date} → {last_date}")
    
    # Örnek teknik indikatör verileri
    sample_data = conn.execute("""
        SELECT symbol, date, time, close, rsi_14, macd_26_12, bol_upper_20_2
        FROM enhanced_stock_data 
        WHERE rsi_14 > 0 AND macd_26_12 != 0
        LIMIT 5
    """).fetchall()
    
    if sample_data:
        print(f"\n🔍 ÖRNEK TEKNİK İNDİKATÖR VERİLERİ:")
        print("="*60)
        print(f"{'Symbol':<8} | {'Date':<10} | {'Time':<5} | {'Close':<8} | {'RSI':<6} | {'MACD':<8} | {'BB Upper':<8}")
        print("-" * 60)
        
        for symbol, date, time, close, rsi, macd, bb_upper in sample_data:
            print(f"{symbol:<8} | {date:<10} | {time:<5} | {close:<8.2f} | {rsi:<6.2f} | {macd:<8.4f} | {bb_upper:<8.2f}")
    
    conn.close()
    
    print(f"\n✅ Test tamamlandı! Database: {test_db}")
    print(f"📁 Test dosyalar: {test_dir}")

if __name__ == "__main__":
    test_with_sample_files()
