#!/usr/bin/env python3
"""
Railway üzerinde SQLite -> PostgreSQL migration
Bu script Railway'e deploy edilip çalıştırılacak
"""
import sqlite3
import psycopg2
import psycopg2.extras
import os
import sys
from datetime import datetime

def migrate_on_railway():
    """Railway üzerinde migration yap"""
    print("🚀 Railway PostgreSQL Migration başlıyor...")
    
    # Railway environment variables
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL environment variable bulunamadı!")
        return False
    
    print(f"🔗 PostgreSQL URL: {DATABASE_URL[:50]}...")
    
    # SQLite dosyası lokal - Railway'e upload edilmeli
    sqlite_db = "enhanced_bist_data.db"
    if not os.path.exists(sqlite_db):
        print(f"❌ SQLite database bulunamadı: {sqlite_db}")
        print("📁 Mevcut dosyalar:")
        for file in os.listdir('.'):
            if file.endswith('.db'):
                print(f"  - {file}")
        return False
    
    try:
        print("📊 SQLite'a bağlanıyor...")
        sqlite_conn = sqlite3.connect(sqlite_db)
        sqlite_conn.row_factory = sqlite3.Row
        cursor = sqlite_conn.cursor()
        
        print("🐘 PostgreSQL'e bağlanıyor...")
        pg_conn = psycopg2.connect(DATABASE_URL)
        pg_cursor = pg_conn.cursor()
        
        # Create table with enhanced structure
        print("🏗️ PostgreSQL tablosu oluşturuluyor...")
        create_sql = """
        DROP TABLE IF EXISTS enhanced_stock_data;
        
        CREATE TABLE enhanced_stock_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            time TIME,
            timeframe VARCHAR(20) NOT NULL,
            open DECIMAL(12,6),
            high DECIMAL(12,6),
            low DECIMAL(12,6),
            close DECIMAL(12,6),
            volume BIGINT,
            rsi_14 DECIMAL(12,8),
            macd_26_12 DECIMAL(12,8),
            macd_trigger_9 DECIMAL(12,8),
            bol_upper_20_2 DECIMAL(12,8),
            bol_middle_20_2 DECIMAL(12,8),
            bol_lower_20_2 DECIMAL(12,8),
            atr_14 DECIMAL(12,8),
            adx_14 DECIMAL(12,8),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_enhanced_symbol_timeframe ON enhanced_stock_data(symbol, timeframe);
        CREATE INDEX idx_enhanced_date ON enhanced_stock_data(date DESC);
        CREATE INDEX idx_enhanced_symbol_date ON enhanced_stock_data(symbol, date DESC);
        """
        
        pg_cursor.execute(create_sql)
        pg_conn.commit()
        print("✅ PostgreSQL tablo ve indexler hazır")
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
        total = cursor.fetchone()[0]
        print(f"📊 Toplam kayıt: {total:,}")
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data")
        symbols_count = cursor.fetchone()[0]
        print(f"📈 Sembol sayısı: {symbols_count}")
        
        # Batch migration
        batch_size = 5000  # Railway için optimize
        processed = 0
        
        print("🔄 Veri aktarımı başlıyor...")
        cursor.execute("""
            SELECT symbol, date, time, timeframe, 
                   open, high, low, close, volume,
                   rsi_14, macd_26_12, macd_trigger_9,
                   bol_upper_20_2, bol_middle_20_2, bol_lower_20_2,
                   atr_14, adx_14
            FROM enhanced_stock_data 
            ORDER BY symbol, date DESC
        """)
        
        insert_sql = """
            INSERT INTO enhanced_stock_data 
            (symbol, date, time, timeframe, open, high, low, close, volume,
             rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
             bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
            VALUES %s
        """
        
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            
            values = []
            for row in batch:
                values.append((
                    row['symbol'], row['date'], row['time'], row['timeframe'],
                    row['open'], row['high'], row['low'], row['close'], row['volume'],
                    row['rsi_14'], row['macd_26_12'], row['macd_trigger_9'],
                    row['bol_upper_20_2'], row['bol_middle_20_2'], row['bol_lower_20_2'],
                    row['atr_14'], row['adx_14']
                ))
            
            psycopg2.extras.execute_values(
                pg_cursor, insert_sql, values, template=None, page_size=batch_size
            )
            pg_conn.commit()
            
            processed += len(batch)
            progress = processed / total * 100
            print(f"⚡ Progress: {processed:,}/{total:,} (%{progress:.1f})")
        
        # Final verification
        pg_cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
        pg_total = pg_cursor.fetchone()[0]
        
        pg_cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data") 
        pg_symbols = pg_cursor.fetchone()[0]
        
        print(f"✅ Migration tamamlandı!")
        print(f"📊 PostgreSQL: {pg_total:,} kayıt")
        print(f"📈 Semboller: {pg_symbols} adet")
        print(f"🎯 Başarı: %{(pg_total/total)*100:.1f}")
        
        # Sample data check
        pg_cursor.execute("""
            SELECT symbol, COUNT(*) as records, 
                   MIN(date) as first_date, MAX(date) as last_date
            FROM enhanced_stock_data 
            GROUP BY symbol 
            ORDER BY records DESC 
            LIMIT 5
        """)
        
        print("🔍 Sample verification:")
        for row in pg_cursor.fetchall():
            print(f"  {row[0]}: {row[1]:,} records ({row[2]} → {row[3]})")
        
        pg_conn.close()
        sqlite_conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Railway PostgreSQL Migration Script")
    print("=" * 50)
    
    success = migrate_on_railway()
    
    if success:
        print("\n🎉 SUCCESS! Excel verileri PostgreSQL'e aktarıldı!")
        print("🔗 Railway API artık gerçek verileri kullanacak")
        print("📈 1500 Excel dosyası için hazır!")
    else:
        print("\n💥 FAILED! Migration başarısız")
        sys.exit(1)
