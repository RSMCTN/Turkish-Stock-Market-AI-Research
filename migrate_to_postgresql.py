#!/usr/bin/env python3
"""
SQLite'dan PostgreSQL'e Excel verilerini migrate et
"""
import sqlite3
import psycopg2
import psycopg2.extras
import os
from datetime import datetime

def migrate_data():
    # Database URLs
    sqlite_db = "enhanced_bist_data.db"
    postgres_url = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@postgres.railway.internal:5432/railway"
    
    print("🔄 SQLite -> PostgreSQL migration başlıyor...")
    
    # SQLite connection
    sqlite_conn = sqlite3.connect(sqlite_db)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()
    
    try:
        # PostgreSQL connection - Railway internal network'den
        print("🔌 PostgreSQL'e bağlanıyor...")
        pg_conn = psycopg2.connect(postgres_url)
        pg_cursor = pg_conn.cursor()
        
        # Create table if not exists
        print("📊 PostgreSQL tablosu oluşturuluyor...")
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS enhanced_stock_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            time TIME,
            timeframe VARCHAR(10) NOT NULL,
            open DECIMAL(10,4),
            high DECIMAL(10,4), 
            low DECIMAL(10,4),
            close DECIMAL(10,4),
            volume BIGINT,
            rsi_14 DECIMAL(10,6),
            macd_26_12 DECIMAL(10,6),
            macd_trigger_9 DECIMAL(10,6),
            bol_upper_20_2 DECIMAL(10,6),
            bol_middle_20_2 DECIMAL(10,6),
            bol_lower_20_2 DECIMAL(10,6),
            atr_14 DECIMAL(10,6),
            adx_14 DECIMAL(10,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON enhanced_stock_data(symbol, timeframe);
        CREATE INDEX IF NOT EXISTS idx_date ON enhanced_stock_data(date);
        """
        
        pg_cursor.execute(create_table_sql)
        pg_conn.commit()
        print("✅ PostgreSQL tablosu hazır")
        
        # Clear existing data
        pg_cursor.execute("TRUNCATE enhanced_stock_data RESTART IDENTITY;")
        pg_conn.commit()
        print("🧹 Eski veriler temizlendi")
        
        # Count total records
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
        total_records = cursor.fetchone()[0]
        print(f"📊 Toplam kayıt sayısı: {total_records:,}")
        
        # Batch migration
        batch_size = 1000
        processed = 0
        
        cursor.execute("""
            SELECT symbol, date, time, timeframe, open, high, low, close, volume,
                   rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2, 
                   bol_middle_20_2, bol_lower_20_2, atr_14, adx_14
            FROM enhanced_stock_data
            ORDER BY symbol, date, time
        """)
        
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
                
            # Insert batch
            insert_sql = """
                INSERT INTO enhanced_stock_data 
                (symbol, date, time, timeframe, open, high, low, close, volume,
                 rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2, 
                 bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
                VALUES %s
            """
            
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
            print(f"⚡ İşlendi: {processed:,}/{total_records:,} (%{processed/total_records*100:.1f})")
        
        # Verify migration
        pg_cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
        pg_count = pg_cursor.fetchone()[0]
        
        pg_cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data")
        pg_symbols = pg_cursor.fetchone()[0]
        
        print(f"✅ Migration tamamlandı!")
        print(f"📊 PostgreSQL: {pg_count:,} kayıt, {pg_symbols} sembol")
        print(f"🎯 Başarı oranı: %{pg_count/total_records*100:.1f}")
        
        pg_conn.close()
        
    except Exception as e:
        print(f"❌ Migration hatası: {e}")
        return False
    finally:
        sqlite_conn.close()
        
    return True

if __name__ == "__main__":
    success = migrate_data()
    if success:
        print("🚀 PostgreSQL migration başarılı! Railway'de deploy edilebilir.")
    else:
        print("💥 Migration başarısız!")
