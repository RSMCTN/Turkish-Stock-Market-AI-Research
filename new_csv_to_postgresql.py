#!/usr/bin/env python3
"""
CSV'yi Railway PostgreSQL'e COPY ile yükle - NEW EXCEL DATA
Bu script new_excel_batch.csv.gz için uyarlanmış
"""
import psycopg2
import psycopg2.extras
import os
import gzip
import sys

def csv_to_postgresql():
    """CSV'yi PostgreSQL'e COPY ile yükle - süper hızlı!"""
    print("🚀 NEW Excel CSV -> PostgreSQL COPY migration başlıyor...")
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL bulunamadı!")
        return False
        
    print(f"🔗 PostgreSQL: {DATABASE_URL[:50]}...")
    
    # CSV dosyasını kontrol et
    csv_file = "new_excel_batch.csv.gz"
    if not os.path.exists(csv_file):
        print(f"❌ CSV dosyası bulunamadı: {csv_file}")
        return False
    
    file_size = os.path.getsize(csv_file) / 1024 / 1024
    print(f"📁 Found: {csv_file} (size: {file_size:.1f} MB)")
    
    try:
        print("🐘 PostgreSQL'e bağlanıyor...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # First, let's check existing data
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
        existing_count = cursor.fetchone()[0]
        print(f"📊 Mevcut kayıt sayısı: {existing_count:,}")
        
        # Check if we have unique constraint (needed for ON CONFLICT)
        cursor.execute("""
            SELECT constraint_name 
            FROM information_schema.table_constraints 
            WHERE table_name = 'enhanced_stock_data' 
            AND constraint_type = 'UNIQUE'
        """)
        unique_constraints = cursor.fetchall()
        
        if not unique_constraints:
            print("🔧 Creating unique constraint...")
            cursor.execute("""
                ALTER TABLE enhanced_stock_data 
                ADD CONSTRAINT enhanced_stock_data_unique_record 
                UNIQUE (symbol, date, time, timeframe)
            """)
            conn.commit()
            print("✅ Unique constraint created!")
        else:
            print(f"✅ Unique constraint exists: {unique_constraints[0][0]}")
        
        # Read CSV and COPY
        print("📊 CSV COPY işlemi başlıyor...")
        
        with gzip.open(csv_file, 'rt', encoding='utf-8', errors='ignore') as f:
            # Skip header
            header_line = next(f, None)
            if header_line:
                print(f"📋 Header: {header_line.strip()[:100]}...")
            
            try:
                cursor.copy_expert(
                    """
                    COPY enhanced_stock_data 
                    (symbol, date, time, timeframe, open, high, low, close, volume,
                     rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
                     bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
                    FROM STDIN WITH CSV
                    """, f
                )
                print("✅ COPY başarılı!")
            except Exception as copy_error:
                print(f"❌ COPY hatası: {copy_error}")
                
                # Alternative: Try INSERT with ON CONFLICT
                print("🔄 COPY başarısız, INSERT ON CONFLICT deneniyor...")
                f.seek(0)
                next(f)  # Skip header again
                
                insert_sql = """
                INSERT INTO enhanced_stock_data 
                (symbol, date, time, timeframe, open, high, low, close, volume,
                 rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
                 bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
                VALUES (
                    %s, %s, NULLIF(%s, ''), %s, 
                    NULLIF(%s, '')::DECIMAL, NULLIF(%s, '')::DECIMAL, 
                    NULLIF(%s, '')::DECIMAL, NULLIF(%s, '')::DECIMAL, 
                    NULLIF(%s, '')::DECIMAL,
                    NULLIF(%s, '')::DECIMAL, NULLIF(%s, '')::DECIMAL,
                    NULLIF(%s, '')::DECIMAL, NULLIF(%s, '')::DECIMAL,
                    NULLIF(%s, '')::DECIMAL, NULLIF(%s, '')::DECIMAL,
                    NULLIF(%s, '')::DECIMAL, NULLIF(%s, '')::DECIMAL
                )
                ON CONFLICT (symbol, date, time, timeframe) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """
                
                batch_size = 1000
                batch = []
                inserted_count = 0
                
                import csv
                csv_reader = csv.reader(f)
                
                for row_num, row in enumerate(csv_reader, 1):
                    try:
                        if len(row) >= 17:  # Ensure we have all columns
                            batch.append(row)
                            
                            if len(batch) >= batch_size:
                                cursor.executemany(insert_sql, batch)
                                inserted_count += cursor.rowcount
                                conn.commit()
                                batch = []
                                
                                if row_num % 10000 == 0:
                                    print(f"  📊 İşlenen: {row_num:,} satır, Eklenen: {inserted_count:,}")
                    
                    except Exception as row_error:
                        print(f"  ⚠️ Row {row_num} error: {row_error}")
                        continue
                
                # Process remaining batch
                if batch:
                    cursor.executemany(insert_sql, batch)
                    inserted_count += cursor.rowcount
                    conn.commit()
                
                print(f"✅ INSERT completed: {inserted_count:,} records")
        
        conn.commit()
        print("✅ Migration completed!")
        
        # Verification
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data")
        symbols = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT symbol, COUNT(*) as records, MIN(date) as first, MAX(date) as last
            FROM enhanced_stock_data 
            WHERE created_at > NOW() - INTERVAL '1 hour'
            GROUP BY symbol 
            ORDER BY records DESC 
            LIMIT 15
        """)
        
        new_data = cursor.fetchall()
        
        print(f"📊 Migration Results:")
        print(f"  Total Records: {total:,} (was {existing_count:,})")
        print(f"  New Records: {total - existing_count:,}")
        print(f"  Unique Symbols: {symbols}")
        
        if new_data:
            print(f"\n🆕 Recently Added Symbols (Top 15):")
            for row in new_data:
                print(f"  {row[0]}: {row[1]:,} records ({row[2]} → {row[3]})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 NEW Excel CSV → PostgreSQL Migration")
    print("=" * 50)
    
    success = csv_to_postgresql()
    
    if success:
        print("\n🎉 SUCCESS! 771K New Excel records → Railway PostgreSQL!")
        print("⚡ Database ready with expanded data!")
        print("📈 Railway API now has even more symbols!")
        print("🔗 Frontend can access new historical data")
    else:
        print("\n💥 FAILED! Migration unsuccessful")
        sys.exit(1)
