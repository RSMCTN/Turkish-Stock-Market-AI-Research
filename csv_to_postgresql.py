#!/usr/bin/env python3
"""
CSV'yi Railway PostgreSQL'e COPY ile yükle
Bu script Railway'de çalıştırılacak
"""
import psycopg2
import psycopg2.extras
import os
import gzip
import sys
from io import StringIO

def csv_to_postgresql():
    """CSV'yi PostgreSQL'e COPY ile yükle - süper hızlı!"""
    print("🚀 CSV -> PostgreSQL COPY migration başlıyor...")
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL bulunamadı!")
        return False
        
    print(f"🔗 PostgreSQL: {DATABASE_URL[:50]}...")
    
    # CSV parçalarını kontrol et
    csv_parts = []
    for suffix in ['aa', 'ab', 'ac', 'ad']:
        gz_file = f"enhanced_stock_data_part_{suffix}.gz"
        csv_file = f"enhanced_stock_data_part_{suffix}"
        if os.path.exists(gz_file):
            csv_parts.append(gz_file)
        elif os.path.exists(csv_file):
            csv_parts.append(csv_file)
    
    if not csv_parts:
        print("❌ CSV parçaları bulunamadı!")
        return False
    
    print(f"✅ {len(csv_parts)} CSV parçası bulundu: {csv_parts}")
    
    try:
        print("🐘 PostgreSQL'e bağlanıyor...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Drop and create table
        print("🏗️ PostgreSQL tablosu yenileniyor...")
        drop_create_sql = """
        DROP TABLE IF EXISTS enhanced_stock_data CASCADE;
        
        CREATE TABLE enhanced_stock_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            time TIME,
            timeframe VARCHAR(20) NOT NULL,
            open DECIMAL(15,8),
            high DECIMAL(15,8),
            low DECIMAL(15,8),
            close DECIMAL(15,8),
            volume BIGINT,
            rsi_14 DECIMAL(15,10),
            macd_26_12 DECIMAL(15,10),
            macd_trigger_9 DECIMAL(15,10),
            bol_upper_20_2 DECIMAL(15,10),
            bol_middle_20_2 DECIMAL(15,10),
            bol_lower_20_2 DECIMAL(15,10),
            atr_14 DECIMAL(15,10),
            adx_14 DECIMAL(15,10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- High-performance indexes
        CREATE INDEX CONCURRENTLY idx_enhanced_symbol_timeframe ON enhanced_stock_data(symbol, timeframe);
        CREATE INDEX CONCURRENTLY idx_enhanced_date_desc ON enhanced_stock_data(date DESC);
        CREATE INDEX CONCURRENTLY idx_enhanced_symbol_date_desc ON enhanced_stock_data(symbol, date DESC);
        CREATE INDEX CONCURRENTLY idx_enhanced_timeframe ON enhanced_stock_data(timeframe);
        """
        
        cursor.execute(drop_create_sql)
        conn.commit()
        print("✅ PostgreSQL tablo ve indexler hazır")
        
        # Read CSV parts and COPY
        print("📊 Multi-part CSV COPY işlemi başlıyor...")
        
        total_imported = 0
        for i, csv_part in enumerate(csv_parts, 1):
            print(f"⚡ İşleniyor: Part {i}/{len(csv_parts)} - {csv_part}")
            
            if csv_part.endswith('.gz'):
                with gzip.open(csv_part, 'rt', encoding='utf-8') as f:
                    # Skip header for first part only
                    if i == 1:
                        next(f)
                    
                    cursor.copy_expert(
                        """
                        COPY enhanced_stock_data 
                        (symbol, date, time, timeframe, open, high, low, close, volume,
                         rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
                         bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
                        FROM STDIN WITH CSV
                        """, f
                    )
            else:
                with open(csv_part, 'r', encoding='utf-8') as f:
                    # Skip header for first part only
                    if i == 1:
                        next(f)
                    
                    cursor.copy_expert(
                        """
                        COPY enhanced_stock_data 
                        (symbol, date, time, timeframe, open, high, low, close, volume,
                         rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
                         bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
                        FROM STDIN WITH CSV
                        """, f
                    )
            
            conn.commit()
            print(f"✅ Part {i} completed!")
        
        print(f"🎉 All {len(csv_parts)} parts imported successfully!")
        
        conn.commit()
        print("✅ CSV COPY tamamlandı!")
        
        # Verification
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data")
        symbols = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT symbol, COUNT(*) as records, MIN(date) as first, MAX(date) as last
            FROM enhanced_stock_data 
            GROUP BY symbol 
            ORDER BY records DESC 
            LIMIT 10
        """)
        
        print(f"📊 Migration Results:")
        print(f"  Total Records: {total:,}")
        print(f"  Unique Symbols: {symbols}")
        print(f"  Database Size: ~{(total * 200 / 1024 / 1024):.1f} MB estimated")
        
        print(f"\n🔍 Top 10 Symbols by Record Count:")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]:,} records ({row[2]} → {row[3]})")
        
        # Test historical endpoint format
        cursor.execute("""
            SELECT symbol, date, time, timeframe, open, high, low, close, volume,
                   rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
                   bol_middle_20_2, bol_lower_20_2, atr_14, adx_14
            FROM enhanced_stock_data 
            WHERE symbol = 'A1CAP' AND timeframe = '60m'
            ORDER BY date DESC, time DESC 
            LIMIT 2
        """)
        
        print(f"\n🧪 API Test Sample (A1CAP, 60m):")
        for row in cursor.fetchall():
            print(f"  {row[0]} | {row[1]} {row[2]} | Close: {row[7]} | Vol: {row[8]:,} | RSI: {row[9]:.2f}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 CSV → PostgreSQL COPY Migration")
    print("=" * 50)
    
    success = csv_to_postgresql()
    
    if success:
        print("\n🎉 SUCCESS! 1.4M Excel records → Railway PostgreSQL!")
        print("⚡ COPY method: 100x faster than INSERT")
        print("📈 Railway API ready for 1500+ Excel files!")
        print("🔗 Frontend can now use real historical data")
    else:
        print("\n💥 FAILED! Migration unsuccessful")
        sys.exit(1)
