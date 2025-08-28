import psycopg2
import json
from datetime import datetime

# Railway PostgreSQL connection
DATABASE_URL = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"

def backup_database():
    try:
        print("🔗 PostgreSQL bağlantısı kuruluyor...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        backup_data = {
            "backup_date": datetime.now().isoformat(),
            "database_info": {},
            "stocks": [],
            "historical_data_sample": [],
            "stats": {}
        }
        
        print("📊 Database stats alınıyor...")
        
        # Stocks count
        cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_active = true")
        stocks_count = cursor.fetchone()[0]
        
        # Historical data count  
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        historical_count = cursor.fetchone()[0]
        
        # Date range
        cursor.execute("SELECT MIN(date_time), MAX(date_time) FROM historical_data")
        date_range = cursor.fetchone()
        
        backup_data["stats"] = {
            "total_stocks": stocks_count,
            "total_historical_records": historical_count,
            "date_range_start": str(date_range[0]),
            "date_range_end": str(date_range[1])
        }
        
        print("🏢 Stocks backup alınıyor...")
        cursor.execute("SELECT * FROM stocks WHERE is_active = true LIMIT 100")
        stocks_columns = [desc[0] for desc in cursor.description]
        stocks_data = cursor.fetchall()
        
        backup_data["stocks"] = [
            dict(zip(stocks_columns, row)) for row in stocks_data
        ]
        
        print("📈 Historical data sample alınıyor...")
        cursor.execute("""
            SELECT * FROM historical_data 
            ORDER BY date_time DESC 
            LIMIT 1000
        """)
        hist_columns = [desc[0] for desc in cursor.description]
        hist_data = cursor.fetchall()
        
        backup_data["historical_data_sample"] = [
            dict(zip(hist_columns, [str(val) if val else None for val in row])) 
            for row in hist_data
        ]
        
        # Save backup
        backup_filename = f"bist_postgresql_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        with open(backup_filename, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Backup tamamlandı: {backup_filename}")
        print(f"📊 Stats: {stocks_count} stocks, {historical_count} historical records")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Backup error: {e}")

if __name__ == "__main__":
    backup_database()
