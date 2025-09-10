#!/usr/bin/env python3
"""
Direct import C-D-E-F data to clone table, then merge in PostgreSQL
"""

import psycopg2
import os
import time

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')

def import_cdef_to_clone():
    """Import C-D-E-F data to clone table using existing CSV"""
    
    print('🚀 Importing C-D-E-F data to clone table...')
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Set longer timeout
        cursor.execute('SET statement_timeout = "15min"')
        
        # Use the comprehensive CSV we created
        csv_file = 'cdef_combined.csv'
        
        print(f'📂 Using CSV: {csv_file}')
        
        with open(csv_file, 'r') as f:
            # Skip header
            next(f)
            
            print('📤 Starting COPY to clone table...')
            start_time = time.time()
            
            cursor.copy_expert('''
                COPY enhanced_stock_data_1 
                (symbol, date, time, timeframe, open, high, low, close, volume,
                 rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
                 bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
                FROM STDIN WITH CSV
                ''', f)
        
        conn.commit()
        elapsed = time.time() - start_time
        
        print(f'✅ COPY to clone completed in {elapsed:.1f} seconds!')
        
        # Check clone table results
        cursor.execute('SELECT COUNT(*) FROM enhanced_stock_data_1')
        clone_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data_1")
        clone_symbols = cursor.fetchone()[0]
        
        print(f'📊 Clone table records: {clone_records:,}')
        print(f'🎯 Clone table symbols: {clone_symbols}')
        
        if clone_records > 1000000:
            print('✅ Clone import successful! Ready for merge.')
            return True
        else:
            print('⚠️ Clone import incomplete.')
            return False
            
        conn.close()
        
    except Exception as e:
        print(f'❌ Clone import error: {e}')
        return False

def merge_tables():
    """Merge clone table data into main table using PostgreSQL"""
    
    print('🔄 Merging clone data into main table...')
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        print('📤 Starting PostgreSQL merge...')
        start_time = time.time()
        
        # Insert new records from clone to main table
        cursor.execute('''
            INSERT INTO enhanced_stock_data 
            (symbol, date, time, timeframe, open, high, low, close, volume,
             rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
             bol_middle_20_2, bol_lower_20_2, atr_14, adx_14, created_at)
            SELECT symbol, date, time, timeframe, open, high, low, close, volume,
                   rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
                   bol_middle_20_2, bol_lower_20_2, atr_14, adx_14, CURRENT_TIMESTAMP
            FROM enhanced_stock_data_1
            ON CONFLICT (symbol, date, time, timeframe) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                rsi_14 = EXCLUDED.rsi_14,
                macd_26_12 = EXCLUDED.macd_26_12,
                macd_trigger_9 = EXCLUDED.macd_trigger_9,
                bol_upper_20_2 = EXCLUDED.bol_upper_20_2,
                bol_middle_20_2 = EXCLUDED.bol_middle_20_2,
                bol_lower_20_2 = EXCLUDED.bol_lower_20_2,
                atr_14 = EXCLUDED.atr_14,
                adx_14 = EXCLUDED.adx_14
        ''')
        
        conn.commit()
        elapsed = time.time() - start_time
        
        print(f'✅ Merge completed in {elapsed:.1f} seconds!')
        
        # Final verification
        cursor.execute('SELECT COUNT(*) FROM enhanced_stock_data')
        final_total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM enhanced_stock_data")
        final_symbols = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM enhanced_stock_data WHERE symbol LIKE 'C%' OR symbol LIKE 'D%' OR symbol LIKE 'E%' OR symbol LIKE 'F%'")
        cdef_records = cursor.fetchone()[0]
        
        print(f'🎉 FINAL RESULTS:')
        print(f'📊 Total records: {final_total:,}')
        print(f'🎯 Total symbols: {final_symbols}')
        print(f'📈 C-D-E-F records: {cdef_records:,}')
        
        # Clean up clone table
        cursor.execute('DROP TABLE enhanced_stock_data_1')
        conn.commit()
        print('🧹 Clone table cleaned up')
        
        conn.close()
        
    except Exception as e:
        print(f'❌ Merge error: {e}')

def main():
    if import_cdef_to_clone():
        merge_tables()
    else:
        print('❌ Clone import failed, skipping merge')

if __name__ == "__main__":
    main()