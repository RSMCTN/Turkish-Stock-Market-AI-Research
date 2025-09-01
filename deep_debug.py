#!/usr/bin/env python3
"""
🔍 DEEP DEBUG - Analyze why 0 records are processed
"""

import pandas as pd
from pathlib import Path
import psycopg2
import os

def deep_debug_single_file():
    # Test single file
    file_path = Path("data/New_excell_Graph_C_D/CRDFA_60Dk.xlsx")
    
    print(f"🔍 DEEP DEBUGGING: {file_path.name}")
    
    try:
        # Read Excel
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"📊 Original shape: {df.shape}")
        
        # Strip column names
        df.columns = df.columns.str.strip()
        print(f"📋 Stripped columns: {list(df.columns)}")
        
        # Check first row data
        print(f"\n📊 First row raw data:")
        for col in df.columns:
            print(f"  {col}: {df[col].iloc[0]} (type: {type(df[col].iloc[0])})")
        
        # Column mapping
        column_mapping = {}
        for col in df.columns:
            col_clean = col.lower().strip()
            if 'date' in col_clean:
                column_mapping[col] = 'date'
            elif 'time' in col_clean:
                column_mapping[col] = 'time'
            elif col_clean == 'open':
                column_mapping[col] = 'open'
            elif col_clean == 'high':
                column_mapping[col] = 'high'
            elif col_clean == 'low':
                column_mapping[col] = 'low'
            elif col_clean == 'close':
                column_mapping[col] = 'close'
            elif col_clean == 'volume':
                column_mapping[col] = 'volume'
        
        print(f"\n🔄 Basic column mapping: {column_mapping}")
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Check if we have required columns
        required_cols = ['date', 'open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ Missing required columns: {missing}")
            return
        
        print(f"✅ All required columns present")
        
        # Process dates
        print(f"\n📅 Processing dates...")
        print(f"Sample dates: {df['date'].head(3).tolist()}")
        
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
            print(f"✅ DD.MM.YYYY parsing successful")
        except:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            print(f"✅ Auto parsing successful")
        
        # Check for null dates
        null_dates = df['date'].isna().sum()
        print(f"📅 Null dates: {null_dates}/{len(df)}")
        
        if null_dates == len(df):
            print(f"❌ ALL dates are null!")
            return
        
        # Remove null dates
        df = df.dropna(subset=['date'])
        print(f"📊 After removing null dates: {df.shape}")
        
        if df.empty:
            print(f"❌ DataFrame empty after date processing")
            return
        
        # Create datetime column
        if 'time' in df.columns:
            print(f"⏰ Creating datetime with time column...")
            print(f"Sample times: {df['time'].head(3).tolist()}")
            
            df['datetime'] = pd.to_datetime(
                df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str),
                errors='coerce'
            )
        else:
            print(f"⏰ Creating datetime without time...")
            df['datetime'] = df['date']
        
        # Check datetime processing
        null_datetimes = df['datetime'].isna().sum()
        print(f"⏰ Null datetimes: {null_datetimes}/{len(df)}")
        
        if null_datetimes == len(df):
            print(f"❌ ALL datetimes are null!")
            return
        
        # Remove null datetimes
        df = df.dropna(subset=['datetime'])
        print(f"📊 After datetime processing: {df.shape}")
        
        if df.empty:
            print(f"❌ DataFrame empty after datetime processing")
            return
        
        # Check data types and sample values
        print(f"\n📊 Sample processed data:")
        sample_row = df.iloc[0]
        print(f"  datetime: {sample_row['datetime']}")
        print(f"  date: {sample_row['datetime'].date()}")
        print(f"  time: {sample_row['datetime'].time()}")
        print(f"  open: {sample_row.get('open', 'N/A')} (type: {type(sample_row.get('open'))})")
        print(f"  high: {sample_row.get('high', 'N/A')} (type: {type(sample_row.get('high'))})")
        print(f"  close: {sample_row.get('close', 'N/A')} (type: {type(sample_row.get('close'))})")
        
        # Try to create one record
        print(f"\n🔄 Creating sample record...")
        sample_record = {
            'symbol': 'CRDFA',
            'date': sample_row['datetime'].date(),
            'time': sample_row['datetime'].time(),
            'timeframe': '60min',
            'open': float(sample_row.get('open', 0)) if pd.notna(sample_row.get('open')) else None,
            'high': float(sample_row.get('high', 0)) if pd.notna(sample_row.get('high')) else None,
            'low': float(sample_row.get('low', 0)) if pd.notna(sample_row.get('low')) else None,
            'close': float(sample_row.get('close', 0)) if pd.notna(sample_row.get('close')) else None,
            'volume': float(sample_row.get('volume', 0)) if pd.notna(sample_row.get('volume')) else None,
        }
        
        print(f"✅ Sample record created: {sample_record}")
        
        # Check if any values are zero
        ohlc_values = [sample_record['open'], sample_record['high'], sample_record['low'], sample_record['close']]
        zero_count = sum(1 for v in ohlc_values if v == 0 or v is None)
        print(f"📊 Zero/None OHLC values: {zero_count}/4")
        
        if zero_count == 4:
            print(f"❌ ALL OHLC values are zero/None - this might be the problem!")
        
        # Test database insertion
        print(f"\n💾 Testing database insertion...")
        DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')
        
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            insert_sql = """
            INSERT INTO enhanced_stock_data 
            (symbol, date, time, timeframe, open, high, low, close, volume, created_at)
            VALUES (%(symbol)s, %(date)s, %(time)s, %(timeframe)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, CURRENT_TIMESTAMP)
            ON CONFLICT (symbol, date, time, timeframe) DO NOTHING
            RETURNING id
            """
            
            cursor.execute(insert_sql, sample_record)
            result = cursor.fetchone()
            
            if result:
                print(f"✅ Database insert successful! ID: {result[0]}")
                conn.commit()
            else:
                print(f"⚠️ Insert returned no result (probably duplicate)")
                conn.rollback()
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Database insert error: {e}")
        
    except Exception as e:
        print(f"❌ Deep debug error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    deep_debug_single_file()
