#!/usr/bin/env python3
"""
Convert all historical Excel files to JSON format for frontend consumption
Process both 60-minute and daily data with technical indicators
"""

import pandas as pd
import json
import os
from datetime import datetime
import glob

def convert_historical_excel_to_json():
    """Convert all historical Excel files to structured JSON"""
    
    data_dir = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ"
    output_dir = "/Users/rasimcetin/rasim_claude/MAMUT_R600/trading-dashboard/public/data/historical"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all historical Excel files
    excel_files = glob.glob(os.path.join(data_dir, "*_*.xlsx"))
    excel_files = [f for f in excel_files if not f.endswith('basestock.xlsx') and not f.endswith('basestock2808.xlsx')]
    
    print(f"📊 Found {len(excel_files)} historical Excel files")
    
    # Group by symbol
    symbols_data = {}
    processed_count = 0
    
    for filepath in excel_files:
        filename = os.path.basename(filepath)
        
        # Parse symbol and timeframe from filename
        if '_60Dk.xlsx' in filename:
            symbol = filename.replace('_60Dk.xlsx', '')
            timeframe = '60min'
        elif '_Günlük.xlsx' in filename:
            symbol = filename.replace('_Günlük.xlsx', '')
            timeframe = 'daily'
        else:
            continue
            
        print(f"🔄 Processing: {symbol} ({timeframe})")
        
        try:
            # Read Excel file
            df = pd.read_excel(filepath)
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            # Convert Date column to proper datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
            
            # Create datetime index combining Date and Time
            if 'Time' in df.columns:
                # For 60min data, combine Date and Time
                if timeframe == '60min':
                    df['DateTime'] = df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'].astype(str)
                    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
                else:
                    # For daily data, just use Date
                    df['DateTime'] = df['Date']
            else:
                df['DateTime'] = df['Date']
            
            # Remove rows with invalid dates
            df = df.dropna(subset=['DateTime'])
            
            # Sort by datetime
            df = df.sort_values('DateTime')
            
            # Select relevant columns and rename them
            columns_mapping = {
                'DateTime': 'datetime',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'RSI (14)': 'rsi',
                'MACD (26,12)': 'macd',
                'TRIGGER (9)': 'macd_signal',
                'BOL U (20,2)': 'bb_upper',
                'BOL M (20,2)': 'bb_middle', 
                'BOL D (20,2)': 'bb_lower',
                'ATR (14)': 'atr',
                'ADX (14)': 'adx',
                'Tenkan-sen': 'ichimoku_tenkan',
                'Kijun-sen': 'ichimoku_kijun',
                'Senkou Span A': 'ichimoku_span_a',
                'Senkou Span B': 'ichimoku_span_b',
                'Chikou Span': 'ichimoku_chikou'
            }
            
            # Select available columns
            available_columns = {k: v for k, v in columns_mapping.items() if k in df.columns}
            df_selected = df[list(available_columns.keys())].copy()
            df_selected = df_selected.rename(columns=available_columns)
            
            # Convert datetime to ISO string
            df_selected['datetime'] = df_selected['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Fill NaN values with 0 for technical indicators
            technical_columns = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower', 
                               'atr', 'adx', 'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_span_a', 
                               'ichimoku_span_b', 'ichimoku_chikou']
            
            for col in technical_columns:
                if col in df_selected.columns:
                    df_selected[col] = df_selected[col].fillna(0)
            
            # Convert to dictionary
            data_records = df_selected.to_dict('records')
            
            # Initialize symbol data if not exists
            if symbol not in symbols_data:
                symbols_data[symbol] = {}
            
            # Add timeframe data
            symbols_data[symbol][timeframe] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'total_records': len(data_records),
                'date_range': {
                    'start': data_records[0]['datetime'] if data_records else None,
                    'end': data_records[-1]['datetime'] if data_records else None
                },
                'data': data_records[-1000:] if len(data_records) > 1000 else data_records  # Keep last 1000 records
            }
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"📈 Processed {processed_count} files...")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    # Save individual symbol files
    for symbol, symbol_data in symbols_data.items():
        symbol_file = os.path.join(output_dir, f"{symbol}.json")
        
        with open(symbol_file, 'w', encoding='utf-8') as f:
            json.dump(symbol_data, f, indent=2, ensure_ascii=False)
    
    # Create master index file
    master_index = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_symbols': len(symbols_data),
            'processed_files': processed_count,
            'description': 'Historical OHLCV + Technical Indicators Data'
        },
        'symbols': list(symbols_data.keys()),
        'timeframes': ['60min', 'daily'],
        'available_indicators': [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'adx', 'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_span_a',
            'ichimoku_span_b', 'ichimoku_chikou'
        ]
    }
    
    master_file = os.path.join(output_dir, 'index.json')
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(master_index, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Conversion completed!")
    print(f"📊 Processed: {processed_count} files")
    print(f"📈 Symbols: {len(symbols_data)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Master index: {master_file}")
    
    # Show sample data
    if symbols_data:
        sample_symbol = list(symbols_data.keys())[0]
        sample_data = symbols_data[sample_symbol]
        
        print(f"\n📋 Sample data for {sample_symbol}:")
        for timeframe, data in sample_data.items():
            print(f"   {timeframe}: {data['total_records']} records ({data['date_range']['start']} to {data['date_range']['end']})")
            if data['data']:
                latest = data['data'][-1]
                print(f"     Latest: {latest['datetime']} | Close: {latest['close']} | RSI: {latest.get('rsi', 'N/A')}")

if __name__ == "__main__":
    convert_historical_excel_to_json()
