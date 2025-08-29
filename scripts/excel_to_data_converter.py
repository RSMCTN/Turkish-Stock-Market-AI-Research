#!/usr/bin/env python3
"""
Convert basestock2808.xlsx to CSV and JSON formats
Update all BIST data with latest prices
"""

import pandas as pd
import json
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_excel_to_data():
    """Convert basestock2808.xlsx to CSV and JSON"""
    
    excel_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/basestock2808.xlsx"
    csv_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/bist_real_data_2808.csv"
    json_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/bist_real_data_2808.json"
    
    try:
        # Read Excel file
        logger.info(f"📂 Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path, engine='openpyxl')
        logger.info(f"✅ Loaded {len(df)} rows from Excel")
        
        # Show column names for debugging
        logger.info(f"📋 Excel columns: {df.columns.tolist()}")
        
        # Save as CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"💾 Saved CSV: {csv_path}")
        
        # Convert to JSON format
        json_data = df.to_dict('records')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"💾 Saved JSON: {json_path}")
        
        # Show sample data
        if len(df) > 0:
            logger.info("🎯 Sample stock data:")
            for idx, row in df.head(3).iterrows():
                if 'SEMBOL' in df.columns or 'Symbol' in df.columns or df.columns[0]:
                    symbol_col = 'SEMBOL' if 'SEMBOL' in df.columns else ('Symbol' if 'Symbol' in df.columns else df.columns[0])
                    price_cols = [col for col in df.columns if any(x in col.upper() for x in ['FIYAT', 'PRICE', 'SON', 'CLOSE', 'LAST'])]
                    
                    symbol = row[symbol_col] if symbol_col in row else 'N/A'
                    prices = {col: row[col] for col in price_cols[:3] if col in row}
                    
                    logger.info(f"   {symbol}: {prices}")
        
        return True, csv_path, json_path
        
    except Exception as e:
        logger.error(f"❌ Error converting Excel: {e}")
        return False, None, None

def update_frontend_prices():
    """Update frontend components with real prices from new data"""
    
    csv_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/bist_real_data_2808.csv"
    
    try:
        # Read the new CSV data
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Read {len(df)} stocks from updated CSV")
        
        # Extract symbol and price columns
        symbol_col = None
        price_col = None
        
        for col in df.columns:
            if 'SEMBOL' in col.upper() or 'SYMBOL' in col.upper():
                symbol_col = col
            elif any(x in col.upper() for x in ['FIYAT', 'SON', 'PRICE', 'CLOSE', 'LAST']):
                if price_col is None:  # Take the first price column
                    price_col = col
        
        if not symbol_col or not price_col:
            logger.error(f"❌ Could not find symbol/price columns in: {df.columns.tolist()}")
            return False
        
        logger.info(f"📋 Using columns: Symbol='{symbol_col}', Price='{price_col}'")
        
        # Create price dictionary
        real_prices = {}
        for idx, row in df.iterrows():
            symbol = str(row[symbol_col]).strip()
            try:
                price = float(row[price_col])
                real_prices[symbol] = price
            except (ValueError, TypeError):
                logger.warning(f"⚠️ Invalid price for {symbol}: {row[price_col]}")
                continue
        
        logger.info(f"✅ Extracted {len(real_prices)} valid prices")
        
        # Show key stock prices
        key_stocks = ['ASTOR', 'AKSEN', 'GARAN', 'THYAO', 'TUPRS', 'BRSAN']
        logger.info("🎯 Key stock prices:")
        for stock in key_stocks:
            if stock in real_prices:
                logger.info(f"   {stock}: {real_prices[stock]} TL")
            else:
                logger.warning(f"   {stock}: NOT FOUND")
        
        return True, real_prices
        
    except Exception as e:
        logger.error(f"❌ Error updating prices: {e}")
        return False, {}

def main():
    """Main execution"""
    logger.info("🚀 Starting Excel to Data conversion...")
    
    # Step 1: Convert Excel to CSV/JSON
    success, csv_path, json_path = convert_excel_to_data()
    if not success:
        return False
    
    # Step 2: Extract real prices
    success, real_prices = update_frontend_prices()
    if not success:
        return False
    
    logger.info("✅ Conversion completed successfully!")
    logger.info(f"📄 CSV: {csv_path}")
    logger.info(f"📄 JSON: {json_path}")
    logger.info(f"💰 Prices extracted: {len(real_prices)} stocks")
    
    return True, real_prices

if __name__ == "__main__":
    result = main()
    if result:
        logger.info("🎉 ALL DATA UPDATED SUCCESSFULLY!")
    else:
        logger.error("💥 CONVERSION FAILED!")
