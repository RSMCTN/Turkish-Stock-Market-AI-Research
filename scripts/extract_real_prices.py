#!/usr/bin/env python3
"""
Extract real stock prices from basestock2808.xlsx data
Update frontend components with correct prices
"""

import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_real_prices():
    """Extract all real stock prices from CSV"""
    
    csv_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/bist_real_data_2808.csv"
    
    try:
        # Read CSV with proper encoding
        df = pd.read_csv(csv_path, encoding='utf-8')
        logger.info(f"📊 Read {len(df)} stocks from CSV")
        
        # Extract prices using SEMBOL and SON columns
        real_prices = {}
        invalid_count = 0
        
        for idx, row in df.iterrows():
            try:
                symbol = str(row['SEMBOL']).strip()
                price = float(row['SON'])
                real_prices[symbol] = price
            except (ValueError, TypeError, KeyError) as e:
                invalid_count += 1
                continue
        
        logger.info(f"✅ Extracted {len(real_prices)} valid prices ({invalid_count} invalid)")
        
        # Show key stock prices
        key_stocks = ['AKSEN', 'ASTOR', 'GARAN', 'THYAO', 'TUPRS', 'BRSAN', 'AKBNK', 'ISCTR', 'SISE', 'ARCLK']
        logger.info("🎯 Key stock prices:")
        found_stocks = {}
        for stock in key_stocks:
            if stock in real_prices:
                found_stocks[stock] = real_prices[stock]
                logger.info(f"   {stock}: {real_prices[stock]} TL")
            else:
                logger.warning(f"   {stock}: NOT FOUND")
        
        # Generate JavaScript object string for frontend
        js_object = "const realPrices: { [key: string]: number } = {\n"
        for symbol, price in found_stocks.items():
            js_object += f"  '{symbol}': {price},  // Real closing price\n"
        
        # Add some additional stocks
        additional_stocks = ['KCHOL', 'BIMAS', 'PETKM', 'TTKOM']
        for stock in additional_stocks:
            if stock in real_prices:
                found_stocks[stock] = real_prices[stock]
                js_object += f"  '{stock}': {real_prices[stock]},\n"
        
        js_object += "};"
        
        logger.info("📝 Generated JavaScript object for frontend:")
        logger.info(js_object)
        
        return True, found_stocks, js_object
        
    except Exception as e:
        logger.error(f"❌ Error extracting prices: {e}")
        return False, {}, ""

def main():
    """Main execution"""
    logger.info("🚀 Starting real price extraction...")
    
    success, prices, js_code = extract_real_prices()
    
    if success:
        logger.info("✅ Price extraction completed successfully!")
        logger.info(f"💰 Total prices found: {len(prices)}")
        
        # Save JS code to file for easy copy-paste
        js_file = "/Users/rasimcetin/rasim_claude/MAMUT_R600/scripts/real_prices.js"
        with open(js_file, 'w', encoding='utf-8') as f:
            f.write(js_code)
        logger.info(f"📄 JavaScript code saved to: {js_file}")
        
        return prices
    else:
        logger.error("💥 Price extraction failed!")
        return {}

if __name__ == "__main__":
    prices = main()
