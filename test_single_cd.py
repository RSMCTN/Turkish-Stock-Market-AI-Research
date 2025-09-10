#!/usr/bin/env python3
"""
Test single C-D Excel file processing
"""

import os
import pandas as pd
import psycopg2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway')

def test_single_file():
    """Test with single CANTE file"""
    
    # Test file
    test_file = Path("data/New_excell_Graph_C_D/CANTE_60Dk.xlsx")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print(f"📊 Testing: {test_file}")
    
    # Read Excel
    try:
        df = pd.read_excel(test_file, engine='openpyxl')
        print(f"✅ Excel loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show first few rows
        print(f"\n🔍 First 3 rows:")
        print(df.head(3))
        
        # Check for required columns
        required = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
        else:
            print(f"✅ All required columns present")
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading Excel: {e}")
        return False

if __name__ == "__main__":
    test_single_file()