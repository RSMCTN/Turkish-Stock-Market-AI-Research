#!/usr/bin/env python3

import pandas as pd

excel_path = "/Users/rasimcetin/rasim_claude/MAMUT_R600/data/excell_MIQ/basestock2808.xlsx"
df = pd.read_excel(excel_path)

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Check first few rows
print(f"\n📋 FIRST 5 ROWS:")
for i, (idx, row) in enumerate(df.head().iterrows()):
    print(f"\nRow {i}:")
    print(f"  SEMBOL: '{row.get('SEMBOL', 'NOT_FOUND')}'")
    print(f"  BORSASEMBOLU: '{row.get('BORSASEMBOLU', 'NOT_FOUND')}'") 
    print(f"  ACKL: '{row.get('ACKL', 'NOT_FOUND')}'")
    print(f"  SEKTOR: '{row.get('SEKTOR', 'NOT_FOUND')}'")
    
    # Check if symbol is valid
    sembol = row.get('SEMBOL', '') or ''
    borsasembolu = row.get('BORSASEMBOLU', '') or ''
    symbol = str(sembol or borsasembolu).strip()
    
    print(f"  Processed Symbol: '{symbol}'")
    print(f"  Valid Symbol: {bool(symbol and symbol != 'nan')}")
