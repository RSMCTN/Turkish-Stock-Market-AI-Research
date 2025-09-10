#!/usr/bin/env python3
"""
Debug BIST Query Issue
Find why API returns duplicate AEFES stocks
"""

import psycopg2
import psycopg2.extras

DATABASE_URL = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"

def debug_query():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    cursor = conn.cursor()
    
    print("🔍 DEBUGGING BIST API QUERY ISSUE")
    print("=" * 50)
    
    # Test simple query first
    print("\n1. Simple BIST_30 stocks:")
    cursor.execute("""
        SELECT sm.symbol, sc.category, sc.priority
        FROM stocks_meta sm
        JOIN stock_categories sc ON sm.symbol = sc.symbol
        WHERE sc.category = 'BIST_30'
        AND sm.is_active = TRUE
        ORDER BY sc.priority, sm.symbol
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    print(f"   Found {len(results)} results:")
    unique_symbols = set()
    for row in results:
        if row['symbol'] in unique_symbols:
            print(f"   ⚠️ DUPLICATE: {row['symbol']} (Priority: {row['priority']})")
        else:
            print(f"   ✅ {row['symbol']} (Priority: {row['priority']})")
        unique_symbols.add(row['symbol'])
    
    print(f"   Total unique symbols: {len(unique_symbols)}")
    
    # Check if BRSAN is in results
    brsan_found = any(row['symbol'] == 'BRSAN' for row in results)
    print(f"   BRSAN found: {'✅' if brsan_found else '❌'}")
    
    # Check categories for BRSAN
    print("\n2. BRSAN category check:")
    cursor.execute("""
        SELECT sm.symbol, sc.category, sc.priority
        FROM stocks_meta sm
        JOIN stock_categories sc ON sm.symbol = sc.symbol
        WHERE sm.symbol = 'BRSAN'
        ORDER BY sc.category
    """)
    brsan_categories = cursor.fetchall()
    for row in brsan_categories:
        print(f"   BRSAN in {row['category']} with priority {row['priority']}")
    
    # Check stock_categories table structure
    print("\n3. Stock categories distribution:")
    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM stock_categories
        GROUP BY category
        ORDER BY category
    """)
    category_counts = cursor.fetchall()
    for row in category_counts:
        print(f"   {row['category']}: {row['count']} stocks")
    
    cursor.close()
    conn.close()
    
    return results, brsan_categories

if __name__ == "__main__":
    try:
        results, brsan_categories = debug_query()
        print("\n🎯 SUMMARY:")
        print(f"- Query returned results, but API may have additional JOIN issues")
        print(f"- BRSAN exists in database with categories: {[r['category'] for r in brsan_categories]}")
        print(f"- Check railway_bist_categories.py for complex JOIN logic")
    except Exception as e:
        print(f"❌ Debug failed: {e}")
