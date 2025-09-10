#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Railway PostgreSQL Latest Prices Checker
Tüm sembollerin son fiyatlarını çekip tablo halinde gösterir
"""

import psycopg2
import os
from datetime import datetime
import pandas as pd

def get_latest_prices():
    """Railway PostgreSQL'den tüm sembollerin son fiyatlarını çeker"""
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL bulunamadı!")
        return None
    
    print("🔗 Railway PostgreSQL'e bağlanıyor...")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Her sembol için en son fiyatı al
        query = """
        WITH latest_data AS (
            SELECT 
                symbol,
                date,
                time,
                close,
                volume,
                timeframe,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC, time DESC) as rn
            FROM enhanced_stock_data
            WHERE close IS NOT NULL AND close > 0
        )
        SELECT 
            symbol,
            date,
            time,
            close as latest_price,
            volume,
            timeframe
        FROM latest_data 
        WHERE rn = 1
        ORDER BY symbol;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            print("❌ Veri bulunamadı!")
            return None
            
        print(f"✅ {len(results)} sembolün son fiyatları alındı")
        
        # DataFrame oluştur
        df = pd.DataFrame(results, columns=[
            'Sembol', 'Tarih', 'Saat', 'Son Fiyat (₺)', 'Hacim', 'Timeframe'
        ])
        
        cursor.close()
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"❌ Database hatası: {e}")
        return None

def display_prices_table(df):
    """Fiyatları formatlanmış tablo olarak gösterir"""
    if df is None:
        return
        
    print("\n" + "="*80)
    print("📊 RAİLWAY PostgreSQL - TÜM SEMBOLLERİN SON FİYATLARI")
    print("="*80)
    print(f"Toplam Sembol: {len(df)}")
    print(f"Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Fiyatları düzenle
    df['Son Fiyat (₺)'] = df['Son Fiyat (₺)'].apply(lambda x: f"₺{x:,.2f}")
    df['Hacim'] = df['Hacim'].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "N/A")
    
    # Tabloyu göster (ilk 50 satır)
    print(f"{'Sembol':<8} | {'Son Fiyat':<12} | {'Tarih':<12} | {'Saat':<8} | {'Hacim':<15} | {'TF':<6}")
    print("-" * 80)
    
    for _, row in df.head(50).iterrows():
        print(f"{row['Sembol']:<8} | {row['Son Fiyat (₺)']:<12} | {row['Tarih']:<12} | "
              f"{row['Saat']:<8} | {row['Hacim']:<15} | {row['Timeframe']:<6}")
    
    if len(df) > 50:
        print(f"... ve {len(df)-50} sembol daha")
    
    print("="*80)
    
    # İstatistikler
    numeric_prices = pd.to_numeric(df['Son Fiyat (₺)'].str.replace('₺', '').str.replace(',', ''), errors='coerce')
    print(f"📈 En Yüksek: ₺{numeric_prices.max():,.2f}")
    print(f"📉 En Düşük: ₺{numeric_prices.min():,.2f}")
    print(f"📊 Ortalama: ₺{numeric_prices.mean():,.2f}")
    print("="*80)

def save_to_files(df):
    """Sonuçları CSV ve HTML olarak kaydet"""
    if df is None:
        return
        
    try:
        # CSV olarak kaydet
        csv_file = "railway_latest_prices.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"💾 CSV kaydedildi: {csv_file}")
        
        # HTML tablo olarak kaydet
        html_file = "railway_latest_prices.html"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Railway PostgreSQL - Son Fiyatlar</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .price {{ text-align: right; font-weight: bold; color: #007bff; }}
            </style>
        </head>
        <body>
            <h1>🚀 Railway PostgreSQL - Tüm Sembollerin Son Fiyatları</h1>
            <p>Toplam Sembol: {len(df)} | Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {df.to_html(index=False, classes='table', escape=False)}
        </body>
        </html>
        """
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"🌐 HTML kaydedildi: {html_file}")
        
    except Exception as e:
        print(f"❌ Dosya kaydetme hatası: {e}")

if __name__ == "__main__":
    print("🚀 Railway PostgreSQL Latest Prices Checker")
    print("-" * 50)
    
    # DATABASE_URL'yi set et
    os.environ['DATABASE_URL'] = "postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"
    
    # Son fiyatları al ve göster
    df = get_latest_prices()
    display_prices_table(df)
    save_to_files(df)
    
    print("\n✅ İşlem tamamlandı!")
