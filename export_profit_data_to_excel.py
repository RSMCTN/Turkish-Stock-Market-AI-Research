#!/usr/bin/env python3
"""
Profit.com API'sinden tüm mevcut verileri Excel'e çıkarma
Export all available data from Profit.com API to Excel
"""

import requests
import pandas as pd
import json
from datetime import datetime
import time
import os

class ProfitDataExporter:
    def __init__(self):
        self.api_key = "a9a0bacbab08493d958244c05380da01"
        self.base_url = "https://api.profit.com"
        self.all_data = []
        
    def get_sample_stock_data(self, symbol):
        """Get detailed data for a single stock to see all available fields"""
        url = f'{self.base_url}/data-api/market-data/quote/{symbol}?token={self.api_key}'
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching {symbol}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception for {symbol}: {e}")
            return None

    def get_stocks_list(self):
        """Get Turkish stocks list"""
        url = f"{self.base_url}/data-api/reference/stocks"
        params = {
            'token': self.api_key,
            'country': 'Turkey',
            'limit': 100  # Get first 100 for detailed analysis
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()['data']
            return []
        except Exception as e:
            print(f"Error getting stocks list: {e}")
            return []

    def analyze_api_structure(self):
        """Analyze API response structure"""
        print("🔍 Profit.com API Veri Yapısını İnceliyoruz...")
        print("=" * 60)
        
        # Get a few sample stocks to analyze structure
        sample_symbols = ['AKBNK.IS', 'GARAN.IS', 'TUPRS.IS', 'TCELL.IS', 'KCHOL.IS']
        
        all_fields = set()
        sample_data = {}
        
        for symbol in sample_symbols:
            print(f"📊 {symbol} verisi çekiliyor...")
            data = self.get_sample_stock_data(symbol)
            
            if data:
                sample_data[symbol] = data
                # Collect all possible fields
                all_fields.update(data.keys())
                time.sleep(0.5)  # Rate limiting
        
        print(f"\n✅ {len(sample_data)} hisse senedi analiz edildi")
        print(f"🎯 Toplam {len(all_fields)} farklı veri alanı bulundu\n")
        
        # Show available fields
        print("📋 API'DEN GELEN TÜM VERİ ALANLARI:")
        print("-" * 50)
        for i, field in enumerate(sorted(all_fields), 1):
            print(f"{i:2}. {field}")
        
        # Show sample data structure
        if sample_data:
            first_symbol = list(sample_data.keys())[0]
            first_data = sample_data[first_symbol]
            
            print(f"\n🔍 ÖRNEK VERİ YAPISI ({first_symbol}):")
            print("-" * 50)
            for key, value in first_data.items():
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                print(f"{key:25} : {value_str}")
        
        return all_fields, sample_data

    def fetch_comprehensive_data(self, limit=50):
        """Fetch comprehensive data for multiple stocks"""
        print(f"\n📈 {limit} Hisse Senedi İçin Detaylı Veri Çekiliyor...")
        print("=" * 60)
        
        # Get stocks list first
        stocks = self.get_stocks_list()[:limit]
        
        comprehensive_data = []
        failed_count = 0
        
        for i, stock in enumerate(stocks, 1):
            symbol = stock.get('ticker', stock.get('symbol', ''))
            if not symbol.endswith('.IS'):
                symbol += '.IS'
                
            print(f"📊 {i}/{len(stocks)}: {symbol}")
            
            # Get detailed quote data
            quote_data = self.get_sample_stock_data(symbol)
            
            if quote_data:
                # Combine reference data with quote data
                combined_data = {
                    'fetch_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'reference_symbol': stock.get('symbol', ''),
                    'reference_ticker': stock.get('ticker', ''),
                    'reference_name': stock.get('name', ''),
                    'reference_currency': stock.get('currency', ''),
                    **quote_data  # Add all quote data fields
                }
                comprehensive_data.append(combined_data)
            else:
                failed_count += 1
                
            # Rate limiting
            time.sleep(0.3)
        
        print(f"\n✅ Başarılı: {len(comprehensive_data)}")
        print(f"❌ Başarısız: {failed_count}")
        print(f"📊 Toplam başarı oranı: {len(comprehensive_data)/(len(comprehensive_data)+failed_count)*100:.1f}%")
        
        return comprehensive_data

    def export_to_excel(self, data, filename=None):
        """Export data to Excel with proper formatting"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'profit_api_data_{timestamp}.xlsx'
        
        print(f"\n💾 Excel dosyasına çıkarılıyor: {filename}")
        print("=" * 60)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Show column info
        print(f"📊 Toplam satır: {len(df)}")
        print(f"📊 Toplam sütun: {len(df.columns)}")
        print(f"\n📋 SÜTUN BAŞLIKLARI ({len(df.columns)} adet):")
        print("-" * 50)
        
        for i, col in enumerate(df.columns, 1):
            sample_value = df[col].iloc[0] if len(df) > 0 else "N/A"
            if isinstance(sample_value, str) and len(sample_value) > 30:
                sample_value = sample_value[:27] + "..."
            print(f"{i:2}. {col:30} : {sample_value}")
        
        # Export to Excel with formatting
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Profit_API_Data', index=False)
            
            # Summary sheet
            summary_data = {
                'Bilgi': [
                    'Veri Kaynağı',
                    'API Endpoint',
                    'Toplam Hisse Sayısı',
                    'Toplam Sütun Sayısı',
                    'Veri Çekme Tarihi',
                    'API Anahtarı (Son 4 Karakter)',
                    'Dosya Oluşturma Tarihi'
                ],
                'Değer': [
                    'Profit.com API',
                    'https://api.profit.com/data-api',
                    len(df),
                    len(df.columns),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    self.api_key[-4:],
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Veri_Özeti', index=False)
            
            # Column descriptions sheet if we have data
            if len(df) > 0:
                col_info = []
                for col in df.columns:
                    col_type = str(df[col].dtype)
                    non_null_count = df[col].count()
                    null_count = df[col].isnull().sum()
                    
                    col_info.append({
                        'Sütun Adı': col,
                        'Veri Tipi': col_type,
                        'Dolu Kayıt': non_null_count,
                        'Boş Kayıt': null_count,
                        'Örnek Değer': str(df[col].iloc[0]) if non_null_count > 0 else 'N/A'
                    })
                
                col_df = pd.DataFrame(col_info)
                col_df.to_excel(writer, sheet_name='Sütun_Detayları', index=False)
        
        print(f"✅ Excel dosyası oluşturuldu: {filename}")
        print(f"📁 Dosya boyutu: {os.path.getsize(filename)/1024:.1f} KB")
        
        return filename

    def run_full_export(self, stock_limit=50):
        """Run complete data export process"""
        print("🚀 PROFİT.COM API EXCEL EXPORT BAŞLATILIYOR")
        print("=" * 60)
        print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Hedef Hisse Sayısı: {stock_limit}")
        print()
        
        # Step 1: Analyze API structure
        fields, sample_data = self.analyze_api_structure()
        
        # Step 2: Fetch comprehensive data
        comprehensive_data = self.fetch_comprehensive_data(limit=stock_limit)
        
        if comprehensive_data:
            # Step 3: Export to Excel
            filename = self.export_to_excel(comprehensive_data)
            
            print(f"\n🎉 İŞLEM TAMAMLANDI!")
            print(f"📊 {len(comprehensive_data)} hisse senedi verisi Excel'e aktarıldı")
            print(f"📁 Dosya: {filename}")
            return filename
        else:
            print("❌ Veri çekilemedi!")
            return None

if __name__ == "__main__":
    exporter = ProfitDataExporter()
    exporter.run_full_export(stock_limit=100)  # İlk 100 hisse senedi
