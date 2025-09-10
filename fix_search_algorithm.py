#!/usr/bin/env python3
"""
Fix Search Algorithm Issues
Improve search quality and add missing stocks
"""

import json
import requests
import time
from datetime import datetime

class SearchAlgorithmFixer:
    def __init__(self):
        self.api_key = "a9a0bacbab08493d958244c05380da01"
        self.base_url = "https://api.profit.com"
    
    def verify_missing_stocks(self):
        """Verify if specific stocks exist in API"""
        print("🔍 EKSİK HİSSE DOĞRULAMA")
        print("=" * 30)
        
        missing_stocks = [
            'ZOREN', 'ONRYT', 'GLCVY', 'THYAO', 'CCOLA',
            'BIST', 'GLYHO', 'IHLAS', 'KARSN', 'LOGO'
        ]
        
        verified_stocks = []
        
        for stock in missing_stocks:
            try:
                ticker = f"{stock}.IS"
                url = f"{self.base_url}/data-api/market-data/quote/{ticker}"
                params = {'token': self.api_key}
                
                response = requests.get(url, params=params, timeout=8)
                
                if response.status_code == 200:
                    data = response.json()
                    verified_stock = {
                        "symbol": stock,
                        "ticker": ticker,
                        "name": data.get('name', stock),
                        "sector": "Unknown",
                        "market": "BIST", 
                        "currency": "TRY",
                        "region": "turkey",
                        "verified_price": data.get('price'),
                        "verified_from_api": True
                    }
                    verified_stocks.append(verified_stock)
                    print(f"✅ {stock:6} | {data.get('name', 'N/A')[:30]:30} | ₺{data.get('price', 'N/A')}")
                else:
                    print(f"❌ {stock:6} | API'da bulunamadı (Status: {response.status_code})")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"💥 {stock:6} | Hata: {str(e)[:40]}...")
        
        print(f"\n✅ {len(verified_stocks)} ek hisse doğrulandı")
        return verified_stocks
    
    def improve_search_text(self, stocks):
        """Improve search_text for better matching"""
        print(f"\n🔧 ARAMA METNİ İYİLEŞTİRME")
        print("=" * 30)
        
        improved_stocks = []
        
        for stock in stocks:
            symbol = stock.get('symbol', '')
            name = stock.get('name', '')
            sector = stock.get('sector', '')
            
            # Create comprehensive search text
            search_components = [
                symbol.lower(),
                name.lower(),
                sector.lower(),
                'turkey',
                'bist'
            ]
            
            # Add name words separately for better partial matching
            name_words = name.lower().split()
            search_components.extend(name_words)
            
            # Add symbol substrings for partial matching
            if len(symbol) >= 3:
                for i in range(1, len(symbol)):
                    search_components.append(symbol[:i].lower())
                    search_components.append(symbol[i:].lower())
            
            # Remove duplicates and create search text
            search_text = ' '.join(set(search_components))
            
            improved_stock = {
                **stock,
                "search_text": search_text,
                "improved": True
            }
            
            improved_stocks.append(improved_stock)
        
        print(f"✅ {len(improved_stocks)} hissenin arama metni iyileştirildi")
        return improved_stocks
    
    def create_fixed_database(self):
        """Create fixed database with improved search"""
        print(f"\n🔄 SABİT VERİTABANI OLUŞTURMA")
        print("=" * 30)
        
        # Load existing stocks
        try:
            with open('complete_turkish_stocks.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            existing_stocks = data['all_turkish_stocks']
            print(f"📊 Mevcut hisseler: {len(existing_stocks)}")
        except Exception as e:
            print(f"❌ Mevcut veriler yüklenemedi: {e}")
            return None
        
        # Verify missing stocks
        additional_stocks = self.verify_missing_stocks()
        
        # Combine all stocks
        all_stocks = existing_stocks + additional_stocks
        print(f"📊 Toplam hisseler: {len(all_stocks)}")
        
        # Improve search text for all stocks
        improved_stocks = self.improve_search_text(all_stocks)
        
        # Create final database
        fixed_database = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_stocks": len(improved_stocks),
                "source": "Profit.com API - Search Fixed",
                "version": "2.1-search-fixed",
                "improvements": [
                    "Enhanced search algorithm",
                    "Added missing verified stocks",
                    "Improved partial matching",
                    "Better search_text generation"
                ]
            },
            "all_turkish_stocks": improved_stocks
        }
        
        # Save fixed database
        filename = 'complete_turkish_stocks_fixed.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(fixed_database, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Sabit veritabanı kaydedildi: {filename}")
        
        # Test search on fixed database
        self.test_search_on_fixed_database(improved_stocks)
        
        return filename
    
    def test_search_on_fixed_database(self, stocks):
        """Test search algorithm on fixed database"""
        print(f"\n🧪 SABİT VERİTABANI ARAMA TESTİ")  
        print("=" * 35)
        
        test_queries = ['Z', 'ZOR', 'ZOREN', 'ONRYT', 'GLCVY', 'THY', 'COCA']
        
        for query in test_queries:
            results = self.search_stocks(query, stocks)
            print(f"\n🔍 \"{query}\" araması:")
            print(f"   Sonuç: {len(results)} hisse")
            
            for i, stock in enumerate(results[:5], 1):
                symbol = stock.get('symbol', 'N/A')
                name = stock.get('name', 'N/A')[:25]
                verified = "✓" if stock.get('verified_from_api') else ""
                print(f"   {i}. {symbol:8} | {name:25} {verified}")
    
    def search_stocks(self, query, stocks):
        """Improved search algorithm"""
        query = query.lower().strip()
        
        if len(query) < 1:
            return []
        
        results = []
        for stock in stocks:
            symbol = stock.get('symbol', '').lower()
            name = stock.get('name', '').lower()
            search_text = stock.get('search_text', '').lower()
            
            # Scoring system
            score = 0
            
            # Perfect symbol match (highest score)
            if symbol == query:
                score += 100
            # Symbol starts with query
            elif symbol.startswith(query):
                score += 50
            # Query in symbol
            elif query in symbol:
                score += 30
            
            # Name matches
            if query in name:
                score += 20
                if name.startswith(query):
                    score += 10
            
            # Search text match
            if query in search_text:
                score += 10
            
            if score > 0:
                results.append((score, stock))
        
        # Sort by score (descending) and return stocks
        results.sort(key=lambda x: x[0], reverse=True)
        return [stock for score, stock in results[:8]]
    
    def run_fix(self):
        """Run complete fix process"""
        print("🚀 SEARCH ALGORITHM FIX")
        print("=" * 30)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        filename = self.create_fixed_database()
        
        if filename:
            print(f"\n🎊 FIX TAMAMLANDI!")
            print(f"📁 Dosya: {filename}")
            print("💡 Bu dosyayı dashboard'a kopyalayın")
        else:
            print("❌ Fix başarısız!")

if __name__ == "__main__":
    fixer = SearchAlgorithmFixer()
    fixer.run_fix()
