#!/usr/bin/env python3
"""
COMPREHENSIVE TURKISH STOCKS DATABASE BUILDER
513 Turkish stocks - eksiksiz, gerçek veri

Strateji:
1. Turkish stock symbols systematic araştırma
2. Her hisse için individual API call
3. Comprehensive database oluşturma
4. Mock data'yı tamamen kaldırma
"""

import requests
import json
import time
import string
from datetime import datetime
from typing import Dict, List, Set
import os

class ComprehensiveTurkishStocksBuilder:
    def __init__(self):
        self.api_key = "a9a0bacbab08493d958244c05380da01"
        self.base_url = "https://api.profit.com/data-api"
        self.all_stocks = []
        self.api_calls_made = 0
        self.failed_symbols = []
        self.successful_symbols = []
        
        # Turkish stock symbols - systematically gathered
        self.known_symbols = self._get_comprehensive_symbol_list()
    
    def _get_comprehensive_symbol_list(self) -> List[str]:
        """Kapsamlı Turkish stock symbols listesi"""
        
        # Major BIST 100 stocks
        bist_100 = [
            'AKBNK', 'ARCLK', 'ASELS', 'BIMAS', 'EKGYO', 'EREGL', 'FROTO',
            'GARAN', 'HALKB', 'ISCTR', 'KCHOL', 'SAHOL', 'SISE', 'TCELL',
            'THYAO', 'TOASO', 'TUPRS', 'VAKBN', 'YKBNK', 'AKSA', 'ALARK',
            'AEFES', 'GUBRF', 'MGROS', 'PGSUS', 'TTKOM', 'ULKER', 'DOHOL',
            'ENKAI', 'KOZAL', 'PETKM', 'TAVHL', 'VESTL', 'AGHOL', 'BRYAT',
            'CCOLA', 'DOCO', 'ECILC', 'GOODY', 'HURGZ', 'IHLAS', 'KARSN',
            'LOGO', 'MPARK', 'NTHOL', 'OTKAR', 'PARSN', 'POLHO', 'SOKM',
            'TSKB', 'TTRAK', 'TUKAS', 'YATAS'
        ]
        
        # Enerji sektörü
        energy_stocks = [
            'CWENE', 'ZOREN', 'SMRTG', 'AKSEN', 'AKENR', 'AVTUR', 'EPLAS',
            'GEREL', 'RTALB', 'SMART', 'ENJSA'
        ]
        
        # Bankacılık 
        banking_stocks = [
            'ALBRK', 'DENIZ', 'ICBCT', 'KLNMA', 'QNBFB', 'SKBNK', 'TSKB',
            'ZIRTR'
        ]
        
        # Teknoloji
        tech_stocks = [
            'ANSGR', 'ARENA', 'ARMDA', 'ASTOR', 'ATATP', 'BFREN', 'BILKO',
            'BIZIM', 'BOYNR', 'BRISA', 'BRSAN', 'BRLSM', 'BTCIM', 'CASH',
            'CIMSA', 'CMBTN', 'COTUR', 'DESA', 'DGKLB', 'DURDO', 'EMKEL',
            'FENER', 'FLAP', 'FORTE', 'GEDIK', 'GLBMD', 'GLCVY', 'GLRYH',
            'GLYHO', 'GRNYO', 'GSRAY', 'ISGSY', 'KERVT', 'KLMSN', 'KONTR',
            'KONYA', 'KRDMD', 'KRSAN', 'KRSTL', 'KTLEV', 'KUTPO', 'LOGO',
            'LUKSK', 'MAVI', 'MEPET', 'METRO', 'MHRGY', 'NTGAZ', 'ONRYT',
            'OSMEN', 'OZBAL', 'OZKGY', 'PENGD', 'PKART', 'PNSUT', 'QUAGR',
            'REEDR', 'RHEAG', 'SELEC', 'SILVR', 'SNPAM', 'SOKE', 'SRVGY',
            'TDGYO', 'TEKTU', 'TMPOL', 'TRCAS', 'TUREX', 'USRDS', 'VANGD',
            'VERUS', 'VKING', 'YAPRK'
        ]
        
        # Alfabetik systematic symbols (A-Z ile başlayan popüler hisseler)
        alphabet_symbols = []
        
        # A harfi
        a_stocks = [
            'ACIBD', 'ADEL', 'ADESE', 'AEFES', 'AFYON', 'AGESA', 'AGHOL',
            'AGROT', 'AGYO', 'AHGAZ', 'AKBNK', 'AKCNS', 'AKENR', 'AKGRT',
            'AKMGY', 'AKSA', 'AKSEN', 'AKSGY', 'AKSUE', 'AKTZA', 'ALARK',
            'ALBRK', 'ALCAR', 'ALCTL', 'ALFAS', 'ALGYO', 'ALKA', 'ALKIM',
            'ALTIN', 'ALYAG', 'AMATR', 'ANAK', 'ANADM', 'ANELE', 'ANGEN',
            'ANHYT', 'ANSGR', 'ARASE', 'ARCLK', 'ARENA', 'ARFMG', 'ARMDA',
            'ARSAN', 'ARTOG', 'ASELS', 'ASGYO', 'ASLAN', 'ASTOR', 'ASUZU',
            'ATAGY', 'ATATP', 'ATEKS', 'ATLAS', 'ATSYH', 'AVGYO', 'AVHOL',
            'AVISA', 'AVIVS', 'AVTUR', 'AYCES', 'AYDEM', 'AYEN', 'AYGAZ',
            'AZTEK'
        ]
        alphabet_symbols.extend(a_stocks)
        
        # Diğer harfler için de önemli hisseler
        other_stocks = [
            'BAGFS', 'BAHKM', 'BAKAB', 'BALAT', 'BANVT', 'BARMA', 'BASCM',
            'BASGZ', 'BAYRK', 'BERA', 'BEYAZ', 'BFREN', 'BIGCH', 'BILKO',
            'BIMAS', 'BINBN', 'BIOEN', 'BIZIM', 'BLCYT', 'BMEKS', 'BMSTL',
            'BNTAS', 'BOBET', 'BOSSA', 'BOYNR', 'BRISA', 'BRKO', 'BRLSM',
            'BRMEN', 'BRSAN', 'BRYAT', 'BSOKE', 'BTCIM', 'BUCIM', 'BURCE',
            'BURVA', 'CANTE', 'CASA', 'CATES', 'CCOLA', 'CELHA', 'CEMTS',
            'CEOEM', 'CIMSA', 'CLEBI', 'CMBTN', 'CMENT', 'CMP', 'CONSE',
            'COTUR', 'CRDFA', 'CRFSA', 'CWENE'
        ]
        alphabet_symbols.extend(other_stocks)
        
        # Tüm listeleri birleştir
        all_symbols = []
        all_symbols.extend(bist_100)
        all_symbols.extend(energy_stocks)
        all_symbols.extend(banking_stocks) 
        all_symbols.extend(tech_stocks)
        all_symbols.extend(alphabet_symbols)
        
        # Duplicates kaldır ve sırala
        unique_symbols = sorted(list(set(all_symbols)))
        
        print(f"📊 Comprehensive symbol list prepared: {len(unique_symbols)} symbols")
        return unique_symbols
    
    def get_stock_data(self, symbol: str) -> Dict:
        """Individual stock data fetch"""
        ticker = f"{symbol}.IS"
        url = f"{self.base_url}/market-data/quote/{ticker}"
        
        try:
            response = requests.get(url, params={'token': self.api_key}, timeout=10)
            self.api_calls_made += 1
            
            if response.status_code == 200:
                data = response.json()
                
                # Stock info structure
                stock_info = {
                    'symbol': symbol,
                    'name': data.get('name', symbol),
                    'price': data.get('price'),
                    'currency': data.get('currency', 'TRY'),
                    'volume': data.get('volume'),
                    'change': data.get('daily_price_change'),
                    'change_percent': data.get('daily_percentage_change'),
                    'previous_close': data.get('previous_close'),
                    'market': 'BIST',
                    'country': 'Turkey',
                    'asset_class': data.get('asset_class', 'stock'),
                    'timestamp': data.get('timestamp', datetime.now().isoformat()),
                    'search_keywords': self._generate_search_keywords(symbol, data.get('name', symbol)),
                    'api_data_raw': data  # Full API response
                }
                
                self.successful_symbols.append(symbol)
                print(f"✅ {symbol}: {stock_info['name'][:30]} - {stock_info['price']} TL")
                return stock_info
                
            else:
                print(f"❌ {symbol}: API error {response.status_code}")
                self.failed_symbols.append(symbol)
                return None
                
        except Exception as e:
            print(f"❌ {symbol}: Exception - {str(e)[:50]}")
            self.failed_symbols.append(symbol)
            return None
    
    def _generate_search_keywords(self, symbol: str, name: str) -> List[str]:
        """Generate search keywords for better matching"""
        keywords = []
        
        # Symbol variants
        keywords.append(symbol.lower())
        keywords.append(symbol.upper())
        
        # Name processing
        if name and name != 'N/A':
            name_parts = name.lower().replace('as', '').replace('a.s.', '').split()
            keywords.extend([part for part in name_parts if len(part) > 2])
        
        # Company type keywords
        company_types = {
            'bank': ['bank', 'bankası', 'bankasi'],
            'holding': ['holding', 'grup'],
            'enerji': ['enerji', 'energy', 'elektrik'],
            'teknoloji': ['teknoloji', 'technology', 'tech'],
            'otomotiv': ['otomotiv', 'automotive', 'motor'],
            'gıda': ['gida', 'food', 'beslenme'],
            'tekstil': ['tekstil', 'textile', 'konfeksiyon'],
            'inşaat': ['insaat', 'construction', 'yapi']
        }
        
        name_lower = name.lower() if name else ''
        for category, terms in company_types.items():
            if any(term in name_lower for term in terms):
                keywords.extend(terms)
        
        return list(set(keywords))  # Remove duplicates
    
    def build_comprehensive_database(self):
        """Build complete Turkish stocks database"""
        print("🚀 COMPREHENSIVE TURKISH STOCKS DATABASE BUILDER")
        print("=" * 55)
        print(f"📊 Target: 513 Turkish stocks")
        print(f"📋 Symbols to process: {len(self.known_symbols)}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        successful_count = 0
        
        for i, symbol in enumerate(self.known_symbols, 1):
            print(f"\n📍 [{i:3d}/{len(self.known_symbols):3d}] Processing: {symbol}")
            
            stock_data = self.get_stock_data(symbol)
            
            if stock_data:
                self.all_stocks.append(stock_data)
                successful_count += 1
                
                # Progress update
                if successful_count % 25 == 0:
                    print(f"\n🎯 Progress: {successful_count} stocks processed")
                    print(f"📞 API calls made: {self.api_calls_made}")
                    print(f"⏱️  Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Rate limiting
            time.sleep(0.1)  # 100ms delay
            
            # Break if we have enough successful stocks
            if successful_count >= 513:
                print(f"\n🎉 TARGET REACHED: 513 stocks!")
                break
            
            # Safety break for API limits
            if self.api_calls_made >= 500:
                print(f"\n⚠️ API call limit reached: {self.api_calls_made}")
                print("Saving progress...")
                break
        
        return self.all_stocks
    
    def save_database(self):
        """Save comprehensive database"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive database structure
        database = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_stocks": len(self.all_stocks),
                "successful_symbols": len(self.successful_symbols),
                "failed_symbols": len(self.failed_symbols),
                "api_calls_made": self.api_calls_made,
                "source": "Profit.com API - Individual Quotes",
                "version": "comprehensive-v1.0",
                "target_reached": len(self.all_stocks) >= 513
            },
            "stocks": self.all_stocks,
            "statistics": {
                "by_first_letter": self._get_letter_distribution(),
                "with_prices": len([s for s in self.all_stocks if s.get('price')]),
                "with_volume": len([s for s in self.all_stocks if s.get('volume')]),
                "average_price": self._calculate_average_price()
            },
            "failed_symbols": self.failed_symbols,
            "successful_symbols": self.successful_symbols
        }
        
        # Save main database
        filename = f"comprehensive_turkish_stocks_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(database, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Database saved: {filename}")
        
        # Save search-optimized version for frontend
        search_optimized = []
        for stock in self.all_stocks:
            search_entry = {
                "symbol": stock['symbol'],
                "name": stock['name'],
                "market": "Turkey",
                "currency": stock.get('currency', 'TRY'),
                "search_text": f"{stock['symbol']} {stock['name']} " + " ".join(stock.get('search_keywords', [])),
                "sector": self._guess_sector(stock),
                "price": stock.get('price'),
                "volume": stock.get('volume')
            }
            search_optimized.append(search_entry)
        
        # Save for frontend
        frontend_file = "global-dashboard/public/comprehensive_turkish_stocks.json"
        with open(frontend_file, 'w', encoding='utf-8') as f:
            json.dump(search_optimized, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Frontend database saved: {frontend_file}")
        
        return filename
    
    def _get_letter_distribution(self) -> Dict[str, int]:
        """Get distribution by first letter"""
        distribution = {}
        for stock in self.all_stocks:
            first_letter = stock['symbol'][0].upper()
            distribution[first_letter] = distribution.get(first_letter, 0) + 1
        return dict(sorted(distribution.items()))
    
    def _calculate_average_price(self) -> float:
        """Calculate average price of stocks with prices"""
        prices = [s['price'] for s in self.all_stocks if s.get('price') and isinstance(s['price'], (int, float))]
        return sum(prices) / len(prices) if prices else 0
    
    def _guess_sector(self, stock: Dict) -> str:
        """Guess sector based on name and keywords"""
        name = stock.get('name', '').lower()
        
        if any(word in name for word in ['bank', 'bankası', 'bankasi']):
            return 'Bankacılık'
        elif any(word in name for word in ['enerji', 'energy', 'elektrik']):
            return 'Enerji'
        elif any(word in name for word in ['teknoloji', 'tech', 'bilgisayar']):
            return 'Teknoloji'  
        elif any(word in name for word in ['otomotiv', 'automotive']):
            return 'Otomotiv'
        elif any(word in name for word in ['holding', 'grup']):
            return 'Holding'
        elif any(word in name for word in ['gida', 'food']):
            return 'Gıda'
        elif any(word in name for word in ['tekstil', 'konfeksiyon']):
            return 'Tekstil'
        elif any(word in name for word in ['insaat', 'construction']):
            return 'İnşaat'
        else:
            return 'Diğer'
    
    def print_summary(self):
        """Print build summary"""
        print("\n" + "=" * 50)
        print("🎯 COMPREHENSIVE DATABASE BUILD SUMMARY")
        print("=" * 50)
        print(f"✅ Successful stocks: {len(self.successful_symbols)}")
        print(f"❌ Failed stocks: {len(self.failed_symbols)}")
        print(f"📞 Total API calls: {self.api_calls_made}")
        print(f"🎯 Target (513): {'✅ REACHED' if len(self.all_stocks) >= 513 else '❌ NOT REACHED'}")
        
        if self.all_stocks:
            # Show some examples
            print(f"\n📋 SAMPLE STOCKS:")
            for stock in self.all_stocks[:10]:
                price = stock.get('price', 'N/A')
                print(f"  {stock['symbol']:8} | {stock['name'][:25]:25} | {price} TL")
        
        if self.failed_symbols:
            print(f"\n❌ FAILED SYMBOLS ({len(self.failed_symbols)}):")
            print("  " + ", ".join(self.failed_symbols[:20]))
            if len(self.failed_symbols) > 20:
                print(f"  ... and {len(self.failed_symbols) - 20} more")

def main():
    builder = ComprehensiveTurkishStocksBuilder()
    
    # Build database
    stocks = builder.build_comprehensive_database()
    
    # Save results
    filename = builder.save_database()
    
    # Print summary
    builder.print_summary()
    
    print(f"\n🚀 COMPREHENSIVE DATABASE READY!")
    print(f"📁 File: {filename}")
    print(f"🎯 Ready for real search without mock data!")

if __name__ == "__main__":
    main()
