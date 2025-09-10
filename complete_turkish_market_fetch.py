#!/usr/bin/env python3
"""
🎯 COMPLETE TURKISH MARKET FETCH
Türk market için eksiksiz veri çekme: 513 Normal + 199 VİOP = 712 total

HEDEF:
• Normal hisseler: 513 unique
• VİOP hisseler: 199 unique (F_ prefix)
• TOPLAM: 712 unique stocks

API LIMITS:
• Kullanılmış: 717 / 150,000
• Kalan: 149,283 calls
• İhtiyaç: ~600 calls (güvenli!)
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Set
import os

# API Configuration
PROFIT_API_KEY = "a9a0bacbab08493d958244c05380da01"
BASE_URL = "https://api.profit.com/data-api"

def load_existing_stocks() -> tuple[List[Dict], Set[str]]:
    """Mevcut hisse listesini yükle ve unique symbols bul"""
    try:
        with open('complete_turkish_stocks_fixed.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        existing_stocks = data.get('all_turkish_stocks', [])
        existing_symbols = set()
        
        # Unique symbols bul
        for stock in existing_stocks:
            symbol = stock.get('symbol', '').upper().strip()
            if symbol:
                existing_symbols.add(symbol)
        
        print(f"📊 Mevcut data: {len(existing_stocks)} entries")
        print(f"📊 Unique symbols: {len(existing_symbols)}")
        
        return existing_stocks, existing_symbols
        
    except FileNotFoundError:
        print("❌ Mevcut dosya bulunamadı, sıfırdan başlıyoruz")
        return [], set()

def fetch_all_turkish_stocks_smart() -> List[Dict]:
    """
    Smart fetch: Sadece eksik hisseleri çek
    """
    print("🚀 SMART TURKISH STOCK FETCH BAŞLADI")
    print("=" * 50)
    
    # Mevcut verileri yükle
    existing_stocks, existing_symbols = load_existing_stocks()
    
    print(f"📍 Mevcut unique: {len(existing_symbols)}")
    print(f"🎯 Hedef: 712 (513 normal + 199 VİOP)")
    print(f"❓ Eksik: {712 - len(existing_symbols)}")
    
    all_stocks = []
    new_stocks_count = 0
    api_calls_made = 0
    
    # Pagination ile tüm hisseleri çek
    limit = 100
    offset = 0
    total_pages_estimated = 8  # 700+ hisse için tahmini
    
    while True:
        print(f"\n📄 Sayfa {offset//limit + 1}/{total_pages_estimated} işleniyor...")
        
        url = f"{BASE_URL}/market-data/turkey/stocks"
        params = {
            "token": PROFIT_API_KEY,
            "limit": limit,
            "offset": offset
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            api_calls_made += 1
            
            print(f"API Call #{api_calls_made}: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                if response.status_code == 429:
                    print("⏳ Rate limit, 60 saniye bekle...")
                    time.sleep(60)
                    continue
                break
                
            data = response.json()
            page_stocks = data.get('data', [])
            
            if not page_stocks:
                print("✅ Son sayfa ulaşıldı")
                break
            
            # Bu sayfadaki yeni hisseleri işle
            page_new_count = 0
            for stock in page_stocks:
                symbol = stock.get('symbol', '').upper().strip()
                
                if symbol and symbol not in existing_symbols:
                    # Yeni hisse bulundu!
                    all_stocks.append(stock)
                    existing_symbols.add(symbol)
                    new_stocks_count += 1
                    page_new_count += 1
                    
                    # VİOP mi normal mi?
                    stock_type = "VİOP" if symbol.startswith('F_') else "Normal"
                    print(f"  ✅ YENİ: {symbol} ({stock_type})")
            
            print(f"📊 Bu sayfada {page_new_count} yeni hisse")
            print(f"📊 Toplam yeni: {new_stocks_count}")
            print(f"📊 API calls: {api_calls_made}")
            
            # Sayfa arttır
            offset += limit
            
            # Rate limiting
            time.sleep(1)
            
            # Güvenlik: 10 sayfadan fazla olmasın
            if offset >= 1000:
                print("⚠️  Güvenlik limiti: 10 sayfa")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            time.sleep(5)
            continue
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
    
    # Sonuçlar
    print("\n" + "=" * 50)
    print("📊 FETCH SONUÇLARI:")
    print(f"✅ Yeni bulunan hisse: {new_stocks_count}")
    print(f"📞 API calls kullanıldı: {api_calls_made}")
    print(f"📈 Total unique stocks: {len(existing_symbols)}")
    
    # Kategorilere ayır
    normal_count = sum(1 for s in existing_symbols if not s.startswith('F_'))
    viop_count = sum(1 for s in existing_symbols if s.startswith('F_'))
    
    print(f"📊 Normal hisse: {normal_count} / 513")
    print(f"📊 VİOP hisse: {viop_count} / 199")
    
    return existing_stocks + all_stocks

def save_complete_data(all_stocks: List[Dict]):
    """Tamamlanmış veriyi kaydet"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ana veri
    complete_data = {
        "fetch_info": {
            "timestamp": timestamp,
            "total_stocks": len(all_stocks),
            "source": "Profit.com API - Complete Turkish Market",
            "target": "513 Normal + 199 VİOP = 712 total"
        },
        "all_turkish_stocks": all_stocks
    }
    
    # JSON olarak kaydet
    filename = f"complete_turkish_market_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(complete_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Kaydedildi: {filename}")
    
    # Global dashboard için de güncelle
    try:
        # Global stocks data güncelle
        global_file = "global-dashboard/public/global_stocks_data.json"
        if os.path.exists(global_file):
            with open(global_file, 'r', encoding='utf-8') as f:
                global_data = json.load(f)
            
            # Turkish stocks'u değiştir
            non_turkish_stocks = [s for s in global_data if not s.get('market', '').lower() == 'turkey']
            
            # Yeni Turkish stocks ekle
            for stock in all_stocks:
                stock_for_global = {
                    "symbol": stock.get('symbol', ''),
                    "name": stock.get('name', ''),
                    "market": "Turkey",
                    "search_text": f"{stock.get('symbol', '')} {stock.get('name', '')} Turkey BIST".lower()
                }
                non_turkish_stocks.append(stock_for_global)
            
            # Kaydet
            with open(global_file, 'w', encoding='utf-8') as f:
                json.dump(non_turkish_stocks, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Global dashboard güncellendi: {len(non_turkish_stocks)} stocks")
            
    except Exception as e:
        print(f"⚠️  Global dashboard güncelleme hatası: {e}")

def main():
    """Ana fonksiyon"""
    print("🎯 COMPLETE TURKISH MARKET FETCH")
    print("=" * 40)
    print("Hedef: 513 Normal + 199 VİOP = 712 total")
    print("=" * 40)
    
    # Fetch yap
    all_stocks = fetch_all_turkish_stocks_smart()
    
    if all_stocks:
        # Kaydet
        save_complete_data(all_stocks)
        
        print("\n🎉 BAŞARI!")
        print(f"📊 Total stocks: {len(all_stocks)}")
        print("🚀 Dashboard ready!")
        
    else:
        print("❌ Hiç veri çekilemedi")

if __name__ == "__main__":
    main()
