#!/usr/bin/env python3
"""
Get complete Turkish stocks list from Profit.com API
"""

import requests
import json

def get_all_turkish_stocks():
    api_key = 'a9a0bacbab08493d958244c05380da01'
    url = 'https://api.profit.com/data-api/reference/stocks'

    # Get all Turkish stocks (increase limit to get all)
    params = {
        'token': api_key,
        'country': 'Turkey',
        'limit': 500  # High limit to get all Turkish stocks
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            stocks = data.get('data', [])
            total = data.get('total', 0)
            
            print('🇹🇷 PROFIT.COM - TÜM TÜRK HİSSE SENETLERİ')
            print('=' * 80)
            print(f'Toplam {total} Turkish stock found, showing {len(stocks)} results')
            print()
            print('🏦 BIST HİSSE LİSTESİ:')
            print('-' * 80)
            
            header = f"{'SYMBOL':<15} | {'TICKER':<15} | {'NAME':<40} | {'CURRENCY':<8}"
            print(header)
            print('-' * 80)
            
            # Group by sectors/categories for better organization
            banking = []
            industrial = []
            telecom = []
            other = []
            
            for stock in stocks:
                symbol = stock.get('symbol', 'N/A')
                ticker = stock.get('ticker', 'N/A') 
                name = stock.get('name', 'N/A')
                currency = stock.get('currency', 'N/A')
                
                # Simple categorization based on name
                name_lower = name.lower()
                if any(word in name_lower for word in ['bank', 'finansbank', 'garanti', 'akbank', 'vakif']):
                    banking.append(stock)
                elif any(word in name_lower for word in ['telekom', 'türk telekom', 'turkcell']):
                    telecom.append(stock)
                elif any(word in name_lower for word in ['aselsan', 'tupras', 'petkim', 'enerjisa']):
                    industrial.append(stock)
                else:
                    other.append(stock)
                
                row = f"{symbol:<15} | {ticker:<15} | {name[:40]:<40} | {currency:<8}"
                print(row)
            
            print()
            print('📊 KATEGORI ÖZETİ:')
            print('-' * 40) 
            print(f'🏦 Bankacılık/Finans: {len(banking)} stocks')
            print(f'📱 Telekomünikasyon: {len(telecom)} stocks')
            print(f'🏭 Sanayi/Enerji: {len(industrial)} stocks')
            print(f'🏢 Diğer Sektörler: {len(other)} stocks')
            print(f'📈 TOPLAM: {len(stocks)} stocks')
            
            # Save to file
            with open('turkish_stocks_profit.json', 'w', encoding='utf-8') as f:
                json.dump(stocks, f, indent=2, ensure_ascii=False)
            print(f'\n💾 Full list saved to: turkish_stocks_profit.json')
            
            # Show detailed categories
            show_detailed_categories(banking, telecom, industrial, other)
            
            return stocks
            
        else:
            print(f'❌ Error: {response.status_code}')
            print(f'Response: {response.text}')
            return []
            
    except Exception as e:
        print(f'💥 Error: {e}')
        return []

def show_detailed_categories(banking, telecom, industrial, other):
    """Show detailed breakdown by categories"""
    
    print('\n' + '='*60)
    print('📋 DETAYLI SEKTÖR BREAKDOWN')
    print('='*60)
    
    # Show major bank stocks specifically
    print('\n🏦 BANKACILIK SEKTÖRÜ:')
    print('-' * 60)
    for stock in banking:
        symbol = stock.get('symbol', '')
        ticker = stock.get('ticker', '')  
        name = stock.get('name', '')
        print(f'  🏛️ {symbol:<8} | {ticker:<12} | {name}')
    
    # Show telecom
    print('\n📱 TELEKOMÜNİKASYON:')
    print('-' * 40)
    for stock in telecom:
        symbol = stock.get('symbol', '')
        ticker = stock.get('ticker', '')
        name = stock.get('name', '')
        print(f'  📞 {symbol:<8} | {ticker:<12} | {name}')
    
    # Show energy/industrial
    print('\n⚡ ENERJİ & SANAYİ:')
    print('-' * 40)
    for stock in industrial:
        symbol = stock.get('symbol', '')
        ticker = stock.get('ticker', '')
        name = stock.get('name', '')
        print(f'  🏭 {symbol:<8} | {ticker:<12} | {name}')
    
    # Show top holdings/conglomerates from other category
    print('\n🏢 HOLDİNG & KONGLOMERALAR:')
    print('-' * 40)
    holdings = []
    for stock in other:
        name = stock.get('name', '').lower()
        if 'holding' in name or 'koc' in name or 'sabanci' in name:
            holdings.append(stock)
    
    for stock in holdings[:10]:  # Show first 10
        symbol = stock.get('symbol', '')
        ticker = stock.get('ticker', '')
        name = stock.get('name', '')
        print(f'  🏗️ {symbol:<8} | {ticker:<12} | {name}')

def test_popular_stocks():
    """Test popular BIST stocks for real-time prices"""
    api_key = 'a9a0bacbab08493d958244c05380da01'
    
    # Most popular BIST stocks to test
    popular_symbols = [
        'AKBNK.IS',   # Akbank
        'GARAN.IS',   # Garanti BBVA
        'ISCTR.IS',   # İş Bankası C
        'YKBNK.IS',   # Yapı Kredi
        'VAKBN.IS',   # Vakıfbank
        'TUPRS.IS',   # Tüpraş
        'ASELS.IS',   # Aselsan
        'TCELL.IS',   # Turkcell
        'SAHOL.IS',   # Sabancı Holding
        'KCHOL.IS',   # Koç Holding
        'THYAO.IS',   # THY
        'PETKM.IS',   # Petkim
    ]

    print('\n' + '='*70)
    print('🔥 EN POPÜLER BIST HİSSELERİ - CANLI FİYATLAR')
    print('='*70)
    
    header = f"{'SYMBOL':<12} | {'NAME':<35} | {'PRICE':<10} | {'CHANGE':<8}"
    print(header)
    print('-' * 70)

    successful = 0
    failed = 0

    for symbol in popular_symbols:
        try:
            url = f'https://api.profit.com/data-api/market-data/quote/{symbol}?token={api_key}'
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                name = data.get('name', symbol)[:35]
                price = data.get('price', 0)
                change = data.get('daily_percentage_change', 0)
                
                status = '🟢' if change >= 0 else '🔴'
                row = f'{status} {symbol:<10} | {name:<35} | ₺{price:8.2f} | {change:+6.2f}%'
                print(row)
                successful += 1
            else:
                error_row = f'❌ {symbol:<10} | Error: {response.status_code}'
                print(error_row)
                failed += 1
                
        except Exception as e:
            error_row = f'💥 {symbol:<10} | Connection error'
            print(error_row)
            failed += 1

    print('-' * 70)
    print(f'✅ Successful: {successful} | ❌ Failed: {failed}')
    print(f'📊 Success Rate: {successful/(successful+failed)*100:.1f}%')

if __name__ == "__main__":
    stocks = get_all_turkish_stocks()
    if stocks:
        test_popular_stocks()
        print(f'\n✅ Total Turkish stocks available: {len(stocks)}')
        print('💡 All symbols can be used with .IS extension (e.g., AKBNK.IS)')
