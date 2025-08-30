#!/usr/bin/env python3
"""
Excel dosyalarının yapısını analiz etme
MAMUT_R600 için detaylı indikatör ve grafik veri analizi
"""
import pandas as pd
import os
from pathlib import Path
import sys

def analyze_excel_file(file_path):
    """Excel dosyasının yapısını analiz et"""
    try:
        print(f"\n🔍 Analiz ediliyor: {os.path.basename(file_path)}")
        print("=" * 60)
        
        # Excel dosyasını oku
        df = pd.read_excel(file_path)
        
        print(f"📏 Boyut: {df.shape[0]} satır, {df.shape[1]} sütun")
        print(f"📅 Tarih aralığı: {df.iloc[0, 0] if len(df) > 0 else 'N/A'} - {df.iloc[-1, 0] if len(df) > 0 else 'N/A'}")
        
        print("\n📊 Sütun Başlıkları:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print("\n📈 İlk 3 Satır Örneği:")
        print(df.head(3).to_string(index=False, max_cols=8))
        
        print("\n📉 Son 2 Satır Örneği:")
        print(df.tail(2).to_string(index=False, max_cols=8))
        
        # Numerik sütunları identifiy et
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        print(f"\n🔢 Numerik Sütunlar ({len(numeric_cols)}):")
        for col in numeric_cols[:10]:  # İlk 10'u göster
            print(f"  - {col}")
        if len(numeric_cols) > 10:
            print(f"  ... ve {len(numeric_cols) - 10} sütun daha")
            
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'numeric_columns': numeric_cols,
            'date_range': (str(df.iloc[0, 0]) if len(df) > 0 else None,
                          str(df.iloc[-1, 0]) if len(df) > 0 else None)
        }
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return None

def main():
    """Ana analiz fonksiyonu"""
    data_dir = Path("data/New_excell_Graph_Sample")
    
    if not data_dir.exists():
        print(f"❌ Klasör bulunamadı: {data_dir}")
        return
    
    excel_files = list(data_dir.glob("*.xlsx"))
    print(f"🔍 {len(excel_files)} Excel dosyası bulundu")
    
    # Farklı zaman dilimlerinden örnekler al
    sample_files = []
    
    # 30dk örneği
    for f in excel_files:
        if "_30Dk.xlsx" in f.name:
            sample_files.append(f)
            break
    
    # 60dk örneği
    for f in excel_files:
        if "_60Dk.xlsx" in f.name:
            sample_files.append(f)
            break
    
    # Günlük örneği
    for f in excel_files:
        if "_Günlük.xlsx" in f.name:
            sample_files.append(f)
            break
    
    print(f"\n📋 {len(sample_files)} örnek dosya analiz edilecek:")
    
    all_analyses = []
    for file_path in sample_files[:3]:  # İlk 3 örneği analiz et
        analysis = analyze_excel_file(file_path)
        if analysis:
            all_analyses.append((file_path.name, analysis))
    
    # Ortak sütunları bul
    if all_analyses:
        print("\n" + "="*80)
        print("🔗 ORTAK SÜTUN ANALİZİ")
        print("="*80)
        
        all_columns = [set(analysis['columns']) for _, analysis in all_analyses]
        common_columns = set.intersection(*all_columns) if all_columns else set()
        
        print(f"📊 Tüm dosyalarda ortak sütunlar ({len(common_columns)}):")
        for col in sorted(common_columns):
            print(f"  ✅ {col}")
        
        # Farklı sütunları da göster
        unique_columns = set()
        for _, analysis in all_analyses:
            unique_columns.update(set(analysis['columns']) - common_columns)
        
        if unique_columns:
            print(f"\n📈 Bazı dosyalarda farklı sütunlar ({len(unique_columns)}):")
            for col in sorted(list(unique_columns)[:10]):
                print(f"  ⚡ {col}")
    
    # Database önerileri
    print("\n" + "="*80)
    print("💾 DATABASE ŞEMA ÖNERİLERİ")
    print("="*80)
    
    if all_analyses:
        sample_columns = all_analyses[0][1]['columns']
        
        print("🔧 Gerekli yeni tablolar/sütunlar:")
        
        # Teknik indikatörler için
        technical_indicators = [col for col in sample_columns if any(term in col.upper() for term in 
                               ['RSI', 'MACD', 'BB', 'MA', 'EMA', 'SMA', 'STOCH', 'ATR', 'CCI', 'WILLIAMS'])]
        
        if technical_indicators:
            print("📊 Technical Indicators tablosu:")
            for indicator in technical_indicators:
                print(f"  - {indicator}")
        
        # OHLCV veriler için
        ohlcv_columns = [col for col in sample_columns if any(term in col.upper() for term in 
                        ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'ACİLİS', 'YUKSEK', 'DUSUK', 'KAPANIŞ', 'HACİM'])]
        
        if ohlcv_columns:
            print("📈 OHLCV veriler:")
            for col in ohlcv_columns:
                print(f"  - {col}")

if __name__ == "__main__":
    main()
