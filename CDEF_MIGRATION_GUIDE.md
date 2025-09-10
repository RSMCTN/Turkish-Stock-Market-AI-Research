# C-D-E-F Hisse Verileri Migration Rehberi

## 📋 Genel Bakış
Bu belge, BIST C-D-E-F harf grubu hisse verilerinin Railway PostgreSQL veritabanına aktarım sürecini documenta eder.

## 🎯 Proje Durumu (2 Eylül 2025)

### ✅ Başarılı Migration
- **Toplam Kayıt:** 2,624,832 
- **Toplam Hisse:** 218 sembol
- **A-B Harfleri:** 117 sembol (1,399,201 kayıt) - Önceden aktarıldı
- **C-D-E-F Harfleri:** 101 sembol (1,225,631 kayıt) - Yeni eklendi

### 📁 Kaynak Dosyalar
- **C-D-E Klasörü:** `data/New_excell_Graph_C_D/` (275 Excel dosyası)
- **F Klasörü:** `data/New_excel_Graph_F/` (27 Excel dosyası)
- **Toplam:** 302 Excel dosyası işlendi

## 🔧 Teknik Çözüm: Clone Table Yaklaşımı

### ❌ Sorun
Railway PostgreSQL COPY işlemleri sürekli timeout alıyordu (5+ dakika), direkt import başarısız oldu.

### ✅ Çözüm: İki Aşamalı Clone Yaklaşımı

#### 1. Aşama: Clone Table Oluşturma
```sql
CREATE TABLE enhanced_stock_data_1 (LIKE enhanced_stock_data INCLUDING ALL)
```

#### 2. Aşama: Clone Table'a C-D-E-F Import
```python
# comprehensive_cdef_processor.py ile 1.2M+ kayıt CSV oluşturuldu
# Clone table'a COPY işlemi: 50.2 saniye ✅
cursor.copy_expert('''
    COPY enhanced_stock_data_1 
    (symbol, date, time, timeframe, open, high, low, close, volume,
     rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
     bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
    FROM STDIN WITH CSV
''', f)
```

#### 3. Aşama: PostgreSQL İçinde Merge
```sql
-- 54.3 saniyede tamamlandı ✅
INSERT INTO enhanced_stock_data 
(symbol, date, time, timeframe, open, high, low, close, volume,
 rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
 bol_middle_20_2, bol_lower_20_2, atr_14, adx_14, created_at)
SELECT symbol, date, time, timeframe, open, high, low, close, volume,
       rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
       bol_middle_20_2, bol_lower_20_2, atr_14, adx_14, CURRENT_TIMESTAMP
FROM enhanced_stock_data_1
ON CONFLICT (symbol, date, time, timeframe) DO UPDATE SET ...
```

## 🛠️ Gelecek Migration'lar İçin Süreç

### 1. Excel Dosyalarını Hazırlama
```bash
# Yeni harf grubu klasörü kontrol et
ls data/New_excel_Graph_[HARF]/
```

### 2. Comprehensive Processor Çalıştırma
```python
# comprehensive_cdef_processor.py'yi yeni harf için adapte et
folders = [
    "data/New_excel_Graph_[YENİ_HARF]"
]
```

### 3. Clone Table Yaklaşımı (Railway Timeout'lar İçin)
```python
# Adım 1: Clone table oluştur
CREATE TABLE enhanced_stock_data_temp (LIKE enhanced_stock_data INCLUDING ALL)

# Adım 2: CSV'yi clone table'a COPY et
cursor.copy_expert('''
    COPY enhanced_stock_data_temp (...) FROM STDIN WITH CSV
''', f)

# Adım 3: PostgreSQL içinde merge
INSERT INTO enhanced_stock_data (...) 
SELECT ... FROM enhanced_stock_data_temp
ON CONFLICT (...) DO UPDATE SET ...

# Adım 4: Clean up
DROP TABLE enhanced_stock_data_temp
```

## 📊 Dosya İsimlendirme Konvansiyonları

### Excel Dosya Formatı
```
SYMBOL_TIMEFRAME.xlsx
Örnek: CANTE_60Dk.xlsx, FRIGO_Günlük.xlsx
```

### Timeframe Mapping
```python
timeframe_map = {
    '30Dk': '30min',
    '60Dk': '60min', 
    'Günlük': 'daily',
    '20Dk': '20min'
}
```

## 🔍 Kritik Çözümler

### 1. Pandas Series Ambiguity Error
```python
# ❌ Hatalı
any([record['open'], record['high'], record['low'], record['close']])

# ✅ Doğru
any(val is not None for val in ohlc_values)
```

### 2. Excel Column Whitespace Issue
```python
# ✅ Kritik: Column name'leri temizle
df.columns = [col.strip() for col in df.columns]
```

### 3. Railway PostgreSQL Performance
- **Direkt COPY:** Timeout (5+ dakika)
- **Clone Table + Merge:** Başarılı (50+54 = 104 saniye)

## 📈 Performance Metrikleri

### C-D-E-F Processing (2 Eylül 2025)
- **Excel İşleme:** 302 dosya, ~4 dakika
- **CSV Oluşturma:** 1.2M+ kayıt, ~2 saniye  
- **Clone Import:** 50.2 saniye ✅
- **PostgreSQL Merge:** 54.3 saniye ✅
- **Toplam Süre:** ~7 dakika

## 🚀 Sonraki Harf Grupları İçin Ready Script

```python
#!/usr/bin/env python3
"""
Gelecek harf grupları için hazır migration script
"""

import psycopg2
import os
from pathlib import Path

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://...')

def migrate_new_letter_group(letter_folder, csv_name):
    """Yeni harf grubu migration'ı"""
    
    # 1. Comprehensive processor çalıştır
    # 2. Clone table oluştur
    # 3. CSV'yi clone'a COPY et
    # 4. PostgreSQL merge yap
    # 5. Clean up
    
    pass

# Örnek kullanım
# migrate_new_letter_group("data/New_excel_Graph_G", "g_data.csv")
```

## 📚 İlgili Dosyalar

- `comprehensive_cdef_processor.py` - Ana işlem dosyası
- `direct_clone_import.py` - Clone table yaklaşımı
- `RAILWAY_MIGRATION_GUIDE.md` - A-B migration rehberi
- `cdef_combined.csv` - C-D-E-F verileri (84MB, 1.2M+ kayıt)

## 💡 Önemli Notlar

1. **Railway Timeout:** Büyük COPY işlemleri için clone table yaklaşımı kullan
2. **Column Cleaning:** Excel column name'lerinde whitespace olabilir
3. **Timeframe Mapping:** Türkçe timeframe'leri standart formata çevir
4. **Conflict Handling:** ON CONFLICT ile duplicate prevention
5. **Performance:** PostgreSQL içi merge çok daha hızlı (54 saniye vs timeout)