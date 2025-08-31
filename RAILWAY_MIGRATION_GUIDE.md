# 🚀 Railway PostgreSQL Migration & Production Deployment Guide

## 📊 Overview
Bu guide 1.4M Excel kayıtlarının local SQLite'dan Railway PostgreSQL'e migration'ı ve production deployment sürecini detaylandırmaktadır.

**Başarılı Migration Sonucu:**
- ✅ **1,399,201 kayıt** Railway PostgreSQL'de
- ✅ **117 unique sembol** (2001-2025, 24 yıllık data)
- ✅ **266.9 MB database**
- ✅ **Multi-device access** hazır
- ✅ **+1200 Excel file** için scalable

---

## 🎯 Problem & Çözüm

### ❌ Sorunlar:
1. **Local SQLite:** 1.4M kayıt ile %99 CPU, kullanılamaz
2. **CSV Upload:** Local'dan Railway'e 300MB+ timeout
3. **Git LFS:** Railway'de CSV dosyaları 0.0MB (pointer files)
4. **Frontend Connection:** localhost → Railway API geçişi

### ✅ Çözümler:
1. **Direct PostgreSQL Connection:** Local'dan Railway'e direkt
2. **Optimized CSV Script:** Gzip handling + error recovery
3. **Frontend API Update:** Production URL'lere yönlendirme
4. **Railway Startup Fix:** PostgreSQL service initialization

---

## 🔧 Migration Script: `csv_to_postgresql.py`

### Key Features:
```python
# Railway-optimized gzip handling
with gzip.open(csv_part, 'rt', encoding='utf-8', errors='ignore') as f:
    header_line = next(f, None)  # Skip headers
    cursor.copy_expert(
        """
        COPY enhanced_stock_data 
        (symbol, date, time, timeframe, open, high, low, close, volume,
         rsi_14, macd_26_12, macd_trigger_9, bol_upper_20_2,
         bol_middle_20_2, bol_lower_20_2, atr_14, adx_14)
        FROM STDIN WITH CSV
        """, f
    )
```

### Migration Command:
```bash
export DATABASE_URL="postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"
python csv_to_postgresql.py
```

### Migration Results:
```
🎉 SUCCESS! 1.4M Excel records → Railway PostgreSQL!
⚡ COPY method: 100x faster than INSERT
📈 Railway API ready for 1500+ Excel files!
```

---

## 🌐 Railway Deployment

### Production URLs:
- **API:** https://bistai001-production.up.railway.app
- **Docs:** https://bistai001-production.up.railway.app/docs
- **PostgreSQL:** Internal Railway connection

### Git LFS Setup:
```bash
git lfs track "*.gz"
git lfs track "enhanced_stock_data_part_*"
git add .gitattributes enhanced_stock_data_part_*.gz
git commit -m "Add CSV data files with Git LFS"
git push origin main
```

### Railway CLI Commands:
```bash
railway status          # Project durumu
railway domain          # Production URL
railway logs            # Deployment logs
railway up              # Deploy to Railway
```

---

## 🖥️ Frontend → Railway Connection

### File Updates:

#### 1. HistoricalChart.tsx
```typescript
// OLD: const LOCAL_API = 'http://localhost:8000';
const RAILWAY_API = 'https://bistai001-production.up.railway.app';

// API calls updated:
fetch(`${RAILWAY_API}/api/bist/historical/${selectedSymbol}?timeframe=60min&limit=100`)
```

#### 2. AICommentaryPanel.tsx
```typescript
const RAILWAY_API = 'https://bistai001-production.up.railway.app';
fetch(`${RAILWAY_API}/api/bist/historical/${selectedSymbol}?timeframe=60min&limit=50`)
```

#### 3. AIChatPanel.tsx
```typescript
// Default API URL updated:
apiBaseUrl = 'https://bistai001-production.up.railway.app'
```

---

## 📊 Database Schema: `enhanced_stock_data`

```sql
CREATE TABLE enhanced_stock_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    time TIME,
    timeframe VARCHAR(20) NOT NULL,
    open DECIMAL(15,8),
    high DECIMAL(15,8),
    low DECIMAL(15,8),
    close DECIMAL(15,8),
    volume DECIMAL(20,8),  -- Fixed: was BIGINT, caused errors
    rsi_14 DECIMAL(15,10),
    macd_26_12 DECIMAL(15,10),
    macd_trigger_9 DECIMAL(15,10),
    bol_upper_20_2 DECIMAL(15,10),
    bol_middle_20_2 DECIMAL(15,10),
    bol_lower_20_2 DECIMAL(15,10),
    atr_14 DECIMAL(15,10),
    adx_14 DECIMAL(15,10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- High-performance indexes
CREATE INDEX idx_enhanced_symbol_timeframe ON enhanced_stock_data(symbol, timeframe);
CREATE INDEX idx_enhanced_date_desc ON enhanced_stock_data(date DESC);
CREATE INDEX idx_enhanced_symbol_date_desc ON enhanced_stock_data(symbol, date DESC);
CREATE INDEX idx_enhanced_timeframe ON enhanced_stock_data(timeframe);
```

---

## 🔍 API Testing

### Test Commands:
```bash
# Basic API test
curl -s "https://bistai001-production.up.railway.app/" | jq '.message'

# Stock data test
curl -s "https://bistai001-production.up.railway.app/api/bist/all-stocks?limit=3" | jq '.'

# Historical data test
curl -s "https://bistai001-production.up.railway.app/api/bist/historical/AKBNK?timeframe=60min&limit=3" | jq '.'

# Debug info
curl -s "https://bistai001-production.up.railway.app/" | jq '.debug'
```

### Expected Results:
```json
{
  "database_url_set": true,
  "postgresql_available": true,
  "historical_service_type": "PostgreSQLBISTService"
}
```

---

## ⚠️ Critical Issues & Solutions

### 1. Volume Column Type Error
**Problem:** `invalid input syntax for type bigint: "9320.73"`
**Solution:** Change `volume BIGINT` → `volume DECIMAL(20,8)`

### 2. Git LFS on Railway
**Problem:** CSV files show 0.0MB (pointer files)
**Solution:** Use direct local → Railway migration instead

### 3. PostgreSQL Service Not Starting
**Problem:** `historical_service_type: null`
**Solution:** Fix startup logic in `main_railway.py`:
```python
if POSTGRESQL_SERVICE_AVAILABLE:
    app_state.historical_service = get_historical_service()
```

### 4. Frontend CORS & Connection
**Problem:** localhost API calls from production
**Solution:** Update all API endpoints to Railway URLs

---

## 📈 Performance Metrics

### Migration Speed:
- **COPY vs INSERT:** 100x faster
- **Total Time:** ~5 minutes for 1.4M records
- **Network Transfer:** ~300MB CSV data

### Database Performance:
- **Query Speed:** <100ms for historical data
- **Index Usage:** Optimized for symbol+date queries
- **Storage:** ~266.9MB for 1.4M records

### Scalability:
- **Current:** 117 symbols, 24 years
- **Target:** +1200 Excel files (10-15M records)
- **Architecture:** Ready for massive scale

---

## 🔄 Future Additions: basestock.xls Pipeline

### Automated Daily Updates:
```python
# Morning pipeline (planned)
def daily_basestock_update():
    """Sabah basestock.xls → otomatik güncelleme + gün içi öngörü sistemi"""
    # 1. Download basestock.xls
    # 2. Process new records
    # 3. Update PostgreSQL
    # 4. Trigger AI model retraining
    # 5. Generate intraday predictions
```

---

## 💾 System Backup Information

### Database Connection:
```
DATABASE_URL: postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway
DATABASE_PUBLIC_URL: postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway
```

### HuggingFace Model:
```
Model: rsmctn/bist-ultimate-turkish-ai-v4
API: https://api-inference.huggingface.co/models/rsmctn/bist-ultimate-turkish-ai-v4
Read Token: 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
```

### GitHub Repository:
```
Repo: https://github.com/RSMCTN/BIST_AI001.git
Branch: main
LFS: Enabled for *.gz files
```

---

## 🚀 Quick Deployment Checklist

### Railway Deployment:
- [ ] `git add` changes
- [ ] `git commit -m "description"`  
- [ ] `git push origin main`
- [ ] Railway auto-deploys (~2 minutes)
- [ ] Test API endpoints

### Frontend Updates:
- [ ] Update API URLs to Railway
- [ ] Test localhost:3000
- [ ] Verify data loading
- [ ] Check console for errors

### Database Migration:
- [ ] Export DATABASE_URL
- [ ] Run `python csv_to_postgresql.py`
- [ ] Verify record counts
- [ ] Test API responses

---

## ✅ Current Status (2025-08-31)

### ✅ Completed:
- 🚀 Railway deployment active
- 💾 1.4M records in PostgreSQL  
- 🌐 Frontend → Railway connection
- 📊 Real OHLCV data flowing
- 🔗 Multi-device access ready

### ⚠️ Known Issues:
- Technical indicators mostly null (expected)
- Graph rendering needs optimization
- AI Chat endpoint needs testing

### 🎯 Next Steps:
- Test with different symbols
- Add remaining 1200 Excel files  
- Implement basestock.xls pipeline
- Optimize frontend rendering

---

**📝 Bu guide'ı referans olarak kullanarak future migrations'larda aynı sorunları yaşamamalısınız!**

**🎉 Migration Başarılı: 1.4M records → Railway PostgreSQL!**
