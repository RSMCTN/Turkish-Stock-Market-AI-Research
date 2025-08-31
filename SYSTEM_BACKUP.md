# 🔄 MAMUT R600 System Backup & Configuration

**Backup Date:** 2025-08-31  
**Status:** Production Ready - 1.4M records migrated successfully

---

## 🔐 Critical Credentials & URLs

### Railway Production:
```
Project: bist-dp-lstm-trading
Environment: production
Service: BIST_AI001
Production URL: https://bistai001-production.up.railway.app
Deployment: Auto from GitHub main branch
```

### PostgreSQL Database:
```
DATABASE_URL: postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway
DATABASE_PUBLIC_URL: postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway

Records: 1,399,201
Symbols: 117 unique
Data Range: 2001-2025 (24 years)
Size: ~266.9 MB
```

### HuggingFace AI Model:
```
Model: rsmctn/bist-ultimate-turkish-ai-v4
API Endpoint: https://api-inference.huggingface.co/models/rsmctn/bist-ultimate-turkish-ai-v4
Read Token: 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
Write Token: [Stored separately for security]
Training Data: 87 Q&A + 400 sentiment + 10K historical records
```

### GitHub Repository:
```
Repo: https://github.com/RSMCTN/BIST_AI001.git
Branch: main
Git LFS: Enabled for CSV files (*.gz, enhanced_stock_data_part_*)
Last Commit: Frontend → Railway API connection
```

---

## 📁 Key Files & Scripts

### Migration Script:
- **File:** `csv_to_postgresql.py`
- **Purpose:** Local CSV → Railway PostgreSQL migration
- **Features:** Gzip handling, error recovery, COPY optimization
- **Usage:** `python csv_to_postgresql.py`

### Frontend Components:
- `trading-dashboard/src/components/trading/HistoricalChart.tsx`
- `trading-dashboard/src/components/trading/AICommentaryPanel.tsx`  
- `trading-dashboard/src/components/trading/AIChatPanel.tsx`
- **Status:** All connected to Railway API

### API Configuration:
- **File:** `src/api/main_railway.py`
- **PostgreSQL Service:** Active and initialized
- **Port:** 8080 (Railway), 8000 (Local)
- **Features:** Real-time OHLCV data, technical indicators, AI chat

---

## 🔧 Development Environment

### Local Setup:
```bash
# Backend (FastAPI)
cd MAMUT_R600
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_railway.txt
python src/api/main_railway.py  # Port 8000

# Frontend (React/Next.js)
cd trading-dashboard
npm install
npm run dev  # Port 3000
```

### Environment Variables:
```bash
export DATABASE_URL="postgresql://postgres:EXCVEzFQvQOgwovvdONbrJofBYeJTKJT@shuttle.proxy.rlwy.net:49135/railway"
```

---

## 📊 Database Schema Backup

### Main Table: `enhanced_stock_data`
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
    volume DECIMAL(20,8),  -- Critical: Must be DECIMAL not BIGINT
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

-- Indexes
CREATE INDEX idx_enhanced_symbol_timeframe ON enhanced_stock_data(symbol, timeframe);
CREATE INDEX idx_enhanced_date_desc ON enhanced_stock_data(date DESC);
CREATE INDEX idx_enhanced_symbol_date_desc ON enhanced_stock_data(symbol, date DESC);
CREATE INDEX idx_enhanced_timeframe ON enhanced_stock_data(timeframe);
```

### Top Symbols by Record Count:
```
ATSYH: 21,655 records (2001-08-24 → 2025-08-29)
BRMEN: 20,115 records (2001-11-07 → 2025-08-29)
BRKO: 19,844 records (2009-04-30 → 2025-08-29)
BASCM: 19,096 records (2012-08-09 → 2025-08-29)
AYES: 18,796 records (2013-02-05 → 2025-08-29)
```

---

## 🚀 API Endpoints Status

### Working Endpoints:
```bash
GET  /                           # System info
GET  /health                     # Health check  
GET  /api/bist/all-stocks        # Stock list
GET  /api/bist/stock/{symbol}    # Individual stock
GET  /api/bist/historical/{symbol} # Historical OHLCV
GET  /api/forecast/{symbol}      # Price forecast
POST /migrate                    # PostgreSQL migration
GET  /debug/files               # Debug info
```

### Test Samples:
```bash
# System status
curl "https://bistai001-production.up.railway.app/"

# AKBNK historical data (confirmed working)
curl "https://bistai001-production.up.railway.app/api/bist/historical/AKBNK?timeframe=60min&limit=3"
```

---

## 📈 Current System Metrics

### Performance:
- **API Response:** <100ms for historical queries
- **Database Query:** Optimized with indexes
- **Migration Speed:** 1.4M records in ~5 minutes
- **Storage:** 266.9MB for 24 years of data

### Scalability Readiness:
- **Current:** 117 symbols
- **Target:** +1200 Excel files (10-15M records)
- **Architecture:** Railway PostgreSQL can handle massive scale
- **Migration Script:** Optimized for large datasets

---

## 🔄 Recovery Procedures

### If Railway Goes Down:
1. **Local Fallback:** Switch frontend API URLs back to localhost
2. **Database Access:** Connect directly via DATABASE_URL
3. **Backup Data:** Export via `pg_dump` if needed

### If Database Corrupts:
1. **Re-run Migration:** `python csv_to_postgresql.py`
2. **CSV Files:** Available in Git LFS
3. **Local SQLite:** `enhanced_bist_data.db` as backup

### If Git Repository Issues:
1. **Manual Deployment:** Railway CLI `railway up`
2. **Code Backup:** All files in local MAMUT_R600 folder
3. **Configuration:** This backup document

---

## 🎯 Future Expansion Plans

### Immediate Tasks:
- [ ] Technical indicators data population  
- [ ] Graph rendering optimization
- [ ] AI Chat endpoint testing

### Medium Term:
- [ ] Add remaining 1200 Excel files
- [ ] Implement basestock.xls daily pipeline
- [ ] Advanced technical indicator calculations

### Long Term:
- [ ] Real-time data feeds
- [ ] Multi-user support
- [ ] Advanced AI trading signals

---

## ⚠️ Known Issues & Workarounds

### 1. Technical Indicators Null
- **Status:** Expected - need calculation service
- **Workaround:** Basic OHLCV data works perfectly
- **Fix:** Implement technical indicators calculator

### 2. Graph Rendering Optimization Needed
- **Status:** Data loads, rendering needs optimization
- **Workaround:** Works but could be smoother
- **Fix:** Frontend performance improvements

### 3. AI Chat Endpoint Status
- **Status:** Backend ready, needs frontend testing
- **Workaround:** Use forecast endpoints for now
- **Fix:** Complete AI integration testing

---

## 🔧 Emergency Contacts & Resources

### Railway Support:
- **Console:** https://railway.app/dashboard
- **Docs:** https://docs.railway.app
- **CLI:** `railway --help`

### HuggingFace Support:
- **Console:** https://huggingface.co/rsmctn/bist-ultimate-turkish-ai-v4
- **API Docs:** https://huggingface.co/docs/api-inference/

### GitHub Repository:
- **URL:** https://github.com/RSMCTN/BIST_AI001
- **Issues:** https://github.com/RSMCTN/BIST_AI001/issues
- **Actions:** Auto-deployment configured

---

**🎉 BACKUP COMPLETE - SYSTEM FULLY DOCUMENTED**

**Last Updated:** 2025-08-31 17:45 UTC  
**Migration Status:** ✅ SUCCESSFUL (1,399,201 records)  
**Production Status:** ✅ ACTIVE (Railway + PostgreSQL)  
**Next Review:** After adding 1200+ Excel files**
