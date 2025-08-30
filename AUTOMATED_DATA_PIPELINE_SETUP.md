# 🚀 **AUTOMATED BIST DATA PIPELINE - SETUP GUIDE**

## 📋 **SYSTEM OVERVIEW**

Bu sistem **1200 Excel dosyası** ile tarihi verileri import edip, **günlük basestock.xls** güncellemeleri ile otomatik AI pipeline çalıştırır!

---

## 🗂 **DIRECTORY STRUCTURE**

```
MAMUT_R600/
├── automated_data_pipeline.py          # Ana pipeline sistemi
├── data_pipeline_config.json           # Konfigürasyon
├── historical_excel_data/               # 1200 Excel dosyalarının yeri
│   ├── 2020_01_BIST_data.xlsx
│   ├── 2020_02_BIST_data.xlsx
│   ├── ... (1200 files total)
├── daily_updates/                       # Günlük basestock.xls buraya
│   ├── basestock.xls                   # Her sabah/akşam güncellenen
│   └── backups/                        # Otomatik backup'lar
├── data/                               # Database ve cache
│   ├── bist_comprehensive.db           # Ana SQLite database
│   └── processing_cache/
├── logs/                               # System logs
│   ├── data_pipeline.log
│   └── error_logs/
└── ai_models/                          # Trained AI models
    ├── dp_lstm_model.pkl
    ├── turkish_qa_model/
    └── sentiment_model.pkl
```

---

## 🔧 **SETUP ADIMLARI**

### **1️⃣ Environment Hazırlığı:**

```bash
# Python dependencies install
pip install pandas numpy sqlite3 schedule watchdog openpyxl xlrd asyncio hashlib logging pathlib

# Directories oluştur
mkdir -p historical_excel_data daily_updates data logs ai_models
mkdir -p daily_updates/backups data/processing_cache logs/error_logs
```

### **2️⃣ Historical Data (1200 Excel) Hazırlığı:**

```bash
# 1200 Excel dosyanızı historical_excel_data/ klasörüne koyun
# Dosya formatları desteklenir:
# - .xlsx (Excel 2007+)
# - .xls (Excel 97-2003)

# Expected columns (herhangi bir isimde olabilir):
# - Date/Tarih/DATE
# - Symbol/Sembol/Code
# - Open/Açılış/Aç
# - High/Yüksek/Max
# - Low/Düşük/Min  
# - Close/Kapanış/Son
# - Volume/Hacim/Miktar
```

### **3️⃣ Günlük Update Hazırlığı:**

```bash
# basestock.xls formatını daily_updates/ klasöründe hazırlayın
# Sistem otomatik detect edecek ve process edecek

# File formats supported:
# - basestock.xls
# - basestock.xlsx
# - daily_*.xls*
```

---

## 🚀 **SISTEM ÇALIŞTIRMA**

### **Manual Start:**

```bash
cd MAMUT_R600
python automated_data_pipeline.py
```

### **Production Deployment:**

```bash
# Systemd service olarak çalıştır (Linux)
sudo cp bist_pipeline.service /etc/systemd/system/
sudo systemctl enable bist_pipeline
sudo systemctl start bist_pipeline

# Ya da Docker container
docker build -t bist-pipeline .
docker run -d --name bist-pipeline -v ./data:/app/data bist-pipeline
```

### **Background Process:**

```bash
# nohup ile background'da çalıştır
nohup python automated_data_pipeline.py > pipeline.log 2>&1 &

# Screen session ile
screen -S bist_pipeline
python automated_data_pipeline.py
# Ctrl+A, D ile detach
```

---

## 📊 **SYSTEM FEATURES**

### **🔄 Automated Processing:**
- **Historical Import**: 1200 Excel dosyasını paralel işleme
- **Daily Monitoring**: basestock.xls değişikliklerini real-time izleme  
- **Change Detection**: Hash-based file change detection
- **Database Updates**: Incremental updates, duplicate detection

### **📈 Technical Analysis:**
- **RSI (14)**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands
- **Ichimoku Cloud**: Comprehensive trend analysis
- **Custom Indicators**: Extensible indicator system

### **🤖 AI Integration:**
- **Auto Retraining**: New data trigger model retraining
- **Prediction Generation**: Multi-horizon price predictions  
- **Model Versioning**: Track different model versions
- **Performance Monitoring**: Model accuracy tracking

### **🔍 Monitoring & Logging:**
- **Comprehensive Logs**: All operations logged
- **Performance Metrics**: Processing time, record counts
- **Error Handling**: Graceful error recovery
- **System Health**: Database stats, uptime monitoring

---

## ⚡ **PERFORMANCE OPTIMIZATIONS**

### **Database Optimizations:**
```sql
-- Automatic indexing
CREATE INDEX idx_historical_symbol_date ON historical_data(symbol, date);
CREATE INDEX idx_technical_symbol_date ON technical_indicators(symbol, date);

-- Batch processing
INSERT INTO historical_data VALUES ... (1000 records at once)

-- Query optimizations
SELECT * FROM historical_data WHERE symbol = ? AND date >= ? ORDER BY date;
```

### **Memory Management:**
- **Chunk Processing**: Large files processed in chunks
- **Memory Limits**: 4GB default limit, configurable
- **Cache System**: Intelligent caching for frequent queries
- **Garbage Collection**: Automatic memory cleanup

### **Parallel Processing:**
- **Multi-threading**: Excel files processed in parallel
- **Async Operations**: Database operations asynchronous
- **Queue System**: Processing queue for large batches

---

## 🔧 **CONFIGURATION OPTIONS**

### **Historical Data Settings:**
```json
{
  "historical_data": {
    "parallel_processing": true,
    "max_workers": 4,
    "batch_size": 1000,
    "data_validation": {
      "min_price": 0.01,
      "max_price": 10000,
      "remove_weekends": true
    }
  }
}
```

### **AI Processing Settings:**
```json
{
  "ai_processing": {
    "retrain_threshold": {
      "new_records_count": 100,
      "days_since_last_training": 7
    },
    "prediction_horizon_days": [1, 3, 7, 14, 30],
    "auto_deployment": true
  }
}
```

### **Monitoring Settings:**
```json
{
  "monitoring": {
    "logging_level": "INFO",
    "email_alerts": true,
    "dashboard_port": 8080
  }
}
```

---

## 📡 **API ENDPOINTS**

Sistem çalıştıktan sonra HTTP API'ler kullanılabilir:

```bash
# System status
curl http://localhost:8080/api/status

# Database statistics  
curl http://localhost:8080/api/stats

# Recent predictions
curl http://localhost:8080/api/predictions?symbol=GARAN&days=7

# Trigger manual retraining
curl -X POST http://localhost:8080/api/retrain?symbol=AKBNK

# Get technical indicators
curl http://localhost:8080/api/indicators?symbol=THYAO&period=30
```

---

## 🎯 **WORKFLOW EXAMPLE**

### **Day 1: Initial Setup**
```
09:00 - 1200 Excel files historical_excel_data/ klasörüne konur
09:15 - python automated_data_pipeline.py çalıştırılır
09:16 - System 1200 dosyayı import etmeye başlar (parallel processing)
11:30 - Import tamamlanır: 2.5M historical records imported
11:35 - Technical indicators calculation başlar
12:45 - All indicators calculated, system ready
```

### **Day 2+: Daily Operations**
```
08:00 - Yeni basestock.xls daily_updates/ klasörüne konur
08:01 - System otomatik detect eder, processing başlar
08:02 - 150 new records imported, 50 records updated
08:05 - Technical indicators updated for affected symbols
08:10 - AI models retrained (if threshold met)
08:15 - New predictions generated and stored
08:20 - System ready for queries
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues:**

**1. Excel Import Errors:**
```bash
# Check file permissions
ls -la historical_excel_data/

# Test single file
python -c "import pandas as pd; print(pd.read_excel('test.xlsx'))"

# Check column names
python -c "import pandas as pd; print(pd.read_excel('test.xlsx').columns)"
```

**2. Database Lock Errors:**
```bash
# Check database permissions
sqlite3 data/bist_comprehensive.db ".databases"

# Restart with clean database
rm data/bist_comprehensive.db
python automated_data_pipeline.py
```

**3. Memory Issues:**
```bash
# Reduce batch size in config
"batch_size": 500  # Instead of 1000

# Limit parallel workers  
"max_workers": 2   # Instead of 4
```

### **Logs Location:**
```bash
# Main log
tail -f logs/data_pipeline.log

# Error logs
tail -f logs/error_logs/$(date +%Y-%m-%d).log

# System monitoring
htop  # CPU/Memory usage
```

---

## 🎉 **EXPECTED RESULTS**

### **After 1200 Excel Import:**
- ✅ **2M+ historical records** in database
- ✅ **500+ unique symbols** tracked  
- ✅ **Technical indicators** calculated for all symbols
- ✅ **Database optimized** with proper indexes
- ✅ **System ready** for daily updates

### **Daily Operations:**
- ✅ **5-10 seconds** basestock.xls processing
- ✅ **Real-time updates** to affected symbols
- ✅ **Auto AI retraining** when thresholds met
- ✅ **New predictions** generated automatically
- ✅ **Zero manual intervention** required

### **Performance Metrics:**
- 📊 **Import Speed**: ~2000 records/second
- 🔄 **Daily Updates**: <10 seconds end-to-end
- 🤖 **AI Training**: <5 minutes for incremental
- 💾 **Memory Usage**: <2GB typical, <4GB peak
- 🗄️ **Database Size**: ~500MB for 2M records

---

## 🎯 **SONUÇ**

Bu sistem ile:
- **Manual work eliminated** - Tamamen otomatik
- **Real-time processing** - Anında güncellemeler  
- **AI-powered insights** - Otomatik tahminler
- **Scalable architecture** - Büyümeye hazır
- **Production-ready** - 7/24 çalışmaya hazır

**1200 Excel + daily basestock.xls = Tamamen automated BIST AI system!** 🚀

**Setup tamamlandıktan sonra sadece basestock.xls'i güncellemek yeterli, geri kalanı otomatik!** ✅
