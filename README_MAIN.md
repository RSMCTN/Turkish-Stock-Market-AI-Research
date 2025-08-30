# 🚀 MAMUT_R600: BIST DP-LSTM Trading System

## 🎯 Amaç
BIST hisseleri için haber + fiyat verilerini kullanarak **DP-LSTM tabanlı kısa-orta vade fiyat/işlem sinyali** üretmek.

## 📊 Ana Özellikler
- **SentimentARMA**: VADER tabanlı duygu puanları
- **Diferansiyel Gizlilik**: Gauss gürültüsü ile privacy-preserving ML  
- **Feature Factory**: 131+ teknik indikatör yönetimi
- **DP-LSTM**: 10 günlük rolling window, min-max normalizasyon
- **Multi-timeframe**: 1m-5m-15m-60m-günlük analiz
- **Real-time Trading**: Canlı sinyal üretimi ve backtest

## 🎯 Hedef Metrikler (MVP)
- **Trend İsabeti**: ≥ %58 (MVP) → %65 (Production)
- **Sharpe Ratio**: ≥ 0.8 (MVP) → 1.2 (Production)
- **Max Drawdown**: ≤ %15 (MVP) → %10 (Production)
- **Prediction Latency**: <200ms (MVP) → <100ms (Production)

## 📁 Proje Yapısı
```
MAMUT_R600/
├── 📂 data/           # Veri katmanı
├── 📂 src/            # Ana kaynak kodu  
├── 📂 config/         # Konfigürasyon
├── 📂 notebooks/      # Jupyter araştırma
├── 📂 tests/          # Unit/integration testler
├── 📂 docs/           # Dokümantasyon
└── 📂 scripts/        # Deployment/otomasyon
```

## 🔄 Geliştirme Fazları
1. **Phase 1** (Weeks 1-4): Foundation - Data pipeline, ARIMA baseline
2. **Phase 2** (Weeks 5-8): Core ML - DP-LSTM, Feature engineering  
3. **Phase 3** (Weeks 9-12): Production - Microservices, Real-time system
4. **Phase 4** (Weeks 13-16): Advanced - Transformer models, Advanced execution

## 🚀 Hızlı Başlangıç
```bash
# Todo manager kullanımı
python scripts/todo_manager.py status
python scripts/todo_manager.py phase 1
```

## 📞 İletişim
**Proje**: MAMUT_R600 BIST DP-LSTM Trading System  
**Versiyon**: 0.1.0 (Development)  
**Estimated Budget**: ~$19,600/month