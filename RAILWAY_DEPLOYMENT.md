# 🚂 Railway Deployment Guide

Bu rehber BIST DP-LSTM Trading System'i Railway'e deploy etmek için gerekli adımları içerir.

## 🚀 Hızlı Deploy (Automated)

```bash
# Railway deployment script'ini çalıştır
chmod +x scripts/deploy_railway.py
python scripts/deploy_railway.py
```

## 📋 Manuel Deploy Adımları

### 1. Railway CLI Kurulumu

```bash
# macOS
curl -fsSL https://railway.app/install.sh | sh

# Alternatif: Homebrew
brew install railway

# Linux
curl -fsSL https://railway.app/install.sh | sh
```

### 2. Railway Login

```bash
railway login
```

### 3. Proje Oluştur

```bash
# Yeni proje oluştur
railway init bist-dp-lstm-trading

# Veya var olan projeye bağlan
railway link
```

### 4. Servisleri Ekle

```bash
# PostgreSQL ekle
railway add postgresql

# Redis ekle  
railway add redis
```

### 5. Environment Variables Ayarla

```bash
# Production ortamı
railway variables set ENVIRONMENT=production
railway variables set LOG_LEVEL=INFO
railway variables set PORT=8000

# Trading ayarları
railway variables set INITIAL_CAPITAL=100000.0
railway variables set MAX_POSITIONS=10
railway variables set BUY_THRESHOLD=0.65
railway variables set SELL_THRESHOLD=0.65

# Daha fazla env var için config/railway.env dosyasına bakın
```

### 6. Deploy Et

```bash
# Dockerfile ile deploy
railway up --detach

# Build loglarını izle
railway logs --follow
```

## 🔧 Railway Özel Dosyalar

### `railway.json`
```json
{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile.railway"
  },
  "deploy": {
    "numReplicas": 1,
    "sleepApplication": false
  }
}
```

### `Dockerfile.railway`
Railway için optimize edilmiş Docker container:
- Minimal base image
- Railway-specific environment variables
- Health check endpoints
- Production optimizations

## 🌐 Deployment Sonrası

### Erişim URLs:
- **API Docs**: `https://yourapp.railway.app/docs`
- **Health Check**: `https://yourapp.railway.app/health`
- **System Metrics**: `https://yourapp.railway.app/metrics/system`
- **Portfolio**: `https://yourapp.railway.app/portfolio/summary`

### Railway Dashboard:
- **Logs**: `railway logs`
- **Status**: `railway status`
- **Variables**: `railway variables`
- **Connect**: `railway connect`

## 📊 Monitoring

### Logs İzleme:
```bash
# Canlı logları izle
railway logs --follow

# Son 100 satır
railway logs --tail 100

# Belirli service
railway logs --service bist-trading-api
```

### Metrics:
- CPU ve Memory kullanımı Railway dashboard'da
- Application metrics: `/metrics/system` endpoint
- Trading metrics: API endpoints

## 🔍 Troubleshooting

### Common Issues:

1. **Build Failure**:
   ```bash
   railway logs --deployment <deployment-id>
   ```

2. **Environment Variables**:
   ```bash
   railway variables list
   railway variables set KEY=VALUE
   ```

3. **Database Connection**:
   - Railway otomatik olarak `DATABASE_URL` sağlar
   - Redis için `REDIS_URL` otomatik oluşur

4. **Port Issues**:
   - Railway otomatik olarak `PORT` environment variable set eder
   - Application `$PORT` değerini kullanmalı

### Health Check Fails:
```bash
curl https://yourapp.railway.app/health
```

### API Test:
```bash
# Signal generation test
curl -X POST https://yourapp.railway.app/signals/generate \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AKBNK", "include_features": true}'
```

## 🚀 Production Checklist

- [ ] Environment variables configured
- [ ] Database services added (PostgreSQL, Redis)
- [ ] Health check endpoint working
- [ ] API endpoints functional
- [ ] Trading system initialized
- [ ] Monitoring configured
- [ ] Logs accessible
- [ ] Performance acceptable

## 📈 Scaling

Railway otomatik scaling için:
- Memory ve CPU limits ayarla
- Horizontal scaling için replicas arttır
- Database connection pooling ekle

## 💡 Tips

1. **Free Tier**: Railway $5/ay sonra ücretli
2. **Databases**: PostgreSQL ve Redis için ayrı servisler
3. **Logs**: Railway 7 gün log saklar
4. **Domains**: Custom domain eklenebilir
5. **GitHub**: Otomatik deployment için GitHub bağla

## 📞 Support

- **Railway Docs**: https://docs.railway.app
- **Discord**: https://discord.gg/railway
- **Status**: https://status.railway.app

---

🎯 **Railway deployment ile BIST DP-LSTM Trading System production'da!**
