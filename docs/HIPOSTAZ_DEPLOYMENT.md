# 🌐 hipostaz.ai Deployment Guide

## 🎯 Deployment Overview

This guide covers the deployment of the MAMUT R600 Turkish Trading Dashboard to the **hipostaz.ai** domain, transitioning from Railway development to a production-ready environment.

## 📋 Pre-Deployment Checklist

### ✅ **Infrastructure Requirements**
- [ ] Domain: hipostaz.ai configured with DNS
- [ ] SSL Certificate: Let's Encrypt or commercial SSL
- [ ] Server: VPS/Cloud instance with Docker support
- [ ] Database: PostgreSQL instance (Railway or dedicated)
- [ ] Cache: Redis instance (Railway or dedicated)
- [ ] CDN: Optional for static assets

### ✅ **Environment Configuration**
- [ ] Environment variables secured
- [ ] API keys configured
- [ ] Database connections tested
- [ ] SSL certificates installed
- [ ] Backup strategy implemented

## 🏗️ Deployment Architecture

```mermaid
graph TB
    subgraph "hipostaz.ai Domain"
        A[Nginx Proxy]
        B[Next.js Frontend :3000]
        C[FastAPI Backend :8000]
    end
    
    subgraph "External Services"  
        D[PostgreSQL DB]
        E[Redis Cache]
        F[Profit.com API]
        G[TradingView Widgets]
    end
    
    Internet --> A
    A --> B
    A --> C
    C --> D
    C --> E
    C --> F
    B --> G
    
    style A fill:#ff9800
    style B fill:#2196f3
    style C fill:#4caf50
```

## 🐳 Docker Configuration

### 1. **Production Dockerfile**
```dockerfile
# Dockerfile.production
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Create non-root user
RUN useradd --create-home --shell /bin/bash mamut
USER mamut

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. **Frontend Dockerfile**
```dockerfile
# global-dashboard/Dockerfile.production
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000
CMD ["node", "server.js"]
```

### 3. **Docker Compose Production**
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    container_name: mamut_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/ssl:ro
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - frontend
      - backend
    restart: unless-stopped
    networks:
      - mamut_network

  frontend:
    build:
      context: ./global-dashboard
      dockerfile: Dockerfile.production
    container_name: mamut_frontend
    environment:
      - NEXT_PUBLIC_API_URL=https://api.hipostaz.ai
      - NEXT_PUBLIC_TRADING_VIEW_KEY=${TRADINGVIEW_KEY}
    restart: unless-stopped
    networks:
      - mamut_network

  backend:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: mamut_backend
    environment:
      - DATABASE_URL=${POSTGRESQL_URL}
      - REDIS_URL=${REDIS_URL}
      - PROFIT_API_KEY=${PROFIT_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
    restart: unless-stopped
    networks:
      - mamut_network
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    container_name: mamut_redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - mamut_network

networks:
  mamut_network:
    driver: bridge

volumes:
  redis_data:
```

## 🔧 Nginx Configuration

### `/etc/nginx/nginx.conf`
```nginx
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server frontend:3000;
    }
    
    upstream backend {
        server backend:8000;
    }
    
    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name hipostaz.ai www.hipostaz.ai;
        return 301 https://$server_name$request_uri;
    }
    
    # Main HTTPS server
    server {
        listen 443 ssl http2;
        server_name hipostaz.ai www.hipostaz.ai;
        
        # SSL Configuration
        ssl_certificate /etc/ssl/hipostaz.ai.crt;
        ssl_certificate_key /etc/ssl/hipostaz.ai.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        
        # Security Headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # Gzip Compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1000;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
        
        # Frontend (Dashboard)
        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }
        
        # Backend API
        location /api {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Rate limiting
            limit_req zone=api burst=20 nodelay;
        }
        
        # Static files caching
        location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
    
    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
}
```

## 🔐 Environment Variables

### Production Environment File
```bash
# .env.production
DATABASE_URL=postgresql://username:password@db.hipostaz.ai:5432/mamut_production
REDIS_URL=redis://redis.hipostaz.ai:6379/0
PROFIT_API_KEY=your_profit_api_key_here
SECRET_KEY=your_super_secret_key_here
ENVIRONMENT=production

# Next.js Frontend
NEXT_PUBLIC_API_URL=https://api.hipostaz.ai
NEXT_PUBLIC_TRADING_VIEW_KEY=your_tradingview_key

# Security
ALLOWED_HOSTS=hipostaz.ai,www.hipostaz.ai
CORS_ORIGINS=https://hipostaz.ai,https://www.hipostaz.ai

# Performance
WORKERS=4
MAX_CONNECTIONS=100
CACHE_TTL=300
```

## 🚀 Deployment Process

### 1. **Initial Server Setup**
```bash
# On your VPS/Cloud server
sudo apt update && sudo apt upgrade -y
sudo apt install docker.io docker-compose certbot nginx -y

# Configure firewall
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

### 2. **SSL Certificate Setup**
```bash
# Using Let's Encrypt
sudo certbot --nginx -d hipostaz.ai -d www.hipostaz.ai

# Or upload your commercial certificate to:
# /etc/ssl/hipostaz.ai.crt
# /etc/ssl/hipostaz.ai.key
```

### 3. **Deploy Application**
```bash
# Clone your repository
git clone <your-repo-url> /opt/mamut-r600
cd /opt/mamut-r600

# Set up environment
cp .env.example .env.production
nano .env.production  # Configure your variables

# Build and start services
docker-compose -f docker-compose.production.yml up --build -d

# Check status
docker-compose ps
docker-compose logs -f
```

### 4. **Database Migration**
```bash
# Run database migrations
docker-compose exec backend python -c "
from src.utils.database import create_tables
create_tables()
"

# Import Turkish stocks data
docker-compose exec backend python -c "
from src.services.data_import import import_turkish_stocks
import_turkish_stocks('/app/data/comprehensive_turkish_stocks.json')
"
```

## 📊 Monitoring & Maintenance

### 1. **Health Checks**
```bash
#!/bin/bash
# health_check.sh
curl -f https://hipostaz.ai/api/health || echo "API Down"
curl -f https://hipostaz.ai || echo "Frontend Down"
```

### 2. **Log Management**
```bash
# Set up log rotation
sudo nano /etc/logrotate.d/mamut
```

```
/opt/mamut-r600/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 root root
}
```

### 3. **Backup Strategy**
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups"

# Database backup
docker-compose exec postgres pg_dump -U username mamut_production > $BACKUP_DIR/db_$DATE.sql

# Application backup  
tar -czf $BACKUP_DIR/app_$DATE.tar.gz /opt/mamut-r600

# Upload to cloud storage (optional)
# aws s3 cp $BACKUP_DIR/ s3://mamut-backups/ --recursive
```

### 4. **Performance Monitoring**
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs -y

# Monitor containers
docker stats
docker-compose logs --tail=100 -f
```

## 🔄 CI/CD Pipeline (Optional)

### GitHub Actions Deployment
```yaml
# .github/workflows/deploy.yml
name: Deploy to hipostaz.ai

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to server
        uses: appleboy/ssh-action@v0.1.5
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd /opt/mamut-r600
            git pull origin main
            docker-compose -f docker-compose.production.yml down
            docker-compose -f docker-compose.production.yml up --build -d
```

## 🎯 Post-Deployment Verification

### 1. **Functionality Tests**
- [ ] Homepage loads: https://hipostaz.ai
- [ ] API health check: https://hipostaz.ai/api/health
- [ ] Stock search works
- [ ] TradingView widgets load
- [ ] Real-time price updates

### 2. **Performance Tests**
- [ ] Page load time < 3 seconds
- [ ] API response time < 200ms
- [ ] Search response time < 100ms
- [ ] Mobile responsiveness

### 3. **Security Tests**
- [ ] SSL certificate valid
- [ ] Security headers present
- [ ] Rate limiting works
- [ ] No exposed sensitive data

## 📞 Support & Maintenance

### Emergency Contacts
- **Server Issues**: [Your server provider support]
- **Domain Issues**: [Your domain registrar support]
- **SSL Issues**: [Your certificate provider support]

### Regular Maintenance Schedule
- **Daily**: Health checks, log review
- **Weekly**: Security updates, backup verification
- **Monthly**: Performance review, capacity planning
- **Quarterly**: Security audit, dependency updates

---

**Deployment Target**: hipostaz.ai  
**Status**: Ready for deployment  
**Estimated Downtime**: < 30 minutes for initial setup  
**Rollback Plan**: Previous Docker images retained for quick rollback
