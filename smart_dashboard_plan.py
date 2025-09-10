#!/usr/bin/env python3
"""
MAMUT R600 - Smart Dashboard Plan with API Limit Optimization
Profit.com API ile 150,000 günlük limit'e optimize edilmiş dashboard planı
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class StockTier:
    name: str
    stocks: List[str]
    update_interval_minutes: int
    priority: int
    description: str

class SmartDashboardPlanner:
    def __init__(self):
        self.daily_api_limit = 150000
        self.minutes_per_day = 1440
        
        # Define stock tiers
        self.tiers = {
            'vip': StockTier(
                name="VIP Tier - BIST 30",
                stocks=[
                    'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'YKBNK.IS', 'VAKBN.IS',
                    'TUPRS.IS', 'ASELS.IS', 'PETKM.IS', 'EREGL.IS', 'TCELL.IS',
                    'THYAO.IS', 'KCHOL.IS', 'SAHOL.IS', 'BIMAS.IS', 'MGROS.IS',
                    'ARCLK.IS', 'CCOLA.IS', 'AEFES.IS', 'ULKER.IS', 'SISE.IS',
                    'FROTO.IS', 'TOASO.IS', 'OTKAR.IS', 'PGSUS.IS', 'TTKOM.IS',
                    'HALKB.IS', 'CIMSA.IS', 'AKCNS.IS', 'AYGAZ.IS', 'TKFEN.IS'
                ],
                update_interval_minutes=5,  # Her 5 dakika
                priority=1,
                description="En büyük 30 Türk hissesi - Ultra hızlı güncelleme"
            ),
            
            'premium': StockTier(
                name="Premium Tier - BIST 70",
                stocks=[],  # Will be populated with remaining BIST 100
                update_interval_minutes=15,  # Her 15 dakika
                priority=2,
                description="BIST 100'ün geri kalanı - Hızlı güncelleme"
            ),
            
            'standard': StockTier(
                name="Standard Tier - Diğer Türk",
                stocks=[],  # Remaining Turkish stocks
                update_interval_minutes=60,  # Her 1 saat
                priority=3,
                description="Diğer Türk hisseleri - Normal güncelleme"
            ),
            
            'global': StockTier(
                name="Global Tier - Dünya",
                stocks=[
                    # Top US stocks
                    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA',
                    'JPM', 'JNJ', 'V', 'WMT', 'PG', 'UNH', 'HD', 'MA',
                    # European
                    'ASML', 'AAPH.L', 'RMS.PA', 'MC.PA', 'OR.PA',
                    # Asian
                    '7203.T', '6758.T', '8306.T'  # Toyota, Sony, MUFG
                ],
                update_interval_minutes=240,  # Her 4 saat
                priority=4,
                description="Seçili dünya hisseleri - Yavaş güncelleme"
            )
        }

    def calculate_daily_api_usage(self) -> Dict:
        """Calculate daily API usage for each tier"""
        results = {}
        total_calls = 0
        
        for tier_name, tier in self.tiers.items():
            calls_per_day = (self.minutes_per_day // tier.update_interval_minutes) * len(tier.stocks)
            total_calls += calls_per_day
            
            results[tier_name] = {
                'tier': tier.name,
                'stock_count': len(tier.stocks),
                'update_interval': f"{tier.update_interval_minutes} min",
                'daily_calls': calls_per_day,
                'percentage': calls_per_day / self.daily_api_limit * 100
            }
        
        results['total'] = {
            'daily_calls': total_calls,
            'percentage': total_calls / self.daily_api_limit * 100,
            'remaining': self.daily_api_limit - total_calls
        }
        
        return results

    def generate_dashboard_config(self):
        """Generate dashboard configuration"""
        config = {
            "dashboard_name": "MAMUT R600 Smart Dashboard",
            "api_limit": self.daily_api_limit,
            "generated_at": datetime.now().isoformat(),
            
            "update_strategy": {
                "vip_stocks": {
                    "symbols": self.tiers['vip'].stocks,
                    "update_seconds": self.tiers['vip'].update_interval_minutes * 60,
                    "cache_ttl": 300,  # 5 minutes
                    "websocket_enabled": True
                },
                "premium_stocks": {
                    "count": 70,
                    "update_seconds": self.tiers['premium'].update_interval_minutes * 60,
                    "cache_ttl": 900,  # 15 minutes
                    "websocket_enabled": True
                },
                "standard_stocks": {
                    "count": 560,
                    "update_seconds": self.tiers['standard'].update_interval_minutes * 60,
                    "cache_ttl": 3600,  # 1 hour
                    "websocket_enabled": False
                },
                "global_stocks": {
                    "symbols": self.tiers['global'].stocks,
                    "update_seconds": self.tiers['global'].update_interval_minutes * 60,
                    "cache_ttl": 14400,  # 4 hours
                    "websocket_enabled": False
                }
            },
            
            "user_interface": {
                "default_view": "heat_map",
                "views": [
                    {
                        "name": "heat_map",
                        "title": "🔥 Heat Map",
                        "description": "Tüm hisseler renk kodlu görünüm",
                        "data_sources": ["vip", "premium"],
                        "refresh_rate": "5s"
                    },
                    {
                        "name": "watchlist",
                        "title": "⭐ Watchlist", 
                        "description": "Kullanıcı seçili hisseler",
                        "data_sources": ["user_selection"],
                        "refresh_rate": "1s"
                    },
                    {
                        "name": "explorer",
                        "title": "🔍 Explorer",
                        "description": "Tüm hisseler arama/filtreleme",
                        "data_sources": ["all"],
                        "refresh_rate": "60s"
                    },
                    {
                        "name": "analytics",
                        "title": "📊 Analytics",
                        "description": "Detaylı analiz ve grafikler",
                        "data_sources": ["historical", "calculated"],
                        "refresh_rate": "300s"
                    }
                ]
            },
            
            "performance_optimization": {
                "redis_cache": {
                    "enabled": True,
                    "hot_data_ttl": 300,      # 5 min
                    "warm_data_ttl": 1800,    # 30 min
                    "cold_data_ttl": 14400    # 4 hours
                },
                "api_rate_limiting": {
                    "max_concurrent": 10,
                    "request_delay": 0.1,
                    "daily_quota_monitoring": True
                },
                "data_compression": {
                    "enabled": True,
                    "algorithm": "gzip",
                    "level": 6
                }
            }
        }
        
        return config

    def export_plan(self, filename: str = None):
        """Export the complete dashboard plan"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'dashboard_plan_{timestamp}.json'
        
        # Calculate API usage
        usage = self.calculate_daily_api_usage()
        
        # Generate configuration
        config = self.generate_dashboard_config()
        
        # Combine everything
        plan = {
            "meta": {
                "title": "MAMUT R600 Smart Dashboard Plan",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "api_daily_limit": self.daily_api_limit
            },
            "api_usage_analysis": usage,
            "dashboard_config": config,
            "implementation_notes": [
                "VIP tier için WebSocket real-time updates",
                "Redis cache ile performance optimization", 
                "API rate limiting ile quota protection",
                "Responsive design mobile-first approach",
                "Progressive loading ile fast initial render",
                "Error handling ve fallback mechanisms"
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        
        return filename, plan

if __name__ == "__main__":
    planner = SmartDashboardPlanner()
    
    print("🎯 MAMUT R600 SMART DASHBOARD PLAN")
    print("=" * 50)
    
    # Calculate usage
    usage = planner.calculate_daily_api_usage()
    
    print("\n📊 API USAGE ANALYSIS:")
    print("-" * 30)
    for tier_name, data in usage.items():
        if tier_name != 'total':
            print(f"{data['tier']:20} | {data['stock_count']:3} hisse | "
                  f"{data['update_interval']:8} | {data['daily_calls']:6,} call | "
                  f"{data['percentage']:5.1f}%")
    
    total = usage['total']
    print(f"\n📈 TOPLAM: {total['daily_calls']:,} call ({total['percentage']:.1f}%)")
    print(f"🚀 KALAN: {total['remaining']:,} call")
    
    # Export plan
    filename, plan = planner.export_plan()
    print(f"\n💾 Plan exported to: {filename}")
    print(f"📏 Total tiers: {len(planner.tiers)}")
    print(f"🎯 Total stocks: {sum(len(tier.stocks) for tier in planner.tiers.values())}")
    
    if total['percentage'] <= 80:
        print("\n✅ PLAN APPROVED - API limit içinde güvenli!")
    elif total['percentage'] <= 95:
        print("\n⚠️ PLAN CAUTION - API limit'e yakın!")
    else:
        print("\n❌ PLAN REJECTED - API limit aşıyor!")
