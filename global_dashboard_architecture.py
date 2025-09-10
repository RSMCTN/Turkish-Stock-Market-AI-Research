#!/usr/bin/env python3
"""
MAMUT R600 - Global Multi-Market Dashboard Architecture
Railway PostgreSQL + Redis + TradingView + Multi-Language Sentiment
"""

import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MarketTier:
    region: str
    markets: List[str]
    stocks_count: int
    update_interval_minutes: int
    language_support: List[str]
    sentiment_priority: str
    daily_calls: int

class GlobalDashboardArchitect:
    def __init__(self):
        self.daily_api_limit = 150000
        self.allocations = {
            'turkey': int(self.daily_api_limit * 0.25),      # 37,500
            'global_top10': int(self.daily_api_limit * 0.50), # 75,000  
            'reserve': int(self.daily_api_limit * 0.25)       # 37,500
        }
        
        # Market tier definitions
        self.tiers = {
            'turkey': MarketTier(
                region="Türkiye",
                markets=["BIST"],
                stocks_count=660,
                update_interval_minutes=5,  # BIST 30: 5min, others: 15-60min
                language_support=["Turkish"],
                sentiment_priority="High",
                daily_calls=28800  # Calculated from previous analysis
            ),
            
            'english_markets': MarketTier(
                region="English Speaking",
                markets=["NYSE", "NASDAQ", "LSE", "TSX", "ASX", "HKEX"],
                stocks_count=310,  # Top stocks from each
                update_interval_minutes=15,
                language_support=["English"],
                sentiment_priority="High", 
                daily_calls=29760  # (1440/15) * 310
            ),
            
            'major_eu': MarketTier(
                region="Major European",
                markets=["XETRA", "Euronext"],
                stocks_count=50,   # DAX 30 + CAC 20
                update_interval_minutes=30,
                language_support=["German", "French"],
                sentiment_priority="Medium",
                daily_calls=2400   # (1440/30) * 50
            ),
            
            'asian_markets': MarketTier(
                region="Asian",
                markets=["TSE", "SSE", "TWSE"],
                stocks_count=70,   # Nikkei 30 + Shanghai 20 + Taiwan 20
                update_interval_minutes=60,
                language_support=["Japanese", "Chinese"],
                sentiment_priority="Low",
                daily_calls=1680   # (1440/60) * 70
            )
        }

    def calculate_usage(self) -> Dict:
        """Calculate API usage across all tiers"""
        usage = {}
        total_calls = 0
        
        for tier_name, tier in self.tiers.items():
            total_calls += tier.daily_calls
            usage[tier_name] = {
                'region': tier.region,
                'daily_calls': tier.daily_calls,
                'stocks': tier.stocks_count,
                'update_freq': f"{tier.update_interval_minutes}min",
                'languages': tier.language_support,
                'sentiment': tier.sentiment_priority
            }
        
        usage['summary'] = {
            'total_daily_calls': total_calls,
            'turkey_allocation': self.allocations['turkey'],
            'global_allocation': self.allocations['global_top10'],
            'reserve_allocation': self.allocations['reserve'],
            'usage_percentage': round(total_calls / self.daily_api_limit * 100, 1),
            'remaining_calls': self.daily_api_limit - total_calls
        }
        
        return usage

    def generate_dashboard_config(self) -> Dict:
        """Generate comprehensive dashboard configuration"""
        
        config = {
            "meta": {
                "name": "MAMUT R600 Global Dashboard",
                "version": "2.0 - Global Multi-Market",
                "created": datetime.now().isoformat(),
                "api_daily_limit": self.daily_api_limit,
                "railway_project": "https://railway.com/project/5f254ec7-154c-421b-8d57-d7ed461ec6ee"
            },
            
            "infrastructure": {
                "database": {
                    "primary": "PostgreSQL (Railway)",
                    "cache": "Redis (Railway)",
                    "status": "✅ Ready"
                },
                "backend": {
                    "api": "FastAPI",
                    "deployment": "Railway",
                    "status": "✅ Running"
                },
                "frontend": {
                    "framework": "Next.js + React",
                    "styling": "TailwindCSS", 
                    "charts": "TradingView Widgets",
                    "status": "🔄 Updating"
                }
            },
            
            "market_coverage": {
                "turkey": {
                    "allocation": "25% (37,500 calls)",
                    "exchanges": ["BIST"],
                    "stocks": {
                        "vip": {"count": 30, "interval": "5min", "desc": "BIST 30"},
                        "premium": {"count": 70, "interval": "15min", "desc": "BIST 100-30"},
                        "standard": {"count": 560, "interval": "60min", "desc": "Other Turkish"}
                    },
                    "languages": ["Turkish"],
                    "sentiment": "✅ Ready"
                },
                "global": {
                    "allocation": "50% (75,000 calls)",
                    "priority_markets": {
                        "english": {
                            "markets": ["NYSE", "NASDAQ", "LSE", "TSX", "ASX", "HKEX"],
                            "stocks": 310,
                            "interval": "15min",
                            "sentiment": "🔄 Phase 1"
                        },
                        "european": {
                            "markets": ["XETRA", "Euronext"],
                            "stocks": 50,
                            "interval": "30min", 
                            "sentiment": "📋 Phase 2"
                        },
                        "asian": {
                            "markets": ["TSE", "SSE", "TWSE"],
                            "stocks": 70,
                            "interval": "60min",
                            "sentiment": "📋 Phase 3"
                        }
                    }
                }
            },
            
            "dashboard_views": {
                "heat_map": {
                    "title": "🔥 Global Heat Map",
                    "description": "Multi-market overview with color coding",
                    "data_sources": ["turkey_vip", "english_markets", "eu_major"],
                    "update_frequency": "5-15min",
                    "tradingview_widget": "Market Overview"
                },
                "turkey_focus": {
                    "title": "🇹🇷 Turkey Deep Dive", 
                    "description": "Comprehensive BIST analysis",
                    "data_sources": ["all_turkish_stocks"],
                    "update_frequency": "5-60min",
                    "tradingview_widget": "Advanced Chart + Screener"
                },
                "global_watch": {
                    "title": "🌍 Global Watchlist",
                    "description": "International portfolio tracking",
                    "data_sources": ["user_selected_global"],
                    "update_frequency": "15min",
                    "tradingview_widget": "Mini Charts Grid"
                },
                "sentiment_radar": {
                    "title": "🎯 Multi-Language Sentiment",
                    "description": "AI sentiment across markets", 
                    "data_sources": ["turkish_sentiment", "english_sentiment"],
                    "update_frequency": "60min",
                    "tradingview_widget": "Custom Sentiment Display"
                }
            },
            
            "tradingview_integration": {
                "plan": "FREE (initially)",
                "upgrade_path": "PRO ($14.95/month) after user validation",
                "widgets": {
                    "advanced_chart": {
                        "use_case": "Main stock analysis",
                        "features": ["Candlesticks", "Volume", "Technical indicators"],
                        "placement": "Turkey Focus + Global Watch"
                    },
                    "market_overview": {
                        "use_case": "Heat map view",
                        "features": ["Multi-asset grid", "Color coding", "Performance sorting"],
                        "placement": "Heat Map view"
                    },
                    "mini_chart": {
                        "use_case": "Stock list previews",
                        "features": ["Compact design", "Quick overview"],
                        "placement": "Search results, watchlists"
                    },
                    "screener": {
                        "use_case": "Stock filtering",
                        "features": ["Custom filters", "Sortable columns", "Export"],
                        "placement": "Explorer/Search page"
                    }
                }
            },
            
            "sentiment_analysis": {
                "current": {
                    "turkish": {
                        "status": "✅ Production Ready",
                        "models": ["Turkish BERT", "Custom LSTM"],
                        "sources": ["KAP", "Turkish financial news"]
                    }
                },
                "roadmap": {
                    "phase_1_english": {
                        "timeline": "2-3 weeks", 
                        "models": ["FinBERT", "RoBERTa"],
                        "sources": ["Reuters", "Bloomberg", "Yahoo Finance"],
                        "markets": ["US", "UK", "Canada", "Australia"]
                    },
                    "phase_2_european": {
                        "timeline": "3-4 weeks",
                        "models": ["German BERT", "French FinancialBERT"],
                        "sources": ["Handelsblatt", "Les Echos"],
                        "markets": ["Germany", "France"]
                    },
                    "phase_3_asian": {
                        "timeline": "4-6 weeks",
                        "models": ["Multilingual BERT", "Japanese FinBERT"],
                        "sources": ["Nikkei", "China Daily"],
                        "markets": ["Japan", "China", "Hong Kong"]
                    }
                }
            },
            
            "user_experience": {
                "language_selector": {
                    "primary": "Turkish",
                    "secondary": ["English"],
                    "future": ["German", "French", "Japanese", "Chinese"]
                },
                "market_selector": {
                    "default": "Turkey",
                    "available": ["Turkey", "US", "UK", "Germany", "Japan", "Global"],
                    "quick_switch": True
                },
                "responsive_design": {
                    "mobile_first": True,
                    "breakpoints": ["mobile", "tablet", "desktop", "ultra-wide"],
                    "pwa_ready": True
                },
                "performance": {
                    "lazy_loading": True,
                    "infinite_scroll": True,
                    "virtual_scrolling": "For large lists",
                    "caching_strategy": "Redis + Browser cache"
                }
            }
        }
        
        return config

    def export_architecture(self, filename: str = None) -> tuple:
        """Export complete architecture plan"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'global_dashboard_architecture_{timestamp}.json'
        
        # Calculate usage
        usage = self.calculate_usage()
        
        # Generate configuration
        config = self.generate_dashboard_config()
        
        # Complete architecture
        architecture = {
            "title": "MAMUT R600 - Global Multi-Market Dashboard Architecture",
            "api_usage_analysis": usage,
            "dashboard_configuration": config,
            "implementation_priorities": [
                {
                    "priority": "P1 - Critical",
                    "tasks": [
                        "TradingView widget integration",
                        "English sentiment analysis setup", 
                        "Global market data pipeline",
                        "Multi-language UI framework"
                    ]
                },
                {
                    "priority": "P2 - High",
                    "tasks": [
                        "Heat map view implementation",
                        "Advanced filtering system",
                        "Mobile responsive optimization",
                        "Performance monitoring"
                    ]
                },
                {
                    "priority": "P3 - Medium",
                    "tasks": [
                        "European markets integration",
                        "Advanced charting features",
                        "Portfolio tracking",
                        "Alert system"
                    ]
                }
            ],
            "success_metrics": {
                "technical": {
                    "api_usage": "< 80% daily limit",
                    "response_time": "< 2s page load",
                    "uptime": "> 99.5%",
                    "cache_hit_rate": "> 85%"
                },
                "business": {
                    "user_engagement": "Daily active users growth",
                    "market_coverage": "Turkey + Top 5 global markets",
                    "sentiment_accuracy": "> 75% precision",
                    "mobile_usage": "> 60% traffic"
                }
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(architecture, f, indent=2, ensure_ascii=False)
        
        return filename, architecture

if __name__ == "__main__":
    architect = GlobalDashboardArchitect()
    
    print("🌍 MAMUT R600 - GLOBAL DASHBOARD ARCHITECTURE")
    print("=" * 60)
    
    # Calculate and display usage
    usage = architect.calculate_usage()
    
    print("\n📊 API USAGE SUMMARY:")
    print("-" * 30)
    for tier, data in usage.items():
        if tier != 'summary':
            print(f"{data['region']:15} | {data['stocks']:3} stocks | "
                  f"{data['update_freq']:5} | {data['daily_calls']:5,} calls | "
                  f"Sentiment: {data['sentiment']}")
    
    summary = usage['summary']
    print(f"\n📈 TOTAL USAGE: {summary['total_daily_calls']:,} calls "
          f"({summary['usage_percentage']}%)")
    print(f"🎯 REMAINING: {summary['remaining_calls']:,} calls")
    
    # Export architecture
    filename, arch = architect.export_architecture()
    print(f"\n💾 Architecture exported to: {filename}")
    
    if summary['usage_percentage'] <= 75:
        print("\n✅ ARCHITECTURE APPROVED - Excellent capacity utilization!")
    else:
        print("\n⚠️ ARCHITECTURE NEEDS REVIEW - High capacity usage!")
