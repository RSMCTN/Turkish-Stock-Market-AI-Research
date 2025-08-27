"""
BIST Real Data Service - Excel Based
Uses real BIST data from basestock.xlsx for accurate stock information
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BISTRealDataService:
    """Real BIST data service using Excel data source"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or str(Path(__file__).parent.parent.parent.parent / "data" / "excell_MIQ")
        self.stocks_data: List[Dict] = []
        self.stocks_dict: Dict[str, Dict] = {}
        self._load_real_data()
        
    def _load_real_data(self):
        """Load real BIST data from JSON file"""
        try:
            json_path = Path(self.data_path) / "bist_real_data.json"
            
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.stocks_data = json.load(f)
                
                # Create quick lookup dictionary
                self.stocks_dict = {stock['symbol']: stock for stock in self.stocks_data}
                logger.info(f"✅ Loaded {len(self.stocks_data)} real BIST stocks from JSON")
                
            else:
                logger.warning(f"JSON file not found: {json_path}")
                self._fallback_load_from_excel()
                
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            self._fallback_load_from_excel()
    
    def _fallback_load_from_excel(self):
        """Fallback: Load directly from Excel if JSON fails"""
        try:
            excel_path = Path(self.data_path) / "basestock.xlsx"
            if excel_path.exists():
                df = pd.read_excel(excel_path)
                self.stocks_data = []
                
                for _, row in df.iterrows():
                    stock = {
                        'symbol': str(row['SEMBOL']),
                        'name': str(row['ACKL']),
                        'name_turkish': str(row['ACKL']),
                        'sector': str(row['SEKTOR']),
                        'sector_turkish': str(row['SEKTOR']),
                        'last_price': float(row['SON']) if pd.notna(row['SON']) else 0.0,
                        'change': float(row['FARK']) if pd.notna(row['FARK']) else 0.0,
                        'change_percent': float(row['%FARK']) if pd.notna(row['%FARK']) else 0.0,
                        'volume': float(row['T.HACİM']) if pd.notna(row['T.HACİM']) else 0.0,
                        'market_cap': float(row['PIY.DEG']) if pd.notna(row['PIY.DEG']) else 0.0,
                        'bist_30': float(row.get('XU030 DAKI AG.', 0)) > 0,
                        'bist_50': float(row.get('XU050 DEKI AG.', 0)) > 0,
                        'bist_100': float(row.get('XU100 DEKI AG.', 0)) > 0,
                        'is_active': True,
                        'last_updated': '2025-08-27T20:00:00'
                    }
                    self.stocks_data.append(stock)
                
                # Create quick lookup dictionary
                self.stocks_dict = {stock['symbol']: stock for stock in self.stocks_data}
                logger.info(f"✅ Loaded {len(self.stocks_data)} real BIST stocks from Excel fallback")
            else:
                logger.error(f"Excel file not found: {excel_path}")
                
        except Exception as e:
            logger.error(f"Error loading Excel fallback: {e}")
    
    def get_all_stocks(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get all stocks with optional limit"""
        stocks = self.stocks_data[:limit] if limit else self.stocks_data
        
        # Format for API response
        formatted_stocks = []
        for stock in stocks:
            formatted_stock = {
                "symbol": stock["symbol"],
                "name": stock["name"],
                "name_turkish": stock["name_turkish"],
                "sector": stock["sector"],
                "sector_turkish": stock["sector_turkish"],
                "market_cap": stock["market_cap"],
                "last_price": stock["last_price"],
                "change": stock["change"],
                "change_percent": stock["change_percent"],
                "volume": stock["volume"],
                "bist_markets": self._get_bist_markets(stock),
                "market_segment": "yildiz_pazar" if stock.get("bist_30") else "ana_pazar",
                "is_active": stock["is_active"],
                "last_updated": datetime.now().isoformat()
            }
            formatted_stocks.append(formatted_stock)
            
        return formatted_stocks
    
    def get_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get specific stock by symbol"""
        stock = self.stocks_dict.get(symbol.upper())
        if not stock:
            return None
            
        return {
            "symbol": stock["symbol"],
            "name": stock["name"],
            "name_turkish": stock["name_turkish"],
            "sector": stock["sector"],
            "sector_turkish": stock["sector_turkish"],
            "market_cap": stock["market_cap"],
            "last_price": stock["last_price"],
            "change": stock["change"],
            "change_percent": stock["change_percent"],
            "volume": stock["volume"],
            "bist_markets": self._get_bist_markets(stock),
            "market_segment": "yildiz_pazar" if stock.get("bist_30") else "ana_pazar",
            "is_active": stock["is_active"],
            "last_updated": datetime.now().isoformat()
        }
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview with real data"""
        active_stocks = [s for s in self.stocks_data if s["is_active"]]
        
        # Calculate market statistics
        total_volume = sum(s["volume"] for s in active_stocks if s["volume"] > 0)
        total_value = sum(s["last_price"] * s["volume"] for s in active_stocks if s["volume"] > 0 and s["last_price"] > 0)
        
        rising_stocks = len([s for s in active_stocks if s["change_percent"] > 0])
        falling_stocks = len([s for s in active_stocks if s["change_percent"] < 0])
        unchanged_stocks = len([s for s in active_stocks if s["change_percent"] == 0])
        
        # BIST 100 calculation (simplified)
        bist_100_stocks = [s for s in active_stocks if s.get("bist_100")]
        bist_100_change = sum(s["change_percent"] for s in bist_100_stocks) / len(bist_100_stocks) if bist_100_stocks else 0
        
        # BIST 30 calculation
        bist_30_stocks = [s for s in active_stocks if s.get("bist_30")]
        bist_30_change = sum(s["change_percent"] for s in bist_30_stocks) / len(bist_30_stocks) if bist_30_stocks else 0
        
        return {
            "bist_100": {
                "value": 10500.0 + (bist_100_change * 10),  # Simulated index value
                "change": bist_100_change,
                "change_direction": "up" if bist_100_change > 0 else "down" if bist_100_change < 0 else "flat"
            },
            "bist_30": {
                "value": 11200.0 + (bist_30_change * 15),  # Simulated index value
                "change": bist_30_change,
                "change_direction": "up" if bist_30_change > 0 else "down" if bist_30_change < 0 else "flat"
            },
            "market_statistics": {
                "total_volume": int(total_volume),
                "total_value": int(total_value),
                "rising_stocks": rising_stocks,
                "falling_stocks": falling_stocks,
                "unchanged_stocks": unchanged_stocks
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def get_sectors(self) -> List[Dict[str, Any]]:
        """Get unique sectors"""
        sectors = {}
        for stock in self.stocks_data:
            sector = stock["sector"]
            if sector not in sectors:
                sectors[sector] = {
                    "name": sector,
                    "name_turkish": stock["sector_turkish"],
                    "stock_count": 0
                }
            sectors[sector]["stock_count"] += 1
            
        return list(sectors.values())
    
    def get_markets(self) -> List[Dict[str, Any]]:
        """Get market segments"""
        return [
            {"name": "bist_30", "name_turkish": "BIST 30", "description": "BIST 30 Endeksi"},
            {"name": "bist_50", "name_turkish": "BIST 50", "description": "BIST 50 Endeksi"},
            {"name": "bist_100", "name_turkish": "BIST 100", "description": "BIST 100 Endeksi"},
            {"name": "yildiz_pazar", "name_turkish": "Yıldız Pazar", "description": "Yıldız Pazar"},
            {"name": "ana_pazar", "name_turkish": "Ana Pazar", "description": "Ana Pazar"}
        ]
    
    def search_stocks(self, query: str) -> List[Dict[str, Any]]:
        """Search stocks by symbol or name"""
        query_lower = query.lower()
        results = []
        
        for stock in self.stocks_data:
            if (query_lower in stock["symbol"].lower() or 
                query_lower in stock["name"].lower() or
                query_lower in stock["name_turkish"].lower()):
                results.append(self.get_stock(stock["symbol"]))
                
        return results[:20]  # Limit to 20 results
    
    def _get_bist_markets(self, stock: Dict) -> List[str]:
        """Get BIST market memberships for a stock"""
        markets = []
        if stock.get("bist_30"):
            markets.append("bist_30")
        if stock.get("bist_50"):
            markets.append("bist_50")  
        if stock.get("bist_100"):
            markets.append("bist_100")
        if stock.get("bist_30"):  # BIST 30 stocks are typically in Yıldız Pazar
            markets.append("yildiz_pazar")
        else:
            markets.append("ana_pazar")
        return markets

# Service instance factory
_real_service_instance = None

def get_real_bist_service() -> BISTRealDataService:
    """Get or create the real BIST data service instance"""
    global _real_service_instance
    if _real_service_instance is None:
        _real_service_instance = BISTRealDataService()
    return _real_service_instance
