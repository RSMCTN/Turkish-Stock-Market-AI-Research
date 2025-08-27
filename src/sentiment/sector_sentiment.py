"""
Sector-Based Sentiment Analysis for BIST Markets
Groups companies by sectors and analyzes sentiment impact
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class SectorInfo:
    """Information about a BIST sector"""
    sector_id: str
    name: str
    name_turkish: str
    keywords: List[str]
    companies: List[str]  # BIST symbols in this sector
    weight: float  # Importance weight for market sentiment


class BISTSectorAnalyzer:
    """Analyzes sentiment by BIST sectors"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sectors = self._initialize_bist_sectors()
        self.markets = self._initialize_bist_markets()
        
        print(f"🏭 BIST SECTOR ANALYZER:")
        print(f"   📊 Total sectors: {len(self.sectors)}")
        print(f"   📈 BIST markets: {len(self.markets)}")
        print(f"   🏢 Total companies mapped: {sum(len(s.companies) for s in self.sectors.values())}")
        
        self.logger.info("BIST Sector Analyzer initialized")
    
    def _initialize_bist_sectors(self) -> Dict[str, SectorInfo]:
        """Initialize BIST sectors with companies and keywords"""
        return {
            'banking': SectorInfo(
                sector_id='banking',
                name='Banking',
                name_turkish='Bankacılık',
                keywords=[
                    'banka', 'bankası', 'banking', 'kredi', 'loan', 'mevduat', 'deposit',
                    'faiz', 'interest', 'tcmb', 'merkez bankası', 'monetary policy', 
                    'npv', 'takipteki krediler', 'non-performing', 'tier1', 'basel',
                    'anapara', 'taksit', 'mortgage', 'konut kredisi', 'bireysel bankacılık'
                ],
                companies=[
                    'AKBNK', 'GARAN', 'ISCTR', 'YKBNK', 'HALKB', 'VAKBN', 'ALBRK', 'ICBCT', 'TSKB'
                ],
                weight=0.25  # High weight - banks are critical
            ),
            
            'aviation': SectorInfo(
                sector_id='aviation',
                name='Aviation',
                name_turkish='Havacılık',
                keywords=[
                    'havayolu', 'aircraft', 'uçak', 'airport', 'havaalanı', 'flight', 'uçuş',
                    'passenger', 'yolcu', 'cargo', 'kargo', 'fuel', 'yakıt', 'jet fuel',
                    'iata', 'icao', 'aviation', 'airline', 'thy', 'turkish airlines'
                ],
                companies=[
                    'THYAO', 'PGSUS'  # Turkish Airlines, Pegasus
                ],
                weight=0.15
            ),
            
            'industrials': SectorInfo(
                sector_id='industrials',
                name='Industrials',
                name_turkish='Sanayi',
                keywords=[
                    'sanayi', 'industrial', 'manufacturing', 'imalat', 'factory', 'fabrika',
                    'production', 'üretim', 'machinery', 'makine', 'steel', 'çelik',
                    'iron', 'demir', 'metal', 'aluminum', 'alüminyum', 'chemicals', 'kimya',
                    'petrochemical', 'petrokimya', 'energy', 'enerji', 'electricity', 'elektrik'
                ],
                companies=[
                    'KRDMD', 'EREGL', 'ARCLK', 'VESTL', 'PETKM', 'TUPRS', 'SODA', 'TRKCM',
                    'ASELS', 'TOASO', 'FROTO', 'OTKAR'
                ],
                weight=0.20
            ),
            
            'retail': SectorInfo(
                sector_id='retail',
                name='Retail',
                name_turkish='Perakende',
                keywords=[
                    'perakende', 'retail', 'market', 'mağaza', 'store', 'shopping', 'alışveriş',
                    'consumer', 'tüketici', 'food', 'gıda', 'supermarket', 'hypermarket',
                    'chain', 'zincir', 'franchise', 'brand', 'marka', 'sales', 'satış'
                ],
                companies=[
                    'BIMAS', 'MGROS', 'CCOLA', 'ULKER'
                ],
                weight=0.12
            ),
            
            'technology': SectorInfo(
                sector_id='technology',
                name='Technology',
                name_turkish='Teknoloji',
                keywords=[
                    'teknoloji', 'technology', 'software', 'yazılım', 'digital', 'dijital',
                    'it', 'bilişim', 'computer', 'bilgisayar', 'internet', 'cloud', 'bulut',
                    'artificial intelligence', 'ai', 'machine learning', 'data', 'veri',
                    'cybersecurity', 'siber güvenlik', 'fintech', 'blockchain'
                ],
                companies=[
                    'LOGO', 'NETAS', 'TTKOM', 'TCELL'
                ],
                weight=0.18
            ),
            
            'telecommunications': SectorInfo(
                sector_id='telecommunications',
                name='Telecommunications',
                name_turkish='Telekomünikasyon',
                keywords=[
                    'telekom', 'telecom', 'telecommunications', 'phone', 'telefon', 'mobile',
                    'mobil', 'gsm', '3g', '4g', '5g', 'network', 'ağ', 'internet', 'broadband',
                    'fiber', 'cable', 'kablo', 'subscriber', 'abone', 'roaming'
                ],
                companies=[
                    'TTKOM', 'TCELL', 'NETAS'
                ],
                weight=0.14
            ),
            
            'construction': SectorInfo(
                sector_id='construction',
                name='Construction & Real Estate',
                name_turkish='İnşaat ve Gayrimenkul',
                keywords=[
                    'inşaat', 'construction', 'building', 'bina', 'real estate', 'gayrimenkul',
                    'house', 'ev', 'apartment', 'daire', 'residential', 'konut', 'commercial', 'ticari',
                    'project', 'proje', 'contractor', 'müteahhit', 'cement', 'çimento',
                    'infrastructure', 'altyapı', 'road', 'yol', 'bridge', 'köprü'
                ],
                companies=[
                    'ENKAI', 'TKFEN'
                ],
                weight=0.16
            ),
            
            'automotive': SectorInfo(
                sector_id='automotive',
                name='Automotive',
                name_turkish='Otomotiv',
                keywords=[
                    'otomotiv', 'automotive', 'car', 'araba', 'vehicle', 'araç', 'truck', 'kamyon',
                    'bus', 'otobüs', 'engine', 'motor', 'spare parts', 'yedek parça',
                    'assembly', 'montaj', 'export', 'ihracat', 'domestic market', 'yerli pazar'
                ],
                companies=[
                    'TOASO', 'FROTO', 'OTKAR', 'TIRE', 'BRISA', 'BFREN'
                ],
                weight=0.17
            ),
            
            'pharmaceuticals': SectorInfo(
                sector_id='pharmaceuticals',
                name='Pharmaceuticals',
                name_turkish='İlaç',
                keywords=[
                    'ilaç', 'pharmaceutical', 'medicine', 'drug', 'health', 'sağlık',
                    'hospital', 'hastane', 'clinic', 'klinik', 'medical', 'tıbbi',
                    'vaccine', 'aşı', 'treatment', 'tedavi', 'research', 'araştırma'
                ],
                companies=[
                    'DEVA', 'ECZYT'
                ],
                weight=0.10
            ),
            
            'fertilizers': SectorInfo(
                sector_id='fertilizers',
                name='Fertilizers & Agriculture',
                name_turkish='Gübre ve Tarım',
                keywords=[
                    'gübre', 'fertilizer', 'agriculture', 'tarım', 'farming', 'çiftçilik',
                    'crop', 'ürün', 'harvest', 'hasat', 'seed', 'tohum', 'soil', 'toprak',
                    'nitrogen', 'azot', 'phosphate', 'fosfat', 'potash', 'potaş'
                ],
                companies=[
                    'GUBRF', 'BAGFS'
                ],
                weight=0.08
            ),
            
            'holdings': SectorInfo(
                sector_id='holdings',
                name='Holdings',
                name_turkish='Holding Şirketleri',
                keywords=[
                    'holding', 'group', 'grup', 'conglomerate', 'diversified', 'çeşitlendirilmiş',
                    'subsidiary', 'bağlı şirket', 'affiliate', 'iştirak', 'investment', 'yatırım',
                    'portfolio', 'portföy'
                ],
                companies=[
                    'KCHOL', 'SAHOL', 'DOHOL', 'GSDHO'
                ],
                weight=0.15
            ),
            
            'sports': SectorInfo(
                sector_id='sports',
                name='Sports',
                name_turkish='Spor',
                keywords=[
                    'spor', 'sports', 'football', 'futbol', 'soccer', 'team', 'takım',
                    'match', 'maç', 'player', 'oyuncu', 'stadium', 'stadyum', 'league', 'lig',
                    'championship', 'şampiyonluk', 'transfer', 'sponsor'
                ],
                companies=[
                    'BJKAS', 'FENER', 'GSDHO'  # Football clubs
                ],
                weight=0.05
            )
        }
    
    def _initialize_bist_markets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize BIST market categories"""
        return {
            'bist_30': {
                'name': 'BIST 30',
                'description': 'Top 30 most liquid stocks',
                'companies': [
                    'AKBNK', 'GARAN', 'ISCTR', 'YKBNK', 'HALKB', 'VAKBN',  # Banks
                    'THYAO', 'PGSUS',  # Aviation
                    'ARCLK', 'VESTL',  # Consumer
                    'KCHOL', 'SAHOL',  # Holdings
                    'BIMAS', 'MGROS',  # Retail
                    'ASELS',  # Defense
                    'TUPRS', 'PETKM',  # Energy/Petrochemicals
                    'KRDMD', 'EREGL',  # Steel
                    'TTKOM', 'TCELL',  # Telecom
                ],
                'weight': 0.40,
                'priority': 'critical'
            },
            
            'bist_50': {
                'name': 'BIST 50',
                'description': 'Top 50 stocks by market cap and liquidity',
                'extends': 'bist_30',  # Includes BIST 30
                'additional_companies': [
                    'SISE', 'TRKCM',  # Glass/Industrial
                    'CCOLA', 'ULKER',  # Food & Beverage
                    'TOASO', 'FROTO', 'OTKAR',  # Automotive
                    'ENKAI', 'TKFEN',  # Construction
                    'ALBRK', 'ICBCT',  # Additional Banks
                ],
                'weight': 0.30,
                'priority': 'high'
            },
            
            'bist_100': {
                'name': 'BIST 100',
                'description': 'Top 100 stocks - main benchmark index',
                'extends': 'bist_50',
                'additional_companies': [
                    'DEVA', 'ECZYT',  # Pharmaceuticals
                    'LOGO', 'NETAS',  # Technology
                    'GUBRF', 'BAGFS',  # Agriculture/Fertilizers
                    'TIRE', 'BRISA', 'BFREN',  # Automotive parts
                    'DOHOL',  # Media holdings
                    'BJKAS', 'FENER', 'GSDHO',  # Sports
                ],
                'weight': 0.25,
                'priority': 'medium'
            },
            
            'yildiz_pazar': {
                'name': 'Yıldız Pazar (Star Market)',
                'description': 'Premium market segment for blue-chip companies',
                'companies': [
                    'AKBNK', 'GARAN', 'ISCTR', 'YKBNK', 'HALKB', 'VAKBN',
                    'THYAO', 'ASELS', 'KCHOL', 'SAHOL', 'ARCLK', 'BIMAS',
                    'TUPRS', 'KRDMD', 'EREGL', 'TTKOM', 'TCELL'
                ],
                'weight': 0.35,
                'priority': 'critical',
                'market_type': 'premium'
            },
            
            'ana_pazar': {
                'name': 'Ana Pazar (Main Market)',
                'description': 'Main market segment',
                'companies': [
                    'VESTL', 'MGROS', 'CCOLA', 'ULKER', 'PGSUS', 'SISE',
                    'TOASO', 'FROTO', 'OTKAR', 'ENKAI', 'TKFEN'
                ],
                'weight': 0.20,
                'priority': 'medium',
                'market_type': 'main'
            }
        }
    
    def get_sector_for_symbol(self, symbol: str) -> Optional[SectorInfo]:
        """Get sector information for a given symbol"""
        for sector in self.sectors.values():
            if symbol in sector.companies:
                return sector
        return None
    
    def get_market_for_symbol(self, symbol: str) -> List[str]:
        """Get market categories for a given symbol"""
        markets = []
        for market_id, market_info in self.markets.items():
            if symbol in market_info.get('companies', []):
                markets.append(market_id)
            # Check if it's in extended markets
            if market_info.get('extends') and symbol in self.markets.get(market_info['extends'], {}).get('companies', []):
                markets.append(market_id)
        return markets
    
    def analyze_sector_sentiment(self, news_articles: List[Dict], symbol: str = None) -> Dict[str, Any]:
        """Analyze sentiment impact by sectors"""
        sector_sentiments = {}
        
        # Initialize sector sentiment scores
        for sector_id, sector in self.sectors.items():
            sector_sentiments[sector_id] = {
                'name': sector.name,
                'name_turkish': sector.name_turkish,
                'sentiment_score': 0.0,
                'article_count': 0,
                'relevant_articles': [],
                'companies_mentioned': set(),
                'weight': sector.weight
            }
        
        # Analyze each article
        for article in news_articles:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            full_text = f"{title} {content}"
            
            # Check which sectors are mentioned
            for sector_id, sector in self.sectors.items():
                relevance_score = 0
                
                # Check for sector keywords
                for keyword in sector.keywords:
                    if keyword.lower() in full_text:
                        relevance_score += 1
                
                # Check for company mentions
                for company in sector.companies:
                    if company.lower() in full_text:
                        relevance_score += 2  # Company mentions are more important
                        sector_sentiments[sector_id]['companies_mentioned'].add(company)
                
                # If relevant, add to sector sentiment
                if relevance_score > 0:
                    sentiment_score = article.get('sentiment', {}).get('compound', 0.0)
                    sector_sentiments[sector_id]['sentiment_score'] += sentiment_score * relevance_score
                    sector_sentiments[sector_id]['article_count'] += 1
                    sector_sentiments[sector_id]['relevant_articles'].append({
                        'title': article.get('title'),
                        'sentiment': sentiment_score,
                        'relevance': relevance_score,
                        'source': article.get('source')
                    })
        
        # Calculate average sentiment for each sector
        for sector_id in sector_sentiments:
            if sector_sentiments[sector_id]['article_count'] > 0:
                total_score = sector_sentiments[sector_id]['sentiment_score']
                count = sector_sentiments[sector_id]['article_count']
                sector_sentiments[sector_id]['average_sentiment'] = round(total_score / count, 3)
            else:
                sector_sentiments[sector_id]['average_sentiment'] = 0.0
            
            # Convert set to list for JSON serialization
            sector_sentiments[sector_id]['companies_mentioned'] = list(sector_sentiments[sector_id]['companies_mentioned'])
        
        # Calculate overall market sentiment
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for sector_data in sector_sentiments.values():
            if sector_data['article_count'] > 0:
                weighted_sentiment += sector_data['average_sentiment'] * sector_data['weight']
                total_weight += sector_data['weight']
        
        overall_sentiment = round(weighted_sentiment / total_weight, 3) if total_weight > 0 else 0.0
        
        # If specific symbol requested, get its sector
        symbol_sector = None
        if symbol:
            symbol_sector_info = self.get_sector_for_symbol(symbol)
            if symbol_sector_info:
                symbol_sector = {
                    'sector_id': symbol_sector_info.sector_id,
                    'sector_name': symbol_sector_info.name_turkish,
                    'sentiment': sector_sentiments[symbol_sector_info.sector_id]['average_sentiment'],
                    'article_count': sector_sentiments[symbol_sector_info.sector_id]['article_count'],
                    'markets': self.get_market_for_symbol(symbol)
                }
        
        return {
            'overall_market_sentiment': overall_sentiment,
            'sector_breakdown': sector_sentiments,
            'symbol_sector': symbol_sector,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_articles_analyzed': len(news_articles)
        }
    
    def get_sector_keywords(self, sector_id: str) -> List[str]:
        """Get keywords for a specific sector"""
        return self.sectors.get(sector_id, SectorInfo('', '', '', [], [], 0)).keywords
    
    def get_all_sectors(self) -> Dict[str, SectorInfo]:
        """Get all sector information"""
        return self.sectors
    
    def get_all_markets(self) -> Dict[str, Dict[str, Any]]:
        """Get all BIST market information"""
        return self.markets


if __name__ == "__main__":
    # Test the sector analyzer
    analyzer = BISTSectorAnalyzer()
    
    print(f"\n🧪 TESTING SECTOR ANALYZER:")
    print(f"📊 GARAN sector: {analyzer.get_sector_for_symbol('GARAN')}")
    print(f"📈 GARAN markets: {analyzer.get_market_for_symbol('GARAN')}")
    
    # Mock articles for testing
    mock_articles = [
        {
            'title': 'Garanti Bankası kâr açıkladı',
            'content': 'Bankacılık sektörü güçlü performans',
            'sentiment': {'compound': 0.6}
        },
        {
            'title': 'Havayolu sektörü zorlanıyor',
            'content': 'THY passenger numbers declining',
            'sentiment': {'compound': -0.4}
        }
    ]
    
    result = analyzer.analyze_sector_sentiment(mock_articles, 'GARAN')
    print(f"🎯 Sentiment analysis result: {result['overall_market_sentiment']}")
    print(f"🏦 Banking sector sentiment: {result['sector_breakdown']['banking']['average_sentiment']}")
