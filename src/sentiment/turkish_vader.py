"""
Turkish VADER Sentiment Analysis Adapter
Optimized for Financial/Stock Market Sentiment Analysis
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging


class TurkishVaderAnalyzer:
    """
    Turkish adaptation of VADER Sentiment Analysis
    Specifically tuned for financial/stock market sentiment
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Turkish financial sentiment lexicon
        self.turkish_financial_lexicon = self._build_turkish_financial_lexicon()
        
        # Turkish linguistic features
        self.turkish_intensifiers = self._build_turkish_intensifiers()
        self.turkish_negations = self._build_turkish_negations()
        
        # Company name patterns for BIST
        self.bist_companies = self._build_bist_company_patterns()
        
        # Update VADER lexicon with Turkish terms
        self._update_vader_lexicon()
        
        print(f"🏢 BIST COMPANY RECOGNITION:")
        print(f"   📊 Total companies: {len(self.bist_companies)}")
        print(f"   🔍 Total name variants: {sum(len(variants) for variants in self.bist_companies.values())}")
        self.logger.info("Turkish VADER Analyzer initialized")
    
    def _build_turkish_financial_lexicon(self) -> Dict[str, float]:
        """Build Turkish financial sentiment dictionary"""
        return {
            # Positive financial terms
            'yükseliş': 2.5, 'artış': 2.0, 'büyüme': 2.2, 'kâr': 2.8, 'kazanç': 2.5,
            'gelir': 1.8, 'başarı': 2.3, 'güçlü': 2.0, 'sağlam': 1.9, 'istikrarlı': 1.7,
            'pozitif': 2.1, 'olumlu': 2.0, 'iyi': 1.5, 'mükemmel': 2.8, 'harika': 2.5,
            'yüksek': 1.8, 'artan': 1.9, 'çıkış': 1.6, 'rekor': 2.4, 'maksimum': 2.0,
            'patlama': 2.2, 'sıçrama': 2.3, 'yükselme': 2.1, 'tırmanış': 2.0,
            'genişleme': 1.8, 'iyileşme': 2.0, 'toparlanma': 2.1, 'canlanma': 2.2,
            'güven': 1.9, 'umut': 1.7, 'fırsat': 1.8, 'potansiyel': 1.6,
            'verimli': 1.9, 'karlı': 2.3, 'başarılı': 2.1, 'etkili': 1.8,
            
            # Negative financial terms
            'düşüş': -2.5, 'azalış': -2.0, 'küçülme': -2.2, 'zarar': -2.8, 'kayıp': -2.5,
            'gerileme': -2.1, 'başarısızlık': -2.3, 'zayıf': -2.0, 'kırılgan': -1.9,
            'istikrarsız': -2.2, 'negatif': -2.1, 'olumsuz': -2.0, 'kötü': -1.8,
            'berbat': -2.8, 'feci': -2.5, 'düşük': -1.8, 'azalan': -1.9, 'iniş': -1.6,
            'minimum': -2.0, 'çöküş': -2.8, 'düşme': -2.1, 'gerileme': -2.0,
            'daralma': -2.2, 'kötüleşme': -2.3, 'bozulma': -2.1, 'çözülme': -2.0,
            'endişe': -1.9, 'korku': -2.2, 'panik': -2.5, 'kriz': -2.8,
            'verimsiz': -1.9, 'zararlı': -2.3, 'başarısız': -2.1, 'etkisiz': -1.8,
            'tehlike': -2.4, 'risk': -1.8, 'belirsizlik': -1.7, 'sorun': -1.9,
            
            # Market specific terms
            'boğa': 2.5, 'ayı': -2.5, 'rallisi': 2.3, 'satış': -1.5, 'alım': 1.8,
            'hacim': 1.2, 'likidite': 1.5, 'volatilite': -1.2, 'manipülasyon': -2.5,
            'spekülasyon': -1.8, 'investisyon': 1.9, 'yatırım': 1.7, 'portföy': 1.2,
            'hisse': 1.0, 'borsa': 1.1, 'endeks': 1.0, 'piyasa': 0.8,
            
            # Intensifiers in Turkish context
            'çok': 0.8, 'oldukça': 0.6, 'son derece': 1.0, 'fevkalade': 1.2,
            'kesinlikle': 0.9, 'mutlaka': 0.7, 'gerçekten': 0.8, 'özellikle': 0.6,
            'büyük': 0.7, 'küçük': -0.5, 'hafif': -0.3, 'şiddetli': 0.9,
            
            # Banking specific terms
            'kredi': 0.5, 'faiz': -0.8, 'enflasyon': -1.5, 'deflasyon': -1.2,
            'devalüasyon': -2.0, 'revalüasyon': 1.8, 'dolar': 0.2, 'euro': 0.2,
            'merkez bankası': 0.5, 'politika': 0.3, 'karar': 0.8, 'açıklama': 0.5,
            
            # EXPANDED LEXICON - Additional 200+ Terms
            # Additional Positive Financial Terms
            'temettü': 2.4, 'bonus': 2.2, 'ödül': 2.0, 'ikramiye': 2.1, 'prim': 1.8,
            'faydalar': 1.9, 'avantaj': 2.0, 'üstünlük': 2.1, 'liderlik': 2.2, 'öncülük': 2.0,
            'yenilikçi': 2.1, 'innovatif': 2.2, 'gelişmiş': 1.9, 'modern': 1.8, 'teknolojik': 1.7,
            'dijital': 1.6, 'akıllı': 1.8, 'verimlilik': 2.0, 'optimizasyon': 1.9, 'etkinlik': 1.8,
            'sürdürülebilir': 2.0, 'dayanıklı': 1.9, 'güvenilir': 2.1, 'istikrar': 1.8, 'denge': 1.6,
            'hedef': 1.7, 'amaç': 1.6, 'vizyon': 1.9, 'misyon': 1.7, 'strateji': 1.8,
            'planlı': 1.6, 'organize': 1.7, 'sistemli': 1.8, 'düzenli': 1.7, 'kontrollü': 1.6,
            'sertifika': 1.8, 'lisans': 1.7, 'onay': 1.9, 'kabul': 1.8, 'approval': 1.9,
            'yatırımcı': 1.8, 'sponsor': 1.9, 'partner': 2.0, 'işbirliği': 2.1, 'ortaklık': 2.0,
            'birleşme': 1.9, 'devralma': 1.8, 'satın alma': 1.7, 'expansion': 1.8, 'büyütme': 1.9,
            'gelişim': 1.8, 'ilerleme': 1.9, 'progress': 1.8, 'development': 1.9, 'yenilik': 2.0,
            'keşif': 1.8, 'buluş': 1.9, 'patent': 2.0, 'marka': 1.7, 'brand': 1.7,
            'kalite': 2.0, 'quality': 2.0, 'excellence': 2.3, 'mükemmellik': 2.3, 'üstün': 2.1,
            'çok iyi': 2.0, 'outstanding': 2.4, 'superior': 2.2, 'premium': 2.1, 'luxury': 1.9,
            'profesyonel': 1.9, 'uzman': 2.0, 'expert': 2.0, 'deneyimli': 1.9, 'tecrübeli': 1.9,
            
            # Trading & Investment Positives
            'bull market': 2.6, 'boğa piyasası': 2.6, 'rising market': 2.3, 'yükseliş piyasası': 2.3,
            'momentum': 2.2, 'ivme': 2.2, 'trend': 1.8, 'trend yönü': 1.8, 'destek': 1.9,
            'breakout': 2.3, 'kırılım': 2.3, 'yukarı kırılım': 2.5, 'rally': 2.4,
            'recovery': 2.1, 'bounce': 2.0, 'uptrend': 2.4, 'yükseliş trendi': 2.4, 'upturn': 2.2,
            'oversold': 1.8, 'aşırı satım': 1.8, 'undervalued': 2.1, 'değer altında': 2.1,
            'bargain': 2.0, 'pazarlık': 2.0, 'opportunity': 1.8, 'chance': 1.7,
            'volume surge': 2.1, 'hacim artışı': 2.1, 'high volume': 1.8, 'yüksek hacim': 1.8,
            
            # Additional Negative Financial Terms
            'iflas': -2.9, 'bankruptcy': -2.9, 'konkordato': -2.7, 'restructuring': -2.2,
            'tasfiye': -2.8, 'liquidation': -2.8, 'kapanma': -2.6, 'closure': -2.6,
            'fire': -2.7, 'işten çıkarma': -2.4, 'layoffs': -2.4, 'downsizing': -2.3,
            'maliyet': -1.8, 'cost': -1.6, 'expense': -1.7, 'gider': -1.8, 'harcama': -1.6,
            'vergi': -1.5, 'tax': -1.5, 'taxation': -1.6, 'vergilendirme': -1.6,
            'ceza': -2.4, 'penalty': -2.4, 'fine': -2.3, 'para cezası': -2.3, 'sanctions': -2.6,
            'investigation': -2.2, 'soruşturma': -2.2, 'inquiry': -2.0, 'inceleme': -2.0,
            'fraud': -2.8, 'dolandırıcılık': -2.8, 'scam': -2.7, 'corruption': -2.9,
            'scandal': -2.6, 'skandal': -2.6, 'illegal': -2.5, 'yasadışı': -2.5,
            'violation': -2.4, 'ihlal': -2.4, 'breach': -2.3, 'non-compliance': -2.2,
            'default': -2.7, 'temerrüt': -2.7, 'delay': -2.0, 'gecikme': -2.0,
            'debt': -2.1, 'borç': -2.1, 'liability': -1.9, 'yükümlülük': -1.8,
            'crisis': -2.8, 'recession': -2.6, 'durgunluk': -2.6, 'depression': -2.9,
            
            # Trading & Investment Negatives  
            'bear market': -2.6, 'ayı piyasası': -2.6, 'falling market': -2.3,
            'correction': -2.1, 'düzeltme': -2.1, 'pullback': -2.0, 'geri çekilme': -2.0,
            'downtrend': -2.4, 'düşüş trendi': -2.4, 'downturn': -2.2, 'decline': -2.1,
            'sell-off': -2.3, 'satış dalgası': -2.3, 'panic selling': -2.7, 'panik satış': -2.7,
            'overbought': -1.8, 'aşırı alım': -1.8, 'overvalued': -2.1, 'değer üstünde': -2.1,
            'resistance': -1.7, 'direnç': -1.7, 'breakdown': -2.3, 'aşağı kırılım': -2.5,
            'low volume': -1.8, 'düşük hacim': -1.8, 'illiquid': -2.0, 'likit olmayan': -2.0,
            'volatile': -1.8, 'oynak': -1.8, 'unstable': -2.0, 'kararsız': -2.0,
            
            # Extended Market Terms
            'dividends': 2.4, 'yield': 2.1, 'getiri': 2.1, 'return': 2.0, 'profit margin': 2.3,
            'earnings': 2.2, 'revenue': 2.0, 'turnover': 1.8, 'ciro': 1.8, 'sales': 1.7,
            'assets': 1.6, 'varlıklar': 1.6, 'equity': 1.7, 'özsermaye': 1.7, 'capital': 1.8,
            'ipo': 2.2, 'halka arz': 2.2, 'listing': 2.0, 'kotasyon': 2.0,
            'merger': 1.9, 'acquisition': 1.8, 'takeover': 1.7, 'subsidiary': 1.5,
            'holding': 1.4, 'market cap': 1.7, 'piyasa değeri': 1.7, 'capitalization': 1.6,
            'institutional': 1.7, 'kurumsal': 1.7, 'retail': 1.5, 'bireysel': 1.5,
            'fund': 1.6, 'fon': 1.6, 'mutual fund': 1.7, 'yatırım fonu': 1.7, 'etf': 1.8,
            
            # Sector Specific Terms
            'automotive': 1.4, 'otomotiv': 1.4, 'manufacturing': 1.5, 'imalat': 1.5,
            'construction': 1.3, 'inşaat': 1.3, 'real estate': 1.5, 'gayrimenkul': 1.5,
            'retail': 1.2, 'perakende': 1.2, 'wholesale': 1.3, 'toptan': 1.3,
            'export': 1.9, 'ihracat': 1.9, 'import': -1.2, 'ithalat': -1.2,
            'tourism': 1.6, 'turizm': 1.6, 'hospitality': 1.5, 'otelcilik': 1.5,
            'healthcare': 1.7, 'sağlık': 1.7, 'pharmaceutical': 1.6, 'ilaç': 1.6,
            'mining': 1.2, 'madencilik': 1.2, 'steel': 1.3, 'çelik': 1.3,
            'chemicals': 1.1, 'kimya': 1.1, 'petrochemicals': 1.2, 'petrokimya': 1.2,
            'textiles': 1.1, 'tekstil': 1.1, 'agriculture': 1.4, 'tarım': 1.4,
            'technology': 1.8, 'teknoloji': 1.8, 'fintech': 1.9, 'blockchain': 1.7,
            
            # Economic Indicators
            'gdp': 1.6, 'gsyh': 1.6, 'growth rate': 2.0, 'büyüme oranı': 2.0,
            'unemployment': -2.1, 'işsizlik': -2.1, 'employment': 1.8, 'istihdam': 1.8,
            'inflation': -1.8, 'interest rates': -0.8, 'faiz oranları': -0.8,
            'monetary policy': 0.3, 'para politikası': 0.3, 'fiscal policy': 0.4,
            
            # Time/Trend Modifiers  
            'daily': 0.5, 'günlük': 0.5, 'weekly': 0.6, 'haftalık': 0.6, 'monthly': 0.7,
            'quarterly': 0.8, 'üç aylık': 0.8, 'annual': 0.9, 'yıllık': 0.9,
            'short term': 0.4, 'kısa vadeli': 0.4, 'medium term': 0.6, 'orta vadeli': 0.6,
            'long term': 0.8, 'uzun vadeli': 0.8, 'permanent': 0.7, 'kalıcı': 0.7,
            'temporary': -0.5, 'geçici': -0.5, 'seasonal': 0.2, 'mevsimsel': 0.2,
            
            # Quantitative Terms
            'percent': 0.3, 'yüzde': 0.3, 'million': 0.6, 'milyon': 0.6, 'billion': 0.8,
            'milyar': 0.8, 'double': 1.8, 'iki katı': 1.8, 'triple': 2.1, 'üç katı': 2.1,
            'half': -0.8, 'yarım': -0.8, 'quarter': -0.6, 'çeyrek': -0.6,
        }
    
    def _build_turkish_intensifiers(self) -> Dict[str, float]:
        """Turkish intensifier words"""
        return {
            'çok': 0.293, 'oldukça': 0.293, 'son derece': 0.457, 'fevkalade': 0.596,
            'kesinlikle': 0.293, 'mutlaka': 0.293, 'gerçekten': 0.293, 'özellikle': 0.293,
            'büyük ölçüde': 0.457, 'tamamen': 0.596, 'tümüyle': 0.596, 'bütünüyle': 0.596,
            'aşırı': 0.596, 'ultra': 0.596, 'süper': 0.596, 'mega': 0.596,
            'maksimum': 0.596, 'minimum': -0.596, 'az': -0.293, 'biraz': -0.293,
        }
    
    def _build_turkish_negations(self) -> List[str]:
        """Turkish negation words"""
        return [
            'değil', 'değildir', 'olmayan', 'olmamış', 'değilmiş', 'değilse',
            'hayır', 'yok', 'yoktur', 'yoxdur', 'asla', 'hiç', 'hiçbir',
            'ne', 'nerede', 'nasıl', 'niçin', 'niye', 'neden',
            'mümkün değil', 'imkansız', 'olamaz', 'olmaz',
            'karşı', 'karşıt', 'ters', 'zıt', 'aksi',
            'red', 'reddediyor', 'reddetti', 'ret',
        ]
    
    def _build_bist_company_patterns(self) -> Dict[str, List[str]]:
        """BIST company name patterns for entity recognition"""
        return {
            # Major BIST companies and their variants
            'AKBNK': ['akbank', 'ak bank', 'ak bankası', 'akbankası'],
            'GARAN': ['garanti', 'garanti bankası', 'tgb', 'türkiye garanti bankası'],
            'ISCTR': ['işbank', 'iş bankası', 'türkiye iş bankası', 'işbankası'],
            'YKBNK': ['yapı kredi', 'yapı kredi bankası', 'ykb', 'yapıkredi'],
            'HALKB': ['halkbank', 'halk bankası', 'türkiye halk bankası'],
            'VAKBN': ['vakıfbank', 'vakıf bankası', 'vakıf bank'],
            'SISE': ['şişe cam', 'şişecam', 'türkiye şişe ve cam'],
            'THYAO': ['thy', 'türk hava yolları', 'turkish airlines'],
            'BIMAS': ['bim', 'birleşik mağazalar'],
            'MGROS': ['migros', 'migros ticaret'],
            'CCOLA': ['coca cola', 'koka kola', 'coca-cola'],
            'PETKM': ['petkim', 'petrokimya'],
            'TUPRS': ['tüpraş', 'türkiye petrol rafinerileri'],
            'KRDMD': ['kardemir', 'karabük demir çelik'],
            'EREGL': ['erdemir', 'ereğli demir çelik'],
            'ARCLK': ['arçelik', 'koç arçelik'],
            'VESTL': ['vestel', 'vestel elektronik'],
            'TTKOM': ['türk telekom', 'türktelekom', 'tt'],
            'TCELL': ['turkcell', 'türk cell', 'turk cell', 'türkcell', 'türk hücre'],
            # Additional major BIST companies (NEW)
            'ASELS': ['aselsan', 'asel', 'savunma sanayi', 'military electronics', 'askeri'],
            'KCHOL': ['koç holding', 'koc holding', 'koç', 'koc', 'koç grup'],
            'SAHOL': ['sabancı holding', 'sabanci holding', 'sabancı', 'sabanci'],
            'DOHOL': ['doğan holding', 'dogan holding', 'doğan', 'dogan', 'doğan grup'],
            'TOASO': ['tofaş', 'tofas', 'türk otomobil', 'fiat tofaş'],
            'FROTO': ['ford otosan', 'ford oto', 'ford otomotiv', 'ford turkey'],
            'OTKAR': ['otokar', 'oto kar', 'otobüs', 'otobus'],
            'ULKER': ['ülker', 'ulker', 'ülker bisküvi', 'ulker biscuit'],
            'TKFEN': ['tekfen', 'tekfen holding', 'tekfen construction'],
            'ENKAI': ['enka', 'enka insaat', 'enka construction'],
            'GSDHO': ['gsd holding', 'galatasaray', 'gs holding'],
            'BJKAS': ['beşiktaş', 'besiktas', 'bjk', 'kartal'],
            'FENER': ['fenerbahçe', 'fenerbahce', 'fb', 'fener'],
            'PGSUS': ['pegasus', 'pgs', 'pegasus airlines', 'pgs hava'],
            'SODA': ['şişe cam soda', 'soda sanayii', 'soda ash'],
            'TRKCM': ['türk cam', 'turk cam', 'türkcam', 'cam sanayi'],
            'DEVA': ['deva holding', 'deva ilaç', 'deva pharmaceutical'],
            'ECZYT': ['eczacıbaşı', 'eczacibasi', 'ecza', 'pharmaceutical'],
            'LOGO': ['logo yazılım', 'logo software', 'logo bilgisayar'],
            'NETAS': ['netaş', 'netas', 'telekomünikasyon', 'telecom equipment'],
            'BRISA': ['brisa', 'sabancı lastik', 'bridgestone sabanci'],
            'TIRE': ['petlas tire', 'petlas', 'lastik', 'tire'],
            'GUBRF': ['gübre fabrikaları', 'gubrf', 'fertilizer'],
            'BAGFS': ['bagfaş', 'bag fash', 'çanta'],
            'BFREN': ['bosch fren', 'fren sistemleri', 'brake systems']
        }
    
    def _update_vader_lexicon(self):
        """Update VADER's lexicon with Turkish financial terms"""
        for word, score in self.turkish_financial_lexicon.items():
            self.analyzer.lexicon[word.lower()] = score
            
        print(f"📊 TURKISH FINANCIAL LEXICON STATS:")
        pos_terms = len([v for v in self.turkish_financial_lexicon.values() if v > 0.5])
        neg_terms = len([v for v in self.turkish_financial_lexicon.values() if v < -0.5])
        neutral_terms = len([v for v in self.turkish_financial_lexicon.values() if -0.5 <= v <= 0.5])
        print(f"   ✅ TOTAL: {len(self.turkish_financial_lexicon)} terms")
        print(f"   📈 POSITIVE: {pos_terms} terms")
        print(f"   📉 NEGATIVE: {neg_terms} terms") 
        print(f"   ⚖️  NEUTRAL: {neutral_terms} terms")
        self.logger.info(f"Updated VADER lexicon with {len(self.turkish_financial_lexicon)} Turkish terms")
    
    def preprocess_turkish_text(self, text: str) -> str:
        """Preprocess Turkish text for better sentiment analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Turkish character normalization
        turkish_chars = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            # Keep original Turkish chars too for lexicon matching
        }
        
        # Clean extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle Turkish punctuation
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions
        
        return text
    
    def extract_company_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract BIST company mentions from text"""
        entities = []
        text_lower = text.lower()
        
        for symbol, variants in self.bist_companies.items():
            for variant in variants:
                if variant in text_lower:
                    # Find actual position for context
                    start_pos = text_lower.find(variant)
                    entities.append((symbol, variant))
        
        return list(set(entities))  # Remove duplicates
    
    def analyze_sentiment(self, text: str, consider_entities: bool = True) -> Dict[str, Any]:
        """
        Analyze sentiment of Turkish text with financial context
        
        Args:
            text: Input text to analyze
            consider_entities: Whether to boost/modify sentiment based on entity mentions
            
        Returns:
            Dictionary with sentiment scores and metadata
        """
        if not text or not text.strip():
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
                'confidence': 0.0,
                'entities': [],
                'processed_text': '',
                'language': 'tr'
            }
        
        # Preprocess text
        processed_text = self.preprocess_turkish_text(text)
        
        # Extract company entities
        entities = []
        if consider_entities:
            entities = self.extract_company_entities(text)
        
        # Get base VADER scores
        vader_scores = self.analyzer.polarity_scores(processed_text)
        
        # Adjust scores based on Turkish linguistic features
        adjusted_scores = self._adjust_for_turkish_features(
            processed_text, vader_scores, entities
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(adjusted_scores, len(entities))
        
        return {
            'compound': adjusted_scores['compound'],
            'pos': adjusted_scores['pos'],
            'neu': adjusted_scores['neu'],
            'neg': adjusted_scores['neg'],
            'confidence': confidence,
            'entities': entities,
            'processed_text': processed_text,
            'language': 'tr'
        }
    
    def _adjust_for_turkish_features(self, text: str, scores: Dict[str, float], 
                                   entities: List[Tuple[str, str]]) -> Dict[str, float]:
        """Adjust sentiment scores for Turkish linguistic features"""
        
        adjusted = scores.copy()
        
        # Boost sentiment if financial entities are mentioned
        if entities:
            entity_boost = min(0.1 * len(entities), 0.3)  # Max 30% boost
            
            if adjusted['compound'] > 0:
                adjusted['compound'] = min(1.0, adjusted['compound'] + entity_boost)
                adjusted['pos'] = min(1.0, adjusted['pos'] + entity_boost * 0.5)
            elif adjusted['compound'] < 0:
                adjusted['compound'] = max(-1.0, adjusted['compound'] - entity_boost)
                adjusted['neg'] = min(1.0, adjusted['neg'] + entity_boost * 0.5)
        
        # Turkish intensifier patterns
        intensifier_pattern = r'\b(çok|son derece|oldukça|kesinlikle)\s+(\w+)'
        intensifiers = re.findall(intensifier_pattern, text)
        
        if intensifiers:
            intensifier_boost = len(intensifiers) * 0.05  # Small boost per intensifier
            if adjusted['compound'] > 0:
                adjusted['compound'] = min(1.0, adjusted['compound'] + intensifier_boost)
            elif adjusted['compound'] < 0:
                adjusted['compound'] = max(-1.0, adjusted['compound'] - intensifier_boost)
        
        # Turkish negation handling - more sophisticated
        negation_words = self.turkish_negations
        for neg_word in negation_words:
            if neg_word in text:
                # Flip sentiment if strong negation detected
                if abs(adjusted['compound']) > 0.1:  # Only if there's clear sentiment
                    adjusted['compound'] *= -0.8  # Partial reversal, not complete
                    # Swap pos/neg scores
                    adjusted['pos'], adjusted['neg'] = adjusted['neg'] * 0.8, adjusted['pos'] * 0.8
                break
        
        # Ensure scores are normalized
        total = adjusted['pos'] + adjusted['neu'] + adjusted['neg']
        if total > 0:
            adjusted['pos'] /= total
            adjusted['neu'] /= total
            adjusted['neg'] /= total
        
        return adjusted
    
    def _calculate_confidence(self, scores: Dict[str, float], entity_count: int) -> float:
        """Calculate confidence score for the sentiment analysis"""
        
        # Base confidence from compound score strength
        base_confidence = abs(scores['compound'])
        
        # Boost confidence if entities are mentioned (more context)
        entity_boost = min(entity_count * 0.1, 0.2)
        
        # Boost confidence if sentiment is clear (not neutral)
        clarity_boost = 0.0
        if scores['pos'] > 0.6 or scores['neg'] > 0.6:
            clarity_boost = 0.1
        elif scores['neu'] > 0.8:  # Very neutral
            base_confidence *= 0.7
        
        confidence = min(1.0, base_confidence + entity_boost + clarity_boost)
        return round(confidence, 3)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts"""
        results = []
        
        for text in texts:
            try:
                result = self.analyze_sentiment(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing text: {str(e)}")
                results.append({
                    'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0,
                    'confidence': 0.0, 'entities': [], 'error': str(e)
                })
        
        return results


def test_turkish_vader():
    """Test function for Turkish VADER analyzer"""
    
    analyzer = TurkishVaderAnalyzer()
    
    test_sentences = [
        "Akbank hisseleri bugün %5 yükseldi, çok güçlü performans!",
        "Garanti Bankası'nın kârları düştü, yatırımcılar endişeli",
        "BIST 100 endeksi rekor kırdı, boğa piyasası devam ediyor",
        "Türk Telekom'un hisseleri çok kötü performans gösteriyor, büyük zarar",
        "Piyasalar pozitif, tüm bankacılık hisseleri yükselişte",
        "Ekonomik belirsizlik artıyor, yatırımcılar temkinli",
        "THY hisselerinde muhteşem artış, rekor kırıyor!",
        "Bim'in satışları çok iyi, müthiş büyüme",
        "Kredi faizleri yükseldi, bankalar için olumsuz haber değil"
    ]
    
    print("🧠 Turkish VADER Sentiment Analysis Test")
    print("=" * 60)
    
    for i, sentence in enumerate(test_sentences, 1):
        result = analyzer.analyze_sentiment(sentence)
        
        print(f"\n{i}. Text: {sentence}")
        print(f"   Compound: {result['compound']:.3f}")
        print(f"   Pos: {result['pos']:.3f}, Neu: {result['neu']:.3f}, Neg: {result['neg']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        if result['entities']:
            print(f"   Entities: {result['entities']}")
    
    print("\n✅ Turkish VADER test completed!")


if __name__ == "__main__":
    test_turkish_vader()
