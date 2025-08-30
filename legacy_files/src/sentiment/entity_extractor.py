"""
BIST Company Entity Extraction
Advanced Named Entity Recognition for Turkish Financial News
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging


@dataclass
class EntityMatch:
    """Represents a matched entity in text"""
    symbol: str
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str  # Surrounding text


class BISTEntityExtractor:
    """
    Advanced entity extractor for BIST companies
    Extracts company mentions from Turkish financial news
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Build comprehensive BIST company database
        self.company_database = self._build_company_database()
        
        # Build search patterns
        self._build_search_patterns()
        
        # Context patterns that indicate financial relevance
        self.financial_contexts = self._build_financial_contexts()
        
        self.logger.info(f"BIST Entity Extractor initialized with {len(self.company_database)} companies")
    
    def _build_company_database(self) -> Dict[str, Dict[str, any]]:
        """Build comprehensive BIST company database"""
        return {
            # Banks
            'AKBNK': {
                'full_name': 'Akbank T.A.Ş.',
                'variants': ['akbank', 'ak bank', 'ak bankası', 'akbankası', 'akbank taş'],
                'sector': 'Banking',
                'aliases': ['ak', 'akb'],
                'market_cap_tier': 'large'
            },
            
            'GARAN': {
                'full_name': 'Türkiye Garanti Bankası A.Ş.',
                'variants': ['garanti', 'garanti bankası', 'tgb', 'türkiye garanti bankası', 'garanti bank'],
                'sector': 'Banking',
                'aliases': ['garan', 'garanti', 'tgb'],
                'market_cap_tier': 'large'
            },
            
            'ISCTR': {
                'full_name': 'Türkiye İş Bankası A.Ş.',
                'variants': ['işbank', 'iş bankası', 'türkiye iş bankası', 'işbankası', 'iş bank'],
                'sector': 'Banking',
                'aliases': ['işbank', 'işb', 'iş'],
                'market_cap_tier': 'large'
            },
            
            'YKBNK': {
                'full_name': 'Yapı ve Kredi Bankası A.Ş.',
                'variants': ['yapı kredi', 'yapı kredi bankası', 'ykb', 'yapıkredi', 'yapı ve kredi'],
                'sector': 'Banking',
                'aliases': ['ykb', 'yapı kredi'],
                'market_cap_tier': 'large'
            },
            
            'HALKB': {
                'full_name': 'Türkiye Halk Bankası A.Ş.',
                'variants': ['halkbank', 'halk bankası', 'türkiye halk bankası', 'halk bank'],
                'sector': 'Banking',
                'aliases': ['halk', 'halkb'],
                'market_cap_tier': 'large'
            },
            
            'VAKBN': {
                'full_name': 'Türkiye Vakıflar Bankası T.A.O.',
                'variants': ['vakıfbank', 'vakıf bankası', 'vakıf bank', 'vakıflar bankası'],
                'sector': 'Banking',
                'aliases': ['vakıf', 'vakbn'],
                'market_cap_tier': 'large'
            },
            
            # Industrial Companies
            'SISE': {
                'full_name': 'Türkiye Şişe ve Cam Fabrikaları A.Ş.',
                'variants': ['şişe cam', 'şişecam', 'türkiye şişe ve cam', 'şişe ve cam'],
                'sector': 'Glass & Ceramics',
                'aliases': ['şişecam', 'sise'],
                'market_cap_tier': 'large'
            },
            
            'THYAO': {
                'full_name': 'Türk Hava Yolları A.O.',
                'variants': ['thy', 'türk hava yolları', 'turkish airlines', 'hava yolları'],
                'sector': 'Airlines',
                'aliases': ['thy', 'türk hava yolları'],
                'market_cap_tier': 'large'
            },
            
            # Retail
            'BIMAS': {
                'full_name': 'BİM Birleşik Mağazalar A.Ş.',
                'variants': ['bim', 'birleşik mağazalar', 'bim mağazaları', 'bim market'],
                'sector': 'Retail',
                'aliases': ['bim'],
                'market_cap_tier': 'large'
            },
            
            'MGROS': {
                'full_name': 'Migros Ticaret A.Ş.',
                'variants': ['migros', 'migros ticaret', 'migros market'],
                'sector': 'Retail',
                'aliases': ['migros'],
                'market_cap_tier': 'medium'
            },
            
            # Telecom
            'TTKOM': {
                'full_name': 'Türk Telekomünikasyon A.Ş.',
                'variants': ['türk telekom', 'türktelekom', 'tt', 'telekomünikasyon'],
                'sector': 'Telecommunications',
                'aliases': ['tt', 'türk telekom'],
                'market_cap_tier': 'large'
            },
            
            'TCELL': {
                'full_name': 'Turkcell İletişim Hizmetleri A.Ş.',
                'variants': ['turkcell', 'türk cell', 'turk cell', 'türkcell'],
                'sector': 'Telecommunications',
                'aliases': ['turkcell'],
                'market_cap_tier': 'large'
            },
            
            # Energy & Chemicals
            'TUPRS': {
                'full_name': 'Türkiye Petrol Rafinerileri A.Ş.',
                'variants': ['tüpraş', 'türkiye petrol rafinerileri', 'petrol rafinerileri'],
                'sector': 'Oil Refinery',
                'aliases': ['tüpraş'],
                'market_cap_tier': 'large'
            },
            
            'PETKM': {
                'full_name': 'Petkim Petrokimya Holding A.Ş.',
                'variants': ['petkim', 'petrokimya', 'petkim petrokimya'],
                'sector': 'Petrochemicals',
                'aliases': ['petkim'],
                'market_cap_tier': 'medium'
            },
            
            # Steel & Mining
            'EREGL': {
                'full_name': 'Ereğli Demir ve Çelik Fabrikaları T.A.Ş.',
                'variants': ['erdemir', 'ereğli demir çelik', 'demir çelik', 'ereğli'],
                'sector': 'Steel',
                'aliases': ['erdemir'],
                'market_cap_tier': 'large'
            },
            
            'KRDMD': {
                'full_name': 'Kardemir Karabük Demir Çelik Sanayi ve Ticaret A.Ş.',
                'variants': ['kardemir', 'karabük demir çelik', 'karabük demir'],
                'sector': 'Steel',
                'aliases': ['kardemir'],
                'market_cap_tier': 'medium'
            },
            
            # Consumer Goods
            'ARCLK': {
                'full_name': 'Arçelik A.Ş.',
                'variants': ['arçelik', 'koç arçelik', 'arcelik'],
                'sector': 'Consumer Electronics',
                'aliases': ['arçelik'],
                'market_cap_tier': 'large'
            },
            
            'VESTL': {
                'full_name': 'Vestel Elektronik Sanayi ve Ticaret A.Ş.',
                'variants': ['vestel', 'vestel elektronik', 'vestel beyaz eşya'],
                'sector': 'Consumer Electronics',
                'aliases': ['vestel'],
                'market_cap_tier': 'medium'
            },
            
            # Beverages
            'CCOLA': {
                'full_name': 'Coca-Cola İçecek A.Ş.',
                'variants': ['coca cola', 'koka kola', 'coca-cola', 'cola'],
                'sector': 'Beverages',
                'aliases': ['coca cola', 'kola'],
                'market_cap_tier': 'large'
            },
        }
    
    def _build_search_patterns(self):
        """Build optimized regex patterns for company matching"""
        self.patterns = {}
        self.reverse_lookup = {}  # text -> symbol mapping
        
        for symbol, info in self.company_database.items():
            patterns = []
            
            # Add symbol itself
            patterns.append(symbol.lower())
            self.reverse_lookup[symbol.lower()] = symbol
            
            # Add variants
            for variant in info['variants']:
                patterns.append(variant.lower())
                self.reverse_lookup[variant.lower()] = symbol
            
            # Add aliases
            for alias in info.get('aliases', []):
                patterns.append(alias.lower())
                self.reverse_lookup[alias.lower()] = symbol
            
            # Build combined regex pattern for this company
            # Sort by length (descending) to match longer patterns first
            patterns.sort(key=len, reverse=True)
            
            # Escape special regex characters and create word boundaries
            escaped_patterns = []
            for pattern in patterns:
                escaped = re.escape(pattern)
                # Add word boundaries for better matching
                word_boundary_pattern = r'\b' + escaped + r'\b'
                escaped_patterns.append(word_boundary_pattern)
            
            combined_pattern = '|'.join(escaped_patterns)
            self.patterns[symbol] = re.compile(combined_pattern, re.IGNORECASE | re.UNICODE)
    
    def _build_financial_contexts(self) -> List[str]:
        """Build patterns that indicate financial relevance"""
        return [
            # Price movements
            'yükseldi', 'düştü', 'arttı', 'azaldı', 'çıktı', 'indi',
            'rekor', 'zirvede', 'dip', 'tavan', 'taban',
            
            # Financial terms
            'hisse', 'hisseleri', 'borsa', 'endeks', 'piyasa',
            'kâr', 'zarar', 'gelir', 'ciro', 'satış',
            'yatırım', 'yatırımcı', 'portföy',
            
            # Percentages and numbers
            '%', 'yüzde', 'puan', 'lira', 'tl', 'dolar', 'euro',
            'milyon', 'milyar', 'trilyon',
            
            # Actions
            'açıkladı', 'duyurdu', 'bildirdi', 'karar', 'plan',
            'strategi', 'hedef', 'beklenti', 'öngörü',
            
            # Market sentiment
            'pozitif', 'negatif', 'olumlu', 'olumsuz',
            'güven', 'endişe', 'beklenti', 'umut'
        ]
    
    def extract_entities(self, text: str, min_confidence: float = 0.5) -> List[EntityMatch]:
        """
        Extract BIST company entities from text
        
        Args:
            text: Input text to analyze
            min_confidence: Minimum confidence threshold for matches
            
        Returns:
            List of EntityMatch objects
        """
        if not text or not text.strip():
            return []
        
        text_lower = text.lower()
        entities = []
        
        # Track found positions to avoid overlaps
        used_positions = set()
        
        for symbol, pattern in self.patterns.items():
            matches = pattern.finditer(text_lower)
            
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                matched_text = text[start_pos:end_pos]  # Preserve original case
                
                # Check for overlaps
                if any(pos in used_positions for pos in range(start_pos, end_pos)):
                    continue
                
                # Calculate confidence
                confidence = self._calculate_confidence(text, matched_text, start_pos, end_pos, symbol)
                
                if confidence >= min_confidence:
                    # Get context
                    context = self._extract_context(text, start_pos, end_pos)
                    
                    entity = EntityMatch(
                        symbol=symbol,
                        matched_text=matched_text,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        context=context
                    )
                    
                    entities.append(entity)
                    
                    # Mark positions as used
                    for pos in range(start_pos, end_pos):
                        used_positions.add(pos)
        
        # Sort by confidence (descending) and position
        entities.sort(key=lambda x: (-x.confidence, x.start_pos))
        
        return entities
    
    def _calculate_confidence(self, text: str, matched_text: str, start_pos: int, 
                            end_pos: int, symbol: str) -> float:
        """Calculate confidence score for entity match"""
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on match characteristics
        company_info = self.company_database[symbol]
        
        # 1. Exact symbol match gets highest boost
        if matched_text.upper() == symbol:
            confidence += 0.4
        
        # 2. Full company name match
        elif matched_text.lower() in [v.lower() for v in company_info['variants'][:2]]:  # First 2 are usually official names
            confidence += 0.3
        
        # 3. Common variant match
        elif matched_text.lower() in [v.lower() for v in company_info['variants']]:
            confidence += 0.2
        
        # 4. Financial context boost
        context_window = text[max(0, start_pos-100):min(len(text), end_pos+100)].lower()
        financial_terms_found = sum(1 for term in self.financial_contexts if term in context_window)
        
        if financial_terms_found > 0:
            context_boost = min(0.2, financial_terms_found * 0.05)  # Max 20% boost
            confidence += context_boost
        
        # 5. Market cap tier boost (larger companies more likely to be mentioned)
        tier = company_info.get('market_cap_tier', 'medium')
        if tier == 'large':
            confidence += 0.1
        elif tier == 'medium':
            confidence += 0.05
        
        # 6. Position context (beginning/end of sentences get slight penalty)
        if start_pos < 10 or end_pos > len(text) - 10:
            confidence -= 0.05
        
        # 7. Length bonus (longer matches are usually more specific)
        if len(matched_text) > 10:
            confidence += 0.1
        elif len(matched_text) < 4:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, window_size: int = 50) -> str:
        """Extract surrounding context for the match"""
        
        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)
        
        context = text[context_start:context_end]
        
        # Mark the matched entity in context
        relative_start = start_pos - context_start
        relative_end = end_pos - context_start
        
        context_with_marker = (
            context[:relative_start] + 
            '**' + context[relative_start:relative_end] + '**' + 
            context[relative_end:]
        )
        
        return context_with_marker.strip()
    
    def extract_with_sentiment_context(self, text: str, sentiment_result: Dict) -> Dict:
        """
        Extract entities and enhance with sentiment context
        
        Args:
            text: Input text
            sentiment_result: Result from TurkishVaderAnalyzer
            
        Returns:
            Enhanced result with entity-sentiment mapping
        """
        
        entities = self.extract_entities(text)
        
        # Group entities by symbol
        entity_groups = {}
        for entity in entities:
            if entity.symbol not in entity_groups:
                entity_groups[entity.symbol] = []
            entity_groups[entity.symbol].append(entity)
        
        # Create entity-sentiment mapping
        entity_sentiments = {}
        for symbol, entity_list in entity_groups.items():
            # Use highest confidence match for this symbol
            best_entity = max(entity_list, key=lambda x: x.confidence)
            
            entity_sentiments[symbol] = {
                'symbol': symbol,
                'company_name': self.company_database[symbol]['full_name'],
                'mentions': len(entity_list),
                'best_match': {
                    'text': best_entity.matched_text,
                    'confidence': best_entity.confidence,
                    'context': best_entity.context
                },
                'sector': self.company_database[symbol]['sector'],
                'market_cap_tier': self.company_database[symbol]['market_cap_tier']
            }
        
        return {
            'entities': entities,
            'entity_sentiments': entity_sentiments,
            'overall_sentiment': sentiment_result,
            'entity_count': len(entity_groups)
        }


def test_entity_extractor():
    """Test function for BIST entity extractor"""
    
    extractor = BISTEntityExtractor()
    
    test_texts = [
        "Akbank hisseleri bugün %5 yükseldi, güçlü performans gösterdi",
        "Garanti Bankası ve İşbank'ın kârları arttı, BIST 100'de pozitif hava",
        "THY'nin satışları rekor kırdı, yatırımcılar memnun",
        "Türk Telekom ve Turkcell arasındaki rekabet kızışıyor",
        "BİM ve Migros mağaza sayısını artırıyor, perakende sektörü güçleniyor",
        "Tüpraş ve Petkim'in üretim kapasitesi artırılacak",
        "Arçelik ve Vestel'in ihracat hedefleri büyük",
        "EREGL ve Kardemir çelik fiyatlarındaki artıştan faydalanıyor"
    ]
    
    print("🔍 BIST Company Entity Extraction Test")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        
        entities = extractor.extract_entities(text)
        
        if entities:
            print(f"   Found {len(entities)} entities:")
            for entity in entities:
                print(f"     • {entity.symbol} ({entity.matched_text}) - Confidence: {entity.confidence:.3f}")
                print(f"       Context: {entity.context[:80]}...")
        else:
            print("   No entities found")
    
    print("\n✅ Entity extraction test completed!")


if __name__ == "__main__":
    test_entity_extractor()
