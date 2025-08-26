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
                'variants': ['akbank', 'ak bank', 'ak bankası', 'akbankası'],
                'sector': 'Banking',
                'aliases': ['ak'],
                'market_cap_tier': 'large'
            },
            
            'GARAN': {
                'full_name': 'Türkiye Garanti Bankası A.Ş.',
                'variants': ['garanti', 'garanti bankası', 'türkiye garanti bankası'],
                'sector': 'Banking',
                'aliases': ['garan'],
                'market_cap_tier': 'large'
            },
            
            'ISCTR': {
                'full_name': 'Türkiye İş Bankası A.Ş.',
                'variants': ['işbank', 'iş bankası', 'türkiye iş bankası'],
                'sector': 'Banking',
                'aliases': ['iş'],
                'market_cap_tier': 'large'
            },
            
            'THYAO': {
                'full_name': 'Türk Hava Yolları A.O.',
                'variants': ['thy', 'türk hava yolları', 'turkish airlines'],
                'sector': 'Airlines',
                'aliases': ['thy'],
                'market_cap_tier': 'large'
            },
            
            'BIMAS': {
                'full_name': 'BİM Birleşik Mağazalar A.Ş.',
                'variants': ['bim', 'birleşik mağazalar', 'bim market'],
                'sector': 'Retail',
                'aliases': ['bim'],
                'market_cap_tier': 'large'
            },
            
            'TTKOM': {
                'full_name': 'Türk Telekomünikasyon A.Ş.',
                'variants': ['türk telekom', 'türktelekom'],
                'sector': 'Telecommunications',
                'aliases': ['tt'],
                'market_cap_tier': 'large'
            },
            
            'TCELL': {
                'full_name': 'Turkcell İletişim Hizmetleri A.Ş.',
                'variants': ['turkcell', 'türkcell'],
                'sector': 'Telecommunications',
                'aliases': ['turkcell'],
                'market_cap_tier': 'large'
            },
        }
    
    def _build_search_patterns(self):
        """Build optimized regex patterns for company matching"""
        self.patterns = {}
        
        for symbol, info in self.company_database.items():
            patterns = []
            
            # Add variants
            for variant in info['variants']:
                patterns.append(variant.lower())
            
            # Add aliases
            for alias in info.get('aliases', []):
                patterns.append(alias.lower())
            
            # Sort by length (descending) to match longer patterns first
            patterns.sort(key=len, reverse=True)
            
            # Create regex pattern
            escaped_patterns = [re.escape(p) for p in patterns]
            combined_pattern = r'\b(' + '|'.join(escaped_patterns) + r')\b'
            self.patterns[symbol] = re.compile(combined_pattern, re.IGNORECASE)
    
    def _build_financial_contexts(self) -> List[str]:
        """Build patterns that indicate financial relevance"""
        return [
            'yükseldi', 'düştü', 'arttı', 'azaldı', 'rekor', 'hisse', 'hisseleri',
            'borsa', 'endeks', 'piyasa', 'kâr', 'zarar', 'gelir', 'yatırım',
            'pozitif', 'negatif', 'olumlu', 'olumsuz', '%', 'yüzde', 'puan'
        ]
    
    def extract_entities(self, text: str, min_confidence: float = 0.5) -> List[EntityMatch]:
        """Extract BIST company entities from text"""
        if not text or not text.strip():
            return []
        
        entities = []
        text_lower = text.lower()
        
        for symbol, pattern in self.patterns.items():
            matches = pattern.finditer(text_lower)
            
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                matched_text = text[start_pos:end_pos]  # Preserve original case
                
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
        
        # Remove overlapping matches (keep highest confidence)
        entities = self._remove_overlaps(entities)
        
        # Sort by confidence and position
        entities.sort(key=lambda x: (-x.confidence, x.start_pos))
        
        return entities
    
    def _calculate_confidence(self, text: str, matched_text: str, start_pos: int, 
                            end_pos: int, symbol: str) -> float:
        """Calculate confidence score for entity match"""
        
        confidence = 0.6  # Base confidence
        
        # Financial context boost
        context_window = text[max(0, start_pos-50):min(len(text), end_pos+50)].lower()
        financial_terms_found = sum(1 for term in self.financial_contexts if term in context_window)
        
        if financial_terms_found > 0:
            confidence += min(0.3, financial_terms_found * 0.1)
        
        # Length bonus
        if len(matched_text) > 5:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, window_size: int = 40) -> str:
        """Extract surrounding context for the match"""
        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)
        return text[context_start:context_end].strip()
    
    def _remove_overlaps(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """Remove overlapping entity matches, keeping highest confidence"""
        if not entities:
            return []
        
        # Sort by confidence descending
        sorted_entities = sorted(entities, key=lambda x: -x.confidence)
        
        result = []
        used_positions = set()
        
        for entity in sorted_entities:
            # Check if this entity overlaps with any already selected
            entity_positions = set(range(entity.start_pos, entity.end_pos))
            
            if not entity_positions.intersection(used_positions):
                result.append(entity)
                used_positions.update(entity_positions)
        
        return result


def test_entity_extractor():
    """Test function for BIST entity extractor"""
    
    extractor = BISTEntityExtractor()
    
    test_texts = [
        "Akbank hisseleri bugün %5 yükseldi, güçlü performans gösterdi",
        "Garanti Bankası'nın kârları arttı, pozitif hava var",
        "THY'nin satışları rekor kırdı, yatırımcılar memnun",
        "Türk Telekom ve Turkcell rekabeti kızışıyor",
        "BİM mağaza sayısını artırıyor, perakende güçleniyor",
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
        else:
            print("   No entities found")
    
    print("\n✅ Entity extraction test completed!")


if __name__ == "__main__":
    test_entity_extractor()
