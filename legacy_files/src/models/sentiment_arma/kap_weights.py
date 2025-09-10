"""
KAP Weight Calculator

This module calculates impact weights for KAP (Kamu Aydınlatma Platformu) announcements
based on announcement type, content analysis, and market impact patterns.

KAP Impact Categories:
- High Impact (Weight: 2.0-3.0): Major corporate actions, M&A, significant events
- Medium Impact (Weight: 1.0-2.0): Financial reports, management changes
- Low Impact (Weight: 0.1-1.0): Routine disclosures, administrative announcements

Weight Formula: Wt = base_weight × time_decay × market_condition_factor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from enum import Enum

class KAPAnnouncementType(Enum):
    """KAP announcement type enumeration"""
    ODA_GENEL = "ÖDA"  # Özel Durum Açıklaması (Genel)
    ODA_BIRLESME = "ÖDA_BIRLEŞME"  # Birleşme/Satın alma
    ODA_TEMETTÜ = "ÖDA_TEMETTÜ"  # Temettü dağıtımı
    ODA_BONUS = "ÖDA_BONUS"  # Bonus pay dağıtımı
    ODA_DEVRE_KESICI = "ÖDA_DEVRE_KESICI"  # Devre kesici
    FR_FINANSAL = "FR"  # Finansal Rapor
    FR_FAALIYET = "FR_FAALIYET"  # Faaliyet Raporu
    DG_GENEL = "DG"  # Diğer Genel
    HISSE_GERI_ALIM = "HISSE_GERI_ALIM"  # Pay geri alımı
    SPK_KARAR = "SPK_KARAR"  # SPK kararları
    YONETIM_KURULU = "YONETIM_KURULU"  # Yönetim kurulu değişiklikleri
    ORTAKLIK_DEGISIM = "ORTAKLIK_DEĞIŞIM"  # Ortaklık yapısı değişimi
    TAHVIL_IHRA = "TAHVIL_İHRAÇ"  # Tahvil/Kira sertifikası ihracı

@dataclass
class KAPAnnouncement:
    """KAP announcement data structure"""
    timestamp: pd.Timestamp
    symbol: str
    announcement_type: str
    title: str
    content: str
    source_url: Optional[str] = None
    impact_weight: Optional[float] = None
    confidence: float = 1.0

class KAPWeightCalculator:
    """
    Calculates impact weights for KAP announcements based on:
    1. Announcement type and severity
    2. Content analysis and keywords
    3. Historical market impact patterns  
    4. Time decay and market conditions
    """
    
    def __init__(self):
        """Initialize KAP weight calculator"""
        self.base_weights = self._initialize_base_weights()
        self.keyword_weights = self._initialize_keyword_weights()
        self.decay_params = {
            'half_life_hours': 24,  # 24-hour half-life for impact decay
            'max_impact_hours': 72  # 72-hour maximum impact window
        }
        
    def _initialize_base_weights(self) -> Dict[str, float]:
        """Initialize base impact weights by announcement type"""
        return {
            # High Impact (2.0-3.0)
            'ÖDA_BIRLEŞME': 3.0,
            'ÖDA_DEVRE_KESICI': 2.8,
            'ÖDA_TEMETTÜ': 2.5,
            'ÖDA_BONUS': 2.3,
            'HISSE_GERI_ALIM': 2.2,
            'SPK_KARAR': 2.0,
            
            # Medium Impact (1.0-2.0)
            'ÖDA': 1.8,  # Genel özel durum
            'FR': 1.5,  # Finansal rapor
            'FR_FAALIYET': 1.3,
            'YONETIM_KURULU': 1.2,
            'ORTAKLIK_DEĞIŞIM': 1.1,
            'TAHVIL_İHRAÇ': 1.0,
            
            # Low Impact (0.1-1.0)
            'DG': 0.8,  # Diğer genel
            'GENEL_BILGI': 0.5,
            'RUTINE': 0.3,
            'IDARI': 0.2,
            'DEFAULT': 0.5  # Default weight
        }
    
    def _initialize_keyword_weights(self) -> Dict[str, float]:
        """Initialize keyword-based weight multipliers"""
        return {
            # Very High Impact Keywords (×2.0)
            'birleşme': 2.0,
            'satın alma': 2.0, 
            'devralma': 2.0,
            'devre kesici': 2.0,
            'işlem durdurma': 2.0,
            'iflas': 2.0,
            'konkordato': 2.0,
            
            # High Impact Keywords (×1.5)
            'temettü': 1.5,
            'bonus': 1.5,
            'bedelsiz': 1.5,
            'sermaye artırımı': 1.5,
            'pay geri alım': 1.5,
            'yönetim kurulu': 1.5,
            'genel kurul': 1.5,
            
            # Medium Impact Keywords (×1.2)
            'finansal tablo': 1.2,
            'kar dağıtım': 1.2,
            'faaliyet sonuç': 1.2,
            'ortaklık yapısı': 1.2,
            'hisse devir': 1.2,
            'tahvil ihraç': 1.2,
            
            # Negative Impact Keywords (×1.3)
            'zarar': 1.3,
            'zararın': 1.3,
            'düşüş': 1.2,
            'azalış': 1.2,
            'iptal': 1.2,
            'erteleme': 1.1,
            
            # Positive Impact Keywords (×1.3)
            'kar artış': 1.3,
            'büyüme': 1.3,
            'yatırım': 1.2,
            'genişleme': 1.2,
            'sözleşme': 1.1,
            'anlaşma': 1.1
        }
    
    def classify_announcement_type(self, 
                                 title: str, 
                                 content: str) -> str:
        """
        Classify KAP announcement type based on title and content
        
        Args:
            title: Announcement title
            content: Announcement content
            
        Returns:
            Classified announcement type
        """
        title_lower = title.lower()
        content_lower = content.lower()
        text_combined = f"{title_lower} {content_lower}"
        
        # High priority classifications
        if any(keyword in text_combined for keyword in 
               ['birleşme', 'satın alma', 'devralma']):
            return 'ÖDA_BIRLEŞME'
            
        if any(keyword in text_combined for keyword in 
               ['devre kesici', 'işlem durdurma', 'durdurma bildirimi']):
            return 'ÖDA_DEVRE_KESICI'
            
        if any(keyword in text_combined for keyword in 
               ['temettü', 'kar dağıtım']):
            return 'ÖDA_TEMETTÜ'
            
        if any(keyword in text_combined for keyword in 
               ['bonus', 'bedelsiz pay']):
            return 'ÖDA_BONUS'
            
        if any(keyword in text_combined for keyword in 
               ['pay geri alım', 'hisse geri alım']):
            return 'HISSE_GERI_ALIM'
            
        # Medium priority classifications
        if 'öda' in title_lower or 'özel durum' in text_combined:
            return 'ÖDA'
            
        if any(keyword in text_combined for keyword in 
               ['finansal tablo', 'finansal rapor']):
            return 'FR'
            
        if 'faaliyet raporu' in text_combined:
            return 'FR_FAALIYET'
            
        if any(keyword in text_combined for keyword in 
               ['yönetim kurulu', 'yönetim kurulu üye']):
            return 'YONETIM_KURULU'
            
        if any(keyword in text_combined for keyword in 
               ['ortaklık yapısı', 'pay devir']):
            return 'ORTAKLIK_DEĞIŞIM'
            
        if any(keyword in text_combined for keyword in 
               ['tahvil', 'kira sertifikası', 'ihraç']):
            return 'TAHVIL_İHRAÇ'
            
        # Default classification
        if 'dg' in title_lower:
            return 'DG'
            
        return 'DEFAULT'
    
    def calculate_content_weight_multiplier(self, 
                                          title: str, 
                                          content: str) -> float:
        """
        Calculate weight multiplier based on content analysis
        
        Args:
            title: Announcement title
            content: Announcement content
            
        Returns:
            Content-based weight multiplier
        """
        text_combined = f"{title} {content}".lower()
        multiplier = 1.0
        
        # Apply keyword weights
        for keyword, weight in self.keyword_weights.items():
            if keyword in text_combined:
                multiplier *= weight
                # Cap maximum multiplier at 3.0
                multiplier = min(multiplier, 3.0)
        
        # Text length factor (longer announcements may be more significant)
        content_length = len(content)
        if content_length > 1000:
            multiplier *= 1.1
        elif content_length > 2000:
            multiplier *= 1.2
            
        # UPPERCASE emphasis factor
        uppercase_ratio = sum(c.isupper() for c in title) / max(len(title), 1)
        if uppercase_ratio > 0.3:  # High caps usage
            multiplier *= 1.1
            
        return min(multiplier, 3.0)  # Cap at 3.0
    
    def calculate_time_decay(self, 
                           announcement_time: pd.Timestamp,
                           current_time: pd.Timestamp) -> float:
        """
        Calculate time decay factor for announcement impact
        
        Uses exponential decay: W(t) = W₀ × e^(-λt)
        
        Args:
            announcement_time: When announcement was made
            current_time: Current prediction time
            
        Returns:
            Time decay factor ∈ [0, 1]
        """
        time_diff_hours = (current_time - announcement_time).total_seconds() / 3600
        
        if time_diff_hours < 0:  # Future announcement
            return 0.0
            
        if time_diff_hours > self.decay_params['max_impact_hours']:
            return 0.0  # No impact after max window
            
        # Exponential decay: e^(-ln(2) * t / half_life)
        lambda_decay = np.log(2) / self.decay_params['half_life_hours']
        decay_factor = np.exp(-lambda_decay * time_diff_hours)
        
        return decay_factor
    
    def calculate_market_condition_factor(self,
                                        current_time: pd.Timestamp,
                                        market_volatility: float = None) -> float:
        """
        Calculate market condition adjustment factor
        
        Args:
            current_time: Current time
            market_volatility: Current market volatility (optional)
            
        Returns:
            Market condition factor ∈ [0.8, 1.2]
        """
        # Default neutral factor
        factor = 1.0
        
        # Market session factor
        hour = current_time.hour
        if 10 <= hour <= 18:  # BIST trading hours
            factor *= 1.1  # Higher impact during market hours
        else:
            factor *= 0.9  # Lower impact outside market hours
            
        # Day of week factor
        weekday = current_time.weekday()
        if weekday < 5:  # Monday-Friday
            factor *= 1.0
        else:  # Weekend
            factor *= 0.8
            
        # Volatility adjustment (if provided)
        if market_volatility is not None:
            if market_volatility > 0.02:  # High volatility
                factor *= 1.1
            elif market_volatility < 0.01:  # Low volatility
                factor *= 0.95
                
        return np.clip(factor, 0.8, 1.2)
    
    def calculate_weight(self, 
                        announcement: KAPAnnouncement,
                        current_time: pd.Timestamp,
                        market_volatility: float = None) -> float:
        """
        Calculate comprehensive KAP announcement weight
        
        Weight Formula: Wt = base_weight × content_multiplier × time_decay × market_factor
        
        Args:
            announcement: KAP announcement data
            current_time: Current prediction time
            market_volatility: Current market volatility
            
        Returns:
            Final announcement weight ∈ [0, 3.0]
        """
        # 1. Get base weight by type
        ann_type = self.classify_announcement_type(announcement.title, announcement.content)
        base_weight = self.base_weights.get(ann_type, self.base_weights['DEFAULT'])
        
        # 2. Content analysis multiplier
        content_multiplier = self.calculate_content_weight_multiplier(
            announcement.title, announcement.content
        )
        
        # 3. Time decay factor
        time_decay = self.calculate_time_decay(announcement.timestamp, current_time)
        
        # 4. Market condition factor
        market_factor = self.calculate_market_condition_factor(
            current_time, market_volatility
        )
        
        # 5. Calculate final weight
        final_weight = base_weight * content_multiplier * time_decay * market_factor
        
        # 6. Apply confidence adjustment
        final_weight *= announcement.confidence
        
        # 7. Ensure bounds [0, 3.0]
        final_weight = np.clip(final_weight, 0.0, 3.0)
        
        return final_weight
    
    def calculate_aggregate_weight(self,
                                 announcements: List[KAPAnnouncement],
                                 symbol: str,
                                 current_time: pd.Timestamp,
                                 market_volatility: float = None) -> Dict[str, float]:
        """
        Calculate aggregated KAP weight for a specific symbol
        
        Args:
            announcements: List of KAP announcements
            symbol: Target stock symbol
            current_time: Current prediction time
            market_volatility: Current market volatility
            
        Returns:
            Dictionary with aggregated weights and components
        """
        # Filter announcements for the symbol
        symbol_announcements = [
            ann for ann in announcements 
            if ann.symbol.upper() == symbol.upper()
        ]
        
        if not symbol_announcements:
            return {
                'total_weight': 0.0,
                'announcement_count': 0,
                'components': []
            }
        
        # Calculate individual weights
        components = []
        total_weight = 0.0
        
        for ann in symbol_announcements:
            weight = self.calculate_weight(ann, current_time, market_volatility)
            
            if weight > 0.01:  # Only include meaningful weights
                components.append({
                    'timestamp': ann.timestamp,
                    'type': self.classify_announcement_type(ann.title, ann.content),
                    'title': ann.title[:100],  # Truncate for display
                    'weight': weight,
                    'time_decay': self.calculate_time_decay(ann.timestamp, current_time)
                })
                total_weight += weight
        
        # Cap total weight at 3.0
        total_weight = min(total_weight, 3.0)
        
        return {
            'total_weight': total_weight,
            'announcement_count': len(components),
            'components': sorted(components, key=lambda x: x['weight'], reverse=True)
        }
    
    def get_weight_explanation(self, 
                             announcement: KAPAnnouncement,
                             current_time: pd.Timestamp) -> Dict[str, any]:
        """
        Get detailed explanation of weight calculation
        
        Args:
            announcement: KAP announcement
            current_time: Current time
            
        Returns:
            Dictionary with calculation breakdown
        """
        ann_type = self.classify_announcement_type(announcement.title, announcement.content)
        base_weight = self.base_weights.get(ann_type, self.base_weights['DEFAULT'])
        content_mult = self.calculate_content_weight_multiplier(
            announcement.title, announcement.content
        )
        time_decay = self.calculate_time_decay(announcement.timestamp, current_time)
        market_factor = self.calculate_market_condition_factor(current_time)
        
        final_weight = base_weight * content_mult * time_decay * market_factor * announcement.confidence
        final_weight = np.clip(final_weight, 0.0, 3.0)
        
        return {
            'final_weight': final_weight,
            'announcement_type': ann_type,
            'base_weight': base_weight,
            'content_multiplier': content_mult,
            'time_decay': time_decay,
            'market_factor': market_factor,
            'confidence': announcement.confidence,
            'calculation': f"{base_weight:.2f} × {content_mult:.2f} × {time_decay:.2f} × {market_factor:.2f} × {announcement.confidence:.2f} = {final_weight:.2f}"
        }

# Example usage
if __name__ == "__main__":
    print("KAP Weight Calculator - Test Implementation")
    
    # Create test announcements
    calculator = KAPWeightCalculator()
    
    # High impact announcement
    high_impact_ann = KAPAnnouncement(
        timestamp=pd.Timestamp('2025-08-28 10:30:00'),
        symbol='BRSAN',
        announcement_type='ÖDA',
        title='Şirket Birleşmesi Hakkında Özel Durum Açıklaması',
        content='Şirketimiz ABC A.Ş. ile birleşme kararı almıştır. Bu birleşme sonucunda şirket değeri önemli ölçüde artacaktır.',
        confidence=0.95
    )
    
    # Medium impact announcement  
    medium_impact_ann = KAPAnnouncement(
        timestamp=pd.Timestamp('2025-08-28 08:00:00'),
        symbol='BRSAN',
        announcement_type='FR',
        title='2025 Yılı 6 Aylık Finansal Rapor',
        content='Şirketimizin 2025 yılı ilk 6 ay finansal sonuçları açıklanmıştır. Net kar 100 milyon TL olarak gerçekleşmiştir.',
        confidence=0.9
    )
    
    current_time = pd.Timestamp('2025-08-28 14:00:00')
    
    # Test individual weights
    high_weight = calculator.calculate_weight(high_impact_ann, current_time)
    medium_weight = calculator.calculate_weight(medium_impact_ann, current_time)
    
    print(f"High Impact Weight: {high_weight:.3f}")
    print(f"Medium Impact Weight: {medium_weight:.3f}")
    
    # Test aggregate weight
    announcements = [high_impact_ann, medium_impact_ann]
    aggregate = calculator.calculate_aggregate_weight(
        announcements, 'BRSAN', current_time
    )
    
    print(f"\nAggregate Weight: {aggregate['total_weight']:.3f}")
    print(f"Announcement Count: {aggregate['announcement_count']}")
    
    # Test weight explanation
    explanation = calculator.get_weight_explanation(high_impact_ann, current_time)
    print(f"\nWeight Explanation:")
    print(f"Type: {explanation['announcement_type']}")
    print(f"Calculation: {explanation['calculation']}")
