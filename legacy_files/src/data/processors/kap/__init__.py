"""
KAP (Kamu Aydınlatma Platformu) Data Processing Module

This module processes Turkish stock market announcements from the Public Disclosure Platform.
It fetches, parses, and analyzes KAP announcements to extract sentiment and impact weights
for the sentimentARMA prediction system.

Components:
- KAPFetcher: Retrieves announcements from KAP platform
- KAPParser: Parses and structures announcement data  
- KAPAnalyzer: Analyzes sentiment and calculates impact weights
- KAPIntegrator: Integrates with sentimentARMA system

Academic Integration:
This module provides real-time fundamental analysis data for the academic project:
"Diferansiyel Gizlilikten Esinlenen LSTM ile Finansal Haberleri ve Değerleri 
Kullanarak İsabet Oranı Yüksek Hisse Senedi Tahmini"
"""

from .kap_fetcher import KAPFetcher
from .kap_parser import KAPParser
from .kap_analyzer import KAPAnalyzer
from .kap_integrator import KAPIntegrator

__version__ = "1.0.0"
__author__ = "BIST AI Research Team"

__all__ = [
    'KAPFetcher',
    'KAPParser', 
    'KAPAnalyzer',
    'KAPIntegrator'
]
