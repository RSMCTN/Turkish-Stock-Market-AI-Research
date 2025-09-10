"""
sentimentARMA - Differential Privacy Inspired LSTM with Financial News Integration

This module implements the core mathematical framework combining:
- ARMA (Autoregressive Moving Average) models
- VADER sentiment analysis for Turkish financial news
- KAP (Kamu Aydınlatma Platformu) announcement impact weighting
- Differential Privacy mechanisms

Academic Project: 
"Diferansiyel Gizlilikten Esinlenen LSTM ile Finansal Haberleri ve Değerleri 
Kullanarak İsabet Oranı Yüksek Hisse Senedi Tahmini"

Core Formula:
Yt = ARMA(p,q) × (1 + β × St × Wt)

Where:
- Yt: Predicted stock price at time t
- ARMA(p,q): Traditional autoregressive moving average component
- St: VADER sentiment score ∈ [-1, +1] from financial news
- Wt: KAP announcement impact weight factor
- β: Sentiment sensitivity parameter (learned parameter)
"""

from .sentiment_arma import SentimentARMA
from .arma_base import ARMABase
from .kap_weights import KAPWeightCalculator
from .integration import SentimentARMAIntegrator

__version__ = "1.0.0"
__author__ = "BIST AI Research Team"

__all__ = [
    'SentimentARMA',
    'ARMABase', 
    'KAPWeightCalculator',
    'SentimentARMAIntegrator'
]
