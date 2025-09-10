---
license: mit
task_categories:
- tabular-classification
- time-series-forecasting
language:
- tr
tags:
- finance
- bist
- turkish-stocks
- trading
- time-series
size_categories:
- 10K<n<100K
---

# bist_historical

BIST 30 historical stock data with technical indicators

## Dataset Details

- **Format:** parquet
- **Size:** ~50MB compressed
- **Language:** Turkish (labels), Numeric (data)
- **License:** MIT
- **Created:** 2025-08-27

## Dataset Structure

### BIST Historical Data
- **Symbols:** BIST 30 index stocks
- **Timeframes:** 1m, 5m, 15m, 60m, 1d
- **Features:** OHLCV + 131 technical indicators
- **Date Range:** 2019-2024

### Technical Indicators
- **Trend:** SMA, EMA, MACD, Bollinger Bands
- **Momentum:** RSI, Stochastic, Williams %R
- **Volume:** OBV, A/D Line, Volume Profile
- **Volatility:** ATR, VIX-like indicators

### News Sentiment
- **Sources:** Major Turkish financial news sites
- **Processing:** VADER + Turkish FinBERT
- **Frequency:** Daily sentiment scores
- **Credibility:** Source reliability weighting

## Usage

```python
import pandas as pd

# Load BIST historical data
df = pd.read_parquet('bist_historical.parquet')

# Load sentiment data  
sentiment_df = pd.read_json('turkish_financial_news.json')
```

## Citation

```bibtex
@misc{bist_dataset_2024,
  title={BIST Turkish Stock Market Dataset with Technical Indicators},
  author={RSMCTN},
  year={2024},
  url={https://github.com/RSMCTN/BIST_AI001}
}
```
