---
license: mit
tags:
- finance
- trading
- lstm
- differential-privacy
- turkish-market
- bist
language:
- tr
- en
library_name: pytorch
pipeline_tag: tabular-classification
---

# feature_selector

Automated feature selection pipeline for financial indicators

## Model Details

- **Developed by:** RSMCTN
- **Model type:** sklearn  
- **Language:** Turkish, English
- **License:** MIT
- **Repository:** [BIST_AI001](https://github.com/RSMCTN/BIST_AI001)

## Intended Use

This model is designed for research and educational purposes in the Turkish stock market (BIST). 

**Primary use cases:**
- Stock price direction prediction
- Risk-adjusted return forecasting  
- Financial sentiment analysis
- Academic research in algorithmic trading

## Training Data

- **BIST Historical Data:** 2020-2025
- **Technical Indicators:** 131+ features across multiple timeframes
- **News Sentiment:** Turkish financial news corpus
- **Privacy Protection:** Differential privacy with adaptive noise

## Performance

- **Direction Accuracy (MVP):** ≥68%
- **Direction Accuracy (Production):** ≥75%
- **Sharpe Ratio:** >2.0
- **Max Drawdown:** <15%

## Ethical Considerations

- **Differential Privacy:** Protects individual trader privacy
- **Bias Mitigation:** Diverse training data across market conditions
- **Transparency:** Open-source implementation
- **Risk Warnings:** Educational use only, not financial advice

## Citation

```bibtex
@misc{bist_dp_lstm_2024,
  title={Differential Privacy LSTM for Turkish Stock Market Prediction},
  author={RSMCTN},
  year={2024},
  url={https://github.com/RSMCTN/BIST_AI001}
}
```

## Contact

- **GitHub:** [RSMCTN](https://github.com/RSMCTN)
- **Repository:** [BIST_AI001](https://github.com/RSMCTN/BIST_AI001)
