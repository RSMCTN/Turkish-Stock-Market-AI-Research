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

# BIST DP-LSTM Trading Model

Differentially Private LSTM ensemble for Turkish stock market (BIST) price prediction with sentiment analysis integration.

## Model Details

- **Developed by:** rsmctn
- **Model type:** PyTorch Differential Privacy LSTM Ensemble  
- **Language:** Turkish, English
- **License:** MIT
- **Repository:** [BIST_AI001](https://github.com/RSMCTN/BIST_AI001)

## Model Architecture

This model combines multiple approaches:

1. **DP-LSTM Core**: Multi-task LSTM with differential privacy (Opacus)
2. **Temporal Fusion Transformer**: Advanced attention mechanisms for financial sequences
3. **Simple Financial Transformer**: Lightweight transformer for rapid inference
4. **Ensemble Weighting**: Dynamic model combination with confidence estimation

## Training Data

- **BIST Historical Data**: 2019-2024 (BIST 30 stocks)
- **Technical Indicators**: 131+ features across multiple timeframes (1m, 5m, 15m, 60m, 1d)
- **News Sentiment**: Turkish financial news corpus with VADER + FinBERT
- **Privacy Protection**: ε=1.0 differential privacy with adaptive noise calibration

## Performance Metrics

- **Direction Accuracy (MVP)**: ≥68%
- **Direction Accuracy (Production)**: ≥75%
- **Sharpe Ratio**: >2.0
- **Max Drawdown**: <15%
- **Signal Confidence**: 65-95% range

## Usage

```python
# This is a demo model - full implementation in production system
import torch
from transformers import AutoModel

# Load model (demo)
model = AutoModel.from_pretrained("rsmctn/bist-dp-lstm-trading-model")

# Production usage requires full system:
# https://github.com/RSMCTN/BIST_AI001
```

## Intended Use

**Primary Use Cases:**
- Turkish stock market research
- Algorithmic trading signal generation  
- Financial sentiment analysis
- Academic research in privacy-preserving ML

**Limitations:**
- Demo version for research purposes
- Requires full system for production use
- Not financial advice

## Ethical Considerations

- **Privacy**: Differential privacy protects individual trader data
- **Bias Mitigation**: Diverse training across market conditions
- **Transparency**: Open-source implementation
- **Responsible AI**: Clear disclaimers about financial risks

## Citation

```bibtex
@misc{bist_dp_lstm_2024,
  title={Differential Privacy LSTM for Turkish Stock Market Prediction},
  author={rsmctn},
  year={2024},
  url={https://github.com/RSMCTN/BIST_AI001}
}
```

## Contact

- **GitHub**: [rsmctn](https://github.com/RSMCTN)  
- **Repository**: [BIST_AI001](https://github.com/RSMCTN/BIST_AI001)
- **HF Spaces Demo**: [Trading Dashboard](https://huggingface.co/spaces/rsmctn/bist-dp-lstm-trading-dashboard)

---

⚠️ **Disclaimer**: This model is for research and educational purposes only. Past performance does not guarantee future results. Always consult financial advisors before making investment decisions.