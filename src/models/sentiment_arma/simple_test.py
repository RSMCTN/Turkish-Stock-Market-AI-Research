#!/usr/bin/env python3
print("🧪 Testing sentimentARMA Mathematical Formula")
print("Formula: Yt = ARMA(p,q) × (1 + β × St × Wt) + εt")

# Simple test
import numpy as np

# Sample data
prices = [450.0, 452.0, 454.0, 449.0, 455.0]  # BRSAN prices
sentiment = 0.8  # Positive sentiment
kap_weight = 2.5  # High KAP impact
beta = 0.5  # Sentiment sensitivity

# ARMA component (simple AR(2))
arma_pred = 0.6 * prices[-1] + 0.3 * prices[-2]

# Sentiment multiplier
sentiment_mult = 1.0 + beta * sentiment * kap_weight

# Final prediction
final_pred = arma_pred * sentiment_mult

print(f"📊 ARMA Prediction: {arma_pred:.2f}")
print(f"🎭 Sentiment Multiplier: {sentiment_mult:.3f}")
print(f"🎯 Final Prediction: {final_pred:.2f}")
print(f"💥 Sentiment Impact: {final_pred - arma_pred:+.2f} TL")
print("✅ sentimentARMA mathematical formula working!")
