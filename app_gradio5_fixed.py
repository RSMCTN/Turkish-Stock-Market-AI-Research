#!/usr/bin/env python3
"""
Hugging Face Spaces - BIST DP-LSTM Trading Dashboard
Gradio 5.44.0 Compatible Version - Fixed for HF Spaces deployment
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import os

# BIST 30 symbols for demo
BIST_30_SYMBOLS = [
    "AKBNK", "ARCELIK", "ASELS", "BIMAS", "EKGYO", "EREGL", 
    "FROTO", "GARAN", "HALKB", "ISCTR", "KCHOL", "KOZAL",
    "KOZAA", "KRDMD", "MGROS", "OYAKC", "PETKM", "PGSUS",
    "SAHOL", "SASA", "SISE", "SKBNK", "TAVHL", "TCELL",
    "THYAO", "TOASO", "TUPRS", "VAKBN", "VESTL", "YKBNK"
]

def generate_mock_signals(symbol: str, days: int = 30) -> pd.DataFrame:
    """Generate mock trading signals for demo"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Mock price data with realistic movement
    np.random.seed(hash(symbol) % 2**32)  # Reproducible for same symbol
    base_price = np.random.uniform(20, 80)
    prices = [base_price]
    
    for i in range(1, days):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = max(prices[-1] * (1 + change), 1.0)  # Ensure positive price
        prices.append(new_price)
    
    # Generate signals
    signals = []
    confidences = []
    expected_returns = []
    
    for i in range(days):
        signal_prob = np.random.random()
        if signal_prob > 0.7:
            signal = "BUY"
            confidence = np.random.uniform(0.65, 0.95)
            expected_return = np.random.uniform(0.01, 0.05)
        elif signal_prob < 0.3:
            signal = "SELL" 
            confidence = np.random.uniform(0.65, 0.95)
            expected_return = np.random.uniform(-0.05, -0.01)
        else:
            signal = "HOLD"
            confidence = np.random.uniform(0.4, 0.65)
            expected_return = np.random.uniform(-0.01, 0.01)
            
        signals.append(signal)
        confidences.append(confidence)
        expected_returns.append(expected_return)
    
    return pd.DataFrame({
        'date': dates,
        'symbol': symbol,
        'price': prices,
        'signal': signals,
        'confidence': confidences,
        'expected_return': expected_returns,
        'volume': np.random.uniform(100000, 1000000, days),
        'sentiment_score': np.random.uniform(-1, 1, days)
    })

def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """Create interactive price chart with signals"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price & Trading Signals', 'Volume'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Buy signals
    buy_signals = df[df['signal'] == 'BUY']
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['date'],
                y=buy_signals['price'],
                mode='markers',
                name='BUY',
                marker=dict(color='green', size=12, symbol='triangle-up'),
                text=[f'Conf: {c:.1%}' for c in buy_signals['confidence']],
                hovertemplate='<b>BUY Signal</b><br>Date: %{x}<br>Price: %{y:.2f}<br>%{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Sell signals
    sell_signals = df[df['signal'] == 'SELL']
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['date'],
                y=sell_signals['price'],
                mode='markers',
                name='SELL',
                marker=dict(color='red', size=12, symbol='triangle-down'),
                text=[f'Conf: {c:.1%}' for c in sell_signals['confidence']],
                hovertemplate='<b>SELL Signal</b><br>Date: %{x}<br>Price: %{y:.2f}<br>%{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{df['symbol'].iloc[0]} - BIST Trading Analysis",
        height=600,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (TL)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_metrics_display(df: pd.DataFrame) -> str:
    """Create formatted metrics text"""
    total_signals = len(df[df['signal'] != 'HOLD'])
    buy_signals = len(df[df['signal'] == 'BUY'])
    sell_signals = len(df[df['signal'] == 'SELL'])
    avg_confidence = df[df['signal'] != 'HOLD']['confidence'].mean() * 100 if total_signals > 0 else 0
    
    # Mock performance calculation
    total_return = (df['price'].iloc[-1] / df['price'].iloc[0] - 1) * 100
    win_rate = np.random.uniform(60, 75)
    sharpe_ratio = np.random.uniform(1.2, 2.5)
    max_drawdown = np.random.uniform(5, 15)
    
    return f"""
## 📊 Performance Metrics

**Trading Activity:**
- Total Signals: {total_signals}
- Buy Signals: {buy_signals} 
- Sell Signals: {sell_signals}
- Average Confidence: {avg_confidence:.1f}%

**Performance:**
- Total Return: {total_return:.2f}%
- Win Rate: {win_rate:.1f}%
- Sharpe Ratio: {sharpe_ratio:.2f}
- Max Drawdown: {max_drawdown:.1f}%
"""

def create_signal_display(df: pd.DataFrame) -> str:
    """Create latest signal display"""
    latest = df.iloc[-1]
    sentiment = 'Positive' if latest['sentiment_score'] > 0.1 else 'Negative' if latest['sentiment_score'] < -0.1 else 'Neutral'
    
    return f"""
## 🎯 Latest Signal

**Symbol:** {latest['symbol']}  
**Date:** {latest['date'].strftime('%Y-%m-%d')}  
**Signal:** {latest['signal']}  
**Confidence:** {latest['confidence']:.1%}  
**Expected Return:** {latest['expected_return']:.2%}  
**Current Price:** {latest['price']:.2f} TL  
**Sentiment:** {sentiment}
"""

def analyze_stock(symbol: str, days: int) -> Tuple[go.Figure, str, str]:
    """Main analysis function - returns chart and text displays"""
    # Generate mock data
    df = generate_mock_signals(symbol, days)
    
    # Create chart
    chart = create_price_chart(df)
    
    # Create text displays
    metrics_text = create_metrics_display(df)
    signal_text = create_signal_display(df)
    
    return chart, metrics_text, signal_text

# Create Gradio Interface - Gradio 5.x Compatible
def create_interface():
    with gr.Blocks(
        title="BIST DP-LSTM Trading Dashboard",
        theme=gr.themes.Default()
    ) as demo:
        
        gr.Markdown("""
# 🚀 BIST DP-LSTM Trading System

Advanced stock trading signals for BIST (Borsa Istanbul) using:
- **Differential Privacy LSTM** models
- **Turkish Financial Sentiment Analysis**
- **Technical Indicators** (131+ features)
- **Real-time Risk Management**

⚠️ **Demo Mode:** This interface shows simulated data for demonstration purposes.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                symbol_dropdown = gr.Dropdown(
                    choices=BIST_30_SYMBOLS,
                    value="AKBNK",
                    label="📈 Select BIST Stock",
                    info="Choose from BIST 30 index stocks"
                )
                
                days_slider = gr.Slider(
                    minimum=7,
                    maximum=90,
                    value=30,
                    step=1,
                    label="📅 Analysis Period (Days)",
                    info="Number of days to analyze"
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Stock",
                    variant="primary",
                    size="lg"
                )
        
        with gr.Row():
            with gr.Column():
                metrics_display = gr.Markdown("### 📊 Select a stock to see metrics...")
            with gr.Column():
                signal_display = gr.Markdown("### 🎯 Select a stock to see latest signal...")
        
        with gr.Row():
            price_chart = gr.Plot(label="📊 Price Chart & Trading Signals")
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_stock,
            inputs=[symbol_dropdown, days_slider],
            outputs=[price_chart, metrics_display, signal_display]
        )
        
        gr.Markdown("""
---
## 🔗 Links

- **GitHub Repository:** [RSMCTN/BIST_AI001](https://github.com/RSMCTN/BIST_AI001)
- **Production API:** [Railway Deployment](https://bistai001-production.up.railway.app)
- **Research Paper:** *Differential Privacy LSTM for Turkish Stock Market Prediction*

## 🛠️ System Architecture

- **Data Sources:** MatriksIQ API, Turkish Financial News
- **Models:** DP-LSTM + TFT + Simple Transformer Ensemble
- **Privacy:** Adaptive Differential Privacy (Opacus)
- **Deployment:** Railway + Docker + FastAPI + HF Spaces
- **Monitoring:** Prometheus + Grafana

## ⚖️ Disclaimer

This system is for research and educational purposes only. 
Past performance does not guarantee future results. 
Always consult with financial advisors before making investment decisions.
        """)
    
    return demo

# Create and launch the demo
demo = create_interface()

if __name__ == "__main__":
    demo.launch()
