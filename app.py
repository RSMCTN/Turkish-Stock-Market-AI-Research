#!/usr/bin/env python3
"""
Hugging Face Spaces - BIST DP-LSTM Trading Dashboard
Interactive interface for BIST stock trading signals and analysis
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import json
import requests
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add src to path for imports
sys.path.append('src')

# Mock data for demo purposes (in production would connect to real API)
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
    base_price = np.random.uniform(10, 100)
    prices = [base_price]
    
    for i in range(1, days):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Generate signals
    signals = []
    confidences = []
    expected_returns = []
    
    for i in range(days):
        # Mock signal generation logic
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
        subplot_titles=('Price & Signals', 'Volume'),
        vertical_spacing=0.1,
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
                marker=dict(color='green', size=10, symbol='triangle-up'),
                hovertemplate='<b>BUY Signal</b><br>Date: %{x}<br>Price: %{y:.2f}<br>Confidence: %{customdata:.1%}<extra></extra>',
                customdata=buy_signals['confidence']
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
                marker=dict(color='red', size=10, symbol='triangle-down'),
                hovertemplate='<b>SELL Signal</b><br>Date: %{x}<br>Price: %{y:.2f}<br>Confidence: %{customdata:.1%}<extra></extra>',
                customdata=sell_signals['confidence']
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
        title=f"{df['symbol'].iloc[0]} - Trading Signals & Price Analysis",
        xaxis_title="Date",
        yaxis_title="Price (TL)",
        template="plotly_white",
        height=600,
        showlegend=True
    )
    
    return fig

def create_performance_metrics(df: pd.DataFrame) -> Dict:
    """Calculate performance metrics"""
    total_signals = len(df[df['signal'] != 'HOLD'])
    buy_signals = len(df[df['signal'] == 'BUY'])
    sell_signals = len(df[df['signal'] == 'SELL'])
    avg_confidence = df[df['signal'] != 'HOLD']['confidence'].mean()
    
    # Mock performance calculation
    total_return = (df['price'].iloc[-1] / df['price'].iloc[0] - 1) * 100
    
    return {
        'total_signals': total_signals,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'avg_confidence': avg_confidence * 100,
        'total_return': total_return,
        'win_rate': np.random.uniform(60, 75),  # Mock win rate
        'sharpe_ratio': np.random.uniform(1.2, 2.5),  # Mock Sharpe ratio
        'max_drawdown': np.random.uniform(5, 15)  # Mock max drawdown
    }

def create_sentiment_chart(df: pd.DataFrame) -> go.Figure:
    """Create sentiment analysis chart"""
    fig = go.Figure()
    
    # Color sentiment scores
    colors = ['red' if x < -0.1 else 'yellow' if x < 0.1 else 'green' for x in df['sentiment_score']]
    
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['sentiment_score'],
            name='Sentiment Score',
            marker_color=colors,
            hovertemplate='<b>Sentiment Analysis</b><br>Date: %{x}<br>Score: %{y:.2f}<extra></extra>'
        )
    )
    
    # Add neutral line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    
    fig.update_layout(
        title="Market Sentiment Analysis (Turkish Financial News)",
        xaxis_title="Date",
        yaxis_title="Sentiment Score (-1 to +1)",
        template="plotly_white",
        height=400
    )
    
    return fig

def analyze_symbol(symbol: str, days: int = 30) -> Tuple[go.Figure, go.Figure, str, str]:
    """Main analysis function"""
    # Generate mock data
    df = generate_mock_signals(symbol, days)
    
    # Create charts
    price_chart = create_price_chart(df)
    sentiment_chart = create_sentiment_chart(df)
    
    # Calculate metrics
    metrics = create_performance_metrics(df)
    
    # Format metrics display
    metrics_text = f"""
    ## 📊 Performance Metrics
    
    **Trading Activity:**
    - Total Signals: {metrics['total_signals']}
    - Buy Signals: {metrics['buy_signals']}
    - Sell Signals: {metrics['sell_signals']}
    - Average Confidence: {metrics['avg_confidence']:.1f}%
    
    **Performance:**
    - Total Return: {metrics['total_return']:.2f}%
    - Win Rate: {metrics['win_rate']:.1f}%
    - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    - Max Drawdown: {metrics['max_drawdown']:.1f}%
    """
    
    # Latest signal info
    latest = df.iloc[-1]
    latest_signal = f"""
    ## 🎯 Latest Signal
    
    **Symbol:** {latest['symbol']}  
    **Date:** {latest['date'].strftime('%Y-%m-%d')}  
    **Signal:** {latest['signal']} 
    **Confidence:** {latest['confidence']:.1%}  
    **Expected Return:** {latest['expected_return']:.2%}  
    **Current Price:** {latest['price']:.2f} TL  
    **Sentiment:** {'Positive' if latest['sentiment_score'] > 0.1 else 'Negative' if latest['sentiment_score'] < -0.1 else 'Neutral'}
    """
    
    return price_chart, sentiment_chart, metrics_text, latest_signal

# Create Gradio Interface
with gr.Blocks(title="BIST DP-LSTM Trading Dashboard", theme=gr.themes.Soft()) as demo:
    
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
            
        with gr.Column(scale=2):
            with gr.Row():
                metrics_display = gr.Markdown("Select a stock to see metrics...")
                latest_signal_display = gr.Markdown("Select a stock to see latest signal...")
    
    with gr.Row():
        price_chart = gr.Plot(label="📊 Price Chart & Trading Signals")
        
    with gr.Row():
        sentiment_chart = gr.Plot(label="💭 Sentiment Analysis")
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_symbol,
        inputs=[symbol_dropdown, days_slider],
        outputs=[price_chart, sentiment_chart, metrics_display, latest_signal_display]
    )
    
    # Auto-load on startup
    demo.load(
        fn=analyze_symbol,
        inputs=[symbol_dropdown, days_slider],
        outputs=[price_chart, sentiment_chart, metrics_display, latest_signal_display]
    )

    gr.Markdown("""
    ---
    ## 🔗 Links
    
    - **GitHub Repository:** [RSMCTN/BIST_AI001](https://github.com/RSMCTN/BIST_AI001)
    - **Paper:** *Differential Privacy LSTM for Turkish Stock Market Prediction*
    - **Documentation:** [Full System Documentation](https://github.com/RSMCTN/BIST_AI001/blob/main/README.md)
    
    ## 🛠️ System Architecture
    
    - **Data Sources:** MatriksIQ API, Turkish Financial News
    - **Models:** DP-LSTM + TFT + Simple Transformer Ensemble  
    - **Privacy:** Adaptive Differential Privacy (Opacus)
    - **Deployment:** Railway + Docker + FastAPI
    - **Monitoring:** Prometheus + Grafana
    
    ## ⚖️ Disclaimer
    
    This system is for research and educational purposes. 
    Past performance does not guarantee future results. 
    Always consult with financial advisors before making investment decisions.
    """)

if __name__ == "__main__":
    demo.launch()
