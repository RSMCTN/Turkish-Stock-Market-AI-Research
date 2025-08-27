#!/usr/bin/env python3
"""
BIST DP-LSTM Advanced Trading Dashboard - Professional UI
Gelişmiş trading arayüzü with multi-tab layout, advanced charts, and real-time features
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
import time
import random

# BIST 30 symbols and categories
BIST_30_SYMBOLS = {
    "Banking": ["AKBNK", "GARAN", "HALKB", "ISCTR", "SKBNK", "VAKBN", "YKBNK"],
    "Technology": ["ASELS", "TCELL", "BIMAS"],
    "Energy": ["EREGL", "PETKM", "TUPRS"],
    "Industrial": ["ARCELIK", "FROTO", "VESTL", "TOASO"],
    "Real Estate": ["EKGYO", "KOZAL", "KOZAA"],
    "Consumer": ["MGROS", "SAHOL", "SASA", "TAVHL"],
    "Others": ["KCHOL", "KRDMD", "OYAKC", "PGSUS", "SISE", "THYAO"]
}

ALL_SYMBOLS = [symbol for category in BIST_30_SYMBOLS.values() for symbol in category]

# Trading strategies
STRATEGIES = {
    "DP-LSTM Ensemble": "Multi-task DP-LSTM with privacy protection",
    "Technical Momentum": "RSI, MACD, Bollinger Bands strategy",
    "Sentiment Fusion": "News sentiment + technical analysis",
    "Mean Reversion": "Statistical arbitrage approach",
    "Trend Following": "Moving average crossover strategy"
}

def generate_advanced_data(symbol: str, days: int = 30, timeframe: str = "1D") -> pd.DataFrame:
    """Generate comprehensive trading data with OHLCV and indicators"""
    np.random.seed(hash(symbol) % 2**32)
    
    # Adjust periods based on timeframe
    periods_map = {"1H": days * 24, "4H": days * 6, "1D": days, "1W": max(days // 7, 4)}
    periods = periods_map.get(timeframe, days)
    
    # Generate time index
    if timeframe == "1H":
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
    elif timeframe == "4H":
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='4H')
    elif timeframe == "1W":
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='W')
    else:  # 1D
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
    
    # Generate realistic OHLCV data
    base_price = np.random.uniform(20, 100)
    prices = []
    volumes = []
    
    for i in range(periods):
        if i == 0:
            open_price = base_price
        else:
            # Price gap based on previous close
            gap = np.random.normal(0, 0.01)
            open_price = max(prices[-1]['close'] * (1 + gap), 1.0)
        
        # Intraday volatility
        volatility = np.random.uniform(0.005, 0.03)
        high = open_price * (1 + abs(np.random.normal(0, volatility)))
        low = open_price * (1 - abs(np.random.normal(0, volatility)))
        
        # Ensure high >= open >= close >= low logic
        close_change = np.random.normal(0, volatility)
        close = max(min(open_price * (1 + close_change), high), low)
        
        # Realistic volume (higher volume on bigger price moves)
        price_change_pct = abs(close - open_price) / open_price
        base_volume = np.random.uniform(500000, 2000000)
        volume = base_volume * (1 + price_change_pct * 3)
        
        prices.append({
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2)
        })
        volumes.append(int(volume))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'symbol': symbol,
        'open': [p['open'] for p in prices],
        'high': [p['high'] for p in prices],
        'low': [p['low'] for p in prices],
        'close': [p['close'] for p in prices],
        'volume': volumes
    })
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Add trading signals
    df = add_trading_signals(df)
    
    # Add sentiment data
    df['sentiment'] = np.random.uniform(-1, 1, len(df))
    df['news_count'] = np.random.poisson(3, len(df))
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to DataFrame"""
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=min(20, len(df))).mean()
    df['sma_50'] = df['close'].rolling(window=min(50, len(df))).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    return df

def add_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add trading signals based on technical analysis"""
    signals = []
    confidences = []
    expected_returns = []
    
    for i in range(len(df)):
        # Multi-factor signal generation
        rsi_val = df.iloc[i]['rsi'] if not pd.isna(df.iloc[i]['rsi']) else 50
        macd_val = df.iloc[i]['macd_histogram'] if not pd.isna(df.iloc[i]['macd_histogram']) else 0
        
        # Signal logic
        signal_score = 0
        
        # RSI signals
        if rsi_val > 70:  # Overbought
            signal_score -= 1
        elif rsi_val < 30:  # Oversold
            signal_score += 1
        
        # MACD signals
        if macd_val > 0:
            signal_score += 0.5
        else:
            signal_score -= 0.5
        
        # Price action
        if i > 0:
            price_change = (df.iloc[i]['close'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close']
            if price_change > 0.02:  # Strong up move
                signal_score += 0.5
            elif price_change < -0.02:  # Strong down move
                signal_score -= 0.5
        
        # Determine signal
        if signal_score > 0.5:
            signal = "BUY"
            confidence = min(0.95, 0.6 + abs(signal_score) * 0.2)
            expected_return = np.random.uniform(0.01, 0.05)
        elif signal_score < -0.5:
            signal = "SELL"
            confidence = min(0.95, 0.6 + abs(signal_score) * 0.2)
            expected_return = np.random.uniform(-0.05, -0.01)
        else:
            signal = "HOLD"
            confidence = np.random.uniform(0.4, 0.65)
            expected_return = np.random.uniform(-0.01, 0.01)
        
        signals.append(signal)
        confidences.append(confidence)
        expected_returns.append(expected_return)
    
    df['signal'] = signals
    df['confidence'] = confidences
    df['expected_return'] = expected_returns
    
    return df

def create_candlestick_chart(df: pd.DataFrame, indicators: List[str] = None) -> go.Figure:
    """Create professional candlestick chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Indicators', 'Volume', 'RSI'),
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=df['symbol'].iloc[0],
            increasing=dict(fillcolor='#26a69a', line=dict(color='#26a69a', width=1)),
            decreasing=dict(fillcolor='#ef5350', line=dict(color='#ef5350', width=1))
        ),
        row=1, col=1
    )
    
    # Add technical indicators
    if indicators and 'SMA20' in indicators:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sma_20'], name='SMA 20', 
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if indicators and 'SMA50' in indicators:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sma_50'], name='SMA 50', 
                      line=dict(color='purple', width=1)),
            row=1, col=1
        )
    
    if indicators and 'Bollinger' in indicators:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['bb_upper'], name='BB Upper', 
                      line=dict(color='gray', width=1, dash='dot'),
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['bb_lower'], name='BB Lower', 
                      line=dict(color='gray', width=1, dash='dot'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                      showlegend=False),
            row=1, col=1
        )
    
    # Add trading signals
    buy_signals = df[df['signal'] == 'BUY']
    sell_signals = df[df['signal'] == 'SELL']
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['date'],
                y=buy_signals['low'] * 0.99,
                mode='markers',
                name='BUY',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['date'],
                y=sell_signals['high'] * 1.01,
                mode='markers',
                name='SELL',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
            row=1, col=1
        )
    
    # Volume chart
    colors = ['red' if row['close'] < row['open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # RSI chart
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['rsi'], name='RSI', line=dict(color='blue', width=2)),
        row=3, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line=dict(color='red', dash='dash', width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash', width=1), row=3, col=1)
    fig.add_hline(y=50, line=dict(color='gray', dash='dot', width=1), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{df['symbol'].iloc[0]} - Advanced Trading Analysis",
        height=800,
        showlegend=True,
        template="plotly_dark",
        font=dict(family="Arial", size=12, color="white"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # Remove rangeslider
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

def create_portfolio_chart(symbols: List[str]) -> go.Figure:
    """Create portfolio performance chart"""
    # Generate mock portfolio data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    portfolio_data = []
    for symbol in symbols:
        np.random.seed(hash(symbol))
        returns = np.random.normal(0.001, 0.02, 30)  # Daily returns
        cumulative = np.cumprod(1 + returns) * np.random.uniform(5000, 15000)
        
        portfolio_data.append({
            'symbol': symbol,
            'dates': dates,
            'values': cumulative
        })
    
    fig = go.Figure()
    
    total_value = np.zeros(30)
    for data in portfolio_data:
        fig.add_trace(
            go.Scatter(
                x=data['dates'],
                y=data['values'],
                name=data['symbol'],
                mode='lines',
                stackgroup='one'
            )
        )
        total_value += data['values']
    
    # Add total portfolio line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=total_value,
            name='Total Portfolio',
            mode='lines',
            line=dict(color='gold', width=3)
        )
    )
    
    fig.update_layout(
        title="Portfolio Performance - Last 30 Days",
        xaxis_title="Date",
        yaxis_title="Value (TL)",
        template="plotly_dark",
        height=500,
        showlegend=True
    )
    
    return fig

def create_risk_metrics(df: pd.DataFrame) -> str:
    """Calculate and format risk metrics"""
    returns = df['close'].pct_change().dropna()
    
    # Calculate metrics
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    max_dd = ((df['close'] / df['close'].cummax()) - 1).min() * 100
    
    win_rate = len(df[df['signal'] == 'BUY']) / len(df[df['signal'] != 'HOLD']) * 100 if len(df[df['signal'] != 'HOLD']) > 0 else 0
    avg_confidence = df[df['signal'] != 'HOLD']['confidence'].mean() * 100 if len(df[df['signal'] != 'HOLD']) > 0 else 0
    
    return f"""
## 📊 Risk & Performance Metrics

### 🎯 Performance
- **Total Return:** {total_return:.2f}%
- **Annualized Volatility:** {volatility:.2f}%
- **Sharpe Ratio:** {sharpe:.2f}
- **Maximum Drawdown:** {max_dd:.2f}%

### 📈 Trading Signals  
- **Win Rate:** {win_rate:.1f}%
- **Average Confidence:** {avg_confidence:.1f}%
- **Total Signals:** {len(df[df['signal'] != 'HOLD'])}
- **Buy/Sell Ratio:** {len(df[df['signal'] == 'BUY'])}/{len(df[df['signal'] == 'SELL'])}

### 🔍 Current Status
- **Last Price:** {df['close'].iloc[-1]:.2f} TL
- **RSI:** {df['rsi'].iloc[-1]:.1f}
- **MACD:** {"Bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "Bearish"}
"""

def analyze_advanced(
    symbol: str, 
    timeframe: str, 
    days: int,
    indicators: List[str],
    strategy: str
) -> Tuple[go.Figure, go.Figure, str]:
    """Advanced analysis function"""
    
    # Generate data
    df = generate_advanced_data(symbol, days, timeframe)
    
    # Create charts
    main_chart = create_candlestick_chart(df, indicators)
    
    # Create secondary analysis based on strategy
    if strategy == "Portfolio View":
        secondary_chart = create_portfolio_chart([symbol])
    else:
        # Create comparison chart or additional analysis
        secondary_chart = create_portfolio_chart([symbol])
    
    # Risk metrics
    metrics_text = create_risk_metrics(df)
    
    return main_chart, secondary_chart, metrics_text

# Create Advanced Gradio Interface
def create_advanced_interface():
    
    # Custom CSS for professional styling
    css = """
    .gradio-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    }
    .dark {
        background: #0f172a;
    }
    .block {
        background: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    .gradio-button {
        background: linear-gradient(135deg, #f59e0b, #fbbf24) !important;
        border: none !important;
        color: #000 !important;
        font-weight: 600 !important;
    }
    """
    
    with gr.Blocks(
        title="BIST DP-LSTM Advanced Trading Dashboard", 
        theme=gr.themes.Base(primary_hue=gr.themes.colors.amber),
        css=css
    ) as demo:
        
        gr.Markdown("""
        # 🚀 BIST DP-LSTM Advanced Trading System
        ## Professional Trading Dashboard with Real-time Analytics
        
        **🎯 Features:** Multi-timeframe Analysis • Technical Indicators • Portfolio Management • Risk Analytics
        """)
        
        with gr.Tabs() as tabs:
            
            # Trading Analysis Tab
            with gr.Tab("📈 Trading Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        symbol_dropdown = gr.Dropdown(
                            choices=ALL_SYMBOLS,
                            value="AKBNK",
                            label="🏢 Select Symbol",
                            info="Choose from BIST 30 stocks"
                        )
                        
                        timeframe_radio = gr.Radio(
                            choices=["1H", "4H", "1D", "1W"],
                            value="1D",
                            label="⏰ Timeframe",
                            info="Analysis timeframe"
                        )
                        
                        days_slider = gr.Slider(
                            minimum=7, maximum=180, value=30, step=1,
                            label="📅 Period (Days)",
                            info="Analysis period"
                        )
                        
                        indicators_checkbox = gr.CheckboxGroup(
                            choices=["SMA20", "SMA50", "Bollinger", "MACD"],
                            value=["SMA20", "Bollinger"],
                            label="📊 Technical Indicators",
                            info="Select indicators to display"
                        )
                        
                        strategy_dropdown = gr.Dropdown(
                            choices=list(STRATEGIES.keys()),
                            value="DP-LSTM Ensemble",
                            label="🧠 Strategy",
                            info="Trading strategy"
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 Analyze", 
                            variant="primary", 
                            size="lg"
                        )
                
                with gr.Row():
                    main_chart = gr.Plot(label="📊 Advanced Price Analysis")
                
                with gr.Row():
                    with gr.Column():
                        secondary_chart = gr.Plot(label="📈 Portfolio Performance")
                    with gr.Column():
                        metrics_display = gr.Markdown("### 📊 Select a symbol to see metrics...")
            
            # Portfolio Management Tab  
            with gr.Tab("💼 Portfolio"):
                gr.Markdown("## 💼 Portfolio Management\n*Coming soon: Real portfolio tracking*")
                
            # Risk Analytics Tab
            with gr.Tab("⚠️ Risk Analytics"):
                gr.Markdown("## ⚠️ Advanced Risk Analytics\n*Coming soon: VaR, stress testing, correlation analysis*")
            
            # Settings Tab
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("## ⚙️ System Settings\n*Coming soon: Model parameters, trading preferences*")
        
        # Event handler
        analyze_btn.click(
            fn=analyze_advanced,
            inputs=[symbol_dropdown, timeframe_radio, days_slider, indicators_checkbox, strategy_dropdown],
            outputs=[main_chart, secondary_chart, metrics_display]
        )
        
        gr.Markdown("""
        ---
        ## 🔗 System Links
        
        - **🐙 GitHub:** [RSMCTN/BIST_AI001](https://github.com/RSMCTN/BIST_AI001)  
        - **🚂 Production API:** [Railway Deployment](https://bistai001-production.up.railway.app)
        - **🤖 ML Models:** [HuggingFace Hub](https://huggingface.co/rsmctn/bist-dp-lstm-trading-model)
        
        ## ⚖️ Risk Disclaimer
        This system is for educational and research purposes only. Past performance does not guarantee future results.
        Always consult with qualified financial advisors before making investment decisions.
        """)
    
    return demo

# Initialize the advanced interface
demo = create_advanced_interface()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )
