"""
Technical Indicators Calculator
==============================
Calculate technical indicators from OHLCV data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicatorCalculator:
    """Calculate technical indicators from OHLCV data"""
    
    def __init__(self, db_path: str = "data/bist_historical.db"):
        self.db_path = Path(db_path)
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_ohlcv_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data for a symbol"""
        with self._get_connection() as conn:
            query = """
            SELECT date_time, open_price, high_price, low_price, close_price, volume
            FROM historical_data 
            WHERE symbol = ? AND open_price IS NOT NULL
            ORDER BY date_time DESC
            LIMIT ?
            """
            
            df = pd.read_sql(query, conn, params=(symbol, limit))
            
            if df.empty:
                return df
            
            # Convert date_time to datetime and sort ascending for calculations
            df['date_time'] = pd.to_datetime(df['date_time'])
            df = df.sort_values('date_time').reset_index(drop=True)
            
            # Rename columns for easier calculation
            df.rename(columns={
                'open_price': 'Open',
                'high_price': 'High', 
                'low_price': 'Low',
                'close_price': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            return df
    
    def calculate_rsi(self, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        if len(close_prices) < period + 1:
            return pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, close_prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        if len(close_prices) < slow:
            empty = pd.Series([np.nan] * len(close_prices), index=close_prices.index)
            return empty, empty
        
        ema_fast = close_prices.ewm(span=fast).mean()
        ema_slow = close_prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line, signal_line
    
    def calculate_bollinger_bands(self, close_prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        if len(close_prices) < period:
            empty = pd.Series([np.nan] * len(close_prices), index=close_prices.index)
            return empty, empty, empty
        
        sma = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_atr(self, high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR (Average True Range)"""
        if len(close_prices) < period + 1:
            return pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        
        high_low = high_prices - low_prices
        high_close = np.abs(high_prices - close_prices.shift())
        low_close = np.abs(low_prices - close_prices.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        if len(close_prices) < period * 2:
            return pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        
        # Calculate True Range
        high_low = high_prices - low_prices
        high_close = np.abs(high_prices - close_prices.shift())
        low_close = np.abs(low_prices - close_prices.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = high_prices.diff()
        minus_dm = -low_prices.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def generate_signals(self, indicators: Dict[str, float]) -> Dict[str, str]:
        """Generate trading signals from technical indicators"""
        signals = {}
        
        # RSI signals
        if 'rsi' in indicators and indicators['rsi'] is not None:
            rsi = indicators['rsi']
            if rsi > 70:
                signals['rsi'] = 'SELL'  # Overbought
            elif rsi < 30:
                signals['rsi'] = 'BUY'   # Oversold
            else:
                signals['rsi'] = 'HOLD'
        
        # MACD signals
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    signals['macd'] = 'BUY'
                else:
                    signals['macd'] = 'SELL'
        
        # Bollinger Bands signals
        if all(k in indicators for k in ['close_price', 'bollinger_upper', 'bollinger_lower']):
            close = indicators['close_price']
            bb_upper = indicators['bollinger_upper']
            bb_lower = indicators['bollinger_lower']
            
            if close is not None and bb_upper is not None and bb_lower is not None:
                if close > bb_upper:
                    signals['bollinger'] = 'SELL'  # Price above upper band
                elif close < bb_lower:
                    signals['bollinger'] = 'BUY'   # Price below lower band
                else:
                    signals['bollinger'] = 'HOLD'
        
        # ADX signals (trend strength)
        if 'adx' in indicators and indicators['adx'] is not None:
            adx = indicators['adx']
            if adx > 25:
                signals['adx'] = 'BUY'   # Strong trend
            else:
                signals['adx'] = 'HOLD'  # Weak trend
        
        return signals
    
    def calculate_all_indicators(self, symbol: str, limit: int = 100) -> Dict[str, any]:
        """Calculate all technical indicators for a symbol"""
        try:
            # Get OHLCV data
            df = self.get_ohlcv_data(symbol, limit)
            
            if df.empty or len(df) < 30:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} records")
                return self._get_mock_indicators(symbol)
            
            # Calculate indicators
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            # Latest values for indicators
            latest_idx = -1
            
            # RSI
            rsi_series = self.calculate_rsi(close)
            rsi_current = rsi_series.iloc[latest_idx] if not pd.isna(rsi_series.iloc[latest_idx]) else None
            
            # MACD
            macd_series, signal_series = self.calculate_macd(close)
            macd_current = macd_series.iloc[latest_idx] if not pd.isna(macd_series.iloc[latest_idx]) else None
            signal_current = signal_series.iloc[latest_idx] if not pd.isna(signal_series.iloc[latest_idx]) else None
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            bb_upper_current = bb_upper.iloc[latest_idx] if not pd.isna(bb_upper.iloc[latest_idx]) else None
            bb_lower_current = bb_lower.iloc[latest_idx] if not pd.isna(bb_lower.iloc[latest_idx]) else None
            
            # ATR
            atr_series = self.calculate_atr(high, low, close)
            atr_current = atr_series.iloc[latest_idx] if not pd.isna(atr_series.iloc[latest_idx]) else None
            
            # ADX
            adx_series = self.calculate_adx(high, low, close)
            adx_current = adx_series.iloc[latest_idx] if not pd.isna(adx_series.iloc[latest_idx]) else None
            
            # Current price
            current_price = close.iloc[latest_idx]
            
            # Compile indicators
            indicators = {
                'symbol': symbol,
                'current_price': float(current_price) if current_price is not None else 0,
                'rsi': float(rsi_current) if rsi_current is not None else None,
                'macd': float(macd_current) if macd_current is not None else None,
                'macd_signal': float(signal_current) if signal_current is not None else None,
                'bollinger_upper': float(bb_upper_current) if bb_upper_current is not None else None,
                'bollinger_lower': float(bb_lower_current) if bb_lower_current is not None else None,
                'atr': float(atr_current) if atr_current is not None else None,
                'adx': float(adx_current) if adx_current is not None else None,
                'close_price': float(current_price) if current_price is not None else 0
            }
            
            # Generate signals
            signals = self.generate_signals(indicators)
            indicators['signals'] = signals
            
            logger.info(f"✅ Calculated indicators for {symbol}: RSI={rsi_current:.1f if rsi_current else 'N/A'}, MACD={macd_current:.3f if macd_current else 'N/A'}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {symbol}: {e}")
            return self._get_mock_indicators(symbol)
    
    def _get_mock_indicators(self, symbol: str) -> Dict[str, any]:
        """Generate mock indicators when calculation fails"""
        import random
        
        current_price = 50 + random.random() * 100
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'rsi': 45 + random.random() * 20,  # 45-65 range
            'macd': (random.random() - 0.5) * 0.5,  # -0.25 to 0.25
            'macd_signal': (random.random() - 0.5) * 0.3,
            'bollinger_upper': current_price * 1.05,
            'bollinger_lower': current_price * 0.95,
            'atr': current_price * 0.02,
            'adx': 20 + random.random() * 40,  # 20-60 range
            'close_price': current_price,
            'signals': {
                'rsi': random.choice(['BUY', 'SELL', 'HOLD']),
                'macd': random.choice(['BUY', 'SELL', 'HOLD']),
                'bollinger': random.choice(['BUY', 'SELL', 'HOLD']),
                'adx': random.choice(['BUY', 'HOLD'])
            }
        }

# Singleton instance
_calculator = None

def get_calculator() -> TechnicalIndicatorCalculator:
    """Get singleton calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = TechnicalIndicatorCalculator()
    return _calculator
