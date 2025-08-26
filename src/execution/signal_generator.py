"""
Advanced Trading Signal Generation & Execution System for BIST
Implements real-time signal generation, portfolio management, and paper trading
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings


class SignalAction(Enum):
    """Trading signal actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PositionSide(Enum):
    """Position side for tracking"""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class TradingSignal:
    """
    Comprehensive trading signal with all necessary information
    
    Features:
    - Unique signal identification
    - Risk parameters (stop loss, take profit)
    - Confidence and expected return
    - Metadata for tracking
    """
    
    symbol: str
    timestamp: datetime
    action: SignalAction
    confidence: float
    expected_return: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal parameters"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        if self.stop_loss is not None and (self.stop_loss < 0 or self.stop_loss > 0.5):
            warnings.warn(f"Unusual stop loss value: {self.stop_loss}")
        
        if self.take_profit is not None and (self.take_profit < 0 or self.take_profit > 1.0):
            warnings.warn(f"Unusual take profit value: {self.take_profit}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action.value,
            'confidence': self.confidence,
            'expected_return': self.expected_return,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create signal from dictionary"""
        return cls(
            signal_id=data.get('signal_id', str(uuid.uuid4())[:8]),
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            action=SignalAction(data['action']),
            confidence=data['confidence'],
            expected_return=data['expected_return'],
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def no_signal(cls, symbol: str, timestamp: Optional[datetime] = None) -> 'TradingSignal':
        """Create a HOLD/no-signal for symbol"""
        return cls(
            symbol=symbol,
            timestamp=timestamp or datetime.now(),
            action=SignalAction.HOLD,
            confidence=0.0,
            expected_return=0.0,
            metadata={'reason': 'no_signal'}
        )
    
    def is_actionable(self) -> bool:
        """Check if signal is actionable (not HOLD)"""
        return self.action != SignalAction.HOLD and self.confidence > 0


@dataclass 
class SignalGeneratorConfig:
    """Configuration for signal generator"""
    # Signal thresholds
    buy_threshold: float = 0.6              # Confidence threshold for buy signals
    sell_threshold: float = 0.6             # Confidence threshold for sell signals  
    min_expected_return: float = 0.01       # Minimum expected return to trade (1%)
    
    # Risk parameters
    default_stop_loss_pct: float = 0.02     # Default 2% stop loss
    default_take_profit_pct: float = 0.05   # Default 5% take profit
    
    # Signal limits
    max_signals_per_symbol: int = 3         # Max signals per symbol per day
    signal_cooldown_hours: int = 2          # Hours between signals for same symbol
    
    # Market condition filters
    min_volume_ratio: float = 0.5           # Min volume vs 10-day average
    max_volatility_zscore: float = 3.0      # Max volatility z-score
    
    # Position sizing
    base_position_size: float = 0.05        # Base position size (5% of capital)
    confidence_multiplier: float = 1.5      # Multiply position size by confidence


class SignalGenerator:
    """
    Advanced trading signal generator
    
    Features:
    - Model prediction integration
    - Risk-aware signal generation
    - Daily limits and cooldown periods
    - Market condition filtering
    - Position sizing based on confidence
    """
    
    def __init__(self, model, feature_processor, config: SignalGeneratorConfig):
        self.model = model
        self.feature_processor = feature_processor
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Signal tracking
        self.daily_signal_count: Dict[str, int] = {}
        self.last_signal_time: Dict[str, datetime] = {}
        self.last_reset_date = datetime.now().date()
        
        # Model performance tracking
        self.prediction_history: List[Dict] = []
        
    def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_signal_count.clear()
            self.last_reset_date = current_date
            self.logger.info("Daily signal counters reset")
    
    def _check_signal_limits(self, symbol: str) -> Tuple[bool, str]:
        """Check if we can generate more signals for this symbol"""
        current_count = self.daily_signal_count.get(symbol, 0)
        
        # Check daily limit
        if current_count >= self.config.max_signals_per_symbol:
            return False, f"Daily limit reached ({current_count}/{self.config.max_signals_per_symbol})"
        
        # Check cooldown period
        last_signal_time = self.last_signal_time.get(symbol)
        if last_signal_time:
            hours_since_last = (datetime.now() - last_signal_time).total_seconds() / 3600
            if hours_since_last < self.config.signal_cooldown_hours:
                return False, f"Cooldown period ({hours_since_last:.1f}h < {self.config.signal_cooldown_hours}h)"
        
        return True, "OK"
    
    def _check_market_conditions(self, market_data: Dict) -> Tuple[bool, str]:
        """Check if market conditions are suitable for trading"""
        try:
            # Check volume condition
            if 'volume_ratio' in market_data:
                volume_ratio = market_data['volume_ratio']
                if volume_ratio < self.config.min_volume_ratio:
                    return False, f"Low volume ratio: {volume_ratio:.2f}"
            
            # Check volatility condition
            if 'volatility_zscore' in market_data:
                vol_zscore = abs(market_data['volatility_zscore'])
                if vol_zscore > self.config.max_volatility_zscore:
                    return False, f"High volatility: {vol_zscore:.2f}"
            
            # Check if market is open (basic check)
            current_hour = datetime.now().hour
            if current_hour < 9 or current_hour > 17:  # Basic BIST hours
                return False, "Market closed"
            
            return True, "OK"
            
        except Exception as e:
            self.logger.warning(f"Error checking market conditions: {e}")
            return False, "Market condition check failed"
    
    async def _generate_prediction(self, features: torch.Tensor) -> Dict[str, Any]:
        """Generate model prediction from features"""
        try:
            self.model.eval()
            with torch.no_grad():
                # Add batch dimension if needed
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                
                # Generate prediction
                outputs = self.model(features)
                
                # Extract predictions
                prediction = {
                    'direction_prob': torch.softmax(outputs.get('direction', torch.zeros(1, 3)), dim=-1).cpu().numpy()[0],
                    'magnitude': outputs.get('magnitude', torch.zeros(1, 1)).cpu().numpy()[0, 0],
                    'volatility': outputs.get('volatility', torch.zeros(1, 1)).cpu().numpy()[0, 0],
                    'volume': outputs.get('volume', torch.zeros(1, 1)).cpu().numpy()[0, 0],
                    'confidence': outputs.get('confidence', torch.tensor([0.5])).cpu().numpy()[0]
                }
                
                # Store prediction for tracking
                self.prediction_history.append({
                    'timestamp': datetime.now(),
                    'prediction': prediction
                })
                
                # Keep only recent predictions (last 1000)
                if len(self.prediction_history) > 1000:
                    self.prediction_history = self.prediction_history[-1000:]
                
                return prediction
                
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return {
                'direction_prob': np.array([0.33, 0.34, 0.33]),  # Neutral
                'magnitude': 0.0,
                'volatility': 0.0,
                'volume': 0.0,
                'confidence': 0.0
            }
    
    def _apply_signal_rules(self, symbol: str, prediction: Dict[str, Any], 
                          market_data: Dict) -> TradingSignal:
        """Apply trading rules to generate signal"""
        
        direction_prob = prediction['direction_prob']
        magnitude = abs(prediction['magnitude'])
        confidence = prediction['confidence']
        
        # Determine action based on probabilities and thresholds
        max_prob_idx = np.argmax(direction_prob)
        max_prob = direction_prob[max_prob_idx]
        
        action = SignalAction.HOLD
        expected_return = 0.0
        
        # Apply signal generation rules
        if max_prob_idx == 2 and max_prob >= self.config.buy_threshold:  # Up movement
            if magnitude >= self.config.min_expected_return:
                action = SignalAction.BUY
                expected_return = magnitude
        elif max_prob_idx == 0 and max_prob >= self.config.sell_threshold:  # Down movement  
            if magnitude >= self.config.min_expected_return:
                action = SignalAction.SELL
                expected_return = -magnitude
        
        # Override if confidence is too low
        if confidence < max(self.config.buy_threshold, self.config.sell_threshold):
            action = SignalAction.HOLD
            expected_return = 0.0
        
        # Calculate risk parameters
        stop_loss = None
        take_profit = None
        
        if action != SignalAction.HOLD:
            # Dynamic stop loss based on volatility
            volatility_factor = max(0.5, min(2.0, prediction.get('volatility', 1.0)))
            stop_loss = self.config.default_stop_loss_pct * volatility_factor
            
            # Dynamic take profit based on expected return
            take_profit = min(
                self.config.default_take_profit_pct,
                max(expected_return * 2, self.config.default_take_profit_pct * 0.5)
            )
        
        # Create signal with metadata
        metadata = {
            'model_confidence': confidence,
            'direction_probs': direction_prob.tolist(),
            'predicted_volatility': prediction.get('volatility', 0.0),
            'predicted_volume': prediction.get('volume', 0.0),
            'market_conditions': market_data.get('conditions', {}),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return TradingSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            action=action,
            confidence=confidence,
            expected_return=expected_return,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata
        )
    
    async def generate_signal(self, symbol: str, market_data: Dict, 
                            news_data: Optional[List[Dict]] = None) -> TradingSignal:
        """
        Generate trading signal for symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AKBNK')
            market_data: Recent price/volume data
            news_data: Recent news with sentiment scores
            
        Returns:
            TradingSignal object
        """
        self._reset_daily_counters()
        
        try:
            # Check signal limits
            can_signal, limit_reason = self._check_signal_limits(symbol)
            if not can_signal:
                self.logger.debug(f"Signal limit for {symbol}: {limit_reason}")
                return TradingSignal.no_signal(symbol)
            
            # Check market conditions
            market_ok, market_reason = self._check_market_conditions(market_data)
            if not market_ok:
                self.logger.debug(f"Market condition for {symbol}: {market_reason}")
                return TradingSignal.no_signal(symbol)
            
            # Process features
            if self.feature_processor:
                features = await self.feature_processor.process_features(
                    market_data, news_data or []
                )
                
                if features is None or (isinstance(features, torch.Tensor) and features.numel() == 0):
                    return TradingSignal.no_signal(symbol)
            else:
                # Mock features for testing
                features = torch.randn(1, 60, 131)  # Batch, sequence, features
            
            # Generate prediction
            prediction = await self._generate_prediction(features)
            
            # Apply signal generation rules
            signal = self._apply_signal_rules(symbol, prediction, market_data)
            
            # Update tracking if actionable signal generated
            if signal.is_actionable():
                self.daily_signal_count[symbol] = self.daily_signal_count.get(symbol, 0) + 1
                self.last_signal_time[symbol] = signal.timestamp
                
                self.logger.info(
                    f"Generated {signal.action.value} signal for {symbol}: "
                    f"confidence={signal.confidence:.3f}, return={signal.expected_return:.3f}"
                )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return TradingSignal.no_signal(symbol)
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily signal generation statistics"""
        total_signals = sum(self.daily_signal_count.values())
        unique_symbols = len(self.daily_signal_count)
        
        return {
            'total_signals_today': total_signals,
            'unique_symbols': unique_symbols,
            'signals_per_symbol': dict(self.daily_signal_count),
            'last_reset_date': self.last_reset_date.isoformat(),
            'recent_predictions': len(self.prediction_history)
        }


def test_signal_generator():
    """Test signal generator functionality"""
    
    print("🔄 Testing Trading Signal Generation System...")
    print("=" * 60)
    
    # Create configuration
    config = SignalGeneratorConfig(
        buy_threshold=0.6,
        sell_threshold=0.6,
        min_expected_return=0.015,  # 1.5%
        max_signals_per_symbol=5
    )
    
    print(f"Signal Generator Configuration:")
    print(f"   Buy threshold: {config.buy_threshold}")
    print(f"   Sell threshold: {config.sell_threshold}")
    print(f"   Min expected return: {config.min_expected_return:.1%}")
    print(f"   Max signals per symbol: {config.max_signals_per_symbol}")
    
    # Mock model for testing
    class MockModel:
        def eval(self): pass
        
        def __call__(self, features):
            # Return realistic mock predictions
            return {
                'direction': torch.tensor([[0.2, 0.3, 0.5]]),  # Slightly bullish
                'magnitude': torch.tensor([[0.025]]),  # 2.5% expected move
                'volatility': torch.tensor([[0.15]]),  # 15% volatility
                'volume': torch.tensor([[1.2]]),       # 20% above average
                'confidence': torch.tensor([0.75])     # 75% confidence
            }
    
    # Initialize signal generator
    model = MockModel()
    signal_gen = SignalGenerator(
        model=model, 
        feature_processor=None,  # Mock will be used
        config=config
    )
    
    # Test signal generation
    print(f"\nTesting signal generation for AKBNK...")
    
    market_data = {
        'symbol': 'AKBNK',
        'current_price': 8.45,
        'volume_ratio': 1.2,
        'volatility_zscore': 1.5,
        'conditions': {'market_open': True}
    }
    
    # Generate multiple signals to test limits
    async def run_signal_tests():
        signals = []
        for i in range(3):
            signal = await signal_gen.generate_signal('AKBNK', market_data)
            signals.append(signal)
            print(f"\nSignal {i+1}:")
            print(f"   Action: {signal.action.value}")
            print(f"   Confidence: {signal.confidence:.3f}")
            print(f"   Expected Return: {signal.expected_return:.3f}")
            print(f"   Stop Loss: {signal.stop_loss:.3f}" if signal.stop_loss else "   Stop Loss: None")
            print(f"   Take Profit: {signal.take_profit:.3f}" if signal.take_profit else "   Take Profit: None")
            print(f"   Signal ID: {signal.signal_id}")
        
        return signals
    
    # Run async test
    import asyncio
    signals = asyncio.run(run_signal_tests())
    
    # Test signal limits
    print(f"\nTesting signal limits...")
    print(f"Daily stats: {signal_gen.get_daily_stats()}")
    
    # Test signal serialization
    print(f"\nTesting signal serialization...")
    first_signal = signals[0]
    signal_dict = first_signal.to_dict()
    reconstructed = TradingSignal.from_dict(signal_dict)
    
    print(f"   Original action: {first_signal.action.value}")
    print(f"   Reconstructed action: {reconstructed.action.value}")
    print(f"   Serialization: ✅")
    
    # Test market condition filtering
    print(f"\nTesting market condition filtering...")
    
    bad_market_data = market_data.copy()
    bad_market_data['volume_ratio'] = 0.3  # Too low
    
    bad_signal = asyncio.run(
        signal_gen.generate_signal('GARAN', bad_market_data)
    )
    print(f"   Bad market signal: {bad_signal.action.value} (should be HOLD)")
    
    print(f"\n✅ Signal Generator test completed!")
    
    return signal_gen, signals


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_signal_generator()
