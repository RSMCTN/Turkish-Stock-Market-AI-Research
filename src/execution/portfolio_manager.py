"""
Advanced Portfolio Management & Paper Trading Engine for BIST
Implements position management, risk controls, and realistic trading simulation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import math

# Import with path adjustment for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from signal_generator import TradingSignal, SignalAction

# Define PositionSide here since it's not in signal_generator
class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(Enum):
    """Order types for execution"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Position:
    """Individual position tracking"""
    position_id: str
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize current price to entry price if not set"""
        if self.current_price == 0.0:
            self.current_price = self.entry_price
    
    def update_current_price(self, price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = price
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.size
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.size
    
    def get_current_value(self) -> float:
        """Get current position value"""
        return self.size * self.current_price
    
    def get_total_pnl(self) -> float:
        """Get total P&L including commissions"""
        return self.unrealized_pnl + self.realized_pnl - self.commission_paid
    
    def should_trigger_stop_loss(self) -> bool:
        """Check if stop loss should be triggered"""
        if self.stop_loss_price is None:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss_price
        else:  # SHORT
            return self.current_price >= self.stop_loss_price
    
    def should_trigger_take_profit(self) -> bool:
        """Check if take profit should be triggered"""
        if self.take_profit_price is None:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit_price
        else:  # SHORT
            return self.current_price <= self.take_profit_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'commission_paid': self.commission_paid,
            'total_pnl': self.get_total_pnl(),
            'current_value': self.get_current_value(),
            'metadata': self.metadata
        }


@dataclass
class Trade:
    """Completed trade record"""
    trade_id: str
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    exit_reason: str  # 'manual', 'stop_loss', 'take_profit', 'timeout'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_return_pct(self) -> float:
        """Get return percentage"""
        if self.side == PositionSide.LONG:
            return (self.exit_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - self.exit_price) / self.entry_price
    
    def get_duration_hours(self) -> float:
        """Get trade duration in hours"""
        return (self.exit_time - self.entry_time).total_seconds() / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'pnl': self.pnl,
            'commission': self.commission,
            'return_pct': self.get_return_pct(),
            'duration_hours': self.get_duration_hours(),
            'exit_reason': self.exit_reason,
            'metadata': self.metadata
        }


@dataclass
class PortfolioConfig:
    """Portfolio management configuration"""
    # Capital management
    initial_capital: float = 100000.0
    max_position_size_pct: float = 0.1      # Max 10% per position
    max_total_exposure_pct: float = 0.8     # Max 80% total exposure
    cash_reserve_pct: float = 0.05          # Keep 5% cash reserve
    
    # Risk management
    max_portfolio_drawdown: float = 0.15    # Max 15% portfolio drawdown
    max_position_loss_pct: float = 0.05     # Max 5% loss per position
    max_correlation: float = 0.7            # Max correlation between positions
    
    # Position management
    max_positions: int = 20                 # Maximum concurrent positions
    position_timeout_hours: int = 72        # Close positions after 72h
    rebalance_threshold_pct: float = 0.05   # Rebalance when 5% deviation
    
    # Trading costs
    commission_rate: float = 0.001          # 0.1% commission per trade
    slippage_rate: float = 0.0005          # 0.05% average slippage
    market_impact_rate: float = 0.0002      # 0.02% market impact
    funding_cost_daily: float = 0.0001      # Daily funding cost for shorts


class PortfolioManager:
    """
    Advanced Portfolio Management System
    
    Features:
    - Position tracking with real-time P&L
    - Risk management and position limits
    - Automatic stop loss / take profit execution
    - Performance analytics and reporting
    - Position timeout management
    """
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.initial_capital = config.initial_capital
        self.current_capital = config.initial_capital
        self.available_cash = config.initial_capital
        
        # Position tracking
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_trades: List[Trade] = []
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.now(), config.initial_capital)]
        self.max_equity = config.initial_capital
        self.max_drawdown = 0.0
        
        # Risk monitoring
        self.daily_pnl_history: Dict[str, List[float]] = defaultdict(list)  # date -> pnl list
        
        self.logger.info(f"Portfolio initialized with {config.initial_capital:,.2f} capital")
    
    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate current total portfolio value"""
        if current_prices:
            # Update position prices if provided
            for symbol, price in current_prices.items():
                if symbol in self.positions:
                    self.positions[symbol].update_current_price(price)
        
        # Calculate total value
        positions_value = sum(pos.get_current_value() for pos in self.positions.values())
        total_value = self.available_cash + positions_value
        
        return total_value
    
    def get_total_exposure(self) -> float:
        """Get total portfolio exposure percentage"""
        total_value = self.get_portfolio_value()
        if total_value <= 0:
            return 0.0
        
        positions_value = sum(pos.get_current_value() for pos in self.positions.values())
        return positions_value / total_value
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> float:
        """Get total realized P&L from closed trades"""
        return sum(trade.pnl for trade in self.closed_trades)
    
    def get_total_commission(self) -> float:
        """Get total commission paid"""
        return sum(trade.commission for trade in self.closed_trades)
    
    def calculate_position_size(self, signal: TradingSignal, current_price: float) -> Tuple[float, float]:
        """
        Calculate optimal position size based on signal and risk parameters
        
        Returns:
            (position_size, position_value)
        """
        portfolio_value = self.get_portfolio_value()
        
        # Base position size from config
        base_size_value = portfolio_value * self.config.max_position_size_pct
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + (signal.confidence * 1.5)  # 0.5x to 2.0x based on confidence
        adjusted_size_value = base_size_value * confidence_multiplier
        
        # Ensure we don't exceed available cash
        max_available = self.available_cash * (1 - self.config.cash_reserve_pct)
        adjusted_size_value = min(adjusted_size_value, max_available)
        
        # Calculate position size (number of shares/units)
        position_size = adjusted_size_value / current_price
        actual_position_value = position_size * current_price
        
        return position_size, actual_position_value
    
    def can_open_position(self, signal: TradingSignal, current_price: float) -> Tuple[bool, str]:
        """Check if we can open a new position"""
        
        # Check if position already exists
        if signal.symbol in self.positions:
            return False, "Position already exists for symbol"
        
        # Check maximum positions limit
        if len(self.positions) >= self.config.max_positions:
            return False, "Maximum positions limit reached"
        
        # Calculate position size and value
        position_size, position_value = self.calculate_position_size(signal, current_price)
        
        # Check available cash
        required_cash = position_value * (1 + self.config.commission_rate)  # Including commission
        if required_cash > self.available_cash * (1 - self.config.cash_reserve_pct):
            return False, "Insufficient available cash"
        
        # Check total exposure limit
        current_exposure = self.get_total_exposure()
        new_exposure = (position_value / self.get_portfolio_value()) + current_exposure
        if new_exposure > self.config.max_total_exposure_pct:
            return False, "Would exceed maximum portfolio exposure"
        
        # Check drawdown limit
        current_drawdown = self.get_current_drawdown()
        if current_drawdown > self.config.max_portfolio_drawdown:
            return False, "Portfolio in maximum drawdown"
        
        return True, "OK"
    
    def open_position(self, signal: TradingSignal, current_price: float, 
                     timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Open a new position based on signal"""
        
        timestamp = timestamp or datetime.now()
        
        # Check if we can open position
        can_open, reason = self.can_open_position(signal, current_price)
        if not can_open:
            self.logger.warning(f"Cannot open position for {signal.symbol}: {reason}")
            return {'success': False, 'reason': reason}
        
        try:
            # Calculate position details
            position_size, position_value = self.calculate_position_size(signal, current_price)
            
            # Calculate trading costs
            commission = position_value * self.config.commission_rate
            slippage = position_value * self.config.slippage_rate
            market_impact = position_value * self.config.market_impact_rate
            total_costs = commission + slippage + market_impact
            
            # Adjust for slippage (worse execution price)
            if signal.action == SignalAction.BUY:
                adjusted_price = current_price * (1 + self.config.slippage_rate)
                position_side = PositionSide.LONG
            else:  # SELL
                adjusted_price = current_price * (1 - self.config.slippage_rate)
                position_side = PositionSide.SHORT
            
            # Calculate stop loss and take profit prices
            stop_loss_price = None
            take_profit_price = None
            
            if signal.stop_loss is not None:
                if position_side == PositionSide.LONG:
                    stop_loss_price = adjusted_price * (1 - signal.stop_loss)
                else:
                    stop_loss_price = adjusted_price * (1 + signal.stop_loss)
            
            if signal.take_profit is not None:
                if position_side == PositionSide.LONG:
                    take_profit_price = adjusted_price * (1 + signal.take_profit)
                else:
                    take_profit_price = adjusted_price * (1 - signal.take_profit)
            
            # Create position
            position = Position(
                position_id=f"{signal.symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                symbol=signal.symbol,
                side=position_side,
                size=position_size,
                entry_price=adjusted_price,
                entry_time=timestamp,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                current_price=adjusted_price,
                commission_paid=total_costs,
                metadata={
                    'signal_id': signal.signal_id,
                    'signal_confidence': signal.confidence,
                    'expected_return': signal.expected_return,
                    'original_price': current_price,
                    'slippage': slippage,
                    'market_impact': market_impact
                }
            )
            
            # Update portfolio state
            self.positions[signal.symbol] = position
            self.available_cash -= (position_value + total_costs)
            
            self.logger.info(
                f"Opened {position_side.value} position: {signal.symbol} "
                f"size={position_size:.2f} @ {adjusted_price:.4f} "
                f"(costs={total_costs:.2f})"
            )
            
            return {
                'success': True,
                'position_id': position.position_id,
                'position_size': position_size,
                'entry_price': adjusted_price,
                'total_costs': total_costs,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price
            }
            
        except Exception as e:
            self.logger.error(f"Error opening position for {signal.symbol}: {e}")
            return {'success': False, 'reason': f"Execution error: {str(e)}"}
    
    def close_position(self, symbol: str, current_price: float, 
                      reason: str = "manual", timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Close an existing position"""
        
        timestamp = timestamp or datetime.now()
        
        if symbol not in self.positions:
            return {'success': False, 'reason': 'Position not found'}
        
        try:
            position = self.positions[symbol]
            
            # Calculate trading costs for closing
            position_value = position.size * current_price
            commission = position_value * self.config.commission_rate
            slippage = position_value * self.config.slippage_rate
            market_impact = position_value * self.config.market_impact_rate
            total_costs = commission + slippage + market_impact
            
            # Adjust for slippage
            if position.side == PositionSide.LONG:
                exit_price = current_price * (1 - self.config.slippage_rate)
            else:
                exit_price = current_price * (1 + self.config.slippage_rate)
            
            # Calculate final P&L
            if position.side == PositionSide.LONG:
                trade_pnl = (exit_price - position.entry_price) * position.size
            else:
                trade_pnl = (position.entry_price - exit_price) * position.size
            
            # Subtract total costs (entry + exit)
            net_pnl = trade_pnl - position.commission_paid - total_costs
            
            # Create trade record
            trade = Trade(
                trade_id=f"T_{position.position_id}",
                symbol=symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                exit_price=exit_price,
                entry_time=position.entry_time,
                exit_time=timestamp,
                pnl=net_pnl,
                commission=position.commission_paid + total_costs,
                exit_reason=reason,
                metadata=position.metadata
            )
            
            # Update portfolio state
            self.closed_trades.append(trade)
            cash_returned = position_value - total_costs  # Cash from closing position
            self.available_cash += cash_returned
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.info(
                f"Closed {position.side.value} position: {symbol} "
                f"@ {exit_price:.4f}, P&L={net_pnl:.2f}, reason={reason}"
            )
            
            return {
                'success': True,
                'trade_id': trade.trade_id,
                'exit_price': exit_price,
                'pnl': net_pnl,
                'total_costs': total_costs,
                'return_pct': trade.get_return_pct(),
                'duration_hours': trade.get_duration_hours()
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return {'success': False, 'reason': f"Execution error: {str(e)}"}
    
    def update_positions(self, current_prices: Dict[str, float], 
                        timestamp: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Update all positions with current prices and check for exits"""
        
        timestamp = timestamp or datetime.now()
        executed_exits = []
        
        # Check each position for exit conditions
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.update_current_price(current_price)
                
                # Check stop loss
                if position.should_trigger_stop_loss():
                    positions_to_close.append((symbol, current_price, "stop_loss"))
                
                # Check take profit
                elif position.should_trigger_take_profit():
                    positions_to_close.append((symbol, current_price, "take_profit"))
                
                # Check position timeout
                elif timestamp - position.entry_time > timedelta(hours=self.config.position_timeout_hours):
                    positions_to_close.append((symbol, current_price, "timeout"))
                
                # Check maximum position loss
                elif position.unrealized_pnl < 0:
                    loss_pct = abs(position.unrealized_pnl) / position.get_current_value()
                    if loss_pct > self.config.max_position_loss_pct:
                        positions_to_close.append((symbol, current_price, "max_loss"))
        
        # Execute closures
        for symbol, price, reason in positions_to_close:
            result = self.close_position(symbol, price, reason, timestamp)
            if result['success']:
                executed_exits.append(result)
        
        # Update equity curve
        current_equity = self.get_portfolio_value(current_prices)
        self.equity_curve.append((timestamp, current_equity))
        
        # Update max equity and drawdown
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        current_drawdown = (self.max_equity - current_equity) / self.max_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        return executed_exits
    
    def get_current_drawdown(self) -> float:
        """Get current portfolio drawdown"""
        current_equity = self.get_portfolio_value()
        if self.max_equity <= 0:
            return 0.0
        return (self.max_equity - current_equity) / self.max_equity
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        current_value = self.get_portfolio_value()
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        # Position summary
        position_summary = []
        for position in self.positions.values():
            position_summary.append({
                'symbol': position.symbol,
                'side': position.side.value,
                'size': position.size,
                'current_value': position.get_current_value(),
                'unrealized_pnl': position.unrealized_pnl,
                'return_pct': (position.unrealized_pnl / (position.size * position.entry_price)) if position.size > 0 else 0
            })
        
        # Trade statistics
        if self.closed_trades:
            winning_trades = [t for t in self.closed_trades if t.pnl > 0]
            losing_trades = [t for t in self.closed_trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        else:
            win_rate = avg_win = avg_loss = 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'capital': {
                'initial': self.initial_capital,
                'current_value': current_value,
                'available_cash': self.available_cash,
                'total_return': total_return,
                'total_return_pct': total_return * 100
            },
            'positions': {
                'count': len(self.positions),
                'total_exposure': self.get_total_exposure(),
                'unrealized_pnl': self.get_unrealized_pnl(),
                'positions': position_summary
            },
            'trades': {
                'total_count': len(self.closed_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_commission': self.get_total_commission(),
                'realized_pnl': self.get_realized_pnl()
            },
            'risk': {
                'current_drawdown': self.get_current_drawdown(),
                'max_drawdown': self.max_drawdown,
                'max_equity': self.max_equity
            }
        }


def test_portfolio_manager():
    """Test portfolio management functionality"""
    
    print("📊 Testing Portfolio Management System...")
    print("=" * 60)
    
    # Create configuration
    config = PortfolioConfig(
        initial_capital=100000.0,
        max_position_size_pct=0.15,
        commission_rate=0.001,
        max_positions=5
    )
    
    print(f"Portfolio Configuration:")
    print(f"   Initial Capital: {config.initial_capital:,.2f} TL")
    print(f"   Max Position Size: {config.max_position_size_pct:.1%}")
    print(f"   Commission Rate: {config.commission_rate:.1%}")
    print(f"   Max Positions: {config.max_positions}")
    
    # Initialize portfolio manager
    portfolio = PortfolioManager(config)
    
    # Signal classes already imported at module level
    
    # Test position opening
    print(f"\nTesting position opening...")
    
    signal = TradingSignal(
        symbol='AKBNK',
        timestamp=datetime.now(),
        action=SignalAction.BUY,
        confidence=0.75,
        expected_return=0.03,
        stop_loss=0.02,
        take_profit=0.05
    )
    
    current_price = 8.45
    
    # Test if we can open position
    can_open, reason = portfolio.can_open_position(signal, current_price)
    print(f"   Can open position: {can_open} ({reason})")
    
    if can_open:
        result = portfolio.open_position(signal, current_price)
        print(f"   Position opened: {result['success']}")
        if result['success']:
            print(f"     Position ID: {result['position_id']}")
            print(f"     Position Size: {result['position_size']:.2f}")
            print(f"     Entry Price: {result['entry_price']:.4f}")
            print(f"     Total Costs: {result['total_costs']:.2f}")
    
    # Test portfolio summary
    print(f"\nPortfolio Summary after opening position:")
    summary = portfolio.get_portfolio_summary()
    print(f"   Current Value: {summary['capital']['current_value']:,.2f} TL")
    print(f"   Available Cash: {summary['capital']['available_cash']:,.2f} TL")
    print(f"   Total Return: {summary['capital']['total_return_pct']:.2f}%")
    print(f"   Position Count: {summary['positions']['count']}")
    print(f"   Total Exposure: {summary['positions']['total_exposure']:.1%}")
    
    # Test position updates
    print(f"\nTesting position updates...")
    
    # Simulate price movements
    price_updates = {
        'AKBNK': 8.60  # 1.77% gain
    }
    
    exits = portfolio.update_positions(price_updates)
    print(f"   Price updated to {price_updates['AKBNK']}")
    print(f"   Exits triggered: {len(exits)}")
    
    # Show updated position
    if 'AKBNK' in portfolio.positions:
        pos = portfolio.positions['AKBNK']
        print(f"   Position P&L: {pos.unrealized_pnl:.2f}")
        print(f"   Position Return: {(pos.unrealized_pnl / (pos.size * pos.entry_price)):.2%}")
    
    # Test manual position closing
    print(f"\nTesting manual position closing...")
    if 'AKBNK' in portfolio.positions:
        close_result = portfolio.close_position('AKBNK', 8.70, 'manual')
        if close_result['success']:
            print(f"   Position closed successfully")
            print(f"   Trade P&L: {close_result['pnl']:.2f}")
            print(f"   Return: {close_result['return_pct']:.2%}")
            print(f"   Duration: {close_result['duration_hours']:.1f} hours")
    
    # Final portfolio summary
    print(f"\nFinal Portfolio Summary:")
    final_summary = portfolio.get_portfolio_summary()
    print(f"   Current Value: {final_summary['capital']['current_value']:,.2f} TL")
    print(f"   Total Return: {final_summary['capital']['total_return_pct']:.2f}%")
    print(f"   Completed Trades: {final_summary['trades']['total_count']}")
    print(f"   Total Commission Paid: {final_summary['trades']['total_commission']:.2f}")
    print(f"   Realized P&L: {final_summary['trades']['realized_pnl']:.2f}")
    
    print(f"\n✅ Portfolio Manager test completed!")
    
    return portfolio


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_portfolio_manager()
