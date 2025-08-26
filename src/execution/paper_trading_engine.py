"""
Advanced Paper Trading Engine & Backtesting Framework for BIST
Implements realistic trading simulation with market microstructure effects
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque, defaultdict
import math

# Import local modules with path adjustment
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from signal_generator import TradingSignal, SignalAction, SignalGenerator, SignalGeneratorConfig
from portfolio_manager import PortfolioManager, PortfolioConfig, Position, Trade


@dataclass
class MarketData:
    """Market data point for backtesting"""
    timestamp: datetime
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    
    def get_mid_price(self) -> float:
        """Get mid price (bid + ask) / 2 or close price"""
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2
        return self.close_price
    
    def get_spread_pct(self) -> float:
        """Get bid-ask spread percentage"""
        if self.bid_price is not None and self.ask_price is not None:
            mid = self.get_mid_price()
            if mid > 0:
                return (self.ask_price - self.bid_price) / mid
        return 0.01  # Default 1% spread


@dataclass
class TradingEngineConfig:
    """Configuration for paper trading engine"""
    # Capital and position management
    initial_capital: float = 100000.0
    max_positions: int = 10
    position_timeout_hours: int = 72
    
    # Market simulation
    market_impact_model: str = "sqrt"         # 'linear' or 'sqrt'
    liquidity_factor: float = 1000000.0       # Higher = more liquid market
    volatility_impact: bool = True            # Include volatility in costs
    
    # Execution algorithms
    execution_algorithm: str = "market"       # 'market', 'vwap', 'twap'
    vwap_slices: int = 5                     # Number of VWAP slices
    twap_slices: int = 10                    # Number of TWAP slices
    
    # Risk management
    max_daily_loss: float = 0.05             # Max 5% daily loss
    max_position_size: float = 0.15          # Max 15% per position
    stop_loss_slippage: float = 0.02         # Additional slippage on stops
    
    # Backtesting
    warmup_days: int = 30                    # Days for technical indicators
    commission_model: str = "fixed"          # 'fixed' or 'tiered'
    slippage_model: str = "adaptive"         # 'fixed' or 'adaptive'


class ExecutionAlgorithm:
    """Base class for execution algorithms"""
    
    def __init__(self, config: TradingEngineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute_order(self, signal: TradingSignal, market_data: List[MarketData], 
                           portfolio: PortfolioManager) -> Dict[str, Any]:
        """Execute order using specific algorithm"""
        raise NotImplementedError


class MarketOrderExecution(ExecutionAlgorithm):
    """Market order execution with realistic market impact"""
    
    async def execute_order(self, signal: TradingSignal, market_data: List[MarketData], 
                           portfolio: PortfolioManager) -> Dict[str, Any]:
        """Execute market order immediately"""
        
        if not market_data:
            return {'success': False, 'reason': 'No market data'}
        
        current_data = market_data[-1]  # Use latest data point
        current_price = current_data.get_mid_price()
        
        # Calculate position size
        position_size, position_value = portfolio.calculate_position_size(signal, current_price)
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(position_value, current_data)
        
        # Calculate slippage based on market conditions
        slippage = self._calculate_slippage(signal, current_data, market_impact)
        
        # Adjust execution price
        if signal.action == SignalAction.BUY:
            execution_price = current_price * (1 + slippage + market_impact)
        else:
            execution_price = current_price * (1 - slippage - market_impact)
        
        # Execute through portfolio manager
        result = portfolio.open_position(signal, execution_price, current_data.timestamp)
        
        if result['success']:
            result.update({
                'execution_algorithm': 'market',
                'market_impact': market_impact,
                'slippage': slippage,
                'original_price': current_price
            })
        
        return result
    
    def _calculate_market_impact(self, order_value: float, market_data: MarketData) -> float:
        """Calculate market impact based on order size and liquidity"""
        
        # Simple square root market impact model
        if self.config.market_impact_model == "sqrt":
            impact_factor = math.sqrt(order_value / self.config.liquidity_factor)
        else:  # linear
            impact_factor = order_value / self.config.liquidity_factor
        
        # Scale by volume (higher volume = lower impact)
        volume_factor = max(0.1, min(2.0, market_data.volume / 1000000))  # Normalize to 1M volume
        impact = impact_factor / volume_factor
        
        # Cap market impact
        return min(impact, 0.01)  # Max 1% impact
    
    def _calculate_slippage(self, signal: TradingSignal, market_data: MarketData, 
                          market_impact: float) -> float:
        """Calculate execution slippage"""
        
        base_slippage = 0.0005  # 0.05% base slippage
        
        if self.config.slippage_model == "adaptive":
            # Adjust for spread
            spread_factor = 1 + market_data.get_spread_pct() * 2
            
            # Adjust for volatility (using simple price change proxy)
            price_change = abs((market_data.close_price - market_data.open_price) / market_data.open_price)
            volatility_factor = 1 + price_change * 5
            
            # Adjust for market impact
            impact_factor = 1 + market_impact * 3
            
            adaptive_slippage = base_slippage * spread_factor * volatility_factor * impact_factor
            return min(adaptive_slippage, 0.01)  # Cap at 1%
        
        return base_slippage


class VWAPExecution(ExecutionAlgorithm):
    """VWAP (Volume Weighted Average Price) execution"""
    
    async def execute_order(self, signal: TradingSignal, market_data: List[MarketData], 
                           portfolio: PortfolioManager) -> Dict[str, Any]:
        """Execute order using VWAP slicing"""
        
        if len(market_data) < self.config.vwap_slices:
            # Fall back to market order
            market_exec = MarketOrderExecution(self.config)
            return await market_exec.execute_order(signal, market_data, portfolio)
        
        # Calculate VWAP from recent data
        recent_data = market_data[-self.config.vwap_slices:]
        vwap_price = self._calculate_vwap(recent_data)
        
        # Simulate gradual execution with lower market impact
        total_market_impact = 0.0
        total_slippage = 0.0
        
        for i, data_point in enumerate(recent_data):
            slice_weight = data_point.volume / sum(d.volume for d in recent_data)
            
            # Calculate slice-specific costs
            slice_impact = self._calculate_slice_impact(slice_weight, data_point)
            slice_slippage = 0.0002 * (1 + i * 0.1)  # Increasing slippage per slice
            
            total_market_impact += slice_impact * slice_weight
            total_slippage += slice_slippage * slice_weight
        
        # Execute at VWAP with calculated costs
        if signal.action == SignalAction.BUY:
            execution_price = vwap_price * (1 + total_slippage + total_market_impact)
        else:
            execution_price = vwap_price * (1 - total_slippage - total_market_impact)
        
        result = portfolio.open_position(signal, execution_price, market_data[-1].timestamp)
        
        if result['success']:
            result.update({
                'execution_algorithm': 'vwap',
                'vwap_price': vwap_price,
                'market_impact': total_market_impact,
                'slippage': total_slippage,
                'slices_executed': len(recent_data)
            })
        
        return result
    
    def _calculate_vwap(self, market_data: List[MarketData]) -> float:
        """Calculate Volume Weighted Average Price"""
        total_value = sum(data.get_mid_price() * data.volume for data in market_data)
        total_volume = sum(data.volume for data in market_data)
        
        if total_volume == 0:
            return market_data[-1].get_mid_price()
        
        return total_value / total_volume
    
    def _calculate_slice_impact(self, weight: float, market_data: MarketData) -> float:
        """Calculate market impact for VWAP slice"""
        base_impact = weight * 0.0001  # Lower impact due to slicing
        
        # Adjust for volume
        volume_factor = max(0.5, market_data.volume / 1000000)
        return base_impact / volume_factor


class PaperTradingEngine:
    """
    Advanced Paper Trading Engine with Realistic Market Simulation
    
    Features:
    - Multiple execution algorithms (Market, VWAP, TWAP)
    - Realistic market impact and slippage modeling  
    - Real-time and backtesting modes
    - Comprehensive performance analytics
    - Risk management integration
    """
    
    def __init__(self, config: TradingEngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize portfolio manager
        portfolio_config = PortfolioConfig(
            initial_capital=config.initial_capital,
            max_position_size_pct=config.max_position_size,
            max_positions=config.max_positions,
            position_timeout_hours=config.position_timeout_hours
        )
        self.portfolio = PortfolioManager(portfolio_config)
        
        # Initialize execution algorithms
        self.execution_algorithms = {
            'market': MarketOrderExecution(config),
            'vwap': VWAPExecution(config),
            'twap': VWAPExecution(config)  # Simplified as VWAP for now
        }
        
        # Market data buffer for execution algorithms
        self.market_data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance tracking
        self.daily_pnl_history: List[Tuple[datetime, float]] = []
        self.trade_log: List[Dict] = []
        self.execution_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Risk monitoring
        self.daily_loss_limit_hit = False
        self.last_reset_date = datetime.now().date()
        
        self.logger.info(f"Paper Trading Engine initialized with {config.initial_capital:,.2f} capital")
    
    def add_market_data(self, data: MarketData):
        """Add market data point for simulation"""
        self.market_data_buffer[data.symbol].append(data)
        
        # Update portfolio with current prices
        current_prices = {data.symbol: data.get_mid_price()}
        exits = self.portfolio.update_positions(current_prices, data.timestamp)
        
        # Log any automatic exits
        for exit_info in exits:
            self.logger.info(f"Auto-exit: {exit_info['trade_id']} - {exit_info.get('reason', 'unknown')}")
    
    async def execute_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute trading signal using configured algorithm"""
        
        self._reset_daily_limits()
        
        # Check daily loss limit
        if self.daily_loss_limit_hit:
            return {'success': False, 'reason': 'Daily loss limit hit'}
        
        # Check if we have market data
        if signal.symbol not in self.market_data_buffer or len(self.market_data_buffer[signal.symbol]) == 0:
            return {'success': False, 'reason': 'No market data available'}
        
        try:
            # Get execution algorithm
            algorithm = self.execution_algorithms.get(
                self.config.execution_algorithm, 
                self.execution_algorithms['market']
            )
            
            # Get market data for execution
            market_data = list(self.market_data_buffer[signal.symbol])
            
            # Execute order
            result = await algorithm.execute_order(signal, market_data, self.portfolio)
            
            # Track execution statistics
            if result['success']:
                self._track_execution_stats(signal, result, market_data[-1])
                
                # Log trade
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'signal': signal.to_dict(),
                    'execution': result
                })
            
            # Check daily loss limit after execution
            self._check_daily_loss_limit()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return {'success': False, 'reason': f'Execution error: {str(e)}'}
    
    def _reset_daily_limits(self):
        """Reset daily limits if new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_loss_limit_hit = False
            self.last_reset_date = current_date
            self.logger.info("Daily limits reset")
    
    def _check_daily_loss_limit(self):
        """Check if daily loss limit is hit"""
        if self.daily_loss_limit_hit:
            return
        
        current_value = self.portfolio.get_portfolio_value()
        daily_loss = (self.config.initial_capital - current_value) / self.config.initial_capital
        
        if daily_loss >= self.config.max_daily_loss:
            self.daily_loss_limit_hit = True
            self.logger.warning(f"Daily loss limit hit: {daily_loss:.2%}")
            
            # Close all positions
            current_prices = {}
            for symbol, buffer in self.market_data_buffer.items():
                if buffer:
                    current_prices[symbol] = buffer[-1].get_mid_price()
            
            for symbol in list(self.portfolio.positions.keys()):
                if symbol in current_prices:
                    self.portfolio.close_position(symbol, current_prices[symbol], 'daily_limit')
    
    def _track_execution_stats(self, signal: TradingSignal, result: Dict, market_data: MarketData):
        """Track execution statistics"""
        
        # Track slippage
        if 'slippage' in result:
            self.execution_stats['slippage'].append(result['slippage'])
        
        # Track market impact
        if 'market_impact' in result:
            self.execution_stats['market_impact'].append(result['market_impact'])
        
        # Track execution delay (for future use)
        execution_delay = 0.0  # Placeholder
        self.execution_stats['execution_delay'].append(execution_delay)
        
        # Track spread at execution
        spread = market_data.get_spread_pct()
        self.execution_stats['spread_at_execution'].append(spread)
    
    def run_backtest(self, market_data: List[MarketData], 
                    signal_generator: SignalGenerator, 
                    news_data: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest simulation
        
        Args:
            market_data: Historical market data
            signal_generator: Configured signal generator
            news_data: Optional news data by symbol
            
        Returns:
            Comprehensive backtest results
        """
        
        self.logger.info(f"Starting backtest with {len(market_data)} data points")
        
        # Sort market data by timestamp
        market_data_sorted = sorted(market_data, key=lambda x: x.timestamp)
        
        # Group data by symbol
        data_by_symbol = defaultdict(list)
        for data in market_data_sorted:
            data_by_symbol[data.symbol].append(data)
        
        # Run simulation
        backtest_results = {
            'start_time': market_data_sorted[0].timestamp,
            'end_time': market_data_sorted[-1].timestamp,
            'total_signals': 0,
            'executed_signals': 0,
            'total_trades': 0,
            'performance_metrics': {},
            'execution_stats': {},
            'trade_analysis': {},
            'risk_metrics': {}
        }
        
        signals_generated = []
        executed_trades = []
        
        # Process each data point
        async def run_simulation():
            for data_point in market_data_sorted:
                # Add market data
                self.add_market_data(data_point)
                
                # Generate signal if we have enough warmup data
                warmup_count = len(self.market_data_buffer[data_point.symbol])
                if warmup_count >= self.config.warmup_days:
                    
                    # Prepare market data for signal generation
                    market_data_dict = {
                        'symbol': data_point.symbol,
                        'current_price': data_point.get_mid_price(),
                        'volume_ratio': 1.0,  # Placeholder
                        'volatility_zscore': 0.0,  # Placeholder
                        'timestamp': data_point.timestamp
                    }
                    
                    # Get relevant news data
                    symbol_news = news_data.get(data_point.symbol, []) if news_data else []
                    
                    # Generate signal
                    signal = await signal_generator.generate_signal(
                        data_point.symbol, 
                        market_data_dict, 
                        symbol_news
                    )
                    
                    signals_generated.append(signal)
                    backtest_results['total_signals'] += 1
                    
                    # Execute signal if actionable
                    if signal.is_actionable():
                        execution_result = await self.execute_signal(signal)
                        
                        if execution_result['success']:
                            executed_trades.append(execution_result)
                            backtest_results['executed_signals'] += 1
                
                # Update daily P&L tracking
                if data_point.timestamp.date() != getattr(self, '_last_pnl_date', None):
                    current_value = self.portfolio.get_portfolio_value()
                    daily_return = (current_value - self.config.initial_capital) / self.config.initial_capital
                    self.daily_pnl_history.append((data_point.timestamp, daily_return))
                    self._last_pnl_date = data_point.timestamp.date()
        
        # Run simulation
        import asyncio
        asyncio.run(run_simulation())
        
        # Calculate performance metrics
        backtest_results['total_trades'] = len(self.portfolio.closed_trades)
        backtest_results['performance_metrics'] = self._calculate_performance_metrics()
        backtest_results['execution_stats'] = self._calculate_execution_stats()
        backtest_results['trade_analysis'] = self._analyze_trades()
        backtest_results['risk_metrics'] = self._calculate_risk_metrics()
        
        self.logger.info(f"Backtest completed: {backtest_results['total_signals']} signals, "
                        f"{backtest_results['executed_signals']} executed, "
                        f"{backtest_results['total_trades']} completed trades")
        
        return backtest_results
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        portfolio_summary = self.portfolio.get_portfolio_summary()
        final_value = portfolio_summary['capital']['current_value']
        total_return = portfolio_summary['capital']['total_return']
        
        # Calculate Sharpe ratio (simplified)
        if self.daily_pnl_history:
            daily_returns = [pnl for _, pnl in self.daily_pnl_history]
            returns_std = np.std(daily_returns) if len(daily_returns) > 1 else 0
            sharpe_ratio = np.mean(daily_returns) / returns_std if returns_std > 0 else 0
            sharpe_ratio *= np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.portfolio.max_drawdown,
            'win_rate': portfolio_summary['trades']['win_rate'],
            'avg_win': portfolio_summary['trades']['avg_win'],
            'avg_loss': portfolio_summary['trades']['avg_loss'],
            'total_commission': portfolio_summary['trades']['total_commission']
        }
    
    def _calculate_execution_stats(self) -> Dict[str, float]:
        """Calculate execution statistics"""
        stats = {}
        
        for metric, values in self.execution_stats.items():
            if values:
                stats[f'avg_{metric}'] = np.mean(values)
                stats[f'std_{metric}'] = np.std(values)
                stats[f'max_{metric}'] = np.max(values)
        
        return stats
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze completed trades"""
        if not self.portfolio.closed_trades:
            return {}
        
        trades = [trade.to_dict() for trade in self.portfolio.closed_trades]
        df = pd.DataFrame(trades)
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(df[df['pnl'] > 0]),
            'losing_trades': len(df[df['pnl'] < 0]),
            'avg_duration_hours': df['duration_hours'].mean(),
            'longest_trade_hours': df['duration_hours'].max(),
            'shortest_trade_hours': df['duration_hours'].min(),
            'best_trade_pnl': df['pnl'].max(),
            'worst_trade_pnl': df['pnl'].min(),
            'avg_return_pct': df['return_pct'].mean() * 100
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        if not self.daily_pnl_history:
            return {}
        
        returns = [pnl for _, pnl in self.daily_pnl_history]
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) if len(returns) > 20 else 0
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            'var_95': var_95,
            'volatility': np.std(returns) * np.sqrt(252),  # Annualized
            'max_consecutive_losses': max_consecutive_losses,
            'downside_deviation': np.std([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': self.portfolio.get_portfolio_summary(),
            'market_data_symbols': list(self.market_data_buffer.keys()),
            'daily_loss_limit_hit': self.daily_loss_limit_hit,
            'execution_algorithm': self.config.execution_algorithm,
            'recent_execution_stats': {
                metric: values[-10:] if len(values) >= 10 else values
                for metric, values in self.execution_stats.items()
            }
        }


def test_paper_trading_engine():
    """Test paper trading engine functionality"""
    
    print("🎮 Testing Paper Trading Engine...")
    print("=" * 60)
    
    # Create configuration
    config = TradingEngineConfig(
        initial_capital=50000.0,
        execution_algorithm='vwap',
        max_positions=5,
        max_daily_loss=0.03
    )
    
    print(f"Paper Trading Configuration:")
    print(f"   Initial Capital: {config.initial_capital:,.0f} TL")
    print(f"   Execution Algorithm: {config.execution_algorithm}")
    print(f"   Max Positions: {config.max_positions}")
    print(f"   Max Daily Loss: {config.max_daily_loss:.1%}")
    
    # Initialize engine
    engine = PaperTradingEngine(config)
    
    # Create mock market data
    print(f"\nGenerating mock market data...")
    
    base_time = datetime.now()
    market_data = []
    
    base_price = 10.0
    for i in range(100):
        # Simple random walk for price
        price_change = np.random.normal(0, 0.02)  # 2% daily volatility
        price = base_price * (1 + price_change)
        
        data = MarketData(
            timestamp=base_time + timedelta(minutes=i * 5),
            symbol='TEST',
            open_price=base_price,
            high_price=max(base_price, price),
            low_price=min(base_price, price),
            close_price=price,
            volume=np.random.uniform(100000, 500000),
            bid_price=price * 0.999,
            ask_price=price * 1.001
        )
        
        market_data.append(data)
        base_price = price
    
    print(f"   Generated {len(market_data)} data points")
    print(f"   Price range: {min(d.close_price for d in market_data):.2f} - {max(d.close_price for d in market_data):.2f}")
    
    # Add market data to engine
    print(f"\nAdding market data to engine...")
    for data in market_data:
        engine.add_market_data(data)
    
    # Create and execute test signals
    print(f"\nTesting signal execution...")
    
    test_signals = [
        TradingSignal(
            symbol='TEST',
            timestamp=datetime.now(),
            action=SignalAction.BUY,
            confidence=0.8,
            expected_return=0.025,
            stop_loss=0.02,
            take_profit=0.04
        ),
        TradingSignal(
            symbol='TEST',
            timestamp=datetime.now() + timedelta(minutes=30),
            action=SignalAction.SELL,
            confidence=0.7,
            expected_return=-0.02,
            stop_loss=0.015,
            take_profit=0.03
        )
    ]
    
    # Execute signals
    async def execute_test_signals():
        results = []
        for signal in test_signals:
            result = await engine.execute_signal(signal)
            results.append(result)
            print(f"   Signal {signal.action.value}: {'✅' if result['success'] else '❌'} "
                  f"({result.get('reason', 'Success')})")
            
            if result['success']:
                print(f"     Entry Price: {result['entry_price']:.4f}")
                print(f"     Position Size: {result['position_size']:.2f}")
                print(f"     Market Impact: {result.get('market_impact', 0):.4f}")
                print(f"     Algorithm: {result.get('execution_algorithm', 'unknown')}")
        
        return results
    
    import asyncio
    execution_results = asyncio.run(execute_test_signals())
    
    # Show current status
    print(f"\nCurrent Engine Status:")
    status = engine.get_current_status()
    portfolio = status['portfolio_summary']
    
    print(f"   Portfolio Value: {portfolio['capital']['current_value']:,.2f} TL")
    print(f"   Available Cash: {portfolio['capital']['available_cash']:,.2f} TL")
    print(f"   Total Return: {portfolio['capital']['total_return_pct']:.2f}%")
    print(f"   Open Positions: {portfolio['positions']['count']}")
    print(f"   Completed Trades: {portfolio['trades']['total_count']}")
    
    # Test execution statistics
    if engine.execution_stats:
        print(f"\nExecution Statistics:")
        for metric, values in engine.execution_stats.items():
            if values:
                print(f"   {metric}: avg={np.mean(values):.4f}, max={np.max(values):.4f}")
    
    print(f"\n✅ Paper Trading Engine test completed!")
    
    return engine


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_paper_trading_engine()
