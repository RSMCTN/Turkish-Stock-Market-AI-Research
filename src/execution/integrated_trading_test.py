"""
Integrated Trading System Test
Tests the complete signal generation -> portfolio management -> execution flow
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_generator import (
    SignalGenerator, SignalGeneratorConfig, TradingSignal, SignalAction
)
from portfolio_manager import PortfolioManager, PortfolioConfig
from paper_trading_engine import (
    PaperTradingEngine, TradingEngineConfig, MarketData
)


class MockModel:
    """Mock model for testing signal generation"""
    
    def __init__(self, trend_direction: float = 0.0):
        self.trend_direction = trend_direction  # -1 (bearish) to 1 (bullish)
    
    def eval(self):
        """Set model to evaluation mode"""
        pass
    
    def __call__(self, features):
        """Generate mock predictions"""
        import torch
        
        # Generate realistic predictions based on trend
        if self.trend_direction > 0.5:  # Strong bullish
            direction_probs = [0.15, 0.25, 0.60]  # Up bias
            magnitude = np.random.uniform(0.015, 0.035)  # 1.5% to 3.5%
            confidence = np.random.uniform(0.7, 0.9)
        elif self.trend_direction < -0.5:  # Strong bearish
            direction_probs = [0.60, 0.25, 0.15]  # Down bias  
            magnitude = np.random.uniform(0.015, 0.035)
            confidence = np.random.uniform(0.7, 0.9)
        else:  # Neutral/weak trend
            direction_probs = [0.30, 0.40, 0.30]  # Neutral
            magnitude = np.random.uniform(0.005, 0.02)
            confidence = np.random.uniform(0.4, 0.65)
        
        return {
            'direction': torch.tensor([direction_probs]),
            'magnitude': torch.tensor([[magnitude]]),
            'volatility': torch.tensor([[np.random.uniform(0.1, 0.3)]]),
            'volume': torch.tensor([[np.random.uniform(0.8, 1.5)]]),
            'confidence': torch.tensor([confidence])
        }


class MockFeatureProcessor:
    """Mock feature processor for testing"""
    
    async def process_features(self, market_data, news_data):
        """Return mock features tensor"""
        import torch
        
        # Return realistic feature tensor shape
        return torch.randn(1, 60, 131)  # batch, sequence, features


async def run_integrated_trading_test():
    """Run comprehensive integrated trading system test"""
    
    print("🚀 INTEGRATED TRADING SYSTEM TEST")
    print("=" * 80)
    
    # =============================================================================
    # STEP 1: Initialize System Components
    # =============================================================================
    print("\n📋 Step 1: Initializing System Components...")
    
    # Signal generator configuration
    signal_config = SignalGeneratorConfig(
        buy_threshold=0.65,
        sell_threshold=0.65,
        min_expected_return=0.012,  # 1.2%
        max_signals_per_symbol=8,
        default_stop_loss_pct=0.025,
        default_take_profit_pct=0.06
    )
    
    # Trading engine configuration
    engine_config = TradingEngineConfig(
        initial_capital=100000.0,
        execution_algorithm='vwap',
        max_positions=8,
        max_daily_loss=0.04,
        market_impact_model='sqrt',
        slippage_model='adaptive'
    )
    
    print(f"   Signal Generator: buy/sell threshold = {signal_config.buy_threshold}")
    print(f"   Trading Engine: capital = {engine_config.initial_capital:,.0f} TL")
    print(f"   Max positions = {engine_config.max_positions}")
    
    # Initialize components
    mock_model = MockModel(trend_direction=0.7)  # Bullish bias for testing
    feature_processor = MockFeatureProcessor()
    
    signal_generator = SignalGenerator(
        model=mock_model,
        feature_processor=feature_processor,
        config=signal_config
    )
    
    paper_engine = PaperTradingEngine(engine_config)
    
    print("   ✅ All components initialized")
    
    # =============================================================================
    # STEP 2: Generate Market Data
    # =============================================================================
    print("\n📊 Step 2: Generating Realistic Market Data...")
    
    symbols = ['AKBNK', 'GARAN', 'ISCTR', 'TUPRS', 'ASELS']
    base_prices = {'AKBNK': 8.45, 'GARAN': 32.10, 'ISCTR': 22.50, 'TUPRS': 425.0, 'ASELS': 85.20}
    
    all_market_data = []
    base_time = datetime.now() - timedelta(hours=48)  # Start 48h ago
    
    for symbol in symbols:
        base_price = base_prices[symbol]
        symbol_data = []
        
        # Generate 480 data points (48 hours * 10 per hour = every 6 minutes)
        for i in range(480):
            timestamp = base_time + timedelta(minutes=i * 6)
            
            # Market hours check (simplified)
            hour = timestamp.hour
            if hour < 9 or hour > 17:
                continue  # Skip non-market hours
            
            # Random walk with trend
            trend_factor = 0.0002 * (i / 480)  # Gradual upward trend
            noise = np.random.normal(0, 0.008)  # 0.8% volatility
            price_change = trend_factor + noise
            
            new_price = base_price * (1 + price_change)
            
            # Create OHLC data
            high = new_price * (1 + abs(np.random.normal(0, 0.003)))
            low = new_price * (1 - abs(np.random.normal(0, 0.003)))
            
            data = MarketData(
                timestamp=timestamp,
                symbol=symbol,
                open_price=base_price,
                high_price=max(base_price, new_price, high),
                low_price=min(base_price, new_price, low),
                close_price=new_price,
                volume=np.random.uniform(100000, 800000),
                bid_price=new_price * 0.9995,
                ask_price=new_price * 1.0005
            )
            
            symbol_data.append(data)
            all_market_data.append(data)
            base_price = new_price
    
    # Sort by timestamp
    all_market_data.sort(key=lambda x: x.timestamp)
    
    print(f"   Generated {len(all_market_data)} market data points")
    print(f"   Symbols: {symbols}")
    print(f"   Time range: {all_market_data[0].timestamp} to {all_market_data[-1].timestamp}")
    
    # =============================================================================
    # STEP 3: Run Trading Simulation
    # =============================================================================
    print("\n⚡ Step 3: Running Trading Simulation...")
    
    signals_generated = []
    signals_executed = []
    simulation_log = []
    
    # Process data chronologically
    for i, data_point in enumerate(all_market_data):
        # Add data to engine
        paper_engine.add_market_data(data_point)
        
        # Generate signal every 30 minutes for each symbol
        if i % 5 == 0:  # Every 5th data point (approx 30 min)
            
            market_data_dict = {
                'symbol': data_point.symbol,
                'current_price': data_point.get_mid_price(),
                'volume_ratio': np.random.uniform(0.8, 1.5),
                'volatility_zscore': np.random.uniform(-2, 2),
                'conditions': {'market_open': True}
            }
            
            # Generate signal
            signal = await signal_generator.generate_signal(
                data_point.symbol,
                market_data_dict,
                []  # No news data for this test
            )
            
            signals_generated.append(signal)
            
            # Execute if actionable
            if signal.is_actionable():
                execution_result = await paper_engine.execute_signal(signal)
                
                if execution_result['success']:
                    signals_executed.append(signal)
                    simulation_log.append({
                        'timestamp': data_point.timestamp,
                        'symbol': signal.symbol,
                        'action': signal.action.value,
                        'confidence': signal.confidence,
                        'execution': execution_result
                    })
        
        # Progress indicator
        if i % 100 == 0:
            portfolio_value = paper_engine.portfolio.get_portfolio_value()
            print(f"   Progress: {i}/{len(all_market_data)} | "
                  f"Portfolio: {portfolio_value:,.0f} TL | "
                  f"Signals: {len(signals_executed)}")
    
    print(f"   ✅ Simulation completed")
    
    # =============================================================================
    # STEP 4: Analysis & Results
    # =============================================================================
    print("\n📈 Step 4: Performance Analysis...")
    
    # Get final portfolio status
    final_status = paper_engine.get_current_status()
    portfolio = final_status['portfolio_summary']
    
    # Calculate key metrics
    initial_capital = engine_config.initial_capital
    final_value = portfolio['capital']['current_value']
    total_return = (final_value - initial_capital) / initial_capital
    
    print(f"\n💰 FINANCIAL PERFORMANCE:")
    print(f"   Initial Capital:     {initial_capital:>12,.2f} TL")
    print(f"   Final Value:         {final_value:>12,.2f} TL")
    print(f"   Total Return:        {total_return:>12.2%}")
    print(f"   Available Cash:      {portfolio['capital']['available_cash']:>12,.2f} TL")
    print(f"   Max Drawdown:        {paper_engine.portfolio.max_drawdown:>12.2%}")
    
    print(f"\n📊 TRADING STATISTICS:")
    print(f"   Total Signals:       {len(signals_generated):>12}")
    print(f"   Executed Signals:    {len(signals_executed):>12}")
    print(f"   Execution Rate:      {len(signals_executed)/len(signals_generated):>12.1%}")
    print(f"   Completed Trades:    {portfolio['trades']['total_count']:>12}")
    print(f"   Open Positions:      {portfolio['positions']['count']:>12}")
    print(f"   Win Rate:            {portfolio['trades']['win_rate']:>12.1%}")
    
    if portfolio['trades']['total_count'] > 0:
        print(f"   Average Win:         {portfolio['trades']['avg_win']:>12.2f} TL")
        print(f"   Average Loss:        {portfolio['trades']['avg_loss']:>12.2f} TL")
        print(f"   Total Commission:    {portfolio['trades']['total_commission']:>12.2f} TL")
    
    print(f"\n🎯 EXECUTION ANALYSIS:")
    execution_stats = final_status.get('recent_execution_stats', {})
    for metric, values in execution_stats.items():
        if values:
            avg_val = np.mean(values)
            max_val = np.max(values)
            print(f"   {metric:20s}: avg={avg_val:8.4f}, max={max_val:8.4f}")
    
    print(f"\n📋 SYMBOL BREAKDOWN:")
    # Count signals by symbol
    symbol_signal_counts = {}
    symbol_executions = {}
    
    for signal in signals_generated:
        symbol_signal_counts[signal.symbol] = symbol_signal_counts.get(signal.symbol, 0) + 1
    
    for signal in signals_executed:
        symbol_executions[signal.symbol] = symbol_executions.get(signal.symbol, 0) + 1
    
    for symbol in symbols:
        signals_count = symbol_signal_counts.get(symbol, 0)
        exec_count = symbol_executions.get(symbol, 0)
        exec_rate = exec_count / signals_count if signals_count > 0 else 0
        print(f"   {symbol:8s}: {signals_count:3d} signals, {exec_count:3d} executed ({exec_rate:.1%})")
    
    # =============================================================================
    # STEP 5: System Health Check
    # =============================================================================
    print(f"\n🔍 Step 5: System Health Check...")
    
    health_issues = []
    
    # Check return vs expectations
    if total_return < -0.10:  # More than 10% loss
        health_issues.append("⚠️  High portfolio loss detected")
    
    # Check execution rate
    execution_rate = len(signals_executed) / len(signals_generated) if signals_generated else 0
    if execution_rate < 0.05:  # Less than 5% execution rate
        health_issues.append("⚠️  Very low signal execution rate")
    
    # Check position management
    if portfolio['positions']['count'] > engine_config.max_positions:
        health_issues.append("⚠️  Position limit exceeded")
    
    # Check commission costs
    if portfolio['trades']['total_count'] > 0:
        commission_pct = portfolio['trades']['total_commission'] / initial_capital
        if commission_pct > 0.02:  # More than 2% in commissions
            health_issues.append("⚠️  High commission costs")
    
    if not health_issues:
        print("   ✅ All system health checks passed")
    else:
        print("   System Health Issues:")
        for issue in health_issues:
            print(f"     {issue}")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    print(f"\n🏆 INTEGRATION TEST SUMMARY:")
    print("=" * 80)
    
    if total_return > 0:
        print("   📈 PROFITABLE: System generated positive returns")
    else:
        print("   📉 LOSS: System generated negative returns")
    
    if len(signals_executed) > 10:
        print("   🎯 ACTIVE: System executed multiple trading signals")
    else:
        print("   😴 PASSIVE: System executed few trading signals")
    
    if execution_rate > 0.1:
        print("   ⚡ RESPONSIVE: Good signal execution rate")
    else:
        print("   🐌 CONSERVATIVE: Low signal execution rate")
    
    grade = "A" if (total_return > 0.02 and execution_rate > 0.15 and not health_issues) else \
            "B" if (total_return > 0 and execution_rate > 0.1) else \
            "C" if (total_return > -0.05) else "D"
    
    print(f"\n   🎖️  OVERALL SYSTEM GRADE: {grade}")
    
    print("\n✅ INTEGRATED TRADING SYSTEM TEST COMPLETED!")
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'signals_generated': len(signals_generated),
        'signals_executed': len(signals_executed),
        'execution_rate': execution_rate,
        'completed_trades': portfolio['trades']['total_count'],
        'win_rate': portfolio['trades']['win_rate'],
        'grade': grade,
        'health_issues': health_issues
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during test
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive test
    result = asyncio.run(run_integrated_trading_test())
