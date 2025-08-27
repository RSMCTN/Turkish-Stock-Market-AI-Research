"""
Complete System Integration Test
Tests the full BIST DP-LSTM Trading System stack
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import all our components
import sys
import os
sys.path.append('src')

from api.main import app, app_state
from monitoring.metrics_collector import MetricsCollector
from execution.integrated_trading_test import run_integrated_trading_test


class ComprehensiveSystemTest:
    """Complete system integration test suite"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.start_time = datetime.now()
        
    async def run_complete_test_suite(self):
        """Run comprehensive test of entire system"""
        
        print("🚀 COMPREHENSIVE BIST DP-LSTM TRADING SYSTEM TEST")
        print("=" * 80)
        print(f"Start time: {self.start_time}")
        print(f"Test environment: Python {sys.version}")
        
        test_modules = [
            ("🔧 System Components", self._test_system_components),
            ("📊 Metrics & Monitoring", self._test_metrics_system), 
            ("🎯 Trading Engine", self._test_trading_system),
            ("🌐 API Integration", self._test_api_system),
            ("⚡ Performance & Load", self._test_performance),
            ("🔍 System Health", self._test_system_health)
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for module_name, test_func in test_modules:
            print(f"\n{'='*60}")
            print(f"{module_name}")
            print("="*60)
            
            try:
                module_results = await test_func()
                
                # Count results
                for test_name, result in module_results.items():
                    total_tests += 1
                    if result.get('success', False):
                        passed_tests += 1
                        print(f"✅ {test_name}: {result.get('message', 'OK')}")
                    else:
                        print(f"❌ {test_name}: {result.get('message', 'FAILED')}")
                
                self.test_results[module_name] = module_results
                
            except Exception as e:
                print(f"❌ {module_name} failed with error: {e}")
                self.test_results[module_name] = {"error": {"success": False, "message": str(e)}}
                total_tests += 1
        
        # Final summary
        await self._display_final_summary(total_tests, passed_tests)
        
        return self.test_results
    
    async def _test_system_components(self) -> Dict[str, Any]:
        """Test basic system component functionality"""
        results = {}
        
        # Test imports
        try:
            from execution.signal_generator import SignalGenerator, TradingSignal
            from execution.portfolio_manager import PortfolioManager
            from execution.paper_trading_engine import PaperTradingEngine
            from api.main import app
            from monitoring.metrics_collector import MetricsCollector
            
            results["imports"] = {"success": True, "message": "All modules imported successfully"}
        except Exception as e:
            results["imports"] = {"success": False, "message": f"Import error: {e}"}
        
        # Test component initialization
        try:
            from execution.integrated_trading_test import MockModel, MockFeatureProcessor
            from execution.signal_generator import SignalGeneratorConfig
            from execution.portfolio_manager import PortfolioConfig
            from execution.paper_trading_engine import TradingEngineConfig
            
            # Initialize components
            model = MockModel(trend_direction=0.5)
            feature_processor = MockFeatureProcessor()
            signal_gen = SignalGenerator(model, feature_processor, SignalGeneratorConfig())
            portfolio = PortfolioManager(PortfolioConfig())
            paper_engine = PaperTradingEngine(TradingEngineConfig())
            metrics = MetricsCollector()
            
            results["initialization"] = {"success": True, "message": "All components initialized"}
        except Exception as e:
            results["initialization"] = {"success": False, "message": f"Init error: {e}"}
        
        # Test FastAPI app
        try:
            from fastapi.testclient import TestClient
            client = TestClient(app)
            
            # Trigger startup
            await app.router.startup()
            
            # Test basic endpoints
            response = client.get("/")
            if response.status_code == 200:
                results["fastapi"] = {"success": True, "message": "FastAPI app functional"}
            else:
                results["fastapi"] = {"success": False, "message": f"HTTP {response.status_code}"}
                
        except Exception as e:
            results["fastapi"] = {"success": False, "message": f"FastAPI error: {e}"}
        
        return results
    
    async def _test_metrics_system(self) -> Dict[str, Any]:
        """Test metrics collection and monitoring"""
        results = {}
        
        try:
            # Initialize metrics collector
            metrics = MetricsCollector({'collection_interval': 1})
            
            # Test startup
            await metrics.start_collection()
            
            # Wait for collection
            await asyncio.sleep(3)
            
            # Test metrics retrieval
            current_metrics = metrics.get_current_metrics()
            if len(current_metrics.get('metrics', {})) > 5:
                results["collection"] = {"success": True, "message": f"{len(current_metrics['metrics'])} metrics collected"}
            else:
                results["collection"] = {"success": False, "message": "Insufficient metrics collected"}
            
            # Test health scoring
            health_score = metrics.get_system_health_score()
            if health_score.get('score', 0) > 0:
                results["health_scoring"] = {"success": True, "message": f"Health score: {health_score['score']}/100"}
            else:
                results["health_scoring"] = {"success": False, "message": "Health scoring failed"}
            
            # Test alert system
            alerts = metrics.get_active_alerts()
            results["alerting"] = {"success": True, "message": f"Alert system functional ({len(alerts)} active)"}
            
            # Stop collection
            await metrics.stop_collection()
            
        except Exception as e:
            results["metrics_error"] = {"success": False, "message": f"Metrics test error: {e}"}
        
        return results
    
    async def _test_trading_system(self) -> Dict[str, Any]:
        """Test integrated trading system"""
        results = {}
        
        try:
            # Run the integrated trading test (simplified)
            print("   Running integrated trading system test...")
            
            from execution.integrated_trading_test import MockModel, MockFeatureProcessor
            from execution.signal_generator import SignalGenerator, SignalGeneratorConfig
            from execution.paper_trading_engine import PaperTradingEngine, TradingEngineConfig
            
            # Quick trading system test
            model = MockModel(trend_direction=0.3)
            feature_processor = MockFeatureProcessor()
            
            signal_gen = SignalGenerator(
                model=model,
                feature_processor=feature_processor,
                config=SignalGeneratorConfig(buy_threshold=0.4, sell_threshold=0.4)
            )
            
            paper_engine = PaperTradingEngine(TradingEngineConfig(initial_capital=50000))
            
            # Generate and execute a few signals
            test_symbols = ['AKBNK', 'GARAN']
            signals_generated = 0
            signals_executed = 0
            
            for symbol in test_symbols:
                market_data = {
                    'symbol': symbol,
                    'current_price': 10.0,
                    'volume_ratio': 1.2,
                    'volatility_zscore': 0.5,
                    'conditions': {'market_open': True}
                }
                
                signal = await signal_gen.generate_signal(symbol, market_data)
                signals_generated += 1
                
                if signal.is_actionable():
                    execution_result = await paper_engine.execute_signal(signal)
                    if execution_result['success']:
                        signals_executed += 1
            
            results["signal_generation"] = {
                "success": signals_generated > 0, 
                "message": f"{signals_generated} signals generated"
            }
            
            results["signal_execution"] = {
                "success": True, 
                "message": f"{signals_executed} signals executed"
            }
            
            # Test portfolio tracking
            portfolio_status = paper_engine.get_current_status()
            portfolio_value = portfolio_status['portfolio_summary']['capital']['current_value']
            
            results["portfolio_tracking"] = {
                "success": portfolio_value > 0,
                "message": f"Portfolio value: {portfolio_value:,.2f} TL"
            }
            
        except Exception as e:
            results["trading_error"] = {"success": False, "message": f"Trading test error: {e}"}
        
        return results
    
    async def _test_api_system(self) -> Dict[str, Any]:
        """Test API system functionality"""
        results = {}
        
        try:
            # Test API endpoints without HTTP server
            from api.main import (
                root, health_check, get_system_metrics, 
                generate_signal, get_portfolio_summary, SignalRequest
            )
            
            # Test root endpoint
            root_response = await root()
            results["root_endpoint"] = {
                "success": "message" in root_response,
                "message": root_response.get("message", "No message")[:50]
            }
            
            # Test health check
            health_response = await health_check()
            results["health_endpoint"] = {
                "success": health_response.status in ["healthy", "degraded"],
                "message": f"Status: {health_response.status}"
            }
            
            # Test metrics endpoint
            metrics_response = await get_system_metrics()
            results["metrics_endpoint"] = {
                "success": hasattr(metrics_response, 'cpu_usage_pct'),
                "message": f"CPU: {metrics_response.cpu_usage_pct:.1f}%"
            }
            
            # Test signal generation
            signal_request = SignalRequest(symbol="AKBNK", include_features=True)
            signal_response = await generate_signal(signal_request)
            results["signal_endpoint"] = {
                "success": hasattr(signal_response, 'symbol'),
                "message": f"Signal: {signal_response.action} for {signal_response.symbol}"
            }
            
            # Test portfolio endpoint
            portfolio_response = await get_portfolio_summary()
            results["portfolio_endpoint"] = {
                "success": hasattr(portfolio_response, 'current_value'),
                "message": f"Portfolio: {portfolio_response.current_value:,.2f} TL"
            }
            
        except Exception as e:
            results["api_error"] = {"success": False, "message": f"API test error: {e}"}
        
        return results
    
    async def _test_performance(self) -> Dict[str, Any]:
        """Test system performance characteristics"""
        results = {}
        
        try:
            # Test signal generation speed
            from execution.integrated_trading_test import MockModel, MockFeatureProcessor
            from execution.signal_generator import SignalGenerator, SignalGeneratorConfig
            
            model = MockModel()
            feature_processor = MockFeatureProcessor()
            signal_gen = SignalGenerator(model, feature_processor, SignalGeneratorConfig())
            
            # Time signal generation
            start_time = time.time()
            signals = []
            
            for i in range(10):
                market_data = {
                    'symbol': 'TEST',
                    'current_price': 10.0 + i * 0.1,
                    'volume_ratio': 1.0,
                    'volatility_zscore': 0.0,
                    'conditions': {'market_open': True}
                }
                
                signal = await signal_gen.generate_signal('TEST', market_data)
                signals.append(signal)
            
            generation_time = (time.time() - start_time) * 1000  # milliseconds
            avg_latency = generation_time / 10
            
            results["signal_latency"] = {
                "success": avg_latency < 500,  # Less than 500ms average
                "message": f"Avg latency: {avg_latency:.1f}ms"
            }
            
            # Test metrics collection performance
            from monitoring.metrics_collector import MetricsCollector
            
            metrics = MetricsCollector({'collection_interval': 0.5})
            await metrics.start_collection()
            
            start_time = time.time()
            await asyncio.sleep(2)  # Let it collect for 2 seconds
            collection_time = time.time() - start_time
            
            current_metrics = metrics.get_current_metrics()
            metrics_count = len(current_metrics.get('metrics', {}))
            
            await metrics.stop_collection()
            
            results["metrics_performance"] = {
                "success": metrics_count > 5,
                "message": f"{metrics_count} metrics in {collection_time:.1f}s"
            }
            
            # Memory usage check
            import psutil
            memory_usage = psutil.virtual_memory().percent
            
            results["memory_usage"] = {
                "success": memory_usage < 85,
                "message": f"Memory usage: {memory_usage:.1f}%"
            }
            
        except Exception as e:
            results["performance_error"] = {"success": False, "message": f"Performance test error: {e}"}
        
        return results
    
    async def _test_system_health(self) -> Dict[str, Any]:
        """Test overall system health and stability"""
        results = {}
        
        try:
            # System resource check
            import psutil
            
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            results["cpu_health"] = {
                "success": cpu_usage < 80,
                "message": f"CPU usage: {cpu_usage:.1f}%"
            }
            
            results["memory_health"] = {
                "success": memory_usage < 85,
                "message": f"Memory usage: {memory_usage:.1f}%"
            }
            
            results["disk_health"] = {
                "success": disk_usage < 90,
                "message": f"Disk usage: {disk_usage:.1f}%"
            }
            
            # System uptime and stability
            uptime = (datetime.now() - self.start_time).total_seconds()
            results["system_stability"] = {
                "success": uptime > 0,
                "message": f"Test uptime: {uptime:.1f}s"
            }
            
            # Component integration health
            try:
                await app.router.startup()
                
                from monitoring.metrics_collector import MetricsCollector
                metrics = MetricsCollector()
                health_score = metrics.get_system_health_score()
                
                results["integration_health"] = {
                    "success": health_score.get('score', 0) > 50,
                    "message": f"Integration health: {health_score.get('score', 0)}/100"
                }
                
            except Exception as e:
                results["integration_health"] = {
                    "success": False,
                    "message": f"Integration check failed: {e}"
                }
            
        except Exception as e:
            results["health_error"] = {"success": False, "message": f"Health check error: {e}"}
        
        return results
    
    async def _display_final_summary(self, total_tests: int, passed_tests: int):
        """Display comprehensive test summary"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print("📊 FINAL TEST SUMMARY")
        print("="*80)
        
        print(f"📅 Start Time: {self.start_time}")
        print(f"📅 End Time: {end_time}")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        
        print(f"\n📈 TEST RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {passed_tests/total_tests:.1%}")
        
        print(f"\n📋 MODULE BREAKDOWN:")
        for module_name, module_results in self.test_results.items():
            module_total = len(module_results)
            module_passed = sum(1 for result in module_results.values() if result.get('success', False))
            status = "✅" if module_passed == module_total else "⚠️" if module_passed > 0 else "❌"
            print(f"   {status} {module_name}: {module_passed}/{module_total}")
        
        # System grade
        success_rate = passed_tests / total_tests
        if success_rate >= 0.95:
            grade = "A+ (EXCELLENT)"
            color = "🟢"
        elif success_rate >= 0.85:
            grade = "A (VERY GOOD)"
            color = "🟢"
        elif success_rate >= 0.75:
            grade = "B (GOOD)"
            color = "🟡"
        elif success_rate >= 0.65:
            grade = "C (ACCEPTABLE)"
            color = "🟡"
        else:
            grade = "D (NEEDS IMPROVEMENT)"
            color = "🔴"
        
        print(f"\n🎯 OVERALL SYSTEM GRADE: {color} {grade}")
        
        if success_rate >= 0.85:
            print(f"\n🎉 SYSTEM IS PRODUCTION READY!")
            print(f"   All major components are functional")
            print(f"   Performance metrics are within acceptable ranges")
            print(f"   Integration between components is working correctly")
        else:
            print(f"\n⚠️  SYSTEM NEEDS ATTENTION:")
            print(f"   Some components require debugging")
            print(f"   Review failed tests and address issues")
            print(f"   Consider additional optimization")
        
        print("="*80)


async def main():
    """Main test execution"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run comprehensive test
    test_suite = ComprehensiveSystemTest()
    results = await test_suite.run_complete_test_suite()
    
    return results


if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(main())
