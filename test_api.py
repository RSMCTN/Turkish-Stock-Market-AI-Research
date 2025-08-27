"""
Simple API test script
"""
import asyncio
import requests
from datetime import datetime

async def test_api():
    """Test API functionality"""
    
    print("🧪 TESTING BIST DP-LSTM TRADING SYSTEM API")
    print("=" * 60)
    
    # First test import
    print("\n1. Testing module imports...")
    try:
        import sys
        sys.path.append('src')
        from api.main import app, app_state
        print("   ✅ FastAPI app imported successfully")
        
        # Test startup event
        await app.router.startup()
        print("   ✅ Startup event completed")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return
    
    # Test individual functions
    print("\n2. Testing API functions...")
    
    try:
        from api.main import health_check, root, get_system_metrics
        
        # Test root endpoint
        root_result = await root()
        print(f"   ✅ Root endpoint: {root_result['message']}")
        
        # Test health check
        health_result = await health_check()
        print(f"   ✅ Health check: {health_result.status}")
        
        # Test metrics
        metrics_result = await get_system_metrics()
        print(f"   ✅ System metrics: CPU {metrics_result.cpu_usage_pct:.1f}%, Memory {metrics_result.memory_usage_pct:.1f}%")
        
    except Exception as e:
        print(f"   ❌ Function test failed: {e}")
    
    # Test signal generation
    print("\n3. Testing signal generation...")
    
    try:
        from api.main import generate_signal, SignalRequest
        
        request = SignalRequest(
            symbol="AKBNK",
            include_features=True
        )
        
        signal_result = await generate_signal(request)
        print(f"   ✅ Signal generated: {signal_result.symbol} - {signal_result.action}")
        print(f"      Confidence: {signal_result.confidence:.3f}, Expected return: {signal_result.expected_return:.3f}")
        
    except Exception as e:
        print(f"   ❌ Signal generation failed: {e}")
    
    # Test portfolio
    print("\n4. Testing portfolio management...")
    
    try:
        from api.main import get_portfolio_summary
        
        portfolio_result = await get_portfolio_summary()
        print(f"   ✅ Portfolio summary retrieved")
        print(f"      Value: {portfolio_result.current_value:,.2f} TL, Return: {portfolio_result.total_return_pct:.2f}%")
        print(f"      Positions: {portfolio_result.current_positions}, Trades: {portfolio_result.total_trades}")
        
    except Exception as e:
        print(f"   ❌ Portfolio test failed: {e}")
    
    print(f"\n✅ API FUNCTIONALITY TEST COMPLETED!")
    print(f"   Timestamp: {datetime.now()}")


if __name__ == "__main__":
    asyncio.run(test_api())
