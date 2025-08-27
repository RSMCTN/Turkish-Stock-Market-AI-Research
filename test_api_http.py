"""
HTTP API Test Script
Tests the running FastAPI server via HTTP requests
"""

import requests
import json
import time
from datetime import datetime

def test_api_endpoints():
    """Test API endpoints via HTTP"""
    
    base_url = "http://127.0.0.1:8000"
    
    print("🌐 TESTING API VIA HTTP REQUESTS")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    
    # Wait for server to start
    print("\n⏳ Waiting for server to start...")
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ Server is running (attempt {attempt + 1})")
                break
        except requests.exceptions.RequestException:
            import time as time_module
            time_module.sleep(2)
            if attempt == max_attempts - 1:
                print("   ❌ Server failed to start within timeout")
                return
    
    # Test endpoints
    endpoints = [
        ("GET", "/", "Root endpoint"),
        ("GET", "/health", "Health check"),
        ("GET", "/metrics/system", "System metrics"),
        ("GET", "/portfolio/summary", "Portfolio summary"),
        ("GET", "/portfolio/positions", "Current positions"),
        ("GET", "/portfolio/trades", "Trade history")
    ]
    
    results = []
    
    for method, path, description in endpoints:
        print(f"\n🔍 Testing {description}...")
        print(f"   {method} {path}")
        
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{path}", timeout=10)
            else:
                continue  # Skip non-GET for now
            
            print(f"   Status: {response.status_code}")
            print(f"   Response time: {response.elapsed.total_seconds():.3f}s")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ Success")
                    
                    # Show key info for different endpoints
                    if path == "/":
                        print(f"      Message: {data.get('message', 'N/A')}")
                        print(f"      Version: {data.get('version', 'N/A')}")
                        print(f"      Uptime: {data.get('uptime_seconds', 0):.1f}s")
                    
                    elif path == "/health":
                        print(f"      Status: {data.get('status', 'unknown')}")
                        components = data.get('components', {})
                        for comp, status in components.items():
                            print(f"      {comp}: {status}")
                    
                    elif path == "/metrics/system":
                        print(f"      CPU: {data.get('cpu_usage_pct', 0):.1f}%")
                        print(f"      Memory: {data.get('memory_usage_pct', 0):.1f}%")
                        print(f"      Requests: {data.get('api_requests_total', 0)}")
                    
                    elif path == "/portfolio/summary":
                        print(f"      Portfolio value: {data.get('current_value', 0):,.2f} TL")
                        print(f"      Total return: {data.get('total_return_pct', 0):.2f}%")
                        print(f"      Positions: {data.get('current_positions', 0)}")
                    
                    results.append(("✅", description, response.status_code, response.elapsed.total_seconds()))
                    
                except json.JSONDecodeError:
                    print(f"   ⚠️  Invalid JSON response")
                    results.append(("⚠️", description, response.status_code, response.elapsed.total_seconds()))
            else:
                print(f"   ❌ Error: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"      Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"      Raw response: {response.text[:200]}")
                results.append(("❌", description, response.status_code, response.elapsed.total_seconds()))
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            results.append(("❌", description, "N/A", "N/A"))
    
    # Test POST endpoint (signal generation)
    print(f"\n🔍 Testing Signal generation (POST)...")
    try:
        signal_data = {
            "symbol": "AKBNK",
            "include_features": True
        }
        
        response = requests.post(
            f"{base_url}/signals/generate",
            json=signal_data,
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {response.elapsed.total_seconds():.3f}s")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Signal generated successfully")
            print(f"      Symbol: {data.get('symbol')}")
            print(f"      Action: {data.get('action')}")
            print(f"      Confidence: {data.get('confidence', 0):.3f}")
            print(f"      Expected return: {data.get('expected_return', 0):.3f}")
            
            results.append(("✅", "Signal generation", response.status_code, response.elapsed.total_seconds()))
        else:
            print(f"   ❌ Error: HTTP {response.status_code}")
            results.append(("❌", "Signal generation", response.status_code, response.elapsed.total_seconds()))
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        results.append(("❌", "Signal generation", "N/A", "N/A"))
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for result in results if result[0] == "✅")
    total_count = len(results)
    
    print(f"Total endpoints tested: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success rate: {success_count/total_count:.1%}")
    
    print(f"\nDetailed results:")
    for status, endpoint, code, time in results:
        time_str = f"{time:.3f}s" if isinstance(time, (int, float)) else str(time)
        print(f"  {status} {endpoint:<25} | HTTP {code} | {time_str}")
    
    print(f"\n✅ HTTP API TEST COMPLETED!")
    print(f"Timestamp: {datetime.now()}")
    
    if success_count == total_count:
        print(f"\n🎉 ALL TESTS PASSED! API is fully operational.")
    else:
        print(f"\n⚠️  Some tests failed. Check the API configuration.")

if __name__ == "__main__":
    test_api_endpoints()
