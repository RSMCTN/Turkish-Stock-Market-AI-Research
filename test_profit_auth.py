#!/usr/bin/env python3
"""
Test different authentication methods for profit.com API
"""

import requests
import json

PROFIT_API_KEY = "a9a0bacbab08493d958244c05380da01"

def test_auth_methods():
    print("🔑 TESTING PROFIT.COM AUTHENTICATION METHODS...")
    print("=" * 60)
    
    # Test endpoint - stocks reference data
    base_url = "https://api.profit.com"
    endpoint = "/data-api/reference/stocks"
    
    # Different authentication methods to try
    auth_methods = [
        # Query parameters
        {
            "name": "Query: token",
            "url": f"{base_url}{endpoint}?token={PROFIT_API_KEY}&limit=5",
            "headers": {}
        },
        {
            "name": "Query: api_token", 
            "url": f"{base_url}{endpoint}?api_token={PROFIT_API_KEY}&limit=5",
            "headers": {}
        },
        {
            "name": "Query: api_key",
            "url": f"{base_url}{endpoint}?api_key={PROFIT_API_KEY}&limit=5", 
            "headers": {}
        },
        {
            "name": "Query: apikey",
            "url": f"{base_url}{endpoint}?apikey={PROFIT_API_KEY}&limit=5",
            "headers": {}
        },
        
        # Header methods
        {
            "name": "Header: X-Token",
            "url": f"{base_url}{endpoint}?limit=5",
            "headers": {"X-Token": PROFIT_API_KEY}
        },
        {
            "name": "Header: token",
            "url": f"{base_url}{endpoint}?limit=5", 
            "headers": {"token": PROFIT_API_KEY}
        },
        {
            "name": "Header: Authorization Token",
            "url": f"{base_url}{endpoint}?limit=5",
            "headers": {"Authorization": f"Token {PROFIT_API_KEY}"}
        },
        {
            "name": "Header: X-API-Token", 
            "url": f"{base_url}{endpoint}?limit=5",
            "headers": {"X-API-Token": PROFIT_API_KEY}
        }
    ]
    
    successful_method = None
    
    for method in auth_methods:
        try:
            print(f"\n🔍 Testing: {method['name']}")
            response = requests.get(
                method['url'], 
                headers=method['headers'],
                timeout=10
            )
            
            print(f"   📡 Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS! Authentication working")
                try:
                    data = response.json()
                    print(f"   📊 JSON Response: {len(str(data))} chars")
                    
                    # Look for stock data
                    if 'data' in data:
                        stocks = data['data']
                        print(f"   📈 Found {len(stocks)} stocks")
                        
                        # Show first stock
                        if stocks:
                            first_stock = stocks[0]
                            print(f"   📊 Sample stock: {first_stock.get('symbol', 'N/A')} - {first_stock.get('name', 'N/A')}")
                            
                            # Look for Turkey/BIST data
                            sample_text = str(first_stock).lower()
                            if 'turkey' in sample_text or 'tr' in sample_text or 'ist' in sample_text:
                                print(f"   🇹🇷 POTENTIAL TURKEY DATA FOUND!")
                    
                    successful_method = method
                    break
                    
                except json.JSONDecodeError:
                    print(f"   📄 Non-JSON response: {response.text[:100]}...")
                    successful_method = method
                    break
                    
            elif response.status_code == 401:
                print(f"   🔐 Authentication failed")
            elif response.status_code == 403:
                print(f"   🚫 Forbidden - token might be invalid")
                try:
                    error_data = response.json()
                    print(f"   💬 Error: {error_data.get('message', 'No message')}")
                except:
                    print(f"   💬 Raw error: {response.text[:100]}...")
            elif response.status_code == 429:
                print(f"   ⏰ Rate limited")
            else:
                print(f"   ⚠️ Status {response.status_code}: {response.text[:100]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"   💥 Connection error: {str(e)[:50]}...")
    
    if successful_method:
        print(f"\n🎉 SUCCESSFUL AUTHENTICATION FOUND!")
        print(f"   Method: {successful_method['name']}")
        test_turkey_specific_data(successful_method)
    else:
        print(f"\n❌ NO WORKING AUTHENTICATION METHOD FOUND")
        print(f"💡 Possible issues:")
        print(f"   • API Key might be invalid")
        print(f"   • API Key might be expired") 
        print(f"   • Account might need activation")
        print(f"   • Different endpoint might be needed")

def test_turkey_specific_data(auth_method):
    """Test for Turkey/BIST specific data once we have working auth"""
    print(f"\n🇹🇷 TESTING TURKEY/BIST SPECIFIC DATA...")
    print("-" * 40)
    
    # Try to find Turkish stocks
    base_url = "https://api.profit.com"
    
    # Test different approaches to find Turkey data
    test_endpoints = [
        # Search for country
        ("/data-api/reference/stocks?country=Turkey&limit=10", "Country filter: Turkey"),
        ("/data-api/reference/stocks?country=TR&limit=10", "Country filter: TR"),
        ("/data-api/reference/stocks?exchange=BIST&limit=10", "Exchange filter: BIST"),
        ("/data-api/reference/stocks?exchange=Istanbul&limit=10", "Exchange filter: Istanbul"),
        
        # Search for specific Turkish stocks
        ("/data-api/market-data/quote/GARAN.IS", "Turkish stock: GARAN.IS"),
        ("/data-api/market-data/quote/GARAN", "Turkish stock: GARAN"),
        ("/data-api/market-data/quote/AKBNK.IS", "Turkish stock: AKBNK.IS"),
        ("/data-api/market-data/quote/AKBNK", "Turkish stock: AKBNK"),
    ]
    
    found_turkey = False
    
    for endpoint, description in test_endpoints:
        try:
            if "?" in endpoint:
                # Already has parameters, add auth as additional parameter
                if "token" in auth_method['name'].lower() and "query" in auth_method['name'].lower():
                    # Extract token parameter from successful method
                    token_param = auth_method['url'].split('?')[1].split('&')[0]  # Get first parameter
                    url = f"{base_url}{endpoint}&{token_param}"
                else:
                    url = f"{base_url}{endpoint}"
            else:
                # No parameters, add auth
                if "token" in auth_method['name'].lower() and "query" in auth_method['name'].lower():
                    token_param = auth_method['url'].split('?')[1].split('&')[0]  # Get first parameter  
                    url = f"{base_url}{endpoint}?{token_param}"
                else:
                    url = f"{base_url}{endpoint}"
            
            print(f"🎯 {description}")
            
            response = requests.get(
                url,
                headers=auth_method.get('headers', {}),
                timeout=5
            )
            
            print(f"   📡 Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Check if we got Turkey data
                    data_str = str(data).lower()
                    if any(keyword in data_str for keyword in ['turkey', 'turkish', 'bist', 'istanbul', 'garan', 'akbnk']):
                        print(f"   🇹🇷 TURKEY DATA FOUND!")
                        print(f"   📊 Data: {str(data)[:200]}...")
                        found_turkey = True
                    else:
                        print(f"   📊 Response: {str(data)[:100]}...")
                        
                except json.JSONDecodeError:
                    print(f"   📄 Non-JSON: {response.text[:100]}...")
            elif response.status_code == 404:
                print(f"   ❌ Not found")
            else:
                print(f"   ⚠️ Status: {response.status_code}")
                
        except requests.exceptions.RequestException:
            print(f"   💥 Connection error")
    
    if found_turkey:
        print(f"\n✅ TURKEY/BIST DATA IS AVAILABLE!")
    else:
        print(f"\n⚠️ No Turkey/BIST specific data found yet")
        print(f"💡 May need to explore exchange codes or symbol formats")

if __name__ == "__main__":
    test_auth_methods()
