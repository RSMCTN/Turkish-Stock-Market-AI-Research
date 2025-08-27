#!/usr/bin/env python3
"""
PROFIT.COM PRO API Test Script
Test profit.com PRO API for BIST stock data
"""

import requests
import json
from datetime import datetime

# Profit.com PRO API credentials
PROFIT_API_KEY = "a9a0bacbab08493d958244c05380da01"

# Common base URLs for financial APIs
POSSIBLE_BASE_URLS = [
    "https://api.profit.com/v1",
    "https://api.profit.com/v2", 
    "https://api.profit.com",
    "https://profit.com/api/v1",
    "https://profit.com/api"
]

# Common header formats
HEADER_FORMATS = [
    {"Authorization": f"Bearer {PROFIT_API_KEY}"},
    {"X-API-Key": PROFIT_API_KEY},
    {"API-Key": PROFIT_API_KEY},
    {"Authorization": f"Token {PROFIT_API_KEY}"},
    {"x-api-key": PROFIT_API_KEY}
]

# Common endpoint patterns
ENDPOINTS_TO_TEST = [
    "/stocks",
    "/symbols",
    "/markets", 
    "/stocks/search",
    "/market-data",
    "/equities",
    "/stocks/GARAN",  # Test with BIST stock
    "/stocks/BIST",
    "/turkey/stocks"
]

def test_profit_api():
    print("🔍 TESTING PROFIT.COM PRO API...")
    print("=" * 50)
    
    success_found = False
    
    # Try different base URLs and header formats
    for base_url in POSSIBLE_BASE_URLS:
        print(f"\n🌐 Testing base URL: {base_url}")
        
        for header_format in HEADER_FORMATS:
            print(f"   🔑 Header format: {list(header_format.keys())[0]}")
            
            # Test basic connectivity first
            try:
                response = requests.get(
                    base_url, 
                    headers=header_format,
                    timeout=10
                )
                
                print(f"   📡 Status: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"   ✅ SUCCESS! Base URL works")
                    print(f"   📄 Response preview: {response.text[:200]}...")
                    success_found = True
                    
                    # Test specific endpoints
                    for endpoint in ENDPOINTS_TO_TEST:
                        test_endpoint(base_url, endpoint, header_format)
                        
                elif response.status_code == 401:
                    print(f"   🔐 Authentication issue - trying next header format")
                elif response.status_code == 404:
                    print(f"   ❌ Not found - trying next base URL")
                else:
                    print(f"   ⚠️  Unexpected status: {response.text[:100]}")
                    
            except requests.exceptions.RequestException as e:
                print(f"   💥 Connection error: {str(e)[:50]}...")
            
            if success_found:
                break
        
        if success_found:
            break
    
    if not success_found:
        print("\n❌ Could not connect to profit.com API")
        print("🔍 Let's try alternative approaches...")
        test_alternative_approaches()

def test_endpoint(base_url, endpoint, headers):
    """Test a specific endpoint"""
    try:
        url = f"{base_url}{endpoint}"
        response = requests.get(url, headers=headers, timeout=5)
        
        print(f"      📍 {endpoint}: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"         📊 JSON data found: {len(str(data))} chars")
                
                # Look for BIST/Turkey data specifically
                response_text = str(data).lower()
                if 'garan' in response_text or 'bist' in response_text or 'turkey' in response_text:
                    print(f"         🇹🇷 TURKISH DATA FOUND!")
                    print(f"         📄 Sample: {str(data)[:150]}...")
                    
            except json.JSONDecodeError:
                print(f"         📄 Non-JSON response: {response.text[:100]}...")
        
    except requests.exceptions.RequestException:
        print(f"      📍 {endpoint}: Connection failed")

def test_alternative_approaches():
    """Try alternative API discovery approaches"""
    print("\n🔍 ALTERNATIVE APPROACHES:")
    
    # Try to find API documentation
    doc_urls = [
        "https://profit.com/api/docs",
        "https://profit.com/docs",
        "https://profit.com/api-docs",
        "https://docs.profit.com"
    ]
    
    for doc_url in doc_urls:
        try:
            response = requests.get(doc_url, timeout=5)
            print(f"📚 {doc_url}: {response.status_code}")
            if response.status_code == 200:
                print(f"   📖 Documentation found! Check manually")
        except:
            print(f"📚 {doc_url}: Failed")

if __name__ == "__main__":
    test_profit_api()
