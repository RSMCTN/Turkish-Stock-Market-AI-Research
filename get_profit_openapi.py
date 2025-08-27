#!/usr/bin/env python3
"""
Get and analyze profit.com OpenAPI specification
"""

import requests
import json
from datetime import datetime

PROFIT_API_KEY = "a9a0bacbab08493d958244c05380da01"

def get_openapi_spec():
    print("📋 FETCHING PROFIT.COM OPENAPI SPECIFICATION...")
    print("=" * 60)
    
    try:
        # Get the OpenAPI specification
        response = requests.get("https://api.profit.com/openapi.json", timeout=10)
        
        if response.status_code == 200:
            print("✅ OpenAPI specification loaded successfully")
            
            # Parse JSON
            spec = response.json()
            
            # Save to file for analysis
            with open("profit_openapi_spec.json", "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
            
            print("💾 Full specification saved to: profit_openapi_spec.json")
            
            # Analyze the specification
            analyze_openapi_spec(spec)
            
        else:
            print(f"❌ Failed to load OpenAPI spec: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"💥 Error fetching OpenAPI spec: {str(e)}")

def analyze_openapi_spec(spec):
    """Analyze the OpenAPI specification"""
    print("\n🔍 ANALYZING OPENAPI SPECIFICATION...")
    print("-" * 50)
    
    # Basic info
    if 'info' in spec:
        info = spec['info']
        print(f"📖 Title: {info.get('title', 'N/A')}")
        print(f"📖 Version: {info.get('version', 'N/A')}")
        print(f"📖 Description: {info.get('description', 'N/A')[:100]}...")
    
    # Servers
    if 'servers' in spec:
        print(f"\n🌐 SERVERS:")
        for server in spec['servers']:
            print(f"   🔗 {server.get('url', 'N/A')} - {server.get('description', 'No description')}")
    
    # Security schemes
    if 'components' in spec and 'securitySchemes' in spec['components']:
        print(f"\n🔑 AUTHENTICATION METHODS:")
        for scheme_name, scheme in spec['components']['securitySchemes'].items():
            print(f"   🔐 {scheme_name}: {scheme.get('type', 'unknown')} - {scheme.get('description', 'No description')}")
            if 'name' in scheme:
                print(f"      Header: {scheme['name']}")
    
    # Paths (endpoints)
    if 'paths' in spec:
        print(f"\n📍 AVAILABLE ENDPOINTS: ({len(spec['paths'])} total)")
        
        stock_endpoints = []
        market_endpoints = []
        other_endpoints = []
        
        for path, methods in spec['paths'].items():
            # Categorize endpoints
            path_lower = path.lower()
            if 'stock' in path_lower or 'symbol' in path_lower or 'equity' in path_lower:
                stock_endpoints.append((path, list(methods.keys())))
            elif 'market' in path_lower or 'index' in path_lower:
                market_endpoints.append((path, list(methods.keys())))
            else:
                other_endpoints.append((path, list(methods.keys())))
        
        # Display stock-related endpoints (most important for us)
        if stock_endpoints:
            print(f"\n📈 STOCK/SYMBOL ENDPOINTS: ({len(stock_endpoints)} found)")
            for path, methods in stock_endpoints:
                print(f"   📊 {path} ({', '.join(methods)})")
                
                # Check for parameters in GET method
                if 'get' in methods:
                    get_info = spec['paths'][path]['get']
                    if 'parameters' in get_info:
                        params = [p['name'] for p in get_info['parameters'] if 'name' in p]
                        if params:
                            print(f"      🔧 Parameters: {', '.join(params)}")
        
        # Display market endpoints
        if market_endpoints:
            print(f"\n🏛️ MARKET ENDPOINTS: ({len(market_endpoints)} found)")
            for path, methods in market_endpoints:
                print(f"   📊 {path} ({', '.join(methods)})")
        
        # Display other endpoints
        if other_endpoints:
            print(f"\n🔧 OTHER ENDPOINTS: ({len(other_endpoints)} found)")
            for path, methods in other_endpoints[:10]:  # Show first 10
                print(f"   📊 {path} ({', '.join(methods)})")
            if len(other_endpoints) > 10:
                print(f"   ... and {len(other_endpoints) - 10} more")
    
    # Look for Turkey/BIST specific information
    spec_text = json.dumps(spec).lower()
    turkey_keywords = ['turkey', 'bist', 'istanbul', 'turkish', 'tr', 'garan']
    found_turkey = [kw for kw in turkey_keywords if kw in spec_text]
    
    if found_turkey:
        print(f"\n🇹🇷 TURKEY/BIST KEYWORDS FOUND: {', '.join(found_turkey)}")
    
    # Test some promising endpoints
    test_promising_endpoints(spec)

def test_promising_endpoints(spec):
    """Test the most promising endpoints for our use case"""
    print(f"\n🎯 TESTING PROMISING ENDPOINTS...")
    print("-" * 40)
    
    if 'paths' in spec:
        # Find the most relevant endpoints for stock data
        promising_paths = []
        
        for path, methods in spec['paths'].items():
            path_lower = path.lower()
            if ('stock' in path_lower or 'symbol' in path_lower or 
                'market' in path_lower or 'quote' in path_lower):
                if 'get' in methods:
                    promising_paths.append(path)
        
        # Test top 5 most promising endpoints
        for path in promising_paths[:5]:
            test_single_endpoint(path)

def test_single_endpoint(path):
    """Test a single endpoint with our API key"""
    headers_to_try = [
        {"X-API-Key": PROFIT_API_KEY},
        {"Authorization": f"Bearer {PROFIT_API_KEY}"},
        {"Authorization": f"ApiKey {PROFIT_API_KEY}"}
    ]
    
    for headers in headers_to_try:
        try:
            url = f"https://api.profit.com{path}"
            response = requests.get(url, headers=headers, timeout=5)
            
            auth_method = list(headers.keys())[0]
            print(f"🎯 {path} ({auth_method}): {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ SUCCESS! JSON Response: {str(data)[:150]}...")
                    return True  # Success, no need to try other auth methods
                except:
                    print(f"   📄 Non-JSON response: {response.text[:100]}...")
                    return True
            elif response.status_code == 401:
                print(f"   🔐 Auth failed with {auth_method}")
            elif response.status_code == 400:
                print(f"   ⚠️ Bad request - might need parameters")
            elif response.status_code != 404:
                print(f"   ℹ️ Status {response.status_code}: {response.text[:50]}...")
                
        except requests.exceptions.RequestException as e:
            continue
    
    return False

if __name__ == "__main__":
    get_openapi_spec()
