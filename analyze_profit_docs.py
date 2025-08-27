#!/usr/bin/env python3
"""
Analyze profit.com API documentation
"""

import requests
from bs4 import BeautifulSoup
import re
import json

PROFIT_API_KEY = "a9a0bacbab08493d958244c05380da01"

def analyze_profit_documentation():
    print("📚 ANALYZING PROFIT.COM API DOCUMENTATION...")
    print("=" * 60)
    
    try:
        # Get the documentation page
        response = requests.get("https://api.profit.com", timeout=10)
        
        if response.status_code == 200:
            print("✅ Documentation page loaded successfully")
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            if title:
                print(f"📖 Title: {title.text}")
            
            # Look for API endpoints in the documentation
            print("\n🔍 SEARCHING FOR API ENDPOINTS...")
            
            # Search for common API patterns
            text_content = response.text.lower()
            
            # Find potential endpoints
            endpoint_patterns = [
                r'/api/[a-zA-Z0-9/_-]+',
                r'/v[0-9]/[a-zA-Z0-9/_-]+', 
                r'https://api\.profit\.com/[a-zA-Z0-9/_-]+',
                r'"[a-zA-Z0-9/_-]*stocks[a-zA-Z0-9/_-]*"',
                r'"[a-zA-Z0-9/_-]*market[a-zA-Z0-9/_-]*"',
                r'"[a-zA-Z0-9/_-]*symbol[a-zA-Z0-9/_-]*"'
            ]
            
            found_endpoints = set()
            for pattern in endpoint_patterns:
                matches = re.findall(pattern, text_content)
                for match in matches:
                    found_endpoints.add(match.strip('"'))
            
            if found_endpoints:
                print("📍 POTENTIAL ENDPOINTS FOUND:")
                for endpoint in sorted(found_endpoints):
                    print(f"   • {endpoint}")
            
            # Look for authentication info
            print("\n🔑 SEARCHING FOR AUTHENTICATION INFO...")
            auth_keywords = ['api-key', 'token', 'bearer', 'auth', 'authorization']
            for keyword in auth_keywords:
                if keyword in text_content:
                    print(f"   🔍 Found '{keyword}' in documentation")
            
            # Look for BIST/Turkey specific content
            print("\n🇹🇷 SEARCHING FOR BIST/TURKEY CONTENT...")
            turkey_keywords = ['turkey', 'bist', 'istanbul', 'turkish', 'tl', 'garan', 'akbnk']
            found_turkey = []
            for keyword in turkey_keywords:
                if keyword in text_content:
                    found_turkey.append(keyword)
            
            if found_turkey:
                print(f"   ✅ Turkey-related keywords found: {', '.join(found_turkey)}")
            else:
                print("   ⚠️  No Turkey-specific content found in documentation")
            
            # Try to extract any JSON examples
            print("\n📊 SEARCHING FOR JSON EXAMPLES...")
            json_matches = re.findall(r'\{[^{}]*"[^"]*"[^{}]*\}', text_content)
            for i, match in enumerate(json_matches[:3]):  # Show first 3 JSON examples
                print(f"   📄 JSON Example {i+1}: {match[:100]}...")
            
            # Look for specific API keys or formats
            print("\n🔍 SEARCHING FOR API FORMATS...")
            if 'x-api-key' in text_content:
                print("   📋 Uses X-API-Key header format")
            if 'bearer' in text_content:
                print("   📋 Uses Bearer token format") 
            if 'authorization' in text_content:
                print("   📋 Uses Authorization header")
            
            # Save full documentation for manual review
            with open("profit_api_docs.html", "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"\n💾 Full documentation saved to: profit_api_docs.html")
            
            # Try some educated guesses for endpoints
            test_educated_guesses()
            
        else:
            print(f"❌ Failed to load documentation: {response.status_code}")
            
    except Exception as e:
        print(f"💥 Error analyzing documentation: {str(e)}")

def test_educated_guesses():
    """Test some educated guesses for API endpoints"""
    print("\n🎯 TESTING EDUCATED GUESSES...")
    print("-" * 40)
    
    # Common API endpoint patterns for financial data
    educated_endpoints = [
        "/stocks/search?q=GARAN",
        "/symbols/GARAN",
        "/market/BIST", 
        "/stocks/GARAN.IS",
        "/equities/TR/GARAN",
        "/data/stocks/GARAN",
        "/v1/stocks/GARAN",
        "/market-data/stocks/GARAN"
    ]
    
    headers = {
        "X-API-Key": PROFIT_API_KEY,
        "Authorization": f"Bearer {PROFIT_API_KEY}",
        "User-Agent": "MAMUT_R600/1.0"
    }
    
    for endpoint in educated_endpoints:
        try:
            url = f"https://api.profit.com{endpoint}"
            
            # Try both header formats
            for auth_header in [{"X-API-Key": PROFIT_API_KEY}, {"Authorization": f"Bearer {PROFIT_API_KEY}"}]:
                response = requests.get(url, headers=auth_header, timeout=5)
                
                print(f"🎯 {endpoint}: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"   ✅ SUCCESS! Found working endpoint")
                    try:
                        data = response.json()
                        print(f"   📊 JSON Response: {str(data)[:200]}...")
                        return  # Stop on first success
                    except:
                        print(f"   📄 Non-JSON: {response.text[:100]}...")
                elif response.status_code == 401:
                    print(f"   🔐 Authentication required")
                elif response.status_code != 404:
                    print(f"   ⚠️  Status: {response.status_code}")
                
        except requests.exceptions.RequestException:
            continue

if __name__ == "__main__":
    analyze_profit_documentation()
