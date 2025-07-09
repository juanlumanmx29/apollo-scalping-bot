#!/usr/bin/env python3
"""
Debug script to test Binance API from Railway environment
"""
import requests
import json
import time

def test_binance_endpoints():
    """Test all Binance endpoints and report results"""
    
    # Test simple price endpoint
    print("🧪 Testing simple price endpoint...")
    price_urls = [
        "https://api.binance.com/api/v3/ticker/price",
        "https://api1.binance.com/api/v3/ticker/price",
        "https://api2.binance.com/api/v3/ticker/price"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for url in price_urls:
        try:
            response = requests.get(url, params={"symbol": "ETHUSDT"}, headers=headers, timeout=10)
            print(f"✅ {url}: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"❌ {url}: ERROR - {e}")
    
    print("\n🧪 Testing klines endpoint...")
    klines_urls = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines"
    ]
    
    params = {
        "symbol": "ETHUSDT",
        "interval": "1m",
        "limit": 10
    }
    
    for url in klines_urls:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            data = response.json()
            print(f"✅ {url}: {response.status_code} - {len(data)} candles")
        except Exception as e:
            print(f"❌ {url}: ERROR - {e}")

if __name__ == "__main__":
    print("🔍 Binance API Debug Test")
    print("=" * 50)
    test_binance_endpoints()
    print("=" * 50)
    print("✅ Debug test completed")