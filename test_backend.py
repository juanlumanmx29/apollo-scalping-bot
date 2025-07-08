#!/usr/bin/env python3
"""
Test script to verify backend is working
"""
import requests
import json

url = "https://apollo-scalping-bot-production.up.railway.app"

def test_endpoint(endpoint):
    try:
        full_url = f"{url}{endpoint}"
        print(f"Testing: {full_url}")
        response = requests.get(full_url, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        print("-" * 50)
        return response
    except Exception as e:
        print(f"Error: {e}")
        print("-" * 50)
        return None

if __name__ == "__main__":
    print("🧪 Testing Apollo Scalping Bot Backend")
    print("=" * 50)
    
    # Test root endpoint
    test_endpoint("/")
    
    # Test health endpoint  
    test_endpoint("/health")
    
    # Test a non-existent endpoint to see error handling
    test_endpoint("/nonexistent")