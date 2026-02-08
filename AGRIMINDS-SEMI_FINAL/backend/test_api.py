#!/usr/bin/env python3
"""
AgriMinds API Test Script
Tests all endpoints to verify they work correctly
"""

import requests
import json

API_BASE = "http://localhost:8000"

def test_health():
    """Test the root endpoint"""
    print("🔍 Testing Health Check...")
    response = requests.get(f"{API_BASE}/")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ Health check passed")
    else:
        print("   ❌ Health check failed")
    print()

def test_crop_prediction():
    """Test crop prediction endpoint"""
    print("🌱 Testing Crop Prediction...")
    
    data = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.88,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 202
    }
    
    try:
        response = requests.post(f"{API_BASE}/predict/crop", json=data)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Recommended Crop: {result['recommended_crop']}")
            print(f"   Confidence: {result['confidence']:.2%}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
    print()

def test_fertilizer_recommendation():
    """Test fertilizer recommendation endpoint"""
    print("⚗️  Testing Fertilizer Recommendation...")
    
    data = {
        "N": 40,
        "P": 30,
        "K": 80,
        "temperature": 25,
        "humidity": 70,
        "ph": 6.5,
        "rainfall": 150,
        "NDVI": 0.6,
        "soil_moisture": 60,
        "growth_stage": "vegetative",
        "forecast_rain": 0
    }
    
    try:
        response = requests.post(f"{API_BASE}/predict/fertilizer", json=data)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ NPK Plan:")
            print(f"      N: {result['npk_plan']['N']} kg/ha")
            print(f"      P: {result['npk_plan']['P']} kg/ha")
            print(f"      K: {result['npk_plan']['K']} kg/ha")
            print(f"   Green Score: {result['green_score']}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
    print()

def test_ndvi():
    """Test NDVI endpoint"""
    print("🛰️  Testing NDVI Satellite Data...")
    
    params = {
        "lat": 28.6139,
        "lon": 77.2090
    }
    
    try:
        response = requests.get(f"{API_BASE}/satellite/ndvi", params=params)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ NDVI: {result['ndvi']}")
            print(f"   Interpretation: {result['interpretation']}")
            print(f"   Source: {result['source']}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
    print()

def main():
    print("=" * 60)
    print("🌾 AgriMinds API Test Suite")
    print("=" * 60)
    print()
    
    # Run all tests
    test_health()
    test_crop_prediction()
    test_fertilizer_recommendation()
    test_ndvi()
    
    print("=" * 60)
    print("✅ Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("   Make sure the backend is running on http://localhost:8000")
        print("   Start it with: python main.py")
