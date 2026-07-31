"""
Test what T212 API actually returns for price data.
Run this locally with your T212_API_KEY set:
    set T212_API_KEY=your_key_here
    py test_t212_api.py
"""

import requests
import os
import json

T212_API_KEY = os.environ.get("T212_API_KEY", "")
T212_BASE    = "https://live.trading212.com/api/v0"
headers      = {"Authorization": T212_API_KEY}

if not T212_API_KEY:
    print("ERROR: T212_API_KEY not set")
    print("Run: set T212_API_KEY=your_key_here")
    exit(1)

# Step 1: Get instruments list and check structure
print("=== Step 1: Instruments list structure ===")
resp = requests.get(f"{T212_BASE}/equity/metadata/instruments", headers=headers, timeout=30)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    instruments = resp.json()
    print(f"Total instruments: {len(instruments)}")
    
    # Show first instrument in full
    print("\nFirst instrument (full structure):")
    print(json.dumps(instruments[0], indent=2))
    
    # Find a few US stocks
    us_stocks = [i for i in instruments if i.get("currencyCode") == "USD" 
                 and "AAPL" in str(i.get("ticker","")) or "NVDA" in str(i.get("ticker",""))]
    if us_stocks:
        print("\nSample US stocks:")
        for s in us_stocks[:3]:
            print(json.dumps(s, indent=2))

# Step 2: Check what price endpoints exist
print("\n=== Step 2: Available API endpoints ===")
# T212 beta API endpoints for prices
endpoints_to_try = [
    f"{T212_BASE}/equity/portfolio",
    f"{T212_BASE}/equity/orders",
    f"{T212_BASE}/equity/metadata/exchanges",
]

for ep in endpoints_to_try:
    r = requests.get(ep, headers=headers, timeout=10)
    print(f"{ep}: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"  Sample: {json.dumps(data[0], indent=2)[:300]}")
        elif isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")

# Step 3: Try the instrument price endpoint patterns
print("\n=== Step 3: Price endpoint patterns ===")
# Try a known ticker
test_ticker = "AAPL_US_EQ"
price_endpoints = [
    f"{T212_BASE}/equity/metadata/instruments/{test_ticker}",
    f"{T212_BASE}/equity/portfolio/{test_ticker}",
]

for ep in price_endpoints:
    r = requests.get(ep, headers=headers, timeout=10)
    print(f"\n{ep}")
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print(f"Response: {json.dumps(r.json(), indent=2)[:500]}")
    else:
        print(f"Error: {r.text[:200]}")
