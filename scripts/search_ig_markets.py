#!/usr/bin/env python3
"""
Search IG Markets for available instruments
Usage: python search_ig_markets.py <search_term>
"""
import sys
import os
import requests

def authenticate(api_key, password, username="rehnmarkh", is_demo=False):
    """Authenticate with IG Markets API"""
    base_url = "https://demo-api.ig.com" if is_demo else "https://api.ig.com"
    url = f"{base_url}/gateway/deal/session"

    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json; charset=UTF-8",
        "X-IG-API-KEY": api_key,
        "Version": "2"
    }

    payload = {
        "identifier": username,
        "password": password
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return {
            "CST": response.headers.get("CST"),
            "X-SECURITY-TOKEN": response.headers.get("X-SECURITY-TOKEN"),
            "X-IG-API-KEY": api_key
        }
    else:
        print(f"❌ Authentication failed: {response.status_code}")
        print(response.text)
        return None

def search_markets(auth_headers, search_term, is_demo=False):
    """Search for markets by term"""
    base_url = "https://demo-api.ig.com" if is_demo else "https://api.ig.com"
    url = f"{base_url}/gateway/deal/markets"

    headers = {
        "Accept": "application/json; charset=UTF-8",
        "Content-Type": "application/json; charset=UTF-8",
        "X-IG-API-KEY": auth_headers["X-IG-API-KEY"],
        "CST": auth_headers["CST"],
        "X-SECURITY-TOKEN": auth_headers["X-SECURITY-TOKEN"],
        "Version": "1"
    }

    params = {"searchTerm": search_term}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Search failed: {response.status_code}")
        print(response.text)
        return None

def format_results(data, search_term):
    """Format search results"""
    if not data or "markets" not in data:
        print(f"❌ No results found for '{search_term}'")
        return

    markets = data["markets"]

    if not markets:
        print(f"❌ No markets found for '{search_term}'")
        return

    print(f"\n{'='*100}")
    print(f"🔍 Search results for '{search_term}' - Found {len(markets)} market(s)")
    print(f"{'='*100}\n")

    for market in markets:
        epic = market.get("epic", "N/A")
        name = market.get("instrumentName", "N/A")
        instrument_type = market.get("instrumentType", "N/A")
        market_status = market.get("marketStatus", "N/A")
        streaming = market.get("streamingPricesAvailable", False)
        tradeable = market.get("otcTradeable", False)

        bid = market.get("bid", "N/A")
        offer = market.get("offer", "N/A")

        print(f"📊 {name}")
        print(f"   Epic:              {epic}")
        print(f"   Type:              {instrument_type}")
        print(f"   Market Status:     {market_status}")
        print(f"   Streaming:         {'✅ Yes' if streaming else '❌ No'}")
        print(f"   Tradeable:         {'✅ Yes' if tradeable else '❌ No'}")
        print(f"   Current Prices:    Bid: {bid}, Offer: {offer}")
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python search_ig_markets.py <search_term>")
        print("\nExamples:")
        print("  python search_ig_markets.py gold")
        print("  python search_ig_markets.py solana")
        print("  python search_ig_markets.py bitcoin")
        sys.exit(1)

    search_term = sys.argv[1]

    # Try production credentials first
    api_key = os.getenv("PROD_IG_API_KEY") or os.getenv("IG_API_KEY")
    password = os.getenv("PROD_IG_PWD") or os.getenv("IG_PWD")

    is_demo = False

    # Fall back to demo if no prod credentials
    if not api_key:
        api_key = os.getenv("DEMO_API_KEY")
        password = os.getenv("DEMO_PASSWORD")
        is_demo = True

    if not api_key or not password:
        print("❌ No IG credentials found in environment")
        print("Set PROD_IG_API_KEY/PROD_IG_PWD or DEMO_API_KEY/DEMO_PASSWORD")
        sys.exit(1)

    print(f"🔐 Authenticating with IG Markets ({'DEMO' if is_demo else 'PRODUCTION'})...")
    auth_headers = authenticate(api_key, password, is_demo=is_demo)

    if not auth_headers:
        sys.exit(1)

    print(f"✅ Authenticated successfully")
    print(f"\n🔍 Searching for '{search_term}'...")

    data = search_markets(auth_headers, search_term, is_demo=is_demo)
    format_results(data, search_term)

if __name__ == "__main__":
    main()
