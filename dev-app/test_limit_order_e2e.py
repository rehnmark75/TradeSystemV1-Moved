#!/usr/bin/env python3
"""
End-to-End Test for Limit Order Functionality - GBPUSD

This script tests the complete flow:
1. Authenticates with IG API
2. Gets current market price
3. Places a limit order via the /orders/place-order endpoint
4. Verifies the order was placed correctly

Usage:
    docker exec fastapi-dev python /app/test_limit_order_e2e.py buy
    docker exec fastapi-dev python /app/test_limit_order_e2e.py sell
    docker exec fastapi-dev python /app/test_limit_order_e2e.py list
    docker exec fastapi-dev python /app/test_limit_order_e2e.py cancel
"""

import asyncio
import httpx
import json
import sys
from datetime import datetime

# Import from the app's modules
from services.ig_auth import ig_login
from services.keyvault import get_secret
from services.ig_market import get_current_bid_price
from config import IG_USERNAME, IG_PWD, IG_API_KEY, API_BASE_URL

# GBPUSD Configuration
GBPUSD_EPIC = "GBPUSD.1.MINI"  # Input format for place-order endpoint
GBPUSD_IG_EPIC = "CS.D.GBPUSD.MINI.IP"  # IG API format

# Internal FastAPI URL
FASTAPI_URL = "http://localhost:8000"


async def get_auth_headers():
    """Get authenticated headers for IG API."""
    print("\n🔐 Authenticating with IG API...")

    api_key = get_secret(IG_API_KEY)
    ig_pwd = get_secret(IG_PWD)

    tokens = await ig_login(api_key=api_key, ig_pwd=ig_pwd, ig_usr=IG_USERNAME)

    headers = {
        "Accept": "application/json; charset=UTF-8",
        "Content-Type": "application/json; charset=UTF-8",
        "X-IG-API-KEY": api_key,
        "Version": "2",
        "CST": tokens["CST"],
        "X-SECURITY-TOKEN": tokens["X-SECURITY-TOKEN"],
    }

    print(f"✅ Authenticated successfully")
    print(f"   Account ID: {tokens.get('ACCOUNT_ID', 'N/A')}")

    return headers


async def get_market_price(headers: dict):
    """Get current GBPUSD market price."""
    print("\n📊 Getting current GBPUSD market price...")

    try:
        price_info = await get_current_bid_price(headers, GBPUSD_IG_EPIC)
        bid = price_info.get("bid_price")
        ask = price_info.get("offer_price") or price_info.get("ask_price")

        print(f"   Bid: {bid}")
        print(f"   Ask: {ask}")
        print(f"   Currency: {price_info.get('currency_code')}")
        print(f"   Min Distance: {price_info.get('min_distance')} points")

        return bid, ask, price_info
    except Exception as e:
        print(f"❌ Error getting price: {e}")
        return None, None, None


async def check_open_positions(headers: dict):
    """Check for existing open positions."""
    print("\n📋 Checking open positions...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{API_BASE_URL}/positions", headers=headers)

        if response.status_code == 200:
            positions = response.json().get("positions", [])
            gbpusd_positions = [
                p for p in positions
                if p.get("market", {}).get("epic") == GBPUSD_IG_EPIC
            ]

            if gbpusd_positions:
                print(f"⚠️  Found {len(gbpusd_positions)} open GBPUSD position(s):")
                for pos in gbpusd_positions:
                    p = pos.get("position", {})
                    print(f"      Deal ID: {p.get('dealId')}")
                    print(f"      Direction: {p.get('direction')}")
                    print(f"      Size: {p.get('size')}")
                    print(f"      Level: {p.get('level')}")
                return gbpusd_positions
            else:
                print("   No open GBPUSD positions")
                return []
        else:
            print(f"❌ Error checking positions: {response.status_code}")
            return []


async def check_working_orders(headers: dict):
    """Check for existing working orders."""
    print("\n📋 Checking working orders...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{API_BASE_URL}/workingorders", headers=headers)

        if response.status_code == 200:
            orders = response.json().get("workingOrders", [])
            gbpusd_orders = [
                o for o in orders
                if o.get("marketData", {}).get("epic") == GBPUSD_IG_EPIC
            ]

            if gbpusd_orders:
                print(f"⚠️  Found {len(gbpusd_orders)} GBPUSD working order(s):")
                for order in gbpusd_orders:
                    w = order.get("workingOrderData", {})
                    print(f"      Deal ID: {w.get('dealId')}")
                    print(f"      Direction: {w.get('direction')}")
                    print(f"      Level: {w.get('level')}")
                    print(f"      Size: {w.get('size')}")
                    print(f"      Good Till: {w.get('goodTillDate')}")
                return gbpusd_orders
            else:
                print("   No GBPUSD working orders")
                return []
        else:
            print(f"❌ Error checking working orders: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return []


async def place_limit_order_via_endpoint(direction: str, entry_level: float):
    """
    Place a limit order via the FastAPI /orders/place-order endpoint.
    This tests the full endpoint including all validations.
    """
    print(f"\n🚀 Placing LIMIT ORDER via FastAPI endpoint...")
    print(f"   Direction: {direction}")
    print(f"   Entry Level: {entry_level:.5f}")

    payload = {
        "epic": GBPUSD_EPIC,
        "direction": direction,
        "size": 1.0,
        "order_type": "limit",
        "entry_level": entry_level,
        "stop_distance": 15,      # 15 pips
        "limit_distance": 30,     # 30 pips (2:1 R:R)
        "use_provided_sl_tp": True,
        "limit_expiry_minutes": 6,
        "custom_label": f"E2E_TEST_{direction}"
    }

    # Headers required by FastAPI middleware
    headers = {
        "x-apim-gateway": "verified",  # Required by middleware
        "Content-Type": "application/json"
    }

    print(f"\n📤 Request payload:")
    print(json.dumps(payload, indent=2))

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{FASTAPI_URL}/orders/place-order",
            json=payload,
            headers=headers
        )

        print(f"\n📬 Response Status: {response.status_code}")

        try:
            result = response.json()
            print(f"📬 Response Body:")
            print(json.dumps(result, indent=2, default=str))
        except:
            print(f"📬 Response Text: {response.text[:500]}")
            result = {"error": response.text}

        if response.status_code == 200:
            if result.get("status") == "pending":
                print("\n" + "="*60)
                print("✅ SUCCESS! LIMIT ORDER PLACED")
                print("="*60)
                print(f"   Deal Reference: {result.get('dealReference')}")
                print(f"   Entry Level: {result.get('entry_level')}")
                print(f"   Stop Distance: {result.get('stop_distance')} pips")
                print(f"   Limit Distance: {result.get('limit_distance')} pips")
                print(f"   Expires in: {result.get('expiry_minutes')} minutes")
                return True, result
            else:
                print(f"\n⚠️ Unexpected status: {result.get('status')}")
                return False, result
        elif response.status_code == 409:
            print("\n⚠️ Conflict - position or working order already exists")
            return False, result
        elif response.status_code == 429:
            print("\n⏱️ Trade cooldown active")
            return False, result
        else:
            print(f"\n❌ Order failed: HTTP {response.status_code}")
            return False, result


async def cancel_working_order(headers: dict, deal_id: str):
    """Cancel a working order directly via IG API."""
    print(f"\n🗑️ Cancelling working order: {deal_id}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.delete(
            f"{API_BASE_URL}/workingorders/otc/{deal_id}",
            headers=headers
        )

        print(f"   Status: {response.status_code}")
        if response.status_code in [200, 201]:
            print(f"   ✅ Cancelled successfully")
            return True
        else:
            print(f"   ❌ Failed: {response.text[:200]}")
            return False


async def test_buy():
    """Test placing a BUY limit order."""
    print("\n" + "="*70)
    print("🧪 E2E TEST: LIMIT ORDER BUY - GBPUSD")
    print("="*70)

    # 1. Authenticate
    headers = await get_auth_headers()

    # 2. Check existing positions/orders
    await check_open_positions(headers)
    await check_working_orders(headers)

    # 3. Get current price
    bid, ask, price_info = await get_market_price(headers)

    if not bid:
        print("❌ Cannot proceed without market price")
        return False

    # 4. Calculate entry level (5 pips below bid for BUY limit)
    entry_level = round(bid - 0.0005, 5)
    print(f"\n📍 Entry level calculated: {entry_level:.5f} (5 pips below bid)")

    # 5. Place order via FastAPI endpoint
    success, result = await place_limit_order_via_endpoint("BUY", entry_level)

    # 6. Verify order was created
    if success:
        print("\n🔍 Verifying order was created...")
        await check_working_orders(headers)

    return success


async def test_sell():
    """Test placing a SELL limit order."""
    print("\n" + "="*70)
    print("🧪 E2E TEST: LIMIT ORDER SELL - GBPUSD")
    print("="*70)

    # 1. Authenticate
    headers = await get_auth_headers()

    # 2. Check existing positions/orders
    await check_open_positions(headers)
    await check_working_orders(headers)

    # 3. Get current price
    bid, ask, price_info = await get_market_price(headers)

    if not bid:
        print("❌ Cannot proceed without market price")
        return False

    # 4. Calculate entry level (5 pips above bid for SELL limit)
    entry_level = round(bid + 0.0005, 5)
    print(f"\n📍 Entry level calculated: {entry_level:.5f} (5 pips above bid)")

    # 5. Place order via FastAPI endpoint
    success, result = await place_limit_order_via_endpoint("SELL", entry_level)

    # 6. Verify order was created
    if success:
        print("\n🔍 Verifying order was created...")
        await check_working_orders(headers)

    return success


async def list_orders():
    """List all working orders."""
    print("\n" + "="*70)
    print("📋 LISTING ALL WORKING ORDERS")
    print("="*70)

    headers = await get_auth_headers()
    orders = await check_working_orders(headers)

    return orders


async def cancel_all_gbpusd():
    """Cancel all GBPUSD working orders."""
    print("\n" + "="*70)
    print("🗑️ CANCELLING ALL GBPUSD WORKING ORDERS")
    print("="*70)

    headers = await get_auth_headers()
    orders = await check_working_orders(headers)

    cancelled = 0
    for order in orders:
        deal_id = order.get("workingOrderData", {}).get("dealId")
        if deal_id:
            if await cancel_working_order(headers, deal_id):
                cancelled += 1

    print(f"\n✅ Cancelled {cancelled} order(s)")
    return cancelled


async def main():
    """Main entry point."""
    print("\n" + "="*70)
    print(f"🚀 LIMIT ORDER E2E TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    if len(sys.argv) < 2:
        print("""
Usage: python test_limit_order_e2e.py <command>

Commands:
    buy     - Place a BUY limit order (5 pips below market)
    sell    - Place a SELL limit order (5 pips above market)
    list    - List all working orders
    cancel  - Cancel all GBPUSD working orders
        """)
        return

    action = sys.argv[1].lower()

    try:
        if action == "buy":
            await test_buy()
        elif action == "sell":
            await test_sell()
        elif action == "list":
            await list_orders()
        elif action == "cancel":
            await cancel_all_gbpusd()
        else:
            print(f"Unknown command: {action}")
            print("Use: buy, sell, list, or cancel")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
