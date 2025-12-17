#!/usr/bin/env python3
"""
Test script for limit order functionality with GBPUSD.

This script emulates a signal from SMC_Simple or EMA_Double strategy
and sends it to the /orders/place-order endpoint to test the IG API
limit order integration.

Usage:
    # From dev-app container:
    python test_limit_order_gbpusd.py

    # Or from host:
    docker exec -it fastapi-dev python /app/test_limit_order_gbpusd.py
"""

import asyncio
import httpx
import json
from datetime import datetime

# Configuration
FASTAPI_URL = "http://localhost:8000"  # Inside container
# FASTAPI_URL = "http://fastapi-dev:8000"  # If running from another container

# GBPUSD epic mapping
# EPIC_MAP expects: "GBPUSD.1.MINI" -> "CS.D.GBPUSD.MINI.IP"
GBPUSD_EPIC = "GBPUSD.1.MINI"
GBPUSD_IG_EPIC = "CS.D.GBPUSD.MINI.IP"


async def get_current_price():
    """
    Get current GBPUSD price from the market endpoint.
    This is needed to set a realistic limit order entry level.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get market data to determine current price
            response = await client.get(
                f"{FASTAPI_URL}/market/info",
                params={"epic": GBPUSD_IG_EPIC}
            )
            if response.status_code == 200:
                data = response.json()
                bid = data.get("bid_price") or data.get("bid")
                ask = data.get("ask_price") or data.get("offer")
                if bid:
                    return float(bid), float(ask) if ask else None
            print(f"⚠️ Could not get market price: {response.status_code} - {response.text[:200]}")
            return None, None
    except Exception as e:
        print(f"⚠️ Error getting market price: {e}")
        return None, None


async def check_existing_positions():
    """Check if there's already an open position for GBPUSD."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{FASTAPI_URL}/positions/open")
            if response.status_code == 200:
                positions = response.json().get("positions", [])
                for pos in positions:
                    if pos.get("market", {}).get("epic") == GBPUSD_IG_EPIC:
                        return True, pos
            return False, None
    except Exception as e:
        print(f"⚠️ Error checking positions: {e}")
        return False, None


async def check_existing_working_orders():
    """Check if there's already a working order for GBPUSD."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{FASTAPI_URL}/orders/working-orders")
            if response.status_code == 200:
                orders = response.json().get("workingOrders", [])
                for order in orders:
                    if order.get("marketData", {}).get("epic") == GBPUSD_IG_EPIC:
                        return True, order
            return False, None
    except Exception as e:
        print(f"⚠️ Error checking working orders: {e}")
        return False, None


async def test_limit_order_buy(entry_level: float = None):
    """
    Test placing a BUY limit order for GBPUSD.

    For a BUY limit order:
    - Entry level should be BELOW current market price
    - We want to buy at a cheaper price
    """
    print("\n" + "="*70)
    print("🧪 TEST: LIMIT ORDER BUY - GBPUSD")
    print("="*70)

    # Get current price
    bid, ask = await get_current_price()
    if bid:
        print(f"📊 Current market: Bid={bid:.5f}, Ask={ask:.5f if ask else 'N/A'}")
        # For BUY limit, set entry below current bid (e.g., 5 pips below)
        if entry_level is None:
            entry_level = round(bid - 0.0005, 5)  # 5 pips below bid
    else:
        print("⚠️ Could not get market price, using estimated entry")
        if entry_level is None:
            entry_level = 1.26500  # Fallback estimated level

    print(f"📍 Limit entry level: {entry_level:.5f}")

    # Build the order payload (simulating a signal from SMC_Simple/EMA_Double)
    order_payload = {
        "epic": GBPUSD_EPIC,
        "direction": "BUY",
        "size": 1.0,
        "order_type": "limit",
        "entry_level": entry_level,
        "stop_distance": 15,  # 15 pips stop
        "limit_distance": 30,  # 30 pips take profit (2:1 R:R)
        "use_provided_sl_tp": True,
        "limit_expiry_minutes": 6,  # 6 minute expiry
        "custom_label": "TEST_LIMIT_BUY",
        "risk_reward": 2.0
    }

    print(f"\n📤 Sending limit order request:")
    print(json.dumps(order_payload, indent=2))

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{FASTAPI_URL}/orders/place-order",
                json=order_payload
            )

            print(f"\n📬 Response Status: {response.status_code}")
            print(f"📬 Response Body:")
            print(json.dumps(response.json(), indent=2, default=str))

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "pending":
                    print("\n✅ SUCCESS: Limit order placed successfully!")
                    print(f"   Deal Reference: {data.get('dealReference')}")
                    print(f"   Entry Level: {data.get('entry_level')}")
                    print(f"   Expires in: {data.get('expiry_minutes')} minutes")
                    return True, data
                else:
                    print(f"\n⚠️ Order status: {data.get('status')}")
            elif response.status_code == 409:
                print("\n⚠️ Order skipped - position or working order already exists")
            elif response.status_code == 429:
                print("\n⏱️ Trade cooldown active")
            else:
                print(f"\n❌ Order failed: HTTP {response.status_code}")

            return False, response.json()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False, {"error": str(e)}


async def test_limit_order_sell(entry_level: float = None):
    """
    Test placing a SELL limit order for GBPUSD.

    For a SELL limit order:
    - Entry level should be ABOVE current market price
    - We want to sell at a higher price
    """
    print("\n" + "="*70)
    print("🧪 TEST: LIMIT ORDER SELL - GBPUSD")
    print("="*70)

    # Get current price
    bid, ask = await get_current_price()
    if ask:
        print(f"📊 Current market: Bid={bid:.5f if bid else 'N/A'}, Ask={ask:.5f}")
        # For SELL limit, set entry above current ask (e.g., 5 pips above)
        if entry_level is None:
            entry_level = round(ask + 0.0005, 5)  # 5 pips above ask
    elif bid:
        print(f"📊 Current market: Bid={bid:.5f}")
        if entry_level is None:
            entry_level = round(bid + 0.0005, 5)  # 5 pips above bid
    else:
        print("⚠️ Could not get market price, using estimated entry")
        if entry_level is None:
            entry_level = 1.27000  # Fallback estimated level

    print(f"📍 Limit entry level: {entry_level:.5f}")

    # Build the order payload
    order_payload = {
        "epic": GBPUSD_EPIC,
        "direction": "SELL",
        "size": 1.0,
        "order_type": "limit",
        "entry_level": entry_level,
        "stop_distance": 15,  # 15 pips stop
        "limit_distance": 30,  # 30 pips take profit (2:1 R:R)
        "use_provided_sl_tp": True,
        "limit_expiry_minutes": 6,  # 6 minute expiry
        "custom_label": "TEST_LIMIT_SELL",
        "risk_reward": 2.0
    }

    print(f"\n📤 Sending limit order request:")
    print(json.dumps(order_payload, indent=2))

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{FASTAPI_URL}/orders/place-order",
                json=order_payload
            )

            print(f"\n📬 Response Status: {response.status_code}")
            print(f"📬 Response Body:")
            print(json.dumps(response.json(), indent=2, default=str))

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "pending":
                    print("\n✅ SUCCESS: Limit order placed successfully!")
                    print(f"   Deal Reference: {data.get('dealReference')}")
                    print(f"   Entry Level: {data.get('entry_level')}")
                    print(f"   Expires in: {data.get('expiry_minutes')} minutes")
                    return True, data
                else:
                    print(f"\n⚠️ Order status: {data.get('status')}")
            elif response.status_code == 409:
                print("\n⚠️ Order skipped - position or working order already exists")
            elif response.status_code == 429:
                print("\n⏱️ Trade cooldown active")
            else:
                print(f"\n❌ Order failed: HTTP {response.status_code}")

            return False, response.json()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False, {"error": str(e)}


async def test_market_order():
    """
    Test placing a regular market order (for comparison).
    This is the original behavior before limit orders were added.
    """
    print("\n" + "="*70)
    print("🧪 TEST: MARKET ORDER BUY - GBPUSD")
    print("="*70)

    # Build the order payload (market order - no entry_level)
    order_payload = {
        "epic": GBPUSD_EPIC,
        "direction": "BUY",
        "size": 1.0,
        "order_type": "market",  # Explicitly market order
        # No entry_level for market orders
        "stop_distance": 15,
        "limit_distance": 30,
        "use_provided_sl_tp": True,
        "custom_label": "TEST_MARKET",
        "risk_reward": 2.0
    }

    print(f"\n📤 Sending market order request:")
    print(json.dumps(order_payload, indent=2))

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{FASTAPI_URL}/orders/place-order",
                json=order_payload
            )

            print(f"\n📬 Response Status: {response.status_code}")
            print(f"📬 Response Body:")
            print(json.dumps(response.json(), indent=2, default=str))

            return response.status_code == 200, response.json()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False, {"error": str(e)}


async def list_working_orders():
    """List all current working orders."""
    print("\n" + "="*70)
    print("📋 LISTING ALL WORKING ORDERS")
    print("="*70)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{FASTAPI_URL}/orders/working-orders")

            print(f"\n📬 Response Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                orders = data.get("workingOrders", [])
                print(f"📊 Found {len(orders)} working order(s)")

                for i, order in enumerate(orders, 1):
                    market = order.get("marketData", {})
                    work = order.get("workingOrderData", {})
                    print(f"\n   Order {i}:")
                    print(f"   - Epic: {market.get('epic')}")
                    print(f"   - Direction: {work.get('direction')}")
                    print(f"   - Level: {work.get('level')}")
                    print(f"   - Size: {work.get('size')}")
                    print(f"   - Deal ID: {work.get('dealId')}")
                    print(f"   - Good Till: {work.get('goodTillDate')}")

                return orders
            else:
                print(f"❌ Failed: {response.text}")
                return []

    except Exception as e:
        print(f"❌ Error: {e}")
        return []


async def cancel_working_order(deal_id: str):
    """Cancel a specific working order."""
    print(f"\n🗑️ Cancelling working order: {deal_id}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{FASTAPI_URL}/orders/cancel-working-order/{deal_id}"
            )

            print(f"📬 Response Status: {response.status_code}")
            print(f"📬 Response: {response.json()}")

            return response.status_code == 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("🚀 LIMIT ORDER TEST SUITE - GBPUSD")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Pre-flight checks
    print("\n📋 Pre-flight checks...")

    # Check for existing position
    has_position, position = await check_existing_positions()
    if has_position:
        print(f"⚠️ Existing position found for GBPUSD:")
        print(f"   Deal ID: {position.get('position', {}).get('dealId')}")
        print(f"   Direction: {position.get('position', {}).get('direction')}")
        print("   (Test may fail with 409 Conflict)")
    else:
        print("✅ No existing position for GBPUSD")

    # Check for existing working order
    has_order, order = await check_existing_working_orders()
    if has_order:
        print(f"⚠️ Existing working order found for GBPUSD:")
        print(f"   Deal ID: {order.get('workingOrderData', {}).get('dealId')}")
        print("   (Test may fail with 409 Conflict)")
    else:
        print("✅ No existing working order for GBPUSD")

    # Menu
    print("\n" + "="*70)
    print("Select test to run:")
    print("  1. Test LIMIT ORDER BUY (5 pips below market)")
    print("  2. Test LIMIT ORDER SELL (5 pips above market)")
    print("  3. Test MARKET ORDER (for comparison)")
    print("  4. List all working orders")
    print("  5. Cancel all GBPUSD working orders")
    print("  6. Run all tests")
    print("  0. Exit")
    print("="*70)

    choice = input("\nEnter choice (1-6, 0 to exit): ").strip()

    if choice == "1":
        await test_limit_order_buy()
    elif choice == "2":
        await test_limit_order_sell()
    elif choice == "3":
        await test_market_order()
    elif choice == "4":
        await list_working_orders()
    elif choice == "5":
        orders = await list_working_orders()
        for order in orders:
            if order.get("marketData", {}).get("epic") == GBPUSD_IG_EPIC:
                deal_id = order.get("workingOrderData", {}).get("dealId")
                if deal_id:
                    await cancel_working_order(deal_id)
    elif choice == "6":
        print("\n🔄 Running all tests...")
        await list_working_orders()
        await test_limit_order_buy()
        await list_working_orders()
        # Note: Won't run sell if buy succeeds (would get 409)
    elif choice == "0":
        print("\nExiting...")
    else:
        print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    asyncio.run(main())
