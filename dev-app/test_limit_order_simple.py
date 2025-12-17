#!/usr/bin/env python3
"""
Simple non-interactive test for limit order functionality with GBPUSD.

Usage:
    docker exec -it fastapi-dev python /app/test_limit_order_simple.py [buy|sell|list|cancel]
"""

import asyncio
import httpx
import json
import sys
from datetime import datetime

FASTAPI_URL = "http://localhost:8000"
GBPUSD_EPIC = "GBPUSD.1.MINI"
GBPUSD_IG_EPIC = "CS.D.GBPUSD.MINI.IP"


async def get_current_price():
    """Get current GBPUSD price."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{FASTAPI_URL}/market/info",
                params={"epic": GBPUSD_IG_EPIC}
            )
            if response.status_code == 200:
                data = response.json()
                bid = data.get("bid_price") or data.get("bid")
                ask = data.get("ask_price") or data.get("offer")
                return float(bid) if bid else None, float(ask) if ask else None
            print(f"Market info response: {response.status_code} - {response.text[:500]}")
    except Exception as e:
        print(f"Error getting price: {e}")
    return None, None


async def list_working_orders():
    """List all working orders."""
    print("\n" + "="*60)
    print("📋 LISTING WORKING ORDERS")
    print("="*60)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{FASTAPI_URL}/orders/working-orders")
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                orders = data.get("workingOrders", [])
                print(f"Found {len(orders)} working order(s)\n")

                for order in orders:
                    market = order.get("marketData", {})
                    work = order.get("workingOrderData", {})
                    print(f"  Epic: {market.get('epic')}")
                    print(f"  Direction: {work.get('direction')}")
                    print(f"  Level: {work.get('level')}")
                    print(f"  Size: {work.get('size')}")
                    print(f"  Deal ID: {work.get('dealId')}")
                    print(f"  Good Till: {work.get('goodTillDate')}")
                    print("-" * 40)
                return orders
            else:
                print(f"Response: {response.text[:500]}")
                return []
    except Exception as e:
        print(f"Error: {e}")
        return []


async def test_limit_buy():
    """Test BUY limit order."""
    print("\n" + "="*60)
    print("🧪 TEST: LIMIT ORDER BUY - GBPUSD")
    print("="*60)

    bid, ask = await get_current_price()
    if bid:
        entry_level = round(bid - 0.0005, 5)  # 5 pips below
        print(f"Current bid: {bid:.5f}")
        print(f"Entry level: {entry_level:.5f} (5 pips below)")
    else:
        entry_level = 1.26500
        print(f"Using fallback entry: {entry_level}")

    payload = {
        "epic": GBPUSD_EPIC,
        "direction": "BUY",
        "size": 1.0,
        "order_type": "limit",
        "entry_level": entry_level,
        "stop_distance": 15,
        "limit_distance": 30,
        "use_provided_sl_tp": True,
        "limit_expiry_minutes": 6,
        "custom_label": "TEST_LIMIT_BUY"
    }

    print(f"\nPayload: {json.dumps(payload, indent=2)}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{FASTAPI_URL}/orders/place-order",
                json=payload
            )
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2, default=str)}")

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "pending":
                    print("\n✅ SUCCESS! Limit order placed")
                    return True
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


async def test_limit_sell():
    """Test SELL limit order."""
    print("\n" + "="*60)
    print("🧪 TEST: LIMIT ORDER SELL - GBPUSD")
    print("="*60)

    bid, ask = await get_current_price()
    if bid:
        entry_level = round(bid + 0.0005, 5)  # 5 pips above
        print(f"Current bid: {bid:.5f}")
        print(f"Entry level: {entry_level:.5f} (5 pips above)")
    else:
        entry_level = 1.27000
        print(f"Using fallback entry: {entry_level}")

    payload = {
        "epic": GBPUSD_EPIC,
        "direction": "SELL",
        "size": 1.0,
        "order_type": "limit",
        "entry_level": entry_level,
        "stop_distance": 15,
        "limit_distance": 30,
        "use_provided_sl_tp": True,
        "limit_expiry_minutes": 6,
        "custom_label": "TEST_LIMIT_SELL"
    }

    print(f"\nPayload: {json.dumps(payload, indent=2)}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{FASTAPI_URL}/orders/place-order",
                json=payload
            )
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2, default=str)}")

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "pending":
                    print("\n✅ SUCCESS! Limit order placed")
                    return True
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


async def cancel_gbpusd_orders():
    """Cancel all GBPUSD working orders."""
    print("\n" + "="*60)
    print("🗑️ CANCELLING GBPUSD WORKING ORDERS")
    print("="*60)

    orders = await list_working_orders()
    cancelled = 0

    for order in orders:
        if order.get("marketData", {}).get("epic") == GBPUSD_IG_EPIC:
            deal_id = order.get("workingOrderData", {}).get("dealId")
            if deal_id:
                print(f"\nCancelling order: {deal_id}")
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.delete(
                            f"{FASTAPI_URL}/orders/cancel-working-order/{deal_id}"
                        )
                        print(f"Status: {response.status_code}")
                        print(f"Response: {response.json()}")
                        if response.status_code == 200:
                            cancelled += 1
                except Exception as e:
                    print(f"Error: {e}")

    print(f"\nCancelled {cancelled} order(s)")
    return cancelled


async def main():
    """Run tests based on command line args."""
    print("\n" + "="*60)
    print(f"🚀 LIMIT ORDER TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if len(sys.argv) < 2:
        print("\nUsage: python test_limit_order_simple.py [buy|sell|list|cancel]")
        print("\nOptions:")
        print("  buy    - Place a BUY limit order (5 pips below market)")
        print("  sell   - Place a SELL limit order (5 pips above market)")
        print("  list   - List all working orders")
        print("  cancel - Cancel all GBPUSD working orders")
        return

    action = sys.argv[1].lower()

    if action == "buy":
        await test_limit_buy()
    elif action == "sell":
        await test_limit_sell()
    elif action == "list":
        await list_working_orders()
    elif action == "cancel":
        await cancel_gbpusd_orders()
    else:
        print(f"Unknown action: {action}")
        print("Use: buy, sell, list, or cancel")


if __name__ == "__main__":
    asyncio.run(main())
