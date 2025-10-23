#!/usr/bin/env python3
"""
Test partial close on EURUSD position with size 0.5 -> close 0.25
"""
import asyncio
from services.ig_orders import partial_close_position
from dependencies import get_ig_auth_headers
from datetime import datetime

async def test_eurusd_partial():
    """Test partial close on EURUSD"""

    deal_id = "DIAAAAVGEA7XXAQ"
    epic = "CS.D.EURUSD.CEEM.IP"
    direction = "SELL"
    current_size = 0.5
    size_to_close = 0.25  # Close half of 0.5

    print(f"📊 Testing Partial Close on EURUSD:")
    print(f"   Deal ID: {deal_id}")
    print(f"   Current Size: {current_size}")
    print(f"   Size to Close: {size_to_close}")
    print()

    # Get auth headers
    print("🔐 Getting authentication headers...")
    auth_headers = await get_ig_auth_headers()
    print("✅ Auth headers obtained\n")

    # Execute partial close
    print(f"📤 Attempting to close {size_to_close} of {current_size} position...")
    result = await partial_close_position(
        deal_id=deal_id,
        epic=epic,
        direction=direction,
        size_to_close=size_to_close,
        auth_headers=auth_headers
    )

    print("\n📥 Result:")
    print(f"   Success: {result.get('success')}")
    if result.get('success'):
        print(f"   ✅ Size Closed: {result.get('size_closed')}")
        print(f"   Response: {result.get('response')}")
    else:
        print(f"   ❌ Error: {result.get('error')}")

    return result.get('success')

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 EURUSD PARTIAL CLOSE TEST")
    print("=" * 60)
    print()

    success = asyncio.run(test_eurusd_partial())

    print()
    print("=" * 60)
    if success:
        print("✅ TEST PASSED!")
    else:
        print("❌ TEST FAILED!")
    print("=" * 60)
