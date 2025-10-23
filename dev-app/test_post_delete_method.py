#!/usr/bin/env python3
"""
Test POST with _method:DELETE based on improved error message
"""
import asyncio
import httpx
from dependencies import get_ig_auth_headers
from config import API_BASE_URL
import json

DEAL_ID = "DIAAAAVGE3L9FAV"
EPIC = "CS.D.EURUSD.CEEM.IP"
DIRECTION = "SELL"
SIZE_TO_CLOSE = 0.5

async def test_post_delete():
    """Test POST with _method:DELETE and orderType"""

    print("🔐 Getting authentication headers...")
    auth_headers = await get_ig_auth_headers()
    print("✅ Auth headers obtained\n")

    close_direction = "BUY" if DIRECTION == "SELL" else "SELL"

    # Test with different payload variations
    tests = [
        {
            "name": "With orderType MARKET",
            "payload": {
                "dealId": DEAL_ID,
                "direction": close_direction,
                "size": SIZE_TO_CLOSE,
                "orderType": "MARKET"
            }
        },
        {
            "name": "With orderType MARKET and epic",
            "payload": {
                "dealId": DEAL_ID,
                "epic": EPIC,
                "direction": close_direction,
                "size": SIZE_TO_CLOSE,
                "orderType": "MARKET"
            }
        },
        {
            "name": "With all fields including expiry",
            "payload": {
                "dealId": DEAL_ID,
                "epic": EPIC,
                "expiry": "-",
                "direction": close_direction,
                "size": SIZE_TO_CLOSE,
                "orderType": "MARKET"
            }
        },
        {
            "name": "With orderType LIMIT and level",
            "payload": {
                "dealId": DEAL_ID,
                "direction": close_direction,
                "size": SIZE_TO_CLOSE,
                "orderType": "LIMIT",
                "level": 11600.0  # Current market level
            }
        }
    ]

    for test in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {test['name']}")
        print(f"{'='*60}")
        print(f"Payload: {json.dumps(test['payload'], indent=2)}")

        headers = {
            **auth_headers,
            "_method": "DELETE",
            "Version": "1"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{API_BASE_URL}/positions/otc",
                    headers=headers,
                    json=test['payload'],
                    timeout=30
                )

                print(f"\nResponse Status: {response.status_code}")
                print(f"Response Body: {response.text}")

                if response.status_code in [200, 201]:
                    print(f"✅ SUCCESS!")
                    return True
                else:
                    print(f"❌ FAILED")

        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")

    return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 POST with _method:DELETE Tests")
    print("=" * 60)
    print()

    success = asyncio.run(test_post_delete())

    print("\n" + "=" * 60)
    if success:
        print("✅ SUCCESS - Partial close worked!")
    else:
        print("❌ All variations failed")
    print("=" * 60)
