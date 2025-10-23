#!/usr/bin/env python3
"""
Comprehensive test of DELETE variations for partial close
Based on IG support: DELETE request with size parameter
"""
import asyncio
import httpx
from dependencies import get_ig_auth_headers
from config import API_BASE_URL
import json

# Current open position
DEAL_ID = "DIAAAAVGE3L9FAV"
EPIC = "CS.D.EURUSD.CEEM.IP"
DIRECTION = "SELL"  # Original position
SIZE_TO_CLOSE = 0.5  # Close half

async def test_delete_variation(variation_name, url, headers, payload, method="DELETE"):
    """Test a specific DELETE variation"""
    print(f"\n{'='*60}")
    print(f"TEST: {variation_name}")
    print(f"{'='*60}")
    print(f"Method: {method}")
    print(f"URL: {url}")
    print(f"Headers: {json.dumps({k: v for k, v in headers.items() if k not in ['X-SECURITY-TOKEN', 'CST']}, indent=2)}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()

    try:
        async with httpx.AsyncClient() as client:
            if method == "DELETE":
                response = await client.request(
                    method="DELETE",
                    url=url,
                    headers=headers,
                    content=json.dumps(payload),
                    timeout=30
                )
            elif method == "POST":
                response = await client.post(
                    url=url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

            print(f"Response Status: {response.status_code}")
            print(f"Response Body: {response.text}")

            if response.status_code in [200, 201]:
                print(f"✅ SUCCESS!")
                return True
            else:
                print(f"❌ FAILED")
                return False

    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False

async def run_all_tests():
    """Run all DELETE variations"""

    print("🔐 Getting authentication headers...")
    auth_headers = await get_ig_auth_headers()
    print("✅ Auth headers obtained\n")

    # Close direction (opposite of position)
    close_direction = "BUY" if DIRECTION == "SELL" else "SELL"

    results = {}

    # ========== Test 1: DELETE with dealId in URL, minimal body ==========
    results['test1'] = await test_delete_variation(
        "DELETE /positions/otc/{dealId} - Minimal body",
        f"{API_BASE_URL}/positions/otc/{DEAL_ID}",
        {**auth_headers, "Version": "1"},
        {
            "direction": close_direction,
            "size": SIZE_TO_CLOSE
        }
    )

    # ========== Test 2: DELETE with dealId in URL, full body ==========
    results['test2'] = await test_delete_variation(
        "DELETE /positions/otc/{dealId} - Full body",
        f"{API_BASE_URL}/positions/otc/{DEAL_ID}",
        {**auth_headers, "Version": "1"},
        {
            "direction": close_direction,
            "size": SIZE_TO_CLOSE,
            "orderType": "MARKET"
        }
    )

    # ========== Test 3: DELETE with dealId in body ==========
    results['test3'] = await test_delete_variation(
        "DELETE /positions/otc - dealId in body",
        f"{API_BASE_URL}/positions/otc",
        {**auth_headers, "Version": "1"},
        {
            "dealId": DEAL_ID,
            "direction": close_direction,
            "size": SIZE_TO_CLOSE
        }
    )

    # ========== Test 4: DELETE v2 ==========
    results['test4'] = await test_delete_variation(
        "DELETE v2 /positions/otc/{dealId}",
        f"{API_BASE_URL}/positions/otc/{DEAL_ID}",
        {**auth_headers, "Version": "2"},
        {
            "direction": close_direction,
            "size": SIZE_TO_CLOSE
        }
    )

    # ========== Test 5: POST with _method: DELETE header ==========
    headers_with_method = {
        **auth_headers,
        "_method": "DELETE",
        "Version": "1"
    }
    results['test5'] = await test_delete_variation(
        "POST with _method:DELETE header",
        f"{API_BASE_URL}/positions/otc",
        headers_with_method,
        {
            "dealId": DEAL_ID,
            "direction": close_direction,
            "size": SIZE_TO_CLOSE
        },
        method="POST"
    )

    # ========== Test 6: DELETE with epic and expiry ==========
    results['test6'] = await test_delete_variation(
        "DELETE with epic and expiry",
        f"{API_BASE_URL}/positions/otc/{DEAL_ID}",
        {**auth_headers, "Version": "1"},
        {
            "epic": EPIC,
            "expiry": "-",
            "direction": close_direction,
            "size": SIZE_TO_CLOSE,
            "orderType": "MARKET"
        }
    )

    # ========== Test 7: DELETE with string size ==========
    results['test7'] = await test_delete_variation(
        "DELETE with string size",
        f"{API_BASE_URL}/positions/otc/{DEAL_ID}",
        {**auth_headers, "Version": "1"},
        {
            "direction": close_direction,
            "size": str(SIZE_TO_CLOSE)  # String instead of number
        }
    )

    # ========== Test 8: DELETE with level parameter ==========
    results['test8'] = await test_delete_variation(
        "DELETE with level=null",
        f"{API_BASE_URL}/positions/otc/{DEAL_ID}",
        {**auth_headers, "Version": "1"},
        {
            "direction": close_direction,
            "size": SIZE_TO_CLOSE,
            "level": None
        }
    )

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")

    # Check if any succeeded
    if any(results.values()):
        print(f"\n🎉 At least one test succeeded!")
        return True
    else:
        print(f"\n❌ All tests failed")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 COMPREHENSIVE DELETE PARTIAL CLOSE TESTS")
    print("=" * 60)
    print(f"\nPosition to test:")
    print(f"  Deal ID: {DEAL_ID}")
    print(f"  Epic: {EPIC}")
    print(f"  Direction: {DIRECTION}")
    print(f"  Size to close: {SIZE_TO_CLOSE}")
    print()

    success = asyncio.run(run_all_tests())

    print("\n" + "=" * 60)
    if success:
        print("✅ TESTING COMPLETE - At least one method worked!")
    else:
        print("❌ TESTING COMPLETE - All methods failed")
    print("=" * 60)
