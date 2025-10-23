#!/usr/bin/env python3
"""
Check deal confirmation for the partial close
"""
import asyncio
import httpx
from dependencies import get_ig_auth_headers
from config import API_BASE_URL

async def check_deal(deal_ref):
    """Check deal confirmation"""

    print(f"🔐 Getting authentication headers...")
    auth_headers = await get_ig_auth_headers()
    print("✅ Auth headers obtained\n")

    print(f"📊 Checking deal confirmation for: {deal_ref}")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/confirms/{deal_ref}",
            headers=auth_headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            print(f"\n✅ Deal Confirmation Retrieved:")
            print(f"   Deal Reference: {data.get('dealReference')}")
            print(f"   Deal ID: {data.get('dealId')}")
            print(f"   Deal Status: {data.get('dealStatus')}")
            print(f"   Status: {data.get('status')}")
            print(f"   Reason: {data.get('reason')}")
            print(f"   Epic: {data.get('epic')}")
            print(f"   Direction: {data.get('direction')}")
            print(f"   Size: {data.get('size')}")
            print(f"   Level: {data.get('level')}")
            print(f"   Profit/Loss: {data.get('profit')}")
            print(f"   Profit Currency: {data.get('profitCurrency')}")

            # Print full response for debugging
            print(f"\n📄 Full Response:")
            import json
            print(json.dumps(data, indent=2))

            return data
        else:
            print(f"❌ Error checking deal: {response.status_code}")
            print(f"   {response.text}")
            return None

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 DEAL CONFIRMATION CHECK")
    print("=" * 60)
    print()

    # Deal reference from the partial close
    deal_ref = "MW7RQMSLHZNT28R"

    asyncio.run(check_deal(deal_ref))

    print("\n" + "=" * 60)
