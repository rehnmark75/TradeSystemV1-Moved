#!/usr/bin/env python3
"""
Query full position details from IG to see all available fields
"""
import asyncio
import httpx
from dependencies import get_ig_auth_headers
from config import API_BASE_URL
import json

async def get_position_details(deal_id):
    """Get detailed position information"""

    print(f"🔐 Getting authentication headers...")
    auth_headers = await get_ig_auth_headers()
    print("✅ Auth headers obtained\n")

    # First get all positions to find our position
    print(f"📊 Fetching all positions from IG...")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/positions",
            headers=auth_headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            positions = data.get('positions', [])

            print(f"✅ Found {len(positions)} position(s)\n")

            for pos in positions:
                position = pos.get('position', {})
                market = pos.get('market', {})

                if position.get('dealId') == deal_id:
                    print(f"🎯 FOUND POSITION {deal_id}!\n")
                    print("=" * 60)
                    print("FULL POSITION OBJECT:")
                    print("=" * 60)
                    print(json.dumps(pos, indent=2))
                    print("\n" + "=" * 60)
                    print("POSITION DETAILS:")
                    print("=" * 60)
                    print(json.dumps(position, indent=2))
                    print("\n" + "=" * 60)
                    print("MARKET DETAILS:")
                    print("=" * 60)
                    print(json.dumps(market, indent=2))
                    print("\n" + "=" * 60)

                    # Extract key fields for DELETE
                    print("\n📋 KEY FIELDS FOR DELETE OPERATION:")
                    print(f"   dealId: {position.get('dealId')}")
                    print(f"   direction: {position.get('direction')}")
                    print(f"   size: {position.get('size')}")
                    print(f"   contractSize: {position.get('contractSize')}")
                    print(f"   currency: {position.get('currency')}")
                    print(f"   level: {position.get('level')}")
                    print(f"   epic: {market.get('epic')}")
                    print(f"   expiry: {market.get('expiry')}")
                    print(f"   instrumentType: {market.get('instrumentType')}")

                    return pos

            print(f"❌ Position {deal_id} not found in open positions")
            return None
        else:
            print(f"❌ Error fetching positions: {response.status_code}")
            print(f"   {response.text}")
            return None

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 POSITION DETAILS QUERY")
    print("=" * 60)
    print()

    # Query position for trade 1340
    deal_id = "DIAAAAVFUTEWHAT"

    asyncio.run(get_position_details(deal_id))

    print("\n" + "=" * 60)
