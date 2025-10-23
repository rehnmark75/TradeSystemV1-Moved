#!/usr/bin/env python3
"""
Verify position on IG platform after partial close
"""
import asyncio
import httpx
from dependencies import get_ig_auth_headers
from config import API_BASE_URL

async def verify_positions():
    """Check open positions on IG"""

    print("🔐 Getting authentication headers...")
    auth_headers = await get_ig_auth_headers()
    print("✅ Auth headers obtained\n")

    print("📊 Fetching open positions from IG...")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/positions",
            headers=auth_headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            positions = data.get('positions', [])

            print(f"✅ Found {len(positions)} open position(s)\n")

            # Calculate net position
            net_positions = {}

            for pos in positions:
                market = pos.get('market', {})
                position = pos.get('position', {})

                epic = market.get('epic', 'Unknown')
                deal_id = position.get('dealId', 'Unknown')
                direction = position.get('direction', 'Unknown')
                size = position.get('size', 0)
                level = position.get('level', 0)

                print(f"📍 Position:")
                print(f"   Epic: {epic}")
                print(f"   Deal ID: {deal_id}")
                print(f"   Direction: {direction}")
                print(f"   Size: {size}")
                print(f"   Entry Level: {level}")

                # Track net position per epic
                if epic not in net_positions:
                    net_positions[epic] = {'BUY': 0, 'SELL': 0}

                if direction == 'BUY':
                    net_positions[epic]['BUY'] += size
                elif direction == 'SELL':
                    net_positions[epic]['SELL'] += size

                if deal_id == "DIAAAAVFUTEWHAT":
                    print(f"\n   🎯 THIS IS TRADE 1340!")
                    if size == 0.5:
                        print(f"   ✅ VERIFIED: Position size is 0.5 (partial close successful!)")
                    else:
                        print(f"   ⚠️ WARNING: Expected size 0.5, got {size}")
                print()

            # Show net positions
            print(f"\n📊 NET POSITIONS:")
            for epic, positions in net_positions.items():
                buy_size = positions['BUY']
                sell_size = positions['SELL']
                net_size = buy_size - sell_size
                net_direction = 'BUY' if net_size > 0 else 'SELL' if net_size < 0 else 'FLAT'

                print(f"   {epic}:")
                print(f"      BUY: {buy_size}, SELL: {sell_size}")
                print(f"      NET: {abs(net_size)} {net_direction}")

                if epic == 'CS.D.USDCHF.MINI.IP' and abs(net_size) == 0.5:
                    print(f"      ✅ PARTIAL CLOSE VERIFIED! Net position is 0.5!")
        else:
            print(f"❌ Error fetching positions: {response.status_code}")
            print(f"   {response.text}")

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 IG POSITION VERIFICATION")
    print("=" * 60)
    print()

    asyncio.run(verify_positions())

    print("=" * 60)
