#!/usr/bin/env python3
"""
Check recent activities to see what happened with the partial close attempts
"""
import asyncio
import httpx
from dependencies import get_ig_auth_headers
from config import API_BASE_URL
from datetime import datetime, timedelta

async def check_activities():
    """Check recent activities"""

    print(f"🔐 Getting authentication headers...")
    auth_headers = await get_ig_auth_headers()
    print("✅ Auth headers obtained\n")

    # Get activities from last hour
    to_time = datetime.now()
    from_time = to_time - timedelta(hours=1)

    print(f"📊 Fetching activities from last hour...")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/history/activity",
            headers=auth_headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            activities = data.get('activities', [])

            print(f"✅ Found {len(activities)} activities\n")

            for activity in activities[:20]:  # Show last 20
                deal_id = activity.get('dealId', 'N/A')
                epic = activity.get('epic', 'N/A')
                activity_type = activity.get('type', 'N/A')
                status = activity.get('status', 'N/A')
                direction = activity.get('direction', 'N/A')
                size = activity.get('size', 'N/A')
                level = activity.get('level', 'N/A')
                date = activity.get('date', 'N/A')

                print(f"📝 Activity:")
                print(f"   Date: {date}")
                print(f"   Type: {activity_type}")
                print(f"   Status: {status}")
                print(f"   Epic: {epic}")
                print(f"   Deal ID: {deal_id}")
                print(f"   Direction: {direction}")
                print(f"   Size: {size}")
                print(f"   Level: {level}")

                if "USDCHF" in epic:
                    print(f"   🎯 USDCHF ACTIVITY!")
                print()
        else:
            print(f"❌ Error fetching activities: {response.status_code}")
            print(f"   {response.text}")

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 ACTIVITY HISTORY CHECK")
    print("=" * 60)
    print()

    asyncio.run(check_activities())

    print("=" * 60)
