#!/usr/bin/env python3
"""
Fill specific gap in EURUSD data between 2025-09-01 02:35:00 and 03:00:00
"""

from datetime import datetime, timezone, timedelta
from services.db import SessionLocal
from services.models import IGCandle
from sqlalchemy import and_
from igstream.ig_auth_prod import ig_login
from services.keyvault import get_secret
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fill_specific_gap():
    """Fill the specific gap identified by the user"""
    print("🔧 Filling gap between 2025-09-01 02:35:00 and 03:00:00")
    
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    # Define the gap times
    gap_start = datetime(2025, 9, 1, 2, 35, 0)
    gap_end = datetime(2025, 9, 1, 3, 0, 0)
    
    print(f"📅 Gap period: {gap_start} to {gap_end}")
    print(f"📊 Expected missing candles: {((gap_end - gap_start).total_seconds() // 300) - 1}")  # Minus 1 because we have the start candle
    
    # Get authentication
    try:
        api_key = get_secret("prodapikey")
        ig_pwd = get_secret("prodpwd")
        ig_usr = "rehnmarkh"
        
        auth = await ig_login(api_key, ig_pwd, ig_usr)
        headers = {
            "CST": auth["CST"],
            "X-SECURITY-TOKEN": auth["X-SECURITY-TOKEN"],
            "X-IG-API-KEY": api_key,
            "VERSION": "3"
        }
        print("✅ Authenticated with IG API")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Verify the gap exists
    with SessionLocal() as session:
        print("\n🔍 Verifying gap...")
        
        # Check entries around the gap
        entries_around_gap = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time >= gap_start - timedelta(minutes=10),
                IGCandle.start_time <= gap_end + timedelta(minutes=10)
            )
        ).order_by(IGCandle.start_time).all()
        
        print(f"✅ Found {len(entries_around_gap)} entries around gap:")
        for entry in entries_around_gap:
            print(f"   {entry.start_time}: {entry.close:.5f} ({entry.data_source})")
        
        # Check what's missing in the gap
        expected_times = []
        current_time = gap_start + timedelta(minutes=5)  # Start with next 5-minute interval
        while current_time < gap_end:
            expected_times.append(current_time)
            current_time += timedelta(minutes=5)
        
        existing_times = [entry.start_time for entry in entries_around_gap]
        missing_times = [t for t in expected_times if t not in existing_times]
        
        print(f"\n📋 Missing timestamps:")
        for missing_time in missing_times:
            print(f"   {missing_time}")
        
        if not missing_times:
            print("✅ No gap found - all timestamps are present")
            return True
        
        # Fetch data from API
        print(f"\n📥 Fetching data from API...")
        try:
            # Format times for API (extend range slightly to ensure we get all data)
            from_time = (gap_start - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
            to_time = (gap_end + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
            
            url = f"https://api.ig.com/gateway/deal/prices/{epic}"
            params = {
                "resolution": "MINUTE_5",
                "from": from_time,
                "to": to_time,
                "max": 20
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"❌ API request failed: {response.status_code} - {response.text}")
                return False
            
            data = response.json()
            prices = data.get("prices", [])
            
            print(f"✅ Received {len(prices)} candles from API")
            
            if not prices:
                print("❌ No price data received from API")
                return False
            
            # Process API data and fill missing candles
            candles_added = 0
            for price_data in prices:
                try:
                    # Parse API timestamp (format: "2025/09/01 02:40:00")
                    api_time_str = price_data["snapshotTime"]
                    api_time = datetime.strptime(api_time_str, "%Y/%m/%d %H:%M:%S")
                    
                    # Check if this timestamp is missing
                    if api_time in missing_times:
                        # Extract OHLC data
                        open_price = float(price_data["openPrice"]["bid"])
                        high_price = float(price_data["highPrice"]["bid"])
                        low_price = float(price_data["lowPrice"]["bid"])
                        close_price = float(price_data["closePrice"]["bid"])
                        
                        # Create new candle
                        new_candle = IGCandle(
                            epic=epic,
                            timeframe=timeframe,
                            start_time=api_time,
                            open=open_price,
                            high=high_price,
                            low=low_price,
                            close=close_price,
                            volume=0,  # Volume not available in API response
                            tick_count=0,
                            data_source='api_gap_fill',
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        )
                        
                        session.add(new_candle)
                        candles_added += 1
                        
                        print(f"   ➕ Adding {api_time}: O={open_price:.5f} H={high_price:.5f} L={low_price:.5f} C={close_price:.5f}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing candle: {e}")
                    continue
            
            if candles_added > 0:
                print(f"\n💾 Committing {candles_added} new candles...")
                try:
                    session.commit()
                    print(f"✅ Successfully added {candles_added} candles")
                    
                    # Verify the gap is filled
                    print(f"\n🔍 Verifying gap fill...")
                    verification_entries = session.query(IGCandle).filter(
                        and_(
                            IGCandle.epic == epic,
                            IGCandle.timeframe == timeframe,
                            IGCandle.start_time >= gap_start - timedelta(minutes=10),
                            IGCandle.start_time <= gap_end + timedelta(minutes=10)
                        )
                    ).order_by(IGCandle.start_time).all()
                    
                    print(f"📊 Entries after gap fill ({len(verification_entries)} total):")
                    for entry in verification_entries:
                        print(f"   {entry.start_time}: {entry.close:.5f} ({entry.data_source})")
                    
                    # Check if gap is fully filled
                    verification_times = [entry.start_time for entry in verification_entries]
                    still_missing = [t for t in expected_times if t not in verification_times]
                    
                    if still_missing:
                        print(f"\n⚠️ Still missing {len(still_missing)} timestamps:")
                        for missing_time in still_missing:
                            print(f"   {missing_time}")
                    else:
                        print(f"\n✅ Gap successfully filled - no missing timestamps")
                    
                    return len(still_missing) == 0
                    
                except Exception as e:
                    print(f"❌ Commit failed: {e}")
                    session.rollback()
                    return False
            else:
                print("ℹ️ No candles to add")
                return True
                
        except Exception as e:
            print(f"❌ API request error: {e}")
            return False

if __name__ == "__main__":
    import asyncio
    
    print("🧪 Filling specific gap in EURUSD data...")
    result = asyncio.run(fill_specific_gap())
    
    if result:
        print("\n🏁 SUCCESS: Gap has been filled")
    else:
        print("\n❌ FAILED: Could not fill the gap")