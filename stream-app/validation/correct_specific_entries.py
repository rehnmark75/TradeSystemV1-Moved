#!/usr/bin/env python3
"""
Correct specific chart_streamer entries that contain corrupted data (+8 pips)
Focus on: 2025-09-01 02:40:00, 02:45:00, 02:50:00, 02:55:00
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

async def correct_specific_entries(dry_run=False):
    """Correct the specific corrupted chart_streamer entries"""
    print("🔧 Correcting specific chart_streamer entries with corrupted data")
    
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    # Define the specific timestamps that need correction
    target_timestamps = [
        datetime(2025, 9, 1, 2, 40, 0),
        datetime(2025, 9, 1, 2, 45, 0),
        datetime(2025, 9, 1, 2, 50, 0),
        datetime(2025, 9, 1, 2, 55, 0)
    ]
    
    print(f"📋 Mode: {'DRY RUN' if dry_run else 'LIVE CORRECTION'}")
    print(f"📅 Target timestamps: {len(target_timestamps)} entries")
    for ts in target_timestamps:
        print(f"   {ts}")
    
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
    
    with SessionLocal() as session:
        # Find the specific entries that need correction
        entries_to_correct = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time.in_(target_timestamps),
                IGCandle.data_source == 'chart_streamer'
            )
        ).all()
        
        print(f"\n🔍 Found {len(entries_to_correct)} chart_streamer entries to correct:")
        for entry in entries_to_correct:
            print(f"   {entry.start_time}: {entry.close:.5f} ({entry.data_source})")
        
        if not entries_to_correct:
            print("❌ No chart_streamer entries found for the specified timestamps")
            return False
        
        # Fetch correct data from API for the time range
        print(f"\n📥 Fetching correct data from API...")
        try:
            # Get a wider time range to ensure we get all the data
            from_time = (min(target_timestamps) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
            to_time = (max(target_timestamps) + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")
            
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
            
            # Process corrections
            corrections_made = 0
            
            for entry in entries_to_correct:
                # Find matching API data
                api_data = None
                for price_data in prices:
                    try:
                        # Parse API timestamp (format: "2025/09/01 02:40:00")
                        api_time_str = price_data["snapshotTime"]
                        api_time = datetime.strptime(api_time_str, "%Y/%m/%d %H:%M:%S")
                        
                        if api_time == entry.start_time:
                            api_data = price_data
                            break
                    except Exception as e:
                        continue
                
                if not api_data:
                    print(f"   ❌ No API data found for {entry.start_time}")
                    continue
                
                # Extract correct OHLC values
                correct_open = float(api_data["openPrice"]["bid"])
                correct_high = float(api_data["highPrice"]["bid"])
                correct_low = float(api_data["lowPrice"]["bid"])
                correct_close = float(api_data["closePrice"]["bid"])
                
                # Calculate differences
                close_diff = abs(correct_close - entry.close)
                pip_difference = close_diff * 10000  # EUR/USD pip multiplier
                
                print(f"\n🎯 Processing {entry.start_time}:")
                print(f"   Current: O={entry.open:.5f} H={entry.high:.5f} L={entry.low:.5f} C={entry.close:.5f}")
                print(f"   Correct: O={correct_open:.5f} H={correct_high:.5f} L={correct_low:.5f} C={correct_close:.5f}")
                print(f"   Close difference: {pip_difference:.1f} pips")
                
                if pip_difference > 0.1:
                    print(f"   ✅ Correction needed - applying update...")
                    
                    if not dry_run:
                        # Apply correction using the same method as weekly corrector
                        update_data = {
                            'open': correct_open,
                            'high': correct_high,
                            'low': correct_low,
                            'close': correct_close,
                            'data_source': 'api_backfill_fixed',
                            'updated_at': datetime.now()
                        }
                        
                        rows_updated = session.query(IGCandle).filter(
                            and_(
                                IGCandle.epic == epic,
                                IGCandle.timeframe == timeframe,
                                IGCandle.start_time == entry.start_time
                            )
                        ).update(update_data)
                        
                        if rows_updated > 0:
                            corrections_made += 1
                            print(f"      ✅ Database updated")
                        else:
                            print(f"      ❌ Database update failed - no rows affected")
                    else:
                        corrections_made += 1
                        print(f"      ✅ Would update (dry run)")
                else:
                    print(f"   ⏭️ No significant correction needed")
            
            # Commit changes
            if corrections_made > 0 and not dry_run:
                print(f"\n💾 Committing {corrections_made} corrections...")
                try:
                    session.commit()
                    print(f"✅ Successfully committed corrections")
                    
                    # Verify the corrections
                    print(f"\n🔍 Verifying corrections...")
                    corrected_entries = session.query(IGCandle).filter(
                        and_(
                            IGCandle.epic == epic,
                            IGCandle.timeframe == timeframe,
                            IGCandle.start_time.in_(target_timestamps)
                        )
                    ).order_by(IGCandle.start_time).all()
                    
                    print(f"📊 Entries after correction:")
                    for entry in corrected_entries:
                        print(f"   {entry.start_time}: {entry.close:.5f} ({entry.data_source})")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Commit failed: {e}")
                    session.rollback()
                    return False
            elif corrections_made > 0:
                print(f"\n✅ Dry run complete - {corrections_made} corrections would be applied")
                return True
            else:
                print("\nℹ️ No corrections needed")
                return True
                
        except Exception as e:
            print(f"❌ API request error: {e}")
            return False

if __name__ == "__main__":
    import asyncio
    import sys
    
    dry_run = "--dry-run" in sys.argv
    
    print("🧪 Correcting specific corrupted chart_streamer entries...")
    result = asyncio.run(correct_specific_entries(dry_run=dry_run))
    
    if result:
        if dry_run:
            print("\n🏁 DRY RUN SUCCESS: Ready to correct corrupted entries")
            print("💡 Run without --dry-run to apply corrections")
        else:
            print("\n🏁 SUCCESS: Corrupted entries have been corrected")
    else:
        print("\n❌ FAILED: Could not correct the entries")