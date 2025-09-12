#!/usr/bin/env python3
"""
Check if there's actually corruption in the chart_streamer data by comparing with API
"""

from datetime import datetime, timezone, timedelta
from services.db import SessionLocal
from services.models import IGCandle
from sqlalchemy import and_
from igstream.ig_auth_prod import ig_login
from services.keyvault import get_secret
import requests
import logging

def check_actual_corruption():
    """Check if chart_streamer entries are actually corrupted compared to API"""
    print("🔍 Checking for actual data corruption in chart_streamer entries")
    
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    # Get authentication
    try:
        api_key = get_secret("prodapikey")
        ig_pwd = get_secret("prodpwd") 
        ig_usr = "rehnmarkh"
        
        import asyncio
        async def get_auth():
            return await ig_login(api_key, ig_pwd, ig_usr)
        
        auth = asyncio.run(get_auth())
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
        # Get recent chart_streamer entries
        chart_entries = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.data_source == 'chart_streamer'
            )
        ).order_by(IGCandle.start_time.desc()).limit(5).all()
        
        if not chart_entries:
            print("❌ No chart_streamer entries found")
            return False
        
        print(f"✅ Found {len(chart_entries)} recent chart_streamer entries:")
        for entry in chart_entries:
            print(f"   {entry.start_time}: {entry.close:.5f}")
        
        # Test one entry by fetching API data
        test_entry = chart_entries[0]
        print(f"\n🧪 Testing entry: {test_entry.start_time}")
        
        # Format time for API
        from_time = test_entry.start_time.strftime("%Y-%m-%dT%H:%M:%S")
        to_time = (test_entry.start_time + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
        
        try:
            url = f"https://api.ig.com/gateway/deal/prices/{epic}"
            params = {
                "resolution": "MINUTE_5",
                "from": from_time,
                "to": to_time,
                "max": 10
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])
                
                print(f"   ✅ API response: {len(prices)} candles")
                
                if prices:
                    # Find matching candle
                    for price_data in prices:
                        api_time_str = price_data["snapshotTime"]
                        # Parse the API timestamp
                        api_time = datetime.strptime(api_time_str, "%Y-%m-%dT%H:%M:%S")
                        
                        if api_time == test_entry.start_time:
                            api_close = float(price_data["closePrice"]["bid"])
                            db_close = float(test_entry.close)
                            
                            difference = abs(api_close - db_close)
                            pip_difference = difference * 10000  # EUR/USD pip multiplier
                            
                            print(f"   📊 Comparison for {test_entry.start_time}:")
                            print(f"      DB Close:  {db_close:.5f}")
                            print(f"      API Close: {api_close:.5f}")
                            print(f"      Difference: {pip_difference:.1f} pips")
                            
                            if pip_difference > 0.1:
                                print(f"      ✅ CORRUPTION CONFIRMED: {pip_difference:.1f} pip difference")
                                return True
                            else:
                                print(f"      ❌ No significant corruption found")
                                return False
                    
                    print(f"   ❌ No matching timestamp found in API response")
                    return False
                else:
                    print(f"   ❌ No price data in API response")
                    return False
            else:
                print(f"   ❌ API request failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            print(f"   ❌ API request error: {e}")
            return False

def check_data_source_distribution():
    """Check the distribution of data sources to understand the current state"""
    print("\n🔍 Checking data source distribution")
    
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    with SessionLocal() as session:
        # Get count by data source for recent data (last 24 hours)
        from sqlalchemy import func
        yesterday = datetime.now() - timedelta(days=1)
        
        source_counts = session.query(
            IGCandle.data_source,
            func.count().label('count')
        ).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time >= yesterday
            )
        ).group_by(IGCandle.data_source).all()
        
        print(f"📊 Data sources in last 24 hours:")
        total = 0
        for source, count in source_counts:
            print(f"   {source}: {count} entries")
            total += count
        print(f"   Total: {total} entries")
        
        # Check if most data has been corrected already
        chart_streamer_count = sum(count for source, count in source_counts if source == 'chart_streamer')
        corrected_count = sum(count for source, count in source_counts if source == 'api_backfill_fixed')
        
        if chart_streamer_count == 0:
            print("\n💡 INSIGHT: No chart_streamer entries in last 24 hours")
            print("   All recent data might have already been corrected")
        elif corrected_count > chart_streamer_count:
            print(f"\n💡 INSIGHT: More corrected ({corrected_count}) than chart_streamer ({chart_streamer_count}) entries")
            print("   Most data appears to have been processed already")

if __name__ == "__main__":
    print("🧪 Checking for actual data corruption...")
    
    corruption_found = check_actual_corruption()
    check_data_source_distribution()
    
    print(f"\n🏁 SUMMARY:")
    if corruption_found:
        print("   ✅ Corruption confirmed - weekly corrector should find and fix differences")
        print("   💡 If corrector shows 0 corrections, there might be an issue with the comparison logic")
    else:
        print("   ❌ No significant corruption found in sample")
        print("   💡 Data might already be correct or the +8 pip issue might not exist")
        print("   💡 Or the API might be returning the same incorrect data as stored")