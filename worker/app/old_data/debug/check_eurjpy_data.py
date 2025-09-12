#!/usr/bin/env python3
"""
Check what EURJPY data exists in the database
"""

import sys
sys.path.append('/app/forex_scanner')

from datetime import datetime, timedelta
import pandas as pd
from core.database import DatabaseManager
import config

def check_eurjpy_data():
    """Check what EURJPY data exists"""
    
    print("🔍 CHECKING EURJPY DATA IN DATABASE")
    print("=" * 50)
    
    db_manager = DatabaseManager(config.DATABASE_URL)
    
    # Check all epics with EURJPY
    query1 = """
    SELECT DISTINCT epic FROM candles 
    WHERE epic ILIKE '%EURJPY%' OR epic ILIKE '%EUR%JPY%'
    ORDER BY epic
    """
    
    eurjpy_epics = db_manager.execute_query(query1)
    
    if eurjpy_epics is None or len(eurjpy_epics) == 0:
        print("❌ No EURJPY epics found")
        
        # Check all available epics
        print("\n🔍 Checking all available epics:")
        all_epics = db_manager.execute_query("SELECT DISTINCT epic FROM candles ORDER BY epic LIMIT 10")
        if all_epics is not None and len(all_epics) > 0:
            print("Available epics (first 10):")
            for epic in all_epics['epic'][:10]:
                print(f"  - {epic}")
        return
    
    print(f"✅ Found EURJPY epics:")
    for epic in eurjpy_epics['epic']:
        print(f"  - {epic}")
    
    # Check timeframes for the first EURJPY epic
    first_epic = eurjpy_epics['epic'].iloc[0]
    print(f"\n🔍 Checking timeframes for {first_epic}:")
    
    query2 = """
    SELECT DISTINCT timeframe FROM candles 
    WHERE epic = :epic
    ORDER BY timeframe
    """
    
    timeframes = db_manager.execute_query(query2, {'epic': first_epic})
    
    if timeframes is not None and len(timeframes) > 0:
        print("Available timeframes:")
        for tf in timeframes['timeframe']:
            print(f"  - {tf}")
    else:
        print("❌ No timeframes found")
        return
    
    # Check recent data for 15-minute timeframe (assuming 15 = 15m)
    target_date = datetime(2025, 8, 28)
    start_time = target_date - timedelta(hours=6)
    end_time = target_date + timedelta(hours=6)
    
    for tf in [15, 900]:  # Try both 15 and 900 (15min in seconds)
        print(f"\n🔍 Checking recent data for {first_epic} on timeframe {tf}:")
        
        query3 = """
        SELECT start_time, close_price FROM candles 
        WHERE epic = :epic 
        AND timeframe = :timeframe
        AND start_time >= :start_time 
        AND start_time <= :end_time
        ORDER BY start_time
        LIMIT 10
        """
        
        recent_data = db_manager.execute_query(query3, {
            'epic': first_epic,
            'timeframe': tf,
            'start_time': start_time,
            'end_time': end_time
        })
        
        if recent_data is not None and len(recent_data) > 0:
            print(f"✅ Found {len(recent_data)} candles for timeframe {tf}")
            print("Recent candles:")
            for _, row in recent_data.head().iterrows():
                local_time = pd.to_datetime(row['start_time']) + timedelta(hours=2)
                print(f"  {local_time.strftime('%Y-%m-%d %H:%M')} UTC+2: {row['close_price']:.3f}")
                
            # Check specifically around 05:15 UTC+2 (03:15 UTC)
            target_utc = datetime(2025, 8, 28, 3, 15)
            target_range_start = target_utc - timedelta(minutes=30)
            target_range_end = target_utc + timedelta(minutes=30)
            
            target_data = db_manager.execute_query(query3.replace("LIMIT 10", ""), {
                'epic': first_epic,
                'timeframe': tf, 
                'start_time': target_range_start,
                'end_time': target_range_end
            })
            
            if target_data is not None and len(target_data) > 0:
                print(f"\n🎯 Data around 05:15 UTC+2 (03:15 UTC):")
                for _, row in target_data.iterrows():
                    local_time = pd.to_datetime(row['start_time']) + timedelta(hours=2)
                    print(f"  {local_time.strftime('%H:%M')} UTC+2: {row['close_price']:.3f}")
            else:
                print(f"❌ No data found around target time for timeframe {tf}")
        else:
            print(f"❌ No recent data found for timeframe {tf}")

if __name__ == "__main__":
    check_eurjpy_data()