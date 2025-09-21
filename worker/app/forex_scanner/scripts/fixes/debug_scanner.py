#!/usr/bin/env python3
"""
Debug script to check data timestamps
"""

import sys
import os
sys.path.insert(0, '/app/forex_scanner')

import config
from core.database import DatabaseManager
from core.data_fetcher import DataFetcher

def debug_data_timestamps():
    """Debug data fetcher timestamps"""
    
    # Initialize components
    try:
        db_manager = DatabaseManager(config.DATABASE_URL)
        data_fetcher = DataFetcher(db_manager, config.USER_TIMEZONE)
        print("✅ Components initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Test EUR/USD data fetching
    epic = 'CS.D.EURUSD.CEEM.IP'
    pair = 'EURUSD'
    
    print(f"\n🔍 Testing data fetcher for {epic}...")
    
    try:
        # Get enhanced data
        df = data_fetcher.get_enhanced_data(epic, pair, timeframe='5m')
        
        if df is None or len(df) == 0:
            print("❌ No data returned")
            return
        
        print(f"✅ Data fetched: {len(df)} rows")
        
        # Check the last 10 timestamps
        print(f"\n📅 Last 10 timestamps in data:")
        last_10 = df.tail(10)[['start_time', 'close']].copy()
        
        for idx, row in last_10.iterrows():
            print(f"  {idx}: {row['start_time']} | Close: {row['close']:.5f}")
        
        # Check if timestamps are all the same
        unique_timestamps = df['start_time'].nunique()
        total_rows = len(df)
        
        print(f"\n📊 Timestamp Analysis:")
        print(f"  Total rows: {total_rows}")
        print(f"  Unique timestamps: {unique_timestamps}")
        print(f"  Latest timestamp: {df.iloc[-1]['start_time']}")
        print(f"  Earliest timestamp: {df.iloc[0]['start_time']}")
        
        if unique_timestamps == 1:
            print("⚠️  WARNING: All timestamps are identical!")
            print(f"  All rows have timestamp: {df.iloc[0]['start_time']}")
        elif unique_timestamps == total_rows:
            print("✅ All timestamps are unique (normal)")
        else:
            print(f"⚠️  Some timestamps are duplicated")
        
        # Check latest vs previous timestamp
        if len(df) >= 2:
            latest_time = df.iloc[-1]['start_time']
            previous_time = df.iloc[-2]['start_time']
            print(f"\n🕐 Time Progression:")
            print(f"  Previous: {previous_time}")
            print(f"  Latest:   {latest_time}")
            
            if latest_time == previous_time:
                print("❌ Latest and previous timestamps are identical!")
            else:
                time_diff = latest_time - previous_time
                print(f"  Difference: {time_diff}")
        
    except Exception as e:
        print(f"❌ Data fetching error: {e}")

if __name__ == "__main__":
    debug_data_timestamps()