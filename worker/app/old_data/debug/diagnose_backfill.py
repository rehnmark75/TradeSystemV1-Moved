#!/usr/bin/env python3
"""
Diagnostic script to analyze backfill data quality issues
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up path
sys.path.append('/app/forex_scanner')

from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
import config

def diagnose_recent_data_quality(epic="CS.D.EURUSD.MINI.IP", hours_back=6):
    """
    Analyze recent data quality to understand incomplete candle warnings
    """
    print(f"🔍 DIAGNOSING DATA QUALITY FOR {epic}")
    print("=" * 60)
    
    # Initialize components
    db_manager = DatabaseManager(config.DATABASE_URL)
    data_fetcher = DataFetcher(db_manager)
    
    # Get recent data
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)
    
    print(f"📅 Analyzing data from {start_time} to {end_time}")
    print(f"🔍 Looking for gaps in 5m data that cause incomplete 15m candles")
    
    try:
        # Get raw 5m data
        query = """
        SELECT epic, start_time, close_time, open_price, high_price, low_price, close_price, volume
        FROM candles 
        WHERE epic = :epic 
        AND timeframe = '5m'
        AND start_time >= :start_time 
        AND start_time <= :end_time
        ORDER BY start_time
        """
        
        df = db_manager.execute_query(query, {'epic': epic, 'start_time': start_time, 'end_time': end_time})
        
        if df is None or len(df) == 0:
            print("❌ No 5m data found for the specified period")
            return
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        print(f"✅ Found {len(df)} 5m candles in {hours_back}h period")
        
        # Check for gaps in 5m data
        print(f"\n🔍 CHECKING FOR 5M DATA GAPS:")
        print("-" * 40)
        
        expected_candles = hours_back * 12  # 12 candles per hour for 5m timeframe
        actual_candles = len(df)
        missing_candles = expected_candles - actual_candles
        
        print(f"Expected 5m candles: {expected_candles}")
        print(f"Actual 5m candles: {actual_candles}")
        print(f"Missing candles: {missing_candles}")
        print(f"Completeness: {actual_candles/expected_candles*100:.1f}%")
        
        if missing_candles > 0:
            print(f"⚠️ {missing_candles} candles missing - this causes incomplete 15m candles!")
        
        # Analyze specific gaps
        df_sorted = df.sort_values('start_time')
        df_sorted['time_diff'] = df_sorted['start_time'].diff()
        
        # Find gaps larger than 5 minutes
        gaps = df_sorted[df_sorted['time_diff'] > pd.Timedelta(minutes=5)]
        
        if len(gaps) > 0:
            print(f"\n🚨 FOUND {len(gaps)} DATA GAPS:")
            print("-" * 30)
            for idx, gap in gaps.iterrows():
                gap_duration = gap['time_diff']
                gap_start = df_sorted[df_sorted.index < idx]['start_time'].iloc[-1] if idx > 0 else "Unknown"
                gap_end = gap['start_time']
                missing_periods = int(gap_duration.total_seconds() / 300) - 1  # 300 seconds = 5 minutes
                
                print(f"Gap {len(gaps) - len(gaps) + gaps.index.get_loc(idx) + 1}:")
                print(f"  Duration: {gap_duration}")
                print(f"  From: {gap_start}")
                print(f"  To: {gap_end}")
                print(f"  Missing periods: {missing_periods}")
                print()
        else:
            print("✅ No significant gaps found in 5m data")
        
        # Check 15m synthesis
        print(f"\n🔍 CHECKING 15M SYNTHESIS:")
        print("-" * 30)
        
        # Simulate 15m resampling to see which candles would be incomplete
        df.set_index('start_time', inplace=True)
        
        # Group by 15-minute periods
        df_15m = df.groupby(pd.Grouper(freq='15min', label='left', closed='left', origin='epoch')).agg({
            'open_price': 'first',
            'high_price': 'max', 
            'low_price': 'min',
            'close_price': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Check completeness of each 15m period
        incomplete_15m = []
        for period_start, group in df.groupby(pd.Grouper(freq='15min', label='left', closed='left', origin='epoch')):
            expected_5m = 3  # Should have 3 x 5m candles per 15m period
            actual_5m = len(group)
            
            if actual_5m < expected_5m and actual_5m > 0:  # Incomplete but has some data
                incomplete_15m.append({
                    'period_start': period_start,
                    'expected_5m': expected_5m,
                    'actual_5m': actual_5m,
                    'missing_5m': expected_5m - actual_5m
                })
        
        if incomplete_15m:
            print(f"⚠️ FOUND {len(incomplete_15m)} INCOMPLETE 15M PERIODS:")
            for incomplete in incomplete_15m[-5:]:  # Show last 5
                period = incomplete['period_start']
                print(f"  {period}: {incomplete['actual_5m']}/{incomplete['expected_5m']} 5m candles (missing {incomplete['missing_5m']})")
        else:
            print("✅ All 15m periods have complete 5m data")
            
        # Recent analysis (last 2 hours as per the warning)
        print(f"\n🔍 RECENT 2-HOUR ANALYSIS:")
        print("-" * 35)
        
        recent_cutoff = end_time - timedelta(hours=2)
        recent_incomplete = [x for x in incomplete_15m if x['period_start'] >= recent_cutoff]
        
        print(f"Incomplete 15m candles in last 2 hours: {len(recent_incomplete)}")
        
        if len(recent_incomplete) > 0:
            print("🚨 THIS MATCHES THE WARNING YOU'RE SEEING!")
            print("\n💡 RECOMMENDATIONS:")
            print("1. Check if the backfill service is running properly")
            print("2. Verify IG Markets API connectivity")
            print("3. Look for network interruptions or service restarts")
            print("4. Consider the warning time - was it during market hours?")
            
            # Check market hours
            last_incomplete = recent_incomplete[-1]['period_start']
            day_of_week = last_incomplete.weekday()  # 0=Monday, 6=Sunday
            hour_utc = last_incomplete.hour
            
            if day_of_week >= 5:  # Weekend
                print("5. ⚠️ This occurred on weekend - normal for reduced data")
            elif hour_utc < 22 or hour_utc >= 22:  # Outside major trading hours
                print("5. ⚠️ This was outside major trading hours - potentially normal")
        else:
            print("✅ No recent incomplete candles found")
            print("   The warning may have been resolved or was temporary")
            
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose backfill data quality")
    parser.add_argument("--epic", default="CS.D.EURUSD.MINI.IP", help="Epic to analyze")
    parser.add_argument("--hours", type=int, default=6, help="Hours to look back")
    
    args = parser.parse_args()
    
    diagnose_recent_data_quality(args.epic, args.hours)