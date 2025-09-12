#!/usr/bin/env python3
"""
Debug why EURJPY data is missing despite being configured
"""

import sys
sys.path.append('/app/forex_scanner')

from datetime import datetime, timedelta
from core.database import DatabaseManager
import config

def debug_missing_eurjpy():
    """Debug why EURJPY is missing from database"""
    
    print("🔍 DEBUGGING MISSING EURJPY DATA")
    print("=" * 50)
    
    # Check configured epics
    epics = [
        "CS.D.EURUSD.MINI.IP",
        "CS.D.GBPUSD.MINI.IP", 
        "CS.D.USDJPY.MINI.IP",
        "CS.D.AUDUSD.MINI.IP",
        "CS.D.USDCAD.MINI.IP",
        "CS.D.EURJPY.MINI.IP",  # This should be monitored!
        "CS.D.AUDJPY.MINI.IP",
        "CS.D.NZDUSD.MINI.IP", 
        "CS.D.USDCHF.MINI.IP"
    ]
    
    print("📋 CONFIGURED EPICS:")
    for i, epic in enumerate(epics, 1):
        print(f"  {i}. {epic}")
    
    db_manager = DatabaseManager(config.DATABASE_URL)
    
    print(f"\n🔍 CHECKING DATABASE STATUS FOR EACH EPIC:")
    print("-" * 60)
    
    recent_cutoff = datetime.now() - timedelta(days=1)
    
    for epic in epics:
        # Check if epic exists at all
        existence_query = """
        SELECT COUNT(*) as count 
        FROM candles 
        WHERE epic = :epic
        """
        
        existence_result = db_manager.execute_query(existence_query, {'epic': epic})
        total_count = existence_result['count'].iloc[0] if not existence_result.empty else 0
        
        # Check recent data
        recent_query = """
        SELECT COUNT(*) as count, MAX(start_time) as latest_time
        FROM candles 
        WHERE epic = :epic 
        AND start_time >= :cutoff
        """
        
        recent_result = db_manager.execute_query(recent_query, {
            'epic': epic,
            'cutoff': recent_cutoff
        })
        
        if not recent_result.empty:
            recent_count = recent_result['count'].iloc[0]
            latest_time = recent_result['latest_time'].iloc[0]
        else:
            recent_count = 0
            latest_time = None
            
        # Status indicator
        if total_count == 0:
            status = "❌ NO DATA"
        elif recent_count == 0:
            status = "⚠️ OLD DATA"
        elif recent_count < 50:  # Less than ~12 hours of 15m data
            status = "⚠️ SPARSE"
        else:
            status = "✅ ACTIVE"
            
        print(f"{status} {epic}")
        print(f"     Total records: {total_count:,}")
        print(f"     Recent records (24h): {recent_count}")
        print(f"     Latest data: {latest_time}")
        
        if epic == "CS.D.EURJPY.MINI.IP" and total_count == 0:
            print(f"     🚨 EURJPY MISSING - This explains the missing signal!")
        print()
    
    # Check for any data collection errors
    print(f"\n🔍 CHECKING FOR DATA COLLECTION ISSUES:")
    print("-" * 50)
    
    # Look for recent data to understand the collection pattern
    summary_query = """
    SELECT epic, 
           COUNT(*) as total_records,
           MIN(start_time) as first_record,
           MAX(start_time) as last_record
    FROM candles 
    WHERE start_time >= :cutoff
    GROUP BY epic
    ORDER BY total_records DESC
    """
    
    summary = db_manager.execute_query(summary_query, {'cutoff': recent_cutoff - timedelta(days=7)})
    
    if not summary.empty:
        print("📊 RECENT DATA COLLECTION SUMMARY (Last 7 days):")
        for _, row in summary.iterrows():
            print(f"  {row['epic']}: {row['total_records']} records")
            print(f"    Period: {row['first_record']} to {row['last_record']}")
    
    # Check distinct timeframes
    print(f"\n🔍 AVAILABLE TIMEFRAMES:")
    tf_query = "SELECT DISTINCT timeframe FROM candles ORDER BY timeframe"
    timeframes = db_manager.execute_query(tf_query)
    if not timeframes.empty:
        print(f"  Timeframes: {timeframes['timeframe'].tolist()}")
    
    # Specific EURJPY investigation
    print(f"\n🎯 SPECIFIC EURJPY INVESTIGATION:")
    print("-" * 40)
    
    # Check if there's ANY EURJPY data with slight name variations
    variant_query = """
    SELECT DISTINCT epic 
    FROM candles 
    WHERE epic ILIKE '%EUR%' AND epic ILIKE '%JPY%'
    """
    
    variants = db_manager.execute_query(variant_query)
    if not variants.empty:
        print("✅ Found EURJPY variants:")
        for variant in variants['epic']:
            print(f"  - {variant}")
    else:
        print("❌ No EURJPY data found with any name variation")
        print("\n💡 POSSIBLE CAUSES:")
        print("1. Stream service not collecting EURJPY data")
        print("2. EURJPY epic not active/available from IG Markets")
        print("3. Data collection service stopped/crashed")
        print("4. Database write permissions issue")
        print("5. Epic name mismatch between config and IG API")
        
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("1. Check streaming service logs for EURJPY")
        print("2. Verify IG Markets API availability for EURJPY")
        print("3. Check if backfill service is running")
        print("4. Test manual EURJPY data fetch from IG API")

if __name__ == "__main__":
    debug_missing_eurjpy()