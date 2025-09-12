#!/usr/bin/env python3
"""
Test script for the weekly close price corrector
Tests with a single epic and timeframe to validate functionality
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from weekly_close_price_corrector import WeeklyClosePriceCorrector
from datetime import datetime, timedelta

async def test_single_correction():
    """Test correction with a single epic and timeframe"""
    print("🧪 Testing Weekly Close Price Corrector")
    print("="*50)
    
    # Create corrector in dry-run mode for safety
    corrector = WeeklyClosePriceCorrector(dry_run=True)
    
    # Override to test only EURUSD 5m for last 24 hours
    corrector.forex_pairs = ['CS.D.EURUSD.CEEM.IP']
    corrector.timeframes = [5]
    
    print("🎯 Test Configuration:")
    print(f"   Epic: {corrector.forex_pairs[0]}")
    print(f"   Timeframe: {corrector.timeframes[0]}m")
    print(f"   Mode: DRY RUN (safe testing)")
    print()
    
    # Test authentication
    print("🔐 Testing IG API authentication...")
    if not await corrector.authenticate():
        print("❌ Authentication failed - check credentials")
        return False
    
    print("✅ Authentication successful")
    
    # Test API data fetch
    print("\n📡 Testing API data fetch...")
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=6)  # Last 6 hours for quick test
    
    try:
        api_data = await corrector.fetch_api_data(
            'CS.D.EURUSD.CEEM.IP', 
            5, 
            start_time, 
            end_time
        )
        
        if api_data:
            print(f"✅ API fetch successful - got {len(api_data)} candles")
            
            # Show sample data structure
            if len(api_data) > 0:
                sample = api_data[0]
                print(f"\n📊 Sample API response structure:")
                print(f"   Timestamp: {sample.get('snapshotTime')}")
                
                close_price = corrector.extract_close_price(sample)
                if close_price:
                    print(f"   Close Price: {close_price}")
                
                ohlc = corrector.extract_ohlc_prices(sample)
                if ohlc:
                    print(f"   OHLC: {ohlc}")
                
        else:
            print("⚠️ No API data returned")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    
    # Test database query
    print(f"\n🗃️ Testing database query for corrupted entries...")
    try:
        corrupted_entries = await corrector.get_corrupted_entries(
            'CS.D.EURUSD.CEEM.IP',
            5,
            start_time,
            end_time
        )
        
        print(f"✅ Database query successful - found {len(corrupted_entries)} corrupted entries")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Corrector is ready for use.")
    print("\n💡 Usage examples:")
    print("   # Dry run (safe testing):")
    print("   python weekly_close_price_corrector.py --dry-run")
    print("   ")
    print("   # Test single epic:")
    print("   python weekly_close_price_corrector.py --dry-run --epic CS.D.EURUSD.CEEM.IP")
    print("   ")
    print("   # Live correction (applies changes):")
    print("   python weekly_close_price_corrector.py --live")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_single_correction())