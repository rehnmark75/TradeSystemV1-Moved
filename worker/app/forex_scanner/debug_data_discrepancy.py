#!/usr/bin/env python3
"""
Debug the critical data discrepancy at 2025-09-01 02:45:00
"""

def analyze_data_discrepancy():
    print("🚨" * 30)
    print("CRITICAL DATA DISCREPANCY ANALYSIS")
    print("🚨" * 30)
    
    timestamp = "2025-09-01 02:45:00"
    our_close = 1.17120
    tv_close = 1.16967
    difference = our_close - tv_close
    difference_pips = difference * 10000
    
    print(f"\nTimestamp: {timestamp}")
    print(f"Our Close Price:      {our_close:.5f}")
    print(f"TradingView Close:    {tv_close:.5f}")
    print(f"Difference:           {difference:.5f} ({difference_pips:.1f} pips)")
    print(f"Percentage Error:     {(difference/tv_close)*100:.2f}%")
    
    print("\n" + "="*80)
    print("IMPACT ANALYSIS")
    print("="*80)
    
    # With correct TradingView data
    tv_upper_estimate = 1.17074  # Our calculated upper band
    
    print(f"\nCrossover Analysis with CORRECT data:")
    print(f"TradingView Close:    {tv_close:.5f}")
    print(f"Upper Band (est):     {tv_upper_estimate:.5f}")
    print(f"Close vs Upper:       {tv_close:.5f} {'>' if tv_close > tv_upper_estimate else '<'} {tv_upper_estimate:.5f}")
    print(f"Would trigger:        {'YES (GREEN)' if tv_close > tv_upper_estimate else 'NO (stays RED)'}")
    
    print(f"\nCrossover Analysis with OUR WRONG data:")
    print(f"Our Close:            {our_close:.5f}")
    print(f"Upper Band (est):     {tv_upper_estimate:.5f}")
    print(f"Close vs Upper:       {our_close:.5f} {'>' if our_close > tv_upper_estimate else '<'} {tv_upper_estimate:.5f}")
    print(f"Would trigger:        {'YES (GREEN)' if our_close > tv_upper_estimate else 'NO (stays RED)'}")
    
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    
    print("This data discrepancy explains EVERYTHING:")
    print("1. ✅ TradingView ribbon stays RED because close (1.16967) < upper band")
    print("2. ❌ Our system shows GREEN because wrong close (1.17120) > upper band") 
    print("3. 🎯 The crossover detection logic is CORRECT")
    print("4. 📊 The band calculations are MOSTLY correct")
    print("5. 📅 The DATA SOURCE is WRONG")
    
    print("\n" + "="*80)
    print("POTENTIAL DATA ISSUES")
    print("="*80)
    
    print("1. TIMESTAMP MISALIGNMENT:")
    print("   - Our data might be from a different timestamp")
    print("   - 15m bar alignment issues between our DB and TradingView")
    print("   - Timezone conversion errors")
    print("")
    print("2. DATA FEED DIFFERENCES:")
    print("   - Different broker feed vs TradingView feed")
    print("   - IG Markets data vs TradingView's data source")
    print("   - Price aggregation differences")
    print("")
    print("3. BAR CONSTRUCTION:")
    print("   - Different OHLC aggregation methods")
    print("   - Different handling of weekend gaps")
    print("   - Different tick data sources")
    print("")
    print("4. DATA PROCESSING:")
    print("   - Resampling from 5m to 15m errors")  
    print("   - Forward/backward filling issues")
    print("   - Database storage precision loss")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    print("1. VERIFY EXACT TIMESTAMP:")
    print("   - Check if our 2025-09-01 02:45:00 matches TradingView's bar")
    print("   - Look at surrounding bars to identify timing offset")
    print("")
    print("2. CHECK DATA LINEAGE:")
    print("   - Trace back to original 5m data source")
    print("   - Verify resampling to 15m is correct")
    print("   - Check database vs live feed")
    print("")
    print("3. COMPARE OHLC VALUES:")
    print("   - Not just Close, but Open, High, Low as well")
    print("   - This will reveal the full extent of data differences")

if __name__ == "__main__":
    analyze_data_discrepancy()