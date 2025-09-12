#!/usr/bin/env python3
"""
Debug 15m resampling logic using actual 5m data from database
"""

import pandas as pd
from datetime import datetime

def debug_15m_resampling():
    print("="*80)
    print("15M RESAMPLING DEBUG - ACTUAL DATABASE DATA")
    print("="*80)
    
    # Actual 5m data from database around 2025-09-01 02:45:00
    data = [
        {'start_time': '2025-09-01 02:30:00', 'open': 1.17053, 'high': 1.17058, 'low': 1.17035, 'close': 1.170555, 'ltv': 122},
        {'start_time': '2025-09-01 02:35:00', 'open': 1.1705450000000002, 'high': 1.17073, 'low': 1.17046, 'close': 1.17051, 'ltv': 120}, 
        {'start_time': '2025-09-01 02:40:00', 'open': 1.1705, 'high': 1.17059, 'low': 1.17044, 'close': 1.17055, 'ltv': 82},
        {'start_time': '2025-09-01 02:45:00', 'open': 1.17054, 'high': 1.17065, 'low': 1.1704050000000001, 'close': 1.17048, 'ltv': 105},
        {'start_time': '2025-09-01 02:50:00', 'open': 1.17046, 'high': 1.1709450000000001, 'low': 1.17042, 'close': 1.170915, 'ltv': 112},
        {'start_time': '2025-09-01 02:55:00', 'open': 1.170895, 'high': 1.17135, 'low': 1.170885, 'close': 1.1712, 'ltv': 178}
    ]
    
    df = pd.DataFrame(data)
    df['start_time'] = pd.to_datetime(df['start_time'])
    
    print("5M DATA FROM DATABASE:")
    print(df[['start_time', 'open', 'high', 'low', 'close', 'ltv']].to_string(index=False))
    
    print("\n" + "="*80)
    print("15M RESAMPLING ANALYSIS")
    print("="*80)
    
    # According to pandas resampling with label='left', closed='left':
    # 02:30:00 period includes: 02:30, 02:35, 02:40
    # 02:45:00 period includes: 02:45, 02:50, 02:55
    
    period_1 = df[df['start_time'].isin(['2025-09-01 02:30:00', '2025-09-01 02:35:00', '2025-09-01 02:40:00'])]
    period_2 = df[df['start_time'].isin(['2025-09-01 02:45:00', '2025-09-01 02:50:00', '2025-09-01 02:55:00'])]
    
    print("\n15M PERIOD 1 (02:30:00 - 02:44:59):")
    print("5m candles:", period_1['start_time'].dt.strftime('%H:%M').tolist())
    if len(period_1) > 0:
        open_1 = period_1['open'].iloc[0]      # First candle's open
        high_1 = period_1['high'].max()        # Highest high
        low_1 = period_1['low'].min()          # Lowest low  
        close_1 = period_1['close'].iloc[-1]   # Last candle's close
        volume_1 = period_1['ltv'].sum()       # Sum volumes
        print(f"15M OHLC: O={open_1:.5f}, H={high_1:.5f}, L={low_1:.5f}, C={close_1:.5f}, V={volume_1}")
    
    print("\n15M PERIOD 2 (02:45:00 - 02:59:59) - THE DISCREPANCY PERIOD:")
    print("5m candles:", period_2['start_time'].dt.strftime('%H:%M').tolist())
    if len(period_2) > 0:
        open_2 = period_2['open'].iloc[0]      # First candle's open  
        high_2 = period_2['high'].max()        # Highest high
        low_2 = period_2['low'].min()          # Lowest low
        close_2 = period_2['close'].iloc[-1]   # Last candle's close
        volume_2 = period_2['ltv'].sum()       # Sum volumes
        print(f"15M OHLC: O={open_2:.5f}, H={high_2:.5f}, L={low_2:.5f}, C={close_2:.5f}, V={volume_2}")
        
        print(f"\n🔍 CRITICAL FINDINGS:")
        print(f"Our calculated 15m close: {close_2:.5f}")
        print(f"TradingView close:        1.16967")
        print(f"Our logged close:         1.17120")
        print(f"Difference (calc vs TV):  {close_2 - 1.16967:.5f} ({(close_2 - 1.16967)*10000:.1f} pips)")
        print(f"Difference (logged vs calc): {1.17120 - close_2:.5f} ({(1.17120 - close_2)*10000:.1f} pips)")
        
        print(f"\n💡 ANALYSIS:")
        if abs(close_2 - 1.16967) < abs(1.17120 - 1.16967):
            print("✅ Our resampled close is CLOSER to TradingView than our logged value!")
            print("❌ The logged value (1.17120) seems WRONG")
        else:
            print("❌ Our resampled close is still different from TradingView")
            
        if abs(1.17120 - close_2) > 0.0005:
            print("🚨 MAJOR DISCREPANCY: Logged value differs significantly from our resampling!")
            print("   This suggests the logged value came from a different source/timestamp")

    print("\n" + "="*80)
    print("ROOT CAUSE HYPOTHESIS")  
    print("="*80)
    print("1. The logged value 1.17120 does NOT match our 5m database resampling")
    print("2. This suggests either:")
    print("   a) The logged value came from a different timestamp")
    print("   b) The logged value came from live API data (not resampled)")
    print("   c) There's a bug in how we store/retrieve the data")
    print("3. Our resampled value is much closer to TradingView")
    print("4. Need to investigate WHERE the logged 1.17120 value came from")

if __name__ == "__main__":
    debug_15m_resampling()