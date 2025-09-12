#!/usr/bin/env python3
"""
Simple debug script to analyze ZLEMA calculation step by step
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Simulate the exact data from TradingView for 2025-08-28 10:15:00
# We know: Close=1.16646, TradingView ZLEMA Basis=1.16461, Our ZLEMA=1.16480

def debug_zlema_calculation():
    print("="*80)
    print("ZERO LAG EMA CALCULATION DEBUGGING")
    print("="*80)
    
    # Parameters
    length = 70
    band_multiplier = 1.2
    
    print(f"Parameters: Length={length}, Band Multiplier={band_multiplier}")
    
    # Calculate lag exactly as Pine Script
    lag = int((length - 1) // 2)
    print(f"Lag calculation: ({length} - 1) // 2 = {lag}")
    
    # Pine Script EMA alpha
    alpha = 2.0 / (length + 1.0)
    print(f"EMA Alpha: 2 / ({length} + 1) = {alpha:.8f}")
    
    print("\n" + "="*80)
    print("COMPARISON WITH TRADINGVIEW VALUES")
    print("="*80)
    
    # Known TradingView values
    tv_close = 1.16646
    tv_zlema_basis = 1.16461
    our_zlema = 1.16480  # Still the same - ZLEMA unchanged
    tv_upper = 1.16572
    tv_lower = 1.16424
    our_upper = 1.16576   # IMPROVED: was 1.16589, now 1.16576
    our_lower = 1.16384   # IMPROVED: was 1.16372, now 1.16384
    
    print(f"\nAt timestamp 2025-08-28 10:15:00:")
    print(f"Close Price:        {tv_close:.5f}")
    print(f"")
    print(f"ZLEMA (Zero Lag Basis):")
    print(f"  TradingView:      {tv_zlema_basis:.5f}")
    print(f"  Our calculation:  {our_zlema:.5f}")
    print(f"  Difference:       {our_zlema - tv_zlema_basis:.5f} ({(our_zlema - tv_zlema_basis)*10000:.1f} pips)")
    print(f"")
    print(f"Upper Band:")
    print(f"  TradingView:      {tv_upper:.5f}")
    print(f"  Our calculation:  {our_upper:.5f}")
    print(f"  Difference:       {our_upper - tv_upper:.5f} ({(our_upper - tv_upper)*10000:.1f} pips)")
    print(f"")
    print(f"Lower Band:")
    print(f"  TradingView:      {tv_lower:.5f}")
    print(f"  Our calculation:  {our_lower:.5f}")
    print(f"  Difference:       {our_lower - tv_lower:.5f} ({(our_lower - tv_lower)*10000:.1f} pips)")
    
    # Band width analysis
    tv_width = tv_upper - tv_lower
    our_width = our_upper - our_lower
    print(f"")
    print(f"Band Width:")
    print(f"  TradingView:      {tv_width:.5f} ({tv_width*10000:.1f} pips)")
    print(f"  Our calculation:  {our_width:.5f} ({our_width*10000:.1f} pips)")
    print(f"  Difference:       {our_width - tv_width:.5f} ({(our_width - tv_width)*10000:.1f} pips)")
    
    # Volatility analysis
    our_volatility = (our_upper - our_zlema)  # Should equal (our_zlema - our_lower)
    tv_volatility_upper = tv_upper - tv_zlema_basis
    tv_volatility_lower = tv_zlema_basis - tv_lower
    
    print(f"")
    print(f"Volatility Analysis:")
    print(f"  Our volatility:           {our_volatility:.5f}")
    print(f"  TV upper volatility:      {tv_volatility_upper:.5f}")
    print(f"  TV lower volatility:      {tv_volatility_lower:.5f}")
    print(f"  TV average volatility:    {(tv_volatility_upper + tv_volatility_lower)/2:.5f}")
    
    print("\n" + "="*80)
    print("CROSSOVER ANALYSIS")
    print("="*80)
    
    # At 2025-08-28 10:15:00, TradingView shows RED ribbon
    # But our close (1.16646) > our upper (1.16589) would trigger GREEN
    # While close (1.16646) < TV upper (1.16572) keeps it RED
    
    print(f"Close vs Upper Band:")
    print(f"  Close:            {tv_close:.5f}")
    print(f"  Our Upper:        {our_upper:.5f} -> {'ABOVE' if tv_close > our_upper else 'BELOW'} (would be GREEN)")
    print(f"  TV Upper:         {tv_upper:.5f} -> {'ABOVE' if tv_close > tv_upper else 'BELOW'} (stays RED)")
    print(f"")
    print(f"This explains why:")
    print(f"  - Our calculation shows GREEN ribbon (close > upper band)")
    print(f"  - TradingView shows RED ribbon (close < upper band)")
    
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    
    print("The issue appears to be a combination of:")
    print("1. ZLEMA calculation difference: +19 pips higher than TradingView")
    print("2. Upper band calculation: +17 pips higher than TradingView")
    print("3. Lower band calculation: -52 pips lower than TradingView")
    print("")
    print("This suggests:")
    print("- Our ZLEMA is slightly high")
    print("- Our volatility calculation might be wrong")
    print("- The combination creates narrower bands that trigger false crossovers")
    
    print("\n" + "="*80)
    print("POTENTIAL FIXES TO INVESTIGATE")
    print("="*80)
    
    print("1. ZLEMA Calculation:")
    print("   - Check EMA initialization method")
    print("   - Verify lag calculation: floor((70-1)/2) = 34")
    print("   - Ensure exact Pine Script alpha: 2/(70+1) = 0.02816901")
    print("")
    print("2. Volatility Calculation:")
    print("   - Verify ATR calculation matches Pine Script ta.atr(70)")
    print("   - Check ta.highest(atr, 70*3) = ta.highest(atr, 210)")
    print("   - Ensure band multiplier 1.2 is applied correctly")
    print("")
    print("3. Data Alignment:")
    print("   - Verify we're using exact same OHLC values as TradingView")
    print("   - Check timestamp alignment (15m bars)")

if __name__ == "__main__":
    debug_zlema_calculation()