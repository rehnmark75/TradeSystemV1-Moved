#!/usr/bin/env python3
"""
Quick MACD calculation verification script
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys
import os

# Add the forex_scanner directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.data_fetcher import DataFetcher
from core.strategies.helpers.macd_indicator_calculator import MACDIndicatorCalculator
try:
    import config
except ImportError:
    from forex_scanner import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_manual_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """Manual MACD calculation for verification"""
    # Calculate EMAs manually
    fast_ema = df['close'].ewm(span=fast_period).mean()
    slow_ema = df['close'].ewm(span=slow_period).mean()
    
    # MACD line = Fast EMA - Slow EMA
    macd_line = fast_ema - slow_ema
    
    # Signal line = EMA of MACD line
    signal_line = macd_line.ewm(span=signal_period).mean()
    
    # Histogram = MACD line - Signal line
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def verify_macd_signal():
    """Verify specific MACD signal calculation"""
    
    # Initialize data fetcher
    fetcher = DataFetcher()
    calculator = MACDIndicatorCalculator(logger=logger)
    
    epic = "CS.D.USDJPY.MINI.IP"
    timeframe = "15m"
    
    print(f"🔍 Verifying MACD calculations for {epic}")
    print("=" * 60)
    
    try:
        # Get data around the signal time
        target_time = datetime(2025, 9, 2, 22, 45, 0, tzinfo=timezone.utc)
        
        # Fetch data with sufficient lookback for MACD
        df = fetcher.get_enhanced_data(
            epic=epic,
            timeframe=timeframe,
            days=7,
            indicators_config={
                'ema_periods': [12, 26, 200],
                'macd': True,
                'ema': True
            }
        )
        
        if df is None or df.empty:
            print("❌ No data retrieved")
            return
            
        print(f"✅ Retrieved {len(df)} data points")
        
        # Calculate MACD manually for comparison
        manual_macd, manual_signal, manual_histogram = calculate_manual_macd(df)
        
        # Find the target timestamp
        df['timestamp'] = pd.to_datetime(df.index)
        target_idx = None
        
        # Find closest timestamp
        time_diffs = abs(df['timestamp'] - target_time)
        target_idx = time_diffs.idxmin()
        
        print(f"📅 Target time: {target_time}")
        print(f"📅 Found time: {df.loc[target_idx, 'timestamp']}")
        print(f"📊 Data index: {target_idx}")
        
        # Get values around the signal
        context_range = 3
        start_idx = max(0, target_idx - context_range)
        end_idx = min(len(df), target_idx + context_range + 1)
        
        print(f"\n🔍 MACD Values Around Signal Time:")
        print("=" * 80)
        print(f"{'IDX':<4} {'TIMESTAMP':<20} {'CLOSE':<10} {'MACD_LINE':<12} {'SIGNAL':<12} {'HISTOGRAM':<12}")
        print("-" * 80)
        
        for i in range(start_idx, end_idx):
            if i >= len(df):
                break
                
            row = df.iloc[i]
            timestamp = row['timestamp'].strftime('%m-%d %H:%M')
            close = row['close']
            
            # Get system calculated values
            sys_macd = row.get('macd_line', 0)
            sys_signal = row.get('macd_signal', 0) 
            sys_histogram = row.get('macd_histogram', 0)
            
            # Get manual calculated values
            manual_macd_val = manual_macd.iloc[i] if i < len(manual_macd) else 0
            manual_signal_val = manual_signal.iloc[i] if i < len(manual_signal) else 0
            manual_histogram_val = manual_histogram.iloc[i] if i < len(manual_histogram) else 0
            
            marker = ">>>" if i == target_idx else "   "
            
            print(f"{marker} {i:<4} {timestamp:<20} {close:<10.5f} {sys_macd:<12.6f} {sys_signal:<12.6f} {sys_histogram:<12.6f}")
            
            # Show manual comparison for target row
            if i == target_idx:
                print(f"    Manual:                             {manual_macd_val:<12.6f} {manual_signal_val:<12.6f} {manual_histogram_val:<12.6f}")
                print(f"    Difference:                         {abs(sys_macd - manual_macd_val):<12.6f} {abs(sys_signal - manual_signal_val):<12.6f} {abs(sys_histogram - manual_histogram_val):<12.6f}")
        
        # Check for crossover at target
        target_row = df.iloc[target_idx]
        prev_row = df.iloc[target_idx - 1] if target_idx > 0 else target_row
        
        current_histogram = target_row.get('macd_histogram', 0)
        prev_histogram = prev_row.get('macd_histogram', 0)
        
        print(f"\n🎯 Signal Analysis at {target_time}:")
        print("=" * 50)
        print(f"Current Histogram: {current_histogram:.6f}")
        print(f"Previous Histogram: {prev_histogram:.6f}")
        print(f"Histogram Change: {current_histogram - prev_histogram:.6f}")
        print(f"Crossover Type: {'BULL' if current_histogram > 0 and prev_histogram <= 0 else 'BEAR' if current_histogram < 0 and prev_histogram >= 0 else 'None'}")
        
        # Check strength threshold for JPY pairs
        jpy_threshold = 0.003
        print(f"JPY Threshold: {jpy_threshold}")
        print(f"Meets Threshold: {'✅' if abs(current_histogram) >= jpy_threshold else '❌'}")
        
        # EMA 200 check
        current_price = target_row['close']
        ema_200 = target_row.get('ema_200', 0)
        print(f"Current Price: {current_price:.5f}")
        print(f"EMA 200: {ema_200:.5f}")
        print(f"Price vs EMA200: {'Above' if current_price > ema_200 else 'Below'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_macd_signal()