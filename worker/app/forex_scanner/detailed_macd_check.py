#!/usr/bin/env python3
"""
Detailed MACD verification using larger dataset
"""
import pandas as pd
import numpy as np
import sys
import os

# Add the forex_scanner directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.database import DatabaseManager
try:
    import config
except ImportError:
    from forex_scanner import config

def get_raw_data():
    """Get raw 15m data from database for USDJPY around the signal time"""
    
    db_manager = DatabaseManager(config.DATABASE_URL)
    
    # Query for USDJPY 5m data around the signal (to synthesize 15m like the backtest does)
    query = """
    SELECT start_time, open, high, low, close, volume
    FROM ig_candles 
    WHERE epic = 'CS.D.USDJPY.MINI.IP' 
    AND timeframe = 5
    AND start_time >= '2025-09-01 00:00:00'
    AND start_time <= '2025-09-03 12:00:00'
    ORDER BY start_time
    """
    
    try:
        df = pd.read_sql(query, db_manager.get_engine())
        return df
    except Exception as e:
        print(f"Database error: {e}")
        return None

def resample_5m_to_15m(df_5m):
    """Resample 5m data to 15m to match backtest data synthesis"""
    
    # Set start_time as index for resampling
    df_5m['start_time'] = pd.to_datetime(df_5m['start_time'])
    df_5m.set_index('start_time', inplace=True)
    df_5m.sort_index(inplace=True)
    
    # Resample to 15m intervals
    df_15m = df_5m.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_15m

def calculate_macd_with_more_data():
    """Calculate MACD with sufficient historical data for proper EMA calculation"""
    
    print("🔍 Getting raw 5m data from database...")
    df_5m = get_raw_data()
    
    if df_5m is None or df_5m.empty:
        print("❌ No data retrieved")
        return
        
    print(f"✅ Retrieved {len(df_5m)} 5m data points")
    
    # Synthesize 15m data (same as backtest)
    print("🔄 Synthesizing 15m data from 5m data...")
    df = resample_5m_to_15m(df_5m)
    print(f"✅ Synthesized {len(df)} 15m data points")
    
    # Calculate MACD with standard parameters (12, 26, 9)
    # Use the same method as pandas-ta or the strategy
    close_prices = df['close']
    
    # Calculate EMAs
    fast_ema = close_prices.ewm(span=12, adjust=False).mean()
    slow_ema = close_prices.ewm(span=26, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Add to dataframe
    df['ema_12'] = fast_ema
    df['ema_26'] = slow_ema
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    df['macd_histogram_prev'] = histogram.shift(1)
    
    # Find the target signal time
    target_time = '2025-09-02 22:45:00'
    target_time_parsed = pd.Timestamp(target_time, tz='UTC').tz_localize(None)
    
    # Find closest timestamp
    if target_time_parsed in df.index:
        target_idx = df.index.get_loc(target_time_parsed)
    else:
        # Find closest
        time_diffs = abs(df.index - target_time_parsed)
        target_idx = time_diffs.argmin()
        
    print(f"🎯 Target time: {target_time}")
    print(f"🎯 Found time: {df.index[target_idx]}")
    
    # Show context around signal
    context_range = 5
    start_idx = max(0, target_idx - context_range)
    end_idx = min(len(df), target_idx + context_range + 1)
    
    print(f"\n📊 MACD Values Around Target Signal:")
    print("=" * 100)
    print(f"{'IDX':<4} {'TIMESTAMP':<20} {'CLOSE':<10} {'MACD_LINE':<12} {'SIGNAL':<12} {'HISTOGRAM':<12} {'PREV_HIST':<12}")
    print("-" * 100)
    
    for i in range(start_idx, end_idx):
        if i >= len(df):
            break
            
        row = df.iloc[i]
        timestamp = row.name.strftime('%m-%d %H:%M')
        close = row['close']
        macd_val = row['macd_line']
        signal_val = row['macd_signal']
        hist_val = row['macd_histogram']
        prev_hist = row['macd_histogram_prev'] if pd.notna(row['macd_histogram_prev']) else 0
        
        marker = ">>>" if i == target_idx else "   "
        print(f"{marker} {i:<4} {timestamp:<20} {close:<10.5f} {macd_val:<12.6f} {signal_val:<12.6f} {hist_val:<12.6f} {prev_hist:<12.6f}")
    
    # Get target values
    target_row = df.iloc[target_idx]
    
    # Compare with backtest reported values
    reported_macd = 0.013474
    reported_signal = 0.004546
    reported_histogram = 0.008928
    reported_prev_histogram = -0.000318
    
    calculated_macd = target_row['macd_line']
    calculated_signal = target_row['macd_signal']
    calculated_histogram = target_row['macd_histogram']
    calculated_prev_histogram = target_row['macd_histogram_prev']
    
    print(f"\n🔍 Detailed Comparison:")
    print("=" * 60)
    print(f"{'METRIC':<20} {'CALCULATED':<15} {'REPORTED':<15} {'DIFFERENCE':<15}")
    print("-" * 60)
    print(f"{'MACD Line':<20} {calculated_macd:<15.6f} {reported_macd:<15.6f} {abs(calculated_macd - reported_macd):<15.6f}")
    print(f"{'Signal Line':<20} {calculated_signal:<15.6f} {reported_signal:<15.6f} {abs(calculated_signal - reported_signal):<15.6f}")
    print(f"{'Histogram':<20} {calculated_histogram:<15.6f} {reported_histogram:<15.6f} {abs(calculated_histogram - reported_histogram):<15.6f}")
    print(f"{'Prev Histogram':<20} {calculated_prev_histogram:<15.6f} {reported_prev_histogram:<15.6f} {abs(calculated_prev_histogram - reported_prev_histogram):<15.6f}")
    
    # Check crossover logic
    is_crossover = calculated_histogram > 0 and calculated_prev_histogram <= 0
    meets_threshold = abs(calculated_histogram) >= 0.003  # JPY threshold
    
    print(f"\n🚦 Signal Validation:")
    print("=" * 40)
    print(f"Current Histogram: {calculated_histogram:.6f}")
    print(f"Previous Histogram: {calculated_prev_histogram:.6f}")
    print(f"Crossover (0 → positive): {'✅' if is_crossover else '❌'}")
    print(f"Meets JPY threshold (≥0.003): {'✅' if meets_threshold else '❌'}")
    print(f"Valid BULL signal: {'✅' if is_crossover and meets_threshold else '❌'}")
    
    # Check EMA 200 (if available)
    current_price = target_row['close']
    print(f"\nPrice Analysis:")
    print(f"Current Price: {current_price:.5f}")
    
    return calculated_macd, calculated_signal, calculated_histogram

if __name__ == "__main__":
    calculate_macd_with_more_data()