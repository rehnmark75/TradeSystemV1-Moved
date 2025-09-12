#!/usr/bin/env python3
"""
Simple MACD manual calculation verification
Based on the raw data we got from the backtest output
"""
import pandas as pd
import numpy as np

def calculate_ema(series, period):
    """Calculate EMA manually"""
    return series.ewm(span=period, adjust=False).mean()

def verify_macd_calculation():
    """
    Verify MACD calculation using the price data from backtest output
    Target signal: 2025-09-02 22:45:00 UTC, USDJPY BUY at 148.50000
    """
    
    # Price data extracted from backtest output (last 10 bars around signal)
    # Format: [timestamp, close_price]
    price_data = [
        ('2025-09-02 20:45:00', 148.36800),
        ('2025-09-02 21:00:00', 148.30600),  
        ('2025-09-02 21:15:00', 148.32900),
        ('2025-09-02 21:30:00', 148.34800),
        ('2025-09-02 21:45:00', 148.35300),
        ('2025-09-02 22:00:00', 148.35700),
        ('2025-09-02 22:15:00', 148.35000),
        ('2025-09-02 22:30:00', 148.36700),
        ('2025-09-02 22:45:00', 148.50000),  # <- TARGET SIGNAL
        ('2025-09-02 23:00:00', 148.52300),
        ('2025-09-02 23:15:00', 148.52200),
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(price_data, columns=['timestamp', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print("🔍 MACD Calculation Verification")
    print("=" * 50)
    print(f"Target Signal: 2025-09-02 22:45:00 UTC")
    print(f"Signal Price: 148.50000")
    
    # Calculate MACD components manually
    # Standard MACD: 12-period EMA, 26-period EMA, 9-period signal
    fast_ema = calculate_ema(df['close'], 12)
    slow_ema = calculate_ema(df['close'], 26)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, 9)
    histogram = macd_line - signal_line
    
    # Add to DataFrame
    df['fast_ema_12'] = fast_ema
    df['slow_ema_26'] = slow_ema
    df['macd_line'] = macd_line
    df['signal_line'] = signal_line
    df['histogram'] = histogram
    
    print(f"\n📊 MACD Values (last 5 periods):")
    print("-" * 80)
    print(f"{'TIME':<18} {'CLOSE':<10} {'FAST_EMA':<10} {'SLOW_EMA':<10} {'MACD':<12} {'SIGNAL':<12} {'HIST':<12}")
    print("-" * 80)
    
    # Show last 5 rows
    for i in range(len(df) - 5, len(df)):
        if i < 0:
            continue
        row = df.iloc[i]
        timestamp = row.name.strftime('%m-%d %H:%M')
        close = row['close']
        fast_ema_val = row['fast_ema_12']
        slow_ema_val = row['slow_ema_26']
        macd_val = row['macd_line']
        signal_val = row['signal_line'] 
        hist_val = row['histogram']
        
        marker = ">>>" if timestamp == "09-02 22:45" else "   "
        print(f"{marker} {timestamp:<18} {close:<10.5f} {fast_ema_val:<10.6f} {slow_ema_val:<10.6f} {macd_val:<12.6f} {signal_val:<12.6f} {hist_val:<12.6f}")
    
    # Get target signal values
    target_idx = df.index.get_loc(pd.Timestamp('2025-09-02 22:45:00'))
    target_row = df.iloc[target_idx]
    prev_row = df.iloc[target_idx - 1]
    
    target_macd = target_row['macd_line']
    target_signal = target_row['signal_line']
    target_histogram = target_row['histogram']
    prev_histogram = prev_row['histogram']
    
    # Compare with backtest reported values
    reported_macd = 0.013474
    reported_signal = 0.004546
    reported_histogram = 0.008928
    reported_prev_histogram = -0.000318
    
    print(f"\n🎯 Signal Analysis:")
    print("=" * 50)
    print(f"Manual Calculation:")
    print(f"  MACD Line: {target_macd:.6f}")
    print(f"  Signal Line: {target_signal:.6f}")
    print(f"  Histogram: {target_histogram:.6f}")
    print(f"  Previous Histogram: {prev_histogram:.6f}")
    
    print(f"\nBacktest Reported:")
    print(f"  MACD Line: {reported_macd:.6f}")
    print(f"  Signal Line: {reported_signal:.6f}") 
    print(f"  Histogram: {reported_histogram:.6f}")
    print(f"  Previous Histogram: {reported_prev_histogram:.6f}")
    
    print(f"\nDifferences:")
    print(f"  MACD Line: {abs(target_macd - reported_macd):.6f}")
    print(f"  Signal Line: {abs(target_signal - reported_signal):.6f}")
    print(f"  Histogram: {abs(target_histogram - reported_histogram):.6f}")
    print(f"  Previous Histogram: {abs(prev_histogram - reported_prev_histogram):.6f}")
    
    # Check crossover logic
    crossover_detected = target_histogram > 0 and prev_histogram <= 0
    histogram_strength = abs(target_histogram)
    jpy_threshold = 0.003
    
    print(f"\n🚦 Signal Logic:")
    print("=" * 30)
    print(f"Crossover Detected: {'✅' if crossover_detected else '❌'}")
    print(f"Histogram Strength: {histogram_strength:.6f}")
    print(f"JPY Threshold: {jpy_threshold}")
    print(f"Meets Threshold: {'✅' if histogram_strength >= jpy_threshold else '❌'}")
    print(f"Signal Type: {'BULL' if crossover_detected else 'None'}")
    
    # Verdict
    calculation_accurate = (
        abs(target_macd - reported_macd) < 0.001 and
        abs(target_signal - reported_signal) < 0.001 and
        abs(target_histogram - reported_histogram) < 0.001
    )
    
    print(f"\n🎯 Verification Result:")
    print("=" * 30)
    print(f"Calculations Accurate: {'✅' if calculation_accurate else '❌'}")
    print(f"Signal Logic Correct: {'✅' if crossover_detected and histogram_strength >= jpy_threshold else '❌'}")
    
    return calculation_accurate

if __name__ == "__main__":
    verify_macd_calculation()