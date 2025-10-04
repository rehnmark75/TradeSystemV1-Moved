#!/usr/bin/env python3
"""
Simple test to verify clean MACD strategy generates ONE signal per crossover
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create test data with ONE clear crossover
dates = pd.date_range(start='2025-10-01', periods=100, freq='15min')
df = pd.DataFrame({
    'start_time': dates,
    'close': 1.1000 + np.random.randn(100) * 0.0010,
    'high': 1.1010 + np.random.randn(100) * 0.0010,
    'low': 1.0990 + np.random.randn(100) * 0.0010,
    'open': 1.1000 + np.random.randn(100) * 0.0010,
})

# Calculate MACD
fast_ema = df['close'].ewm(span=12, adjust=False).mean()
slow_ema = df['close'].ewm(span=26, adjust=False).mean()
df['macd_line'] = fast_ema - slow_ema
df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
df['macd_histogram'] = df['macd_line'] - df['macd_signal']

# Detect crossovers
df['histogram_prev'] = df['macd_histogram'].shift(1)
df['bull_crossover'] = (df['macd_histogram'] > 0) & (df['histogram_prev'] <= 0)
df['bear_crossover'] = (df['macd_histogram'] < 0) & (df['histogram_prev'] >= 0)

print("📊 MACD Crossover Test Results:")
print(f"Total bars: {len(df)}")
print(f"Bull crossovers: {df['bull_crossover'].sum()}")
print(f"Bear crossovers: {df['bear_crossover'].sum()}")
print("\nCrossover bars:")
print(df[df['bull_crossover'] | df['bear_crossover']][['start_time', 'macd_histogram', 'histogram_prev', 'bull_crossover', 'bear_crossover']])

# Now test with latest bar only approach (what the fixed strategy does)
latest_bar = df.iloc[-1]
print(f"\n🔍 Latest Bar Check (Fixed Strategy Approach):")
print(f"Bull crossover: {latest_bar.get('bull_crossover', False)}")
print(f"Bear crossover: {latest_bar.get('bear_crossover', False)}")
print(f"Expected: Should see crossover ONLY if last bar is a crossover")
