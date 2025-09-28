#!/usr/bin/env python3
"""
Quick test to see if MACD strategy gets backtest_mode=True in backtests
"""
import sys
sys.path.insert(0, '/app/forex_scanner')

from core.strategies.macd_strategy import MACDStrategy

# Test 1: Create MACD strategy with backtest_mode=True
print("🧪 Test 1: Creating MACD strategy with backtest_mode=True")
strategy = MACDStrategy(backtest_mode=True, epic='CS.D.EURUSD.CEEM.IP', timeframe='15m')
print(f"   Strategy backtest_mode: {strategy.backtest_mode}")

# Test 2: Check if crossover detection uses backtest filtering
print("\n🧪 Test 2: Creating synthetic test data")
import pandas as pd
import numpy as np

# Create minimal test data with 70 bars
dates = pd.date_range('2025-09-26', periods=70, freq='15min')
test_data = pd.DataFrame({
    'start_time': dates,
    'open': np.random.normal(1.1000, 0.001, 70),
    'high': np.random.normal(1.1005, 0.001, 70),
    'low': np.random.normal(0.9995, 0.001, 70),
    'close': np.random.normal(1.1000, 0.001, 70),
    'ltv': np.random.randint(1000, 5000, 70)
})

print(f"   Created test data with {len(test_data)} bars")

# Test signal detection
print("\n🧪 Test 3: Testing signal detection with backtest mode")
result = strategy.detect_signal(test_data, 'CS.D.EURUSD.CEEM.IP', timeframe='15m')
print(f"   Signal detected: {result is not None}")

print("✅ Test complete")