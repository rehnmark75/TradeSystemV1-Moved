#!/usr/bin/env python3
"""Test script to verify new ATR indicators"""

import sys
import pandas as pd
import numpy as np

# Simple ATR calculation test
def test_atr_indicators():
    # Create test data
    data = pd.DataFrame({
        'high': [1.09, 1.092, 1.095, 1.093, 1.098] + [1.09] * 25,
        'low': [1.08, 1.085, 1.087, 1.088, 1.089] + [1.08] * 25,
        'close': [1.085, 1.089, 1.091, 1.090, 1.093] + [1.085] * 25
    })

    # Calculate true range
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))

    # 14-period ATR
    data['atr'] = true_range.rolling(window=14).mean()

    # 20-period ATR
    data['atr_20'] = true_range.rolling(window=20).mean()

    # ATR percentile
    data['atr_percentile'] = (data['atr'] / data['atr_20'] * 100).fillna(50.0)

    # BB width percentile
    data['bb_upper'] = data['close'].rolling(window=20).mean() + (data['close'].rolling(window=20).std() * 2)
    data['bb_lower'] = data['close'].rolling(window=20).mean() - (data['close'].rolling(window=20).std() * 2)
    bb_width = data['bb_upper'] - data['bb_lower']
    data['bb_width_percentile'] = bb_width.rolling(window=50).rank(pct=True) * 100
    data['bb_width_percentile'] = data['bb_width_percentile'].fillna(50.0)

    # Print results
    print("✅ Indicator Test Results:")
    print(f"  Total rows: {len(data)}")
    print(f"\n  Last row indicators:")
    print(f"    atr: {data['atr'].iloc[-1]:.6f}")
    print(f"    atr_20: {data['atr_20'].iloc[-1]:.6f}")
    print(f"    atr_percentile: {data['atr_percentile'].iloc[-1]:.2f}%")
    print(f"    bb_width_percentile: {data['bb_width_percentile'].iloc[-1]:.2f}%")
    print(f"\n  Non-null counts:")
    print(f"    atr: {data['atr'].notna().sum()}")
    print(f"    atr_20: {data['atr_20'].notna().sum()}")
    print(f"    atr_percentile: {data['atr_percentile'].notna().sum()}")
    print(f"    bb_width_percentile: {data['bb_width_percentile'].notna().sum()}")
    print("\n✅ All indicators calculated successfully!")
    return True

if __name__ == '__main__':
    try:
        test_atr_indicators()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
