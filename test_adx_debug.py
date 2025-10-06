#!/usr/bin/env python3
"""Debug ADX crossover detection to see which validation is failing"""

import pandas as pd
import numpy as np

# Simulate the ADX crossover detection logic
def test_adx_crossover_debug():
    """Test ADX crossover with detailed logging"""

    # Configuration values
    adx_crossover_threshold = 25
    adx_min_histogram = 0.0001
    adx_crossover_lookback = 3
    adx_require_expansion = True

    # Create test data with ADX crossing above 25 at bar 40
    data = {
        'adx': [20.0] * 35 + [21.0, 22.0, 23.0, 24.5, 25.5, 26.0, 26.5],  # Crosses at bar 39
        'macd_histogram': [0.0001] * 35 + [0.00012, 0.00015, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026],  # Rising green
    }

    df = pd.DataFrame(data)

    # Add previous values
    df['adx_prev'] = df['adx'].shift(1)
    df['histogram_prev'] = df['macd_histogram'].shift(1)

    # Detect crossover
    df['adx_cross_up'] = (df['adx'] > adx_crossover_threshold) & (df['adx_prev'] <= adx_crossover_threshold)

    # Initialize signals
    df['bull_adx_crossover'] = False
    df['bear_adx_crossover'] = False

    print("=" * 80)
    print("ADX CROSSOVER DETECTION DEBUG")
    print("=" * 80)

    # Find crossover points
    crossover_indices = df[df['adx_cross_up']].index.tolist()
    print(f"\n✓ ADX crossover detected at bars: {crossover_indices}")

    # Process each crossover
    for idx in crossover_indices:
        print(f"\n{'=' * 60}")
        print(f"Processing bar {idx}")
        print(f"{'=' * 60}")

        histogram = df.loc[idx, 'macd_histogram']
        print(f"MACD histogram: {histogram:.6f}")

        # Check 1: Minimum histogram magnitude
        print(f"\n[CHECK 1] Minimum histogram magnitude")
        print(f"  abs(histogram) = {abs(histogram):.6f}")
        print(f"  min_histogram = {adx_min_histogram:.6f}")
        if abs(histogram) < adx_min_histogram:
            print(f"  ❌ FAILED: Histogram too small")
            continue
        else:
            print(f"  ✓ PASSED")

        # Check 2: ADX rising lookback
        print(f"\n[CHECK 2] ADX rising lookback ({adx_crossover_lookback} bars)")
        if adx_crossover_lookback > 0:
            lookback_start = max(0, idx - adx_crossover_lookback)
            lookback_adx = df.iloc[lookback_start:idx + 1]['adx']

            print(f"  Lookback range: bars {lookback_start} to {idx}")
            print(f"  ADX values: {lookback_adx.tolist()}")

            adx_diffs = lookback_adx.diff().dropna()
            print(f"  ADX differences: {adx_diffs.tolist()}")

            all_rising = all(adx_diffs > 0)
            print(f"  All differences > 0? {all_rising}")

            if not all_rising:
                print(f"  ❌ FAILED: ADX not consistently rising")
                continue
            else:
                print(f"  ✓ PASSED")

        # Check 3: Histogram expansion
        print(f"\n[CHECK 3] Histogram expansion")
        if adx_require_expansion:
            histogram_prev = df.loc[idx, 'histogram_prev']
            print(f"  Current histogram: {histogram:.6f}")
            print(f"  Previous histogram: {histogram_prev:.6f}")
            print(f"  abs(current) = {abs(histogram):.6f}")
            print(f"  abs(previous) = {abs(histogram_prev):.6f}")

            is_expanding = abs(histogram) > abs(histogram_prev)
            print(f"  Is expanding? {is_expanding}")

            if not is_expanding:
                print(f"  ❌ FAILED: Histogram not expanding")
                continue
            else:
                print(f"  ✓ PASSED")

        # Check 4: Histogram direction
        print(f"\n[CHECK 4] Histogram direction")
        print(f"  Histogram: {histogram:.6f}")
        if histogram > 0:
            print(f"  ✓ BULL signal (green histogram)")
            df.loc[idx, 'bull_adx_crossover'] = True
        elif histogram < 0:
            print(f"  ✓ BEAR signal (red histogram)")
            df.loc[idx, 'bear_adx_crossover'] = True
        else:
            print(f"  ❌ Zero histogram")

    # Show results
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")

    for idx in crossover_indices:
        print(f"\nBar {idx}:")
        print(f"  ADX: {df.loc[idx, 'adx']:.2f}")
        print(f"  ADX prev: {df.loc[idx, 'adx_prev']:.2f}")
        print(f"  MACD histogram: {df.loc[idx, 'macd_histogram']:.6f}")
        print(f"  ADX cross up: {df.loc[idx, 'adx_cross_up']}")
        print(f"  BULL ADX crossover: {df.loc[idx, 'bull_adx_crossover']}")
        print(f"  BEAR ADX crossover: {df.loc[idx, 'bear_adx_crossover']}")

    # Count signals
    bull_count = df['bull_adx_crossover'].sum()
    bear_count = df['bear_adx_crossover'].sum()

    print(f"\n{'=' * 80}")
    print(f"Total BULL ADX crossovers: {bull_count}")
    print(f"Total BEAR ADX crossovers: {bear_count}")
    print(f"{'=' * 80}\n")

    if bull_count > 0 or bear_count > 0:
        print("✓ SUCCESS: ADX crossover detected!")
        return True
    else:
        print("❌ FAILURE: No ADX crossover detected")
        return False


if __name__ == "__main__":
    success = test_adx_crossover_debug()
    exit(0 if success else 1)
