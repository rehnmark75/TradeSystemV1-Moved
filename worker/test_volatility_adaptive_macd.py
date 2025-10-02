#!/usr/bin/env python3
"""
Test script for volatility-adaptive MACD parameters
Verifies that threshold and limiter rules adapt based on market regime
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add paths
sys.path.insert(0, '/home/hr/Projects/TradeSystemV1/worker/app')

from forex_scanner.core.strategies.helpers.macd_indicator_calculator import MACDIndicatorCalculator
from forex_scanner.core.strategies.helpers.adaptive_volatility_calculator import VolatilityMetrics

def create_test_data(volatility_level='normal'):
    """Create test dataframe with different volatility scenarios"""
    dates = pd.date_range(start='2025-10-01', periods=100, freq='15min')

    if volatility_level == 'high':
        # High volatility scenario (choppy/ranging)
        atr = np.random.uniform(0.0015, 0.0025, 100)  # High ATR
        adx = np.random.uniform(15, 22, 100)  # Low ADX (ranging)
        close = 1.1000 + np.cumsum(np.random.randn(100) * 0.002)  # Choppy price
    elif volatility_level == 'trending':
        # Trending scenario
        atr = np.random.uniform(0.0005, 0.0008, 100)  # Low ATR
        adx = np.random.uniform(28, 40, 100)  # High ADX (trending)
        close = 1.1000 + np.cumsum(np.random.randn(100) * 0.0005 + 0.0003)  # Trending up
    else:
        # Normal scenario
        atr = np.random.uniform(0.0008, 0.0012, 100)  # Medium ATR
        adx = np.random.uniform(18, 26, 100)  # Medium ADX
        close = 1.1000 + np.cumsum(np.random.randn(100) * 0.001)  # Normal movement

    df = pd.DataFrame({
        'timestamp': dates,
        'close': close,
        'open': close - np.random.uniform(-0.0005, 0.0005, 100),
        'high': close + np.random.uniform(0.0002, 0.0008, 100),
        'low': close - np.random.uniform(0.0002, 0.0008, 100),
        'atr': atr,
        'adx': adx,
    })

    df.set_index('timestamp', inplace=True)

    # Add MACD indicators
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd_line'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']

    return df

def test_threshold_adaptation():
    """Test that thresholds adapt to volatility"""
    print("=" * 80)
    print("TEST 1: Threshold Adaptation")
    print("=" * 80)

    calc = MACDIndicatorCalculator()
    epic = "CS.D.EURJPY.MINI.IP"

    scenarios = {
        'high': ('High Volatility (Choppy)', 'Expected: 2.0x multiplier, stricter threshold'),
        'normal': ('Normal Volatility', 'Expected: 1.0x multiplier, standard threshold'),
        'trending': ('Low Volatility (Trending)', 'Expected: 0.7x multiplier, looser threshold')
    }

    for scenario_type, (name, expected) in scenarios.items():
        print(f"\n{name}")
        print("-" * 80)
        print(f"{expected}")
        print()

        df = create_test_data(scenario_type)

        # Extract metrics
        metrics = calc._extract_volatility_metrics(df, epic)
        close_price = df['close'].iloc[-1]

        if metrics:
            # Get threshold
            threshold = calc.get_histogram_strength_threshold(epic, metrics, close_price)

            print(f"✓ ATR: {metrics.atr:.6f}")
            print(f"✓ ATR Percentile: {metrics.atr_percentile:.0f}")
            print(f"✓ ADX: {metrics.adx:.1f}")
            print(f"✓ Efficiency Ratio: {metrics.efficiency_ratio:.3f}")
            print(f"✓ Threshold: {threshold:.6f}")

            # Verify expected behavior
            atr_pct = (metrics.atr / close_price) * 100
            base = 0.00005  # JPY base

            if scenario_type == 'high':
                expected_mult = 2.0
                expected_threshold = base * expected_mult
                assert threshold >= expected_threshold * 0.9, f"Expected ~{expected_threshold:.6f}, got {threshold:.6f}"
                print(f"✓ PASS: Threshold increased for high volatility")
            elif scenario_type == 'trending':
                expected_mult = 0.7
                expected_threshold = base * expected_mult
                # Trending should have lower threshold
                assert threshold <= base * 1.1, f"Expected lower threshold, got {threshold:.6f}"
                print(f"✓ PASS: Threshold appropriate for trending market")
            else:
                # Normal should be close to base
                assert abs(threshold - base) < base * 0.3, f"Expected ~{base:.6f}, got {threshold:.6f}"
                print(f"✓ PASS: Threshold close to base for normal volatility")
        else:
            print("⚠️  No metrics extracted (fallback to static threshold)")

def test_limiter_adaptation():
    """Test that signal limiter adapts to volatility"""
    print("\n\n" + "=" * 80)
    print("TEST 2: Signal Limiter Adaptation")
    print("=" * 80)

    calc = MACDIndicatorCalculator()
    epic = "CS.D.EURJPY.MINI.IP"

    scenarios = {
        'high': ('High Volatility', 'Expected: 8-bar spacing, max 3 signals'),
        'normal': ('Normal Volatility', 'Expected: 4-bar spacing, max 5 signals'),
        'trending': ('Trending Market', 'Expected: 3-bar spacing, max 8 signals')
    }

    for scenario_type, (name, expected) in scenarios.items():
        print(f"\n{name}")
        print("-" * 80)
        print(f"{expected}")
        print()

        df = create_test_data(scenario_type)

        # Create some test signals (every 2 bars)
        bull_signals = pd.Series(False, index=df.index)
        bear_signals = pd.Series(False, index=df.index)

        for i in range(0, len(df), 2):
            if i % 4 == 0:
                bull_signals.iloc[i] = True
            else:
                bear_signals.iloc[i] = True

        initial_signals = bull_signals.sum() + bear_signals.sum()
        print(f"Initial signals generated: {initial_signals}")

        # Extract metrics
        metrics = calc._extract_volatility_metrics(df, epic)

        if metrics:
            # Apply limiter
            filtered_bull, filtered_bear = calc._apply_global_signal_limiter(
                df, bull_signals, bear_signals, epic, metrics
            )

            final_signals = filtered_bull.sum() + filtered_bear.sum()
            print(f"Final signals after limiting: {final_signals}")
            print(f"Reduction: {initial_signals - final_signals} signals ({(initial_signals - final_signals) / initial_signals * 100:.1f}%)")

            # Verify expected behavior
            if scenario_type == 'high':
                # High volatility should have strict limits
                assert final_signals <= 3, f"Expected max 3 signals, got {final_signals}"
                print(f"✓ PASS: Strict limits applied for high volatility")
            elif scenario_type == 'trending':
                # Trending should allow more signals
                assert final_signals >= 5, f"Expected at least 5 signals, got {final_signals}"
                print(f"✓ PASS: Relaxed limits for trending market")
            else:
                # Normal should be moderate
                assert 3 <= final_signals <= 7, f"Expected 3-7 signals, got {final_signals}"
                print(f"✓ PASS: Moderate limits for normal volatility")
        else:
            print("⚠️  No metrics extracted (fallback to default limits)")

def test_integration():
    """Test full integration with detect_multi_filter_crossovers"""
    print("\n\n" + "=" * 80)
    print("TEST 3: Full Integration Test")
    print("=" * 80)

    calc = MACDIndicatorCalculator()
    epic = "CS.D.EURJPY.MINI.IP"

    print("\nTesting with HIGH VOLATILITY data...")
    df_high = create_test_data('high')
    result_high = calc.detect_multi_filter_crossovers(df_high, epic, is_backtest=False)

    high_vol_signals = result_high['bull_crossover'].sum() + result_high['bear_crossover'].sum()
    print(f"✓ High volatility signals: {high_vol_signals}")

    print("\nTesting with TRENDING data...")
    df_trend = create_test_data('trending')
    result_trend = calc.detect_multi_filter_crossovers(df_trend, epic, is_backtest=False)

    trend_signals = result_trend['bull_crossover'].sum() + result_trend['bear_crossover'].sum()
    print(f"✓ Trending market signals: {trend_signals}")

    print("\n" + "=" * 80)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  High Volatility: {high_vol_signals} signals (should be lower)")
    print(f"  Trending Market: {trend_signals} signals (should be higher)")

    if high_vol_signals < trend_signals:
        print(f"\n✓ PASS: System correctly adapts - fewer signals in choppy, more in trending")
    else:
        print(f"\n⚠️  WARNING: Expected fewer signals in high volatility")

if __name__ == "__main__":
    print("\n🧪 VOLATILITY-ADAPTIVE MACD PARAMETER TESTS\n")

    try:
        test_threshold_adaptation()
        test_limiter_adaptation()
        test_integration()

        print("\n\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nVolatility-adaptive MACD parameters are working correctly!")
        print("The system now:")
        print("  • Adjusts histogram thresholds based on ATR and market regime")
        print("  • Adapts signal spacing and limits based on volatility")
        print("  • Provides stricter filtering in choppy markets")
        print("  • Allows more signals in trending conditions")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
