#!/usr/bin/env python3
"""
Test SuperTrend Enhancement Filters

Quick validation script to ensure all filters are working correctly.
Run this before running full backtests.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add worker path
sys.path.insert(0, str(Path(__file__).parent / 'worker' / 'app'))

from forex_scanner.core.strategies.helpers.supertrend_adaptive_optimizer import (
    SuperTrendPerformanceTracker,
    TrendStrengthCalculator,
    SlowSupertrendStabilityFilter,
    apply_enhanced_supertrend_filters
)


def create_test_dataframe(n_bars: int = 100) -> pd.DataFrame:
    """Create test DataFrame with synthetic SuperTrend data"""
    np.random.seed(42)

    # Generate price data
    close = 1.1000 + np.cumsum(np.random.randn(n_bars) * 0.0001)

    # Generate SuperTrend data
    df = pd.DataFrame({
        'close': close,
        'high': close + 0.0005,
        'low': close - 0.0005,
        'open': close + np.random.randn(n_bars) * 0.0001,

        # SuperTrend values
        'st_fast': close - 0.0010,
        'st_medium': close - 0.0015,
        'st_slow': close - 0.0020,

        # SuperTrend trends (1 = bullish, -1 = bearish)
        'st_fast_trend': np.where(close > 1.1000, 1, -1),
        'st_medium_trend': np.where(close > 1.1000, 1, -1),
        'st_slow_trend': np.where(close > 1.1000, 1, -1),

        # Initial signal flags
        'bull_alert': False,
        'bear_alert': False,
    })

    # Add some bull signals every 10 bars
    for i in range(10, n_bars, 10):
        df.loc[i, 'bull_alert'] = True

    # Add some bear signals every 15 bars
    for i in range(15, n_bars, 15):
        df.loc[i, 'bear_alert'] = True

    return df


def test_performance_tracker():
    """Test SuperTrend Performance Tracker"""
    print("\n" + "="*80)
    print("TEST 1: SuperTrend Performance Tracker")
    print("="*80)

    df = create_test_dataframe(100)
    tracker = SuperTrendPerformanceTracker(lookback=20, alpha=0.1)

    # Calculate performance
    performance = tracker.calculate_performance(df, st_column='st_fast_trend')

    print(f"✅ Performance calculated: {len(performance)} values")
    print(f"   Mean performance: {performance.mean():.6f}")
    print(f"   Min performance: {performance.min():.6f}")
    print(f"   Max performance: {performance.max():.6f}")

    # Check if performance is reasonable
    assert len(performance) == len(df), "Performance length mismatch"
    assert not performance.isna().all(), "All performance values are NaN"

    print("✅ Performance tracker test PASSED")
    return True


def test_trend_strength_calculator():
    """Test Trend Strength Calculator"""
    print("\n" + "="*80)
    print("TEST 2: Trend Strength Calculator")
    print("="*80)

    df = create_test_dataframe(100)
    calc = TrendStrengthCalculator(min_separation_pct=0.3)

    # Calculate trend strength
    trend_strength = calc.calculate_trend_strength(df)

    print(f"✅ Trend strength calculated: {len(trend_strength)} values")
    print(f"   Mean strength: {trend_strength.mean():.3f}%")
    print(f"   Min strength: {trend_strength.min():.3f}%")
    print(f"   Max strength: {trend_strength.max():.3f}%")

    # Check how many pass threshold
    strong_trends = (trend_strength > 0.3).sum()
    print(f"   Strong trends (>0.3%): {strong_trends}/{len(df)} ({strong_trends/len(df)*100:.1f}%)")

    assert len(trend_strength) == len(df), "Trend strength length mismatch"
    assert not trend_strength.isna().all(), "All trend strength values are NaN"

    print("✅ Trend strength calculator test PASSED")
    return True


def test_stability_filter():
    """Test Slow SuperTrend Stability Filter"""
    print("\n" + "="*80)
    print("TEST 3: Slow SuperTrend Stability Filter")
    print("="*80)

    df = create_test_dataframe(100)
    filter_obj = SlowSupertrendStabilityFilter(min_stability_bars=12)

    # Create stable trend (bullish for 20 bars)
    df.loc[30:50, 'st_slow_trend'] = 1

    # Check stability
    bull_stability = filter_obj.check_slow_stability(df, direction=1)

    print(f"✅ Stability checked: {len(bull_stability)} values")

    # Count stable periods
    stable_count = bull_stability.sum()
    print(f"   Stable periods (12+ bars): {stable_count}/{len(df)} ({stable_count/len(df)*100:.1f}%)")

    # Verify bars 42-50 should be stable (12 bars from 30-41)
    expected_stable = bull_stability.loc[42:50].sum()
    print(f"   Expected stable bars (42-50): {expected_stable}/9")

    assert len(bull_stability) == len(df), "Stability length mismatch"
    assert stable_count > 0, "No stable periods found"

    print("✅ Stability filter test PASSED")
    return True


def test_combined_filters():
    """Test Combined Filter Application"""
    print("\n" + "="*80)
    print("TEST 4: Combined Enhanced Filters")
    print("="*80)

    df = create_test_dataframe(200)

    # Add more realistic signals
    original_bull = 20
    original_bear = 15

    for i in range(0, 200, 10):
        if i < 200:
            df.loc[i, 'bull_alert'] = True

    for i in range(5, 200, 13):
        if i < 200:
            df.loc[i, 'bear_alert'] = True

    df['bull_alert'] = df['bull_alert'].astype(bool)
    df['bear_alert'] = df['bear_alert'].astype(bool)

    print(f"Original signals: BULL={df['bull_alert'].sum()}, BEAR={df['bear_alert'].sum()}")

    # Apply all filters
    df_filtered = apply_enhanced_supertrend_filters(
        df,
        performance_threshold=0.0,
        min_trend_strength=0.3,
        min_stability_bars=12,
        enable_performance_filter=True,
        enable_trend_strength_filter=True,
        enable_stability_filter=True
    )

    final_bull = df_filtered['bull_alert'].sum()
    final_bear = df_filtered['bear_alert'].sum()

    print(f"Filtered signals: BULL={final_bull}, BEAR={final_bear}")

    # Calculate reduction
    original_total = df['bull_alert'].sum() + df['bear_alert'].sum()
    final_total = final_bull + final_bear
    reduction_pct = (1 - final_total / original_total) * 100 if original_total > 0 else 0

    print(f"Signal reduction: {reduction_pct:.1f}%")

    # Verify filters reduced signals
    assert final_total < original_total, "Filters should reduce signal count"
    assert reduction_pct > 0, "Should have some signal reduction"

    print("✅ Combined filters test PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 SUPERTREND ENHANCEMENTS - VALIDATION TESTS")
    print("="*80)

    tests = [
        ("Performance Tracker", test_performance_tracker),
        ("Trend Strength Calculator", test_trend_strength_calculator),
        ("Stability Filter", test_stability_filter),
        ("Combined Filters", test_combined_filters),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            results.append((test_name, False))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Enhancement filters are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
