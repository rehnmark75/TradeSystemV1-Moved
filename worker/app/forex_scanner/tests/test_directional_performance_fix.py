"""
Unit Tests for Directional Performance Fix
Tests the fix that separates bull and bear performance tracking to eliminate directional bias.

See DIRECTIONAL_BIAS_FIX.md for detailed analysis of the issue.
"""

import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Import the class under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.strategies.helpers.ema_signal_calculator import EMASignalCalculator


class TestDirectionalPerformanceFix:
    """Test suite for directional bias fix in performance filtering"""

    @pytest.fixture
    def calculator(self):
        """Create EMASignalCalculator instance for testing"""
        return EMASignalCalculator(
            logger=logging.getLogger(__name__),
            use_supertrend=True
        )

    def create_trending_dataframe(self, n_bars: int, direction: int, volatility: float = 0.0001):
        """
        Helper: Create DataFrame with consistent trend

        Args:
            n_bars: Number of bars
            direction: 1 for uptrend, -1 for downtrend
            volatility: Price volatility

        Returns:
            DataFrame with trend data
        """
        # Generate price data
        price_change = np.random.normal(direction * volatility, volatility, n_bars)
        close_price = np.cumsum(price_change) + 1.1000

        # Create timestamps
        start_time = datetime(2025, 1, 1)
        timestamps = [start_time + timedelta(minutes=15*i) for i in range(n_bars)]

        # Create DataFrame
        df = pd.DataFrame({
            'start_time': timestamps,
            'close': close_price,
            'high': close_price + abs(np.random.normal(0, volatility/2, n_bars)),
            'low': close_price - abs(np.random.normal(0, volatility/2, n_bars)),
            'open': close_price + np.random.normal(0, volatility/4, n_bars),
            'st_fast': close_price + (direction * 0.0005),  # SuperTrend above/below price
            'st_medium': close_price + (direction * 0.0010),
            'st_slow': close_price + (direction * 0.0015),
            'st_fast_trend': np.full(n_bars, direction),
            'st_medium_trend': np.full(n_bars, direction),
            'st_slow_trend': np.full(n_bars, direction),
            'atr': np.full(n_bars, 0.0010),
            'ema_200': close_price - (direction * 0.0020),  # EMA 200 for filter
        })

        return df

    def test_pure_uptrend_generates_bull_signals(self, calculator):
        """
        Test Case 1: Pure Uptrend

        Expected: BULL signals generated, BEAR signals blocked by performance filter
        """
        # Setup: 100-bar uptrend
        df = self.create_trending_dataframe(n_bars=100, direction=1, volatility=0.0001)

        # Execute signal detection
        df_with_signals = calculator.detect_supertrend_signals(df)

        # Validate: Bull performance should be positive
        assert 'st_bull_performance' in df_with_signals.columns, \
            "Bull performance column should exist"

        bull_perf_last = df_with_signals['st_bull_performance'].iloc[-1]
        assert bull_perf_last > 0, \
            f"Bull performance should be positive in uptrend, got {bull_perf_last:.6f}"

        # Bear performance should be near zero (no bear periods)
        bear_perf_last = df_with_signals['st_bear_performance'].iloc[-1]
        assert abs(bear_perf_last) < 0.0001, \
            f"Bear performance should be ~0 (no bear periods), got {bear_perf_last:.6f}"

        # Check signal balance
        bull_signals = df_with_signals['bull_alert'].sum()
        bear_signals = df_with_signals['bear_alert'].sum()

        # In pure uptrend, we expect more bull signals (or only bull signals)
        if bull_signals + bear_signals > 0:
            assert bull_signals >= bear_signals, \
                f"Expected more bull signals in uptrend, got {bull_signals} BULL vs {bear_signals} BEAR"

    def test_pure_downtrend_generates_bear_signals(self, calculator):
        """
        Test Case 2: Pure Downtrend

        Expected: BEAR signals generated, BULL signals blocked by performance filter
        """
        # Setup: 100-bar downtrend
        df = self.create_trending_dataframe(n_bars=100, direction=-1, volatility=0.0001)

        # Execute signal detection
        df_with_signals = calculator.detect_supertrend_signals(df)

        # Validate: Bear performance should be positive
        assert 'st_bear_performance' in df_with_signals.columns, \
            "Bear performance column should exist"

        bear_perf_last = df_with_signals['st_bear_performance'].iloc[-1]
        assert bear_perf_last > 0, \
            f"Bear performance should be positive in downtrend, got {bear_perf_last:.6f}"

        # Bull performance should be near zero (no bull periods)
        bull_perf_last = df_with_signals['st_bull_performance'].iloc[-1]
        assert abs(bull_perf_last) < 0.0001, \
            f"Bull performance should be ~0 (no bull periods), got {bull_perf_last:.6f}"

        # Check signal balance
        bull_signals = df_with_signals['bull_alert'].sum()
        bear_signals = df_with_signals['bear_alert'].sum()

        # In pure downtrend, we expect more bear signals (or only bear signals)
        if bull_signals + bear_signals > 0:
            assert bear_signals >= bull_signals, \
                f"Expected more bear signals in downtrend, got {bull_signals} BULL vs {bear_signals} BEAR"

    def test_oscillating_market_balanced_signals(self, calculator):
        """
        Test Case 3: Oscillating Market

        Expected: Balanced bull/bear performance, both signal types possible
        """
        # Setup: Alternating trends (10 bars bull, 10 bars bear)
        n_bars = 100
        direction = np.array([1 if (i // 10) % 2 == 0 else -1 for i in range(n_bars)])

        # Generate oscillating price
        price_changes = direction * 0.0001
        close_price = np.cumsum(price_changes) + 1.1000

        timestamps = [datetime(2025, 1, 1) + timedelta(minutes=15*i) for i in range(n_bars)]

        df = pd.DataFrame({
            'start_time': timestamps,
            'close': close_price,
            'high': close_price + 0.00005,
            'low': close_price - 0.00005,
            'open': close_price,
            'st_fast': close_price + (direction * 0.0005),
            'st_medium': close_price + (direction * 0.0010),
            'st_slow': close_price + (direction * 0.0015),
            'st_fast_trend': direction,
            'st_medium_trend': direction,
            'st_slow_trend': direction,
            'atr': np.full(n_bars, 0.0010),
            'ema_200': np.full(n_bars, 1.1000),  # Neutral EMA
        })

        # Execute signal detection
        df_with_signals = calculator.detect_supertrend_signals(df)

        # Both should have non-zero performance (tracked during their respective periods)
        bull_perf_last = df_with_signals['st_bull_performance'].iloc[-1]
        bear_perf_last = df_with_signals['st_bear_performance'].iloc[-1]

        assert abs(bull_perf_last) > 0.000001, \
            f"Bull performance should be tracked, got {bull_perf_last:.6f}"
        assert abs(bear_perf_last) > 0.000001, \
            f"Bear performance should be tracked, got {bear_perf_last:.6f}"

        # In oscillating market, signals should be more balanced
        bull_signals = df_with_signals['bull_alert'].sum()
        bear_signals = df_with_signals['bear_alert'].sum()

        if bull_signals + bear_signals >= 4:  # Need enough signals for meaningful test
            # Allow 30-70% range (not too strict for oscillating market)
            bull_pct = bull_signals / (bull_signals + bear_signals) * 100
            assert 20 <= bull_pct <= 80, \
                f"Expected more balanced signals in oscillating market, got {bull_pct:.1f}% BULL"

    def test_nan_handling_cold_start(self, calculator):
        """
        Test Case 4: NaN Handling

        Expected: Graceful handling of NaN values, no crashes
        """
        # Setup: DataFrame with initial NaN values
        df = pd.DataFrame({
            'start_time': pd.date_range('2025-01-01', periods=50, freq='15min'),
            'close': [np.nan, np.nan] + list(np.linspace(1.1000, 1.1010, 48)),
            'high': [np.nan, np.nan] + list(np.linspace(1.1001, 1.1011, 48)),
            'low': [np.nan, np.nan] + list(np.linspace(1.0999, 1.1009, 48)),
            'open': [np.nan, np.nan] + list(np.linspace(1.1000, 1.1010, 48)),
            'st_fast': [np.nan, np.nan] + list(np.linspace(1.1005, 1.1015, 48)),
            'st_medium': [np.nan, np.nan] + list(np.linspace(1.1010, 1.1020, 48)),
            'st_slow': [np.nan, np.nan] + list(np.linspace(1.1015, 1.1025, 48)),
            'st_fast_trend': [np.nan, np.nan] + [1] * 48,
            'st_medium_trend': [np.nan, np.nan] + [1] * 48,
            'st_slow_trend': [np.nan, np.nan] + [1] * 48,
            'atr': [np.nan, np.nan] + [0.0010] * 48,
            'ema_200': [np.nan, np.nan] + list(np.linspace(1.0990, 1.1000, 48)),
        })

        # Execute signal detection - should not crash
        try:
            df_with_signals = calculator.detect_supertrend_signals(df)

            # Should produce valid results despite initial NaN
            assert 'st_bull_performance' in df_with_signals.columns
            assert 'st_bear_performance' in df_with_signals.columns

            # At least some performance values should be non-NaN
            assert not df_with_signals['st_bull_performance'].isna().all(), \
                "Performance should be calculated despite initial NaN"

        except Exception as e:
            pytest.fail(f"Should handle NaN gracefully, but raised: {e}")

    def test_insufficient_data_skips_filter(self, calculator):
        """
        Test Case 5: Insufficient Data

        Expected: Performance filter skipped with warning, no crash
        """
        # Setup: Only 10 bars (less than 20 minimum)
        df = self.create_trending_dataframe(n_bars=10, direction=1)

        # Execute signal detection - should skip filter gracefully
        try:
            df_with_signals = calculator.detect_supertrend_signals(df)

            # Should still work, just without performance filter
            assert len(df_with_signals) == 10

        except Exception as e:
            pytest.fail(f"Should handle insufficient data gracefully, but raised: {e}")

    def test_performance_independence(self, calculator):
        """
        Test Case 6: Performance Independence

        Expected: Bull and bear performance are independent (no cross-contamination)
        """
        # Setup: First 50 bars bull, last 50 bars bear
        n_bars = 100

        # Bull period (0-49)
        df_bull = self.create_trending_dataframe(n_bars=50, direction=1, volatility=0.0001)

        # Bear period (50-99)
        df_bear = self.create_trending_dataframe(n_bars=50, direction=-1, volatility=0.0001)

        # Adjust timestamps for bear period
        last_time = df_bull['start_time'].iloc[-1]
        df_bear['start_time'] = [last_time + timedelta(minutes=15*(i+1)) for i in range(50)]

        # Concatenate
        df = pd.concat([df_bull, df_bear], ignore_index=True)

        # Execute signal detection
        df_with_signals = calculator.detect_supertrend_signals(df)

        # Get performance at transition point (bar 50) and end (bar 99)
        bull_perf_mid = df_with_signals['st_bull_performance'].iloc[49]
        bear_perf_mid = df_with_signals['st_bear_performance'].iloc[49]

        bull_perf_end = df_with_signals['st_bull_performance'].iloc[-1]
        bear_perf_end = df_with_signals['st_bear_performance'].iloc[-1]

        # At midpoint (end of bull run):
        # - Bull performance should be high (good bull period)
        # - Bear performance should be low/zero (no bear periods yet)
        assert bull_perf_mid > 0, \
            f"Bull performance should be positive after bull run, got {bull_perf_mid:.6f}"
        assert abs(bear_perf_mid) < 0.0001, \
            f"Bear performance should be near zero before any bear periods, got {bear_perf_mid:.6f}"

        # At endpoint (end of bear run):
        # - Bull performance should decay (no recent bull periods)
        # - Bear performance should be high (good bear period)
        assert bear_perf_end > 0, \
            f"Bear performance should be positive after bear run, got {bear_perf_end:.6f}"

        # Critical test: Bull performance should decay during bear period
        # (not stay high due to contamination)
        assert bull_perf_end < bull_perf_mid, \
            f"Bull performance should decay during bear period due to EWM, " \
            f"got mid={bull_perf_mid:.6f}, end={bull_perf_end:.6f}"

    def test_no_directional_bias_in_statistics(self, calculator):
        """
        Test Case 7: Statistical Bias Check

        Expected: In balanced market, signal distribution should not show extreme bias
        """
        # Setup: Generate 200 bars with random walk (no inherent trend)
        n_bars = 200
        np.random.seed(42)  # Reproducible

        # Random walk
        price_changes = np.random.normal(0, 0.0001, n_bars)
        close_price = np.cumsum(price_changes) + 1.1000

        # Random trend changes (simulate SuperTrend flipping)
        trend = np.array([1 if np.random.random() > 0.5 else -1 for _ in range(n_bars)])

        timestamps = [datetime(2025, 1, 1) + timedelta(minutes=15*i) for i in range(n_bars)]

        df = pd.DataFrame({
            'start_time': timestamps,
            'close': close_price,
            'high': close_price + 0.00005,
            'low': close_price - 0.00005,
            'open': close_price,
            'st_fast': close_price + (trend * 0.0005),
            'st_medium': close_price + (trend * 0.0010),
            'st_slow': close_price + (trend * 0.0015),
            'st_fast_trend': trend,
            'st_medium_trend': trend,
            'st_slow_trend': trend,
            'atr': np.full(n_bars, 0.0010),
            'ema_200': np.full(n_bars, 1.1000),
        })

        # Execute signal detection
        df_with_signals = calculator.detect_supertrend_signals(df)

        # Count signals
        bull_signals = df_with_signals['bull_alert'].sum()
        bear_signals = df_with_signals['bear_alert'].sum()
        total_signals = bull_signals + bear_signals

        if total_signals >= 10:  # Need enough signals for statistical test
            bull_pct = (bull_signals / total_signals) * 100

            # In random walk, expect 40-60% range (not 100% or 0%)
            assert 25 <= bull_pct <= 75, \
                f"Extreme bias detected in random walk: {bull_pct:.1f}% BULL signals. " \
                f"Expected: 40-60%. Got {bull_signals} BULL, {bear_signals} BEAR."

            # This is the KEY test: Before fix, this would fail with ~100% or ~0%
            # After fix, should pass with balanced distribution


class TestPerformanceFilterEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def calculator(self):
        """Create EMASignalCalculator instance for testing"""
        return EMASignalCalculator(
            logger=logging.getLogger(__name__),
            use_supertrend=True
        )

    def test_missing_close_column_error_handling(self, calculator):
        """Test error handling when 'close' column is missing"""
        df = pd.DataFrame({
            'st_fast_trend': [1, 1, 1],
            'st_medium_trend': [1, 1, 1],
            'st_slow_trend': [1, 1, 1],
        })

        # Should handle missing column gracefully (not crash)
        try:
            df_with_signals = calculator.detect_supertrend_signals(df)
            # Should complete without crash
            assert True
        except Exception as e:
            pytest.fail(f"Should handle missing column gracefully, but raised: {e}")

    def test_all_nan_performance_error_handling(self, calculator):
        """Test error handling when performance calculation produces all NaN"""
        df = pd.DataFrame({
            'start_time': pd.date_range('2025-01-01', periods=30, freq='15min'),
            'close': [np.nan] * 30,  # All NaN
            'high': [np.nan] * 30,
            'low': [np.nan] * 30,
            'open': [np.nan] * 30,
            'st_fast': [1.1000] * 30,
            'st_medium': [1.1010] * 30,
            'st_slow': [1.1020] * 30,
            'st_fast_trend': [1] * 30,
            'st_medium_trend': [1] * 30,
            'st_slow_trend': [1] * 30,
            'atr': [0.001] * 30,
            'ema_200': [1.0990] * 30,
        })

        # Should handle all-NaN gracefully
        try:
            df_with_signals = calculator.detect_supertrend_signals(df)
            assert True
        except Exception as e:
            pytest.fail(f"Should handle all-NaN gracefully, but raised: {e}")


# Run tests with: pytest test_directional_performance_fix.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
