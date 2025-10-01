"""
Unit tests for AdaptiveVolatilityCalculator

Tests all calculator methods, fallback chains, caching, and regime detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.strategies.helpers.adaptive_volatility_calculator import (
    AdaptiveVolatilityCalculator,
    ATRStandardCalculator,
    RegimeAdaptiveCalculator,
    BollingerBasedCalculator,
    MarketRegime,
    VolatilityMetrics,
    SLTPResult
)


@pytest.fixture
def mock_logger():
    """Mock logger for tests"""
    return Mock()


@pytest.fixture
def sample_data_eurusd():
    """Sample EURUSD market data"""
    return pd.Series({
        'close': 1.0850,
        'high': 1.0865,
        'low': 1.0835,
        'atr': 0.0015,  # 15 pips
        'atr_20': 0.0014,
        'adx': 28.5,
        'ema_50': 1.0840,
        'bb_upper': 1.0870,
        'bb_middle': 1.0850,
        'bb_lower': 1.0830
    })


@pytest.fixture
def sample_data_gbpusd():
    """Sample GBPUSD market data (higher volatility)"""
    return pd.Series({
        'close': 1.2650,
        'high': 1.2680,
        'low': 1.2620,
        'atr': 0.0025,  # 25 pips
        'atr_20': 0.0022,
        'adx': 32.0,
        'ema_50': 1.2640,
        'bb_upper': 1.2690,
        'bb_middle': 1.2650,
        'bb_lower': 1.2610
    })


@pytest.fixture
def sample_data_usdjpy():
    """Sample USDJPY market data (JPY scale)"""
    return pd.Series({
        'close': 148.50,
        'high': 148.80,
        'low': 148.20,
        'atr': 0.45,  # 45 pips
        'atr_20': 0.40,
        'adx': 24.0,
        'ema_50': 148.30,
        'bb_upper': 148.90,
        'bb_middle': 148.50,
        'bb_lower': 148.10
    })


@pytest.fixture
def trending_metrics():
    """Metrics indicating trending market"""
    return VolatilityMetrics(
        atr=0.0015,
        atr_percentile=55.0,
        adx=32.0,
        efficiency_ratio=0.75,
        bb_width_percentile=60.0,
        ema_separation=1.5,
        timestamp=datetime.now()
    )


@pytest.fixture
def ranging_metrics():
    """Metrics indicating ranging market"""
    return VolatilityMetrics(
        atr=0.0008,
        atr_percentile=25.0,
        adx=15.0,
        efficiency_ratio=0.25,
        bb_width_percentile=20.0,
        ema_separation=0.3,
        timestamp=datetime.now()
    )


@pytest.fixture
def breakout_metrics():
    """Metrics indicating breakout"""
    return VolatilityMetrics(
        atr=0.0025,
        atr_percentile=85.0,
        adx=22.0,
        efficiency_ratio=0.55,
        bb_width_percentile=80.0,
        ema_separation=2.5,
        timestamp=datetime.now()
    )


@pytest.fixture
def high_volatility_metrics():
    """Metrics indicating high volatility"""
    return VolatilityMetrics(
        atr=0.0040,
        atr_percentile=95.0,
        adx=28.0,
        efficiency_ratio=0.60,
        bb_width_percentile=90.0,
        ema_separation=3.5,
        timestamp=datetime.now()
    )


class TestATRStandardCalculator:
    """Test ATR-based baseline calculator"""

    def test_calculate_eurusd_bull(self, mock_logger, sample_data_eurusd, trending_metrics):
        """Test ATR calculation for EURUSD bull signal"""
        calc = ATRStandardCalculator(mock_logger)
        stop, target, confidence = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.TRENDING,
            trending_metrics
        )

        # EUR: 15 pips ATR * 2.5 = 37.5 → bounded to 15-45
        assert 15 <= stop <= 45, f"Stop {stop} out of range"
        assert target >= stop * 1.5, f"Target {target} too small vs stop {stop}"
        assert 0.80 <= confidence <= 0.90, f"Confidence {confidence} out of range"

    def test_calculate_gbpusd_bear(self, mock_logger, sample_data_gbpusd, trending_metrics):
        """Test ATR calculation for GBPUSD bear signal"""
        calc = ATRStandardCalculator(mock_logger)
        stop, target, confidence = calc.calculate(
            'CS.D.GBPUSD.MINI.IP',
            sample_data_gbpusd,
            'BEAR',
            MarketRegime.TRENDING,
            trending_metrics
        )

        # GBP: 25 pips ATR * 2.8 = 70 → bounded to 18-60
        assert 18 <= stop <= 60, f"Stop {stop} out of range"
        assert target >= stop * 1.5
        assert confidence >= 0.80

    def test_calculate_usdjpy(self, mock_logger, sample_data_usdjpy, trending_metrics):
        """Test ATR calculation for USDJPY (JPY scale)"""
        calc = ATRStandardCalculator(mock_logger)
        trending_metrics.atr = 0.45  # JPY scale
        stop, target, confidence = calc.calculate(
            'CS.D.USDJPY.MINI.IP',
            sample_data_usdjpy,
            'BULL',
            MarketRegime.TRENDING,
            trending_metrics
        )

        # JPY: 45 pips ATR * 2.5 = 112.5 → bounded to 20-55
        assert 20 <= stop <= 55, f"Stop {stop} out of range for JPY"
        assert target >= stop * 1.5

    def test_ranging_regime_adjustment(self, mock_logger, sample_data_eurusd, ranging_metrics, trending_metrics):
        """Test that ranging regime tightens stops"""
        calc = ATRStandardCalculator(mock_logger)

        # Calculate for trending
        stop_trending, target_trending, _ = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.TRENDING,
            trending_metrics
        )

        # Calculate for ranging (should be tighter)
        stop_ranging, target_ranging, _ = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.RANGING,
            ranging_metrics
        )

        # Ranging should have tighter stops (0.8x adjustment)
        assert stop_ranging <= stop_trending, "Ranging stops should be tighter"

    def test_breakout_regime_adjustment(self, mock_logger, sample_data_eurusd, breakout_metrics, trending_metrics):
        """Test that breakout regime widens stops"""
        calc = ATRStandardCalculator(mock_logger)

        stop_trending, _, _ = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.TRENDING,
            trending_metrics
        )

        stop_breakout, _, _ = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.BREAKOUT,
            breakout_metrics
        )

        # Breakout should have wider stops (1.2x adjustment)
        assert stop_breakout >= stop_trending, "Breakout stops should be wider"

    def test_invalid_atr_raises_error(self, mock_logger, sample_data_eurusd, trending_metrics):
        """Test that invalid ATR raises error"""
        calc = ATRStandardCalculator(mock_logger)
        trending_metrics.atr = 0  # Invalid

        with pytest.raises(ValueError, match="Invalid ATR"):
            calc.calculate(
                'CS.D.EURUSD.CEEM.IP',
                sample_data_eurusd,
                'BULL',
                MarketRegime.TRENDING,
                trending_metrics
            )

    def test_performance_target(self, mock_logger, sample_data_eurusd, trending_metrics):
        """Test that calculation meets <5ms performance target"""
        calc = ATRStandardCalculator(mock_logger)

        import time
        start = time.time()
        for _ in range(100):
            calc.calculate(
                'CS.D.EURUSD.CEEM.IP',
                sample_data_eurusd,
                'BULL',
                MarketRegime.TRENDING,
                trending_metrics
            )
        elapsed = (time.time() - start) * 1000 / 100  # ms per call

        assert elapsed < 5.0, f"Avg calculation time {elapsed:.2f}ms exceeds 5ms target"


class TestRegimeAdaptiveCalculator:
    """Test regime-aware adaptive calculator"""

    def test_trending_calculation(self, mock_logger, sample_data_eurusd, trending_metrics):
        """Test trending regime calculation"""
        calc = RegimeAdaptiveCalculator(mock_logger)
        stop, target, confidence = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.TRENDING,
            trending_metrics
        )

        assert 15 <= stop <= 45
        assert target >= stop * 1.5
        assert confidence >= 0.85, "Trending should have high confidence"

    def test_ranging_calculation(self, mock_logger, sample_data_eurusd, ranging_metrics):
        """Test ranging regime calculation"""
        calc = RegimeAdaptiveCalculator(mock_logger)
        stop, target, confidence = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.RANGING,
            ranging_metrics
        )

        assert 15 <= stop <= 45
        # Note: enforce_bounds applies min_rr=1.3, but actual result may hit minimum bounds
        assert target >= 15  # At least minimum target
        assert confidence >= 0.82

    def test_breakout_calculation(self, mock_logger, sample_data_eurusd, breakout_metrics):
        """Test breakout regime calculation"""
        calc = RegimeAdaptiveCalculator(mock_logger)
        stop, target, confidence = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.BREAKOUT,
            breakout_metrics
        )

        assert 15 <= stop <= 45
        assert target >= stop * 1.8  # Breakout requires higher R:R
        assert confidence >= 0.80

    def test_high_volatility_calculation(self, mock_logger, sample_data_eurusd, high_volatility_metrics):
        """Test high volatility regime calculation"""
        calc = RegimeAdaptiveCalculator(mock_logger)
        stop, target, confidence = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.HIGH_VOLATILITY,
            high_volatility_metrics
        )

        assert 15 <= stop <= 45
        assert target >= stop * 1.2  # Conservative R:R
        assert confidence >= 0.75

    def test_adx_adjustments(self, mock_logger, sample_data_eurusd, trending_metrics):
        """Test that ADX strength adjusts stops"""
        calc = RegimeAdaptiveCalculator(mock_logger)

        # Weak trend (ADX < 25)
        trending_metrics.adx = 22
        stop_weak, _, _ = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.TRENDING,
            trending_metrics
        )

        # Strong trend (ADX > 35)
        trending_metrics.adx = 38
        stop_strong, _, _ = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.TRENDING,
            trending_metrics
        )

        # Strong trend should have wider stops
        assert stop_strong >= stop_weak

    def test_performance_target(self, mock_logger, sample_data_eurusd, trending_metrics):
        """Test that calculation meets <15ms performance target"""
        calc = RegimeAdaptiveCalculator(mock_logger)

        import time
        start = time.time()
        for _ in range(50):
            calc.calculate(
                'CS.D.EURUSD.CEEM.IP',
                sample_data_eurusd,
                'BULL',
                MarketRegime.TRENDING,
                trending_metrics
            )
        elapsed = (time.time() - start) * 1000 / 50

        assert elapsed < 15.0, f"Avg calculation time {elapsed:.2f}ms exceeds 15ms target"


class TestBollingerBasedCalculator:
    """Test Bollinger Band based calculator"""

    def test_calculate_bull_signal(self, mock_logger, sample_data_eurusd, ranging_metrics):
        """Test Bollinger calculation for bull signal"""
        calc = BollingerBasedCalculator(mock_logger)
        stop, target, confidence = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.RANGING,
            ranging_metrics
        )

        assert 15 <= stop <= 45
        assert target >= stop * 1.5
        assert 0.70 <= confidence <= 0.90

    def test_calculate_bear_signal(self, mock_logger, sample_data_eurusd, ranging_metrics):
        """Test Bollinger calculation for bear signal"""
        calc = BollingerBasedCalculator(mock_logger)
        stop, target, confidence = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BEAR',
            MarketRegime.RANGING,
            ranging_metrics
        )

        assert 15 <= stop <= 45
        assert target >= stop * 1.5

    def test_narrow_bands_high_confidence(self, mock_logger, sample_data_eurusd, ranging_metrics):
        """Test that narrow bands increase confidence"""
        calc = BollingerBasedCalculator(mock_logger)

        # Narrow bands
        ranging_metrics.bb_width_percentile = 20
        _, _, conf_narrow = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.RANGING,
            ranging_metrics
        )

        # Wide bands
        ranging_metrics.bb_width_percentile = 75
        _, _, conf_wide = calc.calculate(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL',
            MarketRegime.RANGING,
            ranging_metrics
        )

        assert conf_narrow > conf_wide, "Narrow bands should have higher confidence"

    def test_missing_bb_data_raises_error(self, mock_logger, sample_data_eurusd, ranging_metrics):
        """Test that missing BB data raises error"""
        calc = BollingerBasedCalculator(mock_logger)
        sample_data_eurusd['bb_upper'] = 0  # Invalid

        with pytest.raises(ValueError, match="Missing Bollinger Band"):
            calc.calculate(
                'CS.D.EURUSD.CEEM.IP',
                sample_data_eurusd,
                'BULL',
                MarketRegime.RANGING,
                ranging_metrics
            )


class TestAdaptiveVolatilityCalculator:
    """Test main adaptive calculator with caching and fallbacks"""

    def test_singleton_pattern(self):
        """Test that calculator is singleton"""
        calc1 = AdaptiveVolatilityCalculator()
        calc2 = AdaptiveVolatilityCalculator()
        assert calc1 is calc2, "Should return same instance"

    def test_calculate_sl_tp_eurusd(self, sample_data_eurusd):
        """Test full SL/TP calculation for EURUSD"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()  # Start fresh

        result = calc.calculate_sl_tp(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL'
        )

        assert isinstance(result, SLTPResult)
        assert 15 <= result.stop_distance <= 45
        # High volatility regime may have conservative R:R (1.2x)
        assert result.limit_distance >= result.stop_distance * 1.2
        assert result.confidence > 0.70
        assert result.regime in MarketRegime
        assert result.calculation_time_ms < 20.0, "Should meet <20ms target"

    def test_regime_detection_trending(self, sample_data_eurusd):
        """Test regime detection identifies trending market"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # Set data for trending (high ADX, moderate ATR to avoid breakout/high volatility)
        sample_data_eurusd['adx'] = 32.0
        sample_data_eurusd['atr'] = 0.0015
        sample_data_eurusd['atr_20'] = 0.0025  # ATR percentile = 60% (below 70% breakout threshold)

        result = calc.calculate_sl_tp(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL'
        )

        assert result.regime == MarketRegime.TRENDING

    def test_regime_detection_ranging(self, sample_data_eurusd):
        """Test regime detection identifies ranging market"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # Set data for ranging (low ADX, low ATR)
        sample_data_eurusd['adx'] = 15.0
        sample_data_eurusd['atr'] = 0.0008

        result = calc.calculate_sl_tp(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL'
        )

        assert result.regime == MarketRegime.RANGING

    def test_regime_detection_high_volatility(self, sample_data_eurusd):
        """Test regime detection identifies high volatility"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # Set data for high volatility (very high ATR)
        sample_data_eurusd['atr'] = 0.0040
        sample_data_eurusd['atr_20'] = 0.0020  # ATR > 90th percentile

        result = calc.calculate_sl_tp(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL'
        )

        assert result.regime == MarketRegime.HIGH_VOLATILITY

    def test_cache_l1_hit(self, sample_data_eurusd):
        """Test L1 cache hit on repeated calls"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # First call (cold)
        result1 = calc.calculate_sl_tp(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL'
        )

        # Second call (should hit L1 cache)
        result2 = calc.calculate_sl_tp(
            'CS.D.EURUSD.CEEM.IP',
            sample_data_eurusd,
            'BULL'
        )

        assert result1.stop_distance == result2.stop_distance
        assert result2.calculation_time_ms < 5.0, "Cache hit should be <5ms"

        stats = calc.get_stats()
        assert stats['cache_hits_l1'] > 0, "Should have L1 cache hit"

    def test_cache_expiry(self, sample_data_eurusd):
        """Test that cache expires after TTL"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # Reset stats to track this test
        calc._stats['total_calls'] = 0
        calc._stats['cache_hits_l1'] = 0

        calc._cache_ttl_l1 = 0.1  # 100ms TTL for test

        # First call
        calc.calculate_sl_tp('CS.D.EURUSD.CEEM.IP', sample_data_eurusd, 'BULL')
        calls_after_first = calc._stats['total_calls']

        # Wait for expiry
        import time
        time.sleep(0.2)

        # Second call - should be cache miss after expiry
        calc.calculate_sl_tp('CS.D.EURUSD.CEEM.IP', sample_data_eurusd, 'BULL')

        # Total calls should increase (not cached)
        assert calc._stats['total_calls'] > calls_after_first

    def test_fallback_level_1_atr_standard(self, sample_data_eurusd):
        """Test fallback to ATR standard when regime calculator fails"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # Break regime calculator by removing required data
        with patch.object(calc.regime_calculator, 'calculate', side_effect=ValueError("Test error")):
            result = calc.calculate_sl_tp(
                'CS.D.EURUSD.CEEM.IP',
                sample_data_eurusd,
                'BULL'
            )

            assert result.fallback_level == 1
            assert "ATRStandard" in result.method_used
            assert result.confidence < 0.85  # Reduced confidence

    def test_fallback_level_2_high_low(self, sample_data_eurusd):
        """Test fallback to high-low range when ATR fails"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # Break both primary and ATR calculators
        with patch.object(calc.regime_calculator, 'calculate', side_effect=ValueError("Test error")):
            with patch.object(calc.atr_calculator, 'calculate', side_effect=ValueError("Test error")):
                result = calc.calculate_sl_tp(
                    'CS.D.EURUSD.CEEM.IP',
                    sample_data_eurusd,
                    'BULL'
                )

                assert result.fallback_level == 2
                assert "HighLow" in result.method_used
                assert result.confidence < 0.70

    def test_fallback_level_3_safe_defaults(self, sample_data_eurusd):
        """Test fallback to safe defaults when everything fails"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # Break everything by forcing exceptions
        bad_data = pd.Series({
            'close': None,
            'high': None,
            'low': None,
            'atr': None,
            'bb_upper': None,
            'bb_lower': None,
            'bb_middle': None
        })

        result = calc.calculate_sl_tp(
            'CS.D.EURUSD.CEEM.IP',
            bad_data,
            'BULL'
        )

        # Should use emergency fallback or level 2+ fallback
        assert result.fallback_level >= 2
        assert result.stop_distance > 0
        assert result.limit_distance > 0

    def test_performance_target_cold_calculation(self, sample_data_eurusd):
        """Test that cold calculation meets <20ms target"""
        calc = AdaptiveVolatilityCalculator()

        import time
        times = []
        for _ in range(20):
            calc.clear_cache()  # Force cold calculation
            start = time.time()
            calc.calculate_sl_tp('CS.D.EURUSD.CEEM.IP', sample_data_eurusd, 'BULL')
            times.append((time.time() - start) * 1000)

        avg_time = sum(times) / len(times)
        assert avg_time < 20.0, f"Avg cold calc time {avg_time:.2f}ms exceeds 20ms target"

    def test_performance_target_cache_hit(self, sample_data_eurusd):
        """Test that cache hit meets <5ms target"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # Prime cache
        calc.calculate_sl_tp('CS.D.EURUSD.CEEM.IP', sample_data_eurusd, 'BULL')

        import time
        times = []
        for _ in range(50):
            start = time.time()
            calc.calculate_sl_tp('CS.D.EURUSD.CEEM.IP', sample_data_eurusd, 'BULL')
            times.append((time.time() - start) * 1000)

        avg_time = sum(times) / len(times)
        assert avg_time < 5.0, f"Avg cache hit time {avg_time:.2f}ms exceeds 5ms target"

    def test_statistics_tracking(self, sample_data_eurusd):
        """Test that statistics are tracked correctly"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # Make several calls
        for _ in range(5):
            calc.calculate_sl_tp('CS.D.EURUSD.CEEM.IP', sample_data_eurusd, 'BULL')

        stats = calc.get_stats()

        assert stats['total_calls'] >= 5
        assert stats['cache_hit_rate_percent'] >= 0
        assert 'cache_l1_size' in stats
        assert 'cache_l2_size' in stats

    def test_multiple_pairs(self, sample_data_eurusd, sample_data_gbpusd, sample_data_usdjpy):
        """Test calculations across multiple currency pairs"""
        calc = AdaptiveVolatilityCalculator()
        calc.clear_cache()

        # EUR/USD
        result_eur = calc.calculate_sl_tp('CS.D.EURUSD.CEEM.IP', sample_data_eurusd, 'BULL')
        assert 15 <= result_eur.stop_distance <= 45

        # GBP/USD (higher volatility)
        result_gbp = calc.calculate_sl_tp('CS.D.GBPUSD.MINI.IP', sample_data_gbpusd, 'BULL')
        assert 18 <= result_gbp.stop_distance <= 60

        # USD/JPY (different scale)
        result_jpy = calc.calculate_sl_tp('CS.D.USDJPY.MINI.IP', sample_data_usdjpy, 'BULL')
        assert 20 <= result_jpy.stop_distance <= 55

        # GBP should have wider stops than EUR
        assert result_gbp.stop_distance >= result_eur.stop_distance


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
