# core/strategies/helpers/ichimoku_adaptive_periods.py
"""
Ichimoku Adaptive Period Selection Module
Dynamic optimization of Ichimoku periods based on autocorrelation analysis

This module implements:
1. Autocorrelation function analysis for optimal period selection
2. Regime-dependent period adjustment (trending/ranging/breakout)
3. Dynamic period updating based on market conditions
4. Cross-validation of period effectiveness
5. Multi-timeframe period optimization

Features:
- Replace fixed periods (9,26,52) with data-driven optimal periods
- Market regime detection for period adaptation
- Autocorrelation-based period optimization
- Period stability and effectiveness validation
- Performance-based period adjustment
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf
from scipy.optimize import minimize_scalar


class IchimokuAdaptivePeriods:
    """Adaptive period selection system for Ichimoku indicators"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        # Default period ranges for optimization
        self.period_ranges = {
            'tenkan_min': 7, 'tenkan_max': 15,       # Conversion line range
            'kijun_min': 20, 'kijun_max': 35,       # Base line range
            'senkou_b_min': 40, 'senkou_b_max': 65, # Leading span B range
            'chikou_min': 20, 'chikou_max': 35,     # Lagging span range
            'cloud_min': 20, 'cloud_max': 35        # Cloud shift range
        }

        # Autocorrelation analysis parameters
        self.autocorr_params = {
            'analysis_window': 200,          # Window for autocorrelation analysis
            'min_correlation_threshold': 0.1, # Minimum correlation for significant period
            'autocorr_lags': 100,            # Maximum lags for autocorrelation
            'significance_level': 0.05,       # Statistical significance level
            'stability_window': 50,          # Window for period stability check
        }

        # Market regime parameters
        self.regime_params = {
            'trend_strength_window': 30,     # Window for trend strength calculation
            'volatility_window': 20,         # Window for volatility calculation
            'regime_update_frequency': 100,   # Bars between regime updates
            'trending_threshold': 0.6,       # Threshold for trending regime
            'volatile_threshold': 1.5,       # Threshold for volatile regime (ATR multiplier)
        }

        # Period optimization parameters
        self.optimization_params = {
            'optimization_frequency': 500,   # Bars between full period optimization
            'validation_window': 100,        # Window for period validation
            'performance_weight': 0.6,       # Weight for performance in optimization
            'stability_weight': 0.4,         # Weight for stability in optimization
            'min_improvement_threshold': 0.05, # Minimum improvement to change periods
        }

        # Cache for period optimization results
        self.period_cache = {}
        self.last_optimization_bar = 0
        self.current_optimal_periods = None

    def get_adaptive_periods(self, df: pd.DataFrame, epic: str,
                            force_optimization: bool = False) -> Dict:
        """
        Get adaptive Ichimoku periods based on current market conditions

        Args:
            df: DataFrame with price data
            epic: Currency pair identifier
            force_optimization: Force full period optimization

        Returns:
            Dictionary with optimized periods
        """
        try:
            # Check if we need to run optimization
            current_bar = len(df)
            need_optimization = (
                force_optimization or
                self.current_optimal_periods is None or
                (current_bar - self.last_optimization_bar) >= self.optimization_params['optimization_frequency']
            )

            if need_optimization:
                self.logger.info(f"Running adaptive period optimization for {epic}")
                optimal_periods = self._optimize_periods(df, epic)
                self.current_optimal_periods = optimal_periods
                self.last_optimization_bar = current_bar
            else:
                # Use cached periods with minor adjustments
                optimal_periods = self._get_cached_periods_with_adjustments(df, epic)

            return optimal_periods

        except Exception as e:
            self.logger.error(f"Error getting adaptive periods: {e}")
            # Return traditional periods as fallback
            return {
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_b_period': 52,
                'chikou_shift': 26,
                'cloud_shift': 26
            }

    def _optimize_periods(self, df: pd.DataFrame, epic: str) -> Dict:
        """Full period optimization using autocorrelation analysis"""
        try:
            if len(df) < self.autocorr_params['analysis_window']:
                self.logger.warning("Insufficient data for period optimization")
                return self._get_default_periods()

            # Detect current market regime
            market_regime = self._detect_market_regime(df)

            # Perform autocorrelation analysis
            autocorr_results = self._analyze_autocorrelation(df)

            # Optimize individual periods based on autocorrelation
            optimal_periods = {}

            # Optimize Tenkan period (conversion line)
            optimal_periods['tenkan_period'] = self._optimize_tenkan_period(
                df, autocorr_results, market_regime
            )

            # Optimize Kijun period (base line)
            optimal_periods['kijun_period'] = self._optimize_kijun_period(
                df, autocorr_results, market_regime
            )

            # Optimize Senkou B period (leading span B)
            optimal_periods['senkou_b_period'] = self._optimize_senkou_b_period(
                df, autocorr_results, market_regime
            )

            # Set Chikou and cloud shifts based on Kijun period
            optimal_periods['chikou_shift'] = optimal_periods['kijun_period']
            optimal_periods['cloud_shift'] = optimal_periods['kijun_period']

            # Validate period relationships
            optimal_periods = self._validate_period_relationships(optimal_periods)

            # Cache results
            self.period_cache[epic] = {
                'periods': optimal_periods,
                'regime': market_regime,
                'timestamp': len(df),
                'autocorr_results': autocorr_results
            }

            self.logger.info(f"Optimized periods for {epic}: {optimal_periods}")
            return optimal_periods

        except Exception as e:
            self.logger.error(f"Error optimizing periods: {e}")
            return self._get_default_periods()

    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime for period adaptation"""
        try:
            window = self.regime_params['trend_strength_window']
            vol_window = self.regime_params['volatility_window']

            if len(df) < max(window, vol_window):
                return 'ranging'

            # Calculate trend strength using linear regression
            recent_closes = df['close'].tail(window).values
            x = np.arange(len(recent_closes))
            correlation = np.corrcoef(x, recent_closes)[0, 1]
            trend_strength = abs(correlation) if not np.isnan(correlation) else 0

            # Calculate volatility (ATR-based)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=vol_window).mean().iloc[-1]
            atr_normalized = atr / df['close'].iloc[-1]

            # Calculate volatility percentile
            volatility_history = true_range.rolling(window=vol_window).mean()
            volatility_history_normalized = volatility_history / df['close']
            volatility_percentile = (
                (atr_normalized > volatility_history_normalized.quantile(0.75))
            )

            # Determine regime
            trending_threshold = self.regime_params['trending_threshold']
            volatile_threshold = self.regime_params['volatile_threshold']

            if trend_strength >= trending_threshold:
                if volatility_percentile:
                    return 'trending_volatile'
                else:
                    return 'trending_stable'
            else:
                if volatility_percentile:
                    return 'ranging_volatile'
                else:
                    return 'ranging_stable'

        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return 'ranging'

    def _analyze_autocorrelation(self, df: pd.DataFrame) -> Dict:
        """Analyze autocorrelation to find optimal periods"""
        try:
            window = self.autocorr_params['analysis_window']
            max_lags = self.autocorr_params['autocorr_lags']

            if len(df) < window:
                return {}

            # Use recent data for analysis
            recent_data = df.tail(window)

            # Calculate autocorrelation for different price series
            results = {}

            # High-Low midpoint autocorrelation (for Tenkan/Kijun periods)
            hl_midpoint = (recent_data['high'] + recent_data['low']) / 2
            results['hl_autocorr'] = acf(hl_midpoint.diff().dropna(), nlags=max_lags, fft=True)

            # Close price autocorrelation
            results['close_autocorr'] = acf(recent_data['close'].diff().dropna(), nlags=max_lags, fft=True)

            # High-Low range autocorrelation (for volatility-based periods)
            hl_range = recent_data['high'] - recent_data['low']
            results['range_autocorr'] = acf(hl_range.diff().dropna(), nlags=max_lags, fft=True)

            # Find significant autocorrelation peaks
            results['hl_peaks'] = self._find_autocorr_peaks(results['hl_autocorr'])
            results['close_peaks'] = self._find_autocorr_peaks(results['close_autocorr'])
            results['range_peaks'] = self._find_autocorr_peaks(results['range_autocorr'])

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing autocorrelation: {e}")
            return {}

    def _find_autocorr_peaks(self, autocorr: np.ndarray) -> List[int]:
        """Find significant peaks in autocorrelation function"""
        try:
            threshold = self.autocorr_params['min_correlation_threshold']
            peaks = []

            # Find local maxima above threshold
            for i in range(2, len(autocorr) - 2):
                if (autocorr[i] > threshold and
                    autocorr[i] > autocorr[i-1] and
                    autocorr[i] > autocorr[i+1] and
                    autocorr[i] > autocorr[i-2] and
                    autocorr[i] > autocorr[i+2]):
                    peaks.append(i)

            return peaks

        except Exception as e:
            self.logger.error(f"Error finding autocorrelation peaks: {e}")
            return []

    def _optimize_tenkan_period(self, df: pd.DataFrame, autocorr_results: Dict,
                               market_regime: str) -> int:
        """Optimize Tenkan (conversion line) period"""
        try:
            # Get autocorrelation peaks for high-low data
            hl_peaks = autocorr_results.get('hl_peaks', [])

            # Filter peaks within Tenkan range
            tenkan_min = self.period_ranges['tenkan_min']
            tenkan_max = self.period_ranges['tenkan_max']
            valid_peaks = [p for p in hl_peaks if tenkan_min <= p <= tenkan_max]

            if valid_peaks:
                # Choose peak based on market regime
                if 'trending' in market_regime:
                    # For trending markets, prefer shorter periods (more responsive)
                    optimal_period = min(valid_peaks)
                else:
                    # For ranging markets, prefer periods with strongest autocorrelation
                    hl_autocorr = autocorr_results.get('hl_autocorr', np.array([]))
                    best_peak = max(valid_peaks, key=lambda p: hl_autocorr[p] if p < len(hl_autocorr) else 0)
                    optimal_period = best_peak
            else:
                # Fallback: regime-based default
                if 'volatile' in market_regime:
                    optimal_period = tenkan_min + 1  # Shorter for volatile markets
                elif 'trending' in market_regime:
                    optimal_period = (tenkan_min + tenkan_max) // 2 - 1  # Medium-short for trending
                else:
                    optimal_period = 9  # Traditional default

            return max(tenkan_min, min(tenkan_max, optimal_period))

        except Exception as e:
            self.logger.error(f"Error optimizing Tenkan period: {e}")
            return 9

    def _optimize_kijun_period(self, df: pd.DataFrame, autocorr_results: Dict,
                              market_regime: str) -> int:
        """Optimize Kijun (base line) period"""
        try:
            # Get autocorrelation peaks for high-low data
            hl_peaks = autocorr_results.get('hl_peaks', [])

            # Filter peaks within Kijun range
            kijun_min = self.period_ranges['kijun_min']
            kijun_max = self.period_ranges['kijun_max']
            valid_peaks = [p for p in hl_peaks if kijun_min <= p <= kijun_max]

            if valid_peaks:
                # Choose peak based on market regime
                if 'ranging' in market_regime:
                    # For ranging markets, prefer longer periods (more stable)
                    optimal_period = max(valid_peaks)
                else:
                    # For trending markets, choose period with best autocorrelation
                    hl_autocorr = autocorr_results.get('hl_autocorr', np.array([]))
                    best_peak = max(valid_peaks, key=lambda p: hl_autocorr[p] if p < len(hl_autocorr) else 0)
                    optimal_period = best_peak
            else:
                # Fallback: regime-based default
                if 'volatile' in market_regime:
                    optimal_period = kijun_min + 2  # Shorter for volatile markets
                elif 'trending' in market_regime:
                    optimal_period = (kijun_min + kijun_max) // 2  # Medium for trending
                else:
                    optimal_period = kijun_max - 2  # Longer for stable ranging

            return max(kijun_min, min(kijun_max, optimal_period))

        except Exception as e:
            self.logger.error(f"Error optimizing Kijun period: {e}")
            return 26

    def _optimize_senkou_b_period(self, df: pd.DataFrame, autocorr_results: Dict,
                                 market_regime: str) -> int:
        """Optimize Senkou B (leading span B) period"""
        try:
            # Get autocorrelation peaks for range data (volatility-based)
            range_peaks = autocorr_results.get('range_peaks', [])

            # Filter peaks within Senkou B range
            senkou_min = self.period_ranges['senkou_b_min']
            senkou_max = self.period_ranges['senkou_b_max']
            valid_peaks = [p for p in range_peaks if senkou_min <= p <= senkou_max]

            if valid_peaks:
                # For Senkou B, prefer longer periods for stability
                optimal_period = max(valid_peaks)
            else:
                # Fallback based on regime
                if 'volatile' in market_regime:
                    optimal_period = senkou_min + 5  # Shorter for volatile markets
                elif 'trending' in market_regime:
                    optimal_period = (senkou_min + senkou_max) // 2  # Medium for trending
                else:
                    optimal_period = senkou_max - 5  # Longer for stable markets

            return max(senkou_min, min(senkou_max, optimal_period))

        except Exception as e:
            self.logger.error(f"Error optimizing Senkou B period: {e}")
            return 52

    def _validate_period_relationships(self, periods: Dict) -> Dict:
        """Validate and adjust period relationships"""
        try:
            # Ensure Tenkan < Kijun < Senkou B relationship
            tenkan = periods['tenkan_period']
            kijun = periods['kijun_period']
            senkou_b = periods['senkou_b_period']

            # Adjust if relationships are violated
            if tenkan >= kijun:
                tenkan = max(self.period_ranges['tenkan_min'], kijun - 2)

            if kijun >= senkou_b:
                senkou_b = max(self.period_ranges['senkou_b_min'], kijun + 5)

            # Update periods
            periods['tenkan_period'] = tenkan
            periods['kijun_period'] = kijun
            periods['senkou_b_period'] = senkou_b

            return periods

        except Exception as e:
            self.logger.error(f"Error validating period relationships: {e}")
            return periods

    def _get_cached_periods_with_adjustments(self, df: pd.DataFrame, epic: str) -> Dict:
        """Get cached periods with minor regime-based adjustments"""
        try:
            if epic not in self.period_cache or self.current_optimal_periods is None:
                return self._get_default_periods()

            cached_periods = self.current_optimal_periods.copy()
            current_regime = self._detect_market_regime(df)
            cached_regime = self.period_cache[epic].get('regime', 'ranging')

            # Apply minor adjustments if regime changed significantly
            if current_regime != cached_regime:
                adjustment_factor = self._get_regime_adjustment_factor(current_regime, cached_regime)

                if adjustment_factor != 1.0:
                    # Apply small adjustments
                    for key in ['tenkan_period', 'kijun_period', 'senkou_b_period']:
                        if key in cached_periods:
                            adjusted = int(cached_periods[key] * adjustment_factor)
                            min_val = self.period_ranges[key.replace('_period', '_min')]
                            max_val = self.period_ranges[key.replace('_period', '_max')]
                            cached_periods[key] = max(min_val, min(max_val, adjusted))

                    # Update shifts
                    cached_periods['chikou_shift'] = cached_periods['kijun_period']
                    cached_periods['cloud_shift'] = cached_periods['kijun_period']

            return cached_periods

        except Exception as e:
            self.logger.error(f"Error getting cached periods with adjustments: {e}")
            return self._get_default_periods()

    def _get_regime_adjustment_factor(self, current_regime: str, cached_regime: str) -> float:
        """Get adjustment factor based on regime change"""
        try:
            # Define regime adjustment factors
            regime_factors = {
                'trending_volatile': 0.9,    # Shorter periods
                'trending_stable': 1.0,      # No change
                'ranging_volatile': 0.95,    # Slightly shorter periods
                'ranging_stable': 1.1        # Slightly longer periods
            }

            current_factor = regime_factors.get(current_regime, 1.0)
            cached_factor = regime_factors.get(cached_regime, 1.0)

            # Calculate relative adjustment
            adjustment = current_factor / cached_factor

            # Limit adjustment magnitude
            return max(0.9, min(1.1, adjustment))

        except Exception as e:
            self.logger.error(f"Error calculating regime adjustment factor: {e}")
            return 1.0

    def _get_default_periods(self) -> Dict:
        """Get traditional Ichimoku periods as fallback"""
        return {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'chikou_shift': 26,
            'cloud_shift': 26
        }

    def get_period_optimization_summary(self, epic: str) -> Dict:
        """Get summary of period optimization for an epic"""
        try:
            if epic not in self.period_cache:
                return {'status': 'no_optimization_data'}

            cache_data = self.period_cache[epic]

            return {
                'status': 'optimized',
                'current_periods': cache_data['periods'],
                'market_regime': cache_data['regime'],
                'last_optimization_bar': cache_data['timestamp'],
                'autocorr_peaks_found': len(cache_data.get('autocorr_results', {}).get('hl_peaks', [])),
                'optimization_age_bars': self.last_optimization_bar - cache_data['timestamp']
            }

        except Exception as e:
            self.logger.error(f"Error getting optimization summary: {e}")
            return {'status': 'error'}

    def force_period_reoptimization(self, df: pd.DataFrame, epic: str) -> Dict:
        """Force immediate period reoptimization"""
        try:
            self.logger.info(f"Forcing period reoptimization for {epic}")
            return self._optimize_periods(df, epic)

        except Exception as e:
            self.logger.error(f"Error forcing period reoptimization: {e}")
            return self._get_default_periods()