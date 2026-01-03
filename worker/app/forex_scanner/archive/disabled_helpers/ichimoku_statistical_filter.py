# core/strategies/helpers/ichimoku_statistical_filter.py
"""
Ichimoku Statistical Filter Module
Advanced statistical filtering for Ichimoku signals to optimize quality over quantity

This module implements:
1. Volatility-normalized signal validation using ATR-based thresholds
2. Dynamic cloud thickness filtering based on volatility percentiles
3. Instrument-specific z-score normalization for cross-market consistency
4. Statistical outlier detection and noise filtering

Enhanced Quality Features:
- ATR-normalized TK cross strength validation
- Volatility-adaptive cloud thickness thresholds
- Z-score based signal strength normalization
- Historical volatility regime detection
- Statistical confidence scoring
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.stats import zscore


class IchimokuStatisticalFilter:
    """Advanced statistical filtering for Ichimoku signal quality optimization"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        # Statistical parameters for quality filtering
        self.default_params = {
            'atr_period': 20,                    # Period for ATR calculation
            'volatility_lookback': 100,          # Lookback for volatility percentiles
            'zscore_window': 50,                 # Window for z-score normalization
            'min_atr_multiplier': 2.0,           # Minimum ATR multiplier for signals
            'cloud_thickness_percentile': 60,    # Percentile threshold for cloud thickness
            'signal_strength_zscore_threshold': 1.0,  # Z-score threshold for signal strength
            'volatility_regime_threshold': 1.5,  # Threshold for high volatility regime
            'outlier_zscore_threshold': 3.0,     # Outlier detection threshold
        }

    def apply_statistical_filters(self, df: pd.DataFrame, ichimoku_config: Dict = None) -> pd.DataFrame:
        """
        Apply comprehensive statistical filtering to Ichimoku signals

        Args:
            df: DataFrame with Ichimoku indicators and signals
            ichimoku_config: Ichimoku configuration parameters

        Returns:
            DataFrame with enhanced statistical filtering applied
        """
        try:
            df_filtered = df.copy()

            # 1. Calculate volatility metrics (ATR, rolling volatility)
            df_filtered = self._calculate_volatility_metrics(df_filtered)

            # 2. Apply volatility-normalized signal validation
            df_filtered = self._apply_volatility_normalized_validation(df_filtered)

            # 3. Apply dynamic cloud thickness filtering
            df_filtered = self._apply_dynamic_cloud_filtering(df_filtered)

            # 4. Apply z-score normalization for signal strength
            df_filtered = self._apply_zscore_normalization(df_filtered)

            # 5. Detect and filter statistical outliers
            df_filtered = self._filter_statistical_outliers(df_filtered)

            # 6. Calculate enhanced confidence scores
            df_filtered = self._calculate_enhanced_confidence(df_filtered)

            self.logger.info("Statistical filtering applied successfully")
            return df_filtered

        except Exception as e:
            self.logger.error(f"Error applying statistical filters: {e}")
            return df

    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility metrics for normalization"""
        try:
            atr_period = self.default_params['atr_period']

            # Calculate True Range components
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))

            # True Range (TR)
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

            # Average True Range (ATR)
            df['atr'] = df['true_range'].rolling(window=atr_period).mean()

            # Normalized ATR (ATR as percentage of price)
            df['atr_normalized'] = df['atr'] / df['close']

            # Rolling volatility (price changes)
            df['price_change'] = df['close'].pct_change()
            df['rolling_volatility'] = df['price_change'].rolling(window=atr_period).std()

            # Volatility regime detection
            volatility_threshold = self.default_params['volatility_regime_threshold']
            volatility_median = df['atr_normalized'].rolling(window=self.default_params['volatility_lookback']).median()
            df['high_volatility_regime'] = df['atr_normalized'] > (volatility_median * volatility_threshold)

            self.logger.debug("Volatility metrics calculated successfully")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {e}")
            return df

    def _apply_volatility_normalized_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ATR-normalized validation for TK cross and cloud signals"""
        try:
            min_atr_multiplier = self.default_params['min_atr_multiplier']

            # ATR-normalized TK cross strength validation
            df['tk_separation_normalized'] = abs(df['tenkan_sen'] - df['kijun_sen']) / df['atr']
            df['tk_cross_valid_atr'] = df['tk_separation_normalized'] >= min_atr_multiplier

            # ATR-normalized cloud breakout validation
            df['cloud_distance_bull'] = np.where(
                df['close'] > df['cloud_top'],
                (df['close'] - df['cloud_top']) / df['atr'],
                0
            )

            df['cloud_distance_bear'] = np.where(
                df['close'] < df['cloud_bottom'],
                (df['cloud_bottom'] - df['close']) / df['atr'],
                0
            )

            df['cloud_breakout_valid_atr'] = (
                (df['cloud_distance_bull'] >= min_atr_multiplier) |
                (df['cloud_distance_bear'] >= min_atr_multiplier)
            )

            # Enhanced signal validation combining original and ATR-based validation
            df['tk_bull_cross_enhanced'] = (
                df.get('tk_bull_cross', False) &
                df['tk_cross_valid_atr']
            )

            df['tk_bear_cross_enhanced'] = (
                df.get('tk_bear_cross', False) &
                df['tk_cross_valid_atr']
            )

            df['cloud_bull_breakout_enhanced'] = (
                df.get('cloud_bull_breakout', False) &
                df['cloud_breakout_valid_atr']
            )

            df['cloud_bear_breakout_enhanced'] = (
                df.get('cloud_bear_breakout', False) &
                df['cloud_breakout_valid_atr']
            )

            self.logger.debug("Volatility-normalized validation applied")
            return df

        except Exception as e:
            self.logger.error(f"Error applying volatility-normalized validation: {e}")
            return df

    def _apply_dynamic_cloud_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply dynamic cloud thickness filtering based on volatility percentiles"""
        try:
            lookback = self.default_params['volatility_lookback']
            percentile_threshold = self.default_params['cloud_thickness_percentile']

            # Calculate cloud thickness as percentage of price
            df['cloud_thickness_pct'] = df['cloud_thickness'] / df['close']

            # Dynamic cloud thickness threshold based on historical percentiles
            df['cloud_thickness_threshold_dynamic'] = (
                df['cloud_thickness_pct']
                .rolling(window=lookback, min_periods=20)
                .quantile(percentile_threshold / 100.0)
            )

            # Cloud thickness validation
            df['cloud_thick_enough'] = (
                df['cloud_thickness_pct'] >= df['cloud_thickness_threshold_dynamic']
            )

            # Adjust based on volatility regime
            df['cloud_thickness_adjusted'] = np.where(
                df['high_volatility_regime'],
                df['cloud_thickness_threshold_dynamic'] * 1.5,  # Higher threshold in high volatility
                df['cloud_thickness_threshold_dynamic']
            )

            df['cloud_valid_dynamic'] = (
                df['cloud_thickness_pct'] >= df['cloud_thickness_adjusted']
            )

            self.logger.debug("Dynamic cloud filtering applied")
            return df

        except Exception as e:
            self.logger.error(f"Error applying dynamic cloud filtering: {e}")
            return df

    def _apply_zscore_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization for signal strength consistency"""
        try:
            zscore_window = self.default_params['zscore_window']
            zscore_threshold = self.default_params['signal_strength_zscore_threshold']

            # Z-score normalize TK cross strength
            df['tk_cross_strength_zscore'] = (
                df['tk_cross_strength']
                .rolling(window=zscore_window, min_periods=10)
                .apply(lambda x: zscore(x)[-1] if len(x) >= 10 else 0)
            )

            # Z-score normalize cloud breakout strength
            if 'cloud_breakout_strength_bull' in df.columns:
                df['cloud_breakout_bull_zscore'] = (
                    df['cloud_breakout_strength_bull']
                    .rolling(window=zscore_window, min_periods=10)
                    .apply(lambda x: zscore(x)[-1] if len(x) >= 10 and x.std() > 0 else 0)
                )

            if 'cloud_breakout_strength_bear' in df.columns:
                df['cloud_breakout_bear_zscore'] = (
                    df['cloud_breakout_strength_bear']
                    .rolling(window=zscore_window, min_periods=10)
                    .apply(lambda x: zscore(x)[-1] if len(x) >= 10 and x.std() > 0 else 0)
                )

            # Signal strength validation based on z-scores
            df['signal_strength_valid_zscore'] = (
                (abs(df['tk_cross_strength_zscore']) >= zscore_threshold) |
                (abs(df.get('cloud_breakout_bull_zscore', 0)) >= zscore_threshold) |
                (abs(df.get('cloud_breakout_bear_zscore', 0)) >= zscore_threshold)
            )

            self.logger.debug("Z-score normalization applied")
            return df

        except Exception as e:
            self.logger.error(f"Error applying z-score normalization: {e}")
            return df

    def _filter_statistical_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out statistical outliers that could represent false signals"""
        try:
            outlier_threshold = self.default_params['outlier_zscore_threshold']

            # Detect price outliers (extreme price movements)
            df['price_change_zscore'] = abs(zscore(df['price_change'].fillna(0)))
            df['price_outlier'] = df['price_change_zscore'] > outlier_threshold

            # Detect ATR outliers (extreme volatility spikes)
            df['atr_zscore'] = abs(zscore(df['atr_normalized'].fillna(0)))
            df['atr_outlier'] = df['atr_zscore'] > outlier_threshold

            # Filter signals during outlier conditions
            df['outlier_filter_passed'] = ~(df['price_outlier'] | df['atr_outlier'])

            # Apply outlier filtering to enhanced signals
            df['tk_bull_cross_filtered'] = (
                df.get('tk_bull_cross_enhanced', False) &
                df['outlier_filter_passed']
            )

            df['tk_bear_cross_filtered'] = (
                df.get('tk_bear_cross_enhanced', False) &
                df['outlier_filter_passed']
            )

            df['cloud_bull_breakout_filtered'] = (
                df.get('cloud_bull_breakout_enhanced', False) &
                df['outlier_filter_passed']
            )

            df['cloud_bear_breakout_filtered'] = (
                df.get('cloud_bear_breakout_enhanced', False) &
                df['outlier_filter_passed']
            )

            self.logger.debug("Statistical outlier filtering applied")
            return df

        except Exception as e:
            self.logger.error(f"Error filtering statistical outliers: {e}")
            return df

    def _calculate_enhanced_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced confidence scores incorporating statistical factors"""
        try:
            # Base confidence factors
            base_factors = []

            # Volatility-adjusted confidence
            volatility_confidence = np.where(
                df['high_volatility_regime'],
                0.8,  # Lower confidence in high volatility
                1.2   # Higher confidence in normal volatility
            )
            base_factors.append(volatility_confidence)

            # ATR validation confidence
            atr_validation_confidence = np.where(
                df.get('tk_cross_valid_atr', False) | df.get('cloud_breakout_valid_atr', False),
                1.3,  # Boost for ATR-validated signals
                0.7   # Penalty for weak signals
            )
            base_factors.append(atr_validation_confidence)

            # Cloud thickness confidence
            cloud_confidence = np.where(
                df.get('cloud_valid_dynamic', False),
                1.2,  # Boost for thick enough cloud
                0.8   # Penalty for thin cloud
            )
            base_factors.append(cloud_confidence)

            # Z-score strength confidence
            zscore_confidence = np.where(
                df.get('signal_strength_valid_zscore', False),
                1.1,  # Boost for statistically significant strength
                0.9   # Small penalty for weak statistical strength
            )
            base_factors.append(zscore_confidence)

            # Outlier filter confidence
            outlier_confidence = np.where(
                df.get('outlier_filter_passed', True),
                1.0,  # Neutral for normal conditions
                0.5   # Strong penalty for outlier conditions
            )
            base_factors.append(outlier_confidence)

            # Calculate composite statistical confidence multiplier
            df['statistical_confidence_multiplier'] = np.prod(base_factors, axis=0)

            # Ensure reasonable bounds (0.3 to 1.5)
            df['statistical_confidence_multiplier'] = np.clip(
                df['statistical_confidence_multiplier'], 0.3, 1.5
            )

            # Enhanced signal flags for final validation
            df['ichimoku_signal_enhanced'] = (
                df.get('tk_bull_cross_filtered', False) |
                df.get('tk_bear_cross_filtered', False) |
                df.get('cloud_bull_breakout_filtered', False) |
                df.get('cloud_bear_breakout_filtered', False)
            )

            self.logger.debug("Enhanced confidence calculation completed")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating enhanced confidence: {e}")
            return df

    def get_signal_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate signal quality metrics for monitoring and optimization"""
        try:
            if len(df) == 0:
                return {}

            metrics = {}

            # Filter efficiency metrics
            original_signals = (
                df.get('tk_bull_cross', False).sum() +
                df.get('tk_bear_cross', False).sum() +
                df.get('cloud_bull_breakout', False).sum() +
                df.get('cloud_bear_breakout', False).sum()
            )

            filtered_signals = df.get('ichimoku_signal_enhanced', False).sum()

            metrics['signal_reduction_ratio'] = (
                1.0 - (filtered_signals / original_signals) if original_signals > 0 else 0
            )

            # Quality indicators
            metrics['avg_atr_normalized_strength'] = df['tk_separation_normalized'].mean()
            metrics['avg_cloud_thickness_pct'] = df['cloud_thickness_pct'].mean()
            metrics['high_volatility_regime_pct'] = df['high_volatility_regime'].mean() * 100
            metrics['outlier_periods_pct'] = (~df['outlier_filter_passed']).mean() * 100

            # Statistical confidence metrics
            metrics['avg_statistical_confidence'] = df['statistical_confidence_multiplier'].mean()
            metrics['min_statistical_confidence'] = df['statistical_confidence_multiplier'].min()
            metrics['max_statistical_confidence'] = df['statistical_confidence_multiplier'].max()

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating signal quality metrics: {e}")
            return {}

    def validate_signal_statistical_quality(self, latest_row: pd.Series, signal_type: str) -> Dict:
        """
        Validate signal quality using statistical filters

        Args:
            latest_row: Latest data row with Ichimoku indicators
            signal_type: 'BULL' or 'BEAR'

        Returns:
            Dictionary with validation results and confidence adjustments
        """
        try:
            validation_result = {
                'atr_validation_passed': False,
                'cloud_thickness_passed': False,
                'zscore_strength_passed': False,
                'outlier_filter_passed': True,
                'statistical_confidence_multiplier': 1.0,
                'quality_score': 0.0
            }

            # ATR validation
            tk_separation_norm = latest_row.get('tk_separation_normalized', 0)
            min_atr_mult = self.default_params['min_atr_multiplier']
            validation_result['atr_validation_passed'] = tk_separation_norm >= min_atr_mult

            # Cloud thickness validation
            validation_result['cloud_thickness_passed'] = latest_row.get('cloud_valid_dynamic', False)

            # Z-score strength validation
            validation_result['zscore_strength_passed'] = latest_row.get('signal_strength_valid_zscore', False)

            # Outlier validation
            validation_result['outlier_filter_passed'] = latest_row.get('outlier_filter_passed', True)

            # Statistical confidence multiplier
            validation_result['statistical_confidence_multiplier'] = latest_row.get(
                'statistical_confidence_multiplier', 1.0
            )

            # Calculate overall quality score (0-100)
            quality_factors = [
                validation_result['atr_validation_passed'],
                validation_result['cloud_thickness_passed'],
                validation_result['zscore_strength_passed'],
                validation_result['outlier_filter_passed']
            ]

            validation_result['quality_score'] = (
                sum(quality_factors) / len(quality_factors) * 100 *
                validation_result['statistical_confidence_multiplier']
            )

            return validation_result

        except Exception as e:
            self.logger.error(f"Error validating signal statistical quality: {e}")
            return {
                'atr_validation_passed': False,
                'cloud_thickness_passed': False,
                'zscore_strength_passed': False,
                'outlier_filter_passed': False,
                'statistical_confidence_multiplier': 0.5,
                'quality_score': 0.0
            }