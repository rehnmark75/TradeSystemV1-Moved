# core/strategies/helpers/ichimoku_indicator_calculator.py
"""
Ichimoku Indicator Calculator Module
Handles Ichimoku Kinko Hyo calculations, signal detection and data validation

Ichimoku Components:
- Tenkan-sen (Conversion Line): (High9 + Low9) / 2
- Kijun-sen (Base Line): (High26 + Low26) / 2
- Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted +26
- Senkou Span B (Leading Span B): (High52 + Low52) / 2, shifted +26
- Chikou Span (Lagging Span): Close shifted -26

Signal Types:
- TK Cross: Tenkan crosses above/below Kijun
- Cloud Breakout: Price breaks above/below cloud (Senkou A/B range)
- Chikou Confirmation: Chikou span clear of historical price action
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class IchimokuIndicatorCalculator:
    """Calculates Ichimoku indicators and detects cloud-based signals"""

    def __init__(self, logger: logging.Logger = None, eps: float = 1e-8):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = eps

        # Default Ichimoku parameters (traditional: 9-26-52-26)
        self.default_config = {
            'tenkan_period': 9,      # Conversion line
            'kijun_period': 26,      # Base line
            'senkou_b_period': 52,   # Leading span B
            'chikou_shift': 26,      # Lagging span displacement
            'cloud_shift': 26,       # Cloud forward displacement
            'cloud_thickness_threshold': 0.0001,
            'tk_cross_strength_threshold': 0.5,
            'chikou_clear_threshold': 0.0002
        }

    def get_required_indicators(self, ichimoku_config: Dict = None) -> List[str]:
        """Get list of required indicators for Ichimoku strategy"""
        config = ichimoku_config or self.default_config
        return [
            'tenkan_sen',          # Conversion line
            'kijun_sen',          # Base line
            'senkou_span_a',      # Leading span A (future cloud edge)
            'senkou_span_b',      # Leading span B (future cloud edge)
            'chikou_span',        # Lagging span (displaced close)
            'cloud_top',          # Upper cloud boundary
            'cloud_bottom',       # Lower cloud boundary
            'tk_bull_cross',      # TK bullish crossover
            'tk_bear_cross',      # TK bearish crossover
            'cloud_bull_breakout', # Bullish cloud breakout
            'cloud_bear_breakout', # Bearish cloud breakout
            'close',              # Current price
            'high',               # High prices
            'low'                 # Low prices
        ]

    def validate_data_requirements(self, df: pd.DataFrame, min_bars: int = 80) -> bool:
        """Validate that we have enough data for Ichimoku calculations"""
        if len(df) < min_bars:
            self.logger.debug(f"Insufficient data: {len(df)} bars (need {min_bars})")
            return False

        required_columns = ['open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False

        return True

    def ensure_ichimoku_indicators(self, df: pd.DataFrame, ichimoku_config: Dict = None) -> pd.DataFrame:
        """
        Calculate Ichimoku indicators if not present

        Args:
            df: DataFrame with price data
            ichimoku_config: Ichimoku configuration parameters

        Returns:
            DataFrame with Ichimoku indicators added
        """
        config = ichimoku_config or self.default_config
        df_copy = df.copy()

        try:
            # Sort by time and reset index for proper calculation
            df_copy = df_copy.sort_values('start_time').reset_index(drop=True)

            # Calculate Ichimoku components
            df_copy = self._calculate_ichimoku_lines(df_copy, config)

            # Calculate cloud boundaries
            df_copy = self._calculate_cloud_boundaries(df_copy)

            # Add volume and volatility context if available
            df_copy = self._add_market_context(df_copy)

            self.logger.debug(f"Ichimoku indicators calculated for {len(df_copy)} bars")

            return df_copy

        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku indicators: {e}")
            return df_copy

    def _calculate_ichimoku_lines(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Calculate the five Ichimoku lines"""
        try:
            tenkan_period = config['tenkan_period']
            kijun_period = config['kijun_period']
            senkou_b_period = config['senkou_b_period']
            chikou_shift = config['chikou_shift']
            cloud_shift = config['cloud_shift']

            # 1. Tenkan-sen (Conversion Line): (High9 + Low9) / 2
            if 'tenkan_sen' not in df.columns:
                df['tenkan_high'] = df['high'].rolling(window=tenkan_period).max()
                df['tenkan_low'] = df['low'].rolling(window=tenkan_period).min()
                df['tenkan_sen'] = (df['tenkan_high'] + df['tenkan_low']) / 2
                self.logger.debug(f"Calculated Tenkan-sen with period {tenkan_period}")

            # 2. Kijun-sen (Base Line): (High26 + Low26) / 2
            if 'kijun_sen' not in df.columns:
                df['kijun_high'] = df['high'].rolling(window=kijun_period).max()
                df['kijun_low'] = df['low'].rolling(window=kijun_period).min()
                df['kijun_sen'] = (df['kijun_high'] + df['kijun_low']) / 2
                self.logger.debug(f"Calculated Kijun-sen with period {kijun_period}")

            # 3. Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward
            if 'senkou_span_a' not in df.columns:
                senkou_a_base = (df['tenkan_sen'] + df['kijun_sen']) / 2
                # Shift forward by cloud_shift periods (plot into future)
                df['senkou_span_a'] = senkou_a_base.shift(-cloud_shift)
                self.logger.debug(f"Calculated Senkou Span A with shift {cloud_shift}")

            # 4. Senkou Span B (Leading Span B): (High52 + Low52) / 2, shifted forward
            if 'senkou_span_b' not in df.columns:
                df['senkou_b_high'] = df['high'].rolling(window=senkou_b_period).max()
                df['senkou_b_low'] = df['low'].rolling(window=senkou_b_period).min()
                senkou_b_base = (df['senkou_b_high'] + df['senkou_b_low']) / 2
                # Shift forward by cloud_shift periods (plot into future)
                df['senkou_span_b'] = senkou_b_base.shift(-cloud_shift)
                self.logger.debug(f"Calculated Senkou Span B with period {senkou_b_period}")

            # 5. Chikou Span (Lagging Span): Close shifted backward
            if 'chikou_span' not in df.columns:
                # Shift backward by chikou_shift periods (plot into past)
                df['chikou_span'] = df['close'].shift(chikou_shift)
                self.logger.debug(f"Calculated Chikou Span with shift {chikou_shift}")

            return df

        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku lines: {e}")
            return df

    def _calculate_cloud_boundaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cloud top and bottom boundaries"""
        try:
            # Cloud boundaries are determined by Senkou Span A and B
            # Cloud top = max(Senkou A, Senkou B)
            # Cloud bottom = min(Senkou A, Senkou B)
            df['cloud_top'] = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
            df['cloud_bottom'] = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)

            # Calculate cloud thickness for signal strength
            df['cloud_thickness'] = df['cloud_top'] - df['cloud_bottom']

            # Cloud color (green if Senkou A > Senkou B, red otherwise)
            df['cloud_is_green'] = df['senkou_span_a'] > df['senkou_span_b']
            df['cloud_is_red'] = df['senkou_span_a'] <= df['senkou_span_b']

            self.logger.debug("Calculated cloud boundaries and thickness")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating cloud boundaries: {e}")
            return df

    def _add_market_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional market context indicators"""
        try:
            # Price position relative to cloud
            df['price_above_cloud'] = df['close'] > df['cloud_top']
            df['price_below_cloud'] = df['close'] < df['cloud_bottom']
            df['price_in_cloud'] = (df['close'] >= df['cloud_bottom']) & (df['close'] <= df['cloud_top'])

            # TK line relationship
            df['tenkan_above_kijun'] = df['tenkan_sen'] > df['kijun_sen']
            df['tenkan_below_kijun'] = df['tenkan_sen'] < df['kijun_sen']

            # Price vs TK lines
            df['price_above_tenkan'] = df['close'] > df['tenkan_sen']
            df['price_above_kijun'] = df['close'] > df['kijun_sen']

            return df

        except Exception as e:
            self.logger.error(f"Error adding market context: {e}")
            return df

    def detect_ichimoku_signals(self, df: pd.DataFrame, ichimoku_config: Dict = None) -> pd.DataFrame:
        """
        Detect Ichimoku signals: TK crosses and cloud breakouts

        Args:
            df: DataFrame with Ichimoku indicators
            ichimoku_config: Configuration parameters

        Returns:
            DataFrame with signal columns added
        """
        config = ichimoku_config or self.default_config

        try:
            # Get previous values for crossover detection
            df['prev_tenkan'] = df['tenkan_sen'].shift(1)
            df['prev_kijun'] = df['kijun_sen'].shift(1)
            df['prev_close'] = df['close'].shift(1)
            df['prev_cloud_top'] = df['cloud_top'].shift(1)
            df['prev_cloud_bottom'] = df['cloud_bottom'].shift(1)

            # TK Line Crossovers
            df = self._detect_tk_crossovers(df, config)

            # Cloud Breakouts
            df = self._detect_cloud_breakouts(df, config)

            # Combined signal strength
            df = self._calculate_signal_strength(df, config)

            return df

        except Exception as e:
            self.logger.error(f"Error detecting Ichimoku signals: {e}")
            return df

    def _detect_tk_crossovers(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect Tenkan-Kijun line crossovers with harmonic mean validation"""
        try:
            # Enhanced TK Cross Detection with Harmonic Mean Analysis
            df = self._calculate_harmonic_mean_analysis(df)

            # TK Bull Cross: Tenkan crosses above Kijun
            df['tk_bull_cross'] = (
                (df['prev_tenkan'] <= df['prev_kijun'] + self.eps) &  # Was below/equal
                (df['tenkan_sen'] > df['kijun_sen'] + self.eps)       # Now above
            )

            # TK Bear Cross: Tenkan crosses below Kijun
            df['tk_bear_cross'] = (
                (df['prev_tenkan'] >= df['prev_kijun'] - self.eps) &  # Was above/equal
                (df['tenkan_sen'] < df['kijun_sen'] - self.eps)       # Now below
            )

            # TK Cross strength (distance between lines)
            df['tk_cross_strength'] = abs(df['tenkan_sen'] - df['kijun_sen']) / (df['close'] + self.eps)

            # Enhanced TK crosses with harmonic mean validation
            df['tk_bull_cross_enhanced'] = (
                df['tk_bull_cross'] &
                df.get('harmonic_divergence_valid', True) &
                df.get('genuine_trend_change', True)
            )

            df['tk_bear_cross_enhanced'] = (
                df['tk_bear_cross'] &
                df.get('harmonic_divergence_valid', True) &
                df.get('genuine_trend_change', True)
            )

            return df

        except Exception as e:
            self.logger.error(f"Error detecting TK crossovers: {e}")
            return df

    def _calculate_harmonic_mean_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate harmonic mean analysis for genuine TK cross validation

        This method uses harmonic vs arithmetic mean divergence to detect
        genuine trend changes vs noise-based crossovers.
        """
        try:
            # Parameters for harmonic analysis
            harmonic_window = 10  # Window for harmonic/arithmetic mean comparison
            divergence_threshold = 0.02  # Minimum divergence for validation
            trend_change_threshold = 0.015  # Threshold for genuine trend change

            # Calculate harmonic and arithmetic means for Tenkan and Kijun
            df = self._calculate_harmonic_and_arithmetic_means(df, harmonic_window)

            # Calculate divergence between harmonic and arithmetic means
            df = self._calculate_harmonic_divergence(df, divergence_threshold)

            # Detect genuine trend changes using harmonic analysis
            df = self._detect_genuine_trend_changes(df, trend_change_threshold)

            self.logger.debug("Harmonic mean analysis completed")
            return df

        except Exception as e:
            self.logger.error(f"Error in harmonic mean analysis: {e}")
            return df

    def _calculate_harmonic_and_arithmetic_means(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling harmonic and arithmetic means for TK lines"""
        try:
            # Harmonic mean calculation (safe for positive values)
            def safe_harmonic_mean(values):
                """Calculate harmonic mean safely, handling zeros and negatives"""
                positive_values = values[values > 1e-10]  # Filter out zeros/negatives
                if len(positive_values) < len(values) * 0.5:  # Too many zeros/negatives
                    return np.nan
                if len(positive_values) == 0:
                    return np.nan
                return len(positive_values) / np.sum(1.0 / positive_values)

            # Tenkan harmonic and arithmetic means
            df['tenkan_harmonic_mean'] = df['tenkan_sen'].rolling(window=window).apply(
                safe_harmonic_mean, raw=True
            )
            df['tenkan_arithmetic_mean'] = df['tenkan_sen'].rolling(window=window).mean()

            # Kijun harmonic and arithmetic means
            df['kijun_harmonic_mean'] = df['kijun_sen'].rolling(window=window).apply(
                safe_harmonic_mean, raw=True
            )
            df['kijun_arithmetic_mean'] = df['kijun_sen'].rolling(window=window).mean()

            # TK spread harmonic and arithmetic means
            df['tk_spread'] = abs(df['tenkan_sen'] - df['kijun_sen'])
            df['tk_spread_harmonic_mean'] = df['tk_spread'].rolling(window=window).apply(
                safe_harmonic_mean, raw=True
            )
            df['tk_spread_arithmetic_mean'] = df['tk_spread'].rolling(window=window).mean()

            return df

        except Exception as e:
            self.logger.error(f"Error calculating harmonic/arithmetic means: {e}")
            return df

    def _calculate_harmonic_divergence(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Calculate divergence between harmonic and arithmetic means"""
        try:
            # Tenkan divergence
            df['tenkan_harmonic_divergence'] = abs(
                (df['tenkan_harmonic_mean'] - df['tenkan_arithmetic_mean']) /
                (df['tenkan_arithmetic_mean'] + self.eps)
            )

            # Kijun divergence
            df['kijun_harmonic_divergence'] = abs(
                (df['kijun_harmonic_mean'] - df['kijun_arithmetic_mean']) /
                (df['kijun_arithmetic_mean'] + self.eps)
            )

            # TK spread divergence
            df['tk_spread_harmonic_divergence'] = abs(
                (df['tk_spread_harmonic_mean'] - df['tk_spread_arithmetic_mean']) /
                (df['tk_spread_arithmetic_mean'] + self.eps)
            )

            # Composite divergence score
            df['composite_harmonic_divergence'] = (
                df['tenkan_harmonic_divergence'] +
                df['kijun_harmonic_divergence'] +
                df['tk_spread_harmonic_divergence']
            ) / 3

            # Divergence validation
            df['harmonic_divergence_valid'] = (
                df['composite_harmonic_divergence'] >= threshold
            )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating harmonic divergence: {e}")
            return df

    def _detect_genuine_trend_changes(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Detect genuine trend changes using harmonic mean analysis"""
        try:
            # Calculate trend change momentum using harmonic means
            df['tenkan_harmonic_momentum'] = df['tenkan_harmonic_mean'].pct_change()
            df['kijun_harmonic_momentum'] = df['kijun_harmonic_mean'].pct_change()

            # Trend alignment using harmonic means
            df['harmonic_trend_alignment'] = np.where(
                (df['tenkan_harmonic_momentum'] > 0) & (df['kijun_harmonic_momentum'] > 0),
                'bullish',
                np.where(
                    (df['tenkan_harmonic_momentum'] < 0) & (df['kijun_harmonic_momentum'] < 0),
                    'bearish',
                    'neutral'
                )
            )

            # Genuine trend change detection
            df['harmonic_momentum_strength'] = abs(
                df['tenkan_harmonic_momentum'] + df['kijun_harmonic_momentum']
            )

            df['genuine_trend_change'] = (
                df['harmonic_momentum_strength'] >= threshold
            )

            # Enhanced trend change validation with directional confirmation
            df['prev_harmonic_trend'] = df['harmonic_trend_alignment'].shift(1)
            df['trend_change_confirmed'] = (
                (df['harmonic_trend_alignment'] != df['prev_harmonic_trend']) &
                (df['harmonic_trend_alignment'] != 'neutral') &
                df['genuine_trend_change']
            )

            # Harmonic strength ratio (harmonic vs arithmetic)
            df['tenkan_harmonic_ratio'] = df['tenkan_harmonic_mean'] / (df['tenkan_arithmetic_mean'] + self.eps)
            df['kijun_harmonic_ratio'] = df['kijun_harmonic_mean'] / (df['kijun_arithmetic_mean'] + self.eps)

            # Quality score for harmonic analysis
            df['harmonic_quality_score'] = (
                df['composite_harmonic_divergence'] *
                df['harmonic_momentum_strength'] *
                np.where(df['trend_change_confirmed'], 1.2, 1.0)  # Bonus for confirmed changes
            )

            return df

        except Exception as e:
            self.logger.error(f"Error detecting genuine trend changes: {e}")
            return df

    def _detect_cloud_breakouts(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect cloud breakouts"""
        try:
            # Cloud Bull Breakout: Price breaks above cloud
            df['cloud_bull_breakout'] = (
                (df['prev_close'] <= df['prev_cloud_top'] + self.eps) &  # Was at/below cloud
                (df['close'] > df['cloud_top'] + self.eps)                # Now above cloud
            )

            # Cloud Bear Breakout: Price breaks below cloud
            df['cloud_bear_breakout'] = (
                (df['prev_close'] >= df['prev_cloud_bottom'] - self.eps) &  # Was at/above cloud
                (df['close'] < df['cloud_bottom'] - self.eps)                # Now below cloud
            )

            # Cloud breakout strength (distance from cloud)
            df['cloud_breakout_strength_bull'] = np.where(
                df['close'] > df['cloud_top'],
                (df['close'] - df['cloud_top']) / (df['close'] + self.eps),
                0
            )

            df['cloud_breakout_strength_bear'] = np.where(
                df['close'] < df['cloud_bottom'],
                (df['cloud_bottom'] - df['close']) / (df['close'] + self.eps),
                0
            )

            return df

        except Exception as e:
            self.logger.error(f"Error detecting cloud breakouts: {e}")
            return df

    def _calculate_signal_strength(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Calculate overall signal strength"""
        try:
            # Bull signal strength combines TK cross and cloud position
            df['bull_signal_strength'] = np.where(
                df['tk_bull_cross'] | df['cloud_bull_breakout'],
                (df['tk_cross_strength'] + df['cloud_breakout_strength_bull'] +
                 df['cloud_thickness'] / (df['close'] + self.eps)) / 3,
                0
            )

            # Bear signal strength combines TK cross and cloud position
            df['bear_signal_strength'] = np.where(
                df['tk_bear_cross'] | df['cloud_bear_breakout'],
                (df['tk_cross_strength'] + df['cloud_breakout_strength_bear'] +
                 df['cloud_thickness'] / (df['close'] + self.eps)) / 3,
                0
            )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return df

    def validate_chikou_span(self, df: pd.DataFrame, signal_type: str, chikou_periods: int = 26) -> bool:
        """
        Validate Chikou Span confirmation for signal

        Args:
            df: DataFrame with Ichimoku data
            signal_type: 'BULL' or 'BEAR'
            chikou_periods: Number of periods to look back for Chikou validation

        Returns:
            True if Chikou span confirms the signal
        """
        try:
            if len(df) < chikou_periods + 10:  # Need enough data
                return False

            # Get current Chikou value and historical price range
            latest_row = df.iloc[-1]
            chikou_value = latest_row.get('chikou_span', 0)

            # Look at price action chikou_periods ago (where Chikou is plotted)
            historical_start = max(0, len(df) - chikou_periods - 5)
            historical_end = len(df) - chikou_periods + 5
            historical_data = df.iloc[historical_start:historical_end]

            if len(historical_data) == 0:
                return False

            historical_high = historical_data['high'].max()
            historical_low = historical_data['low'].min()

            # For bull signals: Chikou should be above historical price action
            if signal_type == 'BULL':
                chikou_clear = chikou_value > historical_high + self.eps
                self.logger.debug(f"Bull Chikou validation: {chikou_value:.5f} vs high {historical_high:.5f} = {chikou_clear}")
                return chikou_clear

            # For bear signals: Chikou should be below historical price action
            elif signal_type == 'BEAR':
                chikou_clear = chikou_value < historical_low - self.eps
                self.logger.debug(f"Bear Chikou validation: {chikou_value:.5f} vs low {historical_low:.5f} = {chikou_clear}")
                return chikou_clear

            return False

        except Exception as e:
            self.logger.error(f"Error validating Chikou span: {e}")
            return False

    def get_cloud_thickness_ratio(self, latest_row: pd.Series) -> float:
        """Get cloud thickness as ratio of price (for signal filtering)"""
        try:
            cloud_thickness = latest_row.get('cloud_thickness', 0)
            price = latest_row.get('close', 1)  # Avoid division by zero
            return cloud_thickness / price if price > 0 else 0

        except Exception as e:
            self.logger.error(f"Error calculating cloud thickness ratio: {e}")
            return 0

    def is_cloud_thick_enough(self, latest_row: pd.Series, min_thickness_ratio: float = 0.001) -> bool:
        """Check if cloud is thick enough for reliable signals"""
        thickness_ratio = self.get_cloud_thickness_ratio(latest_row)
        return thickness_ratio >= min_thickness_ratio