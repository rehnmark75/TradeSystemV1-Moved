# core/strategies/macd_strategy.py
"""
CLEAN MACD STRATEGY - Rebuilt from scratch
Simple, reliable MACD crossover detection with swing proximity validation
Works identically in both live and backtest modes
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
from .helpers.swing_proximity_validator import SwingProximityValidator
from .helpers.smc_market_structure import SMCMarketStructure

try:
    from configdata import config
    from configdata.strategies import config_macd_strategy
except ImportError:
    from forex_scanner.configdata import config
    try:
        from forex_scanner.configdata.strategies import config_macd_strategy
    except ImportError:
        config_macd_strategy = None


class MACDStrategy(BaseStrategy):
    """
    Clean MACD Strategy - Simple and Reliable

    Features:
    - Standard MACD (12, 26, 9) crossover detection
    - Swing proximity validation (don't buy near swing highs, don't sell near swing lows)
    - ADX filter (only trade in trending markets)
    - ONE signal per crossover event
    - Works identically in live and backtest modes
    """

    def __init__(self, data_fetcher=None, backtest_mode: bool = False, epic: str = None,
                 timeframe: str = '15m', **kwargs):
        # Initialize
        self.name = 'macd'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(logging.INFO)

        # Basic config
        self.backtest_mode = backtest_mode
        self.epic = epic
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher
        self.price_adjuster = PriceAdjuster()

        # MACD parameters (standard settings)
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9

        # Validation thresholds (pair-specific ADX with fallback to global default)
        if config_macd_strategy and hasattr(config_macd_strategy, 'get_macd_min_adx'):
            self.min_adx = config_macd_strategy.get_macd_min_adx(epic) if epic else getattr(config_macd_strategy, 'MACD_MIN_ADX', 20)
        else:
            self.min_adx = getattr(config_macd_strategy, 'MACD_MIN_ADX', 20) if config_macd_strategy else 20

        self.min_confidence = 0.60  # 60% minimum confidence (lowered to allow more signals)

        epic_display = f"{epic} " if epic else ""
        self.logger.info(f"🎯 {epic_display}MACD ADX threshold set to: {self.min_adx} (pair-specific)")

        # EMA filter configuration (from config file)
        ema_filter_config = getattr(config_macd_strategy, 'MACD_EMA_FILTER', {}) if config_macd_strategy else {}
        self.ema_filter_enabled = ema_filter_config.get('enabled', False)  # Default: disabled
        self.ema_filter_period = ema_filter_config.get('ema_period', 50)  # Default: 50

        if self.ema_filter_enabled:
            self.logger.info(f"✅ EMA {self.ema_filter_period} filter ENABLED for trend alignment")
        else:
            self.logger.info(f"⚪ EMA filter DISABLED - all trending signals allowed")

        # Initialize swing validator
        swing_config = getattr(config_macd_strategy, 'MACD_SWING_VALIDATION', {}) if config_macd_strategy else {}
        if swing_config.get('enabled', True):
            smc_analyzer = SMCMarketStructure(logger=self.logger)
            self.swing_validator = SwingProximityValidator(
                smc_analyzer=smc_analyzer,
                config=swing_config,
                logger=self.logger
            )
        else:
            self.swing_validator = None

        # ADX Crossover Trigger Configuration (NEW FEATURE)
        self.adx_crossover_enabled = getattr(config_macd_strategy, 'MACD_ADX_CROSSOVER_ENABLED', True) if config_macd_strategy else True

        # Use same pair-specific ADX threshold for crossover trigger (aligned with main ADX threshold)
        self.adx_crossover_threshold = self.min_adx  # Use same threshold as main MACD signals for consistency

        self.adx_crossover_lookback = getattr(config_macd_strategy, 'MACD_ADX_CROSSOVER_LOOKBACK', 3) if config_macd_strategy else 3
        self.adx_min_histogram = getattr(config_macd_strategy, 'MACD_ADX_MIN_HISTOGRAM', 0.0001) if config_macd_strategy else 0.0001
        self.adx_require_expansion = getattr(config_macd_strategy, 'MACD_ADX_REQUIRE_EXPANSION', True) if config_macd_strategy else True
        self.adx_min_confidence = getattr(config_macd_strategy, 'MACD_ADX_MIN_CONFIDENCE', 0.50) if config_macd_strategy else 0.50

        if self.adx_crossover_enabled:
            self.logger.info(f"🚀 {epic_display}ADX Crossover Trigger ENABLED - ADX crosses {self.adx_crossover_threshold} with MACD alignment")
        else:
            self.logger.info(f"⚪ {epic_display}ADX Crossover Trigger DISABLED")

        # ATR-based SL/TP configuration
        self.stop_atr_multiplier = getattr(config_macd_strategy, 'MACD_STOP_LOSS_ATR_MULTIPLIER', 1.8) if config_macd_strategy else 1.8
        self.target_atr_multiplier = getattr(config_macd_strategy, 'MACD_TAKE_PROFIT_ATR_MULTIPLIER', 4.0) if config_macd_strategy else 4.0
        self.use_structure_stops = getattr(config_macd_strategy, 'MACD_USE_STRUCTURE_STOPS', True) if config_macd_strategy else True
        self.min_stop_pips = getattr(config_macd_strategy, 'MACD_MIN_STOP_DISTANCE_PIPS', 12.0) if config_macd_strategy else 12.0
        self.max_stop_pips = getattr(config_macd_strategy, 'MACD_MAX_STOP_DISTANCE_PIPS', 30.0) if config_macd_strategy else 30.0

        # Minimum histogram thresholds (pair-specific to prevent tiny crossovers)
        self.min_histogram_thresholds = getattr(config_macd_strategy, 'MACD_MIN_HISTOGRAM_THRESHOLDS', {
            'default': 0.00003,
            'EURJPY': 0.030, 'GBPJPY': 0.020, 'AUDJPY': 0.012,
            'NZDJPY': 0.010, 'USDJPY': 0.015, 'CADJPY': 0.012, 'CHFJPY': 0.015
        }) if config_macd_strategy else {}

        # Histogram expansion confirmation settings
        self.expansion_enabled = getattr(config_macd_strategy, 'MACD_EXPANSION_ENABLED', True) if config_macd_strategy else True
        self.expansion_window_bars = getattr(config_macd_strategy, 'MACD_EXPANSION_WINDOW_BARS', 3) if config_macd_strategy else 3
        self.expansion_allow_immediate = getattr(config_macd_strategy, 'MACD_EXPANSION_ALLOW_IMMEDIATE', True) if config_macd_strategy else True
        self.expansion_debug = getattr(config_macd_strategy, 'MACD_EXPANSION_DEBUG_LOGGING', True) if config_macd_strategy else True

        # ADX trend validation settings
        self.require_adx_rising = getattr(config_macd_strategy, 'MACD_REQUIRE_ADX_RISING', False) if config_macd_strategy else False
        self.adx_rising_lookback = getattr(config_macd_strategy, 'MACD_ADX_RISING_LOOKBACK', 2) if config_macd_strategy else 2
        self.adx_min_increase = getattr(config_macd_strategy, 'MACD_ADX_MIN_INCREASE', 0.5) if config_macd_strategy else 0.5

        # ADX catch-up window settings (allows ADX to reach threshold within N bars AFTER MACD threshold is met)
        self.adx_catchup_enabled = getattr(config_macd_strategy, 'MACD_ADX_CATCHUP_ENABLED', True) if config_macd_strategy else True
        self.adx_catchup_window_bars = getattr(config_macd_strategy, 'MACD_ADX_CATCHUP_WINDOW_BARS', 3) if config_macd_strategy else 3
        self.adx_catchup_allow_immediate = getattr(config_macd_strategy, 'MACD_ADX_ALLOW_IMMEDIATE', True) if config_macd_strategy else True

        self.logger.info(f"✅ Clean MACD Strategy initialized - ADX >= {self.min_adx}, Swing validation: {self.swing_validator is not None}")
        self.logger.info(f"📊 SL/TP: {self.stop_atr_multiplier}x ATR stop, {self.target_atr_multiplier}x ATR target (Structure stops: {self.use_structure_stops})")
        self.logger.info(f"📏 Histogram thresholds: Default={self.min_histogram_thresholds.get('default', 0.00003)}, JPY pairs scaled 50-100x higher")

        if self.expansion_enabled:
            self.logger.info(f"📊 Histogram Expansion Confirmation ENABLED: Wait up to {self.expansion_window_bars} bars for expansion")
        else:
            self.logger.info(f"📊 Histogram Expansion Confirmation DISABLED: Immediate crossover signals")

        if self.adx_catchup_enabled:
            self.logger.info(f"⏳ ADX Catch-up Window ENABLED: ADX can reach {self.min_adx} within {self.adx_catchup_window_bars} bars AFTER MACD threshold met")
        else:
            self.logger.info(f"⏳ ADX Catch-up Window DISABLED: ADX must be >= {self.min_adx} immediately")

        if self.require_adx_rising:
            self.logger.info(f"📈 ADX Rising Validation ENABLED: Require ADX increasing over {self.adx_rising_lookback} bars (min +{self.adx_min_increase})")
        else:
            self.logger.info(f"📈 ADX Rising Validation DISABLED")

        # Ranging Market Penalty Settings (NEW - Additional Quality Filter)
        self.ranging_penalty_enabled = getattr(config_macd_strategy, 'MACD_RANGING_PENALTY_ENABLED', True) if config_macd_strategy else True

        if self.ranging_penalty_enabled:
            ranging_thresholds = getattr(config_macd_strategy, 'MACD_RANGING_THRESHOLDS', {}) if config_macd_strategy else {}
            self.ranging_high_threshold = ranging_thresholds.get('high_ranging', 0.55)
            self.ranging_moderate_threshold = ranging_thresholds.get('moderate_ranging', 0.45)

            ranging_penalties = getattr(config_macd_strategy, 'MACD_RANGING_PENALTIES', {}) if config_macd_strategy else {}
            self.ranging_high_penalty = ranging_penalties.get('high_ranging_penalty', -0.15)
            self.ranging_moderate_penalty = ranging_penalties.get('moderate_ranging_penalty', -0.08)

            self.log_ranging_analysis = getattr(config_macd_strategy, 'MACD_LOG_RANGING_ANALYSIS', True) if config_macd_strategy else True

            self.logger.info(f"📊 Ranging Market Penalty ENABLED:")
            self.logger.info(f"   High Ranging (>= {self.ranging_high_threshold:.0%}): {self.ranging_high_penalty:+.0%} confidence")
            self.logger.info(f"   Moderate Ranging (>= {self.ranging_moderate_threshold:.0%}): {self.ranging_moderate_penalty:+.0%} confidence")
        else:
            self.logger.info(f"⚪ Ranging Market Penalty DISABLED")

        # Market Bias Conflict Detection Settings (NEW - Market Direction Alignment)
        self.market_bias_filter_enabled = getattr(config_macd_strategy, 'MACD_ENABLE_MARKET_BIAS_FILTER', True) if config_macd_strategy else True

        if self.market_bias_filter_enabled:
            self.market_bias_conflict_penalty = getattr(config_macd_strategy, 'MACD_MARKET_BIAS_CONFLICT_PENALTY', -0.10) if config_macd_strategy else -0.10
            self.market_bias_strong_consensus_threshold = getattr(config_macd_strategy, 'MACD_STRONG_CONSENSUS_THRESHOLD', 0.8) if config_macd_strategy else 0.8
            self.market_bias_strong_consensus_penalty = getattr(config_macd_strategy, 'MACD_STRONG_CONSENSUS_PENALTY', -0.15) if config_macd_strategy else -0.15

            self.logger.info(f"📊 Market Bias Conflict Detection ENABLED:")
            self.logger.info(f"   Normal conflict penalty: {self.market_bias_conflict_penalty:+.0%}")
            self.logger.info(f"   Strong consensus (>{self.market_bias_strong_consensus_threshold:.0%}) penalty: {self.market_bias_strong_consensus_penalty:+.0%}")
        else:
            self.logger.info(f"⚪ Market Bias Conflict Detection DISABLED")

        # KAMA Efficiency Validation Settings (NEW - Non-Blocking Quality Filter)
        self.use_kama_efficiency = getattr(config_macd_strategy, 'MACD_USE_KAMA_EFFICIENCY', True) if config_macd_strategy else True

        if self.use_kama_efficiency:
            # Load thresholds
            kama_thresholds = getattr(config_macd_strategy, 'MACD_KAMA_EFFICIENCY_THRESHOLDS', {}) if config_macd_strategy else {}
            self.kama_excellent_threshold = kama_thresholds.get('excellent', 0.50)
            self.kama_good_threshold = kama_thresholds.get('good', 0.30)
            self.kama_acceptable_threshold = kama_thresholds.get('acceptable', 0.15)
            self.kama_poor_threshold = kama_thresholds.get('poor', 0.10)
            self.kama_very_poor_threshold = kama_thresholds.get('very_poor', 0.05)

            # Load confidence adjustments
            kama_adjustments = getattr(config_macd_strategy, 'MACD_KAMA_CONFIDENCE_ADJUSTMENTS', {}) if config_macd_strategy else {}
            self.kama_excellent_boost = kama_adjustments.get('excellent_boost', 0.10)
            self.kama_good_boost = kama_adjustments.get('good_boost', 0.05)
            self.kama_acceptable_neutral = kama_adjustments.get('acceptable_neutral', 0.0)
            self.kama_poor_penalty = kama_adjustments.get('poor_penalty', -0.05)
            self.kama_very_poor_penalty = kama_adjustments.get('very_poor_penalty', -0.10)

            # Optional trend alignment check
            self.require_kama_trend_alignment = getattr(config_macd_strategy, 'MACD_REQUIRE_KAMA_TREND_ALIGNMENT', False) if config_macd_strategy else False
            self.kama_trend_conflict_penalty = getattr(config_macd_strategy, 'MACD_KAMA_TREND_CONFLICT_PENALTY', -0.08) if config_macd_strategy else -0.08

            # Logging
            self.log_kama_efficiency = getattr(config_macd_strategy, 'MACD_LOG_KAMA_EFFICIENCY', True) if config_macd_strategy else True

            self.logger.info(f"🎯 KAMA Efficiency Validation ENABLED:")
            self.logger.info(f"   Excellent (>= {self.kama_excellent_threshold:.2f}): {self.kama_excellent_boost:+.0%} confidence")
            self.logger.info(f"   Good (>= {self.kama_good_threshold:.2f}): {self.kama_good_boost:+.0%} confidence")
            self.logger.info(f"   Acceptable (>= {self.kama_acceptable_threshold:.2f}): {self.kama_acceptable_neutral:+.0%} confidence")
            self.logger.info(f"   Poor (>= {self.kama_poor_threshold:.2f}): {self.kama_poor_penalty:+.0%} confidence")
            self.logger.info(f"   Very Poor (< {self.kama_poor_threshold:.2f}): {self.kama_very_poor_penalty:+.0%} confidence")
            if self.require_kama_trend_alignment:
                self.logger.info(f"   KAMA Trend Alignment REQUIRED (conflict penalty: {self.kama_trend_conflict_penalty:+.0%})")
        else:
            self.logger.info(f"⚪ KAMA Efficiency Validation DISABLED")

        self.logger.info(f"🔧 CRITICAL DEBUG: Using REBUILT MACD with expansion + ADX rising validation (v3.0)")

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        df = df.copy()

        # Calculate EMAs
        fast_ema = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=self.slow_period, adjust=False).mean()

        # MACD line and signal line
        df['macd_line'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd_line'].ewm(span=self.signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']

        return df

    def _ensure_kama_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure KAMA efficiency_ratio is available in dataframe
        If not present (backtest mode), calculate a simple version
        """
        if 'efficiency_ratio' in df.columns and not df['efficiency_ratio'].isna().all():
            return df  # Already has KAMA efficiency

        try:
            # Calculate simple KAMA efficiency ratio
            # ER = Net Change / Total Change
            period = 10  # KAMA period

            # Net change (absolute distance from start to end)
            net_change = abs(df['close'].diff(period))

            # Total change (sum of absolute price movements)
            total_change = df['close'].diff().abs().rolling(period).sum()

            # Efficiency Ratio = Net / Total (0 to 1 scale)
            df['efficiency_ratio'] = net_change / total_change

            # Fill NaN values with small default (low efficiency)
            df['efficiency_ratio'] = df['efficiency_ratio'].fillna(0.1)

            # Cap at 1.0
            df['efficiency_ratio'] = df['efficiency_ratio'].clip(upper=1.0)

            self.logger.debug(f"✅ Calculated KAMA efficiency ratio for MACD strategy")

        except Exception as e:
            self.logger.warning(f"Failed to calculate KAMA efficiency: {e}")
            df['efficiency_ratio'] = 0.1  # Default to low efficiency

        return df

    def detect_crossover(self, df: pd.DataFrame, epic: str = None) -> pd.DataFrame:
        """
        Detect MACD histogram crossovers with expansion confirmation and ADX trend validation

        Features:
        1. Detects zero-line crossovers
        2. Waits for histogram expansion (configurable)
        3. Validates ADX is rising (strengthening trend)

        Args:
            df: DataFrame with MACD and ADX data
            epic: Epic code for pair-specific thresholds

        Returns:
            DataFrame with 'bull_crossover' and 'bear_crossover' columns
        """
        df = df.copy()

        # Ensure ADX columns exist for trend validation
        if 'adx' not in df.columns:
            df = self._calculate_adx(df)

        # Shift histogram to get previous value
        df['histogram_prev'] = df['macd_histogram'].shift(1)

        # Detect INITIAL zero-line crossovers
        df['bull_cross_initial'] = (df['macd_histogram'] > 0) & (df['histogram_prev'] <= 0)
        df['bear_cross_initial'] = (df['macd_histogram'] < 0) & (df['histogram_prev'] >= 0)

        # If expansion disabled, use immediate crossover (old behavior)
        if not self.expansion_enabled:
            df['bull_crossover'] = df['bull_cross_initial']
            df['bear_crossover'] = df['bear_cross_initial']
            df.drop(['histogram_prev', 'bull_cross_initial', 'bear_cross_initial'],
                    axis=1, inplace=True, errors='ignore')
            return df

        # EXPANSION CONFIRMATION LOGIC
        # Get pair-specific threshold
        if epic:
            min_histogram = self._get_min_histogram(epic)
        else:
            min_histogram = self.min_histogram_thresholds.get('default', 0.00003)

        # Create "crossover window" - marks bars within N bars after initial cross
        bull_window = df['bull_cross_initial'].copy()
        bear_window = df['bear_cross_initial'].copy()

        for i in range(1, self.expansion_window_bars + 1):
            bull_window = bull_window | df['bull_cross_initial'].shift(i).fillna(False)
            bear_window = bear_window | df['bear_cross_initial'].shift(i).fillna(False)

        df['bull_window'] = bull_window
        df['bear_window'] = bear_window

        # BULL EXPANSION: In window AND histogram >= threshold (MACD requirement - must be met first)
        df['bull_expansion_met'] = df['bull_window'] & (df['macd_histogram'] >= min_histogram)

        # BEAR EXPANSION: In window AND |histogram| >= threshold (MACD requirement - must be met first)
        df['bear_expansion_met'] = df['bear_window'] & (abs(df['macd_histogram']) >= min_histogram)

        # ADX CATCH-UP WINDOW: Check if ADX reaches threshold within window AFTER MACD threshold is met
        # CRITICAL: MACD expansion_met must be True first, then we check for ADX within the same window
        if self.adx_catchup_enabled:
            # For each bar where MACD expansion is met, check forward window for ADX >= threshold
            df['adx_met_in_window'] = False

            # Get indices where MACD expansion is met (these are our anchor points)
            bull_expansion_indices = df[df['bull_expansion_met']].index
            bear_expansion_indices = df[df['bear_expansion_met']].index

            # For each MACD expansion point, check if ADX >= threshold in the catchup window
            for idx in bull_expansion_indices:
                idx_loc = df.index.get_loc(idx)
                # Check current bar and next N bars for ADX >= threshold
                for offset in range(self.adx_catchup_window_bars + 1):
                    check_loc = idx_loc + offset
                    if check_loc < len(df):
                        check_idx = df.index[check_loc]
                        adx_value = df.loc[check_idx, 'adx']
                        if not pd.isna(adx_value) and adx_value >= self.min_adx:
                            # Mark all bars in the window from MACD expansion to ADX confirmation
                            for mark_offset in range(offset + 1):
                                mark_loc = idx_loc + mark_offset
                                if mark_loc < len(df):
                                    df.iloc[mark_loc, df.columns.get_loc('adx_met_in_window')] = True
                            break

            for idx in bear_expansion_indices:
                idx_loc = df.index.get_loc(idx)
                for offset in range(self.adx_catchup_window_bars + 1):
                    check_loc = idx_loc + offset
                    if check_loc < len(df):
                        check_idx = df.index[check_loc]
                        adx_value = df.loc[check_idx, 'adx']
                        if not pd.isna(adx_value) and adx_value >= self.min_adx:
                            for mark_offset in range(offset + 1):
                                mark_loc = idx_loc + mark_offset
                                if mark_loc < len(df):
                                    df.iloc[mark_loc, df.columns.get_loc('adx_met_in_window')] = True
                            break

            # Apply ADX catch-up filter: MACD expansion met AND ADX met in window
            df['bull_expansion_met'] = df['bull_expansion_met'] & df['adx_met_in_window']
            df['bear_expansion_met'] = df['bear_expansion_met'] & df['adx_met_in_window']

        # Add ADX trend validation if enabled
        if self.require_adx_rising:
            # Calculate ADX change over lookback period
            df['adx_prev'] = df['adx'].shift(self.adx_rising_lookback)
            df['adx_change'] = df['adx'] - df['adx_prev']
            df['adx_is_rising'] = df['adx_change'] >= self.adx_min_increase

            # Apply ADX filter to expansion conditions
            df['bull_expansion_met'] = df['bull_expansion_met'] & df['adx_is_rising']
            df['bear_expansion_met'] = df['bear_expansion_met'] & df['adx_is_rising']

        # Calculate "bars since crossover" for logging
        df['bull_bars_since_cross'] = 0
        df['bear_bars_since_cross'] = 0

        for idx in df[df['bull_cross_initial']].index:
            for offset in range(self.expansion_window_bars + 1):
                try:
                    future_idx = df.index.get_loc(idx) + offset
                    if future_idx < len(df):
                        df.iloc[future_idx, df.columns.get_loc('bull_bars_since_cross')] = offset
                except:
                    pass

        for idx in df[df['bear_cross_initial']].index:
            for offset in range(self.expansion_window_bars + 1):
                try:
                    future_idx = df.index.get_loc(idx) + offset
                    if future_idx < len(df):
                        df.iloc[future_idx, df.columns.get_loc('bear_bars_since_cross')] = offset
                except:
                    pass

        # TRIGGER SIGNAL: First bar in window where expansion AND ADX conditions are met
        df['bull_crossover'] = (
            df['bull_expansion_met'] &
            (~df['bull_expansion_met'].shift(1).fillna(False))  # First time all conditions met
        )

        df['bear_crossover'] = (
            df['bear_expansion_met'] &
            (~df['bear_expansion_met'].shift(1).fillna(False))
        )

        # DETAILED LOGGING for latest bar (continued in next edit due to size)
        if self.expansion_debug and len(df) > 0:
            self._log_expansion_status(df, min_histogram, epic)

        # Clean up intermediate columns
        cleanup_cols = ['histogram_prev', 'bull_cross_initial', 'bear_cross_initial',
                        'bull_window', 'bear_window', 'bull_expansion_met', 'bear_expansion_met',
                        'bull_bars_since_cross', 'bear_bars_since_cross']
        if self.require_adx_rising:
            cleanup_cols.extend(['adx_prev', 'adx_change', 'adx_is_rising'])

        df.drop(cleanup_cols, axis=1, inplace=True, errors='ignore')

        return df

    def detect_adx_crossover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect ADX crossovers above threshold (NEW FEATURE)

        Triggers when:
        1. ADX crosses above threshold (e.g., 25)
        2. MACD histogram is already in correct direction (green for BULL, red for BEAR)
        3. ADX has been rising for lookback period (prevents whipsaws)
        4. MACD histogram is expanding (optional, prevents catching tops/bottoms)

        This catches trend acceleration earlier than MACD histogram crossover
        """
        df = df.copy()

        # Ensure histogram_prev exists (should be created by detect_crossover)
        if 'histogram_prev' not in df.columns:
            df['histogram_prev'] = df['macd_histogram'].shift(1)

        # Shift ADX to get previous values
        df['adx_prev'] = df['adx'].shift(1)

        # Detect ADX crossing above threshold
        df['adx_cross_up'] = (df['adx'] > self.adx_crossover_threshold) & (df['adx_prev'] <= self.adx_crossover_threshold)

        # Initialize ADX crossover signals as False
        df['bull_adx_crossover'] = False
        df['bear_adx_crossover'] = False

        # For each ADX crossover, check if MACD histogram is aligned
        for idx in df[df['adx_cross_up']].index:
            try:
                # Get current bar data
                histogram = df.loc[idx, 'macd_histogram']

                # Check minimum histogram magnitude
                if abs(histogram) < self.adx_min_histogram:
                    self.logger.debug(f"ADX cross at {idx}: histogram too small ({abs(histogram):.6f} < {self.adx_min_histogram:.6f})")
                    continue

                # Check if ADX has been rising (lookback period)
                if self.adx_crossover_lookback > 0:
                    # Get lookback bars
                    lookback_start = max(0, df.index.get_loc(idx) - self.adx_crossover_lookback)
                    lookback_adx = df.iloc[lookback_start:df.index.get_loc(idx) + 1]['adx']

                    # Check if ADX is monotonically increasing
                    adx_diffs = lookback_adx.diff().dropna()
                    if not all(adx_diffs > 0):
                        self.logger.debug(f"ADX cross at {idx}: ADX not consistently rising (diffs: {adx_diffs.tolist()})")
                        continue

                # Check if MACD histogram is expanding (optional)
                if self.adx_require_expansion:
                    histogram_prev = df.loc[idx, 'histogram_prev']
                    if abs(histogram) <= abs(histogram_prev):
                        self.logger.debug(f"ADX cross at {idx}: histogram not expanding ({abs(histogram):.6f} <= {abs(histogram_prev):.6f})")
                        continue

                # Determine signal type based on MACD histogram color
                if histogram > 0:
                    # MACD green = BULL signal
                    df.loc[idx, 'bull_adx_crossover'] = True
                    self.logger.info(f"✅ ADX BULL crossover detected at {idx}: ADX={df.loc[idx, 'adx']:.2f}, histogram={histogram:.6f}")
                elif histogram < 0:
                    # MACD red = BEAR signal
                    df.loc[idx, 'bear_adx_crossover'] = True
                    self.logger.info(f"✅ ADX BEAR crossover detected at {idx}: ADX={df.loc[idx, 'adx']:.2f}, histogram={histogram:.6f}")

            except Exception as e:
                self.logger.debug(f"Error processing ADX crossover at {idx}: {e}")
                continue

        return df

    def validate_adx_signal(self, row: pd.Series, signal_type: str, epic: str = None) -> bool:
        """
        Validate ADX crossover signal meets quality requirements

        Simplified validation for ADX crossover signals (less strict than MACD crossover)
        ADX crossing above threshold already indicates trend strength
        """
        # Format epic for logging
        epic_display = f"[{epic}] " if epic else ""

        # Check histogram is in correct direction (already checked in detect_adx_crossover, but double-check)
        histogram = row.get('macd_histogram', 0)
        if signal_type == 'BULL' and histogram <= 0:
            self.logger.debug(f"❌ {epic_display}ADX BULL rejected: negative histogram {histogram:.6f}")
            return False
        if signal_type == 'BEAR' and histogram >= 0:
            self.logger.debug(f"❌ {epic_display}ADX BEAR rejected: positive histogram {histogram:.6f}")
            return False

        # Check MACD line position (same filter as regular signals)
        macd_line = row.get('macd_line', 0)
        if signal_type == 'BEAR' and macd_line > 0.05:
            self.logger.info(f"❌ {epic_display}ADX BEAR rejected: MACD line too positive {macd_line:.6f}")
            return False
        if signal_type == 'BULL' and macd_line < -0.05:
            self.logger.info(f"❌ {epic_display}ADX BULL rejected: MACD line too negative {macd_line:.6f}")
            return False

        # Check RSI (optional - avoid extreme zones)
        rsi = row.get('rsi', 50)
        if signal_type == 'BULL' and rsi > 70:
            self.logger.debug(f"❌ {epic_display}ADX BULL rejected: RSI {rsi:.1f} overbought")
            return False
        if signal_type == 'BEAR' and rsi < 30:
            self.logger.debug(f"❌ {epic_display}ADX BEAR rejected: RSI {rsi:.1f} oversold")
            return False

        # EMA filter (if enabled)
        if self.ema_filter_enabled:
            price = row.get('close', 0)
            ema_col = f'ema_{self.ema_filter_period}'
            ema_value = row.get(ema_col, 0)

            if pd.isna(ema_value) or pd.isna(price) or ema_value <= 0 or price <= 0:
                self.logger.info(f"❌ {epic_display}ADX {signal_type} rejected: Invalid EMA or price")
                return False

            if signal_type == 'BULL' and price < ema_value:
                self.logger.debug(f"❌ {epic_display}ADX BULL rejected: price below EMA{self.ema_filter_period}")
                return False
            if signal_type == 'BEAR' and price > ema_value:
                self.logger.debug(f"❌ {epic_display}ADX BEAR rejected: price above EMA{self.ema_filter_period}")
                return False

        self.logger.info(f"✅ {epic_display}ADX {signal_type} signal validated")
        return True

    def validate_signal(self, row: pd.Series, signal_type: str, epic: str = None) -> bool:
        """
        Validate signal meets quality requirements

        Checks:
        1. ADX >= 30 (strong trend)
        2. Histogram in correct direction AND magnitude
        3. RSI in reasonable zone
        4. Dynamic EMA trend filter (ADX-based volatility adaptation)
        """
        # Format epic for logging
        epic_display = f"[{epic}] " if epic else ""

        # Check ADX (handle NaN values)
        # NOTE: If ADX catch-up window is enabled, ADX validation is already done in expansion window logic
        # This check is only for immediate validation when catch-up is disabled
        if not self.adx_catchup_enabled:
            adx = row.get('adx', 0)

            # CRITICAL: Check for NaN or invalid ADX values
            if pd.isna(adx) or adx <= 0:
                self.logger.info(f"❌ {epic_display}{signal_type} rejected: Invalid ADX ({adx})")
                return False

            if adx < self.min_adx:
                self.logger.info(f"❌ {epic_display}{signal_type} rejected: ADX {adx:.1f} < {self.min_adx}")
                return False

            # Log ADX for signals that pass (for debugging)
            self.logger.info(f"✅ {epic_display}{signal_type} passed ADX filter: ADX={adx:.1f} (min={self.min_adx})")
        else:
            # ADX catch-up enabled - validation already done in expansion window
            adx = row.get('adx', 0)
            self.logger.info(f"⏳ {epic_display}{signal_type} using ADX catch-up window (current ADX={adx:.1f}, validated in expansion window)")

        # ENABLED: Histogram magnitude check - ensures signal has sufficient momentum
        histogram = row.get('macd_histogram', 0)
        if epic:
            min_histogram = self._get_min_histogram(epic)
            if abs(histogram) < min_histogram:
                self.logger.info(f"❌ {epic_display}{signal_type} rejected: Histogram {abs(histogram):.6f} too small (min={min_histogram:.6f} for {epic})")
                return False
            else:
                self.logger.info(f"✅ {epic_display}{signal_type} histogram magnitude OK: {abs(histogram):.6f} >= {min_histogram:.6f}")

        # Check histogram direction
        histogram = row.get('macd_histogram', 0)
        if signal_type == 'BULL' and histogram <= 0:
            self.logger.debug(f"❌ {epic_display}BULL rejected: negative histogram {histogram:.6f}")
            return False
        if signal_type == 'BEAR' and histogram >= 0:
            self.logger.debug(f"❌ {epic_display}BEAR rejected: positive histogram {histogram:.6f}")
            return False

        # CRITICAL FIX: Check MACD line position (zero line filter)
        # BULL signals should have MACD line trending upward (can be below zero if reversing)
        # BEAR signals should have MACD line trending downward (can be above zero if reversing)
        # But we should avoid signals when MACD line contradicts direction strongly
        macd_line = row.get('macd_line', 0)

        # For BEAR signals, if MACD line is strongly positive (>0.1 for forex), it's a bad signal
        # This prevents selling during strong uptrends just because histogram dipped slightly
        if signal_type == 'BEAR' and macd_line > 0.05:
            self.logger.info(f"❌ {epic_display}BEAR rejected: MACD line too positive {macd_line:.6f} (still in bullish territory)")
            return False

        # For BULL signals, if MACD line is strongly negative (<-0.1), it's a bad signal
        # This prevents buying during strong downtrends just because histogram rose slightly
        if signal_type == 'BULL' and macd_line < -0.05:
            self.logger.info(f"❌ {epic_display}BULL rejected: MACD line too negative {macd_line:.6f} (still in bearish territory)")
            return False

        # Check RSI (optional - just avoid extreme zones)
        rsi = row.get('rsi', 50)
        if signal_type == 'BULL' and rsi > 70:
            self.logger.debug(f"❌ {epic_display}BULL rejected: RSI {rsi:.1f} overbought")
            return False
        if signal_type == 'BEAR' and rsi < 30:
            self.logger.debug(f"❌ {epic_display}BEAR rejected: RSI {rsi:.1f} oversold")
            return False

        # EMA FILTER (configurable via config file)
        if self.ema_filter_enabled:
            price = row.get('close', 0)
            ema_col = f'ema_{self.ema_filter_period}'
            ema_value = row.get(ema_col, 0)

            # Validate EMA and price are valid numbers
            if pd.isna(ema_value) or pd.isna(price) or ema_value <= 0 or price <= 0:
                self.logger.info(f"❌ {epic_display}{signal_type} rejected: Invalid EMA{self.ema_filter_period} or price (price={price}, EMA={ema_value})")
                return False

            # Apply EMA trend filter
            if signal_type == 'BULL' and price < ema_value:
                self.logger.debug(f"❌ {epic_display}BULL rejected: price {price:.5f} below EMA{self.ema_filter_period} {ema_value:.5f} (ADX={adx:.1f})")
                return False
            if signal_type == 'BEAR' and price > ema_value:
                self.logger.debug(f"❌ {epic_display}BEAR rejected: price {price:.5f} above EMA{self.ema_filter_period} {ema_value:.5f} (ADX={adx:.1f})")
                return False

            self.logger.debug(f"✅ {epic_display}{signal_type} passed EMA{self.ema_filter_period} filter: price={price:.5f}, EMA={ema_value:.5f} (ADX={adx:.1f})")
        else:
            self.logger.debug(f"✅ {epic_display}{signal_type} passed (no EMA filter) - ADX={adx:.1f}")

        return True

    def calculate_confidence(self, row: pd.Series, signal_type: str) -> float:
        """
        Calculate signal confidence (simple, transparent)

        Base: 50%
        + ADX strength: up to +20%
        + Histogram strength: up to +15%
        + RSI confluence: up to +15%
        """
        confidence = 0.50  # Base

        # ADX boost
        adx = row.get('adx', 0)
        if adx >= 40:
            confidence += 0.20
        elif adx >= 35:
            confidence += 0.15
        elif adx >= 30:
            confidence += 0.10

        # Histogram strength boost
        histogram = abs(row.get('macd_histogram', 0))
        if histogram > 0.002:
            confidence += 0.15
        elif histogram > 0.001:
            confidence += 0.10
        elif histogram > 0.0005:
            confidence += 0.05

        # RSI confluence boost
        rsi = row.get('rsi', 50)
        if signal_type == 'BULL' and rsi < 40:
            confidence += 0.15
        elif signal_type == 'BULL' and rsi < 50:
            confidence += 0.10
        elif signal_type == 'BEAR' and rsi > 60:
            confidence += 0.15
        elif signal_type == 'BEAR' and rsi > 50:
            confidence += 0.10

        return min(0.95, confidence)

    def create_signal(self, row: pd.Series, signal_type: str, epic: str, timeframe: str, trigger_type: str = 'macd') -> Optional[Dict]:
        """
        Create signal dictionary

        Args:
            trigger_type: 'macd' for histogram crossover, 'adx' for ADX crossover trigger
        """
        # Format epic for logging
        epic_display = f"[{epic}] " if epic else ""

        try:
            # Calculate base confidence
            confidence = self.calculate_confidence(row, signal_type)

            # KAMA Efficiency Validation (NEW - Non-Blocking Quality Filter)
            kama_adjustment = 0.0
            kama_quality = "N/A"
            kama_efficiency_value = None

            if self.use_kama_efficiency:
                # Note: We pass the row index, but since row is a Series, we need to handle this
                # For now, we'll extract the data directly from the row
                kama_efficiency_value = row.get('efficiency_ratio', None)
                kama_trend = row.get('kama_trend', 0)

                if kama_efficiency_value is not None and not pd.isna(kama_efficiency_value):
                    # Determine confidence adjustment based on efficiency tier
                    if kama_efficiency_value >= self.kama_excellent_threshold:
                        kama_adjustment = self.kama_excellent_boost
                        kama_quality = "EXCELLENT"
                        emoji = "⭐⭐⭐⭐⭐"
                    elif kama_efficiency_value >= self.kama_good_threshold:
                        kama_adjustment = self.kama_good_boost
                        kama_quality = "GOOD"
                        emoji = "⭐⭐⭐"
                    elif kama_efficiency_value >= self.kama_acceptable_threshold:
                        kama_adjustment = self.kama_acceptable_neutral
                        kama_quality = "ACCEPTABLE"
                        emoji = "⭐⭐"
                    elif kama_efficiency_value >= self.kama_poor_threshold:
                        kama_adjustment = self.kama_poor_penalty
                        kama_quality = "POOR"
                        emoji = "⭐"
                    else:
                        kama_adjustment = self.kama_very_poor_penalty
                        kama_quality = "VERY POOR"
                        emoji = "❌"

                    # Optional: Check KAMA trend alignment
                    trend_note = ""
                    if self.require_kama_trend_alignment:
                        trend_aligned = (
                            (signal_type == 'BULL' and kama_trend > 0) or
                            (signal_type == 'BEAR' and kama_trend < 0)
                        )
                        if not trend_aligned:
                            kama_adjustment += self.kama_trend_conflict_penalty
                            trend_note = f" (TREND CONFLICT: kama_trend={kama_trend:+.1f})"
                            kama_quality += " + CONFLICT"

                    # Log KAMA analysis
                    if self.log_kama_efficiency:
                        self.logger.info(
                            f"🎯 {epic_display}KAMA Efficiency: {kama_efficiency_value:.3f} {emoji} {kama_quality}{trend_note} "
                            f"→ Confidence adjustment: {kama_adjustment:+.1%}"
                        )

                    # Apply KAMA adjustment to confidence
                    confidence += kama_adjustment

            # Ranging Market Penalty (NEW - Additional Quality Filter)
            ranging_adjustment = 0.0
            ranging_quality = "N/A"
            ranging_score_value = None

            if self.ranging_penalty_enabled:
                # Try to get ranging score from row (if available from market intelligence)
                ranging_score_value = row.get('ranging_score', None)

                # FALLBACK: If ranging score not available (e.g., in backtest mode),
                # calculate a simple ranging indicator using KAMA efficiency
                if ranging_score_value is None or pd.isna(ranging_score_value):
                    if kama_efficiency_value is not None and not pd.isna(kama_efficiency_value):
                        # Low KAMA efficiency suggests ranging market
                        # Convert KAMA efficiency (0-1) to ranging score (0-1)
                        # Lower efficiency = higher ranging score
                        ranging_score_value = 1.0 - kama_efficiency_value

                        if self.log_ranging_analysis:
                            self.logger.debug(
                                f"📊 {epic_display}Ranging score calculated from KAMA efficiency "
                                f"(1.0 - {kama_efficiency_value:.3f} = {ranging_score_value:.3f})"
                            )

                if ranging_score_value is not None and not pd.isna(ranging_score_value):
                    # Determine ranging penalty based on ranging score
                    if ranging_score_value >= self.ranging_high_threshold:
                        ranging_adjustment = self.ranging_high_penalty
                        ranging_quality = "HIGH RANGING"
                        emoji = "🔀"
                    elif ranging_score_value >= self.ranging_moderate_threshold:
                        ranging_adjustment = self.ranging_moderate_penalty
                        ranging_quality = "MODERATE RANGING"
                        emoji = "↔️"
                    else:
                        ranging_adjustment = 0.0
                        ranging_quality = "LOW RANGING"
                        emoji = "➡️"

                    # Log ranging analysis
                    if self.log_ranging_analysis:
                        self.logger.info(
                            f"📊 {epic_display}Ranging Score: {ranging_score_value:.1%} {emoji} {ranging_quality} "
                            f"→ Confidence adjustment: {ranging_adjustment:+.1%}"
                        )

                    # Apply ranging adjustment to confidence
                    confidence += ranging_adjustment

            # Market Bias Conflict Detection (NEW - Alignment with broader market direction)
            bias_adjustment = 0.0
            bias_conflict = False
            market_bias = None
            directional_consensus = None

            if self.market_bias_filter_enabled:
                # Try to extract market bias from row (populated by market intelligence)
                market_bias = row.get('market_bias', None)
                directional_consensus = row.get('directional_consensus', None)

                if market_bias:
                    # Check for directional conflict
                    is_bullish_signal = signal_type == 'BULL'
                    is_bearish_signal = signal_type == 'BEAR'
                    market_is_bullish = market_bias.lower() in ['bullish', 'bull']
                    market_is_bearish = market_bias.lower() in ['bearish', 'bear']

                    # Detect conflict: BULL signal in bearish market or BEAR signal in bullish market
                    if (is_bullish_signal and market_is_bearish) or (is_bearish_signal and market_is_bullish):
                        bias_conflict = True

                        # Determine penalty based on consensus strength
                        if directional_consensus and directional_consensus > self.market_bias_strong_consensus_threshold:
                            # Strong consensus against signal = larger penalty
                            bias_adjustment = self.market_bias_strong_consensus_penalty
                            self.logger.warning(
                                f"⚠️ {epic_display}STRONG market bias conflict: {signal_type} signal in {market_bias} market "
                                f"(consensus: {directional_consensus:.1%}) → Confidence penalty: {bias_adjustment:+.1%}"
                            )
                        else:
                            # Normal conflict penalty
                            bias_adjustment = self.market_bias_conflict_penalty
                            self.logger.info(
                                f"⚠️ {epic_display}Market bias conflict: {signal_type} signal in {market_bias} market "
                                f"→ Confidence penalty: {bias_adjustment:+.1%}"
                            )

                        # Apply bias conflict penalty
                        confidence += bias_adjustment
                    else:
                        self.logger.debug(
                            f"✅ {epic_display}Market bias aligned: {signal_type} signal with {market_bias} market bias"
                        )

            # Use different minimum confidence for ADX crossover signals
            min_conf = self.adx_min_confidence if trigger_type == 'adx' else self.min_confidence

            if confidence < min_conf:
                self.logger.debug(f"❌ {epic_display}{signal_type} rejected: confidence {confidence:.1%} < {min_conf:.1%}")
                return None

            # Record which EMA filter was used (if any)
            if self.ema_filter_enabled:
                ema_filter_used = f'EMA{self.ema_filter_period}'
            else:
                ema_filter_used = 'None'

            # Create signal
            signal = {
                'epic': epic,
                'direction': signal_type,
                'signal_type': signal_type,  # For validator compatibility
                'strategy': self.name,
                'timeframe': timeframe,
                'price': row.get('close', 0),
                'confidence': confidence,
                'confidence_score': confidence,  # For validator compatibility
                'timestamp': row.name if hasattr(row, 'name') else datetime.now(),
                'trigger_type': trigger_type,  # 'macd' or 'adx' - indicates which trigger fired

                # MACD data
                'macd_line': row.get('macd_line', 0),
                'macd_signal': row.get('macd_signal', 0),
                'macd_histogram': row.get('macd_histogram', 0),

                # Indicators
                'adx': row.get('adx', 0),
                'rsi': row.get('rsi', 50),

                # EMA trend data (for validator and logging)
                'ema_50': row.get('ema_50', 0),
                'ema_100': row.get('ema_100', 0),
                'ema_200': row.get('ema_200', 0),
                'ema_filter_used': ema_filter_used,  # Which EMA was used for filtering

                # KAMA efficiency data (NEW - for analysis and tracking)
                'kama_efficiency': kama_efficiency_value,
                'kama_efficiency_adjustment': kama_adjustment,
                'kama_quality': kama_quality,
                'kama_trend': row.get('kama_trend', None),

                # Ranging market data (NEW - for analysis and tracking)
                'ranging_score': ranging_score_value,
                'ranging_adjustment': ranging_adjustment,
                'ranging_quality': ranging_quality,

                # Market bias tracking (NEW - for conflict detection and database storage)
                'market_bias': market_bias,
                'market_bias_conflict': bias_conflict,
                'directional_consensus': directional_consensus,
            }

            # Calculate SL/TP using base class method (consistent with other strategies)
            sl_tp = self.calculate_optimal_sl_tp(signal, epic, row, 0)  # spread_pips not used in new calc
            signal['stop_distance'] = sl_tp['stop_distance']
            signal['limit_distance'] = sl_tp['limit_distance']

            return signal

        except Exception as e:
            self.logger.error(f"{epic_display}Error creating signal: {e}")
            return None

    def detect_signal(self, df: pd.DataFrame, epic: str, spread_pips: float = 1.5,
                     timeframe: str = '15m', **kwargs) -> Optional[Dict]:
        """
        Main signal detection method
        Works identically in both live and backtest modes
        """
        try:
            if len(df) < self.slow_period + self.signal_period:
                self.logger.warning(f"Insufficient data: {len(df)} bars")
                return None

            # Calculate MACD
            df = self.calculate_macd(df)

            # Calculate other indicators if not present
            if 'adx' not in df.columns:
                df = self._calculate_adx(df)

            if 'rsi' not in df.columns:
                df = self._calculate_rsi(df)

            # Identify swing points for proximity validation and strategy_indicators
            if self.swing_validator and self.swing_validator.smc_analyzer:
                try:
                    df = self.swing_validator.smc_analyzer.identify_swing_points(df)
                    self.logger.debug(f"Swing points identified for {epic}")
                except Exception as e:
                    self.logger.debug(f"Swing point identification skipped: {e}")

            # Ensure KAMA efficiency is available (for quality filtering)
            if self.use_kama_efficiency or self.ranging_penalty_enabled:
                df = self._ensure_kama_efficiency(df)

            # Detect MACD histogram crossovers with expansion + ADX trend validation (Priority 1)
            df = self.detect_crossover(df, epic=epic)

            # Detect ADX crossovers if enabled (Priority 2)
            if self.adx_crossover_enabled:
                df = self.detect_adx_crossover(df)

            # CRITICAL FIX: Only check the LATEST bar for crossover
            # The backtest engine calls us once per bar with growing data
            # The last bar is always the NEW bar to check
            # This prevents re-detecting the same crossover multiple times
            latest_bar = df.iloc[-1]

            # CRITICAL FIX: Only generate signals on COMPLETE bars
            # Incomplete bars have changing histogram values that can cause false triggers
            # Wait for bar to complete before checking crossover conditions
            if not self.backtest_mode:  # In live mode, check if bar is complete
                if 'is_complete' in df.columns and not latest_bar.get('is_complete', True):
                    self.logger.debug(f"⏳ Skipping incomplete bar for {epic} - waiting for bar to complete")
                    return None

            # PRIORITY 1: Check for MACD histogram BULL crossover (stronger signal)
            if latest_bar.get('bull_crossover', False):
                self.logger.info(f"🎯 BULL crossover triggered, validating signal for {epic}...")
                if self.validate_signal(latest_bar, 'BULL', epic=epic):
                    signal = self.create_signal(latest_bar, 'BULL', epic, timeframe, trigger_type='macd')
                    if signal:
                        # Swing proximity validation (uses full DataFrame for context)
                        if self.swing_validator:
                            validation = self.swing_validator.validate_entry_proximity(
                                df, latest_bar['close'], 'BULL', epic
                            )
                            if not validation['valid']:
                                self.logger.info(f"❌ BULL rejected: {validation.get('rejection_reason', 'swing proximity')}")
                                return None
                            # Apply confidence penalty if needed
                            signal['confidence'] -= validation.get('confidence_penalty', 0)

                        self.logger.info(f"✅ MACD BULL signal: {latest_bar.name}, ADX={latest_bar['adx']:.1f}, confidence={signal['confidence']:.1%}")
                        return signal

            # PRIORITY 1: Check for MACD histogram BEAR crossover (stronger signal)
            if latest_bar.get('bear_crossover', False):
                self.logger.info(f"🎯 BEAR crossover triggered, validating signal for {epic}...")
                if self.validate_signal(latest_bar, 'BEAR', epic=epic):
                    signal = self.create_signal(latest_bar, 'BEAR', epic, timeframe, trigger_type='macd')
                    if signal:
                        # Swing proximity validation (uses full DataFrame for context)
                        if self.swing_validator:
                            validation = self.swing_validator.validate_entry_proximity(
                                df, latest_bar['close'], 'BEAR', epic
                            )
                            if not validation['valid']:
                                self.logger.info(f"❌ BEAR rejected: {validation.get('rejection_reason', 'swing proximity')}")
                                return None
                            # Apply confidence penalty if needed
                            signal['confidence'] -= validation.get('confidence_penalty', 0)

                        self.logger.info(f"✅ MACD BEAR signal: {latest_bar.name}, ADX={latest_bar['adx']:.1f}, confidence={signal['confidence']:.1%}")
                        return signal

            # PRIORITY 2: Check for ADX BULL crossover (earlier entry signal)
            if self.adx_crossover_enabled and latest_bar.get('bull_adx_crossover', False):
                if self.validate_adx_signal(latest_bar, 'BULL', epic=epic):
                    signal = self.create_signal(latest_bar, 'BULL', epic, timeframe, trigger_type='adx')
                    if signal:
                        # Swing proximity validation
                        if self.swing_validator:
                            validation = self.swing_validator.validate_entry_proximity(
                                df, latest_bar['close'], 'BULL', epic
                            )
                            if not validation['valid']:
                                self.logger.info(f"❌ ADX BULL rejected: {validation.get('rejection_reason', 'swing proximity')}")
                                return None
                            signal['confidence'] -= validation.get('confidence_penalty', 0)

                        self.logger.info(f"✅ ADX BULL signal: {latest_bar.name}, ADX={latest_bar['adx']:.1f}, confidence={signal['confidence']:.1%} (early entry)")
                        return signal

            # PRIORITY 2: Check for ADX BEAR crossover (earlier entry signal)
            if self.adx_crossover_enabled and latest_bar.get('bear_adx_crossover', False):
                if self.validate_adx_signal(latest_bar, 'BEAR', epic=epic):
                    signal = self.create_signal(latest_bar, 'BEAR', epic, timeframe, trigger_type='adx')
                    if signal:
                        # Swing proximity validation
                        if self.swing_validator:
                            validation = self.swing_validator.validate_entry_proximity(
                                df, latest_bar['close'], 'BEAR', epic
                            )
                            if not validation['valid']:
                                self.logger.info(f"❌ ADX BEAR rejected: {validation.get('rejection_reason', 'swing proximity')}")
                                return None
                            signal['confidence'] -= validation.get('confidence_penalty', 0)

                        self.logger.info(f"✅ ADX BEAR signal: {latest_bar.name}, ADX={latest_bar['adx']:.1f}, confidence={signal['confidence']:.1%} (early entry)")
                        return signal

            return None

        except Exception as e:
            self.logger.error(f"Error detecting signal: {e}", exc_info=True)
            return None

    def _log_expansion_status(self, df: pd.DataFrame, min_histogram: float, epic: str = None):
        """
        Log detailed expansion confirmation status for latest bar

        Args:
            df: DataFrame with expansion tracking columns
            min_histogram: Minimum histogram threshold
            epic: Epic code for pair identification in logs
        """
        try:
            latest = df.iloc[-1]

            # Format epic for logging
            epic_display = f"[{epic}] " if epic else ""

            # Get ADX trend info
            adx_info = ""
            if self.require_adx_rising and 'adx_change' in df.columns:
                adx_current = latest['adx']
                adx_prev = latest['adx_prev'] if pd.notna(latest.get('adx_prev')) else adx_current
                adx_change = latest['adx_change'] if pd.notna(latest.get('adx_change')) else 0
                adx_rising = latest.get('adx_is_rising', False)
                adx_status = "✅ rising" if adx_rising else "❌ falling/flat"
                adx_info = f", ADX: {adx_prev:.1f}→{adx_current:.1f} ({adx_change:+.1f}) {adx_status}"

            # Log BULL crossover events
            if latest.get('bull_cross_initial', False):
                hist = latest['macd_histogram']
                meets_threshold = hist >= min_histogram
                threshold_emoji = "✅" if meets_threshold else "❌"
                self.logger.info(f"🔵 {epic_display}BULL CROSSOVER DETECTED (Bar 0/3)")
                self.logger.info(f"   📊 Histogram: {hist:.6f} {threshold_emoji} (threshold: {min_histogram:.6f}){adx_info}")

                if self.expansion_allow_immediate and hist >= min_histogram:
                    if not self.require_adx_rising or latest.get('adx_is_rising', True):
                        self.logger.info(f"   ✅ IMMEDIATE TRIGGER APPROVED - All conditions met on crossover bar")
                    else:
                        self.logger.info(f"   ❌ IMMEDIATE TRIGGER BLOCKED - ADX not rising")
                else:
                    self.logger.info(f"   ⏳ WAITING FOR EXPANSION - Tracking for next 3 bars...")

            elif latest.get('bull_window', False):
                bars_since = int(latest.get('bull_bars_since_cross', 0))
                hist = latest['macd_histogram']
                hist_ok = hist >= min_histogram
                adx_ok = latest.get('adx_is_rising', True) if self.require_adx_rising else True

                hist_emoji = "✅" if hist_ok else "❌"
                adx_emoji = "✅" if adx_ok else "❌"

                self.logger.info(f"🔵 {epic_display}BULL EXPANSION CHECK (Bar {bars_since}/{self.expansion_window_bars})")
                self.logger.info(f"   📊 Histogram: {hist:.6f} {hist_emoji} (need: {min_histogram:.6f}){adx_info}")

                # Check ADX catch-up window status (if enabled)
                if self.adx_catchup_enabled:
                    adx_current = latest.get('adx', 0)
                    adx_met_in_window = latest.get('adx_met_in_window', False)
                    adx_catchup_emoji = "✅" if adx_met_in_window else "❌"
                    self.logger.info(f"   🎯 ADX Catch-up: {adx_catchup_emoji} ADX={adx_current:.1f} (need {self.min_adx} within {self.adx_catchup_window_bars} bars)")

                if latest.get('bull_crossover', False):
                    self.logger.info(f"   🎯 ✅ {epic_display}EXPANSION CONFIRMED - Signal triggered on bar {bars_since}!")
                elif bars_since >= self.expansion_window_bars:
                    reasons = []
                    if not hist_ok:
                        reasons.append(f"histogram never reached {min_histogram:.6f} (max: {hist:.6f})")
                    if self.require_adx_rising and not adx_ok:
                        reasons.append("ADX not rising")
                    # Add ADX catch-up window failure reason
                    if self.adx_catchup_enabled and hist_ok:
                        adx_current = latest.get('adx', 0)
                        adx_met_in_window = latest.get('adx_met_in_window', False)
                        if not adx_met_in_window:
                            reasons.append(f"ADX never reached {self.min_adx} within {self.adx_catchup_window_bars} bars (current: {adx_current:.1f})")
                    self.logger.info(f"   ⏰ ❌ EXPANSION WINDOW EXPIRED - Signal abandoned: {', '.join(reasons) if reasons else 'Unknown reason'}")
                else:
                    waiting_for = []
                    if not hist_ok:
                        waiting_for.append(f"histogram to reach {min_histogram:.6f} (current: {hist:.6f})")
                    if self.require_adx_rising and not adx_ok:
                        waiting_for.append("ADX to rise")
                    # Add ADX catch-up waiting message if histogram is met but ADX isn't
                    if self.adx_catchup_enabled and hist_ok:
                        adx_current = latest.get('adx', 0)
                        adx_met_in_window = latest.get('adx_met_in_window', False)
                        if not adx_met_in_window:
                            waiting_for.append(f"ADX to reach {self.min_adx} (current: {adx_current:.1f}, window: {bars_since}/{self.adx_catchup_window_bars} bars)")
                    if waiting_for:
                        self.logger.info(f"   ⏳ Still waiting: {', '.join(waiting_for)}")

            # Log BEAR crossover events
            if latest.get('bear_cross_initial', False):
                hist = latest['macd_histogram']
                meets_threshold = abs(hist) >= min_histogram
                threshold_emoji = "✅" if meets_threshold else "❌"
                self.logger.info(f"🔴 {epic_display}BEAR CROSSOVER DETECTED (Bar 0/3)")
                self.logger.info(f"   📊 Histogram: {hist:.6f} {threshold_emoji} (threshold: {min_histogram:.6f}){adx_info}")

                if self.expansion_allow_immediate and abs(hist) >= min_histogram:
                    if not self.require_adx_rising or latest.get('adx_is_rising', True):
                        self.logger.info(f"   ✅ IMMEDIATE TRIGGER APPROVED - All conditions met on crossover bar")
                    else:
                        self.logger.info(f"   ❌ IMMEDIATE TRIGGER BLOCKED - ADX not rising")
                else:
                    self.logger.info(f"   ⏳ WAITING FOR EXPANSION - Tracking for next 3 bars...")

            elif latest.get('bear_window', False):
                bars_since = int(latest.get('bear_bars_since_cross', 0))
                hist = latest['macd_histogram']
                hist_ok = abs(hist) >= min_histogram
                adx_ok = latest.get('adx_is_rising', True) if self.require_adx_rising else True

                hist_emoji = "✅" if hist_ok else "❌"
                adx_emoji = "✅" if adx_ok else "❌"

                self.logger.info(f"🔴 {epic_display}BEAR EXPANSION CHECK (Bar {bars_since}/{self.expansion_window_bars})")
                self.logger.info(f"   📊 Histogram: {hist:.6f} (|{abs(hist):.6f}|) {hist_emoji} (need: {min_histogram:.6f}){adx_info}")

                # Check ADX catch-up window status (if enabled)
                if self.adx_catchup_enabled:
                    adx_current = latest.get('adx', 0)
                    adx_met_in_window = latest.get('adx_met_in_window', False)
                    adx_catchup_emoji = "✅" if adx_met_in_window else "❌"
                    self.logger.info(f"   🎯 ADX Catch-up: {adx_catchup_emoji} ADX={adx_current:.1f} (need {self.min_adx} within {self.adx_catchup_window_bars} bars)")

                if latest.get('bear_crossover', False):
                    self.logger.info(f"   🎯 ✅ {epic_display}EXPANSION CONFIRMED - Signal triggered on bar {bars_since}!")
                elif bars_since >= self.expansion_window_bars:
                    reasons = []
                    if not hist_ok:
                        reasons.append(f"|histogram| never reached {min_histogram:.6f} (max: {abs(hist):.6f})")
                    if self.require_adx_rising and not adx_ok:
                        reasons.append("ADX not rising")
                    # Add ADX catch-up window failure reason
                    if self.adx_catchup_enabled and hist_ok:
                        adx_current = latest.get('adx', 0)
                        adx_met_in_window = latest.get('adx_met_in_window', False)
                        if not adx_met_in_window:
                            reasons.append(f"ADX never reached {self.min_adx} within {self.adx_catchup_window_bars} bars (current: {adx_current:.1f})")
                    self.logger.info(f"   ⏰ ❌ EXPANSION WINDOW EXPIRED - Signal abandoned: {', '.join(reasons) if reasons else 'Unknown reason'}")
                else:
                    waiting_for = []
                    if not hist_ok:
                        waiting_for.append(f"|histogram| to reach {min_histogram:.6f} (current: {abs(hist):.6f})")
                    if self.require_adx_rising and not adx_ok:
                        waiting_for.append("ADX to rise")
                    # Add ADX catch-up waiting message if histogram is met but ADX isn't
                    if self.adx_catchup_enabled and hist_ok:
                        adx_current = latest.get('adx', 0)
                        adx_met_in_window = latest.get('adx_met_in_window', False)
                        if not adx_met_in_window:
                            waiting_for.append(f"ADX to reach {self.min_adx} (current: {adx_current:.1f}, window: {bars_since}/{self.adx_catchup_window_bars} bars)")
                    if waiting_for:
                        self.logger.info(f"   ⏳ Still waiting: {', '.join(waiting_for)}")

        except Exception as e:
            self.logger.debug(f"Error logging expansion status: {e}")

    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX indicator using Wilder's smoothing (matches TradingView)

        IMPORTANT: ADX uses Wilder's smoothing (RMA), not simple moving average
        """
        df = df.copy()
        period = 14

        # True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Wilder's smoothing (RMA) = EWM with alpha=1/period
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_di_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()

        # Add ATR to DataFrame for SL/TP calculation
        df['atr'] = atr

        # Directional Indicators
        plus_di = 100 * (plus_di_smooth / atr)
        minus_di = 100 * (minus_di_smooth / atr)

        # Add directional indicators to DataFrame
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_plus'] = plus_di  # Alternative name
        df['di_minus'] = minus_di  # Alternative name

        # DX and ADX with Wilder's smoothing
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['dx'] = dx
        df['adx'] = dx.ewm(alpha=1/period, adjust=False).mean()

        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator"""
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def _validate_kama_efficiency(self, df: pd.DataFrame, idx: int, signal_type: str) -> tuple:
        """
        Validate KAMA efficiency and calculate confidence adjustment

        Non-blocking: Only adjusts confidence, never blocks signals

        KAMA efficiency measures market directional clarity:
        - High efficiency (>0.5): Strong trending with minimal noise
        - Medium efficiency (0.15-0.5): Moderate trend clarity
        - Low efficiency (<0.15): Choppy/ranging conditions

        Args:
            df: DataFrame with KAMA indicators
            idx: Current row index
            signal_type: 'BULL' or 'BEAR'

        Returns:
            tuple: (confidence_adjustment, quality_label, efficiency_ratio)
        """
        try:
            # Get KAMA data from current row
            current_row = df.iloc[idx]
            kama_efficiency = current_row.get('efficiency_ratio', None)
            kama_trend = current_row.get('kama_trend', 0)

            # If KAMA data not available, return neutral (no adjustment)
            if kama_efficiency is None or pd.isna(kama_efficiency):
                if self.log_kama_efficiency:
                    self.logger.debug("⚪ KAMA efficiency data not available - no adjustment")
                return (0.0, "N/A", None)

            # Determine confidence adjustment based on efficiency tier
            if kama_efficiency >= self.kama_excellent_threshold:
                adjustment = self.kama_excellent_boost
                quality = "EXCELLENT"
                emoji = "⭐⭐⭐⭐⭐"
            elif kama_efficiency >= self.kama_good_threshold:
                adjustment = self.kama_good_boost
                quality = "GOOD"
                emoji = "⭐⭐⭐"
            elif kama_efficiency >= self.kama_acceptable_threshold:
                adjustment = self.kama_acceptable_neutral
                quality = "ACCEPTABLE"
                emoji = "⭐⭐"
            elif kama_efficiency >= self.kama_poor_threshold:
                adjustment = self.kama_poor_penalty
                quality = "POOR"
                emoji = "⭐"
            else:
                adjustment = self.kama_very_poor_penalty
                quality = "VERY POOR"
                emoji = "❌"

            # Optional: Check KAMA trend alignment with signal direction
            trend_note = ""
            if self.require_kama_trend_alignment:
                trend_aligned = (
                    (signal_type == 'BULL' and kama_trend > 0) or
                    (signal_type == 'BEAR' and kama_trend < 0)
                )

                if not trend_aligned:
                    adjustment += self.kama_trend_conflict_penalty
                    trend_note = f" (TREND CONFLICT: kama_trend={kama_trend:+.1f})"
                    quality += " + CONFLICT"

            # Log the analysis
            if self.log_kama_efficiency:
                self.logger.info(
                    f"🎯 KAMA Efficiency: {kama_efficiency:.3f} {emoji} {quality}{trend_note} "
                    f"→ Confidence adjustment: {adjustment:+.1%}"
                )

            return (adjustment, quality, kama_efficiency)

        except Exception as e:
            self.logger.error(f"❌ KAMA efficiency validation error: {e}")
            return (0.0, "ERROR", None)

    def _get_pip_value(self, epic: str) -> float:
        """
        Get pip value for epic/instrument

        Returns:
            Pip value (e.g., 0.0001 for EUR pairs, 0.01 for JPY pairs)
        """
        # JPY pairs use 0.01 as pip value (2 decimal places)
        if 'JPY' in epic or 'jpy' in epic.lower():
            return 0.01
        # Most other forex pairs use 0.0001 (4 decimal places)
        else:
            return 0.0001

    def _get_min_histogram(self, epic: str) -> float:
        """
        Get minimum histogram threshold for this epic/pair

        JPY pairs need much larger histogram movement to be visible
        because their price values are 100x larger (e.g., 150 vs 1.5)

        Supports both legacy format (float) and new format (dict with 'histogram' and 'min_adx')

        Returns:
            Minimum histogram value required for valid signal
        """
        if not epic:
            default_config = self.min_histogram_thresholds.get('default', 0.0002)
            if isinstance(default_config, dict):
                return default_config.get('histogram', 0.0002)
            return default_config

        # Extract pair name from epic (e.g., CS.D.EURJPY.MINI.IP -> EURJPY)
        pair = epic.upper()

        # Try to find matching pair in thresholds
        for pair_name, config in self.min_histogram_thresholds.items():
            if pair_name == 'default':
                continue
            if pair_name in pair:
                # Handle both old format (float) and new format (dict)
                if isinstance(config, dict):
                    threshold = config.get('histogram', 0.015)
                    self.logger.debug(f"📏 {pair_name} minimum histogram: {threshold}")
                    return threshold
                else:
                    # Legacy format: config is a float
                    self.logger.debug(f"📏 {pair_name} minimum histogram: {config}")
                    return config

        # Default for pairs not found
        default_config = self.min_histogram_thresholds.get('default', 0.0002)
        if isinstance(default_config, dict):
            threshold = default_config.get('histogram', 0.0002)
        else:
            threshold = default_config
        self.logger.debug(f"📏 Default minimum histogram: {threshold}")
        return threshold

    # Required abstract methods from BaseStrategy
    def get_required_indicators(self):
        """Return list of required indicators"""
        return ['macd_line', 'macd_signal', 'macd_histogram', 'adx', 'rsi']

    # Compatibility methods for backtest system
    def detect_signal_auto(self, df: pd.DataFrame, epic: str = None, spread_pips: float = 0,
                          timeframe: str = '15m', **kwargs) -> Optional[Dict]:
        """Auto detection wrapper"""
        return self.detect_signal(df, epic, spread_pips, timeframe, **kwargs)

    def enable_forex_integration(self, epic):
        """Compatibility method"""
        pass
