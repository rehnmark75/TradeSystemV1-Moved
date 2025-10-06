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

        # Validation thresholds (read from config)
        self.min_adx = getattr(config_macd_strategy, 'MACD_MIN_ADX', 20) if config_macd_strategy else 20
        self.min_confidence = 0.60  # 60% minimum confidence (lowered to allow more signals)

        self.logger.info(f"🎯 MACD ADX threshold set to: {self.min_adx} (from config)")

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
        self.adx_crossover_threshold = getattr(config_macd_strategy, 'MACD_ADX_CROSSOVER_THRESHOLD', 25) if config_macd_strategy else 25
        self.adx_crossover_lookback = getattr(config_macd_strategy, 'MACD_ADX_CROSSOVER_LOOKBACK', 3) if config_macd_strategy else 3
        self.adx_min_histogram = getattr(config_macd_strategy, 'MACD_ADX_MIN_HISTOGRAM', 0.0001) if config_macd_strategy else 0.0001
        self.adx_require_expansion = getattr(config_macd_strategy, 'MACD_ADX_REQUIRE_EXPANSION', True) if config_macd_strategy else True
        self.adx_min_confidence = getattr(config_macd_strategy, 'MACD_ADX_MIN_CONFIDENCE', 0.50) if config_macd_strategy else 0.50

        if self.adx_crossover_enabled:
            self.logger.info(f"🚀 ADX Crossover Trigger ENABLED - ADX crosses {self.adx_crossover_threshold} with MACD alignment")
        else:
            self.logger.info(f"⚪ ADX Crossover Trigger DISABLED")

        # ATR-based SL/TP configuration
        self.stop_atr_multiplier = getattr(config_macd_strategy, 'MACD_STOP_LOSS_ATR_MULTIPLIER', 1.8) if config_macd_strategy else 1.8
        self.target_atr_multiplier = getattr(config_macd_strategy, 'MACD_TAKE_PROFIT_ATR_MULTIPLIER', 4.0) if config_macd_strategy else 4.0
        self.use_structure_stops = getattr(config_macd_strategy, 'MACD_USE_STRUCTURE_STOPS', True) if config_macd_strategy else True
        self.min_stop_pips = getattr(config_macd_strategy, 'MACD_MIN_STOP_DISTANCE_PIPS', 12.0) if config_macd_strategy else 12.0
        self.max_stop_pips = getattr(config_macd_strategy, 'MACD_MAX_STOP_DISTANCE_PIPS', 30.0) if config_macd_strategy else 30.0

        # Minimum histogram thresholds (pair-specific to prevent tiny crossovers)
        self.min_histogram_thresholds = getattr(config_macd_strategy, 'MACD_MIN_HISTOGRAM_THRESHOLDS', {
            'default': 0.0002,
            'EURJPY': 0.020, 'GBPJPY': 0.020, 'AUDJPY': 0.012,
            'NZDJPY': 0.010, 'USDJPY': 0.015, 'CADJPY': 0.012, 'CHFJPY': 0.015
        }) if config_macd_strategy else {}

        self.logger.info(f"✅ Clean MACD Strategy initialized - ADX >= {self.min_adx}, Swing validation: {self.swing_validator is not None}")
        self.logger.info(f"📊 SL/TP: {self.stop_atr_multiplier}x ATR stop, {self.target_atr_multiplier}x ATR target (Structure stops: {self.use_structure_stops})")
        self.logger.info(f"📏 Histogram thresholds: Default={self.min_histogram_thresholds.get('default', 0.0002)}, JPY pairs scaled 50-100x higher")
        self.logger.info(f"🔧 CRITICAL DEBUG: Using REBUILT MACD with NaN validation and dynamic EMA filter (v2.0)")

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

    def detect_crossover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect MACD histogram crossovers (zero line crosses)
        Returns ONE signal per crossover event
        """
        df = df.copy()

        # Shift histogram to get previous value
        df['histogram_prev'] = df['macd_histogram'].shift(1)

        # Detect crossovers: current > 0 and previous <= 0 (or vice versa)
        df['bull_crossover'] = (df['macd_histogram'] > 0) & (df['histogram_prev'] <= 0)
        df['bear_crossover'] = (df['macd_histogram'] < 0) & (df['histogram_prev'] >= 0)

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

    def validate_adx_signal(self, row: pd.Series, signal_type: str) -> bool:
        """
        Validate ADX crossover signal meets quality requirements

        Simplified validation for ADX crossover signals (less strict than MACD crossover)
        ADX crossing above threshold already indicates trend strength
        """
        # Check histogram is in correct direction (already checked in detect_adx_crossover, but double-check)
        histogram = row.get('macd_histogram', 0)
        if signal_type == 'BULL' and histogram <= 0:
            self.logger.debug(f"❌ ADX BULL rejected: negative histogram {histogram:.6f}")
            return False
        if signal_type == 'BEAR' and histogram >= 0:
            self.logger.debug(f"❌ ADX BEAR rejected: positive histogram {histogram:.6f}")
            return False

        # Check MACD line position (same filter as regular signals)
        macd_line = row.get('macd_line', 0)
        if signal_type == 'BEAR' and macd_line > 0.05:
            self.logger.info(f"❌ ADX BEAR rejected: MACD line too positive {macd_line:.6f}")
            return False
        if signal_type == 'BULL' and macd_line < -0.05:
            self.logger.info(f"❌ ADX BULL rejected: MACD line too negative {macd_line:.6f}")
            return False

        # Check RSI (optional - avoid extreme zones)
        rsi = row.get('rsi', 50)
        if signal_type == 'BULL' and rsi > 70:
            self.logger.debug(f"❌ ADX BULL rejected: RSI {rsi:.1f} overbought")
            return False
        if signal_type == 'BEAR' and rsi < 30:
            self.logger.debug(f"❌ ADX BEAR rejected: RSI {rsi:.1f} oversold")
            return False

        # EMA filter (if enabled)
        if self.ema_filter_enabled:
            price = row.get('close', 0)
            ema_col = f'ema_{self.ema_filter_period}'
            ema_value = row.get(ema_col, 0)

            if pd.isna(ema_value) or pd.isna(price) or ema_value <= 0 or price <= 0:
                self.logger.info(f"❌ ADX {signal_type} rejected: Invalid EMA or price")
                return False

            if signal_type == 'BULL' and price < ema_value:
                self.logger.debug(f"❌ ADX BULL rejected: price below EMA{self.ema_filter_period}")
                return False
            if signal_type == 'BEAR' and price > ema_value:
                self.logger.debug(f"❌ ADX BEAR rejected: price above EMA{self.ema_filter_period}")
                return False

        self.logger.info(f"✅ ADX {signal_type} signal validated")
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
        # Check ADX (handle NaN values)
        adx = row.get('adx', 0)

        # CRITICAL: Check for NaN or invalid ADX values
        if pd.isna(adx) or adx <= 0:
            self.logger.info(f"❌ {signal_type} rejected: Invalid ADX ({adx})")
            return False

        if adx < self.min_adx:
            self.logger.info(f"❌ {signal_type} rejected: ADX {adx:.1f} < {self.min_adx}")
            return False

        # Log ADX for signals that pass (for debugging)
        self.logger.info(f"✅ {signal_type} passed ADX filter: ADX={adx:.1f} (min={self.min_adx})")

        # Check histogram magnitude (pair-specific thresholds)
        histogram = row.get('macd_histogram', 0)
        if epic:
            min_histogram = self._get_min_histogram(epic)
            if abs(histogram) < min_histogram:
                self.logger.info(f"❌ {signal_type} rejected: Histogram {histogram:.6f} too small (min={min_histogram:.6f}) - not visible on chart")
                return False
            else:
                self.logger.debug(f"✅ Histogram magnitude OK: {abs(histogram):.6f} >= {min_histogram:.6f}")

        # Check histogram direction
        histogram = row.get('macd_histogram', 0)
        if signal_type == 'BULL' and histogram <= 0:
            self.logger.debug(f"❌ BULL rejected: negative histogram {histogram:.6f}")
            return False
        if signal_type == 'BEAR' and histogram >= 0:
            self.logger.debug(f"❌ BEAR rejected: positive histogram {histogram:.6f}")
            return False

        # CRITICAL FIX: Check MACD line position (zero line filter)
        # BULL signals should have MACD line trending upward (can be below zero if reversing)
        # BEAR signals should have MACD line trending downward (can be above zero if reversing)
        # But we should avoid signals when MACD line contradicts direction strongly
        macd_line = row.get('macd_line', 0)

        # For BEAR signals, if MACD line is strongly positive (>0.1 for forex), it's a bad signal
        # This prevents selling during strong uptrends just because histogram dipped slightly
        if signal_type == 'BEAR' and macd_line > 0.05:
            self.logger.info(f"❌ BEAR rejected: MACD line too positive {macd_line:.6f} (still in bullish territory)")
            return False

        # For BULL signals, if MACD line is strongly negative (<-0.1), it's a bad signal
        # This prevents buying during strong downtrends just because histogram rose slightly
        if signal_type == 'BULL' and macd_line < -0.05:
            self.logger.info(f"❌ BULL rejected: MACD line too negative {macd_line:.6f} (still in bearish territory)")
            return False

        # Check RSI (optional - just avoid extreme zones)
        rsi = row.get('rsi', 50)
        if signal_type == 'BULL' and rsi > 70:
            self.logger.debug(f"❌ BULL rejected: RSI {rsi:.1f} overbought")
            return False
        if signal_type == 'BEAR' and rsi < 30:
            self.logger.debug(f"❌ BEAR rejected: RSI {rsi:.1f} oversold")
            return False

        # EMA FILTER (configurable via config file)
        if self.ema_filter_enabled:
            price = row.get('close', 0)
            ema_col = f'ema_{self.ema_filter_period}'
            ema_value = row.get(ema_col, 0)

            # Validate EMA and price are valid numbers
            if pd.isna(ema_value) or pd.isna(price) or ema_value <= 0 or price <= 0:
                self.logger.info(f"❌ {signal_type} rejected: Invalid EMA{self.ema_filter_period} or price (price={price}, EMA={ema_value})")
                return False

            # Apply EMA trend filter
            if signal_type == 'BULL' and price < ema_value:
                self.logger.debug(f"❌ BULL rejected: price {price:.5f} below EMA{self.ema_filter_period} {ema_value:.5f} (ADX={adx:.1f})")
                return False
            if signal_type == 'BEAR' and price > ema_value:
                self.logger.debug(f"❌ BEAR rejected: price {price:.5f} above EMA{self.ema_filter_period} {ema_value:.5f} (ADX={adx:.1f})")
                return False

            self.logger.debug(f"✅ {signal_type} passed EMA{self.ema_filter_period} filter: price={price:.5f}, EMA={ema_value:.5f} (ADX={adx:.1f})")
        else:
            self.logger.debug(f"✅ {signal_type} passed (no EMA filter) - ADX={adx:.1f}")

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
        try:
            # Calculate confidence
            confidence = self.calculate_confidence(row, signal_type)

            # Use different minimum confidence for ADX crossover signals
            min_conf = self.adx_min_confidence if trigger_type == 'adx' else self.min_confidence

            if confidence < min_conf:
                self.logger.debug(f"❌ {signal_type} rejected: confidence {confidence:.1%} < {min_conf:.1%}")
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
            }

            # Calculate ATR-based SL/TP
            atr = row.get('atr', None)
            if atr and atr > 0:
                # Get pip value for this epic
                pip_value = self._get_pip_value(epic)

                # Calculate ATR in pips
                atr_pips = atr / pip_value

                # Calculate SL/TP based on ATR multipliers
                stop_distance = max(self.min_stop_pips, min(self.max_stop_pips, atr_pips * self.stop_atr_multiplier))
                limit_distance = atr_pips * self.target_atr_multiplier

                self.logger.debug(f"📊 ATR-based SL/TP: ATR={atr_pips:.1f} pips, SL={stop_distance:.1f}, TP={limit_distance:.1f}")
            else:
                # Fallback to defaults if ATR not available
                stop_distance = self.min_stop_pips
                limit_distance = self.min_stop_pips * 2.5
                self.logger.warning(f"⚠️ ATR not available, using default SL/TP: {stop_distance}/{limit_distance} pips")

            signal['stop_distance'] = stop_distance
            signal['limit_distance'] = limit_distance

            return signal

        except Exception as e:
            self.logger.error(f"Error creating signal: {e}")
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

            # Detect MACD histogram crossovers (Priority 1)
            df = self.detect_crossover(df)

            # Detect ADX crossovers if enabled (Priority 2)
            if self.adx_crossover_enabled:
                df = self.detect_adx_crossover(df)

            # CRITICAL FIX: Only check the LATEST bar for crossover
            # The backtest engine calls us once per bar with growing data
            # The last bar is always the NEW bar to check
            # This prevents re-detecting the same crossover multiple times
            latest_bar = df.iloc[-1]

            # PRIORITY 1: Check for MACD histogram BULL crossover (stronger signal)
            if latest_bar.get('bull_crossover', False):
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
                if self.validate_adx_signal(latest_bar, 'BULL'):
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
                if self.validate_adx_signal(latest_bar, 'BEAR'):
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

        Returns:
            Minimum histogram value required for valid signal
        """
        # Extract pair name from epic (e.g., CS.D.EURJPY.MINI.IP -> EURJPY)
        pair = epic.upper()
        for p in ['EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'USDJPY', 'CADJPY', 'CHFJPY']:
            if p in pair:
                threshold = self.min_histogram_thresholds.get(p, 0.015)
                self.logger.debug(f"📏 {p} minimum histogram: {threshold}")
                return threshold

        # Default for non-JPY pairs
        default = self.min_histogram_thresholds.get('default', 0.0002)
        self.logger.debug(f"📏 Default minimum histogram: {default}")
        return default

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
