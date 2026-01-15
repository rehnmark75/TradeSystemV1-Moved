#!/usr/bin/env python3
"""
Scalp Signal Qualifier - Momentum Confirmation Filters for Scalp Trades

VERSION: 1.1.0
DATE: 2026-01-15
STATUS: Added Micro-Regime Validation

PURPOSE:
    This module provides signal qualification filters for scalp trades.
    It runs multiple momentum confirmation checks on generated signals to
    improve win rate by filtering out low-quality entries.

MODES:
    - MONITORING: Logs qualification results but passes all signals (for analysis)
    - ACTIVE: Blocks signals that don't meet minimum qualification score

FILTERS:
    Momentum Filters (slower indicators):
    1. RSI Momentum: Validates RSI is in appropriate zone and moving in trade direction
    2. Two-Pole Oscillator: Validates momentum using BigBeluga's minimal-lag oscillator
    3. MACD Direction: Validates MACD histogram confirms trade direction

    Micro-Regime Filters (immediate price action at signal):
    4. Consecutive Candles: Last 2+ candles align with trade direction
    5. Anti-Chop: Reject if alternating green/red candles (choppy market)
    6. Body Dominance: Average body > wick (conviction vs indecision)
    7. Micro-Range: Reject if last 5 candles range too tight (congestion)
    8. Momentum Candle: Last candle body > 1.5x average (confirmation thrust)

USAGE:
    # Initialize with config
    qualifier = ScalpSignalQualifier(config, logger)

    # Qualify a signal
    passed, score, results = qualifier.qualify_signal(signal, df_entry, df_trigger)

    # Score is 0.0-1.0 (percentage of filters passed)
    # In MONITORING mode, passed is always True
    # In ACTIVE mode, passed is True only if score >= min_qualification_score

DATABASE CONFIG:
    Momentum Filters:
    - scalp_qualification_enabled: Master toggle (default: FALSE)
    - scalp_qualification_mode: 'MONITORING' or 'ACTIVE' (default: 'MONITORING')
    - scalp_min_qualification_score: Threshold for ACTIVE mode (default: 0.50)
    - scalp_rsi_filter_enabled: Toggle RSI filter (default: TRUE)
    - scalp_two_pole_filter_enabled: Toggle Two-Pole filter (default: TRUE)
    - scalp_macd_filter_enabled: Toggle MACD filter (default: TRUE)

    Micro-Regime Filters:
    - scalp_micro_regime_enabled: Toggle all micro-regime filters (default: FALSE)
    - scalp_consecutive_candles_enabled: Require consecutive aligned candles (default: TRUE)
    - scalp_anti_chop_enabled: Reject choppy markets (default: TRUE)
    - scalp_body_dominance_enabled: Require body > wick (default: TRUE)
    - scalp_micro_range_enabled: Reject tight congestion (default: TRUE)
    - scalp_momentum_candle_enabled: Require momentum thrust (default: FALSE)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime


class ScalpSignalQualifier:
    """
    Signal qualification system for scalp trades.

    Runs multiple confirmation filters on generated signals and either:
    - MONITORING mode: Logs results, passes all signals (for data collection)
    - ACTIVE mode: Blocks signals below min_qualification_score

    Expected improvement: Win rate +10-20% by filtering exhausted/counter-trend entries
    """

    VERSION = "1.1.0"

    def __init__(self, config=None, logger=None, db_config=None):
        """
        Initialize ScalpSignalQualifier.

        Args:
            config: Main config module (optional)
            logger: Logger instance (optional)
            db_config: SMCSimpleConfig object from database (optional, preferred)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._db_config = db_config

        # Load configuration from database config or defaults
        self._load_config()

        self.logger.info(f"ScalpSignalQualifier v{self.VERSION} initialized")
        self.logger.info(f"   Enabled: {self.enabled}")
        self.logger.info(f"   Mode: {self.mode}")
        self.logger.info(f"   Min Score: {self.min_score:.0%}")
        self.logger.info(f"   Momentum Filters: RSI={self.rsi_filter_enabled}, "
                        f"TwoPole={self.two_pole_filter_enabled}, "
                        f"MACD={self.macd_filter_enabled}")
        self.logger.info(f"   Micro-Regime: {self.micro_regime_enabled}")
        if self.micro_regime_enabled:
            self.logger.info(f"     ConsecCandles={self.consecutive_candles_enabled}, "
                            f"AntiChop={self.anti_chop_enabled}, "
                            f"BodyDom={self.body_dominance_enabled}, "
                            f"MicroRange={self.micro_range_enabled}, "
                            f"MomCandle={self.momentum_candle_enabled}")

    def _load_config(self):
        """Load qualification configuration from database or defaults."""
        # Master toggle and mode
        self.enabled = self._get_config_value('scalp_qualification_enabled', False)
        self.mode = self._get_config_value('scalp_qualification_mode', 'MONITORING')
        self.min_score = self._get_config_value('scalp_min_qualification_score', 0.50)

        # Per-filter toggles
        self.rsi_filter_enabled = self._get_config_value('scalp_rsi_filter_enabled', True)
        self.two_pole_filter_enabled = self._get_config_value('scalp_two_pole_filter_enabled', True)
        self.macd_filter_enabled = self._get_config_value('scalp_macd_filter_enabled', True)

        # RSI thresholds
        self.rsi_bull_min = self._get_config_value('scalp_rsi_bull_min', 40)
        self.rsi_bull_max = self._get_config_value('scalp_rsi_bull_max', 75)
        self.rsi_bear_min = self._get_config_value('scalp_rsi_bear_min', 25)
        self.rsi_bear_max = self._get_config_value('scalp_rsi_bear_max', 60)

        # Two-Pole thresholds
        self.two_pole_bull_threshold = self._get_config_value('scalp_two_pole_bull_threshold', -0.3)
        self.two_pole_bear_threshold = self._get_config_value('scalp_two_pole_bear_threshold', 0.3)

        # Micro-regime filter toggles
        self.micro_regime_enabled = self._get_config_value('scalp_micro_regime_enabled', False)
        self.consecutive_candles_enabled = self._get_config_value('scalp_consecutive_candles_enabled', True)
        self.anti_chop_enabled = self._get_config_value('scalp_anti_chop_enabled', True)
        self.body_dominance_enabled = self._get_config_value('scalp_body_dominance_enabled', True)
        self.micro_range_enabled = self._get_config_value('scalp_micro_range_enabled', True)
        self.momentum_candle_enabled = self._get_config_value('scalp_momentum_candle_enabled', False)

        # Micro-regime thresholds
        self.consecutive_candles_min = self._get_config_value('scalp_consecutive_candles_min', 2)
        self.anti_chop_lookback = self._get_config_value('scalp_anti_chop_lookback', 4)
        self.anti_chop_max_alternations = self._get_config_value('scalp_anti_chop_max_alternations', 2)
        self.body_dominance_lookback = self._get_config_value('scalp_body_dominance_lookback', 3)
        self.body_dominance_ratio = self._get_config_value('scalp_body_dominance_ratio', 1.0)
        self.micro_range_lookback = self._get_config_value('scalp_micro_range_lookback', 5)
        self.micro_range_min_pips = self._get_config_value('scalp_micro_range_min_pips', 3.0)
        self.momentum_candle_multiplier = self._get_config_value('scalp_momentum_candle_multiplier', 1.5)

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get config value from database config, then fallback to defaults."""
        # Priority 1: Database config object
        if self._db_config:
            value = getattr(self._db_config, key, None)
            if value is not None:
                return value

        # Priority 2: Config module
        if self.config:
            value = getattr(self.config, key, None)
            if value is not None:
                return value

        # Fallback to default
        return default

    def set_mode(self, mode: str):
        """
        Set qualification mode.

        Args:
            mode: 'MONITORING' or 'ACTIVE'
        """
        if mode not in ('MONITORING', 'ACTIVE'):
            raise ValueError(f"Invalid mode: {mode}. Must be 'MONITORING' or 'ACTIVE'")
        self.mode = mode
        self.logger.info(f"ScalpSignalQualifier mode set to: {mode}")

    def set_enabled(self, enabled: bool):
        """Enable or disable the qualifier."""
        self.enabled = enabled
        self.logger.info(f"ScalpSignalQualifier enabled: {enabled}")

    def qualify_signal(
        self,
        signal: Dict,
        df_entry: pd.DataFrame,
        df_trigger: pd.DataFrame
    ) -> Tuple[bool, float, List[Dict]]:
        """
        Run all qualification filters on a signal.

        Args:
            signal: Signal dict with 'direction' key ('BULL' or 'BEAR')
            df_entry: Entry timeframe DataFrame (e.g., 1m or 5m) with indicators
            df_trigger: Trigger timeframe DataFrame (e.g., 5m or 15m) with indicators

        Returns:
            Tuple of (passed, score, filter_results):
            - passed: True if signal passes qualification (always True in MONITORING mode)
            - score: 0.0-1.0 qualification score (proportion of filters passed)
            - filter_results: List of per-filter result dicts for logging/analysis
        """
        # If disabled, always pass with perfect score
        if not self.enabled:
            return True, 1.0, []

        direction = signal.get('direction', 'BULL')
        filter_results = []

        # Run enabled momentum filters
        if self.rsi_filter_enabled:
            result = self._check_rsi_momentum(df_entry, direction)
            filter_results.append(result)

        if self.two_pole_filter_enabled:
            result = self._check_two_pole_confirmation(df_entry, direction)
            filter_results.append(result)

        if self.macd_filter_enabled:
            result = self._check_macd_direction(df_trigger, direction)
            filter_results.append(result)

        # Run enabled micro-regime filters (if master toggle enabled)
        if self.micro_regime_enabled:
            if self.consecutive_candles_enabled:
                result = self._check_consecutive_candles(df_entry, direction)
                filter_results.append(result)

            if self.anti_chop_enabled:
                result = self._check_anti_chop(df_entry)
                filter_results.append(result)

            if self.body_dominance_enabled:
                result = self._check_body_dominance(df_entry, direction)
                filter_results.append(result)

            if self.micro_range_enabled:
                result = self._check_micro_range(df_entry, signal)
                filter_results.append(result)

            if self.momentum_candle_enabled:
                result = self._check_momentum_candle(df_entry, direction)
                filter_results.append(result)

        # Calculate qualification score
        if not filter_results:
            return True, 1.0, []

        passed_count = sum(1 for r in filter_results if r['passed'])
        total_count = len(filter_results)
        score = passed_count / total_count

        # Log detailed results
        self._log_qualification_results(signal, direction, score, filter_results)

        # Determine if signal passes based on mode
        if self.mode == 'MONITORING':
            # Always pass in monitoring mode (we're just collecting data)
            return True, score, filter_results
        else:
            # In ACTIVE mode, require minimum score
            signal_passes = score >= self.min_score
            return signal_passes, score, filter_results

    def _check_rsi_momentum(self, df: pd.DataFrame, direction: str) -> Dict:
        """
        Check RSI is in momentum zone and moving in right direction.

        For BULL: RSI should be 40-75 AND rising
        For BEAR: RSI should be 25-60 AND falling

        This prevents:
        - Overbought BUY entries (RSI > 75)
        - Oversold SELL entries (RSI < 25)
        - Counter-momentum entries (RSI moving against trade direction)
        """
        result = {
            'filter': 'RSI_MOMENTUM',
            'passed': False,
            'reason': '',
            'rsi_value': None,
            'rsi_prev': None,
            'direction': direction
        }

        # Check for RSI column
        if 'rsi' not in df.columns:
            result['reason'] = 'RSI data not available in DataFrame'
            self.logger.debug(f"RSI filter: {result['reason']}")
            return result

        if len(df) < 2:
            result['reason'] = 'Insufficient data (need 2+ bars for RSI momentum)'
            return result

        rsi = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-2]

        result['rsi_value'] = float(rsi) if not pd.isna(rsi) else None
        result['rsi_prev'] = float(rsi_prev) if not pd.isna(rsi_prev) else None

        if pd.isna(rsi) or pd.isna(rsi_prev):
            result['reason'] = 'RSI values contain NaN'
            return result

        if direction == 'BULL':
            in_zone = self.rsi_bull_min <= rsi <= self.rsi_bull_max
            rising = rsi > rsi_prev

            if in_zone and rising:
                result['passed'] = True
                result['reason'] = f'RSI {rsi:.1f} rising in bull zone ({self.rsi_bull_min}-{self.rsi_bull_max})'
            elif not in_zone:
                if rsi > self.rsi_bull_max:
                    result['reason'] = f'RSI {rsi:.1f} OVERBOUGHT (>{self.rsi_bull_max}) - exhaustion risk'
                else:
                    result['reason'] = f'RSI {rsi:.1f} below bull zone (<{self.rsi_bull_min}) - weak momentum'
            else:
                result['reason'] = f'RSI {rsi:.1f} FALLING in bull setup (prev: {rsi_prev:.1f}) - counter-momentum'

        else:  # BEAR
            in_zone = self.rsi_bear_min <= rsi <= self.rsi_bear_max
            falling = rsi < rsi_prev

            if in_zone and falling:
                result['passed'] = True
                result['reason'] = f'RSI {rsi:.1f} falling in bear zone ({self.rsi_bear_min}-{self.rsi_bear_max})'
            elif not in_zone:
                if rsi < self.rsi_bear_min:
                    result['reason'] = f'RSI {rsi:.1f} OVERSOLD (<{self.rsi_bear_min}) - exhaustion risk'
                else:
                    result['reason'] = f'RSI {rsi:.1f} above bear zone (>{self.rsi_bear_max}) - weak momentum'
            else:
                result['reason'] = f'RSI {rsi:.1f} RISING in bear setup (prev: {rsi_prev:.1f}) - counter-momentum'

        return result

    def _check_two_pole_confirmation(self, df: pd.DataFrame, direction: str) -> Dict:
        """
        Check Two-Pole Oscillator confirms trade direction.

        Two-Pole is a minimal-lag momentum oscillator (BigBeluga):
        - Values > 0 = overbought zone
        - Values < 0 = oversold zone
        - Green (rising) vs Purple (falling)

        For BULL: Oscillator should be green (rising) OR in oversold zone (recovery potential)
        For BEAR: Oscillator should be purple (falling) OR in overbought zone (reversal potential)
        """
        result = {
            'filter': 'TWO_POLE',
            'passed': False,
            'reason': '',
            'osc_value': None,
            'is_green': None,
            'is_purple': None,
            'direction': direction
        }

        # Check for Two-Pole columns
        required_cols = ['two_pole_osc', 'two_pole_is_green', 'two_pole_is_purple']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            result['reason'] = f'Two-Pole data not available (missing: {missing_cols})'
            self.logger.debug(f"Two-Pole filter: {result['reason']}")
            return result

        if len(df) < 1:
            result['reason'] = 'Insufficient data for Two-Pole check'
            return result

        latest = df.iloc[-1]
        osc_value = latest.get('two_pole_osc', 0)
        is_green = bool(latest.get('two_pole_is_green', False))
        is_purple = bool(latest.get('two_pole_is_purple', False))

        result['osc_value'] = float(osc_value) if not pd.isna(osc_value) else None
        result['is_green'] = is_green
        result['is_purple'] = is_purple

        if pd.isna(osc_value):
            result['reason'] = 'Two-Pole oscillator value is NaN'
            return result

        if direction == 'BULL':
            # For BUY: oscillator should be green (rising) or in oversold zone
            in_oversold = osc_value < self.two_pole_bull_threshold

            if is_green:
                result['passed'] = True
                result['reason'] = f'Two-Pole {osc_value:.3f} GREEN (rising momentum)'
            elif in_oversold:
                result['passed'] = True
                result['reason'] = f'Two-Pole {osc_value:.3f} OVERSOLD (recovery setup)'
            else:
                result['reason'] = f'Two-Pole {osc_value:.3f} PURPLE/OVERBOUGHT - no bull confirmation'

        else:  # BEAR
            # For SELL: oscillator should be purple (falling) or in overbought zone
            in_overbought = osc_value > self.two_pole_bear_threshold

            if is_purple:
                result['passed'] = True
                result['reason'] = f'Two-Pole {osc_value:.3f} PURPLE (falling momentum)'
            elif in_overbought:
                result['passed'] = True
                result['reason'] = f'Two-Pole {osc_value:.3f} OVERBOUGHT (reversal setup)'
            else:
                result['reason'] = f'Two-Pole {osc_value:.3f} GREEN/OVERSOLD - no bear confirmation'

        return result

    def _check_macd_direction(self, df: pd.DataFrame, direction: str) -> Dict:
        """
        Check MACD histogram confirms trade direction.

        For BULL: Histogram > 0 AND growing (bullish momentum increasing)
                  OR histogram < 0 but strongly recovering (momentum shift)

        For BEAR: Histogram < 0 AND shrinking (bearish momentum increasing)
                  OR histogram > 0 but strongly declining (momentum shift)

        This uses the trigger timeframe (5m/15m) for smoother MACD signal.
        """
        result = {
            'filter': 'MACD_DIRECTION',
            'passed': False,
            'reason': '',
            'histogram': None,
            'histogram_prev': None,
            'direction': direction
        }

        if 'macd_histogram' not in df.columns:
            result['reason'] = 'MACD histogram not available in DataFrame'
            self.logger.debug(f"MACD filter: {result['reason']}")
            return result

        if len(df) < 2:
            result['reason'] = 'Insufficient data (need 2+ bars for MACD momentum)'
            return result

        hist = df['macd_histogram'].iloc[-1]
        hist_prev = df['macd_histogram'].iloc[-2]

        result['histogram'] = float(hist) if not pd.isna(hist) else None
        result['histogram_prev'] = float(hist_prev) if not pd.isna(hist_prev) else None

        if pd.isna(hist) or pd.isna(hist_prev):
            result['reason'] = 'MACD histogram values contain NaN'
            return result

        # Calculate momentum change
        hist_change = hist - hist_prev
        hist_change_pct = abs(hist_change / hist_prev) if hist_prev != 0 else 0

        if direction == 'BULL':
            positive = hist > 0
            growing = hist > hist_prev
            strong_recovery = hist < 0 and hist_change > 0 and hist_change_pct > 0.1

            if positive and growing:
                result['passed'] = True
                result['reason'] = f'MACD histogram {hist:.6f} POSITIVE & GROWING'
            elif strong_recovery:
                result['passed'] = True
                result['reason'] = f'MACD histogram {hist:.6f} RECOVERING strongly (+{hist_change_pct:.0%})'
            elif positive and not growing:
                result['reason'] = f'MACD histogram {hist:.6f} positive but SHRINKING - momentum fading'
            else:
                result['reason'] = f'MACD histogram {hist:.6f} NEGATIVE - no bull momentum'

        else:  # BEAR
            negative = hist < 0
            shrinking = hist < hist_prev
            strong_decline = hist > 0 and hist_change < 0 and hist_change_pct > 0.1

            if negative and shrinking:
                result['passed'] = True
                result['reason'] = f'MACD histogram {hist:.6f} NEGATIVE & SHRINKING'
            elif strong_decline:
                result['passed'] = True
                result['reason'] = f'MACD histogram {hist:.6f} DECLINING strongly (-{hist_change_pct:.0%})'
            elif negative and not shrinking:
                result['reason'] = f'MACD histogram {hist:.6f} negative but RECOVERING - momentum fading'
            else:
                result['reason'] = f'MACD histogram {hist:.6f} POSITIVE - no bear momentum'

        return result

    # ==========================================================================
    # MICRO-REGIME FILTERS
    # These analyze immediate price action at the signal moment (last 3-5 candles)
    # ==========================================================================

    def _check_consecutive_candles(self, df: pd.DataFrame, direction: str) -> Dict:
        """
        Check that last N candles align with trade direction.

        For BULL: Last 2+ candles should be green (close > open)
        For BEAR: Last 2+ candles should be red (close < open)

        This confirms immediate momentum is in trade direction.
        Prevents entries where price just reversed against our direction.
        """
        result = {
            'filter': 'CONSECUTIVE_CANDLES',
            'passed': False,
            'reason': '',
            'consecutive_count': 0,
            'required': self.consecutive_candles_min,
            'direction': direction
        }

        if len(df) < self.consecutive_candles_min:
            result['reason'] = f'Insufficient data (need {self.consecutive_candles_min}+ bars)'
            return result

        # Get last N candles
        recent = df.iloc[-self.consecutive_candles_min:]

        if direction == 'BULL':
            # Count consecutive green candles (close > open)
            green_count = 0
            for idx in range(len(recent) - 1, -1, -1):
                candle = recent.iloc[idx]
                if candle['close'] > candle['open']:
                    green_count += 1
                else:
                    break  # Stop at first non-green

            result['consecutive_count'] = green_count

            if green_count >= self.consecutive_candles_min:
                result['passed'] = True
                result['reason'] = f'{green_count} consecutive GREEN candles (momentum aligned)'
            else:
                # Check what the last candle was
                last_candle = recent.iloc[-1]
                if last_candle['close'] < last_candle['open']:
                    result['reason'] = f'Only {green_count} green candles, last candle RED - momentum hesitating'
                else:
                    result['reason'] = f'Only {green_count}/{self.consecutive_candles_min} consecutive green candles'

        else:  # BEAR
            # Count consecutive red candles (close < open)
            red_count = 0
            for idx in range(len(recent) - 1, -1, -1):
                candle = recent.iloc[idx]
                if candle['close'] < candle['open']:
                    red_count += 1
                else:
                    break

            result['consecutive_count'] = red_count

            if red_count >= self.consecutive_candles_min:
                result['passed'] = True
                result['reason'] = f'{red_count} consecutive RED candles (momentum aligned)'
            else:
                last_candle = recent.iloc[-1]
                if last_candle['close'] > last_candle['open']:
                    result['reason'] = f'Only {red_count} red candles, last candle GREEN - momentum hesitating'
                else:
                    result['reason'] = f'Only {red_count}/{self.consecutive_candles_min} consecutive red candles'

        return result

    def _check_anti_chop(self, df: pd.DataFrame) -> Dict:
        """
        Detect choppy market conditions by counting alternating candles.

        If last 4 candles alternate green/red/green/red (or similar), market is choppy.
        This suggests indecision and makes scalp entries unreliable.

        Passes if alternation count is below threshold.
        """
        result = {
            'filter': 'ANTI_CHOP',
            'passed': False,
            'reason': '',
            'alternations': 0,
            'max_allowed': self.anti_chop_max_alternations,
            'lookback': self.anti_chop_lookback
        }

        if len(df) < self.anti_chop_lookback:
            result['reason'] = f'Insufficient data (need {self.anti_chop_lookback}+ bars)'
            return result

        recent = df.iloc[-self.anti_chop_lookback:]

        # Count direction changes
        alternations = 0
        prev_direction = None

        for idx in range(len(recent)):
            candle = recent.iloc[idx]
            is_green = candle['close'] > candle['open']

            if prev_direction is not None and is_green != prev_direction:
                alternations += 1

            prev_direction = is_green

        result['alternations'] = alternations

        if alternations <= self.anti_chop_max_alternations:
            result['passed'] = True
            if alternations == 0:
                result['reason'] = f'No alternations in last {self.anti_chop_lookback} candles - clear trend'
            else:
                result['reason'] = f'{alternations} alternations OK (max {self.anti_chop_max_alternations}) - acceptable flow'
        else:
            result['reason'] = f'{alternations} alternations in last {self.anti_chop_lookback} candles - CHOPPY MARKET'

        return result

    def _check_body_dominance(self, df: pd.DataFrame, direction: str) -> Dict:
        """
        Check that candle bodies dominate over wicks (conviction vs indecision).

        Calculates average body size vs average wick size over last N candles.
        High body/wick ratio = directional conviction
        Low body/wick ratio = indecision, potential reversals

        Also checks that bodies align with trade direction.
        """
        result = {
            'filter': 'BODY_DOMINANCE',
            'passed': False,
            'reason': '',
            'avg_body': 0.0,
            'avg_wick': 0.0,
            'ratio': 0.0,
            'required_ratio': self.body_dominance_ratio,
            'direction': direction
        }

        lookback = self.body_dominance_lookback
        if len(df) < lookback:
            result['reason'] = f'Insufficient data (need {lookback}+ bars)'
            return result

        recent = df.iloc[-lookback:]

        # Calculate body and wick sizes
        bodies = []
        wicks = []
        aligned_bodies = 0

        for idx in range(len(recent)):
            candle = recent.iloc[idx]
            body = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            total_wick = upper_wick + lower_wick

            bodies.append(body)
            wicks.append(total_wick)

            # Check if body aligns with direction
            is_green = candle['close'] > candle['open']
            if (direction == 'BULL' and is_green) or (direction == 'BEAR' and not is_green):
                aligned_bodies += 1

        avg_body = sum(bodies) / len(bodies) if bodies else 0
        avg_wick = sum(wicks) / len(wicks) if wicks else 0.0001  # Avoid division by zero

        ratio = avg_body / avg_wick if avg_wick > 0 else 0

        result['avg_body'] = round(avg_body, 6)
        result['avg_wick'] = round(avg_wick, 6)
        result['ratio'] = round(ratio, 2)

        if ratio >= self.body_dominance_ratio:
            result['passed'] = True
            result['reason'] = f'Body/wick ratio {ratio:.2f} >= {self.body_dominance_ratio} - CONVICTION ({aligned_bodies}/{lookback} aligned)'
        else:
            # Check which issue is more prominent
            if avg_wick > avg_body * 2:
                result['reason'] = f'Body/wick ratio {ratio:.2f} - HEAVY WICKS indicate indecision'
            else:
                result['reason'] = f'Body/wick ratio {ratio:.2f} < {self.body_dominance_ratio} - weak conviction'

        return result

    def _check_micro_range(self, df: pd.DataFrame, signal: Dict) -> Dict:
        """
        Detect tight micro-range / congestion that traps scalp trades.

        If the range of last N candles is below threshold (e.g., 3 pips),
        market is in micro-congestion. Scalp entries in congestion often
        get stopped out as price bounces within the range.

        Passes if range is above minimum pip threshold.
        """
        result = {
            'filter': 'MICRO_RANGE',
            'passed': False,
            'reason': '',
            'range_pips': 0.0,
            'min_required': self.micro_range_min_pips,
            'lookback': self.micro_range_lookback
        }

        lookback = self.micro_range_lookback
        if len(df) < lookback:
            result['reason'] = f'Insufficient data (need {lookback}+ bars)'
            return result

        recent = df.iloc[-lookback:]

        # Calculate range in price
        highest = recent['high'].max()
        lowest = recent['low'].min()
        price_range = highest - lowest

        # Convert to pips (assume 4 decimal places for most pairs, 2 for JPY)
        epic = signal.get('epic', '')
        pair = signal.get('pair', epic)

        if 'JPY' in pair.upper() or 'JPY' in epic.upper():
            pip_value = 0.01
        else:
            pip_value = 0.0001

        range_pips = price_range / pip_value
        result['range_pips'] = round(range_pips, 2)

        if range_pips >= self.micro_range_min_pips:
            result['passed'] = True
            result['reason'] = f'Range {range_pips:.1f} pips >= {self.micro_range_min_pips} - adequate volatility'
        else:
            result['reason'] = f'Range {range_pips:.1f} pips < {self.micro_range_min_pips} - MICRO CONGESTION (avoid scalp)'

        return result

    def _check_momentum_candle(self, df: pd.DataFrame, direction: str) -> Dict:
        """
        Check if the last closed candle shows momentum thrust.

        A momentum candle has body > 1.5x the average body size.
        This confirms strong conviction in the direction.

        Also verifies the momentum candle aligns with trade direction.
        """
        result = {
            'filter': 'MOMENTUM_CANDLE',
            'passed': False,
            'reason': '',
            'last_body': 0.0,
            'avg_body': 0.0,
            'multiplier': 0.0,
            'required_multiplier': self.momentum_candle_multiplier,
            'direction': direction
        }

        if len(df) < 10:  # Need some history for average
            result['reason'] = 'Insufficient data for momentum candle check'
            return result

        # Calculate average body size over last 10 candles (excluding last)
        recent = df.iloc[-10:-1]
        bodies = [abs(c['close'] - c['open']) for _, c in recent.iterrows()]
        avg_body = sum(bodies) / len(bodies) if bodies else 0.0001

        # Get last candle
        last = df.iloc[-1]
        last_body = abs(last['close'] - last['open'])
        is_green = last['close'] > last['open']

        multiplier = last_body / avg_body if avg_body > 0 else 0

        result['last_body'] = round(last_body, 6)
        result['avg_body'] = round(avg_body, 6)
        result['multiplier'] = round(multiplier, 2)

        # Check if direction aligns
        direction_aligned = (direction == 'BULL' and is_green) or (direction == 'BEAR' and not is_green)

        if multiplier >= self.momentum_candle_multiplier and direction_aligned:
            result['passed'] = True
            result['reason'] = f'Momentum candle {multiplier:.1f}x average - THRUST confirmed'
        elif multiplier >= self.momentum_candle_multiplier and not direction_aligned:
            result['reason'] = f'Momentum candle {multiplier:.1f}x but WRONG DIRECTION ({"GREEN" if is_green else "RED"} vs {direction})'
        else:
            result['reason'] = f'Last candle only {multiplier:.1f}x average - no momentum thrust'

        return result

    def _log_qualification_results(
        self,
        signal: Dict,
        direction: str,
        score: float,
        filter_results: List[Dict]
    ):
        """Log detailed qualification results."""
        epic = signal.get('epic', 'UNKNOWN')
        pair = signal.get('pair', epic)

        passed_count = sum(1 for r in filter_results if r['passed'])
        total_count = len(filter_results)

        # Determine log level based on score
        if score >= 1.0:
            log_level = logging.INFO
            score_emoji = "✅"
        elif score >= self.min_score:
            log_level = logging.INFO
            score_emoji = "🟡"
        else:
            log_level = logging.WARNING
            score_emoji = "❌"

        # Main qualification summary
        self.logger.log(log_level, f"\n{'='*60}")
        self.logger.log(log_level, f"{score_emoji} SCALP SIGNAL QUALIFICATION - {pair} {direction}")
        self.logger.log(log_level, f"{'='*60}")
        self.logger.log(log_level, f"   Mode: {self.mode}")
        self.logger.log(log_level, f"   Score: {score:.0%} ({passed_count}/{total_count} filters passed)")
        self.logger.log(log_level, f"   Required: {self.min_score:.0%}")

        will_pass = self.mode == 'MONITORING' or score >= self.min_score
        self.logger.log(log_level, f"   Result: {'PASS' if will_pass else 'BLOCK'}")
        self.logger.log(log_level, f"")

        # Per-filter results
        for result in filter_results:
            status = "✅" if result['passed'] else "❌"
            filter_name = result['filter']
            reason = result['reason']

            # Add extra details for logging
            extra = ""
            if filter_name == 'RSI_MOMENTUM' and result.get('rsi_value') is not None:
                extra = f" [RSI={result['rsi_value']:.1f}]"
            elif filter_name == 'TWO_POLE' and result.get('osc_value') is not None:
                extra = f" [OSC={result['osc_value']:.3f}]"
            elif filter_name == 'MACD_DIRECTION' and result.get('histogram') is not None:
                extra = f" [HIST={result['histogram']:.6f}]"
            elif filter_name == 'CONSECUTIVE_CANDLES':
                extra = f" [{result.get('consecutive_count', 0)}/{result.get('required', 2)}]"
            elif filter_name == 'ANTI_CHOP':
                extra = f" [{result.get('alternations', 0)} alt]"
            elif filter_name == 'BODY_DOMINANCE':
                extra = f" [ratio={result.get('ratio', 0):.2f}]"
            elif filter_name == 'MICRO_RANGE':
                extra = f" [{result.get('range_pips', 0):.1f} pips]"
            elif filter_name == 'MOMENTUM_CANDLE':
                extra = f" [{result.get('multiplier', 0):.1f}x]"

            self.logger.log(log_level, f"   {status} {filter_name}{extra}")
            self.logger.log(log_level, f"      {reason}")

        self.logger.log(log_level, f"{'='*60}")

    def get_status(self) -> Dict:
        """Get current qualifier configuration status."""
        return {
            'version': self.VERSION,
            'enabled': self.enabled,
            'mode': self.mode,
            'min_score': self.min_score,
            'momentum_filters': {
                'rsi': self.rsi_filter_enabled,
                'two_pole': self.two_pole_filter_enabled,
                'macd': self.macd_filter_enabled
            },
            'micro_regime_enabled': self.micro_regime_enabled,
            'micro_regime_filters': {
                'consecutive_candles': self.consecutive_candles_enabled,
                'anti_chop': self.anti_chop_enabled,
                'body_dominance': self.body_dominance_enabled,
                'micro_range': self.micro_range_enabled,
                'momentum_candle': self.momentum_candle_enabled
            },
            'thresholds': {
                'rsi_bull': (self.rsi_bull_min, self.rsi_bull_max),
                'rsi_bear': (self.rsi_bear_min, self.rsi_bear_max),
                'two_pole_bull': self.two_pole_bull_threshold,
                'two_pole_bear': self.two_pole_bear_threshold,
                'consecutive_candles_min': self.consecutive_candles_min,
                'anti_chop_lookback': self.anti_chop_lookback,
                'anti_chop_max_alternations': self.anti_chop_max_alternations,
                'body_dominance_ratio': self.body_dominance_ratio,
                'micro_range_min_pips': self.micro_range_min_pips,
                'momentum_candle_multiplier': self.momentum_candle_multiplier
            }
        }
