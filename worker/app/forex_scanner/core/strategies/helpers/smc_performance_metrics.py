"""
SMC Simple Strategy - Enhanced Performance Metrics Calculator
==============================================================

Calculates additional performance metrics for trade analysis without
modifying the core signal detection logic.

Metrics Calculated:
- Kaufman Efficiency Ratio (ER) - trend quality measurement
- Market Regime Classification - trending/ranging/breakout/high_vol
- Bollinger Band Width Percentile - volatility context
- Entry Quality Score - distance from optimal Fib zone
- Multi-Timeframe Confluence - alignment across timeframes
- Volume Profile Metrics - volume quality at key points

Author: Trading System V1
Version: 1.0.0
Created: 2025-01-01
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """
    Enhanced performance metrics for SMC Simple signals.

    These metrics are calculated for BOTH accepted and rejected signals
    to enable comprehensive analysis of trade quality factors.
    """
    # Kaufman Efficiency Ratio
    efficiency_ratio: Optional[float] = None  # 0.0 (choppy) to 1.0 (trending)
    er_period: int = 10  # Period used for ER calculation

    # Market Regime
    market_regime: str = "unknown"  # trending, ranging, breakout, high_volatility
    regime_confidence: Optional[float] = None  # 0-1 confidence in regime classification

    # Volatility Context
    bb_width_percentile: Optional[float] = None  # Current BB width vs 50-period history
    atr_percentile: Optional[float] = None  # Current ATR vs 20-period history
    volatility_state: str = "normal"  # low, normal, high, extreme

    # Entry Quality
    entry_quality_score: Optional[float] = None  # 0-1 score based on Fib zone accuracy
    distance_from_optimal_fib: Optional[float] = None  # Distance from 38.2-50% zone
    entry_candle_momentum: Optional[float] = None  # Body as % of range (0-1)

    # Multi-Timeframe Confluence
    mtf_confluence_score: Optional[float] = None  # 0-1 alignment score
    htf_candle_position: str = "unknown"  # start, middle, end of 4H candle
    all_timeframes_aligned: bool = False

    # Volume Profile
    volume_at_swing_break: Optional[float] = None  # Volume ratio at Tier 2
    volume_trend: str = "unknown"  # increasing, decreasing, stable
    volume_quality_score: Optional[float] = None  # 0-1 overall volume quality

    # ADX Components (for regime detection)
    adx_value: Optional[float] = None
    adx_plus_di: Optional[float] = None
    adx_minus_di: Optional[float] = None
    adx_trend_strength: str = "unknown"  # weak, moderate, strong

    # Timestamp
    calculated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        result = asdict(self)
        result['calculated_at'] = datetime.utcnow().isoformat()
        return result

    def to_json_safe(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary (handles numpy types)"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, (np.integer, np.floating)):
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif value is None or isinstance(value, (str, int, float, bool)):
                result[key] = value
            else:
                result[key] = str(value)
        result['calculated_at'] = datetime.utcnow().isoformat()
        return result


class SMCPerformanceMetricsCalculator:
    """
    Calculates enhanced performance metrics for SMC Simple strategy.

    This calculator is designed to be called AFTER signal detection
    to enrich signals with additional analysis data without affecting
    the core signal logic.

    Usage:
        calculator = SMCPerformanceMetricsCalculator()
        metrics = calculator.calculate_metrics(
            df_5m=df_5m,
            df_15m=df_15m,
            df_4h=df_4h,
            signal_data=signal_dict,
            epic=epic
        )
        signal['performance_metrics'] = metrics.to_dict()
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Configuration
        self.er_period = 10  # Efficiency Ratio lookback
        self.bb_lookback = 50  # Bollinger Band percentile lookback
        self.atr_percentile_lookback = 20  # ATR percentile lookback
        self.volume_trend_lookback = 10  # Volume trend analysis bars

        # Optimal Fib zone (from SMC Simple config)
        self.fib_optimal_min = 0.382
        self.fib_optimal_max = 0.500

        self.logger.debug("SMCPerformanceMetricsCalculator initialized")

    def calculate_metrics(
        self,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        signal_data: Optional[Dict] = None,
        epic: str = ""
    ) -> PerformanceMetrics:
        """
        Calculate all enhanced performance metrics.

        Args:
            df_5m: 5-minute OHLCV DataFrame
            df_15m: 15-minute OHLCV DataFrame
            df_4h: 4-hour OHLCV DataFrame
            signal_data: Signal dictionary from detect_signal()
            epic: Trading pair epic code

        Returns:
            PerformanceMetrics dataclass with all calculated metrics
        """
        metrics = PerformanceMetrics()

        try:
            # Use the most granular available timeframe for calculations
            df_primary = df_5m if df_5m is not None and len(df_5m) > 0 else df_15m

            if df_primary is None or len(df_primary) < 20:
                self.logger.warning(f"Insufficient data for metrics calculation: {epic}")
                return metrics

            # 1. Kaufman Efficiency Ratio
            metrics.efficiency_ratio, metrics.er_period = self._calculate_efficiency_ratio(df_primary)

            # 2. Market Regime Detection
            regime_result = self._detect_market_regime(df_primary, df_4h)
            metrics.market_regime = regime_result['regime']
            metrics.regime_confidence = regime_result['confidence']
            metrics.adx_value = regime_result.get('adx')
            metrics.adx_plus_di = regime_result.get('plus_di')
            metrics.adx_minus_di = regime_result.get('minus_di')
            metrics.adx_trend_strength = regime_result.get('trend_strength', 'unknown')

            # 3. Volatility Context
            vol_result = self._calculate_volatility_context(df_primary)
            metrics.bb_width_percentile = vol_result['bb_percentile']
            metrics.atr_percentile = vol_result['atr_percentile']
            metrics.volatility_state = vol_result['state']

            # 4. Entry Quality (if signal data provided)
            if signal_data:
                entry_result = self._calculate_entry_quality(df_primary, signal_data)
                metrics.entry_quality_score = entry_result['score']
                metrics.distance_from_optimal_fib = entry_result['fib_distance']
                metrics.entry_candle_momentum = entry_result['candle_momentum']

            # 5. Multi-Timeframe Confluence
            if df_4h is not None and signal_data:
                mtf_result = self._calculate_mtf_confluence(df_5m, df_15m, df_4h, signal_data)
                metrics.mtf_confluence_score = mtf_result['score']
                metrics.htf_candle_position = mtf_result['htf_position']
                metrics.all_timeframes_aligned = mtf_result['aligned']

            # 6. Volume Profile
            vol_profile = self._calculate_volume_profile(df_primary, signal_data)
            metrics.volume_at_swing_break = vol_profile['at_break']
            metrics.volume_trend = vol_profile['trend']
            metrics.volume_quality_score = vol_profile['quality_score']

            self.logger.debug(
                f"Metrics calculated for {epic}: "
                f"ER={metrics.efficiency_ratio if metrics.efficiency_ratio is not None else 'N/A'}, "
                f"Regime={metrics.market_regime}, "
                f"BB%={metrics.bb_width_percentile if metrics.bb_width_percentile is not None else 'N/A'}"
            )

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")

        return metrics

    def _calculate_efficiency_ratio(
        self,
        df: pd.DataFrame,
        period: int = None
    ) -> Tuple[Optional[float], int]:
        """
        Calculate Kaufman's Efficiency Ratio.

        ER = abs(price_change) / sum(abs(price_changes))

        Values:
        - 1.0 = Perfect trend (price moved in one direction)
        - 0.0 = No net movement (choppy/ranging)
        - Typically: >0.6 = trending, <0.3 = ranging

        Returns:
            Tuple of (efficiency_ratio, period_used)
        """
        period = period or self.er_period

        if len(df) < period + 1:
            return None, period

        try:
            close = df['close'].values

            # Direction: net price change over period
            direction = abs(close[-1] - close[-period-1])

            # Volatility: sum of absolute price changes
            volatility = sum(abs(close[i] - close[i-1]) for i in range(-period, 0))

            if volatility == 0:
                return 0.0, period

            er = direction / volatility
            return round(float(er), 4), period

        except Exception as e:
            self.logger.warning(f"ER calculation error: {e}")
            return None, period

    def _detect_market_regime(
        self,
        df: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Detect current market regime using ADX and efficiency ratio.

        Regimes:
        - TRENDING: ADX > 25, ER > 0.5
        - RANGING: ADX < 20, ER < 0.3
        - BREAKOUT: ATR expanding, ADX rising
        - HIGH_VOLATILITY: ATR > 90th percentile
        """
        result = {
            'regime': 'unknown',
            'confidence': 0.0,
            'adx': None,
            'plus_di': None,
            'minus_di': None,
            'trend_strength': 'unknown'
        }

        try:
            # Calculate ADX if not present
            if 'adx' in df.columns:
                adx = float(df['adx'].iloc[-1])
            else:
                adx = self._calculate_adx(df)

            result['adx'] = round(adx, 2) if adx else None

            # Get +DI/-DI if available
            if 'plus_di' in df.columns:
                result['plus_di'] = round(float(df['plus_di'].iloc[-1]), 2)
            if 'minus_di' in df.columns:
                result['minus_di'] = round(float(df['minus_di'].iloc[-1]), 2)

            # Calculate efficiency ratio for this check
            er, _ = self._calculate_efficiency_ratio(df)

            # Calculate ATR percentile
            atr_pct = self._get_atr_percentile(df)

            # Regime classification logic
            if atr_pct and atr_pct > 90:
                result['regime'] = MarketRegime.HIGH_VOLATILITY.value
                result['confidence'] = 0.85
                result['trend_strength'] = 'extreme'
            elif adx and adx > 25 and er and er > 0.5:
                result['regime'] = MarketRegime.TRENDING.value
                result['confidence'] = min(0.9, 0.5 + (adx - 25) / 50 + er * 0.3)
                result['trend_strength'] = 'strong' if adx > 35 else 'moderate'
            elif atr_pct and atr_pct > 70 and adx and adx > 18:
                result['regime'] = MarketRegime.BREAKOUT.value
                result['confidence'] = 0.75
                result['trend_strength'] = 'moderate'
            elif adx and adx < 20 and er and er < 0.3:
                result['regime'] = MarketRegime.RANGING.value
                result['confidence'] = min(0.85, 0.5 + (20 - adx) / 40 + (0.3 - er))
                result['trend_strength'] = 'weak'
            else:
                # Default to trending with lower confidence
                result['regime'] = MarketRegime.TRENDING.value
                result['confidence'] = 0.5
                result['trend_strength'] = 'weak' if adx and adx < 25 else 'moderate'

        except Exception as e:
            self.logger.warning(f"Regime detection error: {e}")

        return result

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate ADX if not present in DataFrame"""
        if len(df) < period * 2:
            return None

        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # True Range
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    abs(high[1:] - close[:-1]),
                    abs(low[1:] - close[:-1])
                )
            )

            # +DM and -DM
            plus_dm = np.where(
                (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                np.maximum(high[1:] - high[:-1], 0),
                0
            )
            minus_dm = np.where(
                (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                np.maximum(low[:-1] - low[1:], 0),
                0
            )

            # Smoothed averages (Wilder's smoothing)
            atr = self._wilder_smooth(tr, period)
            plus_di = 100 * self._wilder_smooth(plus_dm, period) / atr
            minus_di = 100 * self._wilder_smooth(minus_dm, period) / atr

            # DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = self._wilder_smooth(dx, period)

            return float(adx[-1]) if len(adx) > 0 else None

        except Exception as e:
            self.logger.warning(f"ADX calculation error: {e}")
            return None

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing (same as EMA with alpha = 1/period)"""
        alpha = 1.0 / period
        result = np.zeros_like(data, dtype=float)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    def _get_atr_percentile(self, df: pd.DataFrame) -> Optional[float]:
        """Get current ATR as percentile of recent history"""
        if 'atr' not in df.columns or len(df) < self.atr_percentile_lookback:
            return None

        try:
            atr_values = df['atr'].dropna().values[-self.atr_percentile_lookback:]
            current_atr = atr_values[-1]
            percentile = (np.sum(atr_values < current_atr) / len(atr_values)) * 100
            return round(float(percentile), 1)
        except:
            return None

    def _calculate_volatility_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate volatility context including BB width percentile.
        """
        result = {
            'bb_percentile': None,
            'atr_percentile': None,
            'state': 'normal'
        }

        try:
            # Bollinger Band Width Percentile
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_width = df['bb_upper'] - df['bb_lower']
            else:
                # Calculate BB width
                close = df['close']
                sma = close.rolling(20).mean()
                std = close.rolling(20).std()
                bb_width = 4 * std  # 2 std above + 2 std below

            if len(bb_width.dropna()) >= self.bb_lookback:
                recent_widths = bb_width.dropna().values[-self.bb_lookback:]
                current_width = recent_widths[-1]
                percentile = (np.sum(recent_widths < current_width) / len(recent_widths)) * 100
                result['bb_percentile'] = round(float(percentile), 1)

            # ATR Percentile
            result['atr_percentile'] = self._get_atr_percentile(df)

            # Volatility State Classification
            atr_pct = result['atr_percentile']
            if atr_pct:
                if atr_pct > 90:
                    result['state'] = 'extreme'
                elif atr_pct > 70:
                    result['state'] = 'high'
                elif atr_pct < 20:
                    result['state'] = 'low'
                else:
                    result['state'] = 'normal'

        except Exception as e:
            self.logger.warning(f"Volatility context error: {e}")

        return result

    def _calculate_entry_quality(
        self,
        df: pd.DataFrame,
        signal_data: Dict
    ) -> Dict[str, Any]:
        """
        Calculate entry quality based on Fib zone accuracy and candle momentum.
        """
        result = {
            'score': None,
            'fib_distance': None,
            'candle_momentum': None
        }

        try:
            # Get pullback depth from signal
            pullback_depth = signal_data.get('pullback_depth')

            if pullback_depth is not None:
                # Distance from optimal zone (38.2% - 50%)
                if self.fib_optimal_min <= pullback_depth <= self.fib_optimal_max:
                    result['fib_distance'] = 0.0
                    fib_score = 1.0
                elif pullback_depth < self.fib_optimal_min:
                    result['fib_distance'] = round(self.fib_optimal_min - pullback_depth, 3)
                    fib_score = max(0, 1 - result['fib_distance'] * 2)
                else:
                    result['fib_distance'] = round(pullback_depth - self.fib_optimal_max, 3)
                    fib_score = max(0, 1 - result['fib_distance'] * 2)
            else:
                fib_score = 0.5  # Unknown, neutral score

            # Entry candle momentum (body as % of range)
            if len(df) > 0:
                last_candle = df.iloc[-1]
                candle_range = last_candle['high'] - last_candle['low']
                if candle_range > 0:
                    body = abs(last_candle['close'] - last_candle['open'])
                    result['candle_momentum'] = round(float(body / candle_range), 3)
                else:
                    result['candle_momentum'] = 0.0

            # Combined entry quality score
            momentum_score = result['candle_momentum'] if result['candle_momentum'] else 0.5
            result['score'] = round((fib_score * 0.6 + momentum_score * 0.4), 3)

        except Exception as e:
            self.logger.warning(f"Entry quality calculation error: {e}")

        return result

    def _calculate_mtf_confluence(
        self,
        df_5m: Optional[pd.DataFrame],
        df_15m: Optional[pd.DataFrame],
        df_4h: pd.DataFrame,
        signal_data: Dict
    ) -> Dict[str, Any]:
        """
        Calculate multi-timeframe confluence score.
        """
        result = {
            'score': 0.0,
            'htf_position': 'unknown',
            'aligned': False
        }

        try:
            signal_type = signal_data.get('signal_type', signal_data.get('signal', ''))
            is_bullish = signal_type.upper() in ['BULL', 'BUY', 'LONG']

            alignment_count = 0
            total_checks = 0

            # 4H candle position (where are we in the HTF candle?)
            if len(df_4h) > 0:
                htf_candle = df_4h.iloc[-1]
                htf_range = htf_candle['high'] - htf_candle['low']
                if htf_range > 0:
                    # Current price position in 4H candle
                    current_price = signal_data.get('entry_price', htf_candle['close'])
                    position_pct = (current_price - htf_candle['low']) / htf_range

                    if position_pct < 0.33:
                        result['htf_position'] = 'start'
                    elif position_pct < 0.67:
                        result['htf_position'] = 'middle'
                    else:
                        result['htf_position'] = 'end'

                # 4H trend alignment
                htf_bullish = htf_candle['close'] > htf_candle['open']
                if (is_bullish and htf_bullish) or (not is_bullish and not htf_bullish):
                    alignment_count += 1
                total_checks += 1

            # 15m alignment
            if df_15m is not None and len(df_15m) > 0:
                mtf_candle = df_15m.iloc[-1]
                mtf_bullish = mtf_candle['close'] > mtf_candle['open']
                if (is_bullish and mtf_bullish) or (not is_bullish and not mtf_bullish):
                    alignment_count += 1
                total_checks += 1

            # 5m alignment
            if df_5m is not None and len(df_5m) > 0:
                ltf_candle = df_5m.iloc[-1]
                ltf_bullish = ltf_candle['close'] > ltf_candle['open']
                if (is_bullish and ltf_bullish) or (not is_bullish and not ltf_bullish):
                    alignment_count += 1
                total_checks += 1

            # Calculate confluence score
            if total_checks > 0:
                result['score'] = round(alignment_count / total_checks, 3)
                result['aligned'] = alignment_count == total_checks

        except Exception as e:
            self.logger.warning(f"MTF confluence calculation error: {e}")

        return result

    def _calculate_volume_profile(
        self,
        df: pd.DataFrame,
        signal_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate volume profile metrics.
        """
        result = {
            'at_break': None,
            'trend': 'unknown',
            'quality_score': None
        }

        try:
            # Get volume column (prefer 'ltv' for IG data, fallback to 'volume')
            vol_col = 'ltv' if 'ltv' in df.columns else 'volume'

            if vol_col not in df.columns or df[vol_col].isna().all():
                return result

            volume = df[vol_col].dropna()

            if len(volume) < self.volume_trend_lookback:
                return result

            # Volume at entry (current bar)
            current_vol = volume.iloc[-1]
            vol_sma = volume.rolling(20).mean().iloc[-1]

            if vol_sma > 0:
                result['at_break'] = round(float(current_vol / vol_sma), 3)

            # Volume trend (increasing/decreasing/stable)
            recent_vol = volume.values[-self.volume_trend_lookback:]
            vol_change = (recent_vol[-1] - recent_vol[0]) / (recent_vol[0] + 1e-10)

            if vol_change > 0.2:
                result['trend'] = 'increasing'
            elif vol_change < -0.2:
                result['trend'] = 'decreasing'
            else:
                result['trend'] = 'stable'

            # Volume quality score
            # Higher volume + increasing trend = better quality
            vol_ratio = result['at_break'] if result['at_break'] else 1.0
            trend_bonus = 0.2 if result['trend'] == 'increasing' else (-0.1 if result['trend'] == 'decreasing' else 0)

            # Score: 0-1 based on volume ratio with trend adjustment
            base_score = min(1.0, vol_ratio / 2.0)  # 2.0x SMA = perfect score
            result['quality_score'] = round(max(0, min(1, base_score + trend_bonus)), 3)

        except Exception as e:
            self.logger.warning(f"Volume profile calculation error: {e}")

        return result

    def calculate_quick_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate a subset of quick metrics for rejection tracking.

        This is a lighter-weight version for use in rejection context
        where we may not have all timeframe data available.
        """
        metrics = {
            'efficiency_ratio': None,
            'market_regime': 'unknown',
            'atr_percentile': None,
            'volatility_state': 'normal'
        }

        try:
            if df is None or len(df) < 20:
                return metrics

            # Quick ER calculation
            er, _ = self._calculate_efficiency_ratio(df)
            metrics['efficiency_ratio'] = er

            # Quick regime detection
            regime = self._detect_market_regime(df)
            metrics['market_regime'] = regime['regime']

            # ATR percentile
            metrics['atr_percentile'] = self._get_atr_percentile(df)

            # Volatility state
            if metrics['atr_percentile']:
                if metrics['atr_percentile'] > 90:
                    metrics['volatility_state'] = 'extreme'
                elif metrics['atr_percentile'] > 70:
                    metrics['volatility_state'] = 'high'
                elif metrics['atr_percentile'] < 20:
                    metrics['volatility_state'] = 'low'

        except Exception as e:
            self.logger.warning(f"Quick metrics calculation error: {e}")

        return metrics


# Singleton instance for reuse
_calculator_instance: Optional[SMCPerformanceMetricsCalculator] = None


def get_performance_metrics_calculator(
    logger: Optional[logging.Logger] = None
) -> SMCPerformanceMetricsCalculator:
    """
    Get singleton instance of performance metrics calculator.

    Usage:
        from forex_scanner.core.strategies.helpers.smc_performance_metrics import (
            get_performance_metrics_calculator
        )

        calculator = get_performance_metrics_calculator()
        metrics = calculator.calculate_metrics(df_5m, df_15m, df_4h, signal_data, epic)
    """
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = SMCPerformanceMetricsCalculator(logger)
    return _calculator_instance
