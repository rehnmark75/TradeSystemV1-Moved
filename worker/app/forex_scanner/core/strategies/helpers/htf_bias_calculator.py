"""
HTF (Higher Timeframe) Bias Score Calculator

Professional approach to HTF alignment: continuous bias score instead of binary filter.

Design Principles:
1. Continuous over binary - Bias score 0.0-1.0 instead of YES/NO
2. Separation of concerns - HTF measures trend alignment only
3. Confidence multiplier - Bad alignment reduces confidence rather than rejecting
4. Single responsibility - One method calculates bias; caller decides usage

Score Interpretation:
    0.0-0.3: Strong counter-trend (significantly misaligned)
    0.3-0.5: Weak counter-trend (slightly misaligned)
    0.5-0.7: Neutral/mixed (no strong bias)
    0.7-0.9: Aligned (favorable trend)
    0.9-1.0: Strong alignment (optimal conditions)

Author: Claude Code
Version: 1.0.0 (January 2026)
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


class HTFBiasCalculator:
    """
    Calculates HTF (4H) bias alignment score for SMC Simple strategy.

    Replaces the binary HTF alignment filter with a continuous score
    that can be used as a confidence multiplier or threshold filter.
    """

    # Component weights (must sum to 1.0)
    WEIGHT_CANDLE_BODY = 0.40  # 40% weight on candle body direction/strength
    WEIGHT_EMA_SLOPE = 0.30    # 30% weight on EMA trend slope
    WEIGHT_MACD_MOMENTUM = 0.30  # 30% weight on MACD histogram

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_bias_score(
        self,
        df_4h: pd.DataFrame,
        direction: str,
        epic: str = ""
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate HTF bias alignment score.

        Args:
            df_4h: 4-hour DataFrame with OHLC, EMA, MACD data
            direction: Signal direction ('BULL' or 'BEAR')
            epic: Epic identifier for logging

        Returns:
            Tuple of (bias_score, component_details)
            - bias_score: 0.0 to 1.0
            - component_details: Dictionary with individual component scores
        """
        if df_4h is None or len(df_4h) < 5:
            self.logger.debug(f"[HTF BIAS] {epic}: Insufficient 4H data, returning neutral")
            return 0.5, {'error': 'insufficient_data', 'candles_available': len(df_4h) if df_4h is not None else 0}

        direction_normalized = direction.upper()
        if direction_normalized not in ('BULL', 'BEAR'):
            self.logger.warning(f"[HTF BIAS] {epic}: Invalid direction '{direction}', returning neutral")
            return 0.5, {'error': 'invalid_direction'}

        # Calculate component scores
        candle_score, candle_details = self._score_candle_bodies(df_4h, direction_normalized)
        ema_score, ema_details = self._score_ema_slope(df_4h, direction_normalized)
        macd_score, macd_details = self._score_macd_momentum(df_4h, direction_normalized)

        # Weighted combination
        final_score = (
            candle_score * self.WEIGHT_CANDLE_BODY +
            ema_score * self.WEIGHT_EMA_SLOPE +
            macd_score * self.WEIGHT_MACD_MOMENTUM
        )

        # Clamp to valid range
        final_score = max(0.0, min(1.0, final_score))
        final_score = round(final_score, 3)

        # Build details dict
        details = {
            'direction': direction_normalized,
            'final_score': final_score,
            'components': {
                'candle_body': {
                    'score': round(candle_score, 3),
                    'weight': self.WEIGHT_CANDLE_BODY,
                    'weighted': round(candle_score * self.WEIGHT_CANDLE_BODY, 3),
                    **candle_details
                },
                'ema_slope': {
                    'score': round(ema_score, 3),
                    'weight': self.WEIGHT_EMA_SLOPE,
                    'weighted': round(ema_score * self.WEIGHT_EMA_SLOPE, 3),
                    **ema_details
                },
                'macd_momentum': {
                    'score': round(macd_score, 3),
                    'weight': self.WEIGHT_MACD_MOMENTUM,
                    'weighted': round(macd_score * self.WEIGHT_MACD_MOMENTUM, 3),
                    **macd_details
                }
            },
            'interpretation': self._interpret_score(final_score, direction_normalized)
        }

        self.logger.debug(
            f"[HTF BIAS] {epic}: score={final_score:.3f} "
            f"(candle={candle_score:.2f}, ema={ema_score:.2f}, macd={macd_score:.2f}) "
            f"→ {details['interpretation']}"
        )

        return final_score, details

    def _score_candle_bodies(
        self,
        df_4h: pd.DataFrame,
        direction: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score based on 4H candle body analysis.

        Measures both direction AND strength (body size relative to ATR).
        A 50-pip bullish candle is different from a 2-pip bullish candle.
        """
        details: Dict[str, Any] = {}

        try:
            # Get last closed candle (iloc[-2]) and previous (iloc[-3])
            curr_open = df_4h['open'].iloc[-2]
            curr_close = df_4h['close'].iloc[-2]
            prev_open = df_4h['open'].iloc[-3]
            prev_close = df_4h['close'].iloc[-3]

            # Calculate ATR for normalization
            if 'atr' in df_4h.columns and not pd.isna(df_4h['atr'].iloc[-2]):
                atr = df_4h['atr'].iloc[-2]
            else:
                # Fallback: average range of last 5 candles
                atr = (df_4h['high'] - df_4h['low']).iloc[-5:].mean()

            if atr <= 0:
                atr = 0.0001  # Prevent division by zero

            # Calculate body sizes normalized by ATR
            # Positive = bullish, Negative = bearish
            curr_body_raw = curr_close - curr_open
            prev_body_raw = prev_close - prev_open
            curr_body_norm = curr_body_raw / atr
            prev_body_norm = prev_body_raw / atr

            details['curr_candle'] = {
                'direction': 'BULLISH' if curr_body_raw > 0 else 'BEARISH' if curr_body_raw < 0 else 'NEUTRAL',
                'body_pips': curr_body_raw,
                'body_atr_ratio': round(curr_body_norm, 3)
            }
            details['prev_candle'] = {
                'direction': 'BULLISH' if prev_body_raw > 0 else 'BEARISH' if prev_body_raw < 0 else 'NEUTRAL',
                'body_pips': prev_body_raw,
                'body_atr_ratio': round(prev_body_norm, 3)
            }

            # Score current candle (contributes 0-0.6 of component score)
            # Strong aligned candle = 0.6, strong opposing = 0.0, neutral = 0.3
            if direction == 'BULL':
                curr_alignment = curr_body_norm  # Positive for bullish
            else:
                curr_alignment = -curr_body_norm  # Negative for bearish (inverted)

            # Map alignment to 0-0.6 score
            # Alignment > 0.5 ATR = max score, < -0.5 ATR = min score
            curr_score = 0.3 + (curr_alignment * 0.6)
            curr_score = max(0.0, min(0.6, curr_score))

            # Score previous candle (contributes 0-0.4 of component score)
            if direction == 'BULL':
                prev_alignment = prev_body_norm
            else:
                prev_alignment = -prev_body_norm

            prev_score = 0.2 + (prev_alignment * 0.4)
            prev_score = max(0.0, min(0.4, prev_score))

            total_score = curr_score + prev_score

            details['curr_component_score'] = round(curr_score, 3)
            details['prev_component_score'] = round(prev_score, 3)

            return total_score, details

        except Exception as e:
            self.logger.debug(f"[HTF BIAS] Candle body scoring error: {e}")
            return 0.5, {'error': str(e)}

    def _score_ema_slope(
        self,
        df_4h: pd.DataFrame,
        direction: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score based on 4H EMA slope direction and strength.

        A rising EMA with accelerating slope is stronger than a flat EMA.
        """
        details: Dict[str, Any] = {}

        # Try different EMA columns
        ema_col = None
        for col in ['ema_50', 'ema_21', 'ema_20', 'ema']:
            if col in df_4h.columns:
                ema_col = col
                break

        if ema_col is None or len(df_4h) < 5:
            details['error'] = 'no_ema_data'
            return 0.5, details

        try:
            # Calculate EMA slope over last 3 candles
            ema_current = df_4h[ema_col].iloc[-2]
            ema_prev = df_4h[ema_col].iloc[-4]  # 2 candles ago

            if pd.isna(ema_current) or pd.isna(ema_prev) or ema_prev == 0:
                return 0.5, {'error': 'invalid_ema_values'}

            # Calculate percentage slope
            slope_pct = ((ema_current - ema_prev) / ema_prev) * 100

            details['ema_column'] = ema_col
            details['ema_current'] = round(ema_current, 5)
            details['ema_prev'] = round(ema_prev, 5)
            details['slope_pct'] = round(slope_pct, 4)

            # Normalize slope to score
            # Typical 4H EMA moves ~0.05-0.3% over 2 candles for forex
            # Map: -0.2% or worse = 0.0, +0.2% or better = 1.0

            if direction == 'BULL':
                # Positive slope = higher score for BULL
                normalized = (slope_pct + 0.2) / 0.4  # Map [-0.2, 0.2] to [0, 1]
            else:
                # Negative slope = higher score for BEAR
                normalized = (-slope_pct + 0.2) / 0.4

            score = max(0.0, min(1.0, normalized))

            details['slope_direction'] = 'RISING' if slope_pct > 0.01 else 'FALLING' if slope_pct < -0.01 else 'FLAT'

            return score, details

        except Exception as e:
            self.logger.debug(f"[HTF BIAS] EMA slope scoring error: {e}")
            return 0.5, {'error': str(e)}

    def _score_macd_momentum(
        self,
        df_4h: pd.DataFrame,
        direction: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score based on 4H MACD histogram direction.

        MACD histogram shows momentum change:
        - Positive histogram = bullish momentum
        - Rising histogram = increasing momentum
        """
        details: Dict[str, Any] = {}

        # Find MACD histogram column
        hist_col = None
        for col in ['macd_histogram', 'macd_hist', 'histogram']:
            if col in df_4h.columns:
                hist_col = col
                break

        if hist_col is None or len(df_4h) < 3:
            details['error'] = 'no_macd_data'
            return 0.5, details

        try:
            hist_current = df_4h[hist_col].iloc[-2]
            hist_prev = df_4h[hist_col].iloc[-3]

            if pd.isna(hist_current) or pd.isna(hist_prev):
                return 0.5, {'error': 'invalid_macd_values'}

            hist_rising = hist_current > hist_prev
            hist_positive = hist_current > 0

            details['histogram_column'] = hist_col
            details['hist_current'] = round(float(hist_current), 6)
            details['hist_prev'] = round(float(hist_prev), 6)
            details['hist_positive'] = hist_positive
            details['hist_rising'] = hist_rising

            # Score based on direction alignment
            if direction == 'BULL':
                if hist_positive and hist_rising:
                    score = 0.95  # Strong bullish momentum
                    details['momentum_state'] = 'STRONG_BULLISH'
                elif hist_positive and not hist_rising:
                    score = 0.70  # Bullish but slowing
                    details['momentum_state'] = 'BULLISH_SLOWING'
                elif not hist_positive and hist_rising:
                    score = 0.55  # Bearish but improving
                    details['momentum_state'] = 'BEARISH_IMPROVING'
                else:
                    score = 0.25  # Bearish and falling
                    details['momentum_state'] = 'STRONG_BEARISH'
            else:  # BEAR
                if not hist_positive and not hist_rising:
                    score = 0.95  # Strong bearish momentum
                    details['momentum_state'] = 'STRONG_BEARISH'
                elif not hist_positive and hist_rising:
                    score = 0.70  # Bearish but slowing
                    details['momentum_state'] = 'BEARISH_SLOWING'
                elif hist_positive and not hist_rising:
                    score = 0.55  # Bullish but weakening
                    details['momentum_state'] = 'BULLISH_WEAKENING'
                else:
                    score = 0.25  # Bullish and rising
                    details['momentum_state'] = 'STRONG_BULLISH'

            return score, details

        except Exception as e:
            self.logger.debug(f"[HTF BIAS] MACD scoring error: {e}")
            return 0.5, {'error': str(e)}

    def _interpret_score(self, score: float, direction: str) -> str:
        """Convert score to human-readable interpretation."""
        if score >= 0.9:
            return f"STRONG_ALIGNMENT ({direction})"
        elif score >= 0.7:
            return f"ALIGNED ({direction})"
        elif score >= 0.5:
            return "NEUTRAL"
        elif score >= 0.3:
            return f"WEAK_COUNTER ({direction})"
        else:
            return f"STRONG_COUNTER ({direction})"

    def get_confidence_multiplier(self, bias_score: float) -> float:
        """
        Convert bias score to confidence multiplier.

        Returns:
            Multiplier in range [0.7, 1.3]
            - Score 0.0 → 0.7 (reduce confidence by 30%)
            - Score 0.5 → 1.0 (no change)
            - Score 1.0 → 1.3 (boost confidence by 30%)
        """
        # Linear mapping: score 0.0 = 0.7, score 1.0 = 1.3
        multiplier = 0.7 + (bias_score * 0.6)
        return round(multiplier, 3)

    def should_filter(
        self,
        bias_score: float,
        threshold: float,
        mode: str = 'active'
    ) -> Tuple[bool, str]:
        """
        Determine if signal should be filtered based on bias score.

        Args:
            bias_score: Calculated bias score (0.0-1.0)
            threshold: Minimum score required (e.g., 0.4)
            mode: 'active' (filter), 'monitor' (log only), 'disabled'

        Returns:
            Tuple of (should_reject, reason)
        """
        if mode == 'disabled':
            return False, "HTF bias filter disabled"

        if mode == 'monitor':
            if bias_score < threshold:
                return False, f"HTF bias {bias_score:.2f} < {threshold} (MONITOR ONLY - would reject)"
            return False, f"HTF bias {bias_score:.2f} >= {threshold} (MONITOR ONLY - would pass)"

        # Active mode
        if bias_score < threshold:
            return True, f"HTF bias {bias_score:.2f} < {threshold}"

        return False, f"HTF bias {bias_score:.2f} >= {threshold}"


# Singleton instance for easy access
_calculator_instance: Optional[HTFBiasCalculator] = None


def get_htf_bias_calculator(logger: Optional[logging.Logger] = None) -> HTFBiasCalculator:
    """Get or create singleton HTFBiasCalculator instance."""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = HTFBiasCalculator(logger)
    elif logger is not None:
        _calculator_instance.logger = logger
    return _calculator_instance
