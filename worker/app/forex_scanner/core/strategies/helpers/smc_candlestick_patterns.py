#!/usr/bin/env python3
"""
SMC Candlestick Pattern Detector
Detects price action rejection patterns for structure-based entries
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging


class SMCCandlestickPatterns:
    """
    Detects candlestick patterns for SMC structure-based trading

    Patterns detected:
    - Pin bars (rejection candles)
    - Engulfing patterns
    - Hammers & Shooting stars
    - Inside bars
    - Doji patterns
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # Pattern strength thresholds
        self.min_body_ratio = 0.25  # Minimum body to range ratio
        self.min_wick_ratio = 0.60  # Minimum wick to range ratio for pin bars
        self.min_engulfing_ratio = 1.0  # Engulfing must be at least equal size

    def detect_rejection_pattern(
        self,
        df: pd.DataFrame,
        direction: str,
        min_strength: float = 0.70
    ) -> Optional[Dict]:
        """
        Detect rejection candlestick pattern confirming the trend direction

        Args:
            df: OHLCV DataFrame (last 5-10 bars)
            direction: 'BULL' or 'BEAR' - expected trend direction
            min_strength: Minimum pattern strength (0-1)

        Returns:
            Dict with pattern details or None if no valid pattern
            {
                'pattern_type': str,  # 'pin_bar', 'engulfing', 'hammer', etc.
                'strength': float,    # 0-1 pattern quality score
                'candle_index': int,  # Index of pattern candle
                'entry_price': float, # Suggested entry price
                'rejection_level': float,  # Level where rejection occurred
                'description': str    # Human readable description
            }
        """
        if len(df) < 3:
            return None

        # Check last 3 candles for patterns (most recent first)
        for i in range(1, min(4, len(df))):
            idx = -i
            candle = df.iloc[idx]
            prev_candle = df.iloc[idx - 1] if idx - 1 >= -len(df) else None

            # Try each pattern type
            patterns = [
                self._detect_pin_bar(df, idx, direction),
                self._detect_engulfing(df, idx, direction),
                self._detect_hammer_shooter(df, idx, direction),
            ]

            # Return first valid pattern that meets strength requirement
            for pattern in patterns:
                if pattern and pattern['strength'] >= min_strength:
                    pattern['candle_index'] = len(df) + idx  # Convert to positive index
                    return pattern

        return None

    def _detect_pin_bar(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: str
    ) -> Optional[Dict]:
        """
        Detect pin bar (rejection candle with long wick)

        Bullish pin bar: Long lower wick, small body, rejection of lows
        Bearish pin bar: Long upper wick, small body, rejection of highs
        """
        candle = df.iloc[idx]

        # Calculate candle components
        body_size = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        total_range = candle['high'] - candle['low']

        if total_range == 0:
            return None

        # Calculate ratios
        body_ratio = body_size / total_range
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range

        # Bullish pin bar (rejection of lows)
        if direction == 'BULL':
            if (lower_wick_ratio >= self.min_wick_ratio and
                body_ratio <= self.min_body_ratio and
                upper_wick_ratio < lower_wick_ratio * 0.5):

                strength = min(1.0, (lower_wick_ratio + (1 - body_ratio)) / 2)

                return {
                    'pattern_type': 'bullish_pin_bar',
                    'strength': strength,
                    'entry_price': candle['high'],  # Enter above pin bar high
                    'rejection_level': candle['low'],
                    'description': f"Bullish pin bar: {lower_wick_ratio*100:.0f}% lower wick, body {body_ratio*100:.0f}%"
                }

        # Bearish pin bar (rejection of highs)
        elif direction == 'BEAR':
            if (upper_wick_ratio >= self.min_wick_ratio and
                body_ratio <= self.min_body_ratio and
                lower_wick_ratio < upper_wick_ratio * 0.5):

                strength = min(1.0, (upper_wick_ratio + (1 - body_ratio)) / 2)

                return {
                    'pattern_type': 'bearish_pin_bar',
                    'strength': strength,
                    'entry_price': candle['low'],  # Enter below pin bar low
                    'rejection_level': candle['high'],
                    'description': f"Bearish pin bar: {upper_wick_ratio*100:.0f}% upper wick, body {body_ratio*100:.0f}%"
                }

        return None

    def _detect_engulfing(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: str
    ) -> Optional[Dict]:
        """
        Detect engulfing pattern (current candle engulfs previous)

        Bullish engulfing: Green candle engulfs previous red candle
        Bearish engulfing: Red candle engulfs previous green candle
        """
        if idx - 1 < -len(df):
            return None

        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        current_body_size = abs(current['close'] - current['open'])
        prev_body_size = abs(previous['close'] - previous['open'])

        if prev_body_size == 0:
            return None

        engulfing_ratio = current_body_size / prev_body_size

        # Bullish engulfing
        if direction == 'BULL':
            is_bullish_current = current['close'] > current['open']
            is_bearish_prev = previous['close'] < previous['open']
            engulfs = (current['open'] <= previous['close'] and
                      current['close'] >= previous['open'])

            if is_bullish_current and is_bearish_prev and engulfs and engulfing_ratio >= self.min_engulfing_ratio:
                strength = min(1.0, engulfing_ratio / 2.0)

                return {
                    'pattern_type': 'bullish_engulfing',
                    'strength': strength,
                    'entry_price': current['close'],
                    'rejection_level': min(current['low'], previous['low']),
                    'description': f"Bullish engulfing: {engulfing_ratio:.1f}x previous candle"
                }

        # Bearish engulfing
        elif direction == 'BEAR':
            is_bearish_current = current['close'] < current['open']
            is_bullish_prev = previous['close'] > previous['open']
            engulfs = (current['open'] >= previous['close'] and
                      current['close'] <= previous['open'])

            if is_bearish_current and is_bullish_prev and engulfs and engulfing_ratio >= self.min_engulfing_ratio:
                strength = min(1.0, engulfing_ratio / 2.0)

                return {
                    'pattern_type': 'bearish_engulfing',
                    'strength': strength,
                    'entry_price': current['close'],
                    'rejection_level': max(current['high'], previous['high']),
                    'description': f"Bearish engulfing: {engulfing_ratio:.1f}x previous candle"
                }

        return None

    def _detect_hammer_shooter(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: str
    ) -> Optional[Dict]:
        """
        Detect hammer (bullish) or shooting star (bearish)

        Hammer: Small body at top, long lower wick (bullish reversal)
        Shooting star: Small body at bottom, long upper wick (bearish reversal)
        """
        candle = df.iloc[idx]

        # Calculate candle components
        body_size = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        total_range = candle['high'] - candle['low']

        if total_range == 0:
            return None

        body_ratio = body_size / total_range
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range

        # Hammer (bullish)
        if direction == 'BULL':
            is_hammer = (
                lower_wick_ratio >= 0.60 and  # Long lower wick
                body_ratio <= 0.30 and          # Small body
                upper_wick_ratio <= 0.15        # Very small upper wick
            )

            if is_hammer:
                strength = min(1.0, lower_wick_ratio + (1 - body_ratio) / 2)

                return {
                    'pattern_type': 'hammer',
                    'strength': strength,
                    'entry_price': candle['high'],
                    'rejection_level': candle['low'],
                    'description': f"Hammer: {lower_wick_ratio*100:.0f}% lower wick"
                }

        # Shooting star (bearish)
        elif direction == 'BEAR':
            is_shooter = (
                upper_wick_ratio >= 0.60 and   # Long upper wick
                body_ratio <= 0.30 and           # Small body
                lower_wick_ratio <= 0.15        # Very small lower wick
            )

            if is_shooter:
                strength = min(1.0, upper_wick_ratio + (1 - body_ratio) / 2)

                return {
                    'pattern_type': 'shooting_star',
                    'strength': strength,
                    'entry_price': candle['low'],
                    'rejection_level': candle['high'],
                    'description': f"Shooting star: {upper_wick_ratio*100:.0f}% upper wick"
                }

        return None

    def get_pattern_summary(self, pattern: Dict) -> str:
        """Get human-readable summary of detected pattern"""
        if not pattern:
            return "No pattern detected"

        return (f"{pattern['pattern_type'].replace('_', ' ').title()} "
                f"(strength: {pattern['strength']*100:.0f}%) - "
                f"{pattern['description']}")
