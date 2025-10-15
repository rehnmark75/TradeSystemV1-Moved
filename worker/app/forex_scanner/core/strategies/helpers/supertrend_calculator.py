# core/strategies/helpers/supertrend_calculator.py
"""
Supertrend Calculator
Calculates Supertrend indicator values for trend-following strategy.

Supertrend Formula:
- Basic Upperband = (HIGH + LOW) / 2 + (Multiplier × ATR)
- Basic Lowerband = (HIGH + LOW) / 2 - (Multiplier × ATR)
- Final bands adjust based on price position relative to previous bands
- Trend is bullish when price > Final Lowerband, bearish when price < Final Upperband
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging


class SupertrendCalculator:
    """
    Calculate Supertrend indicator for multi-timeframe trend analysis

    Supertrend is a trend-following indicator that provides dynamic support/resistance
    based on Average True Range (ATR) volatility.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize Supertrend calculator

        Args:
            logger: Logger instance for debugging
        """
        self.logger = logger or logging.getLogger(__name__)

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default 14)

        Returns:
            Series containing ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Average True Range using EMA (more responsive than SMA)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def calculate_supertrend(
        self,
        df: pd.DataFrame,
        period: int = 14,
        multiplier: float = 2.0,
        atr_period: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate Supertrend indicator

        Args:
            df: DataFrame with OHLC data
            period: ATR period for volatility calculation
            multiplier: ATR multiplier for band width
            atr_period: Optional separate ATR period (uses period if not specified)

        Returns:
            Dictionary with:
                - 'supertrend': Supertrend line values
                - 'trend': Trend direction (1=bullish, -1=bearish)
                - 'upper_band': Upper band values
                - 'lower_band': Lower band values
                - 'atr': ATR values
        """
        if atr_period is None:
            atr_period = period

        # Calculate ATR
        atr = self.calculate_atr(df, atr_period)

        # Calculate basic bands
        hl_avg = (df['high'] + df['low']) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)

        # Initialize final bands and trend
        final_upper_band = pd.Series(index=df.index, dtype=float)
        final_lower_band = pd.Series(index=df.index, dtype=float)
        supertrend = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)

        # First value initialization
        final_upper_band.iloc[0] = upper_band.iloc[0]
        final_lower_band.iloc[0] = lower_band.iloc[0]

        # Initial trend based on close position
        if df['close'].iloc[0] <= final_upper_band.iloc[0]:
            trend.iloc[0] = -1  # Bearish
            supertrend.iloc[0] = final_upper_band.iloc[0]
        else:
            trend.iloc[0] = 1   # Bullish
            supertrend.iloc[0] = final_lower_band.iloc[0]

        # Calculate for remaining values
        for i in range(1, len(df)):
            # Final Upper Band
            if (upper_band.iloc[i] < final_upper_band.iloc[i-1]) or \
               (df['close'].iloc[i-1] > final_upper_band.iloc[i-1]):
                final_upper_band.iloc[i] = upper_band.iloc[i]
            else:
                final_upper_band.iloc[i] = final_upper_band.iloc[i-1]

            # Final Lower Band
            if (lower_band.iloc[i] > final_lower_band.iloc[i-1]) or \
               (df['close'].iloc[i-1] < final_lower_band.iloc[i-1]):
                final_lower_band.iloc[i] = lower_band.iloc[i]
            else:
                final_lower_band.iloc[i] = final_lower_band.iloc[i-1]

            # Trend determination
            if trend.iloc[i-1] == 1:  # Previous trend was bullish
                if df['close'].iloc[i] <= final_lower_band.iloc[i]:
                    trend.iloc[i] = -1  # Flip to bearish
                    supertrend.iloc[i] = final_upper_band.iloc[i]
                else:
                    trend.iloc[i] = 1   # Stay bullish
                    supertrend.iloc[i] = final_lower_band.iloc[i]
            else:  # Previous trend was bearish
                if df['close'].iloc[i] >= final_upper_band.iloc[i]:
                    trend.iloc[i] = 1   # Flip to bullish
                    supertrend.iloc[i] = final_lower_band.iloc[i]
                else:
                    trend.iloc[i] = -1  # Stay bearish
                    supertrend.iloc[i] = final_upper_band.iloc[i]

        return {
            'supertrend': supertrend,
            'trend': trend,
            'upper_band': final_upper_band,
            'lower_band': final_lower_band,
            'atr': atr
        }

    def calculate_multi_supertrend(
        self,
        df: pd.DataFrame,
        fast_period: int = 7,
        fast_multiplier: float = 1.5,
        medium_period: int = 14,
        medium_multiplier: float = 2.0,
        slow_period: int = 21,
        slow_multiplier: float = 3.0,
        atr_period: int = 14
    ) -> Dict[str, Dict[str, pd.Series]]:
        """
        Calculate multiple Supertrend indicators for confluence analysis

        Args:
            df: DataFrame with OHLC data
            fast_period: Fast Supertrend ATR period
            fast_multiplier: Fast Supertrend multiplier
            medium_period: Medium Supertrend ATR period
            medium_multiplier: Medium Supertrend multiplier
            slow_period: Slow Supertrend ATR period
            slow_multiplier: Slow Supertrend multiplier
            atr_period: ATR calculation period

        Returns:
            Dictionary with 'fast', 'medium', 'slow' Supertrend results
        """
        results = {}

        # Calculate Fast Supertrend
        results['fast'] = self.calculate_supertrend(
            df, fast_period, fast_multiplier, atr_period
        )

        # Calculate Medium Supertrend
        results['medium'] = self.calculate_supertrend(
            df, medium_period, medium_multiplier, atr_period
        )

        # Calculate Slow Supertrend
        results['slow'] = self.calculate_supertrend(
            df, slow_period, slow_multiplier, atr_period
        )

        return results

    def get_supertrend_confluence(
        self,
        supertrends: Dict[str, Dict[str, pd.Series]],
        index: int = -1
    ) -> Dict[str, any]:
        """
        Analyze Supertrend confluence at specific index

        Args:
            supertrends: Multi-Supertrend results from calculate_multi_supertrend
            index: Index to analyze (default -1 for latest)

        Returns:
            Dictionary with confluence analysis:
                - 'bullish_count': Number of bullish Supertrends
                - 'bearish_count': Number of bearish Supertrends
                - 'confluence_pct': Confluence percentage (0-100)
                - 'direction': 'bullish', 'bearish', or 'neutral'
                - 'strength': 'strong', 'medium', or 'weak'
        """
        fast_trend = supertrends['fast']['trend'].iloc[index]
        medium_trend = supertrends['medium']['trend'].iloc[index]
        slow_trend = supertrends['slow']['trend'].iloc[index]

        bullish_count = sum([
            fast_trend == 1,
            medium_trend == 1,
            slow_trend == 1
        ])

        bearish_count = sum([
            fast_trend == -1,
            medium_trend == -1,
            slow_trend == -1
        ])

        total = 3
        confluence_pct = max(bullish_count, bearish_count) / total * 100

        # Determine direction
        if bullish_count == 3:
            direction = 'bullish'
            strength = 'strong'
        elif bearish_count == 3:
            direction = 'bearish'
            strength = 'strong'
        elif bullish_count == 2:
            direction = 'bullish'
            strength = 'medium'
        elif bearish_count == 2:
            direction = 'bearish'
            strength = 'medium'
        else:
            direction = 'neutral'
            strength = 'weak'

        return {
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'confluence_pct': confluence_pct,
            'direction': direction,
            'strength': strength,
            'fast_trend': int(fast_trend),
            'medium_trend': int(medium_trend),
            'slow_trend': int(slow_trend)
        }

    def detect_supertrend_flip(
        self,
        supertrend_data: Dict[str, pd.Series],
        lookback: int = 2
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if Supertrend has recently flipped trend

        Args:
            supertrend_data: Single Supertrend result
            lookback: Number of bars to check for flip

        Returns:
            Tuple of (has_flipped, direction)
            - has_flipped: True if trend changed in lookback period
            - direction: 'bullish' or 'bearish' if flipped, None otherwise
        """
        trend = supertrend_data['trend']

        if len(trend) < lookback + 1:
            return False, None

        current_trend = trend.iloc[-1]

        for i in range(1, lookback + 1):
            if trend.iloc[-i-1] != current_trend:
                direction = 'bullish' if current_trend == 1 else 'bearish'
                return True, direction

        return False, None
