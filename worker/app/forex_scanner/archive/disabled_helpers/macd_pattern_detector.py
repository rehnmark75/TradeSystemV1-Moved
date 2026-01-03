# core/strategies/helpers/macd_pattern_detector.py
"""
Candlestick Pattern Detector for MACD Confluence Strategy

Detects high-probability price action patterns on 15M timeframe
that serve as entry triggers when price is at Fibonacci confluence zones.

Patterns Detected:
1. Bullish/Bearish Engulfing
2. Pin Bar (Hammer/Shooting Star)
3. Inside Bar Breakout

All patterns are validated for quality (body size, wick ratios, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging


class CandlestickPatternDetector:
    """
    Detect candlestick patterns for entry confirmation.

    Used by MACD Confluence Strategy to identify high-probability
    entry points when price reaches Fibonacci zones.
    """

    def __init__(self,
                 min_body_ratio: float = 0.6,
                 min_engulf_ratio: float = 1.1,
                 max_pin_body_ratio: float = 0.3,
                 min_pin_wick_ratio: float = 2.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize pattern detector.

        Args:
            min_body_ratio: Minimum body size as ratio of total candle range (for engulfing)
            min_engulf_ratio: Minimum size increase for engulfing (1.1 = 10% larger)
            max_pin_body_ratio: Maximum body size for pin bar (0.3 = 30% of range)
            min_pin_wick_ratio: Minimum wick to body ratio for pin bar (2.0 = wick 2x body)
            logger: Optional logger instance
        """
        self.min_body_ratio = min_body_ratio
        self.min_engulf_ratio = min_engulf_ratio
        self.max_pin_body_ratio = max_pin_body_ratio
        self.min_pin_wick_ratio = min_pin_wick_ratio
        self.logger = logger or logging.getLogger(__name__)

    def _get_candle_metrics(self, candle: pd.Series) -> Dict:
        """
        Calculate candle metrics for pattern detection.

        Args:
            candle: Series with OHLC data

        Returns:
            Dict with candle metrics
        """
        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']

        # Body
        body_size = abs(close - open_price)
        body_top = max(open_price, close)
        body_bottom = min(open_price, close)

        # Wicks
        upper_wick = high - body_top
        lower_wick = body_bottom - low

        # Total range
        total_range = high - low

        # Ratios
        body_ratio = body_size / total_range if total_range > 0 else 0
        upper_wick_ratio = upper_wick / body_size if body_size > 0 else 0
        lower_wick_ratio = lower_wick / body_size if body_size > 0 else 0

        # Direction
        is_bullish = close > open_price
        is_bearish = close < open_price

        return {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'body_size': body_size,
            'body_top': body_top,
            'body_bottom': body_bottom,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'total_range': total_range,
            'body_ratio': body_ratio,
            'upper_wick_ratio': upper_wick_ratio,
            'lower_wick_ratio': lower_wick_ratio,
            'is_bullish': is_bullish,
            'is_bearish': is_bearish
        }

    def detect_bullish_engulfing(self, df: pd.DataFrame, position: int = -1) -> Optional[Dict]:
        """
        Detect bullish engulfing pattern.

        Requirements:
        - Previous candle: bearish (red)
        - Current candle: bullish (green) and engulfs previous candle body
        - Current body > 60% of total range (strong candle)
        - Current body >= 1.1x previous body (clear engulfing)

        Args:
            df: DataFrame with OHLC data
            position: Position to check (-1 = most recent)

        Returns:
            Dict with pattern data or None if not found
        """
        if len(df) < 2:
            return None

        current = self._get_candle_metrics(df.iloc[position])
        previous = self._get_candle_metrics(df.iloc[position - 1])

        # Check basic requirements
        if not current['is_bullish'] or not previous['is_bearish']:
            return None

        # Check body size quality
        if current['body_ratio'] < self.min_body_ratio:
            return None

        # Check engulfing - current body completely covers previous body
        engulfs = (current['body_bottom'] <= previous['body_bottom'] and
                  current['body_top'] >= previous['body_top'])

        if not engulfs:
            return None

        # Check size increase
        size_ratio = current['body_size'] / previous['body_size'] if previous['body_size'] > 0 else 0
        if size_ratio < self.min_engulf_ratio:
            return None

        quality_score = min(100, int(
            (current['body_ratio'] * 50) +  # Body strength
            (size_ratio * 30) +              # Engulfing strength
            20                                # Base score
        ))

        return {
            'pattern': 'bullish_engulfing',
            'signal': 'BULL',
            'quality_score': quality_score,
            'entry_price': current['close'],
            'confirmation_price': current['high'],  # Break above high confirms
            'invalidation_price': current['low'],    # Break below low invalidates
            'candle_data': current,
            'previous_candle_data': previous
        }

    def detect_bearish_engulfing(self, df: pd.DataFrame, position: int = -1) -> Optional[Dict]:
        """
        Detect bearish engulfing pattern.

        Requirements:
        - Previous candle: bullish (green)
        - Current candle: bearish (red) and engulfs previous candle body
        - Current body > 60% of total range (strong candle)
        - Current body >= 1.1x previous body (clear engulfing)

        Args:
            df: DataFrame with OHLC data
            position: Position to check (-1 = most recent)

        Returns:
            Dict with pattern data or None if not found
        """
        if len(df) < 2:
            return None

        current = self._get_candle_metrics(df.iloc[position])
        previous = self._get_candle_metrics(df.iloc[position - 1])

        # Check basic requirements
        if not current['is_bearish'] or not previous['is_bullish']:
            return None

        # Check body size quality
        if current['body_ratio'] < self.min_body_ratio:
            return None

        # Check engulfing
        engulfs = (current['body_top'] >= previous['body_top'] and
                  current['body_bottom'] <= previous['body_bottom'])

        if not engulfs:
            return None

        # Check size increase
        size_ratio = current['body_size'] / previous['body_size'] if previous['body_size'] > 0 else 0
        if size_ratio < self.min_engulf_ratio:
            return None

        quality_score = min(100, int(
            (current['body_ratio'] * 50) +
            (size_ratio * 30) +
            20
        ))

        return {
            'pattern': 'bearish_engulfing',
            'signal': 'BEAR',
            'quality_score': quality_score,
            'entry_price': current['close'],
            'confirmation_price': current['low'],   # Break below low confirms
            'invalidation_price': current['high'],   # Break above high invalidates
            'candle_data': current,
            'previous_candle_data': previous
        }

    def detect_bullish_pin_bar(self, df: pd.DataFrame, position: int = -1) -> Optional[Dict]:
        """
        Detect bullish pin bar (hammer).

        Requirements:
        - Long lower wick (rejection of lower prices)
        - Small body (< 30% of total range)
        - Lower wick >= 2x body size
        - Little to no upper wick

        Args:
            df: DataFrame with OHLC data
            position: Position to check (-1 = most recent)

        Returns:
            Dict with pattern data or None if not found
        """
        if len(df) < 1:
            return None

        current = self._get_candle_metrics(df.iloc[position])

        # Check body size (must be small)
        if current['body_ratio'] > self.max_pin_body_ratio:
            return None

        # Check lower wick dominance
        if current['lower_wick_ratio'] < self.min_pin_wick_ratio:
            return None

        # Check upper wick is minimal (< 50% of lower wick)
        if current['upper_wick'] > current['lower_wick'] * 0.5:
            return None

        quality_score = min(100, int(
            (current['lower_wick_ratio'] * 30) +     # Wick strength
            ((1 - current['body_ratio']) * 40) +      # Body smallness
            30                                         # Base score
        ))

        return {
            'pattern': 'bullish_pin_bar',
            'signal': 'BULL',
            'quality_score': quality_score,
            'entry_price': current['close'],
            'confirmation_price': current['high'],
            'invalidation_price': current['low'],
            'candle_data': current
        }

    def detect_bearish_pin_bar(self, df: pd.DataFrame, position: int = -1) -> Optional[Dict]:
        """
        Detect bearish pin bar (shooting star).

        Requirements:
        - Long upper wick (rejection of higher prices)
        - Small body (< 30% of total range)
        - Upper wick >= 2x body size
        - Little to no lower wick

        Args:
            df: DataFrame with OHLC data
            position: Position to check (-1 = most recent)

        Returns:
            Dict with pattern data or None if not found
        """
        if len(df) < 1:
            return None

        current = self._get_candle_metrics(df.iloc[position])

        # Check body size (must be small)
        if current['body_ratio'] > self.max_pin_body_ratio:
            return None

        # Check upper wick dominance
        if current['upper_wick_ratio'] < self.min_pin_wick_ratio:
            return None

        # Check lower wick is minimal
        if current['lower_wick'] > current['upper_wick'] * 0.5:
            return None

        quality_score = min(100, int(
            (current['upper_wick_ratio'] * 30) +
            ((1 - current['body_ratio']) * 40) +
            30
        ))

        return {
            'pattern': 'bearish_pin_bar',
            'signal': 'BEAR',
            'quality_score': quality_score,
            'entry_price': current['close'],
            'confirmation_price': current['low'],
            'invalidation_price': current['high'],
            'candle_data': current
        }

    def detect_inside_bar(self, df: pd.DataFrame, position: int = -1) -> Optional[Dict]:
        """
        Detect inside bar pattern.

        Requirements:
        - Current candle high < previous candle high
        - Current candle low > previous candle low
        - (Current candle completely inside previous candle range)

        Note: Direction determined by breakout direction

        Args:
            df: DataFrame with OHLC data
            position: Position to check (-1 = most recent)

        Returns:
            Dict with pattern data or None if not found
        """
        if len(df) < 2:
            return None

        current = self._get_candle_metrics(df.iloc[position])
        previous = self._get_candle_metrics(df.iloc[position - 1])

        # Check if current is inside previous
        is_inside = (current['high'] < previous['high'] and
                    current['low'] > previous['low'])

        if not is_inside:
            return None

        # Quality based on how tight the consolidation is
        range_ratio = current['total_range'] / previous['total_range'] if previous['total_range'] > 0 else 0
        quality_score = min(100, int(
            ((1 - range_ratio) * 70) +  # Tighter = better
            30                           # Base score
        ))

        return {
            'pattern': 'inside_bar',
            'signal': 'PENDING',  # Direction determined by breakout
            'quality_score': quality_score,
            'entry_price_bull': previous['high'],  # Breakout above
            'entry_price_bear': previous['low'],   # Breakout below
            'invalidation_price': current['close'],
            'candle_data': current,
            'previous_candle_data': previous
        }

    def detect_all_patterns(self, df: pd.DataFrame, signal_direction: str = None) -> List[Dict]:
        """
        Detect all applicable patterns and return sorted by quality.

        Args:
            df: DataFrame with OHLC data
            signal_direction: Optional filter ('BULL' or 'BEAR')

        Returns:
            List of detected patterns sorted by quality score (highest first)
        """
        patterns = []

        # Detect bullish patterns
        if not signal_direction or signal_direction == 'BULL':
            bullish_engulfing = self.detect_bullish_engulfing(df)
            if bullish_engulfing:
                patterns.append(bullish_engulfing)

            bullish_pin = self.detect_bullish_pin_bar(df)
            if bullish_pin:
                patterns.append(bullish_pin)

        # Detect bearish patterns
        if not signal_direction or signal_direction == 'BEAR':
            bearish_engulfing = self.detect_bearish_engulfing(df)
            if bearish_engulfing:
                patterns.append(bearish_engulfing)

            bearish_pin = self.detect_bearish_pin_bar(df)
            if bearish_pin:
                patterns.append(bearish_pin)

        # Inside bar (works for both directions)
        inside_bar = self.detect_inside_bar(df)
        if inside_bar:
            patterns.append(inside_bar)

        # Sort by quality score (highest first)
        patterns.sort(key=lambda x: x['quality_score'], reverse=True)

        return patterns

    def get_best_pattern(self, df: pd.DataFrame, signal_direction: str) -> Optional[Dict]:
        """
        Get the highest quality pattern matching signal direction.

        Args:
            df: DataFrame with OHLC data
            signal_direction: 'BULL' or 'BEAR'

        Returns:
            Best pattern dict or None
        """
        patterns = self.detect_all_patterns(df, signal_direction)

        if not patterns:
            return None

        # Return highest quality pattern
        best_pattern = patterns[0]

        self.logger.info(f"🎯 Best pattern: {best_pattern['pattern']} "
                        f"(quality: {best_pattern['quality_score']}/100)")

        return best_pattern


def test_pattern_detector():
    """Quick test of pattern detector"""
    import pandas as pd

    # Test data for bullish engulfing
    data = {
        'open': [1.0900, 1.0895],
        'high': [1.0905, 1.0920],
        'low': [1.0890, 1.0885],
        'close': [1.0892, 1.0918]
    }
    df = pd.DataFrame(data)

    detector = CandlestickPatternDetector()

    # Test engulfing
    engulfing = detector.detect_bullish_engulfing(df)
    if engulfing:
        print(f"✅ Bullish engulfing detected - Quality: {engulfing['quality_score']}/100")
    else:
        print("❌ No bullish engulfing")

    # Test all patterns
    all_patterns = detector.detect_all_patterns(df, 'BULL')
    print(f"✅ Total patterns found: {len(all_patterns)}")

    # Test best pattern
    best = detector.get_best_pattern(df, 'BULL')
    if best:
        print(f"✅ Best pattern: {best['pattern']}")


if __name__ == '__main__':
    test_pattern_detector()
