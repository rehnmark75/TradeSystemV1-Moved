# core/strategies/helpers/macd_fibonacci_calculator.py
"""
Fibonacci Retracement Calculator for MACD Confluence Strategy

Detects swing highs/lows and calculates Fibonacci retracement levels
for identifying high-probability entry zones during pullbacks.

Key Features:
- Swing point detection (configurable strength)
- Fibonacci level calculation (38.2%, 50%, 61.8%, 78.6%)
- Multi-timeframe support (primarily H1 for structure)
- Price zone identification for confluence analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime


class FibonacciCalculator:
    """
    Calculate Fibonacci retracement levels from swing points.

    Used by MACD Confluence Strategy to identify high-probability
    entry zones during trend retracements.
    """

    # Standard Fibonacci retracement levels
    FIB_LEVELS = {
        '0': 0.0,
        '23.6': 0.236,
        '38.2': 0.382,
        '50.0': 0.5,
        '61.8': 0.618,
        '78.6': 0.786,
        '100': 1.0
    }

    def __init__(self,
                 lookback_bars: int = 50,
                 swing_strength: int = 5,
                 min_swing_size_pips: float = 15.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Fibonacci calculator.

        Args:
            lookback_bars: How many bars to look back for swing detection
            swing_strength: Minimum bars on left/right for valid swing (5 = 11-bar swing)
            min_swing_size_pips: Minimum swing move size in pips (filters noise)
            logger: Optional logger instance
        """
        self.lookback_bars = lookback_bars
        self.swing_strength = swing_strength
        self.min_swing_size_pips = min_swing_size_pips
        self.logger = logger or logging.getLogger(__name__)

    def detect_swing_high(self, df: pd.DataFrame, position: int) -> bool:
        """
        Detect if a position is a swing high.

        A swing high requires:
        - Current high > all highs N bars to the left
        - Current high > all highs N bars to the right

        Args:
            df: DataFrame with price data (must have 'high' column)
            position: Index position to check

        Returns:
            True if position is a swing high
        """
        if position < self.swing_strength or position >= len(df) - self.swing_strength:
            return False

        current_high = df['high'].iloc[position]

        # Check left side
        left_highs = df['high'].iloc[position - self.swing_strength:position]
        if (left_highs >= current_high).any():
            return False

        # Check right side
        right_highs = df['high'].iloc[position + 1:position + self.swing_strength + 1]
        if (right_highs >= current_high).any():
            return False

        return True

    def detect_swing_low(self, df: pd.DataFrame, position: int) -> bool:
        """
        Detect if a position is a swing low.

        A swing low requires:
        - Current low < all lows N bars to the left
        - Current low < all lows N bars to the right

        Args:
            df: DataFrame with price data (must have 'low' column)
            position: Index position to check

        Returns:
            True if position is a swing low
        """
        if position < self.swing_strength or position >= len(df) - self.swing_strength:
            return False

        current_low = df['low'].iloc[position]

        # Check left side
        left_lows = df['low'].iloc[position - self.swing_strength:position]
        if (left_lows <= current_low).any():
            return False

        # Check right side
        right_lows = df['low'].iloc[position + 1:position + self.swing_strength + 1]
        if (right_lows <= current_low).any():
            return False

        return True

    def find_recent_swing_points(self, df: pd.DataFrame, epic: str = None) -> Dict:
        """
        Find the most recent significant swing high and swing low.

        Args:
            df: DataFrame with OHLC data
            epic: Currency pair epic (for pip calculation)

        Returns:
            Dict with swing high/low data:
            {
                'swing_high': {'price': float, 'index': int, 'time': datetime},
                'swing_low': {'price': float, 'index': int, 'time': datetime},
                'swing_size_pips': float
            }
        """
        if len(df) < self.lookback_bars:
            self.logger.warning(f"Not enough data: {len(df)} bars (need {self.lookback_bars})")
            return None

        # Determine pip multiplier
        pip_multiplier = 100 if epic and 'JPY' in epic else 10000

        # Find swing highs in lookback period
        swing_highs = []
        for i in range(len(df) - self.swing_strength - 1, max(len(df) - self.lookback_bars, self.swing_strength), -1):
            if self.detect_swing_high(df, i):
                swing_highs.append({
                    'price': df['high'].iloc[i],
                    'index': i,
                    'time': df.index[i] if hasattr(df.index[i], 'strftime') else None
                })
                break  # Take most recent swing high

        # Find swing lows in lookback period
        swing_lows = []
        for i in range(len(df) - self.swing_strength - 1, max(len(df) - self.lookback_bars, self.swing_strength), -1):
            if self.detect_swing_low(df, i):
                swing_lows.append({
                    'price': df['low'].iloc[i],
                    'index': i,
                    'time': df.index[i] if hasattr(df.index[i], 'strftime') else None
                })
                break  # Take most recent swing low

        if not swing_highs or not swing_lows:
            self.logger.debug("No swing points found in lookback period")
            return None

        swing_high = swing_highs[0]
        swing_low = swing_lows[0]

        # Calculate swing size in pips
        swing_size = abs(swing_high['price'] - swing_low['price'])
        swing_size_pips = swing_size * pip_multiplier

        # Filter out small swings (noise)
        if swing_size_pips < self.min_swing_size_pips:
            self.logger.debug(f"Swing too small: {swing_size_pips:.1f} pips (min: {self.min_swing_size_pips})")
            return None

        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'swing_size_pips': swing_size_pips
        }

    def calculate_fibonacci_levels(self,
                                   swing_high: float,
                                   swing_low: float,
                                   trend_direction: str = 'bullish') -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.

        For bullish retracement (after uptrend):
        - 0% = swing high (most recent high)
        - 100% = swing low (start of move)
        - Retracement levels calculated from high downwards

        For bearish retracement (after downtrend):
        - 0% = swing low (most recent low)
        - 100% = swing high (start of move)
        - Retracement levels calculated from low upwards

        Args:
            swing_high: Price of swing high
            swing_low: Price of swing low
            trend_direction: 'bullish' or 'bearish' (direction before retracement)

        Returns:
            Dict of Fibonacci levels {level_name: price}
        """
        swing_range = swing_high - swing_low
        fib_levels = {}

        if trend_direction == 'bullish':
            # Uptrend retracement: measure from high down to low
            for name, ratio in self.FIB_LEVELS.items():
                fib_levels[name] = swing_high - (swing_range * ratio)
        else:
            # Downtrend retracement: measure from low up to high
            for name, ratio in self.FIB_LEVELS.items():
                fib_levels[name] = swing_low + (swing_range * ratio)

        return fib_levels

    def get_fibonacci_zones(self,
                           df: pd.DataFrame,
                           epic: str = None,
                           current_trend: str = 'bullish') -> Optional[Dict]:
        """
        Get Fibonacci retracement zones for current market structure.

        This is the main method called by the strategy.

        Args:
            df: DataFrame with OHLC data
            epic: Currency pair epic
            current_trend: Current trend direction from MACD H4 filter

        Returns:
            Dict with Fibonacci data:
            {
                'swing_high': {...},
                'swing_low': {...},
                'swing_size_pips': float,
                'fib_levels': {'23.6': price, '38.2': price, ...},
                'trend_direction': 'bullish' or 'bearish',
                'key_zones': {
                    '50.0': {'price': float, 'is_key': True},
                    '61.8': {'price': float, 'is_key': True},
                    ...
                }
            }
        """
        # Find recent swing points
        swing_data = self.find_recent_swing_points(df, epic)
        if not swing_data:
            return None

        swing_high_price = swing_data['swing_high']['price']
        swing_low_price = swing_data['swing_low']['price']

        # Calculate Fibonacci levels
        fib_levels = self.calculate_fibonacci_levels(
            swing_high_price,
            swing_low_price,
            current_trend
        )

        # Identify key retracement zones (50%, 61.8%)
        key_zones = {}
        for level_name in ['50.0', '61.8', '38.2']:  # Priority order
            if level_name in fib_levels:
                key_zones[level_name] = {
                    'price': fib_levels[level_name],
                    'is_key': level_name in ['50.0', '61.8']  # 50% and 61.8% are most important
                }

        result = {
            'swing_high': swing_data['swing_high'],
            'swing_low': swing_data['swing_low'],
            'swing_size_pips': swing_data['swing_size_pips'],
            'fib_levels': fib_levels,
            'trend_direction': current_trend,
            'key_zones': key_zones
        }

        self.logger.info(f"📊 Fibonacci zones calculated - Swing: {swing_data['swing_size_pips']:.1f} pips, "
                        f"Key levels: {', '.join([f'{k}:{v['price']:.5f}' for k,v in key_zones.items()])}")

        return result

    def is_price_at_fib_level(self,
                              current_price: float,
                              fib_level_price: float,
                              tolerance_pips: float = 5.0,
                              epic: str = None) -> bool:
        """
        Check if current price is at/near a Fibonacci level.

        Args:
            current_price: Current market price
            fib_level_price: Fibonacci level price to check
            tolerance_pips: How close price must be (in pips)
            epic: Currency pair for pip calculation

        Returns:
            True if price is within tolerance of Fibonacci level
        """
        pip_multiplier = 100 if epic and 'JPY' in epic else 10000
        tolerance = tolerance_pips / pip_multiplier

        distance = abs(current_price - fib_level_price)
        return distance <= tolerance

    def get_nearest_fib_level(self,
                             current_price: float,
                             fib_data: Dict,
                             epic: str = None) -> Optional[Tuple[str, float, float]]:
        """
        Find the nearest Fibonacci level to current price.

        Args:
            current_price: Current market price
            fib_data: Fibonacci data from get_fibonacci_zones()
            epic: Currency pair

        Returns:
            Tuple of (level_name, level_price, distance_pips) or None
        """
        if not fib_data or 'fib_levels' not in fib_data:
            return None

        pip_multiplier = 100 if epic and 'JPY' in epic else 10000

        nearest = None
        min_distance = float('inf')

        for level_name, level_price in fib_data['fib_levels'].items():
            distance = abs(current_price - level_price)
            distance_pips = distance * pip_multiplier

            if distance < min_distance:
                min_distance = distance
                nearest = (level_name, level_price, distance_pips)

        return nearest


def test_fibonacci_calculator():
    """Quick test of Fibonacci calculator"""
    import pandas as pd

    # Create test data with clear swing
    dates = pd.date_range('2025-01-01', periods=100, freq='H')
    prices = {
        'high': [1.0900 + i*0.0001 if i < 50 else 1.0950 - (i-50)*0.00005 for i in range(100)],
        'low': [1.0890 + i*0.0001 if i < 50 else 1.0940 - (i-50)*0.00005 for i in range(100)],
        'close': [1.0895 + i*0.0001 if i < 50 else 1.0945 - (i-50)*0.00005 for i in range(100)]
    }
    df = pd.DataFrame(prices, index=dates)

    calc = FibonacciCalculator(lookback_bars=50, swing_strength=5)

    # Test swing detection
    swing_data = calc.find_recent_swing_points(df, 'EURUSD')
    if swing_data:
        print(f"✅ Swing found: {swing_data['swing_size_pips']:.1f} pips")

        # Test Fibonacci calculation
        fib_zones = calc.get_fibonacci_zones(df, 'EURUSD', 'bullish')
        if fib_zones:
            print(f"✅ Fibonacci levels: {list(fib_zones['fib_levels'].keys())}")
            print(f"✅ Key zones: {list(fib_zones['key_zones'].keys())}")
    else:
        print("❌ No swing found")


if __name__ == '__main__':
    test_fibonacci_calculator()
