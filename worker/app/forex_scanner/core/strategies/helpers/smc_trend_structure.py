#!/usr/bin/env python3
"""
SMC Trend Structure Analyzer
Identifies higher timeframe trend direction using structure-based analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class SMCTrendStructure:
    """
    Analyzes trend structure for SMC strategy entries

    Methods:
    - Higher highs/higher lows detection for uptrends
    - Lower highs/lower lows detection for downtrends
    - Trend strength measurement
    - Pullback detection within trends
    - Structure break identification
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # Detection parameters (loosened for better swing detection)
        self.swing_strength = 3  # Bars on each side for swing detection (reduced from 5)
        self.min_swing_significance_pips = 5  # Minimum 5 pips swing size (more realistic)
        self.pullback_ratio = 0.382  # Fibonacci 38.2% minimum pullback

        # PRIORITY 1B: Improved bearish pattern detection
        # More sensitive swing detection for lows to catch bearish structures
        self.min_swing_significance_pips_lows = 3  # Lower threshold for swing lows (bearish)

    def analyze_trend(
        self,
        df: pd.DataFrame,
        epic: str,
        lookback: int = 100
    ) -> Dict:
        """
        Analyze higher timeframe trend structure

        Args:
            df: OHLCV DataFrame (should be higher timeframe like 4H)
            epic: Currency pair for pip calculation
            lookback: Bars to analyze

        Returns:
            {
                'trend': str,  # 'BULL', 'BEAR', 'RANGING', 'UNKNOWN'
                'strength': float,  # 0-1 trend strength score
                'swing_highs': List[Dict],  # List of swing high dicts
                'swing_lows': List[Dict],  # List of swing low dicts
                'structure_type': str,  # 'HH_HL', 'LH_LL', 'MIXED'
                'last_swing_high': Dict,  # Most recent swing high
                'last_swing_low': Dict,  # Most recent swing low
                'in_pullback': bool,  # Currently in pullback?
                'pullback_depth': float,  # Pullback depth as ratio (0-1)
                'description': str  # Human-readable summary
            }
        """
        if len(df) < lookback:
            lookback = len(df)

        recent_data = df.tail(lookback).copy()
        recent_data.reset_index(drop=True, inplace=True)

        # Get pip value
        pair = epic.split('.')[2] if '.' in epic else epic
        pip_value = 0.01 if 'JPY' in pair else 0.0001

        # Find swing highs and lows
        swing_highs = self._find_swing_highs(recent_data, pip_value)
        swing_lows = self._find_swing_lows(recent_data, pip_value)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                'trend': 'UNKNOWN',
                'strength': 0.0,
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'structure_type': 'INSUFFICIENT_DATA',
                'last_swing_high': swing_highs[-1] if swing_highs else None,
                'last_swing_low': swing_lows[-1] if swing_lows else None,
                'in_pullback': False,
                'pullback_depth': 0.0,
                'description': 'Insufficient swing points for trend analysis'
            }

        # Determine structure type (HH/HL vs LH/LL)
        structure_type = self._determine_structure_type(swing_highs, swing_lows)

        # Calculate trend and strength
        trend, strength = self._calculate_trend_and_strength(
            structure_type, swing_highs, swing_lows, recent_data
        )

        # Check for pullback
        in_pullback, pullback_depth = self._detect_pullback(
            recent_data, swing_highs, swing_lows, trend
        )

        # Build description
        description = self._build_trend_description(
            trend, strength, structure_type, in_pullback, pullback_depth
        )

        return {
            'trend': trend,
            'strength': strength,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'structure_type': structure_type,
            'last_swing_high': swing_highs[-1],
            'last_swing_low': swing_lows[-1],
            'in_pullback': in_pullback,
            'pullback_depth': pullback_depth,
            'description': description
        }

    def _find_swing_highs(
        self,
        df: pd.DataFrame,
        pip_value: float
    ) -> List[Dict]:
        """
        Find significant swing high points

        Returns list of dicts with:
        - index: Position in dataframe
        - price: High price
        - timestamp: Bar timestamp
        """
        swing_highs = []

        for i in range(self.swing_strength, len(df) - self.swing_strength):
            window = df['high'].iloc[i - self.swing_strength:i + self.swing_strength + 1]

            # Check if current bar is highest in window
            if df['high'].iloc[i] == window.max():
                # Check significance - compare to previous swing, not adjacent bar
                if len(swing_highs) > 0:
                    # Calculate movement in pips from last swing high
                    movement_pips = abs(df['high'].iloc[i] - swing_highs[-1]['price']) / pip_value

                    if movement_pips >= self.min_swing_significance_pips:
                        swing_highs.append({
                            'index': i,
                            'price': df['high'].iloc[i],
                            'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                        })
                else:
                    # First swing is always valid (no previous swing to compare)
                    swing_highs.append({
                        'index': i,
                        'price': df['high'].iloc[i],
                        'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                    })

        return swing_highs

    def _find_swing_lows(
        self,
        df: pd.DataFrame,
        pip_value: float
    ) -> List[Dict]:
        """
        Find significant swing low points

        PRIORITY 1B: More sensitive detection for bearish structures
        Uses lower pip threshold (3 vs 5) to catch more swing lows

        Returns list of dicts with:
        - index: Position in dataframe
        - price: Low price
        - timestamp: Bar timestamp
        """
        swing_lows = []

        for i in range(self.swing_strength, len(df) - self.swing_strength):
            window = df['low'].iloc[i - self.swing_strength:i + self.swing_strength + 1]

            # Check if current bar is lowest in window
            if df['low'].iloc[i] == window.min():
                # Check significance - compare to previous swing, not adjacent bar
                if len(swing_lows) > 0:
                    # Calculate movement in pips from last swing low
                    movement_pips = abs(df['low'].iloc[i] - swing_lows[-1]['price']) / pip_value

                    # PRIORITY 1B: Use lower threshold for swing lows (better bearish detection)
                    if movement_pips >= self.min_swing_significance_pips_lows:
                        swing_lows.append({
                            'index': i,
                            'price': df['low'].iloc[i],
                            'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                        })
                else:
                    # First swing is always valid (no previous swing to compare)
                    swing_lows.append({
                        'index': i,
                        'price': df['low'].iloc[i],
                        'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                    })

        return swing_lows

    def _determine_structure_type(
        self,
        swing_highs: List[Dict],
        swing_lows: List[Dict]
    ) -> str:
        """
        Determine if structure is making HH/HL (bullish) or LH/LL (bearish)

        Returns:
            'HH_HL': Higher highs and higher lows (bullish)
            'LH_LL': Lower highs and lower lows (bearish)
            'MIXED': Conflicting structure
        """
        # Need at least 2 of each to compare
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'INSUFFICIENT_DATA'

        # Compare last 3 swing highs (if available)
        num_higher_highs = 0
        num_lower_highs = 0

        check_count = min(3, len(swing_highs))
        for i in range(1, check_count):
            if swing_highs[-i]['price'] > swing_highs[-i-1]['price']:
                num_higher_highs += 1
            else:
                num_lower_highs += 1

        # Compare last 3 swing lows (if available)
        num_higher_lows = 0
        num_lower_lows = 0

        check_count = min(3, len(swing_lows))
        for i in range(1, check_count):
            if swing_lows[-i]['price'] > swing_lows[-i-1]['price']:
                num_higher_lows += 1
            else:
                num_lower_lows += 1

        # Determine structure type
        if num_higher_highs > num_lower_highs and num_higher_lows > num_lower_lows:
            return 'HH_HL'  # Bullish structure
        elif num_lower_highs > num_higher_highs and num_lower_lows > num_higher_lows:
            return 'LH_LL'  # Bearish structure
        else:
            return 'MIXED'  # Conflicting signals

    def _calculate_trend_and_strength(
        self,
        structure_type: str,
        swing_highs: List[Dict],
        swing_lows: List[Dict],
        df: pd.DataFrame
    ) -> Tuple[str, float]:
        """
        Calculate trend direction and strength

        Returns:
            (trend, strength) where trend is 'BULL', 'BEAR', 'RANGING'
            and strength is 0-1
        """
        if structure_type == 'INSUFFICIENT_DATA':
            return 'UNKNOWN', 0.0

        # Base trend from structure
        if structure_type == 'HH_HL':
            base_trend = 'BULL'
            base_strength = 0.60
        elif structure_type == 'LH_LL':
            base_trend = 'BEAR'
            base_strength = 0.60
        else:
            base_trend = 'RANGING'
            base_strength = 0.30

        # Adjust strength based on momentum
        # Check if price is moving away from last swing or pulling back
        current_price = df['close'].iloc[-1]
        last_swing_high = swing_highs[-1]['price']
        last_swing_low = swing_lows[-1]['price']

        if base_trend == 'BULL':
            # In uptrend, strength increases if price is near highs
            range_size = last_swing_high - last_swing_low
            if range_size > 0:
                position_in_range = (current_price - last_swing_low) / range_size
                # Higher position = stronger trend
                strength = base_strength + (0.40 * position_in_range)
            else:
                strength = base_strength

        elif base_trend == 'BEAR':
            # In downtrend, strength increases if price is near lows
            range_size = last_swing_high - last_swing_low
            if range_size > 0:
                position_in_range = (last_swing_high - current_price) / range_size
                # Lower position = stronger trend
                strength = base_strength + (0.40 * position_in_range)
            else:
                strength = base_strength

        else:
            strength = base_strength

        return base_trend, min(1.0, strength)

    def _detect_pullback(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict],
        swing_lows: List[Dict],
        trend: str
    ) -> Tuple[bool, float]:
        """
        Detect if price is currently in a pullback within the trend

        Returns:
            (in_pullback, pullback_depth) where pullback_depth is 0-1
        """
        if trend not in ['BULL', 'BEAR']:
            return False, 0.0

        current_price = df['close'].iloc[-1]
        last_swing_high = swing_highs[-1]['price']
        last_swing_low = swing_lows[-1]['price']

        if trend == 'BULL':
            # In uptrend, pullback means price came down from recent high
            # Check if we've made a new high then pulled back
            recent_high = df['high'].iloc[-20:].max() if len(df) >= 20 else df['high'].max()

            # Pullback if price is below recent high and above last swing low
            if current_price < recent_high and current_price > last_swing_low:
                range_size = recent_high - last_swing_low
                if range_size > 0:
                    pullback_depth = (recent_high - current_price) / range_size

                    # Only consider it a pullback if it's significant (>38.2%)
                    if pullback_depth >= self.pullback_ratio:
                        return True, pullback_depth

            return False, 0.0

        else:  # BEAR
            # In downtrend, pullback means price came up from recent low
            recent_low = df['low'].iloc[-20:].min() if len(df) >= 20 else df['low'].min()

            # Pullback if price is above recent low and below last swing high
            if current_price > recent_low and current_price < last_swing_high:
                range_size = last_swing_high - recent_low
                if range_size > 0:
                    pullback_depth = (current_price - recent_low) / range_size

                    # Only consider it a pullback if it's significant (>38.2%)
                    if pullback_depth >= self.pullback_ratio:
                        return True, pullback_depth

            return False, 0.0

    def _build_trend_description(
        self,
        trend: str,
        strength: float,
        structure_type: str,
        in_pullback: bool,
        pullback_depth: float
    ) -> str:
        """Build human-readable trend description"""

        # Base description
        if trend == 'BULL':
            desc = f"Bullish trend (strength {strength*100:.0f}%, {structure_type})"
        elif trend == 'BEAR':
            desc = f"Bearish trend (strength {strength*100:.0f}%, {structure_type})"
        elif trend == 'RANGING':
            desc = f"Ranging market (strength {strength*100:.0f}%, {structure_type})"
        else:
            desc = f"Unknown trend ({structure_type})"

        # Add pullback info
        if in_pullback:
            desc += f" - IN PULLBACK ({pullback_depth*100:.0f}% retracement)"

        return desc

    def is_structure_aligned(
        self,
        trend_analysis: Dict,
        signal_direction: str
    ) -> bool:
        """
        Check if signal direction aligns with trend structure

        Args:
            trend_analysis: Result from analyze_trend()
            signal_direction: 'BULL' or 'BEAR'

        Returns:
            True if aligned, False otherwise
        """
        # Must have clear trend
        if trend_analysis['trend'] not in ['BULL', 'BEAR']:
            return False

        # Signal must match trend
        if signal_direction != trend_analysis['trend']:
            return False

        # Trend must be strong enough
        if trend_analysis['strength'] < 0.50:
            return False

        return True

    def get_trend_summary(self, trend_analysis: Dict) -> str:
        """Get human-readable summary of trend analysis"""
        return trend_analysis['description']
