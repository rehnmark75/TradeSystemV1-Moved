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
        self.swing_strength = 2  # Bars on each side for swing detection (relaxed for 4H = 20 hours)
        self.min_swing_significance_pips = 5  # Minimum 5 pips swing size (adjusted per pair type)
        self.min_swing_significance_pips_jpy = 20  # Minimum 20 pips for JPY pairs (higher volatility)
        self.pullback_ratio = 0.382  # Fibonacci 38.2% minimum pullback

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

        # Determine minimum swing significance based on pair type
        is_jpy_pair = 'JPY' in pair
        min_swing_pips = self.min_swing_significance_pips_jpy if is_jpy_pair else self.min_swing_significance_pips

        # Find swing highs and lows
        swing_highs = self._find_swing_highs(recent_data, pip_value, min_swing_pips)
        swing_lows = self._find_swing_lows(recent_data, pip_value, min_swing_pips)

        # FALLBACK: If insufficient swings detected, use simple price action analysis
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            self.logger.warning(f"⚠️  Insufficient swings detected (Highs: {len(swing_highs)}, Lows: {len(swing_lows)})")
            self.logger.warning(f"   Using fallback price action analysis (20-bar comparison)")

            # Use simple price comparison over last 20 bars
            fallback_trend, fallback_strength = self._fallback_trend_analysis(recent_data)

            return {
                'trend': fallback_trend,
                'strength': fallback_strength,
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'structure_type': 'FALLBACK_ANALYSIS',
                'last_swing_high': swing_highs[-1] if swing_highs else None,
                'last_swing_low': swing_lows[-1] if swing_lows else None,
                'in_pullback': False,
                'pullback_depth': 0.0,
                'description': f'Fallback analysis: {fallback_trend} trend ({fallback_strength*100:.0f}% strength)'
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
        pip_value: float,
        min_swing_pips: float = None
    ) -> List[Dict]:
        """
        Find significant swing high points

        Args:
            df: OHLC DataFrame
            pip_value: Pip value for the pair
            min_swing_pips: Minimum pip movement for swing significance (uses default if None)

        Returns list of dicts with:
        - index: Position in dataframe
        - price: High price
        - timestamp: Bar timestamp
        """
        if min_swing_pips is None:
            min_swing_pips = self.min_swing_significance_pips

        swing_highs = []

        for i in range(self.swing_strength, len(df) - self.swing_strength):
            window = df['high'].iloc[i - self.swing_strength:i + self.swing_strength + 1]

            # Check if current bar is highest in window
            if df['high'].iloc[i] == window.max():
                # Check significance - compare to previous swing, not adjacent bar
                if len(swing_highs) > 0:
                    # Calculate movement in pips from last swing high
                    movement_pips = abs(df['high'].iloc[i] - swing_highs[-1]['price']) / pip_value

                    if movement_pips >= min_swing_pips:
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
        pip_value: float,
        min_swing_pips: float = None
    ) -> List[Dict]:
        """
        Find significant swing low points

        Args:
            df: OHLC DataFrame
            pip_value: Pip value for the pair
            min_swing_pips: Minimum pip movement for swing significance (uses default if None)

        Returns list of dicts with:
        - index: Position in dataframe
        - price: Low price
        - timestamp: Bar timestamp
        """
        if min_swing_pips is None:
            min_swing_pips = self.min_swing_significance_pips

        swing_lows = []

        for i in range(self.swing_strength, len(df) - self.swing_strength):
            window = df['low'].iloc[i - self.swing_strength:i + self.swing_strength + 1]

            # Check if current bar is lowest in window
            if df['low'].iloc[i] == window.min():
                # Check significance - compare to previous swing, not adjacent bar
                if len(swing_lows) > 0:
                    # Calculate movement in pips from last swing low
                    movement_pips = abs(df['low'].iloc[i] - swing_lows[-1]['price']) / pip_value

                    if movement_pips >= min_swing_pips:
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
        Calculate trend direction and strength using DYNAMIC multi-factor analysis

        PHASE 1 ENHANCEMENT: Replaces hardcoded 60% base with quality-based calculation

        Returns:
            (trend, strength) where trend is 'BULL', 'BEAR', 'RANGING'
            and strength is 0.30-1.0 (true distribution, not clustered at 60%)
        """
        if structure_type == 'INSUFFICIENT_DATA':
            return 'UNKNOWN', 0.0

        # Determine base trend direction from structure
        if structure_type == 'HH_HL':
            base_trend = 'BULL'
        elif structure_type == 'LH_LL':
            base_trend = 'BEAR'
        else:
            base_trend = 'RANGING'
            # Ranging markets get low fixed strength
            return base_trend, 0.30

        # Calculate DYNAMIC strength using multi-factor analysis
        strength = self._calculate_dynamic_htf_strength(
            swing_highs, swing_lows, df, base_trend
        )

        return base_trend, min(1.0, max(0.30, strength))

    def _calculate_dynamic_htf_strength(
        self,
        swing_highs: List[Dict],
        swing_lows: List[Dict],
        df: pd.DataFrame,
        trend_direction: str
    ) -> float:
        """
        PHASE 1: Dynamic HTF Strength Calculation - Multi-Factor Quality Scoring

        Replaces hardcoded 60% base with true quality assessment using 5 factors:
        1. Swing Consistency (20%) - Regular swing spacing = institutional trend
        2. Swing Size vs ATR (20%) - Large swings = strong momentum
        3. Pullback Depth (20%) - Shallow pullbacks = strong trend
        4. Price Momentum (20%) - Position in range + velocity
        5. Volume Profile (20%) - Higher impulse volume = institutions

        Expected Output:
        - Weak choppy trends: 30-45%
        - Clean moderate trends: 50-60%
        - Strong institutional trends: 65-85%
        - Exceptional trending markets: 85-100%

        Args:
            swing_highs: List of swing high dicts
            swing_lows: List of swing low dicts
            df: OHLCV DataFrame
            trend_direction: 'BULL' or 'BEAR'

        Returns:
            float: Quality-based strength score (0.30-1.0)
        """
        factors = []

        # Factor 1: Swing Consistency (20%) - More regular swings = stronger trend
        consistency_score = self._calculate_swing_consistency(swing_highs, swing_lows)
        factors.append(consistency_score * 0.20)

        # Factor 2: Swing Size vs ATR (20%) - Larger swings = institutional activity
        swing_size_score = self._calculate_swing_size_strength(swing_highs, swing_lows, df)
        factors.append(swing_size_score * 0.20)

        # Factor 3: Pullback Depth (20%) - Shallower pullbacks = stronger trend
        pullback_score = self._calculate_pullback_strength(swing_highs, swing_lows, trend_direction)
        factors.append(pullback_score * 0.20)

        # Factor 4: Price Momentum (20%) - Current position + velocity
        momentum_score = self._calculate_price_momentum(df, swing_highs, swing_lows, trend_direction)
        factors.append(momentum_score * 0.20)

        # Factor 5: Volume Profile (20%) - Higher impulse volume = institutions
        volume_score = self._calculate_volume_profile(df, swing_highs, swing_lows)
        factors.append(volume_score * 0.20)

        # Sum all factors (0.30-1.0 range)
        total_strength = sum(factors)

        # Log detailed breakdown for analysis
        self.logger.debug(f"🎯 DYNAMIC HTF STRENGTH BREAKDOWN:")
        self.logger.debug(f"   Swing Consistency: {consistency_score*100:.1f}% → {consistency_score*0.20*100:.1f}%")
        self.logger.debug(f"   Swing Size/ATR:    {swing_size_score*100:.1f}% → {swing_size_score*0.20*100:.1f}%")
        self.logger.debug(f"   Pullback Depth:    {pullback_score*100:.1f}% → {pullback_score*0.20*100:.1f}%")
        self.logger.debug(f"   Price Momentum:    {momentum_score*100:.1f}% → {momentum_score*0.20*100:.1f}%")
        self.logger.debug(f"   Volume Profile:    {volume_score*100:.1f}% → {volume_score*0.20*100:.1f}%")
        self.logger.debug(f"   TOTAL STRENGTH:    {total_strength*100:.1f}%")

        return max(0.30, total_strength)

    def _calculate_swing_consistency(
        self,
        swing_highs: List[Dict],
        swing_lows: List[Dict]
    ) -> float:
        """
        Calculate swing consistency score

        More regular swing spacing = stronger institutional trend
        Erratic swings = choppy retail noise

        Returns: 0.0-1.0 score
        """
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return 0.5  # Neutral if insufficient data

        # Calculate distances between consecutive swings
        high_distances = []
        for i in range(1, len(swing_highs)):
            distance = swing_highs[i]['index'] - swing_highs[i-1]['index']
            high_distances.append(distance)

        low_distances = []
        for i in range(1, len(swing_lows)):
            distance = swing_lows[i]['index'] - swing_lows[i-1]['index']
            low_distances.append(distance)

        # Calculate coefficient of variation (lower = more consistent)
        def calc_consistency(distances):
            if not distances or len(distances) < 2:
                return 0.5
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            if mean_dist == 0:
                return 0.5
            cv = std_dist / mean_dist  # Coefficient of variation
            # Convert to 0-1 score (lower CV = higher score)
            # CV of 0.5 = perfect consistency (1.0), CV of 2.0+ = poor (0.0)
            consistency = max(0.0, min(1.0, 1.0 - (cv / 2.0)))
            return consistency

        high_consistency = calc_consistency(high_distances)
        low_consistency = calc_consistency(low_distances)

        # Average both
        return (high_consistency + low_consistency) / 2

    def _calculate_swing_size_strength(
        self,
        swing_highs: List[Dict],
        swing_lows: List[Dict],
        df: pd.DataFrame
    ) -> float:
        """
        Calculate swing size relative to ATR

        Larger swings relative to ATR = institutional activity
        Small swings = low volatility trend

        Returns: 0.0-1.0 score
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.5

        # Calculate ATR (14-period)
        atr = self._calculate_atr(df, period=14)
        if atr == 0:
            return 0.5

        # Calculate average swing size
        swing_sizes = []
        for i in range(1, min(len(swing_highs), len(swing_lows))):
            swing_range = abs(swing_highs[i]['price'] - swing_lows[i]['price'])
            swing_sizes.append(swing_range)

        if not swing_sizes:
            return 0.5

        avg_swing_size = np.mean(swing_sizes)

        # Ratio: avg swing / (ATR * 2)
        # ATR * 2 = typical swing size
        # Ratio > 1.0 = strong swings, < 0.5 = weak swings
        swing_ratio = avg_swing_size / (atr * 2.0)

        # Convert to 0-1 score (cap at 1.0)
        score = min(1.0, swing_ratio)

        return score

    def _calculate_pullback_strength(
        self,
        swing_highs: List[Dict],
        swing_lows: List[Dict],
        trend_direction: str
    ) -> float:
        """
        Calculate pullback depth quality

        Shallow pullbacks (< 38.2% Fib) = strong trend
        Deep pullbacks (> 61.8% Fib) = weak trend

        Returns: 0.0-1.0 score (higher = shallower pullbacks)
        """
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return 0.5

        pullback_depths = []

        if trend_direction == 'BULL':
            # Measure retracement from swing high to next swing low
            for i in range(1, min(len(swing_highs), len(swing_lows))):
                prev_low = swing_lows[i-1]['price']
                swing_high = swing_highs[i]['price']
                next_low = swing_lows[i]['price']

                swing_range = swing_high - prev_low
                if swing_range > 0:
                    pullback = swing_high - next_low
                    pullback_ratio = pullback / swing_range
                    pullback_depths.append(pullback_ratio)

        else:  # BEAR
            # Measure retracement from swing low to next swing high
            for i in range(1, min(len(swing_lows), len(swing_highs))):
                prev_high = swing_highs[i-1]['price']
                swing_low = swing_lows[i]['price']
                next_high = swing_highs[i]['price']

                swing_range = prev_high - swing_low
                if swing_range > 0:
                    pullback = next_high - swing_low
                    pullback_ratio = pullback / swing_range
                    pullback_depths.append(pullback_ratio)

        if not pullback_depths:
            return 0.5

        avg_pullback = np.mean(pullback_depths)

        # Score: shallow pullbacks = high score
        # < 0.382 (38.2%) = 1.0, > 0.618 (61.8%) = 0.0
        score = 1.0 - min(1.0, avg_pullback / 0.618)

        return max(0.0, score)

    def _calculate_price_momentum(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict],
        swing_lows: List[Dict],
        trend_direction: str
    ) -> float:
        """
        Calculate price momentum from position in range + velocity

        Returns: 0.0-1.0 score
        """
        current_price = df['close'].iloc[-1]
        last_swing_high = swing_highs[-1]['price']
        last_swing_low = swing_lows[-1]['price']

        # Component 1: Position in range (50% weight)
        range_size = last_swing_high - last_swing_low
        if range_size > 0:
            if trend_direction == 'BULL':
                # Higher in range = stronger
                position = (current_price - last_swing_low) / range_size
            else:  # BEAR
                # Lower in range = stronger
                position = (last_swing_high - current_price) / range_size

            position_score = max(0.0, min(1.0, position))
        else:
            position_score = 0.5

        # Component 2: Velocity (50% weight)
        # Compare last 5 closes to previous 5 closes
        lookback = min(10, len(df))
        if lookback >= 10:
            recent = df['close'].iloc[-5:].mean()
            previous = df['close'].iloc[-10:-5].mean()

            if previous != 0:
                change_pct = (recent - previous) / previous

                if trend_direction == 'BULL':
                    # Positive change = strong
                    velocity_score = max(0.0, min(1.0, change_pct * 50))  # Scale: 2% = 1.0
                else:  # BEAR
                    # Negative change = strong
                    velocity_score = max(0.0, min(1.0, -change_pct * 50))
            else:
                velocity_score = 0.5
        else:
            velocity_score = 0.5

        # Combine both components
        momentum = (position_score * 0.5) + (velocity_score * 0.5)

        return momentum

    def _calculate_volume_profile(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict],
        swing_lows: List[Dict]
    ) -> float:
        """
        Calculate volume profile ratio (impulse vs pullback)

        Higher volume on impulse moves = institutional participation
        Higher volume on pullbacks = distribution/accumulation

        Returns: 0.0-1.0 score
        """
        # Check if volume data exists
        if 'volume' not in df.columns and 'ltv' not in df.columns:
            # No volume data - return neutral
            return 0.5

        volume_col = 'volume' if 'volume' in df.columns else 'ltv'

        # Not enough swings to determine impulse/pullback
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.5

        try:
            # Identify impulse vs pullback candles
            impulse_volumes = []
            pullback_volumes = []

            # Simplified approach: last swing to current
            last_high_idx = swing_highs[-1]['index']
            last_low_idx = swing_lows[-1]['index']

            # Determine which is more recent
            if last_high_idx > last_low_idx:
                # Bullish impulse (from low to high)
                impulse_start = last_low_idx
                impulse_end = last_high_idx
                impulse_volumes = df[volume_col].iloc[impulse_start:impulse_end].tolist()

                # Pullback (from high to current)
                if impulse_end < len(df) - 1:
                    pullback_volumes = df[volume_col].iloc[impulse_end:].tolist()
            else:
                # Bearish impulse (from high to low)
                impulse_start = last_high_idx
                impulse_end = last_low_idx
                impulse_volumes = df[volume_col].iloc[impulse_start:impulse_end].tolist()

                # Pullback (from low to current)
                if impulse_end < len(df) - 1:
                    pullback_volumes = df[volume_col].iloc[impulse_end:].tolist()

            # Calculate averages
            if impulse_volumes and pullback_volumes:
                avg_impulse = np.mean(impulse_volumes)
                avg_pullback = np.mean(pullback_volumes)

                if avg_pullback > 0:
                    volume_ratio = avg_impulse / avg_pullback
                    # Ratio > 1.5 = institutional (1.0), < 1.0 = weak (0.0)
                    score = max(0.0, min(1.0, (volume_ratio - 1.0) / 0.5))
                    return score

            # Default to neutral if can't calculate
            return 0.5

        except Exception as e:
            self.logger.warning(f"Volume profile calculation failed: {e}")
            return 0.5

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range

        Args:
            df: OHLCV DataFrame
            period: ATR period (default 14)

        Returns:
            float: ATR value
        """
        if len(df) < period + 1:
            # Fallback: use simple range
            return (df['high'].max() - df['low'].min()) / len(df)

        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR (simple moving average of TR)
        atr = true_range.rolling(window=period).mean().iloc[-1]

        return atr if not np.isnan(atr) else 0.0001  # Fallback to small value

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

    def _fallback_trend_analysis(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Fallback trend analysis when swing detection fails
        Uses simple price action over last 20 bars

        Args:
            df: OHLC DataFrame

        Returns:
            Tuple of (trend_string, strength_float)
        """
        lookback = min(20, len(df))
        if lookback < 10:
            return 'UNKNOWN', 0.0

        recent = df.tail(lookback)

        # Compare first half vs second half
        first_half_high = recent['high'].iloc[:lookback//2].max()
        first_half_low = recent['low'].iloc[:lookback//2].min()
        second_half_high = recent['high'].iloc[lookback//2:].max()
        second_half_low = recent['low'].iloc[lookback//2:].min()

        # Calculate directional bias
        higher_highs = second_half_high > first_half_high
        higher_lows = second_half_low > first_half_low
        lower_highs = second_half_high < first_half_high
        lower_lows = second_half_low < first_half_low

        # Determine trend
        if higher_highs and higher_lows:
            trend = 'BULL'
            # Strength based on how much higher
            range_size = first_half_high - first_half_low
            if range_size > 0:
                high_improvement = (second_half_high - first_half_high) / range_size
                low_improvement = (second_half_low - first_half_low) / range_size
                strength = min(0.95, 0.60 + (high_improvement + low_improvement) * 0.5)
            else:
                strength = 0.60
        elif lower_highs and lower_lows:
            trend = 'BEAR'
            # Strength based on how much lower
            range_size = first_half_high - first_half_low
            if range_size > 0:
                high_decline = (first_half_high - second_half_high) / range_size
                low_decline = (first_half_low - second_half_low) / range_size
                strength = min(0.95, 0.60 + (high_decline + low_decline) * 0.5)
            else:
                strength = 0.60
        else:
            # Mixed signals = ranging or transitioning
            trend = 'RANGING'
            strength = 0.30

        self.logger.info(f"   Fallback analysis: {trend} (strength: {strength*100:.0f}%)")
        self.logger.info(f"   First half: H={first_half_high:.5f}, L={first_half_low:.5f}")
        self.logger.info(f"   Second half: H={second_half_high:.5f}, L={second_half_low:.5f}")

        return trend, strength

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
