#!/usr/bin/env python3
"""
SMC Support/Resistance Detector
Identifies horizontal price levels where price reacts (supply/demand zones)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class SMCSupportResistance:
    """
    Detects support and resistance levels for structure-based trading

    Methods:
    - Horizontal level detection from swing highs/lows
    - Level strength scoring (touch count, age, recent reaction)
    - Proximity checking (is price near a level?)
    - Supply/demand zone identification
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # Detection parameters
        self.swing_strength = 3  # Bars on each side for swing detection
        self.level_cluster_pips = 10  # Pips tolerance for level clustering
        self.min_touches = 2  # Minimum touches for valid level
        self.max_level_age_bars = 200  # Maximum age for level relevance

    def detect_levels(
        self,
        df: pd.DataFrame,
        epic: str,
        lookback: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Detect support and resistance levels from price action

        Args:
            df: OHLCV DataFrame
            epic: Currency pair for pip calculation
            lookback: Bars to analyze

        Returns:
            {
                'support': [level_dict, ...],
                'resistance': [level_dict, ...],
                'supply_zones': [zone_dict, ...],
                'demand_zones': [zone_dict, ...]
            }
        """
        if len(df) < lookback:
            lookback = len(df)

        recent_data = df.tail(lookback).copy()
        recent_data.reset_index(drop=True, inplace=True)

        # Get pip value
        pair = epic.split('.')[2] if '.' in epic else epic
        pip_value = 0.01 if 'JPY' in pair else 0.0001

        # Find swing points
        swing_highs = self._find_swing_highs(recent_data)
        swing_lows = self._find_swing_lows(recent_data)

        # Cluster into levels
        resistance_levels = self._cluster_levels(swing_highs, pip_value, 'resistance')
        support_levels = self._cluster_levels(swing_lows, pip_value, 'support')

        # Score level strength
        resistance_levels = self._score_levels(resistance_levels, recent_data, pip_value, 'resistance')
        support_levels = self._score_levels(support_levels, recent_data, pip_value, 'support')

        # Filter by minimum touches and sort by strength
        resistance_levels = [l for l in resistance_levels if l['touch_count'] >= self.min_touches]
        support_levels = [l for l in support_levels if l['touch_count'] >= self.min_touches]

        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        support_levels.sort(key=lambda x: x['strength'], reverse=True)

        # Identify supply/demand zones (stronger levels with confirmation)
        supply_zones = [l for l in resistance_levels if l['strength'] >= 0.70 and l['recent_rejection']]
        demand_zones = [l for l in support_levels if l['strength'] >= 0.70 and l['recent_rejection']]

        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'supply_zones': supply_zones,
            'demand_zones': demand_zones
        }

    def find_nearest_level(
        self,
        current_price: float,
        levels: List[Dict],
        proximity_pips: float,
        pip_value: float
    ) -> Optional[Dict]:
        """
        Find nearest level within proximity range

        Args:
            current_price: Current market price
            levels: List of level dicts
            proximity_pips: Maximum distance in pips
            pip_value: Pip value for pair

        Returns:
            Nearest level dict or None
        """
        proximity_distance = proximity_pips * pip_value
        nearest_level = None
        min_distance = float('inf')

        for level in levels:
            distance = abs(current_price - level['price'])

            if distance <= proximity_distance and distance < min_distance:
                min_distance = distance
                nearest_level = level.copy()
                nearest_level['distance_pips'] = distance / pip_value

        return nearest_level

    def _find_swing_highs(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Find swing high points (local highs)

        Returns list of (index, price) tuples
        """
        swing_highs = []

        for i in range(self.swing_strength, len(df) - self.swing_strength):
            window = df['high'].iloc[i - self.swing_strength:i + self.swing_strength + 1]

            if df['high'].iloc[i] == window.max():
                swing_highs.append((i, df['high'].iloc[i]))

        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Find swing low points (local lows)

        Returns list of (index, price) tuples
        """
        swing_lows = []

        for i in range(self.swing_strength, len(df) - self.swing_strength):
            window = df['low'].iloc[i - self.swing_strength:i + self.swing_strength + 1]

            if df['low'].iloc[i] == window.min():
                swing_lows.append((i, df['low'].iloc[i]))

        return swing_lows

    def _cluster_levels(
        self,
        swing_points: List[Tuple[int, float]],
        pip_value: float,
        level_type: str
    ) -> List[Dict]:
        """
        Cluster swing points into horizontal levels

        Args:
            swing_points: List of (index, price) tuples
            pip_value: Pip value for clustering tolerance
            level_type: 'support' or 'resistance'

        Returns:
            List of level dictionaries
        """
        if not swing_points:
            return []

        cluster_tolerance = self.level_cluster_pips * pip_value
        levels = []

        # Sort by price
        sorted_points = sorted(swing_points, key=lambda x: x[1])

        current_cluster = [sorted_points[0]]

        for point in sorted_points[1:]:
            # Check if point belongs to current cluster
            if abs(point[1] - current_cluster[0][1]) <= cluster_tolerance:
                current_cluster.append(point)
            else:
                # Create level from current cluster
                if len(current_cluster) >= 1:
                    levels.append(self._create_level_dict(current_cluster, level_type))

                # Start new cluster
                current_cluster = [point]

        # Add final cluster
        if len(current_cluster) >= 1:
            levels.append(self._create_level_dict(current_cluster, level_type))

        return levels

    def _create_level_dict(
        self,
        cluster: List[Tuple[int, float]],
        level_type: str
    ) -> Dict:
        """Create level dictionary from cluster of swing points"""
        prices = [p[1] for p in cluster]
        indices = [p[0] for p in cluster]

        return {
            'price': np.mean(prices),  # Average price of cluster
            'type': level_type,
            'touch_count': len(cluster),
            'first_touch_index': min(indices),
            'last_touch_index': max(indices),
            'price_range': (min(prices), max(prices)),
            'strength': 0.0,  # To be calculated
            'recent_rejection': False  # To be calculated
        }

    def _score_levels(
        self,
        levels: List[Dict],
        df: pd.DataFrame,
        pip_value: float,
        level_type: str
    ) -> List[Dict]:
        """
        Score level strength based on:
        - Touch count (more touches = stronger)
        - Recency (recent touches = more relevant)
        - Rejection quality (strong rejection = higher score)
        - Age (too old = lower score)
        """
        current_idx = len(df) - 1

        for level in levels:
            score = 0.0

            # Touch count score (0-0.40)
            touch_score = min(0.40, level['touch_count'] * 0.10)
            score += touch_score

            # Recency score (0-0.30)
            bars_since_last = current_idx - level['last_touch_index']
            if bars_since_last < 20:
                recency_score = 0.30
                level['recent_rejection'] = True
            elif bars_since_last < 50:
                recency_score = 0.20
            elif bars_since_last < 100:
                recency_score = 0.10
            else:
                recency_score = 0.0

            score += recency_score

            # Age penalty (0-0.30)
            age_bars = current_idx - level['first_touch_index']
            if age_bars < 50:
                age_score = 0.30
            elif age_bars < 100:
                age_score = 0.20
            elif age_bars < self.max_level_age_bars:
                age_score = 0.10
            else:
                age_score = 0.0  # Too old

            score += age_score

            level['strength'] = min(1.0, score)
            level['age_bars'] = age_bars
            level['bars_since_last'] = bars_since_last

        return levels

    def get_level_summary(self, level: Dict, pip_value: float) -> str:
        """Get human-readable summary of level"""
        price_range = level['price_range']
        range_pips = (price_range[1] - price_range[0]) / pip_value

        return (f"{level['type'].capitalize()} at {level['price']:.5f} "
                f"({level['touch_count']} touches, strength {level['strength']*100:.0f}%, "
                f"last {level['bars_since_last']} bars ago, range {range_pips:.1f} pips)")
