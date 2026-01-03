# core/strategies/helpers/macd_confluence_analyzer.py
"""
Confluence Zone Analyzer for MACD Confluence Strategy

Analyzes and scores price zones based on multiple alignment factors:
1. Fibonacci retracement levels
2. Swing high/low support/resistance
3. Round number psychological levels
4. EMA dynamic support/resistance

Confluence = Multiple factors aligning at same price level = Higher probability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from decimal import Decimal, ROUND_HALF_UP


class ConfluenceZoneAnalyzer:
    """
    Analyze and score confluence zones for trade entry.

    Identifies high-probability entry zones where multiple
    technical factors align (Fibonacci + swing levels + EMAs + round numbers).
    """

    def __init__(self,
                 confluence_mode: str = 'moderate',
                 proximity_tolerance_pips: float = 5.0,
                 min_confluence_score: float = 2.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize confluence analyzer.

        Args:
            confluence_mode: 'strict', 'moderate', or 'loose'
                - strict: Requires 3+ factors aligned
                - moderate: Requires 2+ factors aligned (Fib + 1 other)
                - loose: Requires Fib level only
            proximity_tolerance_pips: How close factors must be to count as aligned
            min_confluence_score: Minimum score to consider zone valid
            logger: Optional logger instance
        """
        self.confluence_mode = confluence_mode
        self.proximity_tolerance_pips = proximity_tolerance_pips
        self.min_confluence_score = min_confluence_score
        self.logger = logger or logging.getLogger(__name__)

        # Score weights for different confluence factors
        self.score_weights = {
            'fibonacci': 2.0,        # Base factor (always present)
            'swing_level': 1.5,      # Strong resistance/support
            'round_number': 0.5,     # Psychological level
            'ema_21': 0.75,          # Dynamic support (fast EMA)
            'ema_50': 1.0,           # Dynamic support (medium EMA)
            'previous_high_low': 1.0 # Historical S/R
        }

    def _get_pip_value(self, epic: str = None) -> float:
        """Get pip value for currency pair"""
        return 0.01 if epic and 'JPY' in epic else 0.0001

    def _get_pip_multiplier(self, epic: str = None) -> float:
        """Get pip multiplier for currency pair"""
        return 100 if epic and 'JPY' in epic else 10000

    def _is_near_round_number(self, price: float, epic: str = None) -> Tuple[bool, Optional[float]]:
        """
        Check if price is near a round number.

        Round numbers for forex:
        - Major levels: .00 (1.1000, 1.2000)
        - Secondary levels: .50 (1.1050, 1.2050)
        - Minor levels: .00 for JPY pairs (150.00, 151.00)

        Args:
            price: Price to check
            epic: Currency pair

        Returns:
            Tuple of (is_near_round, round_number_price)
        """
        pip_value = self._get_pip_value(epic)
        tolerance = self.proximity_tolerance_pips * pip_value

        # Check major round numbers (.00)
        if epic and 'JPY' in epic:
            # JPY pairs: 150.00, 151.00, etc.
            round_price = round(price)
            if abs(price - round_price) <= tolerance:
                return True, round_price
        else:
            # Standard pairs: 1.1000, 1.1050, etc.
            # Check .00 levels
            round_price_major = round(price, 4)  # 1.1000
            decimal_part = Decimal(str(round_price_major)).as_tuple().digits[-2:]
            if decimal_part == (0, 0) and abs(price - round_price_major) <= tolerance:
                return True, round_price_major

            # Check .50 levels
            round_price_fifty = round(price * 200) / 200  # 1.1050
            if abs(price - round_price_fifty) <= tolerance:
                decimal_part_fifty = Decimal(str(round_price_fifty)).as_tuple().digits[-2:]
                if decimal_part_fifty[0] == 5 and decimal_part_fifty[1] == 0:
                    return True, round_price_fifty

        return False, None

    def _is_near_swing_level(self,
                            price: float,
                            swing_levels: List[float],
                            epic: str = None) -> Tuple[bool, Optional[float]]:
        """
        Check if price is near a swing high/low level.

        Args:
            price: Current price
            swing_levels: List of swing high/low prices
            epic: Currency pair

        Returns:
            Tuple of (is_near_swing, swing_level_price)
        """
        pip_value = self._get_pip_value(epic)
        tolerance = self.proximity_tolerance_pips * pip_value

        for swing_level in swing_levels:
            if abs(price - swing_level) <= tolerance:
                return True, swing_level

        return False, None

    def _is_near_ema(self,
                    price: float,
                    ema_values: Dict[str, float],
                    epic: str = None) -> List[Tuple[str, float]]:
        """
        Check if price is near EMA levels.

        Args:
            price: Current price
            ema_values: Dict of {'ema_21': value, 'ema_50': value, ...}
            epic: Currency pair

        Returns:
            List of tuples [(ema_name, ema_value), ...] for all EMAs near price
        """
        pip_value = self._get_pip_value(epic)
        tolerance = self.proximity_tolerance_pips * pip_value

        near_emas = []
        for ema_name, ema_value in ema_values.items():
            if ema_value and abs(price - ema_value) <= tolerance:
                near_emas.append((ema_name, ema_value))

        return near_emas

    def analyze_confluence_zone(self,
                               fib_level_price: float,
                               fib_level_name: str,
                               current_price: float,
                               swing_highs: List[float],
                               swing_lows: List[float],
                               ema_values: Dict[str, float],
                               epic: str = None) -> Dict:
        """
        Analyze confluence at a specific Fibonacci level.

        Args:
            fib_level_price: Fibonacci level price to analyze
            fib_level_name: Name of Fib level (e.g., '50.0', '61.8')
            current_price: Current market price
            swing_highs: List of recent swing high prices
            swing_lows: List of recent swing low prices
            ema_values: Dict of EMA values {'ema_21': float, 'ema_50': float}
            epic: Currency pair

        Returns:
            Dict with confluence analysis:
            {
                'fib_level': '50.0',
                'price': 1.0950,
                'confluence_score': 4.5,
                'factors': ['fibonacci', 'swing_high', 'ema_50'],
                'is_valid': True,
                'quality': 'high',  # 'low', 'medium', 'high', 'excellent'
                'distance_pips': 2.5
            }
        """
        pip_multiplier = self._get_pip_multiplier(epic)
        factors = ['fibonacci']  # Fib level is always present
        score = self.score_weights['fibonacci']

        # Check swing level confluence
        all_swing_levels = swing_highs + swing_lows
        is_near_swing, swing_price = self._is_near_swing_level(fib_level_price, all_swing_levels, epic)
        if is_near_swing:
            factors.append('swing_level')
            score += self.score_weights['swing_level']

        # Check round number confluence
        is_near_round, round_price = self._is_near_round_number(fib_level_price, epic)
        if is_near_round:
            factors.append('round_number')
            score += self.score_weights['round_number']

        # Check EMA confluence
        near_emas = self._is_near_ema(fib_level_price, ema_values, epic)
        for ema_name, ema_value in near_emas:
            factors.append(ema_name)
            score += self.score_weights.get(ema_name, 0.5)

        # Calculate distance from current price
        distance = abs(current_price - fib_level_price)
        distance_pips = distance * pip_multiplier

        # Determine quality based on score
        if score >= 5.0:
            quality = 'excellent'
        elif score >= 4.0:
            quality = 'high'
        elif score >= 3.0:
            quality = 'medium'
        else:
            quality = 'low'

        # Validate based on confluence mode
        is_valid = self._validate_confluence_mode(factors, score)

        result = {
            'fib_level': fib_level_name,
            'price': fib_level_price,
            'confluence_score': round(score, 2),
            'factors': factors,
            'factor_count': len(factors),
            'is_valid': is_valid,
            'quality': quality,
            'distance_from_price_pips': round(distance_pips, 1)
        }

        return result

    def _validate_confluence_mode(self, factors: List[str], score: float) -> bool:
        """
        Validate confluence based on mode setting.

        Args:
            factors: List of factors present at confluence
            score: Confluence score

        Returns:
            True if valid for current mode
        """
        factor_count = len(factors)

        if self.confluence_mode == 'strict':
            # Requires 3+ factors (Fib + 2 others)
            return factor_count >= 3 and score >= 4.0

        elif self.confluence_mode == 'moderate':
            # Requires 2+ factors (Fib + 1 other)
            return factor_count >= 2 and score >= self.min_confluence_score

        else:  # loose
            # Fib level is enough
            return score >= self.score_weights['fibonacci']

    def find_all_confluence_zones(self,
                                 fib_data: Dict,
                                 current_price: float,
                                 swing_highs: List[float],
                                 swing_lows: List[float],
                                 ema_values: Dict[str, float],
                                 epic: str = None) -> List[Dict]:
        """
        Find all valid confluence zones from Fibonacci levels.

        Args:
            fib_data: Fibonacci data from FibonacciCalculator
            current_price: Current market price
            swing_highs: Recent swing high prices
            swing_lows: Recent swing low prices
            ema_values: Current EMA values
            epic: Currency pair

        Returns:
            List of confluence zones sorted by score (highest first)
        """
        if not fib_data or 'fib_levels' not in fib_data:
            return []

        confluence_zones = []

        # Analyze each Fibonacci level
        for level_name, level_price in fib_data['fib_levels'].items():
            # Skip 0% and 100% (not retracement levels)
            if level_name in ['0', '100']:
                continue

            zone = self.analyze_confluence_zone(
                fib_level_price=level_price,
                fib_level_name=level_name,
                current_price=current_price,
                swing_highs=swing_highs,
                swing_lows=swing_lows,
                ema_values=ema_values,
                epic=epic
            )

            if zone['is_valid']:
                confluence_zones.append(zone)

        # Sort by confluence score (highest first)
        confluence_zones.sort(key=lambda x: x['confluence_score'], reverse=True)

        if confluence_zones:
            self.logger.info(f"🎯 Found {len(confluence_zones)} valid confluence zones")
            for zone in confluence_zones[:3]:  # Log top 3
                self.logger.info(f"  • {zone['fib_level']}%: {zone['price']:.5f} "
                               f"(score: {zone['confluence_score']}, quality: {zone['quality']}, "
                               f"factors: {', '.join(zone['factors'])})")

        return confluence_zones

    def is_price_at_confluence_zone(self,
                                   current_price: float,
                                   confluence_zones: List[Dict],
                                   epic: str = None,
                                   min_quality: str = 'low') -> Optional[Dict]:
        """
        Check if current price is at any confluence zone.

        Args:
            current_price: Current market price
            confluence_zones: List of confluence zones from find_all_confluence_zones()
            epic: Currency pair
            min_quality: Minimum quality required ('low', 'medium', 'high', 'excellent')

        Returns:
            Confluence zone dict if price is at zone, None otherwise
        """
        quality_rank = {'low': 1, 'medium': 2, 'high': 3, 'excellent': 4}
        min_quality_rank = quality_rank.get(min_quality, 1)

        pip_value = self._get_pip_value(epic)
        tolerance = self.proximity_tolerance_pips * pip_value

        for zone in confluence_zones:
            # Check quality requirement
            if quality_rank.get(zone['quality'], 0) < min_quality_rank:
                continue

            # Check if price is at zone
            if abs(current_price - zone['price']) <= tolerance:
                self.logger.info(f"✅ Price at confluence zone: {zone['fib_level']}% "
                               f"({zone['quality']} quality, score: {zone['confluence_score']})")
                return zone

        return None

    def get_best_confluence_zone(self, confluence_zones: List[Dict]) -> Optional[Dict]:
        """
        Get the highest quality confluence zone.

        Args:
            confluence_zones: List of confluence zones

        Returns:
            Best zone or None
        """
        if not confluence_zones:
            return None

        # Already sorted by score, so first is best
        return confluence_zones[0]


def test_confluence_analyzer():
    """Quick test of confluence analyzer"""

    analyzer = ConfluenceZoneAnalyzer(confluence_mode='moderate')

    # Test data
    fib_level_price = 1.0950
    current_price = 1.0948
    swing_highs = [1.0975, 1.0952]  # Near Fib level
    swing_lows = [1.0920]
    ema_values = {'ema_21': 1.0935, 'ema_50': 1.0951}  # ema_50 near Fib

    zone = analyzer.analyze_confluence_zone(
        fib_level_price=fib_level_price,
        fib_level_name='50.0',
        current_price=current_price,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        ema_values=ema_values,
        epic='EURUSD'
    )

    print(f"✅ Confluence zone analyzed")
    print(f"   Score: {zone['confluence_score']}")
    print(f"   Quality: {zone['quality']}")
    print(f"   Factors: {', '.join(zone['factors'])}")
    print(f"   Valid: {zone['is_valid']}")


if __name__ == '__main__':
    test_confluence_analyzer()
