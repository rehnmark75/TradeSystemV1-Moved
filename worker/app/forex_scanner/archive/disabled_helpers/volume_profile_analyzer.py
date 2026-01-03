# core/strategies/helpers/volume_profile_analyzer.py
"""
Volume Profile Analyzer - Helper Functions for Signal Generation

This module provides convenience methods for analyzing VolumeProfile data
and identifying trading opportunities. Used by VolumeProfileStrategy for
signal detection.

Key Functions:
- Find nearest HVN/LVN zones to current price
- Check if price is at institutional levels
- Calculate distance metrics for signal confidence
- Identify support/resistance levels from volume nodes
- Analyze price position relative to Value Area
"""

from typing import Optional, List, Tuple, Dict
import numpy as np

try:
    from analysis.volume_profile import VolumeProfile, VolumeNode
except ImportError:
    from forex_scanner.analysis.volume_profile import VolumeProfile, VolumeNode


class VolumeProfileAnalyzer:
    """
    Analyzes VolumeProfile data to generate trading signals and insights.

    This class provides methods for:
    - Finding nearest HVN/LVN zones
    - Checking price position relative to key levels
    - Calculating distance metrics
    - Identifying support/resistance levels
    - Analyzing volume distribution patterns
    """

    def __init__(self, pip_value: float = 0.0001):
        """
        Initialize analyzer with pip value for distance calculations.

        Args:
            pip_value: Pip value for the currency pair (0.0001 for most pairs, 0.01 for JPY)
        """
        self.pip_value = pip_value

    # ==================== PROXIMITY CHECKS ====================

    def is_at_hvn(self, current_price: float, profile: VolumeProfile,
                   threshold_pips: float = 5.0) -> Tuple[bool, Optional[VolumeNode]]:
        """
        Check if current price is at/near a High Volume Node.

        HVN zones act as support/resistance where institutions accumulated.
        Being at HVN suggests potential bounce or rejection.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile
            threshold_pips: Distance threshold in pips (default 5.0)

        Returns:
            Tuple of (is_at_hvn: bool, nearest_hvn: Optional[VolumeNode])
        """
        if not profile.hvn_zones:
            return False, None

        threshold = threshold_pips * self.pip_value
        nearest_hvn = self.get_nearest_hvn(current_price, profile)

        if nearest_hvn is None:
            return False, None

        # Check if price is within HVN zone boundaries
        if nearest_hvn.price_low <= current_price <= nearest_hvn.price_high:
            return True, nearest_hvn

        # Check if price is near HVN zone (within threshold)
        distance_to_zone = min(
            abs(current_price - nearest_hvn.price_low),
            abs(current_price - nearest_hvn.price_high),
            abs(current_price - nearest_hvn.price_center)
        )

        is_near = distance_to_zone <= threshold
        return is_near, nearest_hvn if is_near else None

    def is_at_lvn(self, current_price: float, profile: VolumeProfile,
                   threshold_pips: float = 5.0) -> Tuple[bool, Optional[VolumeNode]]:
        """
        Check if current price is at/near a Low Volume Node.

        LVN zones are rejection areas with minimal volume. Price tends to
        move quickly through these zones, making them potential breakout levels.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile
            threshold_pips: Distance threshold in pips (default 5.0)

        Returns:
            Tuple of (is_at_lvn: bool, nearest_lvn: Optional[VolumeNode])
        """
        if not profile.lvn_zones:
            return False, None

        threshold = threshold_pips * self.pip_value
        nearest_lvn = self.get_nearest_lvn(current_price, profile)

        if nearest_lvn is None:
            return False, None

        # Check if price is within LVN zone boundaries
        if nearest_lvn.price_low <= current_price <= nearest_lvn.price_high:
            return True, nearest_lvn

        # Check if price is near LVN zone
        distance_to_zone = min(
            abs(current_price - nearest_lvn.price_low),
            abs(current_price - nearest_lvn.price_high),
            abs(current_price - nearest_lvn.price_center)
        )

        is_near = distance_to_zone <= threshold
        return is_near, nearest_lvn if is_near else None

    def is_within_value_area(self, current_price: float, profile: VolumeProfile) -> bool:
        """
        Check if current price is within the Value Area (VAL to VAH).

        Value Area contains 70% of volume and represents fair value range.
        Price inside Value Area is considered balanced/consolidating.
        Price outside Value Area suggests trending or extreme conditions.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile

        Returns:
            True if price is within VAL and VAH boundaries
        """
        return profile.val <= current_price <= profile.vah

    def is_at_poc(self, current_price: float, profile: VolumeProfile,
                   threshold_pips: float = 3.0) -> bool:
        """
        Check if current price is at/near Point of Control.

        POC is the highest volume price level - acts as a magnet and pivot point.
        Price often gravitates toward POC or bounces when it reaches it.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile
            threshold_pips: Distance threshold in pips (default 3.0)

        Returns:
            True if price is within threshold of POC
        """
        threshold = threshold_pips * self.pip_value
        distance = abs(current_price - profile.poc)
        return distance <= threshold

    # ==================== DISTANCE CALCULATIONS ====================

    def distance_to_poc_pips(self, current_price: float, profile: VolumeProfile) -> float:
        """
        Calculate distance from current price to Point of Control in pips.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile

        Returns:
            Distance in pips (positive = above POC, negative = below POC)
        """
        distance = current_price - profile.poc
        return distance / self.pip_value

    def distance_to_value_area_pips(self, current_price: float, profile: VolumeProfile) -> float:
        """
        Calculate distance from current price to nearest Value Area boundary.

        Returns 0 if price is within Value Area.
        Returns positive value if above VAH.
        Returns negative value if below VAL.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile

        Returns:
            Distance in pips to nearest VA boundary (0 if inside VA)
        """
        if self.is_within_value_area(current_price, profile):
            return 0.0

        if current_price > profile.vah:
            distance = current_price - profile.vah
            return distance / self.pip_value
        else:  # price < profile.val
            distance = current_price - profile.val
            return distance / self.pip_value

    def distance_to_nearest_hvn_pips(self, current_price: float, profile: VolumeProfile) -> Optional[float]:
        """
        Calculate distance from current price to nearest HVN center.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile

        Returns:
            Distance in pips (None if no HVN zones exist)
        """
        nearest_hvn = self.get_nearest_hvn(current_price, profile)
        if nearest_hvn is None:
            return None

        distance = current_price - nearest_hvn.price_center
        return distance / self.pip_value

    # ==================== LEVEL IDENTIFICATION ====================

    def get_nearest_hvn(self, current_price: float, profile: VolumeProfile) -> Optional[VolumeNode]:
        """
        Find the nearest High Volume Node to current price.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile

        Returns:
            Nearest VolumeNode or None if no HVN zones exist
        """
        if not profile.hvn_zones:
            return None

        # Find HVN with minimum distance to current price
        nearest = min(
            profile.hvn_zones,
            key=lambda hvn: abs(current_price - hvn.price_center)
        )
        return nearest

    def get_nearest_lvn(self, current_price: float, profile: VolumeProfile) -> Optional[VolumeNode]:
        """
        Find the nearest Low Volume Node to current price.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile

        Returns:
            Nearest VolumeNode or None if no LVN zones exist
        """
        if not profile.lvn_zones:
            return None

        # Find LVN with minimum distance to current price
        nearest = min(
            profile.lvn_zones,
            key=lambda lvn: abs(current_price - lvn.price_center)
        )
        return nearest

    def get_support_resistance_levels(self, profile: VolumeProfile) -> Dict[str, List[float]]:
        """
        Extract key support/resistance levels from Volume Profile.

        Returns HVN centers as support/resistance levels, plus POC, VAH, and VAL.
        These levels are used for stop loss/take profit placement and signal filtering.

        Args:
            profile: Calculated VolumeProfile

        Returns:
            Dictionary with 'resistance' and 'support' lists of price levels
        """
        # All key levels (HVNs + POC + VAH + VAL)
        all_levels = [profile.poc, profile.vah, profile.val]
        all_levels.extend([hvn.price_center for hvn in profile.hvn_zones])

        # Sort levels
        all_levels = sorted(set(all_levels))

        # Split into support (below) and resistance (above) - will be determined by strategy based on current price
        # Here we just return all levels sorted
        return {
            'all_levels': all_levels,
            'hvn_levels': sorted([hvn.price_center for hvn in profile.hvn_zones]),
            'lvn_levels': sorted([lvn.price_center for lvn in profile.lvn_zones]),
            'key_levels': {
                'poc': profile.poc,
                'vah': profile.vah,
                'val': profile.val
            }
        }

    def get_hvn_above_below(self, current_price: float, profile: VolumeProfile) -> Tuple[Optional[VolumeNode], Optional[VolumeNode]]:
        """
        Get nearest HVN zones above and below current price.

        Useful for identifying immediate resistance (above) and support (below).

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile

        Returns:
            Tuple of (hvn_above: Optional[VolumeNode], hvn_below: Optional[VolumeNode])
        """
        if not profile.hvn_zones:
            return None, None

        hvn_above = None
        hvn_below = None

        hvns_above = [hvn for hvn in profile.hvn_zones if hvn.price_center > current_price]
        hvns_below = [hvn for hvn in profile.hvn_zones if hvn.price_center < current_price]

        if hvns_above:
            hvn_above = min(hvns_above, key=lambda hvn: hvn.price_center - current_price)

        if hvns_below:
            hvn_below = max(hvns_below, key=lambda hvn: hvn.price_center)

        return hvn_above, hvn_below

    # ==================== SIGNAL ANALYSIS ====================

    def analyze_price_position(self, current_price: float, profile: VolumeProfile) -> Dict[str, any]:
        """
        Comprehensive analysis of price position relative to Volume Profile.

        Returns complete snapshot of price position including:
        - Value Area position
        - POC proximity
        - Nearest HVN/LVN zones
        - Support/resistance levels
        - Distance metrics

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile

        Returns:
            Dictionary with complete position analysis
        """
        is_at_hvn_flag, nearest_hvn = self.is_at_hvn(current_price, profile)
        is_at_lvn_flag, nearest_lvn = self.is_at_lvn(current_price, profile)
        hvn_above, hvn_below = self.get_hvn_above_below(current_price, profile)

        # Determine price position category
        if current_price > profile.vah:
            position = "ABOVE_VALUE_AREA"
            bias = "BULLISH_EXTREME"
        elif current_price < profile.val:
            position = "BELOW_VALUE_AREA"
            bias = "BEARISH_EXTREME"
        elif self.is_at_poc(current_price, profile):
            position = "AT_POC"
            bias = "NEUTRAL_PIVOT"
        else:
            position = "WITHIN_VALUE_AREA"
            bias = "NEUTRAL_BALANCED"

        return {
            # Position categorization
            'position': position,
            'bias': bias,
            'within_value_area': self.is_within_value_area(current_price, profile),
            'at_poc': self.is_at_poc(current_price, profile),
            'at_hvn': is_at_hvn_flag,
            'at_lvn': is_at_lvn_flag,

            # Distance metrics (pips)
            'distance_to_poc_pips': self.distance_to_poc_pips(current_price, profile),
            'distance_to_va_pips': self.distance_to_value_area_pips(current_price, profile),
            'distance_to_hvn_pips': self.distance_to_nearest_hvn_pips(current_price, profile),

            # Nearest zones
            'nearest_hvn': nearest_hvn,
            'nearest_lvn': nearest_lvn,
            'hvn_above': hvn_above,
            'hvn_below': hvn_below,

            # Key levels
            'poc': profile.poc,
            'vah': profile.vah,
            'val': profile.val,

            # Volume statistics
            'volume_skewness': profile.volume_skewness,  # Positive = volume at higher prices
            'hvn_count': len(profile.hvn_zones),
            'lvn_count': len(profile.lvn_zones),
        }

    def get_signal_confidence(self, current_price: float, profile: VolumeProfile,
                             signal_type: str) -> float:
        """
        Calculate signal confidence based on Volume Profile alignment.

        Higher confidence when:
        - BUY: Price at strong HVN support, below Value Area, skewness bullish
        - SELL: Price at strong HVN resistance, above Value Area, skewness bearish

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile
            signal_type: 'BUY' or 'SELL'

        Returns:
            Confidence score 0.0 to 1.0
        """
        position_analysis = self.analyze_price_position(current_price, profile)
        confidence = 0.5  # Base confidence

        if signal_type == 'BUY':
            # Bullish signal confidence factors
            if position_analysis['position'] == 'BELOW_VALUE_AREA':
                confidence += 0.15  # Price below fair value = potential mean reversion

            if position_analysis['at_hvn'] and position_analysis['nearest_hvn']:
                # At HVN support
                hvn_strength = position_analysis['nearest_hvn'].strength
                confidence += 0.20 * hvn_strength  # Stronger HVN = higher confidence

            if profile.volume_skewness > 0.2:
                # Volume concentrated at higher prices = bullish accumulation
                confidence += 0.10

            if position_analysis['distance_to_poc_pips'] and position_analysis['distance_to_poc_pips'] < 0:
                # Price below POC = potential bounce to POC
                confidence += 0.05

        elif signal_type == 'SELL':
            # Bearish signal confidence factors
            if position_analysis['position'] == 'ABOVE_VALUE_AREA':
                confidence += 0.15  # Price above fair value = potential mean reversion

            if position_analysis['at_hvn'] and position_analysis['nearest_hvn']:
                # At HVN resistance
                hvn_strength = position_analysis['nearest_hvn'].strength
                confidence += 0.20 * hvn_strength  # Stronger HVN = higher confidence

            if profile.volume_skewness < -0.2:
                # Volume concentrated at lower prices = bearish distribution
                confidence += 0.10

            if position_analysis['distance_to_poc_pips'] and position_analysis['distance_to_poc_pips'] > 0:
                # Price above POC = potential drop to POC
                confidence += 0.05

        # Clamp confidence to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def should_filter_signal(self, current_price: float, profile: VolumeProfile,
                            signal_type: str, min_confidence: float = 0.6) -> Tuple[bool, str]:
        """
        Determine if a signal should be filtered out based on Volume Profile analysis.

        Filters signals that don't meet minimum confidence or have conflicting volume profile.

        Args:
            current_price: Current market price
            profile: Calculated VolumeProfile
            signal_type: 'BUY' or 'SELL'
            min_confidence: Minimum confidence threshold (default 0.6)

        Returns:
            Tuple of (should_filter: bool, reason: str)
        """
        confidence = self.get_signal_confidence(current_price, profile, signal_type)

        if confidence < min_confidence:
            return True, f"Low confidence ({confidence:.2f} < {min_confidence})"

        position_analysis = self.analyze_price_position(current_price, profile)

        # Filter BUY signals in unfavorable conditions
        if signal_type == 'BUY':
            if position_analysis['at_lvn']:
                return True, "At LVN rejection zone (weak support)"

            if position_analysis['position'] == 'ABOVE_VALUE_AREA' and not position_analysis['at_hvn']:
                # Above value area without HVN support = risky long
                distance = position_analysis['distance_to_va_pips']
                if distance > 20:  # More than 20 pips above VA
                    return True, f"Too far above Value Area ({distance:.1f} pips)"

        # Filter SELL signals in unfavorable conditions
        elif signal_type == 'SELL':
            if position_analysis['at_lvn']:
                return True, "At LVN rejection zone (weak resistance)"

            if position_analysis['position'] == 'BELOW_VALUE_AREA' and not position_analysis['at_hvn']:
                # Below value area without HVN resistance = risky short
                distance = position_analysis['distance_to_va_pips']
                if distance < -20:  # More than 20 pips below VA
                    return True, f"Too far below Value Area ({abs(distance):.1f} pips)"

        return False, "Signal passed Volume Profile filters"
