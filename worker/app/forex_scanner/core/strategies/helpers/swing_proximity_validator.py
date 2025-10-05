# core/strategies/helpers/swing_proximity_validator.py
"""
Swing Proximity Validator
Validates trade entries against recent swing highs and lows to prevent poor entry timing
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, List
try:
    from .smc_market_structure import SMCMarketStructure, SwingType
except ImportError:
    from smc_market_structure import SMCMarketStructure, SwingType


class SwingProximityValidator:
    """
    Validates trade entries to ensure price is not too close to recent swing points

    Purpose:
    - Prevent BUY signals when price is near recent swing highs (resistance)
    - Prevent SELL signals when price is near recent swing lows (support)
    - Improve entry timing by avoiding immediate rejection zones
    """

    def __init__(self, smc_analyzer: Optional[SMCMarketStructure] = None,
                 config: Optional[Dict] = None, logger: logging.Logger = None):
        """
        Initialize swing proximity validator

        Args:
            smc_analyzer: SMC market structure analyzer instance
            config: Configuration dictionary with swing validation parameters
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.smc_analyzer = smc_analyzer
        self.config = config or {}

        # Configuration parameters with defaults
        self.enabled = self.config.get('enabled', True)
        self.min_distance_pips = self.config.get('min_distance_pips', 8)  # Can be dict or int
        self.timeframe_multipliers = self.config.get('timeframe_multipliers', {})
        self.lookback_swings = self.config.get('lookback_swings', 5)
        self.strict_mode = self.config.get('strict_mode', False)
        self.resistance_buffer = self.config.get('resistance_buffer', 1.0)  # No multiplier by default
        self.support_buffer = self.config.get('support_buffer', 1.0)  # No multiplier by default
        self.equal_level_multiplier = self.config.get('equal_level_multiplier', 2.0)  # 2x for EQH/EQL

        self.logger.debug(f"SwingProximityValidator initialized: min_distance={self.min_distance_pips}, "
                         f"lookback={self.lookback_swings}, strict={self.strict_mode}, "
                         f"equal_level_mult={self.equal_level_multiplier}")

    def validate_entry_proximity(self,
                                 df: pd.DataFrame,
                                 current_price: float,
                                 direction: str,
                                 epic: str = None,
                                 timeframe: str = '15m') -> Dict[str, Any]:
        """
        Validate if entry price is at safe distance from recent swing points

        Args:
            df: DataFrame with price data and swing analysis
            current_price: Current market price
            direction: 'BUY' or 'SELL'
            epic: Epic/symbol name (for pip calculation)

        Returns:
            Dictionary with validation result:
            {
                'valid': bool - True if price is safe distance from swings
                'distance_to_swing': float - Distance in pips to nearest critical swing
                'rejection_reason': str - Reason if invalid
                'nearest_swing_price': float - Price of nearest swing
                'swing_type': str - Type of swing (HH/LH/HL/LL)
                'confidence_penalty': float - Penalty to apply if not strict mode
            }
        """
        try:
            if not self.enabled:
                return {
                    'valid': True,
                    'distance_to_swing': None,
                    'rejection_reason': None,
                    'nearest_swing_price': None,
                    'swing_type': None,
                    'confidence_penalty': 0.0
                }

            # Get nearest swing levels
            swing_levels = self._get_swing_levels_from_df(df, current_price, direction)

            if not swing_levels:
                # No swing data available - allow trade but log warning
                self.logger.debug("No swing data available for proximity check")
                return {
                    'valid': True,
                    'distance_to_swing': None,
                    'rejection_reason': None,
                    'nearest_swing_price': None,
                    'swing_type': None,
                    'confidence_penalty': 0.0
                }

            # Calculate pip value based on epic
            pip_value = self._get_pip_value(epic)

            # Get adjusted minimum distance based on pair and timeframe
            adjusted_min_dist = self._get_adjusted_min_distance(epic, timeframe)

            # Validate based on direction
            if direction.upper() in ['BUY', 'BULL']:
                return self._validate_buy_proximity(
                    current_price,
                    swing_levels,
                    pip_value,
                    adjusted_min_dist
                )
            else:  # SELL/BEAR
                return self._validate_sell_proximity(
                    current_price,
                    swing_levels,
                    pip_value,
                    adjusted_min_dist
                )

        except Exception as e:
            self.logger.error(f"Swing proximity validation failed: {e}")
            # On error, allow trade but with warning
            return {
                'valid': True,
                'distance_to_swing': None,
                'rejection_reason': None,
                'nearest_swing_price': None,
                'swing_type': None,
                'confidence_penalty': 0.0
            }

    def _validate_buy_proximity(self,
                                current_price: float,
                                swing_levels: Dict,
                                pip_value: float,
                                base_min_distance: float = None) -> Dict[str, Any]:
        """Validate BUY signal - check distance to nearest swing high (resistance)"""

        nearest_resistance = swing_levels.get('nearest_resistance')
        resistance_type = swing_levels.get('swing_high_type', 'HH')

        if nearest_resistance is None or nearest_resistance <= current_price:
            # No resistance above or already above it
            return {
                'valid': True,
                'distance_to_swing': None,
                'rejection_reason': None,
                'nearest_swing_price': None,
                'swing_type': None,
                'confidence_penalty': 0.0
            }

        # Calculate distance in pips
        distance_pips = (nearest_resistance - current_price) / pip_value

        # Get minimum distance (use provided or fallback to config)
        min_dist = base_min_distance if base_min_distance is not None else self.min_distance_pips
        if isinstance(min_dist, dict):
            min_dist = min_dist.get('default', 12)

        # Apply resistance buffer (more cautious on buys near resistance)
        adjusted_min_distance = min_dist * self.resistance_buffer

        # Apply Equal High multiplier if this is an EQH
        if resistance_type in ['EQH', 'Equal High']:
            adjusted_min_distance *= self.equal_level_multiplier
            self.logger.debug(f"Equal High detected - applying {self.equal_level_multiplier}x distance requirement")

        # Get swing strength for weighting (if available)
        swing_strength = swing_levels.get('swing_strength', 1.0)

        if distance_pips < adjusted_min_distance:
            # Too close to resistance
            confidence_penalty = self._calculate_proximity_penalty(
                distance_pips,
                adjusted_min_distance,
                swing_strength
            )

            rejection_reason = (
                f"BUY signal too close to swing {resistance_type} resistance at {nearest_resistance:.5f} "
                f"(distance: {distance_pips:.1f} pips, minimum: {adjusted_min_distance:.1f} pips, "
                f"strength: {swing_strength:.2f})"
            )

            return {
                'valid': not self.strict_mode,  # Reject if strict mode
                'distance_to_swing': distance_pips,
                'rejection_reason': rejection_reason if self.strict_mode else None,
                'nearest_swing_price': nearest_resistance,
                'swing_type': resistance_type,
                'confidence_penalty': confidence_penalty
            }

        # Safe distance from resistance
        return {
            'valid': True,
            'distance_to_swing': distance_pips,
            'rejection_reason': None,
            'nearest_swing_price': nearest_resistance,
            'swing_type': resistance_type,
            'confidence_penalty': 0.0
        }

    def _validate_sell_proximity(self,
                                 current_price: float,
                                 swing_levels: Dict,
                                 pip_value: float,
                                 base_min_distance: float = None) -> Dict[str, Any]:
        """Validate SELL signal - check distance to nearest swing low (support)"""

        nearest_support = swing_levels.get('nearest_support')
        support_type = swing_levels.get('swing_low_type', 'LL')

        if nearest_support is None or nearest_support >= current_price:
            # No support below or already below it
            return {
                'valid': True,
                'distance_to_swing': None,
                'rejection_reason': None,
                'nearest_swing_price': None,
                'swing_type': None,
                'confidence_penalty': 0.0
            }

        # Calculate distance in pips
        distance_pips = (current_price - nearest_support) / pip_value

        # Get minimum distance (use provided or fallback to config)
        min_dist = base_min_distance if base_min_distance is not None else self.min_distance_pips
        if isinstance(min_dist, dict):
            min_dist = min_dist.get('default', 12)

        # Apply support buffer (more cautious on sells near support)
        adjusted_min_distance = min_dist * self.support_buffer

        # Apply Equal Low multiplier if this is an EQL
        if support_type in ['EQL', 'Equal Low']:
            adjusted_min_distance *= self.equal_level_multiplier
            self.logger.debug(f"Equal Low detected - applying {self.equal_level_multiplier}x distance requirement")

        # Get swing strength for weighting (if available)
        swing_strength = swing_levels.get('swing_strength', 1.0)

        if distance_pips < adjusted_min_distance:
            # Too close to support
            confidence_penalty = self._calculate_proximity_penalty(
                distance_pips,
                adjusted_min_distance,
                swing_strength
            )

            rejection_reason = (
                f"SELL signal too close to swing {support_type} support at {nearest_support:.5f} "
                f"(distance: {distance_pips:.1f} pips, minimum: {adjusted_min_distance:.1f} pips, "
                f"strength: {swing_strength:.2f})"
            )

            return {
                'valid': not self.strict_mode,  # Reject if strict mode
                'distance_to_swing': distance_pips,
                'rejection_reason': rejection_reason if self.strict_mode else None,
                'nearest_swing_price': nearest_support,
                'swing_type': support_type,
                'confidence_penalty': confidence_penalty
            }

        # Safe distance from support
        return {
            'valid': True,
            'distance_to_swing': distance_pips,
            'rejection_reason': None,
            'nearest_swing_price': nearest_support,
            'swing_type': support_type,
            'confidence_penalty': 0.0
        }

    def _calculate_proximity_penalty(self, actual_distance: float, min_distance: float,
                                     swing_strength: float = 1.0) -> float:
        """
        Calculate confidence penalty based on proximity violation and swing strength

        Args:
            actual_distance: Actual distance to swing in pips
            min_distance: Minimum required distance in pips
            swing_strength: Swing strength (0-5 scale, default 1.0)

        Returns penalty between 0.10 and 0.45 based on severity and strength
        """
        if actual_distance >= min_distance:
            return 0.0

        # Calculate violation ratio
        violation_ratio = actual_distance / min_distance  # 0.0 to 1.0

        # Strength multiplier (1.0 to 1.5 based on swing significance)
        # Higher strength swings = higher penalty
        strength_multiplier = 1.0 + (min(swing_strength, 5.0) * 0.1)  # 1.0 to 1.5

        # Base penalty calculation
        max_penalty = 0.30  # Increased from 0.20
        min_penalty = 0.10  # Increased from 0.05

        base_penalty = max_penalty - (violation_ratio * (max_penalty - min_penalty))

        # Apply strength multiplier
        final_penalty = base_penalty * strength_multiplier

        # Cap at maximum of 0.45
        return max(min_penalty, min(0.45, final_penalty))

    def _get_swing_levels_from_df(self,
                                   df: pd.DataFrame,
                                   current_price: float,
                                   direction: str) -> Optional[Dict]:
        """
        Extract swing levels from DataFrame (with swing point columns)
        Falls back to SMC analyzer if available
        """
        try:
            # Check if DataFrame has swing point columns
            if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
                # Try to use SMC analyzer
                if self.smc_analyzer:
                    return self.smc_analyzer.get_nearest_swing_levels(
                        current_price,
                        direction,
                        self.lookback_swings
                    )
                return None

            # Get recent swing highs and lows from DataFrame
            recent_df = df.tail(100)  # Look at last 100 bars

            # Find swing highs (resistance)
            swing_highs = recent_df[recent_df['swing_high'] == True].tail(self.lookback_swings)
            nearest_resistance = None
            swing_high_type = 'HH'

            if not swing_highs.empty:
                # Get the nearest swing high above current price
                highs_above = swing_highs[swing_highs['high'] > current_price]
                if not highs_above.empty:
                    nearest_idx = highs_above['high'].idxmin()
                    nearest_resistance = highs_above.loc[nearest_idx, 'high']
                    if 'swing_type' in highs_above.columns:
                        swing_high_type = highs_above.loc[nearest_idx, 'swing_type']
                else:
                    # All swings below, get the highest one
                    nearest_idx = swing_highs['high'].idxmax()
                    nearest_resistance = swing_highs.loc[nearest_idx, 'high']
                    if 'swing_type' in swing_highs.columns:
                        swing_high_type = swing_highs.loc[nearest_idx, 'swing_type']

            # Find swing lows (support)
            swing_lows = recent_df[recent_df['swing_low'] == True].tail(self.lookback_swings)
            nearest_support = None
            swing_low_type = 'LL'

            if not swing_lows.empty:
                # Get the nearest swing low below current price
                lows_below = swing_lows[swing_lows['low'] < current_price]
                if not lows_below.empty:
                    nearest_idx = lows_below['low'].idxmax()
                    nearest_support = lows_below.loc[nearest_idx, 'low']
                    if 'swing_type' in lows_below.columns:
                        swing_low_type = lows_below.loc[nearest_idx, 'swing_type']
                else:
                    # All swings above, get the lowest one
                    nearest_idx = swing_lows['low'].idxmin()
                    nearest_support = swing_lows.loc[nearest_idx, 'low']
                    if 'swing_type' in swing_lows.columns:
                        swing_low_type = swing_lows.loc[nearest_idx, 'swing_type']

            return {
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'swing_high_type': swing_high_type,
                'swing_low_type': swing_low_type,
                'resistance_distance_pips': None,  # Calculated later
                'support_distance_pips': None      # Calculated later
            }

        except Exception as e:
            self.logger.error(f"Failed to extract swing levels from DataFrame: {e}")
            return None

    def _get_adjusted_min_distance(self, epic: Optional[str], timeframe: str = '15m') -> float:
        """
        Get adjusted minimum distance based on pair and timeframe

        Args:
            epic: Trading pair symbol
            timeframe: Trading timeframe (e.g., '5m', '15m', '1h')

        Returns:
            Adjusted minimum distance in pips
        """
        # Get base distance from config
        if isinstance(self.min_distance_pips, dict):
            # Pair-specific configuration
            base_distance = self.min_distance_pips.get('default', 12)

            if epic:
                # Check for specific pair matches
                for currency in ['JPY', 'GBP', 'EUR', 'AUD', 'NZD', 'CAD', 'CHF']:
                    if currency in epic.upper():
                        base_distance = self.min_distance_pips.get(currency, base_distance)
                        break
        else:
            # Single value configuration
            base_distance = self.min_distance_pips

        # Apply timeframe multiplier
        tf_multiplier = self.timeframe_multipliers.get(timeframe, 1.0)
        adjusted_distance = base_distance * tf_multiplier

        self.logger.debug(f"Adjusted min distance for {epic} on {timeframe}: "
                         f"{base_distance} pips × {tf_multiplier} = {adjusted_distance:.1f} pips")

        return adjusted_distance

    def _get_pip_value(self, epic: Optional[str]) -> float:
        """
        Get pip value based on epic/symbol

        JPY pairs: 0.01 (3 decimals)
        Other pairs: 0.0001 (5 decimals)
        """
        if epic and 'JPY' in epic.upper():
            return 0.01
        return 0.0001
