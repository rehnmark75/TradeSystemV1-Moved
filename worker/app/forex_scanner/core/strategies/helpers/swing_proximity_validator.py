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
        self.min_distance_pips = self.config.get('min_distance_pips', 8)  # 8 pips = 80 IG points (practical for 5m/15m)
        self.lookback_swings = self.config.get('lookback_swings', 5)
        self.strict_mode = self.config.get('strict_mode', False)
        self.resistance_buffer = self.config.get('resistance_buffer', 1.0)  # No multiplier by default
        self.support_buffer = self.config.get('support_buffer', 1.0)  # No multiplier by default

        self.logger.debug(f"SwingProximityValidator initialized: min_distance={self.min_distance_pips} pips, "
                         f"lookback={self.lookback_swings}, strict={self.strict_mode}")

    def validate_entry_proximity(self,
                                 df: pd.DataFrame,
                                 current_price: float,
                                 direction: str,
                                 epic: str = None) -> Dict[str, Any]:
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

            # Validate based on direction
            if direction.upper() in ['BUY', 'BULL']:
                return self._validate_buy_proximity(
                    current_price,
                    swing_levels,
                    pip_value
                )
            else:  # SELL/BEAR
                return self._validate_sell_proximity(
                    current_price,
                    swing_levels,
                    pip_value
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
                                pip_value: float) -> Dict[str, Any]:
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

        # Apply resistance buffer (more cautious on buys near resistance)
        adjusted_min_distance = self.min_distance_pips * self.resistance_buffer

        if distance_pips < adjusted_min_distance:
            # Too close to resistance
            confidence_penalty = self._calculate_proximity_penalty(
                distance_pips,
                adjusted_min_distance
            )

            rejection_reason = (
                f"BUY signal too close to swing {resistance_type} resistance at {nearest_resistance:.5f} "
                f"(distance: {distance_pips:.1f} pips, minimum: {adjusted_min_distance:.1f} pips)"
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
                                 pip_value: float) -> Dict[str, Any]:
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

        # Apply support buffer (more cautious on sells near support)
        adjusted_min_distance = self.min_distance_pips * self.support_buffer

        if distance_pips < adjusted_min_distance:
            # Too close to support
            confidence_penalty = self._calculate_proximity_penalty(
                distance_pips,
                adjusted_min_distance
            )

            rejection_reason = (
                f"SELL signal too close to swing {support_type} support at {nearest_support:.5f} "
                f"(distance: {distance_pips:.1f} pips, minimum: {adjusted_min_distance:.1f} pips)"
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

    def _calculate_proximity_penalty(self, actual_distance: float, min_distance: float) -> float:
        """
        Calculate confidence penalty based on proximity violation

        Returns penalty between 0.05 and 0.20 based on severity
        """
        if actual_distance >= min_distance:
            return 0.0

        # Calculate violation ratio
        violation_ratio = actual_distance / min_distance  # 0.0 to 1.0

        # Linear penalty: closer = higher penalty
        # 0 pips away = 0.20 penalty
        # min_distance pips away = 0.0 penalty
        max_penalty = 0.20
        min_penalty = 0.05

        penalty = max_penalty - (violation_ratio * (max_penalty - min_penalty))

        return max(min_penalty, min(max_penalty, penalty))

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

    def _get_pip_value(self, epic: Optional[str]) -> float:
        """
        Get pip value based on epic/symbol

        JPY pairs: 0.01 (3 decimals)
        Other pairs: 0.0001 (5 decimals)
        """
        if epic and 'JPY' in epic.upper():
            return 0.01
        return 0.0001
