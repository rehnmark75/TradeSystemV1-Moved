"""
Silver Bullet Liquidity Detection Helper

Handles liquidity level detection and sweep identification for the ICT Silver Bullet strategy.

Key Concepts:
- Buy-Side Liquidity (BSL): Stop losses above swing highs - price sweeps these before reversing down
- Sell-Side Liquidity (SSL): Stop losses below swing lows - price sweeps these before reversing up

The Silver Bullet looks for:
1. Liquidity sweep (price takes out highs/lows)
2. Rejection after sweep (confirms it's a sweep, not a breakout)
3. Market Structure Shift in opposite direction
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class LiquidityType(Enum):
    """Types of liquidity"""
    BSL = "BSL"  # Buy-Side Liquidity (above highs)
    SSL = "SSL"  # Sell-Side Liquidity (below lows)


class SweepStatus(Enum):
    """Status of a liquidity sweep"""
    CLEAN = "CLEAN"              # Clean sweep with rejection
    PARTIAL = "PARTIAL"          # Partial sweep
    BREAKOUT = "BREAKOUT"        # Not a sweep - price continued through
    PENDING = "PENDING"          # Sweep happened, waiting for confirmation


@dataclass
class LiquidityLevel:
    """Represents a liquidity level (swing high or low)"""
    price: float
    index: int
    liquidity_type: LiquidityType
    strength: int  # How many bars confirm this as a swing
    touched_count: int = 0
    swept: bool = False
    sweep_index: Optional[int] = None


@dataclass
class LiquiditySweep:
    """Represents a liquidity sweep event"""
    liquidity_level: LiquidityLevel
    sweep_price: float  # How far price went beyond the level
    sweep_index: int
    sweep_pips: float
    status: SweepStatus
    rejection_confirmed: bool = False
    rejection_candles: int = 0  # How many candles since sweep


class SilverBulletLiquidity:
    """
    Detects liquidity levels and sweeps for ICT Silver Bullet strategy.

    The core idea:
    1. Find swing highs/lows (these are liquidity pools)
    2. Wait for price to sweep these levels
    3. Confirm sweep with rejection (price returns)
    4. This triggers the Silver Bullet setup
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.liquidity_levels: List[LiquidityLevel] = []
        self.recent_sweeps: List[LiquiditySweep] = []

    def detect_liquidity_levels(
        self,
        df: pd.DataFrame,
        lookback_bars: int = 20,
        swing_strength: int = 3,
        pip_value: float = 0.0001
    ) -> List[LiquidityLevel]:
        """
        Detect liquidity levels (swing highs and lows) in the price data.

        Args:
            df: OHLCV DataFrame
            lookback_bars: How many bars to look back for levels
            swing_strength: Bars on each side to confirm a swing
            pip_value: Pip value for the pair (0.0001 for most, 0.01 for JPY)

        Returns:
            List of detected liquidity levels
        """
        try:
            self.liquidity_levels = []

            if len(df) < lookback_bars + swing_strength * 2:
                self.logger.warning(f"Insufficient data for liquidity detection")
                return []

            highs = df['high'].values
            lows = df['low'].values

            # Only look at recent bars (within lookback)
            start_idx = max(swing_strength, len(df) - lookback_bars)
            end_idx = len(df) - swing_strength  # Can't confirm swings at the end

            for i in range(start_idx, end_idx):
                # Check for swing high (BSL - buy-side liquidity above)
                is_swing_high = self._is_swing_high(highs, i, swing_strength)
                if is_swing_high:
                    level = LiquidityLevel(
                        price=highs[i],
                        index=i,
                        liquidity_type=LiquidityType.BSL,
                        strength=swing_strength
                    )
                    self.liquidity_levels.append(level)

                # Check for swing low (SSL - sell-side liquidity below)
                is_swing_low = self._is_swing_low(lows, i, swing_strength)
                if is_swing_low:
                    level = LiquidityLevel(
                        price=lows[i],
                        index=i,
                        liquidity_type=LiquidityType.SSL,
                        strength=swing_strength
                    )
                    self.liquidity_levels.append(level)

            # Sort by recency (most recent first)
            self.liquidity_levels.sort(key=lambda x: x.index, reverse=True)

            self.logger.debug(
                f"Detected {len(self.liquidity_levels)} liquidity levels: "
                f"{len([l for l in self.liquidity_levels if l.liquidity_type == LiquidityType.BSL])} BSL, "
                f"{len([l for l in self.liquidity_levels if l.liquidity_type == LiquidityType.SSL])} SSL"
            )

            return self.liquidity_levels

        except Exception as e:
            self.logger.error(f"Error detecting liquidity levels: {e}")
            return []

    def detect_liquidity_sweep(
        self,
        df: pd.DataFrame,
        liquidity_levels: List[LiquidityLevel],
        min_sweep_pips: float = 3.0,
        max_sweep_pips: float = 15.0,
        pip_value: float = 0.0001,
        require_rejection: bool = True,
        max_sweep_age: int = 10
    ) -> Optional[LiquiditySweep]:
        """
        Detect if a liquidity sweep has occurred recently.

        A sweep occurs when:
        1. Price breaks beyond a liquidity level
        2. Price doesn't continue much further (not a breakout)
        3. Price returns back past the level (rejection)

        Args:
            df: OHLCV DataFrame
            liquidity_levels: List of detected liquidity levels
            min_sweep_pips: Minimum sweep beyond level (pips)
            max_sweep_pips: Maximum sweep - beyond this it's a breakout
            pip_value: Pip value for the pair
            require_rejection: Whether to require price rejection after sweep
            max_sweep_age: Maximum bars since sweep to consider valid

        Returns:
            LiquiditySweep if found, None otherwise
        """
        try:
            self.recent_sweeps = []

            if not liquidity_levels or len(df) < 5:
                return None

            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            current_idx = len(df) - 1

            # Check each liquidity level for a sweep
            for level in liquidity_levels:
                # Skip if level is too old
                if current_idx - level.index > max_sweep_age + 10:
                    continue

                sweep = self._check_level_for_sweep(
                    df=df,
                    level=level,
                    min_sweep_pips=min_sweep_pips,
                    max_sweep_pips=max_sweep_pips,
                    pip_value=pip_value,
                    require_rejection=require_rejection,
                    max_sweep_age=max_sweep_age
                )

                if sweep:
                    self.recent_sweeps.append(sweep)

            # Return the most recent valid sweep
            if self.recent_sweeps:
                # Sort by sweep index (most recent first)
                self.recent_sweeps.sort(key=lambda x: x.sweep_index, reverse=True)

                # Return first clean or partial sweep
                for sweep in self.recent_sweeps:
                    if sweep.status in [SweepStatus.CLEAN, SweepStatus.PARTIAL]:
                        self.logger.info(
                            f"Found {sweep.status.value} {sweep.liquidity_level.liquidity_type.value} sweep: "
                            f"Level {sweep.liquidity_level.price:.5f}, "
                            f"Swept by {sweep.sweep_pips:.1f} pips, "
                            f"{sweep.rejection_candles} bars ago"
                        )
                        return sweep

            return None

        except Exception as e:
            self.logger.error(f"Error detecting liquidity sweep: {e}")
            return None

    def _check_level_for_sweep(
        self,
        df: pd.DataFrame,
        level: LiquidityLevel,
        min_sweep_pips: float,
        max_sweep_pips: float,
        pip_value: float,
        require_rejection: bool,
        max_sweep_age: int
    ) -> Optional[LiquiditySweep]:
        """Check if a specific level has been swept"""
        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            current_idx = len(df) - 1

            # Convert pip thresholds to price
            min_sweep_price = min_sweep_pips * pip_value
            max_sweep_price = max_sweep_pips * pip_value

            sweep_idx = None
            sweep_price = None
            sweep_distance = 0

            # Look for sweep in bars after the swing was formed
            search_start = level.index + level.strength
            search_end = min(current_idx + 1, level.index + max_sweep_age + level.strength)

            if level.liquidity_type == LiquidityType.BSL:
                # BSL sweep: price goes above the swing high
                for i in range(search_start, search_end):
                    if highs[i] > level.price:
                        distance = highs[i] - level.price
                        if distance > sweep_distance:
                            sweep_distance = distance
                            sweep_price = highs[i]
                            sweep_idx = i

            else:  # SSL
                # SSL sweep: price goes below the swing low
                for i in range(search_start, search_end):
                    if lows[i] < level.price:
                        distance = level.price - lows[i]
                        if distance > sweep_distance:
                            sweep_distance = distance
                            sweep_price = lows[i]
                            sweep_idx = i

            # No sweep found
            if sweep_idx is None:
                return None

            sweep_pips = sweep_distance / pip_value

            # Check if sweep is within valid range
            if sweep_pips < min_sweep_pips:
                return None  # Too small to be meaningful

            # Determine sweep status
            if sweep_pips > max_sweep_pips:
                # Too far - likely a breakout, not a sweep
                return LiquiditySweep(
                    liquidity_level=level,
                    sweep_price=sweep_price,
                    sweep_index=sweep_idx,
                    sweep_pips=sweep_pips,
                    status=SweepStatus.BREAKOUT,
                    rejection_confirmed=False,
                    rejection_candles=current_idx - sweep_idx
                )

            # Check for rejection (price returning past the level)
            rejection_confirmed = False
            if require_rejection:
                rejection_confirmed = self._check_rejection(
                    df=df,
                    level=level,
                    sweep_idx=sweep_idx,
                    pip_value=pip_value
                )

            # Determine final status
            if rejection_confirmed:
                status = SweepStatus.CLEAN
            elif current_idx - sweep_idx <= 3:
                status = SweepStatus.PENDING  # Still waiting for rejection
            else:
                status = SweepStatus.PARTIAL  # Sweep but no clear rejection yet

            return LiquiditySweep(
                liquidity_level=level,
                sweep_price=sweep_price,
                sweep_index=sweep_idx,
                sweep_pips=sweep_pips,
                status=status,
                rejection_confirmed=rejection_confirmed,
                rejection_candles=current_idx - sweep_idx
            )

        except Exception as e:
            self.logger.error(f"Error checking level for sweep: {e}")
            return None

    def _check_rejection(
        self,
        df: pd.DataFrame,
        level: LiquidityLevel,
        sweep_idx: int,
        pip_value: float
    ) -> bool:
        """
        Check if price rejected after the sweep (returned past the level).

        A rejection confirms the sweep and indicates potential reversal.
        """
        try:
            current_idx = len(df) - 1
            closes = df['close'].values

            if level.liquidity_type == LiquidityType.BSL:
                # For BSL sweep, rejection = price closing back below the level
                for i in range(sweep_idx + 1, current_idx + 1):
                    if closes[i] < level.price:
                        return True
            else:  # SSL
                # For SSL sweep, rejection = price closing back above the level
                for i in range(sweep_idx + 1, current_idx + 1):
                    if closes[i] > level.price:
                        return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking rejection: {e}")
            return False

    def _is_swing_high(self, highs: np.ndarray, index: int, strength: int) -> bool:
        """Check if the given index is a swing high"""
        try:
            for i in range(1, strength + 1):
                if index - i < 0 or index + i >= len(highs):
                    return False
                if highs[index] <= highs[index - i] or highs[index] <= highs[index + i]:
                    return False
            return True
        except Exception:
            return False

    def _is_swing_low(self, lows: np.ndarray, index: int, strength: int) -> bool:
        """Check if the given index is a swing low"""
        try:
            for i in range(1, strength + 1):
                if index - i < 0 or index + i >= len(lows):
                    return False
                if lows[index] >= lows[index - i] or lows[index] >= lows[index + i]:
                    return False
            return True
        except Exception:
            return False

    def get_nearest_liquidity(
        self,
        current_price: float,
        liquidity_type: LiquidityType = None
    ) -> Optional[LiquidityLevel]:
        """
        Get the nearest liquidity level to current price.

        Args:
            current_price: Current price
            liquidity_type: Filter by type (BSL or SSL), or None for any

        Returns:
            Nearest liquidity level or None
        """
        try:
            if not self.liquidity_levels:
                return None

            # Filter by type if specified
            levels = self.liquidity_levels
            if liquidity_type:
                levels = [l for l in levels if l.liquidity_type == liquidity_type]

            if not levels:
                return None

            # Find nearest
            nearest = min(levels, key=lambda l: abs(l.price - current_price))
            return nearest

        except Exception as e:
            self.logger.error(f"Error getting nearest liquidity: {e}")
            return None

    def get_liquidity_above(self, current_price: float) -> List[LiquidityLevel]:
        """Get all liquidity levels above current price (BSL)"""
        return [l for l in self.liquidity_levels
                if l.price > current_price and l.liquidity_type == LiquidityType.BSL]

    def get_liquidity_below(self, current_price: float) -> List[LiquidityLevel]:
        """Get all liquidity levels below current price (SSL)"""
        return [l for l in self.liquidity_levels
                if l.price < current_price and l.liquidity_type == LiquidityType.SSL]

    def get_sweep_target(
        self,
        sweep: LiquiditySweep,
        pip_value: float = 0.0001
    ) -> Tuple[float, float]:
        """
        Calculate the target price based on the sweep (opposite liquidity).

        For bullish setups (SSL sweep), target the nearest BSL above.
        For bearish setups (BSL sweep), target the nearest SSL below.

        Args:
            sweep: The liquidity sweep
            pip_value: Pip value for the pair

        Returns:
            Tuple of (target_price, target_pips)
        """
        try:
            sweep_price = sweep.liquidity_level.price

            if sweep.liquidity_level.liquidity_type == LiquidityType.SSL:
                # SSL sweep = bullish setup, target BSL above
                targets = self.get_liquidity_above(sweep_price)
                if targets:
                    target = min(targets, key=lambda l: l.price)
                    target_pips = (target.price - sweep_price) / pip_value
                    return (target.price, target_pips)
            else:
                # BSL sweep = bearish setup, target SSL below
                targets = self.get_liquidity_below(sweep_price)
                if targets:
                    target = max(targets, key=lambda l: l.price)
                    target_pips = (sweep_price - target.price) / pip_value
                    return (target.price, target_pips)

            return (0.0, 0.0)

        except Exception as e:
            self.logger.error(f"Error calculating sweep target: {e}")
            return (0.0, 0.0)

    def calculate_sweep_quality(self, sweep: LiquiditySweep) -> float:
        """
        Calculate the quality score of a sweep (0.0 to 1.0).

        Quality factors:
        - Clean rejection after sweep
        - Sweep depth (not too shallow, not too deep)
        - Recency of sweep
        - Level strength
        """
        try:
            quality = 0.5  # Base quality

            # Rejection confirmation (+30%)
            if sweep.rejection_confirmed:
                quality += 0.30

            # Sweep depth (optimal is 5-10 pips)
            if 5 <= sweep.sweep_pips <= 10:
                quality += 0.15
            elif 3 <= sweep.sweep_pips <= 15:
                quality += 0.10

            # Recency (fresher is better)
            if sweep.rejection_candles <= 3:
                quality += 0.10
            elif sweep.rejection_candles <= 6:
                quality += 0.05

            # Level strength
            if sweep.liquidity_level.strength >= 4:
                quality += 0.10
            elif sweep.liquidity_level.strength >= 3:
                quality += 0.05

            return min(quality, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating sweep quality: {e}")
            return 0.5

    def get_summary(self) -> Dict:
        """Get a summary of detected liquidity and sweeps"""
        return {
            'total_levels': len(self.liquidity_levels),
            'bsl_levels': len([l for l in self.liquidity_levels if l.liquidity_type == LiquidityType.BSL]),
            'ssl_levels': len([l for l in self.liquidity_levels if l.liquidity_type == LiquidityType.SSL]),
            'recent_sweeps': len(self.recent_sweeps),
            'clean_sweeps': len([s for s in self.recent_sweeps if s.status == SweepStatus.CLEAN]),
            'partial_sweeps': len([s for s in self.recent_sweeps if s.status == SweepStatus.PARTIAL])
        }
