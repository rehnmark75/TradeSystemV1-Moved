# dev-app/trailing_class.py

"""
Advanced Trailing Stop Strategies - ENHANCED WITH CRITICAL FIXES

This module implements multiple sophisticated trailing stop techniques that adapt
to market conditions and volatility for better profit protection and trend following.

CRITICAL FIXES APPLIED:
- Fixed safe trail calculation direction bug
- Enhanced point-to-price conversion accuracy  
- Improved status management (pending → break_even → trailing)
- Added comprehensive validation and error handling
- Fixed BUY/SELL direction logic inconsistencies
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum

from services.models import TradeLog, IGCandle
from sqlalchemy.orm import Session
from utils import get_point_value, convert_price_to_points, calculate_move_points


class TrailingMethod(Enum):
    """Different trailing stop methods available"""
    FIXED_POINTS = "fixed_points"           # Original simple method
    PERCENTAGE = "percentage"               # Percentage-based trailing
    ATR_BASED = "atr_based"                # ATR volatility-based
    CHANDELIER = "chandelier"              # Chandelier exit
    PARABOLIC_SAR = "parabolic_sar"        # Parabolic SAR-style
    SMART_TRAIL = "smart_trail"            # Multi-condition intelligent trailing
    SUPPORT_RESISTANCE = "support_resistance"  # Technical level-based
    PROGRESSIVE_3_STAGE = "progressive_3_stage"  # 3-stage progressive trailing (Stage 3: percentage-based)


@dataclass
class TrailingConfig:
    """Configuration for advanced trailing strategies"""
    method: TrailingMethod = TrailingMethod.PROGRESSIVE_3_STAGE

    # Epic identifier for pair-specific settings
    epic: Optional[str] = None

    # Universal settings
    initial_trigger_points: int = 7
    break_even_trigger_points: int = 3  # UPDATED: move to BE after +3 points (was 7)
    min_trail_distance: int = 5
    max_trail_distance: int = 50

    monitor_interval_seconds: int = 30  # ✅ NEW: poll interval between trade checks

    # Standard 3-stage trailing settings (simplified)
    stage1_trigger_points: int = 12   # Fallback: Break-even trigger (IG min+4 used when available)
    stage1_lock_points: int = 2       # Fallback: Minimum profit guarantee (IG min used when available)
    stage2_trigger_points: int = 16   # Profit lock trigger
    stage2_lock_points: int = 10      # Profit guarantee
    stage3_trigger_points: int = 17   # Percentage-based trailing trigger
    stage3_atr_multiplier: float = 0.8 # ATR multiplier for stage 3 (tighter)
    stage3_min_distance: int = 2      # Minimum distance for stage 3

    # EMA Exit compatibility (disabled by default for progressive trailing)
    enable_ema_exit: bool = False

    # Percentage-based
    trail_percentage: float = 1.5

    # ATR-based
    atr_multiplier: float = 2.0
    atr_period: int = 14
    atr_timeframe: int = 60

    # Chandelier settings
    chandelier_period: int = 22
    chandelier_multiplier: float = 3.0

    # Parabolic SAR
    sar_initial_af: float = 0.02
    sar_max_af: float = 0.20
    sar_increment: float = 0.02

    # Smart trailing
    volatility_threshold: float = 1.5
    momentum_lookback: int = 5
    trend_strength_min: float = 0.6

    @classmethod
    def from_epic(cls, epic: str, method: TrailingMethod = TrailingMethod.PROGRESSIVE_3_STAGE):
        """
        Create TrailingConfig with pair-specific settings from config.py

        Args:
            epic: Trading symbol (e.g., 'CS.D.EURUSD.CEEM.IP')
            method: Trailing method to use (default: PROGRESSIVE_3_STAGE)

        Returns:
            TrailingConfig instance with pair-specific values

        Note: IG's min_stop_distance_points from trade_log ALWAYS takes priority when available
        """
        from config import get_trailing_config_for_epic

        pair_config = get_trailing_config_for_epic(epic)

        return cls(
            method=method,
            epic=epic,
            # Use pair-specific values
            stage1_trigger_points=pair_config['stage1_trigger_points'],
            stage1_lock_points=pair_config['stage1_lock_points'],
            stage2_trigger_points=pair_config['stage2_trigger_points'],
            stage2_lock_points=pair_config['stage2_lock_points'],
            stage3_trigger_points=pair_config['stage3_trigger_points'],
            stage3_atr_multiplier=pair_config['stage3_atr_multiplier'],
            stage3_min_distance=pair_config['stage3_min_distance'],
            min_trail_distance=pair_config['min_trail_distance'],
            break_even_trigger_points=pair_config['break_even_trigger_points'],
            # Keep other defaults for non-stage-specific settings
            initial_trigger_points=7,
            max_trail_distance=50,
            monitor_interval_seconds=30,
            enable_ema_exit=False,
            trail_percentage=1.5,
            atr_multiplier=2.0,
            atr_period=14,
            atr_timeframe=60,
            chandelier_period=14,
            chandelier_multiplier=3.0,
            sar_initial_af=0.02,
            sar_max_af=0.20,
            sar_increment=0.02,
            volatility_threshold=1.5,
            momentum_lookback=5,
            trend_strength_min=0.6,
        )

class TrailingStrategy(ABC):
    """Base class for trailing stop strategies"""
    
    def __init__(self, config: TrailingConfig, logger):
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def calculate_trail_level(self, trade: TradeLog, current_price: float, 
                            candle_data: List[IGCandle], db: Session) -> Optional[float]:
        """Calculate the new trailing stop level"""
        pass
    
    @abstractmethod
    def should_trail(self, trade: TradeLog, current_price: float,
                    candle_data: List[IGCandle]) -> bool:
        """Determine if we should update the trailing stop"""
        pass


class FixedPointsTrailing(TrailingStrategy):
    """Fixed points trailing that intelligently respects current stop position"""
    
    def calculate_trail_level(self, trade: TradeLog, current_price: float,
                            candle_data: List[IGCandle], db: Session) -> Optional[float]:
        direction = trade.direction.upper()
        point_value = self._get_point_value(trade.symbol)
        
        # Get safe distance
        try:
            safe_distance = self._get_safe_distance_from_processor(trade)
        except:
            safe_distance = self.config.min_trail_distance
        
        trail_distance_price = safe_distance * point_value
        
        # Get current stop level
        current_stop = trade.sl_price or 0.0
        
        # ✅ INTELLIGENT TRAILING: Calculate both options and choose the better one
        
        # Option 1: Trail from current price (traditional)
        if direction == "BUY":
            trail_from_current = current_price - trail_distance_price
        else:
            trail_from_current = current_price + trail_distance_price
        
        # Option 2: Move current stop by minimum increment (progressive trailing)
        if current_stop > 0:
            min_increment = 1 * point_value  # Move by 1 point minimum
            if direction == "BUY":
                # For BUY: only move stop UP (higher)
                trail_from_stop = current_stop + min_increment
                # But don't go beyond safe distance from current
                max_allowed = current_price - trail_distance_price
                trail_from_stop = min(trail_from_stop, max_allowed)
            else:
                # For SELL: only move stop DOWN (lower)  
                trail_from_stop = current_stop - min_increment
                # But don't go beyond safe distance from current
                min_allowed = current_price + trail_distance_price
                trail_from_stop = max(trail_from_stop, min_allowed)
        else:
            trail_from_stop = trail_from_current
        
        # ✅ CHOOSE THE BETTER OPTION:
        # For BUY: higher stop is better
        # For SELL: lower stop is better
        if direction == "BUY":
            # Only move stop UP (higher) for BUY positions
            if current_stop > 0:
                trail_level = max(trail_from_current, current_stop)  # Don't force +1pt increment
            else:
                trail_level = trail_from_current
        else:
            # Only move stop DOWN (lower) for SELL positions
            if current_stop > 0:
                trail_level = min(trail_from_current, current_stop)  # Don't force -1pt increment
            else:
                trail_level = trail_from_current
        
        # Ensure we don't violate minimum distance
        if direction == "BUY":
            min_allowed = current_price - trail_distance_price
            trail_level = min(trail_level, min_allowed)
        else:
            max_allowed = current_price + trail_distance_price
            trail_level = max(trail_level, max_allowed)
        
        # Round to appropriate precision
        trail_level = round(trail_level, 5)
        
        # Calculate distances for logging
        distance_from_current = abs(current_price - trail_level) / point_value
        
        self.logger.info(f"[INTELLIGENT TRAIL] {trade.symbol} {direction}: "
                        f"current={current_price:.5f}, current_stop={current_stop:.5f}, "
                        f"trail_from_current={trail_from_current:.5f}, trail_level={trail_level:.5f}, "
                        f"distance_from_current={distance_from_current:.1f}pts")
        
        # Final validation: only return if it's actually an improvement
        if current_stop > 0:
            if direction == "BUY":
                is_improvement = trail_level > current_stop
            else:
                is_improvement = trail_level < current_stop
            
            if not is_improvement:
                self.logger.warning(f"[TRAIL REJECT] {trade.symbol}: "
                                  f"Calculated trail {trail_level:.5f} not better than current {current_stop:.5f}")
                return None
        
        return trail_level
    
    def should_trail(self, trade: TradeLog, current_price: float,
                    candle_data: List[IGCandle]) -> bool:
        """More intelligent should_trail logic"""
        direction = trade.direction.upper()
        current_stop = trade.sl_price or 0.0
        
        if current_stop <= 0:
            return True  # No current stop, definitely trail
        
        # Check if price has moved favorably enough to justify trailing
        point_value = self._get_point_value(trade.symbol)
        min_move_required = self._get_safe_distance_from_processor(trade)
        
        if direction == "BUY":
            # For BUY: check if current price is significantly above current stop
            distance_from_stop = (current_price - current_stop) / point_value
            # Trail if we're at least 2x the minimum distance above current stop
            should_trail = distance_from_stop >= (min_move_required * 2)
        else:
            # For SELL: check if current price is significantly below current stop
            distance_from_stop = (current_stop - current_price) / point_value
            should_trail = distance_from_stop >= (min_move_required * 2)
        
        self.logger.debug(f"[SHOULD TRAIL INTELLIGENT] {trade.symbol}: "
                         f"distance_from_stop={distance_from_stop:.1f}pts, "
                         f"required={min_move_required * 2}pts, should_trail={should_trail}")
        
        return should_trail
    
    def _get_safe_distance_from_processor(self, trade: TradeLog) -> int:
        """Get the same safe distance that the EnhancedTradeProcessor uses"""
        ig_min_distance = getattr(trade, 'min_stop_distance_points', None)
        config_min_distance = self.config.min_trail_distance
        
        if ig_min_distance:
            safe_distance = max(ig_min_distance, config_min_distance)
        else:
            safe_distance = config_min_distance
        
        return safe_distance
    
    def _get_point_value(self, epic: str) -> float:
        """Get point value for the instrument"""
        if "JPY" in epic:
            return 0.01
        elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return 0.0001
        return 1.0


class PercentageTrailing(TrailingStrategy):
    """Percentage-based trailing stop - ENHANCED"""
    
    def calculate_trail_level(self, trade: TradeLog, current_price: float,
                            candle_data: List[IGCandle], db: Session) -> Optional[float]:
        direction = trade.direction.upper()
        trail_percentage = self.config.trail_percentage / 100
        
        # ✅ ENHANCED: More sophisticated percentage calculation
        if direction == "BUY":
            trail_level = current_price * (1 - trail_percentage)
        else:
            trail_level = current_price * (1 + trail_percentage)
        
        # Validate minimum distance from current price
        point_value = self._get_point_value(trade.symbol)
        min_distance_price = self.config.min_trail_distance * point_value
        
        if direction == "BUY":
            min_trail_level = current_price - min_distance_price
            trail_level = min(trail_level, min_trail_level)
        else:
            min_trail_level = current_price + min_distance_price
            trail_level = max(trail_level, min_trail_level)
        
        return round(trail_level, 5)
    
    def should_trail(self, trade: TradeLog, current_price: float,
                    candle_data: List[IGCandle]) -> bool:
        return True  # Always trail with percentage method
    
    def _get_point_value(self, epic: str) -> float:
        if "JPY" in epic:
            return 0.01
        elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return 0.0001
        return 1.0


class ATRTrailing(TrailingStrategy):
    """ATR-based adaptive trailing stop - ENHANCED WITH VALIDATION"""
    
    def calculate_trail_level(self, trade: TradeLog, current_price: float,
                            candle_data: List[IGCandle], db: Session) -> Optional[float]:
        atr = self._calculate_atr(candle_data)
        if not atr:
            # Fallback to percentage method
            fallback = PercentageTrailing(self.config, self.logger)
            return fallback.calculate_trail_level(trade, current_price, candle_data, db)
        
        direction = trade.direction.upper()
        trail_distance = atr * self.config.atr_multiplier
        
        # Apply min/max constraints
        point_value = self._get_point_value(trade.symbol)
        min_distance = self.config.min_trail_distance * point_value
        max_distance = self.config.max_trail_distance * point_value
        trail_distance = max(min_distance, min(max_distance, trail_distance))
        
        # ✅ CRITICAL FIX: Correct direction logic
        if direction == "BUY":
            trail_level = current_price - trail_distance
        else:
            trail_level = current_price + trail_distance
        
        self.logger.debug(f"[ATR TRAIL] {trade.symbol} {direction}: "
                         f"ATR={atr:.5f}, multiplier={self.config.atr_multiplier}, "
                         f"trail_distance={trail_distance:.5f}, trail_level={trail_level:.5f}")
        
        return round(trail_level, 5)
    
    def should_trail(self, trade: TradeLog, current_price: float,
                    candle_data: List[IGCandle]) -> bool:
        if not trade.last_trigger_price:
            return False
        
        atr = self._calculate_atr(candle_data)
        if not atr:
            return True  # Trail if no ATR data
        
        # Trail when move is at least 50% of ATR
        direction = trade.direction.upper()
        move = abs(current_price - trade.last_trigger_price)
        return move >= (atr * 0.5)
    
    def _calculate_atr(self, candles: List[IGCandle]) -> Optional[float]:
        """Calculate ATR from candle data - ENHANCED WITH VALIDATION"""
        if len(candles) < self.config.atr_period + 1:
            return None
        
        # Sort by time and take recent candles
        sorted_candles = sorted(candles, key=lambda x: x.start_time)[-self.config.atr_period-1:]
        
        true_ranges = []
        for i in range(1, len(sorted_candles)):
            prev_close = sorted_candles[i-1].close
            current = sorted_candles[i]
            
            tr = max(
                current.high - current.low,
                abs(current.high - prev_close),
                abs(current.low - prev_close)
            )
            true_ranges.append(tr)
        
        if not true_ranges:
            return None
        
        return sum(true_ranges) / len(true_ranges)
    
    def _get_point_value(self, epic: str) -> float:
        if "JPY" in epic:
            return 0.01
        elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return 0.0001
        return 1.0


class ChandelierTrailing(TrailingStrategy):
    """Chandelier Exit trailing stop method - ENHANCED"""
    
    def calculate_trail_level(self, trade: TradeLog, current_price: float,
                            candle_data: List[IGCandle], db: Session) -> Optional[float]:
        direction = trade.direction.upper()
        
        # Get recent candles for calculation
        recent_candles = sorted(candle_data, key=lambda x: x.start_time)[-self.config.chandelier_period:]
        
        if len(recent_candles) < self.config.chandelier_period:
            # Fallback to ATR method
            fallback = ATRTrailing(self.config, self.logger)
            return fallback.calculate_trail_level(trade, current_price, candle_data, db)
        
        # Calculate ATR
        atr = self._calculate_atr(recent_candles)
        if not atr:
            return None
        
        # ✅ ENHANCED: Better Chandelier calculation
        if direction == "BUY":
            # For long positions: Highest high - (ATR * multiplier)
            highest_high = max(candle.high for candle in recent_candles)
            trail_level = highest_high - (atr * self.config.chandelier_multiplier)
        else:
            # For short positions: Lowest low + (ATR * multiplier)
            lowest_low = min(candle.low for candle in recent_candles)
            trail_level = lowest_low + (atr * self.config.chandelier_multiplier)
        
        # Validate against minimum distance
        point_value = self._get_point_value(trade.symbol)
        min_distance = self.config.min_trail_distance * point_value
        
        if direction == "BUY":
            min_trail_level = current_price - min_distance
            trail_level = min(trail_level, min_trail_level)
        else:
            min_trail_level = current_price + min_distance
            trail_level = max(trail_level, min_trail_level)
        
        self.logger.debug(f"[CHANDELIER] {trade.symbol} {direction}: "
                         f"ATR={atr:.5f}, multiplier={self.config.chandelier_multiplier}, "
                         f"trail_level={trail_level:.5f}")
        
        return round(trail_level, 5)
    
    def should_trail(self, trade: TradeLog, current_price: float,
                    candle_data: List[IGCandle]) -> bool:
        return True  # Always calculate new Chandelier level
    
    def _calculate_atr(self, candles: List[IGCandle]) -> Optional[float]:
        if len(candles) < 2:
            return None
        
        true_ranges = []
        for i in range(1, len(candles)):
            prev_close = candles[i-1].close
            current = candles[i]
            
            tr = max(
                current.high - current.low,
                abs(current.high - prev_close),
                abs(current.low - prev_close)
            )
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges) if true_ranges else None
    
    def _get_point_value(self, epic: str) -> float:
        if "JPY" in epic:
            return 0.01
        elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return 0.0001
        return 1.0


class SmartTrailing(TrailingStrategy):
    """Intelligent multi-condition trailing strategy - ENHANCED WITH FIXES"""
    
    def __init__(self, config: TrailingConfig, logger):
        super().__init__(config, logger)
        self.atr_strategy = ATRTrailing(config, logger)
        self.chandelier_strategy = ChandelierTrailing(config, logger)
        self.fixed_strategy = FixedPointsTrailing(config, logger)
    
    def calculate_trail_level(self, trade: TradeLog, current_price: float,
                            candle_data: List[IGCandle], db: Session) -> Optional[float]:
        
        # Analyze market conditions
        market_condition = self._analyze_market_condition(candle_data, current_price, trade)
        
        trail_level = None
        
        if market_condition == "high_volatility":
            # Use wider ATR-based trailing in volatile markets
            self.logger.debug(f"[SMART] Using ATR trailing for {trade.symbol} (high volatility)")
            trail_level = self.atr_strategy.calculate_trail_level(trade, current_price, candle_data, db)
        
        elif market_condition == "strong_trend":
            # Use Chandelier for strong trending markets
            self.logger.debug(f"[SMART] Using Chandelier trailing for {trade.symbol} (strong trend)")
            trail_level = self.chandelier_strategy.calculate_trail_level(trade, current_price, candle_data, db)
        
        else:
            # Use tighter fixed trailing for choppy/weak trend markets
            self.logger.debug(f"[SMART] Using fixed trailing for {trade.symbol} (choppy market)")
            trail_level = self.fixed_strategy.calculate_trail_level(trade, current_price, candle_data, db)
        
        # ✅ ENHANCED: Final validation and safety check
        if trail_level is not None:
            # Ensure trail level respects minimum distance
            point_value = self._get_point_value(trade.symbol)
            min_distance = self.config.min_trail_distance * point_value
            direction = trade.direction.upper()
            
            if direction == "BUY":
                safe_max = current_price - min_distance
                trail_level = min(trail_level, safe_max)
            else:
                safe_min = current_price + min_distance
                trail_level = max(trail_level, safe_min)
        
        return trail_level
    
    def should_trail(self, trade: TradeLog, current_price: float,
                    candle_data: List[IGCandle]) -> bool:
        # More conservative trailing in choppy markets
        market_condition = self._analyze_market_condition(candle_data, current_price, trade)
        
        if market_condition == "choppy":
            # Require larger moves to trail in choppy conditions
            if not trade.last_trigger_price:
                return False
            
            move_points = self._calculate_move_points(trade.last_trigger_price, 
                                                    current_price, trade.direction, trade.symbol)
            return move_points >= (self.config.min_trail_distance * 2)
        
        # Use underlying strategy logic for other conditions
        return True
    
    def _analyze_market_condition(self, candles: List[IGCandle], current_price: float, 
                                trade: TradeLog) -> str:
        """Analyze current market conditions - ENHANCED"""
        if len(candles) < 10:
            return "unknown"
        
        recent_candles = sorted(candles, key=lambda x: x.start_time)[-10:]
        
        try:
            # Calculate volatility (ATR relative to price)
            atr = self._calculate_atr(recent_candles)
            if atr and current_price > 0:
                volatility_ratio = atr / current_price
                
                if volatility_ratio > (self.config.volatility_threshold / 100):
                    return "high_volatility"
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(recent_candles)
            
            if trend_strength > self.config.trend_strength_min:
                return "strong_trend"
            elif trend_strength < 0.3:
                return "choppy"
            else:
                return "weak_trend"
                
        except Exception as e:
            self.logger.error(f"[SMART ANALYSIS ERROR] {trade.symbol}: {e}")
            return "unknown"
    
    def _calculate_trend_strength(self, candles: List[IGCandle]) -> float:
        """Calculate trend strength (0 = no trend, 1 = perfect trend) - ENHANCED"""
        if len(candles) < 5:
            return 0.0
        
        closes = [c.close for c in candles]
        
        # Count consecutive moves in same direction
        up_moves = 0
        down_moves = 0
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                up_moves += 1
            elif closes[i] < closes[i-1]:
                down_moves += 1
        
        total_moves = len(closes) - 1
        if total_moves == 0:
            return 0.0
        
        # Trend strength is the dominance of one direction
        dominant_moves = max(up_moves, down_moves)
        return dominant_moves / total_moves
    
    def _calculate_atr(self, candles: List[IGCandle]) -> Optional[float]:
        """Calculate ATR - ENHANCED WITH ERROR HANDLING"""
        if len(candles) < 2:
            return None
        
        try:
            true_ranges = []
            for i in range(1, len(candles)):
                prev_close = candles[i-1].close
                current = candles[i]
                
                tr = max(
                    current.high - current.low,
                    abs(current.high - prev_close),
                    abs(current.low - prev_close)
                )
                true_ranges.append(tr)
            
            return sum(true_ranges) / len(true_ranges) if true_ranges else None
            
        except Exception:
            return None
    
    def _calculate_move_points(self, from_price: float, to_price: float, 
                             direction: str, symbol: str) -> int:
        """Calculate movement in points - ENHANCED"""
        point_value = self._get_point_value(symbol)
        
        if direction.upper() == "BUY":
            move = to_price - from_price
        else:
            move = from_price - to_price
        
        return max(0, int(move / point_value))
    
    def _get_point_value(self, epic: str) -> float:
        if "JPY" in epic:
            return 0.01
        elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return 0.0001
        return 1.0


class Progressive3StageTrailing(TrailingStrategy):
    """
    4-Stage Progressive Trailing Strategy - Optimized for Small Account Protection

    Based on MAE Analysis (Dec 2025):
    - Winners only dip 3 pips on average (median 2.7 pips, 75th percentile 3.5 pips)
    - Losers dip 15 pips on average before hitting SL
    - Conclusion: Good trades barely dip, so early breakeven is safe

    Stage 0: EARLY BREAKEVEN at +6 pts → SL to entry + 1 pt (NEW - capital protection)
    Stage 1: Profit Lock at +10 pts → SL to entry + 5 pts (guaranteed small profit)
    Stage 2: Profit Lock-in at +15 pts → SL to entry + 10 pts (meaningful profit secured)
    Stage 3: Percentage-based trailing at +20+ pts (reliable trend following)

    Stage 3 uses tiered percentage retracement:
    - 50+ points profit: 15% retracement allowed
    - 25-49 points profit: 20% retracement allowed
    - 20-24 points profit: 25% retracement allowed

    This strategy prevents scenarios like GBPUSD (+15 → 0) where profit is given back.
    """

    def __init__(self, config: TrailingConfig, logger):
        super().__init__(config, logger)
        # No longer using ATR strategy - Stage 3 now uses percentage-based trailing

    def calculate_trail_level(self, trade: TradeLog, current_price: float,
                            candle_data: List[IGCandle], db: Session) -> Optional[float]:
        direction = trade.direction.upper()
        point_value = self._get_point_value(trade.symbol)

        # Use 4-stage configuration with early breakeven (Stage 0)
        from config import (EARLY_BREAKEVEN_TRIGGER_POINTS, STAGE1_TRIGGER_POINTS,
                           STAGE2_TRIGGER_POINTS, STAGE3_TRIGGER_POINTS)

        # Get pair-specific early breakeven config if available
        pair_config = self._get_pair_config(trade.symbol)
        early_be_trigger = pair_config.get('early_breakeven_trigger_points', EARLY_BREAKEVEN_TRIGGER_POINTS)
        stage1_trigger = pair_config.get('stage1_trigger_points', STAGE1_TRIGGER_POINTS)
        stage2_trigger = pair_config.get('stage2_trigger_points', STAGE2_TRIGGER_POINTS)
        stage3_trigger = pair_config.get('stage3_trigger_points', STAGE3_TRIGGER_POINTS)

        # Calculate current profit in points
        if direction == "BUY":
            profit_points = int((current_price - trade.entry_price) / point_value)
        else:  # SELL
            profit_points = int((trade.entry_price - current_price) / point_value)

        current_stop = trade.sl_price or 0.0

        # 🆕 Stage 2.5: MFE Protection Check
        # Estimate MFE from current stop position (higher stop = higher MFE was reached)
        mfe_protection_level = self._check_mfe_protection(trade, current_price, current_stop, profit_points, point_value)
        if mfe_protection_level is not None:
            self.logger.info(f"[MFE PROTECTION] {trade.symbol}: "
                           f"Profit declined, protecting at {mfe_protection_level:.5f}")
            return mfe_protection_level

        # Determine which stage we're in and calculate appropriate trail level
        # NEW 4-STAGE SYSTEM: Stage 0 (Early BE) → Stage 1 (Profit Lock) → Stage 2 → Stage 3
        if profit_points >= stage3_trigger:
            # Stage 3: ATR-based trailing for trend following
            trail_level = self._calculate_stage3_trail(trade, current_price, candle_data, current_stop)
            stage = 3

        elif profit_points >= stage2_trigger:
            # Stage 2: Profit lock-in
            trail_level = self._calculate_stage2_trail(trade, current_price, current_stop)
            stage = 2

        elif profit_points >= stage1_trigger:
            # Stage 1: Profit lock (5 pts guaranteed)
            trail_level = self._calculate_stage1_trail(trade, current_price, current_stop)
            stage = 1

        elif profit_points >= early_be_trigger:
            # Stage 0: Early breakeven protection (NEW - small account safety)
            trail_level = self._calculate_early_breakeven(trade, current_price, current_stop)
            stage = 0

        else:
            # Not ready for any trailing yet
            return None

        if trail_level is None:
            return None

        # Validate that this is actually an improvement
        if current_stop > 0:
            if direction == "BUY":
                is_improvement = trail_level > current_stop
            else:
                is_improvement = trail_level < current_stop

            if not is_improvement:
                self.logger.debug(f"[PROGRESSIVE STAGE {stage}] {trade.symbol}: "
                                f"Trail level {trail_level:.5f} not better than current {current_stop:.5f}")
                return None

        self.logger.info(f"[PROGRESSIVE STAGE {stage}] {trade.symbol}: "
                       f"Profit: {profit_points}pts → Trail: {trail_level:.5f}")

        return trail_level

    def _get_pair_config(self, symbol: str) -> dict:
        """Get pair-specific trailing configuration from config.py"""
        from config import get_trailing_config_for_epic
        return get_trailing_config_for_epic(symbol)

    def _calculate_early_breakeven(self, trade: TradeLog, current_price: float, current_stop: float) -> Optional[float]:
        """
        Stage 0: Early Breakeven Protection (NEW - Small Account Safety)

        Based on MAE analysis (Dec 2025):
        - Winners only dip 3 pips on average (median 2.7 pips)
        - Move to breakeven at +6 pips to protect capital early
        - SL moves to entry + 1 pip (covers spread)

        This is a ONE-TIME move when trade first reaches +6 pips profit.
        Prevents scenarios like GBPUSD (+15 → 0) where profit is given back.
        """
        # Check if already moved to early breakeven or beyond
        if trade.moved_to_breakeven:
            self.logger.debug(f"[STAGE 0 SKIP] Trade {trade.id}: Already at breakeven or beyond")
            return None

        direction = trade.direction.upper()
        point_value = self._get_point_value(trade.symbol)

        # Get pair-specific buffer or use default (1 point)
        pair_config = self._get_pair_config(trade.symbol)
        buffer_points = pair_config.get('early_breakeven_buffer_points', 1)

        # Calculate breakeven level with small buffer (covers spread)
        if direction == "BUY":
            trail_level = trade.entry_price + (buffer_points * point_value)
        else:  # SELL
            trail_level = trade.entry_price - (buffer_points * point_value)

        # Ensure we're actually improving the stop (not moving it worse)
        if current_stop > 0:
            if direction == "BUY" and trail_level <= current_stop:
                self.logger.debug(f"[STAGE 0 SKIP] Trade {trade.id}: Early BE level {trail_level:.5f} not better than current {current_stop:.5f}")
                return None
            elif direction == "SELL" and trail_level >= current_stop:
                self.logger.debug(f"[STAGE 0 SKIP] Trade {trade.id}: Early BE level {trail_level:.5f} not better than current {current_stop:.5f}")
                return None

        self.logger.info(f"🛡️ [STAGE 0 EARLY BE] Trade {trade.id} {trade.symbol}: "
                        f"Moving stop to EARLY breakeven at {trail_level:.5f} (entry + {buffer_points}pts buffer)")

        return round(trail_level, 5)

    def _calculate_stage1_trail(self, trade: TradeLog, current_price: float, current_stop: float) -> Optional[float]:
        """
        Stage 1: Profit Lock - Lock in 5 pts profit when trade reaches +10 pts

        This is a ONE-TIME move that triggers after Stage 0 (early breakeven).
        At +10 pts profit, move SL to entry + 5 pts to guarantee small profit.
        """
        # Check if already in Stage 1 or beyond (status-based check)
        if hasattr(trade, 'status') and trade.status in ['stage1_profit_lock', 'stage2_profit_lock', 'stage3_trailing']:
            self.logger.debug(f"[STAGE 1 SKIP] Trade {trade.id}: Already in Stage 1+ ({trade.status})")
            return None

        direction = trade.direction.upper()
        point_value = self._get_point_value(trade.symbol)

        # Get pair-specific lock points or use default (5 points)
        pair_config = self._get_pair_config(trade.symbol)
        lock_points = pair_config.get('stage1_lock_points', 5)

        # Calculate profit lock level
        if direction == "BUY":
            trail_level = trade.entry_price + (lock_points * point_value)
        else:  # SELL
            trail_level = trade.entry_price - (lock_points * point_value)

        # Ensure we're improving from Stage 0 (early breakeven)
        if current_stop > 0:
            if direction == "BUY" and trail_level <= current_stop:
                self.logger.debug(f"[STAGE 1 SKIP] Trade {trade.id}: Stage 1 level {trail_level:.5f} not better than current {current_stop:.5f}")
                return None
            elif direction == "SELL" and trail_level >= current_stop:
                self.logger.debug(f"[STAGE 1 SKIP] Trade {trade.id}: Stage 1 level {trail_level:.5f} not better than current {current_stop:.5f}")
                return None

        self.logger.info(f"💰 [STAGE 1 PROFIT LOCK] Trade {trade.id} {trade.symbol}: "
                        f"Moving stop to lock {lock_points}pts profit at {trail_level:.5f}")
        return round(trail_level, 5)

    def _calculate_stage2_trail(self, trade: TradeLog, current_price: float, current_stop: float) -> Optional[float]:
        """Stage 2: Profit lock-in - ONE-TIME move to lock in 10 points profit when price reaches 16+ points"""

        # ✅ CRITICAL FIX: Stage 2 should only move ONCE to lock in profit, not continuously trail
        # Check both status AND the moved_to_stage2 flag
        if getattr(trade, 'moved_to_stage2', False):
            self.logger.debug(f"[STAGE 2 SKIP] Trade {trade.id}: Stage 2 already executed (moved_to_stage2=True)")
            return None

        if hasattr(trade, 'status') and trade.status in ['stage2_profit_lock', 'stage3_trailing']:
            self.logger.debug(f"[STAGE 2 SKIP] Trade {trade.id}: Already in Stage 2/3 ({trade.status}), no further Stage 2 action needed")
            return None

        direction = trade.direction.upper()
        point_value = self._get_point_value(trade.symbol)

        # ✅ NEW: DEFENSIVE CHECK - Ensure profit actually >= 16 points before Stage 2
        # This protects against Stage 2 being called incorrectly or for legacy trades
        from config import STAGE2_TRIGGER_POINTS
        if direction == "BUY":
            current_profit_points = int((current_price - trade.entry_price) / point_value)
        else:  # SELL
            current_profit_points = int((trade.entry_price - current_price) / point_value)

        if current_profit_points < STAGE2_TRIGGER_POINTS:
            self.logger.warning(f"⚠️ [STAGE 2 GUARD] Trade {trade.id}: Profit {current_profit_points}pts < Stage 2 trigger {STAGE2_TRIGGER_POINTS}pts - REJECTING Stage 2")
            return None

        # ✅ FIXED: Stage 2 locks in 10 points profit (not config value)
        # When price reaches 16+ points, move stop to entry + 10 points
        profit_lock_points = 10

        if direction == "BUY":
            trail_level = trade.entry_price + (profit_lock_points * point_value)
        else:  # SELL
            trail_level = trade.entry_price - (profit_lock_points * point_value)

        # Ensure we don't move backwards from Stage 1
        if current_stop > 0:
            if direction == "BUY":
                trail_level = max(trail_level, current_stop)
            else:
                trail_level = min(trail_level, current_stop)

        self.logger.info(f"💰 [STAGE 2 PROFIT LOCK] Trade {trade.id}: Moving stop to profit lock at {trail_level:.5f} (locking in {profit_lock_points}pts profit)")
        return round(trail_level, 5)

    def _calculate_stage3_trail(self, trade: TradeLog, current_price: float,
                              candle_data: List[IGCandle], current_stop: float) -> Optional[float]:
        """Stage 3: Percentage-based dynamic trailing (replaces ATR-based)"""
        direction = trade.direction.upper()
        point_value = self._get_point_value(trade.symbol)

        # Calculate current profit in points
        if direction == "BUY":
            current_profit_points = (current_price - trade.entry_price) / point_value
        else:  # SELL
            current_profit_points = (trade.entry_price - current_price) / point_value

        # Calculate maximum achieved profit (from current stop level)
        if current_stop > 0:
            if direction == "BUY":
                max_profit_from_stop = (current_stop - trade.entry_price) / point_value
            else:  # SELL
                max_profit_from_stop = (trade.entry_price - current_stop) / point_value

            # Use the better of current profit or profit implied by current stop
            effective_profit = max(current_profit_points, max_profit_from_stop + 2)  # +2pts buffer
        else:
            effective_profit = current_profit_points

        # Tiered percentage trailing based on effective profit level
        if effective_profit >= 50:
            retracement_percentage = 0.15  # 15% retracement for big profits (50+ points)
        elif effective_profit >= 25:
            retracement_percentage = 0.20  # 20% retracement for medium profits (25-49 points)
        else:
            retracement_percentage = 0.25  # 25% retracement for smaller profits (18-24 points)

        # Calculate trail distance in points, with IG minimum protection
        # ✅ CRITICAL FIX: Always respect IG's min_stop_distance_points to avoid rejection
        ig_min_distance = getattr(trade, 'min_stop_distance_points', None)
        if ig_min_distance:
            min_distance = max(ig_min_distance, self.config.stage3_min_distance)
            self.logger.debug(f"[IG MIN CHECK] {trade.symbol}: Using IG min {ig_min_distance}pts vs config {self.config.stage3_min_distance}pts = {min_distance}pts")
        else:
            min_distance = self.config.stage3_min_distance
            self.logger.debug(f"[FALLBACK MIN] {trade.symbol}: No IG minimum, using config {min_distance}pts")

        trail_distance_points = max(
            min_distance,  # IG minimum distance (prevents rejection)
            effective_profit * retracement_percentage  # Percentage-based distance
        )

        trail_distance_price = trail_distance_points * point_value

        if direction == "BUY":
            trail_level = current_price - trail_distance_price
        else:  # SELL
            trail_level = current_price + trail_distance_price

        self.logger.debug(f"[TRAIL CALC] {trade.symbol}: current_profit={current_profit_points:.1f}pts, "
                         f"effective_profit={effective_profit:.1f}pts, retracement={retracement_percentage:.0%}, "
                         f"trail_distance={trail_distance_points:.1f}pts")

        # Ensure we don't move backwards from previous stages
        if current_stop > 0:
            if direction == "BUY":
                # For BUY: only move stop UP (higher) - trail_level should be higher than current_stop
                if trail_level > current_stop:
                    self.logger.debug(f"[TRAIL MOVE] BUY: Moving stop UP from {current_stop:.5f} to {trail_level:.5f}")
                else:
                    self.logger.debug(f"[TRAIL HOLD] BUY: Keeping stop at {current_stop:.5f} (calculated {trail_level:.5f} not better)")
                    trail_level = current_stop  # Don't move backwards
            else:  # SELL
                # For SELL: only move stop DOWN (lower) - trail_level should be lower than current_stop
                if trail_level < current_stop:
                    self.logger.debug(f"[TRAIL MOVE] SELL: Moving stop DOWN from {current_stop:.5f} to {trail_level:.5f}")
                else:
                    self.logger.debug(f"[TRAIL HOLD] SELL: Keeping stop at {current_stop:.5f} (calculated {trail_level:.5f} not better)")
                    trail_level = current_stop  # Don't move backwards

        self.logger.info(f"[PERCENTAGE TRAIL] {trade.symbol}: Profit {current_profit_points:.1f}pts → "
                        f"{retracement_percentage*100:.0f}% retracement = {trail_distance_points:.1f}pts trail distance")

        return round(trail_level, 5)

    def _check_mfe_protection(self, trade: TradeLog, current_price: float,
                              current_stop: float, current_profit_points: float,
                              point_value: float) -> Optional[float]:
        """
        Stage 2.5: MFE Protection Rule

        When profit has reached 70% of target AND then declines 10% from peak (MFE),
        lock 60% of MFE to prevent giving back significant profits.

        MFE is estimated from the current stop position - if stop is in profit,
        it indicates how far in profit the trade has gone (via previous stage progressions).

        Args:
            trade: The trade being monitored
            current_price: Current market price
            current_stop: Current stop loss price
            current_profit_points: Current profit in points
            point_value: Point value for the pair (e.g., 0.0001 for EURUSD)

        Returns:
            Trail level if MFE protection should trigger, None otherwise
        """
        direction = trade.direction.upper()

        # Get target profit from trade's TP or config default
        if trade.tp_price and trade.entry_price:
            if direction == "BUY":
                target_points = abs(trade.tp_price - trade.entry_price) / point_value
            else:
                target_points = abs(trade.entry_price - trade.tp_price) / point_value
        else:
            # Default target if TP not set
            target_points = 30.0  # Default 30 points

        # MFE Protection configuration (from config_trailing_stops.py defaults)
        mfe_threshold_pct = 0.70   # Trigger when profit >= 70% of target
        mfe_decline_pct = 0.10     # Trigger on 10% decline from MFE
        mfe_lock_pct = 0.60        # Lock 60% of MFE

        # Estimate MFE from current stop position
        # If stop is in profit (moved via stages), the profit locked indicates approximate MFE
        if current_stop > 0:
            if direction == "BUY":
                mfe_from_stop = (current_stop - trade.entry_price) / point_value
            else:
                mfe_from_stop = (trade.entry_price - current_stop) / point_value

            # MFE is the better of current profit or profit indicated by stop
            estimated_mfe = max(current_profit_points, mfe_from_stop + 5)  # +5 buffer
        else:
            # No stop in profit, use current profit as MFE
            estimated_mfe = current_profit_points

        # Check if MFE threshold is reached (70% of target)
        mfe_threshold = target_points * mfe_threshold_pct
        if estimated_mfe < mfe_threshold:
            # MFE hasn't reached threshold yet
            return None

        # Check for profit decline (10% drop from MFE)
        if estimated_mfe > 0:
            decline_pct = 1.0 - (current_profit_points / estimated_mfe)
            if decline_pct >= mfe_decline_pct:
                # MFE Protection triggered!
                protected_profit_points = estimated_mfe * mfe_lock_pct

                # Calculate trail level that locks this profit
                if direction == "BUY":
                    mfe_trail_level = trade.entry_price + (protected_profit_points * point_value)
                else:
                    mfe_trail_level = trade.entry_price - (protected_profit_points * point_value)

                # Only return if it's an improvement over current stop
                if current_stop > 0:
                    if direction == "BUY" and mfe_trail_level <= current_stop:
                        return None
                    if direction == "SELL" and mfe_trail_level >= current_stop:
                        return None

                self.logger.info(
                    f"🛡️ [MFE PROTECTION] {trade.symbol}: MFE={estimated_mfe:.1f}pts, "
                    f"Current={current_profit_points:.1f}pts, Decline={decline_pct*100:.1f}%, "
                    f"Locking {mfe_lock_pct*100:.0f}% = {protected_profit_points:.1f}pts"
                )
                return round(mfe_trail_level, 5)

        return None

    def should_trail(self, trade: TradeLog, current_price: float,
                    candle_data: List[IGCandle]) -> bool:
        """Determine if we should update trailing stop for any of the 3 stages"""
        direction = trade.direction.upper()
        point_value = self._get_point_value(trade.symbol)

        # Calculate current profit
        if direction == "BUY":
            profit_points = int((current_price - trade.entry_price) / point_value)
        else:  # SELL
            profit_points = int((trade.entry_price - current_price) / point_value)

        # Use standard stage1 trigger (IG minimum distance + offset)
        # Use config value only (no dynamic calculation)
        stage1_trigger = self.config.stage1_trigger_points

        # Always try to trail if we have enough profit for any stage
        return profit_points >= stage1_trigger

    def _calculate_atr(self, candles: List[IGCandle]) -> Optional[float]:
        """Calculate ATR for Stage 3 trailing"""
        if len(candles) < 2:
            return None

        # Sort and get recent candles
        recent_candles = sorted(candles, key=lambda x: x.start_time)[-15:]

        true_ranges = []
        for i in range(1, len(recent_candles)):
            prev_close = recent_candles[i-1].close
            current = recent_candles[i]

            tr = max(
                current.high - current.low,
                abs(current.high - prev_close),
                abs(current.low - prev_close)
            )
            true_ranges.append(tr)

        return sum(true_ranges) / len(true_ranges) if true_ranges else None

    def _get_point_value(self, epic: str) -> float:
        if "JPY" in epic:
            return 0.01
        elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return 0.0001
        return 1.0


class AdvancedTrailingManager:
    """Manager class for advanced trailing strategies - ENHANCED WITH FIXES"""
    
    def __init__(self, config: TrailingConfig, logger):
        self.config = config
        self.logger = logger
        self.strategies = {
            TrailingMethod.FIXED_POINTS: FixedPointsTrailing(config, logger),
            TrailingMethod.PERCENTAGE: PercentageTrailing(config, logger),
            TrailingMethod.ATR_BASED: ATRTrailing(config, logger),
            TrailingMethod.CHANDELIER: ChandelierTrailing(config, logger),
            TrailingMethod.SMART_TRAIL: SmartTrailing(config, logger),
            TrailingMethod.PROGRESSIVE_3_STAGE: Progressive3StageTrailing(config, logger),
        }
    
    def get_candle_data(self, db: Session, symbol: str, timeframe: int = 60, 
                       limit: int = 50) -> List[IGCandle]:
        """Get recent candle data for analysis - ENHANCED WITH ERROR HANDLING"""
        try:
            return (db.query(IGCandle)
                   .filter(IGCandle.epic == symbol, IGCandle.timeframe == timeframe)
                   .order_by(IGCandle.start_time.desc())
                   .limit(limit)
                   .all())
        except Exception as e:
            self.logger.error(f"[CANDLE DATA ERROR] {symbol}: {e}")
            return []
    
    def validate_stop_level(self, trade: TradeLog, current_price: float, proposed_stop: float) -> bool:
        """Validate that the proposed stop level is safe and logical - ADDED MISSING METHOD"""
        try:
            direction = trade.direction.upper()
            point_value = self._get_point_value(trade.symbol)
            min_distance = self.config.min_trail_distance * point_value
            
            # Use epsilon for floating-point comparison to avoid precision issues
            epsilon = 0.001  # Allow for tiny floating-point differences
            
            # Calculate actual distance
            if direction == "BUY":
                actual_distance = current_price - proposed_stop
                max_trail = current_price - min_distance
                is_valid = proposed_stop <= max_trail and actual_distance > 0
                
                self.logger.debug(f"[VALIDATION] {trade.symbol} BUY: "
                                f"current={current_price:.5f}, proposed={proposed_stop:.5f}, "
                                f"distance={actual_distance:.5f} ({actual_distance/point_value:.3f}pts), "
                                f"min_required={min_distance:.5f} ({min_distance/point_value:.3f}pts), "
                                f"valid={is_valid}")
                                
                if not is_valid:
                    if proposed_stop > current_price:
                        self.logger.error(f"[VALIDATION FAIL] BUY stop {proposed_stop:.5f} ABOVE current {current_price:.5f}")
                    elif actual_distance < (min_distance - epsilon):
                        self.logger.error(f"[VALIDATION FAIL] BUY distance {actual_distance/point_value:.3f}pts < required {min_distance/point_value:.3f}pts")
                    else:
                        self.logger.debug(f"[VALIDATION PASS] BUY distance {actual_distance/point_value:.3f}pts >= required {min_distance/point_value:.3f}pts")
                        is_valid = True  # Override validation if it's within epsilon tolerance
            else:
                actual_distance = proposed_stop - current_price
                min_trail = current_price + min_distance
                is_valid = proposed_stop >= min_trail and actual_distance > 0
                
                self.logger.debug(f"[VALIDATION] {trade.symbol} SELL: "
                                f"current={current_price:.5f}, proposed={proposed_stop:.5f}, "
                                f"distance={actual_distance:.5f} ({actual_distance/point_value:.3f}pts), "
                                f"min_required={min_distance:.5f} ({min_distance/point_value:.3f}pts), "
                                f"valid={is_valid}")
                                
                if not is_valid:
                    if proposed_stop < current_price:
                        self.logger.error(f"[VALIDATION FAIL] SELL stop {proposed_stop:.5f} BELOW current {current_price:.5f}")
                    elif actual_distance < (min_distance - epsilon):
                        self.logger.error(f"[VALIDATION FAIL] SELL distance {actual_distance/point_value:.3f}pts < required {min_distance/point_value:.3f}pts")
                    else:
                        self.logger.debug(f"[VALIDATION PASS] SELL distance {actual_distance/point_value:.3f}pts >= required {min_distance/point_value:.3f}pts")
                        is_valid = True  # Override validation if it's within epsilon tolerance
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"[VALIDATION ERROR] Trade {trade.id}: {e}")
            return False

    def _get_point_value(self, epic: str) -> float:
        if "JPY" in epic:
            return 0.01
        elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return 0.0001
        return 1.0
    
    def should_update_trail(self, trade: TradeLog, current_price: float, 
                          db: Session) -> bool:
        """Determine if trailing stop should be updated - ENHANCED"""
        try:
            strategy = self.strategies[self.config.method]
            candle_data = self.get_candle_data(db, trade.symbol)
            
            return strategy.should_trail(trade, current_price, candle_data)
        except Exception as e:
            self.logger.error(f"[SHOULD TRAIL ERROR] {trade.symbol}: {e}")
            return False
    
    def calculate_new_trail_level(self, trade: TradeLog, current_price: float,
                                db: Session) -> Optional[float]:
        """Calculate new trailing stop level - ENHANCED WITH VALIDATION"""
        try:
            strategy = self.strategies[self.config.method]
            candle_data = self.get_candle_data(db, trade.symbol)
            
            new_level = strategy.calculate_trail_level(trade, current_price, candle_data, db)
            
            if new_level:
                # ✅ CRITICAL: Validate the calculated level
                if not self._validate_trail_level(trade, current_price, new_level):
                    self.logger.warning(f"[TRAIL VALIDATION FAILED] {trade.symbol}: "
                                      f"Calculated level {new_level:.5f} invalid for current {current_price:.5f}")
                    return None
                
                self.logger.info(f"[{self.config.method.value.upper()}] {trade.symbol} "
                               f"new trail level: {new_level:.5f}")
            
            return new_level
        except Exception as e:
            self.logger.error(f"[TRAIL CALC ERROR] {trade.symbol}: {e}")
            return None
    
    def _validate_trail_level(self, trade: TradeLog, current_price: float, trail_level: float) -> bool:
        """Validate that trail level is safe and logical - ENHANCED WITH FLOATING-POINT FIX"""
        try:
            direction = trade.direction.upper()
            point_value = self._get_point_value(trade.symbol)
            min_distance = self.config.min_trail_distance * point_value
            
            # Use epsilon for floating-point comparison to avoid precision issues
            epsilon = 0.001  # Allow for tiny floating-point differences
            
            # Calculate actual distance
            if direction == "BUY":
                actual_distance = current_price - trail_level
                max_trail = current_price - min_distance
                is_valid = trail_level <= max_trail and actual_distance > 0
                
                self.logger.debug(f"[VALIDATION] {trade.symbol} BUY: "
                                f"current={current_price:.5f}, trail={trail_level:.5f}, "
                                f"distance={actual_distance:.5f} ({actual_distance/point_value:.3f}pts), "
                                f"min_required={min_distance:.5f} ({min_distance/point_value:.3f}pts), "
                                f"valid={is_valid}")
                                
                if not is_valid:
                    if trail_level > current_price:
                        self.logger.error(f"[VALIDATION FAIL] BUY trail {trail_level:.5f} ABOVE current {current_price:.5f}")
                    elif actual_distance < (min_distance - epsilon):  # ✅ FIXED: Use epsilon for comparison
                        self.logger.error(f"[VALIDATION FAIL] BUY distance {actual_distance/point_value:.3f}pts < required {min_distance/point_value:.3f}pts")
                    else:
                        # ✅ NEW: Log successful validation when distance meets requirements
                        self.logger.debug(f"[VALIDATION PASS] BUY distance {actual_distance/point_value:.3f}pts >= required {min_distance/point_value:.3f}pts")
                        is_valid = True  # Override validation if it's within epsilon tolerance
            else:
                actual_distance = trail_level - current_price
                min_trail = current_price + min_distance
                is_valid = trail_level >= min_trail and actual_distance > 0
                
                self.logger.debug(f"[VALIDATION] {trade.symbol} SELL: "
                                f"current={current_price:.5f}, trail={trail_level:.5f}, "
                                f"distance={actual_distance:.5f} ({actual_distance/point_value:.3f}pts), "
                                f"min_required={min_distance:.5f} ({min_distance/point_value:.3f}pts), "
                                f"valid={is_valid}")
                                
                if not is_valid:
                    if trail_level < current_price:
                        self.logger.error(f"[VALIDATION FAIL] SELL trail {trail_level:.5f} BELOW current {current_price:.5f}")
                    elif actual_distance < (min_distance - epsilon):  # ✅ FIXED: Use epsilon for comparison
                        self.logger.error(f"[VALIDATION FAIL] SELL distance {actual_distance/point_value:.3f}pts < required {min_distance/point_value:.3f}pts")
                    else:
                        # ✅ NEW: Log successful validation when distance meets requirements
                        self.logger.debug(f"[VALIDATION PASS] SELL distance {actual_distance/point_value:.3f}pts >= required {min_distance/point_value:.3f}pts")
                        is_valid = True  # Override validation if it's within epsilon tolerance
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"[VALIDATION ERROR] Trade {trade.id}: {e}")
            return False
    
    def get_trail_adjustment_points(self, trade: TradeLog, current_price: float,
                                  new_trail_level: float) -> int:
        """Convert trail level to adjustment points for API - ENHANCED"""
        try:
            direction = trade.direction.upper()
            point_value = self._get_point_value(trade.symbol)
            current_stop = trade.sl_price or 0.0
            
            # Calculate the distance between current stop and new trail level
            if direction == "BUY":
                adjustment_distance = new_trail_level - current_stop
            else:
                adjustment_distance = current_stop - new_trail_level
            
            adjustment_points = int(abs(adjustment_distance) / point_value)
            
            # Ensure minimum adjustment
            return max(1, adjustment_points)
        except Exception as e:
            self.logger.error(f"[ADJUSTMENT CALC ERROR] {trade.symbol}: {e}")
            return 1
    
    def _get_point_value(self, epic: str) -> float:
        if "JPY" in epic:
            return 0.01
        elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return 0.0001
        return 1.0


class EnhancedTradeProcessor:
    """Enhanced trade processor with advanced trailing strategies - CRITICALLY FIXED"""

    def __init__(self, trailing_config: TrailingConfig, order_sender, logger):
        self.trailing_manager = AdvancedTrailingManager(trailing_config, logger)
        self.order_sender = order_sender
        self.logger = logger
        self.config = trailing_config

    async def send_stop_adjustment(self, trade: TradeLog, new_stop: float, reason: str = "") -> bool:
        """
        Async method to send stop adjustment using absolute stop level.
        Used by partial close and other async contexts.

        Args:
            trade: The trade to adjust
            new_stop: Absolute stop level to set
            reason: Reason for the adjustment (for logging)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from services.adjust_stop_service import adjust_stop_logic

            self.logger.info(f"📍 [ASYNC STOP ADJUST] Trade {trade.id} {trade.symbol}: "
                           f"Setting stop to {new_stop:.5f} ({reason})")

            result = await adjust_stop_logic(
                epic=trade.symbol,
                new_stop=new_stop,
                stop_offset_points=None,
                limit_offset_points=None,
                dry_run=False
            )

            status = result.get("status", "unknown")

            if status == "updated":
                sent_payload = result.get("sentPayload", {})
                actual_stop = sent_payload.get("stopLevel")
                self.logger.info(f"[✅ ASYNC STOP UPDATED] {trade.symbol} → IG set stopLevel={actual_stop}")
                return True
            elif status == "closed":
                self.logger.warning(f"[❌ POSITION CLOSED] {trade.symbol}: {result.get('message')}")
                return False
            else:
                self.logger.error(f"[❌ ASYNC ADJUSTMENT FAILED] {trade.symbol}: {result.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            self.logger.error(f"❌ [ASYNC SEND ERROR] Trade {trade.id}: {e}")
            return False

    def calculate_safe_trail_distance(self, trade: TradeLog) -> int:
        """Calculate safe trailing distance respecting IG's minimum requirements - ENHANCED"""
        try:
            # Use IG's minimum distance if available, otherwise use config minimum
            ig_min_distance = getattr(trade, 'min_stop_distance_points', None)
            config_min_distance = self.config.min_trail_distance
            
            if ig_min_distance:
                # Use the larger of IG's requirement or our config
                safe_distance = max(ig_min_distance, config_min_distance)
                self.logger.debug(f"[SAFE DISTANCE] Trade {trade.id} {trade.symbol}: "
                                f"IG min={ig_min_distance}, config min={config_min_distance}, using={safe_distance}")
            else:
                # Fallback to config if IG minimum not stored
                safe_distance = config_min_distance
                self.logger.debug(f"[SAFE DISTANCE] Trade {trade.id} {trade.symbol}: "
                                f"No IG minimum stored, using config={safe_distance}")
            
            self.logger.info(f"[SAFE DISTANCE RESULT] Trade {trade.id} {trade.symbol}: "
                           f"Returning {safe_distance} points for trailing")
            return safe_distance
        except Exception as e:
            self.logger.error(f"[SAFE DISTANCE ERROR] Trade {trade.id}: {e}")
            return self.config.min_trail_distance
    
    def validate_stop_level(self, trade: TradeLog, current_price: float, proposed_stop: float) -> bool:
        """Validate that the proposed stop level is safe - SYNCHRONIZED WITH TRAILING STRATEGY"""
        try:
            direction = trade.direction.upper()
            point_value = get_point_value(trade.symbol)
            
            # ✅ CRITICAL FIX: Use the SAME distance that the trailing strategy uses
            # Instead of calculate_safe_trail_distance(), use the config minimum
            # This ensures consistency between calculation and validation
            strategy_distance = getattr(trade, "min_stop_distance_points", 4)  # USE IG MIN ONLY - min_trail_distance DISABLED
            min_distance_price = strategy_distance * point_value
            
            self.logger.debug(f"[VALIDATION SYNC] {trade.symbol}: Using strategy distance {strategy_distance} points "
                            f"instead of safe distance for consistency")
            
            if direction == "BUY":
                # For BUY: stop must be below current price by at least min distance
                required_stop_max = current_price - min_distance_price
                actual_distance = current_price - proposed_stop
                is_valid = proposed_stop <= required_stop_max and actual_distance > 0
                
                actual_points = actual_distance / point_value
                self.logger.debug(f"[VALIDATION SYNC] {trade.symbol} BUY: "
                                f"current={current_price:.5f}, proposed={proposed_stop:.5f}, "
                                f"actual_distance={actual_points:.1f}pts, required={strategy_distance}pts, "
                                f"valid={is_valid}")
                                
                if not is_valid:
                    if proposed_stop > current_price:
                        self.logger.error(f"[VALIDATION FAIL] BUY stop {proposed_stop:.5f} ABOVE current {current_price:.5f}")
                    elif actual_distance < min_distance_price:
                        self.logger.error(f"[VALIDATION FAIL] BUY distance {actual_points:.1f}pts < required {strategy_distance}pts")
            else:
                # For SELL: stop must be above current price by at least min distance  
                required_stop_min = current_price + min_distance_price
                actual_distance = proposed_stop - current_price
                is_valid = proposed_stop >= required_stop_min and actual_distance > 0
                
                actual_points = actual_distance / point_value
                self.logger.debug(f"[VALIDATION SYNC] {trade.symbol} SELL: "
                                f"current={current_price:.5f}, proposed={proposed_stop:.5f}, "
                                f"actual_distance={actual_points:.1f}pts, required={strategy_distance}pts, "
                                f"valid={is_valid}")
                                
                if not is_valid:
                    if proposed_stop < current_price:
                        self.logger.error(f"[VALIDATION FAIL] SELL stop {proposed_stop:.5f} BELOW current {current_price:.5f}")
                    elif actual_distance < min_distance_price:
                        self.logger.error(f"[VALIDATION FAIL] SELL distance {actual_points:.1f}pts < required {strategy_distance}pts")
            
            return is_valid
        except Exception as e:
            self.logger.error(f"[STOP VALIDATION ERROR] Trade {trade.id}: {e}")
            return False

    def calculate_safe_stop_level(self, trade: TradeLog, current_price: float) -> float:
        """Calculate a safe stop level that respects IG's minimum distance - ENHANCED"""
        try:
            direction = trade.direction.upper()
            safe_distance = self.calculate_safe_trail_distance(trade)
            point_value = get_point_value(trade.symbol)
            safe_distance_price = safe_distance * point_value
            
            if direction == "BUY":
                safe_stop = current_price - safe_distance_price
            else:  # SELL
                safe_stop = current_price + safe_distance_price
            
            return round(safe_stop, 5)
        except Exception as e:
            self.logger.error(f"[SAFE STOP CALC ERROR] Trade {trade.id}: {e}")
            # Fallback calculation
            point_value = get_point_value(trade.symbol)
            fallback_distance = self.config.min_trail_distance * point_value
            if trade.direction.upper() == "BUY":
                return current_price - fallback_distance
            else:
                return current_price + fallback_distance

    async def process_trade_with_advanced_trailing(self, trade: TradeLog, current_price: float, db: Session) -> bool:
        """Process trade with break-even logic and then advanced trailing - CRITICALLY FIXED"""

        # Diagnostic log
        self.logger.info(f"🔧 [ENHANCED] Processing trade {trade.id} {trade.symbol} status={trade.status}")

        try:
            # ✅ NEW: Use pair-specific configuration dynamically
            from config import get_trailing_config_for_epic

            trailing_config = get_trailing_config_for_epic(trade.symbol)
            STAGE1_TRIGGER_POINTS = trailing_config['stage1_trigger_points']
            STAGE1_LOCK_POINTS = trailing_config['stage1_lock_points']
            STAGE2_TRIGGER_POINTS = trailing_config['stage2_trigger_points']
            STAGE2_LOCK_POINTS = trailing_config['stage2_lock_points']
            STAGE3_TRIGGER_POINTS = trailing_config['stage3_trigger_points']

            self.logger.info(f"📊 [PAIR CONFIG] {trade.symbol}: "
                           f"Stage1({STAGE1_TRIGGER_POINTS}pts→{STAGE1_LOCK_POINTS}pts) "
                           f"Stage2({STAGE2_TRIGGER_POINTS}pts→{STAGE2_LOCK_POINTS}pts) "
                           f"Stage3({STAGE3_TRIGGER_POINTS}pts)")

            point_value = get_point_value(trade.symbol)

            # Use config file value only (no dynamic calculation)
            break_even_trigger_points = trailing_config['break_even_trigger_points']
            break_even_trigger = break_even_trigger_points * point_value
            
            # Calculate safe trailing distance
            safe_trail_distance = self.calculate_safe_trail_distance(trade)
            safe_trail_distance_price = safe_trail_distance * point_value

            # ✅ CRITICAL FIX: Calculate current profit correctly
            if trade.direction.upper() == "BUY":
                moved_in_favor = current_price - trade.entry_price
                profit_points = int(abs(moved_in_favor) / point_value)
                is_profitable_for_breakeven = moved_in_favor >= break_even_trigger
                
                self.logger.info(f"📊 [PROFIT] Trade {trade.id} {trade.symbol} BUY: "
                            f"entry={trade.entry_price:.5f}, current={current_price:.5f}, "
                            f"profit={profit_points}pts, trigger={break_even_trigger_points}pts")
                
            elif trade.direction.upper() == "SELL":
                moved_in_favor = trade.entry_price - current_price
                profit_points = int(abs(moved_in_favor) / point_value)
                is_profitable_for_breakeven = moved_in_favor >= break_even_trigger
                
                self.logger.info(f"📊 [PROFIT] Trade {trade.id} {trade.symbol} SELL: "
                            f"entry={trade.entry_price:.5f}, current={current_price:.5f}, "
                            f"profit={profit_points}pts, trigger={break_even_trigger_points}pts")

            # --- STEP 1: Break-even logic ---
            # ✅ CRITICAL FIX: Skip break-even logic if already profit_protected
            if trade.status == "profit_protected":
                self.logger.info(f"🛡️ [PROFIT PROTECTED] Trade {trade.id}: Skipping break-even check, already protected at +10pts")
                # Set moved_to_breakeven to True to allow trailing
                trade.moved_to_breakeven = True
                # ✅ NEW: Also set moved_to_stage2 since profit_protected implies Stage 2 completed
                trade.moved_to_stage2 = True

                # ✅ ROBUST: Wrap commit in try-except
                try:
                    db.commit()
                    self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: Profit protected flags updated")
                except Exception as commit_error:
                    db.rollback()
                    self.logger.error(f"❌ [DB COMMIT FAILED] Trade {trade.id}: {type(commit_error).__name__}: {str(commit_error)}")
                    raise
                # Don't change status if already profit_protected
            elif not getattr(trade, 'moved_to_breakeven', False) and is_profitable_for_breakeven:
                self.logger.info(f"🎯 [BREAK-EVEN TRIGGER] Trade {trade.id}: "
                            f"Profit {profit_points}pts >= trigger {break_even_trigger_points}pts")

                # ========== PARTIAL CLOSE CHECK (separate from break-even) ==========
                # Partial close now triggers at partial_close_trigger_points (default 13 pips)
                # NOT at break-even trigger
                enable_partial_close = trailing_config.get('enable_partial_close', True)
                partial_close_size = trailing_config.get('partial_close_size', 0.5)
                partial_close_trigger = trailing_config.get('partial_close_trigger_points', 13)  # Default 13 pips
                partial_close_succeeded = False

                # Check if profit is sufficient for partial close (separate threshold from BE)
                is_profitable_for_partial_close = profit_points >= partial_close_trigger

                if enable_partial_close and is_profitable_for_partial_close:
                    self.logger.info(f"💰 [PARTIAL CLOSE TRIGGER] Trade {trade.id}: "
                                f"Profit {profit_points}pts >= partial close trigger {partial_close_trigger}pts")
                    # ✅ CRITICAL FIX: ALWAYS check database first with row lock to prevent race conditions
                    # The passed-in trade object may be stale/detached - never trust its partial_close_executed value
                    try:
                        # Get fresh trade with row lock - this is the authoritative source
                        refreshed_trade = db.query(TradeLog).filter(TradeLog.id == trade.id).with_for_update(nowait=False).first()
                        if not refreshed_trade:
                            self.logger.error(f"❌ [PARTIAL CLOSE] Trade {trade.id}: Could not find trade in database")
                            raise Exception("Trade not found in database")

                        # ✅ ALWAYS check the DATABASE value, not the passed-in object
                        if refreshed_trade.partial_close_executed:
                            self.logger.info(f"⏭️ [PARTIAL CLOSE SKIP] Trade {trade.id}: Already executed (detected via database lock)")
                            partial_close_succeeded = True  # Skip break-even logic
                        else:
                            self.logger.info(f"💡 [PARTIAL CLOSE] Trade {trade.id}: Attempting to close {partial_close_size} position (WITH ROW LOCK)")

                            # Import partial close function
                            import asyncio
                            from services.ig_orders import partial_close_position
                            from dependencies import get_ig_auth_headers

                            # Get auth headers asynchronously
                            auth_headers = await get_ig_auth_headers()

                            # Execute partial close asynchronously
                            # ✅ FIX: Use refreshed_trade instead of detached trade object
                            partial_result = await partial_close_position(
                                deal_id=refreshed_trade.deal_id,
                                epic=refreshed_trade.symbol,
                                direction=refreshed_trade.direction,
                                size_to_close=partial_close_size,
                                auth_headers=auth_headers
                            )

                            # ✅ Handle case where position was already closed
                            if partial_result.get("already_closed"):
                                self.logger.warning(f"⚠️ [PARTIAL CLOSE] Trade {trade.id}: Position already closed on IG")
                                # Mark as closed in database
                                refreshed_trade.status = "closed"
                                refreshed_trade.partial_close_executed = True  # Prevent further attempts
                                db.commit()
                                return True  # Exit processing for this trade

                            if partial_result.get("success"):
                                # ✅ Partial close succeeded!
                                self.logger.info(f"✅ [PARTIAL CLOSE SUCCESS] Trade {trade.id} {trade.symbol}: "
                                               f"Closed {partial_close_size}, keeping {1.0 - partial_close_size} with original SL")

                                # Update trade tracking (using refreshed_trade to maintain lock)
                                refreshed_trade.current_size = 1.0 - partial_close_size  # e.g., 0.5 remaining
                                refreshed_trade.partial_close_executed = True
                                refreshed_trade.partial_close_time = datetime.utcnow()
                                refreshed_trade.status = "partial_closed"
                                refreshed_trade.moved_to_breakeven = True  # Enable Stage 2/3 progression

                                # Commit changes
                                try:
                                    db.commit()
                                    self.logger.info(f"✅ [PARTIAL CLOSE DB] Trade {trade.id}: Database updated, "
                                                   f"current_size={refreshed_trade.current_size}, continuing to Stage 2/3")

                                    # ✅ ENHANCEMENT: Move SL to entry + min_distance to guarantee profit
                                    # Calculate profit protection stop level
                                    ig_min_distance = getattr(refreshed_trade, 'min_stop_distance_points', None)
                                    if ig_min_distance:
                                        lock_points = max(1, round(ig_min_distance))
                                    else:
                                        lock_points = trailing_config['stage1_lock_points']

                                    if refreshed_trade.direction.upper() == "BUY":
                                        profit_protection_stop = refreshed_trade.entry_price + (lock_points * point_value)
                                    else:  # SELL
                                        profit_protection_stop = refreshed_trade.entry_price - (lock_points * point_value)

                                    self.logger.info(f"💰 [PARTIAL CLOSE PROTECTION] Trade {trade.id}: Moving SL to entry +/- {lock_points}pts "
                                                   f"to guarantee profit on remaining {refreshed_trade.current_size} position")

                                    # Move the stop loss
                                    try:
                                        adjustment_result = await self.send_stop_adjustment(
                                            trade=refreshed_trade,
                                            new_stop=profit_protection_stop,
                                            reason=f"partial_close_protection_{lock_points}pts"
                                        )

                                        if adjustment_result:
                                            self.logger.info(f"✅ [SL MOVED] Trade {trade.id}: Stop moved to {profit_protection_stop:.5f} "
                                                           f"(entry +/- {lock_points}pts) after partial close")
                                            refreshed_trade.sl_price = profit_protection_stop
                                            db.commit()
                                        else:
                                            self.logger.warning(f"⚠️ [SL MOVE FAILED] Trade {trade.id}: Could not move stop after partial close")
                                    except Exception as sl_error:
                                        self.logger.error(f"❌ [SL MOVE ERROR] Trade {trade.id}: {sl_error}")

                                    # ✅ SUCCESS: Set flag to skip break-even stop move
                                    partial_close_succeeded = True
                                    self.logger.info(f"🎉 [PARTIAL CLOSE COMPLETE] Trade {trade.id}: Banked 50% at +{profit_points}pts, "
                                                   f"remaining 50% protected at entry +/- {lock_points}pts")

                                except Exception as commit_error:
                                    db.rollback()
                                    self.logger.error(f"❌ [PARTIAL CLOSE DB FAIL] Trade {trade.id}: {commit_error}")
                                    # Fall through to try break-even stop move as fallback
                                    raise
                            else:
                                # ❌ Partial close failed
                                error_msg = partial_result.get("error", "Unknown error")
                                self.logger.warning(f"⚠️ [PARTIAL CLOSE FAILED] Trade {trade.id}: {error_msg}")
                                self.logger.info(f"🔄 [FALLBACK] Trade {trade.id}: Will try moving stop to break-even instead")
                                # Fall through to execute break-even stop move logic below

                    except Exception as partial_error:
                        # ❌ Exception during partial close attempt
                        self.logger.error(f"❌ [PARTIAL CLOSE ERROR] Trade {trade.id}: {str(partial_error)}")
                        self.logger.info(f"🔄 [FALLBACK] Trade {trade.id}: Exception occurred, trying break-even stop move")
                        # ✅ FIX: Refresh trade from session to ensure it's attached
                        # Since we no longer use db.expire(), the trade should still be attached
                        # But verify and re-query if needed
                        try:
                            db.refresh(trade)
                            self.logger.debug(f"🔄 [SESSION REFRESH] Trade {trade.id}: Trade refreshed in session")
                        except Exception as refresh_error:
                            # Trade might be detached, re-query it
                            self.logger.warning(f"⚠️ [SESSION REFRESH FAILED] Trade {trade.id}: {refresh_error}, re-querying...")
                            try:
                                trade = db.query(TradeLog).filter(TradeLog.id == trade.id).first()
                                if trade:
                                    self.logger.info(f"✅ [SESSION FIX] Trade {trade.id}: Re-queried successfully")
                            except Exception as query_error:
                                self.logger.error(f"❌ [SESSION FIX FAILED] Trade {trade.id}: {query_error}")
                        # Fall through to execute break-even stop move logic below
                else:
                    if not enable_partial_close:
                        self.logger.info(f"ℹ️ [PARTIAL CLOSE DISABLED] Trade {trade.id}: Feature disabled in config")
                    elif getattr(trade, 'partial_close_executed', False):
                        self.logger.info(f"ℹ️ [PARTIAL CLOSE SKIP] Trade {trade.id}: Already executed")
                # ========== END PARTIAL CLOSE ATTEMPT ==========

                # Only execute break-even stop move if partial close didn't succeed
                if partial_close_succeeded:
                    # Skip break-even stop move entirely - partial close already executed
                    self.logger.info(f"⏭️ [SKIP BE STOP] Trade {trade.id}: Partial close succeeded, skipping break-even stop move")
                else:
                    # Calculate break-even stop level using IG's minimum distance if available
                    # ✅ ENHANCEMENT: Use IG's minimum stop distance for better trade evolution
                    ig_min_distance = getattr(trade, 'min_stop_distance_points', None)
                    if ig_min_distance:
                        lock_points = max(1, round(ig_min_distance))  # Round and ensure minimum 1 point
                        self.logger.info(f"🎯 [USING IG MIN] Trade {trade.id}: Using IG minimum distance {lock_points}pts")
                    else:
                        # ✅ NEW: Use pair-specific configuration
                        lock_points = trailing_config['stage1_lock_points']
                        self.logger.info(f"⚠️ [FALLBACK CONFIG] Trade {trade.id}: No IG minimum distance, using pair config {lock_points}pts")
                    if trade.direction.upper() == "BUY":
                        break_even_stop = trade.entry_price + (lock_points * point_value)  # Entry + lock_points
                    else:
                        break_even_stop = trade.entry_price - (lock_points * point_value)  # Entry - lock_points

                
                    # Enhanced logging: Track exact values used
                    self.logger.info(f"💰 [BREAK-EVEN DETAILED] Trade {trade.id}:")
                    self.logger.info(f"   → IG minimum: {ig_min_distance} points")
                    self.logger.info(f"   → Trigger points: {break_even_trigger_points} points")
                    self.logger.info(f"   → Using lock_points: {lock_points} points")
                    self.logger.info(f"   → Entry: {trade.entry_price:.5f}")
                    self.logger.info(f"   → Break-even stop: {break_even_stop:.5f}")
                    self.logger.info(f"   → Distance from entry: {((break_even_stop - trade.entry_price) / point_value):.1f} points")
    # Calculate actual currency amount for JPY pairs
                    if "JPY" in trade.symbol:
                        currency_amount = int(lock_points * 100)  # 2 points = 200 JPY
                        self.logger.info(f"💰 [BREAK-EVEN CALC] Trade {trade.id}: entry={trade.entry_price:.5f}, "
                                       f"lock_points={lock_points} ({currency_amount} JPY), break_even_stop={break_even_stop:.5f}")
                    else:
                        self.logger.info(f"💰 [BREAK-EVEN CALC] Trade {trade.id}: entry={trade.entry_price:.5f}, "
                                       f"lock_points={lock_points}, break_even_stop={break_even_stop:.5f}")
                
                    # ✅ CRITICAL FIX: For profit_protected trades, check if break-even would worsen the position
                    # ✅ FIX: Re-query trade from current session to get fresh data
                    try:
                        fresh_trade = db.query(TradeLog).filter(TradeLog.id == trade.id).first()
                        if fresh_trade:
                            current_stop = fresh_trade.sl_price or 0.0
                        else:
                            self.logger.warning(f"[TRADE NOT FOUND] Trade {trade.id}: Using cached sl_price")
                            current_stop = trade.sl_price or 0.0
                    except Exception as e:
                        self.logger.warning(f"[DB QUERY WARNING] Trade {trade.id}: Could not query fresh data, using current value: {e}")
                        current_stop = trade.sl_price or 0.0
                
                    # Check if break-even move would worsen protection
                    if trade.direction.upper() == "BUY":
                        would_worsen = break_even_stop < current_stop  # Moving stop down is worse for BUY
                    else:  # SELL
                        would_worsen = break_even_stop > current_stop  # Moving stop up is worse for SELL
                
                    if would_worsen:
                        self.logger.info(f"🛡️ [SKIP BREAK-EVEN] Trade {trade.id}: Current stop ({current_stop:.5f}) is better than break-even ({break_even_stop:.5f})")
                        # Set flag to allow trailing without actually moving stop
                        old_status = trade.status
                        trade.moved_to_breakeven = True
                        # Don't change status if already profit_protected
                        if trade.status != "profit_protected":
                            trade.status = "break_even"

                        # ✅ ROBUST: Wrap commit in try-except
                        self.logger.info(f"💾 [DB PREPARE] Trade {trade.id}: status {old_status} → {trade.status}, moved_to_breakeven=True")
                        try:
                            db.commit()
                            self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: Break-even flags updated (skip scenario)")
                        except Exception as commit_error:
                            db.rollback()
                            self.logger.error(f"❌ [DB COMMIT FAILED] Trade {trade.id}: {type(commit_error).__name__}: {str(commit_error)}")
                            raise
                    else:
                        # ✅ FIXED: Simple validation - stop just needs to be on correct side of current price
                        # Break-even = entry + lock_points is ALWAYS valid as long as:
                        # - For BUY: break_even_stop < current_price (stop below current)
                        # - For SELL: break_even_stop > current_price (stop above current)

                        is_valid_stop = True
                        current_stop = trade.sl_price or 0.0

                        # Basic sanity check: ensure stop is on correct side of current price
                        if trade.direction.upper() == "BUY":
                            if break_even_stop >= current_price:
                                self.logger.error(f"❌ [BREAK-EVEN ERROR] Trade {trade.id}: BUY stop {break_even_stop:.5f} >= current {current_price:.5f}")
                                is_valid_stop = False
                        else:  # SELL
                            if break_even_stop <= current_price:
                                self.logger.error(f"❌ [BREAK-EVEN ERROR] Trade {trade.id}: SELL stop {break_even_stop:.5f} <= current {current_price:.5f}")
                                is_valid_stop = False

                        if is_valid_stop:
                            # Calculate adjustment needed
                            if trade.direction.upper() == "BUY":
                                adjustment_distance = break_even_stop - current_stop
                            else:
                                adjustment_distance = current_stop - break_even_stop

                            # ✅ FIX: Use proper rounding instead of truncation to avoid 0-point adjustments
                            adjustment_points = round(abs(adjustment_distance) / point_value)

                            # ✅ FIX: Handle edge case where adjustment is very small but not zero
                            if adjustment_points == 0 and abs(adjustment_distance) > 0.00001:
                                adjustment_points = 1  # Minimum 1-point adjustment for non-zero distances
                                self.logger.debug(f"[BREAK-EVEN CALC] Trade {trade.id}: Small adjustment rounded up from {abs(adjustment_distance):.6f} to 1pt")

                            # Diagnostic logging for troubleshooting
                            self.logger.debug(f"[BREAK-EVEN CALC] Trade {trade.id}: current_stop={current_stop:.5f}, "
                                            f"break_even_stop={break_even_stop:.5f}, adjustment_distance={adjustment_distance:.6f}, "
                                            f"adjustment_points={adjustment_points}")
                        
                            if adjustment_points > 0:
                                direction_stop = "increase" if trade.direction.upper() == "BUY" else "decrease"

                                self.logger.info(f"📤 [BREAK-EVEN SEND] Trade {trade.id}: "
                                            f"Moving stop to break-even at {break_even_stop:.5f} (entry+{lock_points}pts)")

                                # ✅ CRITICAL FIX: Use absolute stop level instead of offset
                                api_result = self._send_stop_adjustment(
                                    trade, adjustment_points, direction_stop, 0,
                                    new_stop_level=break_even_stop  # Pass absolute level
                                )

                                if isinstance(api_result, dict) and api_result.get("status") == "updated":
                                    # Extract IG's actual stop level for break-even
                                    sent_payload = api_result.get("sentPayload", {})
                                    ig_actual_stop = sent_payload.get("stopLevel")

                                    # Store old values for logging
                                    old_sl = trade.sl_price
                                    old_status = trade.status
                                    old_moved_to_breakeven = getattr(trade, 'moved_to_breakeven', False)

                                    # Update trade object fields
                                    trade.moved_to_breakeven = True
                                    # ✅ IMPORTANT: Don't change status if already profit_protected
                                    if trade.status != "profit_protected":
                                        trade.status = "break_even"

                                    # Use IG's actual stop level
                                    if ig_actual_stop:
                                        trade.sl_price = float(ig_actual_stop)
                                        self.logger.info(f"📍 [IG ACTUAL BE] Trade {trade.id}: IG set break-even to {ig_actual_stop:.5f}")
                                    else:
                                        trade.sl_price = break_even_stop
                                        self.logger.warning(f"⚠️ [FALLBACK BE] Trade {trade.id}: Using calculated BE {break_even_stop:.5f}")

                                    trade.last_trigger_price = current_price
                                    trade.trigger_time = datetime.utcnow()

                                    # ✅ FIX: Increment stop_limit_changes_count
                                    trade.stop_limit_changes_count = (trade.stop_limit_changes_count or 0) + 1

                                    # Log changes before commit
                                    self.logger.info(f"💾 [DB PREPARE] Trade {trade.id}: Preparing database commit")
                                    self.logger.info(f"   → sl_price: {old_sl} → {trade.sl_price}")
                                    self.logger.info(f"   → status: {old_status} → {trade.status}")
                                    self.logger.info(f"   → moved_to_breakeven: {old_moved_to_breakeven} → {trade.moved_to_breakeven}")
                                    self.logger.info(f"   → stop_limit_changes_count: {(trade.stop_limit_changes_count or 1) - 1} → {trade.stop_limit_changes_count}")

                                    # ✅ ROBUST: Wrap commit in try-except with detailed logging
                                    try:
                                        db.commit()
                                        self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: All changes persisted to database")
                                        self.logger.info(f"🎉 [BREAK-EVEN] Trade {trade.id} {trade.symbol} "
                                                    f"moved to break-even: {trade.sl_price:.5f}")
                                    except Exception as commit_error:
                                        db.rollback()
                                        self.logger.error(f"❌ [DB COMMIT FAILED] Trade {trade.id}: Database commit failed!")
                                        self.logger.error(f"   → Error type: {type(commit_error).__name__}")
                                        self.logger.error(f"   → Error message: {str(commit_error)}")
                                        self.logger.error(f"   → IG Markets was updated successfully but database sync failed")
                                        self.logger.error(f"   → This will cause database/broker mismatch!")
                                        # Re-raise to ensure error is visible
                                        raise
                                    # Don't return early - continue to Stage 2/3 progression check
                                else:
                                    error_msg = api_result.get("message", "Unknown error") if isinstance(api_result, dict) else "API call failed"
                                    self.logger.error(f"❌ [BREAK-EVEN FAILED] Trade {trade.id}: {error_msg}")
                            else:
                                self.logger.warning(f"⏸️ [BREAK-EVEN SKIP] Trade {trade.id}: Adjustment points = 0")
                                # Set flag anyway to prevent repeated attempts
                                old_status = trade.status
                                trade.moved_to_breakeven = True
                                if trade.status != "profit_protected":
                                    trade.status = "break_even"

                                self.logger.info(f"💾 [DB PREPARE BE-ZERO] Trade {trade.id}: status {old_status} → {trade.status}")
                                try:
                                    db.commit()
                                    self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: Break-even zero adjustment flags persisted")
                                except Exception as commit_error:
                                    db.rollback()
                                    self.logger.error(f"❌ [DB COMMIT FAILED BE-ZERO] Trade {trade.id}: {type(commit_error).__name__}: {str(commit_error)}")
                                    raise
                        else:
                            self.logger.warning(f"⏸️ [BREAK-EVEN SKIP] Trade {trade.id}: Break-even level validation failed")

            # --- STEP 1.5: Standard Stage Check ---
            # Check for Stage 1, Stage 2 and Stage 3 progression after break-even
            if getattr(trade, 'moved_to_breakeven', False):
                # Use standard 3-stage configuration
                stage1_trigger = STAGE1_TRIGGER_POINTS
                stage2_trigger = STAGE2_TRIGGER_POINTS
                stage3_trigger = STAGE3_TRIGGER_POINTS

                if profit_points >= stage3_trigger:
                    # Stage 3: Percentage-based trailing
                    self.logger.info(f"🚀 [STAGE 3 TRIGGER] Trade {trade.id}: Profit {profit_points}pts >= Stage 3 trigger {stage3_trigger}pts - Percentage trailing")
                    # Use progressive 3-stage strategy directly
                    progressive_strategy = Progressive3StageTrailing(self.config, self.logger)
                    try:
                        new_trail_level = progressive_strategy.calculate_trail_level(trade, current_price, [], db)
                        if new_trail_level:
                            current_stop = trade.sl_price or 0.0
                            # ✅ FIXED: Calculate proper directional adjustment
                            if trade.direction.upper() == "BUY":
                                # BUY: Trail stop should only move UP (increase)
                                if new_trail_level > current_stop:
                                    adjustment_distance = new_trail_level - current_stop
                                    adjustment_points = int(adjustment_distance / point_value)
                                    direction_stop = "increase"
                                else:
                                    adjustment_points = 0  # Don't move backwards
                            else:  # SELL
                                # SELL: Trail stop should only move DOWN (decrease)
                                if new_trail_level < current_stop:
                                    adjustment_distance = current_stop - new_trail_level
                                    adjustment_points = int(adjustment_distance / point_value)
                                    direction_stop = "decrease"
                                else:
                                    adjustment_points = 0  # Don't move backwards

                            # ✅ NEW: Enforce minimum movement for Stage 3 trailing
                            # This prevents tiny incremental movements that are too tight
                            from config import STAGE3_MIN_ADJUSTMENT
                            min_adjustment = STAGE3_MIN_ADJUSTMENT  # Default 5 points per trail movement

                            if adjustment_points >= min_adjustment:
                                # ✅ CRITICAL FIX: Use absolute stop level for Stage 3
                                success = self._send_stop_adjustment(
                                    trade, adjustment_points, direction_stop, 0,
                                    new_stop_level=new_trail_level
                                )
                                if isinstance(success, dict) and success.get("status") == "updated":
                                    old_sl = trade.sl_price
                                    old_status = trade.status

                                    trade.sl_price = new_trail_level
                                    trade.status = "stage3_trailing"
                                    trade.last_trigger_price = current_price
                                    trade.trigger_time = datetime.utcnow()
                                    # ✅ FIX: Increment counter
                                    trade.stop_limit_changes_count = (trade.stop_limit_changes_count or 0) + 1

                                    self.logger.info(f"💾 [DB PREPARE STAGE3] Trade {trade.id}: sl_price {old_sl} → {trade.sl_price}, changes: {trade.stop_limit_changes_count}")

                                    try:
                                        db.commit()
                                        self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: Stage 3 changes persisted")
                                        self.logger.info(f"🎯 [STAGE 3] Trade {trade.id}: Percentage trailing to {new_trail_level:.5f} ({adjustment_points}pts)")
                                        return True
                                    except Exception as commit_error:
                                        db.rollback()
                                        self.logger.error(f"❌ [DB COMMIT FAILED STAGE3] Trade {trade.id}: {type(commit_error).__name__}: {str(commit_error)}")
                                        raise
                            else:
                                self.logger.debug(f"📊 [STAGE 3 SKIP] Trade {trade.id}: Adjustment {adjustment_points}pts < minimum {min_adjustment}pts")
                    except Exception as e:
                        self.logger.error(f"❌ [STAGE 3 ERROR] Trade {trade.id}: {e}")

                elif profit_points >= stage2_trigger:
                    # Stage 2: Profit lock
                    # ✅ CRITICAL FIX: Check if Stage 2 already executed
                    if getattr(trade, 'moved_to_stage2', False):
                        self.logger.debug(f"📊 [STAGE 2 SKIP] Trade {trade.id}: Stage 2 already executed, waiting for Stage 3")
                    else:
                        self.logger.info(f"💰 [STAGE 2 TRIGGER] Trade {trade.id}: Profit {profit_points}pts >= Stage 2 trigger {stage2_trigger}pts - Profit lock")

                        # Calculate Stage 2 profit lock level (entry + lock points)
                        lock_points = STAGE2_LOCK_POINTS  # 10 points
                        if trade.direction.upper() == "BUY":
                            stage2_stop = trade.entry_price + (lock_points * point_value)
                        else:
                            stage2_stop = trade.entry_price - (lock_points * point_value)

                        # Check if this improves current stop
                        current_stop = trade.sl_price or 0.0
                        should_update = False
                        if trade.direction.upper() == "BUY":
                            should_update = stage2_stop > current_stop
                        else:
                            should_update = stage2_stop < current_stop

                        if should_update:
                            adjustment_distance = abs(stage2_stop - current_stop)
                            adjustment_points = int(adjustment_distance / point_value)
                            if adjustment_points > 0:
                                direction_stop = "increase" if trade.direction.upper() == "BUY" else "decrease"
                                self.logger.info(f"📤 [STAGE 2 SEND] Trade {trade.id}: Moving to profit lock at {stage2_stop:.5f} (+{lock_points}pts)")
                                # ✅ CRITICAL FIX: Use absolute stop level for Stage 2
                                success = self._send_stop_adjustment(
                                    trade, adjustment_points, direction_stop, 0,
                                    new_stop_level=stage2_stop
                                )
                                if isinstance(success, dict) and success.get("status") == "updated":
                                    old_sl = trade.sl_price
                                    old_status = trade.status
                                    trade.sl_price = stage2_stop
                                    trade.status = "stage2_profit_lock"
                                    trade.moved_to_stage2 = True  # ✅ NEW: Set flag to prevent re-execution
                                    trade.last_trigger_price = current_price
                                    trade.trigger_time = datetime.utcnow()
                                    trade.stop_limit_changes_count = (trade.stop_limit_changes_count or 0) + 1

                                    self.logger.info(f"💾 [DB PREPARE STAGE2] Trade {trade.id}: sl_price {old_sl} → {trade.sl_price}, status {old_status} → {trade.status}, changes: {trade.stop_limit_changes_count}")
                                    try:
                                        db.commit()
                                        self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: Stage 2 changes persisted")
                                        self.logger.info(f"💎 [STAGE 2] Trade {trade.id}: Profit locked at {stage2_stop:.5f} (+{lock_points}pts)")
                                        return True
                                    except Exception as commit_error:
                                        db.rollback()
                                        self.logger.error(f"❌ [DB COMMIT FAILED STAGE2] Trade {trade.id}: {type(commit_error).__name__}: {str(commit_error)}")
                                        raise
                        else:
                            # Stage 2 stop not better than current - mark as executed anyway
                            self.logger.info(f"📊 [STAGE 2 SKIP] Trade {trade.id}: Stage 2 stop {stage2_stop:.5f} not better than current {current_stop:.5f}")
                            trade.moved_to_stage2 = True

                            self.logger.info(f"💾 [DB PREPARE STAGE2-SKIP] Trade {trade.id}: moved_to_stage2=True")
                            try:
                                db.commit()
                                self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: Stage 2 skip flag persisted")
                            except Exception as commit_error:
                                db.rollback()
                                self.logger.error(f"❌ [DB COMMIT FAILED STAGE2-SKIP] Trade {trade.id}: {type(commit_error).__name__}: {str(commit_error)}")
                                raise

                elif profit_points >= stage1_trigger:
                    # Stage 1: Initial profit protection
                    # ✅ NEW: Check if Stage 1 already executed
                    if getattr(trade, 'moved_to_stage1', False):
                        self.logger.debug(f"📊 [STAGE 1 SKIP] Trade {trade.id}: Stage 1 already executed, waiting for Stage 2")
                    else:
                        self.logger.info(f"💰 [STAGE 1 TRIGGER] Trade {trade.id}: Profit {profit_points}pts >= Stage 1 trigger {stage1_trigger}pts - Initial profit lock")

                        # Calculate Stage 1 profit lock level (entry + lock points)
                        lock_points = STAGE1_LOCK_POINTS
                        if trade.direction.upper() == "BUY":
                            stage1_stop = trade.entry_price + (lock_points * point_value)
                        else:
                            stage1_stop = trade.entry_price - (lock_points * point_value)

                        # Check if this improves current stop
                        current_stop = trade.sl_price or 0.0
                        should_update = False
                        if trade.direction.upper() == "BUY":
                            should_update = stage1_stop > current_stop
                        else:
                            should_update = stage1_stop < current_stop

                        if should_update:
                            adjustment_distance = abs(stage1_stop - current_stop)
                            adjustment_points = int(adjustment_distance / point_value)
                            if adjustment_points > 0:
                                direction_stop = "increase" if trade.direction.upper() == "BUY" else "decrease"
                                self.logger.info(f"📤 [STAGE 1 SEND] Trade {trade.id}: Moving to profit lock (+{lock_points}pts), adjustment={adjustment_points}pts")
                                success = self._send_stop_adjustment(trade, adjustment_points, direction_stop, 0)
                                if isinstance(success, dict) and success.get("status") == "updated":
                                    old_sl = trade.sl_price
                                    old_status = trade.status
                                    trade.sl_price = stage1_stop
                                    trade.status = "stage1_profit_lock"
                                    trade.moved_to_stage1 = True  # ✅ NEW: Set flag to prevent re-execution
                                    trade.last_trigger_price = current_price
                                    trade.trigger_time = datetime.utcnow()
                                    trade.stop_limit_changes_count = (trade.stop_limit_changes_count or 0) + 1

                                    self.logger.info(f"💾 [DB PREPARE STAGE1] Trade {trade.id}: sl_price {old_sl} → {trade.sl_price}, status {old_status} → {trade.status}, changes: {trade.stop_limit_changes_count}")
                                    try:
                                        db.commit()
                                        self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: Stage 1 changes persisted")
                                        self.logger.info(f"💎 [STAGE 1] Trade {trade.id}: Profit locked at {stage1_stop:.5f} (+{lock_points}pts)")
                                        return True
                                    except Exception as commit_error:
                                        db.rollback()
                                        self.logger.error(f"❌ [DB COMMIT FAILED STAGE1] Trade {trade.id}: {type(commit_error).__name__}: {str(commit_error)}")
                                        raise
                        else:
                            # Stage 1 stop not better than current - mark as executed anyway
                            self.logger.info(f"📊 [STAGE 1 SKIP] Trade {trade.id}: Stage 1 stop {stage1_stop:.5f} not better than current {current_stop:.5f}")
                            trade.moved_to_stage1 = True

                            self.logger.info(f"💾 [DB PREPARE STAGE1-SKIP] Trade {trade.id}: moved_to_stage1=True")
                            try:
                                db.commit()
                                self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: Stage 1 skip flag persisted")
                            except Exception as commit_error:
                                db.rollback()
                                self.logger.error(f"❌ [DB COMMIT FAILED STAGE1-SKIP] Trade {trade.id}: {type(commit_error).__name__}: {str(commit_error)}")
                                raise

                else:
                    # Below Stage 1 trigger, continue normal processing
                    self.logger.debug(f"📊 [STAGE 0] Trade {trade.id}: Profit {profit_points}pts < Stage 1 trigger {stage1_trigger}pts")

            # --- STEP 2: Advanced trailing logic ---
            should_trail = False

            # ✅ CRITICAL FIX: Allow trailing for profit_protected status OR moved_to_breakeven
            trail_ready = getattr(trade, 'moved_to_breakeven', False) or trade.status == "profit_protected"

            self.logger.debug(f"🔧 [TRAIL CHECK] Trade {trade.id}: moved_to_breakeven={getattr(trade, 'moved_to_breakeven', False)}, "
                           f"status={trade.status}, trail_ready={trail_ready}")
            
            if trail_ready:
                # ✅ CRITICAL FIX: For profit_protected trades, calculate trailing from current stop level, not break-even
                if trade.status == "profit_protected":
                    # Calculate additional movement beyond current protected level
                    current_stop = trade.sl_price or 0.0
                    
                    if trade.direction.upper() == "BUY":
                        # For BUY: additional move = how much price moved above current stop + safe distance
                        additional_move = current_price - current_stop
                        should_trail = additional_move >= safe_trail_distance_price
                        self.logger.debug(f"[TRAIL CHECK PROTECTED] BUY: current_price={current_price:.5f}, "
                                        f"protected_stop={current_stop:.5f}, additional_move={additional_move:.5f}, "
                                        f"required={safe_trail_distance_price:.5f}, should_trail={should_trail}")
                    else:  # SELL
                        # For SELL: additional move = how much price moved below current stop + safe distance
                        additional_move = current_stop - current_price
                        should_trail = additional_move >= safe_trail_distance_price
                        self.logger.debug(f"[TRAIL CHECK PROTECTED] SELL: current_price={current_price:.5f}, "
                                        f"protected_stop={current_stop:.5f}, additional_move={additional_move:.5f}, "
                                        f"required={safe_trail_distance_price:.5f}, should_trail={should_trail}")
                else:
                    # Original logic for non-protected trades
                    if trade.direction.upper() == "BUY":
                        additional_move = current_price - trade.entry_price - break_even_trigger
                        should_trail = additional_move >= safe_trail_distance_price
                        self.logger.debug(f"[TRAIL CHECK] BUY additional_move={additional_move:.5f}, "
                                        f"required={safe_trail_distance_price:.5f}, should_trail={should_trail}")
                    else:
                        additional_move = trade.entry_price - current_price - break_even_trigger
                        should_trail = additional_move >= safe_trail_distance_price
                        self.logger.debug(f"[TRAIL CHECK] SELL additional_move={additional_move:.5f}, "
                                        f"required={safe_trail_distance_price:.5f}, should_trail={should_trail}")
                
                if should_trail:
                    self.logger.info(f"🎯 [TRAILING TRIGGER] Trade {trade.id}: Additional movement sufficient for trailing")
                    
                    # Apply the selected trailing strategy using the unified trailing manager
                    try:
                        trailing_manager = AdvancedTrailingManager(self.config, self.logger)

                        # Calculate new trail level using the configured method
                        new_trail_level = trailing_manager.calculate_new_trail_level(trade, current_price, db)

                        if new_trail_level:
                            # Calculate adjustment points for the API
                            adjustment_points = trailing_manager.get_trail_adjustment_points(trade, current_price, new_trail_level)

                            # Determine direction for adjustment
                            direction_stop = "increase" if trade.direction.upper() == "BUY" else "decrease"

                            # Send the adjustment
                            api_result = self._send_stop_adjustment(trade, adjustment_points, direction_stop, 0)

                            if isinstance(api_result, dict) and api_result.get("status") == "updated":
                                # Extract IG's actual stop level from the API response
                                sent_payload = api_result.get("sentPayload", {})
                                ig_actual_stop = sent_payload.get("stopLevel")

                                old_sl = trade.sl_price
                                old_status = trade.status
                                if ig_actual_stop:
                                    # Use IG's actual stop level instead of our calculation
                                    trade.sl_price = float(ig_actual_stop)
                                    self.logger.info(f"📍 [IG ACTUAL] Trade {trade.id}: IG set stop to {ig_actual_stop:.5f} (calculated {new_trail_level:.5f})")
                                else:
                                    # Fallback to our calculation if IG's response is missing
                                    trade.sl_price = new_trail_level
                                    self.logger.warning(f"⚠️ [FALLBACK] Trade {trade.id}: Using calculated stop {new_trail_level:.5f}")

                                trade.last_trigger_price = current_price
                                trade.trigger_time = datetime.utcnow()
                                trade.status = "trailing"
                                trade.stop_limit_changes_count = (trade.stop_limit_changes_count or 0) + 1

                                self.logger.info(f"💾 [DB PREPARE TRAILING] Trade {trade.id}: sl_price {old_sl} → {trade.sl_price}, status {old_status} → {trade.status}, changes: {trade.stop_limit_changes_count}")
                                try:
                                    db.commit()
                                    self.logger.info(f"✅ [DB COMMIT SUCCESS] Trade {trade.id}: Trailing changes persisted")
                                    self.logger.info(f"🎯 [TRAILING SUCCESS] Trade {trade.id} {trade.symbol}: "
                                                   f"Stop moved to {trade.sl_price:.5f} ({adjustment_points} pts)")
                                    return True
                                except Exception as commit_error:
                                    db.rollback()
                                    self.logger.error(f"❌ [DB COMMIT FAILED TRAILING] Trade {trade.id}: {type(commit_error).__name__}: {str(commit_error)}")
                                    raise
                            else:
                                error_msg = api_result.get("message", "Unknown error") if isinstance(api_result, dict) else "API call failed"
                                self.logger.error(f"❌ [TRAILING FAILED] Trade {trade.id}: {error_msg}")
                                return False
                        else:
                            self.logger.debug(f"[NO TRAIL] Trade {trade.id}: No trail level calculated")
                            return True  # Continue monitoring

                    except Exception as e:
                        self.logger.error(f"❌ [TRAILING ERROR] Trade {trade.id}: {e}")
                        return False
                else:
                    self.logger.debug(f"[NO TRAIL] Trade {trade.id}: Insufficient additional movement for trailing")
                    return True  # No action needed, but continue monitoring
            else:
                self.logger.debug(f"[NOT READY] Trade {trade.id}: Not ready for trailing (moved_to_breakeven={getattr(trade, 'moved_to_breakeven', False)}, status={trade.status})")
                return True  # Continue monitoring
                
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED PROCESSING ERROR] Trade {trade.id}: {e}")
            return False

    
    def _send_stop_adjustment(self, trade: TradeLog, stop_points: int,
                            direction_stop: str, limit_points: int,
                            new_stop_level: float = None):
        """
        Send stop adjustment to order system - ENHANCED WITH ABSOLUTE LEVEL SUPPORT

        Args:
            trade: The trade to adjust
            stop_points: Points to adjust (for legacy offset mode)
            direction_stop: 'increase' or 'decrease'
            limit_points: Points to adjust limit (for legacy offset mode)
            new_stop_level: PREFERRED - Absolute stop level to set directly
        """
        try:
            from services.adjust_stop_service import adjust_stop_sync

            # ✅ CRITICAL FIX: Use absolute stop level when provided
            if new_stop_level is not None:
                self.logger.info(f"📍 [ABSOLUTE STOP] Trade {trade.id} {trade.symbol}: Setting stop directly to {new_stop_level:.5f}")

                result = adjust_stop_sync(
                    epic=trade.symbol,
                    new_stop=new_stop_level,  # Use absolute level - no offset calculation needed
                    stop_offset_points=None,
                    limit_offset_points=None,
                    dry_run=False
                )
            else:
                # Legacy offset-based mode (fallback)
                self.logger.info(f"[OFFSET MODE] Trade {trade.id} {trade.symbol}: Adjusting stop by {stop_points}pts ({direction_stop})")

                result = adjust_stop_sync(
                    epic=trade.symbol,
                    stop_offset_points=stop_points,
                    limit_offset_points=limit_points,
                    adjust_direction_stop=direction_stop,
                    adjust_direction_limit="increase",
                    dry_run=False
                )

            status = result.get("status", "unknown")

            if status == "updated":
                sent_payload = result.get("sentPayload", {})
                actual_stop = sent_payload.get("stopLevel")
                self.logger.info(f"[✅ STOP UPDATED] {trade.symbol} → IG set stopLevel={actual_stop}")
                return result
            elif status == "closed":
                self.logger.warning(f"[❌ POSITION CLOSED] {trade.symbol}: {result.get('message')}")
                return result
            else:
                self.logger.error(f"[❌ ADJUSTMENT FAILED] {trade.symbol}: {result.get('message', 'Unknown error')}")
                return result

        except Exception as e:
            self.logger.error(f"❌ [SEND ERROR] Trade {trade.id}: {e}")
            return {"status": "error", "message": str(e)}


# Configuration examples for different market conditions - ENHANCED
SCALPING_CONFIG = TrailingConfig(
    method=TrailingMethod.PROGRESSIVE_3_STAGE,  # UPDATED: Use new progressive system
    break_even_trigger_points=3,  # ✅ ENHANCED: move to break-even after +3 points
    min_trail_distance=2,
    max_trail_distance=10,
    # Progressive settings
    stage1_trigger_points=3,
    stage1_lock_points=1,
    stage2_trigger_points=5,
    stage2_lock_points=3,
    stage3_trigger_points=8,
    stage3_atr_multiplier=1.5,
    stage3_min_distance=2
)

SWING_TRADING_CONFIG = TrailingConfig(
    method=TrailingMethod.ATR_BASED,
    initial_trigger_points=15,
    break_even_trigger_points=10,  # ✅ ENHANCED
    atr_multiplier=2.5,
    min_trail_distance=10,
    max_trail_distance=100
)

TREND_FOLLOWING_CONFIG = TrailingConfig(
    method=TrailingMethod.CHANDELIER,
    initial_trigger_points=20,
    break_even_trigger_points=15,  # ✅ ENHANCED
    chandelier_period=14,
    chandelier_multiplier=3.0,
    min_trail_distance=15
)

ADAPTIVE_CONFIG = TrailingConfig(
    method=TrailingMethod.PROGRESSIVE_3_STAGE,  # UPDATED: Use progressive system
    break_even_trigger_points=3,  # ✅ ENHANCED: More aggressive
    volatility_threshold=2.0,
    trend_strength_min=0.7,
    min_trail_distance=8,
    max_trail_distance=50,
    # Progressive settings for adaptive trading
    stage1_trigger_points=2,  # Even more aggressive for adaptive
    stage1_lock_points=1,
    stage2_trigger_points=4,
    stage2_lock_points=2,
    stage3_trigger_points=6,
    stage3_atr_multiplier=2.0,  # Wider ATR for trend following
    stage3_min_distance=3
)