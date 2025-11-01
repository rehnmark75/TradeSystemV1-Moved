# core/strategies/helpers/smc_market_structure.py
"""
Smart Money Concepts - Market Structure Analysis
Detects Break of Structure (BOS), Change of Character (ChoCH), and Swing Points

Based on institutional trading concepts:
- Higher Highs (HH) / Higher Lows (HL) = Bullish Structure
- Lower Highs (LH) / Lower Lows (LL) = Bearish Structure
- BOS = Break in direction of trend (continuation)
- ChoCH = Break against trend (reversal)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum


class StructureType(Enum):
    """Types of market structure"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SwingType(Enum):
    """Types of swing points"""
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    EQUAL_HIGH = "EQH"
    EQUAL_LOW = "EQL"


@dataclass
class SwingPoint:
    """Represents a swing high or low point"""
    index: int
    price: float
    swing_type: SwingType
    timestamp: pd.Timestamp
    confirmed: bool = False
    strength: float = 0.0


@dataclass
class StructureBreak:
    """Represents a Break of Structure or Change of Character"""
    index: int
    break_type: str  # 'BOS' or 'ChoCH'
    direction: str   # 'bullish' or 'bearish'
    previous_structure: StructureType
    new_structure: StructureType
    break_price: float
    significance: float
    timestamp: pd.Timestamp


class SMCMarketStructure:
    """Smart Money Concepts Market Structure Analyzer"""
    
    def __init__(self, logger: logging.Logger = None, data_fetcher=None):
        self.logger = logger or logging.getLogger(__name__)
        self.swing_points: List[SwingPoint] = []
        self.structure_breaks: List[StructureBreak] = []
        self.current_structure = StructureType.NEUTRAL
        self.data_fetcher = data_fetcher  # For multi-timeframe analysis
        
    def analyze_market_structure(
        self, 
        df: pd.DataFrame, 
        config: Dict,
        epic: str = None,
        timeframe: str = None
    ) -> pd.DataFrame:
        """
        Analyze market structure and add SMC indicators to DataFrame
        
        Args:
            df: OHLCV DataFrame
            config: SMC configuration dictionary
            
        Returns:
            Enhanced DataFrame with structure analysis
        """
        try:
            # CRITICAL FIX: Reset instance state to prevent contamination across scans
            # Without this, bearish/bullish bias from previous analyses propagates to new scans
            # Issue: Shared instance was maintaining state across multiple time periods in backtests
            # Result: First bearish detection would contaminate ALL subsequent analyses
            self.swing_points = []
            self.structure_breaks = []
            self.current_structure = StructureType.NEUTRAL

            df_enhanced = df.copy()
            
            # Detect swing points
            df_enhanced = self._detect_swing_points(df_enhanced, config)
            
            # Analyze structure breaks with epic and timeframe for MTF validation
            df_enhanced = self._analyze_structure_breaks(df_enhanced, config, epic, timeframe)
            
            # Add structure signals
            df_enhanced = self._add_structure_signals(df_enhanced, config)
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Market structure analysis failed: {e}")
            return df
    
    def _detect_swing_points(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect swing highs and lows using pivot analysis"""
        try:
            swing_length = config.get('swing_length', 5)
            
            # Calculate pivot highs and lows
            df['pivot_high'] = self._calculate_pivot_high(df['high'], swing_length)
            df['pivot_low'] = self._calculate_pivot_low(df['low'], swing_length)
            
            # Initialize swing point columns
            df['swing_high'] = False
            df['swing_low'] = False
            df['swing_type'] = ''
            df['swing_strength'] = 0.0
            
            # Track swing points for structure analysis
            swing_highs = []
            swing_lows = []
            
            for i in range(len(df)):
                if not pd.isna(df.iloc[i]['pivot_high']):
                    # Found swing high
                    price = df.iloc[i]['pivot_high']
                    timestamp = df.index[i] if hasattr(df.index, 'to_pydatetime') else pd.Timestamp.now()
                    
                    # Classify swing high
                    swing_type = self._classify_swing_high(price, swing_highs)
                    strength = self._calculate_swing_strength(df, i, 'high', swing_length)
                    
                    swing_point = SwingPoint(
                        index=i,
                        price=price,
                        swing_type=swing_type,
                        timestamp=timestamp,
                        strength=strength
                    )
                    
                    swing_highs.append(swing_point)
                    df.iloc[i, df.columns.get_loc('swing_high')] = True
                    df.iloc[i, df.columns.get_loc('swing_type')] = swing_type.value
                    df.iloc[i, df.columns.get_loc('swing_strength')] = strength
                
                if not pd.isna(df.iloc[i]['pivot_low']):
                    # Found swing low
                    price = df.iloc[i]['pivot_low']
                    timestamp = df.index[i] if hasattr(df.index, 'to_pydatetime') else pd.Timestamp.now()
                    
                    # Classify swing low
                    swing_type = self._classify_swing_low(price, swing_lows)
                    strength = self._calculate_swing_strength(df, i, 'low', swing_length)
                    
                    swing_point = SwingPoint(
                        index=i,
                        price=price,
                        swing_type=swing_type,
                        timestamp=timestamp,
                        strength=strength
                    )
                    
                    swing_lows.append(swing_point)
                    df.iloc[i, df.columns.get_loc('swing_low')] = True
                    df.iloc[i, df.columns.get_loc('swing_type')] = swing_type.value
                    df.iloc[i, df.columns.get_loc('swing_strength')] = strength
            
            # Store swing points for structure analysis
            self.swing_points = swing_highs + swing_lows
            self.swing_points.sort(key=lambda x: x.index)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Swing point detection failed: {e}")
            return df
    
    def _calculate_pivot_high(self, high_series: pd.Series, length: int, allow_progressive: bool = True) -> pd.Series:
        """
        Calculate pivot highs using rolling window with progressive confirmation

        Args:
            high_series: High price series
            length: Number of bars for confirmation
            allow_progressive: If True, allow recent swings with partial confirmation
        """
        try:
            pivot_highs = pd.Series(index=high_series.index, dtype=float)

            # Fully confirmed pivots (original logic)
            for i in range(length, len(high_series) - length):
                center_price = high_series.iloc[i]
                left_prices = high_series.iloc[i-length:i]
                right_prices = high_series.iloc[i+1:i+length+1]

                # Check if center is highest
                if (center_price > left_prices.max()) and (center_price > right_prices.max()):
                    pivot_highs.iloc[i] = center_price

            # Progressive confirmation for recent bars (if enabled)
            if allow_progressive:
                min_confirmation = 2  # Minimum 2 bars for progressive confirmation
                for i in range(max(length, len(high_series) - length), len(high_series)):
                    bars_available = len(high_series) - i - 1
                    if bars_available >= min_confirmation:
                        center_price = high_series.iloc[i]
                        left_prices = high_series.iloc[i-length:i]
                        right_prices = high_series.iloc[i+1:i+1+bars_available]

                        # Check if center is highest with available confirmation
                        if (center_price > left_prices.max()) and (center_price > right_prices.max()):
                            pivot_highs.iloc[i] = center_price

            return pivot_highs

        except Exception as e:
            self.logger.error(f"Pivot high calculation failed: {e}")
            return pd.Series(index=high_series.index, dtype=float)
    
    def _calculate_pivot_low(self, low_series: pd.Series, length: int, allow_progressive: bool = True) -> pd.Series:
        """
        Calculate pivot lows using rolling window with progressive confirmation

        Args:
            low_series: Low price series
            length: Number of bars for confirmation
            allow_progressive: If True, allow recent swings with partial confirmation
        """
        try:
            pivot_lows = pd.Series(index=low_series.index, dtype=float)

            # Fully confirmed pivots (original logic)
            for i in range(length, len(low_series) - length):
                center_price = low_series.iloc[i]
                left_prices = low_series.iloc[i-length:i]
                right_prices = low_series.iloc[i+1:i+length+1]

                # Check if center is lowest
                if (center_price < left_prices.min()) and (center_price < right_prices.min()):
                    pivot_lows.iloc[i] = center_price

            # Progressive confirmation for recent bars (if enabled)
            if allow_progressive:
                min_confirmation = 2  # Minimum 2 bars for progressive confirmation
                for i in range(max(length, len(low_series) - length), len(low_series)):
                    bars_available = len(low_series) - i - 1
                    if bars_available >= min_confirmation:
                        center_price = low_series.iloc[i]
                        left_prices = low_series.iloc[i-length:i]
                        right_prices = low_series.iloc[i+1:i+1+bars_available]

                        # Check if center is lowest with available confirmation
                        if (center_price < left_prices.min()) and (center_price < right_prices.min()):
                            pivot_lows.iloc[i] = center_price

            return pivot_lows

        except Exception as e:
            self.logger.error(f"Pivot low calculation failed: {e}")
            return pd.Series(index=low_series.index, dtype=float)
    
    def _classify_swing_high(self, current_high: float, previous_highs: List[SwingPoint]) -> SwingType:
        """Classify swing high as HH, LH, or EQH"""
        if not previous_highs:
            return SwingType.HIGHER_HIGH
        
        last_high = previous_highs[-1].price
        tolerance = 0.00005  # 0.5 pip tolerance for equal levels
        
        if current_high > last_high + tolerance:
            return SwingType.HIGHER_HIGH
        elif current_high < last_high - tolerance:
            return SwingType.LOWER_HIGH
        else:
            return SwingType.EQUAL_HIGH
    
    def _classify_swing_low(self, current_low: float, previous_lows: List[SwingPoint]) -> SwingType:
        """Classify swing low as HL, LL, or EQL"""
        if not previous_lows:
            return SwingType.HIGHER_LOW
        
        last_low = previous_lows[-1].price
        tolerance = 0.00005  # 0.5 pip tolerance for equal levels
        
        if current_low > last_low + tolerance:
            return SwingType.HIGHER_LOW
        elif current_low < last_low - tolerance:
            return SwingType.LOWER_LOW
        else:
            return SwingType.EQUAL_LOW
    
    def _calculate_swing_strength(
        self, 
        df: pd.DataFrame, 
        index: int, 
        price_type: str, 
        length: int
    ) -> float:
        """Calculate the strength of a swing point based on volume and price action"""
        try:
            if price_type == 'high':
                price = df.iloc[index]['high']
                # Look for volume confirmation and price extension
                volume = df.iloc[index].get('volume', df.iloc[index].get('ltv', 1))
            else:
                price = df.iloc[index]['low']
                volume = df.iloc[index].get('volume', df.iloc[index].get('ltv', 1))
            
            # Calculate average volume in surrounding bars
            start_idx = max(0, index - length)
            end_idx = min(len(df), index + length + 1)
            
            volume_data = []
            for i in range(start_idx, end_idx):
                vol = df.iloc[i].get('volume', df.iloc[i].get('ltv', 1))
                if vol and vol > 0:
                    volume_data.append(vol)
            
            if volume_data:
                avg_volume = sum(volume_data) / len(volume_data)
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # Calculate price extension strength
            if price_type == 'high':
                surrounding_highs = [df.iloc[i]['high'] for i in range(start_idx, end_idx) if i != index]
                if surrounding_highs:
                    max_surrounding = max(surrounding_highs)
                    price_extension = (price - max_surrounding) / max_surrounding if max_surrounding > 0 else 0
                else:
                    price_extension = 0
            else:
                surrounding_lows = [df.iloc[i]['low'] for i in range(start_idx, end_idx) if i != index]
                if surrounding_lows:
                    min_surrounding = min(surrounding_lows)
                    price_extension = (min_surrounding - price) / min_surrounding if min_surrounding > 0 else 0
                else:
                    price_extension = 0
            
            # Combine volume and price factors
            strength = (volume_ratio * 0.6) + (price_extension * 1000 * 0.4)  # Scale price extension
            return min(max(strength, 0.1), 5.0)  # Clamp between 0.1 and 5.0
            
        except Exception as e:
            self.logger.error(f"Swing strength calculation failed: {e}")
            return 1.0
    
    def _analyze_structure_breaks(self, df: pd.DataFrame, config: Dict, epic: str = None, timeframe: str = None) -> pd.DataFrame:
        """Analyze breaks of structure and changes of character"""
        try:
            confirmation_bars = config.get('structure_confirmation', 3)
            bos_threshold = config.get('bos_threshold', 0.0001)
            min_structure_significance = config.get('min_structure_significance', 0.3)

            # Initialize structure columns
            df['structure_break'] = False
            df['break_type'] = ''
            df['break_direction'] = ''
            df['structure_significance'] = 0.0

            if len(self.swing_points) < 2:
                return df

            # Analyze each swing point for structure breaks
            for i, swing_point in enumerate(self.swing_points[1:], 1):
                previous_swing = self.swing_points[i-1]

                # Determine if this creates a structure break
                break_info = self._detect_structure_break(
                    previous_swing,
                    swing_point,
                    df,
                    bos_threshold,
                    confirmation_bars,
                    epic,
                    timeframe,
                    min_structure_significance
                )
                
                if break_info:
                    # Mark structure break in DataFrame
                    break_index = swing_point.index
                    if break_index < len(df):
                        df.iloc[break_index, df.columns.get_loc('structure_break')] = True
                        df.iloc[break_index, df.columns.get_loc('break_type')] = break_info.break_type
                        df.iloc[break_index, df.columns.get_loc('break_direction')] = break_info.direction
                        df.iloc[break_index, df.columns.get_loc('structure_significance')] = break_info.significance
                    
                    # Store structure break
                    self.structure_breaks.append(break_info)
                    
                    # Update current structure
                    self.current_structure = break_info.new_structure
            
            return df
            
        except Exception as e:
            self.logger.error(f"Structure break analysis failed: {e}")
            return df
    
    def _detect_structure_break(
        self,
        previous_swing: SwingPoint,
        current_swing: SwingPoint,
        df: pd.DataFrame,
        threshold: float,
        confirmation_bars: int,
        epic: str = None,
        timeframe: str = None,
        min_structure_significance: float = 0.3
    ) -> Optional[StructureBreak]:
        """Detect if current swing creates a structure break - SIMPLIFIED VERSION"""
        try:
            # SIMPLIFIED: Focus on basic BOS/CHoCH detection without overly strict filters
            break_type = None
            direction = None
            new_structure = None

            # Determine break type based on swing patterns
            if (previous_swing.swing_type in [SwingType.HIGHER_HIGH, SwingType.EQUAL_HIGH] and
                current_swing.swing_type == SwingType.LOWER_HIGH and
                self.current_structure in [StructureType.BULLISH, StructureType.NEUTRAL]):
                # Bearish ChoCH (Change of Character) - Trend reversal
                break_type = "ChoCH"
                direction = "bearish"
                new_structure = StructureType.BEARISH

            elif (previous_swing.swing_type in [SwingType.LOWER_LOW, SwingType.EQUAL_LOW] and
                  current_swing.swing_type == SwingType.HIGHER_LOW and
                  self.current_structure in [StructureType.BEARISH, StructureType.NEUTRAL]):
                # Bullish ChoCH - Trend reversal
                break_type = "ChoCH"
                direction = "bullish"
                new_structure = StructureType.BULLISH

            elif (current_swing.swing_type == SwingType.HIGHER_HIGH and
                  self.current_structure == StructureType.BULLISH):
                # Bullish BOS (Break of Structure) - Trend continuation
                break_type = "BOS"
                direction = "bullish"
                new_structure = StructureType.BULLISH

            elif (current_swing.swing_type == SwingType.LOWER_LOW and
                  self.current_structure == StructureType.BEARISH):
                # Bearish BOS - Trend continuation
                break_type = "BOS"
                direction = "bearish"
                new_structure = StructureType.BEARISH

            # SIMPLIFIED: Also detect from NEUTRAL state
            elif (current_swing.swing_type == SwingType.HIGHER_HIGH and
                  self.current_structure == StructureType.NEUTRAL):
                # First bullish break from neutral
                break_type = "BOS"
                direction = "bullish"
                new_structure = StructureType.BULLISH

            elif (current_swing.swing_type == SwingType.LOWER_LOW and
                  self.current_structure == StructureType.NEUTRAL):
                # First bearish break from neutral
                break_type = "BOS"
                direction = "bearish"
                new_structure = StructureType.BEARISH

            if not break_type:
                return None

            # Basic price movement validation (very lenient threshold)
            price_difference = abs(current_swing.price - previous_swing.price)
            if price_difference < threshold:
                return None

            # Calculate basic significance
            significance = self._calculate_break_significance(
                previous_swing, current_swing, df
            )

            # Apply minimum significance from config (now properly passed as parameter)
            if significance < min_structure_significance:
                return None

            return StructureBreak(
                index=current_swing.index,
                break_type=break_type,
                direction=direction,
                previous_structure=self.current_structure,
                new_structure=new_structure,
                break_price=current_swing.price,
                significance=significance,
                timestamp=current_swing.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced structure break detection failed: {e}")
            return None
    
    def _calculate_break_significance(
        self, 
        previous_swing: SwingPoint, 
        current_swing: SwingPoint,
        df: pd.DataFrame
    ) -> float:
        """Calculate the significance of a structure break"""
        try:
            # Price movement factor
            price_move = abs(current_swing.price - previous_swing.price)
            avg_range = self._get_average_range(df, current_swing.index, 20)
            price_factor = price_move / avg_range if avg_range > 0 else 1.0
            
            # Time factor (breaks over longer periods are more significant)
            time_factor = min((current_swing.index - previous_swing.index) / 20, 2.0)
            
            # Swing strength factor
            strength_factor = (current_swing.strength + previous_swing.strength) / 2
            
            # Volume confirmation
            volume_factor = self._get_volume_confirmation(df, current_swing.index)
            
            # Combine factors
            significance = (price_factor * 0.4 + 
                          time_factor * 0.2 + 
                          strength_factor * 0.2 + 
                          volume_factor * 0.2)
            
            return min(max(significance, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Break significance calculation failed: {e}")
            return 0.5
    
    def _get_average_range(self, df: pd.DataFrame, index: int, periods: int) -> float:
        """Calculate average true range for recent periods"""
        try:
            start_idx = max(0, index - periods)
            ranges = []
            
            for i in range(start_idx, min(index + 1, len(df))):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low']
                ranges.append(high - low)
            
            return sum(ranges) / len(ranges) if ranges else 0.0001
            
        except Exception as e:
            self.logger.error(f"Average range calculation failed: {e}")
            return 0.0001
    
    def _get_volume_confirmation(self, df: pd.DataFrame, index: int) -> float:
        """Get volume confirmation for structure break"""
        try:
            if index >= len(df):
                return 0.5
            
            current_volume = df.iloc[index].get('volume', df.iloc[index].get('ltv', 1))
            if not current_volume or current_volume <= 0:
                return 0.5
            
            # Compare to recent average volume
            lookback = min(20, index)
            if lookback <= 0:
                return 0.5
            
            recent_volumes = []
            for i in range(max(0, index - lookback), index):
                vol = df.iloc[i].get('volume', df.iloc[i].get('ltv', 1))
                if vol and vol > 0:
                    recent_volumes.append(vol)
            
            if not recent_volumes:
                return 0.5
            
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            volume_ratio = current_volume / avg_volume
            
            # Convert to 0-1 scale
            return min(volume_ratio / 2.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Volume confirmation calculation failed: {e}")
            return 0.5
    
    def _add_structure_signals(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Add trading signals based on structure analysis"""
        try:
            # Initialize signal columns
            df['smc_structure_signal'] = ''
            df['smc_signal_strength'] = 0.0
            df['smc_entry_reason'] = ''
            
            confluence_required = config.get('confluence_required', 2)
            
            for i, row in df.iterrows():
                if row['structure_break']:
                    # Found structure break - generate signal
                    break_type = row['break_type']
                    break_direction = row['break_direction']
                    significance = row['structure_significance']
                    
                    # Check confluence factors
                    confluence_score = self._calculate_confluence(df, i, break_direction, config)
                    
                    if confluence_score >= confluence_required:
                        signal_type = 'BULL' if break_direction == 'bullish' else 'BEAR'
                        signal_strength = min(significance * confluence_score / confluence_required, 1.0)
                        
                        df.at[i, 'smc_structure_signal'] = signal_type
                        df.at[i, 'smc_signal_strength'] = signal_strength
                        df.at[i, 'smc_entry_reason'] = f"{break_type}_{break_direction}_confluence_{confluence_score}"
            
            return df
            
        except Exception as e:
            self.logger.error(f"Structure signals generation failed: {e}")
            return df
    
    def _calculate_confluence(
        self, 
        df: pd.DataFrame, 
        index: int, 
        direction: str, 
        config: Dict
    ) -> float:
        """Calculate confluence score for structure break"""
        try:
            confluence_score = 1.0  # Base score for structure break itself
            
            # Check for multiple swing confirmations
            if self._has_multiple_swing_confirmation(index, direction):
                confluence_score += 0.5
            
            # Check for volume confirmation
            volume_conf = self._get_volume_confirmation(df, index)
            if volume_conf > 0.7:
                confluence_score += 0.3
            
            # Check for equal highs/lows (liquidity)
            if self._has_liquidity_nearby(index, direction):
                confluence_score += 0.4
            
            # Check for recent structure alignment
            if self._has_structure_alignment(direction):
                confluence_score += 0.3
            
            return confluence_score
            
        except Exception as e:
            self.logger.error(f"Confluence calculation failed: {e}")
            return 1.0
    
    def _has_multiple_swing_confirmation(self, index: int, direction: str) -> bool:
        """Check if multiple swings confirm the direction"""
        try:
            recent_swings = [sp for sp in self.swing_points if sp.index <= index and sp.index > index - 20]
            
            if direction == 'bullish':
                return any(sp.swing_type in [SwingType.HIGHER_HIGH, SwingType.HIGHER_LOW] 
                          for sp in recent_swings[-3:])
            else:
                return any(sp.swing_type in [SwingType.LOWER_HIGH, SwingType.LOWER_LOW] 
                          for sp in recent_swings[-3:])
                          
        except Exception:
            return False
    
    def _has_liquidity_nearby(self, index: int, direction: str) -> bool:
        """Check for equal highs/lows (liquidity) nearby"""
        try:
            nearby_swings = [sp for sp in self.swing_points 
                           if abs(sp.index - index) <= 10 and sp.index != index]
            
            return any(sp.swing_type in [SwingType.EQUAL_HIGH, SwingType.EQUAL_LOW] 
                      for sp in nearby_swings)
                      
        except Exception:
            return False
    
    def _has_structure_alignment(self, direction: str) -> bool:
        """Check if recent structure breaks align with current direction"""
        try:
            recent_breaks = [sb for sb in self.structure_breaks[-3:]]
            
            if not recent_breaks:
                return False
            
            aligned_breaks = [sb for sb in recent_breaks if sb.direction == direction]
            return len(aligned_breaks) >= len(recent_breaks) / 2
            
        except Exception:
            return False
    
    def get_current_structure(self) -> Dict:
        """Get current market structure state"""
        return {
            'structure_type': self.current_structure.value,
            'recent_breaks': len(self.structure_breaks[-5:]),
            'last_break_type': self.structure_breaks[-1].break_type if self.structure_breaks else None,
            'last_break_direction': self.structure_breaks[-1].direction if self.structure_breaks else None,
            'swing_points_count': len(self.swing_points),
            'strong_swings': len([sp for sp in self.swing_points if sp.strength > 2.0])
        }
    
    def get_structure_levels(self, direction: str = None) -> List[Dict]:
        """Get key structure levels for entry/exit planning"""
        try:
            levels = []
            
            # Get recent significant swing points
            significant_swings = [sp for sp in self.swing_points if sp.strength > 1.5]
            
            for swing in significant_swings[-10:]:  # Last 10 significant swings
                level_type = "resistance" if swing.swing_type in [SwingType.HIGHER_HIGH, SwingType.LOWER_HIGH] else "support"
                
                levels.append({
                    'price': swing.price,
                    'type': level_type,
                    'strength': swing.strength,
                    'age': len(self.swing_points) - swing.index if swing in self.swing_points else 0,
                    'swing_type': swing.swing_type.value
                })
            
            # Sort by strength and recency
            levels.sort(key=lambda x: (x['strength'], -x['age']), reverse=True)
            
            return levels[:5]  # Return top 5 levels
            
        except Exception as e:
            self.logger.error(f"Structure levels calculation failed: {e}")
            return []
    
    def _validate_liquidity_sweep(self, df: pd.DataFrame, current_swing: SwingPoint, level_type: str) -> bool:
        """Validate liquidity sweep with institutional context"""
        try:
            # Look for volume spike during liquidity sweep
            current_index = current_swing.index
            if current_index >= len(df) or current_index < 5:
                return False
            
            # Get volume data around the sweep
            sweep_volume = df.iloc[current_index].get('volume', df.iloc[current_index].get('ltv', 1))
            if not sweep_volume or sweep_volume <= 0:
                return False
            
            # Calculate recent average volume
            recent_volumes = []
            for i in range(max(0, current_index - 10), current_index):
                vol = df.iloc[i].get('volume', df.iloc[i].get('ltv', 1))
                if vol and vol > 0:
                    recent_volumes.append(vol)
            
            if not recent_volumes:
                return False
            
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            volume_ratio = sweep_volume / avg_volume
            
            # Require significant volume increase for liquidity sweep
            min_liquidity_volume = getattr(self, 'min_liquidity_volume', 1.3)
            if volume_ratio < min_liquidity_volume:
                return False
            
            # Check for immediate reversal after sweep (institutional behavior)
            if current_index + 3 < len(df):
                sweep_price = current_swing.price
                
                if level_type == "high":
                    # After sweeping highs, expect immediate reversal down
                    next_prices = [df.iloc[i]['low'] for i in range(current_index + 1, min(current_index + 4, len(df)))]
                    if next_prices and min(next_prices) < sweep_price * 0.9995:  # 0.5 pip reversal
                        return True
                        
                elif level_type == "low":
                    # After sweeping lows, expect immediate reversal up
                    next_prices = [df.iloc[i]['high'] for i in range(current_index + 1, min(current_index + 4, len(df)))]
                    if next_prices and max(next_prices) > sweep_price * 1.0005:  # 0.5 pip reversal
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Liquidity sweep validation failed: {e}")
            return False
    
    def _validate_institutional_context(
        self, 
        df: pd.DataFrame, 
        current_swing: SwingPoint, 
        previous_swing: SwingPoint, 
        direction: str
    ) -> bool:
        """Validate institutional context for structure break"""
        try:
            current_index = current_swing.index
            
            # Volume profile validation
            if not self._validate_volume_profile(df, current_index, direction):
                return False
            
            # Time-based validation (avoid low liquidity periods)
            if not self._validate_trading_session(current_swing.timestamp):
                return False
            
            # Price action validation (avoid spiky/erratic moves)
            if not self._validate_price_action_quality(df, current_index):
                return False
            
            # Swing strength validation
            min_swing_strength = getattr(self, 'min_swing_strength', 1.5)
            if current_swing.strength < min_swing_strength:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Institutional context validation failed: {e}")
            return False
    
    def _validate_volume_profile(self, df: pd.DataFrame, index: int, direction: str) -> bool:
        """Validate volume profile for institutional activity"""
        try:
            if index < 5 or index >= len(df):
                return False
            
            # Get volume data around structure break
            volume_window = []
            for i in range(max(0, index - 3), min(index + 2, len(df))):
                vol = df.iloc[i].get('volume', df.iloc[i].get('ltv', 1))
                if vol and vol > 0:
                    volume_window.append(vol)
            
            if len(volume_window) < 3:
                return False
            
            # Calculate baseline volume (20-period average)
            baseline_volumes = []
            for i in range(max(0, index - 25), max(0, index - 5)):
                vol = df.iloc[i].get('volume', df.iloc[i].get('ltv', 1))
                if vol and vol > 0:
                    baseline_volumes.append(vol)
            
            if not baseline_volumes:
                return False
            
            baseline_avg = sum(baseline_volumes) / len(baseline_volumes)
            break_avg_volume = sum(volume_window) / len(volume_window)
            
            # Require institutional volume (1.5x baseline minimum)
            volume_factor = break_avg_volume / baseline_avg
            return volume_factor >= 1.5
            
        except Exception:
            return False
    
    def _validate_trading_session(self, timestamp: pd.Timestamp) -> bool:
        """Validate that structure break occurs during institutional trading hours"""
        try:
            # Convert to UTC hour for session analysis
            if hasattr(timestamp, 'hour'):
                utc_hour = timestamp.hour
            else:
                return True  # Skip validation if timestamp format unknown
            
            # London session: 08:00-16:00 UTC
            # New York session: 13:00-22:00 UTC  
            # London/NY overlap: 13:00-16:00 UTC (premium time)
            
            london_session = 8 <= utc_hour <= 16
            ny_session = 13 <= utc_hour <= 22
            overlap_session = 13 <= utc_hour <= 16
            
            # Prefer overlap, then major sessions
            if overlap_session:
                return True  # Best time for institutional activity
            elif london_session or ny_session:
                return True  # Good institutional activity
            else:
                # Asian session or off-hours - avoid unless very strong signal
                return False
                
        except Exception:
            return True  # Default to allowing if session check fails
    
    def _validate_price_action_quality(self, df: pd.DataFrame, index: int) -> bool:
        """Validate price action quality (avoid spiky/erratic moves)"""
        try:
            if index < 3 or index + 3 >= len(df):
                return True
            
            # Get price data around the break
            price_window = []
            for i in range(index - 2, index + 3):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low']
                range_val = high - low
                price_window.append(range_val)
            
            if not price_window:
                return True
            
            # Check for excessive volatility (spike detection)
            avg_range = sum(price_window) / len(price_window)
            max_range = max(price_window)
            
            # Reject if any single bar is more than 3x average range (spike)
            if max_range > avg_range * 3:
                return False
            
            return True
            
        except Exception:
            return True
    
    def _calculate_enhanced_break_significance(
        self,
        previous_swing: SwingPoint,
        current_swing: SwingPoint,
        df: pd.DataFrame,
        break_type: str,
        recent_swings: List[SwingPoint]
    ) -> float:
        """Calculate enhanced significance with institutional factors"""
        try:
            base_significance = self._calculate_break_significance(previous_swing, current_swing, df)
            
            # Enhancement factors
            enhancement_factors = []
            
            # 1. Liquidity sweep bonus
            if "LiquiditySweep" in break_type:
                enhancement_factors.append(0.2)
            
            # 2. Multiple swing confirmation
            if len(recent_swings) >= 3:
                consistent_direction = True
                for i in range(len(recent_swings) - 1):
                    if current_swing.swing_type in [SwingType.HIGHER_HIGH, SwingType.HIGHER_LOW]:
                        if recent_swings[i].swing_type not in [SwingType.HIGHER_HIGH, SwingType.HIGHER_LOW]:
                            consistent_direction = False
                            break
                    elif current_swing.swing_type in [SwingType.LOWER_LOW, SwingType.LOWER_HIGH]:
                        if recent_swings[i].swing_type not in [SwingType.LOWER_LOW, SwingType.LOWER_HIGH]:
                            consistent_direction = False
                            break
                
                if consistent_direction:
                    enhancement_factors.append(0.15)
            
            # 3. Volume profile enhancement
            current_index = current_swing.index
            if self._validate_volume_profile(df, current_index, "bullish"):
                enhancement_factors.append(0.1)
            
            # 4. Session timing bonus
            if self._validate_trading_session(current_swing.timestamp):
                # Check if it's overlap session (premium time)
                if hasattr(current_swing.timestamp, 'hour'):
                    utc_hour = current_swing.timestamp.hour
                    if 13 <= utc_hour <= 16:  # London/NY overlap
                        enhancement_factors.append(0.15)
                    else:
                        enhancement_factors.append(0.05)
            
            # Apply enhancements
            total_enhancement = sum(enhancement_factors)
            enhanced_significance = base_significance + total_enhancement
            
            return min(max(enhanced_significance, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Enhanced significance calculation failed: {e}")
            return self._calculate_break_significance(previous_swing, current_swing, df)
    
    def _validate_multi_timeframe_alignment(self, direction: str, significance: float, epic: str = None, current_timeframe: str = None) -> bool:
        """Validate multi-timeframe structure alignment with actual higher TF data"""
        try:
            # If no data_fetcher available, fall back to significance-based validation
            if not self.data_fetcher or not epic:
                return self._fallback_mtf_validation(significance)
            
            # Define timeframe hierarchy for multi-timeframe analysis
            timeframe_hierarchy = {
                '5m': ['15m', '1h'],
                '15m': ['1h', '4h'], 
                '1h': ['4h', '1d'],
                '4h': ['1d', '1w']
            }
            
            current_tf = current_timeframe or '15m'
            higher_timeframes = timeframe_hierarchy.get(current_tf, ['1h', '4h'])
            
            alignment_score = 0
            total_checks = 0
            
            # Check alignment with each higher timeframe
            for htf in higher_timeframes:
                try:
                    # Fetch higher timeframe data
                    htf_structure = self._get_higher_timeframe_structure(epic, htf)
                    
                    if htf_structure:
                        total_checks += 1
                        
                        # Check if higher timeframe structure aligns with signal direction
                        if self._check_structure_alignment(htf_structure, direction):
                            alignment_score += 1
                            
                            # Bonus for strong higher timeframe trends
                            if htf_structure.get('trend_strength', 0) > 0.7:
                                alignment_score += 0.5
                                
                except Exception as e:
                    self.logger.debug(f"Could not fetch {htf} data for MTF analysis: {e}")
                    continue
            
            # Require at least 60% alignment across checked timeframes
            if total_checks == 0:
                return self._fallback_mtf_validation(significance)
            
            alignment_ratio = alignment_score / total_checks
            required_alignment = 0.6  # 60% minimum alignment
            
            # Apply MTF confluence weight from config
            mtf_weight = getattr(self, 'mtf_confluence_weight', 0.8)
            weighted_alignment = alignment_ratio * mtf_weight
            
            is_aligned = weighted_alignment >= required_alignment
            
            if is_aligned:
                self.logger.debug(f"✅ Multi-timeframe alignment confirmed: {alignment_ratio:.1%}")
            else:
                self.logger.debug(f"❌ Multi-timeframe alignment failed: {alignment_ratio:.1%} < {required_alignment:.1%}")
            
            return is_aligned
                
        except Exception as e:
            self.logger.error(f"Multi-timeframe alignment validation failed: {e}")
            return self._fallback_mtf_validation(significance)
    
    def _fallback_mtf_validation(self, significance: float) -> bool:
        """Fallback MTF validation when higher timeframe data unavailable"""
        # High significance suggests multi-timeframe alignment
        if significance >= 0.7:
            return True
        elif significance >= 0.5:
            return True  # Allow medium significance signals  
        else:
            return False
    
    def _get_higher_timeframe_structure(self, epic: str, timeframe: str) -> Dict:
        """Get higher timeframe structure analysis"""
        try:
            if not self.data_fetcher:
                return None
            
            # Extract pair from epic for data fetching
            pair = self._extract_pair_from_epic(epic)
            
            # Fetch higher timeframe data (48 hours for trend analysis)
            htf_data = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=48
            )
            
            if htf_data is None or htf_data.empty or len(htf_data) < 20:
                return None
            
            # Perform basic structure analysis on higher timeframe
            structure_analysis = self._analyze_htf_structure(htf_data)
            
            return structure_analysis
            
        except Exception as e:
            self.logger.debug(f"Higher timeframe structure fetch failed: {e}")
            return None
    
    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic"""
        try:
            if '.D.' in epic and '.MINI.IP' in epic:
                parts = epic.split('.D.')
                if len(parts) > 1:
                    pair_part = parts[1].split('.MINI.IP')[0]
                    return pair_part
            
            # Fallback for unknown format
            return 'EURUSD'
            
        except Exception:
            return 'EURUSD'
    
    def _analyze_htf_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze higher timeframe structure for trend and momentum"""
        try:
            if len(df) < 20:
                return None
            
            # Simple trend analysis using price action
            recent_data = df.tail(20)
            
            # Calculate trend using linear regression-like approach
            prices = recent_data['close'].values
            x = range(len(prices))
            
            # Simple slope calculation
            n = len(prices)
            sum_x = sum(x)
            sum_y = sum(prices)
            sum_xy = sum(x[i] * prices[i] for i in range(n))
            sum_xx = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_xx - sum_x ** 2 == 0:
                slope = 0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            
            # Determine trend direction and strength
            price_range = recent_data['high'].max() - recent_data['low'].min()
            if price_range == 0:
                trend_strength = 0
            else:
                trend_strength = abs(slope) / price_range * 1000  # Scale for readability
                trend_strength = min(trend_strength, 1.0)  # Cap at 1.0
            
            if slope > 0.00001:  # Bullish trend
                trend_direction = 'bullish'
            elif slope < -0.00001:  # Bearish trend
                trend_direction = 'bearish'
            else:
                trend_direction = 'neutral'
            
            # Additional momentum analysis
            momentum = self._calculate_htf_momentum(recent_data)
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'slope': slope,
                'momentum': momentum,
                'price_range': price_range,
                'analysis_period': len(recent_data)
            }
            
        except Exception as e:
            self.logger.error(f"HTF structure analysis failed: {e}")
            return None
    
    def _calculate_htf_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum for higher timeframe analysis"""
        try:
            if len(df) < 10:
                return 0.5
            
            # Simple momentum using rate of change
            recent_close = df['close'].iloc[-1]
            older_close = df['close'].iloc[-10]
            
            if older_close == 0:
                return 0.5
            
            momentum_raw = (recent_close - older_close) / older_close
            
            # Normalize to 0-1 scale (0.5 = neutral)
            momentum_normalized = 0.5 + (momentum_raw * 100)  # Scale up
            return max(0.0, min(1.0, momentum_normalized))  # Clamp to 0-1
            
        except Exception:
            return 0.5
    
    def _check_structure_alignment(self, htf_structure: Dict, signal_direction: str) -> bool:
        """Check if higher timeframe structure aligns with signal direction"""
        try:
            htf_direction = htf_structure.get('trend_direction', 'neutral')
            htf_strength = htf_structure.get('trend_strength', 0)

            # Require minimum trend strength for alignment
            if htf_strength < 0.3:
                return False  # Too weak to provide meaningful alignment

            # Check directional alignment
            if signal_direction == 'bullish' and htf_direction == 'bullish':
                return True
            elif signal_direction == 'bearish' and htf_direction == 'bearish':
                return True
            elif htf_direction == 'neutral':
                # Neutral higher timeframe allows signals with lower confidence
                return htf_strength < 0.5  # Only if really neutral
            else:
                # Opposite direction - reject signal
                return False

        except Exception:
            return False

    def get_nearest_swing_levels(
        self,
        current_price: float,
        direction: str,
        lookback_count: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Get nearest swing high/low relative to current price
        Used by swing proximity validator to prevent poor entry timing

        Args:
            current_price: Current market price
            direction: 'BUY' or 'SELL' - trade direction
            lookback_count: Number of recent swings to analyze

        Returns:
            Dictionary with swing level information:
            {
                'nearest_resistance': float,       # Nearest swing high above price
                'nearest_support': float,          # Nearest swing low below price
                'resistance_distance_pips': float, # Distance in pips
                'support_distance_pips': float,    # Distance in pips
                'swing_high_type': str,            # HH/LH/EQH
                'swing_low_type': str              # HL/LL/EQL
            }
        """
        try:
            if not self.swing_points:
                self.logger.debug("No swing points available for level calculation")
                return None

            # Get recent swing points
            recent_swings = self.swing_points[-min(lookback_count * 2, len(self.swing_points)):]

            # Separate swing highs and lows
            swing_highs = [sp for sp in recent_swings
                          if sp.swing_type in [SwingType.HIGHER_HIGH, SwingType.LOWER_HIGH, SwingType.EQUAL_HIGH]]
            swing_lows = [sp for sp in recent_swings
                         if sp.swing_type in [SwingType.HIGHER_LOW, SwingType.LOWER_LOW, SwingType.EQUAL_LOW]]

            # Find nearest resistance (swing high above current price)
            nearest_resistance = None
            swing_high_type = 'HH'

            if swing_highs:
                # Get swing highs above current price
                highs_above = [sp for sp in swing_highs if sp.price > current_price]

                if highs_above:
                    # Get the nearest one (smallest price above)
                    nearest_swing_high = min(highs_above, key=lambda sp: sp.price)
                    nearest_resistance = nearest_swing_high.price
                    swing_high_type = nearest_swing_high.swing_type.value
                else:
                    # All swings below current price, get the highest recent one
                    nearest_swing_high = max(swing_highs[-lookback_count:], key=lambda sp: sp.price)
                    nearest_resistance = nearest_swing_high.price
                    swing_high_type = nearest_swing_high.swing_type.value

            # Find nearest support (swing low below current price)
            nearest_support = None
            swing_low_type = 'LL'

            if swing_lows:
                # Get swing lows below current price
                lows_below = [sp for sp in swing_lows if sp.price < current_price]

                if lows_below:
                    # Get the nearest one (largest price below)
                    nearest_swing_low = max(lows_below, key=lambda sp: sp.price)
                    nearest_support = nearest_swing_low.price
                    swing_low_type = nearest_swing_low.swing_type.value
                else:
                    # All swings above current price, get the lowest recent one
                    nearest_swing_low = min(swing_lows[-lookback_count:], key=lambda sp: sp.price)
                    nearest_support = nearest_swing_low.price
                    swing_low_type = nearest_swing_low.swing_type.value

            # Calculate distances (will be calculated with proper pip value by validator)
            result = {
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'swing_high_type': swing_high_type,
                'swing_low_type': swing_low_type,
                'resistance_distance_pips': None,  # Calculated by validator
                'support_distance_pips': None      # Calculated by validator
            }

            self.logger.debug(
                f"Swing levels for price {current_price:.5f}: "
                f"resistance={nearest_resistance:.5f} ({swing_high_type}), "
                f"support={nearest_support:.5f} ({swing_low_type})"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to get nearest swing levels: {e}")
            return None

    def get_last_bos_choch_direction(self, df: pd.DataFrame) -> Optional[str]:
        """
        Get the direction of the most recent BOS/CHoCH

        Args:
            df: DataFrame with structure analysis (must have 'structure_break', 'break_direction' columns)

        Returns:
            'bullish', 'bearish', or None if no BOS/CHoCH found
        """
        try:
            if df.empty or 'structure_break' not in df.columns:
                return None

            # Find rows with structure breaks
            breaks = df[df['structure_break'] == True]

            if breaks.empty:
                return None

            # Get the most recent break
            last_break = breaks.iloc[-1]

            return last_break.get('break_direction', None)

        except Exception as e:
            self.logger.error(f"Failed to get last BOS/CHoCH direction: {e}")
            return None