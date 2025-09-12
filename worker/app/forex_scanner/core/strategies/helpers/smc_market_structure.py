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
from typing import Dict, List, Optional, Tuple
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
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.swing_points: List[SwingPoint] = []
        self.structure_breaks: List[StructureBreak] = []
        self.current_structure = StructureType.NEUTRAL
        
    def analyze_market_structure(
        self, 
        df: pd.DataFrame, 
        config: Dict
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
            df_enhanced = df.copy()
            
            # Detect swing points
            df_enhanced = self._detect_swing_points(df_enhanced, config)
            
            # Analyze structure breaks
            df_enhanced = self._analyze_structure_breaks(df_enhanced, config)
            
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
    
    def _calculate_pivot_high(self, high_series: pd.Series, length: int) -> pd.Series:
        """Calculate pivot highs using rolling window"""
        try:
            pivot_highs = pd.Series(index=high_series.index, dtype=float)
            
            for i in range(length, len(high_series) - length):
                center_price = high_series.iloc[i]
                left_prices = high_series.iloc[i-length:i]
                right_prices = high_series.iloc[i+1:i+length+1]
                
                # Check if center is highest
                if (center_price > left_prices.max()) and (center_price > right_prices.max()):
                    pivot_highs.iloc[i] = center_price
            
            return pivot_highs
            
        except Exception as e:
            self.logger.error(f"Pivot high calculation failed: {e}")
            return pd.Series(index=high_series.index, dtype=float)
    
    def _calculate_pivot_low(self, low_series: pd.Series, length: int) -> pd.Series:
        """Calculate pivot lows using rolling window"""
        try:
            pivot_lows = pd.Series(index=low_series.index, dtype=float)
            
            for i in range(length, len(low_series) - length):
                center_price = low_series.iloc[i]
                left_prices = low_series.iloc[i-length:i]
                right_prices = low_series.iloc[i+1:i+length+1]
                
                # Check if center is lowest
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
    
    def _analyze_structure_breaks(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Analyze breaks of structure and changes of character"""
        try:
            confirmation_bars = config.get('structure_confirmation', 3)
            bos_threshold = config.get('bos_threshold', 0.0001)
            
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
                    confirmation_bars
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
        confirmation_bars: int
    ) -> Optional[StructureBreak]:
        """Detect if current swing creates a structure break"""
        try:
            # Determine break type based on swing sequence
            if (previous_swing.swing_type in [SwingType.HIGHER_HIGH, SwingType.EQUAL_HIGH] and
                current_swing.swing_type == SwingType.LOWER_HIGH):
                # Potential bearish ChoCH (Change of Character)
                break_type = "ChoCH"
                direction = "bearish"
                new_structure = StructureType.BEARISH
                
            elif (previous_swing.swing_type in [SwingType.LOWER_LOW, SwingType.EQUAL_LOW] and
                  current_swing.swing_type == SwingType.HIGHER_LOW):
                # Potential bullish ChoCH
                break_type = "ChoCH"
                direction = "bullish"
                new_structure = StructureType.BULLISH
                
            elif (current_swing.swing_type == SwingType.HIGHER_HIGH and 
                  self.current_structure == StructureType.BULLISH):
                # Bullish BOS (Break of Structure)
                break_type = "BOS"
                direction = "bullish"
                new_structure = StructureType.BULLISH
                
            elif (current_swing.swing_type == SwingType.LOWER_LOW and 
                  self.current_structure == StructureType.BEARISH):
                # Bearish BOS
                break_type = "BOS"
                direction = "bearish"
                new_structure = StructureType.BEARISH
                
            else:
                return None
            
            # Check price movement threshold
            price_difference = abs(current_swing.price - previous_swing.price)
            if price_difference < threshold:
                return None
            
            # Calculate significance based on price move and volume
            significance = self._calculate_break_significance(
                previous_swing, current_swing, df
            )
            
            # Require minimum significance
            if significance < 0.3:
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
            self.logger.error(f"Structure break detection failed: {e}")
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