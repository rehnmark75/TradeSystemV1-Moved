# core/strategies/helpers/smc_order_blocks.py
"""
Smart Money Concepts - Enhanced Order Block Detection with Fair Value Gap Analysis
Identifies institutional order blocks based on volume, price action, and FVG validation

Order Blocks represent areas where institutions have placed large orders:
- Bullish Order Block: Strong move up after consolidation/pullback + FVG validation
- Bearish Order Block: Strong move down after consolidation/pullback + FVG validation
- High volume + significant price movement + Fair Value Gap = Institution activity
- FVG analysis enhances order block quality and confirmation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from .smc_fair_value_gaps import SMCFairValueGaps, FVGType, FairValueGap


class OrderBlockType(Enum):
    """Types of order blocks"""
    BULLISH = "bullish"
    BEARISH = "bearish"


class OrderBlockStrength(Enum):
    """Order block strength levels"""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class OrderBlock:
    """Represents an institutional order block enhanced with Fair Value Gap analysis"""
    start_index: int
    end_index: int
    high_price: float
    low_price: float
    block_type: OrderBlockType
    strength: OrderBlockStrength
    volume_factor: float
    price_movement: float
    timestamp: pd.Timestamp
    tested_count: int = 0
    still_valid: bool = True
    confidence: float = 0.0
    # Fair Value Gap enhancements
    associated_fvgs: List[FairValueGap] = None
    fvg_confluence_score: float = 0.0
    has_fvg_support: bool = False
    fvg_alignment_strength: float = 0.0


class SMCOrderBlocks:
    """Smart Money Concepts Order Block Detector Enhanced with Fair Value Gap Analysis"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.order_blocks: List[OrderBlock] = []
        self.fvg_analyzer = SMCFairValueGaps(logger=self.logger)

    def _get_pip_value(self, config: Dict) -> float:
        """Get pip value from config or default based on pair"""
        if 'pip_value' in config:
            return config['pip_value']
        pair = config.get('pair', config.get('epic', ''))
        if 'JPY' in str(pair).upper():
            return 0.01
        return 0.0001
        
    def detect_order_blocks(
        self, 
        df: pd.DataFrame, 
        config: Dict
    ) -> pd.DataFrame:
        """
        Enhanced order block detection with Fair Value Gap analysis
        
        Args:
            df: OHLCV DataFrame
            config: SMC configuration dictionary
            
        Returns:
            Enhanced DataFrame with order block and FVG analysis
        """
        try:
            df_enhanced = df.copy()
            
            # Initialize order block columns
            df_enhanced['order_block_bullish'] = False
            df_enhanced['order_block_bearish'] = False
            df_enhanced['order_block_strength'] = ''
            df_enhanced['order_block_confidence'] = 0.0
            df_enhanced['order_block_high'] = np.nan
            df_enhanced['order_block_low'] = np.nan
            df_enhanced['order_block_fvg_support'] = False
            df_enhanced['order_block_fvg_score'] = 0.0
            
            # Clear previous order blocks
            self.order_blocks = []
            
            # First, detect Fair Value Gaps for validation
            self.logger.debug("🔍 Detecting Fair Value Gaps for order block validation")
            df_enhanced = self.fvg_analyzer.detect_fair_value_gaps(df_enhanced, config)
            
            # Detect order blocks with FVG validation
            self._scan_for_order_blocks_with_fvg(df_enhanced, config)
            
            # Mark order blocks in DataFrame
            for order_block in self.order_blocks:
                self._mark_order_block_in_df(df_enhanced, order_block)
            
            # Add order block zones
            df_enhanced = self._add_order_block_zones(df_enhanced, config)
            
            self.logger.debug(f"📊 Enhanced order block detection complete: {len(self.order_blocks)} blocks found")
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Enhanced order block detection failed: {e}")
            return df
    
    def _scan_for_order_blocks_with_fvg(self, df: pd.DataFrame, config: Dict):
        """Enhanced scan for order blocks with Fair Value Gap validation"""
        try:
            order_block_length = config.get('order_block_length', 3)
            volume_factor = config.get('order_block_volume_factor', 1.5)
            min_price_movement = config.get('bos_threshold', 0.0001) * 2

            # Calculate average volume for comparison
            # Prefer 'ltv' (Last Traded Volume) over 'volume' - IG provides actual data in ltv
            volumes = []
            for i in range(len(df)):
                row = df.iloc[i]
                # Check ltv first (IG's actual volume), then volume, then default to 1
                vol = row.get('ltv', 0) or row.get('volume', 0) or 1
                if vol and vol > 0:
                    volumes.append(vol)

            avg_volume = sum(volumes) / len(volumes) if volumes else 1
            
            # Scan for order block patterns with FVG validation
            for i in range(order_block_length * 2, len(df) - order_block_length):
                self._check_bullish_order_block_with_fvg(df, i, config, avg_volume)
                self._check_bearish_order_block_with_fvg(df, i, config, avg_volume)
            
            # Keep only the strongest order blocks (prioritizing FVG support)
            max_blocks = config.get('max_order_blocks', 5)
            if len(self.order_blocks) > max_blocks:
                # Sort by FVG support first, then confidence
                self.order_blocks.sort(key=lambda x: (x.has_fvg_support, x.confidence), reverse=True)
                self.order_blocks = self.order_blocks[:max_blocks]
                
        except Exception as e:
            self.logger.error(f"Enhanced order block scanning failed: {e}")
    
    def _validate_order_block_with_fvg(
        self,
        df: pd.DataFrame,
        consolidation_start: int,
        consolidation_end: int,
        move_start: int,
        move_end: int,
        block_type: OrderBlockType,
        config: Dict
    ) -> Tuple[bool, float, List[FairValueGap], float]:
        """
        Validate order block with Fair Value Gap analysis

        Returns:
            (has_fvg_support, fvg_confluence_score, associated_fvgs, alignment_strength)
        """
        try:
            associated_fvgs = []
            fvg_confluence_score = 0.0
            alignment_strength = 0.0

            # Get pip value for correct conversions (JPY pairs use 0.01)
            pip_value = self._get_pip_value(config)

            # Check for FVGs in the move period and surrounding areas
            search_start = max(0, consolidation_start - 5)
            search_end = min(len(df), move_end + 5)

            # Look for FVGs that align with the order block direction
            for i in range(search_start, search_end):
                if i >= len(df):
                    continue

                row = df.iloc[i]

                # Check for bullish FVG support for bullish order block
                if (block_type == OrderBlockType.BULLISH and
                    row.get('fvg_bullish', False)):

                    fvg_high = row.get('fvg_high', 0)
                    fvg_low = row.get('fvg_low', 0)

                    if fvg_high > 0 and fvg_low > 0:
                        # Check if FVG overlaps with consolidation area
                        consolidation_data = df.iloc[consolidation_start:consolidation_end + 1]
                        consolidation_low = consolidation_data['low'].min()
                        consolidation_high = consolidation_data['high'].max()

                        # FVG provides support if it's below or within consolidation
                        if (fvg_low <= consolidation_high and fvg_high >= consolidation_low):
                            fvg_gap = FairValueGap(
                                start_index=i,
                                high_price=fvg_high,
                                low_price=fvg_low,
                                gap_type=FVGType.BULLISH,
                                gap_size_pips=(fvg_high - fvg_low) / pip_value,
                                volume_confirmation=row.get('volume', 1),
                                timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else pd.Timestamp.now(),
                                significance=row.get('fvg_significance', 0.5)
                            )
                            associated_fvgs.append(fvg_gap)
                            
                            # Calculate confluence score based on proximity and size
                            proximity_score = 1.0 - abs(i - move_start) / max(10, move_end - move_start)
                            size_score = min(1.0, fvg_gap.gap_size_pips / 50)  # Normalize by 50 pips
                            fvg_confluence_score += proximity_score * size_score * fvg_gap.significance
                
                # Check for bearish FVG support for bearish order block
                elif (block_type == OrderBlockType.BEARISH and
                      row.get('fvg_bearish', False)):

                    fvg_high = row.get('fvg_high', 0)
                    fvg_low = row.get('fvg_low', 0)

                    if fvg_high > 0 and fvg_low > 0:
                        # Check if FVG overlaps with consolidation area
                        consolidation_data = df.iloc[consolidation_start:consolidation_end + 1]
                        consolidation_low = consolidation_data['low'].min()
                        consolidation_high = consolidation_data['high'].max()

                        # FVG provides resistance if it's above or within consolidation
                        if (fvg_high >= consolidation_low and fvg_low <= consolidation_high):
                            fvg_gap = FairValueGap(
                                start_index=i,
                                high_price=fvg_high,
                                low_price=fvg_low,
                                gap_type=FVGType.BEARISH,
                                gap_size_pips=(fvg_high - fvg_low) / pip_value,
                                volume_confirmation=row.get('volume', 1),
                                timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else pd.Timestamp.now(),
                                significance=row.get('fvg_significance', 0.5)
                            )
                            associated_fvgs.append(fvg_gap)
                            
                            # Calculate confluence score
                            proximity_score = 1.0 - abs(i - move_start) / max(10, move_end - move_start)
                            size_score = min(1.0, fvg_gap.gap_size_pips / 50)
                            fvg_confluence_score += proximity_score * size_score * fvg_gap.significance
            
            # Calculate alignment strength
            if associated_fvgs:
                alignment_strength = min(1.0, fvg_confluence_score)
                has_fvg_support = alignment_strength > 0.3  # Threshold for significant FVG support
            else:
                has_fvg_support = False
                alignment_strength = 0.0
            
            return has_fvg_support, fvg_confluence_score, associated_fvgs, alignment_strength
            
        except Exception as e:
            self.logger.error(f"FVG validation failed: {e}")
            return False, 0.0, [], 0.0
    
    def _scan_for_order_blocks(self, df: pd.DataFrame, config: Dict):
        """Scan DataFrame for order block patterns"""
        try:
            order_block_length = config.get('order_block_length', 3)
            volume_factor = config.get('order_block_volume_factor', 1.5)
            min_price_movement = config.get('bos_threshold', 0.0001) * 2  # Minimum price movement

            # Calculate average volume for comparison
            # Prefer 'ltv' (Last Traded Volume) over 'volume' - IG provides actual data in ltv
            volumes = []
            for i in range(len(df)):
                row = df.iloc[i]
                # Check ltv first (IG's actual volume), then volume, then default to 1
                vol = row.get('ltv', 0) or row.get('volume', 0) or 1
                if vol and vol > 0:
                    volumes.append(vol)

            avg_volume = sum(volumes) / len(volumes) if volumes else 1
            
            # Scan for order block patterns
            for i in range(order_block_length * 2, len(df) - order_block_length):
                self._check_bullish_order_block(df, i, config, avg_volume)
                self._check_bearish_order_block(df, i, config, avg_volume)
            
            # Keep only the strongest order blocks
            max_blocks = config.get('max_order_blocks', 5)
            if len(self.order_blocks) > max_blocks:
                # Sort by confidence and keep top blocks
                self.order_blocks.sort(key=lambda x: x.confidence, reverse=True)
                self.order_blocks = self.order_blocks[:max_blocks]
                
        except Exception as e:
            self.logger.error(f"Order block scanning failed: {e}")
    
    def _check_bullish_order_block_with_fvg(
        self, 
        df: pd.DataFrame, 
        current_index: int, 
        config: Dict, 
        avg_volume: float
    ):
        """Enhanced bullish order block detection with Fair Value Gap validation"""
        try:
            block_length = config.get('order_block_length', 3)
            volume_factor = config.get('order_block_volume_factor', 1.5)
            
            # Look back for consolidation period followed by strong move up
            consolidation_start = current_index - block_length * 2
            consolidation_end = current_index - block_length
            move_start = consolidation_end
            move_end = current_index
            
            if consolidation_start < 0:
                return
            
            # Check consolidation phase
            consolidation_data = df.iloc[consolidation_start:consolidation_end + 1]
            if len(consolidation_data) < block_length:
                return
            
            consolidation_high = consolidation_data['high'].max()
            consolidation_low = consolidation_data['low'].min()
            consolidation_range = consolidation_high - consolidation_low
            
            # Check move phase
            move_data = df.iloc[move_start:move_end + 1]
            if len(move_data) < block_length:
                return
            
            move_high = move_data['high'].max()
            move_low = move_data['low'].min()
            move_start_price = df.iloc[move_start]['close']
            move_end_price = df.iloc[move_end]['close']
            
            price_movement = move_end_price - move_start_price
            move_range = move_high - move_low
            
            # Check if this is a bullish pattern
            if price_movement <= 0:
                return
            
            # Check volume confirmation
            move_volumes = []
            for i in range(move_start, move_end + 1):
                vol = df.iloc[i].get('ltv', 0) or df.iloc[i].get('volume', 0) or 1
                if vol and vol > 0:
                    move_volumes.append(vol)
            
            if not move_volumes:
                return
            
            move_avg_volume = sum(move_volumes) / len(move_volumes)
            volume_confirmation = move_avg_volume / avg_volume
            
            # Basic order block criteria
            basic_criteria_met = (volume_confirmation >= volume_factor and 
                                price_movement >= config.get('bos_threshold', 0.0001) and
                                move_range > consolidation_range * 0.5)
            
            if basic_criteria_met:
                # Get pip value for correct conversions (JPY pairs use 0.01)
                pip_value = self._get_pip_value(config)

                # Enhanced FVG validation
                has_fvg_support, fvg_confluence_score, associated_fvgs, alignment_strength = \
                    self._validate_order_block_with_fvg(
                        df, consolidation_start, consolidation_end, move_start, move_end,
                        OrderBlockType.BULLISH, config
                    )

                # Calculate enhanced order block strength (including FVG)
                base_strength = self._calculate_order_block_strength(
                    price_movement, volume_confirmation, move_range, consolidation_range, pip_value
                )

                # Enhance strength with FVG support
                if has_fvg_support:
                    if base_strength == OrderBlockStrength.WEAK:
                        enhanced_strength = OrderBlockStrength.MEDIUM
                    elif base_strength == OrderBlockStrength.MEDIUM:
                        enhanced_strength = OrderBlockStrength.STRONG
                    else:
                        enhanced_strength = OrderBlockStrength.VERY_STRONG
                else:
                    enhanced_strength = base_strength

                # Enhanced confidence calculation (including FVG)
                base_confidence = self._calculate_order_block_confidence(
                    volume_confirmation, price_movement, enhanced_strength, 'bullish', pip_value
                )
                
                # Boost confidence with FVG support
                fvg_boost = min(0.3, alignment_strength * 0.5) if has_fvg_support else 0.0
                enhanced_confidence = min(1.0, base_confidence + fvg_boost)
                
                # Only create order block if it has minimum quality (FVG support or high confidence)
                min_confidence = config.get('order_block_min_confidence', 0.4)
                if enhanced_confidence >= min_confidence or has_fvg_support:
                    # Create enhanced order block with FVG information
                    order_block = OrderBlock(
                        start_index=consolidation_start,
                        end_index=consolidation_end,
                        high_price=consolidation_high,
                        low_price=consolidation_low,
                        block_type=OrderBlockType.BULLISH,
                        strength=enhanced_strength,
                        volume_factor=volume_confirmation,
                        price_movement=price_movement,
                        timestamp=df.index[current_index] if hasattr(df.index, 'to_pydatetime') else pd.Timestamp.now(),
                        confidence=enhanced_confidence,
                        # FVG enhancements
                        associated_fvgs=associated_fvgs,
                        fvg_confluence_score=fvg_confluence_score,
                        has_fvg_support=has_fvg_support,
                        fvg_alignment_strength=alignment_strength
                    )
                    
                    self.order_blocks.append(order_block)
                    self.logger.debug(f"✅ Enhanced bullish order block detected at {current_index} "
                                    f"(confidence: {enhanced_confidence:.2f}, "
                                    f"FVG support: {has_fvg_support}, "
                                    f"FVG score: {fvg_confluence_score:.2f})")
                
        except Exception as e:
            self.logger.error(f"Enhanced bullish order block check failed: {e}")
    
    def _check_bearish_order_block_with_fvg(
        self, 
        df: pd.DataFrame, 
        current_index: int, 
        config: Dict, 
        avg_volume: float
    ):
        """Enhanced bearish order block detection with Fair Value Gap validation"""
        try:
            block_length = config.get('order_block_length', 3)
            volume_factor = config.get('order_block_volume_factor', 1.5)
            
            # Look back for consolidation period followed by strong move down
            consolidation_start = current_index - block_length * 2
            consolidation_end = current_index - block_length
            move_start = consolidation_end
            move_end = current_index
            
            if consolidation_start < 0:
                return
            
            # Check consolidation phase
            consolidation_data = df.iloc[consolidation_start:consolidation_end + 1]
            if len(consolidation_data) < block_length:
                return
            
            consolidation_high = consolidation_data['high'].max()
            consolidation_low = consolidation_data['low'].min()
            consolidation_range = consolidation_high - consolidation_low
            
            # Check move phase
            move_data = df.iloc[move_start:move_end + 1]
            if len(move_data) < block_length:
                return
            
            move_high = move_data['high'].max()
            move_low = move_data['low'].min()
            move_start_price = df.iloc[move_start]['close']
            move_end_price = df.iloc[move_end]['close']
            
            price_movement = move_start_price - move_end_price  # Negative for bearish
            move_range = move_high - move_low
            
            # Check if this is a bearish pattern
            if price_movement <= 0:
                return
            
            # Check volume confirmation
            move_volumes = []
            for i in range(move_start, move_end + 1):
                vol = df.iloc[i].get('ltv', 0) or df.iloc[i].get('volume', 0) or 1
                if vol and vol > 0:
                    move_volumes.append(vol)
            
            if not move_volumes:
                return
            
            move_avg_volume = sum(move_volumes) / len(move_volumes)
            volume_confirmation = move_avg_volume / avg_volume
            
            # Basic order block criteria
            basic_criteria_met = (volume_confirmation >= volume_factor and 
                                price_movement >= config.get('bos_threshold', 0.0001) and
                                move_range > consolidation_range * 0.5)
            
            if basic_criteria_met:
                # Get pip value for correct conversions (JPY pairs use 0.01)
                pip_value = self._get_pip_value(config)

                # Enhanced FVG validation
                has_fvg_support, fvg_confluence_score, associated_fvgs, alignment_strength = \
                    self._validate_order_block_with_fvg(
                        df, consolidation_start, consolidation_end, move_start, move_end,
                        OrderBlockType.BEARISH, config
                    )

                # Calculate enhanced order block strength (including FVG)
                base_strength = self._calculate_order_block_strength(
                    price_movement, volume_confirmation, move_range, consolidation_range, pip_value
                )

                # Enhance strength with FVG support
                if has_fvg_support:
                    if base_strength == OrderBlockStrength.WEAK:
                        enhanced_strength = OrderBlockStrength.MEDIUM
                    elif base_strength == OrderBlockStrength.MEDIUM:
                        enhanced_strength = OrderBlockStrength.STRONG
                    else:
                        enhanced_strength = OrderBlockStrength.VERY_STRONG
                else:
                    enhanced_strength = base_strength

                # Enhanced confidence calculation (including FVG)
                base_confidence = self._calculate_order_block_confidence(
                    volume_confirmation, price_movement, enhanced_strength, 'bearish', pip_value
                )
                
                # Boost confidence with FVG support
                fvg_boost = min(0.3, alignment_strength * 0.5) if has_fvg_support else 0.0
                enhanced_confidence = min(1.0, base_confidence + fvg_boost)
                
                # Only create order block if it has minimum quality (FVG support or high confidence)
                min_confidence = config.get('order_block_min_confidence', 0.4)
                if enhanced_confidence >= min_confidence or has_fvg_support:
                    # Create enhanced order block with FVG information
                    order_block = OrderBlock(
                        start_index=consolidation_start,
                        end_index=consolidation_end,
                        high_price=consolidation_high,
                        low_price=consolidation_low,
                        block_type=OrderBlockType.BEARISH,
                        strength=enhanced_strength,
                        volume_factor=volume_confirmation,
                        price_movement=price_movement,
                        timestamp=df.index[current_index] if hasattr(df.index, 'to_pydatetime') else pd.Timestamp.now(),
                        confidence=enhanced_confidence,
                        # FVG enhancements
                        associated_fvgs=associated_fvgs,
                        fvg_confluence_score=fvg_confluence_score,
                        has_fvg_support=has_fvg_support,
                        fvg_alignment_strength=alignment_strength
                    )
                    
                    self.order_blocks.append(order_block)
                    self.logger.debug(f"✅ Enhanced bearish order block detected at {current_index} "
                                    f"(confidence: {enhanced_confidence:.2f}, "
                                    f"FVG support: {has_fvg_support}, "
                                    f"FVG score: {fvg_confluence_score:.2f})")
                
        except Exception as e:
            self.logger.error(f"Enhanced bearish order block check failed: {e}")
    
    def _check_bullish_order_block(
        self, 
        df: pd.DataFrame, 
        current_index: int, 
        config: Dict, 
        avg_volume: float
    ):
        """Check for bullish order block pattern at current index"""
        try:
            block_length = config.get('order_block_length', 3)
            volume_factor = config.get('order_block_volume_factor', 1.5)
            
            # Look back for consolidation period followed by strong move up
            consolidation_start = current_index - block_length * 2
            consolidation_end = current_index - block_length
            move_start = consolidation_end
            move_end = current_index
            
            if consolidation_start < 0:
                return
            
            # Check consolidation phase
            consolidation_data = df.iloc[consolidation_start:consolidation_end + 1]
            if len(consolidation_data) < block_length:
                return
            
            consolidation_high = consolidation_data['high'].max()
            consolidation_low = consolidation_data['low'].min()
            consolidation_range = consolidation_high - consolidation_low
            
            # Check move phase
            move_data = df.iloc[move_start:move_end + 1]
            if len(move_data) < block_length:
                return
            
            move_high = move_data['high'].max()
            move_low = move_data['low'].min()
            move_start_price = df.iloc[move_start]['close']
            move_end_price = df.iloc[move_end]['close']
            
            # Bullish order block criteria:
            # 1. Strong upward move after consolidation
            # 2. High volume during the move
            # 3. Significant price movement
            
            price_movement = move_end_price - move_start_price
            move_range = move_high - move_low
            
            # Check if this is a bullish pattern
            if price_movement <= 0:
                return
            
            # Check volume confirmation
            move_volumes = []
            for i in range(move_start, move_end + 1):
                vol = df.iloc[i].get('ltv', 0) or df.iloc[i].get('volume', 0) or 1
                if vol and vol > 0:
                    move_volumes.append(vol)
            
            if not move_volumes:
                return
            
            move_avg_volume = sum(move_volumes) / len(move_volumes)
            volume_confirmation = move_avg_volume / avg_volume
            
            # Check criteria
            if (volume_confirmation >= volume_factor and
                price_movement >= config.get('bos_threshold', 0.0001) and
                move_range > consolidation_range * 0.5):  # Move should be significant

                # Get pip value for correct conversions (JPY pairs use 0.01)
                pip_value = self._get_pip_value(config)

                # Calculate order block strength
                strength = self._calculate_order_block_strength(
                    price_movement, volume_confirmation, move_range, consolidation_range, pip_value
                )

                # Calculate confidence
                confidence = self._calculate_order_block_confidence(
                    volume_confirmation, price_movement, strength, 'bullish', pip_value
                )

                # Create order block
                order_block = OrderBlock(
                    start_index=consolidation_start,
                    end_index=consolidation_end,
                    high_price=consolidation_high,
                    low_price=consolidation_low,
                    block_type=OrderBlockType.BULLISH,
                    strength=strength,
                    volume_factor=volume_confirmation,
                    price_movement=price_movement,
                    timestamp=df.index[current_index] if hasattr(df.index, 'to_pydatetime') else pd.Timestamp.now(),
                    confidence=confidence
                )

                self.order_blocks.append(order_block)

        except Exception as e:
            self.logger.error(f"Bullish order block check failed: {e}")
    
    def _check_bearish_order_block(
        self, 
        df: pd.DataFrame, 
        current_index: int, 
        config: Dict, 
        avg_volume: float
    ):
        """Check for bearish order block pattern at current index"""
        try:
            block_length = config.get('order_block_length', 3)
            volume_factor = config.get('order_block_volume_factor', 1.5)
            
            # Look back for consolidation period followed by strong move down
            consolidation_start = current_index - block_length * 2
            consolidation_end = current_index - block_length
            move_start = consolidation_end
            move_end = current_index
            
            if consolidation_start < 0:
                return
            
            # Check consolidation phase
            consolidation_data = df.iloc[consolidation_start:consolidation_end + 1]
            if len(consolidation_data) < block_length:
                return
            
            consolidation_high = consolidation_data['high'].max()
            consolidation_low = consolidation_data['low'].min()
            consolidation_range = consolidation_high - consolidation_low
            
            # Check move phase
            move_data = df.iloc[move_start:move_end + 1]
            if len(move_data) < block_length:
                return
            
            move_high = move_data['high'].max()
            move_low = move_data['low'].min()
            move_start_price = df.iloc[move_start]['close']
            move_end_price = df.iloc[move_end]['close']
            
            # Bearish order block criteria:
            # 1. Strong downward move after consolidation
            # 2. High volume during the move
            # 3. Significant price movement
            
            price_movement = move_start_price - move_end_price  # Positive for downward move
            move_range = move_high - move_low
            
            # Check if this is a bearish pattern
            if price_movement <= 0:
                return
            
            # Check volume confirmation
            move_volumes = []
            for i in range(move_start, move_end + 1):
                vol = df.iloc[i].get('ltv', 0) or df.iloc[i].get('volume', 0) or 1
                if vol and vol > 0:
                    move_volumes.append(vol)
            
            if not move_volumes:
                return
            
            move_avg_volume = sum(move_volumes) / len(move_volumes)
            volume_confirmation = move_avg_volume / avg_volume
            
            # Check criteria
            if (volume_confirmation >= volume_factor and
                price_movement >= config.get('bos_threshold', 0.0001) and
                move_range > consolidation_range * 0.5):  # Move should be significant

                # Get pip value for correct conversions (JPY pairs use 0.01)
                pip_value = self._get_pip_value(config)

                # Calculate order block strength
                strength = self._calculate_order_block_strength(
                    price_movement, volume_confirmation, move_range, consolidation_range, pip_value
                )

                # Calculate confidence
                confidence = self._calculate_order_block_confidence(
                    volume_confirmation, price_movement, strength, 'bearish', pip_value
                )

                # Create order block
                order_block = OrderBlock(
                    start_index=consolidation_start,
                    end_index=consolidation_end,
                    high_price=consolidation_high,
                    low_price=consolidation_low,
                    block_type=OrderBlockType.BEARISH,
                    strength=strength,
                    volume_factor=volume_confirmation,
                    price_movement=price_movement,
                    timestamp=df.index[current_index] if hasattr(df.index, 'to_pydatetime') else pd.Timestamp.now(),
                    confidence=confidence
                )

                self.order_blocks.append(order_block)

        except Exception as e:
            self.logger.error(f"Bearish order block check failed: {e}")
    
    def _calculate_order_block_strength(
        self,
        price_movement: float,
        volume_factor: float,
        move_range: float,
        consolidation_range: float,
        pip_value: float = 0.0001
    ) -> OrderBlockStrength:
        """Calculate the strength of an order block"""
        try:
            # Combine multiple factors for strength assessment
            volume_score = min(volume_factor / 3.0, 1.0)  # Normalize to 0-1
            # Convert to pips using correct pip value and normalize
            price_score = min(price_movement / pip_value, 1.0)
            range_ratio = move_range / consolidation_range if consolidation_range > 0 else 1.0
            range_score = min(range_ratio / 3.0, 1.0)

            overall_score = (volume_score * 0.4 + price_score * 0.4 + range_score * 0.2)

            if overall_score >= 0.8:
                return OrderBlockStrength.VERY_STRONG
            elif overall_score >= 0.6:
                return OrderBlockStrength.STRONG
            elif overall_score >= 0.4:
                return OrderBlockStrength.MEDIUM
            else:
                return OrderBlockStrength.WEAK

        except Exception as e:
            self.logger.error(f"Order block strength calculation failed: {e}")
            return OrderBlockStrength.MEDIUM
    
    def _calculate_order_block_confidence(
        self,
        volume_factor: float,
        price_movement: float,
        strength: OrderBlockStrength,
        block_type: str,
        pip_value: float = 0.0001
    ) -> float:
        """Calculate confidence score for order block"""
        try:
            base_confidence = 0.5

            # Volume factor contribution
            volume_contribution = min((volume_factor - 1.0) * 0.2, 0.3)

            # Price movement contribution (convert to pips using correct pip value)
            price_pips = price_movement / pip_value
            price_contribution = min(price_pips * 0.05, 0.2)

            # Strength contribution
            strength_map = {
                OrderBlockStrength.WEAK: 0.0,
                OrderBlockStrength.MEDIUM: 0.1,
                OrderBlockStrength.STRONG: 0.2,
                OrderBlockStrength.VERY_STRONG: 0.3
            }
            strength_contribution = strength_map.get(strength, 0.1)

            total_confidence = base_confidence + volume_contribution + price_contribution + strength_contribution

            return min(max(total_confidence, 0.1), 0.95)

        except Exception as e:
            self.logger.error(f"Order block confidence calculation failed: {e}")
            return 0.5
    
    def _mark_order_block_in_df(self, df: pd.DataFrame, order_block: OrderBlock):
        """Enhanced method to mark order block with FVG information in DataFrame"""
        try:
            start_idx = order_block.start_index
            end_idx = order_block.end_index
            
            # Mark all candles in the order block
            for i in range(start_idx, min(end_idx + 1, len(df))):
                if order_block.block_type == OrderBlockType.BULLISH:
                    df.iloc[i, df.columns.get_loc('order_block_bullish')] = True
                else:
                    df.iloc[i, df.columns.get_loc('order_block_bearish')] = True
                
                df.iloc[i, df.columns.get_loc('order_block_strength')] = order_block.strength.value
                df.iloc[i, df.columns.get_loc('order_block_confidence')] = order_block.confidence
                df.iloc[i, df.columns.get_loc('order_block_high')] = order_block.high_price
                df.iloc[i, df.columns.get_loc('order_block_low')] = order_block.low_price
                
                # Enhanced FVG information
                df.iloc[i, df.columns.get_loc('order_block_fvg_support')] = order_block.has_fvg_support
                df.iloc[i, df.columns.get_loc('order_block_fvg_score')] = order_block.fvg_confluence_score
                
        except Exception as e:
            self.logger.error(f"Enhanced order block marking failed: {e}")
    
    def _add_order_block_zones(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Add order block zone analysis"""
        try:
            # Get pip value for correct conversions (JPY pairs use 0.01)
            pip_value = self._get_pip_value(config)

            buffer_pips = config.get('order_block_buffer', 2)
            buffer = buffer_pips * pip_value  # Convert pips to price

            df['in_bullish_order_block'] = False
            df['in_bearish_order_block'] = False
            df['distance_to_nearest_ob'] = np.nan

            for i, row in df.iterrows():
                current_price = row['close']

                # Check if price is in any order block zone
                nearest_distance = float('inf')

                for order_block in self.order_blocks:
                    ob_high = order_block.high_price + buffer
                    ob_low = order_block.low_price - buffer

                    # Calculate distance to order block
                    if current_price > ob_high:
                        distance = current_price - ob_high
                    elif current_price < ob_low:
                        distance = ob_low - current_price
                    else:
                        distance = 0  # Inside the zone

                        # Mark as inside order block
                        if order_block.block_type == OrderBlockType.BULLISH:
                            df.at[i, 'in_bullish_order_block'] = True
                        else:
                            df.at[i, 'in_bearish_order_block'] = True

                    nearest_distance = min(nearest_distance, distance)

                if nearest_distance != float('inf'):
                    df.at[i, 'distance_to_nearest_ob'] = nearest_distance / pip_value  # Convert price to pips

            return df

        except Exception as e:
            self.logger.error(f"Order block zones addition failed: {e}")
            return df
    
    def get_order_blocks_near_price(
        self,
        price: float,
        max_distance_pips: float = 10.0,
        pip_value: float = 0.0001
    ) -> List[OrderBlock]:
        """Get order blocks near a specific price"""
        try:
            max_distance = max_distance_pips * pip_value  # Convert pips to price
            nearby_blocks = []

            for order_block in self.order_blocks:
                if not order_block.still_valid:
                    continue

                # Check distance to order block
                if (price >= order_block.low_price - max_distance and
                    price <= order_block.high_price + max_distance):
                    nearby_blocks.append(order_block)

            # Sort by distance to price
            def distance_to_price(ob):
                if price < ob.low_price:
                    return ob.low_price - price
                elif price > ob.high_price:
                    return price - ob.high_price
                else:
                    return 0

            nearby_blocks.sort(key=distance_to_price)
            return nearby_blocks

        except Exception as e:
            self.logger.error(f"Order blocks near price search failed: {e}")
            return []
    
    def get_order_block_signals(
        self,
        df: pd.DataFrame,
        current_index: int,
        config: Dict
    ) -> Dict:
        """Get order block-based trading signals"""
        try:
            if current_index >= len(df):
                return {}

            # Get pip value for correct conversions (JPY pairs use 0.01)
            pip_value = self._get_pip_value(config)

            current_price = df.iloc[current_index]['close']
            max_distance_pips = config.get('max_distance_to_zone', 10)
            max_distance = max_distance_pips * pip_value  # Convert pips to price

            signals = {
                'bullish_ob_signal': False,
                'bearish_ob_signal': False,
                'signal_strength': 0.0,
                'supporting_ob_count': 0,
                'nearest_ob_distance': float('inf'),
                'confluence_factors': []
            }

            nearby_blocks = self.get_order_blocks_near_price(current_price, max_distance_pips, pip_value)

            for order_block in nearby_blocks:
                distance = self._get_distance_to_order_block(current_price, order_block)

                if distance <= max_distance:
                    signals['supporting_ob_count'] += 1
                    signals['nearest_ob_distance'] = min(signals['nearest_ob_distance'], distance / pip_value)

                    # Check for potential reversal at order block
                    if order_block.block_type == OrderBlockType.BULLISH:
                        # Price approaching bullish order block from above = potential buy
                        if current_price >= order_block.low_price and current_price <= order_block.high_price:
                            signals['bullish_ob_signal'] = True
                            signals['signal_strength'] = max(signals['signal_strength'], order_block.confidence)
                            signals['confluence_factors'].append(f"bullish_ob_{order_block.strength.value}")

                    elif order_block.block_type == OrderBlockType.BEARISH:
                        # Price approaching bearish order block from below = potential sell
                        if current_price >= order_block.low_price and current_price <= order_block.high_price:
                            signals['bearish_ob_signal'] = True
                            signals['signal_strength'] = max(signals['signal_strength'], order_block.confidence)
                            signals['confluence_factors'].append(f"bearish_ob_{order_block.strength.value}")

            return signals

        except Exception as e:
            self.logger.error(f"Order block signals generation failed: {e}")
            return {}
    
    def _get_distance_to_order_block(self, price: float, order_block: OrderBlock) -> float:
        """Calculate distance from price to order block"""
        if price < order_block.low_price:
            return order_block.low_price - price
        elif price > order_block.high_price:
            return price - order_block.high_price
        else:
            return 0.0  # Inside the order block
    
    def update_order_block_tests(self, df: pd.DataFrame, current_index: int):
        """Update order block test counts when price revisits"""
        try:
            if current_index >= len(df):
                return
            
            current_price = df.iloc[current_index]['close']
            
            for order_block in self.order_blocks:
                if not order_block.still_valid:
                    continue
                
                # Check if price is testing this order block
                if (current_price >= order_block.low_price and 
                    current_price <= order_block.high_price):
                    order_block.tested_count += 1
                    
                    # Invalidate order block after too many tests
                    if order_block.tested_count > 3:
                        order_block.still_valid = False
                        
        except Exception as e:
            self.logger.error(f"Order block test update failed: {e}")
    
    def get_order_block_summary(self) -> Dict:
        """Get summary of detected order blocks"""
        try:
            valid_blocks = [ob for ob in self.order_blocks if ob.still_valid]
            
            return {
                'total_blocks': len(self.order_blocks),
                'valid_blocks': len(valid_blocks),
                'bullish_blocks': len([ob for ob in valid_blocks if ob.block_type == OrderBlockType.BULLISH]),
                'bearish_blocks': len([ob for ob in valid_blocks if ob.block_type == OrderBlockType.BEARISH]),
                'strong_blocks': len([ob for ob in valid_blocks if ob.strength in [OrderBlockStrength.STRONG, OrderBlockStrength.VERY_STRONG]]),
                'avg_confidence': sum(ob.confidence for ob in valid_blocks) / len(valid_blocks) if valid_blocks else 0.0,
                'highest_confidence': max(ob.confidence for ob in valid_blocks) if valid_blocks else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Order block summary failed: {e}")
            return {}