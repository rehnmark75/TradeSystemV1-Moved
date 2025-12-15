# core/strategies/helpers/smc_fair_value_gaps.py
"""
Smart Money Concepts - Fair Value Gap (FVG) Detection
Identifies price gaps that represent institutional imbalances

Fair Value Gaps occur when:
- Bullish FVG: Current candle's low > previous candle's high (gap up)
- Bearish FVG: Current candle's high < previous candle's low (gap down)
- Gaps indicate institutional order imbalances
- Often act as support/resistance levels
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum


class FVGType(Enum):
    """Types of Fair Value Gaps"""
    BULLISH = "bullish"
    BEARISH = "bearish"


class FVGStatus(Enum):
    """Fair Value Gap status"""
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    EXPIRED = "expired"


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap"""
    start_index: int
    high_price: float
    low_price: float
    gap_type: FVGType
    gap_size_pips: float
    volume_confirmation: float
    timestamp: pd.Timestamp
    status: FVGStatus = FVGStatus.ACTIVE
    fill_percentage: float = 0.0
    age_bars: int = 0
    touched_count: int = 0
    significance: float = 0.0


class SMCFairValueGaps:
    """Smart Money Concepts Fair Value Gap Detector"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.fair_value_gaps: List[FairValueGap] = []

    def _get_pip_value(self, config: Dict) -> float:
        """Get pip value from config or default based on pair"""
        # Check if pip_value is explicitly provided
        if 'pip_value' in config:
            return config['pip_value']
        # Check pair/epic for JPY
        pair = config.get('pair', config.get('epic', ''))
        if 'JPY' in str(pair).upper():
            return 0.01
        return 0.0001

    def detect_fair_value_gaps(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """
        Detect Fair Value Gaps and add to DataFrame
        
        Args:
            df: OHLCV DataFrame
            config: SMC configuration dictionary
            
        Returns:
            Enhanced DataFrame with FVG analysis
        """
        try:
            df_enhanced = df.copy()
            
            # Initialize FVG columns
            df_enhanced['fvg_bullish'] = False
            df_enhanced['fvg_bearish'] = False
            df_enhanced['fvg_high'] = np.nan
            df_enhanced['fvg_low'] = np.nan
            df_enhanced['fvg_size_pips'] = np.nan
            df_enhanced['fvg_significance'] = 0.0
            
            # Clear previous FVGs
            self.fair_value_gaps = []
            
            # Detect FVGs
            self._scan_for_fvgs(df_enhanced, config)
            
            # Update FVG status
            self._update_fvg_status(df_enhanced, config)
            
            # Mark FVGs in DataFrame
            for fvg in self.fair_value_gaps:
                self._mark_fvg_in_df(df_enhanced, fvg)
            
            # Add FVG zones analysis
            df_enhanced = self._add_fvg_zones(df_enhanced, config)
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Fair Value Gap detection failed: {e}")
            return df
    
    def _scan_for_fvgs(self, df: pd.DataFrame, config: Dict):
        """Scan DataFrame for Fair Value Gap patterns"""
        try:
            min_gap_size_pips = config.get('fvg_min_size', 3)
            pip_value = self._get_pip_value(config)
            min_gap_size = min_gap_size_pips * pip_value  # Convert pips to price
            
            # Need at least 3 candles to detect FVG
            for i in range(2, len(df)):
                current_candle = df.iloc[i]
                previous_candle = df.iloc[i-1]
                prev_prev_candle = df.iloc[i-2]

                # Check for bullish FVG
                bullish_fvg = self._check_bullish_fvg(
                    prev_prev_candle, previous_candle, current_candle, min_gap_size, pip_value
                )
                if bullish_fvg:
                    self.fair_value_gaps.append(bullish_fvg)

                # Check for bearish FVG
                bearish_fvg = self._check_bearish_fvg(
                    prev_prev_candle, previous_candle, current_candle, min_gap_size, pip_value
                )
                if bearish_fvg:
                    self.fair_value_gaps.append(bearish_fvg)
            
        except Exception as e:
            self.logger.error(f"FVG scanning failed: {e}")
    
    def _check_bullish_fvg(
        self,
        candle1: pd.Series,
        candle2: pd.Series,
        candle3: pd.Series,
        min_gap_size: float,
        pip_value: float = 0.0001
    ) -> Optional[FairValueGap]:
        """Check for bullish Fair Value Gap pattern"""
        try:
            # Bullish FVG: candle3.low > candle1.high (gap between candle1 and candle3)
            # candle2 is the middle candle that creates the gap

            gap_low = candle3['low']
            gap_high = candle1['high']

            # Check if there's a valid gap
            if gap_low <= gap_high:
                return None

            gap_size = gap_low - gap_high

            # Check minimum gap size
            if gap_size < min_gap_size:
                return None

            # Volume confirmation (middle candle should have good volume)
            volume = candle2.get('volume', candle2.get('ltv', 1))
            volume_confirmation = self._calculate_volume_confirmation(volume, [candle1, candle2, candle3])

            # Calculate significance
            significance = self._calculate_fvg_significance(
                gap_size, volume_confirmation, 'bullish', [candle1, candle2, candle3], pip_value
            )

            return FairValueGap(
                start_index=candle3.name if hasattr(candle3, 'name') else 0,
                high_price=gap_low,  # Top of the gap
                low_price=gap_high,  # Bottom of the gap
                gap_type=FVGType.BULLISH,
                gap_size_pips=gap_size / pip_value,  # Convert price to pips
                volume_confirmation=volume_confirmation,
                timestamp=candle3.get('start_time', pd.Timestamp.now()) if hasattr(candle3, 'get') else pd.Timestamp.now(),
                significance=significance
            )

        except Exception as e:
            self.logger.error(f"Bullish FVG check failed: {e}")
            return None
    
    def _check_bearish_fvg(
        self,
        candle1: pd.Series,
        candle2: pd.Series,
        candle3: pd.Series,
        min_gap_size: float,
        pip_value: float = 0.0001
    ) -> Optional[FairValueGap]:
        """Check for bearish Fair Value Gap pattern"""
        try:
            # Bearish FVG: candle3.high < candle1.low (gap between candle1 and candle3)
            # candle2 is the middle candle that creates the gap

            gap_high = candle3['high']
            gap_low = candle1['low']

            # Check if there's a valid gap
            if gap_high >= gap_low:
                return None

            gap_size = gap_low - gap_high

            # Check minimum gap size
            if gap_size < min_gap_size:
                return None

            # Volume confirmation (middle candle should have good volume)
            volume = candle2.get('volume', candle2.get('ltv', 1))
            volume_confirmation = self._calculate_volume_confirmation(volume, [candle1, candle2, candle3])

            # Calculate significance
            significance = self._calculate_fvg_significance(
                gap_size, volume_confirmation, 'bearish', [candle1, candle2, candle3], pip_value
            )

            return FairValueGap(
                start_index=candle3.name if hasattr(candle3, 'name') else 0,
                high_price=gap_low,   # Top of the gap
                low_price=gap_high,   # Bottom of the gap
                gap_type=FVGType.BEARISH,
                gap_size_pips=gap_size / pip_value,  # Convert price to pips
                volume_confirmation=volume_confirmation,
                timestamp=candle3.get('start_time', pd.Timestamp.now()) if hasattr(candle3, 'get') else pd.Timestamp.now(),
                significance=significance
            )

        except Exception as e:
            self.logger.error(f"Bearish FVG check failed: {e}")
            return None
    
    def _calculate_volume_confirmation(
        self, 
        current_volume: float, 
        candles: List[pd.Series]
    ) -> float:
        """Calculate volume confirmation for FVG"""
        try:
            if not current_volume or current_volume <= 0:
                return 0.5
            
            # Get volumes from all candles
            volumes = []
            for candle in candles:
                vol = candle.get('volume', candle.get('ltv', 1))
                if vol and vol > 0:
                    volumes.append(vol)
            
            if not volumes:
                return 0.5
            
            avg_volume = sum(volumes) / len(volumes)
            return min(current_volume / avg_volume, 3.0) / 3.0  # Normalize to 0-1
            
        except Exception as e:
            self.logger.error(f"Volume confirmation calculation failed: {e}")
            return 0.5
    
    def _calculate_fvg_significance(
        self,
        gap_size: float,
        volume_confirmation: float,
        gap_type: str,
        candles: List[pd.Series],
        pip_value: float = 0.0001
    ) -> float:
        """Calculate the significance of a Fair Value Gap"""
        try:
            # Gap size factor (larger gaps are more significant)
            gap_size_pips = gap_size / pip_value  # Convert to pips using correct pip value
            size_factor = min(gap_size_pips / 10.0, 1.0)  # Normalize to 0-1 (10 pips = max)
            
            # Volume factor
            volume_factor = volume_confirmation
            
            # Price action factor (strength of the move creating the gap)
            price_action_factor = self._calculate_price_action_strength(candles)
            
            # Combine factors
            significance = (size_factor * 0.4 + 
                           volume_factor * 0.3 + 
                           price_action_factor * 0.3)
            
            return min(max(significance, 0.1), 1.0)
            
        except Exception as e:
            self.logger.error(f"FVG significance calculation failed: {e}")
            return 0.5
    
    def _calculate_price_action_strength(self, candles: List[pd.Series]) -> float:
        """Calculate strength of price action that created the FVG"""
        try:
            if len(candles) < 3:
                return 0.5
            
            # Calculate the total move across the three candles
            first_candle = candles[0]
            last_candle = candles[-1]
            
            total_move = abs(last_candle['close'] - first_candle['close'])
            
            # Calculate average body size
            body_sizes = []
            for candle in candles:
                body_size = abs(candle['close'] - candle['open'])
                body_sizes.append(body_size)
            
            avg_body_size = sum(body_sizes) / len(body_sizes) if body_sizes else 0.0001
            
            # Strength is ratio of total move to average body size
            strength = total_move / avg_body_size if avg_body_size > 0 else 1.0
            
            return min(strength / 3.0, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            self.logger.error(f"Price action strength calculation failed: {e}")
            return 0.5
    
    def _update_fvg_status(self, df: pd.DataFrame, config: Dict):
        """Update the status of all Fair Value Gaps"""
        try:
            max_age = config.get('fvg_max_age', 20)
            fill_threshold = config.get('fvg_fill_threshold', 0.5)
            
            current_index = len(df) - 1
            
            for fvg in self.fair_value_gaps:
                # Update age
                fvg.age_bars = current_index - fvg.start_index
                
                # Check if expired by age
                if fvg.age_bars > max_age:
                    fvg.status = FVGStatus.EXPIRED
                    continue
                
                # Check fill status based on recent price action
                fill_percentage = self._calculate_fvg_fill_percentage(fvg, df, current_index)
                fvg.fill_percentage = fill_percentage
                
                if fill_percentage >= 1.0:
                    fvg.status = FVGStatus.FILLED
                elif fill_percentage >= fill_threshold:
                    fvg.status = FVGStatus.PARTIALLY_FILLED
                else:
                    fvg.status = FVGStatus.ACTIVE
                    
        except Exception as e:
            self.logger.error(f"FVG status update failed: {e}")
    
    def _calculate_fvg_fill_percentage(
        self, 
        fvg: FairValueGap, 
        df: pd.DataFrame, 
        current_index: int
    ) -> float:
        """Calculate how much of the FVG has been filled"""
        try:
            gap_size = fvg.high_price - fvg.low_price
            if gap_size <= 0:
                return 1.0
            
            # Check price action since FVG was created
            max_fill = 0.0
            
            for i in range(fvg.start_index + 1, min(current_index + 1, len(df))):
                candle = df.iloc[i]
                candle_high = candle['high']
                candle_low = candle['low']
                
                if fvg.gap_type == FVGType.BULLISH:
                    # For bullish FVG, check how much price has retraced into the gap
                    if candle_low <= fvg.high_price:
                        fill_amount = max(0, fvg.high_price - max(candle_low, fvg.low_price))
                        fill_percentage = fill_amount / gap_size
                        max_fill = max(max_fill, fill_percentage)
                        
                        # Count touches
                        if candle_low <= fvg.high_price and candle_high >= fvg.low_price:
                            fvg.touched_count += 1
                
                elif fvg.gap_type == FVGType.BEARISH:
                    # For bearish FVG, check how much price has retraced into the gap
                    if candle_high >= fvg.low_price:
                        fill_amount = max(0, min(candle_high, fvg.high_price) - fvg.low_price)
                        fill_percentage = fill_amount / gap_size
                        max_fill = max(max_fill, fill_percentage)
                        
                        # Count touches
                        if candle_high >= fvg.low_price and candle_low <= fvg.high_price:
                            fvg.touched_count += 1
            
            return min(max_fill, 1.0)
            
        except Exception as e:
            self.logger.error(f"FVG fill percentage calculation failed: {e}")
            return 0.0
    
    def _mark_fvg_in_df(self, df: pd.DataFrame, fvg: FairValueGap):
        """Mark Fair Value Gap in DataFrame"""
        try:
            start_idx = fvg.start_index
            
            if start_idx < len(df):
                if fvg.gap_type == FVGType.BULLISH:
                    df.iloc[start_idx, df.columns.get_loc('fvg_bullish')] = True
                else:
                    df.iloc[start_idx, df.columns.get_loc('fvg_bearish')] = True
                
                df.iloc[start_idx, df.columns.get_loc('fvg_high')] = fvg.high_price
                df.iloc[start_idx, df.columns.get_loc('fvg_low')] = fvg.low_price
                df.iloc[start_idx, df.columns.get_loc('fvg_size_pips')] = fvg.gap_size_pips
                df.iloc[start_idx, df.columns.get_loc('fvg_significance')] = fvg.significance
                
        except Exception as e:
            self.logger.error(f"FVG marking in DataFrame failed: {e}")
    
    def _add_fvg_zones(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Add FVG zone analysis to DataFrame"""
        try:
            pip_value = self._get_pip_value(config)

            df['in_bullish_fvg'] = False
            df['in_bearish_fvg'] = False
            df['nearest_fvg_distance'] = np.nan
            df['active_fvg_count'] = 0

            for i, row in df.iterrows():
                current_price = row['close']

                nearest_distance = float('inf')
                active_fvg_count = 0

                for fvg in self.fair_value_gaps:
                    if fvg.status not in [FVGStatus.ACTIVE, FVGStatus.PARTIALLY_FILLED]:
                        continue

                    active_fvg_count += 1

                    # Check if price is in FVG zone
                    if fvg.low_price <= current_price <= fvg.high_price:
                        if fvg.gap_type == FVGType.BULLISH:
                            df.at[i, 'in_bullish_fvg'] = True
                        else:
                            df.at[i, 'in_bearish_fvg'] = True
                        nearest_distance = 0
                    else:
                        # Calculate distance to FVG
                        if current_price < fvg.low_price:
                            distance = fvg.low_price - current_price
                        else:
                            distance = current_price - fvg.high_price

                        nearest_distance = min(nearest_distance, distance)

                df.at[i, 'active_fvg_count'] = active_fvg_count
                if nearest_distance != float('inf'):
                    df.at[i, 'nearest_fvg_distance'] = nearest_distance / pip_value  # Convert to pips

            return df
            
        except Exception as e:
            self.logger.error(f"FVG zones addition failed: {e}")
            return df
    
    def get_fvgs_near_price(
        self,
        price: float,
        max_distance_pips: float = 10.0,
        only_active: bool = True,
        pip_value: float = 0.0001
    ) -> List[FairValueGap]:
        """Get Fair Value Gaps near a specific price"""
        try:
            max_distance = max_distance_pips * pip_value  # Convert pips to price
            nearby_fvgs = []
            
            for fvg in self.fair_value_gaps:
                if only_active and fvg.status not in [FVGStatus.ACTIVE, FVGStatus.PARTIALLY_FILLED]:
                    continue
                
                # Check distance to FVG
                if fvg.low_price - max_distance <= price <= fvg.high_price + max_distance:
                    nearby_fvgs.append(fvg)
            
            # Sort by significance
            nearby_fvgs.sort(key=lambda x: x.significance, reverse=True)
            return nearby_fvgs
            
        except Exception as e:
            self.logger.error(f"FVGs near price search failed: {e}")
            return []
    
    def get_fvg_signals(
        self,
        df: pd.DataFrame,
        current_index: int,
        config: Dict
    ) -> Dict:
        """Get FVG-based trading signals"""
        try:
            if current_index >= len(df):
                return {}

            pip_value = self._get_pip_value(config)
            current_price = df.iloc[current_index]['close']

            signals = {
                'bullish_fvg_signal': False,
                'bearish_fvg_signal': False,
                'fvg_confluence_count': 0,
                'nearest_fvg_distance': float('inf'),
                'fvg_strength': 0.0,
                'confluence_factors': []
            }

            nearby_fvgs = self.get_fvgs_near_price(
                current_price, config.get('max_distance_to_zone', 10), pip_value=pip_value
            )

            for fvg in nearby_fvgs:
                # Check if price is approaching or in FVG
                if fvg.low_price <= current_price <= fvg.high_price:
                    # Price is in the FVG
                    signals['fvg_confluence_count'] += 1
                    signals['fvg_strength'] = max(signals['fvg_strength'], fvg.significance)
                    signals['nearest_fvg_distance'] = 0

                    if fvg.gap_type == FVGType.BULLISH:
                        signals['bullish_fvg_signal'] = True
                        signals['confluence_factors'].append(f"bullish_fvg_{fvg.gap_size_pips:.1f}pips")
                    else:
                        signals['bearish_fvg_signal'] = True
                        signals['confluence_factors'].append(f"bearish_fvg_{fvg.gap_size_pips:.1f}pips")
                else:
                    # Calculate distance in pips
                    if current_price < fvg.low_price:
                        distance = (fvg.low_price - current_price) / pip_value
                    else:
                        distance = (current_price - fvg.high_price) / pip_value

                    signals['nearest_fvg_distance'] = min(signals['nearest_fvg_distance'], distance)

            return signals
            
        except Exception as e:
            self.logger.error(f"FVG signals generation failed: {e}")
            return {}
    
    def get_fvg_summary(self) -> Dict:
        """Get summary of detected Fair Value Gaps"""
        try:
            active_fvgs = [fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.ACTIVE]
            filled_fvgs = [fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.FILLED]
            
            return {
                'total_fvgs': len(self.fair_value_gaps),
                'active_fvgs': len(active_fvgs),
                'filled_fvgs': len(filled_fvgs),
                'bullish_fvgs': len([fvg for fvg in active_fvgs if fvg.gap_type == FVGType.BULLISH]),
                'bearish_fvgs': len([fvg for fvg in active_fvgs if fvg.gap_type == FVGType.BEARISH]),
                'avg_gap_size': sum(fvg.gap_size_pips for fvg in active_fvgs) / len(active_fvgs) if active_fvgs else 0.0,
                'avg_significance': sum(fvg.significance for fvg in active_fvgs) / len(active_fvgs) if active_fvgs else 0.0,
                'largest_gap': max(fvg.gap_size_pips for fvg in active_fvgs) if active_fvgs else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"FVG summary failed: {e}")
            return {}