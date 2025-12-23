# core/intelligence/order_flow_analyzer_optimized.py
"""
Enhanced Order Flow Analyzer - Smart Money Phase 1 Implementation
Implements critical fixes for FX trading accuracy:
- Dynamic pip sizing per instrument (JPY, metals, indices)
- Corrected FVG detection logic (3-bar pattern)
- Improved order block detection with displacement context
- Proximity-weighted bias calculation
- Performance optimizations for real-time analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time


class OrderFlowAnalyzer:
    """
    Enhanced Order Flow Analyzer with FX-specific corrections
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration with performance limits
        self.min_ob_size_pips = 8
        self.min_fvg_size_pips = 5
        self.max_lookback_bars = 50  # Limit lookback for performance
        self.max_order_blocks = 10   # Limit number of OBs to track
        self.max_fvgs = 10           # Limit number of FVGs to track
        self.skip_supply_demand = True  # Skip expensive S/D zone calculation
        self.displacement_factor = 1.5  # ATR multiplier for displacement

        # OB Proximity Scoring Configuration
        self.ob_proximity_enabled = True
        self.ob_proximity_threshold_pips = 20  # Max distance for scoring

        self.logger.info("📊 OrderFlowAnalyzer initialized (Enhanced)")
        self.logger.debug(f"   Min OB size: {self.min_ob_size_pips} pips")
        self.logger.debug(f"   Min FVG size: {self.min_fvg_size_pips} pips")
        self.logger.debug(f"   Max lookback: {self.max_lookback_bars} bars")
        self.logger.debug(f"   Displacement factor: {self.displacement_factor}")
    
    def analyze_order_flow(self, df: pd.DataFrame, epic: str, timeframe: str) -> Dict:
        """
        Enhanced order flow analysis with FX-specific fixes
        """
        start_time = time.time()
        
        try:
            # Limit dataframe size for performance
            df_limited = df.tail(self.max_lookback_bars).copy()
            current_price = df_limited['close'].iloc[-1]
            
            # Enhanced order blocks detection with displacement
            order_blocks = self._find_order_blocks_enhanced(df_limited, epic)
            
            # Corrected FVG detection (3-bar pattern)
            fvgs = self._find_fair_value_gaps_corrected(df_limited, epic)
            
            # Skip expensive supply/demand zone calculation by default
            supply_demand_zones = [] if self.skip_supply_demand else self._find_supply_demand_zones_fast(df_limited, epic)
            
            # Enhanced bias calculation with proximity weighting
            order_flow_bias = self._determine_order_flow_bias_weighted(order_blocks, fvgs, current_price)

            # Calculate OB proximity scoring for signal enrichment
            pip_size = self._get_pip_size(epic)
            ob_proximity = self._calculate_ob_proximity_score(
                current_price, order_blocks, fvgs, pip_size
            ) if self.ob_proximity_enabled else {}

            elapsed = time.time() - start_time
            self.logger.debug(f"📊 Enhanced order flow analysis for {epic} completed in {elapsed:.2f}s")

            return {
                'order_blocks': order_blocks[:self.max_order_blocks],
                'fair_value_gaps': fvgs[:self.max_fvgs],
                'supply_demand_zones': supply_demand_zones,
                'order_flow_bias': order_flow_bias,
                'ob_proximity': ob_proximity,
                'analysis_time_seconds': elapsed,
                'current_price': current_price,
                'pip_size': pip_size
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced order flow analysis failed: {e}")
            return {
                'order_blocks': [],
                'fair_value_gaps': [],
                'supply_demand_zones': [],
                'order_flow_bias': 'NEUTRAL',
                'error': str(e)
            }
    
    def _get_pip_size(self, epic: Optional[str] = None) -> float:
        """
        Dynamic pip sizing per instrument - CRITICAL FIX for FX accuracy
        """
        if not epic:
            return 0.0001
        
        e = epic.upper()
        
        # FX pairs - JPY pairs use different pip size
        if any(j in e for j in ["JPY"]):
            return 0.01  # 1 pip = 0.01 for JPY quotes
        
        # Precious metals
        if any(sym in e for sym in ["XAU", "GOLD"]):
            return 0.1  # 1 pip = 0.1 for XAUUSD at most brokers
        if any(sym in e for sym in ["XAG", "SILV"]):
            return 0.01
        
        # Indices & energies (adjust to your broker mapping)
        if any(sym in e for sym in ["US500", "SPX", "DE40", "FTSE", "NAS", "DAX"]):
            return 1.0
        if any(sym in e for sym in ["OIL", "WTI", "BRENT", "USOIL", "UKOIL"]):
            return 0.01
        
        # Default major FX pairs
        return 0.0001
    
    def _find_order_blocks_enhanced(self, df: pd.DataFrame, epic: str) -> List[Dict]:
        """
        Enhanced order block detection with displacement requirement
        Now requires institutional displacement context for better accuracy
        """
        try:
            order_blocks = []
            d = df.copy()
            
            # Calculate ATR for displacement detection
            high, low, close = d['high'], d['low'], d['close']
            tr = np.maximum(high - low, np.maximum(
                abs(high - close.shift(1)), abs(low - close.shift(1))
            ))
            d['atr'] = pd.Series(tr).rolling(14, min_periods=1).mean()
            
            # Calculate candle metrics
            d['body'] = abs(d['close'] - d['open'])
            d['is_bull'] = d['close'] > d['open']
            
            # Find displacement moves (impulsive candles)
            displacement_threshold = d['body'] > (self.displacement_factor * d['atr'])
            
            for i in range(2, len(d)):
                if not displacement_threshold.iloc[i]:
                    continue
                
                # Bullish displacement - look for previous bearish OB
                if d['is_bull'].iloc[i]:
                    # Find last opposite-color candle before displacement
                    j = i - 1
                    while j >= 0 and d['is_bull'].iloc[j]:
                        j -= 1
                    
                    if j >= 0:
                        # Create bullish order block from bearish candle
                        ob_open = float(d['open'].iloc[j])
                        ob_close = float(d['close'].iloc[j])
                        ob_high = max(ob_open, ob_close)
                        ob_low = min(ob_open, ob_close)
                        
                        order_blocks.append({
                            'type': 'BULLISH_OB',
                            'high': ob_high,
                            'low': ob_low,
                            'timestamp': self._timestamp_helper(d, i),
                            'strength': 'MEDIUM',
                            'displacement_index': i,
                            'displacement_size_atr': float(d['body'].iloc[i] / d['atr'].iloc[i])
                        })
                
                # Bearish displacement - look for previous bullish OB
                elif not d['is_bull'].iloc[i]:
                    # Find last opposite-color candle before displacement
                    j = i - 1
                    while j >= 0 and not d['is_bull'].iloc[j]:
                        j -= 1
                    
                    if j >= 0:
                        # Create bearish order block from bullish candle
                        ob_open = float(d['open'].iloc[j])
                        ob_close = float(d['close'].iloc[j])
                        ob_high = max(ob_open, ob_close)
                        ob_low = min(ob_open, ob_close)
                        
                        order_blocks.append({
                            'type': 'BEARISH_OB',
                            'high': ob_high,
                            'low': ob_low,
                            'timestamp': self._timestamp_helper(d, i),
                            'strength': 'MEDIUM',
                            'displacement_index': i,
                            'displacement_size_atr': float(d['body'].iloc[i] / d['atr'].iloc[i])
                        })
            
            # Keep only most recent order blocks
            order_blocks = order_blocks[-self.max_order_blocks:]
            
            self.logger.debug(f"🔍 Found {len(order_blocks)} enhanced order blocks for {epic}")
            return order_blocks
            
        except Exception as e:
            self.logger.error(f"Enhanced order block detection failed: {e}")
            return []
    
    def _find_fair_value_gaps_corrected(self, df: pd.DataFrame, epic: str) -> List[Dict]:
        """
        Corrected FVG detection using proper 3-bar pattern
        FIXED: Now uses correct candle relationships for gap detection
        """
        try:
            fvgs = []
            pip_size = self._get_pip_size(epic)
            
            # Standard 3-bar FVG detection
            # Bullish FVG: low[i+1] > high[i-1] (gap between future low and past high)
            # Bearish FVG: high[i+1] < low[i-1] (gap between future high and past low)
            
            for i in range(1, len(df) - 1):
                current_idx = df.index[i]
                
                # Check for bullish FVG
                if i + 1 < len(df) and i - 1 >= 0:
                    future_low = df['low'].iloc[i + 1]
                    past_high = df['high'].iloc[i - 1]
                    
                    if future_low > past_high:
                        gap_size_pips = (future_low - past_high) / pip_size
                        if gap_size_pips >= self.min_fvg_size_pips:
                            fvgs.append({
                                'type': 'BULLISH_FVG',
                                'top': float(future_low),
                                'bottom': float(past_high),
                                'timestamp': self._timestamp_helper(df, i),
                                'size_pips': float(gap_size_pips),
                                'middle_candle_index': i
                            })
                
                # Check for bearish FVG
                if i + 1 < len(df) and i - 1 >= 0:
                    future_high = df['high'].iloc[i + 1]
                    past_low = df['low'].iloc[i - 1]
                    
                    if future_high < past_low:
                        gap_size_pips = (past_low - future_high) / pip_size
                        if gap_size_pips >= self.min_fvg_size_pips:
                            fvgs.append({
                                'type': 'BEARISH_FVG',
                                'top': float(past_low),
                                'bottom': float(future_high),
                                'timestamp': self._timestamp_helper(df, i),
                                'size_pips': float(gap_size_pips),
                                'middle_candle_index': i
                            })
            
            # Keep only most recent FVGs
            fvgs = fvgs[-self.max_fvgs:]
            
            self.logger.debug(f"🔍 Found {len(fvgs)} corrected FVGs for {epic}")
            return fvgs
            
        except Exception as e:
            self.logger.error(f"Corrected FVG detection failed: {e}")
            return []
    
    def _find_supply_demand_zones_fast(self, df: pd.DataFrame, epic: str) -> List[Dict]:
        """
        Fast supply/demand zone detection - simplified for performance
        """
        if self.skip_supply_demand:
            return []
        
        try:
            zones = []
            
            # Simple swing high/low detection
            df_copy = df.copy()
            df_copy['swing_high'] = (
                (df_copy['high'] > df_copy['high'].shift(1)) & 
                (df_copy['high'] > df_copy['high'].shift(-1))
            )
            df_copy['swing_low'] = (
                (df_copy['low'] < df_copy['low'].shift(1)) & 
                (df_copy['low'] < df_copy['low'].shift(-1))
            )
            
            # Get last few swings only for performance
            swing_highs = df_copy[df_copy['swing_high']].tail(3)
            swing_lows = df_copy[df_copy['swing_low']].tail(3)
            
            # Create supply zones from swing highs
            for idx, row in swing_highs.iterrows():
                zones.append({
                    'type': 'SUPPLY',
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'strength': 'MEDIUM',
                    'timestamp': self._timestamp_helper(df, df.index.get_loc(idx))
                })
            
            # Create demand zones from swing lows
            for idx, row in swing_lows.iterrows():
                zones.append({
                    'type': 'DEMAND',
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'strength': 'MEDIUM',
                    'timestamp': self._timestamp_helper(df, df.index.get_loc(idx))
                })
            
            self.logger.debug(f"🔍 Found {len(zones)} supply/demand zones for {epic}")
            return zones
            
        except Exception as e:
            self.logger.error(f"Fast S/D zone detection failed: {e}")
            return []
    
    def _determine_order_flow_bias_weighted(
        self, 
        order_blocks: List, 
        fvgs: List, 
        current_price: Optional[float] = None
    ) -> str:
        """
        Enhanced bias calculation with proximity weighting
        FIXED: Now considers proximity to current price and uses weighted scoring
        """
        def proximity_weight(level_mid):
            """Closer levels get higher weight"""
            if current_price is None:
                return 1.0
            distance = abs(level_mid - current_price) / current_price  # Relative distance
            return 1.0 / (1.0 + distance * 10)  # Inverse distance weighting
        
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Weight order blocks higher than FVGs
        ob_weight = 2.0
        fvg_weight = 1.0
        
        # Count bullish vs bearish order blocks with proximity weighting
        for ob in order_blocks:
            mid_price = (ob['high'] + ob['low']) / 2.0
            weight = ob_weight * proximity_weight(mid_price)
            
            if 'BULLISH' in ob.get('type', ''):
                bullish_score += weight
            elif 'BEARISH' in ob.get('type', ''):
                bearish_score += weight
        
        # Count bullish vs bearish FVGs with proximity weighting
        for fvg in fvgs:
            mid_price = (fvg['top'] + fvg['bottom']) / 2.0
            weight = fvg_weight * proximity_weight(mid_price)
            
            if 'BULLISH' in fvg.get('type', ''):
                bullish_score += weight
            elif 'BEARISH' in fvg.get('type', ''):
                bearish_score += weight
        
        # Determine bias with threshold
        bias_threshold = 1.0
        if bullish_score > bearish_score + bias_threshold:
            return 'BULLISH'
        elif bearish_score > bullish_score + bias_threshold:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _timestamp_helper(self, df: pd.DataFrame, idx: Optional[int] = None) -> str:
        """
        Centralized timestamp helper for consistent timestamp handling
        """
        try:
            if idx is None:
                idx = -1
            
            if isinstance(df.index, pd.DatetimeIndex):
                return df.index[idx].isoformat()
            elif 'timestamp' in df.columns:
                return pd.to_datetime(df.iloc[idx]['timestamp']).isoformat()
            else:
                return datetime.utcnow().isoformat()
        except Exception:
            return datetime.utcnow().isoformat()
    
    def validate_signal_against_order_flow(
        self, 
        signal: Dict, 
        order_flow_analysis: Dict,
        df: pd.DataFrame
    ) -> Dict:
        """
        Enhanced signal validation with better scoring
        """
        try:
            signal_type = signal.get('signal_type', '').upper()
            order_flow_bias = order_flow_analysis.get('order_flow_bias', 'NEUTRAL')
            current_price = df['close'].iloc[-1] if len(df) > 0 else 0
            
            # Enhanced alignment check
            aligned = False
            base_score = 0.5
            confidence_boost = 0.0
            
            # Primary alignment
            if signal_type in ['BULL', 'BUY'] and order_flow_bias == 'BULLISH':
                aligned = True
                base_score = 0.8
                confidence_boost = 0.2
            elif signal_type in ['BEAR', 'SELL'] and order_flow_bias == 'BEARISH':
                aligned = True
                base_score = 0.8
                confidence_boost = 0.2
            elif order_flow_bias == 'NEUTRAL':
                base_score = 0.6
                confidence_boost = 0.0
            else:
                # Conflicting signals
                base_score = 0.3
                confidence_boost = -0.2
            
            # Check proximity to order flow levels
            nearby_levels = self._find_nearby_levels(
                current_price, 
                order_flow_analysis.get('order_blocks', []),
                order_flow_analysis.get('fair_value_gaps', [])
            )
            
            # Additional context
            validation_reasons = []
            if aligned:
                validation_reasons.append(f"Order flow {order_flow_bias} aligns with {signal_type} signal")
            else:
                validation_reasons.append(f"Order flow {order_flow_bias} conflicts with {signal_type} signal")
            
            if nearby_levels:
                validation_reasons.append(f"Near {len(nearby_levels)} order flow levels")
                confidence_boost += 0.1
            
            final_score = min(1.0, max(0.0, base_score + confidence_boost))
            
            return {
                'order_flow_aligned': aligned,
                'order_flow_score': final_score,
                'validation_reason': "; ".join(validation_reasons),
                'nearby_levels': nearby_levels,
                'confidence_boost': confidence_boost
            }
            
        except Exception as e:
            self.logger.error(f"Order flow validation failed: {e}")
            return {
                'order_flow_aligned': False,
                'order_flow_score': 0.5,
                'validation_reason': f"Validation error: {e}",
                'nearby_levels': [],
                'confidence_boost': 0.0
            }
    
    def _find_nearby_levels(
        self, 
        current_price: float, 
        order_blocks: List[Dict], 
        fvgs: List[Dict],
        proximity_pips: float = 20
    ) -> List[Dict]:
        """
        Find order flow levels near current price
        """
        nearby_levels = []
        pip_size = 0.0001  # Default, should be passed from analysis
        proximity_threshold = proximity_pips * pip_size
        
        try:
            # Check order blocks
            for ob in order_blocks:
                ob_mid = (ob['high'] + ob['low']) / 2.0
                distance = abs(current_price - ob_mid)
                
                if distance <= proximity_threshold:
                    nearby_levels.append({
                        'type': 'ORDER_BLOCK',
                        'level': ob_mid,
                        'distance_pips': distance / pip_size,
                        'ob_type': ob.get('type'),
                        'strength': ob.get('strength', 'MEDIUM')
                    })
            
            # Check FVGs
            for fvg in fvgs:
                fvg_mid = (fvg['top'] + fvg['bottom']) / 2.0
                distance = abs(current_price - fvg_mid)
                
                if distance <= proximity_threshold:
                    nearby_levels.append({
                        'type': 'FAIR_VALUE_GAP',
                        'level': fvg_mid,
                        'distance_pips': distance / pip_size,
                        'fvg_type': fvg.get('type'),
                        'size_pips': fvg.get('size_pips', 0)
                    })
            
            # Sort by proximity
            nearby_levels.sort(key=lambda x: x['distance_pips'])

        except Exception as e:
            self.logger.error(f"Failed to find nearby levels: {e}")

        return nearby_levels

    def _calculate_ob_proximity_score(
        self,
        current_price: float,
        order_blocks: List[Dict],
        fvgs: List[Dict],
        pip_size: float
    ) -> Dict:
        """
        Calculate proximity scores for order blocks and FVGs relative to current price.
        Returns detailed scoring for analytics and confluence.

        Scoring: 1.0 at level, decays to 0 at threshold distance
        """
        try:
            threshold_distance = self.ob_proximity_threshold_pips * pip_size

            # Find nearest bullish and bearish OBs
            nearest_bullish_ob = None
            nearest_bearish_ob = None
            min_bullish_dist = float('inf')
            min_bearish_dist = float('inf')

            for ob in order_blocks:
                ob_mid = (ob['high'] + ob['low']) / 2.0
                distance = abs(current_price - ob_mid)
                distance_pips = distance / pip_size

                if 'BULLISH' in ob.get('type', ''):
                    if distance < min_bullish_dist:
                        min_bullish_dist = distance
                        nearest_bullish_ob = {
                            'high': ob['high'],
                            'low': ob['low'],
                            'mid': ob_mid,
                            'distance_pips': round(distance_pips, 2),
                            'score': round(max(0, 1 - (distance / threshold_distance)), 3),
                            'strength': ob.get('strength', 'MEDIUM'),
                            'displacement_atr': ob.get('displacement_size_atr', 0)
                        }
                elif 'BEARISH' in ob.get('type', ''):
                    if distance < min_bearish_dist:
                        min_bearish_dist = distance
                        nearest_bearish_ob = {
                            'high': ob['high'],
                            'low': ob['low'],
                            'mid': ob_mid,
                            'distance_pips': round(distance_pips, 2),
                            'score': round(max(0, 1 - (distance / threshold_distance)), 3),
                            'strength': ob.get('strength', 'MEDIUM'),
                            'displacement_atr': ob.get('displacement_size_atr', 0)
                        }

            # Find nearest FVGs
            nearest_bullish_fvg = None
            nearest_bearish_fvg = None
            min_bullish_fvg_dist = float('inf')
            min_bearish_fvg_dist = float('inf')

            for fvg in fvgs:
                fvg_mid = (fvg['top'] + fvg['bottom']) / 2.0
                distance = abs(current_price - fvg_mid)
                distance_pips = distance / pip_size

                if 'BULLISH' in fvg.get('type', ''):
                    if distance < min_bullish_fvg_dist:
                        min_bullish_fvg_dist = distance
                        nearest_bullish_fvg = {
                            'top': fvg['top'],
                            'bottom': fvg['bottom'],
                            'mid': fvg_mid,
                            'distance_pips': round(distance_pips, 2),
                            'score': round(max(0, 1 - (distance / threshold_distance)), 3),
                            'size_pips': fvg.get('size_pips', 0)
                        }
                elif 'BEARISH' in fvg.get('type', ''):
                    if distance < min_bearish_fvg_dist:
                        min_bearish_fvg_dist = distance
                        nearest_bearish_fvg = {
                            'top': fvg['top'],
                            'bottom': fvg['bottom'],
                            'mid': fvg_mid,
                            'distance_pips': round(distance_pips, 2),
                            'score': round(max(0, 1 - (distance / threshold_distance)), 3),
                            'size_pips': fvg.get('size_pips', 0)
                        }

            # Calculate overall alignment score
            # Higher score if price is near OBs/FVGs that support potential trades
            bullish_ob_score = nearest_bullish_ob['score'] if nearest_bullish_ob else 0
            bearish_ob_score = nearest_bearish_ob['score'] if nearest_bearish_ob else 0
            bullish_fvg_score = nearest_bullish_fvg['score'] if nearest_bullish_fvg else 0
            bearish_fvg_score = nearest_bearish_fvg['score'] if nearest_bearish_fvg else 0

            # OBs weighted 2x higher than FVGs
            bullish_alignment = (bullish_ob_score * 2 + bullish_fvg_score) / 3
            bearish_alignment = (bearish_ob_score * 2 + bearish_fvg_score) / 3

            # Determine which direction has better proximity support
            if bullish_alignment > bearish_alignment:
                dominant_bias = 'BULLISH'
                alignment_score = bullish_alignment
            elif bearish_alignment > bullish_alignment:
                dominant_bias = 'BEARISH'
                alignment_score = bearish_alignment
            else:
                dominant_bias = 'NEUTRAL'
                alignment_score = max(bullish_alignment, bearish_alignment)

            # Find absolute nearest OB distance for analytics
            nearest_ob_distance = min(
                nearest_bullish_ob['distance_pips'] if nearest_bullish_ob else float('inf'),
                nearest_bearish_ob['distance_pips'] if nearest_bearish_ob else float('inf')
            )
            if nearest_ob_distance == float('inf'):
                nearest_ob_distance = None

            return {
                'nearest_bullish_ob': nearest_bullish_ob,
                'nearest_bearish_ob': nearest_bearish_ob,
                'nearest_bullish_fvg': nearest_bullish_fvg,
                'nearest_bearish_fvg': nearest_bearish_fvg,
                'bullish_alignment_score': round(bullish_alignment, 3),
                'bearish_alignment_score': round(bearish_alignment, 3),
                'dominant_bias': dominant_bias,
                'alignment_score': round(alignment_score, 3),
                'nearest_ob_distance_pips': nearest_ob_distance,
                'total_obs_found': len(order_blocks),
                'total_fvgs_found': len(fvgs)
            }

        except Exception as e:
            self.logger.error(f"OB proximity scoring failed: {e}")
            return {
                'nearest_bullish_ob': None,
                'nearest_bearish_ob': None,
                'nearest_bullish_fvg': None,
                'nearest_bearish_fvg': None,
                'bullish_alignment_score': 0,
                'bearish_alignment_score': 0,
                'dominant_bias': 'NEUTRAL',
                'alignment_score': 0,
                'nearest_ob_distance_pips': None,
                'total_obs_found': 0,
                'total_fvgs_found': 0
            }