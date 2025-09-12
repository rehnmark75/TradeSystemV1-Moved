# core/intelligence/market_structure_analyzer.py
"""
Enhanced Market Structure Analyzer - Smart Money Phase 1 Implementation
Implements critical fixes for accurate SMC structure analysis:
- ATR-normalized swing strength calculation (maintains 0-1 range)
- BOS detection using latest structural pivots (not absolute extremes)
- ChoCh detection aligned to SMC definitions (break of internal pivots)
- Nearest target identification (not absolute max/min)
- Dynamic pip sizing for different instruments
- Close-based confirmation requirements
- Performance optimizations for real-time analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from configdata import config


class StructureType(Enum):
    """Market structure types"""
    HIGHER_HIGH = "HH"
    LOWER_LOW = "LL"
    HIGHER_LOW = "HL" 
    LOWER_HIGH = "LH"
    EQUAL_HIGH = "EH"
    EQUAL_LOW = "EL"


class MarketBias(Enum):
    """Market bias types"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    RANGING = "RANGING"


@dataclass
class SwingPoint:
    """Represents a swing high or low point"""
    index: int
    price: float
    timestamp: datetime
    swing_type: str  # 'high' or 'low'
    strength: float  # How strong the swing is (0-1)
    confirmed: bool = False


@dataclass
class StructureEvent:
    """Represents a market structure event"""
    event_type: str  # 'BOS', 'ChoCh', 'MSB'
    direction: str  # 'bullish' or 'bearish'
    price: float
    timestamp: datetime
    confidence: float
    previous_structure: Optional[SwingPoint]
    current_structure: SwingPoint
    significance: float  # 0-1, how significant this structure change is


class MarketStructureAnalyzer:
    """
    Enhanced Smart Money Market Structure Analysis
    
    Key Improvements:
    - ATR-normalized swing strength for better resolution
    - BOS detection using latest confirmed swings (not absolute extremes)
    - ChoCh detection with proper internal pivot breaks
    - Dynamic pip sizing for different instruments
    - Nearest target identification for tradeable levels
    - Performance optimizations with configurable limits
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.swing_lookback = getattr(config, 'STRUCTURE_SWING_LOOKBACK', 5)
        self.min_swing_strength = getattr(config, 'STRUCTURE_MIN_SWING_STRENGTH', 0.3)
        self.bos_confirmation_pips = getattr(config, 'STRUCTURE_BOS_CONFIRMATION_PIPS', 5)
        self.choch_lookback_periods = getattr(config, 'STRUCTURE_CHOCH_LOOKBACK', 20)
        self.require_close_confirmation = getattr(config, 'STRUCTURE_REQUIRE_CLOSE', True)
        
        # Performance limits
        self.max_structure_events = 20
        self.max_swing_points = 30
        
        # State tracking
        self.structure_history = {}  # epic -> analysis dict
        self.swing_points_cache = {}  # epic -> list of swing points
        
        self.logger.info("🏗️ MarketStructureAnalyzer initialized (Enhanced)")
        self.logger.debug(f"   Swing lookback: {self.swing_lookback}")
        self.logger.debug(f"   BOS confirmation: {self.bos_confirmation_pips} pips")
        self.logger.debug(f"   Close confirmation: {self.require_close_confirmation}")
    
    def analyze_market_structure(
        self, 
        df: pd.DataFrame, 
        epic: str,
        timeframe: str = '5m'
    ) -> Dict:
        """
        Enhanced market structure analysis with FX-specific improvements
        """
        try:
            if len(df) < self.choch_lookback_periods:
                return self._create_neutral_structure_analysis(epic)
            
            # 1. Identify swing points with ATR-normalized strength
            swing_points = self._identify_swing_points_enhanced(df, epic)
            
            # 2. Detect structure events with improved BOS/ChoCh logic
            structure_events = self._detect_structure_events_enhanced(df, swing_points, epic)
            
            # 3. Determine current market bias
            current_bias = self._determine_market_bias(swing_points, structure_events)
            
            # 4. Calculate structure confidence score
            structure_score = self._calculate_structure_confidence(
                swing_points, structure_events, current_bias
            )
            
            # 5. Identify next structure targets (nearest levels)
            next_targets = self._identify_structure_targets_enhanced(swing_points, current_bias, df)
            
            # 6. Cache results for validation functions
            analysis = {
                'epic': epic,
                'timeframe': timeframe,
                'timestamp': datetime.now(),
                'current_bias': current_bias.value,
                'structure_score': structure_score,
                'swing_points': [self._swing_point_to_dict(sp) for sp in swing_points[-self.max_swing_points:]],
                'structure_events': [self._structure_event_to_dict(se) for se in structure_events[-self.max_structure_events:]],
                'next_targets': next_targets,
                'analysis_summary': self._create_structure_summary(
                    current_bias, structure_score, structure_events
                ),
                'pip_size': self._get_pip_size(epic)
            }
            
            # Cache the full analysis for validation functions
            self._cache_structure_analysis(epic, analysis)
            
            self.logger.debug(f"🏗️ Enhanced structure analysis for {epic}: "
                            f"{current_bias.value} bias, score: {structure_score:.3f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced market structure analysis failed for {epic}: {e}")
            return self._create_neutral_structure_analysis(epic)
    
    def _get_pip_size(self, epic: Optional[str] = None) -> float:
        """
        Dynamic pip sizing per instrument - CRITICAL for accurate BOS/ChoCh detection
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
        
        # Indices & energies
        if any(sym in e for sym in ["US500", "SPX", "DE40", "FTSE", "NAS", "DAX"]):
            return 1.0
        if any(sym in e for sym in ["OIL", "WTI", "BRENT", "USOIL", "UKOIL"]):
            return 0.01
        
        # Default major FX pairs
        return 0.0001
    
    def _identify_swing_points_enhanced(self, df: pd.DataFrame, epic: str) -> List[SwingPoint]:
        """Enhanced swing point identification with ATR-normalized strength"""
        swing_points = []
        
        try:
            # Calculate ATR for strength normalization
            high, low, close = df['high'], df['low'], df['close']
            tr = np.maximum(high - low, np.maximum(
                abs(high - close.shift(1)), abs(low - close.shift(1))
            ))
            atr = pd.Series(tr).rolling(14, min_periods=1).mean()
            
            # Use rolling windows to find local highs and lows
            window_size = self.swing_lookback * 2 + 1
            high_rolling = df['high'].rolling(window=window_size, center=True)
            low_rolling = df['low'].rolling(window=window_size, center=True)
            
            # Find swing highs (local maxima) with stricter conditions
            swing_highs = (
                (df['high'] == high_rolling.max()) & 
                (df['high'].shift(1) < df['high']) & 
                (df['high'].shift(-1) < df['high'])
            )
            
            # Find swing lows (local minima) with stricter conditions
            swing_lows = (
                (df['low'] == low_rolling.min()) & 
                (df['low'].shift(1) > df['low']) & 
                (df['low'].shift(-1) > df['low'])
            )
            
            # Process swing highs
            for idx in df[swing_highs].index:
                loc = df.index.get_loc(idx)
                if loc < self.swing_lookback or loc >= len(df) - self.swing_lookback:
                    continue  # Skip edges and recent unconfirmed swings
                
                strength = self._calculate_swing_strength_atr(df, atr, loc, 'high')
                if strength >= self.min_swing_strength:
                    swing_points.append(SwingPoint(
                        index=loc,
                        price=df.iloc[loc]['high'],
                        timestamp=self._timestamp_at(df, loc),
                        swing_type='high',
                        strength=strength,
                        confirmed=True
                    ))
            
            # Process swing lows
            for idx in df[swing_lows].index:
                loc = df.index.get_loc(idx)
                if loc < self.swing_lookback or loc >= len(df) - self.swing_lookback:
                    continue
                
                strength = self._calculate_swing_strength_atr(df, atr, loc, 'low')
                if strength >= self.min_swing_strength:
                    swing_points.append(SwingPoint(
                        index=loc,
                        price=df.iloc[loc]['low'],
                        timestamp=self._timestamp_at(df, loc),
                        swing_type='low',
                        strength=strength,
                        confirmed=True
                    ))
            
            # Sort by index (chronological order)
            swing_points.sort(key=lambda x: x.index)
            
            self.logger.debug(f"🔍 Found {len(swing_points)} enhanced swing points for {epic}")
            return swing_points
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced swing point identification failed for {epic}: {e}")
            return []
    
    def _calculate_swing_strength_atr(
        self, 
        df: pd.DataFrame, 
        atr: pd.Series, 
        idx: int, 
        swing_type: str
    ) -> float:
        """
        ATR-normalized swing strength calculation - FIXED for better resolution
        """
        try:
            lookback = min(self.swing_lookback, idx)
            lookahead = min(self.swing_lookback, len(df) - idx - 1)
            
            if lookback == 0 or lookahead == 0:
                return 0.0
            
            # Get ATR at this point for normalization
            current_atr = atr.iloc[idx]
            if current_atr <= 0:
                current_atr = 1e-8  # Avoid division by zero
            
            if swing_type == 'high':
                center_price = df.iloc[idx]['high']
                left_max = df.iloc[idx - lookback:idx]['high'].max()
                right_max = df.iloc[idx + 1:idx + lookahead + 1]['high'].max()
                
                # ATR-normalized separation
                left_separation = max(0, center_price - left_max)
                right_separation = max(0, center_price - right_max)
                max_separation = max(left_separation, right_separation)
                
            else:  # swing_type == 'low'
                center_price = df.iloc[idx]['low']
                left_min = df.iloc[idx - lookback:idx]['low'].min()
                right_min = df.iloc[idx + 1:idx + lookahead + 1]['low'].min()
                
                # ATR-normalized separation
                left_separation = max(0, left_min - center_price)
                right_separation = max(0, right_min - center_price)
                max_separation = max(left_separation, right_separation)
            
            # Normalize by ATR and clip to 0-1 range
            strength = float(np.clip(max_separation / current_atr, 0.0, 1.0))
            return strength
            
        except Exception as e:
            self.logger.error(f"❌ ATR swing strength calculation failed: {e}")
            return 0.0
    
    def _detect_structure_events_enhanced(
        self, 
        df: pd.DataFrame, 
        swing_points: List[SwingPoint], 
        epic: str
    ) -> List[StructureEvent]:
        """Enhanced structure event detection with proper SMC logic"""
        structure_events = []
        
        if len(swing_points) < 4:
            return structure_events
        
        try:
            current_price = df.iloc[-1]['close']
            
            # Separate highs and lows, keep chronological order
            swing_highs = [sp for sp in swing_points if sp.swing_type == 'high']
            swing_lows = [sp for sp in swing_points if sp.swing_type == 'low']
            
            # Enhanced BOS detection (using latest structural pivots)
            bos_events = self._detect_break_of_structure_enhanced(
                df, swing_highs, swing_lows, epic
            )
            structure_events.extend(bos_events)
            
            # Enhanced ChoCh detection (aligned to SMC definitions)
            choch_events = self._detect_change_of_character_enhanced(
                swing_points, current_price, df
            )
            structure_events.extend(choch_events)
            
            # Sort events by timestamp
            structure_events.sort(key=lambda x: x.timestamp)
            
            self.logger.debug(f"🔍 Found {len(structure_events)} enhanced structure events for {epic}")
            return structure_events
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced structure event detection failed for {epic}: {e}")
            return []
    
    def _detect_break_of_structure_enhanced(
        self, 
        df: pd.DataFrame,
        swing_highs: List[SwingPoint], 
        swing_lows: List[SwingPoint],
        epic: str
    ) -> List[StructureEvent]:
        """
        Enhanced BOS detection using latest structural pivots (not absolute extremes)
        FIXED: Now uses most recent confirmed swings and close-based confirmation
        """
        bos_events = []
        
        if len(df) < 2:
            return bos_events
        
        try:
            close = df['close'].iloc[-1]
            timestamp = self._timestamp_at(df, -1)
            pip_size = self._get_pip_size(epic)
            confirmation_threshold = self.bos_confirmation_pips * pip_size
            
            # Use latest (chronological) confirmed swings - NOT absolute extremes
            last_high = swing_highs[-1] if swing_highs else None
            last_low = swing_lows[-1] if swing_lows else None
            
            # Bullish BOS: close breaks above latest structural high
            if last_high:
                if self.require_close_confirmation:
                    condition = close > last_high.price + confirmation_threshold
                    price_reference = close
                else:
                    condition = df['high'].iloc[-1] > last_high.price + confirmation_threshold
                    price_reference = df['high'].iloc[-1]
                
                if condition:
                    confidence = min(1.0, (price_reference - last_high.price) / (last_high.price * 0.002))
                    bos_events.append(StructureEvent(
                        event_type='BOS',
                        direction='bullish',
                        price=float(price_reference),
                        timestamp=timestamp,
                        confidence=confidence,
                        previous_structure=last_high,
                        current_structure=SwingPoint(
                            index=len(df)-1,
                            price=float(price_reference),
                            timestamp=timestamp,
                            swing_type='high',
                            strength=0.8
                        ),
                        significance=last_high.strength
                    ))
            
            # Bearish BOS: close breaks below latest structural low
            if last_low:
                if self.require_close_confirmation:
                    condition = close < last_low.price - confirmation_threshold
                    price_reference = close
                else:
                    condition = df['low'].iloc[-1] < last_low.price - confirmation_threshold
                    price_reference = df['low'].iloc[-1]
                
                if condition:
                    confidence = min(1.0, (last_low.price - price_reference) / (last_low.price * 0.002))
                    bos_events.append(StructureEvent(
                        event_type='BOS',
                        direction='bearish',
                        price=float(price_reference),
                        timestamp=timestamp,
                        confidence=confidence,
                        previous_structure=last_low,
                        current_structure=SwingPoint(
                            index=len(df)-1,
                            price=float(price_reference),
                            timestamp=timestamp,
                            swing_type='low',
                            strength=0.8
                        ),
                        significance=last_low.strength
                    ))
            
            return bos_events
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced BOS detection failed: {e}")
            return []
    
    def _detect_change_of_character_enhanced(
        self, 
        swing_points: List[SwingPoint], 
        current_price: float,
        df: pd.DataFrame
    ) -> List[StructureEvent]:
        """
        Enhanced ChoCh detection aligned to SMC definitions
        FIXED: Uses most recent internal pivots (LH/HL) and close requirement
        """
        choch_events = []
        
        if len(swing_points) < 4:
            return choch_events
        
        try:
            timestamp = self._timestamp_at(df, -1)
            close = df['close'].iloc[-1]
            
            highs = [s for s in swing_points if s.swing_type == 'high']
            lows = [s for s in swing_points if s.swing_type == 'low']
            
            # Determine prior bias from swing pattern
            pattern = self._analyze_swing_pattern(swing_points[-6:])
            
            # Bullish ChoCh: break above most recent LH after bearish trend
            if pattern['trend'] == 'bearish' and len(highs) >= 2:
                last_lh = highs[-1]  # Most recent Lower High
                if close > last_lh.price:
                    choch_events.append(StructureEvent(
                        event_type='ChoCh',
                        direction='bullish',
                        price=float(close),
                        timestamp=timestamp,
                        confidence=0.75,
                        previous_structure=highs[-2] if len(highs) >= 2 else None,
                        current_structure=SwingPoint(
                            index=len(df)-1,
                            price=float(close),
                            timestamp=timestamp,
                            swing_type='high',
                            strength=0.7
                        ),
                        significance=0.7
                    ))
            
            # Bearish ChoCh: break below most recent HL after bullish trend
            elif pattern['trend'] == 'bullish' and len(lows) >= 2:
                last_hl = lows[-1]  # Most recent Higher Low
                if close < last_hl.price:
                    choch_events.append(StructureEvent(
                        event_type='ChoCh',
                        direction='bearish',
                        price=float(close),
                        timestamp=timestamp,
                        confidence=0.75,
                        previous_structure=lows[-2] if len(lows) >= 2 else None,
                        current_structure=SwingPoint(
                            index=len(df)-1,
                            price=float(close),
                            timestamp=timestamp,
                            swing_type='low',
                            strength=0.7
                        ),
                        significance=0.7
                    ))
            
            return choch_events
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced ChoCh detection failed: {e}")
            return []
    
    def _analyze_swing_pattern(self, swing_points: List[SwingPoint]) -> Dict:
        """Analyze pattern of swing points to determine trend"""
        if len(swing_points) < 3:
            return {'trend': 'neutral', 'confidence': 0.0}
        
        try:
            highs = [sp for sp in swing_points if sp.swing_type == 'high']
            lows = [sp for sp in swing_points if sp.swing_type == 'low']
            
            # Check for higher highs and higher lows (bullish)
            if len(highs) >= 2 and len(lows) >= 2:
                hh_pattern = all(highs[i].price > highs[i-1].price for i in range(1, len(highs)))
                hl_pattern = all(lows[i].price > lows[i-1].price for i in range(1, len(lows)))
                
                if hh_pattern and hl_pattern:
                    return {'trend': 'bullish', 'confidence': 0.8}
                
                # Check for lower lows and lower highs (bearish)
                ll_pattern = all(lows[i].price < lows[i-1].price for i in range(1, len(lows)))
                lh_pattern = all(highs[i].price < highs[i-1].price for i in range(1, len(highs)))
                
                if ll_pattern and lh_pattern:
                    return {'trend': 'bearish', 'confidence': 0.8}
            
            return {'trend': 'neutral', 'confidence': 0.5}
            
        except Exception as e:
            self.logger.error(f"❌ Swing pattern analysis failed: {e}")
            return {'trend': 'neutral', 'confidence': 0.0}
    
    def _determine_market_bias(
        self, 
        swing_points: List[SwingPoint], 
        structure_events: List[StructureEvent]
    ) -> MarketBias:
        """Enhanced market bias determination"""
        try:
            if not swing_points and not structure_events:
                return MarketBias.NEUTRAL
            
            # Use bar-based recency instead of wall-clock time
            recent_events = structure_events[-3:] if len(structure_events) >= 3 else structure_events
            
            bullish_events = len([e for e in recent_events if e.direction == 'bullish'])
            bearish_events = len([e for e in recent_events if e.direction == 'bearish'])
            
            # Check swing point pattern
            pattern_analysis = self._analyze_swing_pattern(swing_points[-6:])
            
            # Combine event analysis with pattern analysis
            if bullish_events > bearish_events and pattern_analysis['trend'] == 'bullish':
                return MarketBias.BULLISH
            elif bearish_events > bullish_events and pattern_analysis['trend'] == 'bearish':
                return MarketBias.BEARISH
            elif pattern_analysis['trend'] == 'neutral':
                return MarketBias.RANGING
            else:
                return MarketBias.NEUTRAL
                
        except Exception as e:
            self.logger.error(f"❌ Market bias determination failed: {e}")
            return MarketBias.NEUTRAL
    
    def _calculate_structure_confidence(
        self, 
        swing_points: List[SwingPoint], 
        structure_events: List[StructureEvent],
        bias: MarketBias
    ) -> float:
        """Calculate confidence score for the current market structure"""
        try:
            if not swing_points:
                return 0.0
            
            confidence_factors = []
            
            # Factor 1: Quality of recent swing points
            if swing_points:
                recent_swings = swing_points[-5:] if len(swing_points) >= 5 else swing_points
                avg_swing_strength = np.mean([sp.strength for sp in recent_swings])
                confidence_factors.append(avg_swing_strength)
            
            # Factor 2: Consistency of structure events
            if structure_events:
                recent_events = structure_events[-3:]
                if recent_events:
                    bias_direction = bias.value.lower()
                    consistent_events = [e for e in recent_events if e.direction == bias_direction]
                    event_consistency = len(consistent_events) / len(recent_events)
                    confidence_factors.append(event_consistency)
            
            # Factor 3: Number of confirmation points (bar-based)
            confirmation_factor = min(1.0, len(swing_points) / 10)
            confidence_factors.append(confirmation_factor)
            
            # Factor 4: Recent structure strength
            if structure_events:
                recent_significance = np.mean([e.significance for e in structure_events[-2:]])
                confidence_factors.append(recent_significance)
            
            # Calculate weighted average
            if confidence_factors:
                return np.mean(confidence_factors)
            else:
                return 0.5  # Neutral confidence
                
        except Exception as e:
            self.logger.error(f"❌ Structure confidence calculation failed: {e}")
            return 0.0
    
    def _identify_structure_targets_enhanced(
        self, 
        swing_points: List[SwingPoint], 
        bias: MarketBias,
        df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Enhanced target identification - nearest levels, not absolute extremes
        FIXED: Now finds nearest above/below current price for tradeable levels
        """
        targets = {
            'next_resistance': None,
            'next_support': None,
            'key_levels': []
        }
        
        try:
            if not swing_points or df is None:
                return targets
            
            current_price = df['close'].iloc[-1]
            highs = sorted([sp.price for sp in swing_points if sp.swing_type == 'high'])
            lows = sorted([sp.price for sp in swing_points if sp.swing_type == 'low'])
            
            # Find nearest resistance above current price
            above_levels = [h for h in highs if h > current_price]
            if above_levels:
                targets['next_resistance'] = min(above_levels)
            
            # Find nearest support below current price
            below_levels = [l for l in lows if l < current_price]
            if below_levels:
                targets['next_support'] = max(below_levels)
            
            # Identify key levels based on strength and proximity
            key_levels = []
            now = datetime.now()
            for sp in swing_points[-10:]:
                if sp.strength > 0.6:  # Strong swing points only
                    # Calculate age in bars rather than wall-clock time
                    bar_age = len(df) - sp.index if hasattr(sp, 'index') else None
                    
                    key_levels.append({
                        'price': sp.price,
                        'type': sp.swing_type,
                        'strength': sp.strength,
                        'age_bars': bar_age,
                        'distance_from_current': abs(sp.price - current_price) / current_price
                    })
            
            # Sort by strength, then by proximity
            targets['key_levels'] = sorted(
                key_levels, 
                key=lambda x: (x['strength'], -x['distance_from_current']), 
                reverse=True
            )[:5]
            
            return targets
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced structure target identification failed: {e}")
            return targets
    
    def validate_signal_against_structure(
        self, 
        signal: Dict,
        structure_analysis: Dict,
        df: pd.DataFrame
    ) -> Dict:
        """
        Enhanced signal validation against market structure
        FIXED: Now works with cached analysis from analyze_market_structure
        """
        try:
            signal_type = signal.get('signal_type', '').upper()
            current_bias = structure_analysis.get('current_bias', 'NEUTRAL')
            structure_score = structure_analysis.get('structure_score', 0.5)
            
            # Check alignment
            if signal_type in ['BUY', 'BULL']:
                structure_aligned = current_bias in ['BULLISH', 'NEUTRAL']
                if structure_aligned:
                    validation_reason = f"Bullish signal aligns with {current_bias} structure"
                    recommended_action = 'PROCEED'
                else:
                    validation_reason = f"Bullish signal conflicts with {current_bias} structure"
                    recommended_action = 'PROCEED_WITH_CAUTION' if structure_score < 0.7 else 'REJECT'
            
            else:  # SELL/BEAR
                structure_aligned = current_bias in ['BEARISH', 'NEUTRAL']
                if structure_aligned:
                    validation_reason = f"Bearish signal aligns with {current_bias} structure"
                    recommended_action = 'PROCEED'
                else:
                    validation_reason = f"Bearish signal conflicts with {current_bias} structure"
                    recommended_action = 'PROCEED_WITH_CAUTION' if structure_score < 0.7 else 'REJECT'
            
            return {
                'structure_aligned': structure_aligned,
                'structure_score': structure_score,
                'current_bias': current_bias,
                'validation_reason': validation_reason,
                'recommended_action': recommended_action,
                'nearest_resistance': structure_analysis.get('next_targets', {}).get('next_resistance'),
                'nearest_support': structure_analysis.get('next_targets', {}).get('next_support')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced signal structure validation failed: {e}")
            return {
                'structure_aligned': True,  # Default to allow
                'structure_score': 0.5,
                'validation_reason': f'Validation error: {e}',
                'recommended_action': 'PROCEED_WITH_CAUTION'
            }
    
    def _timestamp_at(self, df: pd.DataFrame, idx: Optional[int] = None):
        """Centralized timestamp helper for consistent timestamp handling"""
        try:
            if idx is None:
                idx = -1
            
            if isinstance(df.index, pd.DatetimeIndex):
                return df.index[idx].to_pydatetime()
            elif 'timestamp' in df.columns:
                return pd.to_datetime(df.iloc[idx]['timestamp']).to_pydatetime()
            else:
                return datetime.utcnow()
        except Exception:
            return datetime.utcnow()
    
    def _create_neutral_structure_analysis(self, epic: str) -> Dict:
        """Create neutral structure analysis for insufficient data"""
        return {
            'epic': epic,
            'timeframe': '5m',
            'timestamp': datetime.now(),
            'current_bias': MarketBias.NEUTRAL.value,
            'structure_score': 0.5,
            'swing_points': [],
            'structure_events': [],
            'next_targets': {'next_resistance': None, 'next_support': None, 'key_levels': []},
            'analysis_summary': 'Insufficient data for structure analysis',
            'pip_size': self._get_pip_size(epic)
        }
    
    def _create_structure_summary(
        self, 
        bias: MarketBias, 
        score: float, 
        events: List[StructureEvent]
    ) -> str:
        """Create human-readable structure summary"""
        recent_events = len(events[-2:])  # Use bar-based recency
        
        return (f"Market shows {bias.value.lower()} bias with {score:.1%} confidence. "
                f"{recent_events} recent structure events detected.")
    
    def _cache_structure_analysis(self, epic: str, analysis: Dict):
        """
        Cache the full structure analysis - FIXED for validation compatibility
        """
        try:
            cache_key = f"{epic}_structure"
            self.structure_history[cache_key] = analysis
            
            # Keep only recent cache entries
            if len(self.structure_history) > 50:
                oldest_key = min(self.structure_history.keys(), 
                               key=lambda k: self.structure_history[k].get('timestamp', datetime.min))
                del self.structure_history[oldest_key]
                
        except Exception as e:
            self.logger.error(f"❌ Structure cache update failed: {e}")
    
    def _swing_point_to_dict(self, sp: SwingPoint) -> Dict:
        """Convert SwingPoint to dictionary for JSON serialization"""
        return {
            'index': sp.index,
            'price': sp.price,
            'timestamp': sp.timestamp.isoformat() if sp.timestamp else None,
            'swing_type': sp.swing_type,
            'strength': sp.strength,
            'confirmed': sp.confirmed
        }
    
    def _structure_event_to_dict(self, se: StructureEvent) -> Dict:
        """Convert StructureEvent to dictionary for JSON serialization"""
        return {
            'event_type': se.event_type,
            'direction': se.direction,
            'price': se.price,
            'timestamp': se.timestamp.isoformat() if se.timestamp else None,
            'confidence': se.confidence,
            'significance': se.significance
        }
    
    def get_structure_status(self, epic: str) -> Dict:
        """Get current structure status for an epic"""
        cache_key = f"{epic}_structure"
        if cache_key in self.structure_history:
            analysis = self.structure_history[cache_key]
            return {
                'has_structure_data': True,
                'last_update': analysis.get('timestamp'),
                'current_bias': analysis.get('current_bias'),
                'structure_score': analysis.get('structure_score'),
                'swing_points_count': len(analysis.get('swing_points', [])),
                'structure_events_count': len(analysis.get('structure_events', []))
            }
        else:
            return {
                'has_structure_data': False,
                'message': 'No structure analysis available for this epic'
            }