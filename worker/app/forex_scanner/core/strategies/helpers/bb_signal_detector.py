# core/strategies/helpers/bb_signal_detector.py
"""
BB Signal Detector Module - FIXED FOR MORE SIGNALS
🔍 DETECTION: More sensitive BB mean reversion strategy
🎯 PURPOSE: Generate signals in more market conditions
📊 PRINCIPLE: Buy when approaching lower band, sell when approaching upper band

FIXED: More realistic detection thresholds for actual market conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging
from datetime import datetime


class BBSignalDetector:
    """
    🔍 DETECTION: More sensitive BB mean reversion strategy
    
    CORE CONCEPT: 
    - BULL when price approaches lower BB (relaxed oversold)
    - BEAR when price approaches upper BB (relaxed overbought)
    - More realistic thresholds for actual trading conditions
    """
    
    def __init__(self, logger: logging.Logger = None, forex_optimizer=None, validator=None):
        self.logger = logger or logging.getLogger(__name__)
        self.forex_optimizer = forex_optimizer
        self.validator = validator
        
        # More sensitive detection thresholds
        self.detection_config = {
            # Relaxed BB position thresholds
            'bull_max_position': 0.50,    # Up to 50% of BB range (was 0.30)
            'bear_min_position': 0.50,    # From 50% of BB range (was 0.70)
            
            # Band proximity thresholds  
            'band_proximity_pct': 0.25,   # Within 25% of band distance (was 0.10)
            
            # Minimum volatility
            'min_bb_width': 0.0005,       # Reduced minimum width (was 0.0008)
            
            # Trend filter sensitivity
            'trend_override_position': 0.15,  # Very oversold/overbought override
            'trend_tolerance_range': 0.6,     # Allow signals in 60% of middle range
        }
        
        # Signal detection statistics
        self._signals_detected = 0
        self._bull_signals = 0
        self._bear_signals = 0
        self._rejected_signals = 0
        
        self.logger.info("🔍 BB Signal Detector initialized - ENHANCED SENSITIVITY")
        self.logger.info(f"   📊 BULL threshold: ≤{self.detection_config['bull_max_position']:.0%} BB position")
        self.logger.info(f"   📊 BEAR threshold: ≥{self.detection_config['bear_min_position']:.0%} BB position")

    def detect_bb_supertrend_signal(self, current: pd.Series, previous: pd.Series) -> Optional[str]:
        """
        🔍 ENHANCED: More sensitive BB mean reversion signal detection
        """
        try:
            # Basic validation only
            if not self._validate_signal_data(current, previous):
                return None
            
            current_price = current['close']
            previous_price = previous['close']
            bb_upper = current['bb_upper']
            bb_lower = current['bb_lower']
            bb_middle = current['bb_middle']
            
            # Calculate BB position for context
            bb_width = bb_upper - bb_lower
            bb_position = (current_price - bb_lower) / bb_width if bb_width > 0 else 0.5
            
            self.logger.debug(f"🔍 BB Analysis: Price={current_price:.5f}, Position={bb_position:.1%}, Width={bb_width:.6f}")
            
            signal_type = None
            
            # 🟢 BULL SIGNAL: More sensitive oversold detection
            if self._detect_enhanced_bull_signal(current, previous, bb_position):
                signal_type = 'BULL'
                self._signals_detected += 1
                self._bull_signals += 1
                self.logger.info(f"🟢 ENHANCED BB BULL: Price {current_price:.5f} (pos: {bb_position:.1%}) near lower BB {bb_lower:.5f}")
            
            # 🔴 BEAR SIGNAL: More sensitive overbought detection  
            elif self._detect_enhanced_bear_signal(current, previous, bb_position):
                signal_type = 'BEAR'
                self._signals_detected += 1
                self._bear_signals += 1
                self.logger.info(f"🔴 ENHANCED BB BEAR: Price {current_price:.5f} (pos: {bb_position:.1%}) near upper BB {bb_upper:.5f}")
            
            if not signal_type:
                self.logger.debug(f"❌ No signal: Position {bb_position:.1%} in neutral zone ({self.detection_config['bull_max_position']:.0%}-{self.detection_config['bear_min_position']:.0%})")
            
            return signal_type
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced BB signal detection error: {e}")
            self._rejected_signals += 1
            return None

    def _detect_enhanced_bull_signal(self, current: pd.Series, previous: pd.Series, bb_position: float) -> bool:
        """
        🟢 ENHANCED BULL: More sensitive oversold detection
        """
        try:
            current_price = current['close']
            previous_price = previous['close']
            bb_lower = current['bb_lower']
            bb_upper = current['bb_upper']
            bb_middle = current['bb_middle']
            bb_width = bb_upper - bb_lower
            
            # CONDITION 1: More relaxed oversold condition
            oversold_condition = bb_position <= self.detection_config['bull_max_position']
            
            # CONDITION 2: Minimum BB width (more permissive)
            min_width_condition = bb_width > self.detection_config['min_bb_width']
            
            # CONDITION 3: Approaching lower band OR moving toward it
            band_proximity = self.detection_config['band_proximity_pct']
            approaching_lower = (
                current_price <= (bb_lower + bb_width * band_proximity) or  # Near lower band
                (current_price < previous_price and bb_position <= 0.60)     # Moving down in lower half
            )
            
            # CONDITION 4: Enhanced trend filtering
            trend_ok = self._check_enhanced_trend_filter(
                current, bb_position, 'BULL', 
                self.detection_config['trend_override_position'],
                self.detection_config['trend_tolerance_range']
            )
            
            # CONDITION 5: Additional momentum check
            momentum_ok = self._check_momentum_condition(current, previous, 'BULL')
            
            signal_detected = oversold_condition and min_width_condition and approaching_lower and trend_ok and momentum_ok
            
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"🟢 BULL conditions: oversold={oversold_condition} ({bb_position:.1%}≤{self.detection_config['bull_max_position']:.0%}), "
                                f"width_ok={min_width_condition} ({bb_width:.6f}>{self.detection_config['min_bb_width']:.6f}), "
                                f"approaching={approaching_lower}, trend_ok={trend_ok}, momentum_ok={momentum_ok}")
            
            return signal_detected
            
        except Exception as e:
            self.logger.debug(f"Enhanced BULL detection failed: {e}")
            return False

    def _detect_enhanced_bear_signal(self, current: pd.Series, previous: pd.Series, bb_position: float) -> bool:
        """
        🔴 ENHANCED BEAR: More sensitive overbought detection
        """
        try:
            current_price = current['close']
            previous_price = previous['close']
            bb_lower = current['bb_lower']
            bb_upper = current['bb_upper']
            bb_middle = current['bb_middle']
            bb_width = bb_upper - bb_lower
            
            # CONDITION 1: More relaxed overbought condition
            overbought_condition = bb_position >= self.detection_config['bear_min_position']
            
            # CONDITION 2: Minimum BB width (more permissive)
            min_width_condition = bb_width > self.detection_config['min_bb_width']
            
            # CONDITION 3: Approaching upper band OR moving toward it
            band_proximity = self.detection_config['band_proximity_pct']
            approaching_upper = (
                current_price >= (bb_upper - bb_width * band_proximity) or  # Near upper band
                (current_price > previous_price and bb_position >= 0.40)     # Moving up in upper half
            )
            
            # CONDITION 4: Enhanced trend filtering
            trend_ok = self._check_enhanced_trend_filter(
                current, bb_position, 'BEAR',
                1.0 - self.detection_config['trend_override_position'],  # 85% for bear override
                self.detection_config['trend_tolerance_range']
            )
            
            # CONDITION 5: Additional momentum check
            momentum_ok = self._check_momentum_condition(current, previous, 'BEAR')
            
            signal_detected = overbought_condition and min_width_condition and approaching_upper and trend_ok and momentum_ok
            
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"🔴 BEAR conditions: overbought={overbought_condition} ({bb_position:.1%}≥{self.detection_config['bear_min_position']:.0%}), "
                                f"width_ok={min_width_condition} ({bb_width:.6f}>{self.detection_config['min_bb_width']:.6f}), "
                                f"approaching={approaching_upper}, trend_ok={trend_ok}, momentum_ok={momentum_ok}")
            
            return signal_detected
            
        except Exception as e:
            self.logger.debug(f"Enhanced BEAR detection failed: {e}")
            return False

    def _check_enhanced_trend_filter(self, current: pd.Series, bb_position: float, signal_type: str, override_threshold: float, tolerance_range: float) -> bool:
        """
        🎯 Enhanced trend filtering with override conditions
        """
        try:
            # Override for extreme positions (very oversold/overbought)
            if signal_type == 'BULL' and bb_position <= override_threshold:
                return True  # Very oversold - allow signal regardless of trend
            elif signal_type == 'BEAR' and bb_position >= override_threshold:
                return True  # Very overbought - allow signal regardless of trend
            
            # Check SuperTrend if available
            if 'supertrend_direction' in current.index and not pd.isna(current['supertrend_direction']):
                st_direction = current['supertrend_direction']
                
                if signal_type == 'BULL':
                    # Allow BULL signals if SuperTrend is bullish OR we're in tolerance range
                    return (st_direction == 1) or (bb_position <= (0.5 + tolerance_range/2))
                else:  # BEAR
                    # Allow BEAR signals if SuperTrend is bearish OR we're in tolerance range
                    return (st_direction == -1) or (bb_position >= (0.5 - tolerance_range/2))
            
            # No SuperTrend - use BB position relative to middle
            bb_middle = current['bb_middle']
            current_price = current['close']
            
            if signal_type == 'BULL':
                # Allow BULL if price not too far above middle
                return current_price <= (bb_middle * 1.005)  # Within 0.5% above middle
            else:  # BEAR
                # Allow BEAR if price not too far below middle
                return current_price >= (bb_middle * 0.995)  # Within 0.5% below middle
            
        except Exception as e:
            self.logger.debug(f"Enhanced trend filter failed: {e}")
            return True  # Default to allowing signal

    def _check_momentum_condition(self, current: pd.Series, previous: pd.Series, signal_type: str) -> bool:
        """
        📈 Check momentum conditions for signal quality
        """
        try:
            current_price = current['close']
            previous_price = previous['close']
            price_change = current_price - previous_price
            
            # For BB mean reversion, we often want to catch turning points
            # So we're more flexible with momentum requirements
            
            if signal_type == 'BULL':
                # For BULL signals, allow if:
                # 1. Price is stabilizing (small negative move)
                # 2. Already bouncing (positive move)
                # 3. Or just any reasonable move
                return abs(price_change) <= (current_price * 0.002)  # Within 0.2% move
            else:  # BEAR
                # For BEAR signals, similar logic
                return abs(price_change) <= (current_price * 0.002)  # Within 0.2% move
            
        except Exception as e:
            self.logger.debug(f"Momentum check failed: {e}")
            return True  # Default to allowing signal

    def _validate_signal_data(self, current: pd.Series, previous: pd.Series) -> bool:
        """
        ✅ Enhanced but lenient validation
        """
        try:
            # Only require BB data - SuperTrend is optional
            required_fields = ['close', 'bb_upper', 'bb_middle', 'bb_lower']
            
            for field in required_fields:
                if field not in current.index or pd.isna(current[field]):
                    self.logger.debug(f"Missing required field: {field}")
                    return False
                if field not in previous.index or pd.isna(previous[field]):
                    self.logger.debug(f"Missing required field in previous: {field}")
                    return False
            
            # More lenient BB order validation
            bb_upper = current['bb_upper']
            bb_middle = current['bb_middle']
            bb_lower = current['bb_lower']
            
            if not (bb_upper > bb_lower):  # Only require upper > lower
                self.logger.debug("Invalid BB order: upper not > lower")
                return False
            
            # Allow middle to be slightly outside bands (real market conditions)
            bb_width = bb_upper - bb_lower
            tolerance = bb_width * 0.05  # 5% tolerance
            
            if not (bb_lower - tolerance <= bb_middle <= bb_upper + tolerance):
                self.logger.debug("BB middle significantly outside bands")
                return False
            
            # Validate we have meaningful prices
            if current['close'] <= 0:
                self.logger.debug("Invalid price")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Signal data validation failed: {e}")
            return False

    def analyze_signal_strength(self, current: pd.Series, signal_type: str) -> Dict:
        """
        💪 Enhanced signal strength analysis
        """
        try:
            bb_upper = current['bb_upper']
            bb_lower = current['bb_lower'] 
            current_price = current['close']
            bb_width = bb_upper - bb_lower
            bb_position = (current_price - bb_lower) / bb_width if bb_width > 0 else 0.5
            
            if signal_type == 'BULL':
                # For BULL: closer to lower band = stronger
                # Scale from bull_max_position down to 0
                max_pos = self.detection_config['bull_max_position']
                strength = max(0.0, (max_pos - bb_position) / max_pos) if max_pos > 0 else 0.0
            else:  # BEAR
                # For BEAR: closer to upper band = stronger
                # Scale from bear_min_position up to 1.0
                min_pos = self.detection_config['bear_min_position']
                strength = max(0.0, (bb_position - min_pos) / (1.0 - min_pos)) if min_pos < 1.0 else 0.0
            
            return {
                'signal_type': signal_type,
                'bb_position': bb_position,
                'bb_width': bb_width,
                'overall_strength': min(strength, 1.0),
                'detection_thresholds': {
                    'bull_max_position': self.detection_config['bull_max_position'],
                    'bear_min_position': self.detection_config['bear_min_position'],
                    'band_proximity_pct': self.detection_config['band_proximity_pct']
                },
                'enhancement_features': ['relaxed_thresholds', 'momentum_check', 'enhanced_trend_filter']
            }
            
        except Exception as e:
            return {'error': str(e)}

    def get_detection_stats(self) -> Dict:
        """
        📊 Get enhanced detection statistics
        """
        return {
            'total_signals_detected': self._signals_detected,
            'bull_signals': self._bull_signals,
            'bear_signals': self._bear_signals,
            'rejected_signals': self._rejected_signals,
            'bull_percentage': self._bull_signals / max(self._signals_detected, 1) * 100,
            'bear_percentage': self._bear_signals / max(self._signals_detected, 1) * 100,
            'strategy_type': 'enhanced_mean_reversion',
            'purpose': 'more_sensitive_bb_detection',
            'detection_config': self.detection_config,
            'features_enabled': [
                'relaxed_position_thresholds',
                'enhanced_trend_filtering',
                'momentum_validation',
                'band_proximity_detection'
            ]
        }

    def reset_stats(self):
        """🔄 Reset detection statistics"""
        self._signals_detected = 0
        self._bull_signals = 0
        self._bear_signals = 0
        self._rejected_signals = 0
        self.logger.info("🔄 Enhanced BB Signal Detector stats reset")

    def update_detection_config(self, new_config: Dict):
        """⚙️ Update detection configuration at runtime"""
        self.detection_config.update(new_config)
        self.logger.info(f"🔧 Detection config updated: {list(new_config.keys())}")

    def debug_signal_detection(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> Dict:
        """
        🔍 Enhanced debugging with detailed analysis
        """
        try:
            if len(df) < 2:
                return {'error': 'Insufficient data for debugging'}
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Calculate BB position
            bb_width = current['bb_upper'] - current['bb_lower']
            bb_position = (current['close'] - current['bb_lower']) / bb_width if bb_width > 0 else 0.5
            
            # Enhanced debugging info
            debug_info = {
                'epic': epic,
                'timeframe': timeframe,
                'current_price': current['close'],
                'bb_analysis': {
                    'bb_position': bb_position,
                    'bb_position_pct': f"{bb_position:.1%}",
                    'bb_width': bb_width,
                    'bb_upper': current['bb_upper'],
                    'bb_middle': current['bb_middle'],
                    'bb_lower': current['bb_lower'],
                    'supertrend': current.get('supertrend', 'N/A'),
                    'supertrend_direction': current.get('supertrend_direction', 'N/A')
                },
                'detection_analysis': {
                    'bull_eligible': bb_position <= self.detection_config['bull_max_position'],
                    'bear_eligible': bb_position >= self.detection_config['bear_min_position'],
                    'in_neutral_zone': (self.detection_config['bull_max_position'] < bb_position < self.detection_config['bear_min_position']),
                    'bull_threshold': f"≤{self.detection_config['bull_max_position']:.0%}",
                    'bear_threshold': f"≥{self.detection_config['bear_min_position']:.0%}",
                    'neutral_zone': f"{self.detection_config['bull_max_position']:.0%}-{self.detection_config['bear_min_position']:.0%}"
                },
                'detection_config': self.detection_config.copy(),
                'detection_stats': self.get_detection_stats()
            }
            
            # Test actual signal detection
            signal_type = self.detect_bb_supertrend_signal(current, previous)
            debug_info['signal_detected'] = signal_type
            
            if signal_type:
                strength_analysis = self.analyze_signal_strength(current, signal_type)
                debug_info['signal_strength'] = strength_analysis
            
            return debug_info
            
        except Exception as e:
            return {'error': f'Enhanced debug analysis failed: {str(e)}'}