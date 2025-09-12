# core/strategies/helpers/kama_confidence_calculator.py
"""
KAMA-Specific Confidence Calculator
🎯 KAMA OPTIMIZED: Confidence calculation designed specifically for KAMA strategy characteristics
🚫 NO ENHANCED VALIDATOR: Independent confidence calculation without generic validator interference
🔥 FOREX FOCUSED: Understanding of KAMA's behavior in forex markets
⚡ PERFORMANCE: Fast, focused calculations without overhead

This replaces the Enhanced Signal Validator for KAMA confidence calculation
to avoid the generic "15.0%" scores that don't understand KAMA specifics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime
try:
    import config
except ImportError:
    from forex_scanner import config


class KAMAConfidenceCalculator:
    """
    🎯 KAMA-SPECIFIC: Confidence calculator designed exclusively for KAMA strategy
    
    Understanding KAMA's unique characteristics:
    - Efficiency Ratio is the primary driver (market adaptability)
    - Works best in trending markets, struggles in consolidation
    - Price-KAMA distance indicates signal quality
    - Trend consistency is crucial for reliability
    - Volume and momentum provide additional confirmation
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # KAMA-specific confidence thresholds (optimized for forex)
        self.confidence_thresholds = {
            'efficiency_ratio': {
                'excellent': 0.7,    # Strong trending market
                'very_good': 0.5,    # Good trending market  
                'good': 0.35,        # Moderate trending
                'acceptable': 0.25,  # Minimum for signals
                'poor': 0.15,        # Choppy market
                'very_poor': 0.1     # Avoid trading
            },
            'trend_strength': {
                'strong': 0.002,     # Strong KAMA trend
                'moderate': 0.001,   # Moderate trend
                'weak': 0.0005,      # Weak trend
                'minimal': 0.0002    # Minimal trend
            },
            'price_distance': {
                'very_close': 0.002,    # Price very close to KAMA
                'close': 0.005,         # Price close to KAMA
                'moderate': 0.01,       # Moderate distance
                'far': 0.02,            # Price far from KAMA
                'very_far': 0.03        # Price very far from KAMA
            }
        }
        
        # Confidence component weights (KAMA-optimized)
        self.component_weights = {
            'efficiency_ratio': 0.45,      # Most important for KAMA
            'trend_strength': 0.25,        # KAMA trend momentum
            'price_alignment': 0.15,       # Price-KAMA distance
            'signal_strength': 0.10,       # Signal clarity
            'market_context': 0.05         # Additional context
        }
        
        # Confidence bonuses and penalties
        self.adjustments = {
            'high_efficiency_bonus': 0.08,      # ER > 0.6
            'trend_alignment_bonus': 0.06,      # Trend matches signal
            'volume_confirmation_bonus': 0.04,   # Volume supports signal
            'macd_alignment_bonus': 0.03,       # MACD confirms signal
            'session_bonus': 0.02,              # Active trading session
            
            'low_efficiency_penalty': -0.1,     # ER < 0.15
            'trend_contradiction_penalty': -0.08, # Trend opposes signal
            'distance_penalty': -0.06,          # Price too far from KAMA
            'weak_momentum_penalty': -0.04,     # Low signal strength
            'consolidation_penalty': -0.05      # Consolidating market
        }
        
        self.logger.info("🎯 KAMA-Specific Confidence Calculator initialized")
        self.logger.info(f"   Efficiency Ratio Weight: {self.component_weights['efficiency_ratio']:.1%}")
        self.logger.info(f"   Trend Strength Weight: {self.component_weights['trend_strength']:.1%}")

    def calculate_kama_confidence(
        self, 
        signal_data: Dict, 
        df: pd.DataFrame = None, 
        epic: str = None
    ) -> float:
        """
        🔥 MAIN METHOD: Calculate KAMA-optimized confidence score
        
        Args:
            signal_data: Signal data dictionary with KAMA-specific metrics
            df: DataFrame with market data (optional for context)
            epic: Trading epic (optional for pair-specific adjustments)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            # Extract KAMA-specific metrics
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.1)
            kama_trend = signal_data.get('kama_trend', 0)
            signal_strength = signal_data.get('signal_strength', 0)
            signal_type = signal_data.get('signal_type', '')
            price = signal_data.get('price', 0)
            kama_value = signal_data.get('kama_value', 0)
            
            self.logger.debug(f"[KAMA CONFIDENCE] ER: {efficiency_ratio:.3f}, "
                            f"Trend: {kama_trend:.6f}, Strength: {signal_strength:.3f}")
            
            # Calculate core confidence components
            efficiency_score = self._calculate_efficiency_score(efficiency_ratio)
            trend_score = self._calculate_trend_score(kama_trend, signal_type)
            alignment_score = self._calculate_price_alignment_score(price, kama_value)
            strength_score = self._calculate_signal_strength_score(signal_strength)
            context_score = self._calculate_market_context_score(signal_data, df, epic)
            
            # Weighted combination of components
            base_confidence = (
                efficiency_score * self.component_weights['efficiency_ratio'] +
                trend_score * self.component_weights['trend_strength'] +
                alignment_score * self.component_weights['price_alignment'] +
                strength_score * self.component_weights['signal_strength'] +
                context_score * self.component_weights['market_context']
            )
            
            # Apply KAMA-specific adjustments
            adjusted_confidence = self._apply_kama_adjustments(
                base_confidence, signal_data, df, epic
            )
            
            # Final bounds and validation
            final_confidence = max(0.15, min(0.95, adjusted_confidence))
            
            self.logger.info(f"[KAMA CONFIDENCE] {signal_type} - "
                           f"Base: {base_confidence:.1%} → Adjusted: {adjusted_confidence:.1%} → "
                           f"Final: {final_confidence:.1%}")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"[KAMA CONFIDENCE ERROR] {e}")
            return self._fallback_confidence(signal_data)

    def _calculate_efficiency_score(self, efficiency_ratio: float) -> float:
        """
        🔥 EFFICIENCY SCORING: Core KAMA metric - market adaptability
        
        High efficiency = trending market (KAMA's strength)
        Low efficiency = choppy market (KAMA struggles)
        """
        thresholds = self.confidence_thresholds['efficiency_ratio']
        
        if efficiency_ratio >= thresholds['excellent']:
            return 0.95    # Excellent trending market
        elif efficiency_ratio >= thresholds['very_good']:
            return 0.85    # Very good trending
        elif efficiency_ratio >= thresholds['good']:
            return 0.70    # Good trending
        elif efficiency_ratio >= thresholds['acceptable']:
            return 0.55    # Acceptable for signals
        elif efficiency_ratio >= thresholds['poor']:
            return 0.35    # Poor conditions
        else:
            return 0.20    # Very poor - avoid trading

    def _calculate_trend_score(self, kama_trend: float, signal_type: str) -> float:
        """
        📈 TREND SCORING: KAMA trend strength and alignment
        """
        abs_trend = abs(kama_trend)
        thresholds = self.confidence_thresholds['trend_strength']
        
        # Base trend strength score
        if abs_trend >= thresholds['strong']:
            trend_strength = 0.9
        elif abs_trend >= thresholds['moderate']:
            trend_strength = 0.75
        elif abs_trend >= thresholds['weak']:
            trend_strength = 0.6
        elif abs_trend >= thresholds['minimal']:
            trend_strength = 0.4
        else:
            trend_strength = 0.2
        
        # Trend alignment bonus/penalty
        if signal_type in ['BULL', 'BUY'] and kama_trend > 0:
            trend_strength += 0.1  # Trend supports signal
        elif signal_type in ['BEAR', 'SELL'] and kama_trend < 0:
            trend_strength += 0.1  # Trend supports signal
        elif signal_type in ['BULL', 'BUY'] and kama_trend < -thresholds['weak']:
            trend_strength -= 0.15  # Strong opposing trend
        elif signal_type in ['BEAR', 'SELL'] and kama_trend > thresholds['weak']:
            trend_strength -= 0.15  # Strong opposing trend
        
        return max(0.1, min(1.0, trend_strength))

    def _calculate_price_alignment_score(self, price: float, kama_value: float) -> float:
        """
        📏 ALIGNMENT SCORING: Price distance from KAMA
        
        Close price-KAMA distance = better signal quality
        """
        if price <= 0 or kama_value <= 0:
            return 0.5  # Neutral if no data
        
        distance_ratio = abs(price - kama_value) / price
        thresholds = self.confidence_thresholds['price_distance']
        
        if distance_ratio <= thresholds['very_close']:
            return 0.95    # Excellent alignment
        elif distance_ratio <= thresholds['close']:
            return 0.85    # Good alignment
        elif distance_ratio <= thresholds['moderate']:
            return 0.70    # Acceptable alignment
        elif distance_ratio <= thresholds['far']:
            return 0.50    # Poor alignment
        else:
            return 0.25    # Very poor alignment

    def _calculate_signal_strength_score(self, signal_strength: float) -> float:
        """
        💪 SIGNAL STRENGTH: Clarity and conviction of the signal
        """
        if signal_strength >= 0.8:
            return 0.9     # Very strong signal
        elif signal_strength >= 0.6:
            return 0.75    # Strong signal
        elif signal_strength >= 0.4:
            return 0.6     # Moderate signal
        elif signal_strength >= 0.2:
            return 0.45    # Weak signal
        else:
            return 0.25    # Very weak signal

    def _calculate_market_context_score(
        self, 
        signal_data: Dict, 
        df: pd.DataFrame = None, 
        epic: str = None
    ) -> float:
        """
        🌍 MARKET CONTEXT: Additional market conditions affecting KAMA performance
        """
        context_score = 0.5  # Neutral baseline
        
        try:
            # Volume confirmation
            volume_confirmation = signal_data.get('volume_confirmation', None)
            if volume_confirmation is True:
                context_score += 0.2
            elif volume_confirmation is False:
                context_score -= 0.1
            
            # MACD alignment
            macd_histogram = signal_data.get('macd_histogram', 0)
            signal_type = signal_data.get('signal_type', '')
            
            if signal_type in ['BULL', 'BUY'] and macd_histogram > 0.0001:
                context_score += 0.15  # MACD confirms
            elif signal_type in ['BEAR', 'SELL'] and macd_histogram < -0.0001:
                context_score += 0.15  # MACD confirms
            elif signal_type in ['BULL', 'BUY'] and macd_histogram < -0.0001:
                context_score -= 0.2   # MACD contradicts
            elif signal_type in ['BEAR', 'SELL'] and macd_histogram > 0.0001:
                context_score -= 0.2   # MACD contradicts
            
            # Trading session
            session_bonus = self._get_session_bonus()
            context_score += session_bonus
            
            return max(0.1, min(1.0, context_score))
            
        except Exception as e:
            self.logger.debug(f"Market context calculation error: {e}")
            return 0.5

    def _apply_kama_adjustments(
        self, 
        base_confidence: float, 
        signal_data: Dict, 
        df: pd.DataFrame = None, 
        epic: str = None
    ) -> float:
        """
        🔧 KAMA ADJUSTMENTS: Apply KAMA-specific bonuses and penalties
        """
        adjusted = base_confidence
        
        try:
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.1)
            
            # High efficiency bonus
            if efficiency_ratio > 0.6:
                adjusted += self.adjustments['high_efficiency_bonus']
                self.logger.debug(f"[KAMA BONUS] High efficiency: +{self.adjustments['high_efficiency_bonus']:.1%}")
            
            # Low efficiency penalty
            elif efficiency_ratio < 0.15:
                adjusted += self.adjustments['low_efficiency_penalty']
                self.logger.debug(f"[KAMA PENALTY] Low efficiency: {self.adjustments['low_efficiency_penalty']:.1%}")
            
            # Trend alignment
            kama_trend = signal_data.get('kama_trend', 0)
            signal_type = signal_data.get('signal_type', '')
            
            if ((signal_type in ['BULL', 'BUY'] and kama_trend > 0.0005) or 
                (signal_type in ['BEAR', 'SELL'] and kama_trend < -0.0005)):
                adjusted += self.adjustments['trend_alignment_bonus']
                self.logger.debug(f"[KAMA BONUS] Trend alignment: +{self.adjustments['trend_alignment_bonus']:.1%}")
            
            # Volume confirmation
            volume_confirmation = signal_data.get('volume_confirmation', None)
            if volume_confirmation is True:
                adjusted += self.adjustments['volume_confirmation_bonus']
                self.logger.debug(f"[KAMA BONUS] Volume confirmation: +{self.adjustments['volume_confirmation_bonus']:.1%}")
            
            # Market regime penalty for consolidation
            if df is not None and len(df) > 20:
                market_regime = self._detect_simple_market_regime(df)
                if market_regime == 'consolidating':
                    adjusted += self.adjustments['consolidation_penalty']
                    self.logger.debug(f"[KAMA PENALTY] Consolidation: {self.adjustments['consolidation_penalty']:.1%}")
            
            return adjusted
            
        except Exception as e:
            self.logger.debug(f"KAMA adjustments error: {e}")
            return base_confidence

    def _detect_simple_market_regime(self, df: pd.DataFrame) -> str:
        """
        📊 SIMPLE REGIME: Basic market regime detection for KAMA context
        """
        try:
            if len(df) < 20:
                return 'unknown'
            
            recent_data = df.tail(20)
            price_range = recent_data['high'].max() - recent_data['low'].min()
            avg_price = recent_data['close'].mean()
            volatility = price_range / avg_price
            
            if volatility < 0.005:  # Less than 0.5% range
                return 'consolidating'
            elif volatility > 0.02:  # More than 2% range
                return 'volatile'
            else:
                return 'trending'
                
        except Exception as e:
            self.logger.debug(f"Regime detection error: {e}")
            return 'unknown'

    def _get_session_bonus(self) -> float:
        """
        ⏰ SESSION BONUS: Trading session impact on KAMA performance
        """
        try:
            import pytz
            london_tz = pytz.timezone('Europe/London')
            london_time = datetime.now(london_tz)
            hour = london_time.hour
            
            if 8 <= hour < 17:  # London session
                return 0.02
            elif 13 <= hour < 22:  # New York session
                return 0.02
            elif 0 <= hour < 9:  # Sydney session
                return -0.01
            else:  # Tokyo session
                return 0.0
        except:
            return 0.0

    def _fallback_confidence(self, signal_data: Dict) -> float:
        """
        🚨 FALLBACK: Simple confidence when main calculation fails
        """
        efficiency_ratio = signal_data.get('efficiency_ratio', 0.1)
        signal_strength = signal_data.get('signal_strength', 0.5)
        
        # Simple calculation based on key metrics
        fallback = (efficiency_ratio * 0.6 + signal_strength * 0.4)
        
        # Ensure minimum acceptable confidence
        return max(0.25, min(0.85, fallback))

    def get_confidence_breakdown(
        self, 
        signal_data: Dict, 
        df: pd.DataFrame = None, 
        epic: str = None
    ) -> Dict:
        """
        📊 BREAKDOWN: Detailed confidence calculation breakdown for debugging
        """
        try:
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.1)
            kama_trend = signal_data.get('kama_trend', 0)
            signal_strength = signal_data.get('signal_strength', 0)
            signal_type = signal_data.get('signal_type', '')
            price = signal_data.get('price', 0)
            kama_value = signal_data.get('kama_value', 0)
            
            # Calculate individual components
            efficiency_score = self._calculate_efficiency_score(efficiency_ratio)
            trend_score = self._calculate_trend_score(kama_trend, signal_type)
            alignment_score = self._calculate_price_alignment_score(price, kama_value)
            strength_score = self._calculate_signal_strength_score(signal_strength)
            context_score = self._calculate_market_context_score(signal_data, df, epic)
            
            # Calculate weighted base
            base_confidence = (
                efficiency_score * self.component_weights['efficiency_ratio'] +
                trend_score * self.component_weights['trend_strength'] +
                alignment_score * self.component_weights['price_alignment'] +
                strength_score * self.component_weights['signal_strength'] +
                context_score * self.component_weights['market_context']
            )
            
            # Final confidence
            final_confidence = self.calculate_kama_confidence(signal_data, df, epic)
            
            return {
                'components': {
                    'efficiency_score': efficiency_score,
                    'trend_score': trend_score,
                    'alignment_score': alignment_score,
                    'strength_score': strength_score,
                    'context_score': context_score
                },
                'weights': self.component_weights,
                'base_confidence': base_confidence,
                'final_confidence': final_confidence,
                'adjustment': final_confidence - base_confidence,
                'input_metrics': {
                    'efficiency_ratio': efficiency_ratio,
                    'kama_trend': kama_trend,
                    'signal_strength': signal_strength,
                    'signal_type': signal_type,
                    'price_kama_distance': abs(price - kama_value) / price if price > 0 and kama_value > 0 else 0
                }
            }
            
        except Exception as e:
            return {'error': str(e)}


# ===== INTEGRATION HELPER FOR KAMA STRATEGY =====

def integrate_kama_confidence_calculator(kama_strategy_instance):
    """
    🔧 INTEGRATION: Replace Enhanced Signal Validator with KAMA-specific calculator
    
    Usage in KAMAStrategy.calculate_confidence():
    ```python
    from .helpers.kama_confidence_calculator import KAMAConfidenceCalculator
    
    def __init__(self, data_fetcher=None):
        # ... existing code ...
        self.confidence_calculator = KAMAConfidenceCalculator(logger=self.logger)
    
    def calculate_confidence(self, signal_data, df=None, epic=None):
        return self.confidence_calculator.calculate_kama_confidence(signal_data, df, epic)
    ```
    """
    pass