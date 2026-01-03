# core/strategies/helpers/kama_validator.py
"""
KAMA Validator Module - Extracted from KAMA Strategy
🧠 VALIDATION: Signal validation and confidence calculation for KAMA
🎯 FOCUSED: Single responsibility for KAMA signal validation
📊 COMPREHENSIVE: Enhanced validation with market context awareness

This module contains all the validation logic for KAMA strategy,
extracted for better maintainability and testability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime


class KAMAValidator:
    """
    🧠 VALIDATION: Comprehensive signal validation for KAMA strategy
    
    Responsibilities:
    - KAMA efficiency validation
    - Trend alignment validation
    - Market condition assessment
    - Confidence calculation with enhanced validation
    - Signal quality scoring
    """
    
    def __init__(self, logger: logging.Logger = None, forex_optimizer=None):
        self.logger = logger or logging.getLogger(__name__)
        self.forex_optimizer = forex_optimizer  # Will be injected by main strategy
        
        # Validation statistics
        self._validation_count = 0
        self._rejection_count = 0
        self._rejection_reasons = {}
        
        self.logger.info("🧠 KAMA Validator initialized")

    def validate_kama_efficiency(self, signal_data: Dict, epic: str = None) -> Tuple[bool, str]:
        """
        ⚡ KAMA: Validate efficiency ratio requirements
        """
        try:
            efficiency_ratio = signal_data.get('efficiency_ratio', 0)
            
            # Get forex-specific thresholds
            if self.forex_optimizer:
                thresholds = self.forex_optimizer.get_kama_thresholds_for_pair(epic or 'default')
                min_efficiency = thresholds['min_efficiency']
            else:
                min_efficiency = 0.1
            
            if efficiency_ratio < min_efficiency:
                reason = f"Efficiency ratio too low: {efficiency_ratio:.3f} < {min_efficiency}"
                self._track_rejection('low_efficiency')
                return False, reason
            
            # Additional efficiency quality checks
            if efficiency_ratio > 0.8:
                # Very high efficiency - might be a data issue
                self.logger.warning(f"⚠️ Unusually high efficiency ratio: {efficiency_ratio:.3f}")
            
            return True, f"Efficiency acceptable: {efficiency_ratio:.3f}"
            
        except Exception as e:
            reason = f"Efficiency validation error: {e}"
            self._track_rejection('efficiency_error')
            return False, reason

    def validate_kama_trend_alignment(self, signal_data: Dict, epic: str = None) -> Tuple[bool, str]:
        """
        📈 KAMA: Validate KAMA trend alignment with signal direction
        """
        try:
            signal_type = signal_data.get('signal_type')
            kama_trend = signal_data.get('kama_trend', 0)
            kama_value = signal_data.get('kama_value', 0)
            current_price = signal_data.get('price', signal_data.get('current_price', 0))
            
            if not signal_type or kama_value == 0:
                reason = "Missing signal type or KAMA value"
                self._track_rejection('missing_data')
                return False, reason
            
            # Get forex-specific thresholds
            if self.forex_optimizer:
                thresholds = self.forex_optimizer.get_kama_thresholds_for_pair(epic or 'default')
                trend_threshold = thresholds['trend_threshold']
            else:
                trend_threshold = 0.05
            
            # Check trend significance
            if abs(kama_trend) < trend_threshold * 0.5:
                reason = f"KAMA trend too weak: {abs(kama_trend):.4f} < {trend_threshold * 0.5:.4f}"
                self._track_rejection('weak_trend')
                return False, reason
            
            # Check trend alignment with signal
            if signal_type in ['BULL', 'BUY']:
                # For bullish signals, prefer upward KAMA trend and price above KAMA
                if kama_trend < -trend_threshold:
                    reason = f"Bullish signal but KAMA trending down: {kama_trend:.4f}"
                    self._track_rejection('trend_misalignment')
                    return False, reason
                    
                if current_price < kama_value and abs(current_price - kama_value) / kama_value > 0.01:
                    reason = f"Bullish signal but price well below KAMA"
                    self._track_rejection('price_misalignment')
                    return False, reason
                    
            elif signal_type in ['BEAR', 'SELL']:
                # For bearish signals, prefer downward KAMA trend and price below KAMA
                if kama_trend > trend_threshold:
                    reason = f"Bearish signal but KAMA trending up: {kama_trend:.4f}"
                    self._track_rejection('trend_misalignment')
                    return False, reason
                    
                if current_price > kama_value and abs(current_price - kama_value) / kama_value > 0.01:
                    reason = f"Bearish signal but price well above KAMA"
                    self._track_rejection('price_misalignment')
                    return False, reason
            
            return True, f"KAMA trend alignment good: {signal_type} with trend {kama_trend:.4f}"
            
        except Exception as e:
            reason = f"Trend alignment validation error: {e}"
            self._track_rejection('alignment_error')
            return False, reason

    def validate_market_conditions(self, signal_data: Dict, df: pd.DataFrame = None, epic: str = None) -> Tuple[bool, str]:
        """
        🌡️ KAMA: Validate market conditions for KAMA signals
        """
        try:
            # Check volume confirmation if available
            volume_confirmation = signal_data.get('volume_confirmation', True)  # Default to True if not available
            if not volume_confirmation:
                volume_ratio = signal_data.get('volume_ratio', 1.0)
                if volume_ratio < 0.8:  # Very low volume
                    reason = f"Low volume confirmation: {volume_ratio:.2f}"
                    self._track_rejection('low_volume')
                    return False, reason
            
            # Check for extreme market conditions
            atr = signal_data.get('atr', 0.001)
            if atr > 0.01:  # Very high volatility
                self.logger.warning(f"⚠️ High volatility detected: ATR={atr:.4f}")
                # Don't reject, but warn
            
            # Check market regime if available
            if self.forex_optimizer and df is not None and len(df) > 0:
                market_regime = self.forex_optimizer.detect_forex_market_regime(
                    df.iloc[-1], df, epic or 'default', '15m'
                )
                
                # KAMA works best in trending markets
                if market_regime == 'consolidating':
                    efficiency_ratio = signal_data.get('efficiency_ratio', 0)
                    if efficiency_ratio < 0.3:  # Low efficiency in consolidating market
                        reason = f"Consolidating market with low efficiency: {efficiency_ratio:.3f}"
                        self._track_rejection('consolidating_market')
                        return False, reason
            
            return True, "Market conditions acceptable for KAMA"
            
        except Exception as e:
            reason = f"Market conditions validation error: {e}"
            self._track_rejection('market_error')
            return False, reason

    def calculate_enhanced_confidence(self, signal_data: Dict, df: pd.DataFrame = None, epic: str = None) -> float:
        """
        🧠 KAMA-OPTIMIZED: Calculate confidence using KAMA-specific enhanced validation
        
        This method combines:
        1. KAMA-specific factor analysis (efficiency ratio focus)
        2. Enhanced Signal Validator integration (as cross-reference)
        3. Forex-specific adjustments (via forex optimizer)
        """
        try:
            # 🎯 STEP 1: Calculate KAMA-optimized base confidence
            kama_base_confidence = self._calculate_kama_focused_confidence(signal_data, epic)
            
            # 🧠 STEP 2: Try enhanced validation for cross-reference
            enhanced_validator_confidence = 0.5  # Default fallback
            try:
                from ...detection.enhanced_signal_validator import EnhancedSignalValidator
                enhanced_validator = EnhancedSignalValidator(logger=self.logger)
                
                should_trade, validator_confidence, reason, analysis = enhanced_validator.validate_signal_enhanced(signal_data)
                enhanced_validator_confidence = validator_confidence
                
                # Log the comparison
                self.logger.debug(f"[KAMA VALIDATOR] KAMA-focused: {kama_base_confidence:.1%}, Enhanced validator: {enhanced_validator_confidence:.1%}")
                
                # Use KAMA-focused confidence but validate against enhanced validator
                if enhanced_validator_confidence < 0.3:  # Enhanced validator says this is a bad signal
                    self.logger.warning(f"[KAMA VALIDATOR] Enhanced validator flagged signal as poor quality: {enhanced_validator_confidence:.1%}")
                    # Use the lower confidence (more conservative)
                    final_confidence = min(kama_base_confidence, enhanced_validator_confidence + 0.1)  # Small bonus for KAMA specificity
                else:
                    # Enhanced validator thinks it's reasonable, use KAMA-focused confidence
                    final_confidence = kama_base_confidence
                    
            except Exception as validator_error:
                self.logger.debug(f"Enhanced validation failed, using KAMA-focused calculation: {validator_error}")
                final_confidence = kama_base_confidence
            
            # 🌍 STEP 3: Apply forex adjustments if available
            if self.forex_optimizer:
                final_confidence = self.forex_optimizer.apply_forex_confidence_adjustments(
                    final_confidence, epic or '', signal_data
                )
            
            self.logger.debug(f"[KAMA ENHANCED] Final confidence: {final_confidence:.1%}")
            return final_confidence
                
        except Exception as e:
            self.logger.error(f"[KAMA ENHANCED ERROR] {e}")
            return self.calculate_basic_confidence(signal_data)

    def _calculate_kama_focused_confidence(self, signal_data: Dict, epic: str = None) -> float:
        """
        🎯 KAMA-FOCUSED: Calculate confidence specifically optimized for KAMA characteristics
        
        KAMA is unique because:
        - Efficiency ratio is the key metric (market adaptability)
        - Works best in trending/volatile markets
        - Adapts smoothing based on market conditions
        - Less effective in choppy/consolidating markets
        """
        try:
            # Extract KAMA-specific metrics
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.1)
            kama_trend = abs(signal_data.get('kama_trend', 0))
            signal_strength = signal_data.get('signal_strength', 0)
            signal_type = signal_data.get('signal_type', '')
            
            # Additional context metrics
            price = signal_data.get('price', 0)
            kama_value = signal_data.get('kama_value', 0)
            macd_histogram = signal_data.get('macd_histogram', 0)
            
            # 🔥 EFFICIENCY RATIO SCORING (Most important for KAMA)
            # KAMA's effectiveness is directly tied to market efficiency
            efficiency_score = 0.0
            if efficiency_ratio >= 0.7:
                efficiency_score = 0.95  # Excellent - market is very directional
            elif efficiency_ratio >= 0.5:
                efficiency_score = 0.85  # Very good - clear market direction
            elif efficiency_ratio >= 0.35:
                efficiency_score = 0.70  # Good - decent market direction
            elif efficiency_ratio >= 0.25:
                efficiency_score = 0.55  # Acceptable - minimum for good signals
            elif efficiency_ratio >= 0.15:
                efficiency_score = 0.35  # Poor - choppy market
            else:
                efficiency_score = 0.15  # Very poor - avoid trading
            
            # 📈 KAMA TREND SCORING
            # Strong KAMA trends indicate market conviction
            trend_score = min(kama_trend * 2500, 1.0)  # Scale appropriately for forex
            if trend_score < 0.1:
                trend_score = 0.1  # Minimum trend requirement
            
            # 💪 SIGNAL STRENGTH SCORING
            strength_score = min(signal_strength, 0.9)
            if strength_score < 0.2:
                strength_score = 0.2
            
            # 📏 PRICE-KAMA ALIGNMENT SCORING
            alignment_score = 1.0
            if kama_value > 0 and price > 0:
                distance_ratio = abs(price - kama_value) / price
                if distance_ratio > 0.015:  # More than 1.5% away
                    alignment_score = 0.6
                elif distance_ratio > 0.008:  # More than 0.8% away
                    alignment_score = 0.75
                elif distance_ratio > 0.004:  # More than 0.4% away
                    alignment_score = 0.9
            
            # 🎯 KAMA-WEIGHTED COMBINATION
            # Efficiency ratio gets the highest weight because it's KAMA's core strength
            base_confidence = (
                efficiency_score * 0.50 +      # Efficiency is KAMA's most important factor
                trend_score * 0.25 +           # Trend strength
                strength_score * 0.15 +        # Signal strength
                alignment_score * 0.10         # Price-KAMA alignment
            )
            
            # 🎯 KAMA-SPECIFIC BONUSES AND PENALTIES
            
            # High efficiency bonus (KAMA's sweet spot)
            if efficiency_ratio > 0.6:
                base_confidence += 0.08
            elif efficiency_ratio > 0.4:
                base_confidence += 0.05
            
            # Market adaptability bonus (KAMA adapts to volatility)
            if 0.3 <= efficiency_ratio <= 0.8:
                base_confidence += 0.03  # Sweet spot for KAMA
            
            # Trend consistency bonus
            if ((signal_type in ['BULL', 'BUY'] and kama_trend > 0.0005) or 
                (signal_type in ['BEAR', 'SELL'] and kama_trend < -0.0005)):
                base_confidence += 0.06
            
            # MACD confirmation bonus (if available)
            if ((signal_type in ['BULL', 'BUY'] and macd_histogram > 0.0001) or
                (signal_type in ['BEAR', 'SELL'] and macd_histogram < -0.0001)):
                base_confidence += 0.04
            elif ((signal_type in ['BULL', 'BUY'] and macd_histogram < -0.0001) or
                  (signal_type in ['BEAR', 'SELL'] and macd_histogram > 0.0001)):
                base_confidence -= 0.06  # MACD contradiction penalty
            
            # Low efficiency penalty (KAMA struggles in choppy markets)
            if efficiency_ratio < 0.2:
                base_confidence -= 0.10
            elif efficiency_ratio < 0.25:
                base_confidence -= 0.05
            
            # 🔒 BOUNDS: Keep confidence in reasonable range
            return max(0.1, min(0.92, base_confidence))
            
        except Exception as e:
            self.logger.error(f"KAMA focused confidence calculation error: {e}")
            return 0.5

    def calculate_basic_confidence(self, signal_data: Dict) -> float:
        """💪 KAMA: Calculate basic confidence for KAMA signals"""
        try:
            efficiency_ratio = signal_data.get('efficiency_ratio', 0)
            signal_strength = signal_data.get('signal_strength', 0)
            kama_trend = abs(signal_data.get('kama_trend', 0))
            
            # Base KAMA confidence calculation
            base_confidence = (efficiency_ratio * 0.6) + (signal_strength * 0.4)
            
            # Bonus for strong trend
            if kama_trend > 0.05:  # Strong trend threshold
                base_confidence += 0.1
            elif kama_trend > 0.02:
                base_confidence += 0.05
            
            # Apply forex adjustments if available
            if self.forex_optimizer:
                adjusted_confidence = self.forex_optimizer.apply_forex_confidence_adjustments(
                    base_confidence, signal_data.get('epic', ''), signal_data
                )
                return adjusted_confidence
            
            return max(0.3, min(0.8, base_confidence))
            
        except Exception as e:
            self.logger.error(f"Basic confidence calculation error: {e}")
            return 0.5

    def calculate_weighted_confidence(
        self, 
        signal_data: Dict, 
        validations: List[Tuple[bool, str, float]], 
        epic: str = None
    ) -> float:
        """
        ⚖️ KAMA: Calculate weighted confidence based on multiple validation results
        
        Args:
            signal_data: Signal data dictionary
            validations: List of (passed, reason, weight) tuples
            epic: Trading pair epic
        """
        try:
            total_weight = 0
            weighted_score = 0
            
            for passed, reason, weight in validations:
                total_weight += weight
                if passed:
                    weighted_score += weight
            
            if total_weight == 0:
                return 0.3  # Fallback
            
            # Base weighted score
            base_score = weighted_score / total_weight
            
            # Apply KAMA-specific bonuses
            efficiency_ratio = signal_data.get('efficiency_ratio', 0)
            if efficiency_ratio > 0.7:
                base_score += 0.15  # High efficiency bonus
            elif efficiency_ratio > 0.5:
                base_score += 0.10
            elif efficiency_ratio > 0.3:
                base_score += 0.05
            
            # Apply forex adjustments
            if self.forex_optimizer:
                final_confidence = self.forex_optimizer.apply_forex_confidence_adjustments(
                    base_score, epic or '', signal_data
                )
            else:
                final_confidence = base_score
            
            return max(0.2, min(0.95, final_confidence))
            
        except Exception as e:
            self.logger.error(f"Weighted confidence calculation error: {e}")
            return 0.5

    def validate_signal_comprehensive(self, signal_data: Dict, df: pd.DataFrame = None, epic: str = None) -> Dict:
        """
        🔍 COMPREHENSIVE: Perform comprehensive signal validation
        """
        try:
            self._validation_count += 1
            
            validation_results = {
                'passed': False,
                'confidence': 0.0,
                'validations': [],
                'rejections': [],
                'warnings': []
            }
            
            # 1. Efficiency validation
            efficiency_passed, efficiency_reason = self.validate_kama_efficiency(signal_data, epic)
            validation_results['validations'].append(('efficiency', efficiency_passed, efficiency_reason))
            if not efficiency_passed:
                validation_results['rejections'].append(efficiency_reason)
            
            # 2. Trend alignment validation
            trend_passed, trend_reason = self.validate_kama_trend_alignment(signal_data, epic)
            validation_results['validations'].append(('trend_alignment', trend_passed, trend_reason))
            if not trend_passed:
                validation_results['rejections'].append(trend_reason)
            
            # 3. Market conditions validation
            market_passed, market_reason = self.validate_market_conditions(signal_data, df, epic)
            validation_results['validations'].append(('market_conditions', market_passed, market_reason))
            if not market_passed:
                validation_results['rejections'].append(market_reason)
            
            # Overall validation result
            if efficiency_passed and trend_passed and market_passed:
                validation_results['passed'] = True
                validation_results['confidence'] = self.calculate_enhanced_confidence(signal_data, df, epic)
            else:
                self._rejection_count += 1
                validation_results['confidence'] = 0.0
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive validation error: {e}")
            return {
                'passed': False,
                'confidence': 0.0,
                'validations': [],
                'rejections': [f"Validation error: {e}"],
                'warnings': []
            }

    def _track_rejection(self, reason: str):
        """📊 Track rejection reasons for statistics"""
        self._rejection_reasons[reason] = self._rejection_reasons.get(reason, 0) + 1

    def get_validation_stats(self) -> Dict:
        """📊 Get validation statistics"""
        try:
            total_validations = max(self._validation_count, 1)
            return {
                'module': 'kama_validator',
                'total_validations': self._validation_count,
                'total_rejections': self._rejection_count,
                'acceptance_rate': (self._validation_count - self._rejection_count) / total_validations,
                'rejection_reasons': dict(self._rejection_reasons),
                'error': None
            }
        except Exception as e:
            return {'error': str(e)}

    def reset_stats(self):
        """🔄 Reset validation statistics"""
        self._validation_count = 0
        self._rejection_count = 0
        self._rejection_reasons.clear()
        self.logger.debug("🔄 KAMA Validator statistics reset")