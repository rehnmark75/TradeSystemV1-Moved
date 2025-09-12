# core/strategies/helpers/bb_validator.py
"""
BB Validator Module - Extracted from BB Supertrend Strategy
🎯 VALIDATION: Signal validation and confidence calculation for BB+Supertrend
📊 COMPREHENSIVE: Multi-component validation with forex-specific thresholds
🧠 SMART: Integration with Enhanced Signal Validator

This module contains all the validation logic for BB+Supertrend strategy,
extracted for better maintainability and testability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging
from datetime import datetime


class BBValidator:
    """
    🎯 VALIDATION: Comprehensive signal validation for BB+Supertrend strategy
    
    Responsibilities:
    - BB position quality validation
    - SuperTrend alignment validation
    - Market condition validation
    - Confidence score calculation
    - Risk assessment
    - Integration with Enhanced Signal Validator
    """
    
    def __init__(self, logger: logging.Logger = None, forex_optimizer=None):
        self.logger = logger or logging.getLogger(__name__)
        self.forex_optimizer = forex_optimizer  # Will be injected by main strategy
        
        # Validation thresholds optimized for BB+Supertrend
        self.validation_thresholds = {
            'min_bb_threshold': 0.60,           # Lower than EMA strategies (mean reversion)
            'min_bb_position_quality': 0.5,    # Minimum BB position quality
            'min_supertrend_alignment': 0.7,   # SuperTrend alignment threshold
            'min_volatility_score': 0.4,       # Minimum volatility score
            'min_efficiency_ratio': 0.15,      # Minimum for BB (ranging markets OK)
            'max_efficiency_ratio': 0.8        # Maximum efficiency ratio
        }
        
        # Component weights for validation scoring
        self.validation_weights = {
            'bb_position': 0.30,        # BB position quality (30%)
            'bb_width_volatility': 0.25, # BB width/volatility (25%)
            'supertrend_alignment': 0.25, # SuperTrend alignment (25%)
            'market_efficiency': 0.20   # Market efficiency (20%)
        }
        
        # Validation statistics
        self._validations_performed = 0
        self._signals_approved = 0
        self._signals_rejected = 0
        
        self.logger.info("🎯 BB Validator initialized with forex-optimized thresholds")

    def validate_bb_supertrend_signal(
        self, 
        current: pd.Series, 
        previous: pd.Series, 
        signal_type: str, 
        epic: str,
        timeframe: str
    ) -> Tuple[bool, float, str]:
        """
        🎯 Comprehensive BB+SuperTrend signal validation
        """
        try:
            self._validations_performed += 1
            
            # Get cached calculations from forex optimizer if available
            if self.forex_optimizer:
                efficiency_ratio = self.forex_optimizer.calculate_forex_efficiency_ratio(current, previous)
                market_regime = self.forex_optimizer.detect_forex_market_regime(current)
            else:
                efficiency_ratio = 0.25
                market_regime = 'ranging'
            
            # Validate BB data quality first
            if not self._validate_bb_data_quality(current):
                self._signals_rejected += 1
                return False, 0.2, "Invalid BB data quality"
            
            # Calculate validation components
            validation_score = 0.0
            validation_details = []
            
            # Component 1: BB Position Quality (30% weight)
            bb_position_score = self._validate_bb_position_quality(current, signal_type)
            validation_score += bb_position_score * self.validation_weights['bb_position']
            validation_details.append(f"BB Position: {bb_position_score:.2f}")
            
            # Component 2: BB Width/Volatility (25% weight)
            volatility_score = self._validate_bb_width_volatility(current)
            validation_score += volatility_score * self.validation_weights['bb_width_volatility']
            validation_details.append(f"BB Width: {volatility_score:.2f}")
            
            # Component 3: SuperTrend Alignment (25% weight)
            st_score = self._validate_supertrend_alignment(current, signal_type)
            validation_score += st_score * self.validation_weights['supertrend_alignment']
            validation_details.append(f"SuperTrend: {st_score:.2f}")
            
            # Component 4: Market Efficiency (20% weight)
            efficiency_score = self._validate_market_efficiency(efficiency_ratio)
            validation_score += efficiency_score * self.validation_weights['market_efficiency']
            validation_details.append(f"Efficiency: {efficiency_score:.2f} ({efficiency_ratio:.3f})")
            
            # Apply forex-specific adjustments if forex optimizer is available
            if self.forex_optimizer:
                bb_position_quality = self.forex_optimizer.calculate_bb_position_score(current, signal_type)
                validation_score = self.forex_optimizer.apply_forex_confidence_adjustments(
                    validation_score, epic, market_regime, bb_position_quality
                )
            
            # Final decision
            should_trade = validation_score >= self.validation_thresholds['min_bb_threshold']
            
            if should_trade:
                self._signals_approved += 1
            else:
                self._signals_rejected += 1
            
            reason = f"BB+SuperTrend validation: {validation_score:.1%} ({'APPROVED' if should_trade else 'REJECTED'}) - {' | '.join(validation_details)} - Market: {market_regime}"
            
            self.logger.debug(f"🎯 BB Validation: {validation_score:.1%} (threshold: {self.validation_thresholds['min_bb_threshold']:.1%})")
            self.logger.debug(f"   Components: {' | '.join(validation_details)}")
            self.logger.debug(f"   Market Regime: {market_regime}")
            
            return should_trade, validation_score, reason
            
        except Exception as e:
            self.logger.error(f"BB validation error: {e}")
            self._signals_rejected += 1
            return False, 0.20, f"BB validation error: {str(e)}"

    def _validate_bb_data_quality(self, current: pd.Series) -> bool:
        """
        ✅ Validate Bollinger Bands data quality
        """
        try:
            required_bb_cols = ['bb_upper', 'bb_middle', 'bb_lower']
            
            for col in required_bb_cols:
                if col not in current.index or pd.isna(current[col]) or current[col] <= 0:
                    self.logger.debug(f"BB data quality check failed: {col}")
                    return False
            
            # Check logical order: upper > middle > lower
            if not (current['bb_upper'] > current['bb_middle'] > current['bb_lower']):
                self.logger.debug("BB logical order check failed")
                return False
            
            # Check minimum BB width if forex optimizer is available
            if self.forex_optimizer and not self.forex_optimizer.is_bb_width_sufficient(current):
                self.logger.debug("BB width insufficient")
                return False
                
            return True
            
        except Exception as e:
            self.logger.debug(f"BB data quality validation failed: {e}")
            return False

    def _validate_bb_position_quality(self, current: pd.Series, signal_type: str) -> float:
        """
        📍 Validate BB position quality (how close price is to the appropriate band)
        """
        try:
            if self.forex_optimizer:
                return self.forex_optimizer.calculate_bb_position_score(current, signal_type)
            else:
                # Fallback calculation
                bb_width = current['bb_upper'] - current['bb_lower']
                current_price = current['close']
                
                if signal_type == 'BULL':
                    distance_from_lower = current_price - current['bb_lower']
                    position_score = max(0.0, 1.0 - (distance_from_lower / (bb_width * 0.5)))
                else:  # BEAR
                    distance_from_upper = current['bb_upper'] - current_price
                    position_score = max(0.0, 1.0 - (distance_from_upper / (bb_width * 0.5)))
                
                return position_score
                
        except Exception as e:
            self.logger.debug(f"BB position quality validation failed: {e}")
            return 0.0

    def _validate_bb_width_volatility(self, current: pd.Series) -> float:
        """
        📊 Validate BB width and volatility conditions
        """
        try:
            if self.forex_optimizer:
                volatility_level, bb_width_pct = self.forex_optimizer.assess_bb_volatility(current)
                
                if volatility_level == 'high' and bb_width_pct >= 0.003:
                    return 0.95
                elif volatility_level == 'medium' and bb_width_pct >= 0.0015:
                    return 0.80
                elif volatility_level == 'low' and bb_width_pct >= 0.0008:
                    return 0.60
                else:
                    return 0.20
            else:
                # Fallback calculation
                bb_width = current['bb_upper'] - current['bb_lower']
                current_price = current['close']
                bb_width_percentage = bb_width / current_price if current_price > 0 else 0
                
                if bb_width_percentage >= 0.003:
                    return 0.95
                elif bb_width_percentage >= 0.0015:
                    return 0.80
                elif bb_width_percentage >= 0.0008:
                    return 0.60
                else:
                    return 0.20
                    
        except Exception as e:
            self.logger.debug(f"BB width/volatility validation failed: {e}")
            return 0.20

    def _validate_supertrend_alignment(self, current: pd.Series, signal_type: str) -> float:
        """
        📈 Validate SuperTrend alignment with signal direction
        """
        try:
            supertrend_direction = current.get('supertrend_direction', 0)
            expected_direction = 1 if signal_type == 'BULL' else -1
            current_price = current['close']
            
            if supertrend_direction == expected_direction:
                # Calculate distance to SuperTrend line for quality assessment
                st_distance = abs(current_price - current.get('supertrend', current_price))
                st_distance_pct = st_distance / current_price if current_price > 0 else 0
                
                if 0.001 <= st_distance_pct <= 0.01:  # Good SuperTrend distance for forex
                    return 0.90
                elif st_distance_pct < 0.001:  # Too close
                    return 0.60
                else:  # Acceptable distance
                    return 0.75
            else:
                return 0.20  # Wrong direction - penalize heavily
                
        except Exception as e:
            self.logger.debug(f"SuperTrend alignment validation failed: {e}")
            return 0.20

    def _validate_market_efficiency(self, efficiency_ratio: float) -> float:
        """
        📊 Validate market efficiency for BB strategy
        """
        try:
            # BB strategy can work in ranging markets, so more permissive thresholds
            if efficiency_ratio >= 0.35:  # Good directional movement
                return 0.90
            elif efficiency_ratio >= 0.20:  # Acceptable for BB strategy
                return 0.70
            elif efficiency_ratio >= self.validation_thresholds['min_efficiency_ratio']:  # Minimum for BB
                return 0.50
            else:  # Too choppy even for BB
                return 0.20
                
        except Exception as e:
            self.logger.debug(f"Market efficiency validation failed: {e}")
            return 0.20

    def calculate_legacy_confidence(self, signal_data: Dict) -> float:
        """
        🔄 Calculate confidence using legacy format for backward compatibility
        """
        try:
            # Extract data from signal dictionary
            current_data = {
                'close': signal_data.get('entry_price', signal_data.get('price', 0)),
                'bb_upper': signal_data.get('bb_upper', 0),
                'bb_middle': signal_data.get('bb_middle', 0), 
                'bb_lower': signal_data.get('bb_lower', 0),
                'supertrend': signal_data.get('supertrend', 0),
                'supertrend_direction': signal_data.get('supertrend_direction', 0),
                'atr': signal_data.get('atr', 0.001)
            }
            
            current_series = pd.Series(current_data)
            signal_type = signal_data.get('signal_type', 'BULL').upper()
            
            # Use main validation method
            _, confidence, _ = self.validate_bb_supertrend_signal(
                current_series, current_series, signal_type, 
                signal_data.get('epic', ''), signal_data.get('timeframe', '15m')
            )
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Legacy confidence calculation failed: {e}")
            return 0.60  # Default confidence

    def get_validation_stats(self) -> Dict:
        """
        📊 Get validation performance statistics
        """
        total_validations = self._validations_performed
        return {
            'total_validations': total_validations,
            'signals_approved': self._signals_approved,
            'signals_rejected': self._signals_rejected,
            'approval_rate': self._signals_approved / max(total_validations, 1),
            'rejection_rate': self._signals_rejected / max(total_validations, 1),
            'validation_thresholds': self.validation_thresholds,
            'validation_weights': self.validation_weights
        }

    def reset_stats(self):
        """🔄 Reset validation statistics"""
        self._validations_performed = 0
        self._signals_approved = 0
        self._signals_rejected = 0
        self.logger.info("🔄 BB Validator stats reset")

    def get_cache_stats(self) -> Dict:
        """📊 Get cache-related statistics (for consistency with other modules)"""
        return {
            'validator_stats': self.get_validation_stats(),
            'thresholds_count': len(self.validation_thresholds),
            'weights_count': len(self.validation_weights)
        }