# core/strategies/helpers/combined_validator.py
"""
Combined Strategy Validator - MODULAR HELPER
🔥 FOREX OPTIMIZED: Combined strategy specific validation and confidence calculation
🏗️ MODULAR: Focused on validation logic for combined strategy
🎯 MAINTAINABLE: Single responsibility - validation only
⚡ PERFORMANCE: Efficient validation with caching
🚨 SAFETY: Critical contradiction detection and safety filters
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import config
except ImportError:
    from forex_scanner import config


class CombinedValidator:
    """
    🔥 VALIDATOR: Combined strategy specific validation logic
    
    Handles:
    - Data quality validation
    - Enhanced confidence calculation
    - Signal consensus validation
    - Safety filter application
    - Mean reversion strategy bypass logic
    """
    
    def __init__(self, logger: logging.Logger, forex_optimizer=None):
        self.logger = logger
        self.forex_optimizer = forex_optimizer
        
        # Validation thresholds
        self.validation_config = self._load_validation_config()
        
        # Safety filter configuration
        self.safety_filters = self._load_safety_filter_config()
        
        # Mean reversion bypass patterns
        self.mean_reversion_patterns = self._load_mean_reversion_patterns()
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'safety_filter_rejections': 0,
            'mean_reversion_bypasses': 0
        }
        
        self.logger.debug("✅ CombinedValidator initialized")

    def _load_validation_config(self) -> Dict[str, Any]:
        """Load validation configuration from config"""
        return {
            'min_confidence_threshold': getattr(config, 'MIN_COMBINED_CONFIDENCE', 0.75),
            'consensus_threshold': getattr(config, 'CONSENSUS_THRESHOLD', 0.7),
            'min_confirming_indicators': getattr(config, 'MIN_CONFIRMING_INDICATORS', 1),
            'max_contradictions_allowed': getattr(config, 'MAX_CONTRADICTIONS_ALLOWED', 1),
            'require_volume_confirmation': getattr(config, 'REQUIRE_VOLUME_CONFIRMATION', False),
            'ema200_minimum_margin': getattr(config, 'EMA200_MINIMUM_MARGIN', 0.002),
            'macd_strong_threshold': getattr(config, 'MACD_STRONG_THRESHOLD', 0.0001)
        }

    def _load_safety_filter_config(self) -> Dict[str, bool]:
        """Load safety filter configuration"""
        return {
            'ema200_contradiction_filter': getattr(config, 'ENABLE_EMA200_CONTRADICTION_FILTER', True),
            'macd_contradiction_filter': getattr(config, 'ENABLE_MACD_CONTRADICTION_FILTER', True),
            'ema_stack_contradiction_filter': getattr(config, 'ENABLE_EMA_STACK_CONTRADICTION_FILTER', True),
            'require_indicator_consensus': getattr(config, 'REQUIRE_INDICATOR_CONSENSUS', True),
            'emergency_circuit_breaker': getattr(config, 'ENABLE_EMERGENCY_CIRCUIT_BREAKER', True)
        }

    def _load_mean_reversion_patterns(self) -> List[str]:
        """Load patterns that identify mean reversion strategies"""
        return [
            'bollinger',
            'bb_',
            'rsi_reversal',
            'stochastic_reversal',
            'oversold_reversal',
            'overbought_reversal',
            'mean_reversion',
            'support_resistance',
            'pivot_reversal'
        ]

    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality for combined strategy
        
        Args:
            df: DataFrame with market data
            
        Returns:
            True if data quality is acceptable
        """
        try:
            self.validation_stats['total_validations'] += 1
            
            # Basic validation
            if len(df) < self.validation_config.get('minimum_bars', 50):
                self.logger.debug(f"🚫 Insufficient data: {len(df)} bars")
                self.validation_stats['failed_validations'] += 1
                return False
            
            # Check for essential columns
            essential_cols = ['close', 'high', 'low', 'open']
            missing_essential = [col for col in essential_cols if col not in df.columns]
            if missing_essential:
                self.logger.warning(f"⚠️ Missing essential columns: {missing_essential}")
                self.validation_stats['failed_validations'] += 1
                return False
            
            # Check for NaN values in latest data
            latest = df.iloc[-1]
            for col in essential_cols:
                if pd.isna(latest[col]):
                    self.logger.warning(f"⚠️ NaN value in essential column {col}")
                    self.validation_stats['failed_validations'] += 1
                    return False
            
            # Validate price consistency
            if not self._validate_price_consistency(latest):
                self.validation_stats['failed_validations'] += 1
                return False
            
            self.validation_stats['passed_validations'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data quality validation error: {e}")
            self.validation_stats['failed_validations'] += 1
            return False

    def _validate_price_consistency(self, latest_data: pd.Series) -> bool:
        """Validate that price data is consistent"""
        try:
            high = latest_data.get('high', 0)
            low = latest_data.get('low', 0)
            open_price = latest_data.get('open', 0)
            close = latest_data.get('close', 0)
            
            # Basic price relationship checks
            if high < low:
                self.logger.warning(f"⚠️ Invalid price data: high {high} < low {low}")
                return False
            
            if not (low <= open_price <= high):
                self.logger.warning(f"⚠️ Open price {open_price} outside high-low range [{low}, {high}]")
                return False
            
            if not (low <= close <= high):
                self.logger.warning(f"⚠️ Close price {close} outside high-low range [{low}, {high}]")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Price consistency validation failed: {e}")
            return False

    def calculate_enhanced_confidence(self, signal_data: Dict) -> float:
        """
        Calculate enhanced confidence using comprehensive validation factors
        
        Args:
            signal_data: Dictionary containing signal information
            
        Returns:
            Enhanced confidence score (0.0 to 1.0)
        """
        try:
            # Get base confidence
            base_confidence = signal_data.get('raw_confidence', signal_data.get('confidence_score', 0.8))
            
            # Calculate validation factors
            validation_factors = self._calculate_validation_factors(signal_data)
            
            # Apply validation factors
            enhanced_confidence = self._apply_validation_factors(base_confidence, validation_factors)
            
            # Apply final quality checks
            final_confidence = self._apply_quality_checks(enhanced_confidence, signal_data, validation_factors)
            
            # Apply forex adjustments if available
            if self.forex_optimizer:
                epic = signal_data.get('epic', '')
                final_confidence = self.forex_optimizer.apply_forex_confidence_adjustments(final_confidence, epic)
            
            self.logger.debug(f"[ENHANCED CONFIDENCE] Base: {base_confidence:.1%} → Final: {final_confidence:.1%}")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced confidence calculation failed: {e}")
            return min(0.6, signal_data.get('confidence_score', 0.5))

    def _calculate_validation_factors(self, signal_data: Dict) -> Dict[str, float]:
        """Calculate comprehensive validation factors"""
        factors = {}
        
        try:
            # Factor 1: Strategy consensus quality
            factors['strategy_consensus'] = self._validate_strategy_consensus_factor(signal_data)
            
            # Factor 2: Individual strategy confidence quality
            factors['individual_confidence'] = self._validate_individual_confidence_factor(signal_data)
            
            # Factor 3: Market timing factor
            factors['market_timing'] = self._validate_market_timing_factor(signal_data)
            
            # Factor 4: Volume confirmation factor
            factors['volume_confirmation'] = self._validate_volume_confirmation_factor(signal_data)
            
            # Factor 5: Technical alignment factor
            factors['technical_alignment'] = self._validate_technical_alignment_factor(signal_data)
            
            # Factor 6: Trend alignment factor
            factors['trend_alignment'] = self._validate_trend_alignment_factor(signal_data)
            
            # Factor 7: Risk management factor
            factors['risk_management'] = self._validate_risk_management_factor(signal_data)
            
            self.logger.debug(f"[VALIDATION FACTORS] Calculated {len(factors)} factors")
            
        except Exception as e:
            self.logger.error(f"❌ Validation factors calculation failed: {e}")
            # Return conservative factors
            factors = {
                'strategy_consensus': 0.5,
                'individual_confidence': 0.5,
                'market_timing': 0.5,
                'volume_confirmation': 0.5,
                'technical_alignment': 0.5,
                'trend_alignment': 0.5,
                'risk_management': 0.5
            }
        
        return factors

    def _validate_strategy_consensus_factor(self, signal_data: Dict) -> float:
        """Validate strategy consensus quality"""
        try:
            contributing_strategies = signal_data.get('contributing_strategies', [])
            individual_confidences = signal_data.get('individual_confidences', {})
            total_strategies = signal_data.get('total_strategies_attempted', len(contributing_strategies))
            
            if not contributing_strategies:
                return 0.5  # Single strategy - neutral
            
            # Strategy diversity score
            diversity_score = min(1.0, len(contributing_strategies) / 4)  # Max 4 strategies expected
            
            # Average individual confidence
            if individual_confidences:
                avg_confidence = sum(individual_confidences.values()) / len(individual_confidences)
                confidence_bonus = min(1.0, avg_confidence / 0.8)  # Scale to 80% baseline
            else:
                confidence_bonus = 0.5
            
            # Agreement strength
            agreement_strength = len(contributing_strategies) / max(1, total_strategies)
            
            # Combined consensus score
            consensus_score = (diversity_score * 0.4 + confidence_bonus * 0.4 + agreement_strength * 0.2)
            
            return max(0.3, min(1.0, consensus_score))
            
        except Exception:
            return 0.5

    def _validate_individual_confidence_factor(self, signal_data: Dict) -> float:
        """Validate individual strategy confidence levels"""
        try:
            individual_confidences = signal_data.get('individual_confidences', {})
            
            if not individual_confidences:
                return 0.5
            
            # Check for high individual confidences
            high_confidence_count = sum(1 for conf in individual_confidences.values() if conf > 0.8)
            total_strategies = len(individual_confidences)
            
            # Check for very low confidences (red flag)
            low_confidence_count = sum(1 for conf in individual_confidences.values() if conf < 0.6)
            
            # Calculate factor
            high_confidence_ratio = high_confidence_count / total_strategies
            low_confidence_penalty = low_confidence_count / total_strategies
            
            factor = high_confidence_ratio - (low_confidence_penalty * 0.5)
            
            return max(0.2, min(1.0, factor))
            
        except Exception:
            return 0.5

    def _validate_market_timing_factor(self, signal_data: Dict) -> float:
        """Validate market timing quality"""
        try:
            current_hour = datetime.now().hour
            
            # Market session quality (simplified)
            if 8 <= current_hour <= 17:  # London session
                return 0.9
            elif 13 <= current_hour <= 22:  # New York session
                return 0.8
            elif 22 <= current_hour <= 6:  # Asian session
                return 0.6
            else:  # Overlap periods
                return 0.7
                
        except Exception:
            return 0.6

    def _validate_volume_confirmation_factor(self, signal_data: Dict) -> float:
        """Validate volume confirmation"""
        try:
            volume_confirmation = signal_data.get('volume_confirmation', None)
            volume_ratio = signal_data.get('volume_ratio', 1.0)
            
            if volume_confirmation is True:
                return min(1.0, volume_ratio / 2.0)  # Scale by volume ratio
            elif volume_confirmation is False:
                return 0.3  # Explicit volume rejection
            else:
                # Calculate from volume ratio
                if volume_ratio >= 1.5:
                    return 0.8  # High volume
                elif volume_ratio >= 1.2:
                    return 0.6  # Medium volume
                else:
                    return 0.4  # Low volume
                    
        except Exception:
            return 0.5

    def _validate_technical_alignment_factor(self, signal_data: Dict) -> float:
        """Validate technical indicator alignment"""
        try:
            signal_type = signal_data.get('signal_type', '').upper()
            
            # Get technical indicators
            ema_9 = signal_data.get('ema_9', 0)
            ema_21 = signal_data.get('ema_21', 0)
            ema_200 = signal_data.get('ema_200', 0)
            macd_histogram = signal_data.get('macd_histogram', 0)
            current_price = signal_data.get('signal_price', signal_data.get('price', 0))
            
            if not all([ema_9, ema_21, ema_200, current_price]):
                return 0.5  # Missing data
            
            alignment_score = 0.0
            total_checks = 0
            
            # EMA alignment check
            if signal_type in ['BUY', 'BULL']:
                if ema_9 > ema_21 > ema_200:
                    alignment_score += 1.0
                elif ema_9 > ema_21:
                    alignment_score += 0.6
                total_checks += 1
                
                # Price above EMA 200
                if current_price > ema_200:
                    alignment_score += 1.0
                total_checks += 1
                
            elif signal_type in ['SELL', 'BEAR']:
                if ema_9 < ema_21 < ema_200:
                    alignment_score += 1.0
                elif ema_9 < ema_21:
                    alignment_score += 0.6
                total_checks += 1
                
                # Price below EMA 200
                if current_price < ema_200:
                    alignment_score += 1.0
                total_checks += 1
            
            # MACD alignment check
            if macd_histogram != 0:
                if signal_type in ['BUY', 'BULL'] and macd_histogram > 0:
                    alignment_score += 1.0
                elif signal_type in ['SELL', 'BEAR'] and macd_histogram < 0:
                    alignment_score += 1.0
                else:
                    alignment_score += 0.2  # Contradictory but not zero
                total_checks += 1
            
            return alignment_score / max(1, total_checks) if total_checks > 0 else 0.5
            
        except Exception:
            return 0.5

    def _validate_trend_alignment_factor(self, signal_data: Dict) -> float:
        """Validate trend alignment with signal"""
        try:
            signal_type = signal_data.get('signal_type', '').upper()
            ema_trend = signal_data.get('ema_trend', signal_data.get('ema_200', 0))
            price = signal_data.get('signal_price', signal_data.get('price', 0))
            
            if not all([ema_trend, price]):
                return 0.5  # Missing data
            
            # Calculate price distance from trend line
            distance_pct = abs(price - ema_trend) / ema_trend * 100
            
            if signal_type in ['BUY', 'BULL']:
                if price > ema_trend:
                    if distance_pct > 2.0:  # > 2% above trend line
                        return 0.9
                    elif distance_pct > 0.5:  # > 0.5% above trend line
                        return 0.7
                    else:
                        return 0.6
                else:
                    return 0.2  # Bullish signal below trend
                    
            elif signal_type in ['SELL', 'BEAR']:
                if price < ema_trend:
                    if distance_pct > 2.0:  # > 2% below trend line
                        return 0.9
                    elif distance_pct > 0.5:  # > 0.5% below trend line
                        return 0.7
                    else:
                        return 0.6
                else:
                    return 0.2  # Bearish signal above trend
            
            return 0.5
            
        except Exception:
            return 0.5

    def _validate_risk_management_factor(self, signal_data: Dict) -> float:
        """Validate risk management aspects"""
        try:
            stop_loss = signal_data.get('stop_loss')
            take_profit = signal_data.get('take_profit')
            entry_price = signal_data.get('signal_price', signal_data.get('price', 0))
            
            if not all([stop_loss, take_profit, entry_price]):
                return 0.5  # Missing risk management data
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk == 0:
                return 0.3  # Invalid risk management
            
            risk_reward_ratio = reward / risk
            
            # Score based on risk/reward ratio
            if risk_reward_ratio >= 2.0:
                return 0.9  # Excellent risk/reward
            elif risk_reward_ratio >= 1.5:
                return 0.7  # Good risk/reward
            elif risk_reward_ratio >= 1.0:
                return 0.5  # Acceptable risk/reward
            else:
                return 0.3  # Poor risk/reward
                
        except Exception:
            return 0.5

    def _apply_validation_factors(self, base_confidence: float, validation_factors: Dict[str, float]) -> float:
        """Apply validation factors to base confidence"""
        try:
            # Validation factor weights (must sum to 1.0)
            weights = {
                'strategy_consensus': 0.25,
                'individual_confidence': 0.20,
                'technical_alignment': 0.15,
                'trend_alignment': 0.15,
                'volume_confirmation': 0.10,
                'risk_management': 0.10,
                'market_timing': 0.05
            }
            
            # Calculate weighted validation score
            validation_score = 0.0
            total_weight = 0.0
            
            for factor, value in validation_factors.items():
                if factor in weights and value is not None:
                    weight = weights[factor]
                    validation_score += value * weight
                    total_weight += weight
            
            # Normalize validation score
            if total_weight > 0:
                normalized_validation = validation_score / total_weight
            else:
                normalized_validation = 0.5
            
            # Apply validation to confidence
            if normalized_validation < 0.5:
                # Poor validation - reduce confidence significantly
                enhanced_confidence = base_confidence * (normalized_validation * 1.2)
            elif normalized_validation < 0.7:
                # Moderate validation - slight penalty
                enhanced_confidence = base_confidence * (normalized_validation * 1.1)
            else:
                # Good validation - reward confidence
                enhanced_confidence = base_confidence * min(1.2, normalized_validation + 0.2)
            
            # Ensure bounds
            enhanced_confidence = max(0.1, min(0.98, enhanced_confidence))
            
            return enhanced_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Validation factors application failed: {e}")
            return min(base_confidence * 0.8, 0.6)

    def _apply_quality_checks(self, enhanced_confidence: float, signal_data: Dict, validation_factors: Dict) -> float:
        """Apply final quality checks to prevent bad signals"""
        try:
            final_confidence = enhanced_confidence
            
            # Quality Check 1: Minimum validation standards
            critical_factors = ['technical_alignment', 'trend_alignment']
            for factor in critical_factors:
                if validation_factors.get(factor, 0.5) < 0.3:
                    self.logger.debug(f"[QUALITY CHECK] Critical factor {factor} too low: {validation_factors[factor]:.1%}")
                    final_confidence *= 0.7
            
            # Quality Check 2: Strategy consensus requirement for high confidence
            if enhanced_confidence > 0.8:
                strategy_consensus = validation_factors.get('strategy_consensus', 0.5) 
                if strategy_consensus < 0.6:
                    self.logger.debug(f"[QUALITY CHECK] High confidence requires better consensus: {strategy_consensus:.1%}")
                    final_confidence = min(final_confidence, 0.75)
            
            # Quality Check 3: Prevent artificially high confidence
            if final_confidence > 0.85:
                avg_validation = sum(validation_factors.values()) / len(validation_factors)
                if avg_validation < 0.8:
                    self.logger.debug(f"[QUALITY CHECK] High confidence blocked: avg validation {avg_validation:.1%} < 80%")
                    final_confidence = min(final_confidence, 0.8)
            
            # Quality Check 4: Volume confirmation requirement for strong signals
            if final_confidence > 0.8 and validation_factors.get('volume_confirmation', 0.5) < 0.4:
                self.logger.debug(f"[QUALITY CHECK] Strong signal requires volume confirmation")
                final_confidence *= 0.9
            
            # Final bounds check
            final_confidence = max(0.1, min(0.95, final_confidence))
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Quality checks failed: {e}")
            return min(enhanced_confidence * 0.9, 0.8)

    def validate_signal_consensus(self, all_signals: Dict, epic: str) -> Dict:
        """
        Validate consensus among multiple strategies
        
        Args:
            all_signals: Dictionary of signals from all strategies
            epic: Trading epic
            
        Returns:
            Dictionary with consensus validation results
        """
        try:
            valid_signals = {k: v for k, v in all_signals.items() if v is not None}
            
            if len(valid_signals) < 2:
                return {
                    'consensus_achieved': False,
                    'reason': 'insufficient_strategies',
                    'strategy_count': len(valid_signals),
                    'recommendation': 'single_strategy_or_wait'
                }
            
            # Count signals by type
            bull_signals = {k: v for k, v in valid_signals.items() if v.get('signal_type') == 'BULL'}
            bear_signals = {k: v for k, v in valid_signals.items() if v.get('signal_type') == 'BEAR'}
            
            total_signals = len(valid_signals)
            consensus_threshold = self.validation_config['consensus_threshold']
            min_consensus = max(2, int(total_signals * consensus_threshold))
            
            if len(bull_signals) >= min_consensus:
                return {
                    'consensus_achieved': True,
                    'signal_type': 'BULL',
                    'consensus_strength': len(bull_signals) / total_signals,
                    'contributing_strategies': list(bull_signals.keys()),
                    'strategy_count': len(bull_signals),
                    'individual_confidences': {k: v.get('confidence_score', 0) for k, v in bull_signals.items()}
                }
            elif len(bear_signals) >= min_consensus:
                return {
                    'consensus_achieved': True,
                    'signal_type': 'BEAR',
                    'consensus_strength': len(bear_signals) / total_signals,
                    'contributing_strategies': list(bear_signals.keys()),
                    'strategy_count': len(bear_signals),
                    'individual_confidences': {k: v.get('confidence_score', 0) for k, v in bear_signals.items()}
                }
            else:
                return {
                    'consensus_achieved': False,
                    'reason': 'insufficient_consensus',
                    'bull_count': len(bull_signals),
                    'bear_count': len(bear_signals),
                    'required_consensus': min_consensus,
                    'recommendation': 'wait_for_clearer_signal'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Consensus validation failed: {e}")
            return {
                'consensus_achieved': False,
                'reason': 'validation_error',
                'error': str(e)
            }

    def apply_safety_filters(self, signal: Dict, latest_data: pd.Series, epic: str) -> Dict:
        """
        Apply comprehensive safety filters to prevent bad signals
        
        Args:
            signal: Signal dictionary to validate
            latest_data: Latest market data
            epic: Trading epic
            
        Returns:
            Dictionary with safety filter results
        """
        try:
            # Check if this is a mean reversion strategy (bypass safety filters)
            if self._is_mean_reversion_strategy(signal):
                return {
                    'passed_all_filters': True,
                    'bypass_reason': 'mean_reversion_strategy',
                    'filters_applied': [],
                    'mean_reversion_bypass': True
                }
            
            safety_results = {
                'passed_all_filters': True,
                'failed_filters': [],
                'filter_results': {},
                'mean_reversion_bypass': False
            }
            
            # Apply each safety filter
            filters_to_apply = [
                ('ema200_contradiction', self._validate_ema200_filter),
                ('macd_contradiction', self._validate_macd_filter),
                ('ema_stack_contradiction', self._validate_ema_stack_filter),
                ('indicator_consensus', self._validate_indicator_consensus_filter),
                ('emergency_circuit_breaker', self._validate_circuit_breaker_filter)
            ]
            
            for filter_name, filter_func in filters_to_apply:
                if self.safety_filters.get(filter_name, True):  # Default to enabled
                    try:
                        filter_result = filter_func(signal, latest_data, epic)
                        safety_results['filter_results'][filter_name] = filter_result
                        
                        if not filter_result['passed']:
                            safety_results['passed_all_filters'] = False
                            safety_results['failed_filters'].append(filter_name)
                            self.validation_stats['safety_filter_rejections'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"❌ Safety filter {filter_name} failed: {e}")
                        safety_results['filter_results'][filter_name] = {
                            'passed': False,
                            'error': str(e)
                        }
                        safety_results['passed_all_filters'] = False
                        safety_results['failed_filters'].append(filter_name)
            
            return safety_results
            
        except Exception as e:
            self.logger.error(f"❌ Safety filters application failed: {e}")
            return {
                'passed_all_filters': False,
                'error': str(e),
                'failed_filters': ['safety_filter_error']
            }

    def _is_mean_reversion_strategy(self, signal: Dict) -> bool:
        """Check if signal is from mean reversion strategy"""
        try:
            # Check explicit bypass flags
            if signal.get('bypass_ema200_trend_filter', False):
                self.validation_stats['mean_reversion_bypasses'] += 1
                return True
                
            if signal.get('is_mean_reversion_strategy', False):
                self.validation_stats['mean_reversion_bypasses'] += 1
                return True
                
            if signal.get('contra_trend_allowed', False):
                self.validation_stats['mean_reversion_bypasses'] += 1
                return True
            
            # Check strategy type
            strategy_type = signal.get('strategy_type', '').lower()
            if strategy_type in ['mean_reversion', 'reversal', 'contrarian']:
                self.validation_stats['mean_reversion_bypasses'] += 1
                return True
            
            # Check strategy name patterns
            strategy = signal.get('strategy', '').lower()
            for pattern in self.mean_reversion_patterns:
                if pattern in strategy:
                    self.validation_stats['mean_reversion_bypasses'] += 1
                    return True
            
            # Check contributing strategies for mean reversion
            contributing_strategies = signal.get('contributing_strategies', [])
            for strategy_name in contributing_strategies:
                if any(pattern in str(strategy_name).lower() for pattern in self.mean_reversion_patterns):
                    self.validation_stats['mean_reversion_bypasses'] += 1
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Mean reversion check failed: {e}")
            return False

    def _validate_ema200_filter(self, signal: Dict, latest_data: pd.Series, epic: str) -> Dict:
        """Validate signal against EMA 200 trend filter"""
        try:
            signal_type = signal.get('signal_type', '').upper()
            current_price = signal.get('signal_price', signal.get('price', latest_data.get('close', 0)))
            ema_200 = latest_data.get('ema_200', 0)
            
            if ema_200 <= 0:
                return {'passed': True, 'reason': 'ema200_data_unavailable'}
            
            margin = self.validation_config['ema200_minimum_margin']
            price_diff_pct = abs(current_price - ema_200) / ema_200
            
            if signal_type in ['BUY', 'BULL']:
                if current_price > ema_200:
                    return {'passed': True, 'reason': 'price_above_ema200'}
                else:
                    return {
                        'passed': False,
                        'reason': f'bullish_signal_below_ema200',
                        'price': current_price,
                        'ema_200': ema_200,
                        'difference_pct': price_diff_pct
                    }
                    
            elif signal_type in ['SELL', 'BEAR']:
                if current_price < ema_200:
                    return {'passed': True, 'reason': 'price_below_ema200'}
                else:
                    return {
                        'passed': False,
                        'reason': f'bearish_signal_above_ema200',
                        'price': current_price,
                        'ema_200': ema_200,
                        'difference_pct': price_diff_pct
                    }
            
            return {'passed': True, 'reason': 'unknown_signal_type'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _validate_macd_filter(self, signal: Dict, latest_data: pd.Series, epic: str) -> Dict:
        """Validate signal against MACD momentum filter"""
        try:
            signal_type = signal.get('signal_type', '').upper()
            macd_histogram = latest_data.get('macd_histogram', 0)
            threshold = self.validation_config['macd_strong_threshold']
            
            if signal_type in ['BUY', 'BULL'] and macd_histogram < -threshold:
                return {
                    'passed': False,
                    'reason': 'bullish_signal_negative_macd',
                    'macd_histogram': macd_histogram,
                    'threshold': threshold
                }
                
            elif signal_type in ['SELL', 'BEAR'] and macd_histogram > threshold:
                return {
                    'passed': False,
                    'reason': 'bearish_signal_positive_macd',
                    'macd_histogram': macd_histogram,
                    'threshold': threshold
                }
            
            return {'passed': True, 'reason': 'macd_alignment_ok'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _validate_ema_stack_filter(self, signal: Dict, latest_data: pd.Series, epic: str) -> Dict:
        """Validate signal against EMA stack alignment"""
        try:
            signal_type = signal.get('signal_type', '').upper()
            ema_9 = latest_data.get('ema_9', 0)
            ema_21 = latest_data.get('ema_21', 0)
            ema_200 = latest_data.get('ema_200', 0)
            current_price = signal.get('signal_price', latest_data.get('close', 0))
            
            if not all([ema_9, ema_21, ema_200, current_price]):
                return {'passed': True, 'reason': 'incomplete_ema_data'}
            
            if signal_type in ['BUY', 'BULL']:
                perfect_bullish = current_price > ema_9 > ema_21 > ema_200
                if not perfect_bullish:
                    return {
                        'passed': False,
                        'reason': 'bullish_signal_poor_ema_stack',
                        'stack': f'Price({current_price:.5f}) > EMA9({ema_9:.5f}) > EMA21({ema_21:.5f}) > EMA200({ema_200:.5f})'
                    }
                    
            elif signal_type in ['SELL', 'BEAR']:
                perfect_bearish = current_price < ema_9 < ema_21 < ema_200
                if not perfect_bearish:
                    return {
                        'passed': False,
                        'reason': 'bearish_signal_poor_ema_stack',
                        'stack': f'Price({current_price:.5f}) < EMA9({ema_9:.5f}) < EMA21({ema_21:.5f}) < EMA200({ema_200:.5f})'
                    }
            
            return {'passed': True, 'reason': 'ema_stack_aligned'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _validate_indicator_consensus_filter(self, signal: Dict, latest_data: pd.Series, epic: str) -> Dict:
        """Validate multi-indicator consensus"""
        try:
            signal_type = signal.get('signal_type', '').upper()
            confirmations = []
            contradictions = []
            
            # EMA confirmation
            ema_9 = latest_data.get('ema_9', 0)
            ema_21 = latest_data.get('ema_21', 0)
            if signal_type in ['BUY', 'BULL'] and ema_9 > ema_21:
                confirmations.append('ema_bullish')
            elif signal_type in ['SELL', 'BEAR'] and ema_9 < ema_21:
                confirmations.append('ema_bearish')
            else:
                contradictions.append('ema_contradiction')
            
            # MACD confirmation
            macd_histogram = latest_data.get('macd_histogram', 0)
            if signal_type in ['BUY', 'BULL'] and macd_histogram > 0:
                confirmations.append('macd_bullish')
            elif signal_type in ['SELL', 'BEAR'] and macd_histogram < 0:
                confirmations.append('macd_bearish')
            else:
                contradictions.append('macd_contradiction')
            
            # Check consensus requirements
            min_confirmations = self.validation_config['min_confirming_indicators']
            max_contradictions = self.validation_config['max_contradictions_allowed']
            
            passed = (len(confirmations) >= min_confirmations and 
                     len(contradictions) <= max_contradictions)
            
            return {
                'passed': passed,
                'confirmations': confirmations,
                'contradictions': contradictions,
                'min_required': min_confirmations,
                'max_allowed_contradictions': max_contradictions
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _validate_circuit_breaker_filter(self, signal: Dict, latest_data: pd.Series, epic: str) -> Dict:
        """Emergency circuit breaker for multiple critical contradictions"""
        try:
            signal_type = signal.get('signal_type', '').upper()
            current_price = latest_data['close']
            contradictions = 0
            
            # Count critical contradictions
            ema_200 = latest_data.get('ema_200', 0)
            if ema_200:
                price_above_ema200 = current_price > ema_200
                margin = abs(current_price - ema_200) / ema_200
                if signal_type in ['BEAR', 'SELL'] and price_above_ema200 and margin > 0.002:
                    contradictions += 1
            
            macd_histogram = latest_data.get('macd_histogram', 0)
            if signal_type in ['BEAR', 'SELL'] and macd_histogram > 0.0001:
                contradictions += 1
            elif signal_type in ['BULL', 'BUY'] and macd_histogram < -0.0001:
                contradictions += 1
            
            max_allowed = self.validation_config['max_contradictions_allowed']
            
            if contradictions > max_allowed:
                return {
                    'passed': False,
                    'reason': 'emergency_circuit_breaker_triggered',
                    'critical_contradictions': contradictions,
                    'max_allowed': max_allowed
                }
            
            return {
                'passed': True,
                'reason': 'circuit_breaker_ok',
                'contradictions_count': contradictions
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def get_validation_stats(self) -> Dict:
        """Get validation statistics and performance metrics"""
        try:
            total = self.validation_stats['total_validations']
            passed = self.validation_stats['passed_validations']
            
            return {
                'total_validations': total,
                'passed_validations': passed,
                'failed_validations': self.validation_stats['failed_validations'],
                'safety_filter_rejections': self.validation_stats['safety_filter_rejections'],
                'mean_reversion_bypasses': self.validation_stats['mean_reversion_bypasses'],
                'pass_rate': (passed / total * 100) if total > 0 else 0,
                'validation_config': self.validation_config,
                'safety_filters_enabled': self.safety_filters,
                'mean_reversion_patterns_count': len(self.mean_reversion_patterns),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Validation stats collection failed: {e}")
            return {'error': str(e)}

    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics information"""
        return {
            'module_name': 'CombinedValidator',
            'initialization_successful': True,
            'validation_config_loaded': bool(self.validation_config),
            'safety_filters_loaded': bool(self.safety_filters),
            'mean_reversion_patterns_loaded': len(self.mean_reversion_patterns),
            'forex_optimizer_available': self.forex_optimizer is not None,
            'validation_stats': self.get_validation_stats(),
            'timestamp': datetime.now().isoformat()
        }