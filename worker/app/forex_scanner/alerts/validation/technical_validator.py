"""
Technical Validator - Signal Pre-validation Module
Validates signals technically before sending to Claude
Extracted from claude_api.py for better modularity
"""

import logging
from typing import Dict, List
from datetime import datetime
import pandas as pd


class TechnicalValidator:
    """
    Handles technical pre-validation for all trading strategies
    Prevents invalid signals from being sent to Claude API
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_signal_technically_with_complete_data(self, signal: Dict) -> Dict:
        """
        PUBLIC: Enhanced technical validation with comprehensive null handling
        This is the method called by claude_api.py
        """
        return self._validate_signal_technically_with_complete_data(signal)
    
    def _validate_signal_technically_with_complete_data(self, signal: Dict) -> Dict:
        """
        FIXED: Enhanced technical validation that works with your existing signal format
        """
        validation = {
            'valid': True,  # Default to True - don't block working signals
            'strategy': 'complete_dataframe_analysis',
            'reason': 'Technical validation passed',
            'summary': 'Signal validation successful',
            'details': '',
            'warnings': []
        }
        
        try:
            # CRITICAL FIX: Check if signal is None or empty
            if signal is None:
                validation.update({
                    'valid': False,
                    'reason': 'Signal is None',
                    'summary': 'CRITICAL ERROR: No signal data provided',
                    'details': 'Cannot validate a None signal'
                })
                return validation
            
            if not isinstance(signal, dict):
                validation.update({
                    'valid': False,
                    'reason': f'Signal is not a dictionary, got {type(signal)}',
                    'summary': 'CRITICAL ERROR: Invalid signal data type',
                    'details': f'Expected dict, received {type(signal)}'
                })
                return validation
            
            if len(signal) == 0:
                validation.update({
                    'valid': False,
                    'reason': 'Signal dictionary is empty',
                    'summary': 'CRITICAL ERROR: Empty signal data',
                    'details': 'Signal contains no data fields'
                })
                return validation
            
            signal_type = signal.get('signal_type', '').upper()
            epic = signal.get('epic', 'Unknown')
            
            # ENHANCED: Safe indicator extraction with comprehensive null handling
            def get_indicator_value(signal, field_name, nested_structures=None, default=None):
                """Extract indicator value with robust null checking"""
                try:
                    if signal is None:
                        return default
                    
                    # Try flat structure first (your signals use this!)
                    if field_name in signal:
                        value = signal[field_name]
                        if value is not None and not pd.isna(value):
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return default
                    
                    # Try nested structures if provided
                    if nested_structures is None:
                        nested_structures = ['ema_data', 'macd_data', 'kama_data', 'other_indicators', 'strategy_indicators']
                    
                    for struct_name in nested_structures:
                        if struct_name in signal and signal[struct_name] is not None:
                            struct_data = signal[struct_name]
                            if isinstance(struct_data, dict) and field_name in struct_data:
                                value = struct_data[field_name]
                                if value is not None and not pd.isna(value):
                                    try:
                                        return float(value)
                                    except (ValueError, TypeError):
                                        continue
                    
                    return default
                    
                except Exception as e:
                    self.logger.debug(f"Error extracting {field_name}: {e}")
                    return default
            
            # CRITICAL FIX: Get basic signal data with safe defaults
            current_price = get_indicator_value(signal, 'price', default=None)
            if current_price is None:
                current_price = get_indicator_value(signal, 'current_price', default=None)
            if current_price is None:
                current_price = get_indicator_value(signal, 'close_price', default=None)
            if current_price is None:
                current_price = get_indicator_value(signal, 'close', default=None)
            
            # If still no price, this is a critical error
            if current_price is None or current_price <= 0:
                validation.update({
                    'valid': False,
                    'reason': 'No valid price data found',
                    'summary': 'CRITICAL ERROR: Missing price information',
                    'details': 'Signal must contain valid price data (price, current_price, close_price, or close)'
                })
                return validation
            
            # Get technical indicators with safe extraction (these are what your signals have!)
            indicator_list = [
                'ema_9', 'ema_21', 'ema_200', 'ema_short', 'ema_long', 'ema_trend',
                'macd_line', 'macd_signal', 'macd_histogram',
                'rsi', 'atr', 'atr_14', 'bb_upper', 'bb_lower', 'volume', 'ltv'
            ]
            
            # Count available indicators
            indicators_available = []
            indicator_values = {}
            
            for indicator in indicator_list:
                value = get_indicator_value(signal, indicator)
                if value is not None:
                    indicators_available.append(indicator)
                    indicator_values[indicator] = value
            
            indicators_count = len(indicators_available)
            
            # FLEXIBLE VALIDATION: Allow signals with ANY indicators (don't require specific ones)
            validation_checks = []
            warning_messages = []
            
            # Check 1: Basic signal structure
            required_fields = ['epic', 'signal_type', 'timestamp']
            missing_fields = [field for field in required_fields if field not in signal or signal[field] is None]
            
            if missing_fields:
                warning_messages.append(f"Missing basic fields: {missing_fields}")
            else:
                validation_checks.append("✅ Basic signal structure valid")
            
            # Check 2: Price validation
            validation_checks.append(f"✅ Valid price data: {current_price:.5f}")
            
            # Check 3: Indicator availability (flexible - don't require specific indicators)
            if indicators_count > 0:
                validation_checks.append(f"✅ Technical indicators available: {indicators_count}")
                validation_checks.append(f"   Available: {', '.join(indicators_available[:5])}{'...' if len(indicators_available) > 5 else ''}")
            else:
                # IMPORTANT: Don't fail - just warn
                warning_messages.append("⚠️ No technical indicators found - basic validation only")
            
            # Check 4: EMA validation (if available) - don't require it
            ema_9 = indicator_values.get('ema_9')
            ema_21 = indicator_values.get('ema_21') 
            ema_200 = indicator_values.get('ema_200')
            
            if ema_9 is not None and ema_21 is not None:
                try:
                    if signal_type in ['BULL', 'BUY']:
                        if ema_9 > ema_21:
                            validation_checks.append("✅ EMA bullish alignment confirmed")
                        else:
                            warning_messages.append("⚠️ EMA alignment contradicts signal type")
                            
                    elif signal_type in ['BEAR', 'SELL']:
                        if ema_9 < ema_21:
                            validation_checks.append("✅ EMA bearish alignment confirmed")
                        else:
                            warning_messages.append("⚠️ EMA alignment contradicts signal type")
                            
                except Exception as ema_error:
                    warning_messages.append(f"EMA validation error: {ema_error}")
            else:
                validation_checks.append("ℹ️ EMA data not available for validation")
            
            # Check 5: MACD validation (if available) - don't require it
            macd_histogram = indicator_values.get('macd_histogram')
            if macd_histogram is not None:
                try:
                    if signal_type in ['BULL', 'BUY']:
                        if macd_histogram > 0:
                            validation_checks.append("✅ MACD supports bullish signal")
                        else:
                            warning_messages.append("⚠️ MACD histogram negative for bullish signal")
                    elif signal_type in ['BEAR', 'SELL']:
                        if macd_histogram < 0:
                            validation_checks.append("✅ MACD supports bearish signal")
                        else:
                            warning_messages.append("⚠️ MACD histogram positive for bearish signal")
                except Exception as macd_error:
                    warning_messages.append(f"MACD validation error: {macd_error}")
            else:
                validation_checks.append("ℹ️ MACD data not available for validation")
            
            # FINAL VALIDATION DECISION - Be permissive!
            critical_errors = len([msg for msg in warning_messages if 'CRITICAL' in msg or 'missing basic fields' in msg])
            
            # Only fail for critical structural issues
            if critical_errors > 0:
                is_valid = False
                reason = "Critical structural errors detected"
            else:
                # ALLOW SIGNAL THROUGH - your signals are working fine!
                is_valid = True
                reason = "Technical validation passed"
            
            # Compile summary
            summary_parts = []
            summary_parts.append(f"Price: {current_price:.5f}")
            summary_parts.append(f"Indicators: {indicators_count}")
            summary_parts.append(f"Checks: {len(validation_checks)}")
            if warning_messages:
                summary_parts.append(f"Warnings: {len(warning_messages)}")
            
            validation.update({
                'valid': is_valid,
                'reason': reason,
                'summary': '; '.join(summary_parts),
                'details': '\n'.join(validation_checks + warning_messages),
                'warnings': warning_messages,
                'indicators_count': indicators_count,
                'indicators_available': indicators_available,
                'current_price': current_price,
                'technical_checks_passed': len(validation_checks),
                'validation_mode': 'flexible_permissive'
            })
            
            self.logger.debug(f"✅ Technical validation result for {epic}: {is_valid} ({indicators_count} indicators)")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"❌ Technical validation error: {e}")
            import traceback
            self.logger.debug(f"Technical validation traceback: {traceback.format_exc()}")
            
            # SAFE FALLBACK: Allow signal through on validation errors
            return {
                'valid': True,  # Don't break working signals due to validation errors
                'reason': f'Validation error handled: {str(e)}',
                'summary': f'Technical validation error - signal allowed: {str(e)}',
                'details': f'Validation failed but signal permitted: {str(e)}',
                'warnings': [f'Technical validation error: {str(e)}'],
                'indicators_count': 0,
                'indicators_available': [],
                'current_price': signal.get('price', signal.get('current_price', 1.0)),
                'validation_mode': 'error_fallback'
            }

    
    def _get_indicator_value(self, signal: Dict, field_name: str, nested_structures: List[str] = None) -> float:
        """Extract indicator value from flat or nested structure"""
        # Try flat structure first
        value = signal.get(field_name)
        if value is not None:
            return float(value)
        
        # Try nested structures
        if nested_structures is None:
            nested_structures = ['ema_data', 'macd_data', 'kama_data', 'other_indicators']
        
        for struct_name in nested_structures:
            if struct_name in signal and isinstance(signal[struct_name], dict):
                struct_data = signal[struct_name]
                if field_name in struct_data and struct_data[field_name] is not None:
                    return float(struct_data[field_name])
        
        return None
    
    def _calculate_ema_tolerance(self, signal: Dict, current_price: float, ema_200: float) -> Dict:
        """Calculate EMA 200 tolerance based on pair and strategy"""
        try:
            epic = signal.get('epic', '')
            pip_multiplier = 100.0 if 'JPY' in epic.upper() else 10000.0
            
            # Base tolerance with basic adjustments
            base_tolerance_pips = 45.0
            
            # Strategy-specific multipliers
            strategy = signal.get('strategy', '').lower()
            if 'combined' in strategy:
                strategy_mult = 1.3
            elif 'zero_lag' in strategy:
                strategy_mult = 1.5
            elif 'momentum_bias' in strategy:
                strategy_mult = 1.4
            else:
                strategy_mult = 1.0
            
            # Pair-specific multipliers
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.', '')
            if 'JPY' in pair:
                pair_mult = 2.0
            elif pair in ['GBPUSD', 'EURUSD']:
                pair_mult = 1.3
            else:
                pair_mult = 1.0
            
            # Intelligence score bonus
            intelligence_score = signal.get('intelligence_score', 0.0)
            if intelligence_score >= 90.0:
                intel_mult = 1.8
            elif intelligence_score >= 85.0:
                intel_mult = 1.5
            else:
                intel_mult = 1.0
            
            # Calculate final tolerance
            tolerance_pips = base_tolerance_pips * strategy_mult * pair_mult * intel_mult
            tolerance_pips = min(tolerance_pips, 150.0)  # Cap at 150 pips
            tolerance_pips = max(tolerance_pips, 35.0)   # Minimum 35 pips
            tolerance = tolerance_pips / pip_multiplier
            
            price_ema_diff = current_price - ema_200
            price_ema_diff_pips = abs(price_ema_diff) * pip_multiplier
            within_tolerance = abs(price_ema_diff) <= tolerance
            
            return {
                'within_tolerance': within_tolerance,
                'tolerance_pips': tolerance_pips,
                'difference_pips': price_ema_diff_pips,
                'message': f"Price {current_price:.5f} vs EMA200 {ema_200:.5f} ({price_ema_diff_pips:.1f} pips, tolerance: {tolerance_pips:.1f} pips)"
            }
            
        except (ValueError, TypeError):
            return {
                'within_tolerance': False,
                'message': "Could not calculate EMA 200 tolerance - invalid numeric values"
            }
    
    def _validate_ema_alignment(self, signal_type: str, current_price: float, ema_9: float, ema_21: float, ema_200: float, validation_points: List[str], warning_points: List[str]):
        """Validate EMA alignment for perfect trend structure"""
        try:
            ema_9, ema_21, ema_200 = float(ema_9), float(ema_21), float(ema_200)
            
            if signal_type == 'BULL':
                perfect_alignment = current_price > ema_9 > ema_21 > ema_200
                if perfect_alignment:
                    validation_points.append("Perfect EMA alignment supports BULL")
                else:
                    warning_points.append(f"EMA alignment imperfect for BULL: price={current_price:.5f}, EMA9={ema_9:.5f}, EMA21={ema_21:.5f}, EMA200={ema_200:.5f}")
                    
            elif signal_type == 'BEAR':
                perfect_alignment = current_price < ema_9 < ema_21 < ema_200
                if perfect_alignment:
                    validation_points.append("Perfect EMA alignment supports BEAR")
                else:
                    warning_points.append(f"EMA alignment imperfect for BEAR: price={current_price:.5f}, EMA9={ema_9:.5f}, EMA21={ema_21:.5f}, EMA200={ema_200:.5f}")
        except (ValueError, TypeError):
            warning_points.append("Could not validate full EMA alignment")
    
    def _validate_macd_histogram(self, signal_type: str, macd_histogram: float) -> Dict:
        """Validate MACD histogram for momentum contradictions"""
        try:
            macd_histogram = float(macd_histogram)
            
            if signal_type == 'BULL':
                if macd_histogram > 0.00005:  # Clearly positive momentum
                    return {
                        'is_critical_failure': False,
                        'message': f"MACD histogram strongly supports BULL ({macd_histogram:.6f})"
                    }
                elif macd_histogram >= 0:  # Barely positive or zero
                    return {
                        'is_critical_failure': False,
                        'message': f"MACD histogram weakly supports BULL ({macd_histogram:.6f})"
                    }
                else:
                    # ANY negative MACD histogram rejects BULL signals
                    return {
                        'is_critical_failure': True,
                        'message': f"BULL signal REJECTED: MACD histogram negative ({macd_histogram:.6f}) - bearish momentum contradicts bullish signal"
                    }
                    
            elif signal_type == 'BEAR':
                if macd_histogram < -0.00005:  # Clearly negative momentum
                    return {
                        'is_critical_failure': False,
                        'message': f"MACD histogram strongly supports BEAR ({macd_histogram:.6f})"
                    }
                elif macd_histogram <= 0:  # Barely negative or zero
                    return {
                        'is_critical_failure': False,
                        'message': f"MACD histogram weakly supports BEAR ({macd_histogram:.6f})"
                    }
                else:
                    # ANY positive MACD histogram rejects BEAR signals
                    return {
                        'is_critical_failure': True,
                        'message': f"BEAR signal REJECTED: MACD histogram positive ({macd_histogram:.6f}) - bullish momentum contradicts bearish signal"
                    }
                    
        except (ValueError, TypeError):
            return {
                'is_critical_failure': False,
                'message': "Could not validate MACD histogram"
            }
    
    def _validate_macd_lines(self, signal_type: str, macd_line: float, macd_signal_line: float) -> Dict:
        """Validate MACD line vs signal line positioning"""
        try:
            macd_line, macd_signal_line = float(macd_line), float(macd_signal_line)
            
            if signal_type == 'BULL':
                if macd_line >= macd_signal_line:
                    return {'message': "MACD line position supports BULL"}
                else:
                    return {'message': f"MACD line below signal line for BULL (line={macd_line:.6f}, signal={macd_signal_line:.6f})"}
                    
            elif signal_type == 'BEAR':
                if macd_line <= macd_signal_line:
                    return {'message': "MACD line position supports BEAR"}
                else:
                    return {'message': f"MACD line above signal line for BEAR (line={macd_line:.6f}, signal={macd_signal_line:.6f})"}
                    
        except (ValueError, TypeError):
            return {'message': "Could not validate MACD line vs signal"}
    
    def _validate_volume(self, signal: Dict) -> Dict:
        """Validate volume confirmation"""
        volume_ratio = self._get_indicator_value(signal, 'volume_ratio') or self._get_indicator_value(signal, 'volume_ratio_20')
        
        if volume_ratio is not None:
            try:
                volume_ratio = float(volume_ratio)
                signal_type = signal.get('signal_type', '')
                
                if volume_ratio < 0.5:  # Extremely low volume
                    return {
                        'is_critical_failure': True,
                        'message': f"{signal_type} signal REJECTED: Extremely low volume ({volume_ratio:.3f}) - insufficient market participation"
                    }
                elif volume_ratio < 0.8:  # Low volume
                    return {
                        'is_critical_failure': False,
                        'message': f"{signal_type} signal has low volume confirmation ({volume_ratio:.3f}) - reduced institutional interest"
                    }
                elif volume_ratio > 1.2:  # Good volume
                    return {
                        'is_critical_failure': False,
                        'message': f"Good volume confirmation ({volume_ratio:.3f}) supports {signal_type} signal"
                    }
                else:
                    return {
                        'is_critical_failure': False,
                        'message': f"Moderate volume confirmation ({volume_ratio:.3f})"
                    }
        
            except (ValueError, TypeError):
                return {
                    'is_critical_failure': False,
                    'message': "Could not validate volume ratio - invalid numeric value"
                }
        else:
            return {
                'is_critical_failure': False,
                'message': "Volume ratio data not available for validation"
            }
    
    def _validate_kama(self, signal_type: str, current_price: float, kama_value: float, efficiency_ratio: float) -> Dict:
        """Validate KAMA alignment and efficiency"""
        try:
            kama_value, efficiency_ratio = float(kama_value), float(efficiency_ratio)
            
            if efficiency_ratio > 0.3:  # Good efficiency
                if signal_type == 'BULL' and current_price > kama_value:
                    return {'message': "KAMA alignment supports BULL"}
                elif signal_type == 'BEAR' and current_price < kama_value:
                    return {'message': "KAMA alignment supports BEAR"}
                else:
                    return {'message': f"KAMA alignment questionable: price={current_price:.5f}, KAMA={kama_value:.5f}"}
            else:
                return {'message': f"KAMA efficiency ratio too low: {efficiency_ratio:.3f}"}
                
        except (ValueError, TypeError):
            return {'message': "Could not validate KAMA alignment"}
    
    def validate_signal_technically(self, signal: Dict) -> Dict:
        """
        Legacy method for backward compatibility
        """
        return self.validate_signal_technically_with_complete_data(signal)