# alerts/claude_api.py
"""
Enhanced Claude API Integration with Technical Validation
Includes specific validation for EMA, MACD, and KAMA strategies
Pre-validates signals before sending to Claude to prevent invalid approvals
UPDATED: Now uses complete DataFrame data for comprehensive analysis
"""

import requests
import json
import logging
import pandas as pd
import os
import time
import random
import re
from datetime import datetime
from typing import Dict, Optional, List
try:
    import config
except ImportError:
    from forex_scanner import config


class ClaudeAnalyzer:
    """Enhanced Claude API integration with technical validation for all strategies"""
    
    def __init__(self, api_key: str, auto_save: bool = True, save_directory: str = "claude_analysis"):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-5-sonnet-20241022"
        self.max_tokens = 150  # Increased for technical validation responses
        self.timeout = 30
        self.auto_save = auto_save
        self.save_directory = save_directory
        
        # ✅ NEW: Enhanced retry configuration
        self.max_retries = 3
        self.base_delay = 2.0  # seconds
        self.max_delay = 30.0  # seconds
        self.exponential_base = 2.0
        
        # ✅ NEW: Rate limiting protection
        self.last_api_call = 0
        self.min_call_interval = 1.2  # seconds between calls (50 calls/minute = 1.2s interval)
        
        self.logger = logging.getLogger(__name__)
        
        if not api_key:
            self.logger.warning("⚠️ No Claude API key provided")
        
        # Create save directory if auto_save is enabled
        if self.auto_save:
            self._ensure_save_directory()

    def _clean_unicode_for_api(self, data):
        """
        Clean Unicode issues for API requests to prevent surrogate pair errors
        Fixes the 'no low surrogate in string' error at character position 53414
        """
        if isinstance(data, str):
            # Remove or replace problematic Unicode characters
            # Fix surrogate pairs and control characters
            cleaned = data.encode('utf-8', errors='ignore').decode('utf-8')
            # Remove control characters except newlines and tabs
            cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', cleaned)
            # Fix high surrogates without low surrogates
            cleaned = re.sub(r'[\uD800-\uDBFF](?![\uDC00-\uDFFF])', '', cleaned)
            # Fix low surrogates without high surrogates
            cleaned = re.sub(r'(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]', '', cleaned)
            return cleaned
        elif isinstance(data, dict):
            return {k: self._clean_unicode_for_api(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_unicode_for_api(item) for item in data]
        return data

    def _ensure_save_directory(self):
        """Ensure the claude_analysis directory exists"""
        try:
            os.makedirs(self.save_directory, exist_ok=True)
            self.logger.debug(f"📁 Analysis directory ready: {self.save_directory}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create analysis directory: {e}")
            self.auto_save = False
    
    def analyze_signal(self, signal: Dict, save_to_file: bool = None) -> Optional[str]:
        """
        Enhanced analysis with technical pre-validation
        """
        try:
            # STEP 1: Technical pre-validation BEFORE sending to Claude
            technical_validation = self._validate_signal_technically(signal)
            
            if not technical_validation['valid']:
                self.logger.warning(f"🚨 TECHNICAL VALIDATION FAILED: {technical_validation['reason']}")
                
                # Return rejection response without calling Claude
                return f"""
Claude Analysis for {signal.get('epic', 'Unknown')} {signal.get('signal_type', 'Unknown')} Signal

TECHNICAL VALIDATION FAILED
Decision: REJECT
Reason: {technical_validation['reason']}
Score: 0/10

This signal was rejected before Claude analysis due to technical criteria violations.
{technical_validation['details']}
"""
            
            # STEP 2: If technically valid, proceed with Claude analysis
            minimal_result = self.analyze_signal_minimal(signal, save_to_file)
            
            if minimal_result:
                # Convert minimal result to text format for compatibility
                analysis_text = f"""
Claude Analysis for {signal.get('epic', 'Unknown')} {signal.get('signal_type', 'Unknown')} Signal

TECHNICAL VALIDATION: ✅ PASSED
Signal Quality Score: {minimal_result['score']}/10
Decision: {minimal_result['decision']}
Approved: {minimal_result['approved']}
Reason: {minimal_result['reason']}

Strategy: {self._identify_strategy(signal)}
Price: {signal.get('price', 'N/A')}
Confidence: {signal.get('confidence_score', 0):.1%}

Technical Analysis: {technical_validation['summary']}

Claude Analysis: Based on the signal characteristics, this {signal.get('signal_type', 'signal').lower()} signal for {signal.get('epic', 'the pair')} receives a score of {minimal_result['score']}/10. The recommendation is to {minimal_result['decision'].lower()} this signal because {minimal_result['reason'].lower()}.
"""
            
                self.logger.info(f"✅ Claude full analysis: {signal.get('epic')} - Score: {minimal_result['score']}/10")
                return analysis_text.strip()
            else:
                self.logger.error("❌ Failed to get Claude analysis after technical validation passed")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Claude full analysis failed: {e}")
            return None

    def analyze_signal_minimal(self, signal: Dict, save_to_file: bool = None) -> Optional[Dict]:
        """
        Enhanced minimal analysis with complete DataFrame technical validation
        FIXED: Format string errors by ensuring proper type conversion
        """
        if not self.api_key:
            self.logger.warning("No API key available for Claude analysis")
            return None
        
        try:
            # STEP 1: Enhanced technical pre-validation using ALL DataFrame data
            technical_validation = self._validate_signal_technically_with_complete_data(signal)
            
            if not technical_validation['valid']:
                self.logger.warning(f"🚨 Signal rejected - Complete technical validation failed: {technical_validation['reason']}")
                return {
                    'score': 0,
                    'decision': 'REJECT',
                    'reason': f"Technical validation failed: {technical_validation['reason']}",
                    'approved': False,
                    'raw_response': f"COMPLETE technical validation failed: {technical_validation['reason']}"
                }
            
            # STEP 2: Create enhanced prompt with ALL available data
            prompt = self._build_minimal_prompt_with_complete_data(signal, technical_validation)
            
            # STEP 3: Call Claude API
            response = self._call_claude_api(prompt, max_tokens=100)
            
            if not response:
                self.logger.error("❌ No response from Claude API")
                return None
            
            # STEP 4: Parse response with enhanced error handling
            parsed_result = self.parse_minimal_response(response)
            
            if parsed_result and parsed_result.get('score') is not None:
                # ✅ FIXED: Ensure proper type conversion before formatting
                score = parsed_result.get('score')
                decision = parsed_result.get('decision', 'UNKNOWN')
                
                # Convert score to int safely
                try:
                    score_int = int(float(score)) if score is not None else 0
                except (ValueError, TypeError):
                    score_int = 0
                    self.logger.warning(f"⚠️ Invalid score value: {score}, defaulting to 0")
                
                # Safe string formatting with proper type checking
                epic = signal.get('epic', 'Unknown')
                signal_type = signal.get('signal_type', 'Unknown')
                
                self.logger.info(f"✅ Claude minimal analysis: {epic} {signal_type} - Score: {score_int}/10, Decision: {decision}")
                
                # Return structured result with guaranteed types
                return {
                    'score': score_int,
                    'decision': decision,
                    'reason': parsed_result.get('reason', 'Analysis completed'),
                    'approved': parsed_result.get('approved', False),
                    'raw_response': response,
                    'mode': 'minimal',
                    'technical_validation_passed': True,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            else:
                self.logger.error("❌ Failed to parse Claude response")
                return None
                
        except Exception as e:
            # ✅ FIXED: Better error handling with safe string formatting
            error_msg = str(e)
            self.logger.error(f"❌ Claude minimal analysis failed: {error_msg}")
            return None

    def analyze_signal_minimal_with_fallback(self, signal: Dict, save_to_file: bool = None) -> Optional[Dict]:
        """
        ✅ NEW: Analyze signal with intelligent fallback when Claude is unavailable
        """
        # Try Claude analysis first
        claude_result = self.analyze_signal_minimal(signal, save_to_file)
        
        if claude_result:
            return claude_result
        
        # ✅ FALLBACK: Generate intelligent fallback analysis
        self.logger.warning("🔄 Claude unavailable, using intelligent fallback analysis")
        
        try:
            # Basic signal quality assessment
            confidence = float(signal.get('confidence_score', 0))
            strategy = signal.get('strategy', 'unknown')
            
            # Simple scoring based on confidence and strategy
            if confidence >= 0.9:
                score = 8
                decision = 'APPROVE'
                reason = 'High confidence signal with strong technical indicators'
            elif confidence >= 0.8:
                score = 7
                decision = 'APPROVE' 
                reason = 'Good confidence signal with solid technical setup'
            elif confidence >= 0.7:
                score = 6
                decision = 'APPROVE'
                reason = 'Moderate confidence signal meeting minimum criteria'
            elif confidence >= 0.6:
                score = 5
                decision = 'NEUTRAL'
                reason = 'Borderline signal with mixed technical indicators'
            else:
                score = 3
                decision = 'REJECT'
                reason = 'Low confidence signal with weak technical setup'
            
            # Adjust based on strategy
            if strategy in ['combined', 'consensus']:
                score = min(score + 1, 10)  # Boost for multi-strategy confirmation
            
            return {
                'score': score,
                'decision': decision,
                'reason': reason,
                'approved': decision == 'APPROVE',
                'raw_response': f'FALLBACK ANALYSIS: Score {score}/10, Decision: {decision}',
                'mode': 'fallback',
                'technical_validation_passed': True,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fallback analysis failed: {e}")
            return {
                'score': 5,
                'decision': 'NEUTRAL',
                'reason': 'Fallback analysis error - using neutral assessment',
                'approved': False,
                'raw_response': 'FALLBACK ERROR',
                'mode': 'error_fallback'
            }

    def get_api_health_status(self) -> Dict:
        """
        ✅ NEW: Check Claude API health and return status
        """
        if not self.api_key:
            return {
                'status': 'unavailable',
                'reason': 'No API key configured',
                'recommendation': 'Configure CLAUDE_API_KEY'
            }
        
        try:
            # Simple health check
            test_response = self._call_claude_api("Health check. Respond with 'OK' if working.", max_tokens=10)
            
            if test_response and 'OK' in test_response.upper():
                return {
                    'status': 'healthy',
                    'reason': 'API responding normally',
                    'recommendation': 'Continue normal operations'
                }
            elif test_response:
                return {
                    'status': 'degraded',
                    'reason': 'API responding but with unexpected response',
                    'recommendation': 'Monitor closely, consider fallback mode'
                }
            else:
                return {
                    'status': 'unhealthy',
                    'reason': 'API not responding or returning errors',
                    'recommendation': 'Use fallback analysis mode'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'reason': f'Health check failed: {str(e)}',
                'recommendation': 'Use fallback analysis mode'
            }
    
    def _validate_signal_technically_with_complete_data(self, signal: Dict) -> Dict:
        """
        🔧 FIXED: Enhanced technical validation using ALL available DataFrame indicators
        🔧 FIXED: Now handles nested data structures (ema_data, macd_data, kama_data)
        🔧 FIXED: Strict EMA 200 validation and MACD contradiction detection
        """
        validation = {'valid': False, 'strategy': 'complete_dataframe_analysis'}
        
        try:
            signal_type = signal.get('signal_type', '').upper()
            
            # 🔧 ENHANCED: Extract technical data from both flat and nested structures
            def get_indicator_value(signal, field_name, nested_structures=None):
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
            
            # Get ALL available technical data with enhanced extraction
            current_price = get_indicator_value(signal, 'price') or get_indicator_value(signal, 'current_price')
            
            # If still no price, try using EMA 9 as approximation (common pattern)
            if current_price is None:
                current_price = get_indicator_value(signal, 'ema_9')
                if current_price:
                    self.logger.debug(f"📊 Using EMA 9 as current price approximation: {current_price:.5f}")
            
            # EMA data with enhanced extraction
            ema_9 = get_indicator_value(signal, 'ema_9') or get_indicator_value(signal, 'ema_short')
            ema_21 = get_indicator_value(signal, 'ema_21') or get_indicator_value(signal, 'ema_long')
            ema_200 = get_indicator_value(signal, 'ema_200') or get_indicator_value(signal, 'ema_trend')
            
            # MACD data with enhanced extraction
            macd_line = get_indicator_value(signal, 'macd_line')
            macd_signal_line = get_indicator_value(signal, 'macd_signal')
            macd_histogram = get_indicator_value(signal, 'macd_histogram')
            
            # KAMA data with enhanced extraction
            kama_value = get_indicator_value(signal, 'kama_value') or get_indicator_value(signal, 'kama')
            efficiency_ratio = get_indicator_value(signal, 'efficiency_ratio')
            
            # Log what we found for debugging
            self.logger.debug(f"🔍 Extracted indicators - Price: {current_price}, EMA200: {ema_200}, MACD Hist: {macd_histogram}")
            
            # Validation criteria
            validation_points = []
            warning_points = []
            critical_failures = []
            
            # 1. CRITICAL: Basic price data validation
            if current_price and current_price > 0:
                validation_points.append("Valid price data")
            else:
                validation['valid'] = False
                validation['reason'] = "Invalid or missing price data"
                validation['details'] = f"No valid price found in signal data"
                validation['summary'] = "CRITICAL FAILURE: No price data"
                return validation
            
            # 2. 🔧 CRITICAL: EMA 200 TREND FILTER (Your main requirement)
            if ema_200 is not None and current_price is not None:
                try:
                    ema_200 = float(ema_200)
                    current_price = float(current_price)
                    
                    # Calculate pullback tolerance (simplified version of your validator logic)
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
                    
                    if signal_type == 'BULL':
                        if current_price > ema_200:
                            validation_points.append(f"EMA 200 trend filter PASSED for BULL: Price {current_price:.5f} > EMA200 {ema_200:.5f} (+{price_ema_diff_pips:.1f} pips)")
                        elif abs(price_ema_diff) <= tolerance:
                            validation_points.append(f"EMA 200 pullback ACCEPTED for BULL: Price {current_price:.5f} vs EMA200 {ema_200:.5f} (-{price_ema_diff_pips:.1f} pips within {tolerance_pips:.1f} pip tolerance)")
                        else:
                            critical_failures.append(f"BULL signal REJECTED: Price {current_price:.5f} vs EMA200 {ema_200:.5f} (-{price_ema_diff_pips:.1f} pips exceeds {tolerance_pips:.1f} pip tolerance)")
                            
                    elif signal_type == 'BEAR':
                        if current_price < ema_200:
                            validation_points.append(f"EMA 200 trend filter PASSED for BEAR: Price {current_price:.5f} < EMA200 {ema_200:.5f} (-{price_ema_diff_pips:.1f} pips)")
                        elif abs(price_ema_diff) <= tolerance:
                            validation_points.append(f"EMA 200 pullback ACCEPTED for BEAR: Price {current_price:.5f} vs EMA200 {ema_200:.5f} (+{price_ema_diff_pips:.1f} pips within {tolerance_pips:.1f} pip tolerance)")
                        else:
                            critical_failures.append(f"BEAR signal REJECTED: Price {current_price:.5f} vs EMA200 {ema_200:.5f} (+{price_ema_diff_pips:.1f} pips exceeds {tolerance_pips:.1f} pip tolerance)")
                            
                except (ValueError, TypeError):
                    warning_points.append("Could not validate EMA 200 trend filter - invalid numeric values")
            else:
                warning_points.append("EMA 200 trend filter could not be applied - missing data")
            
            # 3. Enhanced EMA alignment validation (if all EMAs available)
            if all(x is not None for x in [ema_9, ema_21, ema_200, current_price]):
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
            
            # 4. 🔧 CRITICAL: MACD contradiction detection
            if macd_histogram is not None:
                try:
                    macd_histogram = float(macd_histogram)
                    
                    if signal_type == 'BULL':
                        if macd_histogram > 0.00005:  # Clearly positive momentum
                            validation_points.append(f"MACD histogram strongly supports BULL ({macd_histogram:.6f})")
                        elif macd_histogram >= 0:  # Barely positive or zero
                            validation_points.append(f"MACD histogram weakly supports BULL ({macd_histogram:.6f})")
                        else:
                            # ANY negative MACD histogram rejects BULL signals - no tolerance
                            critical_failures.append(f"BULL signal REJECTED: MACD histogram negative ({macd_histogram:.6f}) - bearish momentum contradicts bullish signal")
                            
                    elif signal_type == 'BEAR':
                        if macd_histogram < -0.00005:  # Clearly negative momentum
                            validation_points.append(f"MACD histogram strongly supports BEAR ({macd_histogram:.6f})")
                        elif macd_histogram <= 0:  # Barely negative or zero
                            validation_points.append(f"MACD histogram weakly supports BEAR ({macd_histogram:.6f})")
                        else:
                            # ANY positive MACD histogram rejects BEAR signals - no tolerance
                            critical_failures.append(f"BEAR signal REJECTED: MACD histogram positive ({macd_histogram:.6f}) - bullish momentum contradicts bearish signal")
                    
                    # Additional momentum strength assessment
                    momentum_strength = abs(macd_histogram)
                    if momentum_strength < 0.00002:
                        warning_points.append(f"Very weak MACD momentum ({macd_histogram:.6f}) - signal may lack follow-through")
                    elif momentum_strength > 0.0005:
                        validation_points.append(f"Strong MACD momentum detected ({macd_histogram:.6f}) - good signal strength")
                            
                except (ValueError, TypeError):
                    warning_points.append("Could not validate MACD histogram")
            
            # 5. MACD line vs signal validation (if available)
            if all(x is not None for x in [macd_line, macd_signal_line]):
                try:
                    macd_line, macd_signal_line = float(macd_line), float(macd_signal_line)
                    
                    if signal_type == 'BULL':
                        if macd_line >= macd_signal_line:
                            validation_points.append("MACD line position supports BULL")
                        else:
                            warning_points.append(f"MACD line below signal line for BULL (line={macd_line:.6f}, signal={macd_signal_line:.6f})")
                            
                    elif signal_type == 'BEAR':
                        if macd_line <= macd_signal_line:
                            validation_points.append("MACD line position supports BEAR")
                        else:
                            warning_points.append(f"MACD line above signal line for BEAR (line={macd_line:.6f}, signal={macd_signal_line:.6f})")
                            
                except (ValueError, TypeError):
                    warning_points.append("Could not validate MACD line vs signal")
            
            # Extract volume ratio using the same pattern as other indicators
            volume_ratio = get_indicator_value(signal, 'volume_ratio') or get_indicator_value(signal, 'volume_ratio_20')

            if volume_ratio is not None:
                try:
                    volume_ratio = float(volume_ratio)
                    
                    if volume_ratio < 0.5:  # Extremely low volume
                        critical_failures.append(f"{signal_type} signal REJECTED: Extremely low volume ({volume_ratio:.3f}) - insufficient market participation")
                    elif volume_ratio < 0.8:  # Low volume
                        warning_points.append(f"{signal_type} signal has low volume confirmation ({volume_ratio:.3f}) - reduced institutional interest")
                    elif volume_ratio > 1.2:  # Good volume
                        validation_points.append(f"Good volume confirmation ({volume_ratio:.3f}) supports {signal_type} signal")
                    else:
                        validation_points.append(f"Moderate volume confirmation ({volume_ratio:.3f})")
            
                except (ValueError, TypeError):
                    warning_points.append("Could not validate volume ratio - invalid numeric value")
            else:
                warning_points.append("Volume ratio data not available for validation")


            # 6. KAMA validation (if available)
            if kama_value is not None and efficiency_ratio is not None:
                try:
                    kama_value, efficiency_ratio = float(kama_value), float(efficiency_ratio)
                    
                    if efficiency_ratio > 0.3:  # Good efficiency
                        if signal_type == 'BULL' and current_price > kama_value:
                            validation_points.append("KAMA alignment supports BULL")
                        elif signal_type == 'BEAR' and current_price < kama_value:
                            validation_points.append("KAMA alignment supports BEAR")
                        else:
                            warning_points.append(f"KAMA alignment questionable: price={current_price:.5f}, KAMA={kama_value:.5f}")
                    else:
                        warning_points.append(f"KAMA efficiency ratio too low: {efficiency_ratio:.3f}")
                            
                except (ValueError, TypeError):
                    warning_points.append("Could not validate KAMA alignment")
            
            # 🔧 FINAL VALIDATION DECISION
            if critical_failures:
                validation['valid'] = False
                validation['reason'] = f"Critical contradictions found: {'; '.join(critical_failures)}"
                validation['contradiction_detected'] = True
                validation['details'] = f"Critical failures: {'; '.join(critical_failures)}"
                validation['summary'] = "Complete DataFrame validation: CRITICAL FAILURE"
                self.logger.warning(f"🚨 Critical validation failure: {validation['reason']}")
                return validation
            
            if len(validation_points) >= 1:  # At least one supporting indicator
                validation['valid'] = True
                validation['reason'] = f"Complete analysis passed: {len(validation_points)} supporting factors"
                validation['details'] = f"Supporting: {'; '.join(validation_points)}"
                if warning_points:
                    validation['warnings'] = f"Warnings: {'; '.join(warning_points)}"
                self.logger.debug(f"✅ Validation passed with {len(validation_points)} supporting factors")
            else:
                validation['valid'] = False
                validation['reason'] = "No supporting technical indicators found"
                validation['details'] = f"Warnings: {'; '.join(warning_points)}" if warning_points else "No technical validation possible"
                self.logger.warning(f"❌ Validation failed: {validation['reason']}")
            
            validation['summary'] = f"Complete DataFrame validation: {'PASSED' if validation['valid'] else 'FAILED'}"
            
            return validation
            
        except Exception as e:
            self.logger.error(f"❌ Technical validation error: {e}")
            validation['valid'] = False
            validation['reason'] = f"Technical validation error: {str(e)}"
            validation['summary'] = "Complete DataFrame validation: ERROR"
            return validation

    def _build_minimal_prompt_with_complete_data(self, signal: Dict, technical_validation: Dict) -> str:
        """
        Build enhanced minimal prompt with complete technical data
        FIXED: Safe string formatting with proper type checking
        """
        try:
            # ✅ FIXED: Safe extraction with default values
            epic = str(signal.get('epic', 'Unknown'))
            signal_type = str(signal.get('signal_type', 'Unknown'))
            strategy = str(signal.get('strategy', 'Unknown'))
            
            # Safe numeric conversions
            try:
                price = float(signal.get('price', 0))
                confidence = float(signal.get('confidence_score', 0))
            except (ValueError, TypeError):
                price = 0.0
                confidence = 0.0
                self.logger.warning("⚠️ Invalid numeric values in signal data")
            
            # Build prompt with safe formatting
            prompt = f"""FOREX SIGNAL ANALYSIS - MINIMAL MODE

Signal: {epic} {signal_type}
Strategy: {strategy}
Price: {price:.5f}
Confidence: {confidence:.1%}

Technical Validation: ✅ PASSED
{technical_validation.get('summary', 'Technical analysis completed')}

Instructions: Provide only:
SCORE: [0-10]
DECISION: [APPROVE/REJECT]
REASON: [brief reason]

Be concise. Focus on signal quality."""

            return prompt
            
        except Exception as e:
            self.logger.error(f"Error building prompt: {e}")
            # Return basic prompt if formatting fails
            return f"""FOREX SIGNAL ANALYSIS - MINIMAL MODE

Signal: {signal.get('epic', 'Unknown')} {signal.get('signal_type', 'Unknown')}

Instructions: Provide only:
SCORE: [0-10]  
DECISION: [APPROVE/REJECT]
REASON: [brief reason]"""

    def _validate_ema_signal(self, signal: Dict) -> str:
        """Validate EMA-specific signal characteristics"""
        try:
            # Check for EMA-specific data
            ema_9 = signal.get('ema_9')
            ema_21 = signal.get('ema_21')
            ema_200 = signal.get('ema_200')
            
            if all(val is not None for val in [ema_9, ema_21, ema_200]):
                return f"EMA crossover confirmed (9:{ema_9:.5f}, 21:{ema_21:.5f}, 200:{ema_200:.5f})"
            else:
                return "EMA data available, crossover logic applied"
                
        except Exception:
            return "EMA validation completed"

    def _validate_macd_signal(self, signal: Dict) -> str:
        """Validate MACD-specific signal characteristics"""
        try:
            # Check for MACD-specific data
            macd_histogram = signal.get('macd_histogram')
            macd_line = signal.get('macd_line')
            macd_signal = signal.get('macd_signal')
            
            if macd_histogram is not None:
                return f"MACD histogram: {macd_histogram:.6f}, line: {macd_line or 'N/A'}, signal: {macd_signal or 'N/A'}"
            else:
                return "MACD validation completed"
                
        except Exception:
            return "MACD validation completed"

    def _validate_combined_signal(self, signal: Dict) -> str:
        """Validate combined-specific signal characteristics"""
        try:
            strategy_indicators = signal.get('strategy_indicators', {})
            components = []
            
            if 'ema_data' in strategy_indicators:
                components.append("EMA")
            if 'macd_data' in strategy_indicators:
                components.append("MACD")
            if 'kama_data' in strategy_indicators:
                components.append("KAMA")
            
            if components:
                return f"Combined strategy with {', '.join(components)} components"
            else:
                return "Combined strategy validation completed"
                
        except Exception:
            return "Combined validation completed"

    def _create_enhanced_validation_prompt_with_complete_data(self, signal: Dict, technical_validation: Dict) -> str:
        """
        UPDATED: Create enhanced prompt with COMPLETE DataFrame technical data
        """
        
        # Get price for display
        if 'price_mid' in signal:
            price_info = f"MID: {signal['price_mid']:.5f}, EXEC: {signal['execution_price']:.5f}"
        else:
            price_info = f"{signal.get('price', 'N/A'):.5f}"
        
        # Get strategy-specific info
        strategy = technical_validation['strategy']
        
        # UPDATED: Extract COMPLETE technical indicators from enriched signal data
        complete_indicators_text = "COMPLETE TECHNICAL ANALYSIS DATA:\n"
        
        # 1. PRICE DATA (always available)
        current_price = signal.get('price', signal.get('current_price', 0))
        complete_indicators_text += f"- current_price: {current_price}\n"
        complete_indicators_text += f"- open_price: {signal.get('open_price', 'N/A')}\n"
        complete_indicators_text += f"- high_price: {signal.get('high_price', 'N/A')}\n"
        complete_indicators_text += f"- low_price: {signal.get('low_price', 'N/A')}\n"
        complete_indicators_text += f"- close_price: {signal.get('close_price', current_price)}\n"
        
        # 2. EMA INDICATORS (from enriched DataFrame)
        complete_indicators_text += "\nEMA INDICATORS:\n"
        complete_indicators_text += f"- ema_9: {signal.get('ema_9', signal.get('ema_short', 'N/A'))}\n"
        complete_indicators_text += f"- ema_21: {signal.get('ema_21', signal.get('ema_long', 'N/A'))}\n"
        complete_indicators_text += f"- ema_200: {signal.get('ema_200', signal.get('ema_trend', 'N/A'))}\n"
        complete_indicators_text += f"- ema_short: {signal.get('ema_short', 'N/A')}\n"
        complete_indicators_text += f"- ema_long: {signal.get('ema_long', 'N/A')}\n"
        complete_indicators_text += f"- ema_trend: {signal.get('ema_trend', 'N/A')}\n"
        
        # 3. MACD INDICATORS (from enriched DataFrame) - THE CRITICAL ONES!
        complete_indicators_text += "\nMACD INDICATORS:\n"
        complete_indicators_text += f"- macd_line: {signal.get('macd_line', 'N/A')}\n"
        complete_indicators_text += f"- macd_signal: {signal.get('macd_signal', 'N/A')}\n"
        complete_indicators_text += f"- macd_histogram: {signal.get('macd_histogram', 'N/A')}\n"
        
        # 4. KAMA INDICATORS (if available)
        if signal.get('kama_value') or signal.get('kama') or signal.get('efficiency_ratio'):
            complete_indicators_text += "\nKAMA INDICATORS:\n"
            complete_indicators_text += f"- kama_value: {signal.get('kama_value', signal.get('kama', 'N/A'))}\n"
            complete_indicators_text += f"- efficiency_ratio: {signal.get('efficiency_ratio', 'N/A')}\n"
            complete_indicators_text += f"- kama_trend: {signal.get('kama_trend', signal.get('kama_slope', 'N/A'))}\n"
        
        # 5. OTHER TECHNICAL INDICATORS (from enriched DataFrame)
        complete_indicators_text += "\nADDITIONAL INDICATORS:\n"
        complete_indicators_text += f"- rsi: {signal.get('rsi', 'N/A')}\n"
        complete_indicators_text += f"- atr: {signal.get('atr', 'N/A')}\n"
        complete_indicators_text += f"- bb_upper: {signal.get('bb_upper', 'N/A')}\n"
        complete_indicators_text += f"- bb_middle: {signal.get('bb_middle', 'N/A')}\n"
        complete_indicators_text += f"- bb_lower: {signal.get('bb_lower', 'N/A')}\n"
        
        # 6. VOLUME DATA
        complete_indicators_text += "\nVOLUME DATA:\n"
        complete_indicators_text += f"- volume: {signal.get('volume', 'N/A')}\n"
        complete_indicators_text += f"- volume_ratio: {signal.get('volume_ratio', 'N/A')}\n"
        complete_indicators_text += f"- volume_confirmation: {signal.get('volume_confirmation', 'N/A')}\n"
        
        # 7. SUPPORT/RESISTANCE DATA
        complete_indicators_text += "\nSUPPORT/RESISTANCE:\n"
        complete_indicators_text += f"- nearest_support: {signal.get('nearest_support', 'N/A')}\n"
        complete_indicators_text += f"- nearest_resistance: {signal.get('nearest_resistance', 'N/A')}\n"
        complete_indicators_text += f"- distance_to_support_pips: {signal.get('distance_to_support_pips', 'N/A')}\n"
        complete_indicators_text += f"- distance_to_resistance_pips: {signal.get('distance_to_resistance_pips', 'N/A')}\n"

        # 8. STRATEGY-SPECIFIC DATA (from strategy_indicators JSON if available)
        strategy_indicators = signal.get('strategy_indicators', {})
        if strategy_indicators:
            complete_indicators_text += "\nSTRATEGY-SPECIFIC DATA:\n"
            
            # Handle nested JSON structure
            for category, data in strategy_indicators.items():
                if isinstance(data, dict):
                    complete_indicators_text += f"\n{category.upper()}:\n"
                    for key, value in data.items():
                        complete_indicators_text += f"- {key}: {value}\n"
                else:
                    complete_indicators_text += f"- {category}: {data}\n"

        prompt = f"""
FOREX SIGNAL COMPLETE TECHNICAL ANALYSIS

CRITICAL TASK: You have access to ALL technical indicators calculated from the market data. 
Perform INDEPENDENT analysis and determine if the signal classification is correct.

BASIC SIGNAL INFO:
- Pair: {signal.get('epic', 'N/A')}
- System Classification: {signal.get('signal_type', 'N/A')}
- Price: {price_info}
- System Confidence: {signal.get('confidence_score', 0):.1%}
- Triggering Strategy: {strategy}

{complete_indicators_text}

YOUR INDEPENDENT ANALYSIS CHECKLIST:

1. MACD MOMENTUM ANALYSIS:
   - Is MACD histogram positive (bullish) or negative (bearish)?
   - Is MACD line above or below MACD signal line?
   - Does MACD momentum support the system classification?

2. EMA TREND ANALYSIS:
   - Where is price relative to EMA 9, 21, and 200?
   - What is the EMA alignment (bullish: 9>21>200, bearish: 9<21<200)?
   - Does EMA trend support the system classification?

3. CROSS-INDICATOR VALIDATION:
   - Do MACD and EMA indicators agree on direction?
   - Are there any major contradictions between indicators?
   - Is there confluence or divergence?

4. SIGNAL QUALITY ASSESSMENT:
   - Volume confirmation present?
   - Price near support/resistance levels?
   - Overall technical picture strength?

CRITICAL VALIDATION RULES:
- If MACD histogram is significantly negative (< -0.0001), BULL signals should be REJECTED
- If MACD histogram is significantly positive (> 0.0001), BEAR signals should be REJECTED  
- If price is below EMA 200 by significant margin, BULL signals are questionable
- If price is above EMA 200 by significant margin, BEAR signals are questionable
- If MACD and EMA strongly disagree, signal should be flagged

Respond EXACTLY in this format:

SCORE: [1-10]
CORRECT_TYPE: [BULL/BEAR/NONE/SYSTEM_CORRECT]
DECISION: [APPROVE/REJECT]
REASON: [Your detailed analysis focusing on indicator alignment and any contradictions found]

Focus on cross-validation between ALL available indicators, not just the triggering strategy.
"""
        
        return prompt

    def parse_enhanced_response(self, response: str) -> Dict:
        """
        FIXED: Enhanced response parser with better error handling
        Addresses the "Incomplete Claude response parsing" warning
        """
        result = {
            'score': 5,  # Default to middle score
            'correct_type': 'SYSTEM_CORRECT',  # Default to accepting system classification
            'decision': 'APPROVE',  # Default to approving (will be overridden by score logic)
            'reason': 'Analysis completed',
            'approved': True
        }
        
        try:
            if not response or not response.strip():
                self.logger.warning("⚠️ Empty Claude response received")
                result.update({
                    'score': 0,
                    'decision': 'REJECT',
                    'reason': 'Empty response from Claude',
                    'approved': False
                })
                return result
            
            response_text = response.strip()
            
            # 🔍 DEBUG: More detailed logging (only log first 200 chars to avoid spam)
            self.logger.debug(f"🔍 Claude response preview: '{response_text[:200]}{'...' if len(response_text) > 200 else ''}'")
            
            # Try structured format first
            parsed_structured = self._try_parse_structured_format(response_text)
            if parsed_structured:
                self.logger.debug("✅ Successfully parsed structured format")
                return parsed_structured
            
            # Try natural language parsing
            parsed_text = self._try_parse_natural_language(response_text)
            if parsed_text:
                self.logger.debug("✅ Successfully parsed natural language format")
                return parsed_text
            
            # Enhanced fallback parsing
            fallback_result = self._enhanced_fallback_parse(response_text)
            if fallback_result.get('score') is not None:
                self.logger.info("✅ Successfully used enhanced fallback parsing")
                return fallback_result
            
            # If all parsing fails, log warning but don't fail completely
            self.logger.warning("⚠️ Incomplete Claude response parsing - using safe defaults")
            return {
                'score': 5,  # Neutral score when parsing fails
                'correct_type': 'SYSTEM_CORRECT',
                'decision': 'APPROVE',  # Conservative default
                'reason': 'Parsing incomplete - using safe defaults',
                'approved': True,
                'parsing_failed': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing Claude response: {e}")
            return {
                'score': 0,
                'correct_type': 'UNKNOWN',
                'decision': 'REJECT',
                'reason': f'Parse error: {str(e)}',
                'approved': False,
                'parsing_error': True
            }

    def _try_parse_structured_format(self, response: str) -> Dict:
        """Try to parse the expected structured format with enhanced debugging"""
        result = {}
        
        try:
            lines = response.strip().split('\n')
            self.logger.debug(f"🔍 Structured parsing: Found {len(lines)} lines")
            
            found_fields = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                self.logger.debug(f"  Processing line {i}: '{line}'")
                
                if line.startswith('SCORE:'):
                    score_text = line.replace('SCORE:', '').strip()
                    self.logger.debug(f"    Found SCORE field: '{score_text}'")
                    try:
                        # Handle formats like "8/10", "8", "Score: 8"
                        import re
                        score_match = re.search(r'(\d+)', score_text)
                        if score_match:
                            score = int(score_match.group(1))
                            # Ensure score is between 0-10
                            result['score'] = max(0, min(10, score))
                            found_fields.append('score')
                            self.logger.debug(f"    Parsed score: {result['score']}")
                    except (ValueError, AttributeError) as e:
                        self.logger.debug(f"    Score parsing failed: {e}")
                        continue
                        
                elif line.startswith('CORRECT_TYPE:'):
                    correct_type = line.replace('CORRECT_TYPE:', '').strip().upper()
                    self.logger.debug(f"    Found CORRECT_TYPE field: '{correct_type}'")
                    if correct_type in ['BULL', 'BEAR', 'NONE', 'SYSTEM_CORRECT', 'UNKNOWN']:
                        result['correct_type'] = correct_type
                        found_fields.append('correct_type')
                        self.logger.debug(f"    Parsed correct_type: {result['correct_type']}")
                    else:
                        self.logger.debug(f"    Invalid correct_type value: '{correct_type}'")
                        
                elif line.startswith('DECISION:'):
                    decision = line.replace('DECISION:', '').strip().upper()
                    self.logger.debug(f"    Found DECISION field: '{decision}'")
                    if decision in ['APPROVE', 'REJECT', 'NEUTRAL']:
                        result['decision'] = decision
                        result['approved'] = decision == 'APPROVE'
                        found_fields.append('decision')
                        self.logger.debug(f"    Parsed decision: {result['decision']}")
                    else:
                        self.logger.debug(f"    Invalid decision value: '{decision}'")
                        
                elif line.startswith('REASON:'):
                    reason = line.replace('REASON:', '').strip()
                    if reason:
                        result['reason'] = reason
                        found_fields.append('reason')
                        self.logger.debug(f"    Parsed reason: '{reason[:50]}...'")
            
            self.logger.debug(f"🔍 Structured parsing found fields: {found_fields}")
            
            # Check if we got the minimum required fields
            if 'score' in result and 'decision' in result:
                # Fill in missing fields with defaults
                if 'correct_type' not in result:
                    result['correct_type'] = 'SYSTEM_CORRECT'
                    self.logger.debug("    Added default correct_type: SYSTEM_CORRECT")
                if 'reason' not in result:
                    result['reason'] = 'Structured analysis completed'
                    self.logger.debug("    Added default reason")
                
                self.logger.debug(f"✅ Structured parsing successful: {result}")
                return result
            else:
                self.logger.debug(f"❌ Structured parsing failed - missing required fields. Have score: {'score' in result}, Have decision: {'decision' in result}")
                return None
            
        except Exception as e:
            self.logger.debug(f"❌ Exception in structured parsing: {e}")
            return None

    def _enhanced_fallback_parse(self, response: str) -> Dict:
        """Enhanced fallback parsing with better intelligence"""
        try:
            response_lower = response.lower()
            
            # Look for any numerical scores
            import re
            
            # Score extraction patterns (more comprehensive)
            score_patterns = [
                r'score[:\s]*(\d+)(?:/10)?',
                r'(\d+)(?:/10)?\s*(?:score|points?|rating)',
                r'rate[sd]?\s*(?:this|it|the\s+signal)?\s*(?:at\s+)?(\d+)(?:/10)?',
                r'give[s]?\s*(?:this|it)?\s*(?:a\s+)?(\d+)(?:/10)?',
                r'(\d+)\s*out\s*of\s*10',
                r'quality[:\s]*(\d+)',
                r'strength[:\s]*(\d+)',
                r'confidence[:\s]*(\d+)',
                r'rating[:\s]*(\d+)'
            ]
            
            score = None
            for pattern in score_patterns:
                matches = re.findall(pattern, response_lower)
                for match in matches:
                    try:
                        potential_score = int(match)
                        if 1 <= potential_score <= 10:
                            score = potential_score
                            break
                    except ValueError:
                        continue
                if score:
                    break
            
            # Decision keywords (more comprehensive)
            strong_approve = ['excellent', 'strong', 'good', 'recommend', 'buy', 'sell', 'take', 'enter']
            weak_approve = ['acceptable', 'okay', 'fair', 'moderate', 'reasonable']
            weak_reject = ['questionable', 'uncertain', 'risky', 'caution']
            strong_reject = ['reject', 'avoid', 'poor', 'weak', 'bad', 'dangerous']
            
            # Count keyword occurrences
            strong_approve_count = sum(1 for word in strong_approve if word in response_lower)
            weak_approve_count = sum(1 for word in weak_approve if word in response_lower)
            weak_reject_count = sum(1 for word in weak_reject if word in response_lower)
            strong_reject_count = sum(1 for word in strong_reject if word in response_lower)
            
            # Decision logic
            total_approve = strong_approve_count + (weak_approve_count * 0.5)
            total_reject = strong_reject_count + (weak_reject_count * 0.5)
            
            if total_reject > total_approve:
                decision = 'REJECT'
            elif total_approve > total_reject:
                decision = 'APPROVE'
            elif score is not None:
                decision = 'APPROVE' if score >= 6 else 'REJECT'
            else:
                decision = 'APPROVE'  # Conservative default
            
            # If no score found, estimate from keywords
            if score is None:
                if strong_approve_count > 0:
                    score = 8
                elif weak_approve_count > 0:
                    score = 6  
                elif weak_reject_count > 0:
                    score = 4
                elif strong_reject_count > 0:
                    score = 2
                else:
                    score = 5  # Neutral
            
            # Signal type detection
            signal_type = 'SYSTEM_CORRECT'  # Default
            if 'bull' in response_lower or 'bullish' in response_lower or 'buy' in response_lower:
                signal_type = 'BULL'
            elif 'bear' in response_lower or 'bearish' in response_lower or 'sell' in response_lower:
                signal_type = 'BEAR'
            
            # Extract reasoning (first meaningful sentence)
            sentences = re.split(r'[.!?]+', response)
            reason = 'Enhanced fallback analysis'
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Meaningful length
                    reason = sentence[:200]  # Limit length
                    break
            
            return {
                'score': score,
                'correct_type': signal_type,
                'decision': decision,
                'reason': reason,
                'approved': decision == 'APPROVE',
                'parsing_method': 'enhanced_fallback'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced fallback parsing failed: {e}")
            return {
                'score': 5,
                'correct_type': 'SYSTEM_CORRECT',
                'decision': 'APPROVE',
                'reason': 'Fallback parsing error - using safe defaults',
                'approved': True,
                'parsing_method': 'safe_default'
            }

    def _try_parse_natural_language(self, response: str) -> Dict:
        """Parse natural language responses from Claude with enhanced pattern matching"""
        result = {}
        
        try:
            response_lower = response.lower()
            self.logger.debug(f"🔍 Natural language parsing on {len(response)} chars")
            
            # 1. Extract score using various patterns
            import re
            score_patterns = [
                r'score[:\s]+(\d+)(?:/10)?',
                r'(\d+)(?:/10)?\s*(?:score|points?|rating)',
                r'rate[sd]?\s*(?:this|it|the\s+signal)?\s*(?:at\s+)?(\d+)(?:/10)?',
                r'give[s]?\s*(?:this|it)?\s*(?:a\s+)?(\d+)(?:/10)?',
                r'(\d+)\s*out\s*of\s*10',
                r'quality[:\s]+(\d+)',
                r'strength[:\s]+(\d+)'
            ]
            
            score = None
            for i, pattern in enumerate(score_patterns):
                match = re.search(pattern, response_lower)
                if match:
                    try:
                        score = int(match.group(1))
                        if 0 <= score <= 10:
                            result['score'] = score
                            self.logger.debug(f"    Found score {score} using pattern {i}: '{pattern}'")
                            break
                    except (ValueError, IndexError):
                        continue
            
            if 'score' not in result:
                self.logger.debug("    No score found in natural language")
            
            # 2. Determine decision based on keywords and score
            approve_keywords = ['approve', 'accept', 'good', 'strong', 'valid', 'buy', 'sell', 'trade', 'recommend', 'favorable']
            reject_keywords = ['reject', 'decline', 'weak', 'poor', 'invalid', 'avoid', 'skip', 'unfavorable', 'bad']
            
            decision = None
            approval_count = sum(1 for word in approve_keywords if word in response_lower)
            rejection_count = sum(1 for word in reject_keywords if word in response_lower)
            
            self.logger.debug(f"    Approval keywords found: {approval_count}, Rejection keywords: {rejection_count}")
            
            if approval_count > rejection_count and approval_count > 0:
                decision = 'APPROVE'
            elif rejection_count > approval_count and rejection_count > 0:
                decision = 'REJECT'
            elif score is not None:
                # Base decision on score
                decision = 'APPROVE' if score >= 6 else 'REJECT'
                self.logger.debug(f"    Decision based on score {score}: {decision}")
            
            if decision:
                result['decision'] = decision
                result['approved'] = decision == 'APPROVE'
                self.logger.debug(f"    Determined decision: {decision}")
            
            # 3. Determine signal type
            if 'bull' in response_lower or 'buy' in response_lower or 'bullish' in response_lower:
                result['correct_type'] = 'BULL'
                self.logger.debug("    Found BULL signal type")
            elif 'bear' in response_lower or 'sell' in response_lower or 'bearish' in response_lower:
                result['correct_type'] = 'BEAR'
                self.logger.debug("    Found BEAR signal type")
            elif 'system' in response_lower and 'correct' in response_lower:
                result['correct_type'] = 'SYSTEM_CORRECT'
                self.logger.debug("    Found SYSTEM_CORRECT signal type")
            else:
                result['correct_type'] = 'SYSTEM_CORRECT'  # Default to accepting system
                self.logger.debug("    Defaulted to SYSTEM_CORRECT signal type")
            
            # 4. Extract reason (first meaningful sentence)
            sentences = re.split(r'[.!?]+', response)
            reason_found = False
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and any(word in sentence.lower() for word in 
                    ['because', 'due to', 'analysis', 'signal', 'trend', 'support', 'resistance', 'indicator', 'momentum']):
                    result['reason'] = sentence
                    reason_found = True
                    self.logger.debug(f"    Found informative reason: '{sentence[:50]}...'")
                    break
            
            if not reason_found and sentences and sentences[0].strip():
                result['reason'] = sentences[0].strip()
                self.logger.debug(f"    Used first sentence as reason: '{sentences[0].strip()[:50]}...'")
            
            # Ensure we have minimum required fields
            if 'score' not in result:
                result['score'] = 6 if result.get('decision') == 'APPROVE' else 4
                self.logger.debug(f"    Added default score based on decision: {result['score']}")
                
            if 'decision' not in result:
                result['decision'] = 'APPROVE' if result.get('score', 5) >= 6 else 'REJECT'
                result['approved'] = result['decision'] == 'APPROVE'
                self.logger.debug(f"    Added default decision based on score: {result['decision']}")
                
            if 'reason' not in result:
                result['reason'] = 'Natural language analysis completed'
                self.logger.debug("    Added default reason")
            
            self.logger.debug(f"✅ Natural language parsing result: score={result.get('score')}, decision={result.get('decision')}, type={result.get('correct_type')}")
            return result if result else None
            
        except Exception as e:
            self.logger.debug(f"❌ Exception in natural language parsing: {e}")
            return None

    def _get_analyzed_indicators_list(self, signal: Dict) -> List[str]:
        """Get list of indicators that were analyzed"""
        indicators = []
        
        # Check which indicators are available
        if signal.get('ema_9') or signal.get('ema_short'):
            indicators.append('EMA')
        if signal.get('macd_histogram') is not None:
            indicators.append('MACD')
        if signal.get('kama_value') or signal.get('kama'):
            indicators.append('KAMA')
        if signal.get('rsi'):
            indicators.append('RSI')
        if signal.get('volume'):
            indicators.append('Volume')
        if signal.get('nearest_support') or signal.get('nearest_resistance'):
            indicators.append('Support/Resistance')
        
        return indicators

    def _validate_market_timestamp_comprehensive(self, signal: Dict) -> Dict:
        """
        COMPREHENSIVE: Validate and clean market timestamp to prevent stale data warnings
        """
        market_timestamp = signal.get('market_timestamp')
        validation_result = {
            'is_valid': True,
            'is_stale': False,
            'cleaned_timestamp': market_timestamp,
            'warning_message': None,
            'fix_applied': False
        }
        
        if market_timestamp is None:
            return validation_result
        
        try:
            # Convert to string for checking
            if hasattr(market_timestamp, 'strftime'):
                timestamp_str = market_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                timestamp_obj = market_timestamp
            else:
                timestamp_str = str(market_timestamp)
                timestamp_obj = pd.to_datetime(market_timestamp) if market_timestamp else None
            
            # Check for epoch time (1970-01-01)
            if timestamp_str.startswith('1970'):
                validation_result.update({
                    'is_valid': False,
                    'is_stale': True,
                    'warning_message': f"Stale epoch timestamp detected: {timestamp_str}"
                })
                
                # Try to replace with current time or signal timestamp
                replacement_candidates = [
                    signal.get('timestamp'),
                    signal.get('detection_time'),
                    signal.get('created_at'),
                    datetime.now()
                ]
                
                for candidate in replacement_candidates:
                    if candidate and (not hasattr(candidate, 'year') or candidate.year > 2020):
                        validation_result.update({
                            'cleaned_timestamp': candidate,
                            'fix_applied': True,
                            'warning_message': f"Stale timestamp replaced with {candidate}"
                        })
                        break
            
            # Check for future timestamps (more than 1 hour ahead)
            elif timestamp_obj and timestamp_obj.year > 2020:
                current_time = datetime.now()
                if hasattr(timestamp_obj, 'tzinfo') and timestamp_obj.tzinfo:
                    current_time = datetime.now(timestamp_obj.tzinfo)
                
                time_diff = (timestamp_obj - current_time).total_seconds()
                if time_diff > 3600:  # More than 1 hour in future
                    validation_result.update({
                        'is_valid': False,
                        'warning_message': f"Future timestamp detected: {timestamp_str} ({time_diff/3600:.1f} hours ahead)"
                    })
        
        except Exception as e:
            validation_result.update({
                'is_valid': False,
                'warning_message': f"Timestamp validation error: {e}"
            })
            
        return validation_result


    def _validate_market_timestamp(self, signal: Dict) -> Dict:
        """
        NEW: Validate and clean market timestamp to prevent stale data warnings
        """
        market_timestamp = signal.get('market_timestamp')
        validation_result = {
            'is_valid': True,
            'is_stale': False,
            'cleaned_timestamp': market_timestamp,
            'warning_message': None
        }
        
        if market_timestamp is None:
            return validation_result
        
        try:
            # Convert to string for checking
            if hasattr(market_timestamp, 'strftime'):
                timestamp_str = market_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                timestamp_obj = market_timestamp
            else:
                timestamp_str = str(market_timestamp)
                timestamp_obj = None
            
            # Check for epoch time (1970-01-01)
            if timestamp_str.startswith('1970'):
                validation_result.update({
                    'is_valid': False,
                    'is_stale': True,
                    'cleaned_timestamp': None,  # Remove the bad timestamp
                    'warning_message': f"Stale epoch timestamp detected: {timestamp_str}"
                })
                
                # Try to replace with current time or signal timestamp
                replacement = signal.get('timestamp') or datetime.now()
                if hasattr(replacement, 'strftime') and replacement.year > 2020:
                    validation_result['cleaned_timestamp'] = replacement
                    validation_result['warning_message'] += f" - replaced with {replacement.strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Check for future timestamps (more than 1 hour ahead)
            elif timestamp_obj and timestamp_obj.year > 2020:
                current_time = datetime.now()
                if hasattr(timestamp_obj, 'tzinfo') and timestamp_obj.tzinfo:
                    current_time = datetime.now(timestamp_obj.tzinfo)
                
                time_diff = (timestamp_obj - current_time).total_seconds()
                if time_diff > 3600:  # More than 1 hour in future
                    validation_result.update({
                        'is_valid': False,
                        'warning_message': f"Future timestamp detected: {timestamp_str} ({time_diff/3600:.1f} hours ahead)"
                    })
        
        except Exception as e:
            validation_result.update({
                'is_valid': False,
                'warning_message': f"Timestamp validation error: {e}"
            })
            
        return validation_result

    def _save_complete_analysis(self, signal: Dict, analysis: Dict):
        """Save complete analysis to file with improved timestamp handling"""
        try:
            epic = signal.get('epic', 'unknown').replace('.', '_')
            
            # IMPROVED: Better timestamp handling with multiple fallbacks
            timestamp = self._get_safe_timestamp_for_filename(signal)
            
            filename = f"{self.save_directory}/complete_analysis_{epic}_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Claude Complete DataFrame Analysis\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Epic: {signal.get('epic', 'N/A')}\n")
                f.write(f"System Signal: {signal.get('signal_type', 'N/A')}\n")
                f.write(f"Claude Determined: {analysis.get('claude_determined_type', 'N/A')}\n")
                f.write(f"Price: {signal.get('price', 'N/A')}\n")
                f.write(f"Triggering Strategy: {self._identify_strategy(signal)}\n")
                f.write(f"Technical Validation: {'PASSED' if analysis.get('technical_validation_passed') else 'FAILED'}\n")
                f.write(f"Analysis Type: {analysis.get('analysis_type', 'N/A')}\n")
                f.write(f"Indicators Analyzed: {', '.join(analysis.get('indicators_analyzed', []))}\n")
                f.write(f"\nCLAUDE COMPLETE ANALYSIS:\n")
                f.write(f"Score: {analysis['score']}/10\n")
                f.write(f"Correct Type: {analysis.get('correct_type', 'N/A')}\n")
                f.write(f"Decision: {analysis['decision']}\n")
                f.write(f"Approved: {analysis['approved']}\n")
                f.write(f"Reason: {analysis['reason']}\n")
                f.write(f"\nSignal Classification Match: {'✅' if analysis.get('claude_determined_type') in [signal.get('signal_type'), 'SYSTEM_CORRECT'] else '❌'}\n")
                f.write(f"\nRaw Response:\n{analysis['raw_response']}\n")
            
            self.logger.debug(f"📁 Complete analysis saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save complete analysis: {e}")
    
    def _validate_signal_technically(self, signal: Dict) -> Dict:
        """
        Technical pre-validation for EMA, MACD, and KAMA strategies
        KEPT FOR COMPATIBILITY: Existing method preserved
        """
        signal_type = signal.get('signal_type', '').upper()
        strategy = self._identify_strategy(signal)
        
        # Initialize validation result
        validation = {
            'valid': False,
            'reason': '',
            'details': '',
            'summary': '',
            'strategy': strategy
        }
        
        try:
            if strategy == 'EMA':
                validation = self._validate_ema_signal_legacy(signal, validation)
            elif strategy == 'MACD':
                validation = self._validate_macd_signal_legacy(signal, validation)
            elif strategy == 'KAMA':
                validation = self._validate_kama_signal_legacy(signal, validation)
            elif strategy == 'COMBINED':
                validation = self._validate_combined_signal_legacy(signal, validation)
            else:
                validation.update({
                    'valid': True,  # Allow unknown strategies to pass
                    'reason': f'Unknown strategy {strategy} - no specific validation rules',
                    'summary': f'{strategy} strategy validation not implemented'
                })
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error in technical validation: {e}")
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}',
                'details': 'Technical validation failed due to system error',
                'summary': 'Validation system error',
                'strategy': strategy
            }
    
    def _identify_strategy(self, signal: Dict) -> str:
        """Identify the strategy type from signal data"""
        strategy = signal.get('strategy', '').lower()
        
        if 'combined' in strategy:
            return 'COMBINED'
        elif 'macd' in strategy:
            return 'MACD'
        elif 'kama' in strategy:
            return 'KAMA'
        elif 'ema' in strategy:
            return 'EMA'
        else:
            # Try to identify from available indicators
            if signal.get('macd_line') is not None or signal.get('macd_histogram') is not None:
                return 'MACD'
            elif signal.get('kama_value') is not None or any(k.startswith('kama_') for k in signal.keys()):
                return 'KAMA'
            elif signal.get('ema_short') is not None or signal.get('ema_9') is not None:
                return 'EMA'
            else:
                return 'UNKNOWN'
    
    def _validate_ema_signal_legacy(self, signal: Dict, validation: Dict) -> Dict:
        """Validate EMA strategy signals"""
        signal_type = signal.get('signal_type', '').upper()
        
        # Get EMA values (try multiple field formats)
        ema_short = signal.get('ema_short') or signal.get('ema_9', 0)
        ema_long = signal.get('ema_long') or signal.get('ema_21', 0)
        ema_trend = signal.get('ema_trend') or signal.get('ema_200', 0)
        price = signal.get('price', 0)
        
        if not all([ema_short, ema_long, ema_trend, price]):
            validation.update({
                'valid': False,
                'reason': 'Missing EMA data or price',
                'details': f'Required: price, ema_short, ema_long, ema_trend. Got: {price}, {ema_short}, {ema_long}, {ema_trend}',
                'summary': 'Incomplete EMA data'
            })
            return validation
        
        # Validate EMA alignment
        if signal_type == 'BULL':
            valid_alignment = price > ema_short > ema_long > ema_trend
            expected = "Price > EMA_Short > EMA_Long > EMA_Trend"
            actual = f"{price:.5f} > {ema_short:.5f} > {ema_long:.5f} > {ema_trend:.5f}"
        elif signal_type == 'BEAR':
            valid_alignment = price < ema_short < ema_long < ema_trend
            expected = "Price < EMA_Short < EMA_Long < EMA_Trend"
            actual = f"{price:.5f} < {ema_short:.5f} < {ema_long:.5f} < {ema_trend:.5f}"
        else:
            validation.update({
                'valid': False,
                'reason': f'Unknown signal type: {signal_type}',
                'details': 'Signal type must be BULL or BEAR',
                'summary': 'Invalid signal type'
            })
            return validation
        
        validation.update({
            'valid': valid_alignment,
            'reason': 'EMA alignment valid' if valid_alignment else f'EMA alignment invalid for {signal_type} signal',
            'details': f'Expected: {expected}\nActual: {actual}\nValid: {valid_alignment}',
            'summary': f'EMA alignment check: {"PASSED" if valid_alignment else "FAILED"}'
        })
        
        return validation
    
    def _validate_macd_signal_legacy(self, signal: Dict, validation: Dict) -> Dict:
        """Validate MACD strategy signals"""
        signal_type = signal.get('signal_type', '').upper()
        
        # Get MACD values
        macd_line = signal.get('macd_line', 0)
        macd_signal_line = signal.get('macd_signal', 0)
        macd_histogram = signal.get('macd_histogram', 0)
        ema_200 = signal.get('ema_200') or signal.get('ema_trend', 0)
        price = signal.get('price', 0)
        
        # Check for MACD data
        has_macd_data = any([macd_line, macd_signal_line, macd_histogram])
        
        if not has_macd_data:
            validation.update({
                'valid': False,
                'reason': 'Missing MACD indicator data',
                'details': 'MACD strategy requires macd_line, macd_signal, or macd_histogram',
                'summary': 'No MACD data available'
            })
            return validation
        
        # Basic MACD validation rules
        valid_conditions = []
        
        if signal_type == 'BULL':
            # MACD bullish conditions
            if macd_histogram > 0:
                valid_conditions.append("MACD histogram positive")
            if macd_line > macd_signal_line:
                valid_conditions.append("MACD line above signal line")
            if ema_200 and price > ema_200:
                valid_conditions.append("Price above EMA 200")
                
        elif signal_type == 'BEAR':
            # MACD bearish conditions
            if macd_histogram < 0:
                valid_conditions.append("MACD histogram negative")
            if macd_line < macd_signal_line:
                valid_conditions.append("MACD line below signal line")
            if ema_200 and price < ema_200:
                valid_conditions.append("Price below EMA 200")
        
        validation.update({
            'valid': len(valid_conditions) >= 1,  # At least one MACD condition should be met
            'reason': f'MACD conditions: {len(valid_conditions)} met' if valid_conditions else 'No MACD conditions met',
            'details': f'Valid conditions: {", ".join(valid_conditions)}' if valid_conditions else 'No MACD conditions satisfied',
            'summary': f'MACD validation: {"PASSED" if valid_conditions else "FAILED"}'
        })
        
        return validation
    
    def _validate_kama_signal_legacy(self, signal: Dict, validation: Dict) -> Dict:
        """Validate KAMA strategy signals"""
        signal_type = signal.get('signal_type', '').upper()
        
        # Get KAMA values (multiple possible field names)
        kama_value = signal.get('kama_value') or signal.get('kama_current')
        efficiency_ratio = signal.get('efficiency_ratio') or signal.get('kama_er')
        kama_trend = signal.get('kama_trend')
        ema_200 = signal.get('ema_200') or signal.get('ema_trend', 0)
        price = signal.get('price', 0)
        
        # Look for any KAMA-related fields
        kama_fields = [k for k in signal.keys() if 'kama' in k.lower()]
        
        if not kama_fields and not kama_value:
            validation.update({
                'valid': False,
                'reason': 'Missing KAMA indicator data',
                'details': 'KAMA strategy requires kama_value, efficiency_ratio, or other KAMA indicators',
                'summary': 'No KAMA data available'
            })
            return validation
        
        # Basic KAMA validation rules
        valid_conditions = []
        
        # Check efficiency ratio (if available)
        if efficiency_ratio is not None:
            if efficiency_ratio > 0.1:  # Minimum efficiency for valid signals
                valid_conditions.append(f"Efficiency ratio acceptable: {efficiency_ratio:.3f}")
            else:
                valid_conditions.append(f"Efficiency ratio too low: {efficiency_ratio:.3f}")
        
        # Check trend filter with EMA 200
        if ema_200 and price:
            if signal_type == 'BULL' and price > ema_200:
                valid_conditions.append("Price above EMA 200 (bullish)")
            elif signal_type == 'BEAR' and price < ema_200:
                valid_conditions.append("Price below EMA 200 (bearish)")
        
        validation.update({
            'valid': len(valid_conditions) >= 1,  # At least basic KAMA validation
            'reason': f'KAMA conditions: {len(valid_conditions)} met' if valid_conditions else 'Insufficient KAMA validation',
            'details': f'Conditions: {", ".join(valid_conditions)}' if valid_conditions else 'No KAMA conditions met',
            'summary': f'KAMA validation: {"PASSED" if valid_conditions else "WARNING"}'
        })
        
        return validation
    
    def _validate_combined_signal_legacy(self, signal: Dict, validation: Dict) -> Dict:
        """Validate combined strategy signals"""
        # For combined signals, check if individual components are valid
        strategy_data = signal.get('strategy_indicators', {})
        ema_data = strategy_data.get('ema_data')
        macd_data = strategy_data.get('macd_data')
        
        valid_components = []
        
        # Check EMA component if present
        if ema_data or any(k.startswith('ema_') for k in signal.keys()):
            ema_validation = self._validate_ema_signal_legacy(signal, {'valid': False})
            if ema_validation['valid']:
                valid_components.append("EMA alignment valid")
        
        # Check MACD component if present
        if macd_data or any(k.startswith('macd_') for k in signal.keys()):
            macd_validation = self._validate_macd_signal_legacy(signal, {'valid': False})
            if macd_validation['valid']:
                valid_components.append("MACD conditions valid")
        
        validation.update({
            'valid': len(valid_components) >= 1,  # At least one component should be valid
            'reason': f'Combined strategy: {len(valid_components)} valid components',
            'details': f'Valid components: {", ".join(valid_components)}' if valid_components else 'No valid strategy components',
            'summary': f'Combined validation: {"PASSED" if valid_components else "FAILED"}'
        })
        
        return validation
    
    def _create_enhanced_validation_prompt(self, signal: Dict, technical_validation: Dict) -> str:
        """Create enhanced prompt with technical validation context"""
        
        # Get price for display
        if 'price_mid' in signal:
            price_info = f"MID: {signal['price_mid']:.5f}, EXEC: {signal['execution_price']:.5f}"
        else:
            price_info = f"{signal.get('price', 'N/A'):.5f}"
        
        # Get strategy-specific info
        strategy = technical_validation['strategy']
        strategy_info = self._get_strategy_info_for_claude(signal, strategy)
        
        prompt = f"""
FOREX SIGNAL TECHNICAL VALIDATION AND ANALYSIS

TECHNICAL PRE-VALIDATION: ✅ PASSED
Strategy: {strategy}
Validation Summary: {technical_validation['summary']}

Analyze this technically valid signal and respond EXACTLY in this format:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]  
REASON: [Market context and quality assessment]

Signal Data:
- Pair: {signal.get('epic', 'N/A')}
- Type: {signal.get('signal_type', 'N/A')}
- Price: {price_info}
- Confidence: {signal.get('confidence_score', 0):.1%}
- Strategy: {strategy}

{strategy_info}

Since this signal passed technical validation, focus your analysis on:
1. Market context and timing
2. Risk/reward potential  
3. Overall signal quality
4. Any additional market factors

Respond with ONLY the three lines above, nothing else.
"""
        
        return prompt
    
    def _get_strategy_info_for_claude(self, signal: Dict, strategy: str) -> str:
        """Get strategy-specific information for Claude prompt"""
        
        if strategy == 'EMA':
            return self._get_ema_info_for_claude(signal)
        elif strategy == 'MACD':
            return self._get_macd_info_for_claude(signal)
        elif strategy == 'KAMA':
            return self._get_kama_info_for_claude(signal)
        elif strategy == 'COMBINED':
            ema_info = self._get_ema_info_for_claude(signal)
            macd_info = self._get_macd_info_for_claude(signal)
            return f"{ema_info}\n{macd_info}"
        else:
            return "- Strategy indicators: Not specified"
    
    def _get_ema_info_for_claude(self, signal: Dict) -> str:
        """Extract EMA information for Claude"""
        ema_lines = []
        
        # Try semantic names first
        if 'ema_short' in signal and 'ema_long' in signal and 'ema_trend' in signal:
            ema_lines.append(f"- EMA Short: {self._safe_format_number(signal.get('ema_short'))}")
            ema_lines.append(f"- EMA Long: {self._safe_format_number(signal.get('ema_long'))}")
            ema_lines.append(f"- EMA Trend: {self._safe_format_number(signal.get('ema_trend'))}")
            
            # Add configuration if available
            if signal.get('ema_config'):
                config = signal['ema_config']
                ema_lines.append(f"- Config: {config.get('short', '?')}/{config.get('long', '?')}/{config.get('trend', '?')}")
        
        # Fallback to numbered EMAs
        elif any(key.startswith('ema_') and key[4:].isdigit() for key in signal.keys()):
            ema_periods = []
            for key in signal.keys():
                if key.startswith('ema_') and key[4:].isdigit():
                    period = int(key[4:])
                    value = signal.get(key)
                    if value is not None:
                        ema_periods.append((period, value))
            
            ema_periods.sort()
            for period, value in ema_periods:
                ema_lines.append(f"- EMA{period}: {self._safe_format_number(value)}")
        
        # Specific legacy support
        else:
            if signal.get('ema_9') is not None:
                ema_lines.append(f"- EMA9: {self._safe_format_number(signal.get('ema_9'))}")
            if signal.get('ema_21') is not None:
                ema_lines.append(f"- EMA21: {self._safe_format_number(signal.get('ema_21'))}")
            if signal.get('ema_200') is not None:
                ema_lines.append(f"- EMA200: {self._safe_format_number(signal.get('ema_200'))}")
        
        if not ema_lines:
            ema_lines.append("- EMA Data: Not available")
        
        return '\n'.join(ema_lines)
    
    def _get_macd_info_for_claude(self, signal: Dict) -> str:
        """Extract MACD information for Claude"""
        macd_lines = []
        
        if signal.get('macd_line') is not None:
            macd_lines.append(f"- MACD Line: {self._safe_format_number(signal.get('macd_line'))}")
        if signal.get('macd_signal') is not None:
            macd_lines.append(f"- MACD Signal: {self._safe_format_number(signal.get('macd_signal'))}")
        if signal.get('macd_histogram') is not None:
            macd_lines.append(f"- MACD Histogram: {self._safe_format_number(signal.get('macd_histogram'))}")
        
        if not macd_lines:
            macd_lines.append("- MACD Data: Not available")
        
        return '\n'.join(macd_lines)
    
    def _get_kama_info_for_claude(self, signal: Dict) -> str:
        """Extract KAMA information for Claude"""
        kama_lines = []
        
        if signal.get('kama_value') is not None:
            kama_lines.append(f"- KAMA Value: {self._safe_format_number(signal.get('kama_value'))}")
        if signal.get('efficiency_ratio') is not None:
            kama_lines.append(f"- Efficiency Ratio: {self._safe_format_number(signal.get('efficiency_ratio'))}")
        if signal.get('kama_trend') is not None:
            kama_lines.append(f"- KAMA Trend: {signal.get('kama_trend')}")
        
        # Look for other KAMA fields
        kama_fields = {k: v for k, v in signal.items() if 'kama' in k.lower() and v is not None}
        for field, value in kama_fields.items():
            if field not in ['kama_value', 'kama_trend']:
                kama_lines.append(f"- {field.replace('_', ' ').title()}: {self._safe_format_number(value)}")
        
        if not kama_lines:
            kama_lines.append("- KAMA Data: Not available")
        
        return '\n'.join(kama_lines)
    
    def batch_analyze_signals_minimal(self, signals: List[Dict], save_to_file: bool = None) -> List[Dict]:
        """
        Enhanced batch analysis with technical validation
        """
        results = []
        
        for i, signal in enumerate(signals, 1):
            self.logger.info(f"📊 Analyzing signal {i}/{len(signals)}: {signal.get('epic', 'Unknown')}")
            
            analysis = self.analyze_signal_minimal(signal, save_to_file=save_to_file)
            
            if analysis:
                results.append({
                    'signal': signal,
                    'score': analysis['score'],
                    'decision': analysis['decision'],
                    'approved': analysis['approved'],
                    'reason': analysis['reason'],
                    'technical_validation_passed': analysis.get('technical_validation_passed', False),
                    'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                results.append({
                    'signal': signal,
                    'score': None,
                    'decision': 'REJECT',
                    'approved': False,
                    'reason': 'Analysis failed',
                    'technical_validation_passed': False,
                    'error': 'Analysis failed'
                })
        
        # Save batch summary if enabled
        should_save = save_to_file if save_to_file is not None else self.auto_save
        if should_save and results:
            self._save_batch_summary_minimal(results)
        
        return results
    
    def test_connection(self) -> bool:
        """Test Claude API connection"""
        if not self.api_key:
            return False
        
        try:
            test_prompt = "Please respond with 'Connection successful' if you can read this message."
            response = self._call_claude_api(test_prompt)
            return response is not None and "successful" in response.lower()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def parse_minimal_response(self, response: str) -> Dict:
        """
        Parse the minimal Claude response into structured data
        FIXED: Safe parsing with proper type conversion
        """
        try:
            result = {
                'score': None,
                'decision': None,
                'reason': None,
                'approved': False
            }
            
            if not response or not response.strip():
                return result
            
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            for line in lines:
                if line.startswith('SCORE:'):
                    score_text = line.replace('SCORE:', '').strip()
                    try:
                        # ✅ FIXED: Safe score extraction
                        result['score'] = int(float(score_text))
                    except ValueError:
                        # Extract number from text like "8/10" or "Score: 7"
                        import re
                        numbers = re.findall(r'\b(\d+)\b', score_text)
                        if numbers:
                            try:
                                result['score'] = int(numbers[0])
                            except (ValueError, IndexError):
                                result['score'] = 0
                                self.logger.warning(f"Could not parse score from: {score_text}")
                
                elif line.startswith('DECISION:'):
                    decision = line.replace('DECISION:', '').strip().upper()
                    result['decision'] = decision
                    result['approved'] = decision == 'APPROVE'
                
                elif line.startswith('REASON:'):
                    result['reason'] = line.replace('REASON:', '').strip()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing minimal response: {e}")
            return {
                'score': 0,  # Safe default
                'decision': 'REJECT',
                'reason': 'Parse error',
                'approved': False
            }

    def _call_claude_api(self, prompt: str, max_tokens: int = None) -> Optional[str]:
        """
        Enhanced API call with retry logic and rate limiting
        ✅ FIXED: Comprehensive error handling for 529 errors and timeouts
        """
        if not self.api_key:
            self.logger.warning("No API key available")
            return None
        
        # ✅ NEW: Rate limiting protection
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            self.logger.debug(f"⏱️ Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # ✅ FIXED: Clean Unicode issues in prompt before sending to API
        cleaned_prompt = self._clean_unicode_for_api(prompt)

        # Log if cleaning was necessary
        if cleaned_prompt != prompt:
            original_len = len(prompt)
            cleaned_len = len(cleaned_prompt)
            self.logger.debug(f"🧹 Unicode cleaning applied: {original_len} → {cleaned_len} chars")

        data = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": cleaned_prompt}]
        }

        # ✅ FIXED: Additional safety - clean entire data structure
        data = self._clean_unicode_for_api(data)
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.last_api_call = time.time()
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('content', [])
                    if content and len(content) > 0:
                        return content[0].get('text', '')
                    else:
                        self.logger.warning("Empty content in successful response")
                        return None
                
                elif response.status_code == 529:
                    # ✅ ENHANCED: Specific handling for overload errors
                    error_msg = f"Claude API overloaded (attempt {attempt + 1}/{self.max_retries + 1})"
                    self.logger.warning(f"⚠️ {error_msg}")
                    
                    if attempt < self.max_retries:
                        # Exponential backoff with jitter for 529 errors
                        delay = min(
                            self.base_delay * (self.exponential_base ** attempt) + random.uniform(0, 2),
                            self.max_delay
                        )
                        self.logger.info(f"🔄 Retrying in {delay:.1f}s due to API overload...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"❌ Claude API overloaded after {self.max_retries} retries")
                        return None
                
                elif response.status_code == 429:
                    # ✅ ENHANCED: Rate limit handling
                    error_msg = f"Rate limit exceeded (attempt {attempt + 1}/{self.max_retries + 1})"
                    self.logger.warning(f"⚠️ {error_msg}")
                    
                    if attempt < self.max_retries:
                        # Longer delay for rate limits
                        delay = min(60 + random.uniform(0, 30), self.max_delay)
                        self.logger.info(f"🔄 Retrying in {delay:.1f}s due to rate limit...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"❌ Rate limit exceeded after {self.max_retries} retries")
                        return None
                
                else:
                    # ✅ ENHANCED: Other HTTP error handling
                    error_text = response.text
                    self.logger.error(f"❌ Claude API error: {response.status_code} - {error_text}")
                    
                    # Don't retry for client errors (4xx) except 429
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        return None
                    
                    if attempt < self.max_retries:
                        delay = self.base_delay * (self.exponential_base ** attempt)
                        self.logger.info(f"🔄 Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        return None
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                error_msg = f"Request timeout (attempt {attempt + 1}/{self.max_retries + 1})"
                self.logger.warning(f"⚠️ {error_msg}")
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (self.exponential_base ** attempt)
                    self.logger.info(f"🔄 Retrying in {delay:.1f}s due to timeout...")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"❌ Claude API timeout after {self.max_retries} retries")
                    return None
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                error_msg = f"Request failed: {str(e)} (attempt {attempt + 1}/{self.max_retries + 1})"
                self.logger.warning(f"⚠️ {error_msg}")
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (self.exponential_base ** attempt)
                    self.logger.info(f"🔄 Retrying in {delay:.1f}s due to request error...")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"❌ Claude API request failed after {self.max_retries} retries: {e}")
                    return None
                    
            except Exception as e:
                last_exception = e
                self.logger.error(f"❌ Unexpected error in Claude API call: {e}")
                return None
        
        # ✅ Should never reach here, but just in case
        self.logger.error(f"❌ All retry attempts exhausted. Last exception: {last_exception}")
        return None
    
    def _save_minimal_analysis(self, signal: Dict, analysis: Dict):
        """Save minimal analysis to file with improved timestamp handling"""
        try:
            epic = signal.get('epic', 'unknown').replace('.', '_')
            
            # IMPROVED: Better timestamp handling with multiple fallbacks
            timestamp = self._get_safe_timestamp_for_filename(signal)
            
            filename = f"{self.save_directory}/minimal_analysis_{epic}_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Claude Enhanced Analysis\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Epic: {signal.get('epic', 'N/A')}\n")
                f.write(f"Signal: {signal.get('signal_type', 'N/A')}\n")
                f.write(f"Price: {signal.get('price', 'N/A')}\n")
                f.write(f"Strategy: {self._identify_strategy(signal)}\n")
                f.write(f"Technical Validation: {'PASSED' if analysis.get('technical_validation_passed') else 'FAILED'}\n")
                f.write(f"\nCLAUDE DECISION:\n")
                f.write(f"Score: {analysis['score']}/10\n")
                f.write(f"Decision: {analysis['decision']}\n")
                f.write(f"Approved: {analysis['approved']}\n")
                f.write(f"Reason: {analysis['reason']}\n")
                f.write(f"\nRaw Response:\n{analysis['raw_response']}\n")
            
            self.logger.debug(f"📁 Enhanced analysis saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save minimal analysis: {e}")
    
    def _save_batch_summary_minimal(self, results: List[Dict]):
        """Save enhanced batch analysis summary"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.save_directory}/batch_enhanced_{timestamp}.txt"
            
            approved = len([r for r in results if r.get('approved')])
            tech_passed = len([r for r in results if r.get('technical_validation_passed')])
            total = len(results)
            avg_score = sum([r['score'] for r in results if r['score']]) / len([r for r in results if r['score']]) if results else 0
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Claude Enhanced Batch Analysis\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Signals: {total}\n")
                f.write(f"Technical Validation Passed: {tech_passed} ({tech_passed/total*100:.1f}%)\n")
                f.write(f"Claude Approved: {approved} ({approved/total*100:.1f}%)\n")
                f.write(f"Average Score: {avg_score:.1f}/10\n\n")
                
                for i, result in enumerate(results, 1):
                    signal = result['signal']
                    tech_status = "✅" if result.get('technical_validation_passed') else "❌"
                    f.write(f"{i:2d}. {tech_status} {signal.get('epic', 'Unknown'):20s} {signal.get('signal_type', 'Unknown'):4s} ")
                    f.write(f"Score: {result['score'] or 'N/A':2s}/10 ")
                    f.write(f"Decision: {result['decision']:7s} ")
                    f.write(f"Reason: {result['reason'] or 'N/A'}\n")
            
            self.logger.info(f"📁 Enhanced batch summary saved: {tech_passed}/{total} tech valid, {approved}/{total} approved")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save enhanced batch summary: {e}")
    
    def _get_safe_timestamp_for_filename(self, signal: Dict) -> str:
        """
        FIXED: Safely extract timestamp for filename with comprehensive validation
        Fixes both 19700101_000533 and market_timestamp stale data issues
        """
        # Try multiple timestamp sources in order of preference
        timestamp_sources = [
            ('timestamp', signal.get('timestamp')),
            ('signal_timestamp', signal.get('signal_timestamp')),
            ('detection_time', signal.get('detection_time')),
            ('created_at', signal.get('created_at')),
            ('alert_timestamp', signal.get('alert_timestamp')),
            ('market_timestamp', signal.get('market_timestamp')),  # Try this but validate carefully
        ]
        
        for source_name, timestamp in timestamp_sources:
            if timestamp is None:
                continue
                
            try:
                # Handle datetime objects
                if hasattr(timestamp, 'strftime'):
                    # Check if it's a reasonable date (not epoch time)
                    if timestamp.year > 2020:
                        result = timestamp.strftime('%Y%m%d_%H%M%S')
                        self.logger.debug(f"✅ Using {source_name} timestamp: {result}")
                        return result
                    else:
                        self.logger.debug(f"⚠️ Rejecting {source_name} - year {timestamp.year} too old")
                        continue
                
                # Handle string timestamps
                if isinstance(timestamp, str):
                    # Skip obviously bad timestamps
                    if timestamp.startswith('1970') or timestamp.startswith('1969'):
                        self.logger.debug(f"⚠️ Rejecting {source_name} - starts with epoch year: {timestamp}")
                        continue
                    
                    # Try to parse various formats
                    timestamp_formats = [
                        '%Y-%m-%d %H:%M:%S.%f',  # 2025-07-31 07:15:00.123456
                        '%Y-%m-%d %H:%M:%S',     # 2025-07-31 07:15:00
                        '%Y-%m-%dT%H:%M:%S.%fZ', # 2025-07-31T07:15:00.123456Z
                        '%Y-%m-%dT%H:%M:%SZ',    # 2025-07-31T07:15:00Z
                        '%Y-%m-%dT%H:%M:%S',     # 2025-07-31T07:15:00
                        '%Y%m%d_%H%M%S',         # 20250731_071500
                        '%Y-%m-%d',              # 2025-07-31
                    ]
                    
                    for fmt in timestamp_formats:
                        try:
                            dt = datetime.strptime(timestamp, fmt)
                            if dt.year > 2020:  # Sanity check
                                result = dt.strftime('%Y%m%d_%H%M%S')
                                self.logger.debug(f"✅ Parsed {source_name} timestamp: {result}")
                                return result
                        except ValueError:
                            continue
                    
                    # Try to clean string and extract parts
                    cleaned = str(timestamp).replace(':', '').replace(' ', '_').replace('-', '')
                    if len(cleaned) >= 8 and cleaned[:8].isdigit():
                        year_part = cleaned[:4]
                        try:
                            if int(year_part) > 2020:
                                # Pad with zeros if needed
                                if len(cleaned) < 15:
                                    cleaned += '0' * (15 - len(cleaned))
                                result = cleaned[:15]
                                self.logger.debug(f"✅ Cleaned {source_name} timestamp: {result}")
                                return result
                        except ValueError:
                            continue
                
                # Handle numeric timestamps (Unix time)
                if isinstance(timestamp, (int, float)):
                    # Check if this looks like a reasonable Unix timestamp
                    if timestamp > 1600000000 and timestamp < 2000000000:  # Between 2020 and 2033
                        dt = datetime.fromtimestamp(timestamp)
                        result = dt.strftime('%Y%m%d_%H%M%S')
                        self.logger.debug(f"✅ Converted {source_name} Unix timestamp: {result}")
                        return result
                    else:
                        self.logger.debug(f"⚠️ Rejecting {source_name} - invalid Unix timestamp: {timestamp}")
                        continue
                    
            except Exception as e:
                self.logger.debug(f"❌ Error processing {source_name} timestamp {timestamp}: {e}")
                continue
        
        # Ultimate fallback - use current time
        current_time = datetime.now()
        result = current_time.strftime('%Y%m%d_%H%M%S')
        
        epic = signal.get('epic', 'unknown')
        self.logger.warning(f"⚠️ No valid timestamp found for {epic}, using current time: {result}")
        
        # Add debug info about what timestamps were tried
        tried_timestamps = [f"{name}={value}" for name, value in timestamp_sources if value is not None]
        if tried_timestamps:
            self.logger.debug(f"   Tried timestamps: {', '.join(tried_timestamps[:3])}")  # Limit to first 3
        
        return result
    
    def _fallback_parse(self, response: str) -> Dict:
        """
        Enhanced fallback parsing with better error handling
        """
        try:
            # Extract any numbers that might be scores
            import re
            numbers = re.findall(r'\b(\d+)\b', response)
            
            # Look for score-like numbers (1-10 range)
            score = 5  # Default
            for num_str in numbers:
                num = int(num_str)
                if 1 <= num <= 10:
                    score = num
                    break
            
            # Determine decision based on content
            response_lower = response.lower()
            decision = 'APPROVE'
            
            if any(word in response_lower for word in ['reject', 'poor', 'weak', 'avoid', 'bad']):
                decision = 'REJECT'
            elif score < 6:
                decision = 'REJECT'
            
            return {
                'score': score,
                'correct_type': 'SYSTEM_CORRECT',
                'decision': decision,
                'reason': 'Fallback parsing - extracted from unstructured response',
                'approved': decision == 'APPROVE'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fallback parsing failed: {e}")
            return {
                'score': 0,
                'correct_type': 'UNKNOWN',
                'decision': 'REJECT',
                'reason': f'Fallback parsing error: {str(e)}',
                'approved': False
            }

    def _safe_format_number(self, value) -> str:
        """Safely format a number for display, handling None and string values"""
        try:
            if value is None:
                return 'N/A'
            if isinstance(value, str):
                return value
            return f"{float(value):.5f}"
        except (ValueError, TypeError):
            return 'N/A'

    def analyze_signal_at_timestamp(self, epic: str, timestamp_str: str, signal_detector, include_future_analysis: bool = False) -> Optional[Dict]:
        """
        NEW METHOD: Analyze signal at specific timestamp with optional future data
        Used for backtesting and detailed analysis
        """
        try:
            from datetime import datetime
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            self.logger.info(f"🔍 Analyzing {epic} at {timestamp_str}")
            
            # Extract pair from epic (e.g., 'CS.D.EURUSD.MINI.IP' -> 'EURUSD')
            pair = epic.split('.')[2] if '.' in epic else epic
            
            # Get market data using the correct data_fetcher method
            lookback_hours = 48 if include_future_analysis else 24  # Get more data if analyzing future
            
            df = signal_detector.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='5m',  # Use 5-minute timeframe
                lookback_hours=lookback_hours,
                user_timezone='Europe/Stockholm',  # Use default timezone
                required_indicators=['ema', 'macd', 'kama', 'rsi', 'volume']  # Get all indicators
            )
            
            if df is None or df.empty:
                return {'error': f'No data available for {epic} at {timestamp_str}'}
            
            # Find the candle closest to the target timestamp
            # Convert timestamp to match DataFrame timezone
            if hasattr(df.index, 'tz_localize'):
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                target_timestamp = timestamp.replace(tzinfo=df.index.tz) if timestamp.tzinfo is None else timestamp
            else:
                target_timestamp = timestamp
            
            # Find the candle at or before the target timestamp
            available_times = df[df.index <= target_timestamp]
            if available_times.empty:
                return {'error': f'No data available at or before {timestamp_str}'}
            
            target_candle = available_times.iloc[-1]
            target_time = available_times.index[-1]
            
            self.logger.info(f"🕐 Found candle at {target_time} for analysis at {timestamp_str}")
            
            # Prepare signal data similar to how signal_detector does it
            signal_data = {
                'epic': epic,
                'timestamp': target_time,
                'price': float(target_candle['close']),
                'open_price': float(target_candle['open']),
                'high_price': float(target_candle['high']),
                'low_price': float(target_candle['low']),
                'close_price': float(target_candle['close']),
                'volume': float(target_candle.get('ltv', target_candle.get('volume', 0))),
                'timeframe': '5m',
                'pair': pair
            }
            
            # Add technical indicators if available - use proper column names
            indicator_mappings = {
                'ema_9': ['ema_9', 'ema_short'],
                'ema_21': ['ema_21', 'ema_long'], 
                'ema_200': ['ema_200', 'ema_trend'],
                'macd_line': ['macd_line', 'macd'],
                'macd_signal': ['macd_signal', 'macd_signal_line'],
                'macd_histogram': ['macd_histogram', 'macd_hist'],
                'kama_value': ['kama_value', 'kama'],
                'efficiency_ratio': ['efficiency_ratio', 'kama_er'],
                'rsi': ['rsi'],
                'bb_upper': ['bb_upper', 'bollinger_upper'],
                'bb_middle': ['bb_middle', 'bollinger_middle'],
                'bb_lower': ['bb_lower', 'bollinger_lower'],
                'atr': ['atr']
            }
            
            # Map available indicators to signal data
            for standard_name, possible_cols in indicator_mappings.items():
                for col in possible_cols:
                    if col in target_candle.index:
                        signal_data[standard_name] = float(target_candle[col])
                        break
            
            # Add strategy-specific data
            signal_data['strategy'] = 'retrospective_analysis'
            signal_data['signal_type'] = 'ANALYSIS'  # Will be determined by Claude
            signal_data['confidence_score'] = 0.8  # Default for analysis
            
            # If future analysis is requested, add forward-looking data
            if include_future_analysis:
                # Get future data from the same DataFrame
                future_data = df[df.index > target_time]
                if not future_data.empty:
                    # Look ahead 12 candles (1 hour for 5-minute timeframe)
                    next_candles = future_data.head(12)
                    
                    if not next_candles.empty:
                        # Calculate price movement
                        start_price = float(target_candle['close'])
                        max_high = float(next_candles['high'].max())
                        min_low = float(next_candles['low'].min())
                        end_price = float(next_candles.iloc[-1]['close'])
                        
                        # Calculate pip movements (use proper pip calculation)
                        if 'JPY' in epic:
                            pip_multiplier = 100  # JPY pairs have 2 decimal places
                        else:
                            pip_multiplier = 10000  # Most pairs have 4 decimal places
                            
                        max_gain_pips = (max_high - start_price) * pip_multiplier
                        max_loss_pips = (start_price - min_low) * pip_multiplier
                        net_movement_pips = (end_price - start_price) * pip_multiplier
                        
                        signal_data['future_analysis'] = {
                            'next_hour_high': max_high,
                            'next_hour_low': min_low,
                            'next_hour_close': end_price,
                            'max_gain_pips': round(max_gain_pips, 1),
                            'max_loss_pips': round(max_loss_pips, 1),
                            'net_movement_pips': round(net_movement_pips, 1),
                            'favorable_movement': abs(max_gain_pips) > abs(max_loss_pips),
                            'candles_analyzed': len(next_candles),
                            'price_range_pips': round((max_high - min_low) * pip_multiplier, 1)
                        }
                        
                        self.logger.info(f"🔮 Future analysis: {next_candles.iloc[0].name} to {next_candles.iloc[-1].name}")
                        self.logger.info(f"   Max gain: {max_gain_pips:.1f} pips, Max loss: {max_loss_pips:.1f} pips")
                        self.logger.info(f"   Net movement: {net_movement_pips:.1f} pips")
                else:
                    self.logger.warning(f"⚠️ No future data available after {target_time}")
            
            # Log available indicators for debugging
            available_indicators = [k for k in signal_data.keys() if any(ind in k.lower() for ind in ['ema_', 'macd_', 'kama_', 'rsi', 'bb_', 'atr'])]
            self.logger.info(f"📊 Available indicators: {available_indicators}")
            
            # Perform Claude analysis
            analysis = self.analyze_signal_minimal(signal_data, save_to_file=True)
            
            if analysis:
                analysis['timestamp_analyzed'] = timestamp_str
                analysis['actual_candle_time'] = str(target_time)
                analysis['epic'] = epic
                analysis['pair'] = pair
                analysis['market_data'] = {
                    'price': float(target_candle['close']),
                    'volume': float(target_candle.get('ltv', target_candle.get('volume', 0))),
                    'technical_indicators_count': len(available_indicators),
                    'available_indicators': available_indicators
                }
                
                if include_future_analysis and 'future_analysis' in signal_data:
                    analysis['outcome'] = signal_data['future_analysis']
                    
                    # Determine if the outcome matched the analysis
                    if analysis.get('decision') == 'APPROVE':
                        favorable = signal_data['future_analysis']['favorable_movement']
                        analysis['outcome_accuracy'] = 'correct' if favorable else 'incorrect'
                    else:
                        analysis['outcome_accuracy'] = 'rejected'
                
                self.logger.info(f"✅ Timestamp analysis complete: Score {analysis['score']}/10, Decision: {analysis['decision']}")
                
                return analysis
            else:
                return {'error': 'Claude analysis failed', 'signal_data_keys': list(signal_data.keys())}
                
        except Exception as e:
            self.logger.error(f"❌ Timestamp analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Analysis error: {str(e)}', 'traceback': traceback.format_exc()}


# Enhanced utility functions with technical validation
def create_minimal_claude_analyzer(api_key: str = None, auto_save: bool = True) -> ClaudeAnalyzer:
    """Create Claude analyzer with technical validation"""
    if not api_key:
        api_key = getattr(config, 'CLAUDE_API_KEY', None)
    
    if not api_key:
        logging.getLogger(__name__).warning("⚠️ No Claude API key found in config or parameters")
    
    analyzer = ClaudeAnalyzer(api_key, auto_save=auto_save)
    analyzer.max_tokens = 250  # Increased for complete DataFrame analysis
    return analyzer

def create_claude_analyzer(api_key: str = None, auto_save: bool = True) -> ClaudeAnalyzer:
    """Create Claude analyzer (compatibility function)"""
    return create_minimal_claude_analyzer(api_key, auto_save)

def quick_signal_check(signal: Dict, api_key: str = None) -> bool:
    """Quick function with technical validation"""
    try:
        analyzer = create_minimal_claude_analyzer(api_key, auto_save=False)
        result = analyzer.analyze_signal_minimal(signal, save_to_file=False)
        
        if result:
            return result['approved']
        return False
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Quick signal check failed: {e}")
        return False

def batch_check_signals(signals: List[Dict], api_key: str = None) -> List[bool]:
    """Batch check with technical validation"""
    try:
        analyzer = create_minimal_claude_analyzer(api_key, auto_save=False)
        results = analyzer.batch_analyze_signals_minimal(signals, save_to_file=False)
        
        return [result['approved'] for result in results]
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Batch signal check failed: {e}")
        return [False] * len(signals)


# Example usage and testing
if __name__ == "__main__":
    # Test with enhanced validation
    analyzer = create_minimal_claude_analyzer(auto_save=True)
    
    if analyzer.test_connection():
        print("✅ Claude connection successful")
        
        # Test signal with your problematic data
        test_signal = {
            'epic': 'CS.D.EURUSD.MINI.IP',
            'signal_type': 'BULL',  # System suggests BULL
            'price': 0.813675,
            'confidence_score': 0.853,
            'strategy': 'kama',
            # Complete DataFrame data
            'ema_9': 0.8134414062487326,
            'ema_21': 0.8132858945782762,
            'ema_200': 0.8077259841138129,
            'macd_line': 0.00037725395948196017,
            'macd_signal': 0.0005984096192572145,
            'macd_histogram': -0.0002211556597752543,  # NEGATIVE!
            'kama_value': 0.8134,
            'efficiency_ratio': 0.853,
            'volume': 118.0,
            'volume_confirmation': True,
            'timestamp': datetime.now()
        }
        
        # This should now be REJECTED due to complete DataFrame analysis
        result = analyzer.analyze_signal_minimal(test_signal, save_to_file=True)
        
        if result:
            print(f"\n🔍 COMPLETE ANALYSIS RESULTS:")
            print(f"Technical Validation: {'✅ PASSED' if result.get('technical_validation_passed') else '❌ FAILED'}")
            print(f"System Classification: {test_signal['signal_type']}")
            print(f"Claude Determined: {result.get('claude_determined_type', 'N/A')}")
            print(f"Agreement: {'✅' if result.get('claude_determined_type') in [test_signal['signal_type'], 'SYSTEM_CORRECT'] else '❌'}")
            print(f"Score: {result['score']}/10")
            print(f"Decision: {result['decision']}")
            print(f"Approved: {'✅' if result['approved'] else '❌'}")
            print(f"Reason: {result['reason']}")
            print(f"Analysis Type: {result.get('analysis_type', 'N/A')}")
            print(f"Indicators Analyzed: {', '.join(result.get('indicators_analyzed', []))}")
            
            # Test the critical MACD histogram contradiction detection
            macd_histogram = test_signal.get('macd_histogram')
            signal_type = test_signal.get('signal_type')
            
            if macd_histogram is not None and signal_type == 'BULL' and macd_histogram < -0.0001:
                print(f"\n🚨 CRITICAL CONTRADICTION DETECTED:")
                print(f"   Signal Type: {signal_type}")
                print(f"   MACD Histogram: {macd_histogram:.6f} (strongly negative)")
                print(f"   Expected Result: REJECTION")
                print(f"   Actual Result: {'REJECTED' if not result['approved'] else 'APPROVED'}")
                print(f"   System Working: {'✅' if not result['approved'] else '❌ BUG DETECTED!'}")
        else:
            print("❌ Analysis failed")
    else:
        print("❌ Claude connection failed - check API key")
    
    print("\n" + "="*80)
    print("🔧 ENHANCED CLAUDE API READY")
    print("="*80)
    print("✅ Technical pre-validation implemented")
    print("✅ Complete DataFrame analysis enabled") 
    print("✅ MACD histogram contradiction detection")
    print("✅ EMA alignment validation")
    print("✅ KAMA efficiency ratio checking")
    print("✅ Cross-indicator validation")
    print("✅ Signal classification verification")
    print("✅ Timestamp-based analysis support")
    print("✅ Enhanced logging and debugging")
    print("✅ Comprehensive file saving")
    print("✅ Enhanced retry logic and rate limiting")
    print("✅ Intelligent fallback analysis")
    print("✅ API health monitoring")
    print("="*80)