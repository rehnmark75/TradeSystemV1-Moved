# core/strategies/helpers/kama_signal_detector.py
"""
KAMA Signal Detector Module - Extracted from KAMA Strategy
🔍 DETECTION: Core signal detection algorithms for KAMA strategy
🎯 FOCUSED: Single responsibility for KAMA signal detection
📊 COMPREHENSIVE: Trend changes, crossovers, momentum analysis

This module contains all the core signal detection logic for KAMA strategy,
extracted for better maintainability and testability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime


class KAMASignalDetector:
    """
    🔍 DETECTION: Core signal detection algorithms for KAMA strategy
    
    Responsibilities:
    - KAMA trend change detection
    - Price-KAMA crossover detection
    - Signal strength analysis
    - Momentum assessment
    - MACD cross-validation
    """
    
    def __init__(self, logger: logging.Logger = None, forex_optimizer=None, validator=None):
        self.logger = logger or logging.getLogger(__name__)
        self.forex_optimizer = forex_optimizer  # Will be injected by main strategy
        self.validator = validator  # Will be injected by main strategy
        
        # Detection statistics
        self._signal_count = 0
        self._signal_types = {'BULL': 0, 'BEAR': 0}
        self._detection_reasons = {}
        
        self.logger.info("🔍 KAMA Signal Detector initialized")

    def detect_kama_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float = 1.5,
        timeframe: str = '15m'
    ) -> Optional[Dict]:
        """
        🔍 Main KAMA signal detection orchestrator
        """
        try:
            # Find KAMA columns
            kama_column, er_column = self._find_kama_columns(df)
            if not kama_column or not er_column:
                return None
            
            # Validate data
            if not self._validate_signal_data(df, kama_column, er_column):
                return None
            
            # Get recent data for signal detection
            recent_data = df.tail(10).copy()
            if len(recent_data) < 3:
                return None
            
            current_row = recent_data.iloc[-1]
            previous_row = recent_data.iloc[-2]
            
            # Extract values
            signal_values = self._extract_signal_values(current_row, previous_row, kama_column, er_column)
            if not signal_values:
                return None
            
            # Detect signal patterns
            signal_data = self._detect_signal_patterns(signal_values, epic)
            if not signal_data:
                return None
            
            # MACD cross-validation
            if not self._validate_macd_alignment(current_row, signal_data['signal_type']):
                return None

            # ADX trend strength validation (CRITICAL NEW ADDITION)
            if not self._validate_adx_trend_strength(current_row, signal_data['signal_type']):
                return None

            # Track detection statistics
            self._track_signal_detection(signal_data['signal_type'], signal_data['trigger_reason'])
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in KAMA signal detection: {e}")
            return None

    def _find_kama_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        🔍 Find KAMA and efficiency ratio columns in DataFrame
        """
        try:
            # Get KAMA ER period from config
            try:
                from configdata.strategies import config_kama_strategy
            except ImportError:
                try:
                    from forex_scanner.configdata.strategies import config_kama_strategy
                except ImportError:
                    from forex_scanner.configdata.strategies import config_kama_strategy as config_kama_strategy
            er_period = getattr(config_kama_strategy, 'KAMA_ER_PERIOD', 14)
            
            # Try different KAMA column names
            possible_kama_cols = [
                f'kama_{er_period}',  # e.g., kama_14
                'kama_10',            # Default fallback
                'kama_14',            # Common period
                'kama_6',             # Alternative
                'kama'                # Generic (if exists)
            ]
            
            possible_er_cols = [
                f'kama_{er_period}_er',  # e.g., kama_14_er
                'kama_10_er',            # Default fallback
                'kama_14_er',            # Common period
                'kama_6_er',             # Alternative
                'efficiency_ratio'       # Generic (if exists)
            ]
            
            # Find first available KAMA column
            kama_column = None
            for col in possible_kama_cols:
                if col in df.columns:
                    kama_column = col
                    break
            
            # Find corresponding ER column
            er_column = None
            for col in possible_er_cols:
                if col in df.columns:
                    er_column = col
                    break
            
            if not kama_column or not er_column:
                self.logger.debug(f"KAMA columns not found. Available: {list(df.columns)}")
                return None, None
            
            return kama_column, er_column
            
        except Exception as e:
            self.logger.error(f"Error finding KAMA columns: {e}")
            return None, None

    def _validate_signal_data(self, df: pd.DataFrame, kama_column: str, er_column: str) -> bool:
        """
        ✅ Validate data quality for signal detection
        """
        try:
            # Check required columns
            required_columns = ['close', kama_column, er_column]
            for col in required_columns:
                if col not in df.columns:
                    self.logger.debug(f"Required column missing: {col}")
                    return False
            
            # Check for excessive null values in recent data
            recent_data = df.tail(10)
            for col in required_columns:
                null_count = recent_data[col].isnull().sum()
                if null_count > 7:
                    self.logger.debug(f"Too many null values in {col}: {null_count}/10")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False

    def _extract_signal_values(self, current_row: pd.Series, previous_row: pd.Series, kama_column: str, er_column: str) -> Optional[Dict]:
        """
        📊 Extract values needed for signal detection
        """
        try:
            current_price = float(current_row['close'])
            previous_price = float(previous_row['close'])
            current_kama = float(current_row[kama_column])
            previous_kama = float(previous_row[kama_column])
            current_er = float(current_row[er_column])
            
            # Calculate KAMA trend and slope
            kama_change = current_kama - previous_kama
            kama_trend = kama_change / previous_kama if previous_kama != 0 else 0
            
            # Price position relative to KAMA
            price_above_kama = current_price > current_kama
            prev_price_above_kama = previous_price > previous_kama
            
            return {
                'current_price': current_price,
                'previous_price': previous_price,
                'current_kama': current_kama,
                'previous_kama': previous_kama,
                'current_er': current_er,
                'kama_trend': kama_trend,
                'price_above_kama': price_above_kama,
                'prev_price_above_kama': prev_price_above_kama,
                'kama_column_used': kama_column,
                'er_column_used': er_column
            }
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Value extraction error: {e}")
            return None

    def _detect_signal_patterns(self, values: Dict, epic: str) -> Optional[Dict]:
        """
        🎯 Detect KAMA signal patterns
        """
        try:
            # Get forex-specific thresholds
            if self.forex_optimizer:
                kama_thresholds = self.forex_optimizer.get_kama_thresholds_for_pair(epic)
            else:
                try:
                    from configdata.strategies import config_kama_strategy
                except ImportError:
                    try:
                        from forex_scanner.configdata.strategies import config_kama_strategy
                    except ImportError:
                        from forex_scanner.configdata.strategies import config_kama_strategy as config_kama_strategy
                kama_thresholds = {
                    'min_efficiency': getattr(config_kama_strategy, 'KAMA_MIN_EFFICIENCY', 0.1),
                    'trend_threshold': getattr(config_kama_strategy, 'KAMA_TREND_THRESHOLD', 0.05)
                }
            
            signal_type = None
            trigger_reason = None
            signal_strength = 0
            
            # 1. KAMA Trend Change Signals (primary signal type)
            if abs(values['kama_trend']) > kama_thresholds['trend_threshold'] and values['current_er'] > kama_thresholds['min_efficiency']:
                if values['kama_trend'] > 0:
                    # KAMA trending up
                    signal_type = 'BULL'
                    trigger_reason = f'KAMA trending up (slope: {values["kama_trend"]:.4f})'
                    signal_strength = min(abs(values['kama_trend']) * 1000, 0.8)
                elif values['kama_trend'] < 0:
                    # KAMA trending down  
                    signal_type = 'BEAR'
                    trigger_reason = f'KAMA trending down (slope: {values["kama_trend"]:.4f})'
                    signal_strength = min(abs(values['kama_trend']) * 1000, 0.8)
            
            # 2. Price-KAMA Crossover Signals (with efficiency confirmation)
            if not signal_type and values['current_er'] > 0.3:  # Higher efficiency required for crossovers
                if not values['prev_price_above_kama'] and values['price_above_kama']:
                    # Bullish crossover
                    signal_type = 'BULL'
                    trigger_reason = 'Price crossed above KAMA'
                    signal_strength = min(values['current_er'], 0.8)
                    
                elif values['prev_price_above_kama'] and not values['price_above_kama']:
                    # Bearish crossover
                    signal_type = 'BEAR'
                    trigger_reason = 'Price crossed below KAMA'
                    signal_strength = min(values['current_er'], 0.8)
            
            if signal_type:
                # Calculate additional KAMA metrics
                kama_distance = abs(values['current_price'] - values['current_kama']) / values['current_kama']
                
                return {
                    'signal_type': signal_type,
                    'trigger_reason': trigger_reason,
                    'signal_strength': signal_strength,
                    'kama_value': values['current_kama'],
                    'efficiency_ratio': values['current_er'],
                    'kama_trend': values['kama_trend'],
                    'kama_distance': kama_distance,
                    'price_above_kama': values['price_above_kama'],
                    'raw_confidence': signal_strength * values['current_er'],
                    'kama_column_used': values['kama_column_used'],
                    'er_column_used': values['er_column_used'],
                    'current_price': values['current_price']
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal pattern detection error: {e}")
            return None

    def _validate_macd_alignment(self, current_row: pd.Series, signal_type: str) -> bool:
        """
        🔄 Validate signal against MACD to prevent contradictions
        """
        try:
            # Look for MACD columns
            macd_histogram_col = None
            macd_line_col = None
            macd_signal_col = None
            
            for col in current_row.index:
                col_lower = col.lower()
                if 'macd_histogram' in col_lower or col_lower == 'macd_hist':
                    macd_histogram_col = col
                elif 'macd_line' in col_lower or col_lower == 'macd':
                    macd_line_col = col
                elif 'macd_signal' in col_lower:
                    macd_signal_col = col
            
            # Perform MACD validation if data is available
            if macd_histogram_col and macd_histogram_col in current_row.index:
                try:
                    macd_histogram = float(current_row[macd_histogram_col])

                    # STRENGTHENED THRESHOLDS: Increased from 0.0001 to 0.0003 for forex
                    MIN_MACD_THRESHOLD = 0.0003  # Minimum threshold for contradiction
                    STRONG_MACD_THRESHOLD = 0.0005  # Strong confirmation threshold

                    # Check for major contradictions (STRICTER THAN BEFORE)
                    if signal_type == 'BULL' and macd_histogram < -MIN_MACD_THRESHOLD:
                        self.logger.warning(
                            f"🚫 KAMA BULL signal REJECTED: Negative MACD histogram "
                            f"({macd_histogram:.6f} < -{MIN_MACD_THRESHOLD:.6f})"
                        )
                        return False

                    elif signal_type == 'BEAR' and macd_histogram > MIN_MACD_THRESHOLD:
                        self.logger.warning(
                            f"🚫 KAMA BEAR signal REJECTED: Positive MACD histogram "
                            f"({macd_histogram:.6f} > {MIN_MACD_THRESHOLD:.6f})"
                        )
                        return False

                    # STRONG CONFIRMATION: Add bonus flag for strong MACD alignment
                    if signal_type == 'BULL' and macd_histogram > STRONG_MACD_THRESHOLD:
                        self.logger.info(
                            f"✅ KAMA BULL signal STRONG MACD confirmation "
                            f"({macd_histogram:.6f} > {STRONG_MACD_THRESHOLD:.6f})"
                        )
                        # This can be used later for confidence boost
                    elif signal_type == 'BEAR' and macd_histogram < -STRONG_MACD_THRESHOLD:
                        self.logger.info(
                            f"✅ KAMA BEAR signal STRONG MACD confirmation "
                            f"({macd_histogram:.6f} < -{STRONG_MACD_THRESHOLD:.6f})"
                        )
                        # This can be used later for confidence boost
                    
                    # Log validation status
                    if signal_type == 'BULL' and macd_histogram >= 0:
                        self.logger.debug(f"✅ KAMA BULL signal validated: MACD histogram positive/neutral ({macd_histogram:.6f})")
                    elif signal_type == 'BEAR' and macd_histogram <= 0:
                        self.logger.debug(f"✅ KAMA BEAR signal validated: MACD histogram negative/neutral ({macd_histogram:.6f})")
                    else:
                        self.logger.debug(f"⚠️ KAMA {signal_type} signal: Minor MACD conflict ({macd_histogram:.6f}), but within tolerance")
                        
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"⚠️ Could not validate MACD histogram: {e}")
            
            # Additional MACD line vs signal validation if available
            if macd_line_col and macd_signal_col and macd_line_col in current_row.index and macd_signal_col in current_row.index:
                try:
                    macd_line = float(current_row[macd_line_col])
                    macd_signal = float(current_row[macd_signal_col])
                    
                    # Check for major MACD line/signal contradictions
                    if signal_type == 'BULL' and macd_line < macd_signal and abs(macd_line - macd_signal) > 0.0002:
                        self.logger.warning(f"🚫 KAMA BULL signal rejected: MACD line well below signal line")
                        return False
                        
                    elif signal_type == 'BEAR' and macd_line > macd_signal and abs(macd_line - macd_signal) > 0.0002:
                        self.logger.warning(f"🚫 KAMA BEAR signal rejected: MACD line well above signal line")
                        return False
                        
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"⚠️ Could not validate MACD line/signal: {e}")
            
            return True  # No contradictions found

        except Exception as e:
            self.logger.debug(f"MACD validation error: {e}")
            return True  # Don't reject on validation errors

    def _validate_adx_trend_strength(self, current_row: pd.Series, signal_type: str) -> bool:
        """
        🎯 ADX TREND STRENGTH VALIDATION: Ensure signals align with strong enough trends

        CRITICAL ADDITION: KAMA previously lacked ADX validation (unlike EMA strategy)
        This prevents signals in weak trends that are likely to reverse.

        Args:
            current_row: Current market data row
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if trend strength is adequate, False otherwise
        """
        try:
            # Look for ADX column
            adx_col = None
            for col in current_row.index:
                col_lower = col.lower()
                if col_lower == 'adx' or 'adx' in col_lower:
                    adx_col = col
                    break

            # If ADX not available, don't reject (graceful degradation)
            if not adx_col:
                self.logger.debug("ADX column not found - skipping ADX validation")
                return True

            try:
                adx_value = float(current_row[adx_col])

                # ADX thresholds for KAMA signals (forex-optimized)
                MIN_ADX = 20  # Minimum trend strength required
                STRONG_ADX = 25  # Strong trend confirmation
                WEAK_ADX_WARNING = 18  # Warning threshold

                # REJECTION: ADX too weak (< 20)
                if adx_value < MIN_ADX:
                    self.logger.warning(
                        f"🚫 KAMA {signal_type} signal REJECTED: ADX too weak "
                        f"({adx_value:.1f} < {MIN_ADX}) - Trend not established"
                    )
                    return False

                # STRONG CONFIRMATION: ADX > 25
                if adx_value >= STRONG_ADX:
                    self.logger.info(
                        f"✅ KAMA {signal_type} signal STRONG ADX confirmation "
                        f"({adx_value:.1f} >= {STRONG_ADX}) - Excellent trend strength"
                    )
                    return True

                # ACCEPTABLE: ADX 20-25 (moderate trend)
                elif adx_value >= MIN_ADX:
                    self.logger.debug(
                        f"✅ KAMA {signal_type} signal validated: ADX acceptable "
                        f"({adx_value:.1f} >= {MIN_ADX}) - Moderate trend"
                    )
                    return True

                # WARNING: ADX 18-20 (marginal)
                elif adx_value >= WEAK_ADX_WARNING:
                    self.logger.warning(
                        f"⚠️ KAMA {signal_type} signal: ADX marginal "
                        f"({adx_value:.1f}) - Weak trend, proceed with caution"
                    )
                    return True  # Allow but warn

            except (ValueError, TypeError) as e:
                self.logger.debug(f"⚠️ Could not parse ADX value: {e}")
                return True  # Don't reject on parsing errors

        except Exception as e:
            self.logger.debug(f"ADX validation error: {e}")
            return True  # Don't reject on validation errors (graceful degradation)

    def detect_kama_trend_change(
        self, 
        df: pd.DataFrame, 
        current_data: pd.Series, 
        prev_data: pd.Series
    ) -> Optional[Dict]:
        """
        📈 Detect KAMA trend change signals (legacy method for compatibility)
        """
        try:
            # Find KAMA columns
            kama_column, er_column = self._find_kama_columns(df)
            if not kama_column or not er_column:
                return None
            
            # Extract values
            values = self._extract_signal_values(current_data, prev_data, kama_column, er_column)
            if not values:
                return None
            
            # Detect patterns
            return self._detect_signal_patterns(values, 'default')
            
        except Exception as e:
            self.logger.error(f"Legacy trend change detection error: {e}")
            return None

    def analyze_kama_momentum_strength(self, signal_data: Dict, df: pd.DataFrame = None) -> float:
        """
        💪 Analyze momentum strength of KAMA signal
        """
        try:
            efficiency_ratio = signal_data.get('efficiency_ratio', 0)
            kama_trend = abs(signal_data.get('kama_trend', 0))
            signal_strength = signal_data.get('signal_strength', 0)
            
            # Base momentum score
            momentum_score = (efficiency_ratio * 0.4) + (kama_trend * 500) + (signal_strength * 0.6)
            
            # Additional momentum factors
            if df is not None and len(df) >= 3:
                # Check for momentum persistence
                recent_kama = df['kama'].tail(3) if 'kama' in df.columns else None
                if recent_kama is not None and len(recent_kama) == 3:
                    kama_direction = recent_kama.iloc[-1] - recent_kama.iloc[0]
                    if abs(kama_direction) > 0.001:  # Consistent direction
                        momentum_score += 0.1
            
            return max(0.0, min(1.0, momentum_score))
            
        except Exception as e:
            self.logger.debug(f"Momentum analysis error: {e}")
            return 0.5

    def _track_signal_detection(self, signal_type: str, trigger_reason: str):
        """
        📊 Track signal detection statistics
        """
        try:
            self._signal_count += 1
            self._signal_types[signal_type] = self._signal_types.get(signal_type, 0) + 1
            self._detection_reasons[trigger_reason] = self._detection_reasons.get(trigger_reason, 0) + 1
        except Exception as e:
            self.logger.debug(f"Signal tracking error: {e}")

    def get_detection_stats(self) -> Dict:
        """
        📊 Get signal detection statistics
        """
        try:
            return {
                'module': 'kama_signal_detector',
                'total_signals': self._signal_count,
                'signal_types': dict(self._signal_types),
                'detection_reasons': dict(self._detection_reasons),
                'bull_percentage': self._signal_types.get('BULL', 0) / max(self._signal_count, 1) * 100,
                'bear_percentage': self._signal_types.get('BEAR', 0) / max(self._signal_count, 1) * 100,
                'error': None
            }
        except Exception as e:
            return {'error': str(e)}

    def reset_stats(self):
        """
        🔄 Reset detection statistics
        """
        self._signal_count = 0
        self._signal_types = {'BULL': 0, 'BEAR': 0}
        self._detection_reasons.clear()
        self.logger.debug("🔄 KAMA Signal Detector statistics reset")

    def debug_signal_detection(self, df: pd.DataFrame, epic: str, spread_pips: float = 1.5, timeframe: str = '15m') -> Dict:
        """
        🔍 Debug signal detection process
        """
        try:
            debug_info = {
                'module': 'kama_signal_detector',
                'epic': epic,
                'timeframe': timeframe,
                'data_length': len(df),
                'detection_steps': [],
                'rejection_reasons': []
            }
            
            # Find KAMA columns
            kama_column, er_column = self._find_kama_columns(df)
            debug_info['kama_columns'] = {'kama': kama_column, 'efficiency_ratio': er_column}
            
            if not kama_column or not er_column:
                debug_info['rejection_reasons'].append("KAMA columns not found")
                return debug_info
            
            debug_info['detection_steps'].append(f"✅ Found KAMA columns: {kama_column}, {er_column}")
            
            # Validate data
            if not self._validate_signal_data(df, kama_column, er_column):
                debug_info['rejection_reasons'].append("Data validation failed")
                return debug_info
            
            debug_info['detection_steps'].append("✅ Data validation passed")
            
            # Try signal detection
            if len(df) >= 3:
                recent_data = df.tail(10)
                current_row = recent_data.iloc[-1]
                previous_row = recent_data.iloc[-2]
                
                # Extract values
                values = self._extract_signal_values(current_row, previous_row, kama_column, er_column)
                debug_info['signal_values'] = values
                
                if values:
                    debug_info['detection_steps'].append("✅ Signal values extracted")
                    
                    # Detect patterns
                    signal_data = self._detect_signal_patterns(values, epic)
                    debug_info['signal_pattern'] = signal_data
                    
                    if signal_data:
                        debug_info['detection_steps'].append(f"✅ Signal detected: {signal_data['signal_type']}")
                        
                        # MACD validation
                        macd_valid = self._validate_macd_alignment(current_row, signal_data['signal_type'])
                        debug_info['macd_validation'] = macd_valid
                        
                        if macd_valid:
                            debug_info['detection_steps'].append("✅ MACD validation passed")
                        else:
                            debug_info['rejection_reasons'].append("MACD validation failed")
                    else:
                        debug_info['rejection_reasons'].append("No signal pattern detected")
                else:
                    debug_info['rejection_reasons'].append("Signal value extraction failed")
            else:
                debug_info['rejection_reasons'].append("Insufficient data for signal detection")
            
            return debug_info
            
        except Exception as e:
            debug_info['error'] = str(e)
            debug_info['rejection_reasons'].append(f"Exception: {e}")
            return debug_info