# core/strategies/helpers/kama_data_helper.py
"""
KAMA Data Helper Module - Extracted from KAMA Strategy
🔧 DATA: Data preparation, enhancement, and manipulation utilities for KAMA
📊 COMPREHENSIVE: KAMA indicators, semantic mapping, signal enhancement
🎯 FOCUSED: Single responsibility for KAMA data handling

This module contains all the data preparation and enhancement logic
for KAMA strategy that was previously embedded in the main strategy file.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging
from datetime import datetime
import hashlib
import json


class KAMADataHelper:
    """
    🔧 DATA: Comprehensive data preparation and enhancement for KAMA strategy
    
    Responsibilities:
    - KAMA indicator calculation and validation
    - Data validation and quality checks
    - Signal data enhancement for database storage
    - Timestamp safety and conversion
    - JSON serialization and data integrity
    """
    
    def __init__(self, logger: logging.Logger = None, forex_optimizer=None):
        self.logger = logger or logging.getLogger(__name__)
        self.forex_optimizer = forex_optimizer  # Will be injected by main strategy
        
        # Data processing statistics
        self._indicators_calculated = 0
        self._signals_enhanced = 0
        self._validation_failures = 0
        
        self.logger.info("🔧 KAMA Data Helper initialized")

    def validate_input_data(self, df: pd.DataFrame, epic: str, min_bars: int) -> bool:
        """
        ✅ Validate input data for KAMA processing
        """
        try:
            if df is None or len(df) < min_bars:
                self.logger.debug(f"Insufficient data for KAMA: {len(df) if df is not None else 0} < {min_bars}")
                self._validation_failures += 1
                return False
            
            # Check for required columns
            if 'close' not in df.columns:
                self.logger.debug("Missing 'close' column for KAMA calculation")
                self._validation_failures += 1
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            self._validation_failures += 1
            return False

    def ensure_kama_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔄 Ensure KAMA indicators are present in the DataFrame
        """
        try:
            # Check if KAMA indicators already exist
            kama_cols = [col for col in df.columns if 'kama' in col.lower()]
            er_cols = [col for col in df.columns if 'efficiency' in col.lower() or 'er' in col.lower()]
            
            if kama_cols and er_cols:
                self.logger.debug("✅ KAMA indicators already present")
                return df
            
            self.logger.info("🔄 Adding missing KAMA indicators")
            df_enhanced = self._calculate_kama_indicators(df.copy())
            self._indicators_calculated += 1
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Error ensuring KAMA indicators: {e}")
            return df

    def _calculate_kama_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📊 Calculate KAMA indicators with proper error handling
        """
        try:
            # Get KAMA parameters from config
            import config
            er_period = getattr(config, 'KAMA_ER_PERIOD', 14)
            fast_sc = getattr(config, 'KAMA_FAST_SC', 2)
            slow_sc = getattr(config, 'KAMA_SLOW_SC', 30)
            
            # Verify DataFrame structure
            if not isinstance(df, pd.DataFrame) or 'close' not in df.columns:
                self.logger.error("Invalid DataFrame for KAMA calculation")
                return df
            
            if len(df) < er_period + 10:
                self.logger.warning(f"⚠️ Insufficient data for KAMA: {len(df)} < {er_period + 10}")
                # Add empty columns to prevent errors
                df['kama'] = df['close']  # Fallback to close price
                df['efficiency_ratio'] = 0.1  # Default ER
                df['kama_slope'] = 0.0
                df['kama_acceleration'] = 0.0
                return df
            
            # Calculate Efficiency Ratio (ER)
            close_series = df['close']
            if not isinstance(close_series, pd.Series):
                close_series = pd.Series(close_series, index=df.index)
            
            # Calculate change over ER period
            change = abs(close_series - close_series.shift(er_period))
            
            # Calculate volatility (sum of absolute changes)
            daily_changes = close_series.diff().abs()
            volatility = daily_changes.rolling(window=er_period, min_periods=1).sum()
            
            # Calculate Efficiency Ratio
            efficiency_ratio = np.where(volatility != 0, change / volatility, 0.1)
            
            # Ensure efficiency_ratio is a pandas Series
            if isinstance(efficiency_ratio, np.ndarray):
                efficiency_ratio = pd.Series(efficiency_ratio, index=df.index)
            
            df['efficiency_ratio'] = efficiency_ratio
            
            # Calculate Smoothing Constant (SC)
            fastest_sc = 2.0 / (fast_sc + 1)
            slowest_sc = 2.0 / (slow_sc + 1)
            
            # SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
            sc = (efficiency_ratio * (fastest_sc - slowest_sc) + slowest_sc) ** 2
            
            # Calculate KAMA
            kama = pd.Series(index=df.index, dtype=float, name='kama')
            
            # Initialize first KAMA value
            start_idx = er_period
            if start_idx < len(df):
                kama.iloc[start_idx] = close_series.iloc[start_idx]
                
                # Calculate subsequent KAMA values
                for i in range(start_idx + 1, len(df)):
                    try:
                        current_sc = sc.iloc[i] if isinstance(sc, pd.Series) else sc[i]
                        current_close = close_series.iloc[i]
                        prev_kama = kama.iloc[i-1]
                        
                        # KAMA calculation: KAMA = prev_KAMA + SC * (Price - prev_KAMA)
                        kama.iloc[i] = prev_kama + current_sc * (current_close - prev_kama)
                        
                    except (IndexError, KeyError) as e:
                        # Use previous value or close price as fallback
                        if i > 0:
                            kama.iloc[i] = kama.iloc[i-1]
                        else:
                            kama.iloc[i] = current_close
            
            # Assign KAMA to DataFrame
            df['kama'] = kama
            
            # Calculate additional KAMA indicators
            df['kama_slope'] = df['kama'].diff().fillna(0)
            df['kama_acceleration'] = df['kama_slope'].diff().fillna(0)
            
            # Fill any remaining NaN values
            df['kama'] = df['kama'].ffill().fillna(df['close'])
            df['efficiency_ratio'] = df['efficiency_ratio'].fillna(0.1)
            df['kama_slope'] = df['kama_slope'].fillna(0)
            df['kama_acceleration'] = df['kama_acceleration'].fillna(0)
            
            self.logger.debug("✅ KAMA indicators calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating KAMA indicators: {e}")
            # Fallback: add dummy indicators to prevent crashes
            try:
                df['kama'] = df['close'] if 'close' in df.columns else 0
                df['efficiency_ratio'] = 0.1
                df['kama_slope'] = 0.0
                df['kama_acceleration'] = 0.0
                self.logger.warning("⚠️ Using fallback KAMA indicators")
            except Exception as fallback_error:
                self.logger.error(f"❌ Even fallback failed: {fallback_error}")
            
            return df

    def create_enhanced_signal_data(self, latest: pd.Series, signal_type: str) -> Dict:
        """
        🧠 Create enhanced signal data structure for validation
        """
        try:
            # Build structured data format for Enhanced Signal Validator
            signal_data = {
                # EMA data structure - use KAMA as proxy for EMAs if not available
                'ema_data': {
                    'ema_short': latest.get('ema_9', latest.get('ema_12', latest.get('kama', 0))),
                    'ema_long': latest.get('ema_21', latest.get('ema_26', latest.get('kama', 0))),
                    'ema_trend': latest.get('ema_200', latest.get('kama', 0)),
                    'ema_9': latest.get('ema_9', latest.get('kama', 0)),
                    'ema_21': latest.get('ema_21', latest.get('kama', 0)),
                    'ema_200': latest.get('ema_200', latest.get('kama', 0))
                },
                # MACD data structure - use available data or defaults
                'macd_data': {
                    'macd_line': latest.get('macd_line', 0),
                    'macd_signal': latest.get('macd_signal', 0),
                    'macd_histogram': latest.get('macd_histogram', 0)
                },
                # KAMA data structure - key for KAMA strategy
                'kama_data': {
                    'kama_value': latest.get('kama', latest.get('close', 0)),
                    'efficiency_ratio': latest.get('efficiency_ratio', 0.25),  # Safe default above 0.200 threshold
                    'kama_trend': 0  # Will be calculated
                }
            }
            
            # Add flat structure for backward compatibility
            signal_data.update({
                'signal_type': signal_type,
                'price': latest.get('close', 0),
                'ema_short': signal_data['ema_data']['ema_short'],
                'ema_long': signal_data['ema_data']['ema_long'], 
                'ema_trend': signal_data['ema_data']['ema_trend'],
                'macd_line': signal_data['macd_data']['macd_line'],
                'macd_signal': signal_data['macd_data']['macd_signal'],
                'macd_histogram': signal_data['macd_data']['macd_histogram'],
                'efficiency_ratio': signal_data['kama_data']['efficiency_ratio'],
                'rsi': latest.get('rsi', 50.0),
                'volume': latest.get('ltv', latest.get('volume', 0)),
                'volume_ratio': latest.get('volume_ratio_20', 1.0),
                'volume_confirmation': latest.get('volume_ratio_20', 1.0) > 1.2,
                'atr': latest.get('atr', 0.001),
                'bb_upper': latest.get('bb_upper', 0.0),
                'bb_middle': latest.get('bb_middle', latest.get('close', 0)),
                'bb_lower': latest.get('bb_lower', 0.0),
                
                # KAMA-specific fields
                'kama_value': signal_data['kama_data']['kama_value'],
                'kama_trend': signal_data['kama_data']['kama_trend']
            })
            
            self.logger.debug(f"Enhanced signal data created with KAMA: value={signal_data['kama_value']:.5f}, efficiency={signal_data['efficiency_ratio']:.3f}")
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Enhanced signal data creation error: {e}")
            return {
                'ema_data': {'ema_short': 0, 'ema_long': 0, 'ema_trend': 0},
                'macd_data': {'macd_line': 0, 'macd_signal': 0, 'macd_histogram': 0},
                'kama_data': {'kama_value': 0, 'efficiency_ratio': 0.25, 'kama_trend': 0}
            }

    def build_complete_signal(
        self, 
        signal_data: Dict, 
        enhanced_signal_data: Dict, 
        adjusted_price: float, 
        confidence: float,
        epic: str, 
        timeframe: str, 
        spread_pips: float
    ) -> Dict:
        """
        🏗️ Build complete signal with all required fields
        """
        try:
            self._signals_enhanced += 1
            
            # Start with signal data
            complete_signal = signal_data.copy()
            
            # Add core fields
            complete_signal.update({
                'signal_price': adjusted_price,
                'confidence_score': confidence,
                'timestamp': datetime.now(),
                'strategy': 'kama',
                'epic': epic,
                'timeframe': timeframe,
                'spread_pips': spread_pips,
                'original_price': signal_data.get('current_price', 0),
                'enhanced_validation': True,
                'performance_optimized': True
            })
            
            # Enhance with complete database population
            enhanced_signal = self._enhance_signal_for_complete_database_save(
                complete_signal, enhanced_signal_data, spread_pips
            )
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Complete signal building error: {e}")
            return signal_data

    def _enhance_signal_for_complete_database_save(self, signal: Dict, enhanced_data: Dict, spread_pips: float) -> Dict:
        """
        🚀 Complete signal enhancement for alert_history table
        """
        try:
            current_price = signal.get('signal_price', enhanced_data.get('price', 0))
            signal_type = signal.get('signal_type')
            epic = signal.get('epic', '')
            timeframe = signal.get('timeframe', '15m')
            confidence = signal.get('confidence_score', 0.0)
            
            # Start with the original signal
            enhanced = signal.copy()
            
            # Extract pair from epic
            if not enhanced.get('pair'):
                enhanced['pair'] = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            
            # Timestamp safety
            timestamp_fields = ['market_timestamp', 'timestamp', 'signal_timestamp', 'candle_timestamp']
            for field in timestamp_fields:
                if field in enhanced and enhanced[field]:
                    enhanced[field] = self._convert_market_timestamp_safe(enhanced[field])
            
            # Core price fields
            enhanced.update({
                'entry_price': current_price,
                'current_price': current_price,
                'price': current_price,
                'close_price': current_price
            })
            
            # Calculate stop loss and take profit
            import config
            default_stop_distance = getattr(config, 'DEFAULT_STOP_DISTANCE', 20)
            pip_size = 0.01 if 'JPY' in epic else 0.0001
            stop_distance_price = default_stop_distance * pip_size
            risk_reward = getattr(config, 'DEFAULT_RISK_REWARD', 2.0)
            
            if signal_type in ['BULL', 'BUY']:
                enhanced['stop_loss'] = current_price - stop_distance_price
                enhanced['take_profit'] = current_price + (stop_distance_price * risk_reward)
            else:
                enhanced['stop_loss'] = current_price + stop_distance_price
                enhanced['take_profit'] = current_price - (stop_distance_price * risk_reward)
            
            # Technical indicator fields
            enhanced.update({
                'kama': enhanced_data.get('kama_value', 0),
                'kama_value': enhanced_data.get('kama_value', 0),
                'efficiency_ratio': enhanced_data.get('efficiency_ratio', 0.1),
                'kama_slope': enhanced_data.get('kama_trend', 0),
                'ema_short': enhanced_data.get('ema_short', 0),
                'ema_long': enhanced_data.get('ema_long', 0),
                'ema_trend': enhanced_data.get('ema_trend', 0),
                'rsi': enhanced_data.get('rsi', 50.0),
                'macd_line': enhanced_data.get('macd_line', 0.0),
                'macd_signal': enhanced_data.get('macd_signal', 0.0),
                'macd_histogram': enhanced_data.get('macd_histogram', 0.0),
                'atr': enhanced_data.get('atr', 0.001),
            })
            
            # Volume fields
            enhanced.update({
                'volume': enhanced_data.get('volume', 0),
                'volume_ratio': enhanced_data.get('volume_ratio', 1.0),
                'volume_confirmation': enhanced_data.get('volume_confirmation', False),
            })
            
            # Market context fields
            enhanced.update({
                'market_regime': self._determine_market_regime_for_db(enhanced),
                'volatility': self._assess_volatility(enhanced_data),
                'trend_strength': self._calculate_kama_trend_strength(enhanced),
                'trading_session': self._determine_trading_session(),
                'market_hours': self._is_market_hours(),
            })
            
            # Strategy metadata
            enhanced.update({
                'strategy_config': {
                    'er_period': getattr(config, 'KAMA_ER_PERIOD', 14),
                    'fast_sc': getattr(config, 'KAMA_FAST_SC', 2),
                    'slow_sc': getattr(config, 'KAMA_SLOW_SC', 30),
                    'spread_pips': spread_pips,
                    'timeframe': timeframe,
                    'enhanced_validation_enabled': True,
                    'performance_caching_enabled': True,
                },
                'strategy_metadata': {
                    'detection_method': 'kama_adaptive_moving_average_modular',
                    'signal_strength': enhanced.get('signal_strength', 0.5),
                    'quality_score': confidence,
                    'enhancement_version': '2.0_modular_enhanced',
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            # Alert fields
            enhanced.update({
                'alert_message': f"{signal_type} KAMA signal for {epic} @ {confidence:.1%} (Modular KAMA Strategy)",
                'alert_level': 'HIGH' if confidence > 0.9 else 'MEDIUM' if confidence > 0.8 else 'INFO',
                'status': 'NEW',
                'processed': False,
            })
            
            # Ensure JSON serializable
            enhanced = self._ensure_json_serializable(enhanced)
            
            self.logger.debug(f"✅ Complete KAMA signal enhancement applied to {epic} - {len(enhanced)} fields populated")
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"❌ Error in complete signal enhancement: {e}")
            return signal

    def _convert_market_timestamp_safe(self, timestamp_value) -> Optional[datetime]:
        """Safe timestamp conversion"""
        if timestamp_value is None:
            return None
            
        try:
            if isinstance(timestamp_value, datetime):
                return timestamp_value
            elif isinstance(timestamp_value, str):
                if 'T' in timestamp_value:
                    return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                else:
                    return datetime.fromisoformat(timestamp_value)
            elif isinstance(timestamp_value, (int, float)):
                if 0 <= timestamp_value <= 4102444800:  # 2100-01-01
                    return datetime.fromtimestamp(timestamp_value)
                else:
                    return None
            elif hasattr(timestamp_value, 'to_pydatetime'):
                return timestamp_value.to_pydatetime()
            else:
                return None
        except Exception as e:
            self.logger.warning(f"⚠️ Timestamp conversion error: {e}")
            return None

    def _determine_market_regime_for_db(self, enhanced: Dict) -> str:
        """Determine market regime for database storage"""
        try:
            efficiency_ratio = enhanced.get('efficiency_ratio', 0)
            kama_slope = enhanced.get('kama_slope', 0)
            
            if efficiency_ratio > 0.6 and abs(kama_slope) > 0.002:
                return 'strong_trending'
            elif efficiency_ratio > 0.3 and abs(kama_slope) > 0.001:
                return 'trending'
            elif efficiency_ratio < 0.2 and abs(kama_slope) < 0.0005:
                return 'consolidating'
            else:
                return 'ranging'
        except:
            return 'unknown'

    def _assess_volatility(self, data: Dict) -> str:
        """Assess market volatility"""
        try:
            atr = data.get('atr', 0.001)
            if atr > 0.01:
                return 'high'
            elif atr > 0.005:
                return 'medium'
            else:
                return 'low'
        except:
            return 'unknown'

    def _calculate_kama_trend_strength(self, enhanced: Dict) -> str:
        """Calculate trend strength based on KAMA parameters"""
        try:
            efficiency_ratio = enhanced.get('efficiency_ratio', 0)
            signal_strength = enhanced.get('signal_strength', 0)
            
            combined_strength = (efficiency_ratio + signal_strength) / 2
            
            if combined_strength > 0.7:
                return 'strong'
            elif combined_strength > 0.4:
                return 'medium'
            else:
                return 'weak'
        except:
            return 'unknown'

    def _determine_trading_session(self) -> str:
        """Determine current trading session"""
        try:
            import pytz
            london_tz = pytz.timezone('Europe/London')
            london_time = datetime.now(london_tz)
            hour = london_time.hour
            
            if 8 <= hour < 17:
                return 'london'
            elif 13 <= hour < 22:
                return 'new_york'
            elif 0 <= hour < 9:
                return 'sydney'
            else:
                return 'tokyo'
        except:
            return 'unknown'

    def _is_market_hours(self) -> bool:
        """Check if current time is during major market hours"""
        try:
            current_hour = datetime.now().hour
            return 1 <= current_hour <= 23
        except:
            return True

    def _ensure_json_serializable(self, signal: Dict) -> Dict:
        """Ensure all signal data is JSON serializable"""
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        return convert_for_json(signal)

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators for KAMA strategy"""
        return ['kama', 'efficiency_ratio', 'close', 'high', 'low']

    def get_semantic_indicators(self) -> List[str]:
        """Required indicators with semantic names"""
        return ['kama', 'efficiency_ratio', 'close']

    def debug_data_validation(self, df: pd.DataFrame, epic: str, min_bars: int) -> Dict:
        """
        🔍 Debug data validation process
        """
        try:
            debug_info = {
                'module': 'kama_data_helper',
                'epic': epic,
                'data_length': len(df) if df is not None else 0,
                'min_bars_required': min_bars,
                'validation_steps': [],
                'issues': []
            }
            
            # Basic validation
            if df is None:
                debug_info['issues'].append("DataFrame is None")
                return debug_info
            
            debug_info['validation_steps'].append(f"✅ DataFrame exists with {len(df)} rows")
            
            if len(df) < min_bars:
                debug_info['issues'].append(f"Insufficient data: {len(df)} < {min_bars}")
            else:
                debug_info['validation_steps'].append(f"✅ Sufficient data: {len(df)} >= {min_bars}")
            
            # Column validation
            if 'close' not in df.columns:
                debug_info['issues'].append("Missing 'close' column")
            else:
                debug_info['validation_steps'].append("✅ 'close' column present")
            
            # KAMA indicators check
            kama_cols = [col for col in df.columns if 'kama' in col.lower()]
            er_cols = [col for col in df.columns if 'efficiency' in col.lower() or 'er' in col.lower()]
            
            debug_info['existing_indicators'] = {
                'kama_columns': kama_cols,
                'efficiency_ratio_columns': er_cols
            }
            
            if not kama_cols:
                debug_info['validation_steps'].append("⚠️ KAMA indicators missing - will be calculated")
            else:
                debug_info['validation_steps'].append(f"✅ KAMA indicators present: {kama_cols}")
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}

    def get_data_stats(self) -> Dict:
        """📊 Get data processing statistics"""
        try:
            return {
                'module': 'kama_data_helper',
                'indicators_calculated': self._indicators_calculated,
                'signals_enhanced': self._signals_enhanced,
                'validation_failures': self._validation_failures,
                'success_rate': (self._indicators_calculated + self._signals_enhanced) / max(
                    self._indicators_calculated + self._signals_enhanced + self._validation_failures, 1
                ) * 100,
                'error': None
            }
        except Exception as e:
            return {'error': str(e)}

    def reset_stats(self):
        """🔄 Reset data processing statistics"""
        self._indicators_calculated = 0
        self._signals_enhanced = 0
        self._validation_failures = 0
        self.logger.debug("🔄 KAMA Data Helper statistics reset")