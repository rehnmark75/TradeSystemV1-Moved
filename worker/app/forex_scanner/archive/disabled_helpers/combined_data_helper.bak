# core/strategies/helpers/combined_data_helper.py
"""
Combined Strategy Data Helper - MODULAR HELPER
🔥 DATA PROCESSING: Comprehensive data preparation and enhancement
🏗️ MODULAR: Focused on data handling for combined strategy
🎯 MAINTAINABLE: Single responsibility - data processing only
⚡ PERFORMANCE: Efficient data enhancement with caching
🛡️ SAFETY: Safe timestamp conversion and data validation
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np
import json
import hashlib


class CombinedDataHelper:
    """
    🔥 DATA HELPER: Comprehensive data processing for combined strategy
    
    Handles:
    - Data preparation and enhancement
    - Indicator calculation and validation
    - Signal enrichment and metadata addition
    - Safe timestamp conversion
    - JSON serialization
    """
    
    def __init__(self, logger: logging.Logger, forex_optimizer=None):
        self.logger = logger
        self.forex_optimizer = forex_optimizer
        
        # Data processing statistics
        self.processing_stats = {
            'total_enhancements': 0,
            'successful_enhancements': 0,
            'timestamp_conversions': 0,
            'indicator_additions': 0,
            'json_serializations': 0,
            'data_validations': 0
        }
        
        # Configuration
        self.helper_config = {
            'enable_comprehensive_enhancement': True,
            'enable_timestamp_safety': True,
            'enable_json_validation': True,
            'max_enhancement_attempts': 3,
            'default_risk_reward_ratio': 2.0,
            'default_stop_distance_pips': 20
        }
        
        # Cache for expensive calculations
        self._cache = {}
        
        self.logger.debug("✅ CombinedDataHelper initialized")

    def ensure_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required indicators are present in the DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Enhanced DataFrame with all indicators
        """
        try:
            if not isinstance(df, pd.DataFrame):
                self.logger.error("❌ Input to ensure_all_indicators is not a DataFrame")
                return df
            
            df_enhanced = df.copy()
            self.processing_stats['total_enhancements'] += 1
            
            # Add EMA indicators if missing
            df_enhanced = self._ensure_ema_indicators(df_enhanced)
            
            # Add MACD indicators if missing
            df_enhanced = self._ensure_macd_indicators(df_enhanced)
            
            # Add volume indicators if missing
            df_enhanced = self._ensure_volume_indicators(df_enhanced)
            
            # Add other technical indicators
            df_enhanced = self._ensure_additional_indicators(df_enhanced)
            
            self.processing_stats['successful_enhancements'] += 1
            self.processing_stats['indicator_additions'] += 1
            
            self.logger.debug(f"✅ Enhanced DataFrame: {len(df_enhanced.columns)} columns, {len(df_enhanced)} rows")
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"❌ Data enhancement failed: {e}")
            return df

    def _ensure_ema_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure EMA indicators are present"""
        try:
            ema_periods = [9, 21, 50, 200]
            missing_emas = []
            
            for period in ema_periods:
                col_name = f'ema_{period}'
                if col_name not in df.columns:
                    missing_emas.append(period)
            
            if missing_emas:
                self.logger.debug(f"Adding missing EMA indicators: {missing_emas}")
                for period in missing_emas:
                    col_name = f'ema_{period}'
                    df[col_name] = df['close'].ewm(span=period).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ EMA indicator addition failed: {e}")
            return df

    def _ensure_macd_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure MACD indicators are present"""
        try:
            macd_indicators = ['macd_line', 'macd_signal', 'macd_histogram']
            missing_macd = [col for col in macd_indicators if col not in df.columns]
            
            if missing_macd:
                self.logger.debug(f"Adding missing MACD indicators: {missing_macd}")
                
                # Calculate MACD
                ema_12 = df['close'].ewm(span=12).mean()
                ema_26 = df['close'].ewm(span=26).mean()
                macd_line = ema_12 - ema_26
                macd_signal = macd_line.ewm(span=9).mean()
                macd_histogram = macd_line - macd_signal
                
                df['macd_line'] = macd_line
                df['macd_signal'] = macd_signal
                df['macd_histogram'] = macd_histogram
                
                # Also add as 'macd' for compatibility
                if 'macd' not in df.columns:
                    df['macd'] = macd_line
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ MACD indicator addition failed: {e}")
            return df

    def _ensure_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure volume indicators are present"""
        try:
            # Check if volume data exists
            volume_col = None
            for col in ['volume', 'ltv', 'tick_volume']:
                if col in df.columns:
                    volume_col = col
                    break
            
            if volume_col:
                # Add volume SMA if missing
                if 'volume_sma' not in df.columns:
                    df['volume_sma'] = df[volume_col].rolling(window=20).mean()
                
                # Standardize volume column name
                if 'volume' not in df.columns:
                    df['volume'] = df[volume_col]
            else:
                # Create dummy volume data if none exists
                self.logger.warning("⚠️ No volume data found, creating dummy volume")
                df['volume'] = 1000.0  # Dummy volume
                df['volume_sma'] = 1000.0
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Volume indicator addition failed: {e}")
            return df

    def _ensure_additional_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure additional technical indicators are present"""
        try:
            # Add ATR if missing
            if 'atr' not in df.columns:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                df['atr'] = true_range.rolling(window=14).mean()
            
            # Add RSI if missing (simplified calculation)
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Additional indicator addition failed: {e}")
            return df

    def enhance_signal_comprehensive(
        self, 
        signal: Dict, 
        latest_data: pd.Series, 
        previous_data: Optional[pd.Series] = None
    ) -> Dict:
        """
        Apply comprehensive enhancement to combined strategy signal
        
        Args:
            signal: Signal dictionary to enhance
            latest_data: Latest market data
            previous_data: Previous market data (optional)
            
        Returns:
            Enhanced signal dictionary
        """
        try:
            self.processing_stats['total_enhancements'] += 1
            
            enhanced_signal = signal.copy()
            
            # Apply timestamp safety conversion
            enhanced_signal = self._apply_timestamp_safety(enhanced_signal)
            
            # Add comprehensive technical data
            enhanced_signal = self._add_comprehensive_technical_data(enhanced_signal, latest_data)
            
            # Add strategy configuration metadata
            enhanced_signal = self._add_strategy_configuration(enhanced_signal)
            
            # Add market context and conditions
            enhanced_signal = self._add_market_context(enhanced_signal, latest_data, previous_data)
            
            # Add risk management and execution data
            enhanced_signal = self._add_risk_management_data(enhanced_signal, latest_data)
            
            # Add deduplication and tracking metadata
            enhanced_signal = self._add_deduplication_metadata(enhanced_signal)
            
            # Ensure JSON serializable
            enhanced_signal = self._ensure_json_serializable(enhanced_signal)
            
            self.processing_stats['successful_enhancements'] += 1
            self.logger.debug(f"✅ Comprehensive enhancement applied: {len(enhanced_signal)} fields")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive enhancement failed: {e}")
            return signal

    def _apply_timestamp_safety(self, signal: Dict) -> Dict:
        """Apply safe timestamp conversion to all timestamp fields"""
        try:
            timestamp_fields = [
                'market_timestamp', 'timestamp', 'signal_timestamp', 
                'candle_timestamp', 'alert_timestamp', 'processing_timestamp'
            ]
            
            for field in timestamp_fields:
                if field in signal:
                    original_value = signal[field]
                    safe_timestamp = self._convert_market_timestamp_safe(original_value)
                    
                    if original_value != safe_timestamp:
                        self.logger.debug(f"🛠️ TIMESTAMP FIX: {field} converted from {original_value} to {safe_timestamp}")
                        self.processing_stats['timestamp_conversions'] += 1
                    
                    signal[field] = safe_timestamp
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Timestamp safety application failed: {e}")
            return signal

    def _convert_market_timestamp_safe(self, timestamp_value) -> Optional[datetime]:
        """
        TIMESTAMP FIX: Safely convert various timestamp formats to datetime object
        """
        if timestamp_value is None:
            return None
            
        try:
            # Case 1: Already a datetime object
            if isinstance(timestamp_value, datetime):
                return timestamp_value
                
            # Case 2: String timestamp (ISO format)
            if isinstance(timestamp_value, str):
                if 'T' in timestamp_value:
                    return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                else:
                    return datetime.fromisoformat(timestamp_value)
                    
            # Case 3: Integer or float (Unix timestamp)
            if isinstance(timestamp_value, (int, float)):
                if 0 <= timestamp_value <= 4102444800:  # Valid range
                    return datetime.fromtimestamp(timestamp_value)
                else:
                    self.logger.warning(f"⚠️ Invalid timestamp integer {timestamp_value}")
                    return None
                    
            # Case 4: Pandas timestamp
            if hasattr(timestamp_value, 'to_pydatetime'):
                return timestamp_value.to_pydatetime()
                
            # Case 5: Unknown type
            self.logger.warning(f"⚠️ Unknown timestamp type {type(timestamp_value)}")
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Timestamp conversion error: {e}")
            return None

    def _add_comprehensive_technical_data(self, signal: Dict, latest_data: pd.Series) -> Dict:
        """Add comprehensive technical indicator data"""
        try:
            current_price = signal.get('signal_price', signal.get('price', latest_data.get('close', 0)))
            
            # EMA data with guaranteed EMA 200
            ema_200_value = float(latest_data.get('ema_200', 0))
            if ema_200_value == 0:
                # Try alternative sources
                ema_200_value = float(latest_data.get('ema_trend', 0))
            
            signal.update({
                'ema_short': float(latest_data.get('ema_9', 0)),
                'ema_long': float(latest_data.get('ema_21', 0)),
                'ema_trend': float(latest_data.get('ema_200', 0)),
                'ema_9': float(latest_data.get('ema_9', 0)),
                'ema_21': float(latest_data.get('ema_21', 0)),
                'ema_50': float(latest_data.get('ema_50', 0)),
                'ema_200': ema_200_value,
                'ema_200_current': ema_200_value
            })
            
            # MACD data
            signal.update({
                'macd_line': float(latest_data.get('macd_line', latest_data.get('macd', 0))),
                'macd_signal': float(latest_data.get('macd_signal', 0)),
                'macd_histogram': float(latest_data.get('macd_histogram', 0))
            })
            
            # Volume data
            volume = latest_data.get('volume', latest_data.get('ltv', 0))
            volume_sma = latest_data.get('volume_sma', 1.0)
            
            signal['volume'] = float(volume) if volume else 0.0
            
            if volume_sma and volume_sma > 0:
                signal['volume_ratio'] = signal['volume'] / volume_sma
                signal['volume_confirmation'] = signal['volume_ratio'] > 1.2
            else:
                signal['volume_ratio'] = 1.0
                signal['volume_confirmation'] = False
            
            # Additional technical indicators
            signal.update({
                'atr': float(latest_data.get('atr', 0.001)),
                'rsi': float(latest_data.get('rsi', 50)),
                'current_price': current_price,
                'high_price': float(latest_data.get('high', current_price)),
                'low_price': float(latest_data.get('low', current_price)),
                'open_price': float(latest_data.get('open', current_price)),
                'close_price': current_price
            })
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Technical data addition failed: {e}")
            return signal

    def _add_strategy_configuration(self, signal: Dict) -> Dict:
        """Add strategy configuration metadata"""
        try:
            contributing_strategies = signal.get('contributing_strategies', [])
            individual_confidences = signal.get('individual_confidences', {})
            combination_mode = signal.get('combination_mode', 'consensus')
            
            # Get configuration from forex optimizer if available
            if self.forex_optimizer:
                config_data = self.forex_optimizer.get_combined_strategy_config()
                weights = self.forex_optimizer.get_normalized_strategy_weights()
            else:
                config_data = {'mode': combination_mode}
                weights = {}
            
            signal['strategy_config'] = {
                'strategy_type': 'combined_strategy_modular',
                'strategy_family': 'ensemble',
                'combination_mode': combination_mode,
                'contributing_strategies': contributing_strategies,
                'strategy_weights': weights,
                'consensus_threshold': config_data.get('consensus_threshold', 0.7),
                'min_combined_confidence': config_data.get('min_combined_confidence', 0.75),
                'total_strategies_attempted': len(individual_confidences),
                'active_strategies_count': len(contributing_strategies),
                'modular_architecture': True,
                'timestamp_safety_enabled': True
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Strategy configuration addition failed: {e}")
            return signal

    def _add_market_context(self, signal: Dict, latest_data: pd.Series, previous_data: Optional[pd.Series]) -> Dict:
        """Add market context and conditions"""
        try:
            current_price = signal.get('current_price', latest_data.get('close', 0))
            
            # Market trend analysis
            ema_200 = signal.get('ema_200', 0)
            if ema_200 > 0:
                if current_price > ema_200 * 1.01:
                    market_trend = 'bullish'
                elif current_price < ema_200 * 0.99:
                    market_trend = 'bearish'
                else:
                    market_trend = 'neutral'
            else:
                market_trend = 'unknown'
            
            # Volatility assessment
            atr = signal.get('atr', 0.001)
            if atr > 0:
                atr_pct = atr / current_price * 100
                if atr_pct > 2.0:
                    volatility = 'high'
                elif atr_pct > 1.0:
                    volatility = 'medium'
                else:
                    volatility = 'low'
            else:
                volatility = 'unknown'
            
            # Trading session
            trading_session = self._determine_trading_session()
            
            # Market regime
            market_regime = self._determine_market_regime(signal, latest_data)
            
            signal['signal_conditions'] = {
                'market_trend': market_trend,
                'signal_type': f'{signal.get("combination_mode", "unknown")}_combined',
                'ensemble_agreement': len(signal.get('contributing_strategies', [])),
                'momentum_direction': signal.get('signal_type', 'neutral').lower(),
                'volatility_assessment': volatility,
                'signal_timing': 'ensemble_consensus',
                'confirmation_level': 'high' if signal.get('confidence_score', 0) > 0.85 else 'medium',
                'market_session': trading_session,
                'strategy_diversity': len(set(signal.get('contributing_strategies', []))),
                'consensus_quality': self._assess_consensus_quality(signal),
                'timestamp_safety_processed': True
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Market context addition failed: {e}")
            return signal

    def _add_risk_management_data(self, signal: Dict, latest_data: pd.Series) -> Dict:
        """Add comprehensive risk management data"""
        try:
            current_price = signal.get('current_price', latest_data.get('close', 0))
            epic = signal.get('epic', '')
            signal_type = signal.get('signal_type', '')
            
            # Calculate pip size
            pip_size = 0.01 if 'JPY' in epic else 0.0001
            
            # Get risk parameters
            default_stop_distance = self.helper_config['default_stop_distance_pips']
            risk_reward = self.helper_config['default_risk_reward_ratio']
            
            # Add stop loss and take profit if not present
            if 'stop_loss' not in signal or 'take_profit' not in signal:
                stop_distance_price = default_stop_distance * pip_size
                
                if signal_type in ['BULL', 'BUY']:
                    signal['stop_loss'] = current_price - stop_distance_price
                    signal['take_profit'] = current_price + (stop_distance_price * risk_reward)
                elif signal_type in ['BEAR', 'SELL']:
                    signal['stop_loss'] = current_price + stop_distance_price
                    signal['take_profit'] = current_price - (stop_distance_price * risk_reward)
                else:
                    signal['stop_loss'] = current_price - stop_distance_price
                    signal['take_profit'] = current_price + (stop_distance_price * risk_reward)
            
            # Calculate comprehensive risk metrics
            stop_loss = signal.get('stop_loss', current_price)
            take_profit = signal.get('take_profit', current_price)
            
            # Risk/Reward calculation
            stop_distance = abs(current_price - stop_loss)
            target_distance = abs(take_profit - current_price)
            
            if stop_distance > 0:
                signal['risk_reward_ratio'] = target_distance / stop_distance
                signal['pip_risk'] = stop_distance / pip_size
                signal['max_risk_percentage'] = min(2.0, max(0.5, stop_distance / current_price * 100))
            else:
                signal['risk_reward_ratio'] = risk_reward
                signal['pip_risk'] = default_stop_distance
                signal['max_risk_percentage'] = 1.0
            
            # Add execution pricing
            spread_pips = signal.get('spread_pips', 1.5)
            spread_adjustment = spread_pips * pip_size
            
            signal.update({
                'spread_pips': spread_pips,
                'bid_price': current_price - spread_adjustment,
                'ask_price': current_price + spread_adjustment,
                'execution_price': current_price,
                'pip_size': pip_size,
                'entry_price': current_price,
                'mid_price': current_price,
                'risk_percent': min(2.0, signal['max_risk_percentage']),
                'position_size_suggestion': 'standard'
            })
            
            # Support/Resistance levels
            signal['nearest_support'] = current_price * 0.99  # Simplified calculation
            signal['nearest_resistance'] = current_price * 1.01
            signal['distance_to_support_pips'] = (current_price * 0.01) / pip_size
            signal['distance_to_resistance_pips'] = (current_price * 0.01) / pip_size
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Risk management data addition failed: {e}")
            return signal

    def _add_deduplication_metadata(self, signal: Dict) -> Dict:
        """Add deduplication and tracking metadata"""
        try:
            epic = signal.get('epic', '')
            signal_type = signal.get('signal_type', '')
            timeframe = signal.get('timeframe', '15m')
            current_price = signal.get('current_price', 0)
            
            # Generate signal hash for deduplication
            signal_hash = self._generate_signal_hash(epic, signal_type, timeframe, current_price)
            
            # Add tracking metadata
            current_time = datetime.now()
            signal.update({
                'signal_hash': signal_hash,
                'market_timestamp': current_time,
                'data_source': 'live_scanner',
                'cooldown_key': f"{epic}_{signal_type}_{timeframe}_{current_time.strftime('%Y%m%d%H')}",
                'alert_timestamp': current_time,
                'processing_timestamp': current_time.isoformat(),
                'pair': epic.replace('CS.D.', '').replace('.MINI.IP', ''),
                'signal_source': 'combined_strategy_modular',
                'status': 'NEW',
                'processed': False,
                'alert_level': 'MEDIUM'
            })
            
            # Add alert message
            contributing_count = len(signal.get('contributing_strategies', []))
            confidence = signal.get('confidence_score', 0)
            signal['alert_message'] = f"Combined {signal_type} signal from {contributing_count} strategies @ {confidence:.1%}"
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Deduplication metadata addition failed: {e}")
            return signal

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

    def _determine_market_regime(self, signal: Dict, latest_data: pd.Series) -> str:
        """Determine market regime"""
        try:
            ema_9 = signal.get('ema_9', 0)
            ema_21 = signal.get('ema_21', 0)
            ema_200 = signal.get('ema_200', 0)
            volatility = signal.get('atr', 0.001) / signal.get('current_price', 1) * 100
            
            # Trend determination
            if ema_9 > ema_21 > ema_200:
                trend = 'bullish_trending'
            elif ema_9 < ema_21 < ema_200:
                trend = 'bearish_trending'
            else:
                trend = 'ranging'
            
            # Volatility overlay
            if volatility > 2.0:
                return f'{trend}_high_volatility'
            elif volatility < 0.5:
                return f'{trend}_low_volatility'
            else:
                return trend
                
        except Exception:
            return 'unknown'

    def _assess_consensus_quality(self, signal: Dict) -> str:
        """Assess the quality of strategy consensus"""
        try:
            contributing_strategies = signal.get('contributing_strategies', [])
            total_strategies = signal.get('total_strategies_attempted', len(contributing_strategies))
            
            if total_strategies == 0:
                return 'unknown'
            
            agreement_ratio = len(contributing_strategies) / total_strategies
            
            if agreement_ratio == 1.0:
                return 'unanimous'
            elif agreement_ratio >= 0.8:
                return 'strong_majority'
            elif agreement_ratio >= 0.6:
                return 'majority'
            elif agreement_ratio >= 0.4:
                return 'plurality'
            else:
                return 'minority'
                
        except Exception:
            return 'unknown'

    def _generate_signal_hash(self, epic: str, signal_type: str, timeframe: str, price: float) -> str:
        """Generate unique hash for signal deduplication"""
        try:
            hash_string = f"{epic}_{signal_type}_{timeframe}_{int(price*10000)}_{datetime.now().strftime('%Y%m%d%H')}"
            return hashlib.md5(hash_string.encode()).hexdigest()[:16]
        except Exception:
            return 'hash_error'

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
        
        try:
            self.processing_stats['json_serializations'] += 1
            return convert_for_json(signal)
        except Exception as e:
            self.logger.error(f"❌ JSON serialization failed: {e}")
            return signal

    def validate_data_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness and quality"""
        try:
            self.processing_stats['data_validations'] += 1
            
            validation_result = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'data_quality_score': 1.0,
                'missing_indicators': [],
                'data_completeness': {}
            }
            
            # Check essential columns
            essential_cols = ['close', 'high', 'low', 'open']
            missing_essential = [col for col in essential_cols if col not in df.columns]
            
            if missing_essential:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Missing essential columns: {missing_essential}")
                validation_result['data_quality_score'] -= 0.5
            
            # Check for NaN values
            for col in essential_cols:
                if col in df.columns and df[col].isna().any():
                    nan_count = df[col].isna().sum()
                    nan_pct = nan_count / len(df) * 100
                    
                    if nan_pct > 10:  # More than 10% NaN
                        validation_result['is_valid'] = False
                        validation_result['issues'].append(f"High NaN percentage in {col}: {nan_pct:.1f}%")
                        validation_result['data_quality_score'] -= 0.3
                    elif nan_pct > 5:  # More than 5% NaN
                        validation_result['warnings'].append(f"Moderate NaN percentage in {col}: {nan_pct:.1f}%")
                        validation_result['data_quality_score'] -= 0.1
            
            # Check indicator availability
            required_indicators = ['ema_9', 'ema_21', 'ema_200', 'macd_line', 'macd_signal', 'macd_histogram']
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            
            if missing_indicators:
                validation_result['missing_indicators'] = missing_indicators
                validation_result['warnings'].append(f"Missing indicators: {missing_indicators}")
                validation_result['data_quality_score'] -= len(missing_indicators) * 0.05
            
            # Data completeness metrics
            validation_result['data_completeness'] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'essential_columns_present': len(essential_cols) - len(missing_essential),
                'indicator_columns_present': len(required_indicators) - len(missing_indicators),
                'overall_completeness_pct': (1 - len(missing_essential + missing_indicators) / (len(essential_cols) + len(required_indicators))) * 100
            }
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            return {
                'is_valid': False,
                'issues': [f"Validation error: {e}"],
                'data_quality_score': 0.0
            }

    def enhance_signal_to_match_individual_strategies(self, signal: Dict) -> Dict:
        """Enhance combined signal to match the richness of individual strategy signals"""
        try:
            enhanced_signal = signal.copy()
            
            # Add all the additional fields that individual strategies have
            enhanced_signal.update({
                # Additional technical analysis fields
                'technical_summary': {
                    'primary_signal': f"Combined {signal.get('signal_type', 'UNKNOWN')} Consensus",
                    'ensemble_decision': f"{len(signal.get('contributing_strategies', []))} strategies agree",
                    'confidence_assessment': self._assess_confidence_level(signal.get('confidence_score', 0)),
                    'combination_method': signal.get('combination_mode', 'consensus').upper(),
                    'strategy_diversity': f"{len(set(signal.get('contributing_strategies', [])))} unique strategies",
                    'timeframe_analysis': signal.get('timeframe', '15m'),
                    'signal_reliability': self._assess_signal_reliability(signal),
                    'consensus_type': self._assess_consensus_quality(signal),
                    'timestamp_safety': 'enabled'
                },
                
                # Combined analysis specific to ensemble
                'combined_analysis': {
                    'ensemble_type': 'multi_strategy_consensus',
                    'decision_process': f'{signal.get("combination_mode", "consensus")}_aggregation',
                    'strategy_performance': self._get_strategy_performance_summary(signal),
                    'consensus_metrics': self._calculate_consensus_metrics(signal),
                    'ensemble_effectiveness': signal.get('confidence_score', 0)
                },
                
                # Market conditions
                'market_regime': self._determine_market_regime(signal, pd.Series()),
                'volatility': signal.get('atr', 0.001) / signal.get('current_price', 1) * 100,
                'trend_strength': self._calculate_trend_strength(signal),
                'trading_session': self._determine_trading_session(),
                'market_hours': self._is_market_hours(),
                
                # Claude analysis placeholder fields (for compatibility)
                'claude_analysis': None,
                'claude_approved': None,
                'claude_score': None,
                'claude_decision': None,
                'claude_reasoning': None
            })
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Signal enhancement to match individual strategies failed: {e}")
            return signal

    def _assess_confidence_level(self, confidence: float) -> str:
        """Assess confidence level description"""
        if confidence > 0.9:
            return 'Very High'
        elif confidence > 0.8:
            return 'High'
        elif confidence > 0.7:
            return 'Medium-High'
        elif confidence > 0.6:
            return 'Medium'
        else:
            return 'Low'

    def _assess_signal_reliability(self, signal: Dict) -> str:
        """Assess signal reliability based on consensus"""
        contributing_count = len(signal.get('contributing_strategies', []))
        confidence = signal.get('confidence_score', 0)
        
        if contributing_count >= 3 and confidence > 0.8:
            return 'High'
        elif contributing_count >= 2 and confidence > 0.7:
            return 'Medium'
        else:
            return 'Low'

    def _get_strategy_performance_summary(self, signal: Dict) -> Dict:
        """Get strategy performance summary"""
        individual_confidences = signal.get('individual_confidences', {})
        
        return {
            strategy_name: {
                'confidence': confidence,
                'signal_type': signal.get('signal_type', 'UNKNOWN'),
                'contribution': confidence  # Simplified contribution calculation
            }
            for strategy_name, confidence in individual_confidences.items()
        }

    def _calculate_consensus_metrics(self, signal: Dict) -> Dict:
        """Calculate consensus metrics"""
        contributing_strategies = signal.get('contributing_strategies', [])
        individual_confidences = signal.get('individual_confidences', {})
        total_strategies = signal.get('total_strategies_attempted', len(contributing_strategies))
        
        if not individual_confidences:
            return {
                'agreement_percentage': 0,
                'average_confidence': 0,
                'confidence_std': 0,
                'strongest_contributor': None
            }
        
        confidences = list(individual_confidences.values())
        
        return {
            'agreement_percentage': len(contributing_strategies) / max(1, total_strategies) * 100,
            'average_confidence': sum(confidences) / len(confidences),
            'confidence_std': float(np.std(confidences)) if len(confidences) > 1 else 0,
            'strongest_contributor': max(individual_confidences.items(), key=lambda x: x[1])[0] if individual_confidences else None
        }

    def _calculate_trend_strength(self, signal: Dict) -> str:
        """Calculate trend strength based on EMA separation"""
        try:
            ema_9 = signal.get('ema_9', 0)
            ema_21 = signal.get('ema_21', 0)
            ema_200 = signal.get('ema_200', 0)
            
            if not all([ema_9, ema_21, ema_200]):
                return 'unknown'
            
            spread_short_long = abs(ema_9 - ema_21) / ema_21 if ema_21 != 0 else 0
            spread_long_trend = abs(ema_21 - ema_200) / ema_200 if ema_200 != 0 else 0
            
            total_spread = spread_short_long + spread_long_trend
            
            if total_spread > 0.01:  # 1%
                return 'strong'
            elif total_spread > 0.005:  # 0.5%
                return 'medium'
            else:
                return 'weak'
        except:
            return 'unknown'

    def _is_market_hours(self) -> bool:
        """Check if current time is during major market hours"""
        try:
            current_hour = datetime.now().hour
            return 1 <= current_hour <= 23
        except:
            return True

    def get_data_helper_stats(self) -> Dict[str, Any]:
        """Get data helper statistics"""
        try:
            return {
                'total_enhancements': self.processing_stats['total_enhancements'],
                'successful_enhancements': self.processing_stats['successful_enhancements'],
                'failed_enhancements': self.processing_stats['total_enhancements'] - self.processing_stats['successful_enhancements'],
                'success_rate_percent': (self.processing_stats['successful_enhancements'] / max(1, self.processing_stats['total_enhancements'])) * 100,
                'timestamp_conversions': self.processing_stats['timestamp_conversions'],
                'indicator_additions': self.processing_stats['indicator_additions'],
                'json_serializations': self.processing_stats['json_serializations'],
                'data_validations': self.processing_stats['data_validations'],
                'cache_size': len(self._cache),
                'helper_config': self.helper_config,
                'forex_optimizer_available': self.forex_optimizer is not None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Data helper stats collection failed: {e}")
            return {'error': str(e)}

    def clear_cache(self) -> None:
        """Clear the data helper cache"""
        try:
            cache_size = len(self._cache)
            self._cache.clear()
            self.logger.debug(f"🧹 Data helper cache cleared: {cache_size} items removed")
        except Exception as e:
            self.logger.error(f"❌ Cache clearing failed: {e}")

    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics information"""
        return {
            'module_name': 'CombinedDataHelper',
            'initialization_successful': True,
            'forex_optimizer_available': self.forex_optimizer is not None,
            'processing_stats': self.get_data_helper_stats(),
            'helper_config': self.helper_config,
            'cache_size': len(self._cache),
            'timestamp_safety_enabled': self.helper_config['enable_timestamp_safety'],
            'comprehensive_enhancement_enabled': self.helper_config['enable_comprehensive_enhancement'],
            'json_validation_enabled': self.helper_config['enable_json_validation'],
            'timestamp': datetime.now().isoformat()
        }