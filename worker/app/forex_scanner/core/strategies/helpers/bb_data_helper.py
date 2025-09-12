# core/strategies/helpers/bb_data_helper.py
"""
BB Data Helper Module - Fixed Indicator Validation
🔧 DATA: Data preparation, enhancement, and manipulation utilities for BB+Supertrend
📊 COMPREHENSIVE: BB/SuperTrend indicators, semantic mapping, signal enhancement
🎯 FOCUSED: Single responsibility for BB data handling
🔄 FIXED: Improved indicator validation to reduce fallback usage

This module contains all the data preparation and enhancement logic
for BB+Supertrend strategy with improved validation logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging
from datetime import datetime
import hashlib
import json
try:
    import config
except ImportError:
    from forex_scanner import config


class BBDataHelper:
    """
    🔧 DATA: Comprehensive data preparation and enhancement for BB+Supertrend strategy
    
    Responsibilities:
    - BB and SuperTrend indicator calculation and validation
    - Data validation and quality checks
    - Signal data enhancement for database storage
    - Timestamp safety and conversion
    - JSON serialization and data integrity
    """
    
    def __init__(self, logger: logging.Logger = None, forex_optimizer=None):
        self.logger = logger or logging.getLogger(__name__)
        self.forex_optimizer = forex_optimizer  # Will be injected by main strategy
        
        # Get BB config from forex optimizer or use defaults
        if self.forex_optimizer:
            self.bb_config = self.forex_optimizer.get_bb_config()
        else:
            self.bb_config = {
                'bb_period': 14,
                'bb_std_dev': 1.8,
                'supertrend_period': 8,
                'supertrend_multiplier': 2.5,
                'base_confidence': 0.60
            }
        
        # Data processing statistics
        self._indicators_calculated = 0
        self._signals_enhanced = 0
        self._validation_failures = 0
        self._fallback_applications = 0
        
        self.logger.info("🔧 BB Data Helper initialized with improved validation")

    def validate_input_data(self, df: pd.DataFrame, epic: str, min_bars: int) -> bool:
        """
        ✅ Validate input data for BB+SuperTrend processing
        """
        try:
            if df is None or len(df) < min_bars:
                self.logger.debug(f"Insufficient data for BB+SuperTrend: {len(df) if df is not None else 0} < {min_bars}")
                self._validation_failures += 1
                return False
            
            # Check for required columns
            required_columns = ['close', 'high', 'low']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.debug(f"Missing required columns: {missing_columns}")
                self._validation_failures += 1
                return False
            
            # Check for sufficient non-null data in recent bars (more lenient)
            recent_data = df.tail(min(20, len(df)))  # Check last 20 bars or available data
            null_counts = recent_data[required_columns].isnull().sum()
            null_percentage = null_counts / len(recent_data)
            
            # Allow up to 10% nulls in recent data
            if null_percentage.max() > 0.1:
                self.logger.debug(f"Too many null values in recent data: {null_percentage.to_dict()}")
                self._validation_failures += 1
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            self._validation_failures += 1
            return False

    def ensure_bb_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔄 Ensure BB and SuperTrend indicators are present in the DataFrame
        """
        try:
            df_enhanced = df.copy()
            
            # Check and calculate BB indicators
            bb_cols = ['bb_upper', 'bb_middle', 'bb_lower']
            if not all(col in df_enhanced.columns for col in bb_cols):
                self.logger.debug("Calculating missing Bollinger Bands indicators")
                df_enhanced = self._calculate_bollinger_bands(df_enhanced)
                self._indicators_calculated += 1
            
            # Check and calculate SuperTrend indicators
            st_cols = ['supertrend', 'supertrend_direction']
            if not all(col in df_enhanced.columns for col in st_cols):
                self.logger.debug("Calculating missing SuperTrend indicators")
                # Ensure ATR is present first
                if 'atr' not in df_enhanced.columns:
                    df_enhanced = self._calculate_atr(df_enhanced)
                df_enhanced = self._calculate_supertrend(df_enhanced)
                self._indicators_calculated += 1
            
            # Improved indicator quality validation
            quality_check = self._validate_indicator_quality(df_enhanced)
            if not quality_check['is_valid']:
                self.logger.debug(f"Indicator quality issues detected: {quality_check['issues']}")
                
                # Try to fix issues before falling back
                df_enhanced = self._fix_indicator_issues(df_enhanced, quality_check['issues'])
                
                # Re-validate after fixes
                recheck = self._validate_indicator_quality(df_enhanced)
                if not recheck['is_valid']:
                    self.logger.warning(f"Using fallback indicators due to: {recheck['issues']}")
                    df_enhanced = self._apply_fallback_indicators(df_enhanced)
                    self._fallback_applications += 1
                else:
                    self.logger.debug("✅ Indicator issues fixed successfully")
            else:
                self.logger.debug("✅ All indicators passed quality validation")
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Error ensuring BB indicators: {e}")
            return self._apply_fallback_indicators(df)

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📊 Calculate Bollinger Bands indicators with improved error handling
        """
        try:
            if len(df) < self.bb_config['bb_period']:
                self.logger.debug(f"Insufficient data for BB calculation: {len(df)} < {self.bb_config['bb_period']}")
                return self._apply_bb_fallback(df)
            
            # Calculate with better error handling
            period = self.bb_config['bb_period']
            std_dev = self.bb_config['bb_std_dev']
            
            # Use SMA for middle band (more robust than EMA for BB)
            df['bb_middle'] = df['close'].rolling(window=period, min_periods=max(1, period//2)).mean()
            
            # Calculate standard deviation with minimum periods
            rolling_std = df['close'].rolling(window=period, min_periods=max(1, period//2)).std()
            
            # Calculate bands
            df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
            df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
            
            # Forward fill any remaining NaNs in the last few rows
            df['bb_middle'] = df['bb_middle'].ffill()
            df['bb_upper'] = df['bb_upper'].ffill()
            df['bb_lower'] = df['bb_lower'].ffill()
            
            # Final fallback for any remaining NaNs
            if df[['bb_upper', 'bb_middle', 'bb_lower']].iloc[-1].isnull().any():
                self.logger.debug("Applying final BB fallback for NaN values")
                df = self._apply_bb_fallback(df)
            
            self.logger.debug(f"✅ BB calculated successfully (period: {period}, std: {std_dev})")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return self._apply_bb_fallback(df)

    def _apply_bb_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔄 Apply BB fallback values when calculation fails
        """
        try:
            if 'close' in df.columns and len(df) > 0:
                close_price = df['close'].iloc[-1]
                # Use 2% bands as fallback
                df['bb_middle'] = close_price
                df['bb_upper'] = close_price * 1.02
                df['bb_lower'] = close_price * 0.98
            else:
                # Ultimate fallback
                df['bb_middle'] = 1.0
                df['bb_upper'] = 1.02
                df['bb_lower'] = 0.98
            
            self.logger.debug("Applied BB fallback values")
            return df
            
        except Exception as e:
            self.logger.error(f"BB fallback failed: {e}")
            return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📊 Calculate ATR with improved robustness
        """
        try:
            if len(df) < 2:
                df['atr'] = df.get('close', 1.0) * 0.01  # 1% of close price
                return df
            
            period = self.bb_config.get('atr_period', 14)
            
            # Calculate True Range components
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            # True Range is the maximum of the three
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate ATR using SMA with minimum periods
            df['atr'] = true_range.rolling(window=period, min_periods=max(1, period//4)).mean()
            
            # Forward fill NaNs
            df['atr'] = df['atr'].ffill()
            
            # Final fallback for ATR
            if df['atr'].iloc[-1] is np.nan or df['atr'].iloc[-1] <= 0:
                df['atr'] = df['close'] * 0.01  # 1% of close price
            
            self.logger.debug(f"✅ ATR calculated (period: {period})")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            df['atr'] = df.get('close', 1.0) * 0.01
            return df

    def _calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📊 Calculate SuperTrend with improved robustness
        """
        try:
            if len(df) < 2:
                df['supertrend'] = df.get('close', 1.0)
                df['supertrend_direction'] = 1
                return df
            
            period = self.bb_config['supertrend_period']
            multiplier = self.bb_config['supertrend_multiplier']
            
            # Ensure ATR is available
            if 'atr' not in df.columns:
                df = self._calculate_atr(df)
            
            # Calculate basic SuperTrend
            hl2 = (df['high'] + df['low']) / 2.0
            atr = df['atr']
            
            # Upper and lower bands
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            supertrend = np.full(len(df), np.nan)
            direction = np.full(len(df), np.nan)
            
            # Initialize first valid values
            first_valid_idx = max(period - 1, 0)
            
            if first_valid_idx < len(df):
                supertrend[first_valid_idx] = lower_band.iloc[first_valid_idx]
                direction[first_valid_idx] = 1
                
                # Calculate SuperTrend for remaining bars
                for i in range(first_valid_idx + 1, len(df)):
                    if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                        # Use previous values if current calculation fails
                        supertrend[i] = supertrend[i-1]
                        direction[i] = direction[i-1]
                        continue
                        
                    # Calculate final upper and lower bands
                    ub = upper_band.iloc[i] if upper_band.iloc[i] < upper_band.iloc[i-1] or df['close'].iloc[i-1] > upper_band.iloc[i-1] else upper_band.iloc[i-1]
                    lb = lower_band.iloc[i] if lower_band.iloc[i] > lower_band.iloc[i-1] or df['close'].iloc[i-1] < lower_band.iloc[i-1] else lower_band.iloc[i-1]
                    
                    # Determine SuperTrend value and direction
                    if (supertrend[i-1] == ub and df['close'].iloc[i] <= ub):
                        supertrend[i] = ub
                        direction[i] = -1
                    elif (supertrend[i-1] == ub and df['close'].iloc[i] > ub):
                        supertrend[i] = lb
                        direction[i] = 1
                    elif (supertrend[i-1] == lb and df['close'].iloc[i] >= lb):
                        supertrend[i] = lb
                        direction[i] = 1
                    elif (supertrend[i-1] == lb and df['close'].iloc[i] < lb):
                        supertrend[i] = ub
                        direction[i] = -1
                    else:
                        supertrend[i] = supertrend[i-1]
                        direction[i] = direction[i-1]
            
            # Assign to DataFrame
            df['supertrend'] = supertrend
            df['supertrend_direction'] = direction
            
            # Forward fill any remaining NaNs
            df['supertrend'] = df['supertrend'].ffill()
            df['supertrend_direction'] = df['supertrend_direction'].ffill()
            
            # Final fallback
            if pd.isna(df['supertrend'].iloc[-1]) or pd.isna(df['supertrend_direction'].iloc[-1]):
                df['supertrend'] = df['close']
                df['supertrend_direction'] = 1
            
            self.logger.debug(f"✅ SuperTrend calculated (period: {period}, multiplier: {multiplier})")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating SuperTrend: {e}")
            df['supertrend'] = df.get('close', 1.0)
            df['supertrend_direction'] = 1
            return df

    def _validate_indicator_quality(self, df: pd.DataFrame) -> Dict:
        """
        ✅ Validate the quality of calculated indicators with detailed feedback
        """
        try:
            if len(df) == 0:
                return {'is_valid': False, 'issues': ['Empty DataFrame']}
            
            latest = df.iloc[-1]
            issues = []
            
            # Check BB indicators
            bb_fields = ['bb_upper', 'bb_middle', 'bb_lower']
            for field in bb_fields:
                if field not in latest.index:
                    issues.append(f"Missing {field}")
                elif pd.isna(latest[field]):
                    issues.append(f"NaN in {field}")
                elif latest[field] <= 0:
                    issues.append(f"Non-positive {field}: {latest[field]}")
            
            # Check BB logical order (more lenient)
            if all(field in latest.index and not pd.isna(latest[field]) for field in bb_fields):
                if not (latest['bb_upper'] > latest['bb_lower']):
                    issues.append(f"BB bands inverted: upper={latest['bb_upper']:.5f}, lower={latest['bb_lower']:.5f}")
                elif latest['bb_upper'] == latest['bb_lower']:
                    issues.append(f"BB bands equal: {latest['bb_upper']:.5f}")
                elif not (latest['bb_upper'] >= latest['bb_middle'] >= latest['bb_lower']):
                    # More flexible middle band check - allow slight variations
                    bb_width = latest['bb_upper'] - latest['bb_lower']
                    middle_tolerance = bb_width * 0.01  # 1% tolerance
                    
                    if not (latest['bb_middle'] >= latest['bb_lower'] - middle_tolerance and 
                           latest['bb_middle'] <= latest['bb_upper'] + middle_tolerance):
                        issues.append(f"BB middle out of range: middle={latest['bb_middle']:.5f}, range=[{latest['bb_lower']:.5f}, {latest['bb_upper']:.5f}]")
            
            # Check SuperTrend indicators (more lenient)
            st_fields = ['supertrend', 'supertrend_direction']
            for field in st_fields:
                if field not in latest.index:
                    issues.append(f"Missing {field}")
                elif pd.isna(latest[field]):
                    issues.append(f"NaN in {field}")
            
            # Check SuperTrend direction is valid (allow 0 as neutral)
            if 'supertrend_direction' in latest.index and not pd.isna(latest['supertrend_direction']):
                if latest['supertrend_direction'] not in [-1, 0, 1]:
                    issues.append(f"Invalid SuperTrend direction: {latest['supertrend_direction']}")
            
            # Check for reasonable BB width (prevent ultra-tight bands)
            if all(field in latest.index and not pd.isna(latest[field]) for field in bb_fields):
                bb_width_pct = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle'] if latest['bb_middle'] > 0 else 0
                if bb_width_pct < 0.001:  # Less than 0.1% width
                    issues.append(f"BB width too narrow: {bb_width_pct:.6f}%")
                elif bb_width_pct > 0.2:  # More than 20% width
                    issues.append(f"BB width too wide: {bb_width_pct:.6f}%")
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues,
                'indicators_checked': len(bb_fields) + len(st_fields),
                'latest_values': {
                    'bb_upper': latest.get('bb_upper'),
                    'bb_middle': latest.get('bb_middle'),
                    'bb_lower': latest.get('bb_lower'),
                    'supertrend': latest.get('supertrend'),
                    'supertrend_direction': latest.get('supertrend_direction')
                }
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'indicators_checked': 0,
                'latest_values': {}
            }

    def _fix_indicator_issues(self, df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """
        🔧 Attempt to fix common indicator issues before falling back
        """
        try:
            df_fixed = df.copy()
            
            for issue in issues:
                if "NaN in" in issue:
                    # Fix NaN values by forward filling
                    field = issue.split("NaN in ")[1]
                    if field in df_fixed.columns:
                        df_fixed[field] = df_fixed[field].ffill()
                        self.logger.debug(f"Fixed NaN in {field} using forward fill")
                
                elif "Non-positive" in issue:
                    # Fix non-positive values
                    field = issue.split("Non-positive ")[1].split(":")[0]
                    if field in df_fixed.columns and 'close' in df_fixed.columns:
                        # Replace with reasonable values based on close price
                        close_price = df_fixed['close'].iloc[-1]
                        if field == 'bb_upper':
                            df_fixed[field] = close_price * 1.02
                        elif field == 'bb_lower':
                            df_fixed[field] = close_price * 0.98
                        elif field == 'bb_middle':
                            df_fixed[field] = close_price
                        elif field == 'supertrend':
                            df_fixed[field] = close_price
                        self.logger.debug(f"Fixed non-positive {field}")
                
                elif "BB bands inverted" in issue or "BB bands equal" in issue:
                    # Fix inverted or equal BB bands
                    if all(col in df_fixed.columns for col in ['bb_upper', 'bb_middle', 'bb_lower', 'close']):
                        close_price = df_fixed['close'].iloc[-1]
                        df_fixed['bb_middle'] = close_price
                        df_fixed['bb_upper'] = close_price * 1.02
                        df_fixed['bb_lower'] = close_price * 0.98
                        self.logger.debug("Fixed BB band ordering")
                
                elif "Invalid SuperTrend direction" in issue:
                    # Fix invalid SuperTrend direction
                    df_fixed['supertrend_direction'] = 1  # Default to bullish
                    self.logger.debug("Fixed SuperTrend direction")
                
                elif "BB width too narrow" in issue:
                    # Widen bands slightly
                    if all(col in df_fixed.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                        middle = df_fixed['bb_middle'].iloc[-1]
                        width = middle * 0.01  # 1% width minimum
                        df_fixed['bb_upper'] = middle + width
                        df_fixed['bb_lower'] = middle - width
                        self.logger.debug("Fixed narrow BB width")
            
            return df_fixed
            
        except Exception as e:
            self.logger.debug(f"Failed to fix indicator issues: {e}")
            return df

    def _apply_fallback_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔄 Apply fallback indicator values to prevent crashes
        """
        try:
            df_fallback = df.copy()
            self._fallback_applications += 1
            
            # BB fallback values
            if 'close' in df_fallback.columns and len(df_fallback) > 0:
                close_price = df_fallback['close'].iloc[-1]
                if pd.isna(close_price) or close_price <= 0:
                    close_price = 1.0
                    
                df_fallback['bb_middle'] = close_price
                df_fallback['bb_upper'] = close_price * 1.02  # 2% above close
                df_fallback['bb_lower'] = close_price * 0.98  # 2% below close
            else:
                df_fallback['bb_middle'] = 1.0
                df_fallback['bb_upper'] = 1.02
                df_fallback['bb_lower'] = 0.98
            
            # SuperTrend fallback values
            df_fallback['supertrend'] = df_fallback.get('close', df_fallback['bb_middle'])
            df_fallback['supertrend_direction'] = 1  # Default to bullish
            
            # ATR fallback
            df_fallback['atr'] = df_fallback.get('close', df_fallback['bb_middle']) * 0.01
            
            self.logger.info("⚠️ Applied fallback indicators to prevent crashes")
            return df_fallback
            
        except Exception as e:
            self.logger.error(f"Fallback indicators failed: {e}")
            # Ultimate fallback
            df_fallback = df.copy()
            df_fallback['bb_middle'] = 1.0
            df_fallback['bb_upper'] = 1.02
            df_fallback['bb_lower'] = 0.98
            df_fallback['supertrend'] = 1.0
            df_fallback['supertrend_direction'] = 1
            df_fallback['atr'] = 0.01
            return df_fallback

    def create_enhanced_signal(
        self,
        signal_type: str,
        epic: str, 
        timeframe: str,
        current: pd.Series,
        previous: pd.Series,
        confidence_score: float,
        spread_pips: float,
        reason: str
    ) -> Dict:
        """
        🎯 Create enhanced signal with all BB+SuperTrend specific data
        """
        try:
            self._signals_enhanced += 1
            
            # Build comprehensive signal
            signal = {
                'signal_type': signal_type,
                'epic': epic,
                'timeframe': timeframe,
                'timestamp': datetime.now(),
                'confidence_score': confidence_score,
                'spread_pips': spread_pips,
                'reason': reason,
                'strategy': 'bb_supertrend_enhanced',
                
                # Price data
                'entry_price': float(current['close']),
                'current_price': float(current['close']),
                'high': float(current['high']),
                'low': float(current['low']),
                
                # BB indicators
                'bb_upper': float(current['bb_upper']),
                'bb_middle': float(current['bb_middle']),
                'bb_lower': float(current['bb_lower']),
                
                # SuperTrend indicators
                'supertrend': float(current['supertrend']),
                'supertrend_direction': int(current['supertrend_direction']),
                
                # Additional indicators
                'atr': float(current.get('atr', 0.001)),
                'volume': float(current.get('volume', current.get('ltv', 0))),
                
                # BB analysis
                'bb_position': self._calculate_bb_position(current),
                'bb_width_pct': self._calculate_bb_width_pct(current),
                
                # Strategy metadata
                'bb_config': self.bb_config.copy(),
                'data_quality': 'fallback_applied' if self._fallback_applications > 0 else 'calculated',
                'validation_score': confidence_score
            }
            
            # Add stop loss and take profit calculations
            signal.update(self._calculate_sl_tp(current, signal_type))
            
            # Add previous bar data for analysis
            signal['previous_close'] = float(previous['close'])
            signal['price_change'] = float(current['close'] - previous['close'])
            signal['price_change_pct'] = float((current['close'] - previous['close']) / previous['close'] * 100)
            
            # Ensure JSON compatibility
            signal = self._make_json_safe(signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Failed to create enhanced signal: {e}")
            # Return minimal signal to prevent crashes
            return {
                'signal_type': signal_type,
                'epic': epic,
                'timeframe': timeframe,
                'timestamp': datetime.now(),
                'confidence_score': confidence_score,
                'entry_price': float(current.get('close', 1.0)),
                'strategy': 'bb_supertrend_enhanced',
                'error': str(e)
            }

    def _calculate_bb_position(self, current: pd.Series) -> float:
        """Calculate normalized BB position (0-1)"""
        try:
            bb_width = current['bb_upper'] - current['bb_lower']
            if bb_width <= 0:
                return 0.5
            return float((current['close'] - current['bb_lower']) / bb_width)
        except:
            return 0.5

    def _calculate_bb_width_pct(self, current: pd.Series) -> float:
        """Calculate BB width as percentage of middle band"""
        try:
            if current['bb_middle'] <= 0:
                return 0.02
            return float((current['bb_upper'] - current['bb_lower']) / current['bb_middle'])
        except:
            return 0.02

    def _calculate_sl_tp(self, current: pd.Series, signal_type: str) -> Dict:
        """Calculate stop loss and take profit levels"""
        try:
            atr = current.get('atr', current['close'] * 0.01)
            
            if signal_type == 'BULL':
                # For BULL signals: SL below lower BB, TP at upper BB
                sl = current['bb_lower'] - (atr * 0.5)
                tp = current['bb_upper']
            else:  # BEAR
                # For BEAR signals: SL above upper BB, TP at lower BB  
                sl = current['bb_upper'] + (atr * 0.5)
                tp = current['bb_lower']
            
            return {
                'stop_loss': float(sl),
                'take_profit': float(tp),
                'sl_pips': float(abs(current['close'] - sl) / 0.0001),  # Assuming 4-digit pairs
                'tp_pips': float(abs(tp - current['close']) / 0.0001),
                'risk_reward_ratio': float(abs(tp - current['close']) / abs(current['close'] - sl)) if abs(current['close'] - sl) > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.debug(f"SL/TP calculation failed: {e}")
            return {
                'stop_loss': float(current['close'] * 0.99),
                'take_profit': float(current['close'] * 1.01),
                'sl_pips': 10.0,
                'tp_pips': 10.0,
                'risk_reward_ratio': 1.0
            }

    def _make_json_safe(self, signal: Dict) -> Dict:
        """Ensure signal data is JSON serializable"""
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

    def get_data_stats(self) -> Dict:
        """📊 Get data processing statistics"""
        return {
            'indicators_calculated': self._indicators_calculated,
            'signals_enhanced': self._signals_enhanced,
            'validation_failures': self._validation_failures,
            'fallback_applications': self._fallback_applications,
            'fallback_rate': self._fallback_applications / max(self._indicators_calculated, 1) * 100,
            'bb_config': self.bb_config
        }

    def reset_stats(self):
        """🔄 Reset data processing statistics"""
        self._indicators_calculated = 0
        self._signals_enhanced = 0
        self._validation_failures = 0
        self._fallback_applications = 0
        self.logger.info("🔄 BB Data Helper stats reset")

    def debug_data_validation(self, df: pd.DataFrame, epic: str, min_bars: int) -> Dict:
        """🔍 Get comprehensive data validation information for debugging"""
        try:
            debug_info = {
                'input_validation': self.validate_input_data(df, epic, min_bars),
                'data_shape': df.shape if df is not None else (0, 0),
                'required_columns': ['close', 'high', 'low'],
                'available_columns': list(df.columns) if df is not None else [],
                'bb_config': self.bb_config,
                'data_stats': self.get_data_stats()
            }
            
            if df is not None and len(df) > 0:
                # Test indicator quality
                quality_check = self._validate_indicator_quality(df)
                debug_info['indicator_quality'] = quality_check
                
                debug_info['data_quality'] = {
                    'null_counts': df.isnull().sum().to_dict(),
                    'data_types': df.dtypes.to_dict(),
                    'latest_values': df.iloc[-1].to_dict() if len(df) > 0 else {}
                }
                
                # Test BB calculation
                if len(df) >= self.bb_config['bb_period']:
                    test_df = self._calculate_bollinger_bands(df.copy())
                    debug_info['bb_calculation_test'] = {
                        'success': all(col in test_df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']),
                        'latest_bb_values': {
                            'bb_upper': test_df['bb_upper'].iloc[-1] if 'bb_upper' in test_df.columns else None,
                            'bb_middle': test_df['bb_middle'].iloc[-1] if 'bb_middle' in test_df.columns else None,
                            'bb_lower': test_df['bb_lower'].iloc[-1] if 'bb_lower' in test_df.columns else None
                        }
                    }
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}