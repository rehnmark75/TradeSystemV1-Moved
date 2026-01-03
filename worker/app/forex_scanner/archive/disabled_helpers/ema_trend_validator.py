# core/strategies/helpers/ema_trend_validator.py
"""
EMA Trend Validator Module
Validates EMA signals against trend indicators and momentum oscillators
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class EMATrendValidator:
    """Handles all trend and momentum validation for EMA signals"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = 1e-8  # Epsilon for stability
    
    def validate_ema_200_trend(self, row: pd.Series, signal_type: str) -> bool:
        """
        EMA 200 TREND FILTER: Ensure signals align with major trend direction
        
        Critical trend filter:
        - BUY signals: Price must be ABOVE EMA 200 (uptrend)
        - SELL signals: Price must be BELOW EMA 200 (downtrend)
        
        Args:
            row: DataFrame row with price and EMA data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if trend is correct, False if against major trend
        """
        try:
            if not getattr(config, 'EMA_200_TREND_FILTER_ENABLED', True):
                return True
            
            buffer_pips = getattr(config, 'EMA_200_BUFFER_PIPS', 1.0)
            
            # Get EMA 200 and price data
            ema_200 = row.get('ema_200', 0)
            if ema_200 == 0:
                self.logger.debug("EMA 200 not available for trend validation")
                return True  # Allow signal if EMA 200 not available
                
            close_price = row.get('close', 0)
            if close_price == 0:
                self.logger.debug("Close price not available for trend validation")
                return True
            
            # Determine pip multiplier based on price level (JPY pairs vs others)
            if close_price > 50:  # Likely JPY pair (e.g., USDJPY ~150)
                pip_multiplier = 100   # JPY pairs: 1 pip = 0.01
            else:  # Standard pairs (e.g., EURUSD ~1.17)
                pip_multiplier = 10000  # Standard pairs: 1 pip = 0.0001
            
            buffer_distance = buffer_pips / pip_multiplier
            
            if signal_type == 'BULL':
                # BUY signals: Price must be ABOVE EMA 200 (+ buffer)
                required_price = ema_200 + buffer_distance
                trend_valid = close_price > required_price
                
                if trend_valid:
                    distance_above = (close_price - ema_200) * pip_multiplier
                    self.logger.debug(f"EMA 200 trend OK for BULL: price {distance_above:.1f} pips above EMA 200")
                    return True
                else:
                    distance_below = (ema_200 - close_price) * pip_multiplier
                    self.logger.info(f"EMA 200 trend INVALID for BULL: price {distance_below:.1f} pips below EMA 200")
                    return False
                    
            elif signal_type == 'BEAR':
                # SELL signals: Price must be BELOW EMA 200 (- buffer)
                required_price = ema_200 - buffer_distance
                trend_valid = close_price < required_price
                
                if trend_valid:
                    distance_below = (ema_200 - close_price) * pip_multiplier
                    self.logger.debug(f"EMA 200 trend OK for BEAR: price {distance_below:.1f} pips below EMA 200")
                    return True
                else:
                    distance_above = (close_price - ema_200) * pip_multiplier
                    self.logger.info(f"EMA 200 trend INVALID for BEAR: price {distance_above:.1f} pips above EMA 200")
                    return False
            
            return False  # Unknown signal type
            
        except Exception as e:
            self.logger.error(f"Error validating EMA 200 trend: {e}")
            return True  # Allow signal on error
    
    def validate_rsi_zone(self, latest_row: pd.Series, signal_type: str) -> tuple:
        """
        RSI TREND CONFIRMATION: Use RSI to confirm trend direction

        CORRECTED LOGIC: RSI > 50 = bullish, RSI < 50 = bearish
        Not using RSI for overbought/oversold - only for trend confirmation

        Args:
            latest_row: DataFrame row with RSI data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            (is_valid, confidence_boost): Tuple of validation result and boost value
        """
        try:
            from configdata.strategies.config_ema_strategy import (
                EMA_USE_RSI_FILTER,
                EMA_RSI_BULL_MIN,
                EMA_RSI_BEAR_MAX,
                EMA_RSI_CONFIDENCE_WEIGHT
            )

            if not EMA_USE_RSI_FILTER:
                return True, 0.0

            rsi = latest_row.get('rsi', 50)

            if signal_type == 'BULL':
                # RSI must be > 50 to confirm bullish momentum
                is_valid = rsi >= EMA_RSI_BULL_MIN

                if not is_valid:
                    self.logger.warning(f"❌ BULL signal REJECTED: RSI confirms bearish trend ({rsi:.1f} < {EMA_RSI_BULL_MIN})")
                    return False, 0.0

                # Calculate confidence boost: Higher RSI = stronger bullish momentum
                # RSI 50 = minimal boost, RSI 70+ = max boost
                if rsi >= 70:
                    boost_factor = 1.0  # Maximum boost
                elif rsi >= 60:
                    boost_factor = 0.75
                else:  # 50-60 range
                    boost_factor = 0.5

                confidence_boost = EMA_RSI_CONFIDENCE_WEIGHT * boost_factor

                self.logger.info(f"✅ RSI confirms BULL trend: {rsi:.1f} (boost: +{confidence_boost:.1%})")
                return True, confidence_boost

            elif signal_type == 'BEAR':
                # RSI must be < 50 to confirm bearish momentum
                is_valid = rsi <= EMA_RSI_BEAR_MAX

                if not is_valid:
                    self.logger.warning(f"❌ BEAR signal REJECTED: RSI confirms bullish trend ({rsi:.1f} > {EMA_RSI_BEAR_MAX})")
                    return False, 0.0

                # Calculate confidence boost: Lower RSI = stronger bearish momentum
                # RSI 50 = minimal boost, RSI 30- = max boost
                if rsi <= 30:
                    boost_factor = 1.0  # Maximum boost
                elif rsi <= 40:
                    boost_factor = 0.75
                else:  # 40-50 range
                    boost_factor = 0.5

                confidence_boost = EMA_RSI_CONFIDENCE_WEIGHT * boost_factor

                self.logger.info(f"✅ RSI confirms BEAR trend: {rsi:.1f} (boost: +{confidence_boost:.1%})")
                return True, confidence_boost

            return False, 0.0

        except Exception as e:
            self.logger.error(f"Error validating RSI zone: {e}")
            return True, 0.0  # Graceful degradation

    def calculate_rejection_candle_quality(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        Calculate quality score for rejection candle pattern

        Higher score for:
        - Strong rejection wick (2x+ opposite wick)
        - Decent candle body (not doji)
        - Close near EMA (perfect bounce)

        Returns:
            Confidence boost (0.0 to 0.10)
        """
        try:
            ema21 = latest_row.get('ema_21', 0)
            close = latest_row.get('close', 0)
            open_price = latest_row.get('open', close)
            high = latest_row.get('high', close)
            low = latest_row.get('low', close)

            # Calculate wick and body metrics
            body = abs(close - open_price)
            candle_range = high - low
            upper_wick = high - max(close, open_price)
            lower_wick = min(close, open_price) - low

            if candle_range < 1e-8:
                return 0.0

            body_ratio = body / candle_range

            if signal_type == 'BULL':
                # Check lower wick (rejection from below)
                if lower_wick < 1e-8:
                    return 0.0
                wick_ratio = lower_wick / (upper_wick + 1e-8)

                # Bonus for perfect bounce (close near EMA)
                distance_to_ema = abs(close - ema21) / (ema21 + 1e-8)
                proximity_bonus = max(0, 1 - (distance_to_ema * 100))  # 0-1 scale

            elif signal_type == 'BEAR':
                # Check upper wick (rejection from above)
                if upper_wick < 1e-8:
                    return 0.0
                wick_ratio = upper_wick / (lower_wick + 1e-8)

                # Bonus for perfect bounce
                distance_to_ema = abs(close - ema21) / (ema21 + 1e-8)
                proximity_bonus = max(0, 1 - (distance_to_ema * 100))
            else:
                return 0.0

            # Quality score components
            wick_score = min(1.0, wick_ratio / 2.0)  # Normalize (2x+ wick = 1.0)
            body_score = min(1.0, body_ratio / 0.5)   # Normalize (50%+ body = 1.0)

            # Combined quality score
            quality = (wick_score * 0.5 + body_score * 0.3 + proximity_bonus * 0.2)
            confidence_boost = 0.10 * quality  # Max 10% boost

            self.logger.debug(f"Rejection quality: {quality:.2f} (wick:{wick_score:.2f}, body:{body_score:.2f}, prox:{proximity_bonus:.2f})")
            return confidence_boost

        except Exception as e:
            self.logger.error(f"Error calculating rejection quality: {e}")
            return 0.0
    
    def detect_macd_histogram_trend(self, df: pd.DataFrame, lookback: int = 3) -> str:
        """
        Detect MACD histogram trend over the last N periods
        
        Args:
            df: DataFrame with MACD histogram data
            lookback: Number of periods to look back for trend calculation (default: 3)
            
        Returns:
            'RISING', 'DESCENDING', or 'NEUTRAL'
        """
        try:
            if df is None or len(df) < lookback + 1:
                self.logger.debug(f"Insufficient data for MACD trend detection: {len(df) if df is not None else 0} bars")
                return 'NEUTRAL'
            
            # Get MACD histogram values
            if 'macd_histogram' not in df.columns:
                self.logger.debug("MACD histogram column not found")
                return 'NEUTRAL'
            
            # Get the last few histogram values
            recent_values = df['macd_histogram'].tail(lookback + 1).values
            
            # Remove NaN values
            recent_values = recent_values[~pd.isna(recent_values)]
            
            if len(recent_values) < 2:
                self.logger.debug("Not enough valid MACD histogram values for trend detection")
                return 'NEUTRAL'
            
            # Calculate trend using linear regression slope
            x = np.arange(len(recent_values))
            if len(x) > 1:
                # Calculate slope using numpy polyfit
                try:
                    slope, _ = np.polyfit(x, recent_values, 1)
                    
                    # Get sensitivity-based slope threshold
                    sensitivity = getattr(config, 'MACD_TREND_SENSITIVITY', 'normal')
                    sensitivity_settings = getattr(config, 'MACD_SENSITIVITY_SETTINGS', {})
                    
                    if sensitivity in sensitivity_settings:
                        min_slope_threshold = sensitivity_settings[sensitivity]['min_slope']
                    else:
                        min_slope_threshold = getattr(config, 'MACD_MIN_SLOPE_THRESHOLD', 0.0001)
                    
                    if slope > min_slope_threshold:
                        trend = 'RISING'
                    elif slope < -min_slope_threshold:
                        trend = 'DESCENDING'
                    else:
                        trend = 'NEUTRAL'
                    
                    self.logger.debug(f"MACD histogram trend: {trend} (slope: {slope:.6f})")
                    return trend
                    
                except Exception as e:
                    self.logger.debug(f"Error calculating MACD histogram slope: {e}")
                    
            # Fallback: Simple comparison of first vs last value
            if recent_values[-1] > recent_values[0]:
                return 'RISING'
            elif recent_values[-1] < recent_values[0]:
                return 'DESCENDING'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            self.logger.error(f"Error detecting MACD histogram trend: {e}")
            return 'NEUTRAL'
    
    def validate_macd_momentum(self, df: pd.DataFrame, signal_type: str) -> bool:
        """
        MACD MOMENTUM FILTER: Flexible momentum validation with multiple modes
        
        Validation Modes:
        - strict_blocking: Block signals when MACD momentum opposes signal direction
        - slope_aware: Consider MACD slope but allow weak opposing momentum  
        - neutral_friendly: Only block very strong opposing MACD momentum
        
        Args:
            df: DataFrame with MACD histogram data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if MACD momentum allows signal, False if momentum strongly opposes signal
        """
        try:
            # Check if MACD momentum filter is enabled
            if not getattr(config, 'MACD_MOMENTUM_FILTER_ENABLED', False):
                self.logger.debug("MACD momentum filter disabled, allowing signal")
                return True
            
            self.logger.debug(f"🔍 MACD momentum filter ENABLED - validating {signal_type} signal")
            
            # Get validation mode and sensitivity settings
            validation_mode = getattr(config, 'MACD_VALIDATION_MODE', 'slope_aware')
            sensitivity = getattr(config, 'MACD_TREND_SENSITIVITY', 'permissive')
            sensitivity_settings = getattr(config, 'MACD_SENSITIVITY_SETTINGS', {})
            validation_modes = getattr(config, 'MACD_VALIDATION_MODES', {})
            
            # Use sensitivity settings if available, otherwise fall back to direct config values
            if sensitivity in sensitivity_settings:
                lookback_periods = sensitivity_settings[sensitivity]['lookback']
                self.logger.debug(f"Using MACD {sensitivity} sensitivity: {lookback_periods} periods lookback")
            else:
                lookback_periods = getattr(config, 'MACD_HISTOGRAM_LOOKBACK', 3)
                self.logger.debug(f"Using default MACD lookback: {lookback_periods} periods")
            
            # Detect current MACD histogram trend
            macd_trend = self.detect_macd_histogram_trend(df, lookback_periods)

            # Get recent MACD values to calculate slope strength
            recent_values = df['macd_histogram'].tail(lookback_periods + 1).dropna()
            if len(recent_values) >= 2:
                x = np.arange(len(recent_values))
                slope, _ = np.polyfit(x, recent_values.values, 1)
                slope_strength = abs(slope)

                # Enhanced logging with MACD values
                latest_macd = recent_values.iloc[-1] if len(recent_values) > 0 else 0.0
                previous_macd = recent_values.iloc[-2] if len(recent_values) > 1 else latest_macd
                macd_change = latest_macd - previous_macd

                self.logger.info(f"🔍 MACD VALIDATION for {signal_type} signal:")
                self.logger.info(f"   📊 MACD values: [{', '.join(f'{v:.6f}' for v in recent_values.values)}]")
                self.logger.info(f"   📈 Latest MACD: {latest_macd:.6f}, Change: {macd_change:+.6f}")
                self.logger.info(f"   📉 Trend: {macd_trend}, Slope: {slope:.6f}, Strength: {slope_strength:.6f}")
                self.logger.info(f"   ⚙️ Settings: mode={validation_mode}, sensitivity={sensitivity}, lookback={lookback_periods}")
            else:
                slope_strength = 0.0
                self.logger.warning(f"⚠️ Insufficient MACD data for validation: {len(recent_values)} values")
            
            # Get validation mode settings
            mode_settings = validation_modes.get(validation_mode, validation_modes.get('slope_aware', {}))
            
            # Apply validation logic based on mode
            if validation_mode == 'strict_blocking':
                # Original strict logic - block all opposing momentum
                if signal_type == 'BULL' and macd_trend == 'DESCENDING':
                    self.logger.warning(f"❌ EMA BULL signal REJECTED: MACD histogram descending (strict mode)")
                    self.logger.warning(f"   🔴 CONTRADICTION: BULL signal but MACD momentum is NEGATIVE (slope: {slope:.6f})")
                    self.logger.warning(f"   📉 This prevents trading against momentum - signal blocked for safety")
                    return False
                elif signal_type == 'BEAR' and macd_trend == 'RISING':
                    self.logger.warning(f"❌ EMA BEAR signal REJECTED: MACD histogram rising (strict mode)")
                    self.logger.warning(f"   🔴 CONTRADICTION: BEAR signal but MACD momentum is POSITIVE (slope: {slope:.6f})")
                    self.logger.warning(f"   📈 This prevents trading against momentum - signal blocked for safety")
                    return False
                elif signal_type == 'BULL' and macd_trend in ['RISING', 'NEUTRAL']:
                    self.logger.info(f"✅ EMA BULL signal ALLOWED: MACD momentum is {macd_trend} (slope: {slope:.6f})")
                    self.logger.info(f"   🟢 MOMENTUM ALIGNMENT: BULL signal with compatible MACD direction")
                elif signal_type == 'BEAR' and macd_trend in ['DESCENDING', 'NEUTRAL']:
                    self.logger.info(f"✅ EMA BEAR signal ALLOWED: MACD momentum is {macd_trend} (slope: {slope:.6f})")
                    self.logger.info(f"   🟢 MOMENTUM ALIGNMENT: BEAR signal with compatible MACD direction")
                
            elif validation_mode == 'slope_aware':
                # Only block strong opposing momentum
                weak_threshold = mode_settings.get('weak_threshold', 0.0002)
                
                if signal_type == 'BULL' and macd_trend == 'DESCENDING' and slope_strength > weak_threshold:
                    self.logger.warning(f"❌ EMA BULL signal REJECTED: Strong MACD histogram descent (slope: {slope:.6f})")
                    return False
                elif signal_type == 'BEAR' and macd_trend == 'RISING' and slope_strength > weak_threshold:
                    self.logger.warning(f"❌ EMA BEAR signal REJECTED: Strong MACD histogram rise (slope: {slope:.6f})")
                    return False
                elif signal_type == 'BULL' and macd_trend == 'DESCENDING':
                    self.logger.debug(f"⚠️ EMA BULL signal ALLOWED: Weak MACD descent (slope: {slope:.6f} < {weak_threshold:.6f})")
                elif signal_type == 'BEAR' and macd_trend == 'RISING':
                    self.logger.debug(f"⚠️ EMA BEAR signal ALLOWED: Weak MACD rise (slope: {slope:.6f} < {weak_threshold:.6f})")
                
            elif validation_mode == 'neutral_friendly':
                # Only block very strong opposing momentum
                strong_threshold = mode_settings.get('strong_threshold', 0.0005)
                
                if signal_type == 'BULL' and macd_trend == 'DESCENDING' and slope_strength > strong_threshold:
                    self.logger.warning(f"❌ EMA BULL signal REJECTED: Very strong MACD histogram descent (slope: {slope:.6f})")
                    return False
                elif signal_type == 'BEAR' and macd_trend == 'RISING' and slope_strength > strong_threshold:
                    self.logger.warning(f"❌ EMA BEAR signal REJECTED: Very strong MACD histogram rise (slope: {slope:.6f})")
                    return False
                elif signal_type == 'BULL' and macd_trend == 'DESCENDING':
                    self.logger.debug(f"✅ EMA BULL signal ALLOWED: Moderate MACD descent (slope: {slope:.6f} < {strong_threshold:.6f})")
                elif signal_type == 'BEAR' and macd_trend == 'RISING':
                    self.logger.debug(f"✅ EMA BEAR signal ALLOWED: Moderate MACD rise (slope: {slope:.6f} < {strong_threshold:.6f})")
            
            # If we reach here, signal is allowed
            self.logger.debug(f"✅ MACD momentum OK for {signal_type}: histogram {macd_trend.lower()} (mode: {validation_mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating MACD momentum: {e}")
            return True  # Allow signal on error to avoid blocking valid trades