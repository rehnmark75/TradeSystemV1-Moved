# core/strategies/helpers/ema_breakout_validator.py
"""
Enhanced EMA Breakout Validator
Reduces false breakouts through multi-factor confirmation system
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class EMABreakoutValidator:
    """
    Advanced breakout validation system to reduce false signals
    
    Features:
    - Multi-candle confirmation requirements
    - Volume spike analysis
    - Support/resistance level validation
    - Market condition filtering
    - Volatility-based filtering
    - Price action confirmation
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = 1e-8
        
        # Configuration
        self.confirmation_candles = getattr(config, 'EMA_CONFIRMATION_CANDLES', 2)
        self.volume_spike_threshold = getattr(config, 'EMA_VOLUME_SPIKE_THRESHOLD', 1.5)
        self.pullback_requirement = getattr(config, 'EMA_REQUIRE_PULLBACK', True)
        self.min_breakout_strength = getattr(config, 'EMA_MIN_BREAKOUT_STRENGTH', 0.3)
        
        # Initialize sub-validators
        self._init_sub_validators()
    
    def _init_sub_validators(self):
        """Initialize supporting validator modules"""
        try:
            # Support/Resistance Validator
            from ...detection.support_resistance_validator import SupportResistanceValidator
            self.sr_validator = SupportResistanceValidator()
            self.logger.info("✅ Support/Resistance validator initialized")
        except ImportError:
            self.sr_validator = None
            self.logger.warning("⚠️ Support/Resistance validator not available")
        
        try:
            # Market Conditions Analyzer
            from ...detection.market_conditions import MarketConditionsAnalyzer
            self.market_conditions = MarketConditionsAnalyzer()
            self.logger.info("✅ Market conditions analyzer initialized")
        except ImportError:
            self.market_conditions = None
            self.logger.warning("⚠️ Market conditions analyzer not available")
        
        try:
            # ADX Calculator
            from .adx_calculator import ADXCalculator
            self.adx_calculator = ADXCalculator()
            self.logger.info("✅ ADX calculator initialized")
        except ImportError:
            self.adx_calculator = None
            self.logger.warning("⚠️ ADX calculator not available")
    
    def validate_breakout(self, df: pd.DataFrame, signal_type: str, epic: str) -> Tuple[bool, float, Dict]:
        """
        Main validation method for EMA breakouts
        
        Args:
            df: DataFrame with price and indicator data
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic
            
        Returns:
            (is_valid, confidence_score, validation_details)
        """
        try:
            if len(df) < self.confirmation_candles + 5:
                return False, 0.0, {"error": "insufficient_data"}
            
            validation_details = {
                "multi_candle_confirmation": False,
                "volume_confirmation": False,
                "support_resistance": False,
                "market_conditions": False,
                "trend_strength": False,
                "price_action": False,
                "pullback_test": False,
                "rsi_momentum": False,
                "macd_trend": False,
                "vwap_flow": False
            }
            
            confidence_factors = []
            
            # 1. Multi-Candle Confirmation
            multi_candle_valid, mc_confidence = self._validate_multi_candle_confirmation(df, signal_type)
            validation_details["multi_candle_confirmation"] = multi_candle_valid
            if multi_candle_valid:
                confidence_factors.append(("multi_candle", mc_confidence))
            
            # 2. Volume Confirmation
            if self._has_volume_data(df):
                volume_valid, vol_confidence = self._validate_volume_breakout(df, signal_type)
                validation_details["volume_confirmation"] = volume_valid
                if volume_valid:
                    confidence_factors.append(("volume", vol_confidence))
            
            # 3. Support/Resistance Validation
            if self.sr_validator:
                sr_valid, sr_confidence = self._validate_support_resistance(df, signal_type, epic)
                validation_details["support_resistance"] = sr_valid
                if sr_valid:
                    confidence_factors.append(("support_resistance", sr_confidence))
            
            # 4. Market Conditions
            if self.market_conditions:
                market_valid, market_confidence = self._validate_market_conditions(df, signal_type)
                validation_details["market_conditions"] = market_valid
                if market_valid:
                    confidence_factors.append(("market_conditions", market_confidence))
            
            # 5. Trend Strength (ADX)
            if self.adx_calculator:
                trend_valid, trend_confidence = self._validate_trend_strength(df, signal_type)
                validation_details["trend_strength"] = trend_valid
                if trend_valid:
                    confidence_factors.append(("trend_strength", trend_confidence))
            
            # 6. Price Action Confirmation
            price_action_valid, pa_confidence = self._validate_price_action(df, signal_type)
            validation_details["price_action"] = price_action_valid
            if price_action_valid:
                confidence_factors.append(("price_action", pa_confidence))
            
            # 7. Pullback Test (if enabled)
            if self.pullback_requirement:
                pullback_valid, pb_confidence = self._validate_pullback_retest(df, signal_type)
                validation_details["pullback_test"] = pullback_valid
                if pullback_valid:
                    confidence_factors.append(("pullback", pb_confidence))
            
            # 8. RSI Momentum Confirmation (NEW - High Win Rate Boost)
            rsi_valid, rsi_confidence = self._validate_rsi_momentum(df, signal_type)
            validation_details["rsi_momentum"] = rsi_valid
            if rsi_valid:
                confidence_factors.append(("rsi_momentum", rsi_confidence))
            
            # 9. MACD Trend Validation (ENHANCED)
            macd_valid, macd_confidence = self._validate_enhanced_macd(df, signal_type)
            validation_details["macd_trend"] = macd_valid
            if macd_valid:
                confidence_factors.append(("macd_trend", macd_confidence))
            
            # 10. VWAP-like Institutional Flow Detection (NEW)
            vwap_valid, vwap_confidence = self._validate_institutional_flow(df, signal_type)
            validation_details["vwap_flow"] = vwap_valid
            if vwap_valid:
                confidence_factors.append(("vwap_flow", vwap_confidence))
            
            # Calculate overall validation
            is_valid, final_confidence = self._calculate_final_validation(
                confidence_factors, validation_details
            )
            
            # Log validation results
            self._log_validation_results(signal_type, is_valid, final_confidence, validation_details)
            
            return is_valid, final_confidence, validation_details
            
        except Exception as e:
            self.logger.error(f"Breakout validation error: {e}")
            return False, 0.0, {"error": str(e)}
    
    def _validate_multi_candle_confirmation(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, float]:
        """
        Validate breakout using multiple candles for confirmation
        Prevents entry on single-candle false breakouts
        """
        try:
            if len(df) < self.confirmation_candles + 2:
                return False, 0.0
            
            recent_candles = df.tail(self.confirmation_candles + 1)
            ema_short = recent_candles['ema_short']
            closes = recent_candles['close']
            
            if signal_type == 'BULL':
                # For bull signals, ensure multiple candles stay above EMA
                confirmed_candles = (closes > ema_short + self.eps).sum()
                strength = confirmed_candles / len(recent_candles)
                
                # Progressive movement check (optional based on config)
                if self.confirmation_candles > 1 and len(closes) > 1:
                    distance_progression = (closes.iloc[-1] - ema_short.iloc[-1]) > (closes.iloc[-2] - ema_short.iloc[-2])
                else:
                    distance_progression = True  # Always true for single candle
                
            else:  # BEAR
                # For bear signals, ensure multiple candles stay below EMA
                confirmed_candles = (closes < ema_short - self.eps).sum()
                strength = confirmed_candles / len(recent_candles)
                
                # Progressive movement check (optional based on config)
                if self.confirmation_candles > 1 and len(closes) > 1:
                    distance_progression = (ema_short.iloc[-1] - closes.iloc[-1]) > (ema_short.iloc[-2] - closes.iloc[-2])
                else:
                    distance_progression = True  # Always true for single candle
            
            # More permissive requirements for forex
            min_strength_required = 0.5 if self.confirmation_candles > 1 else 0.8  # 50% for multi-candle, 80% for single
            
            is_valid = (strength >= min_strength_required)
            if self.confirmation_candles > 1:
                is_valid = is_valid and distance_progression  # Only require progression for multi-candle
                
            confidence = min(strength * 1.1 if distance_progression else strength, 1.0)
            
            return is_valid, confidence
            
        except Exception as e:
            self.logger.error(f"Multi-candle validation error: {e}")
            return False, 0.0
    
    def _validate_volume_breakout(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, float]:
        """
        Validate breakout using volume analysis
        Strong breakouts should have volume spikes
        """
        try:
            # Try different volume column names
            volume_col = None
            for col in ['volume', 'ltv', 'tick_volume']:
                if col in df.columns:
                    volume_col = col
                    break
            
            if volume_col is None:
                return True, 0.5  # Neutral if no volume data
            
            recent_volumes = df[volume_col].tail(10)
            current_volume = recent_volumes.iloc[-1]
            avg_volume = recent_volumes.iloc[:-1].mean()
            
            if avg_volume == 0:
                return True, 0.5
            
            volume_ratio = current_volume / avg_volume
            
            # Volume spike indicates strong breakout
            if volume_ratio >= self.volume_spike_threshold:
                confidence = min(volume_ratio / 3.0, 1.0)  # Scale to 0-1
                return True, confidence
            
            # Low volume breakouts are suspicious
            elif volume_ratio < 0.7:
                return False, 0.0
            
            # Normal volume - neutral
            return True, 0.6
            
        except Exception as e:
            self.logger.error(f"Volume validation error: {e}")
            return True, 0.5
    
    def _validate_support_resistance(self, df: pd.DataFrame, signal_type: str, epic: str) -> Tuple[bool, float]:
        """
        Validate breakout against support/resistance levels
        Avoid breakouts at weak levels
        """
        try:
            if not self.sr_validator:
                return True, 0.5
            
            # Create a signal dict for the validator
            current_price = df['close'].iloc[-1]
            signal = {
                'signal_type': signal_type,
                'price': current_price,
                'epic': epic
            }
            
            # Use the support/resistance validator
            is_valid, reason, details = self.sr_validator.validate_trade_direction(signal, df, epic)
            
            if is_valid:
                # Good breakout at significant level
                confidence = 0.8
                if 'strong' in reason.lower():
                    confidence = 0.9
                return True, confidence
            else:
                # Bad breakout - against significant level
                if 'major' in reason.lower() or 'strong' in reason.lower():
                    return False, 0.0
                else:
                    # Minor issue - reduce confidence but don't reject
                    return True, 0.3
            
        except Exception as e:
            self.logger.error(f"Support/resistance validation error: {e}")
            return True, 0.5
    
    def _validate_market_conditions(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, float]:
        """
        Validate breakout based on market conditions
        Trending markets are better for breakouts
        """
        try:
            # Simple market regime detection using EMA alignment and price action
            if len(df) < 20:
                return True, 0.5
            
            recent = df.tail(20)
            
            # Check EMA alignment strength
            if 'ema_short' in recent.columns and 'ema_long' in recent.columns and 'ema_trend' in recent.columns:
                ema_short = recent['ema_short'].iloc[-1]
                ema_long = recent['ema_long'].iloc[-1]
                ema_trend = recent['ema_trend'].iloc[-1]
                current_price = recent['close'].iloc[-1]
                
                # Calculate EMA separation as percentage
                if current_price > 0:
                    short_long_sep = abs(ema_short - ema_long) / current_price
                    long_trend_sep = abs(ema_long - ema_trend) / current_price
                    price_trend_sep = abs(current_price - ema_trend) / current_price
                    
                    # Strong trending conditions
                    if (short_long_sep > 0.001 and long_trend_sep > 0.002 and price_trend_sep > 0.003):
                        return True, 0.9  # Strong trending market
                    elif (short_long_sep > 0.0005 and long_trend_sep > 0.001):
                        return True, 0.7  # Moderate trending market
                    elif (short_long_sep < 0.0002 and long_trend_sep < 0.0005):
                        return False, 0.0  # Ranging market - bad for breakouts
            
            # Check price volatility using recent range
            highs = recent['high'].tail(10)
            lows = recent['low'].tail(10)
            closes = recent['close'].tail(10)
            
            avg_range = (highs - lows).mean()
            recent_range = closes.max() - closes.min()
            
            if avg_range > 0:
                volatility_ratio = recent_range / avg_range
                
                # High volatility can indicate trending market
                if volatility_ratio > 2.0:
                    return True, 0.8  # High volatility trending
                elif volatility_ratio < 0.5:
                    return False, 0.0  # Low volatility ranging
            
            # Default to neutral
            return True, 0.6
            
        except Exception as e:
            self.logger.error(f"Market conditions validation error: {e}")
            return True, 0.5
    
    def _validate_trend_strength(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, float]:
        """
        Validate trend strength using ADX
        Strong trends are better for breakouts
        """
        try:
            if not self.adx_calculator:
                return True, 0.6  # Neutral if no ADX available
            
            # Calculate ADX if not present
            df_work = df.copy()
            if 'adx' not in df_work.columns:
                try:
                    df_work = self.adx_calculator.calculate_adx(df_work)
                    if 'adx' not in df_work.columns:
                        self.logger.debug("ADX calculation failed - returning neutral")
                        return True, 0.6
                except Exception as e:
                    self.logger.debug(f"ADX calculation error: {e} - returning neutral")
                    return True, 0.6
            
            current_adx = df_work['adx'].iloc[-1]
            
            # Make ADX validation more permissive for forex
            if current_adx >= 20:  # Lowered from 25
                confidence = min((current_adx - 10) / 30.0, 1.0)  # Scale 20-40 to 0.33-1.0
                return True, max(confidence, 0.6)
            elif current_adx >= 15:  # Moderate trend - still allow
                return True, 0.5
            else:
                # Very weak trend - still allow but with low confidence
                return True, 0.3
            
        except Exception as e:
            self.logger.debug(f"Trend strength validation error: {e} - returning neutral")
            return True, 0.6
    
    def _validate_price_action(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, float]:
        """
        Validate using basic price action patterns
        Look for confirming candle patterns
        """
        try:
            if len(df) < 3:
                return True, 0.5
            
            recent = df.tail(3)
            current = recent.iloc[-1]
            previous = recent.iloc[-2]
            
            open_price = current['open']
            close_price = current['close']
            high_price = current['high']
            low_price = current['low']
            
            candle_body = abs(close_price - open_price)
            candle_range = high_price - low_price
            
            # Avoid tiny candles (low conviction)
            if candle_range == 0 or candle_body / candle_range < 0.3:
                return False, 0.0
            
            if signal_type == 'BULL':
                # Look for strong bullish candles
                is_bullish = close_price > open_price
                strong_close = (high_price - close_price) < (close_price - low_price)  # Close near high
                
                if is_bullish and strong_close:
                    confidence = min(candle_body / candle_range, 1.0)
                    return True, confidence
            
            else:  # BEAR
                # Look for strong bearish candles
                is_bearish = close_price < open_price
                strong_close = (close_price - low_price) < (high_price - close_price)  # Close near low
                
                if is_bearish and strong_close:
                    confidence = min(candle_body / candle_range, 1.0)
                    return True, confidence
            
            return True, 0.4  # Neutral price action
            
        except Exception as e:
            self.logger.error(f"Price action validation error: {e}")
            return True, 0.5
    
    def _validate_pullback_retest(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, float]:
        """
        Validate pullback/retest behavior
        Better breakouts often show pullback and hold
        """
        try:
            if len(df) < 5:
                return True, 0.5  # Not enough data
            
            recent = df.tail(5)
            ema_short = recent['ema_short']
            closes = recent['close']
            
            if signal_type == 'BULL':
                # Look for pullback to EMA that holds
                touched_ema = ((closes - ema_short).abs() <= 0.5 * ema_short * 0.0001).any()
                held_above = (closes.iloc[-2:] > ema_short.iloc[-2:]).all()
                
                if touched_ema and held_above:
                    return True, 0.9  # Strong pullback confirmation
                elif held_above:
                    return True, 0.7  # Stayed above EMA
                
            else:  # BEAR
                # Look for pullback to EMA that fails
                touched_ema = ((closes - ema_short).abs() <= 0.5 * ema_short * 0.0001).any()
                held_below = (closes.iloc[-2:] < ema_short.iloc[-2:]).all()
                
                if touched_ema and held_below:
                    return True, 0.9  # Strong pullback confirmation
                elif held_below:
                    return True, 0.7  # Stayed below EMA
            
            return True, 0.5  # No clear pullback pattern
            
        except Exception as e:
            self.logger.error(f"Pullback validation error: {e}")
            return True, 0.5
    
    def _has_volume_data(self, df: pd.DataFrame) -> bool:
        """Check if volume data is available"""
        volume_cols = ['volume', 'ltv', 'tick_volume']
        return any(col in df.columns for col in volume_cols)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using pandas - efficient implementation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + self.eps)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _validate_rsi_momentum(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, float]:
        """
        RSI Momentum Confirmation - MAJOR WIN RATE BOOST
        Based on research: RSI overbought/oversold provides excellent confirmation
        """
        try:
            if len(df) < 20:  # Need enough data for RSI
                return True, 0.5
            
            # Calculate RSI
            rsi = self._calculate_rsi(df['close'], period=14)
            current_rsi = rsi.iloc[-1]
            
            if pd.isna(current_rsi):
                return True, 0.5
            
            if signal_type == 'BULL':
                # BALANCED MODE: Good RSI conditions for bullish signals
                if 20 <= current_rsi <= 40:
                    return True, 0.90  # Excellent oversold bounce zone
                elif 40 < current_rsi <= 60:
                    return True, 0.80  # Good momentum building  
                elif 60 < current_rsi <= 70:
                    return True, 0.65  # Acceptable momentum
                elif current_rsi < 20:
                    return True, 0.85  # Very oversold - good bounce potential
                else:
                    return False, 0.0  # Too overbought - reject
            
            else:  # BEAR
                # BALANCED MODE: Good RSI conditions for bearish signals
                if 60 <= current_rsi <= 80:
                    return True, 0.90  # Excellent overbought reversal zone
                elif 40 <= current_rsi < 60:
                    return True, 0.80  # Good bearish momentum
                elif 30 <= current_rsi < 40:
                    return True, 0.65  # Acceptable momentum
                elif current_rsi > 80:
                    return True, 0.85  # Very overbought - good reversal potential
                else:
                    return False, 0.0  # Too oversold - reject
                    
        except Exception as e:
            self.logger.error(f"RSI momentum validation error: {e}")
            return True, 0.5
    
    def _validate_enhanced_macd(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, float]:
        """
        Enhanced MACD Trend Validation - MAJOR WIN RATE BOOST
        Uses MACD histogram and crossover for momentum confirmation
        """
        try:
            # Check if MACD data is available
            if not all(col in df.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
                return True, 0.5
            
            if len(df) < 5:
                return True, 0.5
            
            recent = df.tail(5)
            macd_line = recent['macd_line']
            macd_signal = recent['macd_signal']
            macd_hist = recent['macd_histogram']
            
            current_hist = macd_hist.iloc[-1]
            prev_hist = macd_hist.iloc[-2]
            
            if signal_type == 'BULL':
                # BALANCED MODE: Good MACD conditions for bullish signals
                bullish_crossover = (macd_line.iloc[-1] > macd_signal.iloc[-1] and 
                                   macd_line.iloc[-2] <= macd_signal.iloc[-2])
                positive_histogram = current_hist > 0
                increasing_momentum = current_hist > prev_hist
                strong_momentum = current_hist > prev_hist * 1.2
                
                if bullish_crossover and strong_momentum:
                    return True, 0.90  # Excellent: Fresh crossover with strong momentum
                elif bullish_crossover and increasing_momentum:
                    return True, 0.85  # Good: Fresh crossover with momentum
                elif positive_histogram and strong_momentum:
                    return True, 0.80  # Good: Strong bullish momentum
                elif positive_histogram and increasing_momentum:
                    return True, 0.75  # Acceptable: Bullish momentum building
                elif positive_histogram:
                    return True, 0.65  # Basic: Bullish but weak momentum
                else:
                    return False, 0.0  # Bearish MACD - reject
                    
            else:  # BEAR
                # BALANCED MODE: Good MACD conditions for bearish signals
                bearish_crossover = (macd_line.iloc[-1] < macd_signal.iloc[-1] and 
                                   macd_line.iloc[-2] >= macd_signal.iloc[-2])
                negative_histogram = current_hist < 0
                decreasing_momentum = current_hist < prev_hist
                strong_momentum = current_hist < prev_hist * 1.2
                
                if bearish_crossover and strong_momentum:
                    return True, 0.90  # Excellent: Fresh crossover with strong momentum
                elif bearish_crossover and decreasing_momentum:
                    return True, 0.85  # Good: Fresh crossover with momentum
                elif negative_histogram and strong_momentum:
                    return True, 0.80  # Good: Strong bearish momentum
                elif negative_histogram and decreasing_momentum:
                    return True, 0.75  # Acceptable: Bearish momentum building
                elif negative_histogram:
                    return True, 0.65  # Basic: Bearish but weak momentum
                else:
                    return False, 0.0  # Bullish MACD - reject
                    
        except Exception as e:
            self.logger.error(f"Enhanced MACD validation error: {e}")
            return True, 0.5
    
    def _validate_institutional_flow(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, float]:
        """
        VWAP-like Institutional Flow Detection - WIN RATE BOOST
        Uses volume-weighted price analysis to detect institutional participation
        """
        try:
            if len(df) < 10:
                return True, 0.5
            
            # Use volume if available, otherwise use typical price volume proxy
            if self._has_volume_data(df):
                volume_col = None
                for col in ['volume', 'ltv', 'tick_volume']:
                    if col in df.columns:
                        volume_col = col
                        break
                        
                if volume_col:
                    recent = df.tail(10)
                    volume = recent[volume_col]
                    typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
                    
                    # Calculate volume-weighted average price (VWAP proxy)
                    vwap = (typical_price * volume).sum() / volume.sum()
                    current_price = df['close'].iloc[-1]
                    
                    price_vs_vwap = (current_price - vwap) / vwap
                    
                    if signal_type == 'BULL':
                        # Bullish when price is close to or above VWAP with volume
                        if price_vs_vwap >= 0.001:  # 0.1% above VWAP
                            return True, 0.8  # Strong institutional buying
                        elif price_vs_vwap >= -0.0005:  # Near VWAP
                            return True, 0.7  # Neutral to bullish
                        else:
                            return True, 0.4  # Below VWAP - weak
                            
                    else:  # BEAR
                        # Bearish when price is below VWAP with volume
                        if price_vs_vwap <= -0.001:  # 0.1% below VWAP
                            return True, 0.8  # Strong institutional selling
                        elif price_vs_vwap <= 0.0005:  # Near VWAP
                            return True, 0.7  # Neutral to bearish
                        else:
                            return True, 0.4  # Above VWAP - weak
            
            # Fallback: Use price momentum and volatility as proxy
            recent = df.tail(10)
            price_momentum = (recent['close'].iloc[-1] / recent['close'].iloc[-5] - 1)
            volatility = recent['high'].std() / recent['close'].mean()
            
            # Higher volatility with strong momentum suggests institutional flow
            if signal_type == 'BULL' and price_momentum > 0.005 and volatility > 0.01:
                return True, 0.7
            elif signal_type == 'BEAR' and price_momentum < -0.005 and volatility > 0.01:
                return True, 0.7
            else:
                return True, 0.5
                
        except Exception as e:
            self.logger.error(f"Institutional flow validation error: {e}")
            return True, 0.5
    
    def _calculate_final_validation(self, confidence_factors: List[Tuple[str, float]], 
                                   validation_details: Dict) -> Tuple[bool, float]:
        """
        Calculate final validation result based on all factors
        """
        if not confidence_factors:
            return False, 0.0
        
        # Weight different factors - OPTIMIZED FOR HIGH WIN RATE
        factor_weights = {
            "multi_candle": 0.20,           # Reduced - still important but not dominant
            "support_resistance": 0.15,     # Key levels still crucial
            "rsi_momentum": 0.20,           # NEW - Major win rate boost 
            "macd_trend": 0.15,             # ENHANCED - Momentum confirmation
            "volume": 0.10,                 # Volume confirmation
            "market_conditions": 0.08,      # Market regime
            "vwap_flow": 0.07,              # NEW - Institutional flow
            "price_action": 0.03,           # Basic candle patterns
            "trend_strength": 0.02          # ADX - less reliable in forex
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor_name, confidence in confidence_factors:
            weight = factor_weights.get(factor_name, 0.1)
            weighted_score += confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            final_confidence = weighted_score / total_weight
        else:
            final_confidence = 0.0
        
        # Require minimum confidence threshold
        min_threshold = getattr(config, 'EMA_ENHANCED_MIN_CONFIDENCE', 0.6)
        is_valid = final_confidence >= min_threshold
        
        # Core requirements - ENHANCED FOR HIGH WIN RATE
        # Prioritize the most effective validations based on research
        has_rsi_momentum = validation_details.get("rsi_momentum", False)
        has_macd_trend = validation_details.get("macd_trend", False)
        has_multi_candle = validation_details.get("multi_candle_confirmation", False)
        has_support_resistance = validation_details.get("support_resistance", False)
        has_vwap_flow = validation_details.get("vwap_flow", False)
        
        # HIGH WIN RATE REQUIREMENTS: 
        # 1. Must meet confidence threshold AND
        # 2. Must have at least TWO high-value confirmations OR one critical confirmation
        high_value_confirmations = sum([
            has_rsi_momentum,      # Most important - momentum confirmation
            has_macd_trend,        # Second most important - trend validation  
            has_multi_candle,      # Third - false breakout reduction
            has_support_resistance # Fourth - key levels
        ])
        
        has_institutional_flow = has_vwap_flow  # Bonus factor
        
        # BALANCED HIGH-QUALITY MODE: 2-3 signals per day (14-21 per week)
        # Goal: 50%+ win rate with good signal frequency
        
        # SIGNAL QUALITY REQUIREMENTS:
        strong_momentum = has_rsi_momentum and has_macd_trend  # Both momentum indicators agree
        good_structure = has_support_resistance or has_multi_candle  # Structure OR timing
        has_flow = has_institutional_flow  # Institutional participation bonus
        strong_core = high_value_confirmations >= 3  # 3+ core validations pass
        decent_core = high_value_confirmations >= 2  # 2+ core validations pass
        
        # TIER 1: EXCELLENT SIGNALS (Target: 75%+ confidence)
        tier1_excellent = (
            (strong_momentum and good_structure and final_confidence >= 0.75) or
            (strong_core and (has_rsi_momentum or has_macd_trend) and final_confidence >= 0.76)
        )
        
        # TIER 2: GOOD SIGNALS (Target: 72%+ confidence) 
        tier2_good = (
            (strong_momentum and final_confidence >= 0.72) or
            (decent_core and (has_rsi_momentum or has_macd_trend) and good_structure and final_confidence >= 0.72) or
            (strong_core and final_confidence >= 0.70)
        )
        
        # TIER 3: ACCEPTABLE SIGNALS (Target: 70%+ confidence with bonus)
        tier3_acceptable = (
            decent_core and (has_rsi_momentum or has_macd_trend) and final_confidence >= 0.70 and has_flow
        )
        
        # BALANCED MODE: Accept TIER 1, TIER 2, or TIER 3
        balanced_validation_passed = tier1_excellent or tier2_good or tier3_acceptable
        
        # BASIC REQUIREMENTS:
        # 1. Must meet minimum confidence (adjusted for backtest mode)
        confidence_threshold = 0.60 if getattr(self, 'backtest_mode', False) else 0.70
        confidence_acceptable = final_confidence >= confidence_threshold
        
        # 2. Must have at least some momentum confirmation
        has_momentum_confirmation = has_rsi_momentum or has_macd_trend
        
        # FINAL BALANCED REQUIREMENTS
        if balanced_validation_passed and confidence_acceptable and has_momentum_confirmation:
            core_requirements_met = is_valid
        else:
            core_requirements_met = False  # Still selective but not extreme
        
        return core_requirements_met, final_confidence
    
    def _log_validation_results(self, signal_type: str, is_valid: bool, confidence: float, details: Dict):
        """Log validation results for debugging"""
        status = "✅ VALID" if is_valid else "❌ REJECTED"
        self.logger.info(f"🔍 Enhanced Breakout Validation - {signal_type}: {status} (confidence: {confidence:.1%})")
        
        for check, result in details.items():
            if isinstance(result, bool):
                symbol = "✅" if result else "❌"
                self.logger.debug(f"   {check}: {symbol}")