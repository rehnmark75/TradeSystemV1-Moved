# core/strategies/scalping_strategy.py
"""
Scalping Strategy Implementation with Comprehensive Enhancement
Fast EMA crossover strategy optimized for 1-5 minute timeframes
ADDED: Comprehensive data enhancement and timestamp safety
🔥 UPDATED: Enhanced confidence validation to prevent bad signals with artificial confidence
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime
import hashlib
import json

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
try:
    import config
except ImportError:
    from forex_scanner import config


class ScalpingStrategy(BaseStrategy):
    """Ultra-fast scalping strategy with rapid EMA crossovers, comprehensive enhancement, and enhanced confidence validation"""
    
    def __init__(self, scalping_mode: str = 'aggressive'):
        super().__init__(f'scalping_{scalping_mode}')
        self.price_adjuster = PriceAdjuster()
        self.scalping_mode = scalping_mode
        
        # Scalping-specific EMA configurations
        self.scalping_configs = {
            'ultra_fast': {'fast': 3, 'slow': 8, 'filter': 21},      # 1m timeframe
            'aggressive': {'fast': 5, 'slow': 13, 'filter': 50},    # 1-5m timeframe  
            'conservative': {'fast': 8, 'slow': 20, 'filter': 50},  # 5m timeframe
            'dual_ma': {'fast': 7, 'slow': 14, 'filter': None}      # Simple dual MA
        }
        
        self.config = self.scalping_configs.get(scalping_mode, self.scalping_configs['aggressive'])
        
        # Scalping-specific settings
        self.min_separation_pips = 0.5  # Minimum EMA separation
        self.volume_threshold = 1.2     # Minimum volume multiplier
        self.max_spread_pips = 2.0      # Maximum allowed spread
        self.quick_exit_enabled = True  # Enable rapid exit signals
        
        self.logger.info(f"🏃 Scalping strategy initialized: {scalping_mode} with comprehensive enhancement and enhanced confidence validation")
        self.logger.info(f"   EMAs: {self.config['fast']}/{self.config['slow']}/{self.config.get('filter', 'None')}")
        self.logger.info(f"   ✅ Timestamp safety: enabled")
        self.logger.info(f"   ✅ Comprehensive enhancement: enabled")
        self.logger.info(f"   🔥 Enhanced confidence validation: enabled")

    def _convert_market_timestamp_safe(self, timestamp_value) -> Optional[datetime]:
        """
        TIMESTAMP FIX: SAFELY convert various timestamp formats to datetime object
        FIXES: market_timestamp integer conversion error (like 429 -> None)
        """
        if timestamp_value is None:
            return None

        try:
            # Case 1: Already a datetime object
            if isinstance(timestamp_value, datetime):
                return timestamp_value

            # Case 2: String timestamp (ISO format)
            if isinstance(timestamp_value, str):
                # Handle ISO format with timezone
                if 'T' in timestamp_value:
                    return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                # Handle simple date/time strings
                else:
                    return datetime.fromisoformat(timestamp_value)

            # Case 3: Pandas timestamp
            if hasattr(timestamp_value, 'to_pydatetime'):
                return timestamp_value.to_pydatetime()

            # Case 4: Integer or float (Unix timestamp) - including numpy types
            # Convert numpy types to Python native types first
            if hasattr(timestamp_value, 'item'):  # numpy.int64, numpy.float64, etc.
                timestamp_value = timestamp_value.item()

            if isinstance(timestamp_value, (int, float)):
                # Unix timestamps are typically > 946684800 (year 2000)
                # Values like 193, 429 are clearly not Unix timestamps (would be 1970 + seconds)
                if timestamp_value < 946684800:  # Before year 2000
                    # This is not a valid timestamp - likely an index or counter
                    # Silently return None without warning (these are expected)
                    return None
                elif timestamp_value <= 4102444800:  # Before 2100-01-01
                    return datetime.fromtimestamp(timestamp_value)
                else:
                    # FIXES: Invalid integer timestamp - silently return None
                    return None

            # Case 5: Unknown type - silently return None
            # (Reduced logging to avoid spam for non-timestamp fields)
            return None

        except Exception as e:
            # Only log actual errors during conversion
            return None
    
    def get_required_indicators(self) -> List[str]:
        """Required indicators for scalping strategy"""
        indicators = [
            f'ema_{self.config["fast"]}',
            f'ema_{self.config["slow"]}',
            'close', 'high', 'low', 'ltv'
        ]
        
        if self.config.get('filter'):
            indicators.append(f'ema_{self.config["filter"]}')
        
        # Add RSI for overbought/oversold confirmation
        indicators.extend(['rsi_2', 'rsi_14'])
        
        # Add Bollinger Bands for volatility
        indicators.extend(['bb_upper', 'bb_lower', 'bb_middle'])
        
        return indicators

    # ================================================================================
    # 🔥 ADAPTIVE MARKET REGIME DETECTION SYSTEM
    # ================================================================================

    def _detect_market_regime(self, df: pd.DataFrame) -> Dict:
        """
        🔥 NEW: Detect current market regime for adaptive indicator selection

        Returns dict with:
        - regime: 'trending', 'ranging', 'volatile', 'quiet'
        - trend_strength: 0-100 (ADX-based)
        - volatility: 'high', 'medium', 'low' (ATR-based)
        - market_phase: 'expansion', 'contraction' (BB width)
        - recommended_indicators: List of optimal indicators for this regime
        """
        latest = df.iloc[-1]

        # Calculate ADX for trend strength (if not already in df)
        adx = self._calculate_adx(df) if 'adx' not in df.columns else latest.get('adx', 20)

        # Calculate ATR for volatility
        atr_20 = df['close'].rolling(20).apply(lambda x: x.max() - x.min()).iloc[-1]
        atr_avg = df['close'].rolling(20).apply(lambda x: x.max() - x.min()).mean()
        volatility_ratio = atr_20 / atr_avg if atr_avg > 0 else 1.0

        # Calculate Bollinger Band width for market phase
        bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle'] if 'bb_upper' in df.columns else 0.02
        bb_avg = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']).tail(20).mean()

        # Determine regime
        regime = self._classify_regime(adx, volatility_ratio, bb_width, bb_avg)

        # Select optimal indicators for this regime
        recommended_indicators = self._select_indicators_for_regime(regime)

        return {
            'regime': regime['type'],
            'trend_strength': adx,
            'volatility': regime['volatility'],
            'market_phase': regime['phase'],
            'recommended_indicators': recommended_indicators,
            'confidence_multiplier': regime['confidence_multiplier'],
            'adx': adx,
            'volatility_ratio': volatility_ratio,
            'bb_width': bb_width
        }

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index) for trend strength"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']

            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = -low.diff()

            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Smooth DM and TR
            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean().iloc[-1]

            return adx if not pd.isna(adx) else 20.0
        except:
            return 20.0  # Default neutral value

    def _calculate_linda_macd(self, df: pd.DataFrame) -> Dict:
        """
        🔥 LINDA RASCHKE MACD: Calculate 3-10-16 oscillator with SMA (not EMA!)

        Settings:
        - Fast MA: 3 periods (SMA)
        - Slow MA: 10 periods (SMA)
        - Signal: 16 periods (SMA)

        This is MUCH faster than standard 12-26-9 EMA MACD
        Designed specifically for scalping and intraday trading

        Returns dict with:
        - macd_line: The main MACD line (3 SMA - 10 SMA)
        - signal_line: The signal line (16 SMA of MACD)
        - histogram: MACD line - signal line
        - macd_line_prev: Previous MACD line value
        - signal_line_prev: Previous signal line value
        - histogram_prev: Previous histogram value
        """
        try:
            close = df['close']

            # Calculate MACD line: 3 SMA - 10 SMA (NOT EMA!)
            sma_3 = close.rolling(window=3).mean()
            sma_10 = close.rolling(window=10).mean()
            macd_line_series = sma_3 - sma_10

            # Calculate Signal line: 16 SMA of MACD line (NOT EMA!)
            signal_line_series = macd_line_series.rolling(window=16).mean()

            # Calculate histogram
            histogram_series = macd_line_series - signal_line_series

            # Get current and previous values
            macd_line = macd_line_series.iloc[-1]
            signal_line = signal_line_series.iloc[-1]
            histogram = histogram_series.iloc[-1]

            macd_line_prev = macd_line_series.iloc[-2] if len(macd_line_series) > 1 else macd_line
            signal_line_prev = signal_line_series.iloc[-2] if len(signal_line_series) > 1 else signal_line
            histogram_prev = histogram_series.iloc[-2] if len(histogram_series) > 1 else histogram

            # Check for NaN values
            if pd.isna(macd_line) or pd.isna(signal_line):
                return None

            return {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'macd_line_prev': macd_line_prev,
                'signal_line_prev': signal_line_prev,
                'histogram_prev': histogram_prev,
                'macd_slope': macd_line - macd_line_prev,
                'signal_slope': signal_line - signal_line_prev,
                'macd_series': macd_line_series,  # For Anti pattern detection
                'signal_series': signal_line_series
            }
        except Exception as e:
            self.logger.debug(f"Linda MACD calculation failed: {e}")
            return None

    def _classify_regime(self, adx: float, volatility_ratio: float, bb_width: float, bb_avg: float) -> Dict:
        """
        Classify market regime based on multiple factors

        Regimes:
        - TRENDING: ADX > 25, clear directional movement
        - RANGING: ADX < 20, sideways movement
        - VOLATILE: High ATR, fast price swings
        - QUIET: Low ATR, low volume, tight ranges
        """
        # Determine trend strength
        if adx > 30:
            regime_type = 'strong_trending'
            confidence_mult = 1.2  # Higher confidence in trending markets
        elif adx > 20:
            regime_type = 'trending'
            confidence_mult = 1.1
        elif adx < 15:
            regime_type = 'ranging'
            confidence_mult = 0.9  # Lower confidence in choppy markets
        else:
            regime_type = 'weak_trending'
            confidence_mult = 1.0

        # Determine volatility
        if volatility_ratio > 1.5:
            volatility = 'high'
            if regime_type == 'ranging':
                regime_type = 'volatile_ranging'  # Choppy, avoid
                confidence_mult = 0.7
        elif volatility_ratio > 1.2:
            volatility = 'medium'
        else:
            volatility = 'low'
            if regime_type != 'ranging':
                regime_type = 'quiet_trending'  # Good for scalping
                confidence_mult = 1.15

        # Determine market phase
        if bb_width > bb_avg * 1.3:
            phase = 'expansion'  # Breakout phase
        elif bb_width < bb_avg * 0.7:
            phase = 'contraction'  # Squeeze, expect breakout
        else:
            phase = 'normal'

        return {
            'type': regime_type,
            'volatility': volatility,
            'phase': phase,
            'confidence_multiplier': confidence_mult
        }

    def _select_indicators_for_regime(self, regime: Dict) -> List[str]:
        """
        Select optimal indicators based on market regime

        Returns list of indicator groups to use:
        - 'macd': Linda Raschke MACD 3-10-16 for trending markets (ADX > 15)
        - 'ema': EMA crossovers for trend following
        - 'rsi': RSI directional (momentum) for ranging
        - 'bb': Bollinger Bands for support/resistance
        - 'volume': Volume confirmation
        """
        regime_type = regime['type']
        indicators = []

        if 'trending' in regime_type:
            # Trending markets (ADX > 20): Use Linda Raschke MACD 3-10-16
            indicators.extend(['macd', 'ema'])
            if regime['volatility'] == 'high':
                indicators.append('volume')  # Confirm with volume

        elif regime_type == 'ranging':
            # Ranging markets (ADX < 15): Use RSI momentum + BB support/resistance
            indicators.extend(['rsi', 'bb'])
            indicators.append('volume')  # Volume confirmation

        elif regime_type == 'volatile_ranging':
            # Choppy market: Avoid or use very conservative signals
            indicators.extend(['rsi', 'bb', 'volume'])  # Require strong confluence

        else:  # quiet_trending or weak_trending (ADX 15-20)
            # 🔥 NEW: Weak trending still has directional bias, use Linda Raschke MACD
            indicators.extend(['macd', 'ema'])

        return indicators

    def detect_signal(
        self,
        df: pd.DataFrame,
        epic: str,
        spread_pips: float = 1.5,
        timeframe: str = '1m'
    ) -> Optional[Dict]:
        """
        🔥 ADAPTIVE: Detect scalping signals with dynamic indicator selection

        Now intelligently adapts to market conditions:
        - Detects market regime (trending/ranging/volatile)
        - Selects optimal indicators for current conditions
        - Adjusts confidence based on regime suitability
        """
        # 🔥 NEW: Detect market regime first
        regime_info = self._detect_market_regime(df)

        self.logger.info(f"[ADAPTIVE] {epic} Regime: {regime_info['regime']} | "
                        f"ADX: {regime_info['trend_strength']:.1f} | "
                        f"Vol: {regime_info['volatility']} | "
                        f"Indicators: {regime_info['recommended_indicators']}")

        # Skip if market is too choppy
        if regime_info['regime'] == 'volatile_ranging':
            self.logger.debug(f"🚫 {epic} Skipping volatile ranging market (choppy conditions)")
            return None
        # Check spread - crucial for scalping profitability
        if spread_pips > self.max_spread_pips:
            self.logger.debug(f"🚫 Spread too wide for scalping: {spread_pips} > {self.max_spread_pips}")
            return None
        
        # Ensure we have required indicators
        df_enhanced = self._ensure_scalping_indicators(df)
        
        if not self.validate_data(df_enhanced) or len(df_enhanced) < 20:
            return None
        
        # Price adjustment for BID/ASK spread
        if config.USE_BID_ADJUSTMENT:
            df_adjusted = self.price_adjuster.adjust_bid_to_mid_prices(df_enhanced, spread_pips)
        else:
            df_adjusted = df_enhanced.copy()
        
        # Get latest data
        latest = df_adjusted.iloc[-1]
        previous = df_adjusted.iloc[-2]
        
        current_price = latest['close']
        fast_ema = latest[f'ema_{self.config["fast"]}']
        slow_ema = latest[f'ema_{self.config["slow"]}']
        fast_ema_prev = previous[f'ema_{self.config["fast"]}']
        slow_ema_prev = previous[f'ema_{self.config["slow"]}']
        
        # 🔥 ADAPTIVE: Use regime-appropriate detection method
        recommended_indicators = regime_info['recommended_indicators']

        if 'macd' in recommended_indicators:
            # 🔥 LINDA RASCHKE: Trending market uses MACD 3-10-16 signals (NOT EMA crossovers!)
            # This generates WAY more signals: zero crosses, signal crosses, momentum, Anti pattern
            signal = self._detect_linda_macd_signals(latest, previous, df_adjusted, epic, timeframe)
        elif 'rsi' in recommended_indicators and 'bb' in recommended_indicators:
            # Ranging market: Use RSI + BB mean reversion strategy
            signal = self._detect_ranging_signal(latest, previous, df_adjusted, epic, timeframe)
        else:
            # Balanced/weak trending: Use standard EMA crossover
            signal = self._detect_standard_scalping_signal(latest, previous, epic, timeframe)

        # Apply regime-based confidence adjustment
        if signal:
            signal['regime'] = regime_info['regime']
            signal['regime_confidence_mult'] = regime_info['confidence_multiplier']
            signal['original_confidence'] = signal.get('confidence_score', 0.5)
        
        # 🔥 ENHANCED CONFIDENCE VALIDATION: Apply enhanced validation instead of old method
        if signal:
            # Create enhanced signal data for proper confidence calculation
            enhanced_signal_data = self._create_enhanced_signal_data(signal, latest, previous, df_adjusted, epic, timeframe)
            enhanced_confidence = self.calculate_confidence(enhanced_signal_data)
            
            # Update signal with enhanced confidence
            signal['confidence_score'] = enhanced_confidence
            
            # Apply minimum confidence filter with SCALPING-SPECIFIC threshold
            # Scalping uses lower threshold (45%) vs general signals (60%)
            min_confidence = getattr(config, 'SCALPING_MIN_CONFIDENCE', 0.45)
            if enhanced_confidence < min_confidence:
                self.logger.debug(f"🚫 [SCALPING] Signal confidence {enhanced_confidence:.1%} below scalping threshold {min_confidence:.1%}")
                return None
            
            # 🔧 COMPREHENSIVE ENHANCEMENT WITH TIMESTAMP SAFETY
            signal = self._enhance_scalping_signal_comprehensive(signal, latest, previous, spread_pips)
            self.logger.debug(f"✅ Complete scalping strategy data enhancement with timestamp safety and enhanced confidence applied to {signal['epic']}")
        
        if signal and config.USE_BID_ADJUSTMENT:
            signal = self.price_adjuster.add_execution_prices(signal, spread_pips)
        
        return signal

    def _create_enhanced_signal_data(self, signal: Dict, latest: pd.Series, previous: pd.Series, 
                                   df: pd.DataFrame, epic: str, timeframe: str) -> Dict:
        """
        🔥 NEW: Create enhanced signal data structure for proper confidence validation
        This ensures all required fields are available for enhanced validation
        """
        try:
            signal_type = signal.get('signal_type', 'UNKNOWN')
            current_price = latest.get('close', 0)
            
            # Calculate volume data
            volume = latest.get('ltv', latest.get('volume', 0))
            volume_avg = latest.get('volume_avg_10', 1.0)
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            
            # Calculate EMA data
            ema_fast = latest.get(f'ema_{self.config["fast"]}', 0)
            ema_slow = latest.get(f'ema_{self.config["slow"]}', 0)
            ema_filter = latest.get(f'ema_{self.config.get("filter", 50)}', ema_slow)
            
            # Calculate efficiency ratio for scalping
            efficiency_ratio = self._calculate_efficiency_ratio(df)
            
            return {
                'signal_type': signal_type,
                'price': current_price,
                'epic': epic,
                'timeframe': timeframe,
                
                # EMA data for scalping
                'ema_short': ema_fast,
                'ema_long': ema_slow,
                'ema_trend': ema_filter,
                'ema_separation': abs(ema_fast - ema_slow),
                'ema_separation_pips': abs(ema_fast - ema_slow) * 10000,
                
                # Volume data
                'volume': volume,
                'volume_ratio': volume_ratio,
                'volume_confirmation': volume_ratio > self.volume_threshold,
                
                # RSI data
                'rsi_2': latest.get('rsi_2', 50),
                'rsi_14': latest.get('rsi_14', 50),
                
                # Bollinger Band data
                'bb_upper': latest.get('bb_upper', 0),
                'bb_middle': latest.get('bb_middle', current_price),
                'bb_lower': latest.get('bb_lower', 0),
                'bb_position': latest.get('bb_position', 0.5),
                
                # Price action data
                'atr': latest.get('atr', 0.001),
                'high': latest.get('high', current_price),
                'low': latest.get('low', current_price),
                
                # Market efficiency and spread
                'efficiency_ratio': efficiency_ratio,
                'spread_pips': signal.get('spread_pips', 1.5),
                
                # Scalping specific data
                'scalping_mode': self.scalping_mode,
                'quick_execution_required': True,
                'strategy_name': 'scalping',
                'validation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced signal data creation failed: {e}")
            # Return minimal valid data
            return {
                'signal_type': signal.get('signal_type', 'UNKNOWN'),
                'price': latest.get('close', 0),
                'efficiency_ratio': 0.5,
                'volume_confirmation': False,
                'strategy_name': 'scalping'
            }
    
    def _calculate_efficiency_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate efficiency ratio for scalping (shorter period for faster response)
        """
        try:
            # Use shorter period for scalping (10 periods instead of 14)
            period = min(10, len(df) - 1)
            if period < 3:
                return 0.3  # Conservative default for insufficient data
            
            # Get price series
            close_prices = df['close'].tail(period + 1)
            
            if len(close_prices) < 2:
                return 0.3
            
            # Calculate directional change (net movement)
            start_price = close_prices.iloc[0]
            end_price = close_prices.iloc[-1]
            direction_change = abs(end_price - start_price)
            
            # Calculate total movement (sum of all price changes)
            price_changes = close_prices.diff().dropna()
            total_movement = price_changes.abs().sum()
            
            # Handle edge cases
            if total_movement == 0 or pd.isna(total_movement):
                return 0.25  # Actually zero efficiency if no movement
            
            # Calculate efficiency
            efficiency = direction_change / total_movement
            final_efficiency = max(0.0, min(1.0, efficiency))
            
            self.logger.debug(f"[SCALPING EFFICIENCY] Period: {period}, Direction: {direction_change:.6f}, Total: {total_movement:.6f}, Efficiency: {final_efficiency:.3f}")
            
            return final_efficiency
            
        except Exception as e:
            self.logger.error(f"❌ Scalping efficiency ratio calculation failed: {e}")
            return 0.25

    # 🔥 ENHANCED CONFIDENCE VALIDATION METHODS FOR SCALPING
    def calculate_confidence(self, signal_data: Dict) -> float:
        """
        🔥 ENHANCED CONFIDENCE VALIDATION: Replaces simple calculation with comprehensive validation
        This fixes any confidence issues with scalping signals by using multiple validation factors
        """
        try:
            self.logger.debug(f"[SCALPING ENHANCED CONFIDENCE] Starting validation for {self.scalping_mode} signal: {signal_data.get('signal_type', 'UNKNOWN')}")
            
            # Get base confidence from signal data or calculate based on scalping mode
            base_confidence = self._get_scalping_base_confidence()
            
            # Create validation factors
            validation_factors = self._calculate_scalping_validation_factors(signal_data)
            
            # Calculate enhanced confidence
            enhanced_confidence = self._apply_scalping_validation_factors(base_confidence, validation_factors)
            
            # Apply final quality checks
            final_confidence = self._apply_scalping_quality_checks(enhanced_confidence, signal_data, validation_factors)
            
            self.logger.debug(f"[SCALPING ENHANCED CONFIDENCE] Base: {base_confidence:.1%} → Enhanced: {enhanced_confidence:.1%} → Final: {final_confidence:.1%}")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Scalping enhanced confidence calculation failed: {e}")
            # Fallback to conservative estimate based on scalping mode
            return self._get_conservative_confidence()

    def _get_scalping_base_confidence(self) -> float:
        """Get base confidence based on scalping mode"""
        base_confidences = {
            'ultra_fast': 0.60,     # Lower base due to very high frequency
            'aggressive': 0.65,     # Moderate base for aggressive scalping
            'conservative': 0.70,   # Higher base for conservative scalping
            'dual_ma': 0.68         # Moderate base for dual MA
        }
        return base_confidences.get(self.scalping_mode, 0.65)

    def _get_conservative_confidence(self) -> float:
        """Get conservative confidence fallback based on scalping mode"""
        conservative_confidences = {
            'ultra_fast': 0.50,     # Very conservative for ultra fast
            'aggressive': 0.55,     # Conservative for aggressive
            'conservative': 0.60,   # Less conservative for conservative mode
            'dual_ma': 0.58         # Moderate conservative for dual MA
        }
        return conservative_confidences.get(self.scalping_mode, 0.55)

    def _calculate_scalping_validation_factors(self, signal_data: Dict) -> Dict[str, float]:
        """
        🔥 CORE FIX: Calculate validation factors specific to scalping strategy
        """
        factors = {}
        
        try:
            # Factor 1: EMA Crossover Quality (0.0 - 1.0)
            factors['ema_crossover_quality'] = self._validate_ema_crossover(signal_data)
            
            # Factor 2: EMA Separation Quality (0.0 - 1.0) - Critical for scalping
            factors['ema_separation_quality'] = self._validate_ema_separation(signal_data)
            
            # Factor 3: Volume Confirmation (0.0 - 1.0)
            factors['volume_confirmation'] = self._validate_volume_confirmation_scalping(signal_data)
            
            # Factor 4: RSI Positioning (0.0 - 1.0) - Important for scalping timing
            factors['rsi_positioning'] = self._validate_rsi_positioning_scalping(signal_data)
            
            # Factor 5: Bollinger Band Position (0.0 - 1.0)
            factors['bb_position_quality'] = self._validate_bb_position_scalping(signal_data)
            
            # Factor 6: Market Timing (0.0 - 1.0)
            factors['market_timing'] = self._validate_market_timing_scalping(signal_data)
            
            # Factor 7: Efficiency Ratio (0.0 - 1.0) - Shorter period for scalping
            factors['efficiency_ratio'] = signal_data.get('efficiency_ratio', 0.5)
            
            # Factor 8: Spread Quality (0.0 - 1.0) - Critical for scalping profitability
            factors['spread_quality'] = self._validate_spread_quality(signal_data)
            
            self.logger.debug(f"[SCALPING VALIDATION FACTORS] Calculated {len(factors)} factors: {factors}")
            
        except Exception as e:
            self.logger.error(f"❌ Scalping validation factors calculation failed: {e}")
            # Return conservative factors
            factors = {
                'ema_crossover_quality': 0.5,
                'ema_separation_quality': 0.5,
                'volume_confirmation': 0.5,
                'rsi_positioning': 0.5,
                'bb_position_quality': 0.5,
                'market_timing': 0.5,
                'efficiency_ratio': 0.5,
                'spread_quality': 0.5
            }
        
        return factors

    def _validate_ema_crossover(self, signal_data: Dict) -> float:
        """Validate EMA crossover quality for scalping"""
        try:
            ema_short = signal_data.get('ema_short', 0)
            ema_long = signal_data.get('ema_long', 0)
            ema_trend = signal_data.get('ema_trend', 0)
            price = signal_data.get('price', 0)
            signal_type = signal_data.get('signal_type', '')
            
            if not all([ema_short, ema_long, price]):
                return 0.3  # Missing data penalty
            
            # For scalping, we need clear crossover alignment
            if signal_type in ['BULL', 'BUY']:
                # Bull signal: price > ema_short > ema_long
                if price > ema_short > ema_long:
                    # Check trend alignment if filter EMA available
                    if ema_trend > 0 and ema_short > ema_trend:
                        return 0.9  # Perfect alignment with trend
                    elif ema_trend > 0:
                        return 0.7  # Good crossover but weak trend alignment
                    else:
                        return 0.8  # Good crossover, no trend filter
                elif price > ema_short:
                    return 0.6  # Partial alignment
                else:
                    return 0.2  # Poor alignment
                    
            elif signal_type in ['BEAR', 'SELL']:
                # Bear signal: price < ema_short < ema_long
                if price < ema_short < ema_long:
                    # Check trend alignment if filter EMA available
                    if ema_trend > 0 and ema_short < ema_trend:
                        return 0.9  # Perfect alignment with trend
                    elif ema_trend > 0:
                        return 0.7  # Good crossover but weak trend alignment
                    else:
                        return 0.8  # Good crossover, no trend filter
                elif price < ema_short:
                    return 0.6  # Partial alignment
                else:
                    return 0.2  # Poor alignment
            
            return 0.5
            
        except Exception as e:
            self.logger.debug(f"EMA crossover validation failed: {e}")
            return 0.3

    def _validate_ema_separation(self, signal_data: Dict) -> float:
        """Validate EMA separation quality - critical for scalping"""
        try:
            ema_separation_pips = signal_data.get('ema_separation_pips', 0)
            
            # For scalping, EMA separation is crucial to avoid choppy markets
            if ema_separation_pips >= 2.0:  # 2+ pips separation
                return 0.9  # Excellent separation
            elif ema_separation_pips >= 1.0:  # 1+ pip separation
                return 0.7  # Good separation
            elif ema_separation_pips >= 0.5:  # 0.5+ pip separation
                return 0.5  # Minimum acceptable
            else:
                return 0.1  # Too close - choppy market
                
        except Exception as e:
            self.logger.debug(f"EMA separation validation failed: {e}")
            return 0.3

    def _validate_volume_confirmation_scalping(self, signal_data: Dict) -> float:
        """Validate volume confirmation for scalping"""
        try:
            volume_confirmation = signal_data.get('volume_confirmation', None)
            volume_ratio = signal_data.get('volume_ratio', 1.0)
            
            if volume_confirmation is True:
                # Scale by volume ratio but cap at reasonable levels
                return min(1.0, volume_ratio / 1.5)  # Scale by 1.5x threshold
            elif volume_confirmation is False:
                return 0.4  # Lower penalty for scalping (less volume dependent)
            else:
                # Calculate from volume ratio
                if volume_ratio >= 1.5:
                    return 0.8  # High volume
                elif volume_ratio >= 1.2:
                    return 0.6  # Medium volume
                else:
                    return 0.5  # Acceptable for scalping
                    
        except Exception as e:
            self.logger.debug(f"Volume confirmation validation failed: {e}")
            return 0.5

    def _validate_rsi_positioning_scalping(self, signal_data: Dict) -> float:
        """Validate RSI positioning for scalping timing"""
        try:
            signal_type = signal_data.get('signal_type', '')
            rsi_2 = signal_data.get('rsi_2', 50)
            rsi_14 = signal_data.get('rsi_14', 50)
            
            if signal_type in ['BULL', 'BUY']:
                # For bull signals, avoid extreme overbought
                if rsi_2 < 80 and 30 < rsi_14 < 75:
                    return 0.8  # Good positioning
                elif rsi_2 < 90 and rsi_14 > 25:
                    return 0.6  # Acceptable positioning
                else:
                    return 0.3  # Poor positioning (too overbought)
                    
            elif signal_type in ['BEAR', 'SELL']:
                # For bear signals, avoid extreme oversold
                if rsi_2 > 20 and 25 < rsi_14 < 70:
                    return 0.8  # Good positioning
                elif rsi_2 > 10 and rsi_14 < 75:
                    return 0.6  # Acceptable positioning
                else:
                    return 0.3  # Poor positioning (too oversold)
            
            return 0.5
            
        except Exception as e:
            self.logger.debug(f"RSI positioning validation failed: {e}")
            return 0.5

    def _validate_bb_position_scalping(self, signal_data: Dict) -> float:
        """Validate Bollinger Band position for scalping"""
        try:
            bb_position = signal_data.get('bb_position', 0.5)
            signal_type = signal_data.get('signal_type', '')
            
            # For scalping, avoid extreme BB positions (prevents buying tops/selling bottoms)
            if signal_type in ['BULL', 'BUY']:
                if bb_position < 0.2:
                    return 0.9  # Excellent - buying near lower band
                elif bb_position < 0.4:
                    return 0.7  # Good position
                elif bb_position < 0.7:
                    return 0.6  # Acceptable
                else:
                    return 0.2  # Poor - buying near upper band
                    
            elif signal_type in ['BEAR', 'SELL']:
                if bb_position > 0.8:
                    return 0.9  # Excellent - selling near upper band
                elif bb_position > 0.6:
                    return 0.7  # Good position
                elif bb_position > 0.3:
                    return 0.6  # Acceptable
                else:
                    return 0.2  # Poor - selling near lower band
            
            return 0.5
            
        except Exception as e:
            self.logger.debug(f"BB position validation failed: {e}")
            return 0.5

    def _validate_market_timing_scalping(self, signal_data: Dict) -> float:
        """Validate market timing for scalping"""
        try:
            from datetime import datetime
            
            current_hour = datetime.now().hour
            
            # Scalping works best during high volatility sessions
            if 13 <= current_hour <= 17:  # London/NY overlap - best for scalping
                return 0.9
            elif 8 <= current_hour <= 12:  # London session
                return 0.8
            elif 18 <= current_hour <= 22:  # NY session
                return 0.7
            elif 1 <= current_hour <= 6:  # Asian session
                return 0.5
            else:  # Low activity periods
                return 0.4
                
        except Exception as e:
            self.logger.debug(f"Market timing validation failed: {e}")
            return 0.6

    def _validate_spread_quality(self, signal_data: Dict) -> float:
        """Validate spread quality - critical for scalping profitability"""
        try:
            spread_pips = signal_data.get('spread_pips', 2.0)
            
            # For scalping, spread is crucial for profitability
            if spread_pips <= 0.8:
                return 1.0  # Excellent spread
            elif spread_pips <= 1.2:
                return 0.8  # Good spread
            elif spread_pips <= 1.6:
                return 0.6  # Acceptable spread
            elif spread_pips <= 2.0:
                return 0.4  # Challenging spread
            else:
                return 0.1  # Too wide for profitable scalping
                
        except Exception as e:
            self.logger.debug(f"Spread quality validation failed: {e}")
            return 0.5

    def _apply_scalping_validation_factors(self, base_confidence: float, validation_factors: Dict[str, float]) -> float:
        """
        🔥 CORE ALGORITHM: Apply validation factors to base confidence for scalping
        """
        try:
            # Validation factor weights for scalping (must sum to 1.0)
            weights = {
                'ema_crossover_quality': 0.20,     # Important for entry timing
                'ema_separation_quality': 0.18,    # Critical for avoiding chop
                'spread_quality': 0.15,            # Critical for scalping profitability
                'volume_confirmation': 0.12,       # Less critical than other strategies
                'rsi_positioning': 0.12,           # Important for timing
                'bb_position_quality': 0.10,       # Helps avoid extremes
                'market_timing': 0.08,             # Session timing
                'efficiency_ratio': 0.05           # Less critical for short-term
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
            
            # 🔥 SCALPING-SPECIFIC VALIDATION APPLICATION
            # Scalping requires different thresholds than longer-term strategies
            if normalized_validation < 0.4:
                # Very poor validation - heavily penalize
                enhanced_confidence = base_confidence * (normalized_validation * 1.0)
            elif normalized_validation < 0.6:
                # Moderate validation - slight penalty
                enhanced_confidence = base_confidence * (normalized_validation * 1.1)
            else:
                # Good validation - reward confidence
                enhanced_confidence = base_confidence * min(1.2, normalized_validation + 0.2)
            
            # Scalping-specific bounds (slightly lower max than other strategies)
            enhanced_confidence = max(0.1, min(0.90, enhanced_confidence))
            
            self.logger.debug(f"[SCALPING VALIDATION APPLIED] Base: {base_confidence:.1%}, Validation: {normalized_validation:.1%}, Enhanced: {enhanced_confidence:.1%}")
            
            return enhanced_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Scalping validation factors application failed: {e}")
            return min(base_confidence * 0.8, 0.6)  # Conservative fallback

    def _apply_scalping_quality_checks(self, enhanced_confidence: float, signal_data: Dict, validation_factors: Dict) -> float:
        """
        🔥 FINAL QUALITY GATE: Apply final quality checks for scalping signals
        """
        try:
            final_confidence = enhanced_confidence
            
            # Quality Check 1: Critical scalping factors
            critical_factors = ['ema_separation_quality', 'spread_quality']
            for factor in critical_factors:
                if validation_factors.get(factor, 0.5) < 0.4:
                    self.logger.debug(f"[SCALPING QUALITY CHECK] Critical factor {factor} too low: {validation_factors[factor]:.1%}")
                    final_confidence *= 0.7  # 30% penalty for critical factor failure
            
            # Quality Check 2: EMA separation requirement (critical for scalping)
            ema_separation_pips = signal_data.get('ema_separation_pips', 0)
            if ema_separation_pips < 0.5:
                self.logger.debug(f"[SCALPING QUALITY CHECK] EMA separation too small: {ema_separation_pips:.2f} pips")
                final_confidence *= 0.6  # 40% penalty for insufficient separation
            
            # Quality Check 3: Spread requirement (critical for scalping profitability)
            spread_pips = signal_data.get('spread_pips', 2.0)
            if spread_pips > 2.0:
                self.logger.debug(f"[SCALPING QUALITY CHECK] Spread too wide: {spread_pips:.1f} pips")
                final_confidence *= 0.5  # 50% penalty for wide spread
            
            # Quality Check 4: High confidence requires excellent conditions
            if enhanced_confidence > 0.8:
                avg_validation = sum(validation_factors.values()) / len(validation_factors)
                if avg_validation < 0.7:  # Scalping requires good overall conditions
                    self.logger.debug(f"[SCALPING QUALITY CHECK] High confidence blocked: avg validation {avg_validation:.1%} < 70%")
                    final_confidence = min(final_confidence, 0.75)  # Cap at 75% for insufficient validation
            
            # Quality Check 5: Ultra-fast mode has stricter requirements
            if self.scalping_mode == 'ultra_fast' and final_confidence > 0.75:
                volume_confirmation = validation_factors.get('volume_confirmation', 0.5)
                if volume_confirmation < 0.6:
                    self.logger.debug(f"[SCALPING QUALITY CHECK] Ultra-fast requires better volume: {volume_confirmation:.1%}")
                    final_confidence = min(final_confidence, 0.70)  # Cap for ultra-fast
            
            # Final bounds check (scalping-specific)
            final_confidence = max(0.1, min(0.88, final_confidence))  # Max 88% for scalping
            
            if final_confidence != enhanced_confidence:
                self.logger.debug(f"[SCALPING QUALITY CHECKS APPLIED] {enhanced_confidence:.1%} → {final_confidence:.1%}")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Scalping quality checks failed: {e}")
            return min(enhanced_confidence * 0.9, 0.7)  # Conservative fallback

    def _enhance_scalping_signal_comprehensive(self, signal: Dict, latest: pd.Series, previous: pd.Series, spread_pips: float) -> Dict:
        """
        🔧 COMPREHENSIVE ENHANCEMENT: Populate ALL database columns for ScalpingStrategy
        ADDED: Safe timestamp conversion and complete data richness matching other strategies
        This ensures scalping signals have the same comprehensive data as EMA/MACD/Combined signals
        """
        try:
            current_price = signal.get('price', latest.get('close', 0))
            signal_type = signal.get('signal_type')
            epic = signal.get('epic')
            timeframe = signal.get('timeframe', '1m')
            confidence = signal.get('confidence_score', 0.7)
            
            # Extract pair from epic
            if not signal.get('pair'):
                signal['pair'] = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            
            # ========== TIMESTAMP SAFETY FIX ==========
            # Apply safe timestamp conversion for any timestamp fields
            timestamp_fields = ['market_timestamp', 'timestamp', 'signal_timestamp', 'candle_timestamp', 'alert_timestamp']
            for field in timestamp_fields:
                if field in signal:
                    original_value = signal[field]
                    safe_timestamp = self._convert_market_timestamp_safe(original_value)
                    
                    if original_value != safe_timestamp:
                        self.logger.debug(f"🛠️ TIMESTAMP FIX: {field} converted from {original_value} to {safe_timestamp}")
                    
                    signal[field] = safe_timestamp
            
            # ========== CORE TECHNICAL DATA ==========
            
            # Scalping-specific EMA data
            signal.update({
                'ema_short': float(signal.get('fast_ema', latest.get(f'ema_{self.config["fast"]}', 0))),
                'ema_long': float(signal.get('slow_ema', latest.get(f'ema_{self.config["slow"]}', 0))),
                'ema_trend': float(signal.get('filter_ema', latest.get(f'ema_{self.config.get("filter", 50)}', signal.get('ema_long', 0)))),
                'ema_9': float(latest.get('ema_9', signal.get('ema_short', 0))),
                'ema_21': float(latest.get('ema_21', signal.get('ema_long', 0))),
                'ema_200': float(latest.get('ema_200', signal.get('ema_trend', 0)))
            })
            
            # MACD data (if available, otherwise calculated)
            signal.update({
                'macd_line': float(latest.get('macd_line', latest.get('macd', 0))),
                'macd_signal': float(latest.get('macd_signal', 0)),
                'macd_histogram': float(latest.get('macd_histogram', 0))
            })
            
            # Volume data - critical for scalping
            volume = latest.get('ltv') or latest.get('volume', 0)
            signal['volume'] = float(volume) if volume else 0.0
            
            # Volume ratio and confirmation
            volume_avg = latest.get('volume_avg_10', 1.0)
            if volume_avg and volume_avg > 0:
                signal['volume_ratio'] = signal['volume'] / volume_avg
                signal['volume_confirmation'] = signal['volume_ratio'] > self.volume_threshold
            else:
                signal['volume_ratio'] = 1.0
                signal['volume_confirmation'] = False
            
            # ========== STRATEGY CONFIGURATION (JSON fields) ==========
            
            # strategy_config - comprehensive scalping configuration data
            signal['strategy_config'] = {
                'strategy_type': 'scalping_strategy',
                'strategy_family': 'high_frequency',
                'scalping_mode': self.scalping_mode,
                'ema_fast_period': self.config['fast'],
                'ema_slow_period': self.config['slow'],
                'ema_filter_period': self.config.get('filter'),
                'min_separation_pips': self.min_separation_pips,
                'volume_threshold': self.volume_threshold,
                'max_spread_pips': self.max_spread_pips,
                'quick_exit_enabled': self.quick_exit_enabled,
                'target_timeframe': timeframe,
                'bid_adjustment_enabled': getattr(config, 'USE_BID_ADJUSTMENT', False),
                'timestamp_safety_enabled': True,  # NEW: indicates timestamp fix applied
                'enhanced_confidence_enabled': True  # NEW: indicates confidence fix applied
            }
            
            # strategy_indicators - all technical indicator values for scalping
            ema_separation = abs(signal['ema_short'] - signal['ema_long'])
            ema_separation_pips = signal.get('ema_separation_pips', ema_separation * 10000)
            
            signal['strategy_indicators'] = {
                'primary_indicator': 'fast_ema_crossover',
                'scalping_mode': self.scalping_mode,
                'ema_fast_value': signal['ema_short'],
                'ema_slow_value': signal['ema_long'],
                'ema_filter_value': signal['ema_trend'],
                'ema_separation': ema_separation,
                'ema_separation_pips': ema_separation_pips,
                'current_price': current_price,
                'previous_price': float(previous.get('close', current_price)),
                'price_ema_fast_distance': current_price - signal['ema_short'],
                'rsi_2': float(signal.get('rsi_2', latest.get('rsi_2', 50))),
                'rsi_14': float(signal.get('rsi_14', latest.get('rsi_14', 50))),
                'bb_position': float(signal.get('bb_position', latest.get('bb_position', 0.5))),
                'volume_ratio': signal['volume_ratio'],
                'volume_confirmed': signal['volume_confirmation'],
                'spread_pips': spread_pips,
                'signal_trigger': f'scalping_{self.scalping_mode}',
                'timestamp_safety_applied': True,  # NEW
                'enhanced_confidence_applied': True  # NEW
            }
            
            # strategy_metadata - comprehensive scalping context
            signal['strategy_metadata'] = {
                'strategy_version': '2.1.0',  # Updated for enhanced confidence
                'signal_basis': f'scalping_{self.scalping_mode}_with_comprehensive_enhancement_and_confidence_fix',
                'confidence_calculation': 'scalping_enhanced_validation',  # UPDATED
                'signal_strength': 'strong' if confidence > 0.8 else 'medium' if confidence > 0.7 else 'weak',
                'market_condition': self._determine_scalping_market_condition(current_price, signal, latest),
                'scalping_performance': {
                    'entry_reason': signal.get('entry_reason', 'ema_crossover'),
                    'mode_specifics': {
                        'mode': self.scalping_mode,
                        'fast_ema_period': self.config['fast'],
                        'slow_ema_period': self.config['slow'],
                        'filter_active': self.config.get('filter') is not None
                    },
                    'market_timing': self._assess_scalping_timing(),
                    'volatility_suitability': self._assess_scalping_volatility(latest)
                },
                'processing_timestamp': datetime.now().isoformat(),
                'enhancement_applied': True,
                'data_completeness': 'full',
                'timestamp_safety_applied': True,  # NEW
                'enhanced_confidence_applied': True,  # NEW
                'confidence_factors': {
                    'ema_crossover_confirmed': True,
                    'volume_confirmation': signal['volume_confirmation'],
                    'rsi_positioning': self._assess_rsi_positioning(signal, signal_type),
                    'bb_positioning': self._assess_bb_positioning(signal),
                    'spread_acceptable': spread_pips <= self.max_spread_pips,
                    'ema_separation_adequate': ema_separation_pips >= self.min_separation_pips,
                    'enhanced_validation_passed': True  # NEW
                }
            }
            
            # signal_conditions - market conditions at signal time for scalping
            signal['signal_conditions'] = {
                'market_trend': self._determine_scalping_trend(current_price, signal),
                'scalping_signal_type': f'{self.scalping_mode}_crossover',
                'price_position': f'{"above" if current_price > signal["ema_short"] else "below"}_fast_ema',
                'momentum_direction': signal_type.lower() if signal_type else 'neutral',
                'volatility_assessment': self._assess_volatility(latest),
                'signal_timing': 'scalping_opportunity',
                'confirmation_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.7 else 'low',
                'market_session': self._determine_trading_session(),
                'scalping_favorability': self._assess_scalping_favorability(latest, spread_pips),
                'rapid_execution_required': True,
                'timestamp_safety_processed': True,  # NEW
                'enhanced_confidence_processed': True  # NEW
            }
            
            # ========== PRICING AND EXECUTION DATA ==========
            
            # Comprehensive pricing data with scalping-specific calculations
            pip_size = 0.01 if 'JPY' in epic else 0.0001
            spread_adjustment = spread_pips * pip_size
            
            signal.update({
                'spread_pips': spread_pips,
                'bid_price': current_price - spread_adjustment,
                'ask_price': current_price + spread_adjustment,
                'execution_price': current_price,
                'pip_size': pip_size
            })
            
            # Scalping-specific risk management (tight stops, quick targets)
            scalping_stop_distance = self._get_scalping_stop_distance(self.scalping_mode)
            scalping_target_distance = self._get_scalping_target_distance(self.scalping_mode)
            
            stop_distance_price = scalping_stop_distance * pip_size
            target_distance_price = scalping_target_distance * pip_size
            
            if signal_type in ['BULL', 'BUY']:
                signal['stop_loss'] = current_price - stop_distance_price
                signal['take_profit'] = current_price + target_distance_price
            elif signal_type in ['BEAR', 'SELL']:
                signal['stop_loss'] = current_price + stop_distance_price
                signal['take_profit'] = current_price - target_distance_price
            else:
                signal['stop_loss'] = current_price - stop_distance_price
                signal['take_profit'] = current_price + target_distance_price
            
            # Support/Resistance levels (if available)
            if 'support' in latest and latest['support'] is not None:
                signal['nearest_support'] = float(latest['support'])
                signal['distance_to_support_pips'] = (current_price - latest['support']) / pip_size
            else:
                signal['nearest_support'] = current_price * 0.999  # Very close for scalping
                signal['distance_to_support_pips'] = (current_price * 0.001) / pip_size
            
            if 'resistance' in latest and latest['resistance'] is not None:
                signal['nearest_resistance'] = float(latest['resistance'])
                signal['distance_to_resistance_pips'] = (latest['resistance'] - current_price) / pip_size
            else:
                signal['nearest_resistance'] = current_price * 1.001  # Very close for scalping
                signal['distance_to_resistance_pips'] = (current_price * 0.001) / pip_size
            
            # Risk/Reward calculation for scalping
            stop_distance = abs(current_price - signal['stop_loss'])
            target_distance = abs(signal['take_profit'] - current_price)
            if stop_distance > 0:
                signal['risk_reward_ratio'] = target_distance / stop_distance
            else:
                signal['risk_reward_ratio'] = scalping_target_distance / scalping_stop_distance
            
            # ========== DEDUPLICATION AND TRACKING ==========
            
            # Signal identification and deduplication with SAFE TIMESTAMP
            signal.update({
                'signal_hash': self._generate_signal_hash(epic, signal_type, timeframe, current_price),
                'market_timestamp': datetime.now(),
                'data_source': 'live_scanner',
                'cooldown_key': f"{epic}_{signal_type}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M')}"  # More granular for scalping
            })
            
            # ========== MARKET CONTEXT ==========
            
            # Market session and timing with SAFE TIMESTAMP
            current_time = datetime.now()
            signal.update({
                'market_session': self._determine_trading_session(),
                'is_market_hours': self._is_market_hours(),
                'alert_timestamp': current_time,
                'processing_timestamp': current_time.isoformat()
            })
            
            # ========== SCALPING-SPECIFIC FIELDS ==========
            
            # Scalping performance and execution data
            signal['scalping_analysis'] = {
                'execution_speed_critical': True,
                'profit_target_pips': scalping_target_distance,
                'stop_loss_pips': scalping_stop_distance,
                'expected_trade_duration': self._get_expected_trade_duration(self.scalping_mode),
                'volume_surge_detected': signal['volume_confirmation'],
                'bb_breakout_potential': self._assess_bb_breakout_potential(latest),
                'rsi_momentum_alignment': signal.get('rsi_14', 50) > 50 if signal_type in ['BULL', 'BUY'] else signal.get('rsi_14', 50) < 50,
                'ema_momentum_strength': ema_separation_pips,
                'scalping_score': confidence * 100,
                'enhanced_confidence_applied': True  # NEW
            }
            
            # Technical analysis summary for scalping
            signal['technical_summary'] = {
                'primary_signal': f"Scalping {signal_type if signal_type else 'UNKNOWN'} ({self.scalping_mode})",
                'execution_urgency': 'IMMEDIATE' if confidence > 0.8 else 'HIGH' if confidence > 0.7 else 'MEDIUM',
                'entry_quality': 'Excellent' if confidence > 0.85 else 'Good' if confidence > 0.75 else 'Fair',
                'scalping_setup': f"EMA {self.config['fast']}/{self.config['slow']} crossover",
                'volume_support': 'Strong' if signal['volume_confirmation'] else 'Weak',
                'timeframe_optimization': timeframe,
                'signal_reliability': 'High' if confidence > 0.8 and signal['volume_confirmation'] else 'Medium',
                'market_microstructure': 'Favorable' if spread_pips <= 1.5 else 'Acceptable' if spread_pips <= 2.0 else 'Challenging',
                'timestamp_safety': 'enabled',  # NEW
                'enhanced_confidence': 'enabled'  # NEW
            }
            
            # ========== ADDITIONAL COMPREHENSIVE FIELDS ==========
            
            # Add all the additional fields that other strategies have
            signal.update({
                # Price fields
                'entry_price': current_price,
                'signal_price': current_price,
                'current_price': current_price,
                'open_price': float(latest.get('open', current_price)),
                'high_price': float(latest.get('high', current_price)),
                'low_price': float(latest.get('low', current_price)),
                'close_price': current_price,
                'mid_price': current_price,
                
                # Risk management
                'risk_percent': getattr(config, 'DEFAULT_RISK_PERCENT', 1.0),  # Lower for scalping
                'pip_risk': scalping_stop_distance,
                'max_risk_percentage': min(1.0, max(0.25, stop_distance / current_price * 100)),  # Lower for scalping
                'position_size_suggestion': 'scalping_optimized',
                
                # Market context
                'market_regime': self._determine_market_regime(latest, signal),
                'volatility': self._assess_volatility(latest),
                'trend_strength': self._calculate_trend_strength_scalping(signal),
                'trading_session': self._determine_trading_session(),
                'market_hours': self._is_market_hours(),
                
                # Alert fields
                'alert_message': f"Scalping {signal_type} ({self.scalping_mode}) @ {confidence:.1%} - Quick execution required",
                'alert_level': 'URGENT' if confidence > 0.85 else 'HIGH' if confidence > 0.75 else 'MEDIUM',
                'status': 'NEW',
                'processed': False,
                
                # Claude analysis placeholder fields
                'claude_analysis': None,
                'claude_approved': None,
                'claude_score': None,
                'claude_decision': None,
                'claude_reasoning': None,
                
                # Scalping-specific execution fields
                'scalping_mode': self.scalping_mode,
                'quick_exit_enabled': self.quick_exit_enabled,
                'target_pips': scalping_target_distance,
                'stop_pips': scalping_stop_distance
            })
            
            # ========== FINAL TIMESTAMP SAFETY CHECK ==========
            # Ensure all timestamp fields are properly converted
            for field_name, field_value in signal.items():
                if 'timestamp' in field_name.lower() and field_value is not None:
                    signal[field_name] = self._convert_market_timestamp_safe(field_value)
            
            # Ensure all JSON fields are properly serializable
            signal = self._ensure_json_serializable(signal)
            
            self.logger.debug(f"✅ Enhanced scalping signal with timestamp safety and enhanced confidence: {len(signal)} fields (comprehensive data richness)")
            
        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive scalping signal enhancement with timestamp and confidence fix: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        return signal

    def _get_scalping_stop_distance(self, mode: str) -> float:
        """Get stop loss distance in pips for scalping mode"""
        distances = {
            'ultra_fast': 2.0,
            'aggressive': 3.0,
            'conservative': 5.0,
            'dual_ma': 3.0
        }
        return distances.get(mode, 3.0)

    def _get_scalping_target_distance(self, mode: str) -> float:
        """Get take profit distance in pips for scalping mode"""
        distances = {
            'ultra_fast': 3.0,
            'aggressive': 5.0,
            'conservative': 8.0,
            'dual_ma': 5.0
        }
        return distances.get(mode, 5.0)

    def _get_expected_trade_duration(self, mode: str) -> str:
        """Get expected trade duration for scalping mode"""
        durations = {
            'ultra_fast': '1-3 minutes',
            'aggressive': '3-10 minutes',
            'conservative': '10-30 minutes',
            'dual_ma': '5-15 minutes'
        }
        return durations.get(mode, '5-10 minutes')

    def _determine_scalping_market_condition(self, current_price: float, signal: Dict, latest: pd.Series) -> str:
        """Determine market condition for scalping"""
        ema_fast = signal.get('ema_short', 0)
        ema_slow = signal.get('ema_long', 0)
        
        if current_price > ema_fast > ema_slow:
            return 'scalping_bullish_momentum'
        elif current_price < ema_fast < ema_slow:
            return 'scalping_bearish_momentum'
        else:
            return 'scalping_consolidation'

    def _assess_scalping_timing(self) -> str:
        """Assess timing favorability for scalping"""
        current_hour = datetime.now().hour
        # London/New York overlap is best for scalping
        if 13 <= current_hour <= 17:  # London/NY overlap
            return 'optimal_volatility'
        elif 8 <= current_hour <= 12 or 18 <= current_hour <= 22:  # Single session active
            return 'good_volatility'
        else:
            return 'low_volatility'

    def _assess_scalping_volatility(self, latest: pd.Series) -> str:
        """Assess volatility suitability for scalping"""
        bb_width = latest.get('bb_upper', 0) - latest.get('bb_lower', 0)
        current_price = latest.get('close', 1)
        
        if bb_width > 0:
            volatility_ratio = bb_width / current_price
            if volatility_ratio > 0.01:  # 1%
                return 'high_suitable'
            elif volatility_ratio > 0.005:  # 0.5%
                return 'medium_suitable'
            else:
                return 'low_challenging'
        return 'unknown'

    def _assess_rsi_positioning(self, signal: Dict, signal_type: str) -> bool:
        """Assess RSI positioning for signal type"""
        rsi_14 = signal.get('rsi_14', 50)
        rsi_2 = signal.get('rsi_2', 50)
        
        if signal_type in ['BULL', 'BUY']:
            return 30 < rsi_14 < 80 and rsi_2 < 90
        else:
            return 20 < rsi_14 < 70 and rsi_2 > 10

    def _assess_bb_positioning(self, signal: Dict) -> bool:
        """Assess Bollinger Band positioning"""
        bb_position = signal.get('bb_position', 0.5)
        return 0.1 < bb_position < 0.9  # Not at extremes

    def _determine_scalping_trend(self, current_price: float, signal: Dict) -> str:
        """Determine trend for scalping purposes"""
        ema_fast = signal.get('ema_short', 0)
        ema_slow = signal.get('ema_long', 0)
        
        if ema_fast > ema_slow:
            return 'short_term_bullish'
        elif ema_fast < ema_slow:
            return 'short_term_bearish'
        else:
            return 'short_term_neutral'

    def _assess_scalping_favorability(self, latest: pd.Series, spread_pips: float) -> str:
        """Assess overall favorability for scalping"""
        factors = []
        
        # Spread check
        if spread_pips <= 1.0:
            factors.append('excellent_spread')
        elif spread_pips <= 1.5:
            factors.append('good_spread')
        else:
            factors.append('challenging_spread')
        
        # Volume check
        volume_ratio = latest.get('ltv', 0) / max(latest.get('volume_avg_10', 1), 1)
        if volume_ratio > 1.5:
            factors.append('high_volume')
        elif volume_ratio > 1.2:
            factors.append('good_volume')
        else:
            factors.append('low_volume')
        
        # Volatility check
        volatility = self._assess_scalping_volatility(latest)
        factors.append(volatility)
        
        if 'excellent' in str(factors) or factors.count('good') >= 2:
            return 'highly_favorable'
        elif 'good' in str(factors):
            return 'favorable'
        else:
            return 'challenging'

    def _assess_bb_breakout_potential(self, latest: pd.Series) -> bool:
        """Assess Bollinger Band breakout potential"""
        bb_position = latest.get('bb_position', 0.5)
        return bb_position > 0.85 or bb_position < 0.15

    def _determine_market_regime(self, latest: pd.Series, signal: Dict) -> str:
        """Determine market regime for scalping"""
        volatility = self._assess_scalping_volatility(latest)
        trend = self._determine_scalping_trend(latest.get('close', 0), signal)
        
        if 'high' in volatility and 'bullish' in trend:
            return 'trending_volatile_bullish'
        elif 'high' in volatility and 'bearish' in trend:
            return 'trending_volatile_bearish'
        elif 'medium' in volatility:
            return 'scalping_optimal'
        else:
            return 'low_volatility_challenging'

    def _calculate_trend_strength_scalping(self, signal: Dict) -> str:
        """Calculate trend strength for scalping"""
        ema_short = signal.get('ema_short', 0)
        ema_long = signal.get('ema_long', 0)
        
        if ema_long == 0:
            return 'unknown'
        
        separation = abs(ema_short - ema_long) / ema_long
        
        if separation > 0.001:  # 0.1%
            return 'strong'
        elif separation > 0.0005:  # 0.05%
            return 'medium'
        else:
            return 'weak'

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
            return 1 <= current_hour <= 23  # Markets active most of the day
        except:
            return True

    def _assess_volatility(self, latest_data: pd.Series) -> str:
        """Assess current market volatility"""
        try:
            if 'high' in latest_data and 'low' in latest_data and 'close' in latest_data:
                daily_range = (latest_data['high'] - latest_data['low']) / latest_data['close']
                if daily_range > 0.01:  # 1%
                    return 'high'
                elif daily_range > 0.005:  # 0.5%
                    return 'medium'
                else:
                    return 'low'
        except:
            pass
        return 'unknown'

    def _generate_signal_hash(self, epic: str, signal_type: str, timeframe: str, price: float) -> str:
        """Generate unique hash for signal deduplication"""
        hash_string = f"{epic}_{signal_type}_{timeframe}_{int(price*10000)}_{datetime.now().strftime('%Y%m%d%H%M')}"  # More granular for scalping
        return hashlib.md5(hash_string.encode()).hexdigest()[:16]

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
    
    def _detect_ultra_fast_signal(
        self, 
        latest: pd.Series, 
        previous: pd.Series, 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Ultra-fast 3/8 EMA crossover for 1-minute scalping"""
        
        current_price = latest['close']
        fast_ema = latest[f'ema_{self.config["fast"]}']
        slow_ema = latest[f'ema_{self.config["slow"]}']
        filter_ema = latest[f'ema_{self.config["filter"]}'] if self.config.get('filter') else None
        
        fast_ema_prev = previous[f'ema_{self.config["fast"]}']
        slow_ema_prev = previous[f'ema_{self.config["slow"]}']
        
        # Ultra-fast crossover detection
        bullish_cross = (fast_ema_prev <= slow_ema_prev) and (fast_ema > slow_ema)
        bearish_cross = (fast_ema_prev >= slow_ema_prev) and (fast_ema < slow_ema)
        
        # Additional filters for ultra-fast mode
        rsi_2 = latest.get('rsi_2', 50)
        volume_ok = latest.get('ltv', 0) > latest.get('volume_avg_10', 1) * self.volume_threshold
        
        # BULL signal
        if bullish_cross:
            # Additional confirmations
            price_above_slow = current_price > slow_ema
            trend_filter = filter_ema is None or current_price > filter_ema
            rsi_not_overbought = rsi_2 < 95  # Very permissive for ultra-fast
            
            if price_above_slow and trend_filter and rsi_not_overbought:
                signal = self.create_base_signal('BULL', epic, timeframe, latest)
                signal.update({
                    'scalping_mode': 'ultra_fast',
                    'entry_reason': 'fast_crossover_up',
                    'fast_ema': fast_ema,
                    'slow_ema': slow_ema,
                    'rsi_2': rsi_2,
                    'volume_confirmed': volume_ok,
                    'confidence_score': self._calculate_ultra_fast_confidence(latest, 'BULL')
                })
                return signal
        
        # BEAR signal
        elif bearish_cross:
            price_below_slow = current_price < slow_ema
            trend_filter = filter_ema is None or current_price < filter_ema
            rsi_not_oversold = rsi_2 > 5  # Very permissive for ultra-fast
            
            if price_below_slow and trend_filter and rsi_not_oversold:
                signal = self.create_base_signal('BEAR', epic, timeframe, latest)
                signal.update({
                    'scalping_mode': 'ultra_fast',
                    'entry_reason': 'fast_crossover_down', 
                    'fast_ema': fast_ema,
                    'slow_ema': slow_ema,
                    'rsi_2': rsi_2,
                    'volume_confirmed': volume_ok,
                    'confidence_score': self._calculate_ultra_fast_confidence(latest, 'BEAR')
                })
                return signal
        
        return None
    
    def _detect_dual_ma_signal(
        self, 
        latest: pd.Series, 
        previous: pd.Series, 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Simple dual MA crossover (7/14 EMAs) with price action confirmation"""
        
        current_price = latest['close']
        fast_ema = latest[f'ema_{self.config["fast"]}']
        slow_ema = latest[f'ema_{self.config["slow"]}']
        fast_ema_prev = previous[f'ema_{self.config["fast"]}']
        slow_ema_prev = previous[f'ema_{self.config["slow"]}']
        
        # Crossover detection
        bullish_cross = (fast_ema_prev <= slow_ema_prev) and (fast_ema > slow_ema)
        bearish_cross = (fast_ema_prev >= slow_ema_prev) and (fast_ema < slow_ema)
        
        # Volume confirmation
        volume_ok = self._check_volume_confirmation(latest)
        
        # Bollinger Band position for context
        bb_position = self._get_bb_position(latest)
        
        if bullish_cross and current_price > fast_ema:
            # Avoid buying at upper BB extreme
            if bb_position < 0.8 or not hasattr(latest, 'bb_position'):
                signal = self.create_base_signal('BULL', epic, timeframe, latest)
                signal.update({
                    'scalping_mode': 'dual_ma',
                    'entry_reason': 'dual_ma_cross_up',
                    'fast_ema': fast_ema,
                    'slow_ema': slow_ema,
                    'volume_confirmed': volume_ok,
                    'bb_position': bb_position,
                    'confidence_score': self._calculate_dual_ma_confidence(latest, 'BULL', volume_ok)
                })
                return signal
        
        elif bearish_cross and current_price < fast_ema:
            # Avoid selling at lower BB extreme
            if bb_position > 0.2 or not hasattr(latest, 'bb_position'):
                signal = self.create_base_signal('BEAR', epic, timeframe, latest)
                signal.update({
                    'scalping_mode': 'dual_ma',
                    'entry_reason': 'dual_ma_cross_down',
                    'fast_ema': fast_ema,
                    'slow_ema': slow_ema,
                    'volume_confirmed': volume_ok,
                    'bb_position': bb_position,
                    'confidence_score': self._calculate_dual_ma_confidence(latest, 'BEAR', volume_ok)
                })
                return signal
        
        return None
    
    def _detect_standard_scalping_signal(
        self, 
        latest: pd.Series, 
        previous: pd.Series, 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Standard scalping with 5/13 or 8/20 EMAs plus RSI confirmation"""
        
        current_price = latest['close']
        fast_ema = latest[f'ema_{self.config["fast"]}']
        slow_ema = latest[f'ema_{self.config["slow"]}']
        filter_ema = latest[f'ema_{self.config["filter"]}'] if self.config.get('filter') else None
        
        fast_ema_prev = previous[f'ema_{self.config["fast"]}']
        slow_ema_prev = previous[f'ema_{self.config["slow"]}']
        
        # Check EMA separation (avoid choppy markets)
        ema_separation_pips = abs(fast_ema - slow_ema) * 10000
        if ema_separation_pips < self.min_separation_pips:
            return None
        
        # Crossover detection
        bullish_cross = (fast_ema_prev <= slow_ema_prev) and (fast_ema > slow_ema)
        bearish_cross = (fast_ema_prev >= slow_ema_prev) and (fast_ema < slow_ema)
        
        # RSI confirmation
        rsi_14 = latest.get('rsi_14', 50)
        rsi_2 = latest.get('rsi_2', 50)
        
        # Volume and volatility checks
        volume_ok = self._check_volume_confirmation(latest)
        bb_position = self._get_bb_position(latest)
        
        if bullish_cross:
            # BULL signal conditions
            price_above_fast = current_price > fast_ema
            trend_filter_ok = filter_ema is None or current_price > filter_ema
            rsi_bullish = rsi_14 > 30 and rsi_2 < 80  # Not oversold on 14, not overbought on 2
            bb_not_extreme = bb_position < 0.9
            
            if price_above_fast and trend_filter_ok and rsi_bullish and bb_not_extreme:
                signal = self.create_base_signal('BULL', epic, timeframe, latest)
                signal.update({
                    'scalping_mode': self.scalping_mode,
                    'entry_reason': 'ema_cross_rsi_confirm',
                    'fast_ema': fast_ema,
                    'slow_ema': slow_ema,
                    'filter_ema': filter_ema,
                    'rsi_14': rsi_14,
                    'rsi_2': rsi_2,
                    'volume_confirmed': volume_ok,
                    'bb_position': bb_position,
                    'ema_separation_pips': ema_separation_pips,
                    'confidence_score': self._calculate_standard_confidence(latest, 'BULL', volume_ok, ema_separation_pips)
                })
                return signal
        
        elif bearish_cross:
            # BEAR signal conditions
            price_below_fast = current_price < fast_ema
            trend_filter_ok = filter_ema is None or current_price < filter_ema
            rsi_bearish = rsi_14 < 70 and rsi_2 > 20  # Not overbought on 14, not oversold on 2
            bb_not_extreme = bb_position > 0.1
            
            if price_below_fast and trend_filter_ok and rsi_bearish and bb_not_extreme:
                signal = self.create_base_signal('BEAR', epic, timeframe, latest)
                signal.update({
                    'scalping_mode': self.scalping_mode,
                    'entry_reason': 'ema_cross_rsi_confirm',
                    'fast_ema': fast_ema,
                    'slow_ema': slow_ema,
                    'filter_ema': filter_ema,
                    'rsi_14': rsi_14,
                    'rsi_2': rsi_2,
                    'volume_confirmed': volume_ok,
                    'bb_position': bb_position,
                    'ema_separation_pips': ema_separation_pips,
                    'confidence_score': self._calculate_standard_confidence(latest, 'BEAR', volume_ok, ema_separation_pips)
                })
                return signal
        
        return None

    # ================================================================================
    # 🔥 ADAPTIVE REGIME-SPECIFIC SIGNAL DETECTION METHODS
    # ================================================================================

    def _detect_trending_signal(
        self,
        latest: pd.Series,
        previous: pd.Series,
        df: pd.DataFrame,
        epic: str,
        timeframe: str
    ) -> Optional[Dict]:
        """
        🔥 NEW: Detect signals in TRENDING markets using MACD + EMA momentum

        Optimized for strong directional moves (ADX > 20)
        Uses MACD histogram expansion + EMA alignment
        """
        current_price = latest['close']
        fast_ema = latest[f'ema_{self.config["fast"]}']
        slow_ema = latest[f'ema_{self.config["slow"]}']
        fast_ema_prev = previous[f'ema_{self.config["fast"]}']
        slow_ema_prev = previous[f'ema_{self.config["slow"]}']

        # Check for EMA crossover
        bullish_cross = (fast_ema_prev <= slow_ema_prev) and (fast_ema > slow_ema)
        bearish_cross = (fast_ema_prev >= slow_ema_prev) and (fast_ema < slow_ema)

        if not bullish_cross and not bearish_cross:
            return None

        # Calculate MACD manually if not present
        if 'macd_line' in df.columns:
            macd_line = latest['macd_line']
            macd_signal = latest['macd_signal']
            macd_hist = latest['macd_histogram']
            macd_hist_prev = previous.get('macd_histogram', 0)
        else:
            # Quick MACD calculation (12-26-9)
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd_line_series = exp1 - exp2
            macd_signal_series = macd_line_series.ewm(span=9, adjust=False).mean()
            macd_hist_series = macd_line_series - macd_signal_series

            macd_line = macd_line_series.iloc[-1]
            macd_signal = macd_signal_series.iloc[-1]
            macd_hist = macd_hist_series.iloc[-1]
            macd_hist_prev = macd_hist_series.iloc[-2]

        # Check for MACD momentum expansion
        macd_expanding = abs(macd_hist) > abs(macd_hist_prev)

        # Volume confirmation
        volume = latest.get('ltv', latest.get('volume', 0))
        volume_avg = latest.get('volume_avg_10', 1.0)
        volume_ok = volume > (volume_avg * self.volume_threshold) if volume_avg > 0 else True

        if bullish_cross:
            # BULL signal: EMA crossover detected, MACD used for confidence only
            macd_aligned = macd_hist > 0  # MACD agreeing gives bonus confidence

            signal = self.create_base_signal('BULL', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'trending_adaptive',
                'entry_reason': 'trending_ema_crossover_adaptive',
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_hist,
                'macd_aligned': macd_aligned,
                'macd_expanding': macd_expanding,
                'fast_ema': fast_ema,
                'slow_ema': slow_ema,
                'volume_confirmed': volume_ok,
                'confidence_score': self._calculate_trending_confidence(
                    latest, macd_hist, macd_expanding, volume_ok, macd_aligned
                )
            })
            return signal

        elif bearish_cross:
            # BEAR signal: EMA crossover detected, MACD used for confidence only
            macd_aligned = macd_hist < 0  # MACD agreeing gives bonus confidence

            signal = self.create_base_signal('BEAR', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'trending_adaptive',
                'entry_reason': 'trending_ema_crossover_adaptive',
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_hist,
                'macd_aligned': macd_aligned,
                'macd_expanding': macd_expanding,
                'fast_ema': fast_ema,
                'slow_ema': slow_ema,
                'volume_confirmed': volume_ok,
                'confidence_score': self._calculate_trending_confidence(
                    latest, macd_hist, macd_expanding, volume_ok, macd_aligned
                )
            })
            return signal

        return None

    def _detect_ranging_signal(
        self,
        latest: pd.Series,
        previous: pd.Series,
        df: pd.DataFrame,
        epic: str,
        timeframe: str
    ) -> Optional[Dict]:
        """
        🔥 UPDATED: Detect signals in RANGING markets using RSI momentum + BB support/resistance

        Optimized for sideways markets (ADX < 20)
        Uses RSI DIRECTIONALLY (momentum confirmation) instead of overbought/oversold
        - RSI > 50 = bullish momentum building
        - RSI < 50 = bearish momentum building

        Looks for BB bounces with RSI momentum confirmation
        """
        current_price = latest['close']

        # Get RSI (using it directionally for momentum)
        rsi_14 = latest.get('rsi_14', 50)
        rsi_14_prev = previous.get('rsi_14', 50)
        rsi_2 = latest.get('rsi_2', 50)

        # Get Bollinger Bands
        bb_upper = latest.get('bb_upper', current_price * 1.02)
        bb_lower = latest.get('bb_lower', current_price * 0.98)
        bb_middle = latest.get('bb_middle', current_price)

        # Calculate BB position (0 = lower band, 1 = upper band)
        bb_range = bb_upper - bb_lower
        bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5

        # Volume confirmation (reduced requirement from 1.5x to 1.2x)
        volume = latest.get('ltv', latest.get('volume', 0))
        volume_avg = latest.get('volume_avg_10', 1.0)
        volume_ok = volume > (volume_avg * 1.2) if volume_avg > 0 else True

        # RSI momentum checks
        rsi_crossing_above_50 = (rsi_14_prev <= 50) and (rsi_14 > 50)
        rsi_crossing_below_50 = (rsi_14_prev >= 50) and (rsi_14 < 50)
        rsi_bullish_momentum = rsi_14 > 50  # Above midpoint = bullish
        rsi_bearish_momentum = rsi_14 < 50  # Below midpoint = bearish

        # BUY: Price near lower BB + RSI showing bullish momentum (above 50 or crossing above)
        if bb_position < 0.4 and (rsi_crossing_above_50 or (rsi_bullish_momentum and bb_position < 0.3)) and volume_ok:
            signal = self.create_base_signal('BULL', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'ranging_momentum',
                'entry_reason': 'ranging_bb_support_rsi_bullish',
                'rsi_14': rsi_14,
                'rsi_2': rsi_2,
                'rsi_momentum': 'bullish',
                'rsi_crossing_50': rsi_crossing_above_50,
                'bb_position': bb_position,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'volume_ok': volume_ok,
                'confidence_score': self._calculate_ranging_confidence(
                    rsi_14, bb_position, volume_ok, 'BULL'
                )
            })
            return signal

        # SELL: Price near upper BB + RSI showing bearish momentum (below 50 or crossing below)
        elif bb_position > 0.6 and (rsi_crossing_below_50 or (rsi_bearish_momentum and bb_position > 0.7)) and volume_ok:
            signal = self.create_base_signal('BEAR', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'ranging_momentum',
                'entry_reason': 'ranging_bb_resistance_rsi_bearish',
                'rsi_14': rsi_14,
                'rsi_2': rsi_2,
                'rsi_momentum': 'bearish',
                'rsi_crossing_50': rsi_crossing_below_50,
                'bb_position': bb_position,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'volume_ok': volume_ok,
                'confidence_score': self._calculate_ranging_confidence(
                    rsi_14, bb_position, volume_ok, 'BEAR'
                )
            })
            return signal

        return None

    def _calculate_trending_confidence(
        self,
        latest: pd.Series,
        macd_hist: float,
        macd_expanding: bool,
        volume_ok: bool,
        macd_aligned: bool = False
    ) -> float:
        """Calculate confidence for trending market signals"""
        confidence = 0.5

        # MACD alignment (agrees with EMA signal direction)
        if macd_aligned:
            confidence += 0.2  # Big bonus for MACD confirming the crossover direction

        # MACD histogram strength
        if abs(macd_hist) > 0.0001:
            confidence += 0.1
        if abs(macd_hist) > 0.0002:
            confidence += 0.05

        # MACD expansion (momentum building)
        if macd_expanding:
            confidence += 0.1

        # Volume confirmation
        if volume_ok:
            confidence += 0.05

        return min(confidence, 1.0)

    def _calculate_ranging_confidence(
        self,
        rsi_14: float,
        bb_position: float,
        volume_ok: bool,
        signal_type: str
    ) -> float:
        """Calculate confidence for ranging market signals using RSI momentum"""
        confidence = 0.5

        # RSI momentum strength (directional, not overbought/oversold)
        if signal_type == 'BULL':
            # Stronger bullish momentum (RSI further above 50) = higher confidence
            if rsi_14 > 55:
                confidence += 0.15
            elif rsi_14 > 50:
                confidence += 0.10
            # Even if RSI just crossed 50, give some confidence
            elif rsi_14 > 45:
                confidence += 0.05
        else:  # BEAR
            # Stronger bearish momentum (RSI further below 50) = higher confidence
            if rsi_14 < 45:
                confidence += 0.15
            elif rsi_14 < 50:
                confidence += 0.10
            # Even if RSI just crossed 50, give some confidence
            elif rsi_14 < 55:
                confidence += 0.05

        # BB position (closer to edge = better entry)
        if signal_type == 'BULL' and bb_position < 0.2:
            confidence += 0.15
        elif signal_type == 'BULL' and bb_position < 0.3:
            confidence += 0.10
        elif signal_type == 'BEAR' and bb_position > 0.8:
            confidence += 0.15
        elif signal_type == 'BEAR' and bb_position > 0.7:
            confidence += 0.10

        # Volume confirmation
        if volume_ok:
            confidence += 0.10

        return min(confidence, 1.0)

    def _detect_linda_macd_signals(
        self,
        latest: pd.Series,
        previous: pd.Series,
        df: pd.DataFrame,
        epic: str,
        timeframe: str
    ) -> Optional[Dict]:
        """
        🔥 LINDA RASCHKE MACD SIGNALS: Detect signals using 3-10-16 MACD

        Multiple signal types:
        1. MACD Zero Cross - MACD line crosses above/below zero
        2. MACD Signal Cross - MACD crosses signal line
        3. MACD Momentum - Histogram expanding strongly
        4. MACD Pullback - First pullback to signal line (Anti pattern setup)

        These generate MORE signals than EMA crossovers!
        """
        # Calculate Linda Raschke MACD 3-10-16
        linda_macd = self._calculate_linda_macd(df)
        if not linda_macd:
            return None

        macd_line = linda_macd['macd_line']
        signal_line = linda_macd['signal_line']
        histogram = linda_macd['histogram']
        macd_line_prev = linda_macd['macd_line_prev']
        signal_line_prev = linda_macd['signal_line_prev']
        histogram_prev = linda_macd['histogram_prev']
        macd_slope = linda_macd['macd_slope']
        signal_slope = linda_macd['signal_slope']

        current_price = latest['close']

        # Volume confirmation
        volume = latest.get('ltv', latest.get('volume', 0))
        volume_avg = latest.get('volume_avg_10', 1.0)
        volume_ok = volume > (volume_avg * 1.2) if volume_avg > 0 else True

        # Signal Type 1: MACD ZERO CROSS (Strong trend initiation)
        macd_zero_cross_bull = (macd_line_prev <= 0) and (macd_line > 0)
        macd_zero_cross_bear = (macd_line_prev >= 0) and (macd_line < 0)

        if macd_zero_cross_bull:
            signal = self.create_base_signal('BULL', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'linda_macd_zero_cross',
                'entry_reason': 'macd_310_zero_cross_bullish',
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'macd_slope': macd_slope,
                'signal_slope': signal_slope,
                'volume_confirmed': volume_ok,
                'confidence_score': 0.75  # High confidence for zero cross
            })
            return signal

        if macd_zero_cross_bear:
            signal = self.create_base_signal('BEAR', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'linda_macd_zero_cross',
                'entry_reason': 'macd_310_zero_cross_bearish',
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'macd_slope': macd_slope,
                'signal_slope': signal_slope,
                'volume_confirmed': volume_ok,
                'confidence_score': 0.75
            })
            return signal

        # Signal Type 2: MACD SIGNAL LINE CROSS (Standard Linda Raschke)
        macd_signal_cross_bull = (macd_line_prev <= signal_line_prev) and (macd_line > signal_line)
        macd_signal_cross_bear = (macd_line_prev >= signal_line_prev) and (macd_line < signal_line)

        if macd_signal_cross_bull and histogram > 0:
            signal = self.create_base_signal('BULL', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'linda_macd_cross',
                'entry_reason': 'macd_310_signal_cross_bullish',
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'macd_slope': macd_slope,
                'signal_slope': signal_slope,
                'volume_confirmed': volume_ok,
                'confidence_score': 0.65
            })
            return signal

        if macd_signal_cross_bear and histogram < 0:
            signal = self.create_base_signal('BEAR', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'linda_macd_cross',
                'entry_reason': 'macd_310_signal_cross_bearish',
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'macd_slope': macd_slope,
                'signal_slope': signal_slope,
                'volume_confirmed': volume_ok,
                'confidence_score': 0.65
            })
            return signal

        # Signal Type 3: MACD MOMENTUM CONTINUATION (Histogram expanding)
        histogram_expanding = abs(histogram) > abs(histogram_prev)
        histogram_strong = abs(histogram) > 0.0001  # Threshold for meaningful histogram

        if histogram > 0 and histogram_expanding and histogram_strong and macd_slope > 0:
            # Bullish momentum continuation
            signal = self.create_base_signal('BULL', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'linda_macd_momentum',
                'entry_reason': 'macd_310_bullish_momentum',
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'macd_slope': macd_slope,
                'signal_slope': signal_slope,
                'histogram_expanding': True,
                'volume_confirmed': volume_ok,
                'confidence_score': 0.60
            })
            return signal

        if histogram < 0 and histogram_expanding and histogram_strong and macd_slope < 0:
            # Bearish momentum continuation
            signal = self.create_base_signal('BEAR', epic, timeframe, latest)
            signal.update({
                'scalping_mode': 'linda_macd_momentum',
                'entry_reason': 'macd_310_bearish_momentum',
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'macd_slope': macd_slope,
                'signal_slope': signal_slope,
                'histogram_expanding': True,
                'volume_confirmed': volume_ok,
                'confidence_score': 0.60
            })
            return signal

        # Signal Type 4: ANTI PATTERN (First pullback - more complex, needs trend context)
        # This looks for MACD line pulling back toward signal line after establishing trend
        # We'll check if MACD has made a new high/low and is now pulling back
        macd_series = linda_macd['macd_series']

        # Check for recent MACD extreme (within last 5 periods)
        recent_macd = macd_series.tail(10)
        if len(recent_macd) >= 5:
            # Bullish Anti: MACD made new high, now pulling back toward signal but still positive
            macd_recent_high = recent_macd.max()
            is_pullback_from_high = (macd_line < macd_recent_high * 0.9) and (macd_line > signal_line) and (macd_line > 0)
            signal_sloping_up = signal_slope > 0
            macd_approaching_signal = abs(macd_line - signal_line) < abs(macd_line_prev - signal_line_prev)

            if is_pullback_from_high and signal_sloping_up and macd_approaching_signal:
                signal = self.create_base_signal('BULL', epic, timeframe, latest)
                signal.update({
                    'scalping_mode': 'linda_anti_pattern',
                    'entry_reason': 'macd_310_anti_bullish_pullback',
                    'macd_line': macd_line,
                    'signal_line': signal_line,
                    'histogram': histogram,
                    'macd_recent_high': macd_recent_high,
                    'pullback_pct': (macd_recent_high - macd_line) / macd_recent_high if macd_recent_high != 0 else 0,
                    'signal_slope': signal_slope,
                    'volume_confirmed': volume_ok,
                    'confidence_score': 0.70  # High confidence for Anti pattern
                })
                return signal

            # Bearish Anti: MACD made new low, now pulling back toward signal but still negative
            macd_recent_low = recent_macd.min()
            is_pullback_from_low = (macd_line > macd_recent_low * 0.9) and (macd_line < signal_line) and (macd_line < 0)
            signal_sloping_down = signal_slope < 0

            if is_pullback_from_low and signal_sloping_down and macd_approaching_signal:
                signal = self.create_base_signal('BEAR', epic, timeframe, latest)
                signal.update({
                    'scalping_mode': 'linda_anti_pattern',
                    'entry_reason': 'macd_310_anti_bearish_pullback',
                    'macd_line': macd_line,
                    'signal_line': signal_line,
                    'histogram': histogram,
                    'macd_recent_low': macd_recent_low,
                    'pullback_pct': (macd_line - macd_recent_low) / abs(macd_recent_low) if macd_recent_low != 0 else 0,
                    'signal_slope': signal_slope,
                    'volume_confirmed': volume_ok,
                    'confidence_score': 0.70
                })
                return signal

        return None

    def _ensure_scalping_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all scalping indicators are present"""
        df_enhanced = df.copy()
        
        # Add fast EMAs
        for period in [self.config['fast'], self.config['slow']]:
            col_name = f'ema_{period}'
            if col_name not in df_enhanced.columns:
                df_enhanced[col_name] = df_enhanced['close'].ewm(span=period).mean()
        
        # Add filter EMA if specified
        if self.config.get('filter'):
            filter_col = f'ema_{self.config["filter"]}'
            if filter_col not in df_enhanced.columns:
                df_enhanced[filter_col] = df_enhanced['close'].ewm(span=self.config['filter']).mean()
        
        # Add RSI indicators
        if 'rsi_2' not in df_enhanced.columns:
            df_enhanced = self._add_rsi(df_enhanced, 2)
        if 'rsi_14' not in df_enhanced.columns:
            df_enhanced = self._add_rsi(df_enhanced, 14)
        
        # Add volume average
        if 'volume_avg_10' not in df_enhanced.columns and 'ltv' in df_enhanced.columns:
            df_enhanced['volume_avg_10'] = df_enhanced['ltv'].rolling(window=10).mean()
        
        # Add Bollinger Bands if not present
        if 'bb_middle' not in df_enhanced.columns:
            df_enhanced = self._add_bollinger_bands(df_enhanced)
        
        return df_enhanced
    
    def _add_rsi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add RSI indicator"""
        df_rsi = df.copy()
        delta = df_rsi['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        df_rsi[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df_rsi
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands"""
        df_bb = df.copy()
        df_bb['bb_middle'] = df_bb['close'].rolling(window=period).mean()
        bb_std = df_bb['close'].rolling(window=period).std()
        df_bb['bb_upper'] = df_bb['bb_middle'] + (bb_std * std_dev)
        df_bb['bb_lower'] = df_bb['bb_middle'] - (bb_std * std_dev)
        df_bb['bb_position'] = (df_bb['close'] - df_bb['bb_lower']) / (df_bb['bb_upper'] - df_bb['bb_lower'])
        return df_bb
    
    def _check_volume_confirmation(self, latest: pd.Series) -> bool:
        """Check if current volume supports the signal"""
        current_volume = latest.get('ltv', 0)
        avg_volume = latest.get('volume_avg_10', 1)
        return current_volume > (avg_volume * self.volume_threshold)
    
    def _get_bb_position(self, latest: pd.Series) -> float:
        """Get Bollinger Band position (0=lower band, 1=upper band)"""
        return latest.get('bb_position', 0.5)
    
    # 🔥 LEGACY METHOD COMPATIBILITY: Keep old methods for backward compatibility but redirect to enhanced validation
    def _calculate_ultra_fast_confidence(self, latest: pd.Series, signal_type: str) -> float:
        """Calculate confidence for ultra-fast scalping (legacy method - now uses enhanced validation)"""
        base_confidence = 0.65  # Lower base due to high-frequency nature
        
        # Volume bonus
        volume_bonus = 0.1 if self._check_volume_confirmation(latest) else 0
        
        # RSI position bonus (not too extreme)
        rsi_2 = latest.get('rsi_2', 50)
        if signal_type == 'BULL' and 20 < rsi_2 < 80:
            rsi_bonus = 0.05
        elif signal_type == 'BEAR' and 20 < rsi_2 < 80:
            rsi_bonus = 0.05
        else:
            rsi_bonus = 0
        
        return min(0.85, base_confidence + volume_bonus + rsi_bonus)  # Max 85% for ultra-fast
    
    def _calculate_dual_ma_confidence(self, latest: pd.Series, signal_type: str, volume_ok: bool) -> float:
        """Calculate confidence for dual MA scalping (legacy method - now uses enhanced validation)"""
        base_confidence = 0.70
        
        # Volume bonus
        volume_bonus = 0.15 if volume_ok else 0
        
        # BB position bonus (avoid extremes)
        bb_position = self._get_bb_position(latest)
        if 0.2 < bb_position < 0.8:
            bb_bonus = 0.1
        else:
            bb_bonus = 0
        
        return min(0.90, base_confidence + volume_bonus + bb_bonus)
    
    def _calculate_standard_confidence(
        self, 
        latest: pd.Series, 
        signal_type: str, 
        volume_ok: bool, 
        ema_separation: float
    ) -> float:
        """Calculate confidence for standard scalping (legacy method - now uses enhanced validation)"""
        base_confidence = 0.75
        
        # Volume bonus
        volume_bonus = 0.1 if volume_ok else 0
        
        # EMA separation bonus (wider separation = stronger signal)
        separation_bonus = min(0.1, ema_separation * 0.02)  # Max 10% bonus
        
        # RSI confirmation bonus
        rsi_14 = latest.get('rsi_14', 50)
        if signal_type == 'BULL' and 30 < rsi_14 < 70:
            rsi_bonus = 0.05
        elif signal_type == 'BEAR' and 30 < rsi_14 < 70:
            rsi_bonus = 0.05
        else:
            rsi_bonus = 0
        
        return min(0.95, base_confidence + volume_bonus + separation_bonus + rsi_bonus)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame has minimum required data for scalping"""
        try:
            # Basic validation
            if len(df) < 20:
                return False
            
            # Check for essential columns
            essential_cols = ['close', 'high', 'low', 'open']
            missing_essential = [col for col in essential_cols if col not in df.columns]
            if missing_essential:
                self.logger.warning(f"⚠️ Missing essential columns: {missing_essential}")
                return False
            
            # Check for NaN values in latest data
            latest = df.iloc[-1]
            for col in essential_cols:
                if pd.isna(latest[col]):
                    self.logger.warning(f"⚠️ NaN value in essential column {col}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data validation error: {e}")
            return False
    
    def create_base_signal(self, signal_type: str, epic: str, timeframe: str, latest: pd.Series) -> Dict:
        """Create base signal dictionary with SAFE TIMESTAMP"""
        return {
            'epic': epic,
            'signal_type': signal_type,
            'strategy': self.name,
            'timeframe': timeframe,
            'price': latest['close'],
            'timestamp': self._convert_market_timestamp_safe(latest.name if hasattr(latest, 'name') else pd.Timestamp.now())
        }
    
    def get_exit_signal(self, entry_signal: Dict, current_data: pd.Series) -> Optional[str]:
        """Generate exit signals for scalping (quick profit taking)"""
        if not self.quick_exit_enabled:
            return None
        
        entry_price = entry_signal.get('execution_price', entry_signal.get('price'))
        current_price = current_data['close']
        signal_type = entry_signal['signal_type']
        
        # Quick profit targets (3-8 pips depending on mode)
        profit_targets = {
            'ultra_fast': 3,    # 3 pips for ultra-fast
            'aggressive': 5,    # 5 pips for aggressive
            'conservative': 8,  # 8 pips for conservative
            'dual_ma': 5        # 5 pips for dual MA
        }
        
        target_pips = profit_targets.get(self.scalping_mode, 5)
        target_price_diff = target_pips / 10000
        
        if signal_type == 'BULL':
            if current_price >= entry_price + target_price_diff:
                return 'TAKE_PROFIT'
            elif current_price <= entry_price - (target_price_diff * 0.6):  # Tight stop loss
                return 'STOP_LOSS'
        else:  # BEAR
            if current_price <= entry_price - target_price_diff:
                return 'TAKE_PROFIT'
            elif current_price >= entry_price + (target_price_diff * 0.6):  # Tight stop loss
                return 'STOP_LOSS'
        
        return None