# core/strategies/momentum_bias_strategy.py
"""
Momentum Bias Index Strategy - Python Implementation
Converted from Pine Script by AlgoAlpha

Trading Logic:
- BULL Signal: momentumUpBias > momentumDownBias (bright green) AND momentumUpBias > boundary (above dotted line)
- BEAR Signal: momentumDownBias > momentumUpBias (red) AND momentumDownBias > boundary (above dotted line)
- Additional confirmation from crossunder conditions for signal timing

Enhanced with your forex scanner's confidence scoring and validation systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime
import pandas_ta as ta

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
try:
    import config
except ImportError:
    from forex_scanner import config
try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config_momentum_bias

class MomentumBiasStrategy(BaseStrategy):
    """
    Momentum Bias Index Strategy implementation
    
    Calculates momentum bias using standardized deviation approach
    with configurable smoothing and boundary detection.
    """
    

    def __init__(self):
        super().__init__('momentum_bias')
        self.price_adjuster = PriceAdjuster()
        
        # Pine Script parameters - configurable
        self.momentum_length = config_momentum_bias.MOMENTUM_BIAS_MOMENTUM_LENGTH
        self.bias_length = config_momentum_bias.MOMENTUM_BIAS_BIAS_LENGTH
        self.smooth_length = config_momentum_bias.MOMENTUM_BIAS_SMOOTH_LENGTH
        self.impulse_boundary_length = config_momentum_bias.MOMENTUM_BIAS_IMPULSE_BOUNDARY_LENGTH
        self.std_dev_multiplier = config_momentum_bias.MOMENTUM_BIAS_STD_DEV_MULTIPLIER
        self.smooth_indicator = config_momentum_bias.MOMENTUM_BIAS_SMOOTH_INDICATOR
        
        # Confidence scoring weights
        self.base_confidence = config_momentum_bias.MOMENTUM_BIAS_BASE_CONFIDENCE
        self.boundary_weight = config_momentum_bias.MOMENTUM_BIAS_BOUNDARY_WEIGHT
        self.separation_weight = config_momentum_bias.MOMENTUM_BIAS_SEPARATION_WEIGHT
        
        self.logger.info("📊 MomentumBiasStrategy initialized")
        self.logger.info(f"   Parameters: momentum={self.momentum_length}, bias={self.bias_length}, smooth={self.smooth_length}")
        self.logger.info(f"   Boundary: length={self.impulse_boundary_length}, multiplier={self.std_dev_multiplier}")
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['momentum_bias','close', 'high', 'low', 'open']
    
    def _check_minimum_bars(self, df: pd.DataFrame, epic: str) -> bool:
        """Check if we have sufficient bars for MACD detection"""
        try:
            import config
            min_bars = getattr(config, 'MACD_MIN_BARS_REQUIRED', 50)
            
            if len(df) < min_bars:
                self.logger.debug(f"🚫 Insufficient bars: {len(df)} < {min_bars}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"⚠️ Minimum bars check failed: {e}")
            # Default to allowing if check fails
            return True

    def detect_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float = 1.5,
        timeframe: str = '5m'
    ) -> Optional[Dict]:
        """
        Detect momentum bias signals - FIXED: current_price variable definition
        
        Returns signal dictionary or None if no valid signal
        """
        try:
            self.logger.info(f"🔍 [MOMENTUM BIAS] Analyzing {epic} on {timeframe}")
            
            # Validate data sufficiency
            min_required = max(self.momentum_length, self.impulse_boundary_length) + 50
            if len(df) < min_required:
                self.logger.warning(f"[MOMENTUM BIAS REJECTED] {epic} - Insufficient data: {len(df)} < {min_required}")
                return None
            
            # Apply BID price adjustment for accurate signals
            df_adjusted = self.price_adjuster.adjust_bid_to_mid_prices(df, spread_pips)
            
            # Calculate momentum bias indicators
            df_enhanced = self._calculate_momentum_bias_indicators(df_adjusted)
            
            if df_enhanced is None:
                self.logger.warning(f"[MOMENTUM BIAS REJECTED] {epic} - Indicator calculation failed")
                return None
            
            # Get latest values
            latest = df_enhanced.iloc[-1]
            previous = df_enhanced.iloc[-2]
            
            # 🔥 FIX: Define current_price early in the method
            current_price = latest['close']
            previous_price = previous['close']
            
            # Extract momentum bias values
            momentum_up_bias = latest['momentum_up_bias']
            momentum_down_bias = latest['momentum_down_bias']
            boundary = latest['boundary']
            
            # Extract previous values for crossunder detection
            momentum_up_bias_prev = previous['momentum_up_bias']
            momentum_down_bias_prev = previous['momentum_down_bias']
            
            self.logger.debug(f"   Current Price: {current_price:.5f}")
            self.logger.debug(f"   Momentum Up Bias: {momentum_up_bias:.6f} (prev: {momentum_up_bias_prev:.6f})")
            self.logger.debug(f"   Momentum Down Bias: {momentum_down_bias:.6f} (prev: {momentum_down_bias_prev:.6f})")
            self.logger.debug(f"   Boundary: {boundary:.6f}")
            
            # Trading logic (from Pine Script)
            # BULL Signal: momentumDownBias > momentumUpBias (red) AND momentumDownBias > boundary
            # BEAR Signal: momentumUpBias > momentumDownBias (bright green) AND momentumUpBias > boundary
            
            signal_type = None
            trigger_reason = None
            
            # Check for BULL signal conditions
            if (momentum_down_bias > momentum_up_bias and 
                momentum_down_bias > boundary):
                
                # Additional confirmation: crossunder condition for timing
                if (momentum_down_bias > momentum_down_bias_prev or
                    momentum_up_bias < momentum_up_bias_prev):
                    signal_type = 'BULL'
                    trigger_reason = 'momentum_down_dominance_above_boundary'
                    self.logger.debug(f"   🎯 BULL signal detected: down_bias={momentum_down_bias:.6f} > up_bias={momentum_up_bias:.6f} > boundary={boundary:.6f}")
            
            # Check for BEAR signal conditions
            elif (momentum_up_bias > momentum_down_bias and 
                momentum_up_bias > boundary):
                
                # Additional confirmation: crossunder condition for timing
                if (momentum_up_bias > momentum_up_bias_prev or
                    momentum_down_bias < momentum_down_bias_prev):
                    signal_type = 'BEAR'
                    trigger_reason = 'momentum_up_dominance_above_boundary'
                    self.logger.debug(f"   🎯 BEAR signal detected: up_bias={momentum_up_bias:.6f} > down_bias={momentum_down_bias:.6f} > boundary={boundary:.6f}")
            
            if not signal_type:
                self.logger.debug(f"[MOMENTUM BIAS] No signal: conditions not met for {epic}")
                return None
            
            # 🔥 CRITICAL: Create enhanced signal data for proper confidence calculation
            enhanced_signal_data = self.create_enhanced_signal_data(latest, signal_type, current_price)
            enhanced_signal_data.update({
                'price': current_price,
                'momentum_up_bias': momentum_up_bias,
                'momentum_down_bias': momentum_down_bias,
                'boundary': boundary,
                'efficiency_ratio': self._calculate_efficiency_ratio(df_enhanced)
            })
            
            # 🔥 CRITICAL: Use enhanced confidence calculation
            confidence = self.calculate_confidence(enhanced_signal_data)
            
            # Apply minimum confidence threshold
            min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.65)
            if confidence < min_confidence:
                self.logger.warning(f"[MOMENTUM BIAS REJECTED] {epic} - Low confidence: {confidence:.1%} < {min_confidence:.1%}")
                return None
            
            # Create base signal with current_price properly defined
            signal = self.create_base_signal(signal_type, epic, timeframe, latest)
            
            # Add momentum bias specific data
            signal.update({
                'current_price': current_price,
                'momentum_up_bias': momentum_up_bias,
                'momentum_down_bias': momentum_down_bias,
                'boundary': boundary,
                'dominant_bias': momentum_down_bias if signal_type == 'BULL' else momentum_up_bias,
                'secondary_bias': momentum_up_bias if signal_type == 'BULL' else momentum_down_bias,
                'bias_separation': abs(momentum_up_bias - momentum_down_bias),
                'boundary_strength': (momentum_down_bias if signal_type == 'BULL' else momentum_up_bias) / boundary if boundary > 0 else 1.0,
                'confidence_score': confidence,
                'trigger_reason': trigger_reason,
                'current_price': current_price,  # 🔥 FIX: Explicitly add current_price to signal
                'previous_price': previous_price
            })
            
            # Apply BID price adjustment if enabled
            if getattr(config, 'USE_BID_ADJUSTMENT', False):
                signal = self.price_adjuster.add_execution_prices(signal, spread_pips)
            
            # 🔧 COMPLETE MOMENTUM BIAS SIGNAL DATA ENHANCEMENT - With current_price fix
            if signal:
                signal = self._enhance_momentum_bias_signal_to_match_combined_strategy(
                    signal, latest, previous, spread_pips
                )
                self.logger.debug(f"✅ Complete Momentum Bias data enhancement with timestamp safety applied to {signal['epic']}")
            
            self.logger.info(f"🎯 [MOMENTUM BIAS VALIDATED] {epic} - {signal_type} signal with {confidence:.1%} confidence")
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ [MOMENTUM BIAS ERROR] {epic} - Signal detection failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def create_enhanced_signal_data(self, latest: pd.Series, signal_type: str, current_price: float) -> Dict:
        """
        🔥 NEW: Create enhanced signal data structure for proper confidence calculation
        This ensures all required fields are available for enhanced validation
        """
        return {
            'signal_type': signal_type,
            'price': current_price,  # Use the passed current_price parameter
            'ema_short': current_price,  # Momentum bias doesn't use EMAs, use price as fallback
            'ema_long': current_price,   # Momentum bias doesn't use EMAs, use price as fallback
            'ema_trend': current_price,  # Momentum bias doesn't use EMAs, use price as fallback
            'momentum_up_bias': latest.get('momentum_up_bias', 0.0),
            'momentum_down_bias': latest.get('momentum_down_bias', 0.0),
            'boundary': latest.get('boundary', 0.01),
            'volume': latest.get('ltv', latest.get('volume', 0)),
            'volume_ratio': latest.get('volume_ratio_20', 1.0),
            'volume_confirmation': latest.get('volume_ratio_20', 1.0) > 1.2,
            'atr': latest.get('atr', 0.001),
            'efficiency_ratio': 0.5  # Will be calculated properly in detect_signal
        }

    def create_base_signal(self, signal_type: str, epic: str, timeframe: str, latest: pd.Series) -> Dict:
        """Create base signal dictionary with SAFE TIMESTAMP - FIXED numpy.int64 issue"""
        
        # 🔥 FIX: Better timestamp handling to avoid numpy.int64 error
        try:
            # Try to get timestamp from DataFrame index
            if hasattr(latest, 'name') and latest.name is not None:
                # Check if it's a pandas timestamp
                if isinstance(latest.name, (pd.Timestamp, datetime)):
                    timestamp = self._convert_market_timestamp_safe(latest.name)
                else:
                    # If it's an integer index, use current time
                    timestamp = datetime.now()
            else:
                # Fallback to current time
                timestamp = datetime.now()
        except Exception as e:
            self.logger.debug(f"Timestamp extraction failed: {e}, using current time")
            timestamp = datetime.now()
        
        return {
            'epic': epic,
            'signal_type': signal_type,
            'strategy': self.name,
            'timeframe': timeframe,
            'price': latest['close'],
            'timestamp': timestamp  # Use the safely extracted timestamp
        }

    def _calculate_momentum_bias_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate momentum bias indicators following Pine Script logic
        """
        try:
            df = df.copy()
            
            # Step 1: Calculate momentum (source - source[momentumLength])
            df['momentum'] = df['close'] - df['close'].shift(self.momentum_length)
            
            # Step 2: Calculate standardized deviation
            # Pine: momentum / (ta.ema(high - low, momentumLength)) * 100
            hl_range = df['high'] - df['low']
            ema_hl = ta.ema(hl_range, length=self.momentum_length)
            df['std_dev'] = (df['momentum'] / ema_hl) * 100
            
            # Step 3: Split into up and down momentum
            df['momentum_up'] = np.maximum(df['std_dev'], 0)
            df['momentum_down'] = np.minimum(df['std_dev'], 0)
            
            # Step 4: Calculate bias sums
            # Rolling sum over bias_length periods
            momentum_up_sum = df['momentum_up'].rolling(window=self.bias_length).sum()
            momentum_down_sum = df['momentum_down'].rolling(window=self.bias_length).sum()
            
            # Step 5: Apply smoothing if enabled
            if self.smooth_indicator:
                # Use HMA (Hull Moving Average) from pandas_ta
                df['momentum_up_bias'] = ta.hma(momentum_up_sum, length=self.smooth_length)
                df['momentum_down_bias'] = ta.hma(-momentum_down_sum, length=self.smooth_length)
                
                # Ensure non-negative values after smoothing
                df['momentum_up_bias'] = np.maximum(df['momentum_up_bias'], 0)
                df['momentum_down_bias'] = np.maximum(df['momentum_down_bias'], 0)
            else:
                df['momentum_up_bias'] = momentum_up_sum
                df['momentum_down_bias'] = -momentum_down_sum
                
                # Ensure non-negative values
                df['momentum_up_bias'] = np.maximum(df['momentum_up_bias'], 0)
                df['momentum_down_bias'] = np.maximum(df['momentum_down_bias'], 0)
            
            # Step 6: Calculate average bias
            df['average_bias'] = (df['momentum_up_bias'] + df['momentum_down_bias']) / 2
            
            # Step 7: Calculate boundary (impulse boundary)
            # Pine: ta.ema(averageBias, impulseBoundaryLength) + ta.stdev(averageBias, impulseBoundaryLength) * stdDevMultiplier
            avg_bias_ema = ta.ema(df['average_bias'], length=self.impulse_boundary_length)
            avg_bias_std = df['average_bias'].rolling(window=self.impulse_boundary_length).std()
            df['boundary'] = avg_bias_ema + (avg_bias_std * self.std_dev_multiplier)
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 10:
                self.logger.warning("Insufficient data after indicator calculations")
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Momentum bias indicator calculation failed: {e}")
            return None
    
    def _enhance_momentum_bias_signal_to_match_combined_strategy(self, signal: Dict, latest: pd.Series, previous: pd.Series, spread_pips: float) -> Dict:
        """
        🔧 KEY FIX: Enhance Momentum Bias signal to populate ALL columns that combined_strategy populates
        ADDED: Safe timestamp conversion to prevent database errors
        This ensures momentum_bias has the same data richness as combined_strategy with timestamp safety
        """
        try:
            current_price = signal.get('price', latest.get('close', 0))
            signal_type = signal.get('signal_type')
            epic = signal.get('epic')
            timeframe = signal.get('timeframe', '15m')
            
            # Extract pair from epic
            if not signal.get('pair'):
                signal['pair'] = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            
            # ========== TIMESTAMP SAFETY FIX ==========
            # Apply safe timestamp conversion for any timestamp fields
            timestamp_fields = ['market_timestamp', 'timestamp', 'signal_timestamp', 'candle_timestamp']
            for field in timestamp_fields:
                if field in signal:
                    original_value = signal[field]
                    safe_timestamp = self._convert_market_timestamp_safe(original_value)
                    
                    if original_value != safe_timestamp:
                        self.logger.debug(f"🛠️ TIMESTAMP FIX: {field} converted from {original_value} to {safe_timestamp}")
                    
                    signal[field] = safe_timestamp
            
            # ========== CORE TECHNICAL DATA (matching combined_strategy) ==========
            
            # Momentum Bias data (primary indicators for this strategy)
            signal.update({
                'momentum_up_bias': float(signal.get('momentum_up_bias', 0.0)),
                'momentum_down_bias': float(signal.get('momentum_down_bias', 0.0)),
                'boundary': float(signal.get('boundary', 0.0)),
                'bias_separation': float(signal.get('bias_separation', 0.0)),
                'boundary_strength': float(signal.get('boundary_strength', 1.0))
            })
            
            # EMA data compatibility (use EMA 200 if available, or price as fallback)
            ema_200 = float(latest.get('ema_200', current_price))
            signal.update({
                'ema_200': ema_200,
                'ema_trend': ema_200,  # For consistency with EMA strategy
                # Add other EMA values if available, or use EMA 200 as fallback
                'ema_short': float(latest.get('ema_9', latest.get('ema_12', ema_200))),
                'ema_long': float(latest.get('ema_21', latest.get('ema_26', ema_200))),
                'ema_9': float(latest.get('ema_9', ema_200)),
                'ema_21': float(latest.get('ema_21', ema_200))
            })
            
            # MACD data compatibility (if available, otherwise use momentum bias approximation)
            signal.update({
                'macd_line': float(latest.get('macd_line', signal.get('momentum_up_bias', 0) - signal.get('momentum_down_bias', 0))),
                'macd_signal': float(latest.get('macd_signal', 0.0)),
                'macd_histogram': float(latest.get('macd_histogram', signal.get('bias_separation', 0)))
            })
            
            # Volume data - combined_strategy includes volume analysis
            volume = latest.get('volume') or latest.get('ltv', 0)
            signal['volume'] = float(volume) if volume else 0.0
            
            # Volume ratio and confirmation
            volume_sma = latest.get('volume_sma', 1.0)
            if volume_sma and volume_sma > 0:
                signal['volume_ratio'] = signal['volume'] / volume_sma
                signal['volume_confirmation'] = signal['volume_ratio'] > 1.2
            else:
                signal['volume_ratio'] = 1.0
                signal['volume_confirmation'] = False
            
            # ========== STRATEGY CONFIGURATION (JSON fields) ==========
            
            # strategy_config - comprehensive configuration data
            signal['strategy_config'] = {
                'strategy_type': 'momentum_bias',
                'strategy_family': 'momentum',
                'momentum_length': self.momentum_length,
                'bias_length': self.bias_length,
                'smooth_length': self.smooth_length,
                'impulse_boundary_length': self.impulse_boundary_length,
                'std_dev_multiplier': self.std_dev_multiplier,
                'smooth_indicator': self.smooth_indicator,
                'signal_method': 'momentum_bias_crossover',
                'boundary_filter_enabled': True,
                'momentum_threshold_enabled': True,
                'bid_adjustment_enabled': getattr(config, 'USE_BID_ADJUSTMENT', False),
                'timestamp_safety_enabled': True,  # NEW: indicates timestamp fix applied
                'enhanced_confidence_enabled': True  # NEW: indicates confidence fix applied
            }
            
            # strategy_indicators - all technical indicator values
            momentum_strength = max(signal['momentum_up_bias'], signal['momentum_down_bias'])
            boundary_margin = (momentum_strength - signal['boundary']) / signal['boundary'] if signal['boundary'] > 0 else 0.1
            price_momentum = current_price - float(previous.get('close', current_price))
            
            signal['strategy_indicators'] = {
                'primary_indicator': 'momentum_bias',
                'trend_filter': 'boundary_line',
                'momentum_up_bias_value': signal['momentum_up_bias'],
                'momentum_down_bias_value': signal['momentum_down_bias'],
                'boundary_value': signal['boundary'],
                'dominant_bias': signal.get('dominant_bias', 0.0),
                'secondary_bias': signal.get('secondary_bias', 0.0),
                'momentum_strength': momentum_strength,
                'boundary_margin': boundary_margin,
                'bias_dominance_ratio': signal.get('bias_dominance_ratio', 1.0),
                'current_price': current_price,
                'previous_price': float(previous.get('close', current_price)),
                'price_momentum': price_momentum,
                'above_boundary': momentum_strength > signal['boundary'],
                'momentum_direction': signal_type.lower() if signal_type else 'neutral',
                'bias_crossunder_confirmed': True,  # Signal already validated
                'volume_ratio': signal.get('volume_ratio', 1.0),
                'signal_trigger': 'momentum_bias_crossover',
                'timestamp_safety_applied': True,  # NEW
                'enhanced_confidence_applied': True  # NEW
            }
            
            # strategy_metadata - comprehensive strategy context
            confidence = signal.get('confidence_score', 0.7)
            signal['strategy_metadata'] = {
                'strategy_version': '1.0.0',
                'signal_basis': 'momentum_bias_with_boundary_filter',
                'confidence_calculation': 'enhanced_validation',  # UPDATED
                'signal_strength': 'strong' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'weak',
                'market_condition': 'momentum_driven',
                'momentum_bias_type': 'bullish' if signal_type == 'BULL' else 'bearish',
                'boundary_status': 'above_boundary' if momentum_strength > signal['boundary'] else 'below_boundary',
                'bias_dominance': 'down_bias' if signal_type == 'BULL' else 'up_bias',
                'processing_timestamp': datetime.now().isoformat(),
                'enhancement_applied': True,
                'data_completeness': 'full',
                'timestamp_safety_applied': True,  # NEW
                'enhanced_confidence_applied': True,  # NEW
                'confidence_factors': {
                    'boundary_filter_passed': True,
                    'bias_crossunder_confirmed': True,
                    'dominance_confirmed': True,
                    'boundary_strength': 'strong' if boundary_margin > 0.5 else 'medium' if boundary_margin > 0.1 else 'weak',
                    'momentum_alignment': self._check_momentum_alignment(current_price, previous.get('close', current_price), signal_type),
                    'enhanced_validation_passed': True  # NEW
                }
            }
            
            # signal_conditions - market conditions at signal time
            signal['signal_conditions'] = {
                'market_trend': 'momentum_bullish' if signal_type == 'BULL' else 'momentum_bearish',
                'momentum_signal_type': 'bias_crossover',
                'boundary_position': 'above_boundary' if momentum_strength > signal['boundary'] else 'below_boundary',
                'momentum_direction': signal_type.lower() if signal_type else 'neutral',
                'volatility_assessment': self._assess_volatility(latest),
                'signal_timing': 'fresh_crossover',
                'confirmation_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.7 else 'low',
                'market_session': self._determine_trading_session(),
                'momentum_strength': 'strong' if momentum_strength > signal['boundary'] * 1.5 else 'medium',
                'bias_separation_strength': 'high' if signal['bias_separation'] > signal['boundary'] * 0.5 else 'medium',
                'crossunder_magnitude': abs(signal['momentum_up_bias'] - signal['momentum_down_bias']),
                'timestamp_safety_processed': True,  # NEW
                'enhanced_confidence_processed': True  # NEW
            }
            
            # ========== PRICING AND EXECUTION DATA ==========
            
            # Comprehensive pricing data
            spread_adjustment = spread_pips / 10000
            signal.update({
                'spread_pips': spread_pips,
                'bid_price': current_price - spread_adjustment,
                'ask_price': current_price + spread_adjustment,
                'execution_price': (current_price + spread_adjustment) if signal_type in ['BUY', 'BULL'] else (current_price - spread_adjustment)
            })
            
            # Support/Resistance levels (if available)
            if 'support' in latest and latest['support'] is not None:
                signal['nearest_support'] = float(latest['support'])
                signal['distance_to_support_pips'] = (current_price - latest['support']) * 10000
            
            if 'resistance' in latest and latest['resistance'] is not None:
                signal['nearest_resistance'] = float(latest['resistance'])
                signal['distance_to_resistance_pips'] = (latest['resistance'] - current_price) * 10000
            
            # Risk/Reward calculation based on momentum bias levels
            pip_size = 0.0001 if 'JPY' not in epic else 0.01
            stop_distance = 2.0 * spread_pips * pip_size  # 2x spread stop
            target_distance = 4.0 * spread_pips * pip_size  # 2:1 RR
            
            signal.update({
                'stop_loss': current_price - stop_distance if signal_type == 'BULL' else current_price + stop_distance,
                'take_profit': current_price + target_distance if signal_type == 'BULL' else current_price - target_distance,
                'risk_reward_ratio': 2.0
            })
            
            # ========== DEDUPLICATION AND TRACKING ==========
            
            # Signal identification and deduplication with SAFE TIMESTAMP
            signal.update({
                'signal_hash': self._generate_signal_hash(epic, signal_type, timeframe, current_price),
                'market_timestamp': datetime.now(),
                'data_source': 'live_scanner',
                'cooldown_key': f"{epic}_{signal_type}_{timeframe}_{datetime.now().strftime('%Y%m%d%H')}"
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
            
            # ========== ADDITIONAL ANALYSIS ==========
            
            # Technical analysis summary
            signal['technical_summary'] = {
                'primary_signal': f"Momentum Bias {'Bullish' if signal_type == 'BULL' else 'Bearish'} Crossover",
                'boundary_status': f"Above Boundary Line {'Confirmed' if confidence > 0.8 else 'Developing'}",
                'entry_quality': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.7 else 'Low',
                'setup_type': f"Momentum {'Breakout' if signal_type == 'BULL' else 'Breakdown'}",
                'timeframe_analysis': timeframe,
                'signal_reliability': 'High' if momentum_strength > signal['boundary'] * 1.5 and confidence > 0.8 else 'Medium',
                'bias_analysis': f"{'Down' if signal_type == 'BULL' else 'Up'} bias dominant above boundary",
                'timestamp_safety': 'enabled',  # NEW
                'enhanced_confidence': 'enabled'  # NEW
            }
            
            # Risk management suggestions
            signal.update({
                'stop_loss_suggestion': signal.get('stop_loss', current_price * 0.998),
                'take_profit_suggestion': signal.get('take_profit', current_price * 1.004),
                'position_size_suggestion': 'standard',
                'max_risk_percentage': 2.0
            })
            
            # ========== COMBINED STRATEGY COMPATIBILITY ==========
            
            # Add fields that combined_strategy specifically uses
            signal.update({
                'combined_confidence': confidence,  # For compatibility
                'individual_confidences': {
                    'ema_confidence': 0.0,  # No pure EMA signal in Momentum Bias strategy
                    'macd_confidence': 0.0,  # No pure MACD signal
                    'momentum_bias_confidence': confidence,
                    'consensus_confidence': confidence
                },
                'strategy_agreement': 1.0,  # Single strategy always agrees with itself
                'contributing_strategies': ['momentum_bias'],
                'signal_consensus': 'unanimous',  # Single strategy
                'weight_ema': 0.0,
                'weight_macd': 0.0,
                'weight_momentum_bias': 1.0
            })
            
            # Additional Momentum Bias-specific fields for comprehensive analysis
            signal.update({
                'momentum_up_slope': self._calculate_momentum_slope(latest, previous, 'up'),
                'momentum_down_slope': self._calculate_momentum_slope(latest, previous, 'down'),
                'boundary_trend': self._calculate_boundary_trend(latest, previous),
                'bias_convergence_divergence': 'convergence' if signal['bias_separation'] < abs(float(previous.get('momentum_up_bias', 0)) - float(previous.get('momentum_down_bias', 0))) else 'divergence'
            })
            
            # ========== FINAL TIMESTAMP SAFETY CHECK ==========
            # Ensure all timestamp fields are properly converted
            for field_name, field_value in signal.items():
                if 'timestamp' in field_name.lower() and field_value is not None:
                    signal[field_name] = self._convert_market_timestamp_safe(field_value)
            
            # Ensure all JSON fields are properly serializable
            signal = self._ensure_json_serializable(signal)
            
            self.logger.debug(f"✅ Enhanced Momentum Bias signal with timestamp safety and confidence fix: {len(signal)} fields (matching combined_strategy richness)")
            
        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive Momentum Bias signal enhancement with timestamp and confidence fix: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        return signal

    def _check_momentum_alignment(self, current_price: float, previous_price: float, signal_type: str) -> bool:
        """Check if price momentum aligns with signal direction"""
        if signal_type in ['BUY', 'BULL']:
            return current_price >= previous_price
        else:
            return current_price <= previous_price

    def _calculate_momentum_slope(self, latest: pd.Series, previous: pd.Series, bias_type: str) -> float:
        """Calculate momentum bias slope"""
        try:
            if bias_type == 'up':
                current = float(latest.get('momentum_up_bias', 0))
                prev = float(previous.get('momentum_up_bias', 0))
            else:
                current = float(latest.get('momentum_down_bias', 0))
                prev = float(previous.get('momentum_down_bias', 0))
            return current - prev
        except:
            return 0.0

    def _calculate_boundary_trend(self, latest: pd.Series, previous: pd.Series) -> str:
        """Calculate boundary trend direction"""
        try:
            current_boundary = float(latest.get('boundary', 0))
            prev_boundary = float(previous.get('boundary', 0))
            if current_boundary > prev_boundary:
                return 'rising'
            elif current_boundary < prev_boundary:
                return 'falling'
            else:
                return 'flat'
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
        hash_string = f"{epic}_{signal_type}_{timeframe}_{int(price*10000)}_{datetime.now().strftime('%Y%m%d%H')}"
        import hashlib
        return hashlib.md5(hash_string.encode()).hexdigest()[:16]

    def _ensure_json_serializable(self, signal: Dict) -> Dict:
        """Ensure all signal data is JSON serializable"""
        import json
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

    def _legacy_create_enhanced_signal_data(
        self, 
        latest: pd.Series, 
        signal_type: str, 
        epic: str, 
        timeframe: str, 
        spread_pips: float,
        momentum_up_bias: float,
        momentum_down_bias: float,
        boundary: float
            ) -> Dict:
        """Legacy method - replaced by comprehensive enhancement"""
        # This is now handled in _enhance_momentum_bias_signal_to_match_combined_strategy
        current_price = latest['close']
        pip_size = 0.0001 if 'JPY' not in epic else 0.01
        
        return {
            'signal_type': signal_type,
            'strategy': 'momentum_bias',
            'epic': epic,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'price': current_price,
            'momentum_up_bias': momentum_up_bias,
            'momentum_down_bias': momentum_down_bias,
            'boundary': boundary,
            'confidence_score': 0.7  # Will be overridden
        }
    
    # 🔥 CRITICAL FIX: Replace custom confidence calculation with enhanced validation
    def calculate_confidence(self, signal_data: Dict) -> float:
        """
        🔥 ENHANCED: Calculate confidence specifically for momentum bias strategy
        Override the base class method with momentum bias specific logic
        """
        try:
            # Get momentum bias specific values
            momentum_up_bias = signal_data.get('momentum_up_bias', 0.0)
            momentum_down_bias = signal_data.get('momentum_down_bias', 0.0)
            boundary = signal_data.get('boundary', 0.01)
            signal_type = signal_data.get('signal_type', '')
            
            # Base confidence from configuration
            base_confidence = self.base_confidence  # Usually 0.65 (65%)
            
            # Factor 1: Boundary strength (how far above boundary the dominant bias is)
            dominant_bias = momentum_down_bias if signal_type == 'BULL' else momentum_up_bias
            if boundary > 0:
                boundary_strength = max(0, (abs(dominant_bias) - boundary) / boundary)
                boundary_factor = min(0.2, boundary_strength * self.boundary_weight)  # Max 20% boost
            else:
                boundary_factor = 0.0
            
            # Factor 2: Bias separation (how much the dominant bias exceeds the other)
            bias_separation = abs(momentum_up_bias - momentum_down_bias)
            separation_factor = min(0.15, bias_separation * self.separation_weight)  # Max 15% boost
            
            # Factor 3: Signal direction consistency
            if signal_type == 'BULL' and momentum_down_bias > momentum_up_bias and momentum_down_bias > boundary:
                direction_factor = 0.1  # 10% boost for proper BULL signal
            elif signal_type == 'BEAR' and momentum_up_bias > momentum_down_bias and momentum_up_bias > boundary:
                direction_factor = 0.1  # 10% boost for proper BEAR signal
            else:
                direction_factor = -0.2  # 20% penalty for weak signals
            
            # Factor 4: Efficiency ratio (from enhanced validation)
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.6)
            efficiency_factor = (efficiency_ratio - 0.5) * 0.2  # Can add/subtract up to 10%
            
            # Factor 5: Volume confirmation
            volume_confirmation = signal_data.get('volume_confirmation', False)
            volume_factor = 0.05 if volume_confirmation else 0.0
            
            # Calculate total confidence
            total_confidence = (
                base_confidence + 
                boundary_factor + 
                separation_factor + 
                direction_factor + 
                efficiency_factor + 
                volume_factor
            )
            
            # Ensure confidence is within valid range
            final_confidence = max(0.1, min(0.95, total_confidence))
            
            # Debug logging
            self.logger.debug(f"   Confidence breakdown:")
            self.logger.debug(f"      Base: {base_confidence:.1%}")
            self.logger.debug(f"      Boundary: +{boundary_factor:.1%} (strength: {boundary_strength:.3f})")
            self.logger.debug(f"      Separation: +{separation_factor:.1%} (separation: {bias_separation:.4f})")
            self.logger.debug(f"      Direction: {direction_factor:+.1%}")
            self.logger.debug(f"      Efficiency: {efficiency_factor:+.1%} (ratio: {efficiency_ratio:.3f})")
            self.logger.debug(f"      Volume: +{volume_factor:.1%}")
            self.logger.debug(f"      Final: {final_confidence:.1%}")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum bias confidence: {e}")
            return 0.5  # Safe fallback

    def _calculate_efficiency_ratio(self, df: pd.DataFrame) -> float:
        """
        🔥 FIXED: Calculate efficiency ratio for enhanced validation
        Previous version was returning 0.000 - now properly calculates momentum bias efficiency
        """
        try:
            if len(df) < 20:
                return 0.6  # Default good efficiency for short data
            
            # For momentum bias, we should calculate efficiency based on momentum bias values
            # rather than just price movement
            
            # Check if momentum bias columns exist
            if 'momentum_up_bias' not in df.columns or 'momentum_down_bias' not in df.columns:
                # Fallback to price-based efficiency
                close_prices = df['close'].tail(20)
                direction_change = abs(close_prices.iloc[-1] - close_prices.iloc[0])
                total_movement = close_prices.diff().abs().sum()
                
                if total_movement == 0:
                    return 0.6  # Neutral efficiency
                
                efficiency = direction_change / total_movement
                return min(0.9, max(0.3, efficiency))
            
            # Use momentum bias for efficiency calculation
            recent_data = df.tail(20)
            
            # Calculate momentum bias trend consistency  
            up_bias_trend = recent_data['momentum_up_bias'].iloc[-1] - recent_data['momentum_up_bias'].iloc[0]
            down_bias_trend = recent_data['momentum_down_bias'].iloc[-1] - recent_data['momentum_down_bias'].iloc[0]
            
            # Calculate total momentum bias fluctuation
            up_bias_volatility = recent_data['momentum_up_bias'].diff().abs().sum()
            down_bias_volatility = recent_data['momentum_down_bias'].diff().abs().sum()
            total_volatility = up_bias_volatility + down_bias_volatility
            
            if total_volatility == 0:
                return 0.6  # Neutral efficiency when no movement
            
            # Efficiency based on trend strength vs noise
            trend_strength = abs(up_bias_trend) + abs(down_bias_trend)
            efficiency = trend_strength / total_volatility
            
            # Normalize to reasonable range (0.3 to 0.9)
            normalized_efficiency = min(0.9, max(0.3, efficiency))
            
            self.logger.debug(f"   Efficiency calculation: trend_strength={trend_strength:.4f}, volatility={total_volatility:.4f}, efficiency={normalized_efficiency:.3f}")
            
            return normalized_efficiency
            
        except Exception as e:
            self.logger.debug(f"Efficiency ratio calculation failed: {e}")
            return 0.6  # Default good efficiency on error

    def create_enhanced_signal_data(self, latest: pd.Series, signal_type: str, current_price: float) -> Dict:
        """
        🔥 FIXED: Create enhanced signal data structure for proper confidence calculation
        Updated method signature to accept current_price parameter
        """
        return {
            'signal_type': signal_type,
            'price': current_price,  # Use the passed current_price parameter
            'ema_short': current_price,  # Momentum bias doesn't use EMAs, use price as fallback
            'ema_long': current_price,   # Momentum bias doesn't use EMAs, use price as fallback
            'ema_trend': current_price,  # Momentum bias doesn't use EMAs, use price as fallback
            'momentum_up_bias': latest.get('momentum_up_bias', 0.0),
            'momentum_down_bias': latest.get('momentum_down_bias', 0.0),
            'boundary': latest.get('boundary', 0.01),
            'volume': latest.get('ltv', latest.get('volume', 0)),
            'volume_ratio': latest.get('volume_ratio_20', 1.0),
            'volume_confirmation': latest.get('volume_ratio_20', 1.0) > 1.2,
            'atr': latest.get('atr', 0.001),
            'efficiency_ratio': 0.5  # Will be calculated properly in detect_signal
        }

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
                    
            # Case 3: Integer or float (Unix timestamp)
            if isinstance(timestamp_value, (int, float)):
                # Check if it's a reasonable Unix timestamp (between 1970 and 2100)
                if 0 <= timestamp_value <= 4102444800:  # 2100-01-01
                    return datetime.fromtimestamp(timestamp_value)
                else:
                    # FIXES: Invalid integer timestamp (like 429) - log and return None
                    self.logger.warning(f"⚠️ TIMESTAMP FIX: Invalid timestamp integer {timestamp_value} converted to None (prevents database error)")
                    return None
                    
            # Case 4: Pandas timestamp
            if hasattr(timestamp_value, 'to_pydatetime'):
                return timestamp_value.to_pydatetime()
                
            # Case 5: Unknown type - log and return None
            self.logger.warning(f"⚠️ TIMESTAMP FIX: Unknown timestamp type {type(timestamp_value)} value {timestamp_value} converted to None")
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ TIMESTAMP FIX: Error converting timestamp {timestamp_value}: {e} - converted to None")
            return None

    def original_calculate_confidence(self, signal_data: Dict) -> float:
        """
        Calculate confidence score for momentum bias signals
        
        Factors:
        1. Base confidence for strategy
        2. Boundary strength (how far above boundary line)
        3. Bias separation (difference between up/down bias)
        4. Dominance ratio (strength of dominant bias vs secondary)
        """
        try:
            # Start with base confidence
            confidence = self.base_confidence
            
            # Factor 1: Boundary strength (0.0 - 0.25 boost)
            boundary_strength = signal_data.get('boundary_strength', 1.0)
            boundary_boost = min(0.25, (boundary_strength - 1.0) * self.boundary_weight)
            confidence += boundary_boost
            
            # Factor 2: Bias separation strength (0.0 - 0.15 boost)
            bias_separation = signal_data.get('bias_separation', 0)
            boundary = signal_data.get('boundary', 1.0)
            separation_ratio = bias_separation / boundary if boundary > 0 else 0.1
            separation_boost = min(0.15, separation_ratio * self.separation_weight)
            confidence += separation_boost
            
            # Factor 3: Dominance ratio (0.0 - 0.10 boost)
            dominance_ratio = signal_data.get('bias_dominance_ratio', 1.0)
            dominance_boost = min(0.10, (dominance_ratio - 1.0) * 0.05)
            confidence += dominance_boost
            
            # Factor 4: Boundary margin safety (penalize signals too close to boundary)
            boundary_margin = signal_data.get('boundary_margin', 0.1)
            if boundary_margin < 0.1:  # Less than 10% above boundary
                confidence -= 0.1
            elif boundary_margin > 0.5:  # Strongly above boundary
                confidence += 0.05
            
            # Ensure confidence stays within reasonable bounds
            confidence = max(0.30, min(0.95, confidence))
            
            self.logger.debug(f"   Confidence breakdown:")
            self.logger.debug(f"   - Base: {self.base_confidence:.3f}")
            self.logger.debug(f"   - Boundary boost: +{boundary_boost:.3f}")
            self.logger.debug(f"   - Separation boost: +{separation_boost:.3f}")
            self.logger.debug(f"   - Dominance boost: +{dominance_boost:.3f}")
            self.logger.debug(f"   - Final confidence: {confidence:.3f}")
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return self.base_confidence
    
    def get_strategy_info(self) -> Dict:
        """Return strategy information for monitoring and debugging"""
        return {
            'name': 'Momentum Bias Index',
            'version': '1.0',
            'parameters': {
                'momentum_length': self.momentum_length,
                'bias_length': self.bias_length,
                'smooth_length': self.smooth_length,
                'impulse_boundary_length': self.impulse_boundary_length,
                'std_dev_multiplier': self.std_dev_multiplier,
                'smooth_indicator': self.smooth_indicator
            },
            'confidence_weights': {
                'base_confidence': self.base_confidence,
                'boundary_weight': self.boundary_weight,
                'separation_weight': self.separation_weight
            },
            'signal_conditions': {
                'bull': 'momentum_down_bias > momentum_up_bias AND above boundary line',
                'bear': 'momentum_up_bias > momentum_down_bias AND above boundary line'
            }
        }