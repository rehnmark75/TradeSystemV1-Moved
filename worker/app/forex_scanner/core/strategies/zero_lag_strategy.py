# core/strategies/zero_lag_strategy.py
"""
Zero Lag + Squeeze Momentum Strategy - Refactored & Simplified
Following the successful EMA strategy refactoring pattern

Strategy Components:
1. Zero Lag Trend - Entry signal detection (GPS)
2. Squeeze Momentum - Momentum & volatility confirmation (engine RPM)
3. EMA200 Filter - Macro trend direction (handled by TradeValidator)

Decision Matrix:
- BULL: Zero Lag bullish + Squeeze momentum positive/rising (+ EMA200 filter via TradeValidator)
- BEAR: Zero Lag bearish + Squeeze momentum negative/falling (+ EMA200 filter via TradeValidator)
- NO TRADE: Conflicting signals or low momentum
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster

# Import new simplified helper modules
from .helpers.zero_lag_indicator_calculator import ZeroLagIndicatorCalculator
from .helpers.zero_lag_signal_calculator import ZeroLagSignalCalculator
from .helpers.zero_lag_trend_validator import ZeroLagTrendValidator
from .helpers.zero_lag_squeeze_analyzer import ZeroLagSqueezeAnalyzer

try:
    import config
except ImportError:
    from forex_scanner import config
try:
    from configdata import config as configdata
except ImportError:
    try:
        from forex_scanner.configdata import config as configdata
    except ImportError:
        configdata = None


class ZeroLagStrategy(BaseStrategy):
    """
    Zero Lag + Squeeze Momentum Strategy

    Simplified architecture following EMA strategy pattern:
    - Main class handles orchestration and public interface
    - Helper modules handle specialized functionality
    - Clean separation of concerns
    - EMA200 validation delegated to TradeValidator for consistency
    """
    
    def __init__(self, data_fetcher=None, epic=None, use_optimal_parameters=False, pipeline_mode=True):
        super().__init__('zero_lag_squeeze')

        # Initialize core components
        self.price_adjuster = PriceAdjuster()
        self.data_fetcher = data_fetcher
        self.epic = epic
        self.use_optimal_parameters = use_optimal_parameters
        self.optimal_params = None  # Initialize to prevent AttributeError in base_strategy
        
        # Enable/disable expensive features based on pipeline mode
        self.enhanced_validation = pipeline_mode and getattr(configdata, 'ZERO_LAG_ENHANCED_VALIDATION', True) if configdata else pipeline_mode

        # Initialize simplified helper modules
        self.indicator_calculator = ZeroLagIndicatorCalculator(logger=self.logger)
        self.trend_validator = ZeroLagTrendValidator(logger=self.logger, enhanced_validation=self.enhanced_validation)
        self.squeeze_analyzer = ZeroLagSqueezeAnalyzer(logger=self.logger, enhanced_validation=self.enhanced_validation)
        self.signal_calculator = ZeroLagSignalCalculator(
            logger=self.logger,
            trend_validator=self.trend_validator,
            squeeze_analyzer=self.squeeze_analyzer,
            enhanced_validation=self.enhanced_validation
        )
        
        # Strategy configuration - Dynamic or Static
        if use_optimal_parameters and epic:
            self._load_optimal_parameters(epic)
        else:
            self._load_static_configuration()
        
        # Strategy feature settings
        self.momentum_bias_enabled = False  # Zero Lag strategy doesn't use momentum bias
        self.mtf_validation_enabled = getattr(self, 'mtf_validation_enabled', False)
        self.smart_money_enabled = getattr(self, 'smart_money_enabled', False)
    
    def _load_optimal_parameters(self, epic: str):
        """Load optimal parameters from database optimization results"""
        try:
            # Import the new parameter service
            from optimization.zerolag_parameter_service import get_zerolag_parameter_service
            
            # Get parameter service and optimal parameters for this epic
            param_service = get_zerolag_parameter_service()
            if not param_service:
                raise Exception("Failed to initialize parameter service")
                
            optimal_config = param_service.get_optimal_parameters(epic)
            if not optimal_config:
                raise Exception(f"No optimal parameters found for {epic}")
            
            # Apply optimal parameters
            self.length = optimal_config.zl_length
            self.band_multiplier = optimal_config.band_multiplier
            self.min_confidence = optimal_config.confidence_threshold
            self.bb_length = optimal_config.bb_length
            self.bb_mult = optimal_config.bb_mult
            self.kc_length = optimal_config.kc_length
            self.kc_mult = optimal_config.kc_mult
            self.smart_money_enabled = optimal_config.smart_money_enabled
            self.mtf_validation_enabled = optimal_config.mtf_validation_enabled
            
            # Store optimization metadata
            self.optimal_stop_loss_pips = optimal_config.stop_loss_pips
            self.optimal_take_profit_pips = optimal_config.take_profit_pips
            self.performance_score = optimal_config.composite_score
            self.win_rate = optimal_config.win_rate
            self.net_pips = optimal_config.net_pips
            self.last_optimized = optimal_config.last_updated
            
            if self.enhanced_validation:
                self.logger.info(f"🔍 Enhanced validation ENABLED - Full Zero Lag analysis")
            else:
                self.logger.info(f"🔧 Enhanced validation DISABLED - Basic Zero Lag testing mode")

            self.logger.info("✅ Zero Lag + Squeeze Momentum Strategy initialized with OPTIMAL PARAMETERS")
            self.logger.info(f"   📊 Epic: {epic}")
            self.logger.info(f"   ⚡ ZL Length: {self.length} (optimized)")
            self.logger.info(f"   📈 Band Multiplier: {self.band_multiplier:.2f} (optimized)")
            self.logger.info(f"   🎯 Min Confidence: {self.min_confidence:.1%} (optimized)")
            self.logger.info(f"   🔍 Squeeze BB: {self.bb_length}/{self.bb_mult:.1f} (optimized)")
            self.logger.info(f"   🔍 Squeeze KC: {self.kc_length}/{self.kc_mult:.1f} (optimized)")
            self.logger.info(f"   🧠 Smart Money: {'Enabled' if self.smart_money_enabled else 'Disabled'}")
            self.logger.info(f"   📊 MTF Validation: {'Enabled' if self.mtf_validation_enabled else 'Disabled'}")
            self.logger.info(f"   📈 Performance Score: {self.performance_score:.2f} | Win Rate: {self.win_rate:.1%}")
            self.logger.info(f"   🎯 Optimal SL/TP: {self.optimal_stop_loss_pips:.0f}/{self.optimal_take_profit_pips:.0f} pips")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load optimal parameters for {epic}: {e}")
            self.logger.info("   📋 Falling back to static configuration")
            self._load_static_configuration()
    
    def _load_static_configuration(self):
        """Load static configuration from config files (fallback)"""
        try:
            # Get configuration from configdata if available, otherwise use defaults
            if configdata and hasattr(configdata, 'strategies'):
                zero_lag_config = getattr(configdata.strategies, 'ZERO_LAG_STRATEGY_CONFIG', {}).get('default', {})
                self.length = zero_lag_config.get('zl_length', 50)
                self.band_multiplier = zero_lag_config.get('band_multiplier', 1.5) 
                self.min_confidence = zero_lag_config.get('min_confidence', 0.55)  # Lowered for backtest validation
                self.bb_length = zero_lag_config.get('bb_length', 20)
                self.bb_mult = zero_lag_config.get('bb_mult', 2.0)
                self.kc_length = zero_lag_config.get('kc_length', 20)
                self.kc_mult = zero_lag_config.get('kc_mult', 1.5)
            else:
                # Fallback defaults
                self.length = 50
                self.band_multiplier = 1.5
                self.min_confidence = 0.55  # Lowered for backtest validation debugging
                self.bb_length = 20
                self.bb_mult = 2.0
                self.kc_length = 20
                self.kc_mult = 1.5
            
            # Feature settings from config
            self.smart_money_enabled = getattr(configdata.strategies if configdata and hasattr(configdata, 'strategies') else config, 'ZERO_LAG_SMART_MONEY_ENABLED', False)
            self.mtf_validation_enabled = getattr(configdata.strategies if configdata and hasattr(configdata, 'strategies') else config, 'ZERO_LAG_MTF_VALIDATION_ENABLED', False)
            
            # Set defaults for optimization metadata
            self.optimal_stop_loss_pips = None
            self.optimal_take_profit_pips = None
            self.performance_score = 0.0
            self.last_optimized = None
            
            self.logger.info("✅ Zero Lag + Squeeze Momentum Strategy initialized with STATIC CONFIGURATION")
            self.logger.info(f"   ⚡ ZL Length: {self.length}")
            self.logger.info(f"   📈 Band Multiplier: {self.band_multiplier:.2f}")
            self.logger.info(f"   🎯 Min Confidence: {self.min_confidence:.1%}")
            self.logger.info(f"   🔍 Squeeze BB: {self.bb_length}/{self.bb_mult:.1f}")
            self.logger.info(f"   🔍 Squeeze KC: {self.kc_length}/{self.kc_mult:.1f}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            # Ultra-safe fallback
            self.length = 50
            self.band_multiplier = 1.5
            self.min_confidence = 0.65
            self.bb_length = 20
            self.bb_mult = 2.0
            self.kc_length = 20
            self.kc_mult = 1.5
            self.smart_money_enabled = False
            self.mtf_validation_enabled = False
    
    def get_strategy_metadata(self) -> Dict:
        """Get strategy configuration and optimization metadata"""
        return {
            'strategy_name': 'Zero Lag + Squeeze Momentum',
            'epic': self.epic,
            'use_optimal_parameters': self.use_optimal_parameters,
            'configuration': {
                'zl_length': self.length,
                'band_multiplier': self.band_multiplier,
                'min_confidence': self.min_confidence,
                'bb_length': self.bb_length,
                'bb_mult': self.bb_mult,
                'kc_length': self.kc_length,
                'kc_mult': self.kc_mult,
                'smart_money_enabled': self.smart_money_enabled,
                'mtf_validation_enabled': self.mtf_validation_enabled
            },
            'optimization_data': {
                'performance_score': getattr(self, 'performance_score', 0.0),
                'optimal_stop_loss_pips': getattr(self, 'optimal_stop_loss_pips', None),
                'optimal_take_profit_pips': getattr(self, 'optimal_take_profit_pips', None),
                'last_optimized': getattr(self, 'last_optimized', None)
            }
        }

    def get_required_indicators(self) -> List[str]:
        """Return list of required indicators for this strategy"""
        return self.indicator_calculator.get_required_indicators(self.length)
    
    def detect_signal(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> Optional[Dict]:
        """
        Main signal detection orchestrating the 3-component strategy
        
        Args:
            df: DataFrame with market data
            epic: Trading instrument identifier
            spread_pips: Spread in pips
            timeframe: Timeframe (e.g., '15m')
            
        Returns:
            Signal dictionary if valid signal found, None otherwise
        """
        #self.logger.info(f"🔍 [Zero Lag + Squeeze] Starting detection for {epic}")
        
        try:
            # Validate input data
            # Pine Script volatility needs length*3 periods for ta.highest(ta.atr(length), length*3)
            volatility_requirement = self.length * 3  # Pine Script exact: ta.highest(ta.atr(length), length*3)
            min_bars = max(volatility_requirement, self.length, self.bb_length, self.kc_length, 200) + 20  # Reduced buffer 
            if not self.indicator_calculator.validate_data_requirements(df, min_bars):
                return None

            # Ensure all required indicators are present
            df_enhanced = self._ensure_all_indicators(df.copy())
            if df_enhanced is None:
                self.logger.debug(f"❌ Failed to enhance data for {epic}")
                return None

            # Detect signals using the 3-component system
            signal = self._detect_three_component_signal(df_enhanced, epic, spread_pips, timeframe)
            
            if signal:
                self.logger.debug(f"✅ Signal detected for {epic}: {signal.get('signal_type')}")
                return signal
            else:
                self.logger.debug(f"❌ No signal for {epic}")
                return None

        except Exception as e:
            self.logger.error(f"Error in Zero Lag signal detection for {epic}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _ensure_all_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Ensure all required indicators are calculated"""
        try:
            # Calculate Zero Lag indicators
            df = self.indicator_calculator.ensure_zero_lag_indicators(
                df, self.length, self.band_multiplier
            )
            
            # Calculate Squeeze Momentum indicators
            df = self.squeeze_analyzer.calculate_squeeze_momentum(
                df, self.bb_length, self.bb_mult, self.kc_length, self.kc_mult
            )
            
            # Detect Zero Lag alerts
            df = self.indicator_calculator.detect_zero_lag_alerts(df)
            
            # Add previous EMA200 for slope calculation
            if 'ema_200' in df.columns:
                df['ema_200_prev'] = df['ema_200'].shift(1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error ensuring indicators: {e}")
            return None

    def _detect_three_component_signal(self, df: pd.DataFrame, epic: str, 
                                       spread_pips: float, timeframe: str) -> Optional[Dict]:
        """
        Detect signals using the 2-component system + TradeValidator:
        1. Zero Lag Trend (Entry Signals)
        2. Squeeze Momentum (Confirmation)
        3. EMA200 Filter (handled by TradeValidator)
        """
        try:
            if len(df) < 2:
                return None
            
            latest_row = df.iloc[-1]
            
            # Check for Zero Lag TREND CROSSOVER signals (ribbon color changes ONLY)
            bull_alert = latest_row.get('bull_alert', False)
            bear_alert = latest_row.get('bear_alert', False)
            
            if not bull_alert and not bear_alert:
                return None
            
            signal_type = 'BULL' if bull_alert else 'BEAR'
            trend = latest_row.get('trend', 0)
            
            # === STRICT VALIDATION SYSTEM ===
            # ZERO LAG IS PRIMARY - trend crossover already detected
            self.logger.debug(f"🎯 Zero Lag trend crossover detected: {signal_type} (ribbon color change)")
            
            # VALIDATION 1: RIBBON COLOR MUST MATCH SIGNAL
            # BULL: trend must be 1 (GREEN ribbon)
            # BEAR: trend must be -1 (RED ribbon)
            if signal_type == 'BULL' and trend != 1:
                self.logger.debug(f"❌ BULL signal but trend={trend} (ribbon not GREEN)")
                return None
            elif signal_type == 'BEAR' and trend != -1:
                self.logger.debug(f"❌ BEAR signal but trend={trend} (ribbon not RED)")
                return None
                
            self.logger.debug(f"✅ Ribbon color matches: {signal_type} with trend={trend}")
            
            # EMA200 validation moved to TradeValidator for consistency across all strategies
            close = latest_row.get('close', 0)
            
            # VALIDATION 3: SQUEEZE MOMENTUM ALIGNMENT + NO SQUEEZE
            squeeze_momentum = latest_row.get('squeeze_momentum', 0)
            squeeze_on = latest_row.get('squeeze_on', False)
            is_lime = latest_row.get('squeeze_is_lime', False)
            is_green = latest_row.get('squeeze_is_green', False)
            is_red = latest_row.get('squeeze_is_red', False)
            is_maroon = latest_row.get('squeeze_is_maroon', False)
            
            # NO SQUEEZE requirement
            if squeeze_on:
                self.logger.debug(f"❌ Signal rejected: squeeze is ON (low volatility)")
                return None
                
            # COLOR ALIGNMENT requirement
            if signal_type == 'BULL':
                if not (is_lime or is_green):
                    self.logger.debug(f"❌ BULL signal but squeeze momentum not LIME/GREEN")
                    return None
                momentum_color = "LIME" if is_lime else "GREEN"
            else:  # BEAR
                if not (is_red or is_maroon):
                    self.logger.debug(f"❌ BEAR signal but squeeze momentum not RED/MAROON")
                    return None
                momentum_color = "RED" if is_red else "MAROON"
                
            self.logger.debug(f"✅ Squeeze validation passed: {momentum_color} color + NO squeeze")

            # VALIDATION 4: RSI CONDITIONS
            # BULL: RSI must be under 70 (not overbought)
            # BEAR: RSI must be over 30 (not oversold)
            if not self.signal_calculator.validate_rsi_conditions(latest_row, signal_type):
                return None

            self.logger.debug(f"✅ RSI validation passed")

            # ALL VALIDATIONS PASSED - CREATE HIGH-QUALITY SIGNAL
            signal = self._create_base_signal(signal_type, epic, timeframe, latest_row, df, spread_pips)

            # VALIDATION 5: MULTI-TIMEFRAME ALIGNMENT (1H + 4H)
            if self.data_fetcher:
                import pandas as pd
                current_timestamp = pd.Timestamp(signal['timestamp']) if isinstance(signal.get('timestamp'), str) else latest_row.name
                if not isinstance(current_timestamp, pd.Timestamp):
                    # Fallback to current time if timestamp parsing fails
                    from datetime import datetime
                    current_timestamp = pd.Timestamp(datetime.now())
                
                mtf_results = self.trend_validator.validate_multi_timeframe_alignment(
                    epic, current_timestamp, signal_type, self.data_fetcher
                )
                
                if not mtf_results['overall_valid']:
                    h1_status = "✓" if mtf_results['h1_validation'] else "✗"
                    h4_status = "✓" if mtf_results['h4_validation'] else "✗"
                    self.logger.debug(f"❌ Multi-timeframe validation failed: 1H={h1_status} 4H={h4_status}")
                    return None
                
                # Add MTF validation results to signal
                signal['mtf_validation'] = mtf_results
                self.logger.debug(f"✅ Multi-timeframe validation passed: 1H=✓ 4H=✓")
            else:
                self.logger.debug("⚠️ No data fetcher available for multi-timeframe validation")
                signal['mtf_validation'] = {'overall_valid': True, 'note': 'no_data_fetcher'}
            
            # DEFENSIVE: Store critical fields before confidence calculation
            signal_epic = signal.get('epic')
            signal_signal_type = signal.get('signal_type')

            # Calculate confidence score (should be high for strict validation)
            confidence = self.signal_calculator.calculate_signal_confidence(
                latest_row, signal_type, signal
            )

            # DEFENSIVE: Restore critical fields after confidence calculation
            signal['epic'] = signal_epic or epic
            signal['signal_type'] = signal_signal_type or signal_type
            signal['confidence'] = confidence
            signal['confidence_score'] = confidence
            
            # Add validation results (EMA200 validation moved to TradeValidator)
            signal['validation_results'] = {
                'zero_lag_trend_crossover': True,  # Already validated
                'ribbon_color_aligned': True,      # Already validated
                'squeeze_momentum_aligned': True,  # Already validated
                'squeeze_off': True,               # Already validated
                'rsi_validation': True,            # Already validated
                'multi_timeframe_validation': signal.get('mtf_validation', {}).get('overall_valid', False),
                'validation_level': 'strict_4_component_with_mtf_plus_tradevalidator'  # Updated
            }
            
            # Add strategy metadata
            signal.update({
                'strategy_type': 'zero_lag_ribbon_crossover',
                'signal_trigger': 'trend_crossover',  # Primary trigger
                'ribbon_color': 'GREEN' if signal_type == 'BULL' else 'RED',
                'momentum_color': momentum_color,
                'squeeze_state': 'OFF',
                'strategy_version': 'v3_pine_script_exact',
            })
            
            # DEBUG: Output exact values for TradingView comparison
            close = latest_row.get('close', 0)
            zlema = latest_row.get('zlema', 0)
            upper_band = latest_row.get('upper_band', 0)
            lower_band = latest_row.get('lower_band', 0)
            volatility = latest_row.get('volatility', 0)
            squeeze_momentum_val = latest_row.get('squeeze_momentum', 0)
            
            # MTF status for logging
            mtf_info = signal.get('mtf_validation', {})
            mtf_status = ""
            if mtf_info.get('overall_valid'):
                h1_status = "1H✓" if mtf_info.get('h1_validation') else "1H✗"
                h4_status = "4H✓" if mtf_info.get('h4_validation') else "4H✗"
                mtf_status = f" + MTF({h1_status},{h4_status})"
            
            # Add RSI info to signal for logging
            rsi_value = latest_row.get('rsi', 50.0)

            # CRITICAL: Ensure required fields are present before return
            if 'epic' not in signal or 'signal_type' not in signal:
                self.logger.error(f"❌ CRITICAL: Signal missing required fields! epic: {signal.get('epic', 'MISSING')}, signal_type: {signal.get('signal_type', 'MISSING')}")
                signal['epic'] = epic  # Force add epic
                signal['signal_type'] = signal_type  # Force add signal_type
                self.logger.info(f"✅ FIXED: Added missing fields - epic: {epic}, signal_type: {signal_type}")

            # ✅ NEW: Calculate optimized SL/TP
            sl_tp = self.calculate_optimal_sl_tp(signal, epic, latest_row, spread_pips)
            signal['stop_distance'] = sl_tp['stop_distance']
            signal['limit_distance'] = sl_tp['limit_distance']

            self.logger.info(f"🎯 HIGH-QUALITY {signal_type} signal: ribbon={trend} + squeeze={momentum_color} + NO_SQUEEZE + RSI={rsi_value:.1f}{mtf_status}, SL/TP={sl_tp['stop_distance']}/{sl_tp['limit_distance']}")
            self.logger.info(f"   📊 Signal validation: epic={signal.get('epic', 'MISSING')}, signal_type={signal.get('signal_type', 'MISSING')}")
            self.logger.info(f"   📊 DEBUG VALUES for TradingView comparison:")
            self.logger.info(f"   📊 Close: {close:.5f}, ZLEMA: {zlema:.5f}")
            self.logger.info(f"   📊 Upper Band: {upper_band:.5f}, Lower Band: {lower_band:.5f}")
            self.logger.info(f"   📊 Volatility: {volatility:.5f}, Trend: {trend}")
            self.logger.info(f"   📊 Squeeze Momentum: {squeeze_momentum_val:.6f}")
            self.logger.info(f"   📊 RSI: {rsi_value:.1f}")
            # EMA200 data included in signal for TradeValidator validation
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in three-component signal detection: {e}")
            return None

    def _create_base_signal(self, signal_type: str, epic: str, timeframe: str, 
                           latest_row: pd.Series, df: pd.DataFrame, spread_pips: float) -> Dict:
        """Create base signal with all required fields"""
        try:
            # Get timestamp
            timestamp = self._get_safe_timestamp(latest_row)
            
            # Core signal data
            signal = {
                'signal_type': signal_type,
                'epic': epic,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'market_timestamp': timestamp,
                'price': float(latest_row['close']),
                'strategy': 'zero_lag_squeeze',
                
                # Zero Lag data
                'zlema': float(latest_row.get('zlema', 0)),
                'upper_band': float(latest_row.get('upper_band', 0)),
                'lower_band': float(latest_row.get('lower_band', 0)),
                'volatility': float(latest_row.get('volatility', 0)),
                'trend': int(latest_row.get('trend', 0)),
                
                # EMA200 data
                'ema_200': float(latest_row.get('ema_200', 0)),
                
                # Squeeze Momentum data
                'squeeze_momentum': float(latest_row.get('squeeze_momentum', 0)),
                'squeeze_state': latest_row.get('squeeze_state', 'unknown'),
                'squeeze_is_lime': latest_row.get('squeeze_is_lime', False),
                'squeeze_is_green': latest_row.get('squeeze_is_green', False),
                'squeeze_is_red': latest_row.get('squeeze_is_red', False),
                'squeeze_is_maroon': latest_row.get('squeeze_is_maroon', False),

                # RSI data
                'rsi': float(latest_row.get('rsi', 50.0)),
                
                # Analysis summaries
                'trend_summary': self.trend_validator.get_trend_summary(latest_row),
                'squeeze_summary': self.squeeze_analyzer.get_squeeze_summary(latest_row),
                
                # Signal strength metrics
                'crossover_strength': self.signal_calculator.calculate_crossover_strength(latest_row, signal_type),
                'trend_strength': self.signal_calculator.calculate_trend_strength(latest_row),
            }
            
            # Add spread and fees with safe type conversion
            try:
                spread_pips_float = float(spread_pips) if spread_pips is not None else 0.0
                signal['spread_pips'] = spread_pips_float
                signal['estimated_fee'] = spread_pips_float * 0.5  # Rough estimate
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid spread_pips value: {spread_pips}, using 0.0")
                signal['spread_pips'] = 0.0
                signal['estimated_fee'] = 0.0
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating base signal: {e}")
            return {}

    def _get_safe_timestamp(self, latest_row: pd.Series) -> str:
        """Get safe timestamp string from row data"""
        try:
            # Try different timestamp columns
            for col in ['start_time', 'datetime_utc', 'timestamp', 'datetime']:
                if col in latest_row and latest_row[col] is not None:
                    timestamp = latest_row[col]
                    if isinstance(timestamp, str):
                        return timestamp
                    elif hasattr(timestamp, 'strftime'):
                        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            # Fallback to current time
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            self.logger.debug(f"Timestamp extraction failed: {e}")
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def calculate_confidence(self, signal_data: Dict) -> float:
        """
        Public interface for confidence calculation
        Delegates to signal_calculator for consistency
        """
        try:
            # Extract latest row data from signal
            latest_row_data = {
                'close': signal_data.get('price', 0),
                'zlema': signal_data.get('zlema', 0),
                'ema_200': signal_data.get('ema_200', 0),
                'trend': signal_data.get('trend', 0),
                'volatility': signal_data.get('volatility', 0),
                'squeeze_momentum': signal_data.get('squeeze_momentum', 0),
                'squeeze_state': signal_data.get('squeeze_state', 'unknown'),
                'squeeze_is_lime': signal_data.get('squeeze_is_lime', False),
                'squeeze_is_green': signal_data.get('squeeze_is_green', False),
                'squeeze_is_red': signal_data.get('squeeze_is_red', False),
                'squeeze_is_maroon': signal_data.get('squeeze_is_maroon', False),
                'rsi': signal_data.get('rsi', 50.0),
            }
            
            # Convert to Series for compatibility
            latest_row = pd.Series(latest_row_data)
            signal_type = signal_data.get('signal_type', '')
            
            return self.signal_calculator.calculate_signal_confidence(
                latest_row, signal_type, signal_data
            )
            
        except Exception as e:
            self.logger.error(f"Error in public confidence calculation: {e}")
            return 0.65  # Safe fallback

    def get_strategy_summary(self) -> Dict:
        """Get comprehensive strategy summary"""
        return {
            'strategy_name': 'Zero Lag + Squeeze Momentum + EMA200',
            'strategy_type': 'zero_lag_squeeze_ema200',
            'architecture': 'simplified_modular',
            'components': {
                'ema200_filter': 'Macro trend direction (guardrail)',
                'zero_lag_trend': 'Entry signal detection (GPS)',
                'squeeze_momentum': 'Momentum & volatility confirmation (engine RPM)'
            },
            'configuration': {
                'zero_lag_length': self.length,
                'band_multiplier': self.band_multiplier,
                'bb_length': self.bb_length,
                'kc_length': self.kc_length,
                'min_confidence': self.min_confidence
            },
            'decision_matrix': {
                'bull_conditions': [
                    'close > EMA200',
                    'Zero Lag bullish crossover',
                    'Squeeze momentum positive/rising'
                ],
                'bear_conditions': [
                    'close < EMA200', 
                    'Zero Lag bearish crossover',
                    'Squeeze momentum negative/falling'
                ],
                'no_trade': [
                    'Conflicting signals',
                    'Price flat near EMA200',
                    'Low momentum/volatility'
                ]
            },
            'helper_modules': [
                'ZeroLagIndicatorCalculator',
                'ZeroLagSignalCalculator', 
                'ZeroLagTrendValidator',
                'ZeroLagSqueezeAnalyzer'
            ]
        }


# Factory functions for easy integration
def create_zero_lag_strategy(data_fetcher=None, epic=None, use_optimal_parameters=False):
    """
    Factory function to create Zero Lag strategy
    
    Args:
        data_fetcher: Data fetcher instance for market data
        epic: Trading pair epic (required for optimal parameters)
        use_optimal_parameters: If True, loads optimized parameters from database
        
    Returns:
        Configured ZeroLagStrategy instance
    """
    return ZeroLagStrategy(
        data_fetcher=data_fetcher,
        epic=epic,
        use_optimal_parameters=use_optimal_parameters
    )

def create_optimized_zero_lag_strategy(epic: str, data_fetcher=None):
    """
    Convenience factory to create Zero Lag strategy with optimal parameters
    
    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
        data_fetcher: Data fetcher instance
        
    Returns:
        ZeroLagStrategy with optimization-driven parameters
    """
    return ZeroLagStrategy(
        data_fetcher=data_fetcher,
        epic=epic,
        use_optimal_parameters=True
    )