# core/strategies/macd_strategy_v2.py
"""
MACD Strategy Implementation - REBUILT FROM SCRATCH (Lightweight Orchestrator)
🎯 BACK TO BASICS: Simple, clean MACD strategy based on core crossover logic
📊 SIMPLE: No complex helpers, minimal features - just solid MACD signal detection  
⚡ LIGHTWEIGHT: ~400 lines vs previous 2400+ lines + 7700 lines of helpers
🏗️ ORCHESTRATOR: Main class coordinates, helpers do the work

Features:
- Simple MACD histogram crossover detection
- EMA 200 trend alignment validation
- Multi-timeframe MACD validation (optional)
- Bull/Bear signal generation with confidence scoring
- Compatible with existing backtest system
- Configurable MACD periods through config
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
from .helpers.macd_indicator_calculator import MACDIndicatorCalculator
from .helpers.macd_signal_calculator import MACDSignalCalculator
from .helpers.macd_trend_validator import MACDTrendValidator
from .helpers.macd_mtf_analyzer import MACDMultiTimeframeAnalyzer

# Import optimization functions
try:
    from optimization.optimal_parameter_service import get_macd_optimal_parameters, is_epic_macd_optimized
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    def get_macd_optimal_parameters(*args, **kwargs):
        raise ImportError("Optimization service not available")
    def is_epic_macd_optimized(*args, **kwargs):
        return False

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config

# Import optimization parameter service
try:
    from optimization.optimal_parameter_service import get_macd_optimal_parameters, is_epic_macd_optimized
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.optimization.optimal_parameter_service import get_macd_optimal_parameters, is_epic_macd_optimized
        OPTIMIZATION_AVAILABLE = True
    except ImportError:
        OPTIMIZATION_AVAILABLE = False


class MACDStrategy(BaseStrategy):
    """
    🎯 LIGHTWEIGHT MACD STRATEGY: Orchestrator pattern implementation
    
    Simple MACD histogram crossover strategy with trend alignment validation.
    Coordinates focused helper modules to keep main class lightweight and maintainable.
    """
    
    def __init__(self, data_fetcher=None, backtest_mode: bool = False, epic: str = None, timeframe: str = '15m', use_optimized_parameters: bool = True):
        # Initialize parent  
        self.name = 'macd'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self._validator = None  # Skip enhanced validator for simplicity
        self.epic = epic
        self.timeframe = timeframe
        self.use_optimized_parameters = use_optimized_parameters
        
        # Basic initialization
        self.backtest_mode = backtest_mode
        self.price_adjuster = PriceAdjuster()
        self.data_fetcher = data_fetcher
        
        # Required attributes for backtest compatibility  
        self.enable_mtf_analysis = self._should_enable_mtf_analysis(timeframe)
        
        # Simple MACD configuration - get from config or use defaults
        self.macd_config = self._get_macd_periods()
        self.fast_ema = self.macd_config.get('fast_ema', 12)
        self.slow_ema = self.macd_config.get('slow_ema', 26)
        self.signal_ema = self.macd_config.get('signal_ema', 9)
        
        # Basic parameters
        self.eps = 1e-8  # Epsilon for stability
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.50)  # MACD needs higher confidence
        self.min_bars = 60  # Minimum bars for stable MACD (26 + 9 + buffer)
        
        # Initialize helper modules (orchestrator pattern)
        self.indicator_calculator = MACDIndicatorCalculator(logger=self.logger, eps=self.eps)
        self.trend_validator = MACDTrendValidator(logger=self.logger)
        self.signal_calculator = MACDSignalCalculator(logger=self.logger, trend_validator=self.trend_validator)
        self.mtf_analyzer = MACDMultiTimeframeAnalyzer(logger=self.logger, data_fetcher=data_fetcher)
        
        # Set MTF analyzer for backtest compatibility
        if self.enable_mtf_analysis and self.mtf_analyzer.is_mtf_enabled():
            self.mtf_analyzer_instance = self.mtf_analyzer  # For compatibility
        else:
            self.mtf_analyzer_instance = None
        
        self.logger.info(f"🎯 MACD Strategy initialized - Periods: {self.fast_ema}/{self.slow_ema}/{self.signal_ema} ({timeframe})")
        self.logger.info(f"🔧 Using lightweight orchestrator pattern with 4 focused helpers")
        if backtest_mode:
            self.logger.info("🔥 BACKTEST MODE: Time restrictions disabled")
    
    def _get_macd_periods(self) -> Dict:
        """Get MACD periods from optimization database or config fallback"""
        try:
            # First priority: Use optimized parameters from database if available
            self.logger.info(f"🔍 DEBUGGING: use_optimized={self.use_optimized_parameters}, "
                            f"available={OPTIMIZATION_AVAILABLE}, epic={self.epic}, timeframe={self.timeframe}")
            
            if self.epic and self.timeframe:
                is_optimized = is_epic_macd_optimized(self.epic, self.timeframe)
                self.logger.info(f"🔍 is_epic_macd_optimized({self.epic}, {self.timeframe}) = {is_optimized}")
            else:
                self.logger.info(f"🔍 Missing epic or timeframe: epic={self.epic}, timeframe={self.timeframe}")
            
            final_condition = (self.use_optimized_parameters and 
                               OPTIMIZATION_AVAILABLE and 
                               self.epic and 
                               is_epic_macd_optimized(self.epic, self.timeframe))
            
            self.logger.info(f"🔍 Final condition: {final_condition}")
            
            if final_condition:
                self.logger.info("🎯 ENTERING OPTIMIZATION BLOCK")
                try:
                    optimal_params = get_macd_optimal_parameters(self.epic, self.timeframe)
                    self.logger.info("🎯 get_macd_optimal_parameters SUCCESS")
                except Exception as opt_e:
                    self.logger.error(f"❌ get_macd_optimal_parameters FAILED: {opt_e}")
                    raise
                
                self.logger.info(f"✅ Using OPTIMIZED MACD parameters for {self.epic} ({self.timeframe}): "
                               f"{optimal_params.fast_ema}/{optimal_params.slow_ema}/{optimal_params.signal_ema} "
                               f"(Score: {optimal_params.performance_score:.6f}, Win Rate: {optimal_params.win_rate:.1%})")
                
                return {
                    'fast_ema': optimal_params.fast_ema,
                    'slow_ema': optimal_params.slow_ema,
                    'signal_ema': optimal_params.signal_ema,
                    'confidence_threshold': optimal_params.confidence_threshold,
                    'histogram_threshold': optimal_params.histogram_threshold,
                    'rsi_filter_enabled': optimal_params.rsi_filter_enabled,
                    'momentum_confirmation': optimal_params.momentum_confirmation,
                    'zero_line_filter': optimal_params.zero_line_filter,
                    'mtf_enabled': optimal_params.mtf_enabled,
                    'stop_loss_pips': optimal_params.stop_loss_pips,
                    'take_profit_pips': optimal_params.take_profit_pips
                }
            
            # Second priority: Use OptimalParameterService timeframe-aware fallback
            else:
                try:
                    
                    self.logger.info(f"📋 Using TIMEFRAME-AWARE fallback parameters for {self.epic or 'default'} ({self.timeframe})")
                    optimal_params = get_macd_optimal_parameters(
                        epic=self.epic or "CS.D.DEFAULT.MINI.IP",
                        timeframe=self.timeframe
                    )
                    
                    return {
                        'fast_ema': optimal_params.fast_ema,
                        'slow_ema': optimal_params.slow_ema,
                        'signal_ema': optimal_params.signal_ema,
                        'confidence_threshold': optimal_params.confidence_threshold,
                        'histogram_threshold': optimal_params.histogram_threshold,
                        'rsi_filter_enabled': optimal_params.rsi_filter_enabled,
                        'momentum_confirmation': optimal_params.momentum_confirmation,
                        'zero_line_filter': optimal_params.zero_line_filter,
                        'mtf_enabled': optimal_params.mtf_enabled,
                        'stop_loss_pips': optimal_params.stop_loss_pips,
                        'take_profit_pips': optimal_params.take_profit_pips
                    }
                    
                except Exception as fallback_error:
                    self.logger.warning(f"⚠️ Timeframe-aware fallback failed: {fallback_error}")
                    
                    # Final fallback: Use config if available, otherwise defaults
                    if hasattr(config, 'MACD_PERIODS'):
                        macd_periods = getattr(config, 'MACD_PERIODS', None)
                        if macd_periods and isinstance(macd_periods, dict):
                            self.logger.info(f"📋 Using CONFIG MACD parameters for {self.epic or 'default'}: "
                                           f"{macd_periods.get('fast_ema', 12)}/{macd_periods.get('slow_ema', 26)}/{macd_periods.get('signal_ema', 9)}")
                            return macd_periods
                    
                    # Ultimate fallback: Standard MACD defaults
                    self.logger.warning(f"⚠️ Using ULTIMATE FALLBACK MACD parameters: 12/26/9")
                    return {'fast_ema': 12, 'slow_ema': 26, 'signal_ema': 9}
            
        except Exception as e:
            self.logger.warning(f"Could not load MACD config: {e}, using defaults")
            return {'fast_ema': 12, 'slow_ema': 26, 'signal_ema': 9}
    
    def _should_enable_mtf_analysis(self, timeframe: str) -> bool:
        """
        Determine if MTF analysis should be enabled based on timeframe
        
        Fast timeframes (5m, 15m) don't benefit from MTF validation and it adds lag
        Higher timeframes (1h, 4h, 1d) benefit from MTF confirmation
        """
        # Check config override first
        config_mtf_enabled = getattr(config, 'MACD_MTF_ENABLED', None)
        if config_mtf_enabled is not None:
            # If explicitly set in config, respect that setting
            if config_mtf_enabled:
                self.logger.info(f"🔧 MTF analysis ENABLED via config override for {timeframe}")
                return True
        
        # Timeframe-based logic
        fast_timeframes = ['1m', '5m', '15m']
        slow_timeframes = ['1h', '4h', '1d', '1w']
        
        if timeframe in fast_timeframes:
            self.logger.info(f"🚫 MTF analysis DISABLED for fast timeframe {timeframe} (reduces lag)")
            return False
        elif timeframe in slow_timeframes:
            self.logger.info(f"✅ MTF analysis ENABLED for slower timeframe {timeframe} (improves accuracy)")
            return True
        else:
            # Unknown timeframe, default to enabled but log warning
            self.logger.warning(f"⚠️ Unknown timeframe {timeframe}, defaulting to MTF ENABLED")
            return True
    
    def get_required_indicators(self) -> List[str]:
        """Required indicators for MACD strategy"""
        return self.indicator_calculator.get_required_indicators(self.macd_config)
    
    def detect_signal_auto(
        self, 
        df: pd.DataFrame, 
        epic: str = None, 
        spread_pips: float = 0,
        timeframe: str = '15m',
        **kwargs
    ) -> Optional[Dict]:
        """
        Auto signal detection wrapper for compatibility with signal detector
        Delegates to detect_signal method
        """
        return self.detect_signal(df, epic, spread_pips, timeframe, **kwargs)
    
    def detect_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float = 1.5,
        timeframe: str = '15m',
        evaluation_time: str = None
    ) -> Optional[Dict]:
        """
        🎯 CORE SIGNAL DETECTION: Simple MACD histogram crossover with trend alignment
        
        Signal Logic:
        1. Calculate MACD indicators if not present
        2. Detect histogram crossovers (above/below zero line)
        3. Validate trend alignment (EMA 200 filter)
        4. Optional multi-timeframe validation
        5. Generate bull/bear signals with confidence
        """
        
        try:
            # Validate data requirements
            if not self.indicator_calculator.validate_data_requirements(df, self.min_bars):
                return None
            
            self.logger.debug(f"Processing {len(df)} bars for {epic}")
            
            # 1. Calculate MACD indicators if not present
            df_enhanced = self.indicator_calculator.ensure_macd_indicators(df.copy(), self.macd_config)
            
            # 2. Detect MACD crossovers (with strength filtering)
            df_with_signals = self.indicator_calculator.detect_macd_crossovers(df_enhanced, epic)
            
            # 3. Check for signals - scan all bars with crossovers (for backtest compatibility)
            if self.backtest_mode:
                # BACKTEST MODE: Check all bars that have crossovers in this window
                for idx, row in df_with_signals.iterrows():
                    bull_alert = row.get('bull_alert', False)
                    bear_alert = row.get('bear_alert', False)
                    
                    if bull_alert or bear_alert:
                        signal = self._check_immediate_signal(row, epic, timeframe, spread_pips, len(df))
                        if signal:
                            # Add timing info to signal for backtest
                            if hasattr(row, 'name'):
                                signal['signal_time'] = row.name
                            elif 'start_time' in row:
                                signal['signal_time'] = row['start_time']
                            return signal
                
                return None
            else:
                # LIVE MODE: Only check latest bar (normal behavior)
                latest_row = df_with_signals.iloc[-1]
                
                # Debug logging for troubleshooting
                if len(df) < 100:  # Only log for small datasets to avoid spam
                    bull_alert = latest_row.get('bull_alert', False)
                    bear_alert = latest_row.get('bear_alert', False)
                    if bull_alert or bear_alert:
                        self.logger.info(f"🎯 MACD Alert detected! Bull: {bull_alert}, Bear: {bear_alert}")
                
                signal = self._check_immediate_signal(latest_row, epic, timeframe, spread_pips, len(df))
                if signal:
                    return signal
                
                return None
            
        except Exception as e:
            self.logger.error(f"Signal detection error: {e}")
            return None
    
    def _check_immediate_signal(self, latest_row: pd.Series, epic: str, timeframe: str, spread_pips: float, bar_count: int) -> Optional[Dict]:
        """Check for immediate MACD crossover signals with all validations"""
        try:
            # Check for bull crossover
            if latest_row.get('bull_alert', False):
                self.logger.info(f"🎯 MACD BULL crossover detected at bar {bar_count}")
                
                # Validate EMA 200 trend filter
                if not self.trend_validator.validate_ema_200_trend(latest_row, 'BULL'):
                    self.logger.warning("❌ MACD BULL signal REJECTED: Price below EMA 200 (against major trend)")
                    return None
                
                # Validate MACD histogram direction
                if not self.trend_validator.validate_macd_histogram_direction(latest_row, 'BULL'):
                    self.logger.warning("❌ MACD BULL signal REJECTED: Negative histogram")
                    return None
                
                # Optional: Multi-timeframe validation
                mtf_passed = True
                if self.enable_mtf_analysis and self.mtf_analyzer.is_mtf_enabled():
                    current_time = latest_row.get('start_time', pd.Timestamp.now())
                    mtf_result = self.mtf_analyzer.validate_higher_timeframe_macd(epic, current_time, 'BULL')
                    mtf_passed = mtf_result.get('validation_passed', True)
                    if not mtf_passed:
                        self.logger.info("❌ MACD BULL signal REJECTED: Multi-timeframe validation failed")
                        return None
                
                signal = self._create_signal(
                    signal_type='BULL',
                    epic=epic,
                    timeframe=timeframe,
                    latest_row=latest_row,
                    spread_pips=spread_pips
                )
                if signal:
                    self.logger.info(f"✅ MACD BULL signal generated: {signal['confidence']:.1%}")
                    return signal
                else:
                    self.logger.info("❌ MACD BULL signal creation failed")
            
            # Check for bear crossover
            if latest_row.get('bear_alert', False):
                self.logger.info(f"🎯 MACD BEAR crossover detected at bar {bar_count}")
                
                # Validate EMA 200 trend filter
                if not self.trend_validator.validate_ema_200_trend(latest_row, 'BEAR'):
                    self.logger.warning("❌ MACD BEAR signal REJECTED: Price above EMA 200 (against major trend)")
                    return None
                
                # Validate MACD histogram direction
                if not self.trend_validator.validate_macd_histogram_direction(latest_row, 'BEAR'):
                    self.logger.warning("❌ MACD BEAR signal REJECTED: Positive histogram")
                    return None
                
                # Optional: Multi-timeframe validation
                mtf_passed = True
                if self.enable_mtf_analysis and self.mtf_analyzer.is_mtf_enabled():
                    current_time = latest_row.get('start_time', pd.Timestamp.now())
                    mtf_result = self.mtf_analyzer.validate_higher_timeframe_macd(epic, current_time, 'BEAR')
                    mtf_passed = mtf_result.get('validation_passed', True)
                    if not mtf_passed:
                        self.logger.info("❌ MACD BEAR signal REJECTED: Multi-timeframe validation failed")
                        return None
                
                signal = self._create_signal(
                    signal_type='BEAR', 
                    epic=epic,
                    timeframe=timeframe,
                    latest_row=latest_row,
                    spread_pips=spread_pips
                )
                if signal:
                    self.logger.info(f"✅ MACD BEAR signal generated: {signal['confidence']:.1%}")
                    return signal
                else:
                    self.logger.info("❌ MACD BEAR signal creation failed")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal detection error: {e}")
            return None
    
    def _create_signal(
        self,
        signal_type: str,
        epic: str, 
        timeframe: str,
        latest_row: pd.Series,
        spread_pips: float
    ) -> Optional[Dict]:
        """Create a signal dictionary with all required fields"""
        try:
            # Create base signal using parent method
            signal = self.create_base_signal(signal_type, epic, timeframe, latest_row)
            
            # Add MACD-specific data
            signal.update({
                'macd_line': latest_row.get('macd_line', 0),
                'macd_signal': latest_row.get('macd_signal', 0),
                'macd_histogram': latest_row.get('macd_histogram', 0),
                'ema_200': latest_row.get('ema_200', 0),
                'bull_crossover': latest_row.get('bull_crossover', False),
                'bear_crossover': latest_row.get('bear_crossover', False)
            })
            
            # Calculate confidence using signal calculator
            confidence = self.signal_calculator.calculate_simple_confidence(latest_row, signal_type)
            
            # Add MTF boost if enabled and available
            if self.enable_mtf_analysis and self.mtf_analyzer.is_mtf_enabled():
                current_time = latest_row.get('start_time', pd.Timestamp.now())
                mtf_result = self.mtf_analyzer.validate_higher_timeframe_macd(epic, current_time, signal_type)
                confidence_boost = mtf_result.get('confidence_boost', 0.0)
                confidence = min(0.95, confidence + confidence_boost)
                
                signal['mtf_analysis'] = {
                    'validation_passed': mtf_result.get('validation_passed', False),
                    'confidence_boost': confidence_boost,
                    'timeframes_aligned': mtf_result.get('timeframes_aligned', [])
                }
            
            # Add confidence and execution prices
            signal['confidence'] = confidence
            signal['confidence_score'] = confidence  # For compatibility
            
            # Add execution prices  
            signal = self.add_execution_prices(signal, spread_pips)
            
            # Validate confidence threshold
            if not self.signal_calculator.validate_confidence_threshold(confidence):
                return None
            
            self.logger.info(f"🎯 MACD {signal_type} signal generated: {confidence:.1%} confidence at {signal['price']:.5f}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating signal: {e}")
            return None
    
    def create_enhanced_signal_data(self, latest_row: pd.Series, signal_type: str) -> Dict:
        """Create signal data for confidence calculation - matches BaseStrategy expected format"""
        try:
            # Get MACD-specific data
            macd_histogram = latest_row.get('macd_histogram', 0)
            macd_line = latest_row.get('macd_line', 0)
            macd_signal = latest_row.get('macd_signal', 0)
            
            # Basic price data
            close = latest_row.get('close', 0)
            open_price = latest_row.get('open', close)
            high = latest_row.get('high', close)
            low = latest_row.get('low', close)
            
            # Calculate efficiency ratio from MACD histogram strength
            histogram_strength = abs(macd_histogram)
            # Normalize histogram strength (typical range 0-0.002)
            efficiency_ratio = min(1.0, histogram_strength / 0.001) if histogram_strength > 0 else 0.3
            
            # Boost efficiency for aligned signals
            if signal_type == 'BULL' and macd_histogram > 0 and macd_line > macd_signal:
                efficiency_ratio = max(efficiency_ratio, 0.6)  # Minimum 60% for aligned bull signals
            elif signal_type == 'BEAR' and macd_histogram < 0 and macd_line < macd_signal:
                efficiency_ratio = max(efficiency_ratio, 0.6)  # Minimum 60% for aligned bear signals
            
            return {
                'macd_data': {
                    'macd_line': macd_line,
                    'macd_signal': macd_signal,
                    'macd_histogram': macd_histogram
                },
                'ema_data': {
                    'ema_short': latest_row.get('ema_12', 0),
                    'ema_long': latest_row.get('ema_26', 0),
                    'ema_trend': latest_row.get('ema_200', 0)
                },
                'kama_data': {
                    'efficiency_ratio': efficiency_ratio,  # Calculated from MACD strength
                    'kama_value': macd_line,  # Use MACD line as proxy
                    'kama_trend': 1.0 if signal_type == 'BULL' else -1.0
                },
                'other_indicators': {
                    'atr': high - low if high > low else 0.0001,  # Simple ATR proxy
                    'bb_middle': close,
                    'rsi': 50,  # Neutral RSI if not available
                    'volume': latest_row.get('ltv', 1000),  # Use ltv for volume
                    'volume_ratio': latest_row.get('volume_ratio_10', 1.0)
                },
                'indicator_count': 6,  # Count of indicators available
                'data_source': 'macd_strategy_v2',
                'signal_type': signal_type,
                'price': close
            }
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced signal data: {e}")
            # Simple fallback
            return {
                'macd_data': {'macd_histogram': 0},
                'ema_data': {'ema_short': 0, 'ema_long': 0, 'ema_trend': 0},
                'kama_data': {'efficiency_ratio': 0.5},
                'other_indicators': {},
                'signal_type': signal_type,
                'price': latest_row.get('close', 0)
            }
    
    def log_modular_status(self):
        """Log modular status for backtest compatibility"""
        self.logger.info("🏗️ MACD Strategy Modular Status:")
        self.logger.info(f"   📊 Indicator Calculator: ✅ Active")
        self.logger.info(f"   🎯 Signal Calculator: ✅ Active") 
        self.logger.info(f"   🛡️ Trend Validator: ✅ Active")
        self.logger.info(f"   ⏱️ MTF Analyzer: {'✅ Active' if self.mtf_analyzer.is_mtf_enabled() else '❌ Disabled'}")
        self.logger.info(f"   🔧 Total helpers: 4 (lightweight)")
    
    def detect_signal_with_mtf(self, df, epic, spread_pips=1.5, timeframe='15m', **kwargs):
        """MTF-enhanced signal detection for backtest compatibility"""
        return self.detect_signal(df, epic, spread_pips, timeframe, **kwargs)
    
    def enable_forex_integration(self, epic):
        """Enable forex integration for specific pair - compatibility method"""
        self.logger.debug(f"Forex integration enabled for {epic} (lightweight strategy - no-op)")
        pass


class LegacyMACDStrategy(MACDStrategy):
    """
    🔄 LEGACY COMPATIBILITY: Wrapper for any code that depends on the old implementation
    
    This ensures backward compatibility while using the new simplified implementation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("📦 Legacy MACDStrategy wrapper initialized - using new lightweight implementation")
        
        # Add compatibility attributes that might be expected by legacy code
        self.enhanced_validator = self.signal_calculator
        self.forex_optimizer = None  # Removed in v2
        self.cache = None  # Removed in v2
        self.data_helper = None  # Removed in v2
        self.signal_detector = self  # Self-contained in v2


def create_macd_strategy(data_fetcher=None, **kwargs) -> MACDStrategy:
    """
    🏭 FACTORY FUNCTION: Create MACD strategy instance
    
    Simple factory function for backward compatibility with existing code.
    """
    return MACDStrategy(data_fetcher=data_fetcher, **kwargs)