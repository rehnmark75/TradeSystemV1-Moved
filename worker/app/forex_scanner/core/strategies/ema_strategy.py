# core/strategies/ema_strategy.py
"""
EMA Strategy Implementation - REBUILT FROM SCRATCH
🎯 BACK TO BASICS: Simple, clean EMA strategy based on core crossover logic
📊 SIMPLE: No helpers, no complex features - just solid EMA signal detection
⚡ LIGHTWEIGHT: ~200 lines vs previous 1000+ lines

Features:
- Simple EMA crossover detection (price vs EMA12)
- EMA trend alignment validation (12 > 50 > 200)
- Bull/Bear signal generation with confidence scoring
- Compatible with existing backtest system
- Configurable EMA periods through config
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
from .helpers.ema_trend_validator import EMATrendValidator
from .helpers.ema_signal_calculator import EMASignalCalculator
from .helpers.ema_mtf_analyzer import EMAMultiTimeframeAnalyzer
from .helpers.ema_indicator_calculator import EMAIndicatorCalculator

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class EMAStrategy(BaseStrategy):
    """
    🎯 MINIMAL EMA STRATEGY: Back to basics implementation
    
    Simple EMA crossover strategy with trend alignment validation.
    Based on the core logic from detect_ema_alerts function.
    """
    
    def __init__(self, 
                 ema_config_name: str = None, 
                 data_fetcher=None, 
                 backtest_mode: bool = False,
                 epic: str = None,
                 use_optimal_parameters: bool = True):
        # Initialize parent but skip the enhanced validator setup
        self.name = 'ema'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self._validator = None  # Skip enhanced validator for simplicity
        
        # Basic initialization
        self.backtest_mode = backtest_mode
        self.price_adjuster = PriceAdjuster()
        self.data_fetcher = data_fetcher  # Store data fetcher for multi-timeframe analysis
        self.epic = epic
        self.use_optimal_parameters = use_optimal_parameters
        
        # Required attributes for backtest compatibility
        self.enable_mtf_analysis = False
        self.mtf_analyzer = None
        
        # EMA configuration - now with dynamic optimization support
        self.ema_config = self._get_ema_periods(epic)
        self.ema_short = self.ema_config.get('short', 12)
        self.ema_long = self.ema_config.get('long', 50) 
        self.ema_trend = self.ema_config.get('trend', 200)
        
        # Basic parameters
        self.eps = 1e-8  # Epsilon for stability (from detect_ema_alerts)
        
        # Dynamic parameter integration
        self.optimal_params = None
        if epic and use_optimal_parameters:
            self._load_optimal_parameters(epic)
        
        # Set confidence threshold (optimal if available, otherwise config default)
        self.min_confidence = self._get_optimal_confidence() or getattr(config, 'MIN_CONFIDENCE', 0.45)
        
        # Use reasonable minimum bars for EMA strategy
        self.min_bars = 50  # 50 bars minimum for EMA 50 to be stable
        
        # Initialize helper modules
        self.trend_validator = EMATrendValidator(logger=self.logger)
        self.signal_calculator = EMASignalCalculator(logger=self.logger, trend_validator=self.trend_validator)
        self.mtf_analyzer = EMAMultiTimeframeAnalyzer(logger=self.logger, data_fetcher=data_fetcher)
        self.indicator_calculator = EMAIndicatorCalculator(logger=self.logger, eps=self.eps)
        
        self.logger.info(f"🎯 EMA Strategy initialized - Periods: {self.ema_short}/{self.ema_long}/{self.ema_trend}")
        self.logger.info(f"🔧 Using simple confidence calculation (no enhanced validator)")
        if backtest_mode:
            self.logger.info("🔥 BACKTEST MODE: Time restrictions disabled")
    
    def _get_ema_periods(self, epic: str = None) -> Dict:
        """Get EMA periods - now with dynamic optimization support"""
        try:
            # NEW: Try to get optimal parameters from optimization results first
            if epic and hasattr(self, 'use_optimal_parameters') and self.use_optimal_parameters:
                try:
                    from optimization.optimal_parameter_service import get_epic_ema_config
                    optimal_config = get_epic_ema_config(epic)
                    self.logger.info(f"🎯 Using optimal EMA periods for {epic}: {optimal_config}")
                    return optimal_config
                except Exception as e:
                    self.logger.warning(f"Could not load optimal parameters for {epic}: {e}, falling back to config")
            
            # FALLBACK: Get EMA configuration from configdata structure
            ema_configs = getattr(config, 'EMA_STRATEGY_CONFIG', {})
            active_config = getattr(config, 'ACTIVE_EMA_CONFIG', 'default')
            
            if active_config in ema_configs:
                return ema_configs[active_config]
            
            # Standard EMA defaults if config not available
            return {'short': 21, 'long': 50, 'trend': 200}
            
        except Exception as e:
            self.logger.warning(f"Could not load EMA config: {e}, using defaults")
            return {'short': 21, 'long': 50, 'trend': 200}
    
    def _load_optimal_parameters(self, epic: str):
        """Load optimal parameters for this epic from optimization results"""
        try:
            from optimization.optimal_parameter_service import get_epic_optimal_parameters
            self.optimal_params = get_epic_optimal_parameters(epic)
            self.logger.info(f"🎯 Loaded optimal parameters for {epic}:")
            self.logger.info(f"   EMA Config: {self.optimal_params.ema_config}")
            self.logger.info(f"   Confidence: {self.optimal_params.confidence_threshold:.0%}")
            self.logger.info(f"   Timeframe: {self.optimal_params.timeframe}")
            self.logger.info(f"   SL/TP: {self.optimal_params.stop_loss_pips:.0f}/{self.optimal_params.take_profit_pips:.0f}")
            self.logger.info(f"   Performance Score: {self.optimal_params.performance_score:.3f}")
        except Exception as e:
            self.logger.warning(f"Could not load optimal parameters for {epic}: {e}")
            self.optimal_params = None
    
    def _get_optimal_confidence(self) -> Optional[float]:
        """Get optimal confidence threshold if available"""
        if self.optimal_params:
            return self.optimal_params.confidence_threshold
        return None
    
    def get_optimal_stop_loss(self) -> Optional[float]:
        """Get optimal stop loss in pips"""
        if self.optimal_params:
            return self.optimal_params.stop_loss_pips
        return None
    
    def get_optimal_take_profit(self) -> Optional[float]:
        """Get optimal take profit in pips"""
        if self.optimal_params:
            return self.optimal_params.take_profit_pips
        return None
    
    def get_optimal_timeframe(self) -> Optional[str]:
        """Get optimal timeframe for this epic"""
        if self.optimal_params:
            return self.optimal_params.timeframe
        return None
    
    def should_enable_smart_money(self) -> bool:
        """Check if smart money analysis should be enabled"""
        if self.optimal_params:
            return self.optimal_params.smart_money_enabled
        return False
    
    def get_required_indicators(self) -> List[str]:
        """Required indicators for basic EMA strategy"""
        return self.indicator_calculator.get_required_indicators(self.ema_config)
    
    def detect_signal_auto(
        self, 
        df: pd.DataFrame, 
        epic: str = None, 
        spread_pips: float = 0,
        timeframe: str = '5m',
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
        timeframe: str = '5m',
        evaluation_time: str = None
    ) -> Optional[Dict]:
        """
        🎯 CORE SIGNAL DETECTION: Simple EMA crossover with trend alignment
        
        Based on detect_ema_alerts logic:
        1. Calculate EMAs if not present
        2. Detect price crossover with short EMA  
        3. Validate trend alignment (EMA cascade)
        4. Generate bull/bear signals with confidence
        """
        
        try:
            # Validate data requirements
            if not self.indicator_calculator.validate_data_requirements(df, self.min_bars):
                return None
            
            self.logger.debug(f"Processing {len(df)} bars for {epic}")
            
            # Calculate EMAs if not present
            df_enhanced = self.indicator_calculator.ensure_emas(df.copy(), self.ema_config)
            
            # Apply core detection logic (based on detect_ema_alerts)
            df_with_signals = self.indicator_calculator.detect_ema_alerts(df_enhanced)
            
            # Simple immediate signal detection (no confirmation candle logic)
            latest_row = df_with_signals.iloc[-1]
            
            # Debug logging for troubleshooting
            if len(df) < 80:  # Only log for small datasets to avoid spam
                bull_alert = latest_row.get('bull_alert', False)
                bear_alert = latest_row.get('bear_alert', False)
                if bull_alert or bear_alert:
                    self.logger.info(f"🎯 EMA Alert detected! Bull: {bull_alert}, Bear: {bear_alert}")
            
            signal = self._check_immediate_signal(latest_row, epic, timeframe, spread_pips, len(df), df_with_signals)
            if signal:
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal detection error: {e}")
            return None
    
    
    def _get_1h_two_pole_color(self, epic: str, current_time: pd.Timestamp) -> Optional[str]:
        """Get the Two-Pole Oscillator color from 1H timeframe"""
        return self.mtf_analyzer.get_1h_two_pole_color(epic, current_time)
    
    def _validate_ema_200_trend(self, row: pd.Series, signal_type: str) -> bool:
        """EMA 200 TREND FILTER: Ensure signals align with major trend direction"""
        return self.trend_validator.validate_ema_200_trend(row, signal_type)
    
    def _check_immediate_signal(self, latest_row: pd.Series, epic: str, timeframe: str, spread_pips: float, bar_count: int, df_with_signals: pd.DataFrame) -> Optional[Dict]:
        """Original immediate signal detection logic (fallback)"""
        try:
            # Check for bull alert with Two-Pole Oscillator color validation
            if latest_row.get('bull_alert', False):
                self.logger.info(f"🎯 EMA BULL alert detected at bar {bar_count}")
                
                # CRITICAL: Reject BULL signals if Two-Pole Oscillator is PURPLE (wrong color)
                if not self.trend_validator.validate_two_pole_color(latest_row, 'BULL'):
                    return None
                    
                # Check 1H Two-Pole color if multi-timeframe validation is enabled
                if getattr(config, 'TWO_POLE_MTF_VALIDATION', True):
                    current_time = latest_row.get('start_time', pd.Timestamp.now())
                    if not self.mtf_analyzer.validate_1h_two_pole(epic, current_time, 'BULL'):
                        return None
                
                # Momentum Bias Index validation
                if not self.trend_validator.validate_momentum_bias(latest_row, 'BULL'):
                    return None
                
                # MACD momentum validation
                if not self.trend_validator.validate_macd_momentum(df_with_signals, 'BULL'):
                    return None
                
                # EMA 200 trend filter check
                trend_valid = self._validate_ema_200_trend(latest_row, 'BULL')
                if not trend_valid:
                    self.logger.warning(f"❌ EMA BULL signal REJECTED: Price below EMA 200 (against major trend)")
                    return None
                
                signal = self._create_signal(
                    signal_type='BULL',
                    epic=epic,
                    timeframe=timeframe,
                    latest_row=latest_row,
                    spread_pips=spread_pips
                )
                if signal:
                    self.logger.info(f"✅ EMA BULL signal generated: {signal['confidence']:.1%}")
                    return signal
                else:
                    self.logger.info(f"❌ EMA BULL signal creation failed")
            
            # Check for bear alert with Two-Pole Oscillator color validation
            if latest_row.get('bear_alert', False):
                self.logger.info(f"🎯 EMA BEAR alert detected at bar {bar_count}")
                
                # CRITICAL: Reject BEAR signals if Two-Pole Oscillator is GREEN (wrong color)
                if not self.trend_validator.validate_two_pole_color(latest_row, 'BEAR'):
                    return None
                    
                # Check 1H Two-Pole color if multi-timeframe validation is enabled
                if getattr(config, 'TWO_POLE_MTF_VALIDATION', True):
                    current_time = latest_row.get('start_time', pd.Timestamp.now())
                    if not self.mtf_analyzer.validate_1h_two_pole(epic, current_time, 'BEAR'):
                        return None
                
                # Momentum Bias Index validation
                if not self.trend_validator.validate_momentum_bias(latest_row, 'BEAR'):
                    return None
                
                # MACD momentum validation
                if not self.trend_validator.validate_macd_momentum(df_with_signals, 'BEAR'):
                    return None
                
                # EMA 200 trend filter check
                trend_valid = self._validate_ema_200_trend(latest_row, 'BEAR')
                if not trend_valid:
                    self.logger.warning(f"❌ EMA BEAR signal REJECTED: Price above EMA 200 (against major trend)")
                    return None
                
                signal = self._create_signal(
                    signal_type='BEAR', 
                    epic=epic,
                    timeframe=timeframe,
                    latest_row=latest_row,
                    spread_pips=spread_pips
                )
                if signal:
                    self.logger.info(f"✅ EMA BEAR signal generated: {signal['confidence']:.1%}")
                    return signal
                else:
                    self.logger.info(f"❌ EMA BEAR signal creation failed")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal detection error: {e}")
            return None
    
    def _ensure_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs using our own periods (delegated to helper)"""
        return self.indicator_calculator.ensure_emas(df, self.ema_config)
    
    def _detect_ema_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        """CORE DETECTION LOGIC: Based on the provided detect_ema_alerts function (delegated to helper)"""
        return self.indicator_calculator.detect_ema_alerts(df)
    
    def _calculate_simple_confidence(self, latest_row: pd.Series, signal_type: str) -> float:
        """SIMPLE CONFIDENCE CALCULATION: Based on EMA alignment and crossover strength (delegated to helper)"""
        return self.signal_calculator.calculate_simple_confidence(latest_row, signal_type)
    
    def _validate_two_pole_oscillator(self, latest_row: pd.Series, signal_type: str) -> float:
        """TWO-POLE OSCILLATOR VALIDATION: Validate EMA signals with momentum oscillator (delegated to helper)"""
        return self.trend_validator.validate_two_pole_oscillator(latest_row, signal_type)
    
    def _create_signal(
        self,
        signal_type: str,
        epic: str, 
        timeframe: str,
        latest_row: pd.Series,
        spread_pips: float
    ) -> Dict:
        """Create a signal dictionary with all required fields"""
        try:
            # Create base signal using parent method
            signal = self.create_base_signal(signal_type, epic, timeframe, latest_row)
            
            # Add EMA-specific data
            signal.update({
                'ema_short': latest_row.get('ema_short', 0),
                'ema_long': latest_row.get('ema_long', 0), 
                'ema_trend': latest_row.get('ema_trend', 0),
                'bull_cross': latest_row.get('bull_cross', False),
                'bear_cross': latest_row.get('bear_cross', False),
                'bull_condition': latest_row.get('bull_condition', False),
                'bear_condition': latest_row.get('bear_condition', False)
            })
            
            # Calculate simple confidence based on EMA alignment and crossover strength
            confidence = self._calculate_simple_confidence(latest_row, signal_type)
            
            # Add confidence and execution prices
            signal['confidence'] = confidence
            signal['confidence_score'] = confidence  # For compatibility
            
            # Add execution prices  
            signal = self.add_execution_prices(signal, spread_pips)
            
            # Validate confidence threshold
            if not self.signal_calculator.validate_confidence_threshold(confidence):
                return None
            
            self.logger.info(f"🎯 EMA {signal_type} signal generated: {confidence:.1%} confidence at {signal['price']:.5f}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating signal: {e}")
            return None
    
    def create_enhanced_signal_data(self, latest_row: pd.Series, signal_type: str) -> Dict:
        """Create signal data for confidence calculation - matches BaseStrategy expected format"""
        try:
            # Calculate a basic efficiency ratio from price action
            close = latest_row.get('close', 0)
            open_price = latest_row.get('open', close)
            high = latest_row.get('high', close)
            low = latest_row.get('low', close)
            
            # Simple efficiency calculation: directional move vs total range
            directional_move = abs(close - open_price)
            total_range = high - low if high > low else 0.0001  # Avoid division by zero
            basic_efficiency = min(1.0, directional_move / total_range) if total_range > 0 else 0.5
            
            # For EMA crossover signals, boost efficiency if we have strong trend alignment
            ema_short = latest_row.get('ema_short', 0)
            ema_long = latest_row.get('ema_long', 0)
            ema_trend = latest_row.get('ema_trend', 0)
            
            # If we have proper EMA alignment, increase efficiency
            if signal_type == 'BULL' and ema_short > ema_long > ema_trend:
                basic_efficiency = max(basic_efficiency, 0.6)  # Minimum 60% for aligned bull signals
            elif signal_type == 'BEAR' and ema_short < ema_long < ema_trend:
                basic_efficiency = max(basic_efficiency, 0.6)  # Minimum 60% for aligned bear signals
            else:
                # No clear trend alignment, keep conservative
                basic_efficiency = max(basic_efficiency, 0.3)  # Minimum 30%
            
            return {
                'ema_data': {
                    'ema_short': ema_short,
                    'ema_long': ema_long, 
                    'ema_trend': ema_trend
                },
                'macd_data': {
                    'macd_line': latest_row.get('macd_line', 0),
                    'macd_signal': latest_row.get('macd_signal', 0),
                    'macd_histogram': latest_row.get('macd_histogram', 0)
                },
                'kama_data': {
                    'efficiency_ratio': basic_efficiency,  # Calculated efficiency!
                    'kama_value': ema_short,  # Use EMA short as KAMA proxy
                    'kama_trend': 1.0 if signal_type == 'BULL' else -1.0
                },
                'other_indicators': {
                    'atr': latest_row.get('atr', total_range),  # Use range as ATR proxy
                    'bb_upper': latest_row.get('bb_upper', 0),
                    'bb_middle': latest_row.get('bb_middle', close),
                    'bb_lower': latest_row.get('bb_lower', 0),
                    'rsi': latest_row.get('rsi', 50),
                    'volume': latest_row.get('ltv', 1000),  # Use ltv for volume
                    'volume_ratio': latest_row.get('volume_ratio_10', 1.0)
                },
                'indicator_count': 8,  # Count of indicators available
                'data_source': 'ema_strategy_enhanced',
                'signal_type': signal_type,
                'price': close
            }
        except Exception as e:
            self.logger.error(f"Error creating signal data: {e}")
            # Simple fallback
            return {
                'ema_data': {'ema_short': 0, 'ema_long': 0, 'ema_trend': 0},
                'macd_data': {'macd_histogram': 0},
                'kama_data': {'efficiency_ratio': 0.5},  # Safe default
                'other_indicators': {},
                'signal_type': signal_type,
                'price': latest_row.get('close', 0)
            }


class LegacyEMAStrategy(EMAStrategy):
    """
    🔄 LEGACY COMPATIBILITY: Wrapper for any code that depends on the old implementation
    
    This ensures backward compatibility while using the new simplified implementation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("📦 Legacy EMAStrategy wrapper initialized - using new simplified implementation")
    
    # Add any legacy method stubs here if needed for compatibility


def create_ema_strategy(data_fetcher=None, **kwargs) -> EMAStrategy:
    """
    🏭 FACTORY FUNCTION: Create EMA strategy instance
    
    Simple factory function for backward compatibility with existing code.
    """
    return EMAStrategy(data_fetcher=data_fetcher, **kwargs)