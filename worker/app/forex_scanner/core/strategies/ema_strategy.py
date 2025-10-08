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
from .helpers.adaptive_volatility_calculator import AdaptiveVolatilityCalculator
from .helpers.adx_calculator import ADXCalculator
from .helpers.swing_proximity_validator import SwingProximityValidator

try:
    from configdata import config
    from configdata.strategies import config_ema_strategy
except ImportError:
    from forex_scanner.configdata import config
    try:
        from forex_scanner.configdata.strategies import config_ema_strategy
    except ImportError:
        config_ema_strategy = None


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
                 use_optimal_parameters: bool = True,
                 pipeline_mode: bool = True):
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
        self.enable_mtf_analysis = True  # ✅ RE-ENABLED: Multi-timeframe analysis
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

        # PHASE 3: Adaptive volatility-based SL/TP calculation
        self.use_adaptive_sl_tp = getattr(config_ema_strategy, 'USE_ADAPTIVE_SL_TP', False) if config_ema_strategy else False

        if self.use_adaptive_sl_tp:
            # Initialize adaptive volatility calculator (singleton)
            self.adaptive_calculator = AdaptiveVolatilityCalculator(logger=self.logger)
            self.logger.info("🧠 Adaptive volatility calculator enabled - Runtime regime-aware SL/TP")
        else:
            # Fallback: Use ATR multipliers from config
            self.adaptive_calculator = None
            self.stop_atr_multiplier = getattr(config_ema_strategy, 'EMA_STOP_LOSS_ATR_MULTIPLIER', 2.0) if config_ema_strategy else 2.0
            self.target_atr_multiplier = getattr(config_ema_strategy, 'EMA_TAKE_PROFIT_ATR_MULTIPLIER', 4.0) if config_ema_strategy else 4.0
            self.logger.info(f"🎯 ATR-based dynamic stops: SL={self.stop_atr_multiplier}x ATR, TP={self.target_atr_multiplier}x ATR")
        
        # Use reasonable minimum bars for EMA strategy
        self.min_bars = 50  # 50 bars minimum for EMA 50 to be stable
        
        # Initialize helper modules
        self.trend_validator = EMATrendValidator(logger=self.logger)
        self.signal_calculator = EMASignalCalculator(logger=self.logger, trend_validator=self.trend_validator)
        self.mtf_analyzer = EMAMultiTimeframeAnalyzer(logger=self.logger, data_fetcher=data_fetcher)
        self.indicator_calculator = EMAIndicatorCalculator(logger=self.logger, eps=self.eps)

        # Initialize ADX calculator for trend strength analysis
        self.adx_period = getattr(config_ema_strategy, 'EMA_ADX_PERIOD', 14) if config_ema_strategy else 14
        self.min_adx = getattr(config_ema_strategy, 'EMA_MIN_ADX', 25) if config_ema_strategy else 25
        self.adx_calculator = ADXCalculator(period=self.adx_period, logger=self.logger)
        self.logger.info(f"📊 ADX calculator initialized - Period: {self.adx_period}, Min ADX: {self.min_adx}")

        # Initialize swing proximity validator for entry quality
        swing_config = getattr(config_ema_strategy, 'EMA_SWING_VALIDATION', {}) if config_ema_strategy else {}
        self.swing_validator = SwingProximityValidator(
            config=swing_config,
            logger=self.logger
        )
        if swing_config.get('enabled', True):
            self.logger.info(f"🎯 Swing proximity validator initialized - Min distance: {swing_config.get('min_distance_pips', 8)} pips")
        else:
            self.logger.info(f"⚪ Swing proximity validator disabled")

        # Initialize enhanced breakout validator only in pipeline mode
        self.enhanced_validation = pipeline_mode and getattr(config, 'EMA_ENHANCED_VALIDATION', True)
        if self.enhanced_validation:
            try:
                from .helpers.ema_breakout_validator import EMABreakoutValidator
                self.breakout_validator = EMABreakoutValidator(logger=self.logger)
                # Pass backtest mode to breakout validator for threshold adjustment
                if hasattr(self, 'backtest_mode'):
                    self.breakout_validator.backtest_mode = self.backtest_mode
                self.logger.info("🔍 Enhanced breakout validator initialized - False breakout reduction enabled")
            except ImportError as e:
                self.breakout_validator = None
                self.enhanced_validation = False
                self.logger.warning(f"⚠️ Enhanced validator not available: {e}")
        else:
            self.breakout_validator = None
            
        self.logger.info(f"🎯 EMA Strategy initialized - Periods: {self.ema_short}/{self.ema_long}/{self.ema_trend}")
        
        if self.enhanced_validation:
            self.logger.info(f"🔍 Enhanced validation ENABLED - Multi-factor breakout confirmation")
        else:
            self.logger.info(f"🔧 Enhanced validation DISABLED - Using original signal detection")
            
        if backtest_mode:
            self.logger.info("🔥 BACKTEST MODE: Time restrictions disabled")

        # Configuration validation check
        self._validate_macd_configuration()
    
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

    def _validate_macd_configuration(self):
        """Validate that MACD momentum filter configuration is loaded correctly"""
        try:
            macd_enabled = getattr(config, 'MACD_MOMENTUM_FILTER_ENABLED', None)
            validation_mode = getattr(config, 'MACD_VALIDATION_MODE', None)
            lookback = getattr(config, 'MACD_HISTOGRAM_LOOKBACK', None)

            self.logger.info("🔧 MACD Configuration Validation:")
            self.logger.info(f"   🎛️ MACD Filter Enabled: {macd_enabled}")
            self.logger.info(f"   ⚙️ Validation Mode: {validation_mode}")
            self.logger.info(f"   📊 Histogram Lookback: {lookback}")

            if macd_enabled is None:
                self.logger.warning("⚠️ WARNING: MACD_MOMENTUM_FILTER_ENABLED not found in config!")
                self.logger.warning("   This means MACD validation might be disabled by default")
                return False

            if not macd_enabled:
                self.logger.warning("⚠️ WARNING: MACD momentum filter is DISABLED in configuration")
                self.logger.warning("   Bear signals may be generated even with positive MACD momentum")
                return False

            if validation_mode != 'strict_blocking':
                self.logger.warning(f"⚠️ WARNING: MACD validation mode is '{validation_mode}', not 'strict_blocking'")
                self.logger.warning("   This may allow signals with opposing momentum to pass through")

            self.logger.info("✅ MACD configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ MACD configuration validation failed: {e}")
            return False

    def _format_log_prefix(self, epic: str = None) -> str:
        """Format consistent log prefix with strategy name and epic"""
        epic_part = f"[{epic}] " if epic else ""
        return f"[EMA] {epic_part}"

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
            # Format log prefix
            log_prefix = self._format_log_prefix(epic)

            # Validate data requirements
            if not self.indicator_calculator.validate_data_requirements(df, self.min_bars):
                self.logger.info(f"❌ {log_prefix}Data validation failed - need {self.min_bars} bars, got {len(df)}")
                return None

            self.logger.info(f"✅ {log_prefix}Processing {len(df)} bars")

            # CRITICAL FIX: Get epic-specific EMA configuration (including optimal parameters)
            epic_ema_config = self._get_ema_periods(epic)

            # Calculate EMAs if not present
            df_enhanced = self.indicator_calculator.ensure_emas(df.copy(), epic_ema_config)

            # Calculate ADX for trend strength analysis
            df_enhanced = self.adx_calculator.calculate_adx(df_enhanced)
            self.logger.debug(f"📊 {log_prefix}ADX calculated")

            # Calculate swing points for proximity validation (if SMC analyzer available)
            try:
                if hasattr(self, 'swing_validator') and self.swing_validator.smc_analyzer:
                    df_enhanced = self.swing_validator.smc_analyzer.identify_swing_points(df_enhanced)
                    self.logger.debug(f"🎯 {log_prefix}Swing points identified")
            except Exception as e:
                self.logger.debug(f"{log_prefix}Swing point calculation skipped: {e}")

            # Apply core detection logic (based on detect_ema_alerts)
            df_with_signals = self.indicator_calculator.detect_ema_alerts(df_enhanced)

            # UNIFIED MODE: Use same logic for both live and backtest
            # The backtest scanner will call this method iteratively for each timestamp
            latest_row = df_with_signals.iloc[-1]
            current_timestamp = latest_row.get('start_time')

            # BACKTEST FIX: Check if any alert occurred at the current timestamp
            if self.backtest_mode:
                # In backtest mode, find if there's an alert specifically at the current timestamp
                alert_at_current_time = df_with_signals[df_with_signals['start_time'] == current_timestamp]
                if len(alert_at_current_time) > 0:
                    latest_row = alert_at_current_time.iloc[-1]  # Use the row at current timestamp
                    self.logger.info(f"🔍 {log_prefix}BACKTEST: Checking alerts at timestamp {current_timestamp}")
                else:
                    # No data at current timestamp, use latest available
                    pass

            # DEBUG: Check alert flags
            bull_alert = latest_row.get('bull_alert', False)
            bear_alert = latest_row.get('bear_alert', False)
            self.logger.info(f"🔍 {log_prefix}Debug: Bull alert: {bull_alert}, Bear alert: {bear_alert}")

            # Debug logging for troubleshooting
            if len(df) < 80:  # Only log for small datasets to avoid spam
                bull_alert = latest_row.get('bull_alert', False)
                bear_alert = latest_row.get('bear_alert', False)
                if bull_alert or bear_alert:
                    self.logger.info(f"🎯 {log_prefix}Alert detected! Bull: {bull_alert}, Bear: {bear_alert}")

            signal = self._check_immediate_signal(latest_row, epic, timeframe, spread_pips, len(df), df_with_signals)
            if signal:
                return signal

            return None

        except Exception as e:
            log_prefix = self._format_log_prefix(epic)
            self.logger.error(f"{log_prefix}Signal detection error: {e}")
            return None

    def _check_backtest_signals(self, df_with_signals: pd.DataFrame, epic: str, timeframe: str, spread_pips: float) -> Optional[Dict]:
        """
        BACKTEST MODE: Find the most recent alert in the dataset and check if it's NEW

        The key insight: We need to find the latest alert timestamp in the data and
        check if it matches the current timestamp we're processing.
        """
        try:
            if len(df_with_signals) == 0:
                return None

            current_timestamp = df_with_signals.iloc[-1].get('start_time')

            # Find all rows with alerts in the current dataset
            alert_rows = df_with_signals[
                (df_with_signals.get('bull_alert', False) == True) |
                (df_with_signals.get('bear_alert', False) == True)
            ].copy()

            if len(alert_rows) == 0:
                return None

            # Get the most recent alert in the data
            latest_alert_row = alert_rows.iloc[-1]
            alert_timestamp = latest_alert_row.get('start_time')

            # Only process the alert if it's at the current timestamp (i.e., it's a NEW alert)
            if alert_timestamp == current_timestamp:
                bull_alert = latest_alert_row.get('bull_alert', False)
                bear_alert = latest_alert_row.get('bear_alert', False)

                self.logger.info(f"🔍 BACKTEST: NEW alert at {current_timestamp} - Bull: {bull_alert}, Bear: {bear_alert}")

                # Use the same signal checking logic as live mode
                signal = self._check_immediate_signal(latest_alert_row, epic, timeframe, spread_pips, len(df_with_signals), df_with_signals)
                if signal:
                    signal['alert_timestamp'] = alert_timestamp
                    signal['backtest_detection_mode'] = 'new_alert_detection'
                    self.logger.info(f"✅ BACKTEST: Signal generated at {alert_timestamp}")
                    return signal
                else:
                    self.logger.info(f"❌ BACKTEST: Alert at {alert_timestamp} failed validation")
            else:
                # This is an old alert, not a new one
                if len(df_with_signals) % 50 == 0:  # Log occasionally
                    self.logger.debug(f"🔍 BACKTEST: Latest alert was at {alert_timestamp}, current time: {current_timestamp}")

            return None

        except Exception as e:
            self.logger.error(f"Error in backtest signal processing: {e}")
            return None


    def _get_1h_two_pole_color(self, epic: str, current_time: pd.Timestamp) -> Optional[str]:
        """Get the Two-Pole Oscillator color from 1H timeframe"""
        return self.mtf_analyzer.get_1h_two_pole_color(epic, current_time)
    
    def _validate_ema_200_trend(self, row: pd.Series, signal_type: str) -> bool:
        """EMA 200 TREND FILTER: Ensure signals align with major trend direction"""
        # ✅ RE-ENABLED: EMA 200 trend validation now works in both live and backtest modes
        return self.trend_validator.validate_ema_200_trend(row, signal_type)
    
    def _check_immediate_signal(self, latest_row: pd.Series, epic: str, timeframe: str, spread_pips: float, bar_count: int, df_with_signals: pd.DataFrame) -> Optional[Dict]:
        """Original immediate signal detection logic (fallback)"""
        try:
            # Format log prefix
            log_prefix = self._format_log_prefix(epic)

            # Check for bull alert with Two-Pole Oscillator color validation
            if latest_row.get('bull_alert', False):
                self.logger.info(f"🎯 {log_prefix}BULL alert detected at bar {bar_count}")

                # CRITICAL: Reject BULL signals if Two-Pole Oscillator is PURPLE (wrong color)
                if not self.trend_validator.validate_two_pole_color(latest_row, 'BULL', self.backtest_mode):
                    self.logger.info(f"❌ {log_prefix}BULL signal REJECTED by Two-Pole Oscillator validation")
                    return None

                # ✅ RE-ENABLED: 1H Two-Pole color validation for both live and backtest modes
                if getattr(config, 'TWO_POLE_MTF_VALIDATION', True):
                    current_time = latest_row.get('start_time', pd.Timestamp.now())
                    if not self.mtf_analyzer.validate_1h_two_pole(epic, current_time, 'BULL'):
                        self.logger.info(f"❌ {log_prefix}BULL signal REJECTED by 1H Two-Pole validation")
                        return None


                # ✅ RE-ENABLED: MACD momentum validation for both live and backtest modes
                if not self.trend_validator.validate_macd_momentum(df_with_signals, 'BULL'):
                    self.logger.info(f"❌ {log_prefix}BULL signal REJECTED by MACD momentum validation")
                    return None

                # EMA 200 trend filter check
                trend_valid = self._validate_ema_200_trend(latest_row, 'BULL')
                if not trend_valid:
                    self.logger.warning(f"❌ {log_prefix}BULL signal REJECTED: Price below EMA 200 (against major trend)")
                    return None

                # ADX TREND STRENGTH VALIDATION
                adx_value = latest_row.get('adx', 0)
                if adx_value < self.min_adx:
                    self.logger.warning(f"❌ {log_prefix}BULL signal REJECTED: ADX {adx_value:.1f} below minimum {self.min_adx} (weak trend)")
                    return None
                else:
                    self.logger.info(f"✅ {log_prefix}ADX validation passed: {adx_value:.1f} >= {self.min_adx}")

                # SWING PROXIMITY VALIDATION
                current_price = latest_row.get('close', 0)
                swing_result = self.swing_validator.validate_entry_proximity(
                    df_with_signals, current_price, 'BUY', epic, timeframe
                )
                if not swing_result['valid']:
                    if self.swing_validator.strict_mode:
                        self.logger.warning(f"❌ {log_prefix}BULL signal REJECTED: {swing_result['rejection_reason']}")
                        return None
                    else:
                        self.logger.info(f"⚠️ {log_prefix}Swing proximity warning: {swing_result['rejection_reason']}")
                else:
                    self.logger.info(f"✅ {log_prefix}Swing proximity validation passed")

                # ENHANCED VALIDATION: Multi-factor breakout confirmation
                if self.enhanced_validation and self.breakout_validator:
                    is_valid_breakout, breakout_confidence, validation_details = self.breakout_validator.validate_breakout(
                        df_with_signals, 'BULL', epic
                    )

                    if not is_valid_breakout:
                        self.logger.warning(f"❌ {log_prefix}BULL signal REJECTED by enhanced validation (confidence: {breakout_confidence:.1%})")
                        self.logger.debug(f"   {log_prefix}Validation details: {validation_details}")
                        return None
                    else:
                        self.logger.info(f"✅ {log_prefix}Enhanced validation passed for BULL signal (confidence: {breakout_confidence:.1%})")

                signal = self._create_signal(
                    signal_type='BULL',
                    epic=epic,
                    timeframe=timeframe,
                    latest_row=latest_row,
                    spread_pips=spread_pips
                )
                if signal:
                    self.logger.info(f"✅ {log_prefix}BULL signal generated: {signal['confidence']:.1%}")
                    return signal
                else:
                    self.logger.info(f"❌ {log_prefix}BULL signal creation failed")
            
            # Check for bear alert with Two-Pole Oscillator color validation
            if latest_row.get('bear_alert', False):
                self.logger.info(f"🎯 {log_prefix}BEAR alert detected at bar {bar_count}")

                # CRITICAL: Reject BEAR signals if Two-Pole Oscillator is GREEN (wrong color)
                if not self.trend_validator.validate_two_pole_color(latest_row, 'BEAR', self.backtest_mode):
                    return None

                # ✅ RE-ENABLED: 1H Two-Pole color validation for both live and backtest modes
                if getattr(config, 'TWO_POLE_MTF_VALIDATION', True):
                    current_time = latest_row.get('start_time', pd.Timestamp.now())
                    if not self.mtf_analyzer.validate_1h_two_pole(epic, current_time, 'BEAR'):
                        self.logger.info(f"❌ {log_prefix}BEAR signal REJECTED by 1H Two-Pole validation")
                        return None


                # ✅ RE-ENABLED: MACD momentum validation for both live and backtest modes
                if not self.trend_validator.validate_macd_momentum(df_with_signals, 'BEAR'):
                    self.logger.info(f"❌ {log_prefix}BEAR signal REJECTED by MACD momentum validation")
                    return None

                # EMA 200 trend filter check
                trend_valid = self._validate_ema_200_trend(latest_row, 'BEAR')
                if not trend_valid:
                    self.logger.warning(f"❌ {log_prefix}BEAR signal REJECTED: Price above EMA 200 (against major trend)")
                    return None

                # ADX TREND STRENGTH VALIDATION
                adx_value = latest_row.get('adx', 0)
                if adx_value < self.min_adx:
                    self.logger.warning(f"❌ {log_prefix}BEAR signal REJECTED: ADX {adx_value:.1f} below minimum {self.min_adx} (weak trend)")
                    return None
                else:
                    self.logger.info(f"✅ {log_prefix}ADX validation passed: {adx_value:.1f} >= {self.min_adx}")

                # SWING PROXIMITY VALIDATION
                current_price = latest_row.get('close', 0)
                swing_result = self.swing_validator.validate_entry_proximity(
                    df_with_signals, current_price, 'SELL', epic, timeframe
                )
                if not swing_result['valid']:
                    if self.swing_validator.strict_mode:
                        self.logger.warning(f"❌ {log_prefix}BEAR signal REJECTED: {swing_result['rejection_reason']}")
                        return None
                    else:
                        self.logger.info(f"⚠️ {log_prefix}Swing proximity warning: {swing_result['rejection_reason']}")
                else:
                    self.logger.info(f"✅ {log_prefix}Swing proximity validation passed")

                # ENHANCED VALIDATION: Multi-factor breakout confirmation
                if self.enhanced_validation and self.breakout_validator:
                    is_valid_breakout, breakout_confidence, validation_details = self.breakout_validator.validate_breakout(
                        df_with_signals, 'BEAR', epic
                    )

                    if not is_valid_breakout:
                        self.logger.warning(f"❌ {log_prefix}BEAR signal REJECTED by enhanced validation (confidence: {breakout_confidence:.1%})")
                        self.logger.debug(f"   {log_prefix}Validation details: {validation_details}")
                        return None
                    else:
                        self.logger.info(f"✅ {log_prefix}Enhanced validation passed for BEAR signal (confidence: {breakout_confidence:.1%})")

                signal = self._create_signal(
                    signal_type='BEAR',
                    epic=epic,
                    timeframe=timeframe,
                    latest_row=latest_row,
                    spread_pips=spread_pips
                )
                if signal:
                    self.logger.info(f"✅ {log_prefix}BEAR signal generated: {signal['confidence']:.1%}")
                    return signal
                else:
                    self.logger.info(f"❌ {log_prefix}BEAR signal creation failed")

            return None

        except Exception as e:
            log_prefix = self._format_log_prefix(epic)
            self.logger.error(f"{log_prefix}Signal detection error: {e}")
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

    def _apply_overextension_filters(self, latest_row: pd.Series, signal_type: str, base_confidence: float) -> float:
        """
        Apply overextension filters to adjust signal confidence

        Args:
            latest_row: Latest market data row
            signal_type: 'bull' or 'bear'
            base_confidence: Original confidence score

        Returns:
            Adjusted confidence score after applying overextension filters
        """
        try:
            # Import the config functions we need
            from configdata.strategies.config_ema_strategy import (
                check_stochastic_overextension,
                check_williams_r_overextension,
                check_rsi_extreme_overextension,
                calculate_composite_overextension_score,
                COMPOSITE_OVEREXTENSION_ENABLED,
                OVEREXTENSION_MODE,
                OVEREXTENSION_DEBUG_LOGGING
            )

            # Get indicator values from the data row
            stoch_k = latest_row.get('stoch_k', 50.0)  # Default to neutral if missing
            stoch_d = latest_row.get('stoch_d', 50.0)
            williams_r = latest_row.get('williams_r', -50.0)  # Default to neutral
            rsi = latest_row.get('rsi', 50.0)  # Default to neutral

            # Convert signal type to direction for overextension functions
            signal_direction = 'long' if signal_type == 'bull' else 'short'

            adjusted_confidence = base_confidence

            if COMPOSITE_OVEREXTENSION_ENABLED:
                # Use composite scoring for multiple oscillator agreement
                composite_result = calculate_composite_overextension_score(
                    stoch_k, stoch_d, williams_r, rsi, signal_direction
                )

                if composite_result['composite_overextended']:
                    if OVEREXTENSION_MODE == 'hard_block':
                        if OVEREXTENSION_DEBUG_LOGGING:
                            self.logger.info(f"🚫 {signal_type.upper()} signal BLOCKED by composite overextension "
                                           f"({composite_result['indicators_triggered']}/3 indicators triggered)")
                        return 0.0  # Block signal completely
                    else:
                        # Apply confidence penalty
                        penalty = composite_result['total_penalty']
                        adjusted_confidence = max(0.0, base_confidence - penalty)
                        if OVEREXTENSION_DEBUG_LOGGING:
                            self.logger.info(f"⚠️ {signal_type.upper()} confidence reduced by {penalty:.3f} "
                                           f"due to composite overextension ({composite_result['indicators_triggered']}/3)")

            else:
                # Check individual filters and apply penalties
                total_penalty = 0.0
                triggered_filters = []

                # Check Stochastic
                stoch_result = check_stochastic_overextension(stoch_k, stoch_d, signal_direction)
                if stoch_result['overextended']:
                    total_penalty += stoch_result['penalty']
                    triggered_filters.append(f"Stochastic({stoch_k:.1f})")

                # Check Williams %R
                williams_result = check_williams_r_overextension(williams_r, signal_direction)
                if williams_result['overextended']:
                    total_penalty += williams_result['penalty']
                    triggered_filters.append(f"Williams%R({williams_r:.1f})")

                # Check RSI
                rsi_result = check_rsi_extreme_overextension(rsi, signal_direction)
                if rsi_result['overextended']:
                    total_penalty += rsi_result['penalty']
                    triggered_filters.append(f"RSI({rsi:.1f})")

                # Apply penalties
                if total_penalty > 0:
                    if OVEREXTENSION_MODE == 'hard_block' and len(triggered_filters) >= 2:
                        if OVEREXTENSION_DEBUG_LOGGING:
                            self.logger.info(f"🚫 {signal_type.upper()} signal BLOCKED by overextension filters: {', '.join(triggered_filters)}")
                        return 0.0  # Block signal if multiple filters triggered
                    else:
                        adjusted_confidence = max(0.0, base_confidence - total_penalty)
                        if OVEREXTENSION_DEBUG_LOGGING and total_penalty > 0:
                            self.logger.info(f"⚠️ {signal_type.upper()} confidence reduced by {total_penalty:.3f} "
                                           f"due to overextension: {', '.join(triggered_filters)}")

            return adjusted_confidence

        except Exception as e:
            self.logger.error(f"Error applying overextension filters: {e}")
            return base_confidence  # Return original confidence on error
    
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
            # Format log prefix
            log_prefix = self._format_log_prefix(epic)

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

            # Apply overextension filters if enabled
            confidence = self._apply_overextension_filters(latest_row, signal_type, confidence)

            # Add confidence and execution prices
            signal['confidence'] = confidence
            signal['confidence_score'] = confidence  # For compatibility

            # Add execution prices
            signal = self.add_execution_prices(signal, spread_pips)

            # ✅ NEW: Calculate optimized SL/TP
            sl_tp = self.calculate_optimal_sl_tp(signal, epic, latest_row, spread_pips)
            signal['stop_distance'] = sl_tp['stop_distance']
            signal['limit_distance'] = sl_tp['limit_distance']

            # Validate confidence threshold
            if not self.signal_calculator.validate_confidence_threshold(confidence):
                return None

            self.logger.info(f"🎯 {log_prefix}{signal_type} signal generated: {confidence:.1%} confidence at {signal['price']:.5f}, SL/TP={sl_tp['stop_distance']}/{sl_tp['limit_distance']}")
            return signal

        except Exception as e:
            log_prefix = self._format_log_prefix(epic)
            self.logger.error(f"{log_prefix}Error creating signal: {e}")
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

    def calculate_optimal_sl_tp(
        self,
        signal: Dict,
        epic: str,
        latest_row: pd.Series,
        spread_pips: float
    ) -> Dict[str, int]:
        """
        PHASE 3: Calculate SL/TP using adaptive volatility calculator OR ATR multipliers

        Priority:
        1. Adaptive volatility calculator (if enabled via USE_ADAPTIVE_SL_TP)
        2. Fallback to ATR-based with config multipliers

        Overrides base class to use EMA-specific calculation logic.
        """
        signal_type = signal.get('signal_type', 'BULL')

        # PHASE 3: Use adaptive volatility calculator if enabled
        if self.use_adaptive_sl_tp and self.adaptive_calculator:
            try:
                result = self.adaptive_calculator.calculate_sl_tp(
                    epic=epic,
                    data=latest_row,
                    signal_type=signal_type
                )

                self.logger.info(
                    f"🧠 Adaptive SL/TP [{result.regime.value}] {result.method_used}: "
                    f"SL={result.stop_distance}p TP={result.limit_distance}p "
                    f"(R:R={result.limit_distance/result.stop_distance:.2f}, "
                    f"conf={result.confidence:.1%}, fallback_lvl={result.fallback_level}, "
                    f"{result.calculation_time_ms:.1f}ms)"
                )

                return {
                    'stop_distance': result.stop_distance,
                    'limit_distance': result.limit_distance
                }

            except Exception as e:
                self.logger.error(f"❌ Adaptive calculator failed: {e}, falling back to ATR method")
                # Fall through to ATR fallback below

        # FALLBACK: ATR-based with config multipliers
        # Get ATR for the pair
        atr = latest_row.get('atr', 0)
        if not atr or atr <= 0:
            # Fallback: estimate from current volatility (high-low range)
            atr = abs(latest_row.get('high', 0) - latest_row.get('low', 0))
            self.logger.warning(f"No ATR indicator, using high-low range: {atr}")

        # Convert ATR to pips/points
        if 'JPY' in epic:
            atr_pips = atr * 100  # JPY pairs: 0.01 = 1 pip
        else:
            atr_pips = atr * 10000  # Standard pairs: 0.0001 = 1 pip

        # Calculate using EMA-specific ATR multipliers (trend-following: wider targets)
        raw_stop = atr_pips * self.stop_atr_multiplier
        raw_target = atr_pips * self.target_atr_multiplier

        # Apply minimum safe distances
        if 'JPY' in epic:
            min_sl = 20  # Minimum 20 pips for JPY
        else:
            min_sl = 15  # Minimum 15 pips for others

        stop_distance = max(int(raw_stop), min_sl)
        limit_distance = int(raw_target)

        # Apply reasonable maximums to prevent excessive risk
        if 'JPY' in epic:
            max_sl = 55
        elif 'GBP' in epic:
            max_sl = 60  # GBP pairs are more volatile
        else:
            max_sl = 45

        if stop_distance > max_sl:
            self.logger.warning(f"Stop distance {stop_distance} exceeds max {max_sl}, capping to maximum")
            stop_distance = max_sl
            limit_distance = int(stop_distance * (self.target_atr_multiplier / self.stop_atr_multiplier))

        self.logger.info(
            f"🎯 EMA ATR-based SL/TP: ATR={atr_pips:.1f} pips, "
            f"SL={stop_distance} ({self.stop_atr_multiplier}x), "
            f"TP={limit_distance} ({self.target_atr_multiplier}x), "
            f"R:R={limit_distance/stop_distance:.2f}"
        )

        return {
            'stop_distance': stop_distance,
            'limit_distance': limit_distance
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