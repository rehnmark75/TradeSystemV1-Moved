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
from .helpers.adaptive_volatility_calculator import AdaptiveVolatilityCalculator

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
    from configdata.strategies import config_macd_strategy
except ImportError:
    from forex_scanner.configdata import config
    try:
        from forex_scanner.configdata.strategies import config_macd_strategy
    except ImportError:
        config_macd_strategy = None

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
    
    def __init__(self, data_fetcher=None, backtest_mode: bool = False, epic: str = None, timeframe: str = '15m', use_optimized_parameters: bool = False, pipeline_mode: bool = True):  # DISABLED optimization - use standard MACD (12,26,9)
        # Initialize parent
        self.name = 'macd'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

        # EMERGENCY: Force INFO level logging for debugging
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.debug(f"MACD Strategy initializing for epic={epic}, timeframe={timeframe}")
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
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.65)  # QUALITY: Raised to 65% for high-quality signals only
        self.min_bars = 60  # Minimum bars for stable MACD (26 + 9 + buffer)
        
        # Initialize helper modules (orchestrator pattern)
        self.indicator_calculator = MACDIndicatorCalculator(logger=self.logger, eps=self.eps)
        self.trend_validator = MACDTrendValidator(logger=self.logger)
        self.signal_calculator = MACDSignalCalculator(logger=self.logger, trend_validator=self.trend_validator)

        # Enable/disable expensive features based on pipeline mode
        self.enhanced_validation = pipeline_mode and getattr(config, 'MACD_ENHANCED_VALIDATION', True)
        if self.enhanced_validation:
            self.mtf_analyzer = MACDMultiTimeframeAnalyzer(logger=self.logger, data_fetcher=data_fetcher)
            self.logger.info("🔍 Enhanced MTF analyzer initialized - Multi-timeframe validation enabled")
        else:
            self.mtf_analyzer = None
            self.logger.info("🔧 Enhanced validation DISABLED - Using basic MACD signal detection")
        
        # Set MTF analyzer for backtest compatibility
        if self.enhanced_validation and self.enable_mtf_analysis and self.mtf_analyzer and self.mtf_analyzer.is_mtf_enabled():
            self.mtf_analyzer_instance = self.mtf_analyzer  # For compatibility
        else:
            self.mtf_analyzer_instance = None
        
        # Add simple caching for performance optimization
        self._enhanced_df_cache = {}  # {epic: {'df': DataFrame, 'timestamp': timestamp}}
        self._cache_ttl_seconds = 300  # 5 minutes cache TTL

        # PHASE 1+2+3: Adaptive volatility-based SL/TP calculation
        # Set optimal_params to None to force dynamic calculation in calculate_optimal_sl_tp()
        self.optimal_params = None

        # Feature flag for adaptive volatility calculator (default: False for gradual rollout)
        self.use_adaptive_sl_tp = getattr(config_macd_strategy, 'USE_ADAPTIVE_SL_TP', False) if config_macd_strategy else False

        if self.use_adaptive_sl_tp:
            # Initialize adaptive volatility calculator (singleton)
            self.adaptive_calculator = AdaptiveVolatilityCalculator(logger=self.logger)
            self.logger.info("🧠 Adaptive volatility calculator enabled - Runtime regime-aware SL/TP")
        else:
            # Fallback: Use ATR multipliers from config for dynamic SL/TP calculation
            self.adaptive_calculator = None
            self.stop_atr_multiplier = getattr(config_macd_strategy, 'MACD_STOP_LOSS_ATR_MULTIPLIER', 2.5) if config_macd_strategy else 2.5
            self.target_atr_multiplier = getattr(config_macd_strategy, 'MACD_TAKE_PROFIT_ATR_MULTIPLIER', 3.0) if config_macd_strategy else 3.0

            # Get pair-specific parameters if available
            if config_macd_strategy and hasattr(config_macd_strategy, 'MACD_PAIR_SPECIFIC_PARAMS') and self.epic:
                # Extract clean pair name (EURUSD, GBPUSD, etc.)
                clean_pair = self.epic.replace('CS.D.', '').replace('.CEEM.IP', '').replace('.MINI.IP', '')
                pair_params = config_macd_strategy.MACD_PAIR_SPECIFIC_PARAMS.get(clean_pair, {})
                if pair_params:
                    self.stop_atr_multiplier = pair_params.get('stop_atr_multiplier', self.stop_atr_multiplier)
                    self.target_atr_multiplier = pair_params.get('target_atr_multiplier', self.target_atr_multiplier)
                    self.logger.info(f"✅ Using pair-specific ATR multipliers for {clean_pair}: SL={self.stop_atr_multiplier}x, TP={self.target_atr_multiplier}x")

            self.logger.info(f"🎯 ATR-based dynamic stops enabled: SL={self.stop_atr_multiplier}x ATR, TP={self.target_atr_multiplier}x ATR")

        self.logger.info(f"🎯 MACD Strategy initialized - Periods: {self.fast_ema}/{self.slow_ema}/{self.signal_ema} ({timeframe})")
        self.logger.info(f"🔧 Using lightweight orchestrator pattern with focused helpers")
        self.logger.info(f"⚡ Enhanced with DataFrame caching (TTL: {self._cache_ttl_seconds}s)")

        if self.enhanced_validation:
            self.logger.info(f"🔍 Enhanced validation ENABLED - Multi-timeframe analysis enabled")
        else:
            self.logger.info(f"🔧 Enhanced validation DISABLED - Basic mode for fast testing")

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
            
            # Force database lookup if optimization is available and epic is provided
            has_epic_data = False
            if self.epic and OPTIMIZATION_AVAILABLE:
                try:
                    has_epic_data = is_epic_macd_optimized(self.epic, self.timeframe)
                    self.logger.info(f"🔍 Database check for {self.epic}: {has_epic_data}")
                except Exception as check_error:
                    self.logger.warning(f"Database check failed: {check_error}")
                    has_epic_data = False

            final_condition = (self.use_optimized_parameters and
                               OPTIMIZATION_AVAILABLE and
                               self.epic and
                               has_epic_data)
            
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
            
            # Fallback: Use database defaults if optimization not available
            else:
                self.logger.info(f"📊 Using DATABASE defaults for {self.epic or 'default'} ({self.timeframe}) - proven 8-17-9 settings")

                # Try to get from database fallback method
                try:
                    from optimization.optimal_parameter_service import OptimalParameterService
                    service = OptimalParameterService()
                    fallback_params = service._get_macd_fallback_parameters(self.epic or 'CS.D.EURUSD.MINI.IP', self.timeframe)

                    return {
                        'fast_ema': fallback_params.fast_ema,
                        'slow_ema': fallback_params.slow_ema,
                        'signal_ema': fallback_params.signal_ema,
                        'confidence_threshold': fallback_params.confidence_threshold,
                        'histogram_threshold': fallback_params.histogram_threshold,
                        'rsi_filter_enabled': fallback_params.rsi_filter_enabled,
                        'momentum_confirmation': fallback_params.momentum_confirmation,
                        'zero_line_filter': fallback_params.zero_line_filter,
                        'mtf_enabled': fallback_params.mtf_enabled,
                        'stop_loss_pips': fallback_params.stop_loss_pips,
                        'take_profit_pips': fallback_params.take_profit_pips
                    }
                except Exception as fallback_error:
                    self.logger.warning(f"Database fallback failed: {fallback_error}, using minimal defaults")
                    # Last resort minimal defaults
                    return {
                        'fast_ema': 8, 'slow_ema': 17, 'signal_ema': 9,
                        'confidence_threshold': 0.6, 'histogram_threshold': 0.00005,
                        'rsi_filter_enabled': True, 'momentum_confirmation': True, 'divergence_detection': True, 'divergence_detection': True,
                        'zero_line_filter': False, 'mtf_enabled': False,
                        'stop_loss_pips': 10.0, 'take_profit_pips': 20.0
                    }
            
        except Exception as e:
            self.logger.warning(f"Could not load MACD config: {e}, using DATABASE defaults")
            # Last resort: Use database stored defaults
            try:
                from optimization.optimal_parameter_service import OptimalParameterService
                service = OptimalParameterService()
                fallback_params = service._get_macd_fallback_parameters(self.epic or 'CS.D.EURUSD.MINI.IP', self.timeframe)
                return {
                    'fast_ema': fallback_params.fast_ema,
                    'slow_ema': fallback_params.slow_ema,
                    'signal_ema': fallback_params.signal_ema,
                    'confidence_threshold': fallback_params.confidence_threshold,
                    'histogram_threshold': fallback_params.histogram_threshold,
                    'rsi_filter_enabled': fallback_params.rsi_filter_enabled,
                    'momentum_confirmation': fallback_params.momentum_confirmation,
                    'zero_line_filter': fallback_params.zero_line_filter,
                    'mtf_enabled': fallback_params.mtf_enabled,
                    'stop_loss_pips': fallback_params.stop_loss_pips,
                    'take_profit_pips': fallback_params.take_profit_pips
                }
            except Exception as fallback_error:
                self.logger.error(f"All config methods failed: {fallback_error}")
                # Absolute last resort hardcoded values
                return {
                    'fast_ema': 8, 'slow_ema': 17, 'signal_ema': 9,
                    'confidence_threshold': 0.6, 'histogram_threshold': 0.00005,
                    'rsi_filter_enabled': True, 'momentum_confirmation': True, 'divergence_detection': True,
                    'zero_line_filter': False, 'mtf_enabled': False,
                    'stop_loss_pips': 10.0, 'take_profit_pips': 20.0
                }
    
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
            self.logger.debug(f"MACD detect_signal called for {epic} with {len(df)} bars (timeframe: {timeframe})")

            # Validate data requirements
            if not self.indicator_calculator.validate_data_requirements(df, self.min_bars):
                self.logger.warning(f"Data validation failed for {epic} - need {self.min_bars} bars, have {len(df)}")
                return None
            
            self.logger.debug(f"Processing {len(df)} bars for {epic}")

            # 0. Check cache first for performance optimization
            cached_df = self._get_cached_enhanced_df(epic, df)
            if cached_df is not None:
                self.logger.debug(f"⚡ [CACHE] Using cached enhanced DataFrame for {epic}")
                df_enhanced = cached_df
            else:
                # 1. Check if MACD indicators already exist (from data_fetcher optimization)
                required_macd_cols = ['macd_line', 'macd_signal', 'macd_histogram']
                macd_exists = all(col in df.columns for col in required_macd_cols)

                # Debug: Show what indicators are available
                indicator_cols = [col for col in df.columns if any(x in col.lower() for x in ['macd', 'rsi', 'adx', 'ema'])]
                self.logger.debug(f"📊 Available indicators for {epic}: {indicator_cols}")

                if macd_exists:
                    # PHASE 2 FIX: Check if existing MACD was calculated with current parameters
                    current_params = f"{self.fast_ema}-{self.slow_ema}-{self.signal_ema}"

                    # If we're using optimized parameters, force recalculation
                    if ((self.fast_ema == 5 and self.slow_ema == 13 and self.signal_ema == 3) or
                        (self.fast_ema == 8 and self.slow_ema == 34 and self.signal_ema == 13) or
                        (self.fast_ema == 21 and self.slow_ema == 55 and self.signal_ema == 9)):
                        self.logger.info(f"🔄 [FORCE RECALC] Using optimized {self.fast_ema}-{self.slow_ema}-{self.signal_ema} parameters - forcing MACD recalculation for {epic}")
                        macd_exists = False  # Force recalculation
                        df_enhanced = df.copy()  # Initialize df_enhanced for recalculation
                    else:
                        self.logger.debug(f"MACD indicators already exist for {epic} - reusing cached values")
                        df_enhanced = df.copy()

                        # Ensure we have EMA 200 for trend validation
                        if 'ema_200' not in df_enhanced.columns:
                            df_enhanced['ema_200'] = df_enhanced['close'].ewm(span=200).mean()

                if not macd_exists:

                    # Ensure we have RSI and ADX for quality scoring
                    if 'rsi' not in df_enhanced.columns or 'adx' not in df_enhanced.columns:
                        self.logger.debug(f"📊 Adding missing RSI/ADX indicators for {epic}")
                        df_enhanced = self.indicator_calculator._add_enhanced_filters(df_enhanced)
                else:
                    self.logger.info(f"📊 [STANDARD MACD] Used standard detection for {epic}")
                    df_enhanced = self.indicator_calculator.ensure_macd_indicators(df.copy(), self.macd_config)

                # Cache the enhanced DataFrame for future use
                self._cache_enhanced_df(epic, df_enhanced)

            # 2. Detect MACD crossovers (with strength filtering)
            self.logger.debug(f"Detecting crossovers for {epic} with {len(df_enhanced)} bars")
            df_with_signals = self.indicator_calculator.detect_macd_crossovers(df_enhanced, epic, is_backtest=self.backtest_mode)

            # EMERGENCY: Check what signals were detected
            bull_signals = df_with_signals.get('bull_alert', pd.Series(False, index=df_with_signals.index)).sum()
            bear_signals = df_with_signals.get('bear_alert', pd.Series(False, index=df_with_signals.index)).sum()
            self.logger.info(f"🚨 CROSSOVER DETECTION RESULT for {epic}: {bull_signals} bull alerts, {bear_signals} bear alerts")

            # DEBUG: Find which bars have alerts
            if bull_signals > 0 or bear_signals > 0:
                bull_alert_bars = df_with_signals[df_with_signals.get('bull_alert', False) == True]
                bear_alert_bars = df_with_signals[df_with_signals.get('bear_alert', False) == True]
                if len(bull_alert_bars) > 0:
                    bull_times = bull_alert_bars['start_time'].tolist() if 'start_time' in bull_alert_bars.columns else bull_alert_bars.index.tolist()
                    self.logger.info(f"📍 Bull alert bars for {epic}: {bull_times[-3:]}")  # Show last 3
                if len(bear_alert_bars) > 0:
                    bear_times = bear_alert_bars['start_time'].tolist() if 'start_time' in bear_alert_bars.columns else bear_alert_bars.index.tolist()
                    self.logger.info(f"📍 Bear alert bars for {epic}: {bear_times[-3:]}")  # Show last 3
            
            # 3. Check for signals - scan all bars with crossovers (for backtest compatibility)
            if self.backtest_mode:
                # BACKTEST MODE: Collect all signals in this window and return the latest one
                # This ensures daily limits and filters were already applied in detect_macd_crossovers
                all_signals = []

                for idx, row in df_with_signals.iterrows():
                    bull_alert = row.get('bull_alert', False)
                    bear_alert = row.get('bear_alert', False)

                    if bull_alert or bear_alert:
                        signal_type = 'BULL' if bull_alert else 'BEAR'
                        self.logger.debug(f"Found {signal_type} alert at index {idx} - validating...")

                        signal = self._check_immediate_signal(row, epic, timeframe, spread_pips, len(df))
                        if signal:
                            self.logger.info(f"✅ Signal VALIDATED: {signal_type} at {idx} with confidence {signal.get('confidence', 'unknown')}")
                            # Add timing info to signal for backtest
                            if hasattr(row, 'name'):
                                signal['signal_time'] = row.name
                            elif 'start_time' in row:
                                signal['signal_time'] = row['start_time']
                            all_signals.append((idx, signal))
                        else:
                            self.logger.info(f"❌ Signal REJECTED: {signal_type} at {idx} failed validation")

                # Return the LATEST signal only (respects the filtering that was already applied)
                if all_signals:
                    # Sort by timestamp and return the latest
                    all_signals.sort(key=lambda x: x[0])
                    latest_signal = all_signals[-1][1]
                    self.logger.debug(f"🎯 Backtest: Found {len(all_signals)} valid signals, returning latest at {latest_signal.get('signal_time', 'unknown')}")
                    return latest_signal

                return None
            else:
                # LIVE MODE: Check latest bar AND recent bars for delayed signals
                latest_row = df_with_signals.iloc[-1]

                # Check latest bar first
                bull_alert = latest_row.get('bull_alert', False)
                bear_alert = latest_row.get('bear_alert', False)
                self.logger.info(f"🔍 LIVE MODE - Latest bar check for {epic}: bull_alert={bull_alert}, bear_alert={bear_alert}")

                if bull_alert or bear_alert:
                    self.logger.info(f"🎯 MACD Alert on LATEST BAR! {epic}: Bull={bull_alert}, Bear={bear_alert}")
                    signal = self._check_immediate_signal(latest_row, epic, timeframe, spread_pips, len(df))
                    if signal:
                        return signal

                # NEW: Check recent bars for delayed signals (allow signals within 5 bars of crossover)
                # This matches backtest behavior where we find crossovers in recent history
                lookback_bars = getattr(config_macd_strategy, 'MACD_CONFIRMATION_LOOKBACK', 5)
                allow_delayed = getattr(config_macd_strategy, 'MACD_ALLOW_DELAYED_SIGNALS', True)

                if allow_delayed and lookback_bars > 0:
                    self.logger.info(f"🔍 Checking last {lookback_bars} bars for recent MACD crossovers on {epic}...")

                    # Check last N bars for crossover signals
                    for i in range(1, min(lookback_bars + 1, len(df_with_signals))):
                        row = df_with_signals.iloc[-(i+1)]  # -2, -3, -4, -5, -6
                        bull_alert = row.get('bull_alert', False)
                        bear_alert = row.get('bear_alert', False)

                        # Debug: Log every bar checked
                        if i <= 3:  # Only log first 3 bars to avoid spam
                            bar_time = row.get('start_time', 'unknown')
                            self.logger.info(f"   Bar -{i+1} ({bar_time}): bull_alert={bull_alert}, bear_alert={bear_alert}")

                        if bull_alert or bear_alert:
                            bar_time = row.get('start_time', 'unknown')
                            signal_type_str = 'BULL' if bull_alert else 'BEAR'
                            self.logger.info(f"🎯 MACD Alert found {i} bars ago (at {bar_time})! {epic}: Bull={bull_alert}, Bear={bear_alert}")

                            # CRITICAL: Validate using crossover bar's data (matches backtest behavior)
                            # This ensures backtest and live use identical validation logic
                            signal = self._check_immediate_signal(row, epic, timeframe, spread_pips, len(df), signal_type=signal_type_str)
                            if signal:
                                self.logger.info(f"✅ Delayed MACD {signal_type_str} signal validated from {i} bars ago")
                                return signal
                            else:
                                self.logger.info(f"❌ Delayed MACD {signal_type_str} signal from {i} bars ago failed validation")

                return None

        except Exception as e:
            self.logger.error(f"Signal detection error: {e}")
            return None

    def _get_cached_enhanced_df(self, epic: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get cached enhanced DataFrame if available and fresh"""
        try:
            if epic not in self._enhanced_df_cache:
                return None

            cache_entry = self._enhanced_df_cache[epic]
            current_time = pd.Timestamp.now()

            # Check if cache is still fresh (TTL not expired)
            if (current_time - cache_entry['timestamp']).seconds > self._cache_ttl_seconds:
                del self._enhanced_df_cache[epic]  # Cleanup expired cache
                return None

            # Check if DataFrame structure matches (basic validation)
            cached_df = cache_entry['df']
            if len(cached_df) >= len(df) and all(col in cached_df.columns for col in df.columns):
                self.logger.debug(f"⚡ [CACHE HIT] Using cached enhanced DataFrame for {epic}")
                # Return latest subset to match input DataFrame length
                return cached_df.tail(len(df)).copy()

            return None

        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
            return None

    def _cache_enhanced_df(self, epic: str, df_enhanced: pd.DataFrame):
        """Cache enhanced DataFrame for future use"""
        try:
            self._enhanced_df_cache[epic] = {
                'df': df_enhanced.copy(),
                'timestamp': pd.Timestamp.now()
            }
            self.logger.debug(f"⚡ [CACHE STORE] Cached enhanced DataFrame for {epic}")

        except Exception as e:
            self.logger.error(f"Cache storage error: {e}")

    def _check_immediate_signal(self, latest_row: pd.Series, epic: str, timeframe: str, spread_pips: float, bar_count: int, signal_type: str = None) -> Optional[Dict]:
        """Check for immediate MACD crossover signals with all validations

        Args:
            signal_type: Optional 'BULL' or 'BEAR' to force validation of that signal type
                        (used for delayed signals where latest_row may not have alert flags)
        """
        try:
            # Determine signal type from latest_row or parameter
            has_bull_alert = latest_row.get('bull_alert', False) or signal_type == 'BULL'
            has_bear_alert = latest_row.get('bear_alert', False) or signal_type == 'BEAR'

            # Check for bull crossover
            if has_bull_alert:
                self.logger.debug(f"Validating BULL crossover for {epic} at bar {bar_count}")
                
                # EMA 200 trend filter - DISABLED (MACD is momentum strategy, can trade counter-trend)
                # if not self.trend_validator.validate_ema_200_trend(latest_row, 'BULL'):
                #     self.logger.info("❌ MACD BULL signal REJECTED: Price below EMA 200 (against major trend)")
                #     return None

                # Validate MACD histogram direction
                histogram_val = latest_row.get('macd_histogram', 0)
                self.logger.info(f"🔍 BULL signal validation: histogram={histogram_val:.6f}, ADX={latest_row.get('adx', 0):.1f}")
                if not self.trend_validator.validate_macd_histogram_direction(latest_row, 'BULL'):
                    self.logger.info(f"❌ MACD BULL signal REJECTED: Negative histogram ({histogram_val:.6f})")
                    return None

                # RSI confluence validation (if enabled)
                if self.macd_config.get('rsi_filter_enabled', False):
                    if not self.trend_validator.validate_rsi_confluence(latest_row, 'BULL'):
                        self.logger.info("❌ MACD BULL signal REJECTED: RSI confluence failed")
                        return None

                # ADX trend strength validation (Phase 2: Market regime awareness)
                min_adx = getattr(config_macd_strategy, 'MACD_MIN_ADX', 30) if config_macd_strategy else 30  # Raised from 20 to 30 for quality signals
                if not self.trend_validator.validate_adx_trend_strength(latest_row, 'BULL', min_adx=min_adx):
                    self.logger.info(f"❌ MACD BULL signal REJECTED: ADX trend strength insufficient (ADX < {min_adx}, ranging market)")
                    return None

                # PHASE 2: Market regime classification and adaptive validation
                market_regime = self.trend_validator.classify_market_regime(latest_row)
                if market_regime['recommendation'] == 'avoid_trading':
                    self.logger.info(f"❌ MACD BULL signal REJECTED: Unfavorable market regime ({market_regime['regime']})")
                    return None
                elif market_regime['recommendation'] == 'trade_conservatively':
                    self.logger.info(f"⚠️ MACD BULL signal: Conservative regime ({market_regime['regime']}) - higher quality required")
                
                # Optional: Multi-timeframe validation (only in pipeline mode)
                mtf_passed = True
                if self.enhanced_validation and self.enable_mtf_analysis and self.mtf_analyzer and self.mtf_analyzer.is_mtf_enabled():
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
                    self.logger.debug(f"✅ MACD BULL signal generated: {signal['confidence']:.1%}")
                    return signal
                else:
                    self.logger.info("❌ MACD BULL signal creation failed")
            
            # Check for bear crossover
            if has_bear_alert:
                self.logger.debug(f"🎯 MACD BEAR crossover detected at bar {bar_count}")
                
                # EMA 200 trend filter - DISABLED (MACD is momentum strategy, can trade counter-trend)
                # if not self.trend_validator.validate_ema_200_trend(latest_row, 'BEAR'):
                #     self.logger.info("❌ MACD BEAR signal REJECTED: Price above EMA 200 (against major trend)")
                #     return None

                # Validate MACD histogram direction
                histogram_val = latest_row.get('macd_histogram', 0)
                self.logger.info(f"🔍 BEAR signal validation: histogram={histogram_val:.6f}, ADX={latest_row.get('adx', 0):.1f}")
                if not self.trend_validator.validate_macd_histogram_direction(latest_row, 'BEAR'):
                    self.logger.info(f"❌ MACD BEAR signal REJECTED: Positive histogram ({histogram_val:.6f})")
                    return None

                # RSI confluence validation (if enabled)
                if self.macd_config.get('rsi_filter_enabled', False):
                    if not self.trend_validator.validate_rsi_confluence(latest_row, 'BEAR'):
                        self.logger.info("❌ MACD BEAR signal REJECTED: RSI confluence failed")
                        return None

                # ADX trend strength validation (Phase 2: Market regime awareness)
                min_adx = getattr(config_macd_strategy, 'MACD_MIN_ADX', 30) if config_macd_strategy else 30  # Raised from 20 to 30 for quality signals
                if not self.trend_validator.validate_adx_trend_strength(latest_row, 'BEAR', min_adx=min_adx):
                    self.logger.info(f"❌ MACD BEAR signal REJECTED: ADX trend strength insufficient (ADX < {min_adx}, ranging market)")
                    return None

                # PHASE 2: Market regime classification and adaptive validation
                market_regime = self.trend_validator.classify_market_regime(latest_row)
                if market_regime['recommendation'] == 'avoid_trading':
                    self.logger.info(f"❌ MACD BEAR signal REJECTED: Unfavorable market regime ({market_regime['regime']})")
                    return None
                elif market_regime['recommendation'] == 'trade_conservatively':
                    self.logger.info(f"⚠️ MACD BEAR signal: Conservative regime ({market_regime['regime']}) - higher quality required")
                
                # Optional: Multi-timeframe validation (only in pipeline mode)
                mtf_passed = True
                if self.enhanced_validation and self.enable_mtf_analysis and self.mtf_analyzer and self.mtf_analyzer.is_mtf_enabled():
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
                    self.logger.debug(f"✅ MACD BEAR signal generated: {signal['confidence']:.1%}")
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
            
            # Calculate confidence using signal calculator (with pair-specific calibration)
            confidence = self.signal_calculator.calculate_simple_confidence(latest_row, signal_type, epic=epic)
            
            # Add MTF boost if enabled and available (only in pipeline mode)
            if self.enhanced_validation and self.enable_mtf_analysis and self.mtf_analyzer and self.mtf_analyzer.is_mtf_enabled():
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
            
            # Add execution prices (ensure spread_pips is numeric)
            try:
                spread_pips_numeric = float(spread_pips) if isinstance(spread_pips, (str, int)) else spread_pips
            except (ValueError, TypeError):
                spread_pips_numeric = 1.5  # Default spread for major pairs
            signal = self.add_execution_prices(signal, spread_pips_numeric)

            # ✅ NEW: Calculate optimized SL/TP
            sl_tp = self.calculate_optimal_sl_tp(signal, epic, latest_row, spread_pips_numeric)
            signal['stop_distance'] = sl_tp['stop_distance']
            signal['limit_distance'] = sl_tp['limit_distance']

            # Convert pip distances to price levels for database storage
            pip_value = 0.01 if 'JPY' in epic else 0.0001
            execution_price = signal.get('execution_price', signal['price'])

            if signal_type == 'BULL':
                signal['stop_loss_price'] = execution_price - (sl_tp['stop_distance'] * pip_value)
                signal['take_profit_price'] = execution_price + (sl_tp['limit_distance'] * pip_value)
            else:  # BEAR
                signal['stop_loss_price'] = execution_price + (sl_tp['stop_distance'] * pip_value)
                signal['take_profit_price'] = execution_price - (sl_tp['limit_distance'] * pip_value)

            # Validate confidence threshold
            if not self.signal_calculator.validate_confidence_threshold(confidence):
                return None

            self.logger.debug(f"🎯 MACD {signal_type} signal generated: {confidence:.1%} confidence at {signal['price']:.5f}, SL/TP={sl_tp['stop_distance']}/{sl_tp['limit_distance']}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating signal: {e}")
            return None
    
    def calculate_optimal_sl_tp(
        self,
        signal: Dict,
        epic: str,
        latest_row: pd.Series,
        spread_pips: float
    ) -> Dict[str, int]:
        """
        PHASE 1+2+3: Calculate SL/TP using adaptive volatility calculator OR ATR multipliers

        Priority:
        1. Adaptive volatility calculator (if enabled via USE_ADAPTIVE_SL_TP)
        2. Fallback to ATR-based with config multipliers

        Overrides base class to use MACD-specific calculation logic.
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

                # ✅ DEBUG: Log actual types and values
                self.logger.info(
                    f"🔍 DEBUG SL/TP types: stop_distance type={type(result.stop_distance)} "
                    f"value={result.stop_distance}, limit_distance type={type(result.limit_distance)} "
                    f"value={result.limit_distance}"
                )

                return {
                    'stop_distance': result.stop_distance,
                    'limit_distance': result.limit_distance
                }

            except Exception as e:
                self.logger.error(f"❌ Adaptive calculator failed: {e}, falling back to ATR method")
                # Fall through to ATR fallback below

        # PHASE 1+2 FALLBACK: ATR-based with config multipliers
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

        # Calculate using MACD-specific ATR multipliers
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
            f"🎯 MACD ATR-based SL/TP: ATR={atr_pips:.1f} pips, "
            f"SL={stop_distance} ({self.stop_atr_multiplier}x), "
            f"TP={limit_distance} ({self.target_atr_multiplier}x), "
            f"R:R={limit_distance/stop_distance:.2f}"
        )

        return {
            'stop_distance': stop_distance,
            'limit_distance': limit_distance
        }

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