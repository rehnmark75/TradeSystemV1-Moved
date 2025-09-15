# core/strategies/ichimoku_strategy.py
"""
Ichimoku Strategy Implementation - CLOUD TRADING SYSTEM
🌥️ ICHIMOKU CLOUD: Traditional Japanese candlestick analysis system
📊 COMPREHENSIVE: TK crosses, cloud breakouts, and Chikou span confirmation
⚡ LIGHTWEIGHT: Orchestrator pattern with focused helper modules
🏗️ ORCHESTRATOR: Main class coordinates, helpers do the specialized work

Features:
- Tenkan-sen/Kijun-sen (TK) crossover detection
- Cloud (Kumo) breakout analysis with leading spans
- Chikou Span momentum confirmation
- Multi-timeframe Ichimoku validation (optional)
- Bull/Bear signal generation with confidence scoring
- Compatible with existing backtest system
- Configurable Ichimoku periods through database optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster

# Import optimization functions
try:
    from optimization.optimal_parameter_service import get_ichimoku_optimal_parameters, is_epic_ichimoku_optimized
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    def get_ichimoku_optimal_parameters(*args, **kwargs):
        raise ImportError("Optimization service not available")
    def is_epic_ichimoku_optimized(*args, **kwargs):
        return False

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class IchimokuStrategy(BaseStrategy):
    """
    🌥️ ICHIMOKU CLOUD STRATEGY: Orchestrator pattern implementation

    Traditional Ichimoku Kinko Hyo (One Glance Equilibrium Chart) strategy with:
    - TK line crossovers (Tenkan-sen vs Kijun-sen)
    - Cloud breakouts (price vs Kumo/Cloud)
    - Chikou span confirmation (lagging span momentum)
    Coordinates focused helper modules to keep main class lightweight and maintainable.
    """

    def __init__(self,
                 data_fetcher=None,
                 backtest_mode: bool = False,
                 epic: str = None,
                 timeframe: str = '15m',
                 use_optimized_parameters: bool = True):
        # Initialize parent
        self.name = 'ichimoku'
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

        # Ichimoku configuration - traditional settings: 9, 26, 52, 26
        self.ichimoku_config = self._get_ichimoku_periods(epic)
        self.tenkan_period = self.ichimoku_config.get('tenkan_period', 9)     # Conversion line
        self.kijun_period = self.ichimoku_config.get('kijun_period', 26)     # Base line
        self.senkou_b_period = self.ichimoku_config.get('senkou_b_period', 52)  # Leading span B
        self.chikou_shift = self.ichimoku_config.get('chikou_shift', 26)     # Lagging span displacement
        self.cloud_shift = self.ichimoku_config.get('cloud_shift', 26)       # Cloud forward displacement

        # Basic parameters
        self.eps = 1e-8  # Epsilon for stability
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.55)  # Ichimoku needs higher confidence
        self.min_bars = 80  # Minimum bars for stable Ichimoku (52 + 26 + buffer)

        # Strategy-specific thresholds
        self.cloud_thickness_threshold = self.ichimoku_config.get('cloud_thickness_threshold', 0.0001)
        self.tk_cross_strength_threshold = self.ichimoku_config.get('tk_cross_strength_threshold', 0.5)
        self.chikou_clear_threshold = self.ichimoku_config.get('chikou_clear_threshold', 0.0002)

        # Will be initialized after helpers are available
        self.indicator_calculator = None
        self.trend_validator = None
        self.signal_calculator = None
        self.mtf_analyzer = None

        # Initialize helper modules (orchestrator pattern)
        self._initialize_helpers()

        # Load optimal parameters if available
        self.optimal_params = None
        if epic and use_optimized_parameters:
            self._load_optimal_parameters(epic)

        # Log initialization
        self._log_initialization()

    def _get_ichimoku_periods(self, epic: str = None) -> Dict:
        """Get Ichimoku periods - with database optimization support"""
        try:
            # Try to get optimal parameters from optimization results first
            if epic and hasattr(self, 'use_optimized_parameters') and self.use_optimized_parameters:
                try:
                    if OPTIMIZATION_AVAILABLE and is_epic_ichimoku_optimized(epic):
                        optimal_params = get_ichimoku_optimal_parameters(epic)
                        config = {
                            'tenkan_period': optimal_params.tenkan_period,
                            'kijun_period': optimal_params.kijun_period,
                            'senkou_b_period': optimal_params.senkou_b_period,
                            'chikou_shift': optimal_params.chikou_shift,
                            'cloud_shift': optimal_params.cloud_shift,
                            'cloud_thickness_threshold': optimal_params.cloud_thickness_threshold,
                            'tk_cross_strength_threshold': optimal_params.tk_cross_strength_threshold,
                            'chikou_clear_threshold': optimal_params.chikou_clear_threshold
                        }
                        self.logger.info(f"🎯 Using optimal Ichimoku periods for {epic}: {config}")
                        return config
                except Exception as e:
                    self.logger.warning(f"Could not load optimal parameters for {epic}: {e}, falling back to config")

            # FALLBACK: Get Ichimoku configuration from configdata structure
            ichimoku_configs = getattr(config, 'ICHIMOKU_STRATEGY_CONFIG', {})
            active_config = getattr(config, 'ACTIVE_ICHIMOKU_CONFIG', 'traditional')

            if active_config in ichimoku_configs:
                return ichimoku_configs[active_config]

            # Traditional Ichimoku defaults: 9-26-52-26
            return {
                'tenkan_period': 9,      # Conversion line (fast)
                'kijun_period': 26,      # Base line (medium)
                'senkou_b_period': 52,   # Leading span B (slow)
                'chikou_shift': 26,      # Lagging span displacement
                'cloud_shift': 26,       # Cloud forward shift
                'cloud_thickness_threshold': 0.0001,
                'tk_cross_strength_threshold': 0.5,
                'chikou_clear_threshold': 0.0002
            }

        except Exception as e:
            self.logger.warning(f"Could not load Ichimoku config: {e}, using traditional defaults")
            return {
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_b_period': 52,
                'chikou_shift': 26,
                'cloud_shift': 26,
                'cloud_thickness_threshold': 0.0001,
                'tk_cross_strength_threshold': 0.5,
                'chikou_clear_threshold': 0.0002
            }

    def _should_enable_mtf_analysis(self, timeframe: str) -> bool:
        """Determine if multi-timeframe analysis should be enabled"""
        if self.backtest_mode:
            return False  # Disable MTF in backtest for performance

        # Enable MTF for lower timeframes to get higher timeframe context
        lower_timeframes = ['1m', '5m', '15m']
        return timeframe in lower_timeframes

    def _initialize_helpers(self):
        """Initialize helper modules with lazy loading fallback"""
        try:
            # Import helpers with fallback pattern
            self.logger.info("Initializing Ichimoku helper modules...")

            # Indicator Calculator
            try:
                from .helpers.ichimoku_indicator_calculator import IchimokuIndicatorCalculator
                self.indicator_calculator = IchimokuIndicatorCalculator(logger=self.logger, eps=self.eps)
                self.logger.info("✅ Ichimoku indicator calculator initialized")
            except ImportError as e:
                self.logger.error(f"❌ Failed to load Ichimoku indicator calculator: {e}")
                raise ImportError("Ichimoku indicator calculator is required")

            # Trend Validator
            try:
                from .helpers.ichimoku_trend_validator import IchimokuTrendValidator
                self.trend_validator = IchimokuTrendValidator(logger=self.logger)
                self.logger.info("✅ Ichimoku trend validator initialized")
            except ImportError as e:
                self.logger.error(f"❌ Failed to load Ichimoku trend validator: {e}")
                raise ImportError("Ichimoku trend validator is required")

            # Signal Calculator
            try:
                from .helpers.ichimoku_signal_calculator import IchimokuSignalCalculator
                self.signal_calculator = IchimokuSignalCalculator(
                    logger=self.logger,
                    trend_validator=self.trend_validator
                )
                self.logger.info("✅ Ichimoku signal calculator initialized")
            except ImportError as e:
                self.logger.error(f"❌ Failed to load Ichimoku signal calculator: {e}")
                raise ImportError("Ichimoku signal calculator is required")

            # MTF Analyzer
            try:
                from .helpers.ichimoku_mtf_analyzer import IchimokuMultiTimeframeAnalyzer
                self.mtf_analyzer = IchimokuMultiTimeframeAnalyzer(
                    logger=self.logger,
                    data_fetcher=self.data_fetcher
                )
                self.logger.info("✅ Ichimoku MTF analyzer initialized")
            except ImportError as e:
                self.logger.warning(f"⚠️ Ichimoku MTF analyzer not available: {e}")
                self.mtf_analyzer = None
                self.enable_mtf_analysis = False

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Ichimoku helpers: {e}")
            raise

    def _load_optimal_parameters(self, epic: str):
        """Load optimal parameters for this epic from optimization results"""
        try:
            if OPTIMIZATION_AVAILABLE:
                self.optimal_params = get_ichimoku_optimal_parameters(epic)
                self.logger.info(f"🎯 Loaded optimal parameters for {epic}:")
                self.logger.info(f"   Periods: {self.optimal_params.tenkan_period}-{self.optimal_params.kijun_period}-{self.optimal_params.senkou_b_period}-{self.optimal_params.chikou_shift}")
                self.logger.info(f"   Confidence: {self.optimal_params.confidence_threshold:.0%}")
                self.logger.info(f"   Timeframe: {self.optimal_params.timeframe}")
                self.logger.info(f"   Cloud Thickness: {self.optimal_params.cloud_thickness_threshold:.6f}")
                self.logger.info(f"   Performance Score: {self.optimal_params.performance_score:.3f}")
        except Exception as e:
            self.logger.warning(f"Could not load optimal parameters for {epic}: {e}")
            self.optimal_params = None

    def _log_initialization(self):
        """Log strategy initialization details"""
        self.logger.info(f"🌥️ Ichimoku Strategy initialized - Periods: {self.tenkan_period}-{self.kijun_period}-{self.senkou_b_period}-{self.chikou_shift}")
        self.logger.info(f"   Cloud shift: {self.cloud_shift} periods")
        self.logger.info(f"   Confidence threshold: {self.min_confidence:.0%}")
        self.logger.info(f"   Cloud thickness threshold: {self.cloud_thickness_threshold:.6f}")
        self.logger.info(f"   TK cross strength: {self.tk_cross_strength_threshold:.1f}")

        if self.backtest_mode:
            self.logger.info("🔥 BACKTEST MODE: Time restrictions and MTF disabled")
        elif self.enable_mtf_analysis:
            self.logger.info("📊 Multi-timeframe analysis ENABLED")
        else:
            self.logger.info("📊 Multi-timeframe analysis DISABLED")

    def get_required_indicators(self) -> List[str]:
        """Required indicators for Ichimoku strategy"""
        if self.indicator_calculator:
            return self.indicator_calculator.get_required_indicators(self.ichimoku_config)

        # Fallback if helper not available
        return ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']

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
        🌥️ CORE SIGNAL DETECTION: Ichimoku Cloud analysis

        Ichimoku signal logic:
        1. Calculate Ichimoku components if not present
        2. Detect TK line crossovers (Tenkan vs Kijun)
        3. Validate cloud breakouts (price vs Kumo)
        4. Confirm with Chikou span momentum
        5. Generate bull/bear signals with confidence
        """

        try:
            # Validate data requirements
            if not self.indicator_calculator.validate_data_requirements(df, self.min_bars):
                self.logger.info(f"🌥️ Ichimoku {epic}: Insufficient data ({len(df)} bars, need {self.min_bars})")
                return None

            self.logger.info(f"🌥️ Ichimoku {epic}: Processing {len(df)} bars for signal detection")

            # Calculate Ichimoku indicators if not present
            df_enhanced = self.indicator_calculator.ensure_ichimoku_indicators(df.copy(), self.ichimoku_config)

            # Apply core Ichimoku detection logic
            df_with_signals = self.indicator_calculator.detect_ichimoku_signals(df_enhanced, self.ichimoku_config)

            # Get latest data for signal evaluation
            latest_row = df_with_signals.iloc[-1]

            # Enhanced logging for monitoring
            tk_bull = latest_row.get('tk_bull_cross', False)
            tk_bear = latest_row.get('tk_bear_cross', False)
            cloud_bull = latest_row.get('cloud_bull_breakout', False)
            cloud_bear = latest_row.get('cloud_bear_breakout', False)

            # Always log what we found (or didn't find)
            if any([tk_bull, tk_bear, cloud_bull, cloud_bear]):
                self.logger.info(f"🌥️ Ichimoku {epic}: Potential signals - TK: Bull={tk_bull}/Bear={tk_bear}, Cloud: Bull={cloud_bull}/Bear={cloud_bear}")
            else:
                self.logger.info(f"🌥️ Ichimoku {epic}: No TK crosses or cloud breakouts detected")

            # Check for immediate signals
            signal = self._check_immediate_signal(latest_row, epic, timeframe, spread_pips, len(df), df_with_signals)
            if signal:
                return signal
            else:
                self.logger.info(f"🌥️ Ichimoku {epic}: Signal validation failed or no valid signals")

            return None

        except Exception as e:
            self.logger.error(f"Signal detection error: {e}")
            return None

    def _check_immediate_signal(self, latest_row: pd.Series, epic: str, timeframe: str, spread_pips: float, bar_count: int, df_with_signals: pd.DataFrame) -> Optional[Dict]:
        """Check for immediate Ichimoku signals"""
        try:
            # Check for bull signal (TK cross OR cloud breakout with confirmations)
            bull_tk_cross = latest_row.get('tk_bull_cross', False)
            bull_cloud_breakout = latest_row.get('cloud_bull_breakout', False)

            if bull_tk_cross or bull_cloud_breakout:
                signal_type = 'TK_BULL' if bull_tk_cross else 'CLOUD_BULL'
                self.logger.info(f"🌥️ Ichimoku {signal_type} signal detected at bar {bar_count}")

                # Validate cloud position (price should be above cloud for bull signals)
                if not self.trend_validator.validate_cloud_position(latest_row, 'BULL'):
                    self.logger.info(f"🌥️ Ichimoku {epic}: BULL signal failed cloud position validation")
                    return None

                # Validate Chikou span (should be clear of historical price action)
                if not self.trend_validator.validate_chikou_span(df_with_signals, 'BULL'):
                    self.logger.info(f"🌥️ Ichimoku {epic}: BULL signal failed Chikou span validation")
                    return None

                # Multi-timeframe validation if enabled
                if self.enable_mtf_analysis and self.mtf_analyzer:
                    current_time = latest_row.get('start_time', pd.Timestamp.now())
                    if not self.mtf_analyzer.validate_mtf_ichimoku(epic, current_time, 'BULL'):
                        self.logger.info(f"🌥️ Ichimoku {epic}: BULL signal failed MTF validation")
                        return None

                # Create bull signal
                signal = self._create_signal(
                    signal_type='BULL',
                    epic=epic,
                    timeframe=timeframe,
                    latest_row=latest_row,
                    spread_pips=spread_pips,
                    signal_source=signal_type
                )
                if signal:
                    self.logger.info(f"✅ Ichimoku BULL signal generated: {signal['confidence']:.1%}")
                    return signal

            # Check for bear signal (TK cross OR cloud breakout with confirmations)
            bear_tk_cross = latest_row.get('tk_bear_cross', False)
            bear_cloud_breakout = latest_row.get('cloud_bear_breakout', False)

            if bear_tk_cross or bear_cloud_breakout:
                signal_type = 'TK_BEAR' if bear_tk_cross else 'CLOUD_BEAR'
                self.logger.info(f"🌥️ Ichimoku {signal_type} signal detected at bar {bar_count}")

                # Validate cloud position (price should be below cloud for bear signals)
                if not self.trend_validator.validate_cloud_position(latest_row, 'BEAR'):
                    self.logger.info(f"🌥️ Ichimoku {epic}: BEAR signal failed cloud position validation")
                    return None

                # Validate Chikou span (should be clear of historical price action)
                if not self.trend_validator.validate_chikou_span(df_with_signals, 'BEAR'):
                    self.logger.info(f"🌥️ Ichimoku {epic}: BEAR signal failed Chikou span validation")
                    return None

                # Multi-timeframe validation if enabled
                if self.enable_mtf_analysis and self.mtf_analyzer:
                    current_time = latest_row.get('start_time', pd.Timestamp.now())
                    if not self.mtf_analyzer.validate_mtf_ichimoku(epic, current_time, 'BEAR'):
                        self.logger.info(f"🌥️ Ichimoku {epic}: BEAR signal failed MTF validation")
                        return None

                # Create bear signal
                signal = self._create_signal(
                    signal_type='BEAR',
                    epic=epic,
                    timeframe=timeframe,
                    latest_row=latest_row,
                    spread_pips=spread_pips,
                    signal_source=signal_type
                )
                if signal:
                    self.logger.info(f"✅ Ichimoku BEAR signal generated: {signal['confidence']:.1%}")
                    return signal

            return None

        except Exception as e:
            self.logger.error(f"Signal evaluation error: {e}")
            return None

    def _create_signal(
        self,
        signal_type: str,
        epic: str,
        timeframe: str,
        latest_row: pd.Series,
        spread_pips: float,
        signal_source: str = 'ICHIMOKU'
    ) -> Dict:
        """Create a signal dictionary with all required fields"""
        try:
            # Create base signal using parent method
            signal = self.create_base_signal(signal_type, epic, timeframe, latest_row)

            # Add Ichimoku-specific data
            signal.update({
                'tenkan_sen': latest_row.get('tenkan_sen', 0),
                'kijun_sen': latest_row.get('kijun_sen', 0),
                'senkou_span_a': latest_row.get('senkou_span_a', 0),
                'senkou_span_b': latest_row.get('senkou_span_b', 0),
                'chikou_span': latest_row.get('chikou_span', 0),
                'cloud_top': latest_row.get('cloud_top', 0),
                'cloud_bottom': latest_row.get('cloud_bottom', 0),
                'tk_bull_cross': latest_row.get('tk_bull_cross', False),
                'tk_bear_cross': latest_row.get('tk_bear_cross', False),
                'cloud_bull_breakout': latest_row.get('cloud_bull_breakout', False),
                'cloud_bear_breakout': latest_row.get('cloud_bear_breakout', False),
                'signal_source': signal_source
            })

            # Calculate confidence based on Ichimoku analysis
            confidence = self.signal_calculator.calculate_ichimoku_confidence(latest_row, signal_type)

            # Add confidence and execution prices
            signal['confidence'] = confidence
            signal['confidence_score'] = confidence  # For compatibility

            # Add execution prices
            signal = self.add_execution_prices(signal, spread_pips)

            # Validate confidence threshold
            if not self.signal_calculator.validate_confidence_threshold(confidence, self.min_confidence):
                return None

            self.logger.info(f"🌥️ Ichimoku {signal_type} signal generated: {confidence:.1%} confidence at {signal['price']:.5f}")
            return signal

        except Exception as e:
            self.logger.error(f"Error creating signal: {e}")
            return None

    def get_optimal_stop_loss(self) -> Optional[float]:
        """Get optimal stop loss in pips"""
        if self.optimal_params:
            return self.optimal_params.stop_loss_pips
        # Ichimoku typically uses wider stops due to cloud thickness
        return 15.0

    def get_optimal_take_profit(self) -> Optional[float]:
        """Get optimal take profit in pips"""
        if self.optimal_params:
            return self.optimal_params.take_profit_pips
        # Ichimoku can target wider moves due to trend-following nature
        return 30.0

    def get_optimal_timeframe(self) -> Optional[str]:
        """Get optimal timeframe for this epic"""
        if self.optimal_params:
            return self.optimal_params.timeframe
        return '15m'  # Good balance for Ichimoku

    def should_enable_smart_money(self) -> bool:
        """Check if smart money analysis should be enabled"""
        if self.optimal_params:
            return getattr(self.optimal_params, 'smart_money_enabled', False)
        return False

    def create_enhanced_signal_data(self, latest_row: pd.Series, signal_type: str) -> Dict:
        """Create signal data for confidence calculation - matches BaseStrategy expected format"""
        try:
            # Get basic OHLC data
            close = latest_row.get('close', 0)
            open_price = latest_row.get('open', close)
            high = latest_row.get('high', close)
            low = latest_row.get('low', close)

            # Calculate basic efficiency from Ichimoku alignment
            tenkan = latest_row.get('tenkan_sen', 0)
            kijun = latest_row.get('kijun_sen', 0)
            cloud_top = latest_row.get('cloud_top', 0)
            cloud_bottom = latest_row.get('cloud_bottom', 0)

            # Ichimoku alignment efficiency
            if signal_type == 'BULL':
                # For bull signals: price > cloud, tenkan > kijun ideal
                price_vs_cloud = 1.0 if close > max(cloud_top, cloud_bottom) else 0.3
                tk_alignment = 1.0 if tenkan > kijun else 0.5
            else:
                # For bear signals: price < cloud, tenkan < kijun ideal
                price_vs_cloud = 1.0 if close < min(cloud_top, cloud_bottom) else 0.3
                tk_alignment = 1.0 if tenkan < kijun else 0.5

            ichimoku_efficiency = (price_vs_cloud + tk_alignment) / 2

            return {
                'ichimoku_data': {
                    'tenkan_sen': tenkan,
                    'kijun_sen': kijun,
                    'senkou_span_a': latest_row.get('senkou_span_a', 0),
                    'senkou_span_b': latest_row.get('senkou_span_b', 0),
                    'chikou_span': latest_row.get('chikou_span', 0),
                    'cloud_top': cloud_top,
                    'cloud_bottom': cloud_bottom
                },
                'ema_data': {
                    'ema_short': tenkan,  # Use Tenkan as fast EMA equivalent
                    'ema_long': kijun,    # Use Kijun as slow EMA equivalent
                    'ema_trend': (cloud_top + cloud_bottom) / 2  # Cloud midpoint as trend
                },
                'kama_data': {
                    'efficiency_ratio': ichimoku_efficiency,  # Ichimoku alignment efficiency
                    'kama_value': tenkan,  # Use Tenkan as KAMA proxy
                    'kama_trend': 1.0 if signal_type == 'BULL' else -1.0
                },
                'other_indicators': {
                    'atr': high - low if high > low else 0.0001,  # Use range as ATR proxy
                    'bb_upper': cloud_top,
                    'bb_middle': (cloud_top + cloud_bottom) / 2,
                    'bb_lower': cloud_bottom,
                    'rsi': 60 if signal_type == 'BULL' else 40,  # Directional bias
                    'volume': latest_row.get('ltv', 1000),
                    'volume_ratio': latest_row.get('volume_ratio_10', 1.0)
                },
                'indicator_count': 10,  # Count of indicators available
                'data_source': 'ichimoku_strategy',
                'signal_type': signal_type,
                'price': close
            }
        except Exception as e:
            self.logger.error(f"Error creating signal data: {e}")
            # Simple fallback
            return {
                'ichimoku_data': {'tenkan_sen': 0, 'kijun_sen': 0, 'cloud_top': 0, 'cloud_bottom': 0},
                'ema_data': {'ema_short': 0, 'ema_long': 0, 'ema_trend': 0},
                'kama_data': {'efficiency_ratio': 0.5},
                'other_indicators': {},
                'signal_type': signal_type,
                'price': latest_row.get('close', 0)
            }


def create_ichimoku_strategy(data_fetcher=None, **kwargs) -> IchimokuStrategy:
    """
    🏭 FACTORY FUNCTION: Create Ichimoku strategy instance

    Simple factory function for backward compatibility with existing code.
    """
    return IchimokuStrategy(data_fetcher=data_fetcher, **kwargs)