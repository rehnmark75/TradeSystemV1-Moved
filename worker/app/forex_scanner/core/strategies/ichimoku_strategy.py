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
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.85)  # Ichimoku needs much higher confidence (was 0.55)
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

        # RAG Enhancement modules
        self.rag_enhancer = None
        self.tradingview_parser = None
        self.market_intelligence_adapter = None
        self.confluence_scorer = None
        self.mtf_rag_validator = None
        self.rag_enabled = getattr(config, 'ENABLE_RAG_ENHANCEMENT', True)

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

            # RAG Enhancement Module
            if self.rag_enabled:
                try:
                    from .helpers.ichimoku_rag_enhancer import IchimokuRAGEnhancer
                    self.rag_enhancer = IchimokuRAGEnhancer(logger=self.logger)
                    self.logger.info("🧠 Ichimoku RAG enhancer initialized")
                except ImportError as e:
                    self.logger.warning(f"⚠️ Ichimoku RAG enhancer not available: {e}")
                    self.rag_enhancer = None

                # TradingView Script Parser
                try:
                    from .helpers.tradingview_script_parser import TradingViewScriptParser
                    self.tradingview_parser = TradingViewScriptParser(logger=self.logger)
                    self.logger.info("📊 TradingView script parser initialized")
                except ImportError as e:
                    self.logger.warning(f"⚠️ TradingView script parser not available: {e}")
                    self.tradingview_parser = None

                # Market Intelligence Adapter
                try:
                    from .helpers.ichimoku_market_intelligence_adapter import IchimokuMarketIntelligenceAdapter
                    self.market_intelligence_adapter = IchimokuMarketIntelligenceAdapter(
                        data_fetcher=self.data_fetcher,
                        logger=self.logger
                    )
                    self.logger.info("🧠 Market Intelligence Adapter initialized")
                except ImportError as e:
                    self.logger.warning(f"⚠️ Market Intelligence Adapter not available: {e}")
                    self.market_intelligence_adapter = None

                # Confluence Scorer
                try:
                    from .helpers.ichimoku_confluence_scorer import IchimokuConfluenceScorer
                    self.confluence_scorer = IchimokuConfluenceScorer(
                        rag_enhancer=self.rag_enhancer,
                        tradingview_parser=self.tradingview_parser,
                        logger=self.logger
                    )
                    self.logger.info("🔗 Confluence Scorer initialized")
                except ImportError as e:
                    self.logger.warning(f"⚠️ Confluence Scorer not available: {e}")
                    self.confluence_scorer = None

                # MTF RAG Validator
                try:
                    from .helpers.ichimoku_mtf_rag_validator import IchimokuMTFRAGValidator
                    self.mtf_rag_validator = IchimokuMTFRAGValidator(
                        data_fetcher=self.data_fetcher,
                        rag_enhancer=self.rag_enhancer,
                        tradingview_parser=self.tradingview_parser,
                        logger=self.logger
                    )
                    self.logger.info("🕐 MTF RAG Validator initialized")
                except ImportError as e:
                    self.logger.warning(f"⚠️ MTF RAG Validator not available: {e}")
                    self.mtf_rag_validator = None
            else:
                self.logger.info("🧠 RAG enhancement disabled by configuration")

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
                # Apply RAG enhancement if available
                enhanced_signal = self._apply_rag_enhancement(signal, df_with_signals, epic, timeframe)
                return enhanced_signal
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

    # ==================== RAG ENHANCEMENT METHODS ====================

    def _apply_rag_enhancement(
        self,
        signal: Dict,
        market_data: pd.DataFrame,
        epic: str,
        timeframe: str
    ) -> Dict:
        """
        🧠 APPLY RAG ENHANCEMENT: Enhance Ichimoku signal using RAG data

        Args:
            signal: Original Ichimoku signal
            market_data: Historical market data
            epic: Currency pair
            timeframe: Trading timeframe

        Returns:
            Enhanced signal with RAG improvements
        """
        try:
            if not self.rag_enhancer or not self.rag_enabled:
                self.logger.debug(f"🧠 RAG enhancement disabled for {epic}")
                return signal

            self.logger.info(f"🧠 Applying RAG enhancement to {epic} signal...")

            # Create Ichimoku data package for RAG analysis
            ichimoku_data = {
                'tenkan_sen': signal.get('tenkan_sen', 0),
                'kijun_sen': signal.get('kijun_sen', 0),
                'senkou_span_a': signal.get('senkou_span_a', 0),
                'senkou_span_b': signal.get('senkou_span_b', 0),
                'chikou_span': signal.get('chikou_span', 0),
                'cloud_top': signal.get('cloud_top', 0),
                'cloud_bottom': signal.get('cloud_bottom', 0),
                'tk_bull_cross': signal.get('tk_bull_cross', False),
                'tk_bear_cross': signal.get('tk_bear_cross', False),
                'cloud_bull_breakout': signal.get('cloud_bull_breakout', False),
                'cloud_bear_breakout': signal.get('cloud_bear_breakout', False),
                'signal_type': signal.get('signal_type', 'UNKNOWN'),
                'confidence': signal.get('confidence', 0.5)
            }

            # Apply RAG enhancement
            enhanced_data = self.rag_enhancer.enhance_ichimoku_signal(
                ichimoku_data=ichimoku_data,
                market_data=market_data,
                epic=epic,
                timeframe=timeframe
            )

            # Apply confluence scoring if available
            confluence_data = self._apply_confluence_scoring(
                signal, market_data, epic, timeframe
            )

            # Merge enhanced data with original signal
            enhanced_signal = signal.copy()
            enhanced_signal.update(enhanced_data)
            enhanced_signal.update(confluence_data)

            # Log enhancement results
            original_confidence = signal.get('confidence', 0.5)
            enhanced_confidence = enhanced_signal.get('rag_enhanced_confidence', original_confidence)
            confidence_boost = enhanced_signal.get('confidence_boost', 0)

            self.logger.info(f"🧠 RAG enhancement complete for {epic}: "
                           f"Confidence {original_confidence:.1%} → {enhanced_confidence:.1%} "
                           f"(boost: {confidence_boost:+.1%})")

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"❌ RAG enhancement failed for {epic}: {e}")
            return signal

    def _apply_confluence_scoring(
        self,
        signal: Dict,
        market_data: pd.DataFrame,
        epic: str,
        timeframe: str
    ) -> Dict:
        """
        🔗 APPLY CONFLUENCE SCORING: Enhance signal with confluence analysis

        Args:
            signal: Original Ichimoku signal
            market_data: Historical market data
            epic: Currency pair
            timeframe: Trading timeframe

        Returns:
            Confluence scoring data
        """
        try:
            if not self.confluence_scorer:
                return {'confluence_analysis_available': False}

            self.logger.info(f"🔗 Applying confluence scoring to {epic} signal...")

            # Get market conditions for confluence analysis
            market_conditions = None
            if self.market_intelligence_adapter:
                market_conditions = self.market_intelligence_adapter.analyze_market_conditions(
                    epic=epic,
                    timeframe=timeframe
                )

            # Calculate comprehensive confluence score
            confluence_score = self.confluence_scorer.calculate_confluence_score(
                ichimoku_signal=signal,
                market_data=market_data,
                epic=epic,
                market_conditions=market_conditions
            )

            # Get detailed confluence breakdown
            confluence_breakdown = self.confluence_scorer.get_confluence_breakdown(
                confluence_score, []  # Indicators are internal to scorer
            )

            # Check if confluence is favorable
            signal_type = signal.get('signal_type', 'UNKNOWN')
            confluence_favorable = self.confluence_scorer.is_confluence_favorable(
                confluence_score, signal_type
            )

            # Calculate confluence-adjusted confidence
            original_confidence = signal.get('confidence', 0.5)
            confluence_adjustment = self._calculate_confluence_confidence_adjustment(
                confluence_score, signal_type
            )
            confluence_adjusted_confidence = max(
                0.3, min(0.95, original_confidence + confluence_adjustment)
            )

            # Prepare confluence data
            confluence_data = {
                'confluence_analysis_available': True,
                'confluence_score': confluence_score.total_score,
                'confluence_level': confluence_score.confluence_level,
                'confluence_favorable': confluence_favorable,
                'confluence_bull_score': confluence_score.bull_score,
                'confluence_bear_score': confluence_score.bear_score,
                'confluence_indicator_count': confluence_score.indicator_count,
                'confluence_high_confidence_indicators': confluence_score.high_confidence_indicators,
                'confluence_weighted_strength': confluence_score.weighted_strength,
                'confluence_confidence_adjustment': confluence_adjustment,
                'confluence_adjusted_confidence': confluence_adjusted_confidence,
                'confluence_breakdown': confluence_breakdown
            }

            self.logger.info(f"🔗 Confluence scoring complete for {epic}: "
                           f"Level={confluence_score.confluence_level}, "
                           f"Score={confluence_score.total_score:.3f}, "
                           f"Favorable={confluence_favorable}")

            return confluence_data

        except Exception as e:
            self.logger.error(f"❌ Confluence scoring failed for {epic}: {e}")
            return {'confluence_analysis_available': False, 'confluence_error': str(e)}

    def _calculate_confluence_confidence_adjustment(
        self,
        confluence_score,
        signal_type: str
    ) -> float:
        """Calculate confidence adjustment based on confluence analysis"""
        try:
            adjustment = 0.0

            # Base adjustment from confluence level
            level_adjustments = {
                'VERY_HIGH': 0.12,
                'HIGH': 0.08,
                'MEDIUM': 0.02,
                'LOW': -0.05
            }

            adjustment += level_adjustments.get(confluence_score.confluence_level, 0.0)

            # Directional confluence adjustment
            if signal_type == 'BULL':
                directional_strength = confluence_score.bull_score - confluence_score.bear_score
            elif signal_type == 'BEAR':
                directional_strength = confluence_score.bear_score - confluence_score.bull_score
            else:
                directional_strength = 0.0

            # Add directional adjustment (up to ±0.08)
            adjustment += max(-0.08, min(0.08, directional_strength * 0.15))

            # High confidence indicator bonus
            if confluence_score.high_confidence_indicators >= 3:
                adjustment += 0.03

            # Multiple indicator bonus
            if confluence_score.indicator_count >= 5:
                adjustment += 0.02

            return max(-0.15, min(0.15, adjustment))

        except Exception:
            return 0.0

    def get_rag_market_insights(self, epic: str, market_conditions: Dict = None) -> Dict:
        """
        🧠 GET RAG MARKET INSIGHTS: Query RAG system for market-specific insights

        Args:
            epic: Currency pair
            market_conditions: Current market conditions

        Returns:
            RAG-powered market insights and recommendations
        """
        try:
            if not self.rag_enhancer:
                return {'error': 'RAG enhancer not available'}

            # Get market recommendations from RAG
            recommendations = self.rag_enhancer.get_rag_market_recommendations(
                epic=epic,
                market_conditions=market_conditions or {}
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"RAG market insights failed: {e}")
            return {'error': str(e)}

    def get_tradingview_parameter_recommendations(
        self,
        epic: str,
        market_conditions: Dict = None
    ) -> Dict:
        """
        📊 GET TRADINGVIEW RECOMMENDATIONS: Extract parameter recommendations from TradingView scripts

        Args:
            epic: Currency pair
            market_conditions: Current market conditions

        Returns:
            TradingView-based parameter recommendations
        """
        try:
            if not self.tradingview_parser:
                return {'error': 'TradingView parser not available'}

            # Get enhanced parameters based on TradingView analysis
            enhanced_params = self.tradingview_parser.generate_enhanced_parameters(
                epic=epic,
                market_conditions=market_conditions or {
                    'regime': 'trending',
                    'timeframe': self.timeframe,
                    'volatility': 'medium'
                }
            )

            return enhanced_params

        except Exception as e:
            self.logger.error(f"TradingView parameter recommendations failed: {e}")
            return {'error': str(e)}

    def adapt_parameters_with_rag(self, epic: str, current_market_regime: str) -> Dict:
        """
        🎯 ADAPT PARAMETERS WITH RAG: Dynamically adapt Ichimoku parameters using RAG insights

        Args:
            epic: Currency pair
            current_market_regime: Current market regime (trending, ranging, breakout)

        Returns:
            Adapted parameter configuration
        """
        try:
            adapted_config = self.ichimoku_config.copy()

            # Get TradingView recommendations
            market_conditions = {
                'regime': current_market_regime,
                'timeframe': self.timeframe,
                'volatility': 'medium'  # Could be detected from market data
            }

            tv_recommendations = self.get_tradingview_parameter_recommendations(
                epic=epic,
                market_conditions=market_conditions
            )

            if not tv_recommendations.get('error'):
                # Apply TradingView parameter suggestions
                tv_params = tv_recommendations.get('parameters', {})
                if tv_params:
                    adapted_config.update(tv_params)
                    self.logger.info(f"🎯 Applied TradingView parameters for {epic}: {tv_params}")

                # Apply recommended adjustments
                adjustments = tv_recommendations.get('recommended_adjustments', {})
                if adjustments:
                    confidence_modifier = adjustments.get('confidence_threshold', 0)
                    adapted_config['confidence_boost'] = confidence_modifier
                    self.logger.info(f"🎯 Applied parameter adjustments for {epic}: {adjustments}")

            return adapted_config

        except Exception as e:
            self.logger.error(f"Parameter adaptation failed: {e}")
            return self.ichimoku_config

    def get_rag_confluence_analysis(self, epic: str, signal_data: Dict) -> Dict:
        """
        🔗 GET RAG CONFLUENCE ANALYSIS: Analyze signal confluence using RAG data

        Args:
            epic: Currency pair
            signal_data: Current signal data

        Returns:
            Confluence analysis results
        """
        try:
            if not self.rag_enhancer:
                return {'error': 'RAG enhancer not available'}

            # Use RAG to find confluence indicators
            confluence_query = f"ichimoku {epic} confluence analysis momentum support resistance"

            # Get confluence recommendations
            if hasattr(self.rag_enhancer, 'rag_interface') and self.rag_enhancer.rag_interface:
                confluence_results = self.rag_enhancer.rag_interface.search_indicators(
                    confluence_query, limit=5
                )

                if not confluence_results.get('error'):
                    confluence_score = self._calculate_confluence_from_results(
                        confluence_results, signal_data
                    )

                    return {
                        'confluence_score': confluence_score,
                        'confluence_indicators': len(confluence_results.get('results', [])),
                        'analysis_source': 'rag_search'
                    }

            return {'confluence_score': 0.5, 'analysis_source': 'fallback'}

        except Exception as e:
            self.logger.error(f"RAG confluence analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_confluence_from_results(self, results: Dict, signal_data: Dict) -> float:
        """Calculate confluence score from RAG search results"""
        try:
            search_results = results.get('results', [])
            if not search_results:
                return 0.5

            # Basic confluence calculation based on result count and similarity
            base_score = min(0.3 + (len(search_results) * 0.1), 0.7)

            # Adjust based on signal strength
            signal_type = signal_data.get('signal_type', 'UNKNOWN')
            if signal_type in ['BULL', 'BEAR']:
                base_score += 0.1

            # Check for cross confirmations
            if signal_data.get('tk_bull_cross') or signal_data.get('tk_bear_cross'):
                base_score += 0.1

            if signal_data.get('cloud_bull_breakout') or signal_data.get('cloud_bear_breakout'):
                base_score += 0.1

            return max(0.3, min(0.9, base_score))

        except Exception:
            return 0.5

    def get_rag_system_status(self) -> Dict:
        """
        📊 GET RAG SYSTEM STATUS: Check status of all RAG components

        Returns:
            Status information for RAG system components
        """
        try:
            status = {
                'rag_enabled': self.rag_enabled,
                'rag_enhancer_available': self.rag_enhancer is not None,
                'tradingview_parser_available': self.tradingview_parser is not None,
                'rag_interface_healthy': False,
                'stats': {}
            }

            # Check RAG enhancer status
            if self.rag_enhancer:
                status['rag_interface_healthy'] = self.rag_enhancer.is_rag_available()
                if status['rag_interface_healthy']:
                    status['stats'] = self.rag_enhancer.get_rag_stats()

            # Check TradingView parser status
            if self.tradingview_parser:
                status['tradingview_stats'] = self.tradingview_parser.get_statistics()

            return status

        except Exception as e:
            return {'error': str(e)}

    # ==================== MARKET INTELLIGENCE INTEGRATION ====================

    def apply_adaptive_parameters(self, epic: str, timeframe: str = None) -> Dict:
        """
        🧠 APPLY ADAPTIVE PARAMETERS: Use market intelligence for dynamic parameter adaptation

        Args:
            epic: Currency pair
            timeframe: Optional timeframe override

        Returns:
            Adapted configuration optimized for current market conditions
        """
        try:
            if not self.market_intelligence_adapter:
                return {'error': 'Market intelligence adapter not available'}

            analysis_timeframe = timeframe or self.timeframe

            # Analyze current market conditions
            market_conditions = self.market_intelligence_adapter.analyze_market_conditions(
                epic=epic,
                timeframe=analysis_timeframe
            )

            # Get RAG recommendations if available
            rag_recommendations = None
            if self.tradingview_parser:
                rag_recommendations = self.tradingview_parser.generate_enhanced_parameters(
                    epic=epic,
                    market_conditions=market_conditions
                )

            # Adapt parameters using market intelligence + RAG
            adapted_config = self.market_intelligence_adapter.adapt_ichimoku_parameters(
                base_config=self.ichimoku_config,
                market_conditions=market_conditions,
                rag_recommendations=rag_recommendations
            )

            # Update internal configuration with adaptations
            self._apply_adapted_config(adapted_config)

            # Get adaptation summary
            adaptation_summary = self.market_intelligence_adapter.get_adaptation_summary(
                epic=epic,
                adapted_config=adapted_config
            )

            self.logger.info(f"🧠 Applied adaptive parameters for {epic}: "
                           f"Regime={market_conditions.get('regime', 'unknown')}, "
                           f"Confidence={adapted_config.get('final_confidence_threshold', 0.55):.1%}")

            return {
                'success': True,
                'market_conditions': market_conditions,
                'adapted_config': adapted_config,
                'adaptation_summary': adaptation_summary,
                'rag_integration': rag_recommendations is not None and not rag_recommendations.get('error')
            }

        except Exception as e:
            self.logger.error(f"Adaptive parameter application failed: {e}")
            return {'error': str(e)}

    def _apply_adapted_config(self, adapted_config: Dict):
        """Apply adapted configuration to strategy instance"""
        try:
            # Update confidence threshold
            if 'final_confidence_threshold' in adapted_config:
                self.min_confidence = adapted_config['final_confidence_threshold']

            # Update cloud thickness threshold
            if 'adapted_cloud_thickness_threshold' in adapted_config:
                self.cloud_thickness_threshold = adapted_config['adapted_cloud_thickness_threshold']

            # Update TK cross threshold
            if 'adapted_tk_cross_threshold' in adapted_config:
                self.tk_cross_strength_threshold = adapted_config['adapted_tk_cross_threshold']

            # Update Chikou clear threshold
            if 'adapted_chikou_clear_threshold' in adapted_config:
                self.chikou_clear_threshold = adapted_config['adapted_chikou_clear_threshold']

            # Update RAG-suggested parameters if available
            for param in ['tenkan_period', 'kijun_period', 'senkou_b_period', 'chikou_shift']:
                rag_param = f'rag_suggested_{param}'
                if rag_param in adapted_config:
                    setattr(self, param, adapted_config[rag_param])
                    self.ichimoku_config[param] = adapted_config[rag_param]

            # Store adaptation metadata
            self.current_adaptations = adapted_config

            self.logger.debug(f"🎯 Internal configuration updated with adaptive parameters")

        except Exception as e:
            self.logger.error(f"Adapted configuration application failed: {e}")

    def get_current_market_regime(self, epic: str) -> Dict:
        """
        🔍 GET CURRENT MARKET REGIME: Analyze current market regime for the epic

        Args:
            epic: Currency pair to analyze

        Returns:
            Current market regime analysis
        """
        try:
            if not self.market_intelligence_adapter:
                return {'error': 'Market intelligence adapter not available'}

            market_conditions = self.market_intelligence_adapter.analyze_market_conditions(
                epic=epic,
                timeframe=self.timeframe
            )

            return {
                'regime': market_conditions.get('regime', 'unknown'),
                'confidence': market_conditions.get('regime_confidence', 0.5),
                'volatility': market_conditions.get('volatility_level', 'medium'),
                'session': market_conditions.get('trading_session', 'unknown'),
                'full_analysis': market_conditions
            }

        except Exception as e:
            self.logger.error(f"Market regime analysis failed: {e}")
            return {'error': str(e)}

    def should_adapt_for_market_conditions(self, epic: str, force_check: bool = False) -> bool:
        """
        🤔 SHOULD ADAPT FOR MARKET CONDITIONS: Check if parameters should be adapted

        Args:
            epic: Currency pair
            force_check: Force market condition check

        Returns:
            True if parameters should be adapted
        """
        try:
            if not self.market_intelligence_adapter:
                return False

            # Check if we have recent adaptations
            if hasattr(self, 'current_adaptations') and not force_check:
                adaptation_time = self.current_adaptations.get('adaptation_timestamp')
                if adaptation_time:
                    # Only re-adapt if more than 1 hour has passed
                    from datetime import datetime, timedelta
                    if datetime.now() - datetime.fromisoformat(adaptation_time.replace('Z', '+00:00')) < timedelta(hours=1):
                        return False

            # Get current market regime
            regime_info = self.get_current_market_regime(epic)

            if regime_info.get('error'):
                return False

            # Adapt if regime confidence is high and regime is not trending (default)
            regime = regime_info.get('regime', 'trending')
            confidence = regime_info.get('confidence', 0.5)

            # Adapt for non-trending regimes or high-confidence regime detection
            return regime != 'trending' or confidence > 0.7

        except Exception as e:
            self.logger.error(f"Market condition adaptation check failed: {e}")
            return False

    def get_adaptive_signal_modifiers(self, epic: str, signal_data: Dict) -> Dict:
        """
        🎯 GET ADAPTIVE SIGNAL MODIFIERS: Get signal modifiers based on market conditions

        Args:
            epic: Currency pair
            signal_data: Current signal data

        Returns:
            Signal modifiers based on current market conditions
        """
        try:
            if not hasattr(self, 'current_adaptations'):
                return {}

            adaptations = self.current_adaptations
            modifiers = {}

            # Apply stop loss modifier
            if 'regime_stop_loss_modifier' in adaptations:
                modifiers['stop_loss_modifier'] = adaptations['regime_stop_loss_modifier']

            # Apply take profit modifier
            if 'regime_take_profit_modifier' in adaptations:
                modifiers['take_profit_modifier'] = adaptations['regime_take_profit_modifier']

            # Apply session volatility factor
            if 'session_volatility_factor' in adaptations:
                modifiers['volatility_factor'] = adaptations['session_volatility_factor']

            # Check if additional confirmations are required
            if 'regime_additional_confirmations' in adaptations:
                required_confirmations = adaptations['regime_additional_confirmations']
                if required_confirmations > 0:
                    modifiers['additional_confirmations_required'] = required_confirmations

            # Check MTF requirement
            if 'regime_mtf_required' in adaptations:
                modifiers['mtf_required'] = adaptations['regime_mtf_required']

            # Apply confidence boost from market conditions
            confidence_boost = 0
            if 'regime_confidence' in adaptations:
                regime_confidence = adaptations['regime_confidence']
                if regime_confidence > 0.8:
                    confidence_boost += 0.05  # High confidence regime
                elif regime_confidence < 0.4:
                    confidence_boost -= 0.05  # Low confidence regime

            modifiers['confidence_boost'] = confidence_boost

            return modifiers

        except Exception as e:
            self.logger.error(f"Adaptive signal modifiers failed: {e}")
            return {}

    def get_market_intelligence_status(self) -> Dict:
        """
        📊 GET MARKET INTELLIGENCE STATUS: Get comprehensive status of market intelligence integration

        Returns:
            Status of all market intelligence components
        """
        try:
            status = {
                'market_intelligence_available': self.market_intelligence_adapter is not None,
                'rag_integration': {
                    'rag_enhancer_available': self.rag_enhancer is not None,
                    'tradingview_parser_available': self.tradingview_parser is not None,
                    'rag_enabled': self.rag_enabled
                },
                'current_adaptations': hasattr(self, 'current_adaptations'),
                'adaptive_parameters_active': False
            }

            # Get market intelligence adapter status
            if self.market_intelligence_adapter:
                adapter_status = self.market_intelligence_adapter.get_market_intelligence_status()
                status['adapter_status'] = adapter_status

            # Check if adaptive parameters are currently active
            if hasattr(self, 'current_adaptations'):
                status['adaptive_parameters_active'] = True
                status['current_regime'] = self.current_adaptations.get('applied_regime', 'unknown')
                status['adaptation_timestamp'] = self.current_adaptations.get('adaptation_timestamp')

            # Get RAG system status
            if self.rag_enhancer:
                rag_status = self.rag_enhancer.get_rag_stats()
                status['rag_stats'] = rag_status

            return status

        except Exception as e:
            self.logger.error(f"Market intelligence status check failed: {e}")
            return {'error': str(e)}

    # ==================== MULTI-TIMEFRAME RAG VALIDATION ====================

    def validate_signal_with_mtf_rag(
        self,
        signal: Dict,
        epic: str,
        trading_style: str = 'day_trading'
    ) -> Dict:
        """
        🕐 VALIDATE SIGNAL WITH MTF RAG: Multi-timeframe validation using RAG templates

        Args:
            signal: Ichimoku signal to validate
            epic: Currency pair
            trading_style: Trading style for timeframe selection

        Returns:
            MTF validation results with RAG enhancement
        """
        try:
            if not self.mtf_rag_validator:
                return {'error': 'MTF RAG validator not available'}

            # Get market conditions if available
            market_conditions = None
            if self.market_intelligence_adapter:
                market_conditions = self.market_intelligence_adapter.analyze_market_conditions(
                    epic=epic,
                    timeframe=self.timeframe
                )

            # Perform MTF validation
            validation_result = self.mtf_rag_validator.validate_mtf_signal(
                primary_signal=signal,
                epic=epic,
                primary_timeframe=self.timeframe,
                market_conditions=market_conditions,
                trading_style=trading_style
            )

            # Get validation summary
            validation_summary = self.mtf_rag_validator.get_mtf_summary(validation_result)

            # Calculate MTF confidence adjustment
            mtf_confidence_adjustment = self._calculate_mtf_confidence_adjustment(validation_result)

            self.logger.info(f"🕐 MTF RAG validation for {epic}: "
                           f"Status={validation_summary.get('validation_status', 'UNKNOWN')}, "
                           f"Bias={validation_result.overall_bias}, "
                           f"Confidence={validation_result.confidence_score:.2f}")

            return {
                'mtf_validation_available': True,
                'validation_result': validation_result,
                'validation_summary': validation_summary,
                'mtf_confidence_adjustment': mtf_confidence_adjustment,
                'timeframe_agreement': validation_result.timeframe_agreement_score,
                'rag_template_consensus': validation_result.rag_template_consensus,
                'higher_timeframe_support': validation_result.higher_tf_support,
                'lower_timeframe_confirmation': validation_result.lower_tf_confirmation
            }

        except Exception as e:
            self.logger.error(f"MTF RAG validation failed: {e}")
            return {'error': str(e)}

    def _calculate_mtf_confidence_adjustment(self, validation_result) -> float:
        """Calculate confidence adjustment based on MTF validation"""
        try:
            adjustment = 0.0

            # Base adjustment from validation pass/fail
            if validation_result.validation_passed:
                adjustment += 0.10
            else:
                adjustment -= 0.08

            # Timeframe agreement bonus/penalty
            agreement_score = validation_result.timeframe_agreement_score
            if agreement_score >= 0.8:
                adjustment += 0.05
            elif agreement_score <= 0.3:
                adjustment -= 0.05

            # Higher timeframe support bonus
            if validation_result.higher_tf_support:
                adjustment += 0.06

            # Lower timeframe confirmation bonus
            if validation_result.lower_tf_confirmation:
                adjustment += 0.03

            # RAG template consensus bonus
            if validation_result.rag_template_consensus >= 0.7:
                adjustment += 0.04

            # Conflicting timeframes penalty
            conflicting_count = len(validation_result.conflicting_timeframes)
            if conflicting_count > 1:
                adjustment -= 0.03 * conflicting_count

            return max(-0.20, min(0.20, adjustment))

        except Exception:
            return 0.0

    def should_apply_mtf_validation(self, signal: Dict, epic: str) -> bool:
        """
        🤔 SHOULD APPLY MTF VALIDATION: Check if MTF validation should be applied

        Args:
            signal: Current signal
            epic: Currency pair

        Returns:
            True if MTF validation should be applied
        """
        try:
            if not self.mtf_rag_validator:
                return False

            # Always apply MTF validation for non-backtest mode
            if not self.backtest_mode:
                return True

            # In backtest mode, only apply for higher timeframes
            if self.timeframe in ['4h', '1d', '1w']:
                return True

            return False

        except Exception:
            return False

    def get_mtf_timeframe_recommendations(self, epic: str, trading_style: str = 'day_trading') -> Dict:
        """
        📊 GET MTF TIMEFRAME RECOMMENDATIONS: Get RAG-based timeframe recommendations

        Args:
            epic: Currency pair
            trading_style: Trading style

        Returns:
            Timeframe recommendations from RAG analysis
        """
        try:
            if not self.mtf_rag_validator:
                return {'error': 'MTF RAG validator not available'}

            # Get supported combinations
            combinations = self.mtf_rag_validator.get_supported_timeframe_combinations()

            # Get market conditions for recommendations
            market_conditions = None
            if self.market_intelligence_adapter:
                market_conditions = self.market_intelligence_adapter.analyze_market_conditions(
                    epic=epic,
                    timeframe=self.timeframe
                )

            # Get recommendations based on current conditions
            recommendations = {
                'current_timeframe': self.timeframe,
                'trading_style': trading_style,
                'recommended_combinations': combinations['combinations'].get(trading_style, []),
                'market_conditions': market_conditions,
                'all_combinations': combinations['combinations'],
                'timeframe_hierarchy': combinations['hierarchy'],
                'regime_weights': combinations['regime_weights']
            }

            return recommendations

        except Exception as e:
            self.logger.error(f"MTF timeframe recommendations failed: {e}")
            return {'error': str(e)}

    def analyze_timeframe_correlation(self, epic: str, target_timeframes: List[str] = None) -> Dict:
        """
        🔗 ANALYZE TIMEFRAME CORRELATION: Analyze Ichimoku correlation across timeframes

        Args:
            epic: Currency pair
            target_timeframes: Specific timeframes to analyze

        Returns:
            Timeframe correlation analysis
        """
        try:
            if not self.mtf_rag_validator:
                return {'error': 'MTF RAG validator not available'}

            # Use default timeframes if none specified
            if not target_timeframes:
                target_timeframes = ['4h', '1h', '15m']

            # Create a mock signal for correlation analysis
            mock_signal = {
                'signal_type': 'BULL',  # Test with bull signal
                'confidence': 0.7,
                'timeframe': self.timeframe
            }

            # Get market conditions
            market_conditions = None
            if self.market_intelligence_adapter:
                market_conditions = self.market_intelligence_adapter.analyze_market_conditions(
                    epic=epic,
                    timeframe=self.timeframe
                )

            # Analyze correlation (simplified - would need actual data in production)
            correlation_analysis = {
                'primary_timeframe': self.timeframe,
                'analyzed_timeframes': target_timeframes,
                'market_conditions': market_conditions,
                'correlation_summary': {
                    'strong_correlation': [],
                    'moderate_correlation': [],
                    'weak_correlation': [],
                    'conflicting_signals': []
                },
                'rag_template_insights': self._get_rag_timeframe_insights(target_timeframes, market_conditions)
            }

            # Simulate correlation results (would be based on actual data)
            for tf in target_timeframes:
                # This would be calculated from actual timeframe analysis
                correlation_strength = 0.7  # Placeholder
                if correlation_strength >= 0.8:
                    correlation_analysis['correlation_summary']['strong_correlation'].append(tf)
                elif correlation_strength >= 0.6:
                    correlation_analysis['correlation_summary']['moderate_correlation'].append(tf)
                elif correlation_strength >= 0.4:
                    correlation_analysis['correlation_summary']['weak_correlation'].append(tf)
                else:
                    correlation_analysis['correlation_summary']['conflicting_signals'].append(tf)

            return correlation_analysis

        except Exception as e:
            self.logger.error(f"Timeframe correlation analysis failed: {e}")
            return {'error': str(e)}

    def _get_rag_timeframe_insights(self, timeframes: List[str], market_conditions: Dict = None) -> Dict:
        """Get RAG insights for specific timeframes"""
        try:
            if not self.rag_enhancer or not self.rag_enhancer.rag_interface:
                return {'insights_available': False}

            insights = {}

            for tf in timeframes:
                try:
                    # Create timeframe-specific query
                    regime = market_conditions.get('regime', 'trending') if market_conditions else 'trending'
                    query = f"ichimoku {tf} timeframe {regime} market strategy insights"

                    # Search for insights
                    search_results = self.rag_enhancer.rag_interface.search_indicators(query, limit=2)

                    if not search_results.get('error'):
                        results = search_results.get('results', [])
                        insights[tf] = {
                            'insights_count': len(results),
                            'top_insights': [result.get('title', 'Unknown') for result in results[:2]]
                        }

                except Exception as e:
                    self.logger.warning(f"RAG insights failed for {tf}: {e}")
                    insights[tf] = {'insights_available': False}

            return {
                'insights_available': True,
                'timeframe_insights': insights
            }

        except Exception:
            return {'insights_available': False}

    def get_enhanced_mtf_status(self) -> Dict:
        """
        📊 GET ENHANCED MTF STATUS: Get comprehensive MTF and RAG integration status

        Returns:
            Complete status of MTF RAG validation system
        """
        try:
            status = {
                'mtf_rag_available': self.mtf_rag_validator is not None,
                'mtf_validation_enabled': self.enable_mtf_analysis,
                'current_timeframe': self.timeframe,
                'backtest_mode': self.backtest_mode,
                'component_status': {
                    'rag_enhancer': self.rag_enhancer is not None,
                    'tradingview_parser': self.tradingview_parser is not None,
                    'market_intelligence_adapter': self.market_intelligence_adapter is not None,
                    'confluence_scorer': self.confluence_scorer is not None,
                    'mtf_rag_validator': self.mtf_rag_validator is not None
                }
            }

            # Get MTF validator capabilities
            if self.mtf_rag_validator:
                status['mtf_capabilities'] = {
                    'validation_available': self.mtf_rag_validator.is_mtf_validation_available(),
                    'supported_combinations': self.mtf_rag_validator.get_supported_timeframe_combinations()
                }

            # Get integration health
            healthy_components = sum(1 for available in status['component_status'].values() if available)
            total_components = len(status['component_status'])
            status['integration_health'] = {
                'healthy_components': healthy_components,
                'total_components': total_components,
                'health_percentage': (healthy_components / total_components) * 100,
                'status': 'HEALTHY' if healthy_components >= 4 else 'PARTIAL' if healthy_components >= 2 else 'LIMITED'
            }

            return status

        except Exception as e:
            self.logger.error(f"Enhanced MTF status check failed: {e}")
            return {'error': str(e)}


def create_ichimoku_strategy(data_fetcher=None, **kwargs) -> IchimokuStrategy:
    """
    🏭 FACTORY FUNCTION: Create Ichimoku strategy instance

    Simple factory function for backward compatibility with existing code.
    """
    return IchimokuStrategy(data_fetcher=data_fetcher, **kwargs)