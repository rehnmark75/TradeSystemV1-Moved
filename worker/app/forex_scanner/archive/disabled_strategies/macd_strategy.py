# core/strategies/macd_strategy.py
"""
MACD Confluence Strategy - Built from Scratch
🎯 Multi-timeframe confluence-based entry system

Strategy Components:
1. H4 MACD Trend Filter: Only trade with higher timeframe momentum
2. H1 Fibonacci Zones: Calculate retracement levels from swing points
3. Confluence Analysis: Score zones based on multiple factor alignment
4. 15M Pattern Entry: Candlestick patterns at confluence zones
5. Structure-Based Stops: Tighter 15M swing-based stop losses

Entry Requirements:
- H4 MACD trending in signal direction
- Price at H1 Fibonacci confluence zone (50%, 61.8% + swing/EMA/round number)
- Valid candlestick pattern on 15M (engulfing, pin bar)
- Moderate confluence: Fib + 1 other factor minimum
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster

# Confluence components
from .helpers.macd_fibonacci_calculator import FibonacciCalculator
from .helpers.macd_pattern_detector import CandlestickPatternDetector
from .helpers.macd_confluence_analyzer import ConfluenceZoneAnalyzer
from .helpers.macd_mtf_confluence_filter import MACDMultiTimeframeFilter
from .helpers.smc_market_structure import SMCMarketStructure

try:
    from configdata import config
    from configdata.strategies import config_macd_strategy
except ImportError:
    from forex_scanner.configdata import config
    try:
        from forex_scanner.configdata.strategies import config_macd_strategy
    except ImportError:
        config_macd_strategy = None


class MACDStrategy(BaseStrategy):
    """
    MACD Confluence Strategy - Clean Implementation

    Uses Fibonacci retracements + price action patterns for high-probability entries
    while filtering with H4 MACD trend direction.
    """

    def __init__(self,
                 data_fetcher=None,
                 backtest_mode: bool = False,
                 epic: str = None,
                 timeframe: str = '15m',
                 **kwargs):
        """
        Initialize MACD Confluence Strategy.

        Args:
            data_fetcher: Data fetcher for multi-timeframe analysis
            backtest_mode: Whether running in backtest mode
            epic: Currency pair epic code
            timeframe: Primary timeframe for entries (IGNORED - always uses 1h)
        """
        # Basic initialization
        self.name = 'macd'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(logging.INFO)

        # 🔒 HARD-CODED: MACD strategy ALWAYS uses 1H timeframe
        # This overrides whatever the scanner is configured to use
        timeframe = '1h'

        # Store parameters
        self.backtest_mode = backtest_mode
        self.epic = epic
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher
        self.price_adjuster = PriceAdjuster()

        self.logger.info(f"🔒 MACD strategy locked to 1H timeframe (ignoring scanner config)")

        # MACD parameters (standard 12, 26, 9)
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9

        # Minimum confidence threshold
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.60)

        self.logger.info(f"🎯 MACD Confluence Strategy - {epic or 'All Pairs'}")

        # Load pair-specific settings
        self._load_pair_settings()

        # Initialize confluence components
        self._init_confluence_components()

        # Initialize stop loss/take profit settings
        self._init_risk_management()

        self.logger.info("✅ MACD Confluence Strategy initialized successfully")

    def _load_pair_settings(self):
        """Load pair-specific configuration settings"""
        self.pair_settings = {}

        if config_macd_strategy and hasattr(config_macd_strategy, 'MACD_CONFLUENCE_PAIR_SETTINGS'):
            all_settings = config_macd_strategy.MACD_CONFLUENCE_PAIR_SETTINGS

            if self.epic:
                # Find matching pair settings
                for pair_name, settings in all_settings.items():
                    if pair_name in self.epic:
                        self.pair_settings = settings
                        self.logger.info(f"📊 Loaded pair-specific settings for {pair_name}")
                        break

        # Load general settings with pair-specific overrides
        self.fib_lookback = self.pair_settings.get('fib_lookback') or \
                           getattr(config_macd_strategy, 'MACD_CONFLUENCE_FIB_LOOKBACK', 50) if config_macd_strategy else 50

        self.swing_strength = getattr(config_macd_strategy, 'MACD_CONFLUENCE_FIB_SWING_STRENGTH', 5) if config_macd_strategy else 5

        self.min_swing_pips = self.pair_settings.get('min_swing_pips') or \
                             getattr(config_macd_strategy, 'MACD_CONFLUENCE_MIN_SWING_PIPS', 15.0) if config_macd_strategy else 15.0

        self.confluence_mode = self.pair_settings.get('confluence_mode') or \
                              getattr(config_macd_strategy, 'MACD_CONFLUENCE_MODE', 'moderate') if config_macd_strategy else 'moderate'

        self.logger.info(f"⚙️  Settings: Fib lookback={self.fib_lookback}, "
                        f"Min swing={self.min_swing_pips} pips, Mode={self.confluence_mode}")

    def _init_confluence_components(self):
        """Initialize all confluence analysis components"""

        # Fibonacci Calculator
        self.fib_calculator = FibonacciCalculator(
            lookback_bars=self.fib_lookback,
            swing_strength=self.swing_strength,
            min_swing_size_pips=self.min_swing_pips,
            logger=self.logger
        )

        # Candlestick Pattern Detector
        self.pattern_detector = CandlestickPatternDetector(
            min_body_ratio=getattr(config_macd_strategy, 'MACD_PATTERN_MIN_BODY_RATIO', 0.6) if config_macd_strategy else 0.6,
            min_engulf_ratio=getattr(config_macd_strategy, 'MACD_PATTERN_MIN_ENGULF_RATIO', 1.1) if config_macd_strategy else 1.1,
            max_pin_body_ratio=getattr(config_macd_strategy, 'MACD_PATTERN_MAX_PIN_BODY_RATIO', 0.3) if config_macd_strategy else 0.3,
            min_pin_wick_ratio=getattr(config_macd_strategy, 'MACD_PATTERN_MIN_PIN_WICK_RATIO', 2.0) if config_macd_strategy else 2.0,
            logger=self.logger
        )

        # Confluence Zone Analyzer
        self.confluence_analyzer = ConfluenceZoneAnalyzer(
            confluence_mode=self.confluence_mode,
            proximity_tolerance_pips=getattr(config_macd_strategy, 'MACD_CONFLUENCE_PROXIMITY_PIPS', 5.0) if config_macd_strategy else 5.0,
            min_confluence_score=getattr(config_macd_strategy, 'MACD_CONFLUENCE_MIN_SCORE', 2.0) if config_macd_strategy else 2.0,
            logger=self.logger
        )

        # Multi-Timeframe MACD Filter
        self.mtf_filter = MACDMultiTimeframeFilter(
            data_fetcher=self.data_fetcher,
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period,
            require_histogram_expansion=getattr(config_macd_strategy, 'MACD_CONFLUENCE_H4_REQUIRE_EXPANSION', True) if config_macd_strategy else True,
            min_histogram_value=getattr(config_macd_strategy, 'MACD_CONFLUENCE_H4_MIN_HISTOGRAM', 0.00001) if config_macd_strategy else 0.00001,
            logger=self.logger
        )

        # Configuration flags
        self.h4_filter_enabled = getattr(config_macd_strategy, 'MACD_CONFLUENCE_H4_FILTER_ENABLED', True) if config_macd_strategy else True
        self.require_pattern = getattr(config_macd_strategy, 'MACD_CONFLUENCE_REQUIRE_PATTERN', True) if config_macd_strategy else True
        self.min_pattern_quality = getattr(config_macd_strategy, 'MACD_CONFLUENCE_MIN_PATTERN_QUALITY', 60) if config_macd_strategy else 60

        # Multi-Timeframe MACD Alignment & Market Structure Tracking
        self.mtf_macd_alignment_enabled = getattr(config_macd_strategy, 'MACD_MTF_MACD_ALIGNMENT_ENABLED', True) if config_macd_strategy else True
        self.mtf_alignment_boost = getattr(config_macd_strategy, 'MACD_MTF_ALIGNMENT_CONFIDENCE_BOOST', 0.10) if config_macd_strategy else 0.10
        self.mtf_require_alignment = getattr(config_macd_strategy, 'MACD_MTF_REQUIRE_ALIGNMENT', False) if config_macd_strategy else False
        self.log_mtf_alignment = getattr(config_macd_strategy, 'MACD_LOG_MTF_ALIGNMENT', True) if config_macd_strategy else True

        # 34 EMA Trend Filter (1H)
        self.ema_filter_enabled = getattr(config_macd_strategy, 'MACD_EMA_FILTER_ENABLED', True) if config_macd_strategy else True
        self.ema_filter_period = getattr(config_macd_strategy, 'MACD_EMA_FILTER_PERIOD', 34) if config_macd_strategy else 34
        self.ema_require_alignment = getattr(config_macd_strategy, 'MACD_EMA_REQUIRE_ALIGNMENT', True) if config_macd_strategy else True

        # Price Extreme Filter (Prevent buying tops / selling bottoms)
        self.price_extreme_filter_enabled = getattr(config_macd_strategy, 'MACD_PRICE_EXTREME_FILTER_ENABLED', False) if config_macd_strategy else False
        self.price_extreme_lookback = getattr(config_macd_strategy, 'MACD_PRICE_EXTREME_LOOKBACK', 200) if config_macd_strategy else 200
        self.price_extreme_threshold = getattr(config_macd_strategy, 'MACD_PRICE_EXTREME_THRESHOLD', 90) if config_macd_strategy else 90

        # Price Structure Validation (Hybrid Approach - MACD + Structure)
        self.structure_validation_enabled = getattr(config_macd_strategy, 'MACD_PRICE_STRUCTURE_VALIDATION_ENABLED', True) if config_macd_strategy else True
        self.structure_lookback = getattr(config_macd_strategy, 'MACD_STRUCTURE_LOOKBACK', 30) if config_macd_strategy else 30
        self.structure_swing_strength = getattr(config_macd_strategy, 'MACD_STRUCTURE_SWING_STRENGTH', 3) if config_macd_strategy else 3
        self.structure_min_swings = getattr(config_macd_strategy, 'MACD_STRUCTURE_MIN_SWINGS', 2) if config_macd_strategy else 2

        # H4 Market Structure Alignment (REQUIRED - we don't trade against market structure)
        self.h4_structure_alignment_enabled = getattr(config_macd_strategy, 'MACD_H4_STRUCTURE_ALIGNMENT_ENABLED', True) if config_macd_strategy else True
        self.h4_require_structure_alignment = getattr(config_macd_strategy, 'MACD_H4_REQUIRE_STRUCTURE_ALIGNMENT', True) if config_macd_strategy else True

        if self.h4_structure_alignment_enabled:
            self.structure_analyzer = SMCMarketStructure(
                logger=self.logger,
                data_fetcher=self.data_fetcher
            )

            # Load structure configuration
            self.h4_structure_config = getattr(config_macd_strategy, 'MACD_H4_STRUCTURE_CONFIG', {
                'swing_length': 5,
                'structure_confirmation': 3,
                'min_structure_significance': 0.5
            }) if config_macd_strategy else {
                'swing_length': 5,
                'structure_confirmation': 3,
                'min_structure_significance': 0.5
            }

            self.h4_structure_lookback = getattr(config_macd_strategy, 'MACD_H4_STRUCTURE_LOOKBACK_BARS', 50) if config_macd_strategy else 50
            self.h4_log_structure = getattr(config_macd_strategy, 'MACD_H4_LOG_STRUCTURE_ANALYSIS', True) if config_macd_strategy else True

            alignment_mode = "BLOCKING" if self.h4_require_structure_alignment else "ADVISORY"
            self.logger.info(f"🏗️  H4 Structure alignment enabled ({alignment_mode} mode)")
        else:
            self.structure_analyzer = None
            self.logger.info(f"🏗️  H4 Structure alignment disabled")

        if self.mtf_macd_alignment_enabled:
            self.logger.info(f"📊 Multi-timeframe MACD alignment check enabled (+{self.mtf_alignment_boost*100:.0f}% boost)")

        self.logger.info(f"🔧 Confluence components: H4 filter={self.h4_filter_enabled}, "
                        f"Require pattern={self.require_pattern}, Min quality={self.min_pattern_quality}")

    def _init_risk_management(self):
        """Initialize stop loss and take profit settings"""
        self.use_15m_stops = getattr(config_macd_strategy, 'MACD_CONFLUENCE_USE_15M_STOPS', True) if config_macd_strategy else True
        self.stop_atr_multiplier = getattr(config_macd_strategy, 'MACD_CONFLUENCE_STOP_ATR_MULTIPLIER', 1.5) if config_macd_strategy else 1.5
        self.tp_atr_multiplier = getattr(config_macd_strategy, 'MACD_CONFLUENCE_TP_ATR_MULTIPLIER', 3.0) if config_macd_strategy else 3.0
        self.min_stop_pips = getattr(config_macd_strategy, 'MACD_CONFLUENCE_MIN_STOP_PIPS', 10.0) if config_macd_strategy else 10.0
        self.max_stop_pips = getattr(config_macd_strategy, 'MACD_CONFLUENCE_MAX_STOP_PIPS', 30.0) if config_macd_strategy else 30.0
        self.min_rr_ratio = getattr(config_macd_strategy, 'MACD_CONFLUENCE_MIN_RR_RATIO', 2.0) if config_macd_strategy else 2.0
        self.use_structure_targets = getattr(config_macd_strategy, 'MACD_CONFLUENCE_USE_STRUCTURE_TARGETS', True) if config_macd_strategy else True

        # Load histogram thresholds for signal quality filtering
        self.histogram_thresholds = getattr(config_macd_strategy, 'MACD_MIN_HISTOGRAM_THRESHOLDS', {
            'default': {'histogram': 0.00005, 'min_adx': 15}
        }) if config_macd_strategy else {'default': {'histogram': 0.00005, 'min_adx': 15}}

        self.logger.info(f"💰 Risk: {self.stop_atr_multiplier}x ATR SL, {self.tp_atr_multiplier}x ATR TP, "
                        f"Min {self.min_stop_pips}-{self.max_stop_pips} pips, Min R:R {self.min_rr_ratio}:1")

    def _calculate_ema(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """Calculate EMA for given period"""
        return df[column].ewm(span=period, adjust=False).mean()

    def _get_current_emas(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get current EMA values for confluence analysis"""
        if len(df) < 50:
            return {}

        ema_21 = self._calculate_ema(df, 21).iloc[-1] if len(df) >= 21 else None
        ema_50 = self._calculate_ema(df, 50).iloc[-1] if len(df) >= 50 else None

        return {
            'ema_21': ema_21,
            'ema_50': ema_50
        }

    def _find_swing_levels(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[List[float], List[float]]:
        """
        Find recent swing highs and lows for confluence analysis.

        Args:
            df: DataFrame with OHLC data
            lookback: Bars to look back

        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        if len(df) < lookback:
            return [], []

        recent_data = df.tail(lookback)

        # Simple swing detection: local highs/lows
        swing_highs = []
        swing_lows = []

        for i in range(self.swing_strength, len(recent_data) - self.swing_strength):
            # Check for swing high
            if recent_data['high'].iloc[i] == recent_data['high'].iloc[i - self.swing_strength:i + self.swing_strength + 1].max():
                swing_highs.append(recent_data['high'].iloc[i])

            # Check for swing low
            if recent_data['low'].iloc[i] == recent_data['low'].iloc[i - self.swing_strength:i + self.swing_strength + 1].min():
                swing_lows.append(recent_data['low'].iloc[i])

        return swing_highs, swing_lows

    def _validate_price_structure(self, df: pd.DataFrame, signal_direction: str, epic: str) -> bool:
        """
        Validate that price structure confirms the signal direction.

        For BULL signals: Requires higher lows (uptrend structure)
        For BEAR signals: Requires lower highs (downtrend structure)

        Args:
            df: DataFrame with OHLC data
            signal_direction: 'BULL' or 'BEAR'
            epic: Currency pair epic

        Returns:
            True if structure confirms signal, False otherwise
        """
        try:
            # Get lookback data
            lookback_bars = min(self.structure_lookback, len(df))
            if lookback_bars < self.structure_swing_strength * 2 + 5:
                self.logger.info(f"   ⚠️ Insufficient data for structure check ({lookback_bars} bars) - skipping")
                return True  # Don't reject if not enough data

            recent_data = df.tail(lookback_bars).copy()
            recent_data.reset_index(drop=True, inplace=True)

            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            swing_high_indices = []
            swing_low_indices = []

            for i in range(self.structure_swing_strength, len(recent_data) - self.structure_swing_strength):
                # Check for swing high
                window_high = recent_data['high'].iloc[i - self.structure_swing_strength:i + self.structure_swing_strength + 1]
                if recent_data['high'].iloc[i] == window_high.max():
                    swing_highs.append(recent_data['high'].iloc[i])
                    swing_high_indices.append(i)

                # Check for swing low
                window_low = recent_data['low'].iloc[i - self.structure_swing_strength:i + self.structure_swing_strength + 1]
                if recent_data['low'].iloc[i] == window_low.min():
                    swing_lows.append(recent_data['low'].iloc[i])
                    swing_low_indices.append(i)

            # Get pip value for logging
            pair = epic.split('.')[2] if '.' in epic else epic
            pip_value = 0.01 if 'JPY' in pair else 0.0001

            if signal_direction == 'BULL':
                # For BULL signals, need higher lows (uptrend)
                if len(swing_lows) < self.structure_min_swings:
                    self.logger.info(f"   ⚠️ Not enough swing lows found ({len(swing_lows)} < {self.structure_min_swings}) - allowing signal")
                    return True

                # Check if recent swing lows are making higher lows
                last_two_lows = swing_lows[-2:]
                last_two_low_indices = swing_low_indices[-2:]

                if last_two_lows[-1] > last_two_lows[-2]:
                    # Higher low confirmed
                    diff_pips = (last_two_lows[-1] - last_two_lows[-2]) / pip_value
                    bars_apart = last_two_low_indices[-1] - last_two_low_indices[-2]
                    self.logger.info(f"   ✅ Higher low detected: {last_two_lows[-2]:.5f} → {last_two_lows[-1]:.5f} (+{diff_pips:.1f} pips, {bars_apart} bars apart)")
                    return True
                else:
                    # Lower low - bearish structure
                    diff_pips = (last_two_lows[-2] - last_two_lows[-1]) / pip_value
                    self.logger.info(f"   ❌ Lower low detected: {last_two_lows[-2]:.5f} → {last_two_lows[-1]:.5f} (-{diff_pips:.1f} pips)")
                    self.logger.info(f"   🚫 Bearish structure conflicts with BULL signal")
                    return False

            else:  # BEAR
                # For BEAR signals, need lower highs (downtrend)
                if len(swing_highs) < self.structure_min_swings:
                    self.logger.info(f"   ⚠️ Not enough swing highs found ({len(swing_highs)} < {self.structure_min_swings}) - allowing signal")
                    return True

                # Check if recent swing highs are making lower highs
                last_two_highs = swing_highs[-2:]
                last_two_high_indices = swing_high_indices[-2:]

                if last_two_highs[-1] < last_two_highs[-2]:
                    # Lower high confirmed
                    diff_pips = (last_two_highs[-2] - last_two_highs[-1]) / pip_value
                    bars_apart = last_two_high_indices[-1] - last_two_high_indices[-2]
                    self.logger.info(f"   ✅ Lower high detected: {last_two_highs[-2]:.5f} → {last_two_highs[-1]:.5f} (-{diff_pips:.1f} pips, {bars_apart} bars apart)")
                    return True
                else:
                    # Higher high - bullish structure
                    diff_pips = (last_two_highs[-1] - last_two_highs[-2]) / pip_value
                    self.logger.info(f"   ❌ Higher high detected: {last_two_highs[-2]:.5f} → {last_two_highs[-1]:.5f} (+{diff_pips:.1f} pips)")
                    self.logger.info(f"   🚫 Bullish structure conflicts with BEAR signal")
                    return False

        except Exception as e:
            self.logger.error(f"Error validating price structure: {e}", exc_info=True)
            return True  # Don't reject on error

    def detect_signal_auto(self,
                          df: pd.DataFrame,
                          epic: str = None,
                          spread_pips: float = 1.5,
                          timeframe: str = '1h',
                          **kwargs) -> Optional[Dict]:
        """
        Auto signal detection wrapper for compatibility with backtest signal detector.
        Delegates to detect_signal method.

        🔒 HARD-CODED: Always forces 1H timeframe regardless of input.

        Args:
            df: OHLC DataFrame with indicators
            epic: Currency pair epic
            spread_pips: Current spread in pips
            timeframe: Timeframe (IGNORED - always uses 1h)
            **kwargs: Additional args (intelligence_data, regime_data, etc.)

        Returns:
            Signal dict or None
        """
        # 🔒 Force 1H timeframe - ignore whatever the scanner passed
        return self.detect_signal(
            df=df,
            epic=epic,
            spread_pips=spread_pips,
            timeframe='1h',  # Hard-coded 1H
            intelligence_data=kwargs.get('intelligence_data'),
            regime_data=kwargs.get('regime_data')
        )

    def detect_signal(self,
                     df: pd.DataFrame,
                     epic: str,
                     spread_pips: float = 1.5,
                     timeframe: str = None,
                     intelligence_data: Dict = None,
                     regime_data: Dict = None) -> Optional[Dict]:
        """
        Detect MACD crossover signals with H4 trend filter.

        SIMPLE STRATEGY:
        1. Check H4 MACD trend (bullish/bearish)
        2. Detect MACD crossover on strategy timeframe
        3. Only take signals in direction of H4 trend

        Args:
            df: Strategy timeframe OHLC DataFrame with MACD indicators
            epic: Currency pair epic
            spread_pips: Current spread in pips
            intelligence_data: Optional market intelligence data
            regime_data: Optional regime data

        Returns:
            Signal dict or None
        """
        try:
            if len(df) < 50:
                self.logger.debug(f"Insufficient data: {len(df)} bars (need 50+)")
                return None

            current_price = df['close'].iloc[-1]
            current_time = df.index[-1] if hasattr(df.index[-1], 'strftime') else None

            # DEBUG: Show data range being analyzed
            if len(df) > 0:
                # Try to get timestamps from index or columns
                first_time = 'unknown'
                last_time = 'unknown'

                if hasattr(df.index[0], 'strftime'):
                    first_time = df.index[0]
                    last_time = df.index[-1]
                elif 'start_time' in df.columns:
                    first_time = df['start_time'].iloc[0]
                    last_time = df['start_time'].iloc[-1]
                elif 'timestamp' in df.columns:
                    first_time = df['timestamp'].iloc[0]
                    last_time = df['timestamp'].iloc[-1]

                self.logger.info(f"📅 Data range: {first_time} to {last_time} ({len(df)} bars)")

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔍 MACD Simple Crossover - {epic} @ {current_price:.5f}")
            self.logger.info(f"{'='*60}")

            # STEP 1: H4 MACD Trend Filter
            self.logger.info("📊 Step 1: Checking H4 MACD trend filter...")

            h4_data = self.mtf_filter.get_h4_trend_direction(epic, current_time)

            if not h4_data:
                self.logger.debug("No H4 data available - normal at start of backtests")
                return None

            h4_trend = h4_data['trend']
            self.logger.info(f"   H4 Trend: {h4_trend.upper()}")

            if h4_trend == 'neutral':
                self.logger.info("   H4 neutral - waiting for clear trend")
                return None

            # STEP 2: Detect MACD Crossover on Strategy Timeframe
            self.logger.info("📊 Step 2: Detecting MACD crossover...")

            # Check if MACD data exists in DataFrame
            if 'macd_line' not in df.columns or 'macd_signal' not in df.columns:
                self.logger.error("MACD indicators not found in DataFrame")
                return None

            # 🔒 LIVE TRADING FIX: Check crossover on CLOSED bars only
            # In live mode, iloc[-1] is the INCOMPLETE current bar (can change mid-bar)
            # We check crossover between iloc[-2] (last closed) and iloc[-3] (previous closed)
            # Then validate the crossover is still valid on the current bar

            # Get values from last 3 bars
            macd_current = df['macd_line'].iloc[-1]      # Current incomplete bar (live) or last bar (backtest)
            signal_current = df['macd_signal'].iloc[-1]
            macd_closed = df['macd_line'].iloc[-2]       # Last CLOSED bar
            signal_closed = df['macd_signal'].iloc[-2]
            macd_prev = df['macd_line'].iloc[-3]         # Previous closed bar
            signal_prev = df['macd_signal'].iloc[-3]

            # Detect crossover on CLOSED bars (iloc[-3] to iloc[-2])
            bullish_cross = (macd_prev <= signal_prev) and (macd_closed > signal_closed)
            bearish_cross = (macd_prev >= signal_prev) and (macd_closed < signal_closed)

            if not bullish_cross and not bearish_cross:
                self.logger.info("   No MACD crossover detected on closed bars")
                return None

            self.logger.info(f"   Crossover detected between bars: iloc[-3] to iloc[-2]")
            self.logger.info(f"   MACD closed: {macd_closed:.6f}, Signal closed: {signal_closed:.6f}")

            # 🔥 CRITICAL FIX: Check histogram direction to avoid false signals
            # We use the CLOSED bar's histogram, not the current incomplete bar
            # Bullish crossover in negative territory = still bearish (bearish bounce)
            # Bearish crossover in positive territory = still bullish (bullish pullback)
            histogram_closed = macd_closed - signal_closed
            histogram_current = macd_current - signal_current

            # Determine signal direction based on crossover AND histogram direction
            if bullish_cross:
                if histogram_closed > 0:
                    signal_direction = 'BULL'
                    self.logger.info(f"   ✅ Bullish MACD crossover detected (histogram closed: {histogram_closed:.6f} > 0)")
                else:
                    # Bullish cross but histogram still negative = bearish bounce, not bullish signal
                    self.logger.info(f"   ❌ Bullish crossover rejected - histogram closed still negative ({histogram_closed:.6f})")
                    return None
            else:  # bearish_cross
                if histogram_closed < 0:
                    signal_direction = 'BEAR'
                    self.logger.info(f"   ✅ Bearish MACD crossover detected (histogram closed: {histogram_closed:.6f} < 0)")
                else:
                    # Bearish cross but histogram still positive = bullish pullback, not bearish signal
                    self.logger.info(f"   ❌ Bearish crossover rejected - histogram closed still positive ({histogram_closed:.6f})")
                    return None

            # STEP 2.5: Validate Histogram Size (with 3-bar expansion window)
            self.logger.info("📊 Step 2.5: Checking histogram strength...")

            # Get pair-specific threshold from config
            pair = epic.split('.')[2] if '.' in epic else epic
            threshold_config = self.histogram_thresholds.get(pair, self.histogram_thresholds.get('default', {'histogram': 0.00005}))

            # Handle both dict and float formats
            if isinstance(threshold_config, dict):
                min_histogram = threshold_config.get('histogram', 0.00005)
            else:
                min_histogram = threshold_config

            # 🔥 IMPROVED: Check CURRENT histogram strength (not just 3-bar MAX)
            # Previous logic allowed weak entries when MAX was strong but current is weak
            # New logic: Current histogram must meet threshold AND show momentum expansion
            histogram_column = 'macd_histogram' if 'macd_histogram' in df.columns else None

            if histogram_column:
                # Get last 3 histogram values
                recent_histograms = df[histogram_column].iloc[-3:].values
            else:
                # Calculate from MACD line and signal
                recent_histograms = []
                for i in range(-3, 0):
                    hist = df['macd_line'].iloc[i] - df['macd_signal'].iloc[i]
                    recent_histograms.append(hist)

            current_histogram = recent_histograms[-1]
            previous_histogram = recent_histograms[-2] if len(recent_histograms) >= 2 else 0

            self.logger.info(f"   Current histogram: {current_histogram:.6f} (abs: {abs(current_histogram):.6f})")
            self.logger.info(f"   Previous histogram: {previous_histogram:.6f} (abs: {abs(previous_histogram):.6f})")
            self.logger.info(f"   Min threshold for {pair}: {min_histogram:.6f}")

            # Check 1: Current histogram must meet minimum threshold
            current_histogram_abs = abs(current_histogram)
            if current_histogram_abs < min_histogram:
                self.logger.info(f"   ❌ Current histogram too weak: {current_histogram_abs:.6f} < {min_histogram:.6f}")
                return None

            # Check 2: Momentum must be expanding (current > previous)
            previous_histogram_abs = abs(previous_histogram)
            momentum_expanding = current_histogram_abs > previous_histogram_abs

            if not momentum_expanding:
                self.logger.info(f"   ❌ Momentum fading: current {current_histogram_abs:.6f} <= previous {previous_histogram_abs:.6f}")
                return None

            self.logger.info(f"   ✅ Histogram strength validated (current: {current_histogram_abs:.6f}, expanding: +{(current_histogram_abs - previous_histogram_abs):.6f})")

            # STEP 2.6: Price Extreme Filter (Prevent buying tops / selling bottoms)
            if self.price_extreme_filter_enabled:
                self.logger.info("📍 Step 2.6: Checking price extreme (prevent buying tops/selling bottoms)...")

                # Get lookback data (ensure we have enough bars)
                lookback_bars = min(self.price_extreme_lookback, len(df))
                if lookback_bars < 50:
                    self.logger.info(f"   ⚠️ Insufficient data for extreme check ({lookback_bars} bars available, need 50+) - skipping")
                else:
                    # Get price data for lookback period
                    lookback_data = df.tail(lookback_bars)
                    current_price = df['close'].iloc[-1]

                    # Calculate price percentile
                    # For BULL: percentile tells us what % of prices are BELOW current price
                    # For BEAR: we need to invert (100 - percentile) to get % of prices ABOVE current price
                    all_prices = lookback_data['close'].values
                    prices_below = (all_prices < current_price).sum()
                    percentile = (prices_below / len(all_prices)) * 100

                    # Get pip value for distance calculations
                    pair = epic.split('.')[2] if '.' in epic else epic
                    pip_value = 0.01 if 'JPY' in pair else 0.0001

                    # Get min/max prices in lookback for context
                    period_low = lookback_data['low'].min()
                    period_high = lookback_data['high'].max()
                    price_range = period_high - period_low
                    distance_from_high = period_high - current_price
                    distance_from_low = current_price - period_low

                    self.logger.info(f"   Lookback: {lookback_bars} bars (~{lookback_bars//24:.1f} days)")
                    self.logger.info(f"   Period range: {period_low:.5f} to {period_high:.5f} ({price_range/pip_value:.1f} pips)")
                    self.logger.info(f"   Current price: {current_price:.5f}")
                    self.logger.info(f"   Distance from high: {distance_from_high/pip_value:.1f} pips")
                    self.logger.info(f"   Distance from low: {distance_from_low/pip_value:.1f} pips")
                    self.logger.info(f"   Price percentile: {percentile:.1f}% (higher than {percentile:.1f}% of prices in period)")

                    # Check for BULL signals at extreme highs
                    if signal_direction == 'BULL':
                        if percentile >= self.price_extreme_threshold:
                            self.logger.info(f"   ❌ BULL signal REJECTED: Buying at extreme high")
                            self.logger.info(f"   🚫 Price is in top {100-self.price_extreme_threshold}% of {lookback_bars}-bar range")
                            self.logger.info(f"   🚫 Only {distance_from_high/pip_value:.1f} pips from period high ({period_high:.5f})")
                            self.logger.info(f"   🚫 This indicates potential exhaustion/reversal point")
                            return None
                        else:
                            self.logger.info(f"   ✅ BULL signal: Price not at extreme (percentile {percentile:.1f}% < threshold {self.price_extreme_threshold}%)")

                    # Check for BEAR signals at extreme lows
                    elif signal_direction == 'BEAR':
                        inverted_percentile = 100 - percentile
                        if inverted_percentile >= self.price_extreme_threshold:
                            self.logger.info(f"   ❌ BEAR signal REJECTED: Selling at extreme low")
                            self.logger.info(f"   🚫 Price is in bottom {100-self.price_extreme_threshold}% of {lookback_bars}-bar range")
                            self.logger.info(f"   🚫 Only {distance_from_low/pip_value:.1f} pips from period low ({period_low:.5f})")
                            self.logger.info(f"   🚫 This indicates potential exhaustion/reversal point")
                            return None
                        else:
                            self.logger.info(f"   ✅ BEAR signal: Price not at extreme (inverted percentile {inverted_percentile:.1f}% < threshold {self.price_extreme_threshold}%)")

            # STEP 2.7: Price Structure Validation (Hybrid Approach)
            if self.structure_validation_enabled:
                self.logger.info("🏗️  Step 2.7: Validating price structure (higher highs/lows)...")

                structure_valid = self._validate_price_structure(df, signal_direction, epic)

                if not structure_valid:
                    self.logger.info(f"   🚫 Signal REJECTED - Price structure does not confirm {signal_direction} direction")
                    return None

                self.logger.info(f"   ✅ Price structure confirms {signal_direction} signal")

            # STEP 3: Validate Against H4 Trend
            self.logger.info("📊 Step 3: Validating against H4 trend...")

            h4_allows_bull = h4_trend == 'bullish'
            h4_allows_bear = h4_trend == 'bearish'

            if signal_direction == 'BULL' and not h4_allows_bull:
                self.logger.info(f"   ❌ Bullish signal rejected - H4 trend is {h4_trend}")
                return None

            if signal_direction == 'BEAR' and not h4_allows_bear:
                self.logger.info(f"   ❌ Bearish signal rejected - H4 trend is {h4_trend}")
                return None

            self.logger.info(f"   ✅ {signal_direction} signal aligns with H4 {h4_trend} trend")

            # STEP 3.5: 34 EMA Trend Filter (1H)
            if self.ema_filter_enabled:
                self.logger.info("📈 Step 3.5: Checking 34 EMA trend filter...")

                # Calculate 34 EMA on 1H data
                ema_34 = df['close'].ewm(span=self.ema_filter_period, adjust=False).mean()
                current_ema = ema_34.iloc[-1]
                current_price = df['close'].iloc[-1]

                # Check alignment
                if signal_direction == 'BULL':
                    ema_aligned = current_price > current_ema
                    if ema_aligned:
                        self.logger.info(f"   ✅ BULL signal: Price ({current_price:.5f}) > EMA34 ({current_ema:.5f})")
                    else:
                        self.logger.info(f"   ❌ BULL signal: Price ({current_price:.5f}) < EMA34 ({current_ema:.5f})")
                        if self.ema_require_alignment:
                            self.logger.info(f"   🚫 Signal REJECTED - Price below EMA (BULL signals require price > EMA)")
                            return None
                else:  # BEAR
                    ema_aligned = current_price < current_ema
                    if ema_aligned:
                        self.logger.info(f"   ✅ BEAR signal: Price ({current_price:.5f}) < EMA34 ({current_ema:.5f})")
                    else:
                        self.logger.info(f"   ❌ BEAR signal: Price ({current_price:.5f}) > EMA34 ({current_ema:.5f})")
                        if self.ema_require_alignment:
                            self.logger.info(f"   🚫 Signal REJECTED - Price above EMA (BEAR signals require price < EMA)")
                            return None

            # STEP 3.6: Multi-Timeframe MACD Alignment Check
            mtf_alignment_data = None
            structure_data = None

            if self.mtf_macd_alignment_enabled:
                self.logger.info("📊 Step 3.6: Checking 1H & 4H MACD alignment...")

                mtf_alignment_data = self._check_mtf_macd_alignment(
                    df_1h=df,
                    h4_data=h4_data,
                    signal_direction=signal_direction
                )

                # Optional: Reject if not aligned (if hard filter enabled)
                if self.mtf_require_alignment and not mtf_alignment_data['aligned']:
                    self.logger.info(f"   🚫 Signal rejected - MACD timeframes not aligned (hard filter enabled)")
                    return None

            # STEP 3.6: Validate H4 Market Structure Alignment
            if self.h4_structure_alignment_enabled:
                self.logger.info("🏗️  Step 3.6: Validating H4 market structure alignment...")

                structure_data = self._track_h4_market_structure(
                    epic=epic,
                    current_time=current_time,
                    signal_direction=signal_direction
                )

                # Block signal if structure doesn't align (we don't trade against market structure)
                if self.h4_require_structure_alignment and structure_data.get('break_type'):
                    if not structure_data.get('aligned', False):
                        self.logger.info(f"   🚫 Signal REJECTED - Market structure misaligned")
                        self.logger.info(f"      Signal: {signal_direction}, Structure: {structure_data['direction']}")
                        self.logger.info(f"      We don't trade against market structure trend!")
                        return None

            # STEP 4: Calculate Stop Loss and Take Profit
            self.logger.info("💰 Step 4: Calculating SL/TP...")

            stop_loss, take_profit = self._calculate_simple_sl_tp(
                df=df,
                epic=epic,
                signal_direction=signal_direction,
                entry_price=current_price
            )

            if not stop_loss or not take_profit:
                self.logger.warning("Could not calculate valid SL/TP")
                return None

            # Validate R:R ratio (risk management filter)
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            rr_ratio = reward / risk if risk > 0 else 0

            if rr_ratio < self.min_rr_ratio:
                self.logger.info(f"   🚫 Signal REJECTED - R:R ratio too low")
                self.logger.info(f"      Current R:R: 1:{rr_ratio:.2f} | Required: 1:{self.min_rr_ratio:.2f}")
                self.logger.info(f"      Risk: {risk:.5f} | Reward: {reward:.5f}")
                return None

            self.logger.info(f"   ✅ SL: {stop_loss:.5f} | TP: {take_profit:.5f} | R:R: 1:{rr_ratio:.2f}")

            # STEP 5: Calculate Simple Confidence
            self.logger.info("📊 Step 5: Calculating confidence...")

            # Base confidence: 60% (simpler strategy = higher base)
            confidence = 0.60

            # H4 trend alignment bonus
            if h4_data.get('histogram_expanding', False):
                confidence += 0.10  # +10% for expanding H4 momentum
                self.logger.info("   +10% for expanding H4 histogram")

            # Strong H4 histogram bonus
            histogram_abs = abs(h4_data.get('histogram', 0))
            if histogram_abs > 0.0001:
                confidence += 0.10  # +10% for strong H4 trend
                self.logger.info("   +10% for strong H4 histogram")

            # Multi-Timeframe MACD Alignment bonus
            if mtf_alignment_data and mtf_alignment_data['aligned']:
                confidence += mtf_alignment_data['confidence_adjustment']
                self.logger.info(f"   +{mtf_alignment_data['confidence_adjustment']*100:.0f}% for 1H & 4H MACD alignment")

            # Cap at 90%
            confidence = min(confidence, 0.90)
            self.logger.info(f"   Final confidence: {confidence:.0%}")

            # Calculate stop and limit distances in points (for API order placement)
            pip_multiplier = 100 if 'JPY' in epic else 10000
            stop_distance_points = int(abs(current_price - stop_loss) * pip_multiplier)
            limit_distance_points = int(abs(take_profit - current_price) * pip_multiplier)

            self.logger.info(f"   SL/TP distances: {stop_distance_points} / {limit_distance_points} points")

            # BUILD SIGNAL (with correct field names for validator)
            signal = {
                # Core fields (validator expects these exact names)
                'signal_type': signal_direction,  # BULL or BEAR
                'confidence_score': round(confidence, 2),  # 0.0 to 1.0
                'price': current_price,  # Entry price
                'epic': epic,

                # Trading levels (prices)
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_price': current_price,  # Also keep for compatibility

                # Trading levels (distances in points - for API order placement)
                'stop_distance': stop_distance_points,
                'limit_distance': limit_distance_points,
                'use_provided_sl_tp': True,  # Tell API to use these values instead of calculating ATR

                # Strategy metadata
                'strategy': 'macd_confluence',  # Keep name for validator compatibility
                'strategy_name': 'MACD Simple Crossover',  # Updated name
                'timeframe': self.timeframe,
                'risk_reward_ratio': round(rr_ratio, 2),

                # Strategy context (simplified)
                'h4_trend': h4_trend,
                'h4_histogram': round(h4_data['histogram'], 6),
                'h4_histogram_expanding': h4_data.get('histogram_expanding', False),
                'macd_line': round(macd_current, 6),
                'macd_signal': round(signal_current, 6),

                # Metadata
                'timestamp': datetime.now().isoformat(),
                'signal': signal_direction  # Also keep for backward compatibility
            }

            # Add MTF MACD alignment metadata if available
            if mtf_alignment_data:
                signal['mtf_macd_aligned'] = mtf_alignment_data['aligned']
                signal['1h_macd_histogram'] = round(mtf_alignment_data['1h_histogram'], 6)
                signal['1h_macd_direction'] = mtf_alignment_data['1h_direction']
                signal['4h_macd_direction'] = mtf_alignment_data['4h_direction']

            # Add H4 market structure metadata if available (metadata only)
            if structure_data and structure_data['break_type'] is not None:
                signal['h4_structure_break_type'] = structure_data['break_type']
                signal['h4_structure_direction'] = structure_data['direction']
                signal['h4_structure_significance'] = round(structure_data['significance'], 2)
                signal['h4_structure_bars_ago'] = structure_data['bars_ago']

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎯 SIGNAL GENERATED: {signal_direction}")
            self.logger.info(f"   Entry: {current_price:.5f}")
            self.logger.info(f"   SL: {stop_loss:.5f} ({stop_distance_points}pt) | TP: {take_profit:.5f} ({limit_distance_points}pt)")
            self.logger.info(f"   R:R: 1:{rr_ratio:.2f} | Confidence: {confidence:.0%}")
            self.logger.info(f"   H4 Trend: {h4_trend} (histogram: {h4_data['histogram']:.6f})")
            if mtf_alignment_data:
                alignment_status = "✅ ALIGNED" if mtf_alignment_data['aligned'] else "❌ NOT ALIGNED"
                self.logger.info(f"   MTF MACD: {alignment_status} (1H: {mtf_alignment_data['1h_direction']}, 4H: {mtf_alignment_data['4h_direction']})")
            if structure_data and structure_data['break_type']:
                self.logger.info(f"   H4 Structure: {structure_data['break_type']} {structure_data['direction']} ({structure_data['bars_ago']} bars ago)")
            self.logger.info(f"   MACD Crossover: {macd_current:.6f} > {signal_current:.6f}")
            self.logger.info(f"{'='*60}\n")

            return signal

        except Exception as e:
            self.logger.error(f"Error in detect_signal: {e}", exc_info=True)
            return None

    def _check_mtf_macd_alignment(self,
                                  df_1h: pd.DataFrame,
                                  h4_data: Dict,
                                  signal_direction: str) -> Dict:
        """
        Check if 1H and 4H MACD both align with signal direction.
        This is a key confluence check that affects confidence.

        Args:
            df_1h: 1H DataFrame with MACD indicators
            h4_data: H4 MACD data from mtf_filter
            signal_direction: 'BULL' or 'BEAR'

        Returns:
            Dict with alignment data:
            {
                'aligned': bool,              # Both timeframes align
                'confidence_adjustment': float,  # +0.10 if aligned, 0 if not
                '1h_histogram': float,
                '4h_histogram': float,
                '1h_direction': str,          # 'bullish', 'bearish', 'neutral'
                '4h_direction': str
            }
        """
        result = {
            'aligned': False,
            'confidence_adjustment': 0.0,
            '1h_histogram': 0.0,
            '4h_histogram': 0.0,
            '1h_direction': 'neutral',
            '4h_direction': 'neutral'
        }

        try:
            # Get 1H MACD histogram
            if 'macd_histogram' in df_1h.columns:
                h1_histogram = df_1h['macd_histogram'].iloc[-1]
            else:
                h1_histogram = df_1h['macd_line'].iloc[-1] - df_1h['macd_signal'].iloc[-1]

            result['1h_histogram'] = h1_histogram

            # Determine 1H direction
            if h1_histogram > 0.00001:
                result['1h_direction'] = 'bullish'
            elif h1_histogram < -0.00001:
                result['1h_direction'] = 'bearish'
            else:
                result['1h_direction'] = 'neutral'

            # Get 4H MACD histogram
            h4_histogram = h4_data.get('histogram', 0.0)
            result['4h_histogram'] = h4_histogram

            # Determine 4H direction
            if h4_histogram > 0.00001:
                result['4h_direction'] = 'bullish'
            elif h4_histogram < -0.00001:
                result['4h_direction'] = 'bearish'
            else:
                result['4h_direction'] = 'neutral'

            # Check alignment
            if signal_direction == 'BULL':
                # For BULL signals, both must be bullish
                result['aligned'] = (result['1h_direction'] == 'bullish' and result['4h_direction'] == 'bullish')
            else:  # BEAR
                # For BEAR signals, both must be bearish
                result['aligned'] = (result['1h_direction'] == 'bearish' and result['4h_direction'] == 'bearish')

            # Apply confidence boost if aligned
            if result['aligned']:
                result['confidence_adjustment'] = self.mtf_alignment_boost

            if self.log_mtf_alignment:
                self.logger.info(f"   1H MACD: {result['1h_direction']} (histogram: {h1_histogram:.6f})")
                self.logger.info(f"   4H MACD: {result['4h_direction']} (histogram: {h4_histogram:.6f})")
                self.logger.info(f"   Alignment: {'✅ ALIGNED' if result['aligned'] else '❌ NOT ALIGNED'}")
                if result['aligned']:
                    self.logger.info(f"   Confidence boost: +{result['confidence_adjustment']*100:.0f}%")

            return result

        except Exception as e:
            self.logger.error(f"Error checking MTF MACD alignment: {e}", exc_info=True)
            return result

    def _track_h4_market_structure(self,
                                   epic: str,
                                   current_time,
                                   signal_direction: str) -> Dict:
        """
        Track the last H4 market structure break (BOS/CHOCH) and validate alignment.

        IMPORTANT: We don't trade against market structure!
        - BULL signals require bullish structure (last BOS/CHOCH was bullish)
        - BEAR signals require bearish structure (last BOS/CHOCH was bearish)

        Args:
            epic: Currency pair epic
            current_time: Current timestamp
            signal_direction: 'BULL' or 'BEAR'

        Returns:
            Dict with structure data:
            {
                'break_type': str,               # 'BOS', 'CHOCH', or None
                'direction': str,                # 'bullish', 'bearish', or 'neutral'
                'significance': float,           # 0-1 score
                'bars_ago': int,                 # How many bars ago the break occurred
                'aligned': bool                  # Whether structure aligns with signal direction
            }
        """
        result = {
            'break_type': None,
            'direction': 'neutral',
            'significance': 0.0,
            'bars_ago': None,
            'aligned': False
        }

        try:
            if not self.structure_analyzer or not self.data_fetcher:
                return result

            # Fetch H4 data for structure analysis
            bars_needed = max(100, self.h4_structure_lookback + 20)

            # Extract pair from epic (e.g., CS.D.EURUSD.CEEM.IP -> EURUSD)
            pair = epic.split('.')[2] if '.' in epic else epic

            # Calculate hours needed (bars * 4 hours per bar)
            lookback_hours = bars_needed * 4

            h4_df = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='4h',
                lookback_hours=lookback_hours
            )

            if h4_df is None or len(h4_df) < 20:
                return result

            # Run SMC structure analysis
            h4_with_structure = self.structure_analyzer.analyze_market_structure(
                df=h4_df,
                config=self.h4_structure_config,
                epic=epic,
                timeframe='4h'
            )

            if h4_with_structure is None or len(h4_with_structure) == 0:
                return result

            # Look for structure breaks in lookback window
            recent_data = h4_with_structure.tail(self.h4_structure_lookback)

            # Find structure breaks
            if 'structure_break' in recent_data.columns:
                structure_breaks = recent_data[recent_data['structure_break'] == True]

                if len(structure_breaks) > 0:
                    # Get most recent structure break
                    last_break = structure_breaks.iloc[-1]

                    result['break_type'] = last_break.get('break_type', None)  # 'BOS' or 'CHOCH'
                    result['direction'] = last_break.get('break_direction', 'neutral')  # 'bullish' or 'bearish'
                    result['significance'] = last_break.get('structure_significance', 0.0)

                    # Calculate how many bars ago
                    break_index = structure_breaks.index[-1]
                    current_index = h4_with_structure.index[-1]
                    result['bars_ago'] = len(h4_with_structure.loc[break_index:current_index]) - 1

                    # Check alignment with signal direction
                    signal_wants_bullish = (signal_direction == 'BULL')
                    structure_is_bullish = (result['direction'] == 'bullish')
                    result['aligned'] = (signal_wants_bullish == structure_is_bullish)

                    if self.h4_log_structure:
                        self.logger.info(f"   Last H4 structure: {result['break_type']} {result['direction']} "
                                        f"({result['bars_ago']} bars ago, sig: {result['significance']:.2f})")
                        alignment_status = "✅ ALIGNED" if result['aligned'] else "❌ MISALIGNED"
                        self.logger.info(f"   Structure vs Signal: {alignment_status} (signal: {signal_direction})")

            return result

        except Exception as e:
            self.logger.error(f"Error tracking H4 market structure: {e}", exc_info=True)
            return result

    def _calculate_simple_sl_tp(self,
                               df: pd.DataFrame,
                               epic: str,
                               signal_direction: str,
                               entry_price: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate simple ATR-based stop loss and take profit.

        Args:
            df: Strategy timeframe OHLC DataFrame
            epic: Currency pair epic
            signal_direction: 'BULL' or 'BEAR'
            entry_price: Entry price

        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            # Calculate ATR if not in DataFrame
            if 'atr' not in df.columns:
                atr_series = self._calculate_atr(df, period=14)
                atr = atr_series.iloc[-1]
            else:
                atr = df['atr'].iloc[-1]

            # Get pip value for this pair
            pip_value = 0.01 if 'JPY' in epic else 0.0001
            pip_multiplier = 100 if 'JPY' in epic else 10000

            # ATR-based stop loss (1.5x ATR)
            stop_distance = atr * 1.5
            if signal_direction == 'BULL':
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance

            # Validate stop distance is within bounds
            stop_distance_pips = abs(entry_price - stop_loss) * pip_multiplier

            # Apply min/max constraints
            if stop_distance_pips < self.min_stop_pips:
                # Widen to minimum
                stop_loss = entry_price - (self.min_stop_pips * pip_value) if signal_direction == 'BULL' \
                           else entry_price + (self.min_stop_pips * pip_value)
            elif stop_distance_pips > self.max_stop_pips:
                # Tighten to maximum
                stop_loss = entry_price - (self.max_stop_pips * pip_value) if signal_direction == 'BULL' \
                           else entry_price + (self.max_stop_pips * pip_value)

            # ATR-based take profit (3.0x ATR for 2:1 R:R)
            tp_distance = atr * 3.0
            if signal_direction == 'BULL':
                take_profit = entry_price + tp_distance
            else:
                take_profit = entry_price - tp_distance

            self.logger.debug(f"Simple SL/TP: ATR={atr:.6f}, SL distance={stop_distance_pips:.1f} pips")

            return stop_loss, take_profit

        except Exception as e:
            self.logger.error(f"Error calculating simple SL/TP: {e}", exc_info=True)
            return None, None

    def _calculate_confidence(self,
                             h4_data: Dict,
                             confluence_zone: Dict,
                             pattern: Optional[Dict],
                             signal_direction: str) -> float:
        """
        Calculate signal confidence based on all factors.

        Args:
            h4_data: H4 MACD trend data
            confluence_zone: Confluence zone data
            pattern: Candlestick pattern data
            signal_direction: BULL or BEAR

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence
        confidence = 0.50  # 50% base

        # H4 trend strength boost
        if h4_data and h4_data.get('is_valid'):
            confidence += 0.10  # +10% for valid H4 trend

            if h4_data.get('histogram_expanding'):
                confidence += 0.05  # +5% for expanding momentum

        # Confluence zone quality boost
        zone_quality = confluence_zone.get('quality', 'low')
        quality_boost = {
            'excellent': 0.20,
            'high': 0.15,
            'medium': 0.10,
            'low': 0.05
        }
        confidence += quality_boost.get(zone_quality, 0.05)

        # Pattern quality boost
        if pattern:
            pattern_score = pattern.get('quality_score', 0)
            if pattern_score >= 80:
                confidence += 0.10  # +10% for excellent pattern
            elif pattern_score >= 70:
                confidence += 0.05  # +5% for good pattern

        # Cap at 95%
        confidence = min(confidence, 0.95)

        return confidence

    def _calculate_sl_tp(self,
                        df: pd.DataFrame,
                        epic: str,
                        signal_direction: str,
                        entry_price: float,
                        fib_zones: Dict,
                        pattern: Optional[Dict]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit levels.

        Args:
            df: 15M OHLC data
            epic: Currency pair
            signal_direction: BULL or BEAR
            entry_price: Entry price
            fib_zones: Fibonacci zones data
            pattern: Candlestick pattern data

        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            pip_multiplier = 100 if 'JPY' in epic else 10000
            pip_value = 1.0 / pip_multiplier

            # Calculate ATR for baseline
            if 'atr' not in df.columns:
                df['atr'] = self._calculate_atr(df, period=14)

            atr = df['atr'].iloc[-1]

            if self.use_15m_stops:
                # Use recent 15M swing for tighter stops
                lookback = 10
                recent_data = df.tail(lookback)

                if signal_direction == 'BULL':
                    swing_low = recent_data['low'].min()
                    stop_loss = swing_low - (2 * pip_value)  # 2 pips below swing low
                else:
                    swing_high = recent_data['high'].max()
                    stop_loss = swing_high + (2 * pip_value)  # 2 pips above swing high

                # Validate stop distance
                stop_distance_pips = abs(entry_price - stop_loss) * pip_multiplier

                if stop_distance_pips < self.min_stop_pips:
                    # Widen to minimum
                    stop_loss = entry_price - (self.min_stop_pips * pip_value) if signal_direction == 'BULL' \
                               else entry_price + (self.min_stop_pips * pip_value)
                elif stop_distance_pips > self.max_stop_pips:
                    # Tighten to maximum
                    stop_loss = entry_price - (self.max_stop_pips * pip_value) if signal_direction == 'BULL' \
                               else entry_price + (self.max_stop_pips * pip_value)

            else:
                # ATR-based stop
                stop_distance = atr * self.stop_atr_multiplier
                stop_loss = entry_price - stop_distance if signal_direction == 'BULL' \
                           else entry_price + stop_distance

            # Calculate take profit
            if self.use_structure_targets and fib_zones:
                # Target next swing level
                if signal_direction == 'BULL':
                    target = fib_zones.get('swing_high', {}).get('price')
                else:
                    target = fib_zones.get('swing_low', {}).get('price')

                if target:
                    take_profit = target
                else:
                    # Fallback to ATR-based
                    tp_distance = atr * self.tp_atr_multiplier
                    take_profit = entry_price + tp_distance if signal_direction == 'BULL' \
                                 else entry_price - tp_distance
            else:
                # ATR-based target
                tp_distance = atr * self.tp_atr_multiplier
                take_profit = entry_price + tp_distance if signal_direction == 'BULL' \
                             else entry_price - tp_distance

            return stop_loss, take_profit

        except Exception as e:
            self.logger.error(f"Error calculating SL/TP: {e}", exc_info=True)
            return None, None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.rolling(window=period).mean()

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators for MACD Confluence Strategy.

        Returns:
            List of indicator names
        """
        return [
            'macd_line',
            'macd_signal',
            'macd_histogram',
            'atr',
            'ema_21',
            'ema_50'
        ]


# Quick test
if __name__ == '__main__':
    print("🎯 MACD Confluence Strategy - Testing")

    # Test initialization
    try:
        strategy = MACDStrategy(epic='CS.D.EURUSD.CEEM.IP')
        print("✅ Strategy initialized successfully")
        print(f"   Components: Fib Calculator, Pattern Detector, Confluence Analyzer, MTF Filter")
        print(f"   Settings: {strategy.confluence_mode} mode, {strategy.fib_lookback} bar lookback")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
