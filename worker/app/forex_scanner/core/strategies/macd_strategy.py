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

            # Get current and previous MACD values
            macd_current = df['macd_line'].iloc[-1]
            signal_current = df['macd_signal'].iloc[-1]
            macd_prev = df['macd_line'].iloc[-2]
            signal_prev = df['macd_signal'].iloc[-2]

            # Detect crossover
            bullish_cross = (macd_prev <= signal_prev) and (macd_current > signal_current)
            bearish_cross = (macd_prev >= signal_prev) and (macd_current < signal_current)

            if not bullish_cross and not bearish_cross:
                self.logger.info("   No MACD crossover detected")
                return None

            # Determine signal direction
            if bullish_cross:
                signal_direction = 'BULL'
                self.logger.info("   ✅ Bullish MACD crossover detected")
            else:
                signal_direction = 'BEAR'
                self.logger.info("   ✅ Bearish MACD crossover detected")

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

            # 🔥 EXPANSION WINDOW: Check histogram over last 3 bars (current + 2 previous)
            # This allows the histogram to "catch up" after crossover
            # 3 bars on 1H = 3 hours - backtesting proves this is optimal
            # Longer windows (4-5 bars) catch reversals and reduce profitability
            expansion_window = 3
            histogram_column = 'macd_histogram' if 'macd_histogram' in df.columns else None

            if histogram_column:
                # Get last N histogram values
                recent_histograms = df[histogram_column].iloc[-expansion_window:].values
            else:
                # Calculate from MACD line and signal
                recent_histograms = []
                for i in range(-expansion_window, 0):
                    hist = df['macd_line'].iloc[i] - df['macd_signal'].iloc[i]
                    recent_histograms.append(hist)

            # Find the maximum absolute histogram value in the window
            max_histogram_abs = max([abs(h) for h in recent_histograms])
            current_histogram = recent_histograms[-1]

            self.logger.info(f"   Current histogram: {current_histogram:.6f} (abs: {abs(current_histogram):.6f})")
            self.logger.info(f"   Max histogram in last {expansion_window} bars: {max_histogram_abs:.6f}")
            self.logger.info(f"   Min threshold for {pair}: {min_histogram:.6f}")

            if max_histogram_abs < min_histogram:
                self.logger.info(f"   ❌ Histogram too weak: max {max_histogram_abs:.6f} < {min_histogram:.6f} (checked {expansion_window} bars)")
                return None

            self.logger.info(f"   ✅ Histogram strength validated (max in {expansion_window}-bar window)")

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

            # Validate R:R ratio
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            rr_ratio = reward / risk if risk > 0 else 0

            if rr_ratio < self.min_rr_ratio:
                self.logger.info(f"   R:R too low: {rr_ratio:.2f} < {self.min_rr_ratio}")
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

            # Cap at 90%
            confidence = min(confidence, 0.90)
            self.logger.info(f"   Final confidence: {confidence:.0%}")

            # BUILD SIGNAL (with correct field names for validator)
            signal = {
                # Core fields (validator expects these exact names)
                'signal_type': signal_direction,  # BULL or BEAR
                'confidence_score': round(confidence, 2),  # 0.0 to 1.0
                'price': current_price,  # Entry price
                'epic': epic,

                # Trading levels
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_price': current_price,  # Also keep for compatibility

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

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎯 SIGNAL GENERATED: {signal_direction}")
            self.logger.info(f"   Entry: {current_price:.5f}")
            self.logger.info(f"   SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            self.logger.info(f"   R:R: 1:{rr_ratio:.2f} | Confidence: {confidence:.0%}")
            self.logger.info(f"   H4 Trend: {h4_trend} (histogram: {h4_data['histogram']:.6f})")
            self.logger.info(f"   MACD Crossover: {macd_current:.6f} > {signal_current:.6f}")
            self.logger.info(f"{'='*60}\n")

            return signal

        except Exception as e:
            self.logger.error(f"Error in detect_signal: {e}", exc_info=True)
            return None

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
