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
            timeframe: Primary timeframe for entries (default: 15m)
        """
        # Basic initialization
        self.name = 'macd'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(logging.INFO)

        # Store parameters
        self.backtest_mode = backtest_mode
        self.epic = epic
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher
        self.price_adjuster = PriceAdjuster()

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

    def detect_signal(self,
                     df: pd.DataFrame,
                     epic: str,
                     spread_pips: float = 1.5,
                     intelligence_data: Dict = None,
                     regime_data: Dict = None) -> Optional[Dict]:
        """
        Detect MACD confluence trading signals.

        Args:
            df: 15M OHLC DataFrame
            epic: Currency pair epic
            spread_pips: Current spread in pips
            intelligence_data: Optional market intelligence data
            regime_data: Optional regime data

        Returns:
            Signal dict or None
        """
        try:
            if len(df) < 100:
                self.logger.debug(f"Insufficient data: {len(df)} bars (need 100+)")
                return None

            current_price = df['close'].iloc[-1]
            current_time = df.index[-1] if hasattr(df.index[-1], 'strftime') else None

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔍 MACD Confluence Analysis - {epic} @ {current_price:.5f}")
            self.logger.info(f"{'='*60}")

            # STEP 1: H4 MACD Trend Filter
            if self.h4_filter_enabled:
                self.logger.info("📊 Step 1: Checking H4 MACD trend...")

                # Try both BULL and BEAR to see which trend we have
                h4_data = self.mtf_filter.get_h4_trend_direction(epic, current_time)

                if not h4_data:
                    self.logger.warning("❌ No H4 data available - cannot validate trend")
                    return None

                h4_trend = h4_data['trend']
                self.logger.info(f"   H4 Trend: {h4_trend.upper()} (histogram: {h4_data['histogram']:.6f})")

                if h4_trend == 'neutral':
                    self.logger.info("⚠️  H4 trend is neutral - no clear direction")
                    return None

                # Determine signal direction from H4
                signal_direction = 'BULL' if h4_trend == 'bullish' else 'BEAR'

            else:
                # Without H4 filter, we'd need another way to determine direction
                # For now, require H4 filter
                self.logger.warning("H4 filter disabled - cannot determine signal direction")
                return None

            # STEP 2: Get H1 Fibonacci Zones
            self.logger.info("📐 Step 2: Calculating H1 Fibonacci levels...")

            h1_data = self.mtf_filter.get_h1_swing_data(epic, current_time)
            if h1_data is None or len(h1_data) < self.fib_lookback:
                self.logger.warning("❌ Insufficient H1 data for Fibonacci calculation")
                return None

            fib_zones = self.fib_calculator.get_fibonacci_zones(
                df=h1_data,
                epic=epic,
                current_trend=h4_trend
            )

            if not fib_zones:
                self.logger.info("   No valid Fibonacci zones found")
                return None

            # STEP 3: Analyze Confluence Zones
            self.logger.info("🎯 Step 3: Analyzing confluence zones...")

            # Get swing levels and EMAs from 15M for confluence
            swing_highs, swing_lows = self._find_swing_levels(df, lookback=50)
            ema_values = self._get_current_emas(df)

            confluence_zones = self.confluence_analyzer.find_all_confluence_zones(
                fib_data=fib_zones,
                current_price=current_price,
                swing_highs=swing_highs,
                swing_lows=swing_lows,
                ema_values=ema_values,
                epic=epic
            )

            if not confluence_zones:
                self.logger.info("   No valid confluence zones")
                return None

            # STEP 4: Check if Price at Confluence Zone
            self.logger.info("📍 Step 4: Checking if price at confluence zone...")

            at_zone = self.confluence_analyzer.is_price_at_confluence_zone(
                current_price=current_price,
                confluence_zones=confluence_zones,
                epic=epic,
                min_quality='low'  # Accept any valid zone
            )

            if not at_zone:
                self.logger.info(f"   Price not at confluence zone (nearest: {confluence_zones[0]['distance_from_price_pips']:.1f} pips away)")
                return None

            self.logger.info(f"   ✅ Price at {at_zone['fib_level']}% Fib level - {at_zone['quality']} quality zone")

            # STEP 5: Detect Candlestick Pattern
            self.logger.info("🕯️  Step 5: Detecting candlestick pattern...")

            pattern = self.pattern_detector.get_best_pattern(df, signal_direction)

            if self.require_pattern and not pattern:
                self.logger.info(f"   No valid {signal_direction} pattern found")
                return None

            if pattern:
                if pattern['quality_score'] < self.min_pattern_quality:
                    self.logger.info(f"   Pattern quality too low: {pattern['quality_score']} < {self.min_pattern_quality}")
                    return None

                self.logger.info(f"   ✅ {pattern['pattern']} detected (quality: {pattern['quality_score']}/100)")

            # STEP 6: Calculate Confidence
            confidence = self._calculate_confidence(
                h4_data=h4_data,
                confluence_zone=at_zone,
                pattern=pattern,
                signal_direction=signal_direction
            )

            if confidence < self.min_confidence:
                self.logger.info(f"   Confidence too low: {confidence:.0%} < {self.min_confidence:.0%}")
                return None

            # STEP 7: Calculate Stop Loss and Take Profit
            stop_loss, take_profit = self._calculate_sl_tp(
                df=df,
                epic=epic,
                signal_direction=signal_direction,
                entry_price=current_price,
                fib_zones=fib_zones,
                pattern=pattern
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
                'strategy': 'macd_confluence',
                'strategy_name': 'MACD Confluence',
                'timeframe': self.timeframe,
                'risk_reward_ratio': round(rr_ratio, 2),

                # Confluence analysis context
                'h4_trend': h4_trend,
                'h4_histogram': h4_data['histogram'],
                'fib_level': at_zone['fib_level'],
                'confluence_score': at_zone['confluence_score'],
                'confluence_factors': ', '.join(at_zone['factors']),  # Join for display
                'pattern': pattern['pattern'] if pattern else None,
                'pattern_quality': pattern['quality_score'] if pattern else None,

                # Metadata
                'timestamp': datetime.now().isoformat(),
                'signal': signal_direction  # Also keep for backward compatibility
            }

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎯 SIGNAL GENERATED: {signal_direction}")
            self.logger.info(f"   Entry: {current_price:.5f}")
            self.logger.info(f"   SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            self.logger.info(f"   R:R: 1:{rr_ratio:.2f} | Confidence: {confidence:.0%}")
            self.logger.info(f"   Confluence: {at_zone['fib_level']}% ({', '.join(at_zone['factors'])})")
            if pattern:
                self.logger.info(f"   Pattern: {pattern['pattern']} ({pattern['quality_score']}/100)")
            self.logger.info(f"{'='*60}\n")

            return signal

        except Exception as e:
            self.logger.error(f"Error in detect_signal: {e}", exc_info=True)
            return None

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
