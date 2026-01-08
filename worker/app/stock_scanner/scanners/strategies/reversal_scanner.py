"""
Reversal Scanner (Merged: Selling Climax + Mean Reversion + Wyckoff Spring)

Consolidated reversal/bottom detection scanner combining three complementary approaches:

1. SELLING CLIMAX (Capitulation):
   - Price makes new 20-day low
   - Volume spike > 2.5x average (panic selling)
   - Close in upper 50%+ of range (absorption)

2. MEAN REVERSION (RSI Oversold):
   - RSI oversold (< 35) with bonus for extreme (< 25)
   - Bullish reversal candlestick pattern
   - Still in overall uptrend (not death cross)

3. WYCKOFF SPRING (Accumulation):
   - Stock in tight consolidation box (2-10% range)
   - Price breaks below support intraday
   - Close recovers above support
   - Light volume (weak selling pressure)

Entry Criteria (any of the above detected):
- Composite scoring favors multiple confluence factors

Stop Logic:
- Below pattern low (climax low / spring low / recent swing)
- 1.5x ATR buffer for safety

Target:
- TP1: 2R or return to mean (SMA20)
- TP2: 3R or top of consolidation box

Best For:
- Capitulation bottoming patterns
- Oversold bounces in quality stocks
- Wyckoff accumulation entries
"""

import numpy as np
import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..base_scanner import (
    BaseScanner, SignalSetup, ScannerConfig,
    SignalType, QualityTier
)
from ..scoring import SignalScorer
from ..exclusion_filters import ExclusionFilterEngine, get_mean_reversion_filter

logger = logging.getLogger(__name__)


class ReversalType(Enum):
    """Type of reversal pattern detected"""
    SELLING_CLIMAX = "selling_climax"
    MEAN_REVERSION = "mean_reversion"
    WYCKOFF_SPRING = "wyckoff_spring"


@dataclass
class ReversalScannerConfig(ScannerConfig):
    """Configuration for unified Reversal Scanner"""

    # ==========================================================================
    # SELLING CLIMAX Parameters
    # ==========================================================================
    climax_new_low_lookback: int = 20  # Period for new low detection
    climax_volume_spike_multiplier: float = 2.5  # Volume must be 2.5x average
    climax_min_close_position: float = 0.5  # Close must be in upper 50% of range

    # ==========================================================================
    # MEAN REVERSION Parameters
    # ==========================================================================
    mean_rev_max_rsi: float = 35.0  # Must be oversold
    mean_rev_extreme_oversold_rsi: float = 25.0  # Bonus for extreme
    mean_rev_require_bullish_pattern: bool = True  # Require reversal candle

    # ==========================================================================
    # WYCKOFF SPRING Parameters
    # ==========================================================================
    spring_box_lookback: int = 20  # Days to calculate consolidation box
    spring_max_box_height_pct: float = 10.0  # Max box height as % of price
    spring_min_box_height_pct: float = 2.0  # Min box height (avoid dead stocks)
    spring_min_close_position: float = 0.5  # Close must be in upper 50%
    spring_max_volume_ratio: float = 1.2  # Volume not excessive (weak selling)

    # ==========================================================================
    # Common Parameters
    # ==========================================================================
    setup_window: int = 3  # Look for patterns in last N days
    volume_avg_period: int = 20  # Period for average volume calculation
    min_relative_volume: float = 0.5  # Minimum relative volume

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    atr_stop_multiplier: float = 1.5
    max_stop_loss_pct: float = 8.0  # Max stop distance

    # Default R:R targets
    tp1_rr_ratio: float = 2.0
    tp2_rr_ratio: float = 3.0

    # ==========================================================================
    # Fundamental Filters
    # ==========================================================================
    max_pe_ratio: float = 50.0  # Not too expensive
    min_profit_margin: float = 0.0  # Can be loss-making (panic oversold)
    max_debt_to_equity: float = 3.0  # Some debt OK for reversal plays
    days_to_earnings_min: int = 7  # Avoid earnings


class ReversalScanner(BaseScanner):
    """
    Unified scanner for reversal/bottom patterns.

    Philosophy:
    - Combines three proven reversal detection methods
    - Higher score when multiple reversal types confirm each other
    - Counter-trend plays require strict risk management
    - Position sizing reduced due to fighting trend
    """

    BULLISH_REVERSAL_PATTERNS = [
        'bullish_engulfing', 'hammer', 'dragonfly_doji',
        'inverted_hammer', 'morning_star', 'bullish_marubozu'
    ]

    def __init__(
        self,
        db_manager,
        config: ReversalScannerConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or ReversalScannerConfig(), scorer)
        self.exclusion_filter = get_mean_reversion_filter()
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "reversal_scanner"

    @property
    def description(self) -> str:
        return "Unified reversal detection: climax, oversold bounce, Wyckoff spring"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute unified reversal scan.

        Steps:
        1. Get qualified tickers from watchlist
        2. For each ticker, check all three reversal patterns
        3. Score with bonuses for multiple pattern confirmation
        4. Create signals with best detected pattern as primary
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            # Use today's date - represents when pipeline ran
            calculation_date = datetime.now().date()

        # Ensure calculation_date is a date object
        if hasattr(calculation_date, 'date'):
            calculation_date = calculation_date.date()

        # Get qualified tickers
        tickers = await self._get_qualified_tickers(calculation_date)
        logger.info(f"Scanning {len(tickers)} stocks for reversal patterns")

        signals = []

        for ticker in tickers:
            signal = await self._scan_ticker(ticker, calculation_date)
            if signal:
                signals.append(signal)
                logger.info(f"  REVERSAL: {ticker} @ {signal.entry_price}")

        # Sort by score
        signals.sort(key=lambda x: x.composite_score, reverse=True)

        # Apply limit
        signals = signals[:self.config.max_signals_per_run]

        # Log summary
        high_quality = sum(1 for s in signals if s.is_high_quality)
        self.log_scan_summary(len(tickers), len(signals), high_quality)

        return signals

    async def _scan_ticker(
        self,
        ticker: str,
        calculation_date: datetime
    ) -> Optional[SignalSetup]:
        """Scan a single ticker for all reversal patterns."""

        # Get daily candles
        candles = await self._get_daily_candles(ticker, limit=100)

        if len(candles) < max(self.config.climax_new_low_lookback,
                             self.config.spring_box_lookback) + 10:
            return None

        # Convert to numpy arrays
        closes = np.array([float(c['close']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])
        volumes = np.array([float(c['volume']) for c in candles])

        # Get the actual candle timestamp for signal_timestamp
        candle_timestamp = candles[-1].get('timestamp') if candles else None

        # Get candidate data for scoring
        candidate = await self._get_candidate_data(ticker, calculation_date)
        if not candidate:
            candidate = {
                'ticker': ticker,
                'current_price': closes[-1],
                'atr_percent': self._calculate_atr_percent(highs, lows, closes),
            }

        # Check all three reversal types
        detected_patterns = []
        pattern_data = {}

        # 1. Check Selling Climax
        climax_result = self._detect_selling_climax(closes, highs, lows, volumes)
        if climax_result:
            detected_patterns.append(ReversalType.SELLING_CLIMAX)
            climax_idx, close_position = climax_result
            pattern_data['climax'] = {
                'idx': climax_idx,
                'close_position': close_position,
                'volume_ratio': volumes[-1] / np.mean(volumes[-self.config.volume_avg_period:]) if len(volumes) > self.config.volume_avg_period else 2.0,
                'low': lows[climax_idx],
                'high': highs[climax_idx],
            }

        # 2. Check Mean Reversion (RSI Oversold)
        mean_rev_result = self._check_mean_reversion(candidate)
        if mean_rev_result:
            detected_patterns.append(ReversalType.MEAN_REVERSION)
            pattern_data['mean_rev'] = mean_rev_result

        # 3. Check Wyckoff Spring
        spring_result = self._detect_wyckoff_spring(closes, highs, lows, volumes)
        if spring_result:
            detected_patterns.append(ReversalType.WYCKOFF_SPRING)
            spring_idx, support, resistance, close_pos, vol_ratio = spring_result
            pattern_data['spring'] = {
                'idx': spring_idx,
                'support': support,
                'resistance': resistance,
                'close_position': close_pos,
                'volume_ratio': vol_ratio,
                'low': lows[spring_idx],
                'high': highs[spring_idx],
                'box_height_pct': ((resistance - support) / support) * 100 if support > 0 else 0,
            }

        # No patterns detected
        if not detected_patterns:
            return None

        # Merge pattern data into candidate
        candidate['detected_patterns'] = [p.value for p in detected_patterns]
        candidate['pattern_data'] = pattern_data

        # Determine primary pattern (prefer climax > spring > mean_rev based on specificity)
        if ReversalType.SELLING_CLIMAX in detected_patterns:
            primary = ReversalType.SELLING_CLIMAX
        elif ReversalType.WYCKOFF_SPRING in detected_patterns:
            primary = ReversalType.WYCKOFF_SPRING
        else:
            primary = ReversalType.MEAN_REVERSION

        candidate['primary_pattern'] = primary.value

        # Calculate entry levels based on primary pattern
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, primary, pattern_data)

        # Score the signal
        score_components = self.scorer.score_bullish(candidate)
        composite_score = score_components.composite_score

        # Bonuses based on patterns
        composite_score = self._apply_pattern_bonuses(
            composite_score, detected_patterns, pattern_data, candidate
        )

        # Multi-confluence bonus (multiple reversal types detected)
        if len(detected_patterns) >= 2:
            composite_score = min(100, composite_score + 12)  # Double confluence
        if len(detected_patterns) == 3:
            composite_score = min(100, composite_score + 8)  # Triple confluence (rare)

        # Apply minimum threshold
        if composite_score < self.config.min_score_threshold:
            return None

        # Calculate risk percent
        risk_pct = abs(float(entry) - float(stop)) / float(entry) * 100 if float(entry) > 0 else 0

        # Build description
        description = self._build_description(ticker, detected_patterns, pattern_data, candidate)

        # Create signal
        signal = SignalSetup(
            ticker=ticker,
            scanner_name=self.scanner_name,
            signal_type=SignalType.BUY,
            signal_timestamp=candle_timestamp if candle_timestamp else datetime.now(),
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=Decimal(str(round(self.config.tp1_rr_ratio, 2))),
            risk_percent=Decimal(str(round(risk_pct, 2))),
            composite_score=composite_score,
            trend_score=Decimal(str(round(score_components.weighted_trend, 2))),
            momentum_score=Decimal(str(round(score_components.weighted_momentum, 2))),
            volume_score=Decimal(str(round(score_components.weighted_volume, 2))),
            pattern_score=Decimal(str(round(score_components.weighted_pattern, 2))),
            confluence_score=Decimal(str(round(score_components.weighted_confluence, 2))),
            setup_description=description,
            confluence_factors=self._build_confluence_factors(detected_patterns, pattern_data, candidate),
            timeframe="daily",
            market_regime=self._determine_market_regime(detected_patterns, pattern_data, candidate),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data=candidate,
        )

        # Calculate position size (reduced for reversal plays - counter-trend)
        signal.suggested_position_size_pct = self.calculate_position_size(
            Decimal(str(self.config.max_risk_per_trade_pct * 0.8)),  # 80% of normal size
            entry,
            stop,
            signal.quality_tier
        )

        return signal

    def _detect_selling_climax(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray
    ) -> Optional[Tuple[int, float]]:
        """
        Detect selling climax pattern.

        Returns (climax_index, close_position_pct) if found.
        """
        lookback = self.config.climax_new_low_lookback
        vol_period = self.config.volume_avg_period

        if len(closes) < lookback + vol_period:
            return None

        for i in range(len(closes) - self.config.setup_window, len(closes)):
            if i < lookback + vol_period:
                continue

            # 1. New low in lookback period
            prior_lows = lows[i - lookback:i]
            if lows[i] >= min(prior_lows):
                continue

            # 2. Volume spike
            avg_volume = np.mean(volumes[i - vol_period:i])
            if volumes[i] < avg_volume * self.config.climax_volume_spike_multiplier:
                continue

            # 3. Reversal close (upper portion of range)
            day_range = highs[i] - lows[i]
            if day_range <= 0:
                continue

            close_position = (closes[i] - lows[i]) / day_range
            if close_position < self.config.climax_min_close_position:
                continue

            return (i, close_position)

        return None

    def _check_mean_reversion(
        self,
        candidate: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check mean reversion (RSI oversold) conditions.

        Returns dict with RSI and pattern info if conditions met.
        """
        rsi_14 = float(candidate.get('rsi_14') or 50)

        if rsi_14 > self.config.mean_rev_max_rsi:
            return None

        # Check we're not in death cross (falling knife)
        sma_cross = candidate.get('sma_cross_signal', '')
        if sma_cross == 'death_cross':
            return None

        # Check for bullish pattern if required
        pattern = candidate.get('candlestick_pattern', '')
        has_bullish_pattern = pattern in self.BULLISH_REVERSAL_PATTERNS

        if self.config.mean_rev_require_bullish_pattern and not has_bullish_pattern:
            return None

        return {
            'rsi': rsi_14,
            'pattern': pattern,
            'extreme_oversold': rsi_14 <= self.config.mean_rev_extreme_oversold_rsi,
        }

    def _detect_wyckoff_spring(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray
    ) -> Optional[Tuple[int, float, float, float, float]]:
        """
        Detect Wyckoff Spring pattern.

        Returns (spring_idx, support, resistance, close_position, volume_ratio) if found.
        """
        box_lookback = self.config.spring_box_lookback
        vol_period = self.config.volume_avg_period

        if len(closes) < box_lookback + vol_period:
            return None

        for i in range(len(closes) - self.config.setup_window, len(closes)):
            if i < box_lookback + vol_period:
                continue

            # 1. Calculate consolidation box
            box_start = i - box_lookback
            box_end = i - 1

            box_highs = highs[box_start:box_end]
            box_lows = lows[box_start:box_end]

            support = np.min(box_lows)
            resistance = np.max(box_highs)

            # 2. Check box tightness
            if support <= 0:
                continue

            box_height_pct = ((resistance - support) / support) * 100

            if box_height_pct > self.config.spring_max_box_height_pct:
                continue
            if box_height_pct < self.config.spring_min_box_height_pct:
                continue

            # 3. Spring: low penetrates support, close recovers
            if lows[i] >= support:
                continue
            if closes[i] <= support:
                continue

            # 4. Close in upper portion
            day_range = highs[i] - lows[i]
            if day_range <= 0:
                continue

            close_position = (closes[i] - lows[i]) / day_range
            if close_position < self.config.spring_min_close_position:
                continue

            # 5. Light volume (weak selling)
            avg_volume = np.mean(volumes[i - vol_period:i])
            if avg_volume <= 0:
                continue

            volume_ratio = volumes[i] / avg_volume
            if volume_ratio > self.config.spring_max_volume_ratio:
                continue

            return (i, support, resistance, close_position, volume_ratio)

        return None

    def _apply_pattern_bonuses(
        self,
        score: float,
        patterns: List[ReversalType],
        pattern_data: Dict[str, Dict],
        candidate: Dict[str, Any]
    ) -> float:
        """Apply bonuses based on detected patterns."""

        # Selling Climax bonuses
        if ReversalType.SELLING_CLIMAX in patterns:
            climax = pattern_data.get('climax', {})
            close_pos = climax.get('close_position', 0)
            vol_ratio = climax.get('volume_ratio', 0)

            if close_pos >= 0.7:
                score = min(100, score + 10)
            elif close_pos >= 0.6:
                score = min(100, score + 5)

            if vol_ratio >= 4.0:
                score = min(100, score + 8)
            elif vol_ratio >= 3.0:
                score = min(100, score + 5)

        # Mean Reversion bonuses
        if ReversalType.MEAN_REVERSION in patterns:
            mean_rev = pattern_data.get('mean_rev', {})

            if mean_rev.get('extreme_oversold'):
                score = min(100, score + 5)

            pattern = mean_rev.get('pattern', '')
            if pattern in ['bullish_engulfing', 'morning_star']:
                score = min(100, score + 3)

        # Wyckoff Spring bonuses
        if ReversalType.WYCKOFF_SPRING in patterns:
            spring = pattern_data.get('spring', {})
            close_pos = spring.get('close_position', 0)
            vol_ratio = spring.get('volume_ratio', 1.0)
            box_height = spring.get('box_height_pct', 0)

            if close_pos >= 0.7:
                score = min(100, score + 10)
            elif close_pos >= 0.6:
                score = min(100, score + 5)

            # Lower volume is better for springs
            if vol_ratio <= 0.8:
                score = min(100, score + 8)
            elif vol_ratio <= 1.0:
                score = min(100, score + 5)

            # Tight consolidation bonus
            if box_height <= 5.0:
                score = min(100, score + 5)

        return score

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        primary_pattern: ReversalType,
        pattern_data: Dict[str, Dict]
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """Calculate entry, stop, and take profit levels based on primary pattern."""

        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)
        atr = current_price * Decimal(str(atr_percent / 100))

        if primary_pattern == ReversalType.SELLING_CLIMAX:
            climax = pattern_data.get('climax', {})
            climax_low = Decimal(str(climax.get('low', float(current_price) * 0.95)))
            climax_high = Decimal(str(climax.get('high', float(current_price))))

            entry = max(current_price, climax_high)
            stop = climax_low - (atr * Decimal('0.5'))

        elif primary_pattern == ReversalType.WYCKOFF_SPRING:
            spring = pattern_data.get('spring', {})
            spring_low = Decimal(str(spring.get('low', float(current_price) * 0.95)))
            spring_high = Decimal(str(spring.get('high', float(current_price))))
            resistance = Decimal(str(spring.get('resistance', float(current_price) * 1.05)))

            entry = max(current_price, spring_high)
            stop = spring_low - (atr * Decimal('0.5'))

            # TP1 at resistance for spring
            risk = entry - stop
            tp1 = max(resistance, entry + (risk * Decimal(str(self.config.tp1_rr_ratio))))
            tp2 = entry + (risk * Decimal(str(self.config.tp2_rr_ratio)))

            # Apply max stop constraint
            max_stop_dist = entry * Decimal(str(self.config.max_stop_loss_pct / 100))
            if entry - stop > max_stop_dist:
                stop = entry - max_stop_dist

            return entry, stop, tp1, tp2

        else:  # Mean Reversion
            entry = current_price
            stop = entry - (atr * Decimal(str(self.config.atr_stop_multiplier)))

            # Target SMA20 if available
            sma_20 = float(candidate.get('sma_20') or 0)
            risk = entry - stop

            if sma_20 > 0 and Decimal(str(sma_20)) > entry:
                tp1 = Decimal(str(sma_20))
            else:
                tp1 = entry + (risk * Decimal(str(self.config.tp1_rr_ratio)))

            tp2 = entry + (risk * Decimal(str(self.config.tp2_rr_ratio)))

            # Apply max stop constraint
            max_stop_dist = entry * Decimal(str(self.config.max_stop_loss_pct / 100))
            if entry - stop > max_stop_dist:
                stop = entry - max_stop_dist

            return entry, stop, tp1, tp2

        # Default TP calculation for climax (fallback)
        risk = entry - stop
        tp1 = entry + (risk * Decimal(str(self.config.tp1_rr_ratio)))
        tp2 = entry + (risk * Decimal(str(self.config.tp2_rr_ratio)))

        # Apply max stop constraint
        max_stop_dist = entry * Decimal(str(self.config.max_stop_loss_pct / 100))
        if entry - stop > max_stop_dist:
            stop = entry - max_stop_dist

        return entry, stop, tp1, tp2

    def _calculate_atr_percent(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate ATR as percentage of price."""
        if len(closes) < period + 1:
            return 3.0

        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        atr = np.mean(tr[-period:])
        return (atr / closes[-1]) * 100 if closes[-1] > 0 else 3.0

    async def _get_daily_candles(self, ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get daily candles for a ticker."""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles_synthesized
            WHERE ticker = $1 AND timeframe = '1d'
            ORDER BY timestamp ASC
            LIMIT $2
        """
        rows = await self.db.fetch(query, ticker, limit)
        return [dict(r) for r in rows]

    async def _get_qualified_tickers(self, calculation_date: datetime) -> List[str]:
        """Get all active tickers for reversal scan."""
        # Scan ALL active stocks - no pre-filtering
        return await self.get_all_active_tickers()

    async def _get_candidate_data(
        self,
        ticker: str,
        calculation_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get full candidate data from watchlist and metrics."""
        query = """
            SELECT
                w.*,
                m.rsi_14, m.rsi_signal,
                m.macd_histogram, m.macd_cross_signal,
                m.trend_strength, m.relative_volume,
                m.atr_percent, m.current_price, m.sma_20
            FROM stock_watchlist w
            JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            WHERE w.ticker = $1 AND w.calculation_date = $2
        """
        row = await self.db.fetchrow(query, ticker, calculation_date)
        return dict(row) if row else None

    def _build_description(
        self,
        ticker: str,
        patterns: List[ReversalType],
        pattern_data: Dict[str, Dict],
        candidate: Dict[str, Any]
    ) -> str:
        """Build human-readable setup description."""
        pattern_names = [p.value.replace('_', ' ').title() for p in patterns]

        desc = f"{ticker} reversal detected: {', '.join(pattern_names)}. "

        # Add pattern-specific details
        if ReversalType.SELLING_CLIMAX in patterns:
            climax = pattern_data.get('climax', {})
            vol_ratio = climax.get('volume_ratio', 0)
            close_pos = climax.get('close_position', 0)
            desc += f"Climax: {vol_ratio:.1f}x volume, {close_pos:.0%} close. "

        if ReversalType.WYCKOFF_SPRING in patterns:
            spring = pattern_data.get('spring', {})
            box_height = spring.get('box_height_pct', 0)
            desc += f"Spring from {box_height:.1f}% box. "

        if ReversalType.MEAN_REVERSION in patterns:
            mean_rev = pattern_data.get('mean_rev', {})
            rsi = mean_rev.get('rsi', 50)
            desc += f"RSI {rsi:.0f}. "

        return desc.strip()

    def _build_confluence_factors(
        self,
        patterns: List[ReversalType],
        pattern_data: Dict[str, Dict],
        candidate: Dict[str, Any]
    ) -> List[str]:
        """Build list of confluence factors."""
        factors = []

        # Pattern counts
        if len(patterns) >= 2:
            factors.append(f"Multiple Reversal Types ({len(patterns)})")

        # Selling Climax factors
        if ReversalType.SELLING_CLIMAX in patterns:
            climax = pattern_data.get('climax', {})
            factors.append("Selling Climax Pattern")

            close_pos = climax.get('close_position', 0)
            if close_pos >= 0.7:
                factors.append("Strong Reversal Close (70%+)")

            vol_ratio = climax.get('volume_ratio', 0)
            if vol_ratio >= 4.0:
                factors.append(f"Extreme Volume ({vol_ratio:.1f}x)")
            elif vol_ratio >= 2.5:
                factors.append(f"Volume Spike ({vol_ratio:.1f}x)")

        # Wyckoff Spring factors
        if ReversalType.WYCKOFF_SPRING in patterns:
            spring = pattern_data.get('spring', {})
            factors.append("Wyckoff Spring Pattern")

            vol_ratio = spring.get('volume_ratio', 1.0)
            if vol_ratio <= 0.8:
                factors.append(f"Light Volume ({vol_ratio:.1f}x)")

            box_height = spring.get('box_height_pct', 0)
            if box_height <= 5.0:
                factors.append(f"Tight Consolidation ({box_height:.1f}%)")

            support = spring.get('support', 0)
            if support > 0:
                factors.append(f"Support: ${support:.2f}")

        # Mean Reversion factors
        if ReversalType.MEAN_REVERSION in patterns:
            mean_rev = pattern_data.get('mean_rev', {})
            rsi = mean_rev.get('rsi', 50)

            if rsi <= 25:
                factors.append(f"Extreme Oversold RSI ({rsi:.0f})")
            else:
                factors.append(f"RSI Oversold ({rsi:.0f})")

            pattern = mean_rev.get('pattern', '')
            if pattern:
                factors.append(f"{pattern.replace('_', ' ').title()} Pattern")

        return factors

    def _determine_market_regime(
        self,
        patterns: List[ReversalType],
        pattern_data: Dict[str, Dict],
        candidate: Dict[str, Any]
    ) -> str:
        """Determine market regime based on detected patterns."""

        if len(patterns) >= 2:
            regime = "Multi-Confluence Reversal"
        elif ReversalType.SELLING_CLIMAX in patterns:
            regime = "Capitulation"
        elif ReversalType.WYCKOFF_SPRING in patterns:
            regime = "Wyckoff Accumulation"
        else:
            regime = "Oversold Bounce"

        # Add context
        if ReversalType.SELLING_CLIMAX in patterns:
            vol_ratio = pattern_data.get('climax', {}).get('volume_ratio', 0)
            if vol_ratio >= 4.0:
                regime += " - Extreme Volume"

        if ReversalType.WYCKOFF_SPRING in patterns:
            box_height = pattern_data.get('spring', {}).get('box_height_pct', 0)
            if box_height <= 5.0:
                regime += " - Tight Range"

        return regime

    def log_scan_summary(
        self,
        candidates_count: int,
        signals_count: int,
        high_quality_count: int
    ) -> None:
        """Log scan summary."""
        logger.info(f"\n{self.scanner_name} scan complete:")
        logger.info(f"  Stocks scanned: {candidates_count}")
        logger.info(f"  Reversal signals: {signals_count}")
        logger.info(f"  High quality (A/A+): {high_quality_count}")
