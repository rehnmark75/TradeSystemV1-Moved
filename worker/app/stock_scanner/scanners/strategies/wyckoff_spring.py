"""
Wyckoff Spring Scanner

Adapted from AlphaSuite's wyckoff_spring.py pattern.
Identifies potential accumulation phases by detecting "spring" patterns -
temporary price dips below support that quickly reverse, suggesting weak selling pressure.

Entry Criteria:
- Stock in tight consolidation range (box) for N days
- Price dips below support (box low) intraday
- Close recovers above support level
- Close in upper 50%+ of day's range (bullish reversal)
- Volume not excessive (weak selling pressure)

Stop Logic:
- Below spring low
- 1.5x ATR buffer

Target:
- TP1: Top of consolidation box
- TP2: 2x box height above resistance

Best For:
- Accumulation phase detection
- Support test and reversal
- Wyckoff methodology entries
"""

import numpy as np
import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..base_scanner import (
    BaseScanner, SignalSetup, ScannerConfig,
    SignalType, QualityTier
)
from ..scoring import SignalScorer
from ..exclusion_filters import ExclusionFilterEngine, get_mean_reversion_filter

logger = logging.getLogger(__name__)


@dataclass
class WyckoffSpringConfig(ScannerConfig):
    """Configuration for Wyckoff Spring Scanner"""

    # ==========================================================================
    # Consolidation Box Parameters
    # ==========================================================================
    box_lookback: int = 20  # Days to calculate consolidation box
    max_box_height_pct: float = 10.0  # Max box height as % of price
    min_box_height_pct: float = 2.0  # Min box height (avoid dead stocks)

    # ==========================================================================
    # Spring Detection Parameters
    # ==========================================================================
    min_close_position: float = 0.5  # Close must be in upper 50% of range
    max_volume_ratio: float = 1.2  # Volume not excessive (weak selling)
    setup_window: int = 3  # Look for spring in last N days

    # ==========================================================================
    # Volume Requirements
    # ==========================================================================
    volume_avg_period: int = 20
    min_relative_volume: float = 0.5  # Can be lower for springs

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    atr_stop_multiplier: float = 1.5
    max_stop_loss_pct: float = 10.0  # Wider for accumulation plays

    # Default R:R targets
    tp1_rr_ratio: float = 2.0
    tp2_rr_ratio: float = 3.0

    # ==========================================================================
    # Fundamental Filters
    # ==========================================================================
    max_pe_ratio: float = 60.0
    min_profit_margin: float = 0.0
    max_debt_to_equity: float = 3.0
    days_to_earnings_min: int = 7


class WyckoffSpringScanner(BaseScanner):
    """
    Scans for Wyckoff Spring accumulation patterns.

    Philosophy:
    - Springs occur when price temporarily breaks below support but quickly recovers
    - Weak volume on the spring indicates sellers are exhausted
    - Strong close indicates buyers absorbing supply
    - Classic Wyckoff accumulation signal before markup phase
    """

    def __init__(
        self,
        db_manager,
        config: WyckoffSpringConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or WyckoffSpringConfig(), scorer)
        self.exclusion_filter = get_mean_reversion_filter()
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "wyckoff_spring"

    @property
    def description(self) -> str:
        return "Wyckoff accumulation spring with support test and reversal"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute Wyckoff Spring scan.

        Steps:
        1. Get qualified tickers from watchlist
        2. Fetch daily candles for each
        3. Identify consolidation boxes
        4. Detect spring patterns (support break + recovery)
        5. Score and create signals
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            from datetime import timedelta
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Ensure calculation_date is a date object
        if hasattr(calculation_date, 'date'):
            calculation_date = calculation_date.date()

        # Get qualified tickers
        tickers = await self._get_qualified_tickers(calculation_date)
        logger.info(f"Scanning {len(tickers)} qualified stocks for Wyckoff Spring")

        signals = []

        for ticker in tickers:
            signal = await self._scan_ticker(ticker, calculation_date)
            if signal:
                signals.append(signal)
                logger.info(f"  SPRING: {ticker} @ {signal.entry_price}")

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
        """Scan a single ticker for Wyckoff Spring pattern."""

        # Get daily candles
        candles = await self._get_daily_candles(ticker, limit=100)

        if len(candles) < self.config.box_lookback + 10:
            return None

        # Convert to numpy arrays
        closes = np.array([float(c['close']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])
        volumes = np.array([float(c['volume']) for c in candles])

        # Detect Wyckoff Spring
        spring_result = self._detect_spring(closes, highs, lows, volumes)

        if spring_result is None:
            return None

        spring_idx, support, resistance, close_position, volume_ratio = spring_result

        # Get candidate data for scoring
        candidate = await self._get_candidate_data(ticker, calculation_date)
        if not candidate:
            candidate = {
                'ticker': ticker,
                'current_price': closes[-1],
                'atr_percent': self._calculate_atr_percent(highs, lows, closes),
            }

        # Add spring-specific data
        candidate['spring_support'] = support
        candidate['spring_resistance'] = resistance
        candidate['spring_close_position'] = close_position
        candidate['spring_volume_ratio'] = volume_ratio
        candidate['spring_low'] = lows[spring_idx]
        candidate['spring_high'] = highs[spring_idx]
        candidate['box_height_pct'] = ((resistance - support) / support) * 100 if support > 0 else 0

        # Calculate entry levels
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, SignalType.BUY)

        # Score the signal
        score_components = self.scorer.score_bullish(candidate)
        composite_score = score_components.composite_score

        # Bonus for strong reversal close
        if close_position >= 0.7:
            composite_score = min(100, composite_score + 10)
        elif close_position >= 0.6:
            composite_score = min(100, composite_score + 5)

        # Bonus for low volume (weak selling)
        if volume_ratio <= 0.8:
            composite_score = min(100, composite_score + 8)
        elif volume_ratio <= 1.0:
            composite_score = min(100, composite_score + 5)

        # Bonus for tight consolidation
        box_height = candidate.get('box_height_pct', 0)
        if box_height <= 5.0:
            composite_score = min(100, composite_score + 5)

        # Apply minimum threshold
        if composite_score < self.config.min_score_threshold:
            return None

        # Calculate risk percent
        risk_pct = abs(float(entry) - float(stop)) / float(entry) * 100 if float(entry) > 0 else 0

        # Build description
        description = self._build_description(
            ticker, support, resistance, close_position, volume_ratio
        )

        # Create signal
        signal = SignalSetup(
            ticker=ticker,
            scanner_name=self.scanner_name,
            signal_type=SignalType.BUY,
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
            confluence_factors=self._build_confluence_factors(candidate),
            timeframe="daily",
            market_regime=self._determine_market_regime(candidate),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data=candidate,
        )

        # Calculate position size
        signal.suggested_position_size_pct = self.calculate_position_size(
            Decimal(str(self.config.max_risk_per_trade_pct)),
            entry,
            stop,
            signal.quality_tier
        )

        return signal

    def _detect_spring(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray
    ) -> Optional[Tuple[int, float, float, float, float]]:
        """
        Detect Wyckoff Spring pattern.

        Returns:
            Tuple of (spring_idx, support, resistance, close_position, volume_ratio)
            if found, None otherwise
        """
        box_lookback = self.config.box_lookback
        vol_period = self.config.volume_avg_period

        if len(closes) < box_lookback + vol_period:
            return None

        # Check last N days for spring
        for i in range(len(closes) - self.config.setup_window, len(closes)):
            if i < box_lookback + vol_period:
                continue

            # 1. Calculate consolidation box from prior period
            box_start = i - box_lookback
            box_end = i - 1  # Exclude current candle from box calculation

            box_highs = highs[box_start:box_end]
            box_lows = lows[box_start:box_end]

            support = np.min(box_lows)
            resistance = np.max(box_highs)

            # 2. Check box tightness
            if support <= 0:
                continue

            box_height_pct = ((resistance - support) / support) * 100

            if box_height_pct > self.config.max_box_height_pct:
                continue  # Box too wide - not consolidation

            if box_height_pct < self.config.min_box_height_pct:
                continue  # Box too tight - dead stock

            # 3. Check for spring: low penetrates support but close recovers
            if lows[i] >= support:
                continue  # No support penetration

            if closes[i] <= support:
                continue  # Close didn't recover above support

            # 4. Check reversal close position
            day_range = highs[i] - lows[i]
            if day_range <= 0:
                continue

            close_position = (closes[i] - lows[i]) / day_range

            if close_position < self.config.min_close_position:
                continue  # Close not in upper portion

            # 5. Check volume (should not be excessive - weak selling)
            avg_volume = np.mean(volumes[i - vol_period:i])
            if avg_volume <= 0:
                continue

            volume_ratio = volumes[i] / avg_volume

            if volume_ratio > self.config.max_volume_ratio:
                continue  # Too much volume - not weak selling

            # All conditions met - Wyckoff Spring detected!
            return (i, support, resistance, close_position, volume_ratio)

        return None

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """
        Calculate entry, stop, and take profit levels for Wyckoff Spring.

        Entry: Above spring candle high (confirmation)
        Stop: Below spring low
        TP1: Top of consolidation box (resistance)
        TP2: 2x box height above resistance
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        spring_low = Decimal(str(candidate.get('spring_low', current_price * Decimal('0.95'))))
        spring_high = Decimal(str(candidate.get('spring_high', current_price)))
        support = Decimal(str(candidate.get('spring_support', current_price * Decimal('0.95'))))
        resistance = Decimal(str(candidate.get('spring_resistance', current_price * Decimal('1.05'))))
        atr_percent = float(candidate.get('atr_percent') or 3.0)

        # Entry above spring high (confirmation of reversal)
        entry = max(current_price, spring_high)

        # Stop below spring low with small buffer
        atr = current_price * Decimal(str(atr_percent / 100))
        stop = spring_low - (atr * Decimal('0.5'))

        # Apply max stop constraint
        max_stop_distance = entry * Decimal(str(self.config.max_stop_loss_pct / 100))
        if entry - stop > max_stop_distance:
            stop = entry - max_stop_distance

        # TP1: Resistance (top of box)
        tp1 = resistance

        # If resistance is below entry (unusual), use R multiple
        if tp1 <= entry:
            risk = entry - stop
            tp1 = entry + (risk * Decimal(str(self.config.tp1_rr_ratio)))

        # TP2: 2x box height above resistance
        box_height = resistance - support
        tp2 = resistance + (box_height * Decimal('2'))

        # Ensure minimum R:R for TP2
        risk = entry - stop
        min_tp2 = entry + (risk * Decimal(str(self.config.tp2_rr_ratio)))
        tp2 = max(tp2, min_tp2)

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
        """Get tickers that meet minimum criteria for spring scan."""
        query = """
            SELECT w.ticker
            FROM stock_watchlist w
            JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            WHERE w.calculation_date = $1
              AND w.tier <= $2
              AND COALESCE(m.relative_volume, 0) >= $3
            ORDER BY w.rank_overall
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.max_tier,
            self.config.min_relative_volume
        )
        return [r['ticker'] for r in rows]

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
                m.atr_percent, m.current_price
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
        support: float,
        resistance: float,
        close_position: float,
        volume_ratio: float
    ) -> str:
        """Build human-readable setup description."""
        box_range = ((resistance - support) / support) * 100 if support > 0 else 0

        vol_desc = "light" if volume_ratio <= 0.8 else "moderate" if volume_ratio <= 1.0 else "average"
        close_desc = "strong" if close_position >= 0.7 else "solid"

        return (
            f"{ticker} Wyckoff Spring detected. "
            f"Price broke below ${support:.2f} support then recovered. "
            f"{close_desc.title()} close ({close_position:.0%} of range) on {vol_desc} volume ({volume_ratio:.1f}x). "
            f"Box range {box_range:.1f}%. Potential accumulation breakout."
        )

    def _build_confluence_factors(self, candidate: Dict[str, Any]) -> List[str]:
        """Build list of confluence factors."""
        factors = ["Wyckoff Spring Pattern"]

        # Close position
        close_pos = candidate.get('spring_close_position', 0)
        if close_pos >= 0.7:
            factors.append("Strong Reversal Close (70%+)")
        elif close_pos >= 0.5:
            factors.append("Reversal Close (50%+)")

        # Volume (lower is better for springs)
        vol_ratio = candidate.get('spring_volume_ratio', 1.0)
        if vol_ratio <= 0.8:
            factors.append(f"Light Volume ({vol_ratio:.1f}x)")
        elif vol_ratio <= 1.0:
            factors.append(f"Below Avg Volume ({vol_ratio:.1f}x)")

        # Box tightness
        box_height = candidate.get('box_height_pct', 0)
        if box_height <= 5.0:
            factors.append(f"Tight Consolidation ({box_height:.1f}%)")
        elif box_height <= 8.0:
            factors.append(f"Consolidation ({box_height:.1f}%)")

        # RSI
        rsi = float(candidate.get('rsi_14') or 50)
        if rsi <= 35:
            factors.append(f"RSI Oversold ({rsi:.0f})")
        elif rsi <= 45:
            factors.append(f"RSI Neutral ({rsi:.0f})")

        # Support level
        support = candidate.get('spring_support', 0)
        if support > 0:
            factors.append(f"Support: ${support:.2f}")

        return factors

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime."""
        box_height = candidate.get('box_height_pct', 0)
        vol_ratio = candidate.get('spring_volume_ratio', 1.0)

        regime = "Wyckoff Accumulation"

        if box_height <= 5.0:
            regime += " - Tight Range"
        elif box_height <= 8.0:
            regime += " - Consolidation"

        if vol_ratio <= 0.8:
            regime += " (Light Vol)"
        elif vol_ratio <= 1.0:
            regime += " (Low Vol)"

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
        logger.info(f"  Spring signals: {signals_count}")
        logger.info(f"  High quality (A/A+): {high_quality_count}")
