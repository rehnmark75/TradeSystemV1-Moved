"""
Selling Climax Scanner

Adapted from AlphaSuite's selling_climax.py pattern.
Identifies potential market bottoming points by detecting capitulation patterns
where panic selling is absorbed by strong buyers.

Entry Criteria:
- Price makes new 20-day low (capitulation)
- Volume spike > 2.5x 20-day average (panic selling)
- Close in upper 50% of day's range (absorption/reversal)

Stop Logic:
- Below the climax candle low
- 1.5x ATR for safety

Target:
- TP1: Prior support level or 2R
- TP2: 50% retracement of decline or 3R

Best For:
- Capitulation bottoming patterns
- Panic selling exhaustion
- High-volume reversal plays
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
class SellingClimaxConfig(ScannerConfig):
    """Configuration for Selling Climax Scanner"""

    # ==========================================================================
    # Climax Detection Parameters
    # ==========================================================================
    new_low_lookback: int = 20  # Period for new low detection
    volume_spike_multiplier: float = 2.5  # Volume must be 2.5x average
    min_close_position: float = 0.5  # Close must be in upper 50% of range
    setup_window: int = 3  # Look for climax in last N days

    # ==========================================================================
    # Volume Requirements
    # ==========================================================================
    volume_avg_period: int = 20  # Period for average volume calculation
    min_relative_volume: float = 1.0  # Minimum relative volume

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    atr_stop_multiplier: float = 1.5  # Stop below climax low
    max_stop_loss_pct: float = 8.0  # Max stop distance

    # Default R:R targets
    tp1_rr_ratio: float = 2.0
    tp2_rr_ratio: float = 3.0

    # ==========================================================================
    # Fundamental Filters (Focus on financial health)
    # ==========================================================================
    max_pe_ratio: float = 50.0  # Not too expensive
    min_profit_margin: float = 0.0  # Can be loss-making (panic oversold)
    max_debt_to_equity: float = 3.0  # Some debt OK for climax plays
    days_to_earnings_min: int = 7  # Avoid earnings


class SellingClimaxScanner(BaseScanner):
    """
    Scans for selling climax / capitulation patterns.

    Philosophy:
    - Capitulation creates opportunity when panic selling exhausts
    - Volume spike + new low + reversal close = potential bottom
    - Strong close in upper range signals buyers absorbing supply
    - Counter-trend play - use strict risk management
    """

    def __init__(
        self,
        db_manager,
        config: SellingClimaxConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or SellingClimaxConfig(), scorer)
        self.exclusion_filter = get_mean_reversion_filter()
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "selling_climax"

    @property
    def description(self) -> str:
        return "Capitulation bottoming with volume spike and reversal close"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute selling climax scan.

        Steps:
        1. Get qualified tickers from watchlist
        2. Fetch daily candles for each
        3. Detect climax pattern (new low + volume spike + reversal close)
        4. Score and create signals
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
        logger.info(f"Scanning {len(tickers)} qualified stocks for selling climax")

        signals = []

        for ticker in tickers:
            signal = await self._scan_ticker(ticker, calculation_date)
            if signal:
                signals.append(signal)
                logger.info(f"  CLIMAX: {ticker} @ {signal.entry_price}")

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
        """Scan a single ticker for selling climax pattern."""

        # Get daily candles
        candles = await self._get_daily_candles(ticker, limit=100)

        if len(candles) < self.config.new_low_lookback + 10:
            return None

        # Convert to numpy arrays for analysis
        closes = np.array([float(c['close']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])
        volumes = np.array([float(c['volume']) for c in candles])

        # Detect selling climax
        climax_result = self._detect_selling_climax(closes, highs, lows, volumes)

        if climax_result is None:
            return None

        climax_idx, close_position_pct = climax_result

        # Get candidate data for scoring
        candidate = await self._get_candidate_data(ticker, calculation_date)
        if not candidate:
            candidate = {
                'ticker': ticker,
                'current_price': closes[-1],
                'atr_percent': self._calculate_atr_percent(highs, lows, closes),
            }

        # Add climax-specific data
        candidate['climax_close_position'] = close_position_pct
        candidate['climax_volume_ratio'] = volumes[-1] / np.mean(volumes[-self.config.volume_avg_period:]) if len(volumes) > self.config.volume_avg_period else 2.0
        candidate['climax_low'] = lows[climax_idx]
        candidate['climax_high'] = highs[climax_idx]

        # Calculate entry levels
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, SignalType.BUY)

        # Score the signal
        score_components = self.scorer.score_bullish(candidate)
        composite_score = score_components.composite_score

        # Bonus for strong reversal close
        if close_position_pct >= 0.7:
            composite_score = min(100, composite_score + 10)
        elif close_position_pct >= 0.6:
            composite_score = min(100, composite_score + 5)

        # Bonus for extreme volume
        vol_ratio = candidate.get('climax_volume_ratio', 0)
        if vol_ratio >= 4.0:
            composite_score = min(100, composite_score + 8)
        elif vol_ratio >= 3.0:
            composite_score = min(100, composite_score + 5)

        # Apply minimum threshold
        if composite_score < self.config.min_score_threshold:
            return None

        # Calculate risk percent
        risk_pct = abs(float(entry) - float(stop)) / float(entry) * 100 if float(entry) > 0 else 0

        # Build description
        description = self._build_description(ticker, close_position_pct, vol_ratio)

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

        # Calculate position size (reduced for reversal plays)
        signal.suggested_position_size_pct = self.calculate_position_size(
            Decimal(str(self.config.max_risk_per_trade_pct * 0.8)),
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

        Returns:
            Tuple of (climax_index, close_position_pct) if found, None otherwise
        """
        lookback = self.config.new_low_lookback
        vol_period = self.config.volume_avg_period

        if len(closes) < lookback + vol_period:
            return None

        # Check last N days for climax
        for i in range(len(closes) - self.config.setup_window, len(closes)):
            if i < lookback + vol_period:
                continue

            # 1. Check for new low (lowest close in lookback period)
            prior_lows = lows[i - lookback:i]
            if lows[i] >= min(prior_lows):
                continue  # Not a new low

            # 2. Check for volume spike
            avg_volume = np.mean(volumes[i - vol_period:i])
            if volumes[i] < avg_volume * self.config.volume_spike_multiplier:
                continue  # Not enough volume

            # 3. Check for reversal close (close in upper portion of range)
            day_range = highs[i] - lows[i]
            if day_range <= 0:
                continue

            close_position = (closes[i] - lows[i]) / day_range

            if close_position < self.config.min_close_position:
                continue  # Close not in upper half

            # All conditions met - selling climax detected!
            return (i, close_position)

        return None

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """
        Calculate entry, stop, and take profit levels for selling climax.

        Entry: Above climax candle high (confirmation)
        Stop: Below climax candle low
        TP1: 2R
        TP2: 3R
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        climax_low = Decimal(str(candidate.get('climax_low', current_price * Decimal('0.95'))))
        climax_high = Decimal(str(candidate.get('climax_high', current_price)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)

        # Entry above climax high (confirmation of reversal)
        entry = max(current_price, climax_high)

        # Stop below climax low
        stop = climax_low - (climax_low * Decimal(str(atr_percent / 200)))  # Small buffer

        # Apply max stop constraint
        max_stop_distance = entry * Decimal(str(self.config.max_stop_loss_pct / 100))
        if entry - stop > max_stop_distance:
            stop = entry - max_stop_distance

        # Take profits based on R multiples
        risk = entry - stop
        tp1 = entry + (risk * Decimal(str(self.config.tp1_rr_ratio)))
        tp2 = entry + (risk * Decimal(str(self.config.tp2_rr_ratio)))

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
            return 3.0  # Default

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
        """Get tickers that meet minimum criteria for climax scan."""
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
        close_position: float,
        volume_ratio: float
    ) -> str:
        """Build human-readable setup description."""
        strength = "strong" if close_position >= 0.7 else "moderate"

        return (
            f"{ticker} selling climax detected. "
            f"New low with {volume_ratio:.1f}x volume spike. "
            f"{strength.title()} reversal close ({close_position:.0%} of range). "
            f"Potential capitulation bottom."
        )

    def _build_confluence_factors(self, candidate: Dict[str, Any]) -> List[str]:
        """Build list of confluence factors."""
        factors = ["Selling Climax Pattern"]

        # Close position
        close_pos = candidate.get('climax_close_position', 0)
        if close_pos >= 0.7:
            factors.append("Strong Reversal Close (70%+)")
        elif close_pos >= 0.5:
            factors.append("Reversal Close (50%+)")

        # Volume
        vol_ratio = candidate.get('climax_volume_ratio', 0)
        if vol_ratio >= 4.0:
            factors.append(f"Extreme Volume ({vol_ratio:.1f}x)")
        elif vol_ratio >= 2.5:
            factors.append(f"Volume Spike ({vol_ratio:.1f}x)")

        # RSI
        rsi = float(candidate.get('rsi_14') or 50)
        if rsi <= 30:
            factors.append(f"RSI Oversold ({rsi:.0f})")
        elif rsi <= 40:
            factors.append(f"RSI Low ({rsi:.0f})")

        # Trend context
        trend = candidate.get('trend_strength', '')
        if 'down' in str(trend).lower():
            factors.append("Downtrend Context")

        return factors

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime."""
        trend = candidate.get('trend_strength', '')
        vol_ratio = candidate.get('climax_volume_ratio', 0)

        regime = "Capitulation"

        if vol_ratio >= 4.0:
            regime += " - Extreme Volume"
        elif vol_ratio >= 3.0:
            regime += " - High Volume"

        if 'strong' in str(trend).lower():
            regime += " (Strong Trend)"
        elif 'down' in str(trend).lower():
            regime += " (Downtrend)"

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
        logger.info(f"  Climax signals: {signals_count}")
        logger.info(f"  High quality (A/A+): {high_quality_count}")
