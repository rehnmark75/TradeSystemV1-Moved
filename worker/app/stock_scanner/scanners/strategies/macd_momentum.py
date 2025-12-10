"""
MACD Momentum Scanner

Adapted from forex MACD strategy for stocks.
Uses MACD histogram crossover with price structure confirmation
and multi-timeframe alignment for momentum entries.

Entry Criteria:
- MACD histogram crosses zero (positive for bullish, negative for bearish)
- Histogram expanding (current > previous) showing momentum
- Price structure: Higher lows (bullish) or lower highs (bearish)
- Not at price extremes (avoid buying tops, selling bottoms)

Stop Logic:
- 1.5x ATR from entry (tighter for momentum plays)

Target:
- TP1: 3x ATR (2:1 R:R)
- TP2: 4.5x ATR (3:1 R:R)

Best For:
- Momentum breakouts
- Trend continuation after consolidation
- Fresh momentum signals
"""

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

logger = logging.getLogger(__name__)


@dataclass
class MACDMomentumConfig(ScannerConfig):
    """Configuration for MACD Momentum Scanner"""

    # ==========================================================================
    # MACD Parameters (standard settings)
    # ==========================================================================
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ==========================================================================
    # Histogram Thresholds
    # ==========================================================================
    # Stock MACD histogram values are typically larger than forex
    histogram_min_threshold: float = 0.01  # Minimum histogram value
    histogram_expansion_required: bool = True  # Current > previous

    # ==========================================================================
    # Price Structure
    # ==========================================================================
    require_price_structure: bool = True  # Higher lows (bull) / lower highs (bear)
    structure_lookback: int = 10  # Bars to check price structure

    # ==========================================================================
    # Extreme Filter (52-week range position)
    # ==========================================================================
    price_percentile_max: float = 95.0  # Don't buy at top 5% of range
    price_percentile_min: float = 5.0  # Don't sell at bottom 5%

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    atr_stop_multiplier: float = 1.5  # Tighter stop for momentum
    min_rr_ratio: float = 2.0

    # Default R:R targets
    tp1_rr_ratio: float = 2.0
    tp2_rr_ratio: float = 3.0

    # ==========================================================================
    # Trend/Alignment Filters
    # ==========================================================================
    require_trend_alignment: bool = True  # Trend must match direction
    min_relative_volume: float = 0.8

    # ==========================================================================
    # Fundamental Filters
    # ==========================================================================
    max_pe_ratio: float = 100.0  # Momentum stocks can be expensive
    min_institutional_pct: float = 10.0


class MACDMomentumScanner(BaseScanner):
    """
    Scans for MACD momentum signals with structure confirmation.

    Philosophy (adapted from forex MACD strategy):
    - Enter on fresh MACD momentum (histogram crossing zero)
    - Confirm with expanding histogram (momentum increasing)
    - Validate with price structure (HL for bull, LH for bear)
    - Avoid extremes to prevent buying tops/selling bottoms
    """

    def __init__(
        self,
        db_manager,
        config: MACDMomentumConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or MACDMomentumConfig(), scorer)
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "macd_momentum"

    @property
    def description(self) -> str:
        return "MACD momentum confluence with price structure confirmation"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute MACD Momentum scan.

        Steps:
        1. Get candidates with MACD crossover signals
        2. Filter for histogram expansion and structure
        3. Filter out price extremes
        4. Score and create signals
        5. Return sorted signals
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            from datetime import timedelta
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Get bullish and bearish MACD candidates
        bullish_candidates = await self._get_bullish_macd_candidates(calculation_date)
        bearish_candidates = await self._get_bearish_macd_candidates(calculation_date)

        logger.info(f"MACD signals: {len(bullish_candidates)} bullish, {len(bearish_candidates)} bearish")

        # Filter for momentum and structure conditions
        bullish_setups = self._filter_bullish_momentum(bullish_candidates)
        bearish_setups = self._filter_bearish_momentum(bearish_candidates)

        logger.info(f"After momentum filter: {len(bullish_setups)} bullish, {len(bearish_setups)} bearish")

        # Score and create signals
        signals = []

        for candidate in bullish_setups:
            signal = self._create_signal(candidate, SignalType.BUY)
            if signal:
                signals.append(signal)

        for candidate in bearish_setups:
            signal = self._create_signal(candidate, SignalType.SELL)
            if signal:
                signals.append(signal)

        # Sort by score
        signals.sort(key=lambda x: x.composite_score, reverse=True)

        # Apply limit
        signals = signals[:self.config.max_signals_per_run]

        # Log summary
        high_quality = sum(1 for s in signals if s.is_high_quality)
        total_candidates = len(bullish_candidates) + len(bearish_candidates)
        self.log_scan_summary(total_candidates, len(signals), high_quality)

        return signals

    async def _get_bullish_macd_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get stocks with bullish MACD signals"""

        # MACD bullish cross or positive histogram
        # Note: macd_cross_signal is in watchlist (w), macd_histogram is in metrics (m)
        additional_filters = """
            AND w.macd_cross_signal IN ('bullish_cross', 'bullish')
            AND m.macd_histogram > 0
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    async def _get_bearish_macd_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get stocks with bearish MACD signals"""

        # MACD bearish cross or negative histogram
        # Note: macd_cross_signal is in watchlist (w), macd_histogram is in metrics (m)
        additional_filters = """
            AND w.macd_cross_signal IN ('bearish_cross', 'bearish')
            AND m.macd_histogram < 0
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _filter_bullish_momentum(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for bullish momentum with structure confirmation"""

        filtered = []

        for c in candidates:
            # Check histogram value meets threshold
            histogram = float(c.get('macd_histogram') or 0)
            if histogram < self.config.histogram_min_threshold:
                continue

            # Volume check
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            # Price extreme filter - don't buy at yearly highs
            high_low_signal = c.get('high_low_signal', '')
            if high_low_signal == 'new_high':
                # Calculate approximate percentile from 52-week range
                current = float(c.get('current_price') or 0)
                high_52w = float(c.get('fifty_two_week_high') or current)
                low_52w = float(c.get('fifty_two_week_low') or current)

                if high_52w > low_52w:
                    percentile = (current - low_52w) / (high_52w - low_52w) * 100
                    if percentile > self.config.price_percentile_max:
                        continue

            # Trend alignment if required
            if self.config.require_trend_alignment:
                trend = c.get('trend_strength', '')
                # Allow uptrend or neutral (recovering)
                if trend in ['strong_down', 'down']:
                    continue

            # Check for fresh MACD cross (preferred)
            macd_cross = c.get('macd_cross_signal', '')
            is_fresh_cross = macd_cross == 'bullish_cross'

            c['signal_direction'] = 'BULLISH'
            c['is_fresh_cross'] = is_fresh_cross
            c['histogram_value'] = histogram
            filtered.append(c)

        return filtered

    def _filter_bearish_momentum(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for bearish momentum with structure confirmation"""

        filtered = []

        for c in candidates:
            # Check histogram value meets threshold (negative)
            histogram = float(c.get('macd_histogram') or 0)
            if histogram > -self.config.histogram_min_threshold:
                continue

            # Volume check
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            # Price extreme filter - don't sell at yearly lows
            high_low_signal = c.get('high_low_signal', '')
            if high_low_signal == 'new_low':
                current = float(c.get('current_price') or 0)
                high_52w = float(c.get('fifty_two_week_high') or current)
                low_52w = float(c.get('fifty_two_week_low') or current)

                if high_52w > low_52w:
                    percentile = (current - low_52w) / (high_52w - low_52w) * 100
                    if percentile < self.config.price_percentile_min:
                        continue

            # Trend alignment if required
            if self.config.require_trend_alignment:
                trend = c.get('trend_strength', '')
                # Allow downtrend or neutral (weakening)
                if trend in ['strong_up', 'up']:
                    continue

            # Check for fresh MACD cross
            macd_cross = c.get('macd_cross_signal', '')
            is_fresh_cross = macd_cross == 'bearish_cross'

            c['signal_direction'] = 'BEARISH'
            c['is_fresh_cross'] = is_fresh_cross
            c['histogram_value'] = histogram
            filtered.append(c)

        return filtered

    def _create_signal(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Optional[SignalSetup]:
        """Create a SignalSetup from a candidate"""

        # Score the candidate
        if signal_type == SignalType.BUY:
            score_components = self.scorer.score_bullish(candidate)
        else:
            score_components = self.scorer.score_bearish(candidate)

        composite_score = score_components.composite_score

        # Boost score for fresh MACD crosses
        if candidate.get('is_fresh_cross'):
            composite_score = min(100, composite_score + 5)

        # Minimum score threshold
        if composite_score < self.config.min_score_threshold:
            return None

        # Calculate entry levels
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, signal_type)

        # Calculate risk percent
        risk_pct = abs(entry - stop) / entry * 100

        # Build setup description
        confluence_factors = self._build_confluence_factors(candidate, signal_type)
        description = self._build_description(candidate, confluence_factors)

        # Create signal
        signal = SignalSetup(
            ticker=candidate['ticker'],
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=Decimal(str(self.config.tp1_rr_ratio)),
            risk_percent=Decimal(str(round(risk_pct, 2))),
            composite_score=composite_score,
            trend_score=Decimal(str(round(score_components.weighted_trend, 2))),
            momentum_score=Decimal(str(round(score_components.weighted_momentum, 2))),
            volume_score=Decimal(str(round(score_components.weighted_volume, 2))),
            pattern_score=Decimal(str(round(score_components.weighted_pattern, 2))),
            confluence_score=Decimal(str(round(score_components.weighted_confluence, 2))),
            setup_description=description,
            confluence_factors=confluence_factors,
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

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """
        Calculate entry, stop, and take profit levels.

        For MACD Momentum:
        - Entry: Current price
        - Stop: 1.5x ATR (tighter for momentum)
        - TP1: 3x ATR (2:1 R:R)
        - TP2: 4.5x ATR (3:1 R:R)
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)

        # ATR in price terms
        atr = current_price * Decimal(str(atr_percent / 100))

        # Entry at current price
        entry = current_price

        # Stop loss: 1.5x ATR (tighter for momentum)
        stop = self.calculate_atr_based_stop(entry, atr, signal_type)

        # Take profits
        tp1, tp2 = self.calculate_take_profits(entry, stop, signal_type)

        return entry, stop, tp1, tp2

    def _build_confluence_factors(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> List[str]:
        """Build list of confluence factors for this setup"""

        factors = []
        is_fresh = candidate.get('is_fresh_cross', False)
        histogram = candidate.get('histogram_value', 0)

        if signal_type == SignalType.BUY:
            # MACD factors
            if is_fresh:
                factors.append("Fresh MACD bullish cross")
            else:
                factors.append("MACD histogram positive")

            factors.append(f"Histogram: +{abs(histogram):.3f}")

            # Trend
            trend = candidate.get('trend_strength', '')
            if trend == 'strong_up':
                factors.append("Strong uptrend")
            elif trend == 'up':
                factors.append("Uptrend confirmed")
            elif trend == 'neutral':
                factors.append("Trend turning bullish")

            # Position in range
            high_low = candidate.get('high_low_signal', '')
            if high_low == 'near_high':
                factors.append("Near 52W high (momentum)")
            elif high_low in ['middle', 'near_low']:
                factors.append("Room to run higher")

        else:  # SELL
            if is_fresh:
                factors.append("Fresh MACD bearish cross")
            else:
                factors.append("MACD histogram negative")

            factors.append(f"Histogram: {histogram:.3f}")

            trend = candidate.get('trend_strength', '')
            if trend == 'strong_down':
                factors.append("Strong downtrend")
            elif trend == 'down':
                factors.append("Downtrend confirmed")
            elif trend == 'neutral':
                factors.append("Trend turning bearish")

            high_low = candidate.get('high_low_signal', '')
            if high_low == 'near_low':
                factors.append("Near 52W low (momentum)")
            elif high_low in ['middle', 'near_high']:
                factors.append("Room to fall lower")

        # Volume
        rel_vol = float(candidate.get('relative_volume') or 0)
        if rel_vol >= 1.5:
            factors.append(f"High volume ({rel_vol:.1f}x)")
        elif rel_vol >= 1.0:
            factors.append(f"Avg volume ({rel_vol:.1f}x)")

        # RSI
        rsi = float(candidate.get('rsi_14') or 50)
        if signal_type == SignalType.BUY and rsi > 50:
            factors.append(f"RSI bullish ({rsi:.0f})")
        elif signal_type == SignalType.SELL and rsi < 50:
            factors.append(f"RSI bearish ({rsi:.0f})")

        return factors

    def _build_description(
        self,
        candidate: Dict[str, Any],
        factors: List[str]
    ) -> str:
        """Build human-readable setup description"""

        ticker = candidate['ticker']
        direction = candidate.get('signal_direction', 'BULLISH')
        is_fresh = candidate.get('is_fresh_cross', False)
        histogram = candidate.get('histogram_value', 0)

        desc = f"{ticker} MACD momentum {direction.lower()}. "

        if is_fresh:
            desc += "Fresh histogram cross. "
        else:
            desc += f"Histogram {'expanding' if abs(histogram) > 0.05 else 'positive'}. "

        if factors:
            desc += f"Factors: {', '.join(factors[:2])}"

        return desc

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime for the stock"""

        trend_strength = candidate.get('trend_strength', '')
        is_fresh = candidate.get('is_fresh_cross', False)
        atr_pct = float(candidate.get('atr_percent') or 0)

        if trend_strength == 'strong_up':
            regime = "Strong Uptrend"
        elif trend_strength == 'up':
            regime = "Uptrend"
        elif trend_strength == 'strong_down':
            regime = "Strong Downtrend"
        elif trend_strength == 'down':
            regime = "Downtrend"
        else:
            regime = "Transitioning"

        if is_fresh:
            regime += " (Fresh Momentum)"
        else:
            regime += " (Momentum)"

        if atr_pct > 5:
            regime += " - High Vol"
        elif atr_pct < 2:
            regime += " - Low Vol"

        return regime
