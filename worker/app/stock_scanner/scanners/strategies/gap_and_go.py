"""
Gap & Go Scanner

Identifies gap continuation opportunities with catalyst confirmation.

Entry Criteria:
- Gap up > 2% (large gap > 4%)
- High pre-market/early volume (> 1.5x average)
- Gap doesn't fill in first hour (holds above gap open)
- Bullish trend context (not gapping into resistance)

Stop Logic:
- Below gap open (low of gap candle)
- Or below pre-market low

Target:
- TP1: Gap extension (50-100% of gap size)
- TP2: Next resistance or measured move

Best For:
- Earnings beats
- News catalysts
- Sector momentum days
- Opening range breakouts
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
from ..exclusion_filters import ExclusionFilterEngine, FilterConfig

logger = logging.getLogger(__name__)


@dataclass
class GapAndGoConfig(ScannerConfig):
    """Configuration specific to Gap & Go Scanner"""

    # Gap requirements
    min_gap_pct: float = 2.0  # Minimum 2% gap
    large_gap_pct: float = 4.0  # Large gap threshold
    max_gap_pct: float = 15.0  # Avoid extreme gaps (risky)

    # Volume requirements
    min_relative_volume: float = 1.5  # Strong volume required
    extreme_volume_threshold: float = 3.0

    # Trend context
    prefer_bullish_trend: bool = True
    avoid_resistance: bool = True  # Don't gap into 52W high resistance

    # Risk
    atr_stop_multiplier: float = 1.5
    max_stop_loss_pct: float = 5.0  # Tighter stops for gaps


class GapAndGoScanner(BaseScanner):
    """
    Scans for gap continuation opportunities.

    Philosophy:
    - Gaps represent overnight information or catalyst
    - Volume confirms institutional interest
    - Gaps in direction of trend more reliable
    - Quick decisions needed - gaps are time-sensitive
    """

    def __init__(
        self,
        db_manager,
        config: GapAndGoConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or GapAndGoConfig(), scorer)
        self.exclusion_filter = ExclusionFilterEngine(FilterConfig(
            max_tier=3,
            min_relative_volume=1.2,
            max_atr_percent=15.0,
            min_score_for_trade=50,
        ))
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "gap_and_go"

    @property
    def description(self) -> str:
        return "Gap continuation plays with catalyst confirmation"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute gap & go scan.

        Steps:
        1. Get candidates with gaps
        2. Filter by gap size and direction
        3. Check volume confirmation
        4. Score and apply filters
        5. Calculate entry/exit levels
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            from datetime import timedelta
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Get gap candidates
        candidates = await self._get_gap_candidates(calculation_date)
        logger.info(f"Found {len(candidates)} gap candidates")

        if not candidates:
            return []

        # Filter for tradeable gaps
        tradeable = self._filter_tradeable_gaps(candidates)
        logger.info(f"Found {len(tradeable)} tradeable gaps")

        # Score and create signals
        signals = []
        for candidate in tradeable:
            signal = self._create_signal(candidate)
            if signal:
                signals.append(signal)

        # Sort by score
        signals.sort(key=lambda x: x.composite_score, reverse=True)

        # Apply limit
        signals = signals[:self.config.max_signals_per_run]

        # Log summary
        high_quality = sum(1 for s in signals if s.is_high_quality)
        self.log_scan_summary(len(candidates), len(signals), high_quality)

        return signals

    async def _get_gap_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get stocks with gap ups"""

        additional_filters = """
            AND w.gap_signal IN ('gap_up', 'gap_up_large')
            AND w.relative_volume >= {min_vol}
        """.format(min_vol=self.config.min_relative_volume)

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _filter_tradeable_gaps(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for tradeable gap setups"""

        tradeable = []

        for c in candidates:
            # Check gap size from price change
            price_change_1d = float(c.get('price_change_1d') or 0)

            # Validate gap size
            if price_change_1d < self.config.min_gap_pct:
                continue

            if price_change_1d > self.config.max_gap_pct:
                # Extreme gaps are risky
                logger.debug(f"Skipping {c['ticker']}: gap too large ({price_change_1d:.1f}%)")
                continue

            # Volume confirmation
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            # Trend context preference
            if self.config.prefer_bullish_trend:
                sma_cross = c.get('sma_cross_signal', '')
                # Don't require bullish, but death cross is bad
                if sma_cross == 'death_cross':
                    continue

            # Avoid gapping into major resistance
            if self.config.avoid_resistance:
                high_low = c.get('high_low_signal', '')
                rsi_signal = c.get('rsi_signal', '')

                # Already at 52W high + overbought = risky
                if high_low == 'new_high' and rsi_signal in ['overbought', 'overbought_extreme']:
                    continue

            tradeable.append(c)

        return tradeable

    def _create_signal(
        self,
        candidate: Dict[str, Any]
    ) -> Optional[SignalSetup]:
        """Create a SignalSetup from a gap candidate"""

        # Score the candidate
        score_components = self.scorer.score_bullish(candidate)
        composite_score = score_components.composite_score

        # Gap size bonus
        price_change_1d = float(candidate.get('price_change_1d') or 0)
        if price_change_1d >= self.config.large_gap_pct:
            composite_score = min(100, composite_score + 5)

        # Extreme volume bonus
        rel_vol = float(candidate.get('relative_volume') or 0)
        if rel_vol >= self.config.extreme_volume_threshold:
            composite_score = min(100, composite_score + 5)

        # Trend alignment bonus
        sma_cross = candidate.get('sma_cross_signal', '')
        if sma_cross in ['golden_cross', 'bullish']:
            composite_score = min(100, composite_score + 3)

        # Apply exclusion filter
        exclusion = self.exclusion_filter.check_bullish_exclusions(
            candidate, composite_score
        )

        if exclusion.is_excluded:
            logger.debug(
                f"Excluded {candidate['ticker']}: {exclusion.summary}"
            )
            return None

        # Calculate entry levels
        entry, stop, tp1, tp2 = self._calculate_entry_levels(
            candidate, SignalType.BUY
        )

        # Calculate risk percent
        risk_pct = abs(entry - stop) / entry * 100

        # Build setup description
        confluence_factors = self.build_confluence_factors(candidate)
        gap_type = "Large Gap" if price_change_1d >= self.config.large_gap_pct else "Gap"
        confluence_factors.insert(0, f'{gap_type} +{price_change_1d:.1f}%')
        description = self._build_description(candidate, confluence_factors)

        # Create signal
        signal = SignalSetup(
            ticker=candidate['ticker'],
            scanner_name=self.scanner_name,
            signal_type=SignalType.BUY,
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
            market_regime="Gap Play",
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
        Calculate entry, stop, and take profit levels for gaps.

        Gap strategy:
        - Entry: Current price or slightly above gap candle high
        - Stop: Below gap open (gap must hold)
        - Target: Gap extension
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)
        price_change_1d = float(candidate.get('price_change_1d') or 2.0)

        # ATR in price terms
        atr = current_price * Decimal(str(atr_percent / 100))

        # Entry at current price
        entry = current_price

        # Stop loss: Below gap open
        # Gap open approximation: current_price / (1 + change%)
        gap_open = current_price / (1 + Decimal(str(price_change_1d / 100)))
        stop = gap_open * Decimal('0.99')  # Slightly below gap open

        # Apply max constraint
        max_stop_distance = entry * Decimal(str(self.config.max_stop_loss_pct / 100))
        if (entry - stop) > max_stop_distance:
            stop = entry - max_stop_distance

        # Take profit 1: Gap extension (add 50-100% of gap size)
        gap_size = current_price - gap_open
        tp1 = entry + (gap_size * Decimal('0.75'))  # 75% extension

        # Take profit 2: Full gap extension or 3R
        risk = entry - stop
        tp2_r = entry + (risk * Decimal('3.0'))
        tp2_gap = entry + gap_size
        tp2 = max(tp2_r, tp2_gap)

        return entry, stop, tp1, tp2

    def _build_description(
        self,
        candidate: Dict[str, Any],
        factors: List[str]
    ) -> str:
        """Build human-readable setup description"""

        ticker = candidate['ticker']
        price_change = float(candidate.get('price_change_1d') or 0)
        rel_vol = float(candidate.get('relative_volume') or 0)
        sma_cross = candidate.get('sma_cross_signal', '')

        desc = f"{ticker} gap continuation. "
        desc += f"Gapped +{price_change:.1f}% on {rel_vol:.1f}x volume. "

        if sma_cross in ['golden_cross', 'bullish']:
            desc += "In uptrend. "

        return desc
