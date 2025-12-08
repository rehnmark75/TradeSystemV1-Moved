"""
Breakout Confirmation Scanner

Identifies volume-confirmed breakouts from consolidation patterns.

Entry Criteria:
- Near 52-week high (within 5%) or making new high
- Volume surge (> 1.5x average)
- Positive gap or strong close
- MACD momentum expanding

Stop Logic:
- Below breakout level (recent consolidation high)
- Or 1.5-2x ATR below entry

Target:
- TP1: Measured move based on consolidation range
- TP2: Extension target (1.618 fib or round number)

Best For:
- Strong momentum markets
- Stocks with catalyst (earnings beat, news)
- Sector leaders breaking out
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
class BreakoutConfig(ScannerConfig):
    """Configuration specific to Breakout Scanner"""

    # Position requirements
    max_pct_from_high: float = 5.0  # Within 5% of 52W high
    require_near_high: bool = True

    # Volume requirements
    min_relative_volume: float = 1.3
    optimal_volume: float = 2.0

    # Gap requirements
    allow_gap_breakout: bool = True
    min_gap_pct: float = 1.0

    # Trend
    require_bullish_trend: bool = True

    # ATR for stops
    atr_stop_multiplier: float = 2.0  # Wider stops for breakouts


class BreakoutConfirmationScanner(BaseScanner):
    """
    Scans for volume-confirmed breakout setups.

    Philosophy:
    - Breakouts need volume confirmation to be valid
    - Near 52W highs means limited overhead resistance
    - Gaps add conviction (institutional interest)
    - Failed breakouts get stopped out quickly
    """

    def __init__(
        self,
        db_manager,
        config: BreakoutConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or BreakoutConfig(), scorer)
        self.exclusion_filter = ExclusionFilterEngine(FilterConfig(
            max_tier=3,
            min_relative_volume=1.0,
            max_atr_percent=12.0,
            min_score_for_trade=55,
        ))
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "breakout_confirmation"

    @property
    def description(self) -> str:
        return "Volume-confirmed breakouts from consolidation patterns"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute breakout scan.

        Steps:
        1. Get candidates near 52W highs
        2. Filter for volume surge
        3. Check for breakout confirmation
        4. Score and apply filters
        5. Calculate entry/exit levels
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            from datetime import timedelta
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Get breakout candidates
        candidates = await self._get_breakout_candidates(calculation_date)
        logger.info(f"Found {len(candidates)} breakout candidates")

        if not candidates:
            return []

        # Filter for confirmation
        confirmed = self._filter_confirmed_breakouts(candidates)
        logger.info(f"Found {len(confirmed)} confirmed breakouts")

        # Score and create signals
        signals = []
        for candidate in confirmed:
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

    async def _get_breakout_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get stocks near 52W highs with volume"""

        additional_filters = """
            AND w.high_low_signal IN ('near_high', 'new_high')
            AND w.relative_volume >= {min_vol}
        """.format(min_vol=self.config.min_relative_volume)

        if self.config.require_bullish_trend:
            additional_filters += """
                AND w.sma_cross_signal IN ('golden_cross', 'bullish')
            """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _filter_confirmed_breakouts(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for confirmed breakout setups"""

        confirmed = []

        for c in candidates:
            # Check position from high
            pct_from_high = float(c.get('pct_from_52w_high') or -100)
            if abs(pct_from_high) > self.config.max_pct_from_high:
                continue

            # Volume confirmation
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            # Gap or strong close confirmation
            gap_signal = c.get('gap_signal', '')
            price_change_1d = float(c.get('price_change_1d') or 0)

            has_gap = gap_signal in ['gap_up', 'gap_up_large']
            has_strong_close = price_change_1d > 2.0

            if not (has_gap or has_strong_close):
                continue

            # MACD momentum
            macd_cross = c.get('macd_cross_signal', '')
            if macd_cross not in ['bullish_cross', 'bullish']:
                continue

            confirmed.append(c)

        return confirmed

    def _create_signal(
        self,
        candidate: Dict[str, Any]
    ) -> Optional[SignalSetup]:
        """Create a SignalSetup from a breakout candidate"""

        # Score the candidate
        score_components = self.scorer.score_bullish(candidate)
        composite_score = score_components.composite_score

        # Breakout bonus for near/new high
        high_low = candidate.get('high_low_signal', '')
        if high_low == 'new_high':
            composite_score = min(100, composite_score + 5)

        # Volume bonus for extreme volume
        rel_vol = float(candidate.get('relative_volume') or 0)
        if rel_vol >= self.config.optimal_volume:
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
        confluence_factors.insert(0, 'Breakout')  # Add breakout factor
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
            market_regime="Breakout",
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
        Calculate entry, stop, and take profit levels for breakouts.

        Breakout strategy uses wider stops (2x ATR) to avoid
        getting shaken out during initial volatility.
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)

        # ATR in price terms
        atr = current_price * Decimal(str(atr_percent / 100))

        # Entry at current price
        entry = current_price

        # Stop loss: 2x ATR below (wider for breakouts)
        stop = entry - (atr * Decimal(str(self.config.atr_stop_multiplier)))

        # Ensure stop isn't too far (max 8%)
        max_stop = entry * Decimal('0.92')
        stop = max(stop, max_stop)

        # Take profits
        tp1, tp2 = self.calculate_take_profits(entry, stop, signal_type)

        return entry, stop, tp1, tp2

    def _build_description(
        self,
        candidate: Dict[str, Any],
        factors: List[str]
    ) -> str:
        """Build human-readable setup description"""

        ticker = candidate['ticker']
        price = float(candidate.get('current_price', 0))
        pct_from_high = float(candidate.get('pct_from_52w_high') or 0)
        rel_vol = float(candidate.get('relative_volume') or 0)
        gap = candidate.get('gap_signal', '')

        desc = f"{ticker} breakout setup. "

        if pct_from_high >= -1:
            desc += "At/near 52W high. "
        else:
            desc += f"{abs(pct_from_high):.1f}% from 52W high. "

        desc += f"Volume {rel_vol:.1f}x. "

        if gap in ['gap_up', 'gap_up_large']:
            desc += "Gap up confirmation. "

        return desc
