"""
Trend Momentum Scanner

Finds pullback entry opportunities in established uptrends.

Entry Criteria:
- Stock in confirmed uptrend (above SMA20, SMA50, ideally SMA200)
- RSI pulled back to 40-60 zone (not overbought)
- MACD bullish or recent bullish cross
- Volume confirmation (relative volume > 1.0)

Stop Logic:
- Below recent swing low or 1.5x ATR below entry

Target:
- TP1: 2R (2x risk)
- TP2: 3R (3x risk) or next resistance

Best For:
- Trending markets
- Continuation plays
- Momentum stocks with institutional interest
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
from ..exclusion_filters import ExclusionFilterEngine, get_momentum_filter

logger = logging.getLogger(__name__)


@dataclass
class TrendMomentumConfig(ScannerConfig):
    """Configuration specific to Trend Momentum Scanner"""

    # Trend requirements
    require_above_sma20: bool = True
    require_above_sma50: bool = True
    require_above_sma200: bool = False  # Nice to have

    # RSI pullback zone
    rsi_min: float = 35.0
    rsi_max: float = 65.0

    # Momentum
    min_20d_change: float = 0.0  # Must be positive or flat
    min_5d_change: float = -5.0  # Allow slight pullback

    # Volume
    min_relative_volume: float = 0.8

    # Trend strength
    min_trend_strength: float = 0.4

    # =========================================================================
    # FUNDAMENTAL FILTERS FOR TREND MOMENTUM
    # Focus: Growth + Quality stocks with institutional backing
    # =========================================================================

    # Valuation - not too expensive for continuation plays
    max_pe_ratio: float = 50.0  # Avoid extremely overvalued

    # Growth - prefer growing companies
    min_earnings_growth: float = 0.0  # At least flat earnings

    # Profitability - quality companies
    min_profit_margin: float = 0.0  # Must be profitable

    # Financial health
    max_debt_to_equity: float = 3.0  # Reasonable debt load

    # Institutional support (momentum stocks need institutional buying)
    min_institutional_pct: float = 20.0

    # Avoid earnings risk (don't enter momentum trade right before earnings)
    days_to_earnings_min: int = 7


class TrendMomentumScanner(BaseScanner):
    """
    Scans for pullback entries in established uptrends.

    Philosophy:
    - Buy strength, not weakness
    - Wait for pullback to get better entry
    - Require multiple confirmations for higher win rate
    - Use ATR-based stops for dynamic risk management
    """

    def __init__(
        self,
        db_manager,
        config: TrendMomentumConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or TrendMomentumConfig(), scorer)
        self.exclusion_filter = get_momentum_filter()
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "trend_momentum"

    @property
    def description(self) -> str:
        return "Pullback entries in established uptrends with momentum confirmation"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute trend momentum scan.

        Steps:
        1. Get candidates in uptrends
        2. Filter for pullback conditions
        3. Score remaining candidates
        4. Apply exclusion filters
        5. Calculate entry/exit levels
        6. Return sorted signals
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            from datetime import timedelta
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Get uptrend candidates
        candidates = await self._get_uptrend_candidates(calculation_date)
        logger.info(f"Found {len(candidates)} uptrend candidates")

        if not candidates:
            return []

        # Filter for pullback conditions
        pullback_candidates = self._filter_pullback_conditions(candidates)
        logger.info(f"Found {len(pullback_candidates)} with pullback conditions")

        # Score and create signals
        signals = []
        for candidate in pullback_candidates:
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

    async def _get_uptrend_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get stocks in confirmed uptrends"""

        additional_filters = """
            AND w.sma_cross_signal IN ('golden_cross', 'bullish')
            AND w.trend_strength IN ('strong_up', 'up')
            AND COALESCE(w.price_change_20d, 0) >= {min_20d}
        """.format(
            min_20d=self.config.min_20d_change
        )

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _filter_pullback_conditions(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for pullback entry conditions"""

        filtered = []

        for c in candidates:
            rsi_14 = float(c.get('rsi_14') or 50)
            rsi_signal = c.get('rsi_signal', '')
            price_change_5d = float(c.get('price_change_5d') or 0)
            macd_cross = c.get('macd_cross_signal', '')

            # RSI in pullback zone (not overbought)
            if not (self.config.rsi_min <= rsi_14 <= self.config.rsi_max):
                continue

            # Not extremely overbought
            if rsi_signal == 'overbought_extreme':
                continue

            # Recent pullback (5-day change can be slightly negative)
            if price_change_5d < self.config.min_5d_change:
                continue

            # MACD still bullish
            if macd_cross not in ['bullish_cross', 'bullish']:
                continue

            # Volume check
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            filtered.append(c)

        return filtered

    def _create_signal(
        self,
        candidate: Dict[str, Any]
    ) -> Optional[SignalSetup]:
        """Create a SignalSetup from a candidate"""

        # Score the candidate
        score_components = self.scorer.score_bullish(candidate)
        composite_score = score_components.composite_score

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

        For trend momentum:
        - Entry: Current price (market order) or slightly below
        - Stop: 1.5x ATR below entry, or below recent support
        - TP1: 2R
        - TP2: 3R
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)

        # ATR in price terms
        atr = current_price * Decimal(str(atr_percent / 100))

        # Entry at current price
        entry = current_price

        # Stop loss: 1.5x ATR below
        stop = self.calculate_atr_based_stop(entry, atr, signal_type)

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
        change_20d = float(candidate.get('price_change_20d') or 0)
        rsi = float(candidate.get('rsi_14') or 50)

        desc = f"{ticker} pullback in uptrend. "
        desc += f"Up {change_20d:.1f}% (20d), RSI {rsi:.0f}. "

        if factors:
            desc += f"Confluence: {', '.join(factors[:3])}"

        return desc

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime for the stock"""

        trend_strength = candidate.get('trend_strength', '')
        atr_pct = float(candidate.get('atr_percent') or 0)

        if trend_strength == 'strong_up':
            regime = "Strong Uptrend"
        elif trend_strength == 'up':
            regime = "Moderate Uptrend"
        else:
            regime = "Weak Uptrend"

        if atr_pct > 5:
            regime += " (High Vol)"
        elif atr_pct < 2:
            regime += " (Low Vol)"

        return regime
