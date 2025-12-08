"""
Mean Reversion Scanner

Identifies oversold bounce opportunities in stocks with underlying strength.

Entry Criteria:
- RSI oversold (< 30) or extreme oversold (< 20)
- Still in overall uptrend (above SMA50 or SMA200)
- Bullish reversal candlestick pattern
- Not in death cross

Stop Logic:
- Below recent low or support level
- 1.5x ATR below entry

Target:
- TP1: Return to mean (20-day SMA)
- TP2: Previous swing high

Best For:
- Oversold bounces in quality stocks
- Market pullbacks in uptrends
- Counter-trend plays with strict risk management
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
from ..exclusion_filters import ExclusionFilterEngine, get_mean_reversion_filter

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionConfig(ScannerConfig):
    """Configuration specific to Mean Reversion Scanner"""

    # RSI requirements
    max_rsi: float = 35.0  # Must be oversold
    extreme_oversold_rsi: float = 25.0  # Bonus for extreme

    # Trend requirements (must have some underlying strength)
    require_above_sma50: bool = False
    require_above_sma200: bool = False
    require_bullish_long_term: bool = True  # At least not death cross

    # Pattern requirements
    require_bullish_pattern: bool = True

    # Volume
    min_relative_volume: float = 0.6  # Can be lower for reversals

    # Risk
    atr_stop_multiplier: float = 1.5
    max_stop_loss_pct: float = 6.0  # Tighter stops for reversals


class MeanReversionScanner(BaseScanner):
    """
    Scans for oversold bounce opportunities.

    Philosophy:
    - Buy fear when underlying trend is still intact
    - Reversal patterns add conviction
    - Tighter stops because these are counter-trend
    - Target return to mean, not new highs
    """

    BULLISH_REVERSAL_PATTERNS = [
        'bullish_engulfing', 'hammer', 'dragonfly_doji',
        'inverted_hammer', 'morning_star', 'bullish_marubozu'
    ]

    def __init__(
        self,
        db_manager,
        config: MeanReversionConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or MeanReversionConfig(), scorer)
        self.exclusion_filter = get_mean_reversion_filter()
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "mean_reversion"

    @property
    def description(self) -> str:
        return "Oversold bounces in uptrends with reversal patterns"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute mean reversion scan.

        Steps:
        1. Get oversold candidates
        2. Filter for underlying strength
        3. Check for reversal patterns
        4. Score and apply filters
        5. Calculate entry/exit levels
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            from datetime import timedelta
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Get oversold candidates
        candidates = await self._get_oversold_candidates(calculation_date)
        logger.info(f"Found {len(candidates)} oversold candidates")

        if not candidates:
            return []

        # Filter for reversal setups
        reversal_setups = self._filter_reversal_setups(candidates)
        logger.info(f"Found {len(reversal_setups)} reversal setups")

        # Score and create signals
        signals = []
        for candidate in reversal_setups:
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

    async def _get_oversold_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get oversold stocks"""

        additional_filters = """
            AND w.rsi_signal IN ('oversold', 'oversold_extreme')
            AND w.sma_cross_signal != 'death_cross'
        """

        if self.config.require_bullish_long_term:
            additional_filters += """
                AND w.sma_cross_signal IN ('golden_cross', 'bullish', 'neutral')
            """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _filter_reversal_setups(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for valid reversal setups"""

        setups = []

        for c in candidates:
            # Check RSI level
            rsi_14 = float(c.get('rsi_14') or 50)
            if rsi_14 > self.config.max_rsi:
                continue

            # Check for bullish pattern if required
            pattern = c.get('candlestick_pattern', '')
            has_bullish_pattern = pattern in self.BULLISH_REVERSAL_PATTERNS

            if self.config.require_bullish_pattern and not has_bullish_pattern:
                continue

            # Not a falling knife (new low + death cross)
            high_low = c.get('high_low_signal', '')
            sma_cross = c.get('sma_cross_signal', '')

            if high_low == 'new_low' and sma_cross == 'death_cross':
                continue

            # Volume check
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            setups.append(c)

        return setups

    def _create_signal(
        self,
        candidate: Dict[str, Any]
    ) -> Optional[SignalSetup]:
        """Create a SignalSetup from a mean reversion candidate"""

        # Score the candidate
        score_components = self.scorer.score_bullish(candidate)
        composite_score = score_components.composite_score

        # Bonus for extreme oversold
        rsi_14 = float(candidate.get('rsi_14') or 50)
        if rsi_14 <= self.config.extreme_oversold_rsi:
            composite_score = min(100, composite_score + 5)

        # Pattern bonus
        pattern = candidate.get('candlestick_pattern', '')
        if pattern in ['bullish_engulfing', 'morning_star']:
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
        confluence_factors.insert(0, f'RSI {rsi_14:.0f}')  # Add RSI
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
            market_regime="Oversold Bounce",
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data=candidate,
        )

        # Calculate position size (smaller for mean reversion)
        signal.suggested_position_size_pct = self.calculate_position_size(
            Decimal(str(self.config.max_risk_per_trade_pct * 0.8)),  # Reduce size
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
        Calculate entry, stop, and take profit levels for mean reversion.

        Mean reversion uses:
        - Tighter stops (we're fighting trend)
        - Target return to mean (SMA20)
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)
        sma_20 = float(candidate.get('sma_20') or 0)

        # ATR in price terms
        atr = current_price * Decimal(str(atr_percent / 100))

        # Entry at current price
        entry = current_price

        # Stop loss: 1.5x ATR below (tighter for reversals)
        stop_distance = atr * Decimal(str(self.config.atr_stop_multiplier))

        # Apply max constraint
        max_stop_distance = entry * Decimal(str(self.config.max_stop_loss_pct / 100))
        stop_distance = min(stop_distance, max_stop_distance)

        stop = entry - stop_distance

        # Take profit 1: Return to SMA20 (mean)
        if sma_20 > 0:
            tp1 = Decimal(str(sma_20))
            # If SMA20 is below current (unlikely for oversold), use standard R
            if tp1 <= entry:
                risk = entry - stop
                tp1 = entry + (risk * Decimal('2.0'))
        else:
            risk = entry - stop
            tp1 = entry + (risk * Decimal('2.0'))

        # Take profit 2: 3R or previous swing high
        risk = entry - stop
        tp2 = entry + (risk * Decimal('3.0'))

        return entry, stop, tp1, tp2

    def _build_description(
        self,
        candidate: Dict[str, Any],
        factors: List[str]
    ) -> str:
        """Build human-readable setup description"""

        ticker = candidate['ticker']
        rsi = float(candidate.get('rsi_14') or 50)
        pattern = candidate.get('candlestick_pattern', '')
        sma_cross = candidate.get('sma_cross_signal', '')

        desc = f"{ticker} oversold bounce. "
        desc += f"RSI at {rsi:.0f}. "

        if pattern in self.BULLISH_REVERSAL_PATTERNS:
            desc += f"{pattern.replace('_', ' ').title()} pattern. "

        if sma_cross == 'bullish':
            desc += "Still in uptrend. "

        return desc
