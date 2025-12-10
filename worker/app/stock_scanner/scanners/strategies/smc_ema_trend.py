"""
SMC EMA Trend Scanner

Adapted from forex SMC Simple strategy for stocks.
Uses EMA for trend bias, swing structure breaks for confirmation,
and Fibonacci pullback zones for optimal entry timing.

Entry Criteria (3-Tier System):
- Tier 1: Price above/below EMA 50 for directional bias
- Tier 2: Recent swing high/low break confirming structure
- Tier 3: Price in Fibonacci pullback zone (23.6%-70%) or momentum continuation

Stop Logic:
- Beyond opposite swing structure + buffer

Target:
- TP1: 1.5R minimum
- TP2: Next swing structure or 2.5R

Best For:
- Trending markets with clear structure
- Pullback continuation plays
- Momentum breakouts with structure confirmation
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
class SMCEmaTrendConfig(ScannerConfig):
    """Configuration for SMC EMA Trend Scanner"""

    # ==========================================================================
    # TIER 1: EMA Bias Determination
    # ==========================================================================
    ema_period: int = 50  # Daily EMA for trend bias (institutional standard)
    ema_buffer_pct: float = 0.5  # Price must be 0.5% from EMA for valid bias

    # ==========================================================================
    # TIER 2: Swing Structure Detection
    # ==========================================================================
    swing_lookback: int = 20  # Bars to find swing points
    swing_strength: int = 2  # Bars on each side to confirm swing
    min_swing_atr_pct: float = 25.0  # Minimum swing size as % of ATR

    # ==========================================================================
    # TIER 3: Pullback/Entry Zone (Fibonacci Levels)
    # ==========================================================================
    pullback_zone_start: float = 0.236  # 23.6% retracement
    pullback_zone_end: float = 0.70  # 70% retracement
    optimal_zone_start: float = 0.382  # Golden zone start
    optimal_zone_end: float = 0.618  # Golden zone end

    # Entry modes
    allow_pullback_entry: bool = True  # Retracement into zone
    allow_momentum_entry: bool = True  # Continuation beyond break

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    min_rr_ratio: float = 1.5
    atr_stop_multiplier: float = 1.5
    swing_stop_buffer_pct: float = 0.5  # Buffer beyond swing for SL

    # Default R:R targets
    tp1_rr_ratio: float = 1.5
    tp2_rr_ratio: float = 2.5

    # ==========================================================================
    # Technical Filters
    # ==========================================================================
    min_relative_volume: float = 0.8
    require_macd_alignment: bool = True  # MACD must align with direction

    # ==========================================================================
    # Fundamental Filters (inherited from ScannerConfig)
    # ==========================================================================
    max_pe_ratio: float = 60.0
    min_institutional_pct: float = 15.0
    days_to_earnings_min: int = 5  # Avoid earnings risk


class SMCEmaTrendScanner(BaseScanner):
    """
    Scans for SMC-style trend continuation setups.

    Philosophy (adapted from forex SMC Simple):
    - Use EMA 50 as institutional bias filter
    - Confirm direction with swing structure breaks
    - Enter on pullbacks to Fibonacci zones for optimal R:R
    - Alternative: momentum continuation entries
    """

    def __init__(
        self,
        db_manager,
        config: SMCEmaTrendConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or SMCEmaTrendConfig(), scorer)
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "smc_ema_trend"

    @property
    def description(self) -> str:
        return "SMC-style EMA trend following with swing structure and pullback zones"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute SMC EMA Trend scan.

        Steps:
        1. Get candidates with EMA alignment (Tier 1)
        2. Filter for swing structure breaks (Tier 2)
        3. Filter for pullback zone or momentum entry (Tier 3)
        4. Score and create signals
        5. Return sorted signals
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            from datetime import timedelta
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Tier 1: Get EMA-aligned candidates (bullish and bearish)
        bullish_candidates = await self._get_bullish_ema_candidates(calculation_date)
        bearish_candidates = await self._get_bearish_ema_candidates(calculation_date)

        logger.info(f"Tier 1 - EMA aligned: {len(bullish_candidates)} bullish, {len(bearish_candidates)} bearish")

        # Tier 2 & 3: Filter for structure and entry conditions
        bullish_setups = self._filter_bullish_setups(bullish_candidates)
        bearish_setups = self._filter_bearish_setups(bearish_candidates)

        logger.info(f"Tier 2&3 - Valid setups: {len(bullish_setups)} bullish, {len(bearish_setups)} bearish")

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

    async def _get_bullish_ema_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get stocks with bullish EMA alignment (Tier 1 bullish)"""

        # Price above SMA50 (using SMA as EMA proxy since SMA50 is available)
        # Also require bullish trend structure
        additional_filters = """
            AND w.sma50_signal IN ('above', 'crossed_above')
            AND w.sma_cross_signal IN ('golden_cross', 'bullish')
            AND w.trend_strength IN ('strong_up', 'up', 'neutral')
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    async def _get_bearish_ema_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get stocks with bearish EMA alignment (Tier 1 bearish)"""

        # Price below SMA50
        # Also require bearish trend structure
        additional_filters = """
            AND w.sma50_signal IN ('below', 'crossed_below')
            AND w.sma_cross_signal IN ('death_cross', 'bearish')
            AND w.trend_strength IN ('strong_down', 'down')
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _filter_bullish_setups(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for bullish swing break + pullback/momentum entry"""

        filtered = []

        for c in candidates:
            # Check MACD alignment if required
            if self.config.require_macd_alignment:
                macd_cross = c.get('macd_cross_signal', '')
                if macd_cross not in ['bullish_cross', 'bullish']:
                    continue

            # Volume check
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            # Check for structure (using trend strength and SMA alignment as proxy)
            # In stocks, we use the existing signals rather than computing swing breaks
            sma_cross = c.get('sma_cross_signal', '')
            trend = c.get('trend_strength', '')

            # Tier 2: Structure confirmation
            # Golden cross or bullish MA alignment indicates swing structure break
            if sma_cross not in ['golden_cross', 'bullish']:
                continue

            # Tier 3: Pullback entry zone
            # Use RSI as pullback indicator - not overbought means potential pullback zone
            rsi_14 = float(c.get('rsi_14') or 50)
            rsi_signal = c.get('rsi_signal', '')

            entry_type = None

            # Pullback entry: RSI pulled back but not oversold
            if self.config.allow_pullback_entry:
                if 35 <= rsi_14 <= 65 and rsi_signal not in ['overbought', 'overbought_extreme']:
                    entry_type = 'PULLBACK'

            # Momentum entry: Strong momentum, higher timeframe aligned
            if self.config.allow_momentum_entry and not entry_type:
                if trend == 'strong_up' and rsi_14 > 50:
                    # Check price is making new highs (momentum continuation)
                    high_low_signal = c.get('high_low_signal', '')
                    if high_low_signal in ['new_high', 'near_high']:
                        entry_type = 'MOMENTUM'

            if entry_type:
                c['entry_type'] = entry_type
                c['signal_direction'] = 'BULLISH'
                filtered.append(c)

        return filtered

    def _filter_bearish_setups(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for bearish swing break + pullback/momentum entry"""

        filtered = []

        for c in candidates:
            # Check MACD alignment if required
            if self.config.require_macd_alignment:
                macd_cross = c.get('macd_cross_signal', '')
                if macd_cross not in ['bearish_cross', 'bearish']:
                    continue

            # Volume check
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            # Check for structure
            sma_cross = c.get('sma_cross_signal', '')
            trend = c.get('trend_strength', '')

            # Tier 2: Structure confirmation
            if sma_cross not in ['death_cross', 'bearish']:
                continue

            # Tier 3: Entry zone
            rsi_14 = float(c.get('rsi_14') or 50)
            rsi_signal = c.get('rsi_signal', '')

            entry_type = None

            # Pullback entry: RSI bounced but not oversold
            if self.config.allow_pullback_entry:
                if 35 <= rsi_14 <= 65 and rsi_signal not in ['oversold', 'oversold_extreme']:
                    entry_type = 'PULLBACK'

            # Momentum entry: Strong downward momentum
            if self.config.allow_momentum_entry and not entry_type:
                if trend == 'strong_down' and rsi_14 < 50:
                    high_low_signal = c.get('high_low_signal', '')
                    if high_low_signal in ['new_low', 'near_low']:
                        entry_type = 'MOMENTUM'

            if entry_type:
                c['entry_type'] = entry_type
                c['signal_direction'] = 'BEARISH'
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

        For SMC EMA Trend:
        - Entry: Current price
        - Stop: ATR-based stop with swing buffer consideration
        - TP1: 1.5R (conservative)
        - TP2: 2.5R (extended target)
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)

        # ATR in price terms
        atr = current_price * Decimal(str(atr_percent / 100))

        # Entry at current price
        entry = current_price

        # Stop loss: ATR-based
        stop = self.calculate_atr_based_stop(entry, atr, signal_type)

        # Take profits using config ratios
        tp1, tp2 = self.calculate_take_profits(entry, stop, signal_type)

        return entry, stop, tp1, tp2

    def _build_confluence_factors(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> List[str]:
        """Build list of confluence factors for this setup"""

        factors = []
        entry_type = candidate.get('entry_type', 'PULLBACK')

        # Direction-specific factors
        if signal_type == SignalType.BUY:
            factors.append(f"Bullish EMA50 bias")

            sma_cross = candidate.get('sma_cross_signal', '')
            if sma_cross == 'golden_cross':
                factors.append("Golden Cross (SMA50 > SMA200)")
            elif sma_cross == 'bullish':
                factors.append("Bullish MA alignment")

            macd_cross = candidate.get('macd_cross_signal', '')
            if macd_cross == 'bullish_cross':
                factors.append("MACD bullish cross")
            elif macd_cross == 'bullish':
                factors.append("MACD histogram positive")

        else:  # SELL
            factors.append(f"Bearish EMA50 bias")

            sma_cross = candidate.get('sma_cross_signal', '')
            if sma_cross == 'death_cross':
                factors.append("Death Cross (SMA50 < SMA200)")
            elif sma_cross == 'bearish':
                factors.append("Bearish MA alignment")

            macd_cross = candidate.get('macd_cross_signal', '')
            if macd_cross == 'bearish_cross':
                factors.append("MACD bearish cross")
            elif macd_cross == 'bearish':
                factors.append("MACD histogram negative")

        # Entry type
        if entry_type == 'PULLBACK':
            factors.append("Pullback entry zone")
        else:
            factors.append("Momentum continuation")

        # Volume
        rel_vol = float(candidate.get('relative_volume') or 0)
        if rel_vol >= 1.5:
            factors.append(f"High volume ({rel_vol:.1f}x)")
        elif rel_vol >= 1.0:
            factors.append(f"Above avg volume ({rel_vol:.1f}x)")

        # Trend strength
        trend = candidate.get('trend_strength', '')
        if trend in ['strong_up', 'strong_down']:
            factors.append("Strong trend")

        return factors

    def _build_description(
        self,
        candidate: Dict[str, Any],
        factors: List[str]
    ) -> str:
        """Build human-readable setup description"""

        ticker = candidate['ticker']
        entry_type = candidate.get('entry_type', 'PULLBACK')
        direction = candidate.get('signal_direction', 'BULLISH')
        rsi = float(candidate.get('rsi_14') or 50)

        desc = f"{ticker} SMC {entry_type.lower()} setup. "
        desc += f"{direction.title()} bias with EMA50 alignment. "
        desc += f"RSI {rsi:.0f}. "

        if factors:
            desc += f"Confluence: {', '.join(factors[:3])}"

        return desc

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime for the stock"""

        trend_strength = candidate.get('trend_strength', '')
        entry_type = candidate.get('entry_type', 'PULLBACK')
        atr_pct = float(candidate.get('atr_percent') or 0)

        if trend_strength == 'strong_up':
            regime = "Strong Uptrend"
        elif trend_strength == 'up':
            regime = "Moderate Uptrend"
        elif trend_strength == 'strong_down':
            regime = "Strong Downtrend"
        elif trend_strength == 'down':
            regime = "Moderate Downtrend"
        else:
            regime = "Ranging"

        if entry_type == 'MOMENTUM':
            regime += " (Momentum)"
        else:
            regime += " (Pullback)"

        if atr_pct > 5:
            regime += " - High Vol"
        elif atr_pct < 2:
            regime += " - Low Vol"

        return regime
