"""
EMA Crossover Scanner

Adapted from forex EMA strategy for stocks.
Uses EMA cascade alignment with price crossover triggers for trend-following entries.

Entry Criteria:
- EMA Cascade: Fast > Medium > Slow (or reverse for bearish)
- Price crosses above/below fast EMA
- RSI in valid zone (not extreme)
- Trend strength confirmation

Stop Logic:
- 2x ATR from entry

Target:
- TP1: 4x ATR (2:1 R:R)
- TP2: 6x ATR (3:1 R:R)

Best For:
- Strong trending markets
- Momentum continuation
- Breakout confirmations
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
class EMACrossoverConfig(ScannerConfig):
    """Configuration for EMA Crossover Scanner"""

    # ==========================================================================
    # EMA Parameters
    # ==========================================================================
    # Using SMA as proxy since SMA 20/50/200 are pre-calculated
    ema_fast: int = 20  # Short-term trend (using SMA20)
    ema_medium: int = 50  # Medium-term trend (using SMA50)
    ema_slow: int = 200  # Long-term trend (using SMA200)

    # ==========================================================================
    # Crossover Requirements
    # ==========================================================================
    require_cascade: bool = True  # EMA fast > medium > slow for bullish
    require_price_cross: bool = True  # Price must cross fast EMA

    # ==========================================================================
    # Trend Strength Filters
    # ==========================================================================
    min_trend_strength: str = 'up'  # Minimum: 'strong_up', 'up', 'neutral'

    # ==========================================================================
    # RSI Confirmation
    # ==========================================================================
    rsi_period: int = 14
    rsi_bullish_min: float = 40.0  # Not oversold for bullish
    rsi_bullish_max: float = 75.0  # Not overbought (leave room to run)
    rsi_bearish_min: float = 25.0  # Not oversold for bearish
    rsi_bearish_max: float = 60.0  # Not overbought

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    atr_stop_multiplier: float = 2.0  # 2x ATR for stop
    atr_tp_multiplier: float = 4.0  # 4x ATR for TP1 (2:1 R:R)

    # Default R:R targets
    tp1_rr_ratio: float = 2.0
    tp2_rr_ratio: float = 3.0

    # ==========================================================================
    # Volume & Filters
    # ==========================================================================
    min_relative_volume: float = 0.8
    require_macd_alignment: bool = False  # Optional MACD confirmation

    # ==========================================================================
    # Fundamental Filters
    # ==========================================================================
    max_pe_ratio: float = 80.0  # Growth stocks can have higher PE
    min_institutional_pct: float = 10.0


class EMACrossoverScanner(BaseScanner):
    """
    Scans for EMA cascade alignment with crossover entries.

    Philosophy (adapted from forex EMA strategy):
    - Follow the trend using EMA cascade alignment
    - Enter on price crossing fast EMA with confirmation
    - Use RSI to avoid chasing extended moves
    - ATR-based risk management for dynamic stops
    """

    def __init__(
        self,
        db_manager,
        config: EMACrossoverConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or EMACrossoverConfig(), scorer)
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "ema_crossover"

    @property
    def description(self) -> str:
        return "EMA cascade crossover with trend alignment confirmation"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute EMA Crossover scan.

        Steps:
        1. Get candidates with EMA cascade alignment
        2. Filter for crossover signals and RSI confirmation
        3. Score and create signals
        4. Return sorted signals
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            from datetime import timedelta
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Get bullish and bearish candidates
        bullish_candidates = await self._get_bullish_candidates(calculation_date)
        bearish_candidates = await self._get_bearish_candidates(calculation_date)

        logger.info(f"EMA aligned: {len(bullish_candidates)} bullish, {len(bearish_candidates)} bearish")

        # Filter for crossover and RSI conditions
        bullish_setups = self._filter_bullish_crossovers(bullish_candidates)
        bearish_setups = self._filter_bearish_crossovers(bearish_candidates)

        logger.info(f"After crossover filter: {len(bullish_setups)} bullish, {len(bearish_setups)} bearish")

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

    async def _get_bullish_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get stocks with bullish EMA cascade (SMA20 > SMA50 > SMA200)"""

        # Bullish cascade alignment
        additional_filters = """
            AND w.sma_cross_signal IN ('golden_cross', 'bullish')
            AND w.sma20_signal IN ('above', 'crossed_above')
            AND w.sma50_signal IN ('above', 'crossed_above')
            AND w.trend_strength IN ('strong_up', 'up', 'neutral')
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    async def _get_bearish_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get stocks with bearish EMA cascade (SMA20 < SMA50 < SMA200)"""

        # Bearish cascade alignment
        additional_filters = """
            AND w.sma_cross_signal IN ('death_cross', 'bearish')
            AND w.sma20_signal IN ('below', 'crossed_below')
            AND w.sma50_signal IN ('below', 'crossed_below')
            AND w.trend_strength IN ('strong_down', 'down')
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _filter_bullish_crossovers(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for bullish crossover signals with RSI confirmation"""

        filtered = []

        for c in candidates:
            # RSI confirmation - in valid bullish zone
            rsi_14 = float(c.get('rsi_14') or 50)
            if not (self.config.rsi_bullish_min <= rsi_14 <= self.config.rsi_bullish_max):
                continue

            # Check for recent price cross above SMA20
            sma20_signal = c.get('sma20_signal', '')
            if self.config.require_price_cross:
                if sma20_signal != 'crossed_above' and sma20_signal != 'above':
                    continue

            # Volume check
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            # Optional MACD alignment
            if self.config.require_macd_alignment:
                macd_cross = c.get('macd_cross_signal', '')
                if macd_cross not in ['bullish_cross', 'bullish']:
                    continue

            # Trend strength check
            trend = c.get('trend_strength', '')
            valid_trends = ['strong_up', 'up']
            if self.config.min_trend_strength == 'neutral':
                valid_trends.append('neutral')
            if trend not in valid_trends:
                continue

            c['signal_direction'] = 'BULLISH'
            c['crossover_type'] = 'price_above_ema20'
            filtered.append(c)

        return filtered

    def _filter_bearish_crossovers(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for bearish crossover signals with RSI confirmation"""

        filtered = []

        for c in candidates:
            # RSI confirmation - in valid bearish zone
            rsi_14 = float(c.get('rsi_14') or 50)
            if not (self.config.rsi_bearish_min <= rsi_14 <= self.config.rsi_bearish_max):
                continue

            # Check for recent price cross below SMA20
            sma20_signal = c.get('sma20_signal', '')
            if self.config.require_price_cross:
                if sma20_signal != 'crossed_below' and sma20_signal != 'below':
                    continue

            # Volume check
            rel_vol = float(c.get('relative_volume') or 0)
            if rel_vol < self.config.min_relative_volume:
                continue

            # Optional MACD alignment
            if self.config.require_macd_alignment:
                macd_cross = c.get('macd_cross_signal', '')
                if macd_cross not in ['bearish_cross', 'bearish']:
                    continue

            # Trend strength check
            trend = c.get('trend_strength', '')
            if trend not in ['strong_down', 'down']:
                continue

            c['signal_direction'] = 'BEARISH'
            c['crossover_type'] = 'price_below_ema20'
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

        For EMA Crossover:
        - Entry: Current price
        - Stop: 2x ATR from entry
        - TP1: 4x ATR (2:1 R:R)
        - TP2: 6x ATR (3:1 R:R)
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)

        # ATR in price terms
        atr = current_price * Decimal(str(atr_percent / 100))

        # Entry at current price
        entry = current_price

        # Stop loss: 2x ATR
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

        if signal_type == SignalType.BUY:
            # EMA cascade
            sma_cross = candidate.get('sma_cross_signal', '')
            if sma_cross == 'golden_cross':
                factors.append("Golden Cross confirmed")
            else:
                factors.append("Bullish EMA cascade (20>50>200)")

            # Price cross
            sma20_signal = candidate.get('sma20_signal', '')
            if sma20_signal == 'crossed_above':
                factors.append("Fresh cross above SMA20")
            else:
                factors.append("Price above SMA20")

            # Trend
            trend = candidate.get('trend_strength', '')
            if trend == 'strong_up':
                factors.append("Strong uptrend")
            elif trend == 'up':
                factors.append("Uptrend confirmed")

        else:  # SELL
            sma_cross = candidate.get('sma_cross_signal', '')
            if sma_cross == 'death_cross':
                factors.append("Death Cross confirmed")
            else:
                factors.append("Bearish EMA cascade (20<50<200)")

            sma20_signal = candidate.get('sma20_signal', '')
            if sma20_signal == 'crossed_below':
                factors.append("Fresh cross below SMA20")
            else:
                factors.append("Price below SMA20")

            trend = candidate.get('trend_strength', '')
            if trend == 'strong_down':
                factors.append("Strong downtrend")
            elif trend == 'down':
                factors.append("Downtrend confirmed")

        # RSI
        rsi = float(candidate.get('rsi_14') or 50)
        if 45 <= rsi <= 55:
            factors.append(f"RSI neutral ({rsi:.0f})")
        elif rsi > 55:
            factors.append(f"RSI momentum ({rsi:.0f})")
        else:
            factors.append(f"RSI pullback ({rsi:.0f})")

        # Volume
        rel_vol = float(candidate.get('relative_volume') or 0)
        if rel_vol >= 1.5:
            factors.append(f"High volume ({rel_vol:.1f}x)")
        elif rel_vol >= 1.0:
            factors.append(f"Avg volume ({rel_vol:.1f}x)")

        return factors

    def _build_description(
        self,
        candidate: Dict[str, Any],
        factors: List[str]
    ) -> str:
        """Build human-readable setup description"""

        ticker = candidate['ticker']
        direction = candidate.get('signal_direction', 'BULLISH')
        crossover = candidate.get('crossover_type', '')
        rsi = float(candidate.get('rsi_14') or 50)

        desc = f"{ticker} EMA crossover {direction.lower()}. "

        if 'above' in crossover:
            desc += "Price crossed above SMA20 with cascade alignment. "
        elif 'below' in crossover:
            desc += "Price crossed below SMA20 with cascade alignment. "

        desc += f"RSI {rsi:.0f}. "

        if factors:
            desc += f"Factors: {', '.join(factors[:2])}"

        return desc

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime for the stock"""

        trend_strength = candidate.get('trend_strength', '')
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
            regime = "Trending"

        regime += " (EMA Aligned)"

        if atr_pct > 5:
            regime += " - High Vol"
        elif atr_pct < 2:
            regime += " - Low Vol"

        return regime
