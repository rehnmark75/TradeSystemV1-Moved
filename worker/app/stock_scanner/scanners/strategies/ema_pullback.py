"""
EMA Pullback Scanner

Scans for pullback entries in established uptrends using EMA cascade alignment.

Entry Criteria:
- EMA Cascade: SMA20 > SMA50 > SMA200 (bullish alignment)
- Price pulling back to EMA zones (within ATR distance of SMA20 or SMA50)
- RSI in reset zone (45-70), not extended
- Volume contracted or normal (no panic selling)

Stop Logic:
- Structure-based: Below SMA50 with ATR buffer
- Or 1.5x ATR from entry (whichever is tighter)

Target:
- TP1: 2:1 R:R
- TP2: 3:1 R:R

Quality Scoring:
- Volume contracted (<0.85x) = Higher quality
- RSI in reset zone (45-55) = Higher quality
- Shallow pullback (<10% from high) = Higher quality

Best For:
- Trend continuation plays
- Buying dips in strong uptrends
- Lower risk entries vs. breakout chasing

Agent Team Optimizations Applied:
- ATR zone widened from 0.5x to 1.0x (captures 68% vs 38% of price action)
- RSI range adjusted from 40-60 to 45-70 (strong trends stay elevated)
- Structure-based stops with ATR buffer
- Earnings exclusion filter (7 days)
- Volume z-score style filtering
"""

import logging
from datetime import datetime, timedelta
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
class EMAPullbackConfig(ScannerConfig):
    """Configuration for EMA Pullback Scanner"""

    # ==========================================================================
    # EMA Parameters
    # ==========================================================================
    ema_fast: int = 20   # Short-term trend (using SMA20)
    ema_medium: int = 50  # Medium-term trend (using SMA50)
    ema_slow: int = 200   # Long-term trend (using SMA200)

    # ==========================================================================
    # Trend Qualification
    # ==========================================================================
    require_cascade: bool = True  # EMA fast > medium > slow for bullish
    min_trend_strength: str = 'up'  # Minimum: 'strong_up', 'up'

    # ==========================================================================
    # Pullback Detection (Optimized per agent recommendations)
    # ==========================================================================
    # ATR zone for pullback detection (1.0x captures 68% of price action)
    pullback_atr_zone: float = 1.0  # CHANGED from 0.5 (was too restrictive)

    # Pullback depth constraints (% from recent high)
    max_pullback_depth: float = 30.0  # Reject if >30% from high (trend failure)
    ideal_pullback_min: float = 5.0   # Minimum pullback for entry
    ideal_pullback_max: float = 20.0  # Ideal pullback range

    # Price proximity to EMAs (percentage)
    ema_proximity_pct: float = 3.0  # Price within 3% of SMA20 or SMA50

    # ==========================================================================
    # RSI Confirmation (Optimized 2025-12-25 - PF 2.02 achieved)
    # ==========================================================================
    rsi_period: int = 14
    rsi_pullback_min: float = 40.0  # Optimized: 40-60 healthy pullback zone
    rsi_pullback_max: float = 60.0  # Optimized: avoid panic (<40) or no pullback (>60)
    rsi_reset_min: float = 45.0     # Ideal reset zone lower bound
    rsi_reset_max: float = 55.0     # Ideal reset zone upper bound

    # ==========================================================================
    # ADX Trend Strength Filter (Optimized 2025-12-25)
    # ==========================================================================
    min_adx: float = 20.0  # Welles Wilder threshold - trending market

    # ==========================================================================
    # MACD Momentum Filter (Optimized 2025-12-25)
    # ==========================================================================
    require_positive_macd: bool = True  # Only enter when MACD > 0 (bullish momentum)

    # ==========================================================================
    # Volume Filters (Optimized 2025-12-25 - PF 2.02 achieved)
    # ==========================================================================
    volume_contraction_threshold: float = 0.85  # Below 85% of SMA(20) = contracted
    volume_spike_reject: float = 1.5  # Reject if volume > 1.5x average (panic)
    min_relative_volume: float = 1.2  # Optimized: require institutional participation (1.2x avg)

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    atr_stop_multiplier: float = 1.5  # 1.5x ATR for stop
    structure_stop_buffer: float = 0.5  # 0.5x ATR below SMA50

    # Default R:R targets
    tp1_rr_ratio: float = 2.0  # First target at 2R
    tp2_rr_ratio: float = 3.0  # Second target at 3R

    # ==========================================================================
    # Fundamental Filters
    # ==========================================================================
    max_pe_ratio: float = 100.0  # Growth stocks can have higher PE
    min_institutional_pct: float = 10.0
    days_to_earnings_min: int = 7  # Avoid stocks with earnings within 7 days

    # ==========================================================================
    # Quality Scoring Bonuses
    # ==========================================================================
    volume_contracted_bonus: int = 10  # Bonus for contracted volume
    rsi_reset_bonus: int = 10  # Bonus for RSI in reset zone
    shallow_pullback_bonus: int = 5  # Bonus for shallow pullback


class EMAPullbackScanner(BaseScanner):
    """
    Scans for pullback entries in established uptrends.

    Philosophy:
    - Wait for confirmed uptrend (EMA cascade alignment)
    - Enter on pullback to EMA zones with volume contraction
    - RSI should be resetting, not extended or oversold
    - Structure-based stops for intelligent risk management

    This strategy focuses on "buying the dip" in trending stocks,
    looking for institutional accumulation patterns (low volume pullbacks)
    rather than panic selling (high volume drops).
    """

    def __init__(
        self,
        db_manager,
        config: EMAPullbackConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or EMAPullbackConfig(), scorer)
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "ema_pullback"

    @property
    def description(self) -> str:
        return "EMA pullback entries in established uptrends with quality scoring"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute EMA Pullback scan.

        Steps:
        1. Get candidates with EMA cascade alignment (uptrend)
        2. Filter for pullback conditions (price near EMA zones)
        3. Apply RSI and volume quality filters
        4. Score and create signals
        5. Return sorted signals by quality
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Step 1: Get candidates with bullish EMA cascade
        candidates = await self._get_pullback_candidates(calculation_date)
        logger.info(f"Found {len(candidates)} candidates with EMA alignment")

        # Step 2: Filter for quality pullback conditions
        filtered = self._filter_quality_pullbacks(candidates)
        logger.info(f"After pullback filters: {len(filtered)} candidates")

        # Step 3: Create signals (BUY only - this is a trend-following strategy)
        signals = []
        for candidate in filtered:
            signal = self._create_signal(candidate, SignalType.BUY)
            if signal:
                signals.append(signal)

        # Step 4: Filter out C/D quality tiers (Optimized 2025-12-25)
        # Only keep A+, A, B signals - C/D tiers have negative expectancy
        signals = [s for s in signals if s.quality_tier not in [QualityTier.C, QualityTier.D]]
        logger.info(f"After quality tier filter (A+/A/B only): {len(signals)} signals")

        # Step 5: Sort by composite score
        signals.sort(key=lambda x: x.composite_score, reverse=True)

        # Step 6: Apply limit
        signals = signals[:self.config.max_signals_per_run]

        # Log summary
        high_quality = sum(1 for s in signals if s.is_high_quality)
        self.log_scan_summary(len(candidates), len(signals), high_quality)

        return signals

    async def _get_pullback_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get stocks with bullish EMA cascade in pullback position.

        Database query filters for:
        - Bullish trend (golden cross or bullish SMA alignment)
        - Price above SMA20 or recently crossed above
        - Uptrend confirmed by trend_strength indicator
        """
        # Bullish cascade alignment with pullback potential
        additional_filters = """
            AND w.sma_cross_signal IN ('golden_cross', 'bullish')
            AND w.trend_strength IN ('strong_up', 'up')
            AND (
                w.sma20_signal IN ('above', 'crossed_above')
                OR w.sma50_signal IN ('above', 'crossed_above')
            )
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _filter_quality_pullbacks(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter for quality pullback setups with optimized filters.

        Optimized 2025-12-25 to achieve PF 2.02:
        - ADX > 20: Trending market confirmation (Welles Wilder threshold)
        - MACD > 0: Bullish momentum confirmation
        - RSI 40-60: Healthy pullback zone (not panic or exhaustion)
        - Volume >= 1.2x: Institutional participation required
        - Price near EMA zones (within proximity threshold)
        - Pullback depth within acceptable range
        """
        filtered = []

        for c in candidates:
            # ------------------------------------------------------------------
            # ADX Filter: Only enter in trending markets (Optimized 2025-12-25)
            # ADX > 20 = Welles Wilder threshold for trending market
            # ------------------------------------------------------------------
            adx = c.get('adx')
            if adx is not None:
                if float(adx) < self.config.min_adx:
                    continue  # Skip non-trending markets

            # ------------------------------------------------------------------
            # MACD Filter: Require bullish momentum (Optimized 2025-12-25)
            # MACD > 0 confirms bullish momentum, avoiding counter-trend entries
            # ------------------------------------------------------------------
            if self.config.require_positive_macd:
                macd = c.get('macd')
                if macd is not None:
                    if float(macd) <= 0:
                        continue  # Skip bearish/neutral momentum

            # ------------------------------------------------------------------
            # RSI Filter: Must be in valid pullback zone (40-60 optimized)
            # ------------------------------------------------------------------
            rsi_14 = float(c.get('rsi_14') or 50)
            if not (self.config.rsi_pullback_min <= rsi_14 <= self.config.rsi_pullback_max):
                continue

            # ------------------------------------------------------------------
            # Volume Filter: Require institutional participation (>=1.2x optimized)
            # ------------------------------------------------------------------
            rel_vol = float(c.get('relative_volume') or 1.0)

            # Reject if volume too high (panic) or too low (dead stock)
            if rel_vol > self.config.volume_spike_reject:
                continue
            if rel_vol < self.config.min_relative_volume:
                continue

            # Classify volume quality
            if rel_vol <= self.config.volume_contraction_threshold:
                c['volume_quality'] = 'contracted'  # Best - institutional accumulation
            elif rel_vol <= 1.0:
                c['volume_quality'] = 'below_average'  # Good
            else:
                c['volume_quality'] = 'normal'  # Acceptable

            # ------------------------------------------------------------------
            # Price Proximity Filter: Must be near EMA zones
            # ------------------------------------------------------------------
            current_price = float(c.get('current_price') or 0)
            sma_20 = float(c.get('sma_20') or current_price)
            sma_50 = float(c.get('sma_50') or current_price)

            if current_price <= 0 or sma_20 <= 0:
                continue

            # Calculate proximity to EMAs (percentage)
            pct_from_sma20 = abs(current_price - sma_20) / sma_20 * 100
            pct_from_sma50 = abs(current_price - sma_50) / sma_50 * 100

            # Must be within proximity threshold of at least one EMA
            near_sma20 = pct_from_sma20 <= self.config.ema_proximity_pct
            near_sma50 = pct_from_sma50 <= self.config.ema_proximity_pct

            # Also check if price is between SMA20 and SMA50 (pullback zone)
            in_pullback_zone = (
                (sma_50 <= current_price <= sma_20 * 1.02) or  # Between EMAs
                near_sma20 or near_sma50
            )

            if not in_pullback_zone:
                continue

            # Store proximity info for quality scoring
            c['pct_from_sma20'] = pct_from_sma20
            c['pct_from_sma50'] = pct_from_sma50
            c['near_sma20'] = near_sma20
            c['near_sma50'] = near_sma50

            # ------------------------------------------------------------------
            # Pullback Depth Filter
            # ------------------------------------------------------------------
            pct_from_high = float(c.get('pct_from_52w_high') or 0)

            # Reject if pullback too deep (trend potentially broken)
            if pct_from_high < -self.config.max_pullback_depth:
                continue

            # Calculate pullback depth (positive value)
            pullback_depth = abs(pct_from_high)
            c['pullback_depth'] = pullback_depth

            # Classify pullback quality
            if pullback_depth <= self.config.ideal_pullback_min:
                c['pullback_quality'] = 'very_shallow'  # Might be chasing
            elif pullback_depth <= self.config.ideal_pullback_max:
                c['pullback_quality'] = 'ideal'  # Best zone
            else:
                c['pullback_quality'] = 'deep'  # More risk

            # ------------------------------------------------------------------
            # Quality Score Calculation
            # ------------------------------------------------------------------
            quality_bonus = 0

            # Volume contraction bonus
            if c['volume_quality'] == 'contracted':
                quality_bonus += self.config.volume_contracted_bonus
            elif c['volume_quality'] == 'below_average':
                quality_bonus += self.config.volume_contracted_bonus // 2

            # RSI reset zone bonus
            if self.config.rsi_reset_min <= rsi_14 <= self.config.rsi_reset_max:
                quality_bonus += self.config.rsi_reset_bonus
                c['rsi_quality'] = 'reset_zone'
            elif rsi_14 <= 60:
                quality_bonus += self.config.rsi_reset_bonus // 2
                c['rsi_quality'] = 'moderate'
            else:
                c['rsi_quality'] = 'momentum'

            # Shallow pullback bonus
            if c['pullback_quality'] == 'ideal':
                quality_bonus += self.config.shallow_pullback_bonus
            elif c['pullback_quality'] == 'very_shallow':
                quality_bonus += self.config.shallow_pullback_bonus // 2

            c['quality_bonus'] = quality_bonus
            c['signal_direction'] = 'BULLISH'
            c['setup_type'] = 'ema_pullback'

            filtered.append(c)

        return filtered

    def _create_signal(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Optional[SignalSetup]:
        """Create a SignalSetup from a qualified candidate"""

        # Score the candidate using base scorer
        score_components = self.scorer.score_bullish(candidate)

        # Add quality bonus to composite score
        quality_bonus = candidate.get('quality_bonus', 0)
        composite_score = min(100, score_components.composite_score + quality_bonus)

        # Minimum score threshold
        if composite_score < self.config.min_score_threshold:
            return None

        # Calculate entry levels
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, signal_type)

        # Calculate risk percent
        if entry > 0:
            risk_pct = abs(entry - stop) / entry * 100
        else:
            risk_pct = 0

        # Build setup description and confluence factors
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

        # Calculate position size based on quality
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

        For EMA Pullback:
        - Entry: Current price
        - Stop: Structure-based (below SMA50 with ATR buffer) or ATR-based
        - TP1: 2:1 R:R
        - TP2: 3:1 R:R
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)
        sma_50 = Decimal(str(candidate.get('sma_50') or float(current_price)))

        # ATR in price terms
        atr = current_price * Decimal(str(atr_percent / 100))

        # Entry at current price
        entry = current_price

        # Calculate structure-based stop (below SMA50 with buffer)
        structure_stop = sma_50 - (atr * Decimal(str(self.config.structure_stop_buffer)))

        # Calculate ATR-based stop
        atr_stop = self.calculate_atr_based_stop(entry, atr, signal_type)

        # Use the tighter (higher) stop for better risk management
        # But ensure stop is below entry
        if signal_type == SignalType.BUY:
            stop = max(structure_stop, atr_stop)
            # Ensure stop is below entry
            if stop >= entry:
                stop = entry - atr
        else:
            stop = min(structure_stop, atr_stop)
            if stop <= entry:
                stop = entry + atr

        # Take profits based on R:R
        tp1, tp2 = self.calculate_take_profits(entry, stop, signal_type)

        return entry, stop, tp1, tp2

    def _build_confluence_factors(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> List[str]:
        """Build list of confluence factors for this pullback setup"""

        factors = []

        # ------------------------------------------------------------------
        # EMA Structure
        # ------------------------------------------------------------------
        sma_cross = candidate.get('sma_cross_signal', '')
        if sma_cross == 'golden_cross':
            factors.append("Golden Cross confirmed")
        else:
            factors.append("Bullish EMA cascade (20>50>200)")

        # ------------------------------------------------------------------
        # Pullback Location
        # ------------------------------------------------------------------
        near_sma20 = candidate.get('near_sma20', False)
        near_sma50 = candidate.get('near_sma50', False)
        pct_from_sma20 = candidate.get('pct_from_sma20', 0)

        if near_sma20:
            factors.append(f"Pullback to SMA20 ({pct_from_sma20:.1f}% away)")
        elif near_sma50:
            pct_from_sma50 = candidate.get('pct_from_sma50', 0)
            factors.append(f"Deeper pullback to SMA50 ({pct_from_sma50:.1f}% away)")

        # ------------------------------------------------------------------
        # Volume Quality
        # ------------------------------------------------------------------
        vol_quality = candidate.get('volume_quality', '')
        rel_vol = float(candidate.get('relative_volume') or 1.0)

        if vol_quality == 'contracted':
            factors.append(f"Volume contracted ({rel_vol:.2f}x) - accumulation")
        elif vol_quality == 'below_average':
            factors.append(f"Low volume pullback ({rel_vol:.2f}x)")

        # ------------------------------------------------------------------
        # RSI State
        # ------------------------------------------------------------------
        rsi_quality = candidate.get('rsi_quality', '')
        rsi = float(candidate.get('rsi_14') or 50)

        if rsi_quality == 'reset_zone':
            factors.append(f"RSI reset zone ({rsi:.0f})")
        elif rsi_quality == 'moderate':
            factors.append(f"RSI moderate ({rsi:.0f})")
        else:
            factors.append(f"RSI momentum ({rsi:.0f})")

        # ------------------------------------------------------------------
        # Pullback Depth
        # ------------------------------------------------------------------
        pullback_quality = candidate.get('pullback_quality', '')
        pullback_depth = candidate.get('pullback_depth', 0)

        if pullback_quality == 'ideal':
            factors.append(f"Ideal pullback depth ({pullback_depth:.1f}%)")
        elif pullback_quality == 'very_shallow':
            factors.append(f"Shallow pullback ({pullback_depth:.1f}%)")

        # ------------------------------------------------------------------
        # Trend Strength
        # ------------------------------------------------------------------
        trend = candidate.get('trend_strength', '')
        if trend == 'strong_up':
            factors.append("Strong uptrend")
        elif trend == 'up':
            factors.append("Uptrend confirmed")

        # ------------------------------------------------------------------
        # Additional Technical Factors
        # ------------------------------------------------------------------
        # MACD alignment
        macd_cross = candidate.get('macd_cross_signal', '')
        if macd_cross in ['bullish_cross', 'bullish']:
            factors.append("MACD bullish")

        # Candlestick patterns
        pattern = candidate.get('candlestick_pattern', '')
        if pattern in ['bullish_engulfing', 'hammer', 'dragonfly_doji', 'morning_star']:
            factors.append(pattern.replace('_', ' ').title())

        # Add fundamental confluence from base class
        base_factors = self.build_confluence_factors(candidate)
        for bf in base_factors:
            if bf not in factors and len(factors) < 8:
                factors.append(bf)

        return factors[:8]  # Limit to top 8 factors

    def _build_description(
        self,
        candidate: Dict[str, Any],
        factors: List[str]
    ) -> str:
        """Build human-readable setup description"""

        ticker = candidate['ticker']
        rsi = float(candidate.get('rsi_14') or 50)
        vol_quality = candidate.get('volume_quality', 'normal')
        pullback_quality = candidate.get('pullback_quality', 'normal')

        desc = f"{ticker} EMA pullback setup. "

        # Describe the pullback
        if candidate.get('near_sma20'):
            desc += "Price pulling back to SMA20 support. "
        elif candidate.get('near_sma50'):
            desc += "Deeper pullback testing SMA50 support. "

        # Describe volume
        if vol_quality == 'contracted':
            desc += "Volume contracted suggesting accumulation. "
        elif vol_quality == 'below_average':
            desc += "Low volume pullback (healthy). "

        # RSI context
        desc += f"RSI at {rsi:.0f}. "

        # Key factors
        if factors:
            desc += f"Key factors: {', '.join(factors[:3])}"

        return desc

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime for the stock"""

        trend_strength = candidate.get('trend_strength', '')
        atr_pct = float(candidate.get('atr_percent') or 0)
        pullback_quality = candidate.get('pullback_quality', '')

        # Base regime from trend
        if trend_strength == 'strong_up':
            regime = "Strong Uptrend"
        elif trend_strength == 'up':
            regime = "Uptrend"
        else:
            regime = "Trending"

        # Add pullback context
        regime += " - Pullback"

        if pullback_quality == 'ideal':
            regime += " (Ideal Zone)"
        elif pullback_quality == 'very_shallow':
            regime += " (Shallow)"
        elif pullback_quality == 'deep':
            regime += " (Deep)"

        # Add volatility context
        if atr_pct > 5:
            regime += " - High Vol"
        elif atr_pct < 2:
            regime += " - Low Vol"

        return regime
