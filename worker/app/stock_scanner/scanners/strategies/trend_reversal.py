"""
Trend Reversal Scanner

Identifies stocks transitioning from downtrend to potential uptrend with
MULTI-DAY CONFIRMATION (not just single candle flips).

Philosophy:
- Look for stocks that WERE in a clear downtrend (below MAs, RSI oversold)
- Detect early signs of reversal over 3+ days (not a single candle bounce)
- Enter early in the reversal before the crowd

Entry Criteria:
1. Prior Downtrend (last 3-5 days):
   - trend_strength was 'down' or 'strong_down'
   - RSI was below 40 (oversold)
   - Price was below SMA20

2. Multi-Day Recovery (3 days confirmation):
   - At least 2 of last 3 days positive
   - RSI improved by 10+ points from the low
   - MACD histogram improving (less negative or turning positive)

3. Current State:
   - RSI now in 40-65 range (recovered but not overbought)
   - trend_strength no longer 'down' or 'strong_down'
   - Not yet extended (RSI < 70, room to run)

Stop Logic:
- 2.5x ATR below entry (wider for reversal plays)

Target:
- TP1: 2R
- TP2: 3R

Best For:
- Catching early trend reversals
- Bottom fishing with confirmation
- Momentum shift plays
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
class TrendReversalConfig(ScannerConfig):
    """Configuration for Trend Reversal Scanner"""

    # ==========================================================================
    # Prior Downtrend Requirements
    # ==========================================================================
    # RSI must have been below this level during downtrend
    rsi_was_oversold_threshold: float = 40.0

    # Required trend_strength values for "was in downtrend"
    # We look for these in historical data
    downtrend_strengths: tuple = ('down', 'strong_down')

    # ==========================================================================
    # Multi-Day Recovery Confirmation
    # ==========================================================================
    confirmation_days: int = 3  # How many days to check for recovery
    min_positive_days: int = 2  # At least 2 of 3 days must be positive
    min_rsi_improvement: float = 10.0  # RSI must improve by at least this much

    # ==========================================================================
    # Current State Requirements
    # ==========================================================================
    rsi_now_min: float = 40.0  # RSI now at least this (recovered)
    rsi_now_max: float = 65.0  # RSI not yet overbought

    # trend_strength must NOT be these (no longer in downtrend)
    exclude_trend_strengths: tuple = ('down', 'strong_down')

    # ==========================================================================
    # Not Too Extended Filters
    # ==========================================================================
    max_rsi_extended: float = 70.0  # Don't chase if RSI > 70
    max_price_change_5d: float = 20.0  # Don't chase if already up 20%+ in 5 days

    # ==========================================================================
    # Volume Requirements
    # ==========================================================================
    min_relative_volume: float = 0.5  # Can be lower for early reversals

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    atr_stop_multiplier: float = 2.5  # Wider stop for reversal plays
    tp1_rr_ratio: float = 2.0
    tp2_rr_ratio: float = 3.0

    # ==========================================================================
    # Fundamental Filters (optional)
    # ==========================================================================
    # For reversals, we want financially healthy companies (not value traps)
    max_debt_to_equity: float = 3.0
    min_current_ratio: float = 0.8


class TrendReversalScanner(BaseScanner):
    """
    Scans for stocks reversing from downtrend with multi-day confirmation.

    Key difference from mean_reversion:
    - Mean reversion: oversold bounce in an UPTREND
    - Trend reversal: actual TREND CHANGE from down to up

    We require:
    1. Stock WAS in downtrend (not just a pullback)
    2. Multiple days of recovery (not just 1 green candle)
    3. RSI significantly improved from oversold
    """

    def __init__(
        self,
        db_manager,
        config: TrendReversalConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or TrendReversalConfig(), scorer)
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "trend_reversal"

    @property
    def description(self) -> str:
        return "Detects downtrend-to-uptrend reversals with 3-day confirmation"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute Trend Reversal scan.

        Steps:
        1. Get candidates showing current recovery signs
        2. Check historical data for prior downtrend + multi-day confirmation
        3. Score and create signals
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Step 1: Get candidates with current recovery characteristics
        candidates = await self._get_current_recovery_candidates(calculation_date)
        logger.info(f"Initial recovery candidates: {len(candidates)}")

        if not candidates:
            return []

        # Step 2: Check historical data for prior downtrend + multi-day confirmation
        confirmed_reversals = await self._confirm_multi_day_reversal(
            candidates, calculation_date
        )
        logger.info(f"Multi-day confirmed reversals: {len(confirmed_reversals)}")

        # Step 3: Score and create signals
        signals = []
        for candidate in confirmed_reversals:
            signal = self._create_signal(candidate, SignalType.BUY)
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

    async def _get_current_recovery_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get stocks showing CURRENT recovery characteristics.

        These are stocks that TODAY look like they might be reversing:
        - RSI in recovery zone (40-65)
        - No longer in downtrend
        - Not too extended
        """
        # Build exclude list for trend_strength
        exclude_trends = "', '".join(self.config.exclude_trend_strengths)

        additional_filters = f"""
            -- RSI in recovery zone (recovered from oversold, not overbought)
            AND m.rsi_14 >= {self.config.rsi_now_min}
            AND m.rsi_14 <= {self.config.rsi_now_max}

            -- No longer in downtrend
            AND (w.trend_strength IS NULL OR w.trend_strength NOT IN ('{exclude_trends}'))

            -- Not too extended
            AND m.rsi_14 < {self.config.max_rsi_extended}
            AND (m.price_change_5d IS NULL OR m.price_change_5d <= {self.config.max_price_change_5d})

            -- Some recent positive momentum (at least today or 5d positive)
            AND (m.price_change_1d > 0 OR m.price_change_5d > 0)
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    async def _confirm_multi_day_reversal(
        self,
        candidates: List[Dict[str, Any]],
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Check historical data to confirm multi-day reversal pattern.

        For each candidate, verify:
        1. It WAS in a downtrend (prior period)
        2. Recovery signs have persisted for 3+ days
        3. RSI improved significantly
        """
        if not candidates:
            return []

        confirmed = []
        tickers = [c['ticker'] for c in candidates]

        # Get historical data: confirmation window + extra days for downtrend check
        days_needed = self.config.confirmation_days + 3
        history = await self._get_historical_metrics(
            tickers, calculation_date, days_needed
        )

        for candidate in candidates:
            ticker = candidate['ticker']
            ticker_history = history.get(ticker, [])

            if len(ticker_history) < self.config.confirmation_days + 1:
                continue  # Not enough history

            # Check 1: Was in downtrend before recovery
            was_in_downtrend, downtrend_rsi_low = self._check_prior_downtrend(ticker_history)
            if not was_in_downtrend:
                continue

            # Check 2: Multi-day recovery confirmation
            reversal_metrics = self._calculate_reversal_metrics(
                ticker_history, downtrend_rsi_low
            )

            if not self._passes_reversal_criteria(reversal_metrics):
                continue

            # Add reversal metrics to candidate for scoring
            candidate['reversal_metrics'] = reversal_metrics
            candidate['downtrend_rsi_low'] = downtrend_rsi_low
            confirmed.append(candidate)

        return confirmed

    async def _get_historical_metrics(
        self,
        tickers: List[str],
        end_date: datetime,
        days_back: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get historical screening metrics for multi-day analysis.

        Returns:
            Dict mapping ticker -> list of daily metrics (oldest to newest)
        """
        # Add buffer for weekends/holidays
        start_date = end_date - timedelta(days=days_back + 5)

        query = """
            SELECT
                ticker,
                calculation_date,
                current_price,
                price_change_1d,
                price_change_5d,
                rsi_14,
                macd_histogram,
                price_vs_sma20,
                price_vs_sma50,
                trend_strength,
                relative_volume,
                sma20_signal
            FROM stock_screening_metrics
            WHERE ticker = ANY($1)
              AND calculation_date BETWEEN $2 AND $3
            ORDER BY ticker, calculation_date ASC
        """

        rows = await self.db.fetch(query, tickers, start_date, end_date)

        # Group by ticker
        history = {}
        for row in rows:
            ticker = row['ticker']
            if ticker not in history:
                history[ticker] = []
            history[ticker].append(dict(row))

        return history

    def _check_prior_downtrend(
        self,
        history: List[Dict[str, Any]]
    ) -> Tuple[bool, float]:
        """
        Verify the stock was in a downtrend before the recent recovery.

        Returns:
            Tuple of (was_in_downtrend, lowest_rsi_during_downtrend)
        """
        if len(history) < self.config.confirmation_days + 1:
            return False, 50.0

        # Look at data BEFORE the most recent confirmation_days
        # e.g., if confirmation_days=3, check days before the last 3
        prior_data = history[:-self.config.confirmation_days]

        if len(prior_data) < 2:
            return False, 50.0

        was_in_downtrend = False
        lowest_rsi = 100.0

        for day in prior_data:
            # Check trend_strength for downtrend
            trend = day.get('trend_strength', '')
            if trend in self.config.downtrend_strengths:
                was_in_downtrend = True

            # Track lowest RSI
            rsi = float(day.get('rsi_14') or 50)
            if rsi < lowest_rsi:
                lowest_rsi = rsi

            # Check if RSI was oversold
            if rsi < self.config.rsi_was_oversold_threshold:
                was_in_downtrend = True

            # Check if price was below SMA20
            price_vs_sma20 = float(day.get('price_vs_sma20') or 0)
            if price_vs_sma20 < -3:  # More than 3% below SMA20
                was_in_downtrend = True

        return was_in_downtrend, lowest_rsi

    def _calculate_reversal_metrics(
        self,
        history: List[Dict[str, Any]],
        downtrend_rsi_low: float
    ) -> Dict[str, Any]:
        """
        Calculate reversal strength metrics over the confirmation window.

        Checks the last N days (confirmation_days) for:
        - How many days were positive
        - RSI improvement from the low
        - MACD histogram improvement
        """
        confirmation_data = history[-self.config.confirmation_days:]

        if len(confirmation_data) < self.config.confirmation_days:
            return {'passes': False}

        metrics = {
            'positive_days': 0,
            'rsi_improvement': 0,
            'macd_improving': False,
            'current_rsi': 50,
            'current_trend': '',
            'sma20_crossed': False,
            'passes': False
        }

        # Count positive days
        for day in confirmation_data:
            price_change = float(day.get('price_change_1d') or 0)
            if price_change > 0:
                metrics['positive_days'] += 1

        # RSI improvement (from downtrend low to current)
        current_rsi = float(confirmation_data[-1].get('rsi_14') or 50)
        metrics['current_rsi'] = current_rsi
        metrics['rsi_improvement'] = current_rsi - downtrend_rsi_low

        # MACD histogram trend (is it improving?)
        macd_values = [float(d.get('macd_histogram') or 0) for d in confirmation_data]
        if len(macd_values) >= 2:
            # Check if MACD is trending up (each value >= previous, or turned positive)
            macd_improving = True
            for i in range(1, len(macd_values)):
                if macd_values[i] < macd_values[i-1] - 0.01:  # Small tolerance
                    macd_improving = False
                    break
            # Also count as improving if it turned positive
            if macd_values[-1] > 0 and macd_values[0] < 0:
                macd_improving = True
            metrics['macd_improving'] = macd_improving

        # Current trend strength
        metrics['current_trend'] = confirmation_data[-1].get('trend_strength', '')

        # Check if SMA20 was crossed
        sma20_signal = confirmation_data[-1].get('sma20_signal', '')
        metrics['sma20_crossed'] = sma20_signal == 'crossed_above'

        return metrics

    def _passes_reversal_criteria(self, metrics: Dict[str, Any]) -> bool:
        """Check if reversal metrics meet minimum criteria."""

        # Must have minimum positive days
        if metrics.get('positive_days', 0) < self.config.min_positive_days:
            return False

        # RSI must have improved significantly
        if metrics.get('rsi_improvement', 0) < self.config.min_rsi_improvement:
            return False

        # Current trend should not still be in downtrend
        current_trend = metrics.get('current_trend', '')
        if current_trend in self.config.exclude_trend_strengths:
            return False

        metrics['passes'] = True
        return True

    def _create_signal(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Optional[SignalSetup]:
        """Create a SignalSetup from a reversal candidate."""

        # Score the candidate using base scorer
        score_components = self.scorer.score_bullish(candidate)

        # Add reversal-specific bonus points
        reversal_metrics = candidate.get('reversal_metrics', {})
        reversal_bonus = self._calculate_reversal_bonus(reversal_metrics)

        composite_score = min(score_components.composite_score + reversal_bonus, 100)

        # Minimum score threshold
        if composite_score < self.config.min_score_threshold:
            return None

        # Calculate entry levels
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, signal_type)

        # Calculate risk percent
        risk_pct = abs(entry - stop) / entry * 100

        # Build setup description
        confluence_factors = self._build_confluence_factors(candidate)
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

    def _calculate_reversal_bonus(self, reversal_metrics: Dict[str, Any]) -> int:
        """
        Calculate bonus points based on reversal strength.

        Max bonus: 25 points
        """
        bonus = 0

        # SMA20 crossed above: +10 points
        if reversal_metrics.get('sma20_crossed'):
            bonus += 10

        # RSI improvement > 20: +8 points, > 15: +5 points
        rsi_improvement = reversal_metrics.get('rsi_improvement', 0)
        if rsi_improvement > 20:
            bonus += 8
        elif rsi_improvement > 15:
            bonus += 5

        # All 3 days positive: +5 points
        if reversal_metrics.get('positive_days', 0) >= 3:
            bonus += 5

        # MACD improving: +5 points
        if reversal_metrics.get('macd_improving'):
            bonus += 5

        return min(bonus, 25)

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """
        Calculate entry, stop, and take profit levels.

        For reversals:
        - Entry: Current price
        - Stop: 2.5x ATR below (wider for reversal plays)
        - TP1: 2R
        - TP2: 3R
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)

        # ATR in price terms
        atr = current_price * Decimal(str(atr_percent / 100))

        # Entry at current price
        entry = current_price

        # Stop loss: 2.5x ATR below entry
        stop_distance = atr * Decimal(str(self.config.atr_stop_multiplier))
        stop = entry - stop_distance

        # Risk amount
        risk = entry - stop

        # Take profits based on R-multiples
        tp1 = entry + (risk * Decimal(str(self.config.tp1_rr_ratio)))
        tp2 = entry + (risk * Decimal(str(self.config.tp2_rr_ratio)))

        return entry, stop, tp1, tp2

    def _build_confluence_factors(self, candidate: Dict[str, Any]) -> List[str]:
        """Build list of confluence factors for reversal setup."""
        factors = []
        reversal_metrics = candidate.get('reversal_metrics', {})

        # Reversal-specific factors
        positive_days = reversal_metrics.get('positive_days', 0)
        factors.append(f"{positive_days}/3 days positive")

        rsi_improvement = reversal_metrics.get('rsi_improvement', 0)
        if rsi_improvement > 0:
            factors.append(f"RSI +{rsi_improvement:.0f} from low")

        if reversal_metrics.get('macd_improving'):
            factors.append("MACD improving")

        if reversal_metrics.get('sma20_crossed'):
            factors.append("Crossed above SMA20")

        # Current RSI level
        current_rsi = reversal_metrics.get('current_rsi', 50)
        factors.append(f"RSI now {current_rsi:.0f}")

        # Prior downtrend depth
        downtrend_rsi_low = candidate.get('downtrend_rsi_low', 50)
        if downtrend_rsi_low < 30:
            factors.append(f"Was deeply oversold ({downtrend_rsi_low:.0f})")
        elif downtrend_rsi_low < 40:
            factors.append(f"Was oversold ({downtrend_rsi_low:.0f})")

        # Volume
        rel_vol = float(candidate.get('relative_volume') or 0)
        if rel_vol >= 1.5:
            factors.append(f"High volume ({rel_vol:.1f}x)")
        elif rel_vol >= 1.0:
            factors.append(f"Avg volume ({rel_vol:.1f}x)")

        # Price momentum
        price_5d = float(candidate.get('price_change_5d') or candidate.get('perf_1w') or 0)
        if price_5d > 5:
            factors.append(f"5d momentum +{price_5d:.1f}%")

        # Use base class for fundamental factors if enabled
        if self.config.include_fundamentals:
            factors.extend(self._build_fundamental_confluence(candidate))

        return factors[:8]  # Limit to 8 factors

    def _build_description(
        self,
        candidate: Dict[str, Any],
        factors: List[str]
    ) -> str:
        """Build human-readable setup description."""
        ticker = candidate['ticker']
        reversal_metrics = candidate.get('reversal_metrics', {})

        positive_days = reversal_metrics.get('positive_days', 0)
        rsi_improvement = reversal_metrics.get('rsi_improvement', 0)
        downtrend_rsi_low = candidate.get('downtrend_rsi_low', 50)

        desc = f"{ticker} trend reversal detected. "
        desc += f"RSI recovered from {downtrend_rsi_low:.0f} (+{rsi_improvement:.0f}). "
        desc += f"{positive_days} of 3 days positive. "

        if reversal_metrics.get('sma20_crossed'):
            desc += "Crossed above SMA20. "
        if reversal_metrics.get('macd_improving'):
            desc += "MACD turning up. "

        return desc

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime for the reversal."""
        reversal_metrics = candidate.get('reversal_metrics', {})
        downtrend_rsi_low = candidate.get('downtrend_rsi_low', 50)

        if downtrend_rsi_low < 25:
            regime = "Deep Reversal"
        elif downtrend_rsi_low < 35:
            regime = "Oversold Reversal"
        else:
            regime = "Trend Reversal"

        # Add confirmation strength
        positive_days = reversal_metrics.get('positive_days', 0)
        if positive_days >= 3:
            regime += " (Strong)"
        elif positive_days >= 2:
            regime += " (Confirmed)"

        if reversal_metrics.get('sma20_crossed'):
            regime += " + MA Cross"

        return regime
