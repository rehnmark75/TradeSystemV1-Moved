"""
Earnings Momentum Scanner

Scans for stocks showing strong momentum following earnings announcements.

Strategy Logic:
1. Identify stocks with earnings released within last 2 trading days
2. Filter for significant price gap (>5%)
3. Require volume surge (>2x average)
4. Check for continuation pattern forming
5. Score based on gap size, volume, and technical alignment

Use Cases:
- Capture post-earnings drift (PEAD)
- Earnings surprise momentum plays
- Beat-and-raise continuation setups
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple

from ..base_scanner import (
    BaseScanner, SignalSetup, SignalType, QualityTier, ScannerConfig
)

logger = logging.getLogger(__name__)


class EarningsMomentumConfig(ScannerConfig):
    """Configuration specific to Earnings Momentum Scanner"""

    # Earnings timing (days since earnings)
    max_days_since_earnings: int = 3  # Look back N days for earnings
    min_days_since_earnings: int = 0  # Allow same-day earnings

    # Gap requirements
    min_gap_percent: float = 3.0      # Minimum gap % after earnings
    ideal_gap_percent: float = 5.0    # Ideal gap for higher score
    max_gap_percent: float = 25.0     # Too big might mean overextension

    # Volume requirements
    min_earnings_volume_ratio: float = 1.5  # Min volume vs avg
    ideal_volume_ratio: float = 2.5         # Ideal volume for higher score

    # Technical requirements
    require_above_ma: bool = True     # Price above key MAs
    max_pullback_pct: float = 3.0     # Max pullback from gap high

    # Risk parameters for earnings plays
    atr_stop_multiplier: float = 2.0  # Wider stops for earnings volatility
    tp1_rr_ratio: float = 1.5         # First target
    tp2_rr_ratio: float = 2.5         # Second target


class EarningsMomentumScanner(BaseScanner):
    """
    Earnings Momentum Scanner - Catches post-earnings drift.

    Scans for stocks that:
    1. Had earnings within last 3 days
    2. Gapped up/down significantly (>3%)
    3. Show strong volume (>1.5x average)
    4. Maintain momentum post-gap
    5. Align with technical trend

    Scoring considers:
    - Gap magnitude (larger = more conviction)
    - Volume surge (higher = more institutional interest)
    - Technical alignment (above MAs = bullish)
    - Pattern quality (continuation vs reversal)
    """

    def __init__(self, db_manager, config: EarningsMomentumConfig = None, scorer=None):
        super().__init__(db_manager, config or EarningsMomentumConfig(), scorer)

    @property
    def scanner_name(self) -> str:
        return "earnings_momentum"

    @property
    def description(self) -> str:
        return "Post-earnings momentum continuation plays"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Scan for earnings momentum setups.

        Args:
            calculation_date: Date to scan

        Returns:
            List of SignalSetup objects
        """
        logger.info("=" * 60)
        logger.info("EARNINGS MOMENTUM SCANNER")
        logger.info("=" * 60)

        if calculation_date is None:
            calculation_date = datetime.now().date()

        # Get candidates with recent earnings
        candidates = await self._get_earnings_candidates(calculation_date)
        logger.info(f"Found {len(candidates)} stocks with recent earnings")

        signals = []
        for candidate in candidates:
            signal = await self._evaluate_candidate(candidate, calculation_date)
            if signal and signal.composite_score >= self.config.min_score_threshold:
                signals.append(signal)

        # Sort by score and limit
        signals.sort(key=lambda x: x.composite_score, reverse=True)
        signals = signals[:self.config.max_signals_per_run]

        # Log summary
        high_quality = sum(1 for s in signals if s.is_high_quality)
        self.log_scan_summary(len(candidates), len(signals), high_quality)

        return signals

    async def _get_earnings_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get stocks with recent earnings that meet initial criteria.

        Filters:
        - Earnings within last N days
        - In watchlist (tier 1-3)
        - Has price and volume data
        """
        config = self.config

        query = """
            SELECT
                w.ticker,
                w.tier,
                w.score as watchlist_score,
                w.current_price,
                w.atr_percent,
                w.relative_volume,
                w.trend_strength,
                w.rsi_signal,
                w.sma_cross_signal,
                w.macd_cross_signal,
                w.candlestick_pattern,
                w.pct_from_52w_high,
                m.price_change_1d,
                m.price_change_5d,
                m.rsi_14,
                m.macd_histogram,
                m.atr_14,
                m.sma_20,
                m.sma_50,
                m.sma_200,
                m.avg_volume_20,
                m.data_quality,
                i.name,
                i.sector,
                i.earnings_date,
                i.short_percent_float,
                i.analyst_rating
            FROM stock_watchlist w
            JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            JOIN stock_instruments i ON w.ticker = i.ticker
            WHERE w.calculation_date = $1
              AND w.tier <= $2
              AND i.earnings_date IS NOT NULL
              AND i.earnings_date >= $3
              AND i.earnings_date <= $4
              AND m.data_quality = 'good'
              AND w.relative_volume >= $5
            ORDER BY w.score DESC
        """

        # Calculate earnings date range
        earnings_start = calculation_date - timedelta(days=config.max_days_since_earnings)
        earnings_end = calculation_date - timedelta(days=config.min_days_since_earnings)

        rows = await self.db.fetch(
            query,
            calculation_date,
            config.max_tier,
            earnings_start,
            earnings_end,
            config.min_earnings_volume_ratio
        )

        return [dict(r) for r in rows]

    async def _evaluate_candidate(
        self,
        candidate: Dict[str, Any],
        calculation_date: datetime
    ) -> Optional[SignalSetup]:
        """
        Evaluate a single earnings candidate.

        Checks:
        1. Gap magnitude
        2. Volume confirmation
        3. Technical alignment
        4. Pattern quality
        """
        ticker = candidate['ticker']
        config = self.config

        # Get gap information (from 1d price change as proxy)
        price_change_1d = float(candidate.get('price_change_1d', 0) or 0)

        # Determine signal direction based on gap
        if abs(price_change_1d) < config.min_gap_percent:
            return None  # Gap not significant enough

        if price_change_1d > 0:
            signal_type = SignalType.BUY
        else:
            signal_type = SignalType.SELL
            price_change_1d = abs(price_change_1d)  # Work with positive value

        # Check gap is not too extreme
        if price_change_1d > config.max_gap_percent:
            logger.debug(f"{ticker}: Gap too large ({price_change_1d:.1f}%), skipping")
            return None

        # Check technical alignment for BUY
        if signal_type == SignalType.BUY:
            if config.require_above_ma:
                sma_signal = candidate.get('sma_cross_signal', '')
                if sma_signal in ['death_cross', 'bearish']:
                    logger.debug(f"{ticker}: Not above MAs, skipping BUY")
                    return None

        # Calculate scores
        trend_score = self._calculate_trend_score(candidate, signal_type)
        momentum_score = self._calculate_momentum_score(candidate, signal_type, price_change_1d)
        volume_score = self._calculate_volume_score(candidate)
        pattern_score = self._calculate_pattern_score(candidate, signal_type)
        confluence_score = self._calculate_confluence_score(candidate, signal_type)

        composite_score = int(
            trend_score + momentum_score + volume_score +
            pattern_score + confluence_score
        )

        # Build signal
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, signal_type)
        risk_pct = abs(entry - stop) / entry * 100

        # Build confluence factors
        confluence_factors = self._build_earnings_confluence(candidate, signal_type, price_change_1d)

        quality_tier = QualityTier.from_score(composite_score)
        position_size = self.calculate_position_size(
            Decimal(str(config.max_risk_per_trade_pct)),
            entry, stop, quality_tier
        )

        return SignalSetup(
            ticker=ticker,
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            signal_timestamp=datetime.now(),
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=Decimal(str(config.tp1_rr_ratio)),
            risk_percent=Decimal(str(risk_pct)),
            composite_score=composite_score,
            quality_tier=quality_tier,
            trend_score=Decimal(str(trend_score)),
            momentum_score=Decimal(str(momentum_score)),
            volume_score=Decimal(str(volume_score)),
            pattern_score=Decimal(str(pattern_score)),
            confluence_score=Decimal(str(confluence_score)),
            setup_description=f"Earnings momentum: {price_change_1d:+.1f}% gap",
            confluence_factors=confluence_factors,
            timeframe="daily",
            market_regime=candidate.get('trend_strength', ''),
            suggested_position_size_pct=position_size,
            max_risk_per_trade_pct=Decimal(str(config.max_risk_per_trade_pct)),
            raw_data=candidate
        )

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """Calculate entry, stop, and take-profit levels."""
        price = Decimal(str(candidate['current_price']))
        atr = Decimal(str(candidate.get('atr_14', 0) or candidate['current_price'] * 0.02))

        # Entry at current price
        entry = price

        # Stop based on ATR (wider for earnings volatility)
        stop = self.calculate_atr_based_stop(entry, atr, signal_type)

        # Take profits
        tp1, tp2 = self.calculate_take_profits(entry, stop, signal_type)

        return entry, stop, tp1, tp2

    def _calculate_trend_score(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> float:
        """Score trend alignment (0-25 points)."""
        score = 0
        config = self.config

        # SMA alignment (0-10)
        sma_signal = candidate.get('sma_cross_signal', '')
        if signal_type == SignalType.BUY:
            if sma_signal == 'golden_cross':
                score += 10
            elif sma_signal == 'bullish':
                score += 7
            elif sma_signal in ['', 'neutral']:
                score += 3
        else:  # SELL
            if sma_signal == 'death_cross':
                score += 10
            elif sma_signal == 'bearish':
                score += 7

        # Trend strength (0-10)
        trend = candidate.get('trend_strength', '')
        if signal_type == SignalType.BUY:
            if trend == 'strong_up':
                score += 10
            elif trend == 'up':
                score += 7
            elif trend == 'neutral':
                score += 3
        else:
            if trend == 'strong_down':
                score += 10
            elif trend == 'down':
                score += 7

        # Price position relative to 52W high (0-5)
        pct_from_high = float(candidate.get('pct_from_52w_high', -50) or -50)
        if signal_type == SignalType.BUY:
            if pct_from_high > -5:  # Near high
                score += 5
            elif pct_from_high > -15:
                score += 3

        return min(25, score)

    def _calculate_momentum_score(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType,
        gap_pct: float
    ) -> float:
        """Score momentum quality (0-25 points)."""
        score = 0
        config = self.config

        # Gap size (0-10)
        if gap_pct >= config.ideal_gap_percent:
            score += 10
        elif gap_pct >= config.min_gap_percent:
            score += int(10 * gap_pct / config.ideal_gap_percent)

        # RSI position (0-8)
        rsi = float(candidate.get('rsi_14', 50) or 50)
        if signal_type == SignalType.BUY:
            if 40 <= rsi <= 70:  # Not overbought
                score += 8
            elif rsi < 40:  # Pullback opportunity
                score += 6
        else:
            if 30 <= rsi <= 60:
                score += 8
            elif rsi > 60:
                score += 6

        # MACD alignment (0-7)
        macd_signal = candidate.get('macd_cross_signal', '')
        if signal_type == SignalType.BUY:
            if macd_signal in ['bullish_cross', 'bullish']:
                score += 7
        else:
            if macd_signal in ['bearish_cross', 'bearish']:
                score += 7

        return min(25, score)

    def _calculate_volume_score(self, candidate: Dict[str, Any]) -> float:
        """Score volume confirmation (0-20 points)."""
        score = 0
        config = self.config

        rel_vol = float(candidate.get('relative_volume', 1) or 1)

        # Volume surge (0-15)
        if rel_vol >= config.ideal_volume_ratio:
            score += 15
        elif rel_vol >= config.min_earnings_volume_ratio:
            score += int(15 * (rel_vol - 1) / (config.ideal_volume_ratio - 1))

        # Tier bonus (0-5) - better liquidity stocks get higher score
        tier = candidate.get('tier', 4)
        if tier == 1:
            score += 5
        elif tier == 2:
            score += 3
        elif tier == 3:
            score += 1

        return min(20, score)

    def _calculate_pattern_score(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> float:
        """Score chart pattern quality (0-15 points)."""
        score = 0

        pattern = candidate.get('candlestick_pattern', '')

        # Bullish patterns for BUY
        bullish_patterns = ['bullish_engulfing', 'hammer', 'dragonfly_doji',
                          'strong_bullish', 'bullish_marubozu']
        # Bearish patterns for SELL
        bearish_patterns = ['bearish_engulfing', 'shooting_star', 'hanging_man',
                          'strong_bearish', 'bearish_marubozu']

        if signal_type == SignalType.BUY:
            if pattern in bullish_patterns:
                score += 10
            elif pattern == 'doji':
                score += 3  # Indecision, might be pause before continuation
        else:
            if pattern in bearish_patterns:
                score += 10
            elif pattern == 'doji':
                score += 3

        # Gap signal bonus (0-5)
        gap_signal = candidate.get('gap_signal', '')
        if signal_type == SignalType.BUY and 'gap_up' in gap_signal:
            score += 5
        elif signal_type == SignalType.SELL and 'gap_down' in gap_signal:
            score += 5

        return min(15, score)

    def _calculate_confluence_score(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> float:
        """Score additional confluence factors (0-15 points)."""
        score = 0

        # Analyst rating (0-4)
        rating = candidate.get('analyst_rating', '')
        if rating and 'buy' in str(rating).lower():
            score += 4
        elif rating and 'hold' in str(rating).lower():
            score += 2

        # Short interest - high short can mean squeeze potential (0-4)
        short_pct = float(candidate.get('short_percent_float', 0) or 0)
        if signal_type == SignalType.BUY and short_pct >= 15:
            score += 4  # Squeeze potential

        # Sector alignment would go here (0-4)
        # (Would need market sector data)

        # Watchlist score bonus (0-3)
        watchlist_score = float(candidate.get('watchlist_score', 50) or 50)
        if watchlist_score >= 80:
            score += 3
        elif watchlist_score >= 70:
            score += 2
        elif watchlist_score >= 60:
            score += 1

        return min(15, score)

    def _build_earnings_confluence(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType,
        gap_pct: float
    ) -> List[str]:
        """Build confluence factors specific to earnings plays."""
        factors = []

        # Core earnings factor
        factors.append(f"Earnings Gap: {gap_pct:+.1f}%")

        # Volume
        rel_vol = float(candidate.get('relative_volume', 1) or 1)
        if rel_vol >= 2.0:
            factors.append(f"Volume Surge: {rel_vol:.1f}x")

        # Add standard technical factors
        factors.extend(self.build_confluence_factors(candidate))

        # Short squeeze potential
        short_pct = float(candidate.get('short_percent_float', 0) or 0)
        if signal_type == SignalType.BUY and short_pct >= 15:
            factors.append(f"Short Squeeze Potential: {short_pct:.1f}%")

        return factors
