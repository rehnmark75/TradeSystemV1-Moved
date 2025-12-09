"""
Short Squeeze Scanner

Identifies potential short squeeze candidates based on:
1. High short interest (>15% of float)
2. High days to cover (>3 days)
3. Breaking above resistance
4. Unusual volume surge
5. Technical momentum building

Short Squeeze Mechanics:
- When heavily shorted stocks rise, shorts must cover (buy)
- This buying pressure accelerates the move higher
- Volume surge indicates covering activity
- Breaking resistance triggers more covering
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple

from ..base_scanner import (
    BaseScanner, SignalSetup, SignalType, QualityTier, ScannerConfig
)

logger = logging.getLogger(__name__)


class ShortSqueezeConfig(ScannerConfig):
    """Configuration specific to Short Squeeze Scanner"""

    # Short interest thresholds
    min_short_percent: float = 15.0    # Minimum short % of float
    ideal_short_percent: float = 25.0  # Ideal for squeeze
    min_days_to_cover: float = 3.0     # Minimum days to cover

    # Volume requirements
    min_volume_surge: float = 1.5      # Min volume vs average
    ideal_volume_surge: float = 2.5    # Ideal volume for squeeze

    # Technical requirements
    require_breaking_resistance: bool = True
    min_price_gain_5d: float = 5.0     # Momentum threshold (%)

    # Risk parameters (tighter for momentum plays)
    atr_stop_multiplier: float = 1.5
    tp1_rr_ratio: float = 2.0
    tp2_rr_ratio: float = 3.0

    # Only BUY signals (squeeze is bullish)
    allow_sell_signals: bool = False


class ShortSqueezeScanner(BaseScanner):
    """
    Short Squeeze Scanner - Identifies squeeze setups.

    A short squeeze occurs when:
    1. Large short position exists (>15% of float)
    2. Price starts moving against shorts (up)
    3. Shorts are forced to cover (buy to close)
    4. This buying creates more upward pressure
    5. Cycle accelerates as more shorts cover

    Key Signals:
    - High short interest with rising price
    - Volume surge indicating covering
    - Breaking above resistance levels
    - Momentum indicators turning bullish
    """

    def __init__(self, db_manager, config: ShortSqueezeConfig = None, scorer=None):
        super().__init__(db_manager, config or ShortSqueezeConfig(), scorer)

    @property
    def scanner_name(self) -> str:
        return "short_squeeze"

    @property
    def description(self) -> str:
        return "High short interest stocks showing squeeze potential"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Scan for short squeeze setups.

        Args:
            calculation_date: Date to scan

        Returns:
            List of SignalSetup objects (BUY only)
        """
        logger.info("=" * 60)
        logger.info("SHORT SQUEEZE SCANNER")
        logger.info("=" * 60)

        if calculation_date is None:
            calculation_date = datetime.now().date()

        # Get high short interest candidates
        candidates = await self._get_squeeze_candidates(calculation_date)
        logger.info(f"Found {len(candidates)} high short interest stocks")

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

    async def _get_squeeze_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get stocks with high short interest meeting squeeze criteria.

        Filters:
        - Short interest >= min_short_percent
        - In watchlist (active trading)
        - Has recent volume data
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
                w.high_low_signal,
                m.price_change_1d,
                m.price_change_5d,
                m.price_change_20d,
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
                i.short_percent_float,
                i.short_ratio,
                i.shares_short,
                i.shares_float,
                i.avg_volume as daily_avg_volume,
                i.analyst_rating,
                i.market_cap
            FROM stock_watchlist w
            JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            JOIN stock_instruments i ON w.ticker = i.ticker
            WHERE w.calculation_date = $1
              AND w.tier <= $2
              AND i.short_percent_float IS NOT NULL
              AND i.short_percent_float >= $3
              AND m.data_quality = 'good'
            ORDER BY i.short_percent_float DESC
        """

        rows = await self.db.fetch(
            query,
            calculation_date,
            config.max_tier,
            config.min_short_percent
        )

        return [dict(r) for r in rows]

    async def _evaluate_candidate(
        self,
        candidate: Dict[str, Any],
        calculation_date: datetime
    ) -> Optional[SignalSetup]:
        """
        Evaluate a single squeeze candidate.

        Checks:
        1. Short interest level
        2. Days to cover
        3. Price momentum
        4. Volume confirmation
        5. Technical breakout signals
        """
        ticker = candidate['ticker']
        config = self.config

        # Always BUY for squeeze
        signal_type = SignalType.BUY

        # Check for positive momentum (shorts getting squeezed)
        price_change_5d = float(candidate.get('price_change_5d', 0) or 0)
        price_change_1d = float(candidate.get('price_change_1d', 0) or 0)

        # Need some upward movement to indicate squeeze starting
        if price_change_5d < 0 and price_change_1d < 0:
            logger.debug(f"{ticker}: No upward momentum, skipping")
            return None

        # Check volume surge
        rel_vol = float(candidate.get('relative_volume', 1) or 1)
        if rel_vol < config.min_volume_surge:
            logger.debug(f"{ticker}: Volume too low ({rel_vol:.1f}x)")
            return None

        # Check for resistance break if required
        if config.require_breaking_resistance:
            high_low_signal = candidate.get('high_low_signal', '')
            sma_cross = candidate.get('sma_cross_signal', '')

            is_breaking = (
                high_low_signal in ['near_high', 'new_high'] or
                sma_cross in ['golden_cross', 'bullish'] or
                price_change_5d >= config.min_price_gain_5d
            )

            if not is_breaking:
                logger.debug(f"{ticker}: Not breaking resistance")
                return None

        # Calculate scores
        squeeze_score = self._calculate_squeeze_score(candidate)
        momentum_score = self._calculate_momentum_score(candidate)
        volume_score = self._calculate_volume_score(candidate)
        technical_score = self._calculate_technical_score(candidate)
        risk_score = self._calculate_risk_score(candidate)

        composite_score = int(
            squeeze_score + momentum_score + volume_score +
            technical_score + risk_score
        )

        # Build signal
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, signal_type)
        risk_pct = abs(entry - stop) / entry * 100

        # Build confluence factors
        short_pct = float(candidate.get('short_percent_float', 0))
        confluence_factors = self._build_squeeze_confluence(candidate, short_pct)

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
            trend_score=Decimal(str(technical_score)),
            momentum_score=Decimal(str(momentum_score)),
            volume_score=Decimal(str(volume_score)),
            pattern_score=Decimal(str(squeeze_score)),
            confluence_score=Decimal(str(risk_score)),
            setup_description=f"Short squeeze: {short_pct:.1f}% short interest",
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
        atr = Decimal(str(candidate.get('atr_14', 0) or candidate['current_price'] * 0.03))

        # Entry at current price
        entry = price

        # Stop based on ATR
        stop = self.calculate_atr_based_stop(entry, atr, signal_type)

        # Take profits (wider for squeeze potential)
        tp1, tp2 = self.calculate_take_profits(entry, stop, signal_type)

        return entry, stop, tp1, tp2

    def _calculate_squeeze_score(self, candidate: Dict[str, Any]) -> float:
        """Score squeeze potential (0-25 points)."""
        score = 0
        config = self.config

        short_pct = float(candidate.get('short_percent_float', 0) or 0)
        short_ratio = float(candidate.get('short_ratio', 0) or 0)  # Days to cover

        # Short interest level (0-15)
        if short_pct >= config.ideal_short_percent:
            score += 15
        elif short_pct >= config.min_short_percent:
            ratio = (short_pct - config.min_short_percent) / (config.ideal_short_percent - config.min_short_percent)
            score += int(15 * ratio)

        # Days to cover (0-10)
        if short_ratio >= 5:
            score += 10  # Very high squeeze potential
        elif short_ratio >= config.min_days_to_cover:
            score += int(10 * (short_ratio - config.min_days_to_cover) / (5 - config.min_days_to_cover))

        return min(25, score)

    def _calculate_momentum_score(self, candidate: Dict[str, Any]) -> float:
        """Score price momentum (0-25 points)."""
        score = 0

        price_change_1d = float(candidate.get('price_change_1d', 0) or 0)
        price_change_5d = float(candidate.get('price_change_5d', 0) or 0)

        # 1-day momentum (0-10)
        if price_change_1d >= 5:
            score += 10
        elif price_change_1d >= 2:
            score += 7
        elif price_change_1d > 0:
            score += 4

        # 5-day momentum (0-10)
        if price_change_5d >= 10:
            score += 10
        elif price_change_5d >= 5:
            score += 7
        elif price_change_5d > 0:
            score += 4

        # RSI momentum (0-5) - Rising but not overbought
        rsi = float(candidate.get('rsi_14', 50) or 50)
        if 50 <= rsi <= 70:
            score += 5  # Building momentum
        elif 40 <= rsi < 50:
            score += 3  # Early stage

        return min(25, score)

    def _calculate_volume_score(self, candidate: Dict[str, Any]) -> float:
        """Score volume confirmation (0-20 points)."""
        score = 0
        config = self.config

        rel_vol = float(candidate.get('relative_volume', 1) or 1)

        # Volume surge (0-15)
        if rel_vol >= 3.0:
            score += 15  # Major covering activity
        elif rel_vol >= config.ideal_volume_surge:
            score += 12
        elif rel_vol >= config.min_volume_surge:
            score += 8

        # Consistent volume (0-5)
        # High volume with upward price indicates accumulation/covering
        price_change_1d = float(candidate.get('price_change_1d', 0) or 0)
        if rel_vol >= 2.0 and price_change_1d > 0:
            score += 5

        return min(20, score)

    def _calculate_technical_score(self, candidate: Dict[str, Any]) -> float:
        """Score technical breakout signals (0-15 points)."""
        score = 0

        # Position relative to 52W high (0-5)
        high_low_signal = candidate.get('high_low_signal', '')
        if high_low_signal == 'new_high':
            score += 5
        elif high_low_signal == 'near_high':
            score += 4

        # MA alignment (0-5)
        sma_signal = candidate.get('sma_cross_signal', '')
        if sma_signal == 'golden_cross':
            score += 5
        elif sma_signal == 'bullish':
            score += 3

        # MACD (0-5)
        macd_signal = candidate.get('macd_cross_signal', '')
        if macd_signal == 'bullish_cross':
            score += 5
        elif macd_signal == 'bullish':
            score += 3

        return min(15, score)

    def _calculate_risk_score(self, candidate: Dict[str, Any]) -> float:
        """Score risk factors - lower risk = higher score (0-15 points)."""
        score = 15  # Start with max, deduct for risks

        # Market cap risk (smaller = more volatile)
        market_cap = candidate.get('market_cap', 0) or 0
        if market_cap < 500_000_000:  # Under $500M
            score -= 5
        elif market_cap < 1_000_000_000:  # Under $1B
            score -= 2

        # Tier risk
        tier = candidate.get('tier', 4)
        if tier >= 3:
            score -= 3

        # Overbought risk
        rsi = float(candidate.get('rsi_14', 50) or 50)
        if rsi > 80:
            score -= 5  # Extremely overbought
        elif rsi > 70:
            score -= 2

        return max(0, score)

    def _build_squeeze_confluence(
        self,
        candidate: Dict[str, Any],
        short_pct: float
    ) -> List[str]:
        """Build confluence factors specific to short squeeze."""
        factors = []

        # Core squeeze factor
        factors.append(f"Short Interest: {short_pct:.1f}%")

        # Days to cover
        short_ratio = float(candidate.get('short_ratio', 0) or 0)
        if short_ratio > 0:
            factors.append(f"Days to Cover: {short_ratio:.1f}")

        # Volume
        rel_vol = float(candidate.get('relative_volume', 1) or 1)
        if rel_vol >= 2.0:
            factors.append(f"Volume Surge: {rel_vol:.1f}x")

        # Price momentum
        price_change_5d = float(candidate.get('price_change_5d', 0) or 0)
        if price_change_5d > 0:
            factors.append(f"5D Momentum: +{price_change_5d:.1f}%")

        # Technical breakout
        high_low = candidate.get('high_low_signal', '')
        if high_low in ['near_high', 'new_high']:
            factors.append("Breaking Resistance")

        # MA cross
        sma_signal = candidate.get('sma_cross_signal', '')
        if sma_signal == 'golden_cross':
            factors.append("Golden Cross")
        elif sma_signal == 'bullish':
            factors.append("Above MAs")

        return factors
