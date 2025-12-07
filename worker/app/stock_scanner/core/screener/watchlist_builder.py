"""
Stock Watchlist Builder

Creates a tiered, scored watchlist from screening metrics.
Focuses on high volume + high movement stocks.

Scoring System (0-100):
- Volume Score (30 pts): Based on dollar volume
- Volatility Score (25 pts): Based on ATR %
- Momentum Score (30 pts): Based on price changes and trend
- Relative Strength Score (15 pts): Based on performance vs market

Tiers:
- Tier 1 (80-100): Top 50 stocks - Day trading candidates
- Tier 2 (65-79): Next 150 stocks - Swing trading
- Tier 3 (50-64): Next 300 stocks - Position trading
- Tier 4 (40-49): Universe - Monitoring only
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for scoring thresholds."""

    # Volume thresholds (dollar volume)
    VOL_ELITE: float = 100_000_000    # $100M+ = 30 pts
    VOL_HIGH: float = 50_000_000      # $50M+ = 25 pts
    VOL_GOOD: float = 25_000_000      # $25M+ = 20 pts
    VOL_MIN: float = 10_000_000       # $10M+ = 10 pts (minimum)

    # ATR thresholds (%)
    ATR_OPTIMAL_MIN: float = 2.5      # Optimal range start
    ATR_OPTIMAL_MAX: float = 6.0      # Optimal range end
    ATR_MIN: float = 1.5              # Minimum for consideration

    # Price thresholds
    PRICE_MIN: float = 5.0            # Minimum price
    PRICE_MAX: float = 1000.0         # Maximum price

    # Tier score thresholds
    TIER_1_MIN: float = 80
    TIER_2_MIN: float = 65
    TIER_3_MIN: float = 50
    TIER_4_MIN: float = 40


class WatchlistBuilder:
    """
    Builds a tiered watchlist from screening metrics.

    Scoring Components:
    1. Volume (30 pts) - Dollar volume based
    2. Volatility (25 pts) - ATR % based
    3. Momentum (30 pts) - Price changes + trend
    4. Relative Strength (15 pts) - vs market average
    """

    def __init__(self, db_manager, config: ScoringConfig = None):
        self.db = db_manager
        self.config = config or ScoringConfig()

    async def build_watchlist(
        self,
        calculation_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Build the complete watchlist for a given date.

        Returns:
            Statistics about the watchlist build
        """
        logger.info("=" * 60)
        logger.info("WATCHLIST BUILD")
        logger.info("=" * 60)

        start_time = datetime.now()

        if calculation_date is None:
            calculation_date = datetime.now().date() - timedelta(days=1)

        # Get metrics for all stocks
        metrics = await self._get_all_metrics(calculation_date)
        logger.info(f"Loaded metrics for {len(metrics)} stocks")

        if not metrics:
            logger.warning("No metrics found for date")
            return {'error': 'No metrics available'}

        # Calculate market averages for relative scoring
        market_avg = self._calculate_market_averages(metrics)
        logger.info(f"Market averages: ROC20={market_avg['avg_roc20']:.2f}%")

        # Score each stock
        scored_stocks = []
        for m in metrics:
            score_data = self._score_stock(m, market_avg)
            if score_data:  # Only include stocks that pass filters
                scored_stocks.append(score_data)

        logger.info(f"Scored {len(scored_stocks)} stocks (passed filters)")

        # Sort by score and assign tiers/ranks
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)

        for i, stock in enumerate(scored_stocks):
            stock['rank_overall'] = i + 1
            stock['tier'] = self._assign_tier(stock['score'])

        # Assign rank within tier
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for stock in scored_stocks:
            tier = stock['tier']
            tier_counts[tier] += 1
            stock['rank_in_tier'] = tier_counts[tier]

        # Detect tier changes from previous day
        await self._detect_tier_changes(scored_stocks, calculation_date)

        # Save to database
        await self._save_watchlist(scored_stocks, calculation_date)

        elapsed = (datetime.now() - start_time).total_seconds()

        stats = {
            'calculation_date': str(calculation_date),
            'total_with_metrics': len(metrics),
            'passed_filters': len(scored_stocks),
            'tier_1': tier_counts[1],
            'tier_2': tier_counts[2],
            'tier_3': tier_counts[3],
            'tier_4': tier_counts[4],
            'duration_seconds': round(elapsed, 2)
        }

        logger.info(f"\nWatchlist build complete:")
        logger.info(f"  Tier 1: {tier_counts[1]} stocks (score 80+)")
        logger.info(f"  Tier 2: {tier_counts[2]} stocks (score 65-79)")
        logger.info(f"  Tier 3: {tier_counts[3]} stocks (score 50-64)")
        logger.info(f"  Tier 4: {tier_counts[4]} stocks (score 40-49)")
        logger.info(f"  Total: {len(scored_stocks)} stocks")
        logger.info(f"  Duration: {elapsed:.2f}s")

        return stats

    def _score_stock(
        self,
        metrics: Dict[str, Any],
        market_avg: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Score a single stock.

        Returns None if stock doesn't pass minimum filters.
        """
        # Extract values (convert Decimal to float for arithmetic)
        price = float(metrics.get('current_price') or 0)
        atr_pct = float(metrics.get('atr_percent') or 0)
        dollar_vol = float(metrics.get('avg_dollar_volume') or 0)
        rvol = float(metrics.get('relative_volume') or 0)
        roc_1d = float(metrics.get('price_change_1d') or 0)
        roc_5d = float(metrics.get('price_change_5d') or 0)
        roc_20d = float(metrics.get('price_change_20d') or 0)
        trend = metrics.get('trend_strength') or 'neutral'
        ma_align = metrics.get('ma_alignment') or 'mixed'

        # === FILTERS (must pass all) ===

        # Price filter
        if price < self.config.PRICE_MIN or price > self.config.PRICE_MAX:
            return None

        # Minimum volume filter
        if dollar_vol < self.config.VOL_MIN:
            return None

        # Minimum ATR filter
        if atr_pct < self.config.ATR_MIN:
            return None

        # === SCORING ===

        # 1. Volume Score (0-30 pts)
        volume_score = self._score_volume(dollar_vol)

        # 2. Volatility Score (0-25 pts)
        volatility_score = self._score_volatility(atr_pct)

        # 3. Momentum Score (0-30 pts)
        momentum_score = self._score_momentum(roc_1d, roc_5d, roc_20d, trend, ma_align)

        # 4. Relative Strength Score (0-15 pts)
        rs_score = self._score_relative_strength(roc_20d, market_avg['avg_roc20'])

        # Total score
        total_score = volume_score + volatility_score + momentum_score + rs_score

        # Only include if score meets minimum tier threshold
        if total_score < self.config.TIER_4_MIN:
            return None

        return {
            'ticker': metrics['ticker'],
            'score': round(total_score, 2),
            'volume_score': round(volume_score, 2),
            'volatility_score': round(volatility_score, 2),
            'momentum_score': round(momentum_score, 2),
            'relative_strength_score': round(rs_score, 2),
            'current_price': price,
            'atr_percent': atr_pct,
            'avg_dollar_volume': dollar_vol,
            'relative_volume': rvol,
            'price_change_20d': roc_20d,
            'trend_strength': trend
        }

    def _score_volume(self, dollar_volume: float) -> float:
        """Score based on dollar volume (0-30 pts)."""
        if dollar_volume >= self.config.VOL_ELITE:
            return 30.0
        elif dollar_volume >= self.config.VOL_HIGH:
            return 25.0
        elif dollar_volume >= self.config.VOL_GOOD:
            return 20.0
        elif dollar_volume >= self.config.VOL_MIN:
            # Linear interpolation between 10-20
            ratio = (dollar_volume - self.config.VOL_MIN) / (self.config.VOL_GOOD - self.config.VOL_MIN)
            return 10.0 + (ratio * 10.0)
        else:
            return 0.0

    def _score_volatility(self, atr_percent: float) -> float:
        """Score based on ATR % (0-25 pts). Optimal: 2.5%-6.0%."""
        if atr_percent < self.config.ATR_MIN:
            return 0.0

        # Optimal range gets full points
        if self.config.ATR_OPTIMAL_MIN <= atr_percent <= self.config.ATR_OPTIMAL_MAX:
            return 25.0

        # Below optimal range
        if atr_percent < self.config.ATR_OPTIMAL_MIN:
            ratio = (atr_percent - self.config.ATR_MIN) / (self.config.ATR_OPTIMAL_MIN - self.config.ATR_MIN)
            return 10.0 + (ratio * 15.0)

        # Above optimal range (slightly penalized for extreme volatility)
        if atr_percent <= 10.0:
            return 20.0  # Still good
        elif atr_percent <= 15.0:
            return 15.0  # Higher risk
        else:
            return 10.0  # Extreme volatility

    def _score_momentum(
        self,
        roc_1d: float,
        roc_5d: float,
        roc_20d: float,
        trend: str,
        ma_alignment: str
    ) -> float:
        """Score based on momentum indicators (0-30 pts)."""
        score = 0.0

        # ROC 1-day (0-5 pts)
        if roc_1d > 3:
            score += 5.0
        elif roc_1d > 1:
            score += 3.0
        elif roc_1d > 0:
            score += 1.0

        # ROC 5-day (0-8 pts)
        if roc_5d > 8:
            score += 8.0
        elif roc_5d > 4:
            score += 6.0
        elif roc_5d > 2:
            score += 4.0
        elif roc_5d > 0:
            score += 2.0

        # ROC 20-day (0-10 pts)
        if roc_20d > 15:
            score += 10.0
        elif roc_20d > 10:
            score += 8.0
        elif roc_20d > 5:
            score += 6.0
        elif roc_20d > 0:
            score += 3.0

        # Trend strength (0-4 pts)
        trend_scores = {
            'strong_up': 4.0,
            'up': 3.0,
            'neutral': 1.0,
            'down': 0.0,
            'strong_down': 0.0
        }
        score += trend_scores.get(trend, 1.0)

        # MA alignment (0-3 pts)
        align_scores = {
            'bullish': 3.0,
            'mixed': 1.0,
            'bearish': 0.0
        }
        score += align_scores.get(ma_alignment, 1.0)

        return min(score, 30.0)

    def _score_relative_strength(
        self,
        stock_roc20: float,
        market_roc20: float
    ) -> float:
        """Score based on relative performance vs market (0-15 pts)."""
        # Outperformance vs market
        outperformance = stock_roc20 - market_roc20

        if outperformance > 20:
            return 15.0
        elif outperformance > 10:
            return 12.0
        elif outperformance > 5:
            return 9.0
        elif outperformance > 0:
            return 6.0
        elif outperformance > -5:
            return 3.0
        else:
            return 0.0

    def _assign_tier(self, score: float) -> int:
        """Assign tier based on score."""
        if score >= self.config.TIER_1_MIN:
            return 1
        elif score >= self.config.TIER_2_MIN:
            return 2
        elif score >= self.config.TIER_3_MIN:
            return 3
        else:
            return 4

    def _calculate_market_averages(
        self,
        metrics: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate market-wide averages for relative scoring."""
        roc20_values = [float(m.get('price_change_20d', 0)) for m in metrics if m.get('price_change_20d')]

        return {
            'avg_roc20': sum(roc20_values) / len(roc20_values) if roc20_values else 0.0
        }

    async def _detect_tier_changes(
        self,
        scored_stocks: List[Dict[str, Any]],
        calculation_date: datetime
    ) -> None:
        """Detect tier changes from previous day."""
        prev_date = calculation_date - timedelta(days=1)

        # Get previous tiers
        query = """
            SELECT ticker, tier
            FROM stock_watchlist
            WHERE calculation_date = $1
        """
        rows = await self.db.fetch(query, prev_date)
        prev_tiers = {r['ticker']: r['tier'] for r in rows}

        for stock in scored_stocks:
            ticker = stock['ticker']
            current_tier = stock['tier']
            prev_tier = prev_tiers.get(ticker)

            if prev_tier is None:
                stock['is_new_to_tier'] = True
                stock['tier_change'] = 0
            elif current_tier < prev_tier:
                stock['is_new_to_tier'] = True
                stock['tier_change'] = 1  # Promoted
            elif current_tier > prev_tier:
                stock['is_new_to_tier'] = True
                stock['tier_change'] = -1  # Demoted
            else:
                stock['is_new_to_tier'] = False
                stock['tier_change'] = 0

    async def _get_all_metrics(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get screening metrics for all stocks."""
        query = """
            SELECT *
            FROM stock_screening_metrics
            WHERE calculation_date = $1
              AND data_quality = 'good'
        """
        rows = await self.db.fetch(query, calculation_date)
        return [dict(r) for r in rows]

    async def _save_watchlist(
        self,
        stocks: List[Dict[str, Any]],
        calculation_date: datetime
    ) -> None:
        """Save watchlist to database."""
        # Delete existing entries for this date
        await self.db.execute(
            "DELETE FROM stock_watchlist WHERE calculation_date = $1",
            calculation_date
        )

        # Insert new entries
        for stock in stocks:
            query = """
                INSERT INTO stock_watchlist (
                    ticker, calculation_date,
                    score, volume_score, volatility_score,
                    momentum_score, relative_strength_score,
                    tier, rank_in_tier, rank_overall,
                    current_price, atr_percent, avg_dollar_volume,
                    relative_volume, price_change_20d, trend_strength,
                    is_new_to_tier, tier_change
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18
                )
            """

            await self.db.execute(
                query,
                stock['ticker'],
                calculation_date,
                stock['score'],
                stock['volume_score'],
                stock['volatility_score'],
                stock['momentum_score'],
                stock['relative_strength_score'],
                stock['tier'],
                stock['rank_in_tier'],
                stock['rank_overall'],
                stock['current_price'],
                stock['atr_percent'],
                stock['avg_dollar_volume'],
                stock['relative_volume'],
                stock['price_change_20d'],
                stock['trend_strength'],
                stock.get('is_new_to_tier', False),
                stock.get('tier_change', 0)
            )

    async def get_tier_stocks(
        self,
        tier: int,
        calculation_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """Get stocks for a specific tier."""
        if calculation_date is None:
            calculation_date = datetime.now().date() - timedelta(days=1)

        query = """
            SELECT *
            FROM stock_watchlist
            WHERE calculation_date = $1 AND tier = $2
            ORDER BY rank_in_tier
        """

        rows = await self.db.fetch(query, calculation_date, tier)
        return [dict(r) for r in rows]

    async def get_watchlist_summary(
        self,
        calculation_date: datetime = None
    ) -> Dict[str, Any]:
        """Get summary statistics for the watchlist."""
        if calculation_date is None:
            calculation_date = datetime.now().date() - timedelta(days=1)

        query = """
            SELECT
                tier,
                COUNT(*) as count,
                ROUND(AVG(score), 2) as avg_score,
                ROUND(AVG(atr_percent), 2) as avg_atr,
                ROUND(AVG(avg_dollar_volume) / 1000000, 1) as avg_dollar_vol_m,
                ROUND(AVG(price_change_20d), 2) as avg_roc20,
                SUM(CASE WHEN tier_change = 1 THEN 1 ELSE 0 END) as promoted,
                SUM(CASE WHEN tier_change = -1 THEN 1 ELSE 0 END) as demoted
            FROM stock_watchlist
            WHERE calculation_date = $1
            GROUP BY tier
            ORDER BY tier
        """

        rows = await self.db.fetch(query, calculation_date)
        return {
            'date': str(calculation_date),
            'tiers': [dict(r) for r in rows]
        }
