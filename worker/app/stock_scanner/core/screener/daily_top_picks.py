"""
Daily Top Picks Generator

Generates 10-20 high-quality BUY candidates each day based on signal confluence,
quality metrics, and risk-appropriate categorization.

Categories:
1. MOMENTUM RIDERS: Trending stocks with confirmation signals
2. BREAKOUT WATCH: Stocks near 52W highs with volume surge
3. BOUNCE PLAYS: Oversold stocks showing reversal patterns

Scoring System:
- Base Quality (40 pts): Tier, volume, ATR
- Signal Confluence (40 pts): Technical signal alignment
- Confluence Bonus (20 pts): Multiple confirming signals
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SetupCategory(Enum):
    """Trading setup categories"""
    MOMENTUM = "Momentum"
    BREAKOUT = "Breakout"
    MEAN_REVERSION = "Mean Reversion"


@dataclass
class TopPickConfig:
    """Configuration for top picks generation"""

    # Category limits
    max_momentum: int = 7
    max_breakout: int = 6
    max_mean_reversion: int = 5

    # Tier limits (1=best, 4=lowest)
    max_tier: int = 3  # Exclude tier 4 (low liquidity)

    # Volume thresholds
    min_relative_volume_momentum: float = 1.0
    min_relative_volume_breakout: float = 1.3
    high_volume_bonus_threshold: float = 1.5

    # ATR sweet spot (for volatility scoring)
    optimal_atr_min: float = 2.0
    optimal_atr_max: float = 8.0

    # Score thresholds
    min_score_momentum: float = 50.0
    min_score_breakout: float = 55.0
    min_score_mean_reversion: float = 45.0

    # Sector diversity (max picks per sector)
    max_per_sector: int = 3


@dataclass
class TopPick:
    """A single top pick recommendation"""
    ticker: str
    category: SetupCategory
    total_score: float
    base_score: float
    signal_score: float
    confluence_bonus: float
    signal_count: int

    # Price data
    current_price: float
    price_change_1d: float
    price_change_5d: float
    atr_percent: float
    relative_volume: float
    tier: int

    # Signals
    rsi_signal: str
    sma_cross_signal: str
    macd_cross_signal: str
    high_low_signal: str
    gap_signal: str
    candlestick_pattern: str

    # Rank within category
    rank: int = 0

    # Entry/Stop suggestions (based on ATR)
    suggested_stop_pct: float = 0.0
    risk_reward_ratio: float = 0.0

    @property
    def signals_summary(self) -> str:
        """Summary of active bullish signals"""
        signals = []
        if self.sma_cross_signal in ['golden_cross', 'bullish']:
            signals.append('SMA Bullish' if self.sma_cross_signal == 'bullish' else 'Golden Cross')
        if self.macd_cross_signal in ['bullish_cross', 'bullish']:
            signals.append('MACD Bullish' if self.macd_cross_signal == 'bullish' else 'MACD Cross')
        if self.high_low_signal in ['new_high', 'near_high']:
            signals.append('Near 52W High' if self.high_low_signal == 'near_high' else 'New 52W High')
        if self.rsi_signal in ['oversold', 'oversold_extreme']:
            signals.append('Oversold')
        if self.candlestick_pattern in ['bullish_engulfing', 'hammer', 'dragonfly_doji', 'bullish_marubozu', 'strong_bullish']:
            signals.append(self.candlestick_pattern.replace('_', ' ').title())
        if self.gap_signal in ['gap_up', 'gap_up_large']:
            signals.append('Gap Up')
        return ', '.join(signals) if signals else 'Mixed'


class DailyTopPicks:
    """
    Generates daily top pick recommendations based on signal confluence.

    Uses a 3-category system:
    1. Momentum Riders - Trending stocks with bullish confirmation
    2. Breakout Watch - Near 52W highs with volume surge
    3. Bounce Plays - Oversold stocks with reversal patterns
    """

    # Bullish candlestick patterns
    BULLISH_PATTERNS = [
        'bullish_engulfing', 'hammer', 'dragonfly_doji',
        'bullish_marubozu', 'strong_bullish', 'inverted_hammer'
    ]

    # Bearish candlestick patterns (exclusion)
    BEARISH_PATTERNS = [
        'bearish_engulfing', 'hanging_man', 'shooting_star',
        'strong_bearish', 'bearish_marubozu', 'gravestone_doji'
    ]

    def __init__(self, db_manager, config: TopPickConfig = None):
        self.db = db_manager
        self.config = config or TopPickConfig()

    async def generate_daily_picks(
        self,
        calculation_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Generate the daily top picks report.

        Returns:
            Dictionary with picks by category and summary statistics
        """
        logger.info("=" * 60)
        logger.info("GENERATING DAILY TOP PICKS")
        logger.info("=" * 60)

        if calculation_date is None:
            # Use today's date - represents when pipeline ran
            calculation_date = datetime.now().date()

        # Get all eligible candidates
        candidates = await self._get_eligible_candidates(calculation_date)
        logger.info(f"Found {len(candidates)} eligible candidates after exclusions")

        if not candidates:
            return {
                'date': str(calculation_date),
                'momentum': [],
                'breakout': [],
                'mean_reversion': [],
                'total_picks': 0
            }

        # Score and categorize all candidates
        scored_candidates = []
        for candidate in candidates:
            pick = self._score_candidate(candidate)
            if pick:
                scored_candidates.append(pick)

        logger.info(f"Scored {len(scored_candidates)} candidates")

        # Separate into categories
        momentum_picks = [p for p in scored_candidates if p.category == SetupCategory.MOMENTUM]
        breakout_picks = [p for p in scored_candidates if p.category == SetupCategory.BREAKOUT]
        reversion_picks = [p for p in scored_candidates if p.category == SetupCategory.MEAN_REVERSION]

        # Sort each category by score
        momentum_picks.sort(key=lambda x: x.total_score, reverse=True)
        breakout_picks.sort(key=lambda x: x.total_score, reverse=True)
        reversion_picks.sort(key=lambda x: x.total_score, reverse=True)

        # Apply limits and assign ranks
        momentum_final = self._apply_limits(momentum_picks, self.config.max_momentum, self.config.min_score_momentum)
        breakout_final = self._apply_limits(breakout_picks, self.config.max_breakout, self.config.min_score_breakout)
        reversion_final = self._apply_limits(reversion_picks, self.config.max_mean_reversion, self.config.min_score_mean_reversion)

        # Assign ranks
        for i, pick in enumerate(momentum_final, 1):
            pick.rank = i
        for i, pick in enumerate(breakout_final, 1):
            pick.rank = i
        for i, pick in enumerate(reversion_final, 1):
            pick.rank = i

        total_picks = len(momentum_final) + len(breakout_final) + len(reversion_final)

        logger.info(f"\nDaily Top Picks Summary:")
        logger.info(f"  Momentum Riders: {len(momentum_final)}")
        logger.info(f"  Breakout Watch: {len(breakout_final)}")
        logger.info(f"  Bounce Plays: {len(reversion_final)}")
        logger.info(f"  Total: {total_picks}")

        return {
            'date': str(calculation_date),
            'momentum': [self._pick_to_dict(p) for p in momentum_final],
            'breakout': [self._pick_to_dict(p) for p in breakout_final],
            'mean_reversion': [self._pick_to_dict(p) for p in reversion_final],
            'total_picks': total_picks,
            'stats': {
                'candidates_analyzed': len(candidates),
                'candidates_scored': len(scored_candidates),
                'avg_score_momentum': round(sum(p.total_score for p in momentum_final) / len(momentum_final), 1) if momentum_final else 0,
                'avg_score_breakout': round(sum(p.total_score for p in breakout_final) / len(breakout_final), 1) if breakout_final else 0,
                'avg_score_reversion': round(sum(p.total_score for p in reversion_final) / len(reversion_final), 1) if reversion_final else 0,
            }
        }

    async def _get_eligible_candidates(
        self,
        calculation_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get eligible candidates after applying hard exclusions.

        Hard Exclusions:
        - Death cross stocks
        - Overbought extreme + bearish SMA
        - Bearish patterns (unless extremely oversold)
        - Large gap down with high volume (panic selling)
        - Tier 4 stocks (low liquidity)
        - New 52W lows with bearish trend
        """
        query = """
            SELECT
                w.ticker,
                w.tier,
                w.score as watchlist_score,
                w.current_price,
                w.atr_percent,
                w.relative_volume,
                w.price_change_20d,
                w.trend_strength,
                w.rsi_signal,
                w.sma20_signal,
                w.sma50_signal,
                w.sma_cross_signal,
                w.macd_cross_signal,
                w.high_low_signal,
                w.gap_signal,
                w.candlestick_pattern,
                w.pct_from_52w_high,
                m.price_change_1d,
                m.price_change_5d,
                m.rsi_14,
                m.macd_histogram,
                m.perf_1w,
                m.perf_1m
            FROM stock_watchlist w
            LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            WHERE w.calculation_date = $1
              -- Hard exclusions
              AND w.tier <= $2  -- Exclude low liquidity
              AND w.sma_cross_signal != 'death_cross'  -- No death crosses
              AND NOT (
                  w.rsi_signal = 'overbought_extreme'
                  AND w.sma_cross_signal = 'bearish'
              )  -- No overbought extreme + bearish
              AND NOT (
                  w.candlestick_pattern IN ('bearish_engulfing', 'hanging_man', 'shooting_star', 'strong_bearish', 'bearish_marubozu')
                  AND w.rsi_signal NOT IN ('oversold', 'oversold_extreme')
              )  -- No bearish patterns unless oversold
              AND NOT (
                  w.gap_signal IN ('gap_down_large', 'gap_down')
                  AND w.relative_volume > 2.0
              )  -- No panic selling
              AND NOT (
                  w.high_low_signal = 'new_low'
                  AND w.sma_cross_signal = 'bearish'
              )  -- No falling knives
            ORDER BY w.score DESC
        """

        rows = await self.db.fetch(query, calculation_date, self.config.max_tier)
        return [dict(r) for r in rows]

    def _score_candidate(self, candidate: Dict[str, Any]) -> Optional[TopPick]:
        """
        Score a candidate and determine its category.

        Returns None if candidate doesn't fit any category.
        """
        # Extract values
        tier = candidate.get('tier', 4)
        relative_volume = float(candidate.get('relative_volume') or 0)
        atr_pct = float(candidate.get('atr_percent') or 0)
        rsi_signal = candidate.get('rsi_signal') or ''
        sma_cross_signal = candidate.get('sma_cross_signal') or ''
        macd_cross_signal = candidate.get('macd_cross_signal') or ''
        high_low_signal = candidate.get('high_low_signal') or ''
        gap_signal = candidate.get('gap_signal') or ''
        candlestick_pattern = candidate.get('candlestick_pattern') or ''
        price_change_1d = float(candidate.get('price_change_1d') or 0)
        price_change_5d = float(candidate.get('price_change_5d') or 0)

        # === BASE SCORE (40 points max) ===
        base_score = 0.0

        # Tier score (0-20 pts)
        tier_scores = {1: 20, 2: 15, 3: 10}
        base_score += tier_scores.get(tier, 0)

        # Volume score (0-10 pts)
        if relative_volume >= self.config.high_volume_bonus_threshold:
            base_score += 10
        elif relative_volume >= 1.2:
            base_score += 5
        elif relative_volume >= 1.0:
            base_score += 2

        # ATR sweet spot (0-10 pts)
        if self.config.optimal_atr_min <= atr_pct <= self.config.optimal_atr_max:
            base_score += 10
        elif atr_pct > 0:
            base_score += 5

        # === SIGNAL SCORE (40 points max) ===
        signal_score = 0.0
        signal_count = 0

        # Trend signals (0-15 pts)
        if sma_cross_signal == 'golden_cross':
            signal_score += 12
            signal_count += 1
        elif sma_cross_signal == 'bullish':
            signal_score += 8
            signal_count += 1

        # Momentum signals (0-15 pts)
        if macd_cross_signal == 'bullish_cross':
            signal_score += 10
            signal_count += 1
        elif macd_cross_signal == 'bullish':
            signal_score += 6
            signal_count += 1

        # Position signals (0-10 pts)
        if high_low_signal in ['new_high', 'near_high']:
            signal_score += 10
            signal_count += 1
        elif rsi_signal in ['oversold', 'oversold_extreme'] and high_low_signal != 'new_low':
            signal_score += 8
            signal_count += 1

        # Pattern confirmation (0-10 pts)
        if candlestick_pattern in ['bullish_engulfing', 'bullish_marubozu']:
            signal_score += 10
            signal_count += 1
        elif candlestick_pattern in ['hammer', 'dragonfly_doji', 'strong_bullish', 'inverted_hammer']:
            signal_score += 7
            signal_count += 1

        # Gap bonus (0-5 pts)
        if gap_signal == 'gap_up_large':
            signal_score += 5
        elif gap_signal == 'gap_up':
            signal_score += 3

        # === CONFLUENCE BONUS (20 points max) ===
        if signal_count >= 4:
            confluence_bonus = 20.0
        elif signal_count >= 3:
            confluence_bonus = 12.0
        elif signal_count >= 2:
            confluence_bonus = 5.0
        else:
            confluence_bonus = 0.0

        total_score = base_score + signal_score + confluence_bonus

        # === DETERMINE CATEGORY ===
        category = self._determine_category(
            rsi_signal, sma_cross_signal, macd_cross_signal,
            high_low_signal, gap_signal, candlestick_pattern,
            relative_volume, price_change_5d
        )

        if category is None:
            return None

        # Calculate suggested stop (based on ATR)
        suggested_stop_pct = min(atr_pct * 1.5, 8.0)  # 1.5x ATR, max 8%

        return TopPick(
            ticker=candidate['ticker'],
            category=category,
            total_score=round(total_score, 2),
            base_score=round(base_score, 2),
            signal_score=round(signal_score, 2),
            confluence_bonus=round(confluence_bonus, 2),
            signal_count=signal_count,
            current_price=float(candidate.get('current_price') or 0),
            price_change_1d=price_change_1d,
            price_change_5d=price_change_5d,
            atr_percent=atr_pct,
            relative_volume=relative_volume,
            tier=tier,
            rsi_signal=rsi_signal,
            sma_cross_signal=sma_cross_signal,
            macd_cross_signal=macd_cross_signal,
            high_low_signal=high_low_signal,
            gap_signal=gap_signal,
            candlestick_pattern=candlestick_pattern,
            suggested_stop_pct=round(suggested_stop_pct, 2),
            risk_reward_ratio=round(2.0 / suggested_stop_pct * atr_pct, 2) if suggested_stop_pct > 0 else 0
        )

    def _determine_category(
        self,
        rsi_signal: str,
        sma_cross_signal: str,
        macd_cross_signal: str,
        high_low_signal: str,
        gap_signal: str,
        candlestick_pattern: str,
        relative_volume: float,
        price_change_5d: float
    ) -> Optional[SetupCategory]:
        """
        Determine which category a candidate belongs to.

        Priority: Breakout > Momentum > Mean Reversion
        """
        # Breakout: Near highs + volume surge + gap
        if (high_low_signal in ['near_high', 'new_high']
            and relative_volume >= self.config.min_relative_volume_breakout
            and (gap_signal in ['gap_up', 'gap_up_large'] or sma_cross_signal == 'golden_cross')):
            return SetupCategory.BREAKOUT

        # Momentum: Bullish trend + not overbought extreme
        if (sma_cross_signal in ['golden_cross', 'bullish']
            and rsi_signal not in ['overbought_extreme']
            and macd_cross_signal in ['bullish_cross', 'bullish']
            and high_low_signal not in ['new_low', 'near_low']):
            return SetupCategory.MOMENTUM

        # Mean Reversion: Oversold + reversal pattern + not death cross
        if (rsi_signal in ['oversold', 'oversold_extreme']
            and sma_cross_signal not in ['death_cross', 'bearish']
            and candlestick_pattern in self.BULLISH_PATTERNS):
            return SetupCategory.MEAN_REVERSION

        # Secondary Momentum: Strong uptrend even without all confirmations
        if (sma_cross_signal in ['golden_cross', 'bullish']
            and price_change_5d > 2.0
            and rsi_signal not in ['overbought_extreme']):
            return SetupCategory.MOMENTUM

        # Secondary Mean Reversion: Oversold without pattern but strong support
        if (rsi_signal in ['oversold', 'oversold_extreme']
            and sma_cross_signal == 'bullish'
            and high_low_signal != 'new_low'):
            return SetupCategory.MEAN_REVERSION

        return None

    def _apply_limits(
        self,
        picks: List[TopPick],
        max_count: int,
        min_score: float
    ) -> List[TopPick]:
        """Apply count limits and minimum score filter"""
        filtered = [p for p in picks if p.total_score >= min_score]
        return filtered[:max_count]

    def _pick_to_dict(self, pick: TopPick) -> Dict[str, Any]:
        """Convert TopPick to dictionary for JSON serialization"""
        return {
            'rank': pick.rank,
            'ticker': pick.ticker,
            'category': pick.category.value,
            'total_score': pick.total_score,
            'base_score': pick.base_score,
            'signal_score': pick.signal_score,
            'confluence_bonus': pick.confluence_bonus,
            'signal_count': pick.signal_count,
            'signals_summary': pick.signals_summary,
            'current_price': pick.current_price,
            'price_change_1d': pick.price_change_1d,
            'price_change_5d': pick.price_change_5d,
            'atr_percent': pick.atr_percent,
            'relative_volume': pick.relative_volume,
            'tier': pick.tier,
            'rsi_signal': pick.rsi_signal,
            'sma_cross_signal': pick.sma_cross_signal,
            'macd_cross_signal': pick.macd_cross_signal,
            'high_low_signal': pick.high_low_signal,
            'gap_signal': pick.gap_signal,
            'candlestick_pattern': pick.candlestick_pattern,
            'suggested_stop_pct': pick.suggested_stop_pct,
            'risk_reward_ratio': pick.risk_reward_ratio
        }

    async def get_formatted_report(
        self,
        calculation_date: datetime = None
    ) -> str:
        """Generate a formatted text report of daily picks"""
        picks = await self.generate_daily_picks(calculation_date)

        lines = []
        lines.append("=" * 70)
        lines.append(f"  DAILY TOP PICKS - {picks['date']}")
        lines.append("=" * 70)
        lines.append("")

        # Momentum section
        lines.append("TREND RIDERS (Momentum Continuation)")
        lines.append("-" * 50)
        if picks['momentum']:
            for p in picks['momentum']:
                lines.append(
                    f"  {p['rank']}. {p['ticker']:6} | Score: {p['total_score']:5.1f} | "
                    f"${p['current_price']:.2f} ({p['price_change_1d']:+.1f}%) | "
                    f"Vol: {p['relative_volume']:.1f}x | Tier {p['tier']}"
                )
                lines.append(f"     Signals: {p['signals_summary']}")
                lines.append(f"     Stop: -{p['suggested_stop_pct']:.1f}% | R/R: {p['risk_reward_ratio']:.1f}:1")
                lines.append("")
        else:
            lines.append("  No momentum setups today")
            lines.append("")

        # Breakout section
        lines.append("BREAKOUT WATCH (Near 52W Highs)")
        lines.append("-" * 50)
        if picks['breakout']:
            for p in picks['breakout']:
                lines.append(
                    f"  {p['rank']}. {p['ticker']:6} | Score: {p['total_score']:5.1f} | "
                    f"${p['current_price']:.2f} ({p['price_change_1d']:+.1f}%) | "
                    f"Vol: {p['relative_volume']:.1f}x | Tier {p['tier']}"
                )
                lines.append(f"     Signals: {p['signals_summary']}")
                lines.append(f"     Stop: -{p['suggested_stop_pct']:.1f}% | R/R: {p['risk_reward_ratio']:.1f}:1")
                lines.append("")
        else:
            lines.append("  No breakout setups today")
            lines.append("")

        # Mean Reversion section
        lines.append("BOUNCE PLAYS (Mean Reversion)")
        lines.append("-" * 50)
        if picks['mean_reversion']:
            for p in picks['mean_reversion']:
                lines.append(
                    f"  {p['rank']}. {p['ticker']:6} | Score: {p['total_score']:5.1f} | "
                    f"${p['current_price']:.2f} ({p['price_change_1d']:+.1f}%) | "
                    f"Vol: {p['relative_volume']:.1f}x | Tier {p['tier']}"
                )
                lines.append(f"     Signals: {p['signals_summary']}")
                lines.append(f"     Stop: -{p['suggested_stop_pct']:.1f}% | R/R: {p['risk_reward_ratio']:.1f}:1")
                lines.append("")
        else:
            lines.append("  No bounce setups today")
            lines.append("")

        # Summary
        lines.append("=" * 70)
        lines.append(f"SUMMARY: {picks['total_picks']} total picks | "
                    f"Momentum: {len(picks['momentum'])} | "
                    f"Breakout: {len(picks['breakout'])} | "
                    f"Bounce: {len(picks['mean_reversion'])}")
        lines.append("=" * 70)

        return "\n".join(lines)
