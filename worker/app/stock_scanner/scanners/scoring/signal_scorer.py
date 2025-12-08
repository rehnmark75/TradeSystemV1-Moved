"""
Signal Scoring System

Multi-component scoring for trading signals:
- Trend Score (30%): Price vs MAs, trend strength, crossovers
- Momentum Score (25%): RSI, MACD, price momentum
- Volume Score (20%): Relative volume, volume trend
- Pattern Score (15%): Candlestick patterns, chart patterns
- Confluence Bonus (10%): Multiple confirming factors

Total composite score: 0-100
Quality Tiers: A+ (85+), A (70-84), B (60-69), C (50-59), D (<50)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from decimal import Decimal
from enum import Enum

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Direction we're scoring for"""
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class ScoreComponents:
    """Individual score components"""
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    pattern_score: float = 0.0
    confluence_bonus: float = 0.0

    # Weights (must sum to 1.0)
    trend_weight: float = 0.30
    momentum_weight: float = 0.25
    volume_weight: float = 0.20
    pattern_weight: float = 0.15
    confluence_weight: float = 0.10

    # Contributing factors for explanation
    trend_factors: List[str] = field(default_factory=list)
    momentum_factors: List[str] = field(default_factory=list)
    volume_factors: List[str] = field(default_factory=list)
    pattern_factors: List[str] = field(default_factory=list)

    @property
    def weighted_trend(self) -> float:
        return self.trend_score * self.trend_weight * 100

    @property
    def weighted_momentum(self) -> float:
        return self.momentum_score * self.momentum_weight * 100

    @property
    def weighted_volume(self) -> float:
        return self.volume_score * self.volume_weight * 100

    @property
    def weighted_pattern(self) -> float:
        return self.pattern_score * self.pattern_weight * 100

    @property
    def weighted_confluence(self) -> float:
        return self.confluence_bonus * self.confluence_weight * 100

    @property
    def composite_score(self) -> int:
        """Calculate total composite score (0-100)"""
        total = (
            self.weighted_trend +
            self.weighted_momentum +
            self.weighted_volume +
            self.weighted_pattern +
            self.weighted_confluence
        )
        return int(min(100, max(0, total)))

    @property
    def all_factors(self) -> List[str]:
        """Get all contributing factors"""
        return (
            self.trend_factors +
            self.momentum_factors +
            self.volume_factors +
            self.pattern_factors
        )


@dataclass
class ScorerConfig:
    """Configuration for signal scorer"""

    # Trend thresholds
    golden_cross_score: float = 1.0
    bullish_ma_score: float = 0.7
    above_sma20_score: float = 0.3
    above_sma50_score: float = 0.3
    above_sma200_score: float = 0.4

    # Momentum thresholds
    macd_cross_score: float = 0.8
    macd_bullish_score: float = 0.5
    rsi_oversold_score: float = 0.9
    rsi_neutral_score: float = 0.5
    strong_momentum_threshold: float = 3.0  # 3% weekly return

    # Volume thresholds
    high_volume_threshold: float = 1.5
    very_high_volume_threshold: float = 2.0
    extreme_volume_threshold: float = 3.0

    # Pattern scores
    strong_pattern_score: float = 1.0
    moderate_pattern_score: float = 0.7
    weak_pattern_score: float = 0.4

    # Confluence
    min_factors_for_bonus: int = 3
    confluence_per_factor: float = 0.15  # 15% bonus per additional factor


class SignalScorer:
    """
    Scores trading signals based on multiple technical factors.

    Usage:
        scorer = SignalScorer()
        components = scorer.score_bullish(candidate_data)
        print(f"Score: {components.composite_score}, Tier: {get_tier(components.composite_score)}")
    """

    # Bullish candlestick patterns by strength
    STRONG_BULLISH_PATTERNS = [
        'bullish_engulfing', 'bullish_marubozu', 'strong_bullish'
    ]
    MODERATE_BULLISH_PATTERNS = [
        'hammer', 'dragonfly_doji', 'inverted_hammer', 'morning_star'
    ]
    WEAK_BULLISH_PATTERNS = [
        'doji', 'spinning_top'
    ]

    # Bearish candlestick patterns
    STRONG_BEARISH_PATTERNS = [
        'bearish_engulfing', 'bearish_marubozu', 'strong_bearish'
    ]
    MODERATE_BEARISH_PATTERNS = [
        'hanging_man', 'shooting_star', 'gravestone_doji', 'evening_star'
    ]

    def __init__(self, config: ScorerConfig = None):
        self.config = config or ScorerConfig()

    def score_bullish(self, candidate: Dict[str, Any]) -> ScoreComponents:
        """
        Score a candidate for bullish (BUY) signal quality.

        Args:
            candidate: Dictionary with technical data

        Returns:
            ScoreComponents with all scoring details
        """
        components = ScoreComponents()

        # Calculate each component
        self._score_trend(candidate, components, SignalDirection.BULLISH)
        self._score_momentum(candidate, components, SignalDirection.BULLISH)
        self._score_volume(candidate, components, SignalDirection.BULLISH)
        self._score_pattern(candidate, components, SignalDirection.BULLISH)
        self._calculate_confluence_bonus(components)

        return components

    def score_bearish(self, candidate: Dict[str, Any]) -> ScoreComponents:
        """
        Score a candidate for bearish (SELL) signal quality.

        Args:
            candidate: Dictionary with technical data

        Returns:
            ScoreComponents with all scoring details
        """
        components = ScoreComponents()

        # Calculate each component
        self._score_trend(candidate, components, SignalDirection.BEARISH)
        self._score_momentum(candidate, components, SignalDirection.BEARISH)
        self._score_volume(candidate, components, SignalDirection.BEARISH)
        self._score_pattern(candidate, components, SignalDirection.BEARISH)
        self._calculate_confluence_bonus(components)

        return components

    # =========================================================================
    # TREND SCORING (30% weight)
    # =========================================================================

    def _score_trend(
        self,
        candidate: Dict[str, Any],
        components: ScoreComponents,
        direction: SignalDirection
    ):
        """
        Score trend alignment.

        Factors:
        - SMA crossover (golden cross / death cross)
        - Price vs SMA20, SMA50, SMA200
        - Overall trend strength
        """
        score = 0.0
        factors = []

        sma_cross = candidate.get('sma_cross_signal', '')
        sma20_signal = candidate.get('sma20_signal', '')
        sma50_signal = candidate.get('sma50_signal', '')
        # trend_strength is categorical: 'strong_up', 'up', 'neutral', 'down', 'strong_down'
        trend_strength = candidate.get('trend_strength', '')
        price_change_20d = float(candidate.get('price_change_20d') or 0)

        if direction == SignalDirection.BULLISH:
            # Golden cross is the strongest bullish signal
            if sma_cross == 'golden_cross':
                score += self.config.golden_cross_score
                factors.append('Golden Cross')
            elif sma_cross == 'bullish':
                score += self.config.bullish_ma_score
                factors.append('Bullish MA Alignment')

            # Price above moving averages
            if sma20_signal in ['above', 'bullish']:
                score += self.config.above_sma20_score
                factors.append('Above SMA20')

            if sma50_signal in ['above', 'bullish']:
                score += self.config.above_sma50_score
                factors.append('Above SMA50')

            # Check 200-day if available
            current_price = float(candidate.get('current_price') or 0)
            sma_200 = float(candidate.get('sma_200') or 0)
            if sma_200 > 0 and current_price > sma_200:
                score += self.config.above_sma200_score
                factors.append('Above SMA200')

            # Trend strength bonus
            if trend_strength == 'strong_up':
                score += 0.3
                factors.append('Strong Uptrend')
            elif trend_strength == 'up':
                score += 0.15

            # 20-day momentum
            if price_change_20d > 10:
                score += 0.2
                factors.append(f'+{price_change_20d:.0f}% (20d)')
            elif price_change_20d > 5:
                score += 0.1

        else:  # BEARISH
            # Death cross is the strongest bearish signal
            if sma_cross == 'death_cross':
                score += self.config.golden_cross_score
                factors.append('Death Cross')
            elif sma_cross == 'bearish':
                score += self.config.bullish_ma_score
                factors.append('Bearish MA Alignment')

            # Price below moving averages
            if sma20_signal in ['below', 'bearish']:
                score += self.config.above_sma20_score
                factors.append('Below SMA20')

            if sma50_signal in ['below', 'bearish']:
                score += self.config.above_sma50_score
                factors.append('Below SMA50')

            # Trend weakness
            if trend_strength in ['strong_down', 'down']:
                score += 0.3
                factors.append('Weak Trend')

            # Negative 20-day momentum
            if price_change_20d < -10:
                score += 0.2
                factors.append(f'{price_change_20d:.0f}% (20d)')
            elif price_change_20d < -5:
                score += 0.1

        # Normalize to 0-1
        components.trend_score = min(1.0, score / 2.5)
        components.trend_factors = factors

    # =========================================================================
    # MOMENTUM SCORING (25% weight)
    # =========================================================================

    def _score_momentum(
        self,
        candidate: Dict[str, Any],
        components: ScoreComponents,
        direction: SignalDirection
    ):
        """
        Score momentum indicators.

        Factors:
        - RSI levels and divergences
        - MACD crossovers and histogram
        - Price momentum (weekly/monthly returns)
        """
        score = 0.0
        factors = []

        rsi_signal = candidate.get('rsi_signal', '')
        macd_cross = candidate.get('macd_cross_signal', '')
        macd_histogram = float(candidate.get('macd_histogram') or 0)
        rsi_14 = float(candidate.get('rsi_14') or 50)
        perf_1w = float(candidate.get('perf_1w') or 0)
        perf_1m = float(candidate.get('perf_1m') or 0)

        if direction == SignalDirection.BULLISH:
            # RSI scoring for bullish
            if rsi_signal in ['oversold', 'oversold_extreme']:
                score += self.config.rsi_oversold_score
                factors.append(f'Oversold RSI ({rsi_14:.0f})')
            elif rsi_signal == 'neutral' and 40 <= rsi_14 <= 60:
                score += self.config.rsi_neutral_score
                factors.append('RSI Room to Run')
            elif rsi_signal == 'overbought_extreme':
                score -= 0.3  # Penalty for extreme overbought

            # MACD scoring
            if macd_cross == 'bullish_cross':
                score += self.config.macd_cross_score
                factors.append('MACD Cross Up')
            elif macd_cross == 'bullish':
                score += self.config.macd_bullish_score
                factors.append('MACD Bullish')

            # Histogram momentum
            if macd_histogram > 0.5:
                score += 0.2
                factors.append('Strong MACD Histogram')

            # Price momentum
            if perf_1w >= self.config.strong_momentum_threshold:
                score += 0.3
                factors.append(f'+{perf_1w:.1f}% (1W)')
            elif perf_1w > 0:
                score += 0.1

            if perf_1m > 10:
                score += 0.2
                factors.append(f'+{perf_1m:.0f}% (1M)')

        else:  # BEARISH
            # RSI scoring for bearish
            if rsi_signal in ['overbought', 'overbought_extreme']:
                score += self.config.rsi_oversold_score
                factors.append(f'Overbought RSI ({rsi_14:.0f})')
            elif rsi_signal == 'neutral' and 40 <= rsi_14 <= 60:
                score += 0.3

            # MACD scoring
            if macd_cross == 'bearish_cross':
                score += self.config.macd_cross_score
                factors.append('MACD Cross Down')
            elif macd_cross == 'bearish':
                score += self.config.macd_bullish_score
                factors.append('MACD Bearish')

            # Histogram momentum
            if macd_histogram < -0.5:
                score += 0.2
                factors.append('Weak MACD Histogram')

            # Negative price momentum
            if perf_1w <= -self.config.strong_momentum_threshold:
                score += 0.3
                factors.append(f'{perf_1w:.1f}% (1W)')

            if perf_1m < -10:
                score += 0.2
                factors.append(f'{perf_1m:.0f}% (1M)')

        # Normalize to 0-1
        components.momentum_score = min(1.0, max(0.0, score / 2.5))
        components.momentum_factors = factors

    # =========================================================================
    # VOLUME SCORING (20% weight)
    # =========================================================================

    def _score_volume(
        self,
        candidate: Dict[str, Any],
        components: ScoreComponents,
        direction: SignalDirection
    ):
        """
        Score volume confirmation.

        Factors:
        - Relative volume vs 20-day average
        - Volume trend (accumulation/distribution)
        - Gap with volume
        """
        score = 0.0
        factors = []

        rel_volume = float(candidate.get('relative_volume') or 1.0)
        gap_signal = candidate.get('gap_signal', '')
        price_change_1d = float(candidate.get('price_change_1d') or 0)

        # Relative volume scoring
        if rel_volume >= self.config.extreme_volume_threshold:
            score += 1.0
            factors.append(f'Extreme Vol ({rel_volume:.1f}x)')
        elif rel_volume >= self.config.very_high_volume_threshold:
            score += 0.8
            factors.append(f'Very High Vol ({rel_volume:.1f}x)')
        elif rel_volume >= self.config.high_volume_threshold:
            score += 0.6
            factors.append(f'High Vol ({rel_volume:.1f}x)')
        elif rel_volume >= 1.0:
            score += 0.3
            factors.append(f'Above Avg Vol ({rel_volume:.1f}x)')
        else:
            # Below average volume is negative
            score -= 0.2

        if direction == SignalDirection.BULLISH:
            # Gap up with volume
            if gap_signal == 'gap_up_large' and rel_volume >= 1.5:
                score += 0.5
                factors.append('Large Gap Up + Volume')
            elif gap_signal == 'gap_up' and rel_volume >= 1.2:
                score += 0.3
                factors.append('Gap Up + Volume')

            # Positive price with high volume = accumulation
            if price_change_1d > 2 and rel_volume >= 1.5:
                score += 0.3
                factors.append('Accumulation Day')

        else:  # BEARISH
            # Gap down with volume
            if gap_signal == 'gap_down_large' and rel_volume >= 1.5:
                score += 0.5
                factors.append('Large Gap Down + Volume')
            elif gap_signal == 'gap_down' and rel_volume >= 1.2:
                score += 0.3
                factors.append('Gap Down + Volume')

            # Negative price with high volume = distribution
            if price_change_1d < -2 and rel_volume >= 1.5:
                score += 0.3
                factors.append('Distribution Day')

        # Normalize to 0-1
        components.volume_score = min(1.0, max(0.0, score / 2.0))
        components.volume_factors = factors

    # =========================================================================
    # PATTERN SCORING (15% weight)
    # =========================================================================

    def _score_pattern(
        self,
        candidate: Dict[str, Any],
        components: ScoreComponents,
        direction: SignalDirection
    ):
        """
        Score candlestick and chart patterns.

        Factors:
        - Candlestick patterns (engulfing, hammer, etc.)
        - 52-week high/low position
        - Support/resistance context
        """
        score = 0.0
        factors = []

        candlestick = candidate.get('candlestick_pattern', '')
        high_low = candidate.get('high_low_signal', '')
        pct_from_high = float(candidate.get('pct_from_52w_high') or 0)

        if direction == SignalDirection.BULLISH:
            # Strong bullish patterns
            if candlestick in self.STRONG_BULLISH_PATTERNS:
                score += self.config.strong_pattern_score
                factors.append(candlestick.replace('_', ' ').title())
            elif candlestick in self.MODERATE_BULLISH_PATTERNS:
                score += self.config.moderate_pattern_score
                factors.append(candlestick.replace('_', ' ').title())
            elif candlestick in self.WEAK_BULLISH_PATTERNS:
                score += self.config.weak_pattern_score

            # Bearish patterns are negative
            if candlestick in self.STRONG_BEARISH_PATTERNS:
                score -= 0.5
            elif candlestick in self.MODERATE_BEARISH_PATTERNS:
                score -= 0.3

            # 52-week position
            if high_low == 'near_high':
                score += 0.6
                factors.append('Near 52W High')
            elif high_low == 'new_high':
                score += 0.8
                factors.append('New 52W High')
            elif high_low == 'new_low':
                score -= 0.3  # Fighting the trend

            # Breakout potential (within 5% of high)
            if -5 <= pct_from_high <= 0:
                score += 0.2
                factors.append(f'{pct_from_high:.1f}% from High')

        else:  # BEARISH
            # Strong bearish patterns
            if candlestick in self.STRONG_BEARISH_PATTERNS:
                score += self.config.strong_pattern_score
                factors.append(candlestick.replace('_', ' ').title())
            elif candlestick in self.MODERATE_BEARISH_PATTERNS:
                score += self.config.moderate_pattern_score
                factors.append(candlestick.replace('_', ' ').title())

            # Bullish patterns are negative for shorts
            if candlestick in self.STRONG_BULLISH_PATTERNS:
                score -= 0.5
            elif candlestick in self.MODERATE_BULLISH_PATTERNS:
                score -= 0.3

            # 52-week position
            if high_low == 'near_low':
                score += 0.6
                factors.append('Near 52W Low')
            elif high_low == 'new_low':
                score += 0.8
                factors.append('New 52W Low')

            # Breakdown potential (far from high)
            if pct_from_high < -20:
                score += 0.3
                factors.append(f'{pct_from_high:.1f}% from High')

        # Normalize to 0-1
        components.pattern_score = min(1.0, max(0.0, score / 2.0))
        components.pattern_factors = factors

    # =========================================================================
    # CONFLUENCE BONUS (10% weight)
    # =========================================================================

    def _calculate_confluence_bonus(self, components: ScoreComponents):
        """
        Calculate confluence bonus based on number of confirming factors.

        More independent confirming signals = higher conviction.
        """
        # Count factors across all categories
        factor_count = (
            len(components.trend_factors) +
            len(components.momentum_factors) +
            len(components.volume_factors) +
            len(components.pattern_factors)
        )

        if factor_count >= self.config.min_factors_for_bonus:
            # Each factor above minimum adds bonus
            extra_factors = factor_count - self.config.min_factors_for_bonus
            bonus = min(1.0, extra_factors * self.config.confluence_per_factor + 0.5)
            components.confluence_bonus = bonus
        else:
            components.confluence_bonus = 0.0


def get_quality_tier(score: int) -> str:
    """Get quality tier from composite score"""
    if score >= 85:
        return "A+"
    elif score >= 70:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C"
    else:
        return "D"
