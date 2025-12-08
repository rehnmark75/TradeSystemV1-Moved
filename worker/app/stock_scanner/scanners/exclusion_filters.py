"""
Exclusion Filter Engine

Hard filters that immediately reject candidates regardless of score.
These represent conditions that historically lead to losing trades.

Categories:
1. Trend Destruction: Death cross, falling knife
2. Exhaustion: Overbought extreme + bearish, panic selling
3. Pattern Conflicts: Bearish patterns in uptrend setups
4. Liquidity Issues: Low volume, wide spreads
5. Earnings Risk: Upcoming earnings announcements
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ExclusionReason(Enum):
    """Reasons for excluding a candidate"""

    # Trend Destruction
    DEATH_CROSS = "death_cross"
    FALLING_KNIFE = "falling_knife"
    BELOW_ALL_MAS = "below_all_moving_averages"

    # Exhaustion Signals
    OVERBOUGHT_EXTREME = "overbought_extreme_with_bearish_trend"
    OVERSOLD_EXTREME_BEARISH = "oversold_extreme_with_bearish_trend"
    PANIC_SELLING = "panic_selling_high_volume_gap_down"

    # Pattern Conflicts
    BEARISH_PATTERN_UPTREND = "bearish_pattern_contradicts_bullish_setup"
    BULLISH_PATTERN_DOWNTREND = "bullish_pattern_contradicts_bearish_setup"

    # Liquidity Issues
    LOW_LIQUIDITY = "low_liquidity_tier"
    INSUFFICIENT_VOLUME = "insufficient_relative_volume"
    EXTREME_VOLATILITY = "extreme_volatility_risk"

    # Position Issues
    NEW_52W_LOW_BEARISH = "new_52w_low_in_bearish_trend"
    FAR_FROM_SUPPORT = "price_far_from_support"

    # Risk Issues
    EXCESSIVE_ATR = "excessive_atr_risk"
    EARNINGS_IMMINENT = "earnings_announcement_imminent"

    # Score Threshold
    BELOW_MIN_SCORE = "below_minimum_score_threshold"


@dataclass
class ExclusionResult:
    """Result of exclusion check"""
    is_excluded: bool
    reasons: List[ExclusionReason]
    details: Dict[str, Any]

    @property
    def summary(self) -> str:
        """Human-readable summary of exclusion reasons"""
        if not self.is_excluded:
            return "Passed all filters"
        return ", ".join([r.value.replace("_", " ").title() for r in self.reasons])


@dataclass
class FilterConfig:
    """Configuration for exclusion filters"""

    # Liquidity thresholds
    max_tier: int = 3  # Exclude tier 4+
    min_relative_volume: float = 0.5  # At least 50% of average
    min_avg_volume: int = 100000  # Minimum 100k average volume

    # Volatility thresholds
    max_atr_percent: float = 15.0  # Max 15% ATR
    panic_volume_threshold: float = 2.0  # 2x volume on gap down = panic

    # Pattern conflict settings
    exclude_bearish_patterns_bullish: bool = True
    exclude_bullish_patterns_bearish: bool = True

    # Score thresholds
    min_score_for_trade: int = 50

    # Earnings buffer (days)
    earnings_buffer_days: int = 5


class ExclusionFilterEngine:
    """
    Hard filter engine that rejects candidates with deal-breaker conditions.

    Usage:
        engine = ExclusionFilterEngine()

        # Check for bullish setup
        result = engine.check_bullish_exclusions(candidate_data)
        if result.is_excluded:
            print(f"Excluded: {result.summary}")

        # Check for bearish setup
        result = engine.check_bearish_exclusions(candidate_data)
    """

    BEARISH_PATTERNS = [
        'bearish_engulfing', 'hanging_man', 'shooting_star',
        'strong_bearish', 'bearish_marubozu', 'gravestone_doji',
        'evening_star'
    ]

    BULLISH_PATTERNS = [
        'bullish_engulfing', 'hammer', 'dragonfly_doji',
        'bullish_marubozu', 'strong_bullish', 'inverted_hammer',
        'morning_star'
    ]

    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()

    def check_bullish_exclusions(
        self,
        candidate: Dict[str, Any],
        score: int = None
    ) -> ExclusionResult:
        """
        Check if candidate should be excluded from bullish (BUY) signals.

        Args:
            candidate: Candidate data dictionary
            score: Optional pre-calculated score

        Returns:
            ExclusionResult with exclusion status and reasons
        """
        reasons = []
        details = {}

        # === TREND DESTRUCTION FILTERS ===
        sma_cross = candidate.get('sma_cross_signal', '')

        # Death Cross - strongest bearish signal
        if sma_cross == 'death_cross':
            reasons.append(ExclusionReason.DEATH_CROSS)
            details['sma_cross'] = sma_cross

        # Falling Knife - new low + bearish trend
        high_low = candidate.get('high_low_signal', '')
        if high_low == 'new_low' and sma_cross in ['death_cross', 'bearish']:
            reasons.append(ExclusionReason.FALLING_KNIFE)
            details['high_low'] = high_low

        # Below all MAs - fighting strong downtrend
        sma20 = candidate.get('sma20_signal', '')
        sma50 = candidate.get('sma50_signal', '')
        if sma20 in ['below', 'bearish'] and sma50 in ['below', 'bearish'] and sma_cross == 'death_cross':
            reasons.append(ExclusionReason.BELOW_ALL_MAS)

        # === EXHAUSTION FILTERS ===
        rsi_signal = candidate.get('rsi_signal', '')

        # Overbought extreme + bearish trend = likely reversal
        if rsi_signal == 'overbought_extreme' and sma_cross in ['bearish', 'death_cross']:
            reasons.append(ExclusionReason.OVERBOUGHT_EXTREME)
            details['rsi_signal'] = rsi_signal

        # Panic selling - large gap down with high volume
        gap_signal = candidate.get('gap_signal', '')
        rel_volume = float(candidate.get('relative_volume') or 0)

        if gap_signal in ['gap_down_large', 'gap_down'] and rel_volume >= self.config.panic_volume_threshold:
            reasons.append(ExclusionReason.PANIC_SELLING)
            details['gap'] = gap_signal
            details['volume'] = rel_volume

        # === PATTERN CONFLICT FILTERS ===
        if self.config.exclude_bearish_patterns_bullish:
            pattern = candidate.get('candlestick_pattern', '')

            # Bearish pattern when not oversold is a red flag
            if pattern in self.BEARISH_PATTERNS and rsi_signal not in ['oversold', 'oversold_extreme']:
                reasons.append(ExclusionReason.BEARISH_PATTERN_UPTREND)
                details['pattern'] = pattern

        # === LIQUIDITY FILTERS ===
        tier = candidate.get('tier', 4)
        if tier > self.config.max_tier:
            reasons.append(ExclusionReason.LOW_LIQUIDITY)
            details['tier'] = tier

        if rel_volume < self.config.min_relative_volume:
            reasons.append(ExclusionReason.INSUFFICIENT_VOLUME)
            details['relative_volume'] = rel_volume

        # === VOLATILITY FILTERS ===
        atr_percent = float(candidate.get('atr_percent') or 0)
        if atr_percent > self.config.max_atr_percent:
            reasons.append(ExclusionReason.EXCESSIVE_ATR)
            details['atr_percent'] = atr_percent

        # === SCORE FILTER ===
        if score is not None and score < self.config.min_score_for_trade:
            reasons.append(ExclusionReason.BELOW_MIN_SCORE)
            details['score'] = score

        return ExclusionResult(
            is_excluded=len(reasons) > 0,
            reasons=reasons,
            details=details
        )

    def check_bearish_exclusions(
        self,
        candidate: Dict[str, Any],
        score: int = None
    ) -> ExclusionResult:
        """
        Check if candidate should be excluded from bearish (SELL) signals.

        Args:
            candidate: Candidate data dictionary
            score: Optional pre-calculated score

        Returns:
            ExclusionResult with exclusion status and reasons
        """
        reasons = []
        details = {}

        # === TREND DESTRUCTION FILTERS ===
        sma_cross = candidate.get('sma_cross_signal', '')

        # Golden Cross - strongest bullish signal, don't short
        if sma_cross == 'golden_cross':
            reasons.append(ExclusionReason.DEATH_CROSS)  # Opposite for bearish
            details['sma_cross'] = sma_cross

        # New high + bullish trend - don't fight momentum
        high_low = candidate.get('high_low_signal', '')
        if high_low in ['new_high', 'near_high'] and sma_cross in ['golden_cross', 'bullish']:
            reasons.append(ExclusionReason.FALLING_KNIFE)
            details['high_low'] = high_low

        # === EXHAUSTION FILTERS ===
        rsi_signal = candidate.get('rsi_signal', '')

        # Oversold extreme + bullish trend = likely bounce
        if rsi_signal in ['oversold', 'oversold_extreme'] and sma_cross == 'bullish':
            reasons.append(ExclusionReason.OVERSOLD_EXTREME_BEARISH)
            details['rsi_signal'] = rsi_signal

        # === PATTERN CONFLICT FILTERS ===
        if self.config.exclude_bullish_patterns_bearish:
            pattern = candidate.get('candlestick_pattern', '')

            # Bullish pattern when not overbought is a red flag for shorts
            if pattern in self.BULLISH_PATTERNS and rsi_signal not in ['overbought', 'overbought_extreme']:
                reasons.append(ExclusionReason.BULLISH_PATTERN_DOWNTREND)
                details['pattern'] = pattern

        # === LIQUIDITY FILTERS ===
        tier = candidate.get('tier', 4)
        if tier > self.config.max_tier:
            reasons.append(ExclusionReason.LOW_LIQUIDITY)
            details['tier'] = tier

        rel_volume = float(candidate.get('relative_volume') or 0)
        if rel_volume < self.config.min_relative_volume:
            reasons.append(ExclusionReason.INSUFFICIENT_VOLUME)
            details['relative_volume'] = rel_volume

        # === VOLATILITY FILTERS ===
        atr_percent = float(candidate.get('atr_percent') or 0)
        if atr_percent > self.config.max_atr_percent:
            reasons.append(ExclusionReason.EXCESSIVE_ATR)
            details['atr_percent'] = atr_percent

        # === SCORE FILTER ===
        if score is not None and score < self.config.min_score_for_trade:
            reasons.append(ExclusionReason.BELOW_MIN_SCORE)
            details['score'] = score

        return ExclusionResult(
            is_excluded=len(reasons) > 0,
            reasons=reasons,
            details=details
        )

    def get_exclusion_stats(
        self,
        candidates: List[Dict[str, Any]],
        direction: str = 'bullish'
    ) -> Dict[str, Any]:
        """
        Get statistics on exclusion reasons for a batch of candidates.

        Args:
            candidates: List of candidate dictionaries
            direction: 'bullish' or 'bearish'

        Returns:
            Dictionary with exclusion statistics
        """
        check_func = (
            self.check_bullish_exclusions if direction == 'bullish'
            else self.check_bearish_exclusions
        )

        reason_counts = {}
        excluded_count = 0
        passed_count = 0

        for candidate in candidates:
            result = check_func(candidate)

            if result.is_excluded:
                excluded_count += 1
                for reason in result.reasons:
                    reason_counts[reason.value] = reason_counts.get(reason.value, 0) + 1
            else:
                passed_count += 1

        return {
            'total_candidates': len(candidates),
            'excluded': excluded_count,
            'passed': passed_count,
            'exclusion_rate': excluded_count / len(candidates) if candidates else 0,
            'reason_breakdown': dict(sorted(
                reason_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        }


class CompositeFilter:
    """
    Combines multiple filter strategies for advanced filtering.

    Allows creating custom filter combinations for different scanner types.
    """

    def __init__(self):
        self.filters: List[ExclusionFilterEngine] = []
        self.custom_rules: List[callable] = []

    def add_filter(self, filter_engine: ExclusionFilterEngine):
        """Add a filter engine"""
        self.filters.append(filter_engine)

    def add_custom_rule(self, rule: callable):
        """
        Add a custom exclusion rule.

        Rule should be a function that takes (candidate, direction) and returns
        Optional[Tuple[ExclusionReason, Dict]] or None if no exclusion.
        """
        self.custom_rules.append(rule)

    def check(
        self,
        candidate: Dict[str, Any],
        direction: str = 'bullish',
        score: int = None
    ) -> ExclusionResult:
        """
        Run all filters and custom rules.

        Args:
            candidate: Candidate data
            direction: 'bullish' or 'bearish'
            score: Optional pre-calculated score

        Returns:
            Combined ExclusionResult
        """
        all_reasons = []
        all_details = {}

        # Run standard filters
        for filter_engine in self.filters:
            if direction == 'bullish':
                result = filter_engine.check_bullish_exclusions(candidate, score)
            else:
                result = filter_engine.check_bearish_exclusions(candidate, score)

            all_reasons.extend(result.reasons)
            all_details.update(result.details)

        # Run custom rules
        for rule in self.custom_rules:
            try:
                rule_result = rule(candidate, direction)
                if rule_result:
                    reason, details = rule_result
                    all_reasons.append(reason)
                    all_details.update(details)
            except Exception as e:
                logger.warning(f"Custom rule failed: {e}")

        return ExclusionResult(
            is_excluded=len(all_reasons) > 0,
            reasons=all_reasons,
            details=all_details
        )


# Pre-built filter configurations for different scanner types
def get_conservative_filter() -> ExclusionFilterEngine:
    """Get conservative filter for lower-risk setups"""
    return ExclusionFilterEngine(FilterConfig(
        max_tier=2,
        min_relative_volume=0.8,
        max_atr_percent=10.0,
        min_score_for_trade=65,
    ))


def get_aggressive_filter() -> ExclusionFilterEngine:
    """Get aggressive filter for higher-risk setups"""
    return ExclusionFilterEngine(FilterConfig(
        max_tier=3,
        min_relative_volume=0.5,
        max_atr_percent=20.0,
        min_score_for_trade=45,
    ))


def get_momentum_filter() -> ExclusionFilterEngine:
    """Get filter optimized for momentum setups"""
    return ExclusionFilterEngine(FilterConfig(
        max_tier=3,
        min_relative_volume=1.0,  # Require at least average volume
        max_atr_percent=12.0,
        min_score_for_trade=55,
        exclude_bearish_patterns_bullish=True,
    ))


def get_mean_reversion_filter() -> ExclusionFilterEngine:
    """Get filter optimized for mean reversion setups"""
    return ExclusionFilterEngine(FilterConfig(
        max_tier=3,
        min_relative_volume=0.6,  # Can be lower for reversals
        max_atr_percent=15.0,
        min_score_for_trade=50,
        exclude_bearish_patterns_bullish=False,  # Bearish patterns OK if oversold
    ))
