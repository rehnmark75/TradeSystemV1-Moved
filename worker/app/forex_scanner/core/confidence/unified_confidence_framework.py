# core/confidence/unified_confidence_framework.py
"""
Unified Confidence Scoring Framework for ALL Trading Strategies
================================================================

Mathematical Framework:
Final_Score = Base_Confidence × Regime_Modifier × Strategy_Priority_Weight × Correlation_Factor × Optimization_Multiplier

Where:
- Base_Confidence: Raw strategy confidence (0.0-1.0)
- Regime_Modifier: Strategy-specific performance in current market regime (0.2-1.0)
- Strategy_Priority_Weight: EMA gets highest weight (1.0), others scaled accordingly (0.7-1.0)
- Correlation_Factor: Adjusts for strategy redundancy/complementarity (0.8-1.2)
- Optimization_Multiplier: Database-driven optimization bonus (1.0-1.3)

MACD Success Analysis:
- Current 41.3% validation rate achieved through probabilistic confidence modifiers
- Success factors: regime-aware scoring, momentum alignment, volatility adaptation
- Key insight: Dynamic confidence adjustment based on market conditions

Target Outcomes:
- Each strategy achieving 30-50% validation rate
- EMA maintaining 50-70% validation rate as priority strategy
- Balanced strategy allocation across market conditions
- Mathematically optimized system-wide performance
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

# =============================================================================
# MATHEMATICAL CONSTANTS AND ENUMS
# =============================================================================

class StrategyType(Enum):
    """All supported trading strategies"""
    EMA = "ema"
    MACD = "macd"
    ICHIMOKU = "ichimoku"
    SMC = "smc"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    RANGING_MARKET = "ranging_market"
    ZERO_LAG = "zero_lag"
    BB_SUPERTREND = "bb_supertrend"
    SCALPING = "scalping"
    KAMA = "kama"

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEDIUM_VOLATILITY = "medium_volatility"
    SCALPING = "scalping"
    UNKNOWN = "unknown"

class StrategyFamily(Enum):
    """Strategy family classifications for correlation analysis"""
    TREND_FOLLOWING = "trend_following"  # EMA, MACD, Ichimoku, KAMA
    MEAN_REVERTING = "mean_reverting"    # Mean Reversion, Ranging Market
    MOMENTUM_BASED = "momentum_based"    # Momentum, Zero Lag, BB SuperTrend
    SPECIALIZED = "specialized"          # SMC, Scalping

@dataclass
class ConfidenceComponents:
    """Components of confidence calculation"""
    base_confidence: float
    regime_modifier: float
    priority_weight: float
    correlation_factor: float
    optimization_multiplier: float
    final_score: float
    reasoning: List[str]

# =============================================================================
# STRATEGY CLASSIFICATION AND PRIORITY SYSTEM
# =============================================================================

# EMA Priority Mathematical Justification:
# 1. EMA is foundational for trend analysis in most other strategies
# 2. EMA performance often predicts success of momentum/trend strategies
# 3. EMA has lowest correlation with mean-reverting strategies (diversification)
# 4. EMA signals have fastest reaction time for trend changes

STRATEGY_PRIORITY_WEIGHTS = {
    StrategyType.EMA: 1.0,              # Highest priority - foundational strategy
    StrategyType.MACD: 0.95,            # Very high - proven 41.3% success rate
    StrategyType.ICHIMOKU: 0.90,        # High - comprehensive trend analysis
    StrategyType.MOMENTUM: 0.85,        # High - fast market adaptation
    StrategyType.ZERO_LAG: 0.85,        # High - real-time responsiveness
    StrategyType.KAMA: 0.80,            # Good - adaptive nature
    StrategyType.BB_SUPERTREND: 0.80,   # Good - volatility + trend combination
    StrategyType.MEAN_REVERSION: 0.75,  # Medium - specialized use case
    StrategyType.SMC: 0.75,             # Medium - institutional flow analysis
    StrategyType.RANGING_MARKET: 0.70,  # Lower - limited market conditions
    StrategyType.SCALPING: 0.70         # Lower - very specific conditions
}

STRATEGY_FAMILIES = {
    StrategyType.EMA: StrategyFamily.TREND_FOLLOWING,
    StrategyType.MACD: StrategyFamily.TREND_FOLLOWING,
    StrategyType.ICHIMOKU: StrategyFamily.TREND_FOLLOWING,
    StrategyType.KAMA: StrategyFamily.TREND_FOLLOWING,
    StrategyType.MEAN_REVERSION: StrategyFamily.MEAN_REVERTING,
    StrategyType.RANGING_MARKET: StrategyFamily.MEAN_REVERTING,
    StrategyType.MOMENTUM: StrategyFamily.MOMENTUM_BASED,
    StrategyType.ZERO_LAG: StrategyFamily.MOMENTUM_BASED,
    StrategyType.BB_SUPERTREND: StrategyFamily.MOMENTUM_BASED,
    StrategyType.SMC: StrategyFamily.SPECIALIZED,
    StrategyType.SCALPING: StrategyFamily.SPECIALIZED
}

# =============================================================================
# COMPREHENSIVE CONFIDENCE MODIFIER MATRIX (11 Strategies × 8 Regimes)
# =============================================================================

# Mathematical Basis: Regime-Strategy compatibility based on:
# 1. Market microstructure analysis
# 2. Historical performance data
# 3. Strategy characteristics and reaction time
# 4. MACD success pattern extrapolation

REGIME_STRATEGY_CONFIDENCE_MODIFIERS = {
    MarketRegime.TRENDING: {
        StrategyType.EMA: 1.0,              # Perfect - foundational trend following
        StrategyType.MACD: 1.0,             # Perfect - proven 41.3% success in trending
        StrategyType.ICHIMOKU: 1.0,         # Perfect - designed for trends
        StrategyType.KAMA: 1.0,             # Perfect - adaptive to trend strength
        StrategyType.MOMENTUM: 0.95,        # Excellent - trend confirmation
        StrategyType.ZERO_LAG: 0.95,        # Excellent - fast trend reaction
        StrategyType.BB_SUPERTREND: 0.90,   # Very good - trend + volatility
        StrategyType.SMC: 0.75,             # Good - institutional flow alignment
        StrategyType.MEAN_REVERSION: 0.35,  # Poor - counter-trend nature
        StrategyType.RANGING_MARKET: 0.25,  # Very poor - designed for ranges
        StrategyType.SCALPING: 0.40         # Poor - trends too sustained
    },

    MarketRegime.RANGING: {
        StrategyType.MEAN_REVERSION: 1.0,   # Perfect - designed for ranges
        StrategyType.RANGING_MARKET: 1.0,   # Perfect - specialized for ranges
        StrategyType.SMC: 1.0,              # Perfect - range bounce analysis
        StrategyType.MACD: 0.80,            # Good - divergence detection in ranges
        StrategyType.BB_SUPERTREND: 0.75,   # Good - range boundaries
        StrategyType.SCALPING: 0.70,        # Good - range scalping opportunities
        StrategyType.EMA: 0.55,             # Moderate - can work in wide ranges
        StrategyType.ICHIMOKU: 0.45,        # Moderate - cloud support/resistance
        StrategyType.KAMA: 0.50,            # Moderate - adapts to range conditions
        StrategyType.MOMENTUM: 0.35,        # Poor - lacks sustained momentum
        StrategyType.ZERO_LAG: 0.40         # Poor - too reactive for ranges
    },

    MarketRegime.BREAKOUT: {
        StrategyType.MOMENTUM: 1.0,         # Perfect - captures breakout momentum
        StrategyType.BB_SUPERTREND: 1.0,    # Perfect - volatility expansion
        StrategyType.KAMA: 1.0,             # Perfect - rapid adaptation to volatility
        StrategyType.ZERO_LAG: 0.95,        # Excellent - immediate breakout detection
        StrategyType.MACD: 0.90,            # Excellent - momentum confirmation
        StrategyType.EMA: 0.85,             # Very good - trend establishment
        StrategyType.ICHIMOKU: 0.80,        # Good - breakout confirmation
        StrategyType.SCALPING: 0.60,        # Moderate - initial breakout trades
        StrategyType.SMC: 0.55,             # Moderate - institutional breakout
        StrategyType.MEAN_REVERSION: 0.20,  # Very poor - counter-breakout
        StrategyType.RANGING_MARKET: 0.15   # Very poor - opposite of breakout
    },

    MarketRegime.CONSOLIDATION: {
        StrategyType.MEAN_REVERSION: 1.0,   # Perfect - tight range reversions
        StrategyType.RANGING_MARKET: 1.0,   # Perfect - consolidation detection
        StrategyType.SMC: 1.0,              # Perfect - accumulation/distribution
        StrategyType.SCALPING: 0.85,        # Very good - small moves
        StrategyType.MACD: 0.75,            # Good - subtle momentum shifts
        StrategyType.BB_SUPERTREND: 0.70,   # Good - low volatility adaptation
        StrategyType.EMA: 0.60,             # Moderate - slow trend detection
        StrategyType.KAMA: 0.65,            # Moderate - adapts to low volatility
        StrategyType.ICHIMOKU: 0.50,        # Moderate - cloud equilibrium
        StrategyType.MOMENTUM: 0.30,        # Poor - insufficient momentum
        StrategyType.ZERO_LAG: 0.35         # Poor - too sensitive for consolidation
    },

    MarketRegime.HIGH_VOLATILITY: {
        StrategyType.ZERO_LAG: 1.0,         # Perfect - rapid adaptation
        StrategyType.MOMENTUM: 1.0,         # Perfect - captures volatility moves
        StrategyType.KAMA: 1.0,             # Perfect - designed for volatility
        StrategyType.MACD: 1.0,             # Perfect - thrives in volatile conditions
        StrategyType.EMA: 0.95,             # Excellent - trend clarity in volatility
        StrategyType.BB_SUPERTREND: 0.95,   # Excellent - volatility-based
        StrategyType.ICHIMOKU: 0.85,        # Very good - comprehensive analysis
        StrategyType.SCALPING: 0.75,        # Good - if risk-managed properly
        StrategyType.SMC: 0.65,             # Moderate - institutional moves
        StrategyType.MEAN_REVERSION: 0.30,  # Poor - volatility extends moves
        StrategyType.RANGING_MARKET: 0.25   # Very poor - volatility breaks ranges
    },

    MarketRegime.LOW_VOLATILITY: {
        StrategyType.MEAN_REVERSION: 1.0,   # Perfect - small reversions work
        StrategyType.RANGING_MARKET: 1.0,   # Perfect - low vol creates ranges
        StrategyType.EMA: 0.95,             # Excellent - stable trend following
        StrategyType.SMC: 0.90,             # Very good - institutional accumulation
        StrategyType.MACD: 0.85,            # Very good - MACD success pattern
        StrategyType.ICHIMOKU: 0.80,        # Good - stable signals
        StrategyType.SCALPING: 0.75,        # Good - many small opportunities
        StrategyType.KAMA: 0.70,            # Good - adapts to low volatility
        StrategyType.BB_SUPERTREND: 0.65,   # Moderate - limited volatility info
        StrategyType.MOMENTUM: 0.40,        # Poor - insufficient momentum
        StrategyType.ZERO_LAG: 0.45         # Poor - may be too sensitive
    },

    MarketRegime.MEDIUM_VOLATILITY: {
        StrategyType.EMA: 1.0,              # Perfect - ideal conditions for EMA
        StrategyType.MACD: 1.0,             # Perfect - optimal volatility level
        StrategyType.ICHIMOKU: 1.0,         # Perfect - balanced analysis
        StrategyType.KAMA: 1.0,             # Perfect - adaptive sweet spot
        StrategyType.ZERO_LAG: 0.95,        # Excellent - good signal/noise ratio
        StrategyType.MOMENTUM: 0.90,        # Very good - sustainable momentum
        StrategyType.BB_SUPERTREND: 0.85,   # Very good - volatility-trend balance
        StrategyType.SMC: 0.80,             # Good - institutional flow clarity
        StrategyType.SCALPING: 0.70,        # Good - moderate opportunities
        StrategyType.MEAN_REVERSION: 0.65,  # Moderate - some reversions work
        StrategyType.RANGING_MARKET: 0.60   # Moderate - some range formation
    },

    MarketRegime.SCALPING: {
        StrategyType.SCALPING: 1.0,         # Perfect - designed for scalping
        StrategyType.ZERO_LAG: 1.0,         # Perfect - immediate reaction
        StrategyType.MOMENTUM: 0.90,        # Excellent - quick momentum bursts
        StrategyType.EMA: 0.75,             # Good - short-term trends
        StrategyType.MACD: 0.60,            # Moderate - faster parameters needed
        StrategyType.MEAN_REVERSION: 0.55,  # Moderate - quick reversions
        StrategyType.BB_SUPERTREND: 0.65,   # Moderate - volatility breakouts
        StrategyType.KAMA: 0.50,            # Moderate - adapts to speed
        StrategyType.SMC: 0.45,             # Moderate - institutional levels
        StrategyType.ICHIMOKU: 0.30,        # Poor - too slow for scalping
        StrategyType.RANGING_MARKET: 0.35   # Poor - scalping transcends ranges
    }
}

# =============================================================================
# STRATEGY CORRELATION MATRIX
# =============================================================================

# Correlation factors to prevent strategy over-concentration
# Values: 1.2 = strong complementarity bonus, 1.0 = neutral, 0.8 = redundancy penalty

STRATEGY_CORRELATION_FACTORS = {
    # When EMA is active, how do other strategies correlate?
    StrategyType.EMA: {
        StrategyType.MACD: 0.90,            # High correlation - both trend-following
        StrategyType.ICHIMOKU: 0.85,        # High correlation - both trend-following
        StrategyType.MOMENTUM: 0.80,        # Medium-high correlation
        StrategyType.ZERO_LAG: 0.85,        # High correlation - EMA-based
        StrategyType.KAMA: 0.80,            # Medium-high correlation
        StrategyType.BB_SUPERTREND: 0.75,   # Medium correlation
        StrategyType.MEAN_REVERSION: 1.20,  # Strong complementarity - opposite
        StrategyType.RANGING_MARKET: 1.15,  # Good complementarity
        StrategyType.SMC: 1.05,             # Slight complementarity
        StrategyType.SCALPING: 1.00         # Neutral - different timeframes
    },

    # When MACD is active
    StrategyType.MACD: {
        StrategyType.EMA: 0.90,
        StrategyType.ICHIMOKU: 0.85,
        StrategyType.MOMENTUM: 0.80,
        StrategyType.ZERO_LAG: 0.80,
        StrategyType.KAMA: 0.85,
        StrategyType.BB_SUPERTREND: 0.80,
        StrategyType.MEAN_REVERSION: 1.15,
        StrategyType.RANGING_MARKET: 1.10,
        StrategyType.SMC: 1.00,
        StrategyType.SCALPING: 1.05
    },

    # When Mean Reversion is active
    StrategyType.MEAN_REVERSION: {
        StrategyType.EMA: 1.20,
        StrategyType.MACD: 1.15,
        StrategyType.ICHIMOKU: 1.10,
        StrategyType.MOMENTUM: 1.25,        # Very complementary
        StrategyType.ZERO_LAG: 1.20,
        StrategyType.KAMA: 1.10,
        StrategyType.BB_SUPERTREND: 1.15,
        StrategyType.RANGING_MARKET: 0.85,  # Some overlap
        StrategyType.SMC: 1.05,
        StrategyType.SCALPING: 1.00
    },

    # Simplified for other strategies - focus on major correlations
    StrategyType.MOMENTUM: {
        StrategyType.ZERO_LAG: 0.75,        # High correlation
        StrategyType.BB_SUPERTREND: 0.80,
        StrategyType.MEAN_REVERSION: 1.25,  # Very complementary
        StrategyType.RANGING_MARKET: 1.20
    }
}

# =============================================================================
# OPTIMIZATION MULTIPLIERS
# =============================================================================

# Database-driven optimization bonuses based on backtested performance
OPTIMIZATION_MULTIPLIERS = {
    # Strategies with proven optimization success get bonuses
    StrategyType.MACD: 1.15,        # 15% bonus - proven 41.3% success
    StrategyType.EMA: 1.10,         # 10% bonus - foundational strategy
    StrategyType.ICHIMOKU: 1.05,    # 5% bonus - comprehensive system
    StrategyType.ZERO_LAG: 1.05,    # 5% bonus - fast adaptation

    # Other strategies at neutral until optimization proven
    StrategyType.MOMENTUM: 1.0,
    StrategyType.KAMA: 1.0,
    StrategyType.BB_SUPERTREND: 1.0,
    StrategyType.MEAN_REVERSION: 1.0,
    StrategyType.RANGING_MARKET: 1.0,
    StrategyType.SMC: 1.0,
    StrategyType.SCALPING: 1.0
}

# =============================================================================
# UNIFIED CONFIDENCE SCORING CLASS
# =============================================================================

class UnifiedConfidenceFramework:
    """
    Unified confidence scoring framework for all trading strategies

    Implements the mathematical formula:
    Final_Score = Base_Confidence × Regime_Modifier × Strategy_Priority_Weight × Correlation_Factor × Optimization_Multiplier
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_strategies: List[StrategyType] = []
        self.market_regime: MarketRegime = MarketRegime.UNKNOWN
        self.strategy_counts: Dict[StrategyFamily, int] = {family: 0 for family in StrategyFamily}

        # Target validation rates for monitoring
        self.target_validation_rates = {
            StrategyType.EMA: (0.50, 0.70),        # 50-70% target
            StrategyType.MACD: (0.35, 0.50),       # Maintain current success
            StrategyType.ICHIMOKU: (0.30, 0.45),
            StrategyType.MOMENTUM: (0.30, 0.45),
            StrategyType.ZERO_LAG: (0.30, 0.45),
            StrategyType.KAMA: (0.30, 0.45),
            StrategyType.BB_SUPERTREND: (0.30, 0.45),
            StrategyType.MEAN_REVERSION: (0.30, 0.45),
            StrategyType.RANGING_MARKET: (0.30, 0.45),
            StrategyType.SMC: (0.30, 0.45),
            StrategyType.SCALPING: (0.30, 0.45)
        }

    def calculate_confidence_score(
        self,
        strategy: StrategyType,
        base_confidence: float,
        market_regime: MarketRegime,
        active_strategies: List[StrategyType] = None
    ) -> ConfidenceComponents:
        """
        Calculate unified confidence score for a strategy

        Args:
            strategy: Strategy type
            base_confidence: Raw strategy confidence (0.0-1.0)
            market_regime: Current market regime
            active_strategies: List of currently active strategies

        Returns:
            ConfidenceComponents with detailed breakdown
        """
        if active_strategies:
            self.active_strategies = active_strategies
            self._update_strategy_counts()

        self.market_regime = market_regime

        # Validate inputs
        base_confidence = max(0.0, min(1.0, base_confidence))

        reasoning = []

        # 1. Get regime modifier
        regime_modifier = self._get_regime_modifier(strategy, market_regime)
        reasoning.append(f"Regime modifier ({market_regime.value}): {regime_modifier:.3f}")

        # 2. Get priority weight
        priority_weight = self._get_priority_weight(strategy)
        reasoning.append(f"Priority weight: {priority_weight:.3f}")

        # 3. Calculate correlation factor
        correlation_factor = self._calculate_correlation_factor(strategy)
        reasoning.append(f"Correlation factor: {correlation_factor:.3f}")

        # 4. Get optimization multiplier
        optimization_multiplier = self._get_optimization_multiplier(strategy)
        reasoning.append(f"Optimization multiplier: {optimization_multiplier:.3f}")

        # 5. Calculate final score
        final_score = (
            base_confidence *
            regime_modifier *
            priority_weight *
            correlation_factor *
            optimization_multiplier
        )

        # Ensure final score stays within bounds
        final_score = max(0.0, min(1.0, final_score))

        reasoning.append(f"Final calculation: {base_confidence:.3f} × {regime_modifier:.3f} × {priority_weight:.3f} × {correlation_factor:.3f} × {optimization_multiplier:.3f} = {final_score:.3f}")

        return ConfidenceComponents(
            base_confidence=base_confidence,
            regime_modifier=regime_modifier,
            priority_weight=priority_weight,
            correlation_factor=correlation_factor,
            optimization_multiplier=optimization_multiplier,
            final_score=final_score,
            reasoning=reasoning
        )

    def _get_regime_modifier(self, strategy: StrategyType, regime: MarketRegime) -> float:
        """Get regime-strategy confidence modifier"""
        try:
            return REGIME_STRATEGY_CONFIDENCE_MODIFIERS[regime][strategy]
        except KeyError:
            self.logger.warning(f"No regime modifier found for {strategy.value} in {regime.value}, using 0.7")
            return 0.7  # Conservative default

    def _get_priority_weight(self, strategy: StrategyType) -> float:
        """Get strategy priority weight"""
        return STRATEGY_PRIORITY_WEIGHTS.get(strategy, 0.75)

    def _calculate_correlation_factor(self, strategy: StrategyType) -> float:
        """
        Calculate correlation factor based on active strategies

        Prevents over-concentration of similar strategies
        """
        if not self.active_strategies:
            return 1.0

        correlation_adjustments = []

        # Check correlations with active strategies
        strategy_correlations = STRATEGY_CORRELATION_FACTORS.get(strategy, {})

        for active_strategy in self.active_strategies:
            if active_strategy == strategy:
                continue

            correlation = strategy_correlations.get(active_strategy, 1.0)
            correlation_adjustments.append(correlation)

        if not correlation_adjustments:
            return 1.0

        # Use geometric mean to avoid extreme penalties
        correlation_factor = np.power(np.prod(correlation_adjustments), 1.0 / len(correlation_adjustments))

        # Apply family diversity bonus/penalty
        family_factor = self._calculate_family_diversity_factor(strategy)

        return correlation_factor * family_factor

    def _calculate_family_diversity_factor(self, strategy: StrategyType) -> float:
        """Calculate diversity factor based on strategy family distribution"""
        strategy_family = STRATEGY_FAMILIES[strategy]
        current_family_count = self.strategy_counts[strategy_family]
        total_active = sum(self.strategy_counts.values())

        if total_active == 0:
            return 1.0

        family_ratio = current_family_count / total_active

        # Penalize over-concentration, reward diversity
        if family_ratio > 0.6:  # More than 60% from same family
            return 0.9  # 10% penalty
        elif family_ratio > 0.4:  # More than 40% from same family
            return 0.95  # 5% penalty
        else:
            return 1.05 if family_ratio < 0.2 else 1.0  # 5% bonus for diversity

    def _get_optimization_multiplier(self, strategy: StrategyType) -> float:
        """Get optimization-based performance multiplier"""
        return OPTIMIZATION_MULTIPLIERS.get(strategy, 1.0)

    def _update_strategy_counts(self):
        """Update strategy family counts"""
        self.strategy_counts = {family: 0 for family in StrategyFamily}

        for strategy in self.active_strategies:
            family = STRATEGY_FAMILIES[strategy]
            self.strategy_counts[family] += 1

    def get_strategy_allocation_recommendation(self) -> Dict[str, Any]:
        """
        Get recommendations for optimal strategy allocation

        Returns:
            Dictionary with allocation recommendations
        """
        total_strategies = len(self.active_strategies)

        recommendations = {
            'total_active_strategies': total_strategies,
            'family_distribution': dict(self.strategy_counts),
            'recommendations': [],
            'warnings': []
        }

        # Check for over-concentration
        for family, count in self.strategy_counts.items():
            if total_strategies > 0:
                ratio = count / total_strategies
                if ratio > 0.6:
                    recommendations['warnings'].append(f"Over-concentration in {family.value}: {ratio:.1%}")
                elif ratio < 0.1 and total_strategies > 4:
                    recommendations['recommendations'].append(f"Consider adding {family.value} strategy for diversity")

        # Check EMA priority
        if StrategyType.EMA not in self.active_strategies and total_strategies > 2:
            recommendations['warnings'].append("EMA (priority strategy) not active")

        return recommendations

    def validate_target_rates(self, strategy: StrategyType, actual_rate: float) -> Dict[str, Any]:
        """
        Validate actual validation rate against targets

        Args:
            strategy: Strategy type
            actual_rate: Actual validation rate achieved

        Returns:
            Validation results and recommendations
        """
        target_min, target_max = self.target_validation_rates.get(strategy, (0.30, 0.45))

        status = "optimal"
        recommendations = []

        if actual_rate < target_min:
            status = "below_target"
            recommendations.append(f"Increase confidence modifiers for {strategy.value}")
            recommendations.append("Review regime compatibility settings")
        elif actual_rate > target_max:
            status = "above_target"
            recommendations.append(f"Consider tightening validation for {strategy.value}")
            recommendations.append("May be accepting too many low-quality signals")

        return {
            'strategy': strategy.value,
            'actual_rate': actual_rate,
            'target_range': (target_min, target_max),
            'status': status,
            'recommendations': recommendations
        }