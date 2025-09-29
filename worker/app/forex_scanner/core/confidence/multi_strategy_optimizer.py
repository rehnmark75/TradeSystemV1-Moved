# core/confidence/multi_strategy_optimizer.py
"""
Multi-Strategy Optimization Framework
====================================

Prevents strategy over-concentration and optimizes system-wide performance

Mathematical Optimization Model:
===============================

Objective Function:
Maximize: Σ(i=1 to n) w_i × c_i × (1 - correlation_penalty_i)

Subject to:
1. Σ(i=1 to n) family_allocation_i ≤ max_family_allocation
2. trend_following_strategies ≤ 0.6 × total_strategies
3. mean_reverting_strategies ≤ 0.4 × total_strategies
4. EMA_priority: w_ema = 1.0, others ≤ 0.95
5. diversity_constraint: min_families ≥ 2 when total > 3

Where:
- w_i = strategy weight
- c_i = confidence score
- correlation_penalty_i = penalty for strategy correlation
- family_allocation_i = allocation within strategy family

Portfolio Theory Application:
============================

Using Modern Portfolio Theory adapted for trading strategies:

Expected_Return = Σ(w_i × E[R_i])
Portfolio_Risk = Σ(w_i × w_j × σ_i × σ_j × ρ_ij)

Where:
- E[R_i] = expected validation rate for strategy i
- σ_i = validation rate volatility for strategy i
- ρ_ij = correlation between strategies i and j

Risk-Adjusted Score = Expected_Return / Portfolio_Risk

Target Allocation:
=================

Optimal strategy mix based on mathematical optimization:
- Trend Following: 40-60% (EMA, MACD, Ichimoku, KAMA)
- Mean Reverting: 20-35% (Mean Reversion, Ranging Market)
- Momentum Based: 15-30% (Momentum, Zero Lag, BB SuperTrend)
- Specialized: 5-15% (SMC, Scalping)
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from scipy.optimize import minimize
from datetime import datetime, timedelta
import itertools

from .unified_confidence_framework import (
    StrategyType, MarketRegime, StrategyFamily,
    UnifiedConfidenceFramework, ConfidenceComponents,
    STRATEGY_PRIORITY_WEIGHTS, STRATEGY_FAMILIES
)

logger = logging.getLogger(__name__)

# =============================================================================
# OPTIMIZATION CONSTANTS AND CONSTRAINTS
# =============================================================================

# Maximum allocation per strategy family (prevent over-concentration)
MAX_FAMILY_ALLOCATION = {
    StrategyFamily.TREND_FOLLOWING: 0.60,   # Max 60% trend-following
    StrategyFamily.MEAN_REVERTING: 0.35,    # Max 35% mean-reverting
    StrategyFamily.MOMENTUM_BASED: 0.30,    # Max 30% momentum-based
    StrategyFamily.SPECIALIZED: 0.15        # Max 15% specialized
}

# Target allocation ranges for optimal diversification
TARGET_ALLOCATION_RANGES = {
    StrategyFamily.TREND_FOLLOWING: (0.40, 0.60),
    StrategyFamily.MEAN_REVERTING: (0.20, 0.35),
    StrategyFamily.MOMENTUM_BASED: (0.15, 0.30),
    StrategyFamily.SPECIALIZED: (0.05, 0.15)
}

# Correlation matrix between strategy families
FAMILY_CORRELATION_MATRIX = np.array([
    [1.00, -0.30, 0.60, 0.20],  # Trend Following
    [-0.30, 1.00, -0.40, 0.10], # Mean Reverting
    [0.60, -0.40, 1.00, 0.30],  # Momentum Based
    [0.20, 0.10, 0.30, 1.00]   # Specialized
])

# Strategy validation rate volatility (historical estimation)
STRATEGY_VOLATILITY = {
    StrategyType.EMA: 0.08,             # Most stable
    StrategyType.MACD: 0.12,            # Moderate volatility
    StrategyType.ICHIMOKU: 0.10,        # Stable
    StrategyType.MOMENTUM: 0.15,        # Higher volatility
    StrategyType.ZERO_LAG: 0.18,        # High volatility
    StrategyType.KAMA: 0.12,            # Moderate volatility
    StrategyType.BB_SUPERTREND: 0.14,   # Moderate-high volatility
    StrategyType.MEAN_REVERSION: 0.16,  # Market dependent
    StrategyType.RANGING_MARKET: 0.20,  # Very market dependent
    StrategyType.SMC: 0.13,             # Moderate volatility
    StrategyType.SCALPING: 0.22         # Highest volatility
}

@dataclass
class OptimizationConstraints:
    """Optimization constraints for strategy allocation"""
    max_total_strategies: int = 6
    min_strategies_for_diversification: int = 3
    max_family_concentration: float = 0.60
    min_ema_weight: float = 1.0
    max_correlation_penalty: float = 0.3
    require_family_diversity: bool = True

@dataclass
class StrategyAllocation:
    """Optimized strategy allocation result"""
    strategy: StrategyType
    weight: float
    confidence: float
    family: StrategyFamily
    allocation_reason: str
    risk_contribution: float

@dataclass
class OptimizationResult:
    """Complete optimization result"""
    allocations: List[StrategyAllocation]
    total_expected_return: float
    portfolio_risk: float
    risk_adjusted_score: float
    family_distribution: Dict[StrategyFamily, float]
    constraints_satisfied: bool
    optimization_warnings: List[str]

# =============================================================================
# MULTI-STRATEGY OPTIMIZER CLASS
# =============================================================================

class MultiStrategyOptimizer:
    """
    Multi-strategy optimization framework for preventing over-concentration
    and maximizing system-wide performance using portfolio theory principles.
    """

    def __init__(self, unified_framework: UnifiedConfidenceFramework):
        self.logger = logging.getLogger(__name__)
        self.framework = unified_framework
        self.constraints = OptimizationConstraints()

        # Historical performance tracking
        self.strategy_performance_history: Dict[StrategyType, List[float]] = {}
        self.regime_performance_history: Dict[MarketRegime, Dict[StrategyType, List[float]]] = {}

        # Optimization state
        self.current_allocations: List[StrategyAllocation] = []
        self.last_optimization_time = datetime.now()

    def optimize_strategy_portfolio(
        self,
        available_strategies: Dict[StrategyType, float],
        market_regime: MarketRegime,
        constraints: Optional[OptimizationConstraints] = None
    ) -> OptimizationResult:
        """
        Optimize strategy portfolio using mathematical optimization

        Args:
            available_strategies: Dictionary of {strategy: base_confidence}
            market_regime: Current market regime
            constraints: Optional optimization constraints

        Returns:
            OptimizationResult with optimal allocations
        """

        if constraints:
            self.constraints = constraints

        self.logger.info(f"🔬 Starting multi-strategy optimization for {len(available_strategies)} strategies")
        self.logger.info(f"🌍 Market regime: {market_regime.value}")

        # Step 1: Calculate enhanced confidence scores for all strategies
        enhanced_scores = self._calculate_enhanced_confidence_scores(
            available_strategies, market_regime
        )

        # Step 2: Apply portfolio optimization
        optimization_result = self._optimize_portfolio(enhanced_scores, market_regime)

        # Step 3: Validate constraints
        optimization_result = self._validate_constraints(optimization_result)

        # Step 4: Store results
        self.current_allocations = optimization_result.allocations
        self.last_optimization_time = datetime.now()

        self.logger.info(f"✅ Optimization complete: {len(optimization_result.allocations)} strategies selected")
        self.logger.info(f"📊 Risk-adjusted score: {optimization_result.risk_adjusted_score:.3f}")

        return optimization_result

    def _calculate_enhanced_confidence_scores(
        self,
        available_strategies: Dict[StrategyType, float],
        market_regime: MarketRegime
    ) -> Dict[StrategyType, ConfidenceComponents]:
        """Calculate enhanced confidence scores with portfolio considerations"""

        enhanced_scores = {}
        active_strategies = list(available_strategies.keys())

        for strategy, base_confidence in available_strategies.items():
            # Get base confidence score
            confidence_components = self.framework.calculate_confidence_score(
                strategy, base_confidence, market_regime, active_strategies
            )

            # Apply portfolio-specific adjustments
            portfolio_adjustment = self._calculate_portfolio_adjustment(
                strategy, active_strategies, market_regime
            )

            # Update final score with portfolio adjustment
            adjusted_score = confidence_components.final_score * portfolio_adjustment
            adjusted_score = max(0.0, min(1.0, adjusted_score))

            confidence_components.final_score = adjusted_score
            confidence_components.reasoning.append(f"Portfolio adjustment: {portfolio_adjustment:.3f}")

            enhanced_scores[strategy] = confidence_components

        return enhanced_scores

    def _calculate_portfolio_adjustment(
        self,
        strategy: StrategyType,
        active_strategies: List[StrategyType],
        market_regime: MarketRegime
    ) -> float:
        """
        Calculate portfolio-specific adjustments for strategy allocation

        Mathematical basis: Portfolio theory risk-return optimization
        """

        # 1. Diversification benefit
        strategy_family = STRATEGY_FAMILIES[strategy]
        family_count = sum(1 for s in active_strategies if STRATEGY_FAMILIES[s] == strategy_family)
        total_strategies = len(active_strategies)

        if total_strategies <= 1:
            diversification_factor = 1.0
        else:
            family_concentration = family_count / total_strategies
            max_allowed = MAX_FAMILY_ALLOCATION[strategy_family]

            if family_concentration > max_allowed:
                # Penalty for over-concentration
                diversification_factor = 0.7 + 0.3 * (max_allowed / family_concentration)
            else:
                # Bonus for good diversification
                diversification_factor = 1.0 + 0.1 * (1.0 - family_concentration / max_allowed)

        # 2. Regime-specific portfolio balance
        regime_balance_factor = self._calculate_regime_balance_factor(
            strategy, active_strategies, market_regime
        )

        # 3. Historical performance factor
        performance_factor = self._calculate_historical_performance_factor(strategy, market_regime)

        # 4. Risk-adjusted factor
        risk_factor = self._calculate_risk_adjustment_factor(strategy)

        # Combine all factors
        portfolio_adjustment = (
            diversification_factor *
            regime_balance_factor *
            performance_factor *
            risk_factor
        )

        return max(0.5, min(1.5, portfolio_adjustment))  # Bound adjustments

    def _optimize_portfolio(
        self,
        enhanced_scores: Dict[StrategyType, ConfidenceComponents],
        market_regime: MarketRegime
    ) -> OptimizationResult:
        """
        Apply mathematical optimization to select optimal strategy portfolio

        Uses constrained optimization to maximize risk-adjusted returns
        while respecting diversification constraints.
        """

        strategies = list(enhanced_scores.keys())
        n_strategies = len(strategies)

        if n_strategies <= self.constraints.max_total_strategies:
            # If we have few strategies, include all that meet minimum threshold
            selected_strategies = [
                (strategy, components.final_score)
                for strategy, components in enhanced_scores.items()
                if components.final_score >= 0.3  # Minimum viability threshold
            ]
        else:
            # Use optimization for strategy selection
            selected_strategies = self._solve_optimization_problem(enhanced_scores, market_regime)

        # Create allocation results
        allocations = []
        total_weight = 0.0

        for strategy, score in selected_strategies:
            # Calculate weight based on score and constraints
            base_weight = score * STRATEGY_PRIORITY_WEIGHTS[strategy]

            # Apply EMA priority
            if strategy == StrategyType.EMA:
                weight = max(base_weight, 0.20)  # Minimum 20% for EMA
            else:
                weight = base_weight

            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for i, (strategy, score) in enumerate(selected_strategies):
                normalized_weight = (
                    score * STRATEGY_PRIORITY_WEIGHTS[strategy] / total_weight
                    if total_weight > 0 else 0.0
                )

                # Special handling for EMA priority
                if strategy == StrategyType.EMA and normalized_weight < 0.15:
                    normalized_weight = 0.15  # Minimum EMA allocation

                allocation = StrategyAllocation(
                    strategy=strategy,
                    weight=normalized_weight,
                    confidence=score,
                    family=STRATEGY_FAMILIES[strategy],
                    allocation_reason=self._get_allocation_reason(strategy, score, market_regime),
                    risk_contribution=self._calculate_risk_contribution(strategy, normalized_weight)
                )
                allocations.append(allocation)

        # Calculate portfolio metrics
        expected_return = sum(alloc.weight * alloc.confidence for alloc in allocations)
        portfolio_risk = self._calculate_portfolio_risk(allocations)
        risk_adjusted_score = expected_return / portfolio_risk if portfolio_risk > 0 else 0.0

        # Calculate family distribution
        family_distribution = {}
        for family in StrategyFamily:
            family_weight = sum(
                alloc.weight for alloc in allocations
                if alloc.family == family
            )
            family_distribution[family] = family_weight

        return OptimizationResult(
            allocations=allocations,
            total_expected_return=expected_return,
            portfolio_risk=portfolio_risk,
            risk_adjusted_score=risk_adjusted_score,
            family_distribution=family_distribution,
            constraints_satisfied=True,  # Will be validated separately
            optimization_warnings=[]
        )

    def _solve_optimization_problem(
        self,
        enhanced_scores: Dict[StrategyType, ConfidenceComponents],
        market_regime: MarketRegime
    ) -> List[Tuple[StrategyType, float]]:
        """
        Solve the mathematical optimization problem for strategy selection

        This is a simplified version - in production, would use more sophisticated
        optimization algorithms like genetic algorithms or simulated annealing.
        """

        # Sort strategies by adjusted score
        sorted_strategies = sorted(
            enhanced_scores.items(),
            key=lambda x: x[1].final_score * STRATEGY_PRIORITY_WEIGHTS[x[0]],
            reverse=True
        )

        selected = []
        family_counts = {family: 0 for family in StrategyFamily}
        total_selected = 0

        # Always include EMA if available and viable
        for strategy, components in sorted_strategies:
            if strategy == StrategyType.EMA and components.final_score >= 0.25:
                selected.append((strategy, components.final_score))
                family_counts[STRATEGY_FAMILIES[strategy]] += 1
                total_selected += 1
                break

        # Select remaining strategies with diversification constraints
        for strategy, components in sorted_strategies:
            if strategy == StrategyType.EMA:
                continue  # Already handled

            if total_selected >= self.constraints.max_total_strategies:
                break

            strategy_family = STRATEGY_FAMILIES[strategy]
            family_ratio = family_counts[strategy_family] / max(total_selected, 1)

            # Check family concentration constraint
            if family_ratio < MAX_FAMILY_ALLOCATION[strategy_family]:
                if components.final_score >= 0.30:  # Minimum viability
                    selected.append((strategy, components.final_score))
                    family_counts[strategy_family] += 1
                    total_selected += 1

        return selected

    def _calculate_portfolio_risk(self, allocations: List[StrategyAllocation]) -> float:
        """
        Calculate portfolio risk using strategy correlations and volatilities

        Mathematical basis: Portfolio variance = Σ Σ w_i * w_j * σ_i * σ_j * ρ_ij
        """

        if len(allocations) <= 1:
            return allocations[0].risk_contribution if allocations else 0.0

        total_risk = 0.0

        for i, alloc_i in enumerate(allocations):
            for j, alloc_j in enumerate(allocations):
                vol_i = STRATEGY_VOLATILITY[alloc_i.strategy]
                vol_j = STRATEGY_VOLATILITY[alloc_j.strategy]

                if i == j:
                    correlation = 1.0
                else:
                    # Estimate correlation based on family membership
                    family_i = STRATEGY_FAMILIES[alloc_i.strategy]
                    family_j = STRATEGY_FAMILIES[alloc_j.strategy]

                    if family_i == family_j:
                        correlation = 0.7  # High correlation within family
                    else:
                        # Use family correlation matrix
                        family_indices = {
                            StrategyFamily.TREND_FOLLOWING: 0,
                            StrategyFamily.MEAN_REVERTING: 1,
                            StrategyFamily.MOMENTUM_BASED: 2,
                            StrategyFamily.SPECIALIZED: 3
                        }
                        idx_i = family_indices[family_i]
                        idx_j = family_indices[family_j]
                        correlation = FAMILY_CORRELATION_MATRIX[idx_i, idx_j]

                total_risk += alloc_i.weight * alloc_j.weight * vol_i * vol_j * correlation

        return np.sqrt(total_risk)

    def _calculate_risk_contribution(self, strategy: StrategyType, weight: float) -> float:
        """Calculate individual strategy risk contribution"""
        return weight * STRATEGY_VOLATILITY[strategy]

    def _validate_constraints(self, result: OptimizationResult) -> OptimizationResult:
        """Validate optimization result against constraints"""

        warnings = []

        # Check family concentration
        for family, allocation in result.family_distribution.items():
            max_allowed = MAX_FAMILY_ALLOCATION[family]
            if allocation > max_allowed:
                warnings.append(f"Family over-concentration: {family.value} = {allocation:.1%} > {max_allowed:.1%}")

        # Check EMA priority
        ema_allocation = sum(
            alloc.weight for alloc in result.allocations
            if alloc.strategy == StrategyType.EMA
        )
        if ema_allocation < 0.10 and len(result.allocations) > 2:
            warnings.append(f"EMA allocation too low: {ema_allocation:.1%}")

        # Check total strategies
        if len(result.allocations) > self.constraints.max_total_strategies:
            warnings.append(f"Too many strategies: {len(result.allocations)} > {self.constraints.max_total_strategies}")

        # Check minimum diversification
        unique_families = len(set(alloc.family for alloc in result.allocations))
        if unique_families < 2 and len(result.allocations) > 3:
            warnings.append("Insufficient family diversification")

        result.optimization_warnings = warnings
        result.constraints_satisfied = len(warnings) == 0

        return result

    def _get_allocation_reason(
        self,
        strategy: StrategyType,
        score: float,
        market_regime: MarketRegime
    ) -> str:
        """Get human-readable reason for strategy allocation"""

        if strategy == StrategyType.EMA:
            return "Priority strategy - mathematical foundation"
        elif score > 0.8:
            return f"High confidence in {market_regime.value} regime"
        elif score > 0.6:
            return "Good regime compatibility"
        elif score > 0.4:
            return "Diversification benefit"
        else:
            return "Minimal allocation for completeness"

    def _calculate_regime_balance_factor(
        self,
        strategy: StrategyType,
        active_strategies: List[StrategyType],
        market_regime: MarketRegime
    ) -> float:
        """Calculate regime-specific portfolio balance factor"""

        # Count strategies suitable for current regime
        regime_suitable_count = 0
        for s in active_strategies:
            from .unified_confidence_framework import REGIME_STRATEGY_CONFIDENCE_MODIFIERS
            regime_suitability = REGIME_STRATEGY_CONFIDENCE_MODIFIERS.get(
                market_regime, {}
            ).get(s, 0.5)

            if regime_suitability > 0.7:
                regime_suitable_count += 1

        # If we have too many highly suitable strategies, reduce bonus
        if regime_suitable_count > 3:
            return 0.9  # Slight penalty for over-concentration
        elif regime_suitable_count < 2:
            return 1.1  # Bonus for adding suitable strategy
        else:
            return 1.0  # Neutral

    def _calculate_historical_performance_factor(
        self,
        strategy: StrategyType,
        market_regime: MarketRegime
    ) -> float:
        """Calculate factor based on historical performance"""

        # Placeholder for historical performance analysis
        # In production, would analyze actual validation rates over time

        # For now, use MACD success as baseline
        if strategy == StrategyType.MACD:
            return 1.05  # 5% bonus for proven performance
        elif strategy == StrategyType.EMA:
            return 1.03  # 3% bonus for foundational role
        else:
            return 1.0   # Neutral for unproven strategies

    def _calculate_risk_adjustment_factor(self, strategy: StrategyType) -> float:
        """Calculate risk adjustment factor based on strategy volatility"""

        volatility = STRATEGY_VOLATILITY[strategy]

        # Penalize high-volatility strategies slightly
        if volatility > 0.18:
            return 0.95  # 5% penalty for high volatility
        elif volatility < 0.10:
            return 1.05  # 5% bonus for low volatility
        else:
            return 1.0   # Neutral

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and metrics"""

        if not self.current_allocations:
            return {'status': 'no_optimization_run'}

        family_distribution = {}
        for family in StrategyFamily:
            family_weight = sum(
                alloc.weight for alloc in self.current_allocations
                if alloc.family == family
            )
            family_distribution[family.value] = family_weight

        return {
            'optimization_time': self.last_optimization_time.isoformat(),
            'total_strategies': len(self.current_allocations),
            'family_distribution': family_distribution,
            'ema_allocation': sum(
                alloc.weight for alloc in self.current_allocations
                if alloc.strategy == StrategyType.EMA
            ),
            'risk_metrics': {
                'total_expected_return': sum(alloc.weight * alloc.confidence for alloc in self.current_allocations),
                'portfolio_risk': self._calculate_portfolio_risk(self.current_allocations),
                'individual_risks': {
                    alloc.strategy.value: alloc.risk_contribution
                    for alloc in self.current_allocations
                }
            },
            'constraints': {
                'max_total_strategies': self.constraints.max_total_strategies,
                'max_family_concentration': self.constraints.max_family_concentration,
                'min_ema_weight': self.constraints.min_ema_weight
            }
        }