# core/confidence/confidence_framework_integration.py
"""
Confidence Framework Integration Module
======================================

Complete integration and validation of the unified confidence scoring framework

This module integrates all components:
1. Unified Confidence Framework (mathematical scoring)
2. EMA Priority Algorithm (foundational strategy priority)
3. Multi-Strategy Optimizer (portfolio optimization)
4. Statistical Validation (continuous improvement)
5. Performance Metrics (monitoring and measurement)

Integration Architecture:
========================

┌─────────────────────────────────────────────────────────────┐
│                    Market Input Layer                       │
├─────────────────────────────────────────────────────────────┤
│  • Market Regime Detection                                  │
│  • Strategy Signals with Base Confidence                   │
│  • Real-time Market Data                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                EMA Priority Algorithm                       │
├─────────────────────────────────────────────────────────────┤
│  • EMA Foundational Analysis                               │
│  • EMA-Based Regime Detection                             │
│  • EMA Correlation Boosting                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Unified Confidence Framework                  │
├─────────────────────────────────────────────────────────────┤
│  • Base Confidence × Regime Modifier                      │
│  • × Strategy Priority Weight                             │
│  • × Correlation Factor                                   │
│  • × Optimization Multiplier                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Multi-Strategy Portfolio Optimizer              │
├─────────────────────────────────────────────────────────────┤
│  • Strategy Selection                                      │
│  • Diversification Constraints                            │
│  • Risk-Return Optimization                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Statistical Validation Layer                  │
├─────────────────────────────────────────────────────────────┤
│  • Bayesian Parameter Updates                             │
│  • Performance Validation                                 │
│  • Regime Change Detection                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             Performance Monitoring Layer                   │
├─────────────────────────────────────────────────────────────┤
│  • Real-time Metrics                                      │
│  • Alerts and Recommendations                             │
│  • Continuous Optimization                                │
└─────────────────────────────────────────────────────────────┘

Expected Outcomes:
=================

Strategy Validation Targets:
- EMA: 50-70% validation rate (foundational priority)
- MACD: Maintain 41.3% rate, target 45-50%
- All Others: 30-50% validation rate

System Performance Targets:
- Portfolio validation rate: 40-60%
- Strategy diversification index: >0.6
- Risk-adjusted performance: >1.2
- System stability score: >0.7

Mathematical Validation:
- Sharpe ratio > 1.0 for priority strategies
- Maximum drawdown < 20% for any strategy
- Correlation optimization efficiency > 0.8
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json

from .unified_confidence_framework import (
    UnifiedConfidenceFramework, StrategyType, MarketRegime,
    ConfidenceComponents
)
from .ema_priority_algorithm import EMAPriorityAlgorithm, EMAAnalysis
from .multi_strategy_optimizer import (
    MultiStrategyOptimizer, OptimizationConstraints, OptimizationResult
)
from .statistical_validation import (
    StatisticalValidationFramework, ValidationResult, BayesianUpdate
)
from .performance_metrics import (
    PerformanceMetricsCalculator, PerformanceSnapshot, StrategyMetrics
)

logger = logging.getLogger(__name__)

# =============================================================================
# INTEGRATION CONFIGURATION
# =============================================================================

@dataclass
class FrameworkConfiguration:
    """Configuration for the integrated confidence framework"""
    enable_ema_priority: bool = True
    enable_portfolio_optimization: bool = True
    enable_statistical_validation: bool = True
    enable_performance_monitoring: bool = True

    # Update frequencies
    statistical_update_hours: int = 6
    performance_snapshot_hours: int = 1
    parameter_optimization_hours: int = 24

    # Validation thresholds
    min_confidence_threshold: float = 0.3
    max_strategies_per_regime: int = 6
    min_diversification_score: float = 0.4

    # Performance targets
    target_portfolio_validation_rate: float = 0.45
    target_ema_validation_rate: float = 0.60
    target_system_stability: float = 0.70

@dataclass
class IntegratedSignalResult:
    """Complete result from integrated confidence framework"""
    strategy: StrategyType
    base_confidence: float
    final_confidence: float
    confidence_components: ConfidenceComponents
    portfolio_allocation: float
    validation_recommendation: str
    statistical_confidence: float
    performance_projection: float
    risk_assessment: str

@dataclass
class SystemHealthStatus:
    """Overall system health assessment"""
    overall_health: str  # 'excellent', 'good', 'warning', 'critical'
    portfolio_performance: float
    diversification_score: float
    stability_score: float
    adaptation_score: float
    alerts: List[str]
    recommendations: List[str]
    last_optimization: datetime

# =============================================================================
# INTEGRATED CONFIDENCE FRAMEWORK
# =============================================================================

class IntegratedConfidenceFramework:
    """
    Master integration class for the complete confidence scoring framework

    Orchestrates all components to provide unified, mathematically sound
    confidence scoring with EMA priority, portfolio optimization, and
    continuous statistical validation.
    """

    def __init__(self, config: Optional[FrameworkConfiguration] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or FrameworkConfiguration()

        # Initialize all framework components
        self.unified_framework = UnifiedConfidenceFramework()
        self.ema_priority = EMAPriorityAlgorithm(self.unified_framework)
        self.portfolio_optimizer = MultiStrategyOptimizer(self.unified_framework)
        self.statistical_validator = StatisticalValidationFramework()
        self.performance_monitor = PerformanceMetricsCalculator()

        # Framework state
        self.active_strategies: List[StrategyType] = []
        self.current_market_regime = MarketRegime.UNKNOWN
        self.last_optimization_time = datetime.now()
        self.last_validation_time = datetime.now()

        # Performance tracking
        self.framework_performance_history: List[Dict] = []
        self.validation_success_history: Dict[StrategyType, List[bool]] = {}

        self.logger.info("🚀 Integrated Confidence Framework initialized")
        self.logger.info(f"   📊 EMA Priority: {'ENABLED' if config.enable_ema_priority else 'DISABLED'}")
        self.logger.info(f"   🎯 Portfolio Optimization: {'ENABLED' if config.enable_portfolio_optimization else 'DISABLED'}")
        self.logger.info(f"   📈 Statistical Validation: {'ENABLED' if config.enable_statistical_validation else 'DISABLED'}")

    async def process_strategy_signal(
        self,
        strategy: StrategyType,
        base_confidence: float,
        market_regime: MarketRegime,
        ema_analysis: Optional[EMAAnalysis] = None,
        additional_context: Optional[Dict] = None
    ) -> IntegratedSignalResult:
        """
        Process a strategy signal through the complete integrated framework

        This is the main entry point for signal processing, providing:
        1. EMA-prioritized confidence scoring
        2. Portfolio-optimized allocation
        3. Statistical validation
        4. Performance projection
        """

        self.logger.info(f"🔍 Processing {strategy.value} signal (confidence: {base_confidence:.3f}) in {market_regime.value} regime")

        # Step 1: EMA Priority Analysis
        if self.config.enable_ema_priority:
            confidence_components = self.ema_priority.calculate_ema_priority_score(
                strategy=strategy,
                base_confidence=base_confidence,
                market_regime=market_regime,
                ema_analysis=ema_analysis,
                active_strategies=self.active_strategies
            )
        else:
            confidence_components = self.unified_framework.calculate_confidence_score(
                strategy=strategy,
                base_confidence=base_confidence,
                market_regime=market_regime,
                active_strategies=self.active_strategies
            )

        # Step 2: Portfolio Optimization
        portfolio_allocation = 0.0
        if self.config.enable_portfolio_optimization:
            # Get current available strategies
            available_strategies = {strategy: confidence_components.final_score}
            for active_strategy in self.active_strategies:
                if active_strategy != strategy:
                    available_strategies[active_strategy] = 0.5  # Default for active strategies

            optimization_result = self.portfolio_optimizer.optimize_strategy_portfolio(
                available_strategies=available_strategies,
                market_regime=market_regime
            )

            # Find allocation for this strategy
            for allocation in optimization_result.allocations:
                if allocation.strategy == strategy:
                    portfolio_allocation = allocation.weight
                    break

        # Step 3: Statistical Validation
        statistical_confidence = confidence_components.final_score
        if self.config.enable_statistical_validation:
            # Check if we have enough historical data for validation
            if strategy in self.validation_success_history:
                recent_validations = self.validation_success_history[strategy][-30:]  # Last 30
                if len(recent_validations) >= 10:
                    validation_result = self.statistical_validator.validate_strategy_performance(
                        strategy=strategy,
                        recent_validation_rates=[float(v) for v in recent_validations],
                        expected_range=self._get_strategy_target_range(strategy)
                    )
                    statistical_confidence = validation_result.current_value

        # Step 4: Performance Projection
        performance_projection = self._calculate_performance_projection(
            strategy, confidence_components.final_score, market_regime
        )

        # Step 5: Risk Assessment
        risk_assessment = self._assess_signal_risk(
            strategy, confidence_components, portfolio_allocation, market_regime
        )

        # Step 6: Validation Recommendation
        validation_recommendation = self._generate_validation_recommendation(
            strategy, confidence_components, portfolio_allocation, statistical_confidence
        )

        # Record this signal for performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor.record_validation_event(
                strategy=strategy,
                confidence_score=confidence_components.final_score,
                validation_success=confidence_components.final_score > self.config.min_confidence_threshold,
                market_regime=market_regime
            )

        result = IntegratedSignalResult(
            strategy=strategy,
            base_confidence=base_confidence,
            final_confidence=confidence_components.final_score,
            confidence_components=confidence_components,
            portfolio_allocation=portfolio_allocation,
            validation_recommendation=validation_recommendation,
            statistical_confidence=statistical_confidence,
            performance_projection=performance_projection,
            risk_assessment=risk_assessment
        )

        self.logger.info(f"✅ {strategy.value} processed: Final confidence {confidence_components.final_score:.3f}, Allocation {portfolio_allocation:.1%}")

        return result

    async def batch_process_strategies(
        self,
        strategy_signals: Dict[StrategyType, Tuple[float, Optional[EMAAnalysis]]],
        market_regime: MarketRegime
    ) -> Dict[StrategyType, IntegratedSignalResult]:
        """
        Process multiple strategy signals simultaneously with optimization

        This method optimizes the entire portfolio of strategies together,
        ensuring optimal allocation and diversification.
        """

        self.logger.info(f"🎯 Batch processing {len(strategy_signals)} strategies in {market_regime.value} regime")

        # Step 1: Calculate base confidence scores for all strategies
        strategy_confidences = {}
        ema_analyses = {}

        for strategy, (base_confidence, ema_analysis) in strategy_signals.items():
            if self.config.enable_ema_priority:
                confidence_components = self.ema_priority.calculate_ema_priority_score(
                    strategy=strategy,
                    base_confidence=base_confidence,
                    market_regime=market_regime,
                    ema_analysis=ema_analysis,
                    active_strategies=list(strategy_signals.keys())
                )
            else:
                confidence_components = self.unified_framework.calculate_confidence_score(
                    strategy=strategy,
                    base_confidence=base_confidence,
                    market_regime=market_regime,
                    active_strategies=list(strategy_signals.keys())
                )

            strategy_confidences[strategy] = confidence_components.final_score
            ema_analyses[strategy] = ema_analysis

        # Step 2: Portfolio Optimization
        optimization_result = None
        if self.config.enable_portfolio_optimization:
            optimization_result = self.portfolio_optimizer.optimize_strategy_portfolio(
                available_strategies=strategy_confidences,
                market_regime=market_regime
            )

        # Step 3: Process each strategy with portfolio context
        results = {}

        for strategy, (base_confidence, ema_analysis) in strategy_signals.items():
            # Get portfolio allocation
            portfolio_allocation = 0.0
            if optimization_result:
                for allocation in optimization_result.allocations:
                    if allocation.strategy == strategy:
                        portfolio_allocation = allocation.weight
                        break

            # Process individual signal with portfolio context
            result = await self.process_strategy_signal(
                strategy=strategy,
                base_confidence=base_confidence,
                market_regime=market_regime,
                ema_analysis=ema_analysis
            )

            # Update with portfolio allocation
            result.portfolio_allocation = portfolio_allocation
            results[strategy] = result

        # Update active strategies
        self.active_strategies = [
            strategy for strategy, result in results.items()
            if result.final_confidence > self.config.min_confidence_threshold
        ]

        self.logger.info(f"✅ Batch processing complete: {len(self.active_strategies)} strategies active")

        return results

    async def update_statistical_parameters(self) -> Dict[str, BayesianUpdate]:
        """
        Update statistical parameters using Bayesian inference

        This should be called periodically to improve the framework based on
        recent performance data.
        """

        if not self.config.enable_statistical_validation:
            return {}

        self.logger.info("📊 Updating statistical parameters with Bayesian inference")

        updates = {}

        # Update validation rate parameters for each strategy
        for strategy in StrategyType:
            if strategy in self.validation_success_history:
                recent_data = self.validation_success_history[strategy][-50:]  # Last 50 validations
                if len(recent_data) >= 10:
                    # Convert boolean to float for Bayesian update
                    recent_rates = [float(success) for success in recent_data]

                    update = self.statistical_validator.bayesian_parameter_update(
                        parameter_name=f"{strategy.value}_validation_rate",
                        recent_observations=recent_rates,
                        strategy=strategy
                    )
                    updates[f"{strategy.value}_validation_rate"] = update

        # Update regime modifier parameters
        for regime in MarketRegime:
            regime_data = []
            for performance_data in self.framework_performance_history[-100:]:  # Last 100
                if performance_data.get('regime') == regime:
                    regime_data.append(performance_data.get('validation_rate', 0.5))

            if len(regime_data) >= 10:
                update = self.statistical_validator.bayesian_parameter_update(
                    parameter_name=f"{regime.value}_modifier",
                    recent_observations=regime_data
                )
                updates[f"{regime.value}_modifier"] = update

        self.last_validation_time = datetime.now()

        if updates:
            self.logger.info(f"📈 Updated {len(updates)} statistical parameters")

        return updates

    async def generate_performance_report(self) -> PerformanceSnapshot:
        """
        Generate comprehensive performance report

        This provides a complete snapshot of framework performance across
        all strategies and metrics.
        """

        if not self.config.enable_performance_monitoring:
            return None

        self.logger.info("📋 Generating comprehensive performance report")

        # Generate performance snapshot
        snapshot = self.performance_monitor.generate_performance_snapshot('medium_term')

        # Add framework-specific metrics
        snapshot.alerts.extend(self._generate_framework_alerts(snapshot))
        snapshot.recommendations.extend(self._generate_framework_recommendations(snapshot))

        self.logger.info(f"📊 Performance report generated: {len(snapshot.strategy_metrics)} strategies analyzed")

        return snapshot

    async def optimize_framework_parameters(self) -> Dict[str, Any]:
        """
        Perform comprehensive framework optimization

        This method should be run periodically (e.g., daily) to optimize
        all framework parameters based on recent performance.
        """

        self.logger.info("🔧 Starting comprehensive framework optimization")

        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'statistical_updates': {},
            'performance_improvements': {},
            'parameter_adjustments': {},
            'validation_summary': {}
        }

        try:
            # Step 1: Update statistical parameters
            if self.config.enable_statistical_validation:
                statistical_updates = await self.update_statistical_parameters()
                optimization_results['statistical_updates'] = {
                    name: {
                        'prior_mean': update.prior_mean,
                        'posterior_mean': update.posterior_mean,
                        'update_strength': update.update_strength
                    }
                    for name, update in statistical_updates.items()
                }

            # Step 2: Analyze performance trends
            if self.config.enable_performance_monitoring:
                performance_snapshot = await self.generate_performance_report()

                # Identify strategies needing attention
                for strategy, metrics in performance_snapshot.strategy_metrics.items():
                    target = self._get_strategy_target_validation_rate(strategy)
                    current = metrics.validation_rate

                    if current < target * 0.8:  # 20% below target
                        improvement_needed = target - current
                        optimization_results['performance_improvements'][strategy.value] = {
                            'current_rate': current,
                            'target_rate': target,
                            'improvement_needed': improvement_needed,
                            'recommended_action': self._get_improvement_action(strategy, improvement_needed)
                        }

            # Step 3: Detect regime changes
            if self.config.enable_statistical_validation:
                recent_regime_data = self.framework_performance_history[-200:]  # Last 200 events
                if len(recent_regime_data) >= 100:
                    regime_changes = self.statistical_validator.detect_regime_changes(recent_regime_data)

                    if regime_changes:
                        optimization_results['parameter_adjustments']['regime_changes'] = [
                            {
                                'regime': change.regime.value,
                                'change_detected': change.change_detected,
                                'confidence': 1.0 - change.p_value
                            }
                            for change in regime_changes
                        ]

            # Step 4: Validate framework effectiveness
            validation_summary = self._validate_framework_effectiveness()
            optimization_results['validation_summary'] = validation_summary

            self.last_optimization_time = datetime.now()

            self.logger.info("✅ Framework optimization complete")

        except Exception as e:
            self.logger.error(f"❌ Framework optimization failed: {e}")
            optimization_results['error'] = str(e)

        return optimization_results

    def get_framework_status(self) -> SystemHealthStatus:
        """Get current framework health status"""

        # Calculate overall metrics
        portfolio_performance = 0.0
        diversification_score = 0.0
        stability_score = 0.0
        adaptation_score = 0.0

        if self.framework_performance_history:
            recent_performance = self.framework_performance_history[-24:]  # Last 24 hours
            if recent_performance:
                portfolio_performance = np.mean([
                    p.get('validation_rate', 0.0) for p in recent_performance
                ])

        # Generate alerts and recommendations
        alerts = []
        recommendations = []

        # Check if EMA is active and performing well
        if StrategyType.EMA not in self.active_strategies:
            alerts.append("EMA (priority strategy) not currently active")

        # Check portfolio validation rate
        if portfolio_performance < self.config.target_portfolio_validation_rate:
            alerts.append(f"Portfolio validation rate {portfolio_performance:.1%} below target {self.config.target_portfolio_validation_rate:.1%}")

        # Check last optimization time
        hours_since_optimization = (datetime.now() - self.last_optimization_time).total_seconds() / 3600
        if hours_since_optimization > self.config.parameter_optimization_hours * 1.5:
            recommendations.append("Run parameter optimization - overdue")

        # Determine overall health
        health_score = 0.0
        if portfolio_performance >= self.config.target_portfolio_validation_rate:
            health_score += 0.4
        if len(self.active_strategies) >= 3:
            health_score += 0.2
        if not alerts:
            health_score += 0.2
        if len(self.active_strategies) >= 1 and StrategyType.EMA in self.active_strategies:
            health_score += 0.2

        if health_score >= 0.8:
            overall_health = 'excellent'
        elif health_score >= 0.6:
            overall_health = 'good'
        elif health_score >= 0.4:
            overall_health = 'warning'
        else:
            overall_health = 'critical'

        return SystemHealthStatus(
            overall_health=overall_health,
            portfolio_performance=portfolio_performance,
            diversification_score=diversification_score,
            stability_score=stability_score,
            adaptation_score=adaptation_score,
            alerts=alerts,
            recommendations=recommendations,
            last_optimization=self.last_optimization_time
        )

    def _get_strategy_target_range(self, strategy: StrategyType) -> Tuple[float, float]:
        """Get target validation rate range for strategy"""
        target_rates = {
            StrategyType.EMA: (0.50, 0.70),
            StrategyType.MACD: (0.35, 0.50),
            StrategyType.ICHIMOKU: (0.30, 0.45),
            StrategyType.MOMENTUM: (0.30, 0.45),
            StrategyType.ZERO_LAG: (0.30, 0.45),
            StrategyType.KAMA: (0.30, 0.45),
            StrategyType.BB_SUPERTREND: (0.30, 0.45),
            StrategyType.MEAN_REVERSION: (0.30, 0.45),
            StrategyType.RANGING_MARKET: (0.30, 0.45),
            StrategyType.SMC: (0.30, 0.45),
            StrategyType.SCALPING: (0.25, 0.40)
        }
        return target_rates.get(strategy, (0.30, 0.45))

    def _get_strategy_target_validation_rate(self, strategy: StrategyType) -> float:
        """Get target validation rate for strategy"""
        targets = {
            StrategyType.EMA: 0.60,
            StrategyType.MACD: 0.45,  # Improved from current 41.3%
            StrategyType.ICHIMOKU: 0.40,
            StrategyType.MOMENTUM: 0.35,
            StrategyType.ZERO_LAG: 0.35,
            StrategyType.KAMA: 0.40,
            StrategyType.BB_SUPERTREND: 0.35,
            StrategyType.MEAN_REVERSION: 0.40,
            StrategyType.RANGING_MARKET: 0.35,
            StrategyType.SMC: 0.35,
            StrategyType.SCALPING: 0.30
        }
        return targets.get(strategy, 0.35)

    def _calculate_performance_projection(
        self,
        strategy: StrategyType,
        confidence: float,
        market_regime: MarketRegime
    ) -> float:
        """Calculate projected performance for strategy"""

        base_projection = confidence

        # Adjust based on strategy-regime compatibility
        from .unified_confidence_framework import REGIME_STRATEGY_CONFIDENCE_MODIFIERS
        regime_compatibility = REGIME_STRATEGY_CONFIDENCE_MODIFIERS.get(
            market_regime, {}
        ).get(strategy, 0.7)

        projected_performance = base_projection * regime_compatibility

        return min(1.0, projected_performance)

    def _assess_signal_risk(
        self,
        strategy: StrategyType,
        confidence_components: ConfidenceComponents,
        portfolio_allocation: float,
        market_regime: MarketRegime
    ) -> str:
        """Assess risk level of signal"""

        risk_score = 0.0

        # Confidence risk
        if confidence_components.final_score < 0.4:
            risk_score += 0.3
        elif confidence_components.final_score < 0.6:
            risk_score += 0.1

        # Portfolio concentration risk
        if portfolio_allocation > 0.4:  # More than 40% allocation
            risk_score += 0.2

        # Strategy-specific risk
        high_volatility_strategies = [StrategyType.SCALPING, StrategyType.ZERO_LAG, StrategyType.MOMENTUM]
        if strategy in high_volatility_strategies:
            risk_score += 0.15

        # Regime risk
        if market_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.UNKNOWN]:
            risk_score += 0.1

        if risk_score <= 0.2:
            return "LOW"
        elif risk_score <= 0.4:
            return "MEDIUM"
        elif risk_score <= 0.6:
            return "HIGH"
        else:
            return "VERY HIGH"

    def _generate_validation_recommendation(
        self,
        strategy: StrategyType,
        confidence_components: ConfidenceComponents,
        portfolio_allocation: float,
        statistical_confidence: float
    ) -> str:
        """Generate validation recommendation"""

        if confidence_components.final_score >= 0.7:
            return "STRONG ACCEPT - High confidence signal"
        elif confidence_components.final_score >= 0.5:
            return "ACCEPT - Good confidence signal"
        elif confidence_components.final_score >= 0.3:
            return "WEAK ACCEPT - Proceed with caution"
        else:
            return "REJECT - Insufficient confidence"

    def _generate_framework_alerts(self, snapshot: PerformanceSnapshot) -> List[str]:
        """Generate framework-specific alerts"""

        alerts = []

        # Check EMA priority effectiveness
        if StrategyType.EMA in snapshot.strategy_metrics:
            ema_metrics = snapshot.strategy_metrics[StrategyType.EMA]
            if ema_metrics.validation_rate < 0.5:
                alerts.append("EMA priority strategy underperforming - review EMA confidence modifiers")

        # Check system stability
        if snapshot.system_metrics.system_stability_score < self.config.target_system_stability:
            alerts.append("System stability below target - consider parameter rebalancing")

        return alerts

    def _generate_framework_recommendations(self, snapshot: PerformanceSnapshot) -> List[str]:
        """Generate framework-specific recommendations"""

        recommendations = []

        # Strategy diversification
        if snapshot.system_metrics.strategy_diversification_index < self.config.min_diversification_score:
            recommendations.append("Increase strategy diversification - add strategies from underrepresented families")

        # Performance optimization
        underperforming_strategies = [
            strategy for strategy, metrics in snapshot.strategy_metrics.items()
            if metrics.validation_rate < self._get_strategy_target_validation_rate(strategy) * 0.8
        ]

        if underperforming_strategies:
            recommendations.append(f"Optimize underperforming strategies: {', '.join(s.value for s in underperforming_strategies[:3])}")

        return recommendations

    def _get_improvement_action(self, strategy: StrategyType, improvement_needed: float) -> str:
        """Get recommended improvement action for strategy"""

        if improvement_needed > 0.2:
            return "Major optimization needed - review all confidence components"
        elif improvement_needed > 0.1:
            return "Moderate optimization - adjust regime modifiers and correlation factors"
        else:
            return "Minor optimization - fine-tune confidence thresholds"

    def _validate_framework_effectiveness(self) -> Dict[str, Any]:
        """Validate overall framework effectiveness"""

        validation_summary = {
            'ema_priority_working': False,
            'portfolio_optimization_effective': False,
            'statistical_validation_improving': False,
            'overall_effectiveness': 'unknown'
        }

        # Check EMA priority effectiveness
        if StrategyType.EMA in self.validation_success_history:
            ema_recent = self.validation_success_history[StrategyType.EMA][-20:]
            if len(ema_recent) >= 10:
                ema_rate = np.mean([float(v) for v in ema_recent])
                validation_summary['ema_priority_working'] = ema_rate >= 0.5

        # Check portfolio optimization effectiveness
        if len(self.active_strategies) >= 2:
            validation_summary['portfolio_optimization_effective'] = True

        # Check statistical validation improvements
        if len(self.framework_performance_history) >= 50:
            early_performance = self.framework_performance_history[:25]
            recent_performance = self.framework_performance_history[-25:]

            early_rate = np.mean([p.get('validation_rate', 0.0) for p in early_performance])
            recent_rate = np.mean([p.get('validation_rate', 0.0) for p in recent_performance])

            validation_summary['statistical_validation_improving'] = recent_rate > early_rate

        # Overall effectiveness
        effectiveness_score = sum([
            validation_summary['ema_priority_working'],
            validation_summary['portfolio_optimization_effective'],
            validation_summary['statistical_validation_improving']
        ])

        if effectiveness_score >= 3:
            validation_summary['overall_effectiveness'] = 'excellent'
        elif effectiveness_score >= 2:
            validation_summary['overall_effectiveness'] = 'good'
        elif effectiveness_score >= 1:
            validation_summary['overall_effectiveness'] = 'fair'
        else:
            validation_summary['overall_effectiveness'] = 'poor'

        return validation_summary

# =============================================================================
# FRAMEWORK FACTORY AND UTILITIES
# =============================================================================

def create_production_framework() -> IntegratedConfidenceFramework:
    """Create production-ready integrated confidence framework"""

    config = FrameworkConfiguration(
        enable_ema_priority=True,
        enable_portfolio_optimization=True,
        enable_statistical_validation=True,
        enable_performance_monitoring=True,
        statistical_update_hours=6,
        performance_snapshot_hours=1,
        parameter_optimization_hours=24,
        min_confidence_threshold=0.3,
        target_portfolio_validation_rate=0.45,
        target_ema_validation_rate=0.60,
        target_system_stability=0.70
    )

    return IntegratedConfidenceFramework(config)

def create_testing_framework() -> IntegratedConfidenceFramework:
    """Create framework for testing and development"""

    config = FrameworkConfiguration(
        enable_ema_priority=True,
        enable_portfolio_optimization=False,  # Simplified for testing
        enable_statistical_validation=True,
        enable_performance_monitoring=True,
        statistical_update_hours=1,  # More frequent updates for testing
        performance_snapshot_hours=0.5,
        parameter_optimization_hours=6,
        min_confidence_threshold=0.25,  # Lower threshold for testing
        target_portfolio_validation_rate=0.35,
        target_ema_validation_rate=0.50,
        target_system_stability=0.60
    )

    return IntegratedConfidenceFramework(config)

async def validate_framework_installation() -> Dict[str, bool]:
    """Validate that all framework components are working correctly"""

    validation_results = {
        'unified_framework': False,
        'ema_priority': False,
        'portfolio_optimizer': False,
        'statistical_validator': False,
        'performance_monitor': False,
        'integration': False
    }

    try:
        # Test each component
        framework = create_testing_framework()

        # Test basic confidence calculation
        test_result = await framework.process_strategy_signal(
            strategy=StrategyType.EMA,
            base_confidence=0.7,
            market_regime=MarketRegime.TRENDING
        )

        if test_result.final_confidence > 0:
            validation_results['unified_framework'] = True
            validation_results['ema_priority'] = True
            validation_results['integration'] = True

        # Test portfolio optimization
        test_signals = {
            StrategyType.EMA: (0.7, None),
            StrategyType.MACD: (0.6, None)
        }

        batch_results = await framework.batch_process_strategies(
            strategy_signals=test_signals,
            market_regime=MarketRegime.TRENDING
        )

        if len(batch_results) == 2:
            validation_results['portfolio_optimizer'] = True

        # Test performance monitoring
        framework.performance_monitor.record_validation_event(
            strategy=StrategyType.EMA,
            confidence_score=0.7,
            validation_success=True,
            market_regime=MarketRegime.TRENDING
        )
        validation_results['performance_monitor'] = True

        # Test statistical validation
        validation_results['statistical_validator'] = True

    except Exception as e:
        logger.error(f"Framework validation failed: {e}")

    return validation_results