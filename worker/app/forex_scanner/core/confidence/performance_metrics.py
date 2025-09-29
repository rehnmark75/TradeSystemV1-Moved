# core/confidence/performance_metrics.py
"""
Performance Measurement Metrics for System-Wide Monitoring
=========================================================

Comprehensive performance measurement system for confidence scoring framework

Key Performance Indicators (KPIs):
=================================

1. STRATEGY-LEVEL METRICS
   - Validation Rate: % of signals that pass validation
   - Signal Quality Score: True positives / (True positives + False positives)
   - Confidence Accuracy: How well confidence predicts success
   - Regime Adaptation Score: Performance across different market regimes

2. SYSTEM-LEVEL METRICS
   - Portfolio Validation Rate: Overall system validation success
   - Strategy Diversification Index: Measure of strategy variety
   - Risk-Adjusted Performance: Return/Risk ratio for the strategy portfolio
   - Efficiency Ratio: Output quality per computational resource

3. COMPARATIVE METRICS
   - EMA Priority Effectiveness: How EMA priority improves overall performance
   - Regime Modifier Accuracy: How well regime modifiers predict performance
   - Correlation Factor Optimization: Effectiveness of correlation adjustments
   - Optimization Multiplier ROI: Return on optimization investments

4. TEMPORAL METRICS
   - Performance Stability: Consistency of validation rates over time
   - Adaptation Speed: How quickly system adapts to regime changes
   - Learning Curve: Improvement in performance over time
   - Seasonality Effects: Performance variation by time patterns

Mathematical Foundations:
========================

Sharpe Ratio Adaptation for Trading Strategies:
Sharpe_Strategy = (E[Validation_Rate] - Risk_Free_Rate) / σ(Validation_Rate)

Information Ratio for Strategy Selection:
IR = (Strategy_Return - Benchmark_Return) / Tracking_Error

Maximum Drawdown for Validation Performance:
MDD = max(Peak_Validation_Rate - Current_Validation_Rate) / Peak_Validation_Rate

Calmar Ratio for Risk-Adjusted Performance:
Calmar = Annual_Return / Maximum_Drawdown

Sortino Ratio (Downside Risk):
Sortino = (E[Return] - Target) / σ(Downside_Returns)

Value at Risk (VaR) for Validation Rates:
VaR_95 = μ - 1.645 * σ (95% confidence)

Expected Shortfall (Conditional VaR):
ES = E[Validation_Rate | Validation_Rate < VaR]
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
import statistics

from .unified_confidence_framework import StrategyType, MarketRegime, StrategyFamily

logger = logging.getLogger(__name__)

# =============================================================================
# PERFORMANCE METRICS CONSTANTS
# =============================================================================

# Risk-free rate for Sharpe ratio calculations (annualized)
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

# Target validation rates for different strategies
TARGET_VALIDATION_RATES = {
    StrategyType.EMA: 0.60,          # 60% target for EMA
    StrategyType.MACD: 0.413,        # Current achieved rate
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

# Performance measurement windows
PERFORMANCE_WINDOWS = {
    'short_term': 24,    # 24 hours
    'medium_term': 168,  # 1 week
    'long_term': 720     # 30 days
}

# Statistical significance thresholds
SIGNIFICANCE_THRESHOLD = 0.05
MIN_SAMPLE_SIZE_FOR_STATS = 30

@dataclass
class StrategyMetrics:
    """Performance metrics for individual strategy"""
    strategy: StrategyType
    validation_rate: float
    signal_quality_score: float
    confidence_accuracy: float
    regime_adaptation_score: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    value_at_risk_95: float
    expected_shortfall: float
    sample_size: int
    measurement_period: str
    last_updated: datetime

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    portfolio_validation_rate: float
    strategy_diversification_index: float
    risk_adjusted_performance: float
    efficiency_ratio: float
    ema_priority_effectiveness: float
    regime_modifier_accuracy: float
    correlation_factor_optimization: float
    optimization_multiplier_roi: float
    overall_sharpe_ratio: float
    system_stability_score: float
    adaptation_speed_score: float
    sample_size: int
    measurement_period: str
    last_updated: datetime

@dataclass
class ComparativeMetrics:
    """Comparative analysis between strategies and benchmarks"""
    strategy_rankings: Dict[StrategyType, int]
    performance_attribution: Dict[str, float]
    regime_performance_matrix: Dict[MarketRegime, Dict[StrategyType, float]]
    correlation_efficiency: Dict[Tuple[StrategyType, StrategyType], float]
    benchmark_comparison: Dict[str, float]
    improvement_opportunities: List[str]

@dataclass
class TemporalMetrics:
    """Time-based performance analysis"""
    performance_stability: float
    trend_analysis: Dict[str, float]
    seasonality_effects: Dict[str, float]
    learning_curve_slope: float
    regime_transition_performance: Dict[str, float]
    volatility_clustering: float

@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot"""
    timestamp: datetime
    strategy_metrics: Dict[StrategyType, StrategyMetrics]
    system_metrics: SystemMetrics
    comparative_metrics: ComparativeMetrics
    temporal_metrics: TemporalMetrics
    alerts: List[str]
    recommendations: List[str]

# =============================================================================
# PERFORMANCE METRICS CALCULATOR
# =============================================================================

class PerformanceMetricsCalculator:
    """
    Comprehensive performance measurement system for confidence scoring framework

    Tracks and analyzes performance across multiple dimensions:
    - Individual strategy performance
    - System-wide performance
    - Comparative analysis
    - Temporal patterns
    """

    def __init__(self, max_history_size: int = 10000):
        self.logger = logging.getLogger(__name__)
        self.max_history_size = max_history_size

        # Historical data storage
        self.validation_history: Dict[StrategyType, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self.confidence_history: Dict[StrategyType, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self.regime_history: deque = deque(maxlen=max_history_size)
        self.system_events: deque = deque(maxlen=max_history_size)

        # Performance snapshots
        self.performance_snapshots: deque = deque(maxlen=100)

        # Benchmarks
        self.benchmarks: Dict[str, float] = {
            'random_baseline': 0.5,
            'simple_ema_baseline': 0.45,
            'macd_baseline': 0.413
        }

    def record_validation_event(
        self,
        strategy: StrategyType,
        confidence_score: float,
        validation_success: bool,
        market_regime: MarketRegime,
        timestamp: Optional[datetime] = None
    ):
        """Record a validation event for performance tracking"""

        if timestamp is None:
            timestamp = datetime.now()

        # Record validation result
        self.validation_history[strategy].append({
            'timestamp': timestamp,
            'confidence_score': confidence_score,
            'validation_success': validation_success,
            'market_regime': market_regime
        })

        # Record confidence accuracy
        self.confidence_history[strategy].append({
            'timestamp': timestamp,
            'predicted_confidence': confidence_score,
            'actual_success': validation_success
        })

        # Record regime context
        self.regime_history.append({
            'timestamp': timestamp,
            'regime': market_regime,
            'strategy': strategy,
            'performance': validation_success
        })

    def calculate_strategy_metrics(
        self,
        strategy: StrategyType,
        period: str = 'medium_term'
    ) -> Optional[StrategyMetrics]:
        """Calculate comprehensive metrics for individual strategy"""

        window_hours = PERFORMANCE_WINDOWS.get(period, 168)
        cutoff_time = datetime.now() - timedelta(hours=window_hours)

        # Filter recent data
        recent_data = [
            event for event in self.validation_history[strategy]
            if event['timestamp'] >= cutoff_time
        ]

        if len(recent_data) < MIN_SAMPLE_SIZE_FOR_STATS:
            return None

        # Calculate basic metrics
        validation_successes = [event['validation_success'] for event in recent_data]
        confidence_scores = [event['confidence_score'] for event in recent_data]

        validation_rate = np.mean(validation_successes)
        signal_quality_score = self._calculate_signal_quality_score(recent_data)
        confidence_accuracy = self._calculate_confidence_accuracy(strategy, period)
        regime_adaptation_score = self._calculate_regime_adaptation_score(strategy, recent_data)

        # Calculate risk metrics
        validation_rates_series = self._get_rolling_validation_rates(
            recent_data, window_size=24
        )

        sharpe_ratio = self._calculate_sharpe_ratio(validation_rates_series, TARGET_VALIDATION_RATES[strategy])
        sortino_ratio = self._calculate_sortino_ratio(validation_rates_series, TARGET_VALIDATION_RATES[strategy])
        maximum_drawdown = self._calculate_maximum_drawdown(validation_rates_series)
        value_at_risk_95 = self._calculate_value_at_risk(validation_rates_series, 0.95)
        expected_shortfall = self._calculate_expected_shortfall(validation_rates_series, 0.95)

        return StrategyMetrics(
            strategy=strategy,
            validation_rate=validation_rate,
            signal_quality_score=signal_quality_score,
            confidence_accuracy=confidence_accuracy,
            regime_adaptation_score=regime_adaptation_score,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            maximum_drawdown=maximum_drawdown,
            value_at_risk_95=value_at_risk_95,
            expected_shortfall=expected_shortfall,
            sample_size=len(recent_data),
            measurement_period=period,
            last_updated=datetime.now()
        )

    def calculate_system_metrics(self, period: str = 'medium_term') -> SystemMetrics:
        """Calculate system-wide performance metrics"""

        window_hours = PERFORMANCE_WINDOWS.get(period, 168)
        cutoff_time = datetime.now() - timedelta(hours=window_hours)

        # Aggregate data across all strategies
        all_recent_data = []
        strategy_data = {}

        for strategy in StrategyType:
            recent_data = [
                event for event in self.validation_history[strategy]
                if event['timestamp'] >= cutoff_time
            ]
            strategy_data[strategy] = recent_data
            all_recent_data.extend(recent_data)

        if not all_recent_data:
            return SystemMetrics(
                portfolio_validation_rate=0.0,
                strategy_diversification_index=0.0,
                risk_adjusted_performance=0.0,
                efficiency_ratio=0.0,
                ema_priority_effectiveness=0.0,
                regime_modifier_accuracy=0.0,
                correlation_factor_optimization=0.0,
                optimization_multiplier_roi=0.0,
                overall_sharpe_ratio=0.0,
                system_stability_score=0.0,
                adaptation_speed_score=0.0,
                sample_size=0,
                measurement_period=period,
                last_updated=datetime.now()
            )

        # Calculate system metrics
        portfolio_validation_rate = np.mean([event['validation_success'] for event in all_recent_data])
        strategy_diversification_index = self._calculate_diversification_index(strategy_data)
        risk_adjusted_performance = self._calculate_system_sharpe_ratio(strategy_data)
        efficiency_ratio = self._calculate_efficiency_ratio(strategy_data)

        # EMA priority effectiveness
        ema_priority_effectiveness = self._calculate_ema_priority_effectiveness(strategy_data)

        # Regime modifier accuracy
        regime_modifier_accuracy = self._calculate_regime_modifier_accuracy(all_recent_data)

        # Correlation factor optimization
        correlation_factor_optimization = self._calculate_correlation_optimization(strategy_data)

        # Optimization multiplier ROI
        optimization_multiplier_roi = self._calculate_optimization_roi(strategy_data)

        # System stability and adaptation
        system_stability_score = self._calculate_system_stability(all_recent_data)
        adaptation_speed_score = self._calculate_adaptation_speed(all_recent_data)

        return SystemMetrics(
            portfolio_validation_rate=portfolio_validation_rate,
            strategy_diversification_index=strategy_diversification_index,
            risk_adjusted_performance=risk_adjusted_performance,
            efficiency_ratio=efficiency_ratio,
            ema_priority_effectiveness=ema_priority_effectiveness,
            regime_modifier_accuracy=regime_modifier_accuracy,
            correlation_factor_optimization=correlation_factor_optimization,
            optimization_multiplier_roi=optimization_multiplier_roi,
            overall_sharpe_ratio=risk_adjusted_performance,
            system_stability_score=system_stability_score,
            adaptation_speed_score=adaptation_speed_score,
            sample_size=len(all_recent_data),
            measurement_period=period,
            last_updated=datetime.now()
        )

    def generate_performance_snapshot(self, period: str = 'medium_term') -> PerformanceSnapshot:
        """Generate complete performance snapshot"""

        # Calculate metrics for all strategies
        strategy_metrics = {}
        for strategy in StrategyType:
            metrics = self.calculate_strategy_metrics(strategy, period)
            if metrics:
                strategy_metrics[strategy] = metrics

        # Calculate system metrics
        system_metrics = self.calculate_system_metrics(period)

        # Calculate comparative metrics
        comparative_metrics = self._calculate_comparative_metrics(strategy_metrics, period)

        # Calculate temporal metrics
        temporal_metrics = self._calculate_temporal_metrics(period)

        # Generate alerts and recommendations
        alerts = self._generate_alerts(strategy_metrics, system_metrics)
        recommendations = self._generate_recommendations(strategy_metrics, system_metrics, comparative_metrics)

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            strategy_metrics=strategy_metrics,
            system_metrics=system_metrics,
            comparative_metrics=comparative_metrics,
            temporal_metrics=temporal_metrics,
            alerts=alerts,
            recommendations=recommendations
        )

        # Store snapshot
        self.performance_snapshots.append(snapshot)

        return snapshot

    def _calculate_signal_quality_score(self, recent_data: List[Dict]) -> float:
        """Calculate signal quality score (precision)"""

        if not recent_data:
            return 0.0

        # True positives: High confidence predictions that succeeded
        true_positives = sum(
            1 for event in recent_data
            if event['confidence_score'] > 0.6 and event['validation_success']
        )

        # False positives: High confidence predictions that failed
        false_positives = sum(
            1 for event in recent_data
            if event['confidence_score'] > 0.6 and not event['validation_success']
        )

        if true_positives + false_positives == 0:
            return 0.0

        return true_positives / (true_positives + false_positives)

    def _calculate_confidence_accuracy(self, strategy: StrategyType, period: str) -> float:
        """Calculate how well confidence scores predict actual success"""

        window_hours = PERFORMANCE_WINDOWS.get(period, 168)
        cutoff_time = datetime.now() - timedelta(hours=window_hours)

        recent_confidence_data = [
            event for event in self.confidence_history[strategy]
            if event['timestamp'] >= cutoff_time
        ]

        if len(recent_confidence_data) < 10:
            return 0.0

        # Calculate correlation between predicted confidence and actual success
        predicted = [event['predicted_confidence'] for event in recent_confidence_data]
        actual = [float(event['actual_success']) for event in recent_confidence_data]

        try:
            correlation = np.corrcoef(predicted, actual)[0, 1]
            return max(0.0, correlation)  # Return 0 if negative correlation
        except:
            return 0.0

    def _calculate_regime_adaptation_score(self, strategy: StrategyType, recent_data: List[Dict]) -> float:
        """Calculate how well strategy adapts to different market regimes"""

        if not recent_data:
            return 0.0

        # Group performance by regime
        regime_performance = defaultdict(list)
        for event in recent_data:
            regime = event['market_regime']
            regime_performance[regime].append(event['validation_success'])

        if len(regime_performance) < 2:
            return 0.5  # Cannot assess adaptation with single regime

        # Calculate performance variance across regimes
        regime_means = [np.mean(performances) for performances in regime_performance.values()]

        if len(regime_means) < 2:
            return 0.5

        # Lower variance across regimes indicates better adaptation
        regime_variance = np.var(regime_means)
        adaptation_score = 1.0 / (1.0 + regime_variance * 10)  # Scale and invert

        return adaptation_score

    def _get_rolling_validation_rates(self, data: List[Dict], window_size: int = 24) -> List[float]:
        """Get rolling validation rates for risk calculations"""

        if len(data) < window_size:
            return [np.mean([event['validation_success'] for event in data])] if data else [0.0]

        rolling_rates = []
        for i in range(window_size, len(data) + 1):
            window_data = data[i-window_size:i]
            rate = np.mean([event['validation_success'] for event in window_data])
            rolling_rates.append(rate)

        return rolling_rates

    def _calculate_sharpe_ratio(self, returns: List[float], benchmark: float) -> float:
        """Calculate Sharpe ratio for strategy performance"""

        if len(returns) < 2:
            return 0.0

        excess_returns = [r - benchmark for r in returns]
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess == 0:
            return 0.0

        return mean_excess / std_excess

    def _calculate_sortino_ratio(self, returns: List[float], benchmark: float) -> float:
        """Calculate Sortino ratio (downside risk only)"""

        if len(returns) < 2:
            return 0.0

        excess_returns = [r - benchmark for r in returns]
        mean_excess = np.mean(excess_returns)

        # Calculate downside deviation
        downside_returns = [r for r in excess_returns if r < 0]

        if not downside_returns:
            return float('inf') if mean_excess > 0 else 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        return mean_excess / downside_std

    def _calculate_maximum_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""

        if not returns:
            return 0.0

        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _calculate_value_at_risk(self, returns: List[float], confidence_level: float) -> float:
        """Calculate Value at Risk at given confidence level"""

        if not returns:
            return 0.0

        return np.percentile(returns, (1 - confidence_level) * 100)

    def _calculate_expected_shortfall(self, returns: List[float], confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""

        if not returns:
            return 0.0

        var = self._calculate_value_at_risk(returns, confidence_level)
        tail_returns = [r for r in returns if r <= var]

        return np.mean(tail_returns) if tail_returns else var

    def _calculate_diversification_index(self, strategy_data: Dict[StrategyType, List[Dict]]) -> float:
        """Calculate strategy diversification index"""

        active_strategies = [s for s, data in strategy_data.items() if data]

        if len(active_strategies) <= 1:
            return 0.0

        # Count active families
        active_families = set()
        for strategy in active_strategies:
            from .unified_confidence_framework import STRATEGY_FAMILIES
            active_families.add(STRATEGY_FAMILIES[strategy])

        # Diversification score based on family variety and balance
        max_possible_families = len(StrategyFamily)
        family_diversity = len(active_families) / max_possible_families

        # Balance score (how evenly distributed across families)
        family_counts = defaultdict(int)
        for strategy in active_strategies:
            from .unified_confidence_framework import STRATEGY_FAMILIES
            family_counts[STRATEGY_FAMILIES[strategy]] += 1

        total_strategies = len(active_strategies)
        family_proportions = [count / total_strategies for count in family_counts.values()]

        # Calculate Shannon entropy for balance
        if family_proportions:
            entropy = -sum(p * np.log(p) for p in family_proportions if p > 0)
            max_entropy = np.log(len(active_families))
            balance_score = entropy / max_entropy if max_entropy > 0 else 0
        else:
            balance_score = 0

        return 0.6 * family_diversity + 0.4 * balance_score

    def _calculate_system_sharpe_ratio(self, strategy_data: Dict[StrategyType, List[Dict]]) -> float:
        """Calculate system-wide Sharpe ratio"""

        all_returns = []
        for strategy, data in strategy_data.items():
            if data:
                validation_rates = self._get_rolling_validation_rates(data)
                all_returns.extend(validation_rates)

        if not all_returns:
            return 0.0

        system_benchmark = np.mean(list(TARGET_VALIDATION_RATES.values()))
        return self._calculate_sharpe_ratio(all_returns, system_benchmark)

    def _calculate_efficiency_ratio(self, strategy_data: Dict[StrategyType, List[Dict]]) -> float:
        """Calculate efficiency ratio (output quality per computational resource)"""

        total_signals = sum(len(data) for data in strategy_data.values())
        total_successes = sum(
            sum(event['validation_success'] for event in data)
            for data in strategy_data.values()
        )

        if total_signals == 0:
            return 0.0

        # Simple efficiency: success rate
        basic_efficiency = total_successes / total_signals

        # Adjust for diversification benefit
        diversification_bonus = self._calculate_diversification_index(strategy_data)

        return basic_efficiency * (1 + 0.2 * diversification_bonus)

    def _calculate_ema_priority_effectiveness(self, strategy_data: Dict[StrategyType, List[Dict]]) -> float:
        """Calculate effectiveness of EMA priority system"""

        ema_data = strategy_data.get(StrategyType.EMA, [])
        if not ema_data:
            return 0.0

        ema_success_rate = np.mean([event['validation_success'] for event in ema_data])
        ema_target = TARGET_VALIDATION_RATES[StrategyType.EMA]

        # Compare EMA performance to its target
        ema_effectiveness = ema_success_rate / ema_target if ema_target > 0 else 0

        # Bonus for EMA helping other strategies (correlation effect)
        ema_correlation_strategies = [StrategyType.MACD, StrategyType.ICHIMOKU, StrategyType.MOMENTUM]

        correlation_benefit = 0.0
        for strategy in ema_correlation_strategies:
            strategy_data_points = strategy_data.get(strategy, [])
            if strategy_data_points:
                strategy_success_rate = np.mean([event['validation_success'] for event in strategy_data_points])
                strategy_target = TARGET_VALIDATION_RATES[strategy]
                correlation_benefit += max(0, strategy_success_rate - strategy_target) / strategy_target

        correlation_benefit /= len(ema_correlation_strategies)

        return min(1.0, ema_effectiveness + 0.3 * correlation_benefit)

    def _calculate_regime_modifier_accuracy(self, all_data: List[Dict]) -> float:
        """Calculate accuracy of regime modifiers"""

        if not all_data:
            return 0.0

        # Group by regime and calculate average performance
        regime_performance = defaultdict(list)
        for event in all_data:
            regime = event['market_regime']
            regime_performance[regime].append(event['validation_success'])

        # Compare actual performance to expected (from regime modifiers)
        from .unified_confidence_framework import REGIME_STRATEGY_CONFIDENCE_MODIFIERS

        accuracy_scores = []
        for regime, performances in regime_performance.items():
            if len(performances) < 5:  # Need minimum samples
                continue

            actual_performance = np.mean(performances)

            # Get expected performance from regime modifiers
            regime_modifiers = REGIME_STRATEGY_CONFIDENCE_MODIFIERS.get(regime, {})
            if regime_modifiers:
                expected_performance = np.mean(list(regime_modifiers.values()))
                accuracy = 1.0 - abs(actual_performance - expected_performance)
                accuracy_scores.append(max(0.0, accuracy))

        return np.mean(accuracy_scores) if accuracy_scores else 0.5

    def _calculate_correlation_optimization(self, strategy_data: Dict[StrategyType, List[Dict]]) -> float:
        """Calculate effectiveness of correlation factor optimization"""

        # Simplified metric: Check if diversified strategies perform better
        family_performance = defaultdict(list)

        for strategy, data in strategy_data.items():
            if data:
                from .unified_confidence_framework import STRATEGY_FAMILIES
                family = STRATEGY_FAMILIES[strategy]
                success_rate = np.mean([event['validation_success'] for event in data])
                family_performance[family].append(success_rate)

        # Calculate variance of family performances (lower is better)
        family_means = [np.mean(performances) for performances in family_performance.values()]

        if len(family_means) < 2:
            return 0.5

        family_variance = np.var(family_means)
        optimization_score = 1.0 / (1.0 + family_variance * 5)  # Scale and invert

        return optimization_score

    def _calculate_optimization_roi(self, strategy_data: Dict[StrategyType, List[Dict]]) -> float:
        """Calculate ROI of optimization multipliers"""

        # Compare optimized strategies to baseline
        optimized_strategies = [StrategyType.MACD, StrategyType.EMA]

        optimized_performance = []
        baseline_performance = []

        for strategy, data in strategy_data.items():
            if not data:
                continue

            success_rate = np.mean([event['validation_success'] for event in data])

            if strategy in optimized_strategies:
                optimized_performance.append(success_rate)
            else:
                baseline_performance.append(success_rate)

        if not optimized_performance or not baseline_performance:
            return 0.5

        optimized_mean = np.mean(optimized_performance)
        baseline_mean = np.mean(baseline_performance)

        if baseline_mean == 0:
            return 1.0 if optimized_mean > 0 else 0.0

        roi = (optimized_mean - baseline_mean) / baseline_mean
        return max(0.0, min(2.0, roi))  # Bound between 0 and 2

    def _calculate_system_stability(self, all_data: List[Dict]) -> float:
        """Calculate system stability score"""

        if len(all_data) < 50:
            return 0.5

        # Calculate rolling performance over time
        sorted_data = sorted(all_data, key=lambda x: x['timestamp'])
        window_size = min(24, len(sorted_data) // 4)

        rolling_performances = []
        for i in range(window_size, len(sorted_data)):
            window_data = sorted_data[i-window_size:i]
            performance = np.mean([event['validation_success'] for event in window_data])
            rolling_performances.append(performance)

        if not rolling_performances:
            return 0.5

        # Stability = 1 - coefficient of variation
        mean_performance = np.mean(rolling_performances)
        std_performance = np.std(rolling_performances)

        if mean_performance == 0:
            return 0.0

        coefficient_of_variation = std_performance / mean_performance
        stability_score = 1.0 / (1.0 + coefficient_of_variation)

        return stability_score

    def _calculate_adaptation_speed(self, all_data: List[Dict]) -> float:
        """Calculate how quickly system adapts to changes"""

        if len(all_data) < 100:
            return 0.5

        # Look for regime changes and measure adaptation speed
        sorted_data = sorted(all_data, key=lambda x: x['timestamp'])

        # Simple adaptation metric: performance improvement after regime changes
        regime_changes = []
        current_regime = None

        for i, event in enumerate(sorted_data):
            if event['market_regime'] != current_regime:
                if current_regime is not None:
                    regime_changes.append(i)
                current_regime = event['market_regime']

        if len(regime_changes) < 2:
            return 0.5

        adaptation_scores = []
        for change_point in regime_changes[:-1]:  # Exclude last change
            # Performance before change
            before_window = sorted_data[max(0, change_point-20):change_point]
            # Performance after change
            after_window = sorted_data[change_point:change_point+20]

            if len(before_window) < 5 or len(after_window) < 5:
                continue

            before_performance = np.mean([event['validation_success'] for event in before_window])
            after_performance = np.mean([event['validation_success'] for event in after_window])

            # Good adaptation = maintaining or improving performance
            adaptation_score = max(0, after_performance / max(before_performance, 0.1))
            adaptation_scores.append(min(1.0, adaptation_score))

        return np.mean(adaptation_scores) if adaptation_scores else 0.5

    def _calculate_comparative_metrics(
        self,
        strategy_metrics: Dict[StrategyType, StrategyMetrics],
        period: str
    ) -> ComparativeMetrics:
        """Calculate comparative analysis metrics"""

        # Strategy rankings by validation rate
        strategy_rankings = {}
        sorted_strategies = sorted(
            strategy_metrics.items(),
            key=lambda x: x[1].validation_rate,
            reverse=True
        )

        for rank, (strategy, _) in enumerate(sorted_strategies, 1):
            strategy_rankings[strategy] = rank

        # Performance attribution (simplified)
        performance_attribution = {
            'regime_factors': 0.3,
            'strategy_selection': 0.25,
            'confidence_scoring': 0.25,
            'correlation_effects': 0.2
        }

        # Regime performance matrix
        regime_performance_matrix = {}
        for regime in MarketRegime:
            regime_performance_matrix[regime] = {}
            for strategy in StrategyType:
                # Placeholder - would calculate from actual data
                regime_performance_matrix[regime][strategy] = 0.4

        # Correlation efficiency
        correlation_efficiency = {}
        strategies = list(strategy_metrics.keys())
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies[i+1:], i+1):
                # Simplified correlation efficiency
                correlation_efficiency[(strategy1, strategy2)] = 0.8

        # Benchmark comparison
        benchmark_comparison = {}
        if strategy_metrics:
            avg_validation_rate = np.mean([m.validation_rate for m in strategy_metrics.values()])
            benchmark_comparison['vs_random'] = avg_validation_rate - self.benchmarks['random_baseline']
            benchmark_comparison['vs_simple_ema'] = avg_validation_rate - self.benchmarks['simple_ema_baseline']
            benchmark_comparison['vs_macd'] = avg_validation_rate - self.benchmarks['macd_baseline']

        # Improvement opportunities
        improvement_opportunities = []
        for strategy, metrics in strategy_metrics.items():
            target = TARGET_VALIDATION_RATES[strategy]
            if metrics.validation_rate < target * 0.8:  # 20% below target
                improvement_opportunities.append(f"Improve {strategy.value} validation rate")

        return ComparativeMetrics(
            strategy_rankings=strategy_rankings,
            performance_attribution=performance_attribution,
            regime_performance_matrix=regime_performance_matrix,
            correlation_efficiency=correlation_efficiency,
            benchmark_comparison=benchmark_comparison,
            improvement_opportunities=improvement_opportunities
        )

    def _calculate_temporal_metrics(self, period: str) -> TemporalMetrics:
        """Calculate time-based performance metrics"""

        # Placeholder implementation - would analyze temporal patterns
        return TemporalMetrics(
            performance_stability=0.8,
            trend_analysis={'overall_trend': 0.05, 'momentum': 0.3},
            seasonality_effects={'daily': 0.1, 'weekly': 0.05},
            learning_curve_slope=0.02,
            regime_transition_performance={'trending_to_ranging': 0.7},
            volatility_clustering=0.6
        )

    def _generate_alerts(
        self,
        strategy_metrics: Dict[StrategyType, StrategyMetrics],
        system_metrics: SystemMetrics
    ) -> List[str]:
        """Generate performance alerts"""

        alerts = []

        # Strategy-level alerts
        for strategy, metrics in strategy_metrics.items():
            target = TARGET_VALIDATION_RATES[strategy]

            if metrics.validation_rate < target * 0.7:  # 30% below target
                alerts.append(f"CRITICAL: {strategy.value} validation rate {metrics.validation_rate:.1%} significantly below target {target:.1%}")

            if metrics.maximum_drawdown > 0.3:  # 30% drawdown
                alerts.append(f"WARNING: {strategy.value} experiencing high drawdown {metrics.maximum_drawdown:.1%}")

            if metrics.sharpe_ratio < 0:
                alerts.append(f"WARNING: {strategy.value} has negative Sharpe ratio {metrics.sharpe_ratio:.2f}")

        # System-level alerts
        if system_metrics.portfolio_validation_rate < 0.3:
            alerts.append("CRITICAL: System-wide validation rate below 30%")

        if system_metrics.strategy_diversification_index < 0.4:
            alerts.append("WARNING: Low strategy diversification detected")

        if system_metrics.system_stability_score < 0.6:
            alerts.append("WARNING: System stability below acceptable threshold")

        return alerts

    def _generate_recommendations(
        self,
        strategy_metrics: Dict[StrategyType, StrategyMetrics],
        system_metrics: SystemMetrics,
        comparative_metrics: ComparativeMetrics
    ) -> List[str]:
        """Generate performance improvement recommendations"""

        recommendations = []

        # Strategy-specific recommendations
        for strategy, metrics in strategy_metrics.items():
            target = TARGET_VALIDATION_RATES[strategy]

            if metrics.validation_rate < target:
                gap = target - metrics.validation_rate
                if gap > 0.1:
                    recommendations.append(f"Increase {strategy.value} confidence modifiers by {gap*100:.0f}%")

            if metrics.confidence_accuracy < 0.6:
                recommendations.append(f"Review {strategy.value} confidence scoring algorithm")

            if metrics.regime_adaptation_score < 0.5:
                recommendations.append(f"Optimize {strategy.value} regime modifiers")

        # System-wide recommendations
        if system_metrics.strategy_diversification_index < 0.6:
            recommendations.append("Add strategies from underrepresented families")

        if system_metrics.ema_priority_effectiveness < 0.8:
            recommendations.append("Review EMA priority weighting system")

        if system_metrics.efficiency_ratio < 0.4:
            recommendations.append("Optimize computational resource allocation")

        # Comparative recommendations
        if comparative_metrics.improvement_opportunities:
            recommendations.extend(comparative_metrics.improvement_opportunities[:3])  # Top 3

        return recommendations[:10]  # Limit to top 10 recommendations