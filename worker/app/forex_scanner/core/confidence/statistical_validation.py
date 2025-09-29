# core/confidence/statistical_validation.py
"""
Statistical Validation Methods for Continuous Improvement
=========================================================

Advanced statistical methods for validating and improving confidence scoring framework

Statistical Methods Implemented:
===============================

1. BAYESIAN CONFIDENCE UPDATING
   Prior: Historical strategy performance
   Likelihood: Recent validation results
   Posterior: Updated confidence modifiers

2. HYPOTHESIS TESTING
   H0: Strategy performance is random
   H1: Strategy has predictive power
   Test: Two-sample t-test with multiple testing correction

3. REGIME CHANGE DETECTION
   Method: Structural break detection using Chow test
   Application: Detect when regime modifiers need updating

4. CROSS-VALIDATION
   Method: Time-series cross-validation with walk-forward analysis
   Purpose: Validate confidence framework out-of-sample

5. CORRELATION ANALYSIS
   Method: Dynamic correlation estimation with exponential weighting
   Purpose: Update strategy correlation factors

6. PERFORMANCE ATTRIBUTION
   Method: Factor decomposition analysis
   Purpose: Identify which components drive success

7. STATISTICAL PROCESS CONTROL
   Method: Control charts for validation rates
   Purpose: Detect performance degradation early

Mathematical Foundation:
=======================

Bayesian Update Formula:
P(θ|data) ∝ P(data|θ) × P(θ)

Where:
- θ = confidence modifier parameters
- P(θ) = prior belief about parameters
- P(data|θ) = likelihood of observed data
- P(θ|data) = posterior updated parameters

Regime Change Detection:
Chow Test Statistic: F = (RSS_r - RSS_u) / RSS_u × (n-2k)/(k)

Where:
- RSS_r = residual sum of squares (restricted model)
- RSS_u = residual sum of squares (unrestricted model)
- n = sample size, k = parameters

Statistical Quality Control:
Control Limits = μ ± 3σ/√n

Where:
- μ = target validation rate
- σ = historical standard deviation
- n = sample size
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import t, chi2, f
from scipy.optimize import minimize
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .unified_confidence_framework import StrategyType, MarketRegime

logger = logging.getLogger(__name__)

# =============================================================================
# STATISTICAL VALIDATION CONSTANTS
# =============================================================================

# Statistical significance levels
ALPHA_LEVEL = 0.05  # 95% confidence level
BONFERRONI_CORRECTION = True  # Apply multiple testing correction

# Bayesian prior parameters
PRIOR_ALPHA = 2.0  # Beta distribution alpha parameter
PRIOR_BETA = 3.0   # Beta distribution beta parameter

# Time series validation parameters
MIN_SAMPLE_SIZE = 30  # Minimum samples for statistical tests
LOOKBACK_WINDOW = 100  # Rolling window for dynamic estimation
UPDATE_FREQUENCY_HOURS = 24  # How often to update parameters

# Control chart parameters
CONTROL_CHART_SIGMA = 3  # Standard deviations for control limits
TREND_DETECTION_POINTS = 7  # Points for trend detection

@dataclass
class ValidationResult:
    """Statistical validation result"""
    strategy: StrategyType
    metric_name: str
    current_value: float
    expected_value: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    recommendation: str

@dataclass
class BayesianUpdate:
    """Bayesian parameter update result"""
    parameter_name: str
    prior_mean: float
    likelihood_estimate: float
    posterior_mean: float
    posterior_variance: float
    credible_interval: Tuple[float, float]
    update_strength: float

@dataclass
class RegimeChangeDetection:
    """Regime change detection result"""
    regime: MarketRegime
    change_detected: bool
    change_point: Optional[datetime]
    test_statistic: float
    p_value: float
    old_parameters: Dict[str, float]
    new_parameters: Dict[str, float]

@dataclass
class PerformanceAttribution:
    """Performance attribution analysis result"""
    strategy: StrategyType
    total_performance: float
    factor_contributions: Dict[str, float]
    residual_performance: float
    r_squared: float
    factor_significance: Dict[str, float]

# =============================================================================
# STATISTICAL VALIDATION CLASS
# =============================================================================

class StatisticalValidationFramework:
    """
    Advanced statistical validation framework for confidence scoring system

    Provides continuous monitoring, validation, and improvement of the
    confidence scoring framework using rigorous statistical methods.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Historical data storage
        self.validation_history: Dict[StrategyType, List[Dict]] = {}
        self.regime_history: List[Dict] = []
        self.performance_history: Dict[StrategyType, List[float]] = {}

        # Bayesian parameters
        self.bayesian_parameters: Dict[str, Dict] = {}
        self.parameter_update_history: List[Dict] = []

        # Control charts
        self.control_charts: Dict[StrategyType, Dict] = {}

        # Last update timestamp
        self.last_update = datetime.now()

        self._initialize_bayesian_priors()
        self._initialize_control_charts()

    def validate_strategy_performance(
        self,
        strategy: StrategyType,
        recent_validation_rates: List[float],
        expected_range: Tuple[float, float]
    ) -> ValidationResult:
        """
        Validate strategy performance using statistical hypothesis testing

        H0: Strategy performance equals expected value
        H1: Strategy performance differs from expected value (two-tailed test)
        """

        if len(recent_validation_rates) < MIN_SAMPLE_SIZE:
            return ValidationResult(
                strategy=strategy,
                metric_name="validation_rate",
                current_value=np.mean(recent_validation_rates) if recent_validation_rates else 0.0,
                expected_value=np.mean(expected_range),
                p_value=1.0,
                is_significant=False,
                confidence_interval=(0.0, 1.0),
                recommendation="Insufficient data for statistical testing"
            )

        # Calculate sample statistics
        sample_mean = np.mean(recent_validation_rates)
        sample_std = np.std(recent_validation_rates, ddof=1)
        sample_size = len(recent_validation_rates)
        expected_mean = np.mean(expected_range)

        # Perform two-sample t-test
        t_statistic = (sample_mean - expected_mean) / (sample_std / np.sqrt(sample_size))
        degrees_freedom = sample_size - 1
        p_value = 2 * (1 - t.cdf(abs(t_statistic), degrees_freedom))

        # Apply Bonferroni correction for multiple testing
        if BONFERRONI_CORRECTION:
            adjusted_alpha = ALPHA_LEVEL / len(StrategyType)
        else:
            adjusted_alpha = ALPHA_LEVEL

        is_significant = p_value < adjusted_alpha

        # Calculate confidence interval
        t_critical = t.ppf(1 - adjusted_alpha/2, degrees_freedom)
        margin_error = t_critical * (sample_std / np.sqrt(sample_size))
        confidence_interval = (
            sample_mean - margin_error,
            sample_mean + margin_error
        )

        # Generate recommendation
        if is_significant:
            if sample_mean < expected_range[0]:
                recommendation = f"Performance significantly below target. Increase confidence modifiers."
            elif sample_mean > expected_range[1]:
                recommendation = f"Performance significantly above target. Consider tightening validation."
            else:
                recommendation = f"Performance significantly different but within acceptable range."
        else:
            recommendation = "Performance within statistical expectations. No action needed."

        return ValidationResult(
            strategy=strategy,
            metric_name="validation_rate",
            current_value=sample_mean,
            expected_value=expected_mean,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=confidence_interval,
            recommendation=recommendation
        )

    def bayesian_parameter_update(
        self,
        parameter_name: str,
        recent_observations: List[float],
        strategy: Optional[StrategyType] = None
    ) -> BayesianUpdate:
        """
        Update confidence parameters using Bayesian inference

        Uses Beta-Binomial conjugate prior for validation rates
        Uses Normal-Normal conjugate prior for continuous parameters
        """

        # Get prior parameters
        prior_params = self.bayesian_parameters.get(parameter_name, {
            'alpha': PRIOR_ALPHA,
            'beta': PRIOR_BETA,
            'mu': 0.5,
            'sigma_squared': 0.1
        })

        if len(recent_observations) == 0:
            return BayesianUpdate(
                parameter_name=parameter_name,
                prior_mean=prior_params['mu'],
                likelihood_estimate=prior_params['mu'],
                posterior_mean=prior_params['mu'],
                posterior_variance=prior_params['sigma_squared'],
                credible_interval=(0.0, 1.0),
                update_strength=0.0
            )

        # For validation rates (bounded 0-1), use Beta-Binomial model
        if 'validation_rate' in parameter_name or 'confidence' in parameter_name:
            return self._bayesian_beta_update(parameter_name, recent_observations, prior_params)
        else:
            # For unbounded parameters, use Normal-Normal model
            return self._bayesian_normal_update(parameter_name, recent_observations, prior_params)

    def _bayesian_beta_update(
        self,
        parameter_name: str,
        observations: List[float],
        prior_params: Dict
    ) -> BayesianUpdate:
        """Bayesian update using Beta-Binomial conjugate prior"""

        # Convert validation rates to successes/failures
        successes = sum(1 for x in observations if x > 0.5)  # Above median performance
        failures = len(observations) - successes

        # Prior parameters
        alpha_prior = prior_params.get('alpha', PRIOR_ALPHA)
        beta_prior = prior_params.get('beta', PRIOR_BETA)

        # Posterior parameters (Beta distribution)
        alpha_posterior = alpha_prior + successes
        beta_posterior = beta_prior + failures

        # Posterior statistics
        posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
        posterior_variance = (alpha_posterior * beta_posterior) / (
            (alpha_posterior + beta_posterior) ** 2 * (alpha_posterior + beta_posterior + 1)
        )

        # 95% credible interval
        credible_interval = (
            stats.beta.ppf(0.025, alpha_posterior, beta_posterior),
            stats.beta.ppf(0.975, alpha_posterior, beta_posterior)
        )

        # Update strength (how much the data changed our beliefs)
        prior_mean = alpha_prior / (alpha_prior + beta_prior)
        likelihood_estimate = np.mean(observations) if observations else prior_mean
        update_strength = abs(posterior_mean - prior_mean) / prior_mean if prior_mean > 0 else 0

        # Store updated parameters
        self.bayesian_parameters[parameter_name] = {
            'alpha': alpha_posterior,
            'beta': beta_posterior,
            'mu': posterior_mean,
            'sigma_squared': posterior_variance
        }

        return BayesianUpdate(
            parameter_name=parameter_name,
            prior_mean=prior_mean,
            likelihood_estimate=likelihood_estimate,
            posterior_mean=posterior_mean,
            posterior_variance=posterior_variance,
            credible_interval=credible_interval,
            update_strength=update_strength
        )

    def _bayesian_normal_update(
        self,
        parameter_name: str,
        observations: List[float],
        prior_params: Dict
    ) -> BayesianUpdate:
        """Bayesian update using Normal-Normal conjugate prior"""

        # Prior parameters
        mu_prior = prior_params.get('mu', 0.5)
        sigma_squared_prior = prior_params.get('sigma_squared', 0.1)

        # Data statistics
        n = len(observations)
        sample_mean = np.mean(observations)
        sample_variance = np.var(observations, ddof=1) if n > 1 else sigma_squared_prior

        # Posterior parameters (assuming known variance)
        precision_prior = 1 / sigma_squared_prior
        precision_data = n / sample_variance
        precision_posterior = precision_prior + precision_data

        mu_posterior = (
            precision_prior * mu_prior + precision_data * sample_mean
        ) / precision_posterior

        sigma_squared_posterior = 1 / precision_posterior

        # 95% credible interval
        credible_interval = (
            mu_posterior - 1.96 * np.sqrt(sigma_squared_posterior),
            mu_posterior + 1.96 * np.sqrt(sigma_squared_posterior)
        )

        # Update strength
        update_strength = abs(mu_posterior - mu_prior) / (abs(mu_prior) + 1e-6)

        # Store updated parameters
        self.bayesian_parameters[parameter_name] = {
            'mu': mu_posterior,
            'sigma_squared': sigma_squared_posterior,
            'alpha': prior_params.get('alpha', PRIOR_ALPHA),
            'beta': prior_params.get('beta', PRIOR_BETA)
        }

        return BayesianUpdate(
            parameter_name=parameter_name,
            prior_mean=mu_prior,
            likelihood_estimate=sample_mean,
            posterior_mean=mu_posterior,
            posterior_variance=sigma_squared_posterior,
            credible_interval=credible_interval,
            update_strength=update_strength
        )

    def detect_regime_changes(
        self,
        regime_data: List[Dict],
        window_size: int = 50
    ) -> List[RegimeChangeDetection]:
        """
        Detect regime changes using structural break tests

        Uses Chow test to detect structural breaks in regime-strategy relationships
        """

        if len(regime_data) < 2 * window_size:
            return []

        regime_changes = []

        # Group data by regime
        regime_groups = {}
        for data_point in regime_data:
            regime = data_point.get('regime')
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(data_point)

        for regime, data_points in regime_groups.items():
            if len(data_points) < 2 * window_size:
                continue

            # Extract validation rates over time
            validation_rates = [dp.get('validation_rate', 0.0) for dp in data_points]
            timestamps = [dp.get('timestamp') for dp in data_points]

            # Test for structural break at midpoint
            midpoint = len(validation_rates) // 2
            first_half = validation_rates[:midpoint]
            second_half = validation_rates[midpoint:]

            # Perform Chow test
            change_detected, test_statistic, p_value = self._chow_test(
                first_half, second_half
            )

            if change_detected:
                change_point = timestamps[midpoint] if timestamps[midpoint] else datetime.now()

                # Calculate old and new parameters
                old_params = {
                    'mean': np.mean(first_half),
                    'std': np.std(first_half),
                    'trend': self._calculate_trend(first_half)
                }

                new_params = {
                    'mean': np.mean(second_half),
                    'std': np.std(second_half),
                    'trend': self._calculate_trend(second_half)
                }

                regime_changes.append(RegimeChangeDetection(
                    regime=MarketRegime(regime),
                    change_detected=True,
                    change_point=change_point,
                    test_statistic=test_statistic,
                    p_value=p_value,
                    old_parameters=old_params,
                    new_parameters=new_params
                ))

        return regime_changes

    def _chow_test(
        self,
        series1: List[float],
        series2: List[float]
    ) -> Tuple[bool, float, float]:
        """
        Perform Chow test for structural break

        H0: No structural break (same parameters)
        H1: Structural break (different parameters)
        """

        if len(series1) < 3 or len(series2) < 3:
            return False, 0.0, 1.0

        # Convert to numpy arrays
        y1 = np.array(series1)
        y2 = np.array(series2)
        y_combined = np.concatenate([y1, y2])

        # Create time indices
        x1 = np.arange(len(y1))
        x2 = np.arange(len(y2))
        x_combined = np.arange(len(y_combined))

        # Fit linear models
        try:
            # Restricted model (combined data)
            p_restricted = np.polyfit(x_combined, y_combined, 1)
            y_pred_restricted = np.polyval(p_restricted, x_combined)
            rss_restricted = np.sum((y_combined - y_pred_restricted) ** 2)

            # Unrestricted models (separate data)
            p1 = np.polyfit(x1, y1, 1)
            p2 = np.polyfit(x2, y2, 1)

            y1_pred = np.polyval(p1, x1)
            y2_pred = np.polyval(p2, x2)

            rss1 = np.sum((y1 - y1_pred) ** 2)
            rss2 = np.sum((y2 - y2_pred) ** 2)
            rss_unrestricted = rss1 + rss2

            # Chow test statistic
            n = len(y_combined)
            k = 2  # Number of parameters in linear model

            if rss_unrestricted > 0 and n > 2 * k:
                f_statistic = (
                    (rss_restricted - rss_unrestricted) / k
                ) / (rss_unrestricted / (n - 2 * k))

                # P-value from F-distribution
                p_value = 1 - f.cdf(f_statistic, k, n - 2 * k)

                # Test for significance
                is_significant = p_value < ALPHA_LEVEL

                return is_significant, f_statistic, p_value

        except (np.linalg.LinAlgError, ValueError) as e:
            self.logger.warning(f"Chow test failed: {e}")

        return False, 0.0, 1.0

    def _calculate_trend(self, series: List[float]) -> float:
        """Calculate linear trend slope"""
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        y = np.array(series)

        try:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        except:
            return 0.0

    def cross_validate_framework(
        self,
        historical_data: Dict[StrategyType, List[Dict]],
        n_splits: int = 5
    ) -> Dict[StrategyType, Dict[str, float]]:
        """
        Perform time-series cross-validation of confidence framework

        Uses walk-forward analysis to validate out-of-sample performance
        """

        validation_results = {}

        for strategy, data in historical_data.items():
            if len(data) < n_splits * 20:  # Need minimum data for meaningful CV
                continue

            # Prepare data
            timestamps = [d['timestamp'] for d in data]
            features = [d.get('features', {}) for d in data]
            labels = [d.get('validation_success', False) for d in data]

            # Time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []

            for train_idx, test_idx in tscv.split(features):
                # Split data
                train_features = [features[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                test_features = [features[i] for i in test_idx]
                test_labels = [labels[i] for i in test_idx]

                # Simple prediction based on confidence scores
                predictions = [
                    f.get('final_confidence', 0.5) > 0.5
                    for f in test_features
                ]

                # Calculate metrics
                if len(test_labels) > 0 and len(predictions) > 0:
                    accuracy = accuracy_score(test_labels, predictions)
                    cv_scores.append(accuracy)

            # Calculate cross-validation statistics
            if cv_scores:
                validation_results[strategy] = {
                    'mean_accuracy': np.mean(cv_scores),
                    'std_accuracy': np.std(cv_scores),
                    'min_accuracy': np.min(cv_scores),
                    'max_accuracy': np.max(cv_scores),
                    'n_folds': len(cv_scores)
                }

        return validation_results

    def performance_attribution_analysis(
        self,
        strategy: StrategyType,
        performance_data: List[Dict]
    ) -> PerformanceAttribution:
        """
        Analyze which factors contribute most to strategy performance

        Uses multiple regression to decompose performance into factor contributions
        """

        if len(performance_data) < 20:
            return PerformanceAttribution(
                strategy=strategy,
                total_performance=0.0,
                factor_contributions={},
                residual_performance=0.0,
                r_squared=0.0,
                factor_significance={}
            )

        # Extract dependent variable (performance)
        y = np.array([d.get('validation_success', 0.0) for d in performance_data])

        # Extract independent variables (factors)
        factors = {}
        for data_point in performance_data:
            confidence_components = data_point.get('confidence_components', {})
            for factor_name, factor_value in confidence_components.items():
                if factor_name not in factors:
                    factors[factor_name] = []
                factors[factor_name].append(factor_value)

        if not factors:
            return PerformanceAttribution(
                strategy=strategy,
                total_performance=np.mean(y) if len(y) > 0 else 0.0,
                factor_contributions={},
                residual_performance=0.0,
                r_squared=0.0,
                factor_significance={}
            )

        # Create design matrix
        factor_names = list(factors.keys())
        X = np.column_stack([factors[name] for name in factor_names])

        # Add intercept
        X = np.column_stack([np.ones(X.shape[0]), X])

        try:
            # Multiple regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            # Predictions and residuals
            y_pred = X @ beta
            residuals = y - y_pred

            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Factor contributions
            factor_contributions = {}
            for i, factor_name in enumerate(factor_names):
                # Contribution = beta * mean(factor)
                factor_contributions[factor_name] = beta[i + 1] * np.mean(factors[factor_name])

            # Statistical significance of factors
            factor_significance = {}
            if X.shape[0] > X.shape[1]:  # Overdetermined system
                # Calculate standard errors (simplified)
                mse = ss_res / (X.shape[0] - X.shape[1])
                var_covar = mse * np.linalg.inv(X.T @ X)
                std_errors = np.sqrt(np.diag(var_covar))

                for i, factor_name in enumerate(factor_names):
                    if std_errors[i + 1] > 0:
                        t_stat = beta[i + 1] / std_errors[i + 1]
                        p_value = 2 * (1 - t.cdf(abs(t_stat), X.shape[0] - X.shape[1]))
                        factor_significance[factor_name] = p_value

            return PerformanceAttribution(
                strategy=strategy,
                total_performance=np.mean(y),
                factor_contributions=factor_contributions,
                residual_performance=np.mean(residuals),
                r_squared=r_squared,
                factor_significance=factor_significance
            )

        except np.linalg.LinAlgError as e:
            self.logger.warning(f"Performance attribution failed for {strategy.value}: {e}")
            return PerformanceAttribution(
                strategy=strategy,
                total_performance=np.mean(y) if len(y) > 0 else 0.0,
                factor_contributions={},
                residual_performance=0.0,
                r_squared=0.0,
                factor_significance={}
            )

    def _initialize_bayesian_priors(self):
        """Initialize Bayesian prior parameters"""

        # Strategy-specific priors based on known characteristics
        strategy_priors = {
            'ema_validation_rate': {'alpha': 4.0, 'beta': 2.0},  # Optimistic for EMA
            'macd_validation_rate': {'alpha': 3.0, 'beta': 4.0},  # Based on 41.3% success
            'momentum_validation_rate': {'alpha': 2.0, 'beta': 4.0},  # Conservative
        }

        # Regime modifier priors
        regime_priors = {
            'trending_modifier': {'mu': 0.8, 'sigma_squared': 0.05},
            'ranging_modifier': {'mu': 0.6, 'sigma_squared': 0.05},
            'volatility_modifier': {'mu': 0.7, 'sigma_squared': 0.1}
        }

        self.bayesian_parameters.update(strategy_priors)
        self.bayesian_parameters.update(regime_priors)

    def _initialize_control_charts(self):
        """Initialize statistical process control charts"""

        for strategy in StrategyType:
            self.control_charts[strategy] = {
                'target_rate': 0.4,  # Default target
                'control_limits': (0.1, 0.7),  # 3-sigma limits
                'recent_points': [],
                'alert_status': 'normal'
            }

        # Special targets for priority strategies
        self.control_charts[StrategyType.EMA]['target_rate'] = 0.6
        self.control_charts[StrategyType.EMA]['control_limits'] = (0.3, 0.9)

    def update_control_charts(
        self,
        strategy: StrategyType,
        validation_rate: float
    ) -> Dict[str, Any]:
        """Update control charts and detect process changes"""

        chart = self.control_charts[strategy]
        chart['recent_points'].append({
            'timestamp': datetime.now(),
            'value': validation_rate
        })

        # Keep only recent points
        if len(chart['recent_points']) > LOOKBACK_WINDOW:
            chart['recent_points'] = chart['recent_points'][-LOOKBACK_WINDOW:]

        # Calculate control limits
        recent_values = [p['value'] for p in chart['recent_points']]
        if len(recent_values) >= MIN_SAMPLE_SIZE:
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)

            ucl = mean_value + CONTROL_CHART_SIGMA * std_value / np.sqrt(len(recent_values))
            lcl = mean_value - CONTROL_CHART_SIGMA * std_value / np.sqrt(len(recent_values))

            chart['control_limits'] = (max(0.0, lcl), min(1.0, ucl))

        # Check for alerts
        alert_status = 'normal'
        alert_message = ""

        if validation_rate > chart['control_limits'][1]:
            alert_status = 'above_control'
            alert_message = f"Validation rate {validation_rate:.1%} above upper control limit"
        elif validation_rate < chart['control_limits'][0]:
            alert_status = 'below_control'
            alert_message = f"Validation rate {validation_rate:.1%} below lower control limit"

        # Check for trends
        if len(recent_values) >= TREND_DETECTION_POINTS:
            last_points = recent_values[-TREND_DETECTION_POINTS:]
            if all(last_points[i] < last_points[i+1] for i in range(len(last_points)-1)):
                alert_status = 'upward_trend'
                alert_message = f"Detected upward trend in validation rate"
            elif all(last_points[i] > last_points[i+1] for i in range(len(last_points)-1)):
                alert_status = 'downward_trend'
                alert_message = f"Detected downward trend in validation rate"

        chart['alert_status'] = alert_status

        return {
            'strategy': strategy.value,
            'current_value': validation_rate,
            'target_rate': chart['target_rate'],
            'control_limits': chart['control_limits'],
            'alert_status': alert_status,
            'alert_message': alert_message,
            'n_points': len(chart['recent_points'])
        }

    def generate_statistical_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistical validation report"""

        report = {
            'timestamp': datetime.now().isoformat(),
            'bayesian_updates': {},
            'control_chart_status': {},
            'recent_regime_changes': [],
            'parameter_recommendations': [],
            'statistical_health': 'unknown'
        }

        # Bayesian parameter status
        for param_name, params in self.bayesian_parameters.items():
            report['bayesian_updates'][param_name] = {
                'current_estimate': params.get('mu', params.get('alpha', 0.0)),
                'confidence_interval': self._calculate_credible_interval(params),
                'last_update_strength': params.get('update_strength', 0.0)
            }

        # Control chart status
        alert_count = 0
        for strategy, chart in self.control_charts.items():
            status = chart['alert_status']
            report['control_chart_status'][strategy.value] = {
                'alert_status': status,
                'target_rate': chart['target_rate'],
                'control_limits': chart['control_limits']
            }

            if status != 'normal':
                alert_count += 1

        # Overall statistical health
        if alert_count == 0:
            report['statistical_health'] = 'excellent'
        elif alert_count <= 2:
            report['statistical_health'] = 'good'
        elif alert_count <= 4:
            report['statistical_health'] = 'warning'
        else:
            report['statistical_health'] = 'critical'

        # Generate recommendations
        if alert_count > 0:
            report['parameter_recommendations'].append(
                f"Review {alert_count} strategies with control chart alerts"
            )

        return report

    def _calculate_credible_interval(self, params: Dict) -> Tuple[float, float]:
        """Calculate credible interval from Bayesian parameters"""

        if 'alpha' in params and 'beta' in params:
            # Beta distribution
            alpha, beta = params['alpha'], params['beta']
            return (
                stats.beta.ppf(0.025, alpha, beta),
                stats.beta.ppf(0.975, alpha, beta)
            )
        elif 'mu' in params and 'sigma_squared' in params:
            # Normal distribution
            mu, sigma_sq = params['mu'], params['sigma_squared']
            sigma = np.sqrt(sigma_sq)
            return (
                mu - 1.96 * sigma,
                mu + 1.96 * sigma
            )
        else:
            return (0.0, 1.0)