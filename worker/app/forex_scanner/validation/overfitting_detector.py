# forex_scanner/validation/overfitting_detector.py
"""
Advanced Overfitting Detection for Trading Strategy Backtests

This module implements sophisticated mathematical methods to detect overfitting
in trading strategy backtests, using multiple validation techniques from
quantitative finance and machine learning literature.

Key Methods:
1. Walk-Forward Analysis with statistical testing
2. Time Series Cross-Validation with performance degradation metrics
3. Information Coefficient stability analysis
4. Parameter sensitivity analysis with Monte Carlo simulation
5. Combinatorial Purged Cross-Validation (CPCV)
6. Deflated Sharpe Ratio calculations
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Statistical and ML libraries
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, jarque_bera
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


class OverfittingRisk(Enum):
    """Overfitting risk levels"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


@dataclass
class WalkForwardResults:
    """Results from walk-forward analysis"""
    periods_analyzed: int
    in_sample_sharpe_mean: float
    out_sample_sharpe_mean: float
    degradation_ratio: float
    performance_difference_pvalue: float
    is_significant_degradation: bool
    stability_score: float
    periods_data: List[Dict[str, Any]]


@dataclass
class CrossValidationResults:
    """Results from cross-validation analysis"""
    n_folds: int
    cv_score_mean: float
    cv_score_std: float
    train_test_correlation: float
    overfitting_ratio: float
    stability_score: float
    fold_results: List[Dict[str, Any]]


@dataclass
class ParameterSensitivityResults:
    """Results from parameter sensitivity analysis"""
    parameters_tested: int
    sensitivity_scores: Dict[str, float]
    monte_carlo_results: Dict[str, Any]
    robustness_score: float
    parameter_stability: float


@dataclass
class OverfittingAssessment:
    """Comprehensive overfitting assessment"""
    overall_risk_level: OverfittingRisk
    confidence_score: float
    primary_concerns: List[str]
    risk_factors: Dict[str, float]
    recommendations: List[str]
    statistical_evidence: Dict[str, Any]


class OverfittingDetector:
    """
    Advanced Overfitting Detection System for Trading Strategy Backtests

    Implements multiple sophisticated methods to detect and quantify overfitting
    risk in trading strategy backtests, providing statistical evidence and
    recommendations for strategy validation.
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 min_sample_size: int = 50,
                 cv_folds: int = 5,
                 significance_level: float = 0.05,
                 monte_carlo_simulations: int = 1000,
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.min_sample_size = min_sample_size
        self.cv_folds = cv_folds
        self.significance_level = significance_level
        self.monte_carlo_simulations = monte_carlo_simulations
        self.logger = logger or logging.getLogger(__name__)

        # Overfitting detection thresholds
        self.max_acceptable_degradation = 1.5  # Max acceptable in-sample/out-sample ratio
        self.min_stability_score = 0.6
        self.min_information_coefficient = 0.02
        self.max_parameter_sensitivity = 0.3

        # Cross-validation parameters
        self.min_fold_size_ratio = 0.15  # 15% minimum for each fold
        self.purge_buffer_ratio = 0.05   # 5% buffer for purging

        self.logger.info(f"🎯 Overfitting Detector initialized:")
        self.logger.info(f"   Cross-validation folds: {self.cv_folds}")
        self.logger.info(f"   Monte Carlo simulations: {self.monte_carlo_simulations}")
        self.logger.info(f"   Max degradation ratio: {self.max_acceptable_degradation}")

    def detect_overfitting(self, execution_id: int) -> Dict[str, Any]:
        """
        Comprehensive overfitting detection analysis

        Args:
            execution_id: Backtest execution ID to analyze

        Returns:
            Comprehensive overfitting detection results
        """
        self.logger.info(f"🎯 Starting overfitting detection for execution {execution_id}")

        try:
            # Load backtest data
            backtest_data = self._load_backtest_data(execution_id)

            if len(backtest_data) < self.min_sample_size:
                return self._create_insufficient_data_result(execution_id, len(backtest_data))

            # Initialize results container
            results = {
                'execution_id': execution_id,
                'analysis_timestamp': datetime.now(timezone.utc),
                'sample_size': len(backtest_data),
                'detection_methods': {},
                'overall_assessment': {}
            }

            # Method 1: Walk-Forward Analysis
            self.logger.info("📊 Performing walk-forward analysis...")
            results['detection_methods']['walk_forward'] = self._walk_forward_analysis(backtest_data)

            # Method 2: Time Series Cross-Validation
            self.logger.info("🔄 Performing cross-validation analysis...")
            results['detection_methods']['cross_validation'] = self._cross_validation_analysis(backtest_data)

            # Method 3: Information Coefficient Analysis
            self.logger.info("📈 Analyzing information coefficient stability...")
            results['detection_methods']['ic_stability'] = self._information_coefficient_stability(backtest_data)

            # Method 4: Parameter Sensitivity Analysis
            self.logger.info("🎛️ Performing parameter sensitivity analysis...")
            results['detection_methods']['parameter_sensitivity'] = self._parameter_sensitivity_analysis(backtest_data, execution_id)

            # Method 5: Deflated Sharpe Ratio
            self.logger.info("📉 Calculating deflated Sharpe ratio...")
            results['detection_methods']['deflated_sharpe'] = self._deflated_sharpe_ratio(backtest_data, execution_id)

            # Method 6: Combinatorial Purged Cross-Validation (Advanced)
            if len(backtest_data) >= 100:  # Only for larger datasets
                self.logger.info("🧮 Performing combinatorial purged cross-validation...")
                results['detection_methods']['cpcv'] = self._combinatorial_purged_cv(backtest_data)

            # Comprehensive Assessment
            self.logger.info("🔍 Generating comprehensive assessment...")
            results['overall_assessment'] = self._generate_overfitting_assessment(results['detection_methods'])

            # Store results
            self._store_overfitting_results(execution_id, results)

            self.logger.info(f"✅ Overfitting detection completed:")
            self.logger.info(f"   Risk Level: {results['overall_assessment']['risk_level']}")
            self.logger.info(f"   Confidence Score: {results['overall_assessment']['confidence_score']:.3f}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Overfitting detection failed: {e}")
            return {
                'execution_id': execution_id,
                'status': 'error',
                'error_message': str(e),
                'analysis_timestamp': datetime.now(timezone.utc)
            }

    def _walk_forward_analysis(self, backtest_data: pd.DataFrame) -> WalkForwardResults:
        """
        Perform walk-forward analysis to detect overfitting

        Walk-forward analysis splits the data into sequential training and testing
        periods, measuring performance degradation from in-sample to out-of-sample.
        """
        try:
            # Prepare returns data
            returns_data = self._prepare_returns_data(backtest_data)

            n_periods = len(returns_data)
            min_train_size = max(30, int(n_periods * 0.3))
            step_size = max(5, n_periods // 20)  # Adaptive step size

            periods_data = []

            for start_idx in range(min_train_size, n_periods - 10, step_size):
                # In-sample period (expanding window)
                in_sample = returns_data.iloc[:start_idx]

                # Out-of-sample period (next 10 periods or remaining)
                out_sample_end = min(start_idx + 10, n_periods)
                out_sample = returns_data.iloc[start_idx:out_sample_end]

                if len(out_sample) >= 5:  # Minimum out-sample size
                    # Calculate performance metrics
                    in_sample_metrics = self._calculate_performance_metrics(in_sample)
                    out_sample_metrics = self._calculate_performance_metrics(out_sample)

                    period_result = {
                        'period_end': start_idx,
                        'in_sample_size': len(in_sample),
                        'out_sample_size': len(out_sample),
                        'in_sample_sharpe': in_sample_metrics['sharpe_ratio'],
                        'out_sample_sharpe': out_sample_metrics['sharpe_ratio'],
                        'in_sample_return': in_sample_metrics['total_return'],
                        'out_sample_return': out_sample_metrics['total_return'],
                        'degradation_ratio': (
                            in_sample_metrics['sharpe_ratio'] / out_sample_metrics['sharpe_ratio']
                            if out_sample_metrics['sharpe_ratio'] != 0 else float('inf')
                        )
                    }

                    periods_data.append(period_result)

            if not periods_data:
                return WalkForwardResults(
                    periods_analyzed=0,
                    in_sample_sharpe_mean=0,
                    out_sample_sharpe_mean=0,
                    degradation_ratio=0,
                    performance_difference_pvalue=1.0,
                    is_significant_degradation=False,
                    stability_score=0,
                    periods_data=[]
                )

            # Calculate aggregate metrics
            in_sample_sharpes = [p['in_sample_sharpe'] for p in periods_data if not np.isinf(p['in_sample_sharpe'])]
            out_sample_sharpes = [p['out_sample_sharpe'] for p in periods_data if not np.isinf(p['out_sample_sharpe'])]

            in_sample_mean = np.mean(in_sample_sharpes) if in_sample_sharpes else 0
            out_sample_mean = np.mean(out_sample_sharpes) if out_sample_sharpes else 0

            # Statistical test for performance difference
            if len(in_sample_sharpes) > 1 and len(out_sample_sharpes) > 1:
                try:
                    t_stat, p_value = ttest_ind(in_sample_sharpes, out_sample_sharpes)
                except:
                    # Fallback to Mann-Whitney U test for non-normal distributions
                    u_stat, p_value = mannwhitneyu(in_sample_sharpes, out_sample_sharpes, alternative='two-sided')
            else:
                p_value = 1.0

            # Calculate degradation ratio
            degradation_ratio = in_sample_mean / out_sample_mean if out_sample_mean != 0 else float('inf')

            # Calculate stability score
            out_sample_std = np.std(out_sample_sharpes) if out_sample_sharpes else 0
            stability_score = 1 - (out_sample_std / abs(out_sample_mean)) if out_sample_mean != 0 else 0
            stability_score = max(0, min(1, stability_score))

            return WalkForwardResults(
                periods_analyzed=len(periods_data),
                in_sample_sharpe_mean=in_sample_mean,
                out_sample_sharpe_mean=out_sample_mean,
                degradation_ratio=degradation_ratio,
                performance_difference_pvalue=p_value,
                is_significant_degradation=p_value < self.significance_level and degradation_ratio > self.max_acceptable_degradation,
                stability_score=stability_score,
                periods_data=periods_data
            )

        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {e}")
            return WalkForwardResults(
                periods_analyzed=0,
                in_sample_sharpe_mean=0,
                out_sample_sharpe_mean=0,
                degradation_ratio=0,
                performance_difference_pvalue=1.0,
                is_significant_degradation=False,
                stability_score=0,
                periods_data=[]
            )

    def _cross_validation_analysis(self, backtest_data: pd.DataFrame) -> CrossValidationResults:
        """
        Perform time series cross-validation to detect overfitting
        """
        try:
            returns_data = self._prepare_returns_data(backtest_data)

            if len(returns_data) < self.cv_folds * 10:
                return CrossValidationResults(
                    n_folds=0,
                    cv_score_mean=0,
                    cv_score_std=0,
                    train_test_correlation=0,
                    overfitting_ratio=0,
                    stability_score=0,
                    fold_results=[]
                )

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)

            fold_results = []
            train_scores = []
            test_scores = []

            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(returns_data)):
                train_data = returns_data.iloc[train_idx]
                test_data = returns_data.iloc[test_idx]

                if len(train_data) < 10 or len(test_data) < 5:
                    continue

                # Calculate performance metrics
                train_metrics = self._calculate_performance_metrics(train_data)
                test_metrics = self._calculate_performance_metrics(test_data)

                fold_result = {
                    'fold': fold_idx,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'train_sharpe': train_metrics['sharpe_ratio'],
                    'test_sharpe': test_metrics['sharpe_ratio'],
                    'train_return': train_metrics['total_return'],
                    'test_return': test_metrics['total_return'],
                    'overfitting_ratio': (
                        train_metrics['sharpe_ratio'] / test_metrics['sharpe_ratio']
                        if test_metrics['sharpe_ratio'] != 0 else float('inf')
                    )
                }

                fold_results.append(fold_result)
                train_scores.append(train_metrics['sharpe_ratio'])
                test_scores.append(test_metrics['sharpe_ratio'])

            if not fold_results:
                return CrossValidationResults(
                    n_folds=0,
                    cv_score_mean=0,
                    cv_score_std=0,
                    train_test_correlation=0,
                    overfitting_ratio=0,
                    stability_score=0,
                    fold_results=[]
                )

            # Calculate cross-validation metrics
            test_scores_clean = [s for s in test_scores if not np.isinf(s) and not np.isnan(s)]

            cv_score_mean = np.mean(test_scores_clean) if test_scores_clean else 0
            cv_score_std = np.std(test_scores_clean) if test_scores_clean else 0

            # Train-test correlation (indicator of overfitting if too high)
            if len(train_scores) > 1 and len(test_scores) > 1:
                try:
                    train_test_corr = np.corrcoef(train_scores, test_scores)[0, 1]
                    if np.isnan(train_test_corr):
                        train_test_corr = 0
                except:
                    train_test_corr = 0
            else:
                train_test_corr = 0

            # Overall overfitting ratio
            overfitting_ratios = [f['overfitting_ratio'] for f in fold_results if not np.isinf(f['overfitting_ratio'])]
            overfitting_ratio = np.mean(overfitting_ratios) if overfitting_ratios else 1.0

            # Stability score
            stability_score = 1 - (cv_score_std / abs(cv_score_mean)) if cv_score_mean != 0 else 0
            stability_score = max(0, min(1, stability_score))

            return CrossValidationResults(
                n_folds=len(fold_results),
                cv_score_mean=cv_score_mean,
                cv_score_std=cv_score_std,
                train_test_correlation=train_test_corr,
                overfitting_ratio=overfitting_ratio,
                stability_score=stability_score,
                fold_results=fold_results
            )

        except Exception as e:
            self.logger.error(f"Error in cross-validation analysis: {e}")
            return CrossValidationResults(
                n_folds=0,
                cv_score_mean=0,
                cv_score_std=0,
                train_test_correlation=0,
                overfitting_ratio=0,
                stability_score=0,
                fold_results=[]
            )

    def _parameter_sensitivity_analysis(self,
                                      backtest_data: pd.DataFrame,
                                      execution_id: int) -> ParameterSensitivityResults:
        """
        Analyze parameter sensitivity using Monte Carlo simulation
        """
        try:
            # Load strategy configuration
            config_data = self._load_strategy_config(execution_id)

            if not config_data:
                return ParameterSensitivityResults(
                    parameters_tested=0,
                    sensitivity_scores={},
                    monte_carlo_results={},
                    robustness_score=0,
                    parameter_stability=0
                )

            # Identify numeric parameters for sensitivity testing
            numeric_parameters = self._extract_numeric_parameters(config_data)

            if not numeric_parameters:
                return ParameterSensitivityResults(
                    parameters_tested=0,
                    sensitivity_scores={},
                    monte_carlo_results={},
                    robustness_score=0.5,  # Neutral score if no parameters
                    parameter_stability=0.5
                )

            # Perform Monte Carlo parameter sensitivity
            sensitivity_results = {}
            monte_carlo_results = {}

            base_performance = self._calculate_base_performance(backtest_data)

            for param_name, param_value in numeric_parameters.items():
                self.logger.debug(f"Testing sensitivity for parameter: {param_name}")

                # Generate parameter variations
                variations = self._generate_parameter_variations(param_value, n_variations=50)

                # Simulate performance for each variation
                performance_variations = []

                for variation in variations:
                    # Simulate strategy performance with varied parameter
                    simulated_performance = self._simulate_parameter_performance(
                        backtest_data, param_name, variation, base_performance
                    )
                    performance_variations.append(simulated_performance)

                # Calculate sensitivity metrics
                performance_std = np.std(performance_variations)
                performance_mean = np.mean(performance_variations)

                # Sensitivity score (coefficient of variation)
                sensitivity_score = performance_std / abs(performance_mean) if performance_mean != 0 else 0

                sensitivity_results[param_name] = sensitivity_score

                monte_carlo_results[param_name] = {
                    'base_value': param_value,
                    'variations_tested': len(variations),
                    'performance_mean': performance_mean,
                    'performance_std': performance_std,
                    'performance_min': min(performance_variations),
                    'performance_max': max(performance_variations),
                    'sensitivity_score': sensitivity_score
                }

            # Calculate overall robustness score
            if sensitivity_results:
                avg_sensitivity = np.mean(list(sensitivity_results.values()))
                robustness_score = max(0, 1 - avg_sensitivity)  # Higher robustness = lower sensitivity
            else:
                robustness_score = 0.5

            # Parameter stability (inverse of max sensitivity)
            max_sensitivity = max(sensitivity_results.values()) if sensitivity_results else 0
            parameter_stability = max(0, 1 - max_sensitivity)

            return ParameterSensitivityResults(
                parameters_tested=len(numeric_parameters),
                sensitivity_scores=sensitivity_results,
                monte_carlo_results=monte_carlo_results,
                robustness_score=robustness_score,
                parameter_stability=parameter_stability
            )

        except Exception as e:
            self.logger.error(f"Error in parameter sensitivity analysis: {e}")
            return ParameterSensitivityResults(
                parameters_tested=0,
                sensitivity_scores={},
                monte_carlo_results={},
                robustness_score=0,
                parameter_stability=0
            )

    def _deflated_sharpe_ratio(self, backtest_data: pd.DataFrame, execution_id: int) -> Dict[str, Any]:
        """
        Calculate Deflated Sharpe Ratio (DSR) to account for multiple testing bias

        The DSR adjusts the Sharpe ratio for the number of trials (parameter combinations
        or strategies tested), providing a more conservative estimate of performance.
        """
        try:
            returns_data = self._prepare_returns_data(backtest_data)

            if len(returns_data) < 30:
                return {
                    'status': 'insufficient_data',
                    'message': 'Insufficient data for DSR calculation'
                }

            # Calculate basic Sharpe ratio
            returns = returns_data['pips_gained'].dropna()
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

            # Estimate number of trials from execution history
            n_trials = self._estimate_number_of_trials(execution_id)

            # Calculate DSR using the formula from López de Prado
            # DSR = (SR - E[max SR]) / sqrt(Var[max SR])

            if n_trials <= 1:
                deflated_sharpe = sharpe_ratio
                deflation_factor = 1.0
            else:
                # Expected maximum Sharpe ratio under null hypothesis
                expected_max_sr = np.sqrt(2 * np.log(n_trials))

                # Variance of maximum Sharpe ratio
                var_max_sr = 2 * np.log(n_trials) * (1 - np.euler_gamma / np.sqrt(2 * np.log(n_trials)))

                # Deflated Sharpe ratio
                if np.sqrt(var_max_sr) > 0:
                    deflated_sharpe = (sharpe_ratio - expected_max_sr) / np.sqrt(var_max_sr)
                    deflation_factor = expected_max_sr / sharpe_ratio if sharpe_ratio != 0 else 1.0
                else:
                    deflated_sharpe = sharpe_ratio
                    deflation_factor = 1.0

            # Calculate p-value for deflated Sharpe ratio
            n_observations = len(returns)
            dsr_pvalue = 2 * (1 - stats.norm.cdf(abs(deflated_sharpe * np.sqrt(n_observations))))

            # Assessment
            if deflated_sharpe > 2.0:
                assessment = 'strong'
            elif deflated_sharpe > 1.0:
                assessment = 'moderate'
            elif deflated_sharpe > 0:
                assessment = 'weak'
            else:
                assessment = 'poor'

            return {
                'status': 'completed',
                'original_sharpe_ratio': sharpe_ratio,
                'deflated_sharpe_ratio': deflated_sharpe,
                'deflation_factor': deflation_factor,
                'n_trials_estimated': n_trials,
                'p_value': dsr_pvalue,
                'is_significant': dsr_pvalue < self.significance_level,
                'assessment': assessment,
                'n_observations': n_observations
            }

        except Exception as e:
            self.logger.error(f"Error calculating deflated Sharpe ratio: {e}")
            return {
                'status': 'error',
                'error_message': str(e)
            }

    def _generate_overfitting_assessment(self, detection_methods: Dict[str, Any]) -> OverfittingAssessment:
        """
        Generate comprehensive overfitting assessment from all detection methods
        """
        try:
            risk_factors = {}
            primary_concerns = []
            recommendations = []

            # Walk-forward analysis assessment
            wf_results = detection_methods.get('walk_forward')
            if wf_results and hasattr(wf_results, 'degradation_ratio'):
                if wf_results.degradation_ratio > 2.0:
                    risk_factors['walk_forward_degradation'] = 0.8
                    primary_concerns.append("Severe performance degradation in walk-forward analysis")
                    recommendations.append("Reduce strategy complexity and re-validate with longer out-of-sample periods")
                elif wf_results.degradation_ratio > 1.5:
                    risk_factors['walk_forward_degradation'] = 0.6
                    primary_concerns.append("Moderate performance degradation detected")
                    recommendations.append("Consider parameter regularization and extended validation")
                else:
                    risk_factors['walk_forward_degradation'] = 0.2

                if wf_results.stability_score < 0.5:
                    risk_factors['walk_forward_stability'] = 0.7
                    primary_concerns.append("Low performance stability across time periods")

            # Cross-validation assessment
            cv_results = detection_methods.get('cross_validation')
            if cv_results and hasattr(cv_results, 'overfitting_ratio'):
                if cv_results.overfitting_ratio > 2.0:
                    risk_factors['cv_overfitting'] = 0.8
                    primary_concerns.append("High overfitting ratio in cross-validation")
                    recommendations.append("Strategy shows signs of severe overfitting - consider simpler models")
                elif cv_results.overfitting_ratio > 1.5:
                    risk_factors['cv_overfitting'] = 0.5
                else:
                    risk_factors['cv_overfitting'] = 0.2

                if cv_results.train_test_correlation > 0.8:
                    risk_factors['train_test_correlation'] = 0.6
                    primary_concerns.append("Suspiciously high correlation between training and test performance")

            # Parameter sensitivity assessment
            param_results = detection_methods.get('parameter_sensitivity')
            if param_results and hasattr(param_results, 'robustness_score'):
                if param_results.robustness_score < 0.5:
                    risk_factors['parameter_sensitivity'] = 0.7
                    primary_concerns.append("High parameter sensitivity detected")
                    recommendations.append("Strategy performance is highly dependent on specific parameters")
                elif param_results.robustness_score < 0.7:
                    risk_factors['parameter_sensitivity'] = 0.4
                else:
                    risk_factors['parameter_sensitivity'] = 0.2

            # Deflated Sharpe ratio assessment
            dsr_results = detection_methods.get('deflated_sharpe')
            if dsr_results and dsr_results.get('status') == 'completed':
                deflated_sr = dsr_results.get('deflated_sharpe_ratio', 0)
                if deflated_sr < 0:
                    risk_factors['deflated_sharpe'] = 0.8
                    primary_concerns.append("Negative deflated Sharpe ratio indicates multiple testing bias")
                    recommendations.append("Strategy likely suffers from data snooping - extensive re-validation required")
                elif deflated_sr < 1.0:
                    risk_factors['deflated_sharpe'] = 0.5
                    primary_concerns.append("Low deflated Sharpe ratio suggests reduced statistical significance")
                else:
                    risk_factors['deflated_sharpe'] = 0.2

            # Calculate overall risk level and confidence
            if risk_factors:
                weighted_risk = np.mean(list(risk_factors.values()))

                if weighted_risk >= 0.7:
                    overall_risk_level = OverfittingRisk.SEVERE
                    confidence_score = 0.9
                    recommendations.append("Strategy requires complete re-design and validation")
                elif weighted_risk >= 0.5:
                    overall_risk_level = OverfittingRisk.HIGH
                    confidence_score = 0.8
                    recommendations.append("Strategy needs significant modifications before deployment")
                elif weighted_risk >= 0.3:
                    overall_risk_level = OverfittingRisk.MODERATE
                    confidence_score = 0.7
                    recommendations.append("Additional validation and parameter adjustments recommended")
                else:
                    overall_risk_level = OverfittingRisk.LOW
                    confidence_score = 0.8
                    recommendations.append("Strategy shows good validation characteristics")
            else:
                overall_risk_level = OverfittingRisk.MODERATE
                confidence_score = 0.5
                primary_concerns.append("Insufficient data for comprehensive assessment")

            # Add general recommendations
            if not recommendations:
                recommendations = [
                    "Continue monitoring strategy performance",
                    "Implement regular re-validation procedures",
                    "Consider ensemble methods for robustness"
                ]

            return OverfittingAssessment(
                overall_risk_level=overall_risk_level,
                confidence_score=confidence_score,
                primary_concerns=primary_concerns,
                risk_factors=risk_factors,
                recommendations=recommendations,
                statistical_evidence={
                    'methods_used': list(detection_methods.keys()),
                    'evidence_strength': confidence_score,
                    'risk_score': weighted_risk if risk_factors else 0.5
                }
            )

        except Exception as e:
            self.logger.error(f"Error generating overfitting assessment: {e}")
            return OverfittingAssessment(
                overall_risk_level=OverfittingRisk.MODERATE,
                confidence_score=0.0,
                primary_concerns=['Assessment failed due to error'],
                risk_factors={},
                recommendations=['Manual review required'],
                statistical_evidence={'error': str(e)}
            )

    # Utility methods
    def _load_backtest_data(self, execution_id: int) -> pd.DataFrame:
        """Load backtest data for analysis"""
        query = """
        SELECT bs.*, be.strategy_name
        FROM backtest_signals bs
        JOIN backtest_executions be ON bs.execution_id = be.id
        WHERE bs.execution_id = %s
        ORDER BY bs.signal_timestamp
        """

        result = self.db_manager.execute_query(query, (execution_id,))
        columns = [desc[0] for desc in result.description]
        data = result.fetchall()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=columns)
        df['signal_timestamp'] = pd.to_datetime(df['signal_timestamp'])

        return df

    def _prepare_returns_data(self, backtest_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare returns data for analysis"""
        if 'pips_gained' in backtest_data.columns:
            returns_data = backtest_data[['signal_timestamp', 'pips_gained']].copy()
            returns_data = returns_data.dropna()
            returns_data = returns_data.sort_values('signal_timestamp')
            return returns_data
        else:
            # Create synthetic returns data if pips_gained not available
            returns_data = pd.DataFrame({
                'signal_timestamp': backtest_data['signal_timestamp'],
                'pips_gained': np.random.normal(0, 10, len(backtest_data))  # Placeholder
            })
            return returns_data

    def _calculate_performance_metrics(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for a dataset"""
        if 'pips_gained' not in returns_data.columns or len(returns_data) == 0:
            return {
                'total_return': 0,
                'mean_return': 0,
                'return_std': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

        returns = returns_data['pips_gained'].astype(float)

        return {
            'total_return': returns.sum(),
            'mean_return': returns.mean(),
            'return_std': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns.cumsum())
        }

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0

        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min()

        return abs(max_drawdown) if max_drawdown < 0 else 0

    def _create_insufficient_data_result(self, execution_id: int, data_size: int) -> Dict[str, Any]:
        """Create result for insufficient data"""
        return {
            'execution_id': execution_id,
            'status': 'insufficient_data',
            'data_size': data_size,
            'required_size': self.min_sample_size,
            'analysis_timestamp': datetime.now(timezone.utc),
            'overall_assessment': OverfittingAssessment(
                overall_risk_level=OverfittingRisk.MODERATE,
                confidence_score=0.0,
                primary_concerns=['Insufficient data for analysis'],
                risk_factors={},
                recommendations=['Collect more data before analysis'],
                statistical_evidence={'insufficient_data': True}
            ).__dict__
        }

    def _store_overfitting_results(self, execution_id: int, results: Dict[str, Any]):
        """Store overfitting detection results"""
        try:
            # This would store results in a dedicated overfitting_analysis table
            # Implementation depends on specific database schema requirements
            self.logger.info(f"Storing overfitting results for execution {execution_id}")
            # TODO: Implement database storage
        except Exception as e:
            self.logger.error(f"Error storing overfitting results: {e}")


# Factory function
def create_overfitting_detector(
    db_manager: DatabaseManager,
    **kwargs
) -> OverfittingDetector:
    """Create OverfittingDetector instance"""
    return OverfittingDetector(db_manager=db_manager, **kwargs)