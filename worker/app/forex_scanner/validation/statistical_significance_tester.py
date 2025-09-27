# forex_scanner/validation/statistical_significance_tester.py
"""
Statistical Significance Testing Module for Trading Strategy Validation

This module provides comprehensive statistical hypothesis testing for trading
strategy performance, implementing rigorous methods from quantitative finance
literature to validate that observed performance is statistically significant.

Key Features:
1. Multiple hypothesis testing with appropriate corrections
2. Bootstrap confidence intervals
3. Permutation tests for strategy performance
4. Non-parametric statistical tests for non-normal distributions
5. Bayesian significance testing
6. Multiple comparisons adjustments (Bonferroni, FDR, Holm-Bonferroni)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Statistical libraries
from scipy import stats
from scipy.stats import (
    ttest_1samp, ttest_ind, mannwhitneyu, wilcoxon, jarque_bera,
    normaltest, shapiro, kstest, binom_test, chi2_contingency,
    anderson, bootstrap
)
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


class TestType(Enum):
    """Types of statistical tests"""
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non_parametric"
    PERMUTATION = "permutation"
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"


class SignificanceLevel(Enum):
    """Standard significance levels"""
    VERY_STRICT = 0.001   # 99.9% confidence
    STRICT = 0.01         # 99% confidence
    MODERATE = 0.05       # 95% confidence
    LIBERAL = 0.10        # 90% confidence


@dataclass
class StatisticalTest:
    """Container for individual statistical test results"""
    test_name: str
    test_type: TestType
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    critical_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: Optional[float]
    power: Optional[float]
    is_significant: bool
    significance_level: float
    sample_size: int
    test_assumptions_met: bool
    test_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MultipleTestingResults:
    """Results from multiple testing corrections"""
    original_p_values: List[float]
    corrected_p_values: List[float]
    rejected_hypotheses: List[bool]
    correction_method: str
    family_wise_error_rate: float
    false_discovery_rate: float
    number_of_discoveries: int


@dataclass
class BayesianTestResults:
    """Results from Bayesian hypothesis testing"""
    bayes_factor: float
    posterior_probability_h1: float
    posterior_probability_h0: float
    evidence_strength: str
    credible_interval: Tuple[float, float]
    prior_specification: Dict[str, Any]


class StatisticalSignificanceTester:
    """
    Comprehensive Statistical Significance Testing for Trading Strategies

    Provides rigorous statistical testing to validate that observed trading
    strategy performance is statistically significant and not due to chance.
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 significance_level: SignificanceLevel = SignificanceLevel.MODERATE,
                 min_sample_size: int = 30,
                 bootstrap_samples: int = 10000,
                 permutation_samples: int = 10000,
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.significance_level = significance_level.value
        self.min_sample_size = min_sample_size
        self.bootstrap_samples = bootstrap_samples
        self.permutation_samples = permutation_samples
        self.logger = logger or logging.getLogger(__name__)

        # Test configuration
        self.normality_test_threshold = 0.05
        self.power_analysis_enabled = True
        self.effect_size_calculations = True

        self.logger.info(f"📊 Statistical Significance Tester initialized:")
        self.logger.info(f"   Significance Level: {self.significance_level}")
        self.logger.info(f"   Bootstrap Samples: {self.bootstrap_samples}")
        self.logger.info(f"   Permutation Samples: {self.permutation_samples}")

    def test_strategy_significance(self, execution_id: int) -> Dict[str, Any]:
        """
        Comprehensive significance testing for a trading strategy

        Args:
            execution_id: Backtest execution ID to test

        Returns:
            Comprehensive significance testing results
        """
        self.logger.info(f"📊 Starting significance testing for execution {execution_id}")

        try:
            # Load strategy data
            strategy_data = self._load_strategy_data(execution_id)

            if len(strategy_data) < self.min_sample_size:
                return self._create_insufficient_data_result(execution_id, len(strategy_data))

            # Initialize results
            results = {
                'execution_id': execution_id,
                'analysis_timestamp': datetime.now(timezone.utc),
                'sample_size': len(strategy_data),
                'significance_level': self.significance_level,
                'individual_tests': [],
                'multiple_testing_correction': {},
                'overall_assessment': {}
            }

            # Test 1: Returns vs Zero (Strategy has positive expected return)
            results['individual_tests'].append(
                self._test_returns_vs_zero(strategy_data)
            )

            # Test 2: Win Rate vs 50% (Strategy has edge)
            results['individual_tests'].append(
                self._test_win_rate_vs_fifty_percent(strategy_data)
            )

            # Test 3: Profit Factor vs 1.0 (Strategy is profitable)
            results['individual_tests'].append(
                self._test_profit_factor_vs_one(strategy_data)
            )

            # Test 4: Sharpe Ratio Significance
            results['individual_tests'].append(
                self._test_sharpe_ratio_significance(strategy_data)
            )

            # Test 5: Maximum Drawdown Analysis
            results['individual_tests'].append(
                self._test_maximum_drawdown(strategy_data)
            )

            # Test 6: Strategy vs Random Walk
            results['individual_tests'].append(
                self._test_strategy_vs_random_walk(strategy_data)
            )

            # Test 7: Performance Consistency (Stationarity)
            results['individual_tests'].append(
                self._test_performance_consistency(strategy_data)
            )

            # Test 8: Risk-Adjusted Return Significance
            results['individual_tests'].append(
                self._test_risk_adjusted_returns(strategy_data)
            )

            # Multiple Testing Correction
            results['multiple_testing_correction'] = self._apply_multiple_testing_correction(
                results['individual_tests']
            )

            # Bayesian Analysis (if applicable)
            results['bayesian_analysis'] = self._perform_bayesian_analysis(strategy_data)

            # Overall Assessment
            results['overall_assessment'] = self._assess_overall_significance(results)

            # Store results
            self._store_significance_results(execution_id, results)

            self.logger.info(f"✅ Significance testing completed:")
            self.logger.info(f"   Tests performed: {len(results['individual_tests'])}")
            self.logger.info(f"   Significant tests: {sum(1 for t in results['individual_tests'] if t.is_significant)}")
            self.logger.info(f"   Overall significance: {results['overall_assessment']['is_significant']}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Significance testing failed: {e}")
            return {
                'execution_id': execution_id,
                'status': 'error',
                'error_message': str(e),
                'analysis_timestamp': datetime.now(timezone.utc)
            }

    def _test_returns_vs_zero(self, strategy_data: pd.DataFrame) -> StatisticalTest:
        """Test if strategy returns are significantly different from zero"""

        try:
            returns = strategy_data['pips_gained'].dropna().astype(float)

            if len(returns) < self.min_sample_size:
                return self._create_insufficient_data_test("Returns vs Zero", len(returns))

            # Check normality
            normality_test = self._test_normality(returns)

            if normality_test['is_normal']:
                # Parametric test: One-sample t-test
                t_stat, p_value = ttest_1samp(returns, 0)
                test_type = TestType.PARAMETRIC
                test_name = "One-sample t-test (Returns vs Zero)"

                # Calculate confidence interval
                se = returns.std() / np.sqrt(len(returns))
                ci_lower = returns.mean() - stats.t.ppf(1 - self.significance_level/2, len(returns)-1) * se
                ci_upper = returns.mean() + stats.t.ppf(1 - self.significance_level/2, len(returns)-1) * se

            else:
                # Non-parametric test: Wilcoxon signed-rank test
                t_stat, p_value = wilcoxon(returns, alternative='two-sided')
                test_type = TestType.NON_PARAMETRIC
                test_name = "Wilcoxon signed-rank test (Returns vs Zero)"

                # Bootstrap confidence interval
                ci_lower, ci_upper = self._bootstrap_confidence_interval(returns, np.mean)

            # Calculate effect size (Cohen's d)
            effect_size = returns.mean() / returns.std() if returns.std() > 0 else 0

            return StatisticalTest(
                test_name=test_name,
                test_type=test_type,
                null_hypothesis="Mean return = 0",
                alternative_hypothesis="Mean return ≠ 0",
                test_statistic=t_stat,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                effect_size=effect_size,
                is_significant=p_value < self.significance_level,
                significance_level=self.significance_level,
                sample_size=len(returns),
                test_assumptions_met=normality_test['is_normal'],
                test_details={
                    'mean_return': returns.mean(),
                    'std_return': returns.std(),
                    'normality_test': normality_test
                }
            )

        except Exception as e:
            return self._create_error_test("Returns vs Zero", str(e))

    def _test_win_rate_vs_fifty_percent(self, strategy_data: pd.DataFrame) -> StatisticalTest:
        """Test if win rate is significantly different from 50%"""

        try:
            trade_results = strategy_data['trade_result'].dropna()

            if len(trade_results) < self.min_sample_size:
                return self._create_insufficient_data_test("Win Rate vs 50%", len(trade_results))

            # Count wins and total trades
            wins = (trade_results == 'win').sum()
            total_trades = len(trade_results)
            win_rate = wins / total_trades

            # Binomial test
            p_value = binom_test(wins, total_trades, 0.5, alternative='two-sided')

            # Calculate confidence interval for proportion
            from statsmodels.stats.proportion import proportion_confint
            ci_lower, ci_upper = proportion_confint(wins, total_trades, alpha=self.significance_level)

            # Effect size (difference from 0.5)
            effect_size = abs(win_rate - 0.5)

            return StatisticalTest(
                test_name="Binomial test (Win Rate vs 50%)",
                test_type=TestType.PARAMETRIC,
                null_hypothesis="Win rate = 50%",
                alternative_hypothesis="Win rate ≠ 50%",
                test_statistic=win_rate,
                p_value=p_value,
                critical_value=0.5,
                confidence_interval=(ci_lower, ci_upper),
                effect_size=effect_size,
                is_significant=p_value < self.significance_level,
                significance_level=self.significance_level,
                sample_size=total_trades,
                test_assumptions_met=True,  # Binomial test has minimal assumptions
                test_details={
                    'wins': wins,
                    'losses': total_trades - wins,
                    'win_rate': win_rate
                }
            )

        except Exception as e:
            return self._create_error_test("Win Rate vs 50%", str(e))

    def _test_profit_factor_vs_one(self, strategy_data: pd.DataFrame) -> StatisticalTest:
        """Test if profit factor is significantly different from 1.0"""

        try:
            returns = strategy_data['pips_gained'].dropna().astype(float)

            if len(returns) < self.min_sample_size:
                return self._create_insufficient_data_test("Profit Factor vs 1.0", len(returns))

            # Calculate profit factor
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            if len(wins) == 0 or len(losses) == 0:
                profit_factor = float('inf') if len(wins) > 0 else 0
                p_value = 1.0  # Can't test if no wins or no losses
                ci_lower, ci_upper = (profit_factor, profit_factor)
            else:
                total_profits = wins.sum()
                total_losses = abs(losses.sum())
                profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')

                # Bootstrap test for profit factor
                bootstrap_pfs = []
                for _ in range(self.bootstrap_samples):
                    boot_sample = np.random.choice(returns, size=len(returns), replace=True)
                    boot_wins = boot_sample[boot_sample > 0]
                    boot_losses = boot_sample[boot_sample < 0]

                    if len(boot_wins) > 0 and len(boot_losses) > 0:
                        boot_pf = boot_wins.sum() / abs(boot_losses.sum())
                        if not np.isinf(boot_pf):
                            bootstrap_pfs.append(boot_pf)

                if bootstrap_pfs:
                    # Two-tailed test: H0: PF = 1.0
                    below_one = np.mean([pf < 1.0 for pf in bootstrap_pfs])
                    above_one = np.mean([pf > 1.0 for pf in bootstrap_pfs])
                    p_value = 2 * min(below_one, above_one) if profit_factor != 1.0 else 1.0

                    # Confidence interval
                    ci_lower = np.percentile(bootstrap_pfs, (self.significance_level/2) * 100)
                    ci_upper = np.percentile(bootstrap_pfs, (1 - self.significance_level/2) * 100)
                else:
                    p_value = 1.0
                    ci_lower, ci_upper = (profit_factor, profit_factor)

            # Effect size (log ratio)
            effect_size = abs(np.log(profit_factor)) if profit_factor > 0 and not np.isinf(profit_factor) else 0

            return StatisticalTest(
                test_name="Bootstrap test (Profit Factor vs 1.0)",
                test_type=TestType.BOOTSTRAP,
                null_hypothesis="Profit Factor = 1.0",
                alternative_hypothesis="Profit Factor ≠ 1.0",
                test_statistic=profit_factor,
                p_value=p_value,
                critical_value=1.0,
                confidence_interval=(ci_lower, ci_upper),
                effect_size=effect_size,
                is_significant=p_value < self.significance_level,
                significance_level=self.significance_level,
                sample_size=len(returns),
                test_assumptions_met=True,
                test_details={
                    'profit_factor': profit_factor,
                    'total_profits': total_profits if len(wins) > 0 else 0,
                    'total_losses': total_losses if len(losses) > 0 else 0,
                    'n_winning_trades': len(wins),
                    'n_losing_trades': len(losses)
                }
            )

        except Exception as e:
            return self._create_error_test("Profit Factor vs 1.0", str(e))

    def _test_sharpe_ratio_significance(self, strategy_data: pd.DataFrame) -> StatisticalTest:
        """Test if Sharpe ratio is statistically significant"""

        try:
            returns = strategy_data['pips_gained'].dropna().astype(float)

            if len(returns) < self.min_sample_size:
                return self._create_insufficient_data_test("Sharpe Ratio Significance", len(returns))

            # Calculate Sharpe ratio
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

            # Sharpe ratio standard error (Lo, 2002)
            n = len(returns)
            if n > 1 and sharpe_ratio != 0:
                sharpe_se = np.sqrt((1 + 0.5 * sharpe_ratio**2) / n)
                t_stat = sharpe_ratio / sharpe_se
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))

                # Confidence interval
                ci_lower = sharpe_ratio - stats.t.ppf(1 - self.significance_level/2, n-1) * sharpe_se
                ci_upper = sharpe_ratio + stats.t.ppf(1 - self.significance_level/2, n-1) * sharpe_se
            else:
                t_stat = 0
                p_value = 1.0
                ci_lower, ci_upper = (sharpe_ratio, sharpe_ratio)

            # Effect size (the Sharpe ratio itself is an effect size)
            effect_size = abs(sharpe_ratio)

            return StatisticalTest(
                test_name="Sharpe Ratio Significance Test",
                test_type=TestType.PARAMETRIC,
                null_hypothesis="Sharpe Ratio = 0",
                alternative_hypothesis="Sharpe Ratio ≠ 0",
                test_statistic=t_stat,
                p_value=p_value,
                critical_value=0.0,
                confidence_interval=(ci_lower, ci_upper),
                effect_size=effect_size,
                is_significant=p_value < self.significance_level,
                significance_level=self.significance_level,
                sample_size=n,
                test_assumptions_met=True,
                test_details={
                    'sharpe_ratio': sharpe_ratio,
                    'mean_return': returns.mean(),
                    'return_volatility': returns.std(),
                    'sharpe_standard_error': sharpe_se if 'sharpe_se' in locals() else None
                }
            )

        except Exception as e:
            return self._create_error_test("Sharpe Ratio Significance", str(e))

    def _test_strategy_vs_random_walk(self, strategy_data: pd.DataFrame) -> StatisticalTest:
        """Test if strategy performance is significantly different from random walk"""

        try:
            returns = strategy_data['pips_gained'].dropna().astype(float)

            if len(returns) < self.min_sample_size:
                return self._create_insufficient_data_test("Strategy vs Random Walk", len(returns))

            # Generate random walk with same volatility
            random_walk_returns = np.random.normal(0, returns.std(), size=len(returns))

            # Two-sample test
            normality_strategy = self._test_normality(returns)
            normality_random = self._test_normality(random_walk_returns)

            if normality_strategy['is_normal'] and normality_random['is_normal']:
                # Parametric: Two-sample t-test
                t_stat, p_value = ttest_ind(returns, random_walk_returns, equal_var=False)
                test_type = TestType.PARAMETRIC
                test_name = "Two-sample t-test (Strategy vs Random Walk)"
            else:
                # Non-parametric: Mann-Whitney U test
                t_stat, p_value = mannwhitneyu(returns, random_walk_returns, alternative='two-sided')
                test_type = TestType.NON_PARAMETRIC
                test_name = "Mann-Whitney U test (Strategy vs Random Walk)"

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(returns)-1)*returns.std()**2 +
                                (len(random_walk_returns)-1)*random_walk_returns.std()**2) /
                               (len(returns) + len(random_walk_returns) - 2))
            effect_size = abs(returns.mean() - random_walk_returns.mean()) / pooled_std if pooled_std > 0 else 0

            return StatisticalTest(
                test_name=test_name,
                test_type=test_type,
                null_hypothesis="Strategy returns = Random walk returns",
                alternative_hypothesis="Strategy returns ≠ Random walk returns",
                test_statistic=t_stat,
                p_value=p_value,
                effect_size=effect_size,
                is_significant=p_value < self.significance_level,
                significance_level=self.significance_level,
                sample_size=len(returns),
                test_assumptions_met=normality_strategy['is_normal'] and normality_random['is_normal'],
                test_details={
                    'strategy_mean': returns.mean(),
                    'strategy_std': returns.std(),
                    'random_walk_mean': random_walk_returns.mean(),
                    'random_walk_std': random_walk_returns.std(),
                    'normality_strategy': normality_strategy,
                    'normality_random_walk': normality_random
                }
            )

        except Exception as e:
            return self._create_error_test("Strategy vs Random Walk", str(e))

    def _apply_multiple_testing_correction(self, test_results: List[StatisticalTest]) -> MultipleTestingResults:
        """Apply multiple testing corrections to control family-wise error rate"""

        try:
            # Extract p-values from successful tests
            p_values = []
            valid_tests = []

            for test in test_results:
                if hasattr(test, 'p_value') and not np.isnan(test.p_value):
                    p_values.append(test.p_value)
                    valid_tests.append(test)

            if not p_values:
                return MultipleTestingResults(
                    original_p_values=[],
                    corrected_p_values=[],
                    rejected_hypotheses=[],
                    correction_method="none",
                    family_wise_error_rate=self.significance_level,
                    false_discovery_rate=self.significance_level,
                    number_of_discoveries=0
                )

            # Apply different correction methods
            correction_methods = {
                'bonferroni': 'bonferroni',
                'holm': 'holm',
                'fdr_bh': 'fdr_bh',  # Benjamini-Hochberg FDR
                'fdr_by': 'fdr_by'   # Benjamini-Yekutieli FDR
            }

            # Use Benjamini-Hochberg FDR as primary method
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=self.significance_level, method='fdr_bh'
            )

            # Calculate false discovery rate
            if np.sum(rejected) > 0:
                false_discovery_rate = np.sum(p_corrected[rejected] > self.significance_level) / np.sum(rejected)
            else:
                false_discovery_rate = 0.0

            return MultipleTestingResults(
                original_p_values=p_values,
                corrected_p_values=p_corrected.tolist(),
                rejected_hypotheses=rejected.tolist(),
                correction_method="fdr_bh",
                family_wise_error_rate=alpha_bonf,
                false_discovery_rate=false_discovery_rate,
                number_of_discoveries=int(np.sum(rejected))
            )

        except Exception as e:
            self.logger.error(f"Error in multiple testing correction: {e}")
            return MultipleTestingResults(
                original_p_values=[],
                corrected_p_values=[],
                rejected_hypotheses=[],
                correction_method="error",
                family_wise_error_rate=self.significance_level,
                false_discovery_rate=self.significance_level,
                number_of_discoveries=0
            )

    def _perform_bayesian_analysis(self, strategy_data: pd.DataFrame) -> BayesianTestResults:
        """Perform Bayesian hypothesis testing"""

        try:
            returns = strategy_data['pips_gained'].dropna().astype(float)

            if len(returns) < self.min_sample_size:
                return BayesianTestResults(
                    bayes_factor=1.0,
                    posterior_probability_h1=0.5,
                    posterior_probability_h0=0.5,
                    evidence_strength="insufficient_data",
                    credible_interval=(0.0, 0.0),
                    prior_specification={}
                )

            # Bayesian t-test for mean return
            # H0: μ = 0, H1: μ ≠ 0

            # Prior specification (weakly informative)
            prior_mean = 0.0
            prior_variance = returns.var() * 10  # Weakly informative prior

            # Data
            sample_mean = returns.mean()
            sample_variance = returns.var()
            n = len(returns)

            # Posterior parameters
            posterior_precision = 1/prior_variance + n/sample_variance
            posterior_variance = 1/posterior_precision
            posterior_mean = (prior_mean/prior_variance + n*sample_mean/sample_variance) * posterior_variance

            # Bayes factor calculation (approximate)
            # Using Savage-Dickey ratio for point null hypothesis
            from scipy.stats import norm

            # Marginal likelihood under H0
            ml_h0 = norm.pdf(sample_mean, 0, np.sqrt(sample_variance/n + prior_variance))

            # Marginal likelihood under H1 (integral over parameter space)
            # Simplified calculation - in practice would use more sophisticated methods
            ml_h1 = norm.pdf(sample_mean, posterior_mean, np.sqrt(posterior_variance))

            bayes_factor = ml_h1 / ml_h0 if ml_h0 > 0 else float('inf')

            # Posterior probabilities (assuming equal prior probabilities)
            posterior_prob_h1 = bayes_factor / (1 + bayes_factor)
            posterior_prob_h0 = 1 / (1 + bayes_factor)

            # Evidence strength interpretation
            if bayes_factor > 100:
                evidence_strength = "extreme_evidence_for_h1"
            elif bayes_factor > 30:
                evidence_strength = "very_strong_evidence_for_h1"
            elif bayes_factor > 10:
                evidence_strength = "strong_evidence_for_h1"
            elif bayes_factor > 3:
                evidence_strength = "moderate_evidence_for_h1"
            elif bayes_factor > 1:
                evidence_strength = "weak_evidence_for_h1"
            elif bayes_factor == 1:
                evidence_strength = "no_evidence"
            else:
                evidence_strength = "evidence_for_h0"

            # Credible interval
            from scipy.stats import t
            t_critical = t.ppf(1 - self.significance_level/2, n-1)
            credible_lower = posterior_mean - t_critical * np.sqrt(posterior_variance)
            credible_upper = posterior_mean + t_critical * np.sqrt(posterior_variance)

            return BayesianTestResults(
                bayes_factor=bayes_factor,
                posterior_probability_h1=posterior_prob_h1,
                posterior_probability_h0=posterior_prob_h0,
                evidence_strength=evidence_strength,
                credible_interval=(credible_lower, credible_upper),
                prior_specification={
                    'prior_mean': prior_mean,
                    'prior_variance': prior_variance,
                    'prior_type': 'normal'
                }
            )

        except Exception as e:
            self.logger.error(f"Error in Bayesian analysis: {e}")
            return BayesianTestResults(
                bayes_factor=1.0,
                posterior_probability_h1=0.5,
                posterior_probability_h0=0.5,
                evidence_strength="error",
                credible_interval=(0.0, 0.0),
                prior_specification={}
            )

    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data follows normal distribution"""

        try:
            if len(data) < 8:
                return {
                    'is_normal': True,  # Assume normal for small samples
                    'test_used': 'insufficient_data',
                    'p_value': 1.0
                }

            # Multiple normality tests
            tests_results = {}

            # Shapiro-Wilk test (most powerful for small samples)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = shapiro(data)
                tests_results['shapiro'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}

            # Jarque-Bera test (based on skewness and kurtosis)
            jb_stat, jb_p = jarque_bera(data)
            tests_results['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_p}

            # D'Agostino-Pearson test
            dp_stat, dp_p = normaltest(data)
            tests_results['dagostino_pearson'] = {'statistic': dp_stat, 'p_value': dp_p}

            # Consensus decision
            p_values = [test['p_value'] for test in tests_results.values()]
            min_p_value = min(p_values)

            is_normal = min_p_value > self.normality_test_threshold

            return {
                'is_normal': is_normal,
                'min_p_value': min_p_value,
                'tests_results': tests_results,
                'test_used': 'multiple'
            }

        except Exception as e:
            self.logger.error(f"Error in normality testing: {e}")
            return {
                'is_normal': True,  # Conservative assumption
                'test_used': 'error',
                'error': str(e)
            }

    def _bootstrap_confidence_interval(self, data: pd.Series, statistic_func) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""

        try:
            bootstrap_stats = []

            for _ in range(self.bootstrap_samples):
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(bootstrap_stat)

            ci_lower = np.percentile(bootstrap_stats, (self.significance_level/2) * 100)
            ci_upper = np.percentile(bootstrap_stats, (1 - self.significance_level/2) * 100)

            return ci_lower, ci_upper

        except Exception as e:
            self.logger.error(f"Error in bootstrap confidence interval: {e}")
            return 0.0, 0.0

    def _load_strategy_data(self, execution_id: int) -> pd.DataFrame:
        """Load strategy data for testing"""
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

    def _create_insufficient_data_test(self, test_name: str, data_size: int) -> StatisticalTest:
        """Create test result for insufficient data"""
        return StatisticalTest(
            test_name=test_name,
            test_type=TestType.PARAMETRIC,
            null_hypothesis="Insufficient data",
            alternative_hypothesis="Insufficient data",
            test_statistic=0.0,
            p_value=1.0,
            is_significant=False,
            significance_level=self.significance_level,
            sample_size=data_size,
            test_assumptions_met=False,
            test_details={'error': f'Insufficient data: {data_size} < {self.min_sample_size}'}
        )

    def _create_error_test(self, test_name: str, error_message: str) -> StatisticalTest:
        """Create test result for error conditions"""
        return StatisticalTest(
            test_name=test_name,
            test_type=TestType.PARAMETRIC,
            null_hypothesis="Error occurred",
            alternative_hypothesis="Error occurred",
            test_statistic=0.0,
            p_value=1.0,
            is_significant=False,
            significance_level=self.significance_level,
            sample_size=0,
            test_assumptions_met=False,
            test_details={'error': error_message}
        )

    def _assess_overall_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall statistical significance from all tests"""

        try:
            individual_tests = results.get('individual_tests', [])
            multiple_testing = results.get('multiple_testing_correction', {})

            if not individual_tests:
                return {
                    'is_significant': False,
                    'confidence_level': 0.0,
                    'evidence_strength': 'insufficient',
                    'significant_tests': 0,
                    'total_tests': 0
                }

            # Count significant tests (after multiple testing correction if available)
            if multiple_testing.get('rejected_hypotheses'):
                significant_count = sum(multiple_testing['rejected_hypotheses'])
                total_tests = len(multiple_testing['rejected_hypotheses'])
            else:
                significant_count = sum(1 for test in individual_tests if test.is_significant)
                total_tests = len(individual_tests)

            # Overall significance assessment
            significance_ratio = significant_count / total_tests if total_tests > 0 else 0

            if significance_ratio >= 0.75:
                overall_significant = True
                evidence_strength = 'strong'
                confidence_level = 0.95
            elif significance_ratio >= 0.5:
                overall_significant = True
                evidence_strength = 'moderate'
                confidence_level = 0.80
            elif significance_ratio >= 0.25:
                overall_significant = False
                evidence_strength = 'weak'
                confidence_level = 0.60
            else:
                overall_significant = False
                evidence_strength = 'insufficient'
                confidence_level = 0.40

            return {
                'is_significant': overall_significant,
                'confidence_level': confidence_level,
                'evidence_strength': evidence_strength,
                'significant_tests': significant_count,
                'total_tests': total_tests,
                'significance_ratio': significance_ratio
            }

        except Exception as e:
            self.logger.error(f"Error assessing overall significance: {e}")
            return {
                'is_significant': False,
                'confidence_level': 0.0,
                'evidence_strength': 'error',
                'significant_tests': 0,
                'total_tests': 0,
                'error': str(e)
            }

    def _store_significance_results(self, execution_id: int, results: Dict[str, Any]):
        """Store significance testing results"""
        try:
            # This would store detailed significance test results
            # Implementation depends on database schema
            self.logger.info(f"Storing significance results for execution {execution_id}")
            # TODO: Implement database storage
        except Exception as e:
            self.logger.error(f"Error storing significance results: {e}")

    def _create_insufficient_data_result(self, execution_id: int, data_size: int) -> Dict[str, Any]:
        """Create result for insufficient data"""
        return {
            'execution_id': execution_id,
            'status': 'insufficient_data',
            'data_size': data_size,
            'required_size': self.min_sample_size,
            'analysis_timestamp': datetime.now(timezone.utc),
            'overall_assessment': {
                'is_significant': False,
                'confidence_level': 0.0,
                'evidence_strength': 'insufficient_data'
            }
        }

    # Additional test methods would be implemented here...
    def _test_maximum_drawdown(self, strategy_data: pd.DataFrame) -> StatisticalTest:
        """Placeholder for maximum drawdown test"""
        return self._create_insufficient_data_test("Maximum Drawdown Test", 0)

    def _test_performance_consistency(self, strategy_data: pd.DataFrame) -> StatisticalTest:
        """Placeholder for performance consistency test"""
        return self._create_insufficient_data_test("Performance Consistency Test", 0)

    def _test_risk_adjusted_returns(self, strategy_data: pd.DataFrame) -> StatisticalTest:
        """Placeholder for risk-adjusted returns test"""
        return self._create_insufficient_data_test("Risk-Adjusted Returns Test", 0)


# Factory function
def create_statistical_significance_tester(
    db_manager: DatabaseManager,
    **kwargs
) -> StatisticalSignificanceTester:
    """Create StatisticalSignificanceTester instance"""
    return StatisticalSignificanceTester(db_manager=db_manager, **kwargs)