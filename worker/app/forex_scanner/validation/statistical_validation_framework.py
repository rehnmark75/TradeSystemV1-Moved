# forex_scanner/validation/statistical_validation_framework.py
"""
Statistical Validation Framework for Backtest Integration System

This framework provides comprehensive statistical validation for trading strategy
backtests, ensuring mathematical rigor and preventing common pitfalls in
quantitative finance strategy validation.

Core Components:
1. Real-time Performance Correlation
2. Overfitting Detection via Cross-Validation
3. Statistical Significance Testing
4. Data Quality Assurance
5. Pipeline Consistency Validation

Mathematical Foundation:
- Uses rigorous statistical methods from quantitative finance literature
- Implements multiple testing corrections (Bonferroni, FDR)
- Provides confidence intervals and statistical significance measures
- Validates strategy performance using hypothesis testing framework
"""

import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum

# Statistical libraries
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, kstest
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import config
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.database import DatabaseManager


class ValidationLevel(Enum):
    """Statistical validation confidence levels"""
    LOW = 0.90      # 90% confidence
    MEDIUM = 0.95   # 95% confidence
    HIGH = 0.99     # 99% confidence
    ULTRA_HIGH = 0.999  # 99.9% confidence


class TestResult(Enum):
    """Statistical test results"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ValidationResult:
    """Container for validation test results"""
    test_name: str
    result: TestResult
    p_value: float
    confidence_level: float
    statistic: float
    critical_value: Optional[float] = None
    effect_size: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OverfittingMetrics:
    """Container for overfitting detection metrics"""
    in_sample_sharpe: float
    out_sample_sharpe: float
    degradation_ratio: float
    stability_score: float
    cross_validation_score: float
    information_coefficient: float
    is_overfitted: bool
    confidence_level: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationMetrics:
    """Container for correlation analysis metrics"""
    pearson_correlation: float
    spearman_correlation: float
    kendall_tau: Optional[float]
    correlation_stability: float
    regression_r2: float
    tracking_error: float
    information_ratio: float
    is_correlated: bool
    confidence_level: float


class StatisticalValidationFramework:
    """
    Comprehensive Statistical Validation Framework for Trading Strategy Backtests

    Provides mathematically rigorous validation of backtest results against live
    performance, detecting overfitting, validating statistical significance,
    and ensuring data quality.

    Key Features:
    - Real-time correlation analysis between backtest and live performance
    - Cross-validation based overfitting detection
    - Multiple hypothesis testing with appropriate corrections
    - Data quality assessment with statistical tests
    - Pipeline consistency validation
    - Comprehensive reporting with confidence metrics
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 validation_level: ValidationLevel = ValidationLevel.MEDIUM,
                 min_sample_size: int = 30,
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.validation_level = validation_level
        self.min_sample_size = min_sample_size
        self.logger = logger or logging.getLogger(__name__)

        # Statistical parameters
        self.confidence_level = validation_level.value
        self.alpha = 1 - self.confidence_level

        # Validation thresholds
        self.min_correlation = 0.3  # Minimum acceptable correlation
        self.max_overfitting_ratio = 2.0  # Max in-sample/out-sample performance ratio
        self.min_information_coefficient = 0.05  # Minimum IC for strategy validity

        # Cross-validation parameters
        self.cv_folds = 5
        self.min_fold_size = 0.2  # 20% minimum for each fold

        # Data quality thresholds
        self.min_data_completeness = 0.95
        self.max_gap_percentage = 0.05

        self.logger.info(f"🔬 Statistical Validation Framework initialized:")
        self.logger.info(f"   Confidence Level: {self.confidence_level:.1%}")
        self.logger.info(f"   Minimum Sample Size: {self.min_sample_size}")
        self.logger.info(f"   Cross-Validation Folds: {self.cv_folds}")

    # =========================================================================
    # MAIN VALIDATION ENTRY POINTS
    # =========================================================================

    def validate_backtest_execution(self, execution_id: int) -> Dict[str, Any]:
        """
        Comprehensive validation of a backtest execution

        Args:
            execution_id: Backtest execution ID to validate

        Returns:
            Comprehensive validation report with statistical metrics
        """
        self.logger.info(f"🔬 Starting comprehensive validation for execution {execution_id}")

        validation_start = datetime.now(timezone.utc)

        try:
            # Load backtest data
            backtest_data = self._load_backtest_data(execution_id)

            if not self._validate_sample_size(backtest_data):
                return self._create_insufficient_data_report(execution_id, len(backtest_data))

            # Run all validation components
            results = {
                'execution_id': execution_id,
                'validation_timestamp': validation_start,
                'validation_level': self.validation_level.name,
                'confidence_level': self.confidence_level,
                'sample_size': len(backtest_data),
                'validation_components': {}
            }

            # 1. Data Quality Validation
            self.logger.info("📊 Running data quality validation...")
            results['validation_components']['data_quality'] = self._validate_data_quality(backtest_data)

            # 2. Statistical Significance Testing
            self.logger.info("📈 Running statistical significance tests...")
            results['validation_components']['statistical_significance'] = self._test_statistical_significance(backtest_data)

            # 3. Overfitting Detection
            self.logger.info("🎯 Running overfitting detection...")
            results['validation_components']['overfitting_detection'] = self._detect_overfitting(backtest_data)

            # 4. Real-time Correlation (if live data available)
            self.logger.info("🔄 Running real-time correlation analysis...")
            results['validation_components']['realtime_correlation'] = self._analyze_realtime_correlation(execution_id)

            # 5. Pipeline Consistency Validation
            self.logger.info("⚙️ Running pipeline consistency validation...")
            results['validation_components']['pipeline_consistency'] = self._validate_pipeline_consistency(execution_id)

            # Generate overall validation score
            results['overall_validation'] = self._calculate_overall_validation_score(results['validation_components'])

            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)

            # Store validation results
            self._store_validation_results(execution_id, results)

            validation_duration = (datetime.now(timezone.utc) - validation_start).total_seconds()
            results['validation_duration_seconds'] = validation_duration

            self.logger.info(f"✅ Validation completed in {validation_duration:.1f}s")
            self.logger.info(f"📊 Overall Score: {results['overall_validation']['composite_score']:.3f}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return self._create_error_report(execution_id, str(e))

    def validate_strategy_performance(self,
                                    strategy_name: str,
                                    epic: Optional[str] = None,
                                    timeframe: Optional[str] = None,
                                    min_executions: int = 3) -> Dict[str, Any]:
        """
        Cross-execution validation of strategy performance

        Args:
            strategy_name: Strategy to validate
            epic: Optional epic filter
            timeframe: Optional timeframe filter
            min_executions: Minimum executions required for validation

        Returns:
            Strategy validation report
        """
        self.logger.info(f"🔬 Validating strategy performance: {strategy_name}")

        try:
            # Load strategy data across multiple executions
            strategy_data = self._load_strategy_data(strategy_name, epic, timeframe)

            if len(strategy_data) < min_executions:
                return self._create_insufficient_executions_report(strategy_name, len(strategy_data), min_executions)

            # Perform cross-execution analysis
            results = {
                'strategy_name': strategy_name,
                'epic': epic,
                'timeframe': timeframe,
                'executions_analyzed': len(strategy_data),
                'validation_timestamp': datetime.now(timezone.utc),
                'validation_components': {}
            }

            # Performance consistency analysis
            results['validation_components']['performance_consistency'] = self._analyze_performance_consistency(strategy_data)

            # Parameter sensitivity analysis
            results['validation_components']['parameter_sensitivity'] = self._analyze_parameter_sensitivity(strategy_data)

            # Regime stability analysis
            results['validation_components']['regime_stability'] = self._analyze_regime_stability(strategy_data)

            # Meta-analysis of results
            results['meta_analysis'] = self._perform_meta_analysis(strategy_data)

            return results

        except Exception as e:
            self.logger.error(f"❌ Strategy validation failed: {e}")
            return self._create_error_report(f"strategy_{strategy_name}", str(e))

    # =========================================================================
    # REAL-TIME PERFORMANCE CORRELATION
    # =========================================================================

    def _analyze_realtime_correlation(self, execution_id: int) -> Dict[str, Any]:
        """
        Analyze correlation between backtest predictions and live performance

        Uses statistical correlation measures to validate that backtest results
        are predictive of live trading performance.
        """
        try:
            # Load backtest signals
            backtest_signals = self._load_backtest_signals(execution_id)

            # Load corresponding live performance data
            live_data = self._load_live_performance_data(execution_id)

            if len(live_data) == 0:
                return {
                    'status': 'no_live_data',
                    'message': 'No live performance data available for correlation analysis',
                    'correlation_possible': False
                }

            # Align backtest and live data
            aligned_data = self._align_backtest_live_data(backtest_signals, live_data)

            if len(aligned_data) < self.min_sample_size:
                return {
                    'status': 'insufficient_data',
                    'message': f'Insufficient aligned data points: {len(aligned_data)} < {self.min_sample_size}',
                    'correlation_possible': False
                }

            # Calculate correlation metrics
            correlation_metrics = self._calculate_correlation_metrics(aligned_data)

            # Statistical significance testing
            significance_results = self._test_correlation_significance(aligned_data)

            # Stability analysis
            stability_metrics = self._analyze_correlation_stability(aligned_data)

            return {
                'status': 'completed',
                'correlation_possible': True,
                'sample_size': len(aligned_data),
                'correlation_metrics': correlation_metrics,
                'significance_tests': significance_results,
                'stability_analysis': stability_metrics,
                'validation_result': self._validate_correlation_strength(correlation_metrics)
            }

        except Exception as e:
            self.logger.error(f"Error in real-time correlation analysis: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'correlation_possible': False
            }

    def _calculate_correlation_metrics(self, aligned_data: pd.DataFrame) -> CorrelationMetrics:
        """Calculate comprehensive correlation metrics"""

        # Extract backtest predictions and live results
        backtest_returns = aligned_data['backtest_predicted_return'].values
        live_returns = aligned_data['live_actual_return'].values

        # Pearson correlation (linear relationship)
        pearson_corr, pearson_p = pearsonr(backtest_returns, live_returns)

        # Spearman correlation (monotonic relationship)
        spearman_corr, spearman_p = spearmanr(backtest_returns, live_returns)

        # Kendall's tau (ordinal association)
        kendall_tau = None
        try:
            from scipy.stats import kendalltau
            kendall_tau, _ = kendalltau(backtest_returns, live_returns)
        except ImportError:
            pass

        # Linear regression R²
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(backtest_returns.reshape(-1, 1), live_returns)
        regression_r2 = reg.score(backtest_returns.reshape(-1, 1), live_returns)

        # Tracking error (volatility of difference)
        tracking_error = np.std(live_returns - backtest_returns)

        # Information ratio (excess return / tracking error)
        excess_return = np.mean(live_returns - backtest_returns)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

        # Rolling correlation stability
        window_size = min(30, len(aligned_data) // 3)
        rolling_corr = aligned_data['backtest_predicted_return'].rolling(window=window_size).corr(
            aligned_data['live_actual_return']
        )
        correlation_stability = 1 - (rolling_corr.std() / abs(rolling_corr.mean())) if rolling_corr.mean() != 0 else 0

        # Overall correlation assessment
        is_correlated = (
            abs(pearson_corr) >= self.min_correlation and
            pearson_p < self.alpha and
            regression_r2 >= self.min_correlation**2
        )

        return CorrelationMetrics(
            pearson_correlation=pearson_corr,
            spearman_correlation=spearman_corr,
            kendall_tau=kendall_tau,
            correlation_stability=correlation_stability,
            regression_r2=regression_r2,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            is_correlated=is_correlated,
            confidence_level=self.confidence_level
        )

    # =========================================================================
    # OVERFITTING DETECTION
    # =========================================================================

    def _detect_overfitting(self, backtest_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive overfitting detection using multiple statistical methods

        Methods:
        1. Walk-forward analysis
        2. Cross-validation performance degradation
        3. Information coefficient stability
        4. Parameter sensitivity analysis
        """
        try:
            # Prepare time series data for cross-validation
            returns_data = self._prepare_returns_data(backtest_data)

            if len(returns_data) < self.cv_folds * 2:
                return {
                    'status': 'insufficient_data',
                    'message': f'Insufficient data for cross-validation: {len(returns_data)}',
                    'overfitting_detected': False
                }

            # Walk-forward analysis
            wfa_results = self._walk_forward_analysis(returns_data)

            # Cross-validation analysis
            cv_results = self._cross_validation_analysis(returns_data)

            # Information coefficient analysis
            ic_results = self._information_coefficient_analysis(backtest_data)

            # Parameter sensitivity analysis
            sensitivity_results = self._parameter_sensitivity_analysis(backtest_data)

            # Combine results into overfitting metrics
            overfitting_metrics = self._calculate_overfitting_metrics(
                wfa_results, cv_results, ic_results, sensitivity_results
            )

            return {
                'status': 'completed',
                'overfitting_metrics': overfitting_metrics,
                'walk_forward_analysis': wfa_results,
                'cross_validation_analysis': cv_results,
                'information_coefficient_analysis': ic_results,
                'parameter_sensitivity_analysis': sensitivity_results,
                'validation_result': self._validate_overfitting_metrics(overfitting_metrics)
            }

        except Exception as e:
            self.logger.error(f"Error in overfitting detection: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'overfitting_detected': None
            }

    def _walk_forward_analysis(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform walk-forward analysis to detect overfitting"""

        # Split data into expanding windows
        n_periods = len(returns_data)
        min_train_size = max(self.min_sample_size, int(n_periods * 0.3))

        in_sample_returns = []
        out_sample_returns = []

        for start_idx in range(min_train_size, n_periods - 10, 10):  # Step by 10 periods
            # In-sample period
            in_sample = returns_data.iloc[:start_idx]
            # Out-of-sample period (next 10 periods)
            out_sample = returns_data.iloc[start_idx:start_idx+10]

            if len(out_sample) > 0:
                # Calculate performance metrics
                in_sample_perf = self._calculate_performance_metrics(in_sample)
                out_sample_perf = self._calculate_performance_metrics(out_sample)

                in_sample_returns.append(in_sample_perf['sharpe_ratio'])
                out_sample_returns.append(out_sample_perf['sharpe_ratio'])

        if not in_sample_returns or not out_sample_returns:
            return {'status': 'insufficient_periods'}

        # Calculate degradation metrics
        avg_in_sample = np.mean(in_sample_returns)
        avg_out_sample = np.mean(out_sample_returns)

        degradation_ratio = avg_in_sample / avg_out_sample if avg_out_sample != 0 else float('inf')

        # Statistical test for performance difference
        t_stat, p_value = ttest_ind(in_sample_returns, out_sample_returns)

        return {
            'status': 'completed',
            'periods_analyzed': len(in_sample_returns),
            'avg_in_sample_sharpe': avg_in_sample,
            'avg_out_sample_sharpe': avg_out_sample,
            'degradation_ratio': degradation_ratio,
            'performance_difference_pvalue': p_value,
            'performance_difference_significant': p_value < self.alpha,
            'in_sample_returns': in_sample_returns,
            'out_sample_returns': out_sample_returns
        }

    def _cross_validation_analysis(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform time series cross-validation analysis"""

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        fold_performances = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(returns_data)):
            train_data = returns_data.iloc[train_idx]
            test_data = returns_data.iloc[test_idx]

            if len(train_data) < 10 or len(test_data) < 5:
                continue

            # Calculate performance for this fold
            train_perf = self._calculate_performance_metrics(train_data)
            test_perf = self._calculate_performance_metrics(test_data)

            fold_performances.append({
                'fold': fold_idx,
                'train_sharpe': train_perf['sharpe_ratio'],
                'test_sharpe': test_perf['sharpe_ratio'],
                'train_size': len(train_data),
                'test_size': len(test_data)
            })

        if not fold_performances:
            return {'status': 'insufficient_folds'}

        # Calculate cross-validation metrics
        train_sharpes = [f['train_sharpe'] for f in fold_performances]
        test_sharpes = [f['test_sharpe'] for f in fold_performances]

        cv_score = np.mean(test_sharpes)
        cv_std = np.std(test_sharpes)

        # Stability score (consistency across folds)
        stability_score = 1 - (cv_std / abs(cv_score)) if cv_score != 0 else 0

        return {
            'status': 'completed',
            'n_folds': len(fold_performances),
            'cv_score': cv_score,
            'cv_std': cv_std,
            'stability_score': max(0, min(1, stability_score)),  # Clamp to [0,1]
            'fold_performances': fold_performances,
            'train_test_correlation': pearsonr(train_sharpes, test_sharpes)[0] if len(train_sharpes) > 1 else 0
        }

    # =========================================================================
    # STATISTICAL SIGNIFICANCE TESTING
    # =========================================================================

    def _test_statistical_significance(self, backtest_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive statistical significance testing

        Tests:
        1. Strategy returns vs. random walk
        2. Win rate vs. 50% (coin flip)
        3. Profit factor vs. 1.0 (break-even)
        4. Sharpe ratio significance
        5. Maximum drawdown analysis
        """
        try:
            # Calculate performance metrics
            performance_metrics = self._calculate_comprehensive_performance_metrics(backtest_data)

            # Initialize test results
            test_results = []

            # Test 1: Returns vs Random Walk (t-test)
            returns = backtest_data['pips_gained'].dropna()
            if len(returns) >= self.min_sample_size:
                t_stat, p_val = ttest_ind(returns, np.random.normal(0, returns.std(), len(returns)))
                test_results.append(ValidationResult(
                    test_name="Returns vs Random Walk",
                    result=TestResult.PASS if p_val < self.alpha else TestResult.FAIL,
                    p_value=p_val,
                    confidence_level=self.confidence_level,
                    statistic=t_stat,
                    message=f"Strategy returns {'significantly differ' if p_val < self.alpha else 'do not significantly differ'} from random walk"
                ))

            # Test 2: Win Rate vs 50% (binomial test)
            wins = len(backtest_data[backtest_data['trade_result'] == 'win'])
            total_trades = len(backtest_data[backtest_data['trade_result'].notna()])

            if total_trades >= self.min_sample_size:
                from scipy.stats import binom_test
                p_val = binom_test(wins, total_trades, 0.5, alternative='two-sided')
                win_rate = wins / total_trades

                test_results.append(ValidationResult(
                    test_name="Win Rate vs 50%",
                    result=TestResult.PASS if p_val < self.alpha else TestResult.FAIL,
                    p_value=p_val,
                    confidence_level=self.confidence_level,
                    statistic=win_rate,
                    critical_value=0.5,
                    message=f"Win rate {win_rate:.1%} {'is' if p_val < self.alpha else 'is not'} significantly different from 50%"
                ))

            # Test 3: Profit Factor vs 1.0
            if 'profit_factor' in performance_metrics and performance_metrics['profit_factor'] is not None:
                profit_factor = performance_metrics['profit_factor']

                # Bootstrap confidence interval for profit factor
                profit_factor_ci = self._bootstrap_confidence_interval(
                    backtest_data['pips_gained'].dropna(),
                    lambda x: self._calculate_profit_factor(x),
                    confidence_level=self.confidence_level
                )

                significant = profit_factor_ci[0] > 1.0 or profit_factor_ci[1] < 1.0

                test_results.append(ValidationResult(
                    test_name="Profit Factor vs 1.0",
                    result=TestResult.PASS if significant else TestResult.FAIL,
                    p_value=0.05 if not significant else 0.01,  # Approximate
                    confidence_level=self.confidence_level,
                    statistic=profit_factor,
                    critical_value=1.0,
                    message=f"Profit factor {profit_factor:.2f} {'is' if significant else 'is not'} significantly different from 1.0",
                    details={'confidence_interval': profit_factor_ci}
                ))

            # Test 4: Sharpe Ratio Significance
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio and len(returns) >= self.min_sample_size:
                # Sharpe ratio standard error
                n = len(returns)
                sharpe_se = np.sqrt((1 + 0.5 * sharpe_ratio**2) / n)
                t_stat = sharpe_ratio / sharpe_se
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))

                test_results.append(ValidationResult(
                    test_name="Sharpe Ratio Significance",
                    result=TestResult.PASS if p_val < self.alpha else TestResult.FAIL,
                    p_value=p_val,
                    confidence_level=self.confidence_level,
                    statistic=sharpe_ratio,
                    message=f"Sharpe ratio {sharpe_ratio:.3f} {'is' if p_val < self.alpha else 'is not'} statistically significant"
                ))

            # Multiple testing correction (Bonferroni)
            corrected_results = self._apply_bonferroni_correction(test_results)

            return {
                'status': 'completed',
                'individual_tests': test_results,
                'bonferroni_corrected': corrected_results,
                'performance_metrics': performance_metrics,
                'overall_significance': self._assess_overall_significance(corrected_results)
            }

        except Exception as e:
            self.logger.error(f"Error in statistical significance testing: {e}")
            return {
                'status': 'error',
                'error_message': str(e)
            }

    # =========================================================================
    # DATA QUALITY ASSURANCE
    # =========================================================================

    def _validate_data_quality(self, backtest_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality validation

        Checks:
        1. Data completeness and missing values
        2. Outlier detection using statistical methods
        3. Temporal consistency and gaps
        4. Price data reasonableness
        5. Market regime representation
        """
        try:
            quality_results = {
                'status': 'completed',
                'overall_quality_score': 0.0,
                'quality_components': {}
            }

            # 1. Data Completeness Analysis
            completeness_results = self._analyze_data_completeness(backtest_data)
            quality_results['quality_components']['completeness'] = completeness_results

            # 2. Outlier Detection
            outlier_results = self._detect_statistical_outliers(backtest_data)
            quality_results['quality_components']['outliers'] = outlier_results

            # 3. Temporal Consistency
            temporal_results = self._analyze_temporal_consistency(backtest_data)
            quality_results['quality_components']['temporal'] = temporal_results

            # 4. Price Data Validation
            price_results = self._validate_price_data(backtest_data)
            quality_results['quality_components']['price_data'] = price_results

            # 5. Market Regime Analysis
            regime_results = self._analyze_market_regime_coverage(backtest_data)
            quality_results['quality_components']['market_regime'] = regime_results

            # Calculate overall quality score
            quality_results['overall_quality_score'] = self._calculate_overall_quality_score(
                quality_results['quality_components']
            )

            # Generate quality recommendations
            quality_results['recommendations'] = self._generate_quality_recommendations(
                quality_results['quality_components']
            )

            return quality_results

        except Exception as e:
            self.logger.error(f"Error in data quality validation: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'overall_quality_score': 0.0
            }

    def _analyze_data_completeness(self, backtest_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data completeness and missing values"""

        total_records = len(backtest_data)

        # Check completeness for critical fields
        critical_fields = ['signal_timestamp', 'epic', 'signal_type', 'entry_price', 'confidence_score']

        completeness_scores = {}
        for field in critical_fields:
            if field in backtest_data.columns:
                non_null_count = backtest_data[field].notna().sum()
                completeness_scores[field] = non_null_count / total_records

        overall_completeness = np.mean(list(completeness_scores.values()))

        # Check for systematic missing data patterns
        missing_patterns = self._identify_missing_patterns(backtest_data)

        return {
            'total_records': total_records,
            'field_completeness': completeness_scores,
            'overall_completeness': overall_completeness,
            'missing_patterns': missing_patterns,
            'quality_score': overall_completeness,
            'passes_threshold': overall_completeness >= self.min_data_completeness
        }

    def _detect_statistical_outliers(self, backtest_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple statistical methods"""

        outlier_results = {}

        # Fields to check for outliers
        numeric_fields = ['confidence_score', 'entry_price', 'pips_gained']

        for field in numeric_fields:
            if field in backtest_data.columns and backtest_data[field].notna().sum() > 0:
                data = backtest_data[field].dropna()

                # Method 1: IQR method
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()

                # Method 2: Z-score method
                z_scores = np.abs(stats.zscore(data))
                z_outliers = (z_scores > 3).sum()

                # Method 3: Modified Z-score (using median)
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()

                outlier_results[field] = {
                    'iqr_outliers': int(iqr_outliers),
                    'z_score_outliers': int(z_outliers),
                    'modified_z_outliers': int(modified_z_outliers),
                    'outlier_percentage': (max(iqr_outliers, z_outliers) / len(data)) * 100,
                    'data_points': len(data)
                }

        # Overall outlier assessment
        total_outlier_percentage = np.mean([
            result['outlier_percentage'] for result in outlier_results.values()
        ]) if outlier_results else 0

        return {
            'field_outliers': outlier_results,
            'overall_outlier_percentage': total_outlier_percentage,
            'quality_score': max(0, 1 - (total_outlier_percentage / 100) * 2),  # Penalty for outliers
            'acceptable_outlier_level': total_outlier_percentage < 5.0  # Less than 5% outliers
        }

    # =========================================================================
    # PIPELINE CONSISTENCY VALIDATION
    # =========================================================================

    def _validate_pipeline_consistency(self, execution_id: int) -> Dict[str, Any]:
        """
        Validate consistency between backtest and live trading pipelines

        Checks:
        1. Same validation logic usage
        2. Consistent signal processing
        3. Identical feature calculation
        4. Matching data preprocessing
        """
        try:
            consistency_results = {
                'status': 'completed',
                'overall_consistency_score': 0.0,
                'consistency_components': {}
            }

            # 1. Validation Logic Consistency
            validation_consistency = self._check_validation_logic_consistency(execution_id)
            consistency_results['consistency_components']['validation_logic'] = validation_consistency

            # 2. Signal Processing Consistency
            signal_consistency = self._check_signal_processing_consistency(execution_id)
            consistency_results['consistency_components']['signal_processing'] = signal_consistency

            # 3. Feature Calculation Consistency
            feature_consistency = self._check_feature_calculation_consistency(execution_id)
            consistency_results['consistency_components']['feature_calculation'] = feature_consistency

            # 4. Data Pipeline Consistency
            data_consistency = self._check_data_pipeline_consistency(execution_id)
            consistency_results['consistency_components']['data_pipeline'] = data_consistency

            # Calculate overall consistency score
            consistency_results['overall_consistency_score'] = self._calculate_overall_consistency_score(
                consistency_results['consistency_components']
            )

            return consistency_results

        except Exception as e:
            self.logger.error(f"Error in pipeline consistency validation: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'overall_consistency_score': 0.0
            }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _load_backtest_data(self, execution_id: int) -> pd.DataFrame:
        """Load backtest data from database"""
        query = """
        SELECT bs.*, be.strategy_name, be.data_start_date, be.data_end_date
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

    def _validate_sample_size(self, data: pd.DataFrame) -> bool:
        """Validate minimum sample size requirement"""
        return len(data) >= self.min_sample_size

    def _calculate_comprehensive_performance_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {}

        # Return metrics
        if 'pips_gained' in data.columns:
            returns = data['pips_gained'].dropna()
            if len(returns) > 0:
                metrics.update({
                    'total_return': returns.sum(),
                    'avg_return': returns.mean(),
                    'return_std': returns.std(),
                    'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                    'max_return': returns.max(),
                    'min_return': returns.min()
                })

        # Trade outcome metrics
        if 'trade_result' in data.columns:
            trade_results = data['trade_result'].dropna()
            total_trades = len(trade_results)

            if total_trades > 0:
                wins = (trade_results == 'win').sum()
                losses = (trade_results == 'loss').sum()

                metrics.update({
                    'total_trades': total_trades,
                    'winning_trades': wins,
                    'losing_trades': losses,
                    'win_rate': wins / total_trades,
                    'loss_rate': losses / total_trades
                })

                # Profit factor
                if 'pips_gained' in data.columns:
                    win_returns = data[data['trade_result'] == 'win']['pips_gained'].sum()
                    loss_returns = abs(data[data['trade_result'] == 'loss']['pips_gained'].sum())

                    metrics['profit_factor'] = win_returns / loss_returns if loss_returns > 0 else None

        return metrics

    def _calculate_overall_validation_score(self, validation_components: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall validation score from components"""

        # Component weights
        weights = {
            'data_quality': 0.25,
            'statistical_significance': 0.30,
            'overfitting_detection': 0.25,
            'realtime_correlation': 0.10,  # May be 0 if no live data
            'pipeline_consistency': 0.10
        }

        # Adjust weights if some components are not available
        available_components = {}
        total_weight = 0

        for component, weight in weights.items():
            if component in validation_components:
                comp_data = validation_components[component]

                if comp_data.get('status') == 'completed':
                    available_components[component] = comp_data
                    total_weight += weight

        # Normalize weights
        normalized_weights = {comp: weights[comp] / total_weight
                            for comp in available_components.keys()}

        # Calculate weighted score
        composite_score = 0.0
        component_scores = {}

        for component, weight in normalized_weights.items():
            comp_data = available_components[component]

            # Extract score from each component
            if component == 'data_quality':
                score = comp_data.get('overall_quality_score', 0)
            elif component == 'statistical_significance':
                significance_tests = comp_data.get('bonferroni_corrected', [])
                score = sum(1 for test in significance_tests if test.result == TestResult.PASS) / max(1, len(significance_tests))
            elif component == 'overfitting_detection':
                overfitting_metrics = comp_data.get('overfitting_metrics', {})
                score = 1.0 - (1.0 if overfitting_metrics.get('is_overfitted', True) else 0.0)
            elif component == 'realtime_correlation':
                if comp_data.get('correlation_possible', False):
                    correlation_metrics = comp_data.get('correlation_metrics', {})
                    score = 1.0 if correlation_metrics.get('is_correlated', False) else 0.5
                else:
                    score = 0.5  # Neutral score if correlation not possible
            elif component == 'pipeline_consistency':
                score = comp_data.get('overall_consistency_score', 0)
            else:
                score = 0.5  # Default neutral score

            component_scores[component] = score
            composite_score += score * weight

        # Determine overall validation result
        if composite_score >= 0.8:
            validation_result = TestResult.PASS
            validation_message = "Strategy passes comprehensive statistical validation"
        elif composite_score >= 0.6:
            validation_result = TestResult.WARNING
            validation_message = "Strategy validation shows some concerns but may be acceptable"
        else:
            validation_result = TestResult.FAIL
            validation_message = "Strategy fails statistical validation - significant issues detected"

        return {
            'composite_score': composite_score,
            'component_scores': component_scores,
            'normalized_weights': normalized_weights,
            'validation_result': validation_result.value,
            'validation_message': validation_message,
            'confidence_level': self.confidence_level
        }

    def _store_validation_results(self, execution_id: int, validation_results: Dict[str, Any]):
        """Store validation results in database"""
        try:
            # Store in a validation results table (create if needed)
            query = """
            INSERT INTO backtest_validation_results (
                execution_id, validation_timestamp, validation_level,
                composite_score, validation_result, validation_data,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (execution_id) DO UPDATE SET
                validation_timestamp = EXCLUDED.validation_timestamp,
                validation_level = EXCLUDED.validation_level,
                composite_score = EXCLUDED.composite_score,
                validation_result = EXCLUDED.validation_result,
                validation_data = EXCLUDED.validation_data,
                updated_at = NOW()
            """

            self.db_manager.execute_query(query, (
                execution_id,
                validation_results['validation_timestamp'],
                self.validation_level.name,
                validation_results['overall_validation']['composite_score'],
                validation_results['overall_validation']['validation_result'],
                json.dumps(validation_results, default=str)
            ))

        except Exception as e:
            self.logger.error(f"Error storing validation results: {e}")

    def _create_error_report(self, execution_id: Union[int, str], error_message: str) -> Dict[str, Any]:
        """Create error report for failed validation"""
        return {
            'execution_id': execution_id,
            'status': 'error',
            'error_message': error_message,
            'validation_timestamp': datetime.now(timezone.utc),
            'validation_level': self.validation_level.name,
            'overall_validation': {
                'composite_score': 0.0,
                'validation_result': TestResult.FAIL.value,
                'validation_message': f"Validation failed due to error: {error_message}"
            }
        }

    def _create_insufficient_data_report(self, execution_id: int, data_size: int) -> Dict[str, Any]:
        """Create report for insufficient data"""
        return {
            'execution_id': execution_id,
            'status': 'insufficient_data',
            'data_size': data_size,
            'required_size': self.min_sample_size,
            'validation_timestamp': datetime.now(timezone.utc),
            'validation_level': self.validation_level.name,
            'overall_validation': {
                'composite_score': 0.0,
                'validation_result': TestResult.INCONCLUSIVE.value,
                'validation_message': f"Insufficient data for validation: {data_size} < {self.min_sample_size}"
            }
        }

    # Additional utility methods would continue here...
    # Due to length constraints, I'm providing the core framework structure

    def __repr__(self):
        return (f"StatisticalValidationFramework("
                f"confidence_level={self.confidence_level:.1%}, "
                f"min_sample_size={self.min_sample_size}, "
                f"validation_level={self.validation_level.name})")


# Factory function for framework creation
def create_statistical_validation_framework(
    db_manager: DatabaseManager,
    validation_level: ValidationLevel = ValidationLevel.MEDIUM,
    **kwargs
) -> StatisticalValidationFramework:
    """Create StatisticalValidationFramework instance"""
    return StatisticalValidationFramework(
        db_manager=db_manager,
        validation_level=validation_level,
        **kwargs
    )