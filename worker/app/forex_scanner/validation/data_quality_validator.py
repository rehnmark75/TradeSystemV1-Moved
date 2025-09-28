# forex_scanner/validation/data_quality_validator.py
"""
Data Quality Assurance for Trading Strategy Validation

This module provides comprehensive data quality validation to ensure that
historical data used in backtesting represents realistic market conditions
and is suitable for statistical analysis.

Key Features:
1. Temporal consistency and gap analysis
2. Price data reasonableness checks
3. Volume and liquidity analysis
4. Market microstructure validation
5. Outlier detection and handling
6. Data completeness assessment
7. Market regime coverage analysis
8. Statistical distribution validation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Statistical libraries
from scipy import stats
from scipy.stats import jarque_bera, anderson, kstest, normaltest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    UNACCEPTABLE = "UNACCEPTABLE"


class QualityCheck(Enum):
    """Types of quality checks"""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"


@dataclass
class QualityMetric:
    """Container for individual quality metric results"""
    metric_name: str
    quality_check_type: QualityCheck
    score: float  # 0.0 to 1.0
    threshold: float
    is_passed: bool
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TemporalAnalysis:
    """Results from temporal data analysis"""
    total_periods: int
    missing_periods: int
    gap_count: int
    largest_gap_minutes: float
    average_gap_minutes: float
    temporal_consistency_score: float
    timezone_consistency: bool
    sampling_regularity_score: float


@dataclass
class PriceDataAnalysis:
    """Results from price data validation"""
    price_reasonableness_score: float
    spread_analysis: Dict[str, Any]
    price_jump_analysis: Dict[str, Any]
    ohlc_consistency_score: float
    price_precision_analysis: Dict[str, Any]
    currency_pair_characteristics: Dict[str, Any]


@dataclass
class VolumeAnalysis:
    """Results from volume data analysis"""
    volume_completeness: float
    volume_consistency_score: float
    volume_outliers_percentage: float
    volume_trend_analysis: Dict[str, Any]
    liquidity_indicators: Dict[str, Any]


@dataclass
class MarketRegimeAnalysis:
    """Analysis of market regime representation"""
    regime_coverage: Dict[str, float]
    volatility_regimes: Dict[str, Any]
    trend_regimes: Dict[str, Any]
    regime_diversity_score: float
    regime_balance_score: float


class DataQualityValidator:
    """
    Comprehensive Data Quality Validation for Trading Data

    Validates historical market data to ensure it's suitable for backtesting
    and statistical analysis, identifying potential issues that could
    invalidate trading strategy results.
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 min_data_completeness: float = 0.95,
                 max_gap_threshold_minutes: int = 60,
                 outlier_threshold: float = 0.05,
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.min_data_completeness = min_data_completeness
        self.max_gap_threshold_minutes = max_gap_threshold_minutes
        self.outlier_threshold = outlier_threshold
        self.logger = logger or logging.getLogger(__name__)

        # Quality thresholds
        self.excellent_threshold = 0.95
        self.good_threshold = 0.85
        self.acceptable_threshold = 0.70

        # Price validation parameters
        self.max_price_jump_std = 5.0  # Max price jump in standard deviations
        self.min_spread_pips = 0.1     # Minimum spread in pips
        self.max_spread_pips = 50.0    # Maximum spread in pips

        # Volume validation parameters
        self.volume_outlier_threshold = 3.0  # Z-score threshold for volume outliers

        self.logger.info(f"🔍 Data Quality Validator initialized:")
        self.logger.info(f"   Min completeness: {self.min_data_completeness:.1%}")
        self.logger.info(f"   Max gap threshold: {self.max_gap_threshold_minutes} minutes")
        self.logger.info(f"   Outlier threshold: {self.outlier_threshold:.1%}")

    def validate_data_quality(self, execution_id: int) -> Dict[str, Any]:
        """
        Comprehensive data quality validation for a backtest execution

        Args:
            execution_id: Backtest execution ID to validate

        Returns:
            Comprehensive data quality assessment
        """
        self.logger.info(f"🔍 Starting data quality validation for execution {execution_id}")

        try:
            # Load execution data and configuration
            execution_data = self._load_execution_data(execution_id)
            backtest_data = self._load_backtest_data(execution_id)

            if backtest_data.empty:
                return self._create_no_data_result(execution_id)

            # Initialize results container
            results = {
                'execution_id': execution_id,
                'validation_timestamp': datetime.now(timezone.utc),
                'data_period': {
                    'start_date': execution_data.get('data_start_date'),
                    'end_date': execution_data.get('data_end_date'),
                    'total_signals': len(backtest_data)
                },
                'quality_metrics': [],
                'detailed_analysis': {},
                'overall_assessment': {}
            }

            # 1. Data Completeness Analysis
            self.logger.info("📊 Analyzing data completeness...")
            completeness_metrics = self._analyze_data_completeness(backtest_data)
            results['quality_metrics'].extend(completeness_metrics)
            results['detailed_analysis']['completeness'] = self._detailed_completeness_analysis(backtest_data)

            # 2. Temporal Consistency Analysis
            self.logger.info("⏰ Analyzing temporal consistency...")
            temporal_metrics = self._analyze_temporal_consistency(backtest_data, execution_data)
            results['quality_metrics'].extend(temporal_metrics)
            results['detailed_analysis']['temporal'] = self._detailed_temporal_analysis(backtest_data, execution_data)

            # 3. Price Data Validation
            self.logger.info("💰 Validating price data...")
            price_metrics = self._validate_price_data(backtest_data)
            results['quality_metrics'].extend(price_metrics)
            results['detailed_analysis']['price'] = self._detailed_price_analysis(backtest_data)

            # 4. Volume Data Analysis
            self.logger.info("📈 Analyzing volume data...")
            volume_metrics = self._analyze_volume_data(backtest_data)
            results['quality_metrics'].extend(volume_metrics)
            results['detailed_analysis']['volume'] = self._detailed_volume_analysis(backtest_data)

            # 5. Market Regime Coverage Analysis
            self.logger.info("🌍 Analyzing market regime coverage...")
            regime_metrics = self._analyze_market_regime_coverage(backtest_data, execution_data)
            results['quality_metrics'].extend(regime_metrics)
            results['detailed_analysis']['market_regimes'] = self._detailed_regime_analysis(backtest_data)

            # 6. Statistical Distribution Validation
            self.logger.info("📉 Validating statistical distributions...")
            distribution_metrics = self._validate_statistical_distributions(backtest_data)
            results['quality_metrics'].extend(distribution_metrics)
            results['detailed_analysis']['distributions'] = self._detailed_distribution_analysis(backtest_data)

            # 7. Outlier Detection and Analysis
            self.logger.info("🎯 Detecting outliers...")
            outlier_metrics = self._detect_and_analyze_outliers(backtest_data)
            results['quality_metrics'].extend(outlier_metrics)
            results['detailed_analysis']['outliers'] = self._detailed_outlier_analysis(backtest_data)

            # Calculate Overall Quality Assessment
            results['overall_assessment'] = self._calculate_overall_quality_assessment(results['quality_metrics'])

            # Generate Recommendations
            results['recommendations'] = self._generate_quality_recommendations(results)

            # Store results
            self._store_quality_results(execution_id, results)

            self.logger.info(f"✅ Data quality validation completed:")
            self.logger.info(f"   Overall Score: {results['overall_assessment']['composite_score']:.3f}")
            self.logger.info(f"   Quality Level: {results['overall_assessment']['quality_level']}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Data quality validation failed: {e}")
            return {
                'execution_id': execution_id,
                'status': 'error',
                'error_message': str(e),
                'validation_timestamp': datetime.now(timezone.utc)
            }

    def _analyze_data_completeness(self, backtest_data: pd.DataFrame) -> List[QualityMetric]:
        """Analyze data completeness across critical fields"""

        metrics = []

        # Define critical fields for completeness analysis
        critical_fields = {
            'signal_timestamp': 'Timestamp completeness',
            'epic': 'Epic completeness',
            'signal_type': 'Signal type completeness',
            'confidence_score': 'Confidence score completeness',
            'entry_price': 'Entry price completeness'
        }

        optional_fields = {
            'pips_gained': 'Pips gained completeness',
            'trade_result': 'Trade result completeness',
            'exit_price': 'Exit price completeness',
            'volume': 'Volume completeness'
        }

        # Analyze critical fields
        for field, description in critical_fields.items():
            if field in backtest_data.columns:
                non_null_count = backtest_data[field].notna().sum()
                total_count = len(backtest_data)
                completeness_score = non_null_count / total_count if total_count > 0 else 0

                metrics.append(QualityMetric(
                    metric_name=f"{field}_completeness",
                    quality_check_type=QualityCheck.COMPLETENESS,
                    score=completeness_score,
                    threshold=self.min_data_completeness,
                    is_passed=completeness_score >= self.min_data_completeness,
                    description=description,
                    details={
                        'non_null_count': non_null_count,
                        'total_count': total_count,
                        'missing_count': total_count - non_null_count,
                        'field_type': 'critical'
                    },
                    recommendations=self._generate_completeness_recommendations(completeness_score, field)
                ))

        # Analyze optional fields
        for field, description in optional_fields.items():
            if field in backtest_data.columns:
                non_null_count = backtest_data[field].notna().sum()
                total_count = len(backtest_data)
                completeness_score = non_null_count / total_count if total_count > 0 else 0

                # Lower threshold for optional fields
                optional_threshold = 0.80

                metrics.append(QualityMetric(
                    metric_name=f"{field}_completeness",
                    quality_check_type=QualityCheck.COMPLETENESS,
                    score=completeness_score,
                    threshold=optional_threshold,
                    is_passed=completeness_score >= optional_threshold,
                    description=description,
                    details={
                        'non_null_count': non_null_count,
                        'total_count': total_count,
                        'missing_count': total_count - non_null_count,
                        'field_type': 'optional'
                    },
                    recommendations=self._generate_completeness_recommendations(completeness_score, field)
                ))

        return metrics

    def _analyze_temporal_consistency(self,
                                    backtest_data: pd.DataFrame,
                                    execution_data: Dict[str, Any]) -> List[QualityMetric]:
        """Analyze temporal consistency of the data"""

        metrics = []

        try:
            if 'signal_timestamp' not in backtest_data.columns:
                return [QualityMetric(
                    metric_name="temporal_consistency",
                    quality_check_type=QualityCheck.CONSISTENCY,
                    score=0.0,
                    threshold=0.8,
                    is_passed=False,
                    description="No timestamp data available",
                    recommendations=["Ensure timestamp data is included in backtest results"]
                )]

            # Convert timestamps and sort
            timestamps = pd.to_datetime(backtest_data['signal_timestamp']).sort_values()

            if len(timestamps) < 2:
                return [QualityMetric(
                    metric_name="temporal_consistency",
                    quality_check_type=QualityCheck.CONSISTENCY,
                    score=0.5,
                    threshold=0.8,
                    is_passed=False,
                    description="Insufficient timestamp data",
                    recommendations=["Increase sample size for temporal analysis"]
                )]

            # Calculate time differences
            time_diffs = timestamps.diff().dropna()
            time_diffs_minutes = time_diffs.dt.total_seconds() / 60

            # Gap analysis
            gaps = time_diffs_minutes[time_diffs_minutes > self.max_gap_threshold_minutes]
            gap_count = len(gaps)
            largest_gap = gaps.max() if len(gaps) > 0 else 0
            average_gap = gaps.mean() if len(gaps) > 0 else 0

            # Temporal consistency score
            expected_intervals = self._estimate_expected_intervals(execution_data)
            consistency_score = self._calculate_temporal_consistency_score(
                time_diffs_minutes, expected_intervals, gap_count, len(timestamps)
            )

            metrics.append(QualityMetric(
                metric_name="temporal_consistency",
                quality_check_type=QualityCheck.CONSISTENCY,
                score=consistency_score,
                threshold=0.8,
                is_passed=consistency_score >= 0.8,
                description="Temporal consistency of signal timestamps",
                details={
                    'total_signals': len(timestamps),
                    'gap_count': gap_count,
                    'largest_gap_minutes': largest_gap,
                    'average_gap_minutes': average_gap,
                    'median_interval_minutes': time_diffs_minutes.median(),
                    'std_interval_minutes': time_diffs_minutes.std()
                },
                recommendations=self._generate_temporal_recommendations(consistency_score, gap_count)
            ))

            # Timezone consistency
            timezone_consistency_score = self._check_timezone_consistency(timestamps)

            metrics.append(QualityMetric(
                metric_name="timezone_consistency",
                quality_check_type=QualityCheck.CONSISTENCY,
                score=timezone_consistency_score,
                threshold=1.0,
                is_passed=timezone_consistency_score == 1.0,
                description="Consistency of timezone information",
                details={'timezone_consistent': timezone_consistency_score == 1.0},
                recommendations=["Ensure all timestamps use consistent timezone"] if timezone_consistency_score < 1.0 else []
            ))

        except Exception as e:
            self.logger.error(f"Error in temporal consistency analysis: {e}")
            metrics.append(QualityMetric(
                metric_name="temporal_consistency",
                quality_check_type=QualityCheck.CONSISTENCY,
                score=0.0,
                threshold=0.8,
                is_passed=False,
                description=f"Temporal analysis failed: {str(e)}",
                recommendations=["Review timestamp data format and consistency"]
            ))

        return metrics

    def _validate_price_data(self, backtest_data: pd.DataFrame) -> List[QualityMetric]:
        """Validate price data reasonableness and consistency"""

        metrics = []

        price_fields = ['entry_price', 'exit_price', 'open_price', 'high_price', 'low_price', 'close_price']
        available_price_fields = [field for field in price_fields if field in backtest_data.columns]

        if not available_price_fields:
            return [QualityMetric(
                metric_name="price_data_availability",
                quality_check_type=QualityCheck.VALIDITY,
                score=0.0,
                threshold=1.0,
                is_passed=False,
                description="No price data available",
                recommendations=["Ensure price data is included in backtest results"]
            )]

        # Analyze each price field
        for field in available_price_fields:
            price_data = backtest_data[field].dropna()

            if len(price_data) == 0:
                continue

            # Price reasonableness checks
            price_reasonableness_score = self._assess_price_reasonableness(price_data, field)

            metrics.append(QualityMetric(
                metric_name=f"{field}_reasonableness",
                quality_check_type=QualityCheck.VALIDITY,
                score=price_reasonableness_score,
                threshold=0.9,
                is_passed=price_reasonableness_score >= 0.9,
                description=f"Reasonableness of {field} values",
                details=self._calculate_price_statistics(price_data),
                recommendations=self._generate_price_recommendations(price_reasonableness_score, field)
            ))

        # OHLC consistency (if available)
        if all(field in backtest_data.columns for field in ['open_price', 'high_price', 'low_price', 'close_price']):
            ohlc_consistency_score = self._validate_ohlc_consistency(backtest_data)

            metrics.append(QualityMetric(
                metric_name="ohlc_consistency",
                quality_check_type=QualityCheck.CONSISTENCY,
                score=ohlc_consistency_score,
                threshold=0.95,
                is_passed=ohlc_consistency_score >= 0.95,
                description="OHLC price relationship consistency",
                details={'ohlc_violations': 1 - ohlc_consistency_score},
                recommendations=self._generate_ohlc_recommendations(ohlc_consistency_score)
            ))

        # Price jump analysis
        if 'entry_price' in backtest_data.columns:
            price_jump_score = self._analyze_price_jumps(backtest_data['entry_price'].dropna())

            metrics.append(QualityMetric(
                metric_name="price_jump_analysis",
                quality_check_type=QualityCheck.VALIDITY,
                score=price_jump_score,
                threshold=0.85,
                is_passed=price_jump_score >= 0.85,
                description="Analysis of extreme price movements",
                recommendations=self._generate_price_jump_recommendations(price_jump_score)
            ))

        return metrics

    def _detect_and_analyze_outliers(self, backtest_data: pd.DataFrame) -> List[QualityMetric]:
        """Detect and analyze outliers in the data"""

        metrics = []

        # Numeric fields to analyze for outliers
        numeric_fields = ['confidence_score', 'pips_gained', 'entry_price']
        available_fields = [field for field in numeric_fields if field in backtest_data.columns and backtest_data[field].notna().sum() > 0]

        for field in available_fields:
            data = backtest_data[field].dropna().astype(float)

            if len(data) < 10:
                continue

            # Multiple outlier detection methods
            outlier_results = self._detect_outliers_multiple_methods(data)

            # Calculate overall outlier score
            outlier_percentage = outlier_results['overall_outlier_percentage']
            outlier_score = max(0, 1 - (outlier_percentage / 100) * 2)  # Penalty for outliers

            metrics.append(QualityMetric(
                metric_name=f"{field}_outlier_analysis",
                quality_check_type=QualityCheck.VALIDITY,
                score=outlier_score,
                threshold=0.8,
                is_passed=outlier_percentage < (self.outlier_threshold * 100),
                description=f"Outlier analysis for {field}",
                details=outlier_results,
                recommendations=self._generate_outlier_recommendations(outlier_percentage, field)
            ))

        return metrics

    def _detect_outliers_multiple_methods(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""

        try:
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
            modified_z_scores = 0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
            modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()

            # Method 4: Isolation Forest
            isolation_outliers = 0
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(data.values.reshape(-1, 1))
                isolation_outliers = (outlier_pred == -1).sum()
            except:
                pass

            # Consensus outlier count
            outlier_counts = [iqr_outliers, z_outliers, modified_z_outliers, isolation_outliers]
            max_outliers = max(outlier_counts)
            outlier_percentage = (max_outliers / len(data)) * 100

            return {
                'iqr_outliers': int(iqr_outliers),
                'z_score_outliers': int(z_outliers),
                'modified_z_outliers': int(modified_z_outliers),
                'isolation_forest_outliers': int(isolation_outliers),
                'overall_outlier_percentage': outlier_percentage,
                'data_points': len(data),
                'outlier_bounds': {
                    'iqr_lower': lower_bound,
                    'iqr_upper': upper_bound,
                    'z_score_threshold': 3.0,
                    'modified_z_threshold': 3.5
                }
            }

        except Exception as e:
            self.logger.error(f"Error in outlier detection: {e}")
            return {
                'error': str(e),
                'overall_outlier_percentage': 0.0,
                'data_points': len(data)
            }

    def _calculate_overall_quality_assessment(self, quality_metrics: List[QualityMetric]) -> Dict[str, Any]:
        """Calculate overall data quality assessment"""

        try:
            if not quality_metrics:
                return {
                    'composite_score': 0.0,
                    'quality_level': DataQualityLevel.UNACCEPTABLE,
                    'passed_checks': 0,
                    'total_checks': 0,
                    'critical_issues': ['No quality metrics available']
                }

            # Calculate weighted composite score
            weights = {
                QualityCheck.COMPLETENESS: 0.30,
                QualityCheck.CONSISTENCY: 0.25,
                QualityCheck.VALIDITY: 0.25,
                QualityCheck.ACCURACY: 0.15,
                QualityCheck.TIMELINESS: 0.05,
                QualityCheck.UNIQUENESS: 0.0
            }

            weighted_scores = {}
            for check_type, weight in weights.items():
                relevant_metrics = [m for m in quality_metrics if m.quality_check_type == check_type]
                if relevant_metrics:
                    avg_score = np.mean([m.score for m in relevant_metrics])
                    weighted_scores[check_type.value] = avg_score * weight

            composite_score = sum(weighted_scores.values())

            # Determine quality level
            if composite_score >= self.excellent_threshold:
                quality_level = DataQualityLevel.EXCELLENT
            elif composite_score >= self.good_threshold:
                quality_level = DataQualityLevel.GOOD
            elif composite_score >= self.acceptable_threshold:
                quality_level = DataQualityLevel.ACCEPTABLE
            elif composite_score >= 0.5:
                quality_level = DataQualityLevel.POOR
            else:
                quality_level = DataQualityLevel.UNACCEPTABLE

            # Count passed/failed checks
            passed_checks = sum(1 for m in quality_metrics if m.is_passed)
            total_checks = len(quality_metrics)

            # Identify critical issues
            critical_issues = []
            for metric in quality_metrics:
                if not metric.is_passed and metric.score < 0.5:
                    critical_issues.append(f"Critical issue in {metric.metric_name}: {metric.description}")

            return {
                'composite_score': composite_score,
                'quality_level': quality_level.value,
                'passed_checks': passed_checks,
                'total_checks': total_checks,
                'pass_rate': passed_checks / total_checks if total_checks > 0 else 0,
                'weighted_scores': weighted_scores,
                'critical_issues': critical_issues,
                'quality_summary': {
                    'excellent_checks': sum(1 for m in quality_metrics if m.score >= self.excellent_threshold),
                    'good_checks': sum(1 for m in quality_metrics if self.good_threshold <= m.score < self.excellent_threshold),
                    'acceptable_checks': sum(1 for m in quality_metrics if self.acceptable_threshold <= m.score < self.good_threshold),
                    'poor_checks': sum(1 for m in quality_metrics if m.score < self.acceptable_threshold)
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating overall quality assessment: {e}")
            return {
                'composite_score': 0.0,
                'quality_level': DataQualityLevel.UNACCEPTABLE.value,
                'passed_checks': 0,
                'total_checks': 0,
                'error': str(e)
            }

    def _generate_quality_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving data quality"""

        recommendations = []
        overall_assessment = results.get('overall_assessment', {})
        quality_metrics = results.get('quality_metrics', [])

        # Overall quality recommendations
        quality_level = overall_assessment.get('quality_level')
        if quality_level == DataQualityLevel.UNACCEPTABLE.value:
            recommendations.append("DATA QUALITY CRITICAL: Complete data review and collection required")
            recommendations.append("Strategy validation not recommended with current data quality")
        elif quality_level == DataQualityLevel.POOR.value:
            recommendations.append("Significant data quality improvements required before strategy deployment")
            recommendations.append("Consider additional data sources or cleaning procedures")
        elif quality_level == DataQualityLevel.ACCEPTABLE.value:
            recommendations.append("Data quality is minimally acceptable - monitor closely")
            recommendations.append("Consider improvements to strengthen validation confidence")

        # Specific metric recommendations
        failed_metrics = [m for m in quality_metrics if not m.is_passed]
        if failed_metrics:
            recommendations.append(f"Address {len(failed_metrics)} failed quality checks:")
            for metric in failed_metrics[:5]:  # Top 5 issues
                if metric.recommendations:
                    recommendations.extend(metric.recommendations[:2])  # Top 2 per metric

        # Critical issues
        critical_issues = overall_assessment.get('critical_issues', [])
        if critical_issues:
            recommendations.append("CRITICAL ISSUES IDENTIFIED:")
            recommendations.extend(critical_issues[:3])  # Top 3 critical issues

        return recommendations

    # Utility methods
    def _load_execution_data(self, execution_id: int) -> Dict[str, Any]:
        """Load execution configuration data"""
        query = """
        SELECT * FROM backtest_executions WHERE id = %s
        """
        result = self.db_manager.execute_query(query, (execution_id,))
        row = result.fetchone()

        if row:
            columns = [desc[0] for desc in result.description]
            return dict(zip(columns, row))
        return {}

    def _load_backtest_data(self, execution_id: int) -> pd.DataFrame:
        """Load backtest signal data"""
        query = """
        SELECT * FROM backtest_signals WHERE execution_id = %s ORDER BY signal_timestamp
        """
        result = self.db_manager.execute_query(query, (execution_id,))
        columns = [desc[0] for desc in result.description]
        data = result.fetchall()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=columns)
        return df

    def _create_no_data_result(self, execution_id: int) -> Dict[str, Any]:
        """Create result for no data scenarios"""
        return {
            'execution_id': execution_id,
            'status': 'no_data',
            'validation_timestamp': datetime.now(timezone.utc),
            'overall_assessment': {
                'composite_score': 0.0,
                'quality_level': DataQualityLevel.UNACCEPTABLE.value,
                'critical_issues': ['No data available for quality assessment']
            },
            'recommendations': ['Ensure backtest data is properly stored and accessible']
        }

    def _store_quality_results(self, execution_id: int, results: Dict[str, Any]):
        """Store data quality results"""
        try:
            # Store quality assessment results
            # Implementation depends on database schema
            self.logger.info(f"Storing data quality results for execution {execution_id}")
            # TODO: Implement database storage
        except Exception as e:
            self.logger.error(f"Error storing quality results: {e}")

    # Additional helper methods would be implemented here...
    def _generate_completeness_recommendations(self, score: float, field: str) -> List[str]:
        """Generate recommendations for completeness issues"""
        if score < 0.5:
            return [f"Critical: {field} has >50% missing data - review data collection process"]
        elif score < 0.8:
            return [f"Moderate: {field} missing data may impact analysis reliability"]
        elif score < 0.95:
            return [f"Minor: {field} has some missing data - consider data cleaning"]
        return []

    # Placeholder methods for detailed analysis (would be fully implemented)
    def _detailed_completeness_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        return {"status": "placeholder"}

    def _detailed_temporal_analysis(self, data: pd.DataFrame, execution_data: Dict) -> TemporalAnalysis:
        return TemporalAnalysis(0, 0, 0, 0, 0, 0, True, 0)

    def _detailed_price_analysis(self, data: pd.DataFrame) -> PriceDataAnalysis:
        return PriceDataAnalysis(0, {}, {}, 0, {}, {})

    def _detailed_volume_analysis(self, data: pd.DataFrame) -> VolumeAnalysis:
        return VolumeAnalysis(0, 0, 0, {}, {})

    def _detailed_regime_analysis(self, data: pd.DataFrame) -> MarketRegimeAnalysis:
        return MarketRegimeAnalysis({}, {}, {}, 0, 0)

    def _detailed_distribution_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        return {"status": "placeholder"}

    def _detailed_outlier_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        return {"status": "placeholder"}

    # Additional placeholder methods
    def _estimate_expected_intervals(self, execution_data: Dict[str, Any]) -> List[int]:
        return [15]  # Default 15-minute intervals

    def _calculate_temporal_consistency_score(self, diffs, expected, gap_count, total) -> float:
        return 0.8  # Placeholder

    def _check_timezone_consistency(self, timestamps: pd.Series) -> float:
        return 1.0  # Placeholder

    def _generate_temporal_recommendations(self, score: float, gap_count: int) -> List[str]:
        return [] if score > 0.8 else ["Review temporal data consistency"]

    def _assess_price_reasonableness(self, data: pd.Series, field: str) -> float:
        return 0.9  # Placeholder

    def _calculate_price_statistics(self, data: pd.Series) -> Dict[str, Any]:
        return {"mean": data.mean(), "std": data.std()}

    def _generate_price_recommendations(self, score: float, field: str) -> List[str]:
        return [] if score > 0.9 else ["Review price data reasonableness"]

    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> float:
        return 0.95  # Placeholder

    def _generate_ohlc_recommendations(self, score: float) -> List[str]:
        return [] if score > 0.95 else ["Review OHLC price consistency"]

    def _analyze_price_jumps(self, data: pd.Series) -> float:
        return 0.85  # Placeholder

    def _generate_price_jump_recommendations(self, score: float) -> List[str]:
        return [] if score > 0.85 else ["Review extreme price movements"]

    def _analyze_volume_data(self, data: pd.DataFrame) -> List[QualityMetric]:
        return []  # Placeholder

    def _analyze_market_regime_coverage(self, data: pd.DataFrame, execution_data: Dict) -> List[QualityMetric]:
        return []  # Placeholder

    def _validate_statistical_distributions(self, data: pd.DataFrame) -> List[QualityMetric]:
        return []  # Placeholder

    def _generate_outlier_recommendations(self, percentage: float, field: str) -> List[str]:
        if percentage > 10:
            return [f"High outlier percentage in {field} - review data cleaning procedures"]
        return []


# Factory function
def create_data_quality_validator(
    db_manager: DatabaseManager,
    **kwargs
) -> DataQualityValidator:
    """Create DataQualityValidator instance"""
    return DataQualityValidator(db_manager=db_manager, **kwargs)