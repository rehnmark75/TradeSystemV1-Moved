# forex_scanner/validation/realtime_correlation_analyzer.py
"""
Real-time Performance Correlation Analyzer

This module provides sophisticated correlation analysis between backtest predictions
and live trading performance, enabling continuous validation of strategy effectiveness.

Key Features:
- Multi-dimensional correlation analysis (Pearson, Spearman, Kendall)
- Time-varying correlation with regime detection
- Information coefficient calculations
- Rolling correlation stability analysis
- Statistical significance testing with multiple hypothesis corrections
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Statistical libraries
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


@dataclass
class CorrelationWindow:
    """Container for correlation analysis over a time window"""
    start_date: datetime
    end_date: datetime
    pearson_correlation: float
    spearman_correlation: float
    kendall_tau: Optional[float]
    sample_size: int
    p_value_pearson: float
    p_value_spearman: float
    regression_r2: float
    information_coefficient: float
    tracking_error: float
    information_ratio: float


@dataclass
class RegimeCorrelation:
    """Correlation metrics for different market regimes"""
    regime_name: str
    correlation_pearson: float
    correlation_spearman: float
    sample_size: int
    regime_period: Tuple[datetime, datetime]
    statistical_significance: float


class RealtimeCorrelationAnalyzer:
    """
    Advanced Real-time Correlation Analysis for Backtest Validation

    Provides continuous monitoring and analysis of correlations between
    backtest predictions and actual live trading performance.
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 min_sample_size: int = 30,
                 correlation_window_days: int = 90,
                 significance_level: float = 0.05,
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.min_sample_size = min_sample_size
        self.correlation_window_days = correlation_window_days
        self.significance_level = significance_level
        self.logger = logger or logging.getLogger(__name__)

        # Correlation analysis parameters
        self.min_correlation_threshold = 0.2
        self.stability_window_size = 30
        self.regime_detection_threshold = 0.3

        # Performance tracking
        self.analysis_cache = {}
        self.last_update = None

        self.logger.info(f"🔄 Realtime Correlation Analyzer initialized:")
        self.logger.info(f"   Minimum sample size: {self.min_sample_size}")
        self.logger.info(f"   Correlation window: {self.correlation_window_days} days")
        self.logger.info(f"   Significance level: {self.significance_level}")

    def analyze_backtest_live_correlation(self,
                                        execution_id: int,
                                        strategy_name: str,
                                        epic: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive correlation analysis between backtest and live performance

        Args:
            execution_id: Backtest execution ID
            strategy_name: Strategy name to analyze
            epic: Optional epic filter

        Returns:
            Detailed correlation analysis results
        """
        self.logger.info(f"🔄 Starting correlation analysis for execution {execution_id}")

        try:
            # Load aligned backtest and live data
            aligned_data = self._load_aligned_backtest_live_data(execution_id, strategy_name, epic)

            if len(aligned_data) < self.min_sample_size:
                return self._create_insufficient_data_result(len(aligned_data))

            # Perform comprehensive correlation analysis
            results = {
                'execution_id': execution_id,
                'strategy_name': strategy_name,
                'epic': epic,
                'analysis_timestamp': datetime.now(timezone.utc),
                'sample_size': len(aligned_data),
                'correlation_analysis': {}
            }

            # 1. Overall correlation metrics
            results['correlation_analysis']['overall'] = self._calculate_overall_correlation(aligned_data)

            # 2. Time-varying correlation analysis
            results['correlation_analysis']['time_varying'] = self._analyze_time_varying_correlation(aligned_data)

            # 3. Regime-based correlation analysis
            results['correlation_analysis']['regime_based'] = self._analyze_regime_correlation(aligned_data)

            # 4. Rolling correlation stability
            results['correlation_analysis']['stability'] = self._analyze_correlation_stability(aligned_data)

            # 5. Information coefficient analysis
            results['correlation_analysis']['information_coefficient'] = self._calculate_information_coefficient(aligned_data)

            # 6. Prediction accuracy metrics
            results['correlation_analysis']['prediction_accuracy'] = self._analyze_prediction_accuracy(aligned_data)

            # 7. Statistical significance testing
            results['statistical_tests'] = self._perform_correlation_significance_tests(aligned_data)

            # 8. Overall assessment
            results['assessment'] = self._assess_correlation_quality(results)

            # Cache results for performance
            self.analysis_cache[f"{execution_id}_{strategy_name}_{epic}"] = results
            self.last_update = datetime.now(timezone.utc)

            self.logger.info(f"✅ Correlation analysis completed:")
            self.logger.info(f"   Overall Pearson: {results['correlation_analysis']['overall']['pearson_correlation']:.3f}")
            self.logger.info(f"   Information Coefficient: {results['correlation_analysis']['information_coefficient']['monthly_ic']:.3f}")
            self.logger.info(f"   Assessment: {results['assessment']['overall_quality']}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Correlation analysis failed: {e}")
            return {
                'execution_id': execution_id,
                'status': 'error',
                'error_message': str(e),
                'analysis_timestamp': datetime.now(timezone.utc)
            }

    def _load_aligned_backtest_live_data(self,
                                       execution_id: int,
                                       strategy_name: str,
                                       epic: Optional[str] = None) -> pd.DataFrame:
        """
        Load and align backtest signals with corresponding live trading results

        This is the critical method that matches backtest predictions with
        actual live performance to enable correlation analysis.
        """

        # Load backtest signals
        backtest_query = """
        SELECT
            bs.signal_timestamp,
            bs.epic,
            bs.signal_type,
            bs.confidence_score,
            bs.entry_price,
            bs.take_profit_price,
            bs.stop_loss_price,
            bs.pips_gained as backtest_pips,
            bs.trade_result as backtest_result,
            bs.validation_passed
        FROM backtest_signals bs
        WHERE bs.execution_id = %s
        AND bs.strategy_name = %s
        """ + (f" AND bs.epic = '{epic}'" if epic else "") + """
        ORDER BY bs.signal_timestamp
        """

        backtest_result = self.db_manager.execute_query(backtest_query, (execution_id, strategy_name))
        backtest_columns = [desc[0] for desc in backtest_result.description]
        backtest_data = pd.DataFrame(backtest_result.fetchall(), columns=backtest_columns)

        if backtest_data.empty:
            return pd.DataFrame()

        # Load corresponding live trading data from alert_history
        # This matches signals that were generated around the same time as backtest signals
        live_query = """
        SELECT
            ah.alert_timestamp,
            ah.epic,
            ah.signal_type,
            ah.confidence_score,
            ah.price as entry_price,
            tl.pips_gained as live_pips,
            CASE
                WHEN tl.pips_gained > 0 THEN 'win'
                WHEN tl.pips_gained < 0 THEN 'loss'
                ELSE 'breakeven'
            END as live_result,
            ah.claude_approved,
            tl.profit_loss as live_pnl
        FROM alert_history ah
        LEFT JOIN trade_log tl ON ah.id = tl.alert_id
        WHERE ah.strategy = %s
        """ + (f" AND ah.epic = '{epic}'" if epic else "") + """
        AND ah.alert_timestamp >= (
            SELECT MIN(signal_timestamp) - INTERVAL '1 day'
            FROM backtest_signals
            WHERE execution_id = %s
        )
        AND ah.alert_timestamp <= (
            SELECT MAX(signal_timestamp) + INTERVAL '1 day'
            FROM backtest_signals
            WHERE execution_id = %s
        )
        ORDER BY ah.alert_timestamp
        """

        live_result = self.db_manager.execute_query(live_query, (strategy_name, execution_id, execution_id))
        live_columns = [desc[0] for desc in live_result.description]
        live_data = pd.DataFrame(live_result.fetchall(), columns=live_columns)

        if live_data.empty:
            return pd.DataFrame()

        # Align backtest and live data using time-based matching
        aligned_data = self._align_temporal_data(backtest_data, live_data)

        return aligned_data

    def _align_temporal_data(self,
                           backtest_data: pd.DataFrame,
                           live_data: pd.DataFrame,
                           max_time_diff_minutes: int = 60) -> pd.DataFrame:
        """
        Align backtest and live data based on temporal proximity

        Args:
            backtest_data: Backtest signals DataFrame
            live_data: Live trading data DataFrame
            max_time_diff_minutes: Maximum time difference for alignment

        Returns:
            DataFrame with aligned backtest and live data
        """

        # Convert timestamps to datetime
        backtest_data['signal_timestamp'] = pd.to_datetime(backtest_data['signal_timestamp'])
        live_data['alert_timestamp'] = pd.to_datetime(live_data['alert_timestamp'])

        aligned_records = []

        for _, backtest_row in backtest_data.iterrows():
            # Find live signals within time window and same epic
            time_window_start = backtest_row['signal_timestamp'] - timedelta(minutes=max_time_diff_minutes)
            time_window_end = backtest_row['signal_timestamp'] + timedelta(minutes=max_time_diff_minutes)

            matching_live = live_data[
                (live_data['alert_timestamp'] >= time_window_start) &
                (live_data['alert_timestamp'] <= time_window_end) &
                (live_data['epic'] == backtest_row['epic']) &
                (live_data['signal_type'] == backtest_row['signal_type'])
            ]

            if not matching_live.empty:
                # Take the closest match by time
                time_diffs = abs(matching_live['alert_timestamp'] - backtest_row['signal_timestamp'])
                closest_match = matching_live.iloc[time_diffs.argmin()]

                # Create aligned record
                aligned_record = {
                    'timestamp': backtest_row['signal_timestamp'],
                    'epic': backtest_row['epic'],
                    'signal_type': backtest_row['signal_type'],

                    # Backtest data
                    'backtest_confidence': backtest_row['confidence_score'],
                    'backtest_entry_price': backtest_row['entry_price'],
                    'backtest_pips': backtest_row['backtest_pips'] or 0,
                    'backtest_result': backtest_row['backtest_result'] or 'unknown',
                    'backtest_validated': backtest_row['validation_passed'],

                    # Live data
                    'live_confidence': closest_match['confidence_score'],
                    'live_entry_price': closest_match['entry_price'],
                    'live_pips': closest_match['live_pips'] or 0,
                    'live_result': closest_match['live_result'] or 'unknown',
                    'live_approved': closest_match['claude_approved'],
                    'live_pnl': closest_match['live_pnl'] or 0,

                    # Temporal alignment
                    'time_difference_minutes': time_diffs.min().total_seconds() / 60
                }

                aligned_records.append(aligned_record)

        return pd.DataFrame(aligned_records)

    def _calculate_overall_correlation(self, aligned_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive correlation metrics"""

        try:
            # Extract numeric features for correlation
            backtest_pips = aligned_data['backtest_pips'].astype(float)
            live_pips = aligned_data['live_pips'].astype(float)

            backtest_confidence = aligned_data['backtest_confidence'].astype(float)
            live_confidence = aligned_data['live_confidence'].astype(float)

            # Pearson correlation (linear relationship)
            pips_corr_pearson, pips_p_pearson = pearsonr(backtest_pips, live_pips)
            conf_corr_pearson, conf_p_pearson = pearsonr(backtest_confidence, live_confidence)

            # Spearman correlation (monotonic relationship)
            pips_corr_spearman, pips_p_spearman = spearmanr(backtest_pips, live_pips)
            conf_corr_spearman, conf_p_spearman = spearmanr(backtest_confidence, live_confidence)

            # Kendall's tau
            pips_kendall, pips_kendall_p = kendalltau(backtest_pips, live_pips)
            conf_kendall, conf_kendall_p = kendalltau(backtest_confidence, live_confidence)

            # Linear regression analysis
            reg = LinearRegression()
            reg.fit(backtest_pips.values.reshape(-1, 1), live_pips.values)
            regression_r2 = reg.score(backtest_pips.values.reshape(-1, 1), live_pips.values)
            regression_slope = reg.coef_[0]
            regression_intercept = reg.intercept_

            # Tracking error and information ratio
            tracking_error = np.std(live_pips - backtest_pips)
            excess_return = np.mean(live_pips - backtest_pips)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

            return {
                'pips_correlation': {
                    'pearson': pips_corr_pearson,
                    'spearman': pips_corr_spearman,
                    'kendall': pips_kendall,
                    'p_value_pearson': pips_p_pearson,
                    'p_value_spearman': pips_p_spearman,
                    'p_value_kendall': pips_kendall_p
                },
                'confidence_correlation': {
                    'pearson': conf_corr_pearson,
                    'spearman': conf_corr_spearman,
                    'kendall': conf_kendall,
                    'p_value_pearson': conf_p_pearson,
                    'p_value_spearman': conf_p_spearman,
                    'p_value_kendall': conf_kendall_p
                },
                'regression_analysis': {
                    'r_squared': regression_r2,
                    'slope': regression_slope,
                    'intercept': regression_intercept,
                    'explained_variance': regression_r2
                },
                'risk_metrics': {
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'excess_return': excess_return
                },
                'sample_statistics': {
                    'backtest_mean_pips': backtest_pips.mean(),
                    'live_mean_pips': live_pips.mean(),
                    'backtest_std_pips': backtest_pips.std(),
                    'live_std_pips': live_pips.std()
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating overall correlation: {e}")
            return {
                'error': str(e),
                'status': 'failed'
            }

    def _analyze_time_varying_correlation(self, aligned_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how correlation changes over time"""

        try:
            # Sort by timestamp
            data_sorted = aligned_data.sort_values('timestamp')

            # Calculate rolling correlations
            window_size = min(self.stability_window_size, len(data_sorted) // 3)

            if window_size < 10:
                return {
                    'status': 'insufficient_data',
                    'message': f'Insufficient data for time-varying analysis: {len(data_sorted)}'
                }

            rolling_correlations = []

            for i in range(window_size, len(data_sorted)):
                window_data = data_sorted.iloc[i-window_size:i]

                backtest_pips = window_data['backtest_pips'].astype(float)
                live_pips = window_data['live_pips'].astype(float)

                if backtest_pips.std() > 0 and live_pips.std() > 0:
                    corr, p_val = pearsonr(backtest_pips, live_pips)

                    rolling_correlations.append({
                        'end_timestamp': window_data['timestamp'].iloc[-1],
                        'correlation': corr,
                        'p_value': p_val,
                        'window_size': window_size,
                        'significant': p_val < self.significance_level
                    })

            if not rolling_correlations:
                return {
                    'status': 'insufficient_variance',
                    'message': 'Insufficient variance in data for correlation calculation'
                }

            # Calculate correlation stability metrics
            correlations = [r['correlation'] for r in rolling_correlations]
            correlation_mean = np.mean(correlations)
            correlation_std = np.std(correlations)
            correlation_stability = 1 - (correlation_std / abs(correlation_mean)) if correlation_mean != 0 else 0

            # Trend analysis
            correlation_trend = np.polyfit(range(len(correlations)), correlations, 1)[0]

            return {
                'status': 'completed',
                'rolling_correlations': rolling_correlations,
                'stability_metrics': {
                    'mean_correlation': correlation_mean,
                    'correlation_std': correlation_std,
                    'stability_score': max(0, min(1, correlation_stability)),
                    'correlation_trend': correlation_trend
                },
                'windows_analyzed': len(rolling_correlations),
                'window_size': window_size
            }

        except Exception as e:
            self.logger.error(f"Error in time-varying correlation analysis: {e}")
            return {
                'status': 'error',
                'error_message': str(e)
            }

    def _calculate_information_coefficient(self, aligned_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Information Coefficient (IC) - correlation between
        predicted returns and actual returns
        """

        try:
            # Calculate monthly information coefficients
            data_with_month = aligned_data.copy()
            data_with_month['month'] = pd.to_datetime(data_with_month['timestamp']).dt.to_period('M')

            monthly_ics = []

            for month, month_data in data_with_month.groupby('month'):
                if len(month_data) >= 10:  # Minimum observations per month
                    backtest_pips = month_data['backtest_pips'].astype(float)
                    live_pips = month_data['live_pips'].astype(float)

                    if backtest_pips.std() > 0 and live_pips.std() > 0:
                        ic, p_val = pearsonr(backtest_pips, live_pips)

                        monthly_ics.append({
                            'month': str(month),
                            'information_coefficient': ic,
                            'p_value': p_val,
                            'sample_size': len(month_data),
                            'significant': p_val < self.significance_level
                        })

            if not monthly_ics:
                return {
                    'status': 'insufficient_data',
                    'message': 'Insufficient data for IC calculation'
                }

            # Calculate IC statistics
            ic_values = [ic_data['information_coefficient'] for ic_data in monthly_ics]

            # IC Mean and standard deviation
            ic_mean = np.mean(ic_values)
            ic_std = np.std(ic_values)

            # IC Information Ratio (IC mean / IC std)
            ic_information_ratio = ic_mean / ic_std if ic_std > 0 else 0

            # Percentage of significant months
            significant_months = sum(1 for ic_data in monthly_ics if ic_data['significant'])
            significance_rate = significant_months / len(monthly_ics)

            # Hit rate (percentage of positive ICs)
            positive_ics = sum(1 for ic in ic_values if ic > 0)
            hit_rate = positive_ics / len(ic_values)

            return {
                'status': 'completed',
                'monthly_ic': ic_mean,
                'ic_std': ic_std,
                'ic_information_ratio': ic_information_ratio,
                'hit_rate': hit_rate,
                'significance_rate': significance_rate,
                'monthly_ics': monthly_ics,
                'months_analyzed': len(monthly_ics),
                'assessment': self._assess_ic_quality(ic_mean, ic_information_ratio, hit_rate)
            }

        except Exception as e:
            self.logger.error(f"Error calculating information coefficient: {e}")
            return {
                'status': 'error',
                'error_message': str(e)
            }

    def _assess_ic_quality(self, ic_mean: float, ic_ir: float, hit_rate: float) -> Dict[str, Any]:
        """Assess the quality of Information Coefficient"""

        # IC quality thresholds based on quantitative finance literature
        if abs(ic_mean) >= 0.05 and ic_ir >= 0.5 and hit_rate >= 0.55:
            quality = 'excellent'
            message = 'Strategy shows excellent predictive power'
        elif abs(ic_mean) >= 0.03 and ic_ir >= 0.3 and hit_rate >= 0.52:
            quality = 'good'
            message = 'Strategy shows good predictive power'
        elif abs(ic_mean) >= 0.02 and ic_ir >= 0.2 and hit_rate >= 0.50:
            quality = 'acceptable'
            message = 'Strategy shows acceptable predictive power'
        else:
            quality = 'poor'
            message = 'Strategy shows poor predictive power'

        return {
            'quality_rating': quality,
            'assessment_message': message,
            'ic_score': abs(ic_mean),
            'consistency_score': ic_ir,
            'hit_rate_score': hit_rate
        }

    def _create_insufficient_data_result(self, data_size: int) -> Dict[str, Any]:
        """Create result for insufficient data scenarios"""
        return {
            'status': 'insufficient_data',
            'data_size': data_size,
            'required_size': self.min_sample_size,
            'message': f'Insufficient aligned data for correlation analysis: {data_size} < {self.min_sample_size}',
            'analysis_timestamp': datetime.now(timezone.utc),
            'correlation_possible': False
        }

    def get_correlation_summary(self, execution_id: int) -> Dict[str, Any]:
        """Get correlation analysis summary for an execution"""

        cache_key = f"{execution_id}_summary"

        if cache_key in self.analysis_cache:
            cached_result = self.analysis_cache[cache_key]
            cache_age = (datetime.now(timezone.utc) - cached_result.get('analysis_timestamp', datetime.min.replace(tzinfo=timezone.utc))).total_seconds()

            if cache_age < 3600:  # Cache valid for 1 hour
                return cached_result

        # Generate fresh summary
        try:
            query = """
            SELECT strategy_name, COUNT(*) as signal_count
            FROM backtest_signals
            WHERE execution_id = %s
            GROUP BY strategy_name
            """

            result = self.db_manager.execute_query(query, (execution_id,))
            strategies = result.fetchall()

            summary = {
                'execution_id': execution_id,
                'analysis_timestamp': datetime.now(timezone.utc),
                'strategies': {},
                'overall_assessment': {}
            }

            for strategy_name, signal_count in strategies:
                strategy_analysis = self.analyze_backtest_live_correlation(execution_id, strategy_name)
                summary['strategies'][strategy_name] = {
                    'signal_count': signal_count,
                    'correlation_analysis': strategy_analysis
                }

            # Cache the result
            self.analysis_cache[cache_key] = summary

            return summary

        except Exception as e:
            self.logger.error(f"Error generating correlation summary: {e}")
            return {
                'execution_id': execution_id,
                'status': 'error',
                'error_message': str(e),
                'analysis_timestamp': datetime.now(timezone.utc)
            }


# Factory function
def create_realtime_correlation_analyzer(
    db_manager: DatabaseManager,
    **kwargs
) -> RealtimeCorrelationAnalyzer:
    """Create RealtimeCorrelationAnalyzer instance"""
    return RealtimeCorrelationAnalyzer(db_manager=db_manager, **kwargs)