# forex_scanner/validation/realtime_validation_monitor.py
"""
Real-time Validation Monitor

This module provides continuous monitoring capabilities for trading strategy
validation, enabling real-time assessment of strategy performance degradation,
data quality issues, and validation metric drift.

Key Features:
1. Continuous performance monitoring
2. Real-time alerting system
3. Validation metric drift detection
4. Automated re-validation triggering
5. Performance degradation alerts
6. Data quality monitoring
7. Statistical control charts
8. Threshold-based notifications
"""

import logging
import numpy as np
import pandas as pd
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from core.database import DatabaseManager
    from .statistical_validation_framework import StatisticalValidationFramework
    from .validation_report_generator import ValidationReportGenerator
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.validation.statistical_validation_framework import StatisticalValidationFramework
    from forex_scanner.validation.validation_report_generator import ValidationReportGenerator


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class MonitoringMetric(Enum):
    """Metrics to monitor"""
    PERFORMANCE_CORRELATION = "performance_correlation"
    VALIDATION_SCORE_DRIFT = "validation_score_drift"
    DATA_QUALITY_DEGRADATION = "data_quality_degradation"
    OVERFITTING_DETECTION = "overfitting_detection"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    PIPELINE_CONSISTENCY = "pipeline_consistency"


@dataclass
class Alert:
    """Container for monitoring alerts"""
    alert_id: str
    severity: AlertSeverity
    metric: MonitoringMetric
    message: str
    details: Dict[str, Any]
    threshold_value: float
    actual_value: float
    execution_id: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class MonitoringThreshold:
    """Threshold configuration for monitoring"""
    metric: MonitoringMetric
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    evaluation_window_minutes: int = 60
    min_samples: int = 10


@dataclass
class ValidationStatus:
    """Current validation status"""
    execution_id: int
    overall_score: float
    component_scores: Dict[str, float]
    last_validation: datetime
    status: str
    active_alerts: List[Alert]
    performance_trend: str
    degradation_detected: bool


class RealtimeValidationMonitor:
    """
    Real-time Validation Monitor

    Continuously monitors trading strategy validation metrics in real-time,
    detecting performance degradation, data quality issues, and other
    validation concerns as they occur.
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 statistical_framework: StatisticalValidationFramework,
                 report_generator: ValidationReportGenerator,
                 monitoring_interval_seconds: int = 300,  # 5 minutes
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.statistical_framework = statistical_framework
        self.report_generator = report_generator
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self.logger = logger or logging.getLogger(__name__)

        # Monitoring state
        self.is_monitoring = False
        self.monitored_executions = set()
        self.active_alerts = {}
        self.metric_history = {}
        self.alert_callbacks = []

        # Threading
        self.monitor_thread = None
        self.stop_event = threading.Event()

        # Default thresholds
        self.default_thresholds = {
            MonitoringMetric.PERFORMANCE_CORRELATION: MonitoringThreshold(
                metric=MonitoringMetric.PERFORMANCE_CORRELATION,
                warning_threshold=0.7,
                critical_threshold=0.5,
                emergency_threshold=0.3,
                evaluation_window_minutes=60,
                min_samples=20
            ),
            MonitoringMetric.VALIDATION_SCORE_DRIFT: MonitoringThreshold(
                metric=MonitoringMetric.VALIDATION_SCORE_DRIFT,
                warning_threshold=0.1,  # 10% drift
                critical_threshold=0.2,  # 20% drift
                emergency_threshold=0.3,  # 30% drift
                evaluation_window_minutes=120,
                min_samples=10
            ),
            MonitoringMetric.DATA_QUALITY_DEGRADATION: MonitoringThreshold(
                metric=MonitoringMetric.DATA_QUALITY_DEGRADATION,
                warning_threshold=0.85,
                critical_threshold=0.7,
                emergency_threshold=0.5,
                evaluation_window_minutes=30,
                min_samples=5
            )
        }

        self.logger.info(f"📊 Real-time Validation Monitor initialized:")
        self.logger.info(f"   Monitoring interval: {self.monitoring_interval_seconds}s")
        self.logger.info(f"   Default thresholds: {len(self.default_thresholds)} metrics")

    def start_monitoring(self, execution_ids: List[int] = None):
        """Start real-time monitoring for specified executions"""

        if self.is_monitoring:
            self.logger.warning("Monitor is already running")
            return

        self.logger.info("🚀 Starting real-time validation monitoring...")

        # Set monitored executions
        if execution_ids:
            self.monitored_executions.update(execution_ids)
        else:
            # Monitor all active executions
            self.monitored_executions.update(self._get_active_executions())

        # Initialize monitoring state
        self.is_monitoring = True
        self.stop_event.clear()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info(f"✅ Monitoring started for {len(self.monitored_executions)} executions")

    def stop_monitoring(self):
        """Stop real-time monitoring"""

        if not self.is_monitoring:
            self.logger.warning("Monitor is not running")
            return

        self.logger.info("⏹️ Stopping real-time validation monitoring...")

        # Signal stop
        self.is_monitoring = False
        self.stop_event.set()

        # Wait for thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)

        self.logger.info("✅ Monitoring stopped")

    def add_execution_to_monitoring(self, execution_id: int):
        """Add execution to monitoring list"""
        self.monitored_executions.add(execution_id)
        self.logger.info(f"Added execution {execution_id} to monitoring")

    def remove_execution_from_monitoring(self, execution_id: int):
        """Remove execution from monitoring list"""
        self.monitored_executions.discard(execution_id)
        # Clean up alerts for this execution
        if execution_id in self.active_alerts:
            del self.active_alerts[execution_id]
        self.logger.info(f"Removed execution {execution_id} from monitoring")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""

        execution_statuses = {}
        for execution_id in self.monitored_executions:
            execution_statuses[str(execution_id)] = self._get_execution_status(execution_id)

        return {
            'is_monitoring': self.is_monitoring,
            'monitoring_interval_seconds': self.monitoring_interval_seconds,
            'monitored_executions': list(self.monitored_executions),
            'total_active_alerts': sum(len(alerts) for alerts in self.active_alerts.values()),
            'execution_statuses': execution_statuses,
            'last_check': datetime.now(timezone.utc).isoformat()
        }

    def get_active_alerts(self, execution_id: Optional[int] = None) -> List[Alert]:
        """Get active alerts for specific execution or all executions"""

        if execution_id:
            return self.active_alerts.get(execution_id, [])

        # Return all alerts
        all_alerts = []
        for alerts in self.active_alerts.values():
            all_alerts.extend(alerts)

        return sorted(all_alerts, key=lambda x: x.timestamp, reverse=True)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""

        for execution_alerts in self.active_alerts.values():
            for alert in execution_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    self.logger.info(f"Alert {alert_id} acknowledged")
                    return True

        self.logger.warning(f"Alert {alert_id} not found")
        return False

    def _monitoring_loop(self):
        """Main monitoring loop"""

        self.logger.info("📊 Monitoring loop started")

        try:
            while self.is_monitoring and not self.stop_event.is_set():
                start_time = time.time()

                # Check all monitored executions
                for execution_id in list(self.monitored_executions):
                    try:
                        self._check_execution_metrics(execution_id)
                    except Exception as e:
                        self.logger.error(f"Error monitoring execution {execution_id}: {e}")

                # Clean up resolved alerts
                self._cleanup_resolved_alerts()

                # Calculate sleep time to maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval_seconds - elapsed)

                if sleep_time > 0:
                    self.stop_event.wait(sleep_time)

        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
        finally:
            self.logger.info("📊 Monitoring loop stopped")

    def _check_execution_metrics(self, execution_id: int):
        """Check all metrics for a specific execution"""

        try:
            # Get current validation results
            current_results = self.statistical_framework.validate_backtest_execution(execution_id)

            if current_results.get('status') == 'error':
                return

            # Update metric history
            self._update_metric_history(execution_id, current_results)

            # Check each monitored metric
            for metric_type, threshold_config in self.default_thresholds.items():
                self._evaluate_metric_threshold(execution_id, metric_type, threshold_config, current_results)

            # Check for performance correlation with live trading
            self._check_performance_correlation(execution_id)

            # Check for validation score drift
            self._check_validation_score_drift(execution_id)

            # Check data quality degradation
            self._check_data_quality_degradation(execution_id, current_results)

        except Exception as e:
            self.logger.error(f"Error checking metrics for execution {execution_id}: {e}")

    def _update_metric_history(self, execution_id: int, validation_results: Dict[str, Any]):
        """Update metric history for trend analysis"""

        if execution_id not in self.metric_history:
            self.metric_history[execution_id] = {
                'timestamps': deque(maxlen=100),
                'overall_scores': deque(maxlen=100),
                'component_scores': {},
                'data_quality_scores': deque(maxlen=100)
            }

        history = self.metric_history[execution_id]
        current_time = datetime.now(timezone.utc)

        # Update overall score
        overall_score = validation_results.get('overall_validation', {}).get('composite_score', 0.0)
        history['timestamps'].append(current_time)
        history['overall_scores'].append(overall_score)

        # Update component scores
        component_scores = validation_results.get('overall_validation', {}).get('component_scores', {})
        for component, score in component_scores.items():
            if component not in history['component_scores']:
                history['component_scores'][component] = deque(maxlen=100)
            history['component_scores'][component].append(score)

        # Update data quality score
        data_quality_score = validation_results.get('validation_components', {}).get(
            'data_quality', {}
        ).get('overall_quality_score', 1.0)
        history['data_quality_scores'].append(data_quality_score)

    def _check_performance_correlation(self, execution_id: int):
        """Check performance correlation between backtest and live trading"""

        try:
            # Get recent correlation analysis
            correlation_results = self._get_recent_correlation_analysis(execution_id)

            if not correlation_results:
                return

            correlation_score = correlation_results.get('correlation_metrics', {}).get('pearson_correlation', 1.0)
            threshold = self.default_thresholds[MonitoringMetric.PERFORMANCE_CORRELATION]

            # Check thresholds
            severity = None
            if correlation_score < threshold.emergency_threshold:
                severity = AlertSeverity.EMERGENCY
            elif correlation_score < threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif correlation_score < threshold.warning_threshold:
                severity = AlertSeverity.WARNING

            if severity:
                alert = Alert(
                    alert_id=f"corr_{execution_id}_{int(time.time())}",
                    severity=severity,
                    metric=MonitoringMetric.PERFORMANCE_CORRELATION,
                    message=f"Performance correlation degraded to {correlation_score:.3f}",
                    details=correlation_results,
                    threshold_value=threshold.warning_threshold,
                    actual_value=correlation_score,
                    execution_id=execution_id
                )

                self._trigger_alert(alert)

        except Exception as e:
            self.logger.error(f"Error checking performance correlation: {e}")

    def _check_validation_score_drift(self, execution_id: int):
        """Check for drift in validation scores over time"""

        try:
            if execution_id not in self.metric_history:
                return

            history = self.metric_history[execution_id]
            scores = list(history['overall_scores'])

            if len(scores) < 10:  # Need minimum history
                return

            # Calculate drift using linear regression
            x = np.arange(len(scores))
            y = np.array(scores)

            # Simple linear regression
            slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0

            # Calculate drift magnitude (negative slope indicates degradation)
            drift_magnitude = abs(slope)
            threshold = self.default_thresholds[MonitoringMetric.VALIDATION_SCORE_DRIFT]

            severity = None
            if drift_magnitude > threshold.emergency_threshold:
                severity = AlertSeverity.EMERGENCY
            elif drift_magnitude > threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif drift_magnitude > threshold.warning_threshold:
                severity = AlertSeverity.WARNING

            if severity:
                alert = Alert(
                    alert_id=f"drift_{execution_id}_{int(time.time())}",
                    severity=severity,
                    metric=MonitoringMetric.VALIDATION_SCORE_DRIFT,
                    message=f"Validation score drift detected: {drift_magnitude:.3f} per period",
                    details={
                        'slope': slope,
                        'drift_magnitude': drift_magnitude,
                        'recent_scores': scores[-10:],
                        'trend_direction': 'degrading' if slope < 0 else 'improving'
                    },
                    threshold_value=threshold.warning_threshold,
                    actual_value=drift_magnitude,
                    execution_id=execution_id
                )

                self._trigger_alert(alert)

        except Exception as e:
            self.logger.error(f"Error checking validation score drift: {e}")

    def _check_data_quality_degradation(self, execution_id: int, validation_results: Dict[str, Any]):
        """Check for data quality degradation"""

        try:
            data_quality_score = validation_results.get('validation_components', {}).get(
                'data_quality', {}
            ).get('overall_quality_score', 1.0)

            threshold = self.default_thresholds[MonitoringMetric.DATA_QUALITY_DEGRADATION]

            severity = None
            if data_quality_score < threshold.emergency_threshold:
                severity = AlertSeverity.EMERGENCY
            elif data_quality_score < threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif data_quality_score < threshold.warning_threshold:
                severity = AlertSeverity.WARNING

            if severity:
                alert = Alert(
                    alert_id=f"quality_{execution_id}_{int(time.time())}",
                    severity=severity,
                    metric=MonitoringMetric.DATA_QUALITY_DEGRADATION,
                    message=f"Data quality degraded to {data_quality_score:.3f}",
                    details=validation_results.get('validation_components', {}).get('data_quality', {}),
                    threshold_value=threshold.warning_threshold,
                    actual_value=data_quality_score,
                    execution_id=execution_id
                )

                self._trigger_alert(alert)

        except Exception as e:
            self.logger.error(f"Error checking data quality: {e}")

    def _trigger_alert(self, alert: Alert):
        """Trigger an alert"""

        # Add to active alerts
        if alert.execution_id not in self.active_alerts:
            self.active_alerts[alert.execution_id] = []

        self.active_alerts[alert.execution_id].append(alert)

        # Log alert
        self.logger.warning(f"🚨 ALERT [{alert.severity.value}] {alert.message}")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

        # Store alert in database
        self._store_alert(alert)

    def _evaluate_metric_threshold(self,
                                 execution_id: int,
                                 metric_type: MonitoringMetric,
                                 threshold_config: MonitoringThreshold,
                                 current_results: Dict[str, Any]):
        """Evaluate a specific metric against its threshold"""

        # This is a placeholder for metric-specific evaluation logic
        # Each metric would have its own evaluation method
        pass

    def _get_recent_correlation_analysis(self, execution_id: int) -> Dict[str, Any]:
        """Get recent correlation analysis results"""
        # Placeholder - would retrieve recent correlation analysis
        return {}

    def _cleanup_resolved_alerts(self):
        """Clean up resolved alerts"""

        for execution_id in list(self.active_alerts.keys()):
            # Remove resolved alerts older than 24 hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            self.active_alerts[execution_id] = [
                alert for alert in self.active_alerts[execution_id]
                if not alert.resolved and alert.timestamp > cutoff_time
            ]

            # Remove empty alert lists
            if not self.active_alerts[execution_id]:
                del self.active_alerts[execution_id]

    def _get_active_executions(self) -> List[int]:
        """Get list of active backtest executions"""

        try:
            query = """
            SELECT id FROM backtest_executions
            WHERE status IN ('running', 'completed')
            AND created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
            """
            result = self.db_manager.execute_query(query)
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting active executions: {e}")
            return []

    def _get_execution_status(self, execution_id: int) -> ValidationStatus:
        """Get current validation status for an execution"""

        # Get latest validation results
        try:
            validation_results = self.statistical_framework.validate_backtest_execution(execution_id)
            overall_score = validation_results.get('overall_validation', {}).get('composite_score', 0.0)
            component_scores = validation_results.get('overall_validation', {}).get('component_scores', {})

            # Determine performance trend
            performance_trend = "stable"
            degradation_detected = False

            if execution_id in self.metric_history:
                recent_scores = list(self.metric_history[execution_id]['overall_scores'])
                if len(recent_scores) >= 5:
                    # Simple trend analysis
                    recent_avg = np.mean(recent_scores[-3:])
                    older_avg = np.mean(recent_scores[-6:-3]) if len(recent_scores) >= 6 else recent_avg

                    if recent_avg < older_avg - 0.05:
                        performance_trend = "degrading"
                        degradation_detected = True
                    elif recent_avg > older_avg + 0.05:
                        performance_trend = "improving"

            return ValidationStatus(
                execution_id=execution_id,
                overall_score=overall_score,
                component_scores=component_scores,
                last_validation=datetime.now(timezone.utc),
                status="monitored",
                active_alerts=self.active_alerts.get(execution_id, []),
                performance_trend=performance_trend,
                degradation_detected=degradation_detected
            )

        except Exception as e:
            self.logger.error(f"Error getting execution status: {e}")
            return ValidationStatus(
                execution_id=execution_id,
                overall_score=0.0,
                component_scores={},
                last_validation=datetime.now(timezone.utc),
                status="error",
                active_alerts=[],
                performance_trend="unknown",
                degradation_detected=False
            )

    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            # Store alert for audit trail and historical analysis
            self.logger.debug(f"Storing alert {alert.alert_id} in database")
            # TODO: Implement database storage
        except Exception as e:
            self.logger.error(f"Error storing alert: {e}")


# Factory function
def create_realtime_validation_monitor(
    db_manager: DatabaseManager,
    statistical_framework: StatisticalValidationFramework,
    report_generator: ValidationReportGenerator,
    **kwargs
) -> RealtimeValidationMonitor:
    """Create RealtimeValidationMonitor instance"""
    return RealtimeValidationMonitor(
        db_manager=db_manager,
        statistical_framework=statistical_framework,
        report_generator=report_generator,
        **kwargs
    )