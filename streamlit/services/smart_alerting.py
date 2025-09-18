"""
Smart Alerting System - Intelligent monitoring and notification for critical log patterns
Provides proactive alerts, pattern detection, and escalation for trading system issues
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time
import threading
import json
from collections import defaultdict, deque

from enhanced_log_parser import EnhancedLogParser, ParsedLogEntry, AlertType, LogLevel
from signal_intelligence import SignalIntelligenceService, TimeRange

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertCategory(Enum):
    SYSTEM_ERROR = "system_error"
    SIGNAL_ANOMALY = "signal_anomaly"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_ISSUE = "security_issue"
    DATA_QUALITY = "data_quality"
    CONNECTIVITY = "connectivity"

@dataclass
class AlertRule:
    """Configuration for an alert rule"""
    id: str
    name: str
    description: str
    category: AlertCategory
    severity: AlertSeverity
    pattern: str  # Regex pattern or keyword
    condition: str  # Condition logic (e.g., "count > 5 in 10m")
    enabled: bool = True
    cooldown_minutes: int = 30  # Minimum time between similar alerts
    escalation_threshold: int = 3  # Number of occurrences before escalation

@dataclass
class Alert:
    """Representation of a triggered alert"""
    id: str
    rule_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    description: str
    log_entries: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class AlertSummary:
    """Summary of alert activity"""
    total_alerts: int = 0
    critical_alerts: int = 0
    high_alerts: int = 0
    medium_alerts: int = 0
    low_alerts: int = 0
    unacknowledged_alerts: int = 0
    recent_alerts: List[Alert] = None

class SmartAlertingSystem:
    """
    Intelligent alerting system for trading system monitoring

    Features:
    - Pattern-based alert rules
    - Smart frequency analysis
    - Alert escalation and cooldowns
    - Real-time monitoring
    - Alert acknowledgment and resolution
    """

    def __init__(self, intelligence_service: SignalIntelligenceService = None):
        """Initialize smart alerting system"""
        self.logger = logging.getLogger(__name__)

        # Core services
        self.intelligence = intelligence_service or SignalIntelligenceService()
        self.log_parser = EnhancedLogParser()

        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self.alert_counters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_alert_times: Dict[str, datetime] = {}

        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Initialize default rules
        self._initialize_default_rules()

        self.logger.info("✅ Smart Alerting System initialized")

    def _initialize_default_rules(self):
        """Initialize default alert rules for common scenarios"""

        default_rules = [
            AlertRule(
                id="critical_errors",
                name="Critical System Errors",
                description="Multiple critical errors detected in short timeframe",
                category=AlertCategory.SYSTEM_ERROR,
                severity=AlertSeverity.CRITICAL,
                pattern=r"ERROR.*(?:failed|exception|critical|fatal)",
                condition="count >= 3 in 5m",
                escalation_threshold=5
            ),
            AlertRule(
                id="signal_rejection_spike",
                name="Signal Rejection Spike",
                description="Unusually high number of signal rejections",
                category=AlertCategory.SIGNAL_ANOMALY,
                severity=AlertSeverity.HIGH,
                pattern=r"signal.*rejected",
                condition="count >= 10 in 15m",
                escalation_threshold=2
            ),
            AlertRule(
                id="connection_failures",
                name="Connection Failures",
                description="Database or API connection failures",
                category=AlertCategory.CONNECTIVITY,
                severity=AlertSeverity.HIGH,
                pattern=r"(?:connection|auth).*(?:failed|lost|timeout)",
                condition="count >= 2 in 10m",
                escalation_threshold=3
            ),
            AlertRule(
                id="data_quality_issues",
                name="Data Quality Issues",
                description="Missing or invalid data detected",
                category=AlertCategory.DATA_QUALITY,
                severity=AlertSeverity.MEDIUM,
                pattern=r"(?:invalid|missing|corrupt).*data",
                condition="count >= 5 in 30m",
                cooldown_minutes=60
            ),
            AlertRule(
                id="performance_degradation",
                name="Performance Degradation",
                description="System performance issues detected",
                category=AlertCategory.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.MEDIUM,
                pattern=r"(?:slow|timeout|lag|delay)",
                condition="count >= 8 in 20m",
                cooldown_minutes=45
            ),
            AlertRule(
                id="authentication_failures",
                name="Authentication Failures",
                description="Multiple authentication failures",
                category=AlertCategory.SECURITY_ISSUE,
                severity=AlertSeverity.HIGH,
                pattern=r"auth.*(?:failed|denied|invalid)",
                condition="count >= 3 in 15m",
                escalation_threshold=2
            )
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        self.alert_rules[rule.id] = rule
        self.logger.info(f"✅ Alert rule added: {rule.name}")

    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"🗑️ Alert rule removed: {rule_id}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)

    def start_monitoring(self, check_interval: int = 30):
        """Start real-time monitoring for alert conditions"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"🔍 Smart alerting monitoring started (interval: {check_interval}s)")

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("⏹️ Smart alerting monitoring stopped")

    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_alert_conditions()
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)

    def _check_alert_conditions(self):
        """Check all alert rules against recent log data"""
        try:
            # Get recent log entries (last 2 hours for pattern analysis)
            recent_logs = self.log_parser.get_recent_logs(
                hours_back=2,
                max_entries=500
            )

            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue

                # Check if rule is in cooldown
                if self._is_in_cooldown(rule):
                    continue

                # Find matching log entries
                matching_logs = self._find_matching_logs(recent_logs, rule)

                if matching_logs:
                    # Evaluate condition
                    if self._evaluate_condition(rule, matching_logs):
                        self._trigger_alert(rule, matching_logs)

        except Exception as e:
            self.logger.error(f"Error checking alert conditions: {e}")

    def _find_matching_logs(self, logs: List[ParsedLogEntry], rule: AlertRule) -> List[ParsedLogEntry]:
        """Find log entries matching the alert rule pattern"""
        import re

        matching_logs = []
        pattern = re.compile(rule.pattern, re.IGNORECASE)

        for log in logs:
            if pattern.search(log.message):
                matching_logs.append(log)

        return matching_logs

    def _evaluate_condition(self, rule: AlertRule, matching_logs: List[ParsedLogEntry]) -> bool:
        """Evaluate if the alert condition is met"""
        try:
            # Parse condition (e.g., "count >= 3 in 5m")
            condition_parts = rule.condition.split()

            if len(condition_parts) >= 5 and condition_parts[0] == "count":
                operator = condition_parts[1]
                threshold = int(condition_parts[2])
                time_unit = condition_parts[4]

                # Parse time unit
                if time_unit.endswith('m'):
                    minutes = int(time_unit[:-1])
                elif time_unit.endswith('h'):
                    minutes = int(time_unit[:-1]) * 60
                else:
                    minutes = 30  # Default

                # Filter logs within time window
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                recent_matches = [
                    log for log in matching_logs
                    if log.timestamp >= cutoff_time
                ]

                count = len(recent_matches)

                # Evaluate operator
                if operator == ">=":
                    return count >= threshold
                elif operator == ">":
                    return count > threshold
                elif operator == "==":
                    return count == threshold
                elif operator == "<=":
                    return count <= threshold
                elif operator == "<":
                    return count < threshold

        except Exception as e:
            self.logger.error(f"Error evaluating condition for rule {rule.id}: {e}")

        return False

    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period"""
        last_alert_time = self.last_alert_times.get(rule.id)
        if last_alert_time:
            cooldown_end = last_alert_time + timedelta(minutes=rule.cooldown_minutes)
            return datetime.now() < cooldown_end
        return False

    def _trigger_alert(self, rule: AlertRule, matching_logs: List[ParsedLogEntry]):
        """Trigger an alert for the given rule"""
        try:
            # Create alert
            alert_id = f"{rule.id}_{int(datetime.now().timestamp())}"

            # Prepare log data for serialization
            log_data = []
            for log in matching_logs[-10:]:  # Include last 10 matching logs
                log_data.append({
                    'timestamp': log.timestamp.isoformat(),
                    'level': log.level,
                    'module': log.module,
                    'message': log.message,
                    'epic': log.epic,
                    'alert_type': log.alert_type
                })

            # Create metadata
            metadata = {
                'pattern': rule.pattern,
                'condition': rule.condition,
                'match_count': len(matching_logs),
                'recent_match_count': len([
                    log for log in matching_logs
                    if log.timestamp >= datetime.now() - timedelta(minutes=60)
                ])
            }

            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                timestamp=datetime.now(),
                severity=rule.severity,
                category=rule.category,
                title=rule.name,
                description=f"{rule.description} - {len(matching_logs)} matches found",
                log_entries=log_data,
                metadata=metadata
            )

            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.last_alert_times[rule.id] = alert.timestamp

            # Log alert
            self.logger.warning(
                f"🚨 ALERT TRIGGERED: {alert.title} "
                f"[{alert.severity.value.upper()}] - {len(matching_logs)} matches"
            )

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")

        except Exception as e:
            self.logger.error(f"Error triggering alert for rule {rule.id}: {e}")

    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"✅ Alert acknowledged: {alert_id} by {user}")
            return True
        return False

    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.acknowledged = True

            # Move to history and remove from active
            del self.active_alerts[alert_id]

            self.logger.info(f"✅ Alert resolved: {alert_id} by {user}")
            return True
        return False

    def get_alert_summary(self, hours_back: int = 24) -> AlertSummary:
        """Get summary of alert activity"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # Get recent alerts from history
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]

        # Count by severity
        severity_counts = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 0,
            AlertSeverity.MEDIUM: 0,
            AlertSeverity.LOW: 0
        }

        for alert in recent_alerts:
            severity_counts[alert.severity] += 1

        # Count unacknowledged
        unacknowledged = len([
            alert for alert in self.active_alerts.values()
            if not alert.acknowledged
        ])

        return AlertSummary(
            total_alerts=len(recent_alerts),
            critical_alerts=severity_counts[AlertSeverity.CRITICAL],
            high_alerts=severity_counts[AlertSeverity.HIGH],
            medium_alerts=severity_counts[AlertSeverity.MEDIUM],
            low_alerts=severity_counts[AlertSeverity.LOW],
            unacknowledged_alerts=unacknowledged,
            recent_alerts=sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)[:20]
        )

    def get_active_alerts(self, severity_filter: AlertSeverity = None) -> List[Alert]:
        """Get currently active alerts"""
        alerts = list(self.active_alerts.values())

        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_rules(self) -> List[AlertRule]:
        """Get all configured alert rules"""
        return list(self.alert_rules.values())

    def export_alert_config(self) -> str:
        """Export alert configuration to JSON"""
        config = {
            'rules': [asdict(rule) for rule in self.alert_rules.values()],
            'export_timestamp': datetime.now().isoformat()
        }
        return json.dumps(config, indent=2, default=str)

    def import_alert_config(self, config_json: str):
        """Import alert configuration from JSON"""
        try:
            config = json.loads(config_json)

            for rule_data in config.get('rules', []):
                # Convert string enums back to enum objects
                rule_data['category'] = AlertCategory(rule_data['category'])
                rule_data['severity'] = AlertSeverity(rule_data['severity'])

                rule = AlertRule(**rule_data)
                self.add_alert_rule(rule)

            self.logger.info(f"✅ Imported {len(config.get('rules', []))} alert rules")

        except Exception as e:
            self.logger.error(f"Error importing alert config: {e}")
            raise

    def test_alert_rule(self, rule_id: str) -> Dict[str, Any]:
        """Test an alert rule against recent data"""
        if rule_id not in self.alert_rules:
            return {'error': 'Rule not found'}

        rule = self.alert_rules[rule_id]

        # Get recent logs for testing
        recent_logs = self.log_parser.get_recent_logs(hours_back=4, max_entries=200)

        # Find matches
        matching_logs = self._find_matching_logs(recent_logs, rule)

        # Test condition
        condition_met = self._evaluate_condition(rule, matching_logs)

        return {
            'rule_name': rule.name,
            'total_logs_checked': len(recent_logs),
            'matching_logs': len(matching_logs),
            'condition_met': condition_met,
            'condition': rule.condition,
            'pattern': rule.pattern,
            'recent_matches': [
                {
                    'timestamp': log.timestamp.isoformat(),
                    'message': log.message[:100],
                    'level': log.level
                }
                for log in matching_logs[-5:]  # Last 5 matches
            ]
        }

def get_smart_alerting_system() -> SmartAlertingSystem:
    """Get a configured smart alerting system instance"""
    return SmartAlertingSystem()

# Example alert callback functions
def console_alert_callback(alert: Alert):
    """Simple console output for alerts"""
    print(f"\n🚨 ALERT: {alert.title}")
    print(f"   Severity: {alert.severity.value.upper()}")
    print(f"   Time: {alert.timestamp}")
    print(f"   Description: {alert.description}")
    print(f"   Matches: {len(alert.log_entries)}")

def email_alert_callback(alert: Alert):
    """Email notification callback (placeholder implementation)"""
    # This would integrate with your email service
    subject = f"[{alert.severity.value.upper()}] Trading System Alert: {alert.title}"
    body = f"""
    Alert: {alert.title}
    Severity: {alert.severity.value.upper()}
    Category: {alert.category.value}
    Time: {alert.timestamp}

    Description: {alert.description}

    Recent log entries ({len(alert.log_entries)} total):
    """ + "\n".join([f"- {entry['timestamp']}: {entry['message']}" for entry in alert.log_entries[-3:]])

    # Send email here
    print(f"EMAIL ALERT: {subject}")

def slack_alert_callback(alert: Alert):
    """Slack notification callback (placeholder implementation)"""
    # This would integrate with Slack webhook
    message = {
        "text": f"🚨 Trading System Alert",
        "attachments": [
            {
                "color": "danger" if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH] else "warning",
                "fields": [
                    {"title": "Alert", "value": alert.title, "short": True},
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                    {"title": "Matches", "value": str(len(alert.log_entries)), "short": True},
                    {"title": "Description", "value": alert.description, "short": False}
                ]
            }
        ]
    }

    # Send to Slack here
    print(f"SLACK ALERT: {alert.title}")

if __name__ == "__main__":
    # Test the smart alerting system
    alerting = SmartAlertingSystem()

    # Add some test callbacks
    alerting.add_alert_callback(console_alert_callback)

    print("Testing Smart Alerting System...")

    # Test a rule
    test_result = alerting.test_alert_rule("critical_errors")
    print(f"\nTest Result for 'critical_errors' rule:")
    for key, value in test_result.items():
        print(f"  {key}: {value}")

    # Get alert summary
    summary = alerting.get_alert_summary(hours_back=24)
    print(f"\nAlert Summary (24h):")
    print(f"  Total: {summary.total_alerts}")
    print(f"  Critical: {summary.critical_alerts}")
    print(f"  High: {summary.high_alerts}")
    print(f"  Medium: {summary.medium_alerts}")
    print(f"  Low: {summary.low_alerts}")
    print(f"  Unacknowledged: {summary.unacknowledged_alerts}")

    # Get active alerts
    active = alerting.get_active_alerts()
    print(f"\nActive Alerts: {len(active)}")

    print("\n🔍 Starting monitoring for 30 seconds...")
    alerting.start_monitoring(check_interval=10)
    time.sleep(30)
    alerting.stop_monitoring()
    print("✅ Monitoring test completed")