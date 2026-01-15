"""
Alert manager service - handles alert creation, deduplication, and notifications.
"""
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from ..config import settings
from ..models import Alert, AlertSeverity, AlertType, ContainerHealth, SystemHealth, ContainerStatus, HealthStatus

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages system alerts with deduplication and notification routing."""

    def __init__(self, metrics_collector=None, telegram_notifier=None, email_notifier=None):
        self.metrics_collector = metrics_collector
        self.telegram_notifier = telegram_notifier
        self.email_notifier = email_notifier

        # Track recent alerts for deduplication
        self._recent_alerts: Dict[str, datetime] = {}
        # Track container states for recovery detection
        self._container_states: Dict[str, Dict] = {}
        # Track restart counts for restart loop detection
        self._restart_history: Dict[str, List[datetime]] = {}

    def _get_alert_key(self, alert_type: AlertType, container_name: Optional[str]) -> str:
        """Generate unique key for alert deduplication."""
        return f"{alert_type.value}:{container_name or 'system'}"

    def _should_alert(self, alert_key: str) -> bool:
        """Check if we should send this alert (not in cooldown)."""
        if alert_key not in self._recent_alerts:
            return True

        last_alert = self._recent_alerts[alert_key]
        cooldown = timedelta(seconds=settings.alert_cooldown)

        return datetime.now(timezone.utc) - last_alert > cooldown

    def _record_alert(self, alert_key: str):
        """Record that we sent an alert."""
        self._recent_alerts[alert_key] = datetime.now(timezone.utc)

    async def process_container_health(self, container: ContainerHealth) -> List[Alert]:
        """
        Process container health and generate any necessary alerts.
        Returns list of generated alerts.
        """
        alerts = []
        name = container.name

        # Get previous state
        prev_state = self._container_states.get(name, {})
        prev_status = prev_state.get("status")
        prev_health = prev_state.get("health_status")

        # Update current state
        self._container_states[name] = {
            "status": container.status,
            "health_status": container.health_status,
            "timestamp": datetime.now(timezone.utc),
        }

        # Check for container down
        if container.status != ContainerStatus.RUNNING:
            alert = await self._create_container_down_alert(container)
            if alert:
                alerts.append(alert)

        # Check for container recovery
        elif prev_status and prev_status != ContainerStatus.RUNNING:
            alert = await self._create_recovery_alert(container, prev_state)
            if alert:
                alerts.append(alert)

        # Check for health check failures
        if container.health_status == HealthStatus.UNHEALTHY:
            alert = await self._create_health_failure_alert(container)
            if alert:
                alerts.append(alert)

        # Check for health check recovery (was unhealthy, now healthy)
        elif container.health_status == HealthStatus.HEALTHY and prev_health == HealthStatus.UNHEALTHY:
            alert = await self._create_health_recovery_alert(container)
            if alert:
                alerts.append(alert)

        # Check for restart loops
        restart_alert = await self._check_restart_loop(container)
        if restart_alert:
            alerts.append(restart_alert)

        # Check for high CPU
        if container.metrics.cpu_percent > settings.cpu_critical_threshold:
            alert = await self._create_high_cpu_alert(container, critical=True)
            if alert:
                alerts.append(alert)
        elif container.metrics.cpu_percent > settings.cpu_warning_threshold:
            alert = await self._create_high_cpu_alert(container, critical=False)
            if alert:
                alerts.append(alert)

        # Check for high memory
        if container.metrics.memory_percent > settings.memory_critical_threshold:
            alert = await self._create_high_memory_alert(container, critical=True)
            if alert:
                alerts.append(alert)
        elif container.metrics.memory_percent > settings.memory_warning_threshold:
            alert = await self._create_high_memory_alert(container, critical=False)
            if alert:
                alerts.append(alert)

        return alerts

    async def _create_container_down_alert(self, container: ContainerHealth) -> Optional[Alert]:
        """Create alert for container down."""
        alert_key = self._get_alert_key(AlertType.CONTAINER_DOWN, container.name)

        if not self._should_alert(alert_key):
            return None

        severity = AlertSeverity.CRITICAL if container.is_critical else AlertSeverity.WARNING

        alert = Alert(
            alert_type=AlertType.CONTAINER_DOWN,
            severity=severity,
            container_name=container.name,
            title=f"Container Down: {container.name}",
            message=f"Container {container.name} is not running.\nStatus: {container.status.value}\nExit code: {container.exit_code}",
            details={
                "status": container.status.value,
                "exit_code": container.exit_code,
                "is_critical": container.is_critical,
                "description": container.description,
            },
        )

        await self._send_alert(alert)
        self._record_alert(alert_key)

        return alert

    async def _create_recovery_alert(self, container: ContainerHealth, prev_state: Dict) -> Optional[Alert]:
        """Create alert for container recovery."""
        alert_key = self._get_alert_key(AlertType.CONTAINER_RECOVERED, container.name)

        if not self._should_alert(alert_key):
            return None

        # Calculate downtime
        prev_timestamp = prev_state.get("timestamp")
        downtime = "unknown"
        if prev_timestamp:
            delta = datetime.now(timezone.utc) - prev_timestamp
            minutes = int(delta.total_seconds() / 60)
            if minutes < 60:
                downtime = f"{minutes} minutes"
            else:
                hours = minutes // 60
                downtime = f"{hours} hours, {minutes % 60} minutes"

        alert = Alert(
            alert_type=AlertType.CONTAINER_RECOVERED,
            severity=AlertSeverity.INFO,
            container_name=container.name,
            title=f"Container Recovered: {container.name}",
            message=f"Container {container.name} is now running.\nDowntime: {downtime}",
            details={
                "downtime": downtime,
                "previous_status": prev_state.get("status", "unknown"),
            },
        )

        await self._send_alert(alert)
        self._record_alert(alert_key)

        return alert

    async def _create_health_failure_alert(self, container: ContainerHealth) -> Optional[Alert]:
        """Create alert for health check failure."""
        alert_key = self._get_alert_key(AlertType.HEALTH_CHECK_FAILED, container.name)

        if not self._should_alert(alert_key):
            return None

        severity = AlertSeverity.WARNING
        if container.is_critical and container.consecutive_failures >= settings.health_check_failures_threshold:
            severity = AlertSeverity.CRITICAL

        alert = Alert(
            alert_type=AlertType.HEALTH_CHECK_FAILED,
            severity=severity,
            container_name=container.name,
            title=f"Health Check Failed: {container.name}",
            message=f"Container {container.name} health check is failing.\nConsecutive failures: {container.consecutive_failures}",
            details={
                "consecutive_failures": container.consecutive_failures,
                "health_endpoint": container.health_endpoint,
            },
        )

        await self._send_alert(alert)
        self._record_alert(alert_key)

        return alert

    async def _create_health_recovery_alert(self, container: ContainerHealth) -> Optional[Alert]:
        """Create alert for health check recovery."""
        alert_key = self._get_alert_key(AlertType.HEALTH_CHECK_RECOVERED, container.name)

        if not self._should_alert(alert_key):
            return None

        alert = Alert(
            alert_type=AlertType.HEALTH_CHECK_RECOVERED,
            severity=AlertSeverity.INFO,
            container_name=container.name,
            title=f"Health Check Recovered: {container.name}",
            message=f"Container {container.name} health check is now passing.",
            details={
                "health_endpoint": container.health_endpoint,
            },
        )

        await self._send_alert(alert)
        self._record_alert(alert_key)

        return alert

    async def _check_restart_loop(self, container: ContainerHealth) -> Optional[Alert]:
        """Check for restart loop and create alert if detected."""
        name = container.name

        # Initialize restart history for this container
        if name not in self._restart_history:
            self._restart_history[name] = []

        # Get current restart count from container
        current_restarts = container.restart_count

        # Check if restart count increased
        prev_state = self._container_states.get(name, {})
        prev_restarts = prev_state.get("restart_count", 0)

        if current_restarts > prev_restarts:
            # Record restart
            self._restart_history[name].append(datetime.now(timezone.utc))

        # Update state with restart count
        self._container_states[name]["restart_count"] = current_restarts

        # Clean up old restart records
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.restart_loop_window)
        self._restart_history[name] = [
            t for t in self._restart_history[name] if t > cutoff
        ]

        # Check for restart loop
        if len(self._restart_history[name]) >= settings.restart_loop_threshold:
            alert_key = self._get_alert_key(AlertType.RESTART_LOOP, container.name)

            if not self._should_alert(alert_key):
                return None

            severity = AlertSeverity.CRITICAL if container.is_critical else AlertSeverity.WARNING

            alert = Alert(
                alert_type=AlertType.RESTART_LOOP,
                severity=severity,
                container_name=container.name,
                title=f"Restart Loop Detected: {container.name}",
                message=f"Container {container.name} has restarted {len(self._restart_history[name])} times in the last {settings.restart_loop_window // 60} minutes.",
                details={
                    "restart_count": len(self._restart_history[name]),
                    "window_minutes": settings.restart_loop_window // 60,
                    "total_restarts": current_restarts,
                },
            )

            await self._send_alert(alert)
            self._record_alert(alert_key)

            return alert

        return None

    async def _create_high_cpu_alert(self, container: ContainerHealth, critical: bool) -> Optional[Alert]:
        """Create alert for high CPU usage."""
        alert_key = self._get_alert_key(AlertType.HIGH_CPU, container.name)

        if not self._should_alert(alert_key):
            return None

        severity = AlertSeverity.CRITICAL if critical else AlertSeverity.WARNING

        alert = Alert(
            alert_type=AlertType.HIGH_CPU,
            severity=severity,
            container_name=container.name,
            title=f"High CPU Usage: {container.name}",
            message=f"Container {container.name} CPU usage is at {container.metrics.cpu_percent:.1f}%",
            details={
                "cpu_percent": container.metrics.cpu_percent,
                "threshold": settings.cpu_critical_threshold if critical else settings.cpu_warning_threshold,
            },
        )

        await self._send_alert(alert)
        self._record_alert(alert_key)

        return alert

    async def _create_high_memory_alert(self, container: ContainerHealth, critical: bool) -> Optional[Alert]:
        """Create alert for high memory usage."""
        alert_key = self._get_alert_key(AlertType.HIGH_MEMORY, container.name)

        if not self._should_alert(alert_key):
            return None

        severity = AlertSeverity.CRITICAL if critical else AlertSeverity.WARNING

        # Format memory in human-readable form
        memory_mb = container.metrics.memory_bytes / (1024 * 1024)
        limit_mb = container.metrics.memory_limit / (1024 * 1024)

        alert = Alert(
            alert_type=AlertType.HIGH_MEMORY,
            severity=severity,
            container_name=container.name,
            title=f"High Memory Usage: {container.name}",
            message=f"Container {container.name} memory usage is at {container.metrics.memory_percent:.1f}% ({memory_mb:.0f}MB / {limit_mb:.0f}MB)",
            details={
                "memory_percent": container.metrics.memory_percent,
                "memory_mb": round(memory_mb, 1),
                "limit_mb": round(limit_mb, 1),
                "threshold": settings.memory_critical_threshold if critical else settings.memory_warning_threshold,
            },
        )

        await self._send_alert(alert)
        self._record_alert(alert_key)

        return alert

    async def _send_alert(self, alert: Alert):
        """Send alert through configured notification channels."""
        # Store in database
        if self.metrics_collector:
            alert_id = self.metrics_collector.store_alert(alert)
            if alert_id:
                alert.id = str(alert_id)

        # Determine channels based on severity
        channels = []
        if alert.severity == AlertSeverity.CRITICAL:
            channels = ["telegram", "email"]
        elif alert.severity == AlertSeverity.WARNING:
            channels = ["telegram"]
        else:
            channels = ["telegram"]

        # Send to Telegram
        if "telegram" in channels and self.telegram_notifier:
            try:
                await self.telegram_notifier.send_alert(alert)
                alert.notification_sent = True
                alert.notification_channels.append("telegram")
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")

        # Send to Email (critical only by default)
        if "email" in channels and self.email_notifier and settings.email_enabled:
            try:
                await self.email_notifier.send_alert(alert)
                alert.notification_channels.append("email")
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")

        logger.info(f"Alert sent: {alert.title} (severity: {alert.severity.value})")

    async def process_system_health(self, system_health: SystemHealth) -> List[Alert]:
        """Process system health and generate alerts for all containers."""
        all_alerts = []

        for container in system_health.containers:
            alerts = await self.process_container_health(container)
            all_alerts.extend(alerts)

        # Update active alert count in system health
        if self.metrics_collector:
            system_health.active_alerts = self.metrics_collector.get_active_alert_count()

        return all_alerts

    def get_recent_alerts(self, limit: int = 50, active_only: bool = False) -> List[Dict]:
        """Get recent alerts from storage."""
        if self.metrics_collector:
            return self.metrics_collector.get_recent_alerts(limit, active_only)
        return []

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str = "admin") -> bool:
        """Acknowledge an alert."""
        if self.metrics_collector:
            return self.metrics_collector.acknowledge_alert(alert_id, acknowledged_by)
        return False

    def resolve_alert(self, alert_id: int) -> bool:
        """Resolve an alert."""
        if self.metrics_collector:
            return self.metrics_collector.resolve_alert(alert_id)
        return False
