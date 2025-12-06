"""
Infrastructure Service - Client for the System Monitor API.
Provides methods to fetch container status, metrics, and alerts.
"""
import os
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Get system monitor URL from environment
SYSTEM_MONITOR_URL = os.getenv("SYSTEM_MONITOR_URL", "http://system-monitor:8095")


class InfrastructureService:
    """Client for the System Monitor API."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or SYSTEM_MONITOR_URL
        self.timeout = 10  # seconds

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request to the system monitor API."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", self.timeout)

        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error to system monitor: {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to system monitor: {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from system monitor: {e}")
            return None
        except Exception as e:
            logger.error(f"Error making request to system monitor: {e}")
            return None

    def is_available(self) -> bool:
        """Check if the system monitor service is available."""
        result = self._make_request("GET", "/health")
        return result is not None and result.get("status") == "healthy"

    def get_system_status(self) -> Optional[Dict]:
        """Get overall system status summary."""
        return self._make_request("GET", "/api/v1/status")

    def get_all_containers(self, include_stopped: bool = True) -> Optional[Dict]:
        """Get health information for all containers."""
        params = {"include_stopped": include_stopped}
        return self._make_request("GET", "/api/v1/containers", params=params)

    def get_container_detail(self, container_name: str) -> Optional[Dict]:
        """Get detailed information for a specific container."""
        return self._make_request("GET", f"/api/v1/containers/{container_name}")

    def get_container_logs(self, container_name: str, lines: int = 100) -> Optional[Dict]:
        """Get recent logs from a container."""
        params = {"lines": lines}
        return self._make_request("GET", f"/api/v1/containers/{container_name}/logs", params=params)

    def restart_container(self, container_name: str) -> Optional[Dict]:
        """Restart a container."""
        return self._make_request("POST", f"/api/v1/containers/{container_name}/restart")

    def get_current_metrics(self) -> Optional[Dict]:
        """Get current metrics snapshot for all containers."""
        return self._make_request("GET", "/api/v1/metrics")

    def get_metrics_history(self, container_name: str = None, hours: int = 24) -> Optional[Dict]:
        """Get historical metrics."""
        params = {"hours": hours}
        if container_name:
            params["container_name"] = container_name
        return self._make_request("GET", "/api/v1/metrics/history", params=params)

    def get_alerts(self, limit: int = 50, active_only: bool = False) -> Optional[Dict]:
        """Get recent alerts."""
        params = {"limit": limit, "active_only": active_only}
        return self._make_request("GET", "/api/v1/alerts", params=params)

    def get_active_alerts(self) -> Optional[Dict]:
        """Get currently active alerts."""
        return self._make_request("GET", "/api/v1/alerts/active")

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str = "admin") -> Optional[Dict]:
        """Acknowledge an alert."""
        data = {"acknowledged_by": acknowledged_by}
        return self._make_request("POST", f"/api/v1/alerts/{alert_id}/acknowledge", json=data)

    def resolve_alert(self, alert_id: int) -> Optional[Dict]:
        """Resolve an alert."""
        return self._make_request("POST", f"/api/v1/alerts/{alert_id}/resolve")

    def get_health_checks(self) -> Optional[Dict]:
        """Get results from all service health checks."""
        return self._make_request("GET", "/api/v1/health-checks")

    def test_notification(self, channel: str = "telegram") -> Optional[Dict]:
        """Test a notification channel."""
        data = {"channel": channel}
        return self._make_request("POST", "/api/v1/test-notification", json=data)

    def get_config(self) -> Optional[Dict]:
        """Get monitor configuration."""
        return self._make_request("GET", "/api/v1/config")


# Singleton instance
_infrastructure_service: Optional[InfrastructureService] = None


def get_infrastructure_service() -> InfrastructureService:
    """Get or create the infrastructure service instance."""
    global _infrastructure_service
    if _infrastructure_service is None:
        _infrastructure_service = InfrastructureService()
    return _infrastructure_service


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable string."""
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.1f} KB"
    elif bytes_value < 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_value / (1024 * 1024 * 1024):.2f} GB"


def format_uptime(seconds: int) -> str:
    """Format uptime seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}d {hours}h"


def get_status_color(status: str) -> str:
    """Get color for status display."""
    status_colors = {
        "running": "#28a745",
        "healthy": "#28a745",
        "stopped": "#dc3545",
        "exited": "#dc3545",
        "unhealthy": "#dc3545",
        "restarting": "#ffc107",
        "degraded": "#fd7e14",
        "warning": "#ffc107",
        "paused": "#6c757d",
        "created": "#6c757d",
    }
    return status_colors.get(status.lower(), "#6c757d")


def get_severity_color(severity: str) -> str:
    """Get color for alert severity."""
    severity_colors = {
        "critical": "#dc3545",
        "warning": "#ffc107",
        "info": "#17a2b8",
    }
    return severity_colors.get(severity.lower(), "#6c757d")
