"""
Health status models for container monitoring.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ContainerStatus(str, Enum):
    """Docker container status."""
    RUNNING = "running"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    PAUSED = "paused"
    EXITED = "exited"
    DEAD = "dead"
    CREATED = "created"
    UNKNOWN = "unknown"


class HealthStatus(str, Enum):
    """Container health check status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    NONE = "none"
    UNKNOWN = "unknown"


class ContainerMetrics(BaseModel):
    """Resource metrics for a container."""
    cpu_percent: float = 0.0
    memory_bytes: int = 0
    memory_limit: int = 0
    memory_percent: float = 0.0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    block_read_bytes: int = 0
    block_write_bytes: int = 0
    pids: int = 0


class ContainerHealth(BaseModel):
    """Complete health information for a single container."""
    name: str
    container_id: str
    status: ContainerStatus
    health_status: HealthStatus
    image: str
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    uptime_seconds: int = 0
    uptime_human: str = ""
    restart_count: int = 0
    exit_code: Optional[int] = None
    ports: Dict[str, Any] = {}
    metrics: ContainerMetrics = ContainerMetrics()
    health_endpoint: Optional[str] = None
    health_check_response_time_ms: Optional[float] = None
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    description: str = ""
    is_critical: bool = False
    warnings: List[str] = []
    errors: List[str] = []

    def get_status_emoji(self) -> str:
        """Get emoji representation of status."""
        if self.status != ContainerStatus.RUNNING:
            return "🔴"
        if self.health_status == HealthStatus.UNHEALTHY:
            return "🔴"
        if self.health_status == HealthStatus.DEGRADED or self.warnings:
            return "🟡"
        return "🟢"


class SystemHealth(BaseModel):
    """Overall system health summary."""
    timestamp: datetime
    total_containers: int = 0
    running_containers: int = 0
    stopped_containers: int = 0
    unhealthy_containers: int = 0
    warning_containers: int = 0
    health_score: float = 100.0
    containers: List[ContainerHealth] = []
    active_alerts: int = 0
    critical_issues: List[str] = []
    warnings: List[str] = []

    def get_overall_status(self) -> str:
        """Get overall system status string."""
        if self.critical_issues:
            return "critical"
        if self.unhealthy_containers > 0 or self.stopped_containers > 0:
            return "degraded"
        if self.warning_containers > 0:
            return "warning"
        return "healthy"

    def get_status_emoji(self) -> str:
        """Get emoji for overall status."""
        status = self.get_overall_status()
        return {
            "critical": "🔴",
            "degraded": "🟠",
            "warning": "🟡",
            "healthy": "🟢",
        }.get(status, "⚪")
