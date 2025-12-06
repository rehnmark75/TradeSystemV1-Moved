"""
Docker monitoring service - connects to Docker daemon to monitor containers.
"""
import docker
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from ..models import ContainerStatus, HealthStatus, ContainerHealth, ContainerMetrics, SystemHealth
from ..config import settings, CONTAINER_DESCRIPTIONS

logger = logging.getLogger(__name__)


class DockerMonitor:
    """Service for monitoring Docker containers via Docker API."""

    def __init__(self):
        self.client = None
        self._connect()

    def _connect(self):
        """Connect to Docker daemon."""
        try:
            self.client = docker.from_env()
            self.client.ping()
            logger.info("Connected to Docker daemon")
        except Exception as e:
            logger.error(f"Failed to connect to Docker daemon: {e}")
            self.client = None

    def is_connected(self) -> bool:
        """Check if connected to Docker daemon."""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def _parse_container_status(self, status: str) -> ContainerStatus:
        """Parse Docker status string to ContainerStatus enum."""
        status_map = {
            "running": ContainerStatus.RUNNING,
            "exited": ContainerStatus.EXITED,
            "paused": ContainerStatus.PAUSED,
            "restarting": ContainerStatus.RESTARTING,
            "dead": ContainerStatus.DEAD,
            "created": ContainerStatus.CREATED,
        }
        return status_map.get(status.lower(), ContainerStatus.UNKNOWN)

    def _parse_health_status(self, health: Optional[Dict]) -> HealthStatus:
        """Parse Docker health check status."""
        if not health:
            return HealthStatus.NONE
        status = health.get("Status", "").lower()
        status_map = {
            "healthy": HealthStatus.HEALTHY,
            "unhealthy": HealthStatus.UNHEALTHY,
            "starting": HealthStatus.STARTING,
        }
        return status_map.get(status, HealthStatus.UNKNOWN)

    def _calculate_uptime(self, started_at: Optional[str]) -> tuple:
        """Calculate uptime in seconds and human-readable format."""
        if not started_at:
            return 0, "Unknown"

        try:
            # Parse Docker timestamp (e.g., "2025-01-15T10:30:00.123456789Z")
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = now - started
            seconds = int(delta.total_seconds())

            # Human-readable format
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            minutes = (seconds % 3600) // 60

            if days > 0:
                human = f"{days}d {hours}h"
            elif hours > 0:
                human = f"{hours}h {minutes}m"
            else:
                human = f"{minutes}m"

            return seconds, human
        except Exception as e:
            logger.warning(f"Failed to parse started_at: {e}")
            return 0, "Unknown"

    def _get_container_metrics(self, container) -> ContainerMetrics:
        """Get resource metrics for a container."""
        try:
            stats = container.stats(stream=False)

            # CPU calculation
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                        stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                           stats["precpu_stats"]["system_cpu_usage"]
            num_cpus = stats["cpu_stats"].get("online_cpus", 1) or len(
                stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1])
            )

            cpu_percent = 0.0
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0

            # Memory calculation
            memory_stats = stats.get("memory_stats", {})
            memory_bytes = memory_stats.get("usage", 0)
            memory_limit = memory_stats.get("limit", 1)
            memory_percent = (memory_bytes / memory_limit) * 100.0 if memory_limit > 0 else 0.0

            # Network calculation
            networks = stats.get("networks", {})
            network_rx = sum(net.get("rx_bytes", 0) for net in networks.values())
            network_tx = sum(net.get("tx_bytes", 0) for net in networks.values())

            # Block I/O
            blkio = stats.get("blkio_stats", {}).get("io_service_bytes_recursive", []) or []
            block_read = sum(item.get("value", 0) for item in blkio if item.get("op") == "read")
            block_write = sum(item.get("value", 0) for item in blkio if item.get("op") == "write")

            # PIDs
            pids = stats.get("pids_stats", {}).get("current", 0)

            return ContainerMetrics(
                cpu_percent=round(cpu_percent, 2),
                memory_bytes=memory_bytes,
                memory_limit=memory_limit,
                memory_percent=round(memory_percent, 2),
                network_rx_bytes=network_rx,
                network_tx_bytes=network_tx,
                block_read_bytes=block_read,
                block_write_bytes=block_write,
                pids=pids,
            )
        except Exception as e:
            logger.warning(f"Failed to get metrics for container: {e}")
            return ContainerMetrics()

    def get_container_health(self, container_name: str) -> Optional[ContainerHealth]:
        """Get health information for a specific container."""
        if not self.is_connected():
            self._connect()
            if not self.is_connected():
                return None

        try:
            container = self.client.containers.get(container_name)
            attrs = container.attrs
            state = attrs.get("State", {})

            # Parse status
            status = self._parse_container_status(state.get("Status", "unknown"))

            # Parse health
            health_data = state.get("Health")
            health_status = self._parse_health_status(health_data)

            # Calculate uptime
            started_at = state.get("StartedAt")
            uptime_seconds, uptime_human = self._calculate_uptime(started_at)

            # Get metrics (only for running containers)
            metrics = ContainerMetrics()
            if status == ContainerStatus.RUNNING:
                metrics = self._get_container_metrics(container)

            # Parse ports
            ports = {}
            network_settings = attrs.get("NetworkSettings", {}).get("Ports", {})
            for container_port, host_bindings in network_settings.items():
                if host_bindings:
                    ports[container_port] = host_bindings[0].get("HostPort", "")

            # Determine warnings
            warnings = []
            if metrics.cpu_percent > settings.cpu_warning_threshold:
                warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            if metrics.memory_percent > settings.memory_warning_threshold:
                warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")

            # Determine errors
            errors = []
            if status != ContainerStatus.RUNNING:
                errors.append(f"Container not running: {status.value}")
            if health_status == HealthStatus.UNHEALTHY:
                errors.append("Health check failing")

            return ContainerHealth(
                name=container_name,
                container_id=container.short_id,
                status=status,
                health_status=health_status,
                image=attrs.get("Config", {}).get("Image", "unknown"),
                created_at=attrs.get("Created"),
                started_at=started_at,
                uptime_seconds=uptime_seconds,
                uptime_human=uptime_human,
                restart_count=state.get("RestartCount", 0),
                exit_code=state.get("ExitCode"),
                ports=ports,
                metrics=metrics,
                description=CONTAINER_DESCRIPTIONS.get(container_name, ""),
                is_critical=container_name in settings.critical_containers,
                warnings=warnings,
                errors=errors,
            )
        except docker.errors.NotFound:
            logger.warning(f"Container not found: {container_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to get container health for {container_name}: {e}")
            return None

    def get_all_containers(self, include_stopped: bool = True) -> List[ContainerHealth]:
        """Get health information for all containers."""
        if not self.is_connected():
            self._connect()
            if not self.is_connected():
                return []

        containers = []
        try:
            docker_containers = self.client.containers.list(all=include_stopped)
            for container in docker_containers:
                health = self.get_container_health(container.name)
                if health:
                    containers.append(health)
        except Exception as e:
            logger.error(f"Failed to list containers: {e}")

        return containers

    def get_system_health(self) -> SystemHealth:
        """Get overall system health summary."""
        containers = self.get_all_containers()

        running = sum(1 for c in containers if c.status == ContainerStatus.RUNNING)
        stopped = sum(1 for c in containers if c.status != ContainerStatus.RUNNING)
        unhealthy = sum(1 for c in containers if c.health_status == HealthStatus.UNHEALTHY)
        warning = sum(1 for c in containers if c.warnings)

        # Calculate health score
        total = len(containers)
        if total > 0:
            # Weight critical containers more heavily
            score = 100.0
            for c in containers:
                weight = 2.0 if c.is_critical else 1.0
                if c.status != ContainerStatus.RUNNING:
                    score -= (15.0 * weight)
                elif c.health_status == HealthStatus.UNHEALTHY:
                    score -= (10.0 * weight)
                elif c.warnings:
                    score -= (5.0 * weight)
            score = max(0.0, score)
        else:
            score = 0.0

        # Collect critical issues and warnings
        critical_issues = []
        warnings_list = []
        for c in containers:
            for error in c.errors:
                if c.is_critical:
                    critical_issues.append(f"{c.name}: {error}")
                else:
                    warnings_list.append(f"{c.name}: {error}")
            for warning in c.warnings:
                warnings_list.append(f"{c.name}: {warning}")

        return SystemHealth(
            timestamp=datetime.now(timezone.utc),
            total_containers=total,
            running_containers=running,
            stopped_containers=stopped,
            unhealthy_containers=unhealthy,
            warning_containers=warning,
            health_score=round(score, 1),
            containers=containers,
            critical_issues=critical_issues,
            warnings=warnings_list,
        )

    def get_container_logs(self, container_name: str, lines: int = 100) -> str:
        """Get recent logs from a container."""
        if not self.is_connected():
            return "Not connected to Docker daemon"

        try:
            container = self.client.containers.get(container_name)
            logs = container.logs(tail=lines, timestamps=True).decode("utf-8")
            return logs
        except docker.errors.NotFound:
            return f"Container not found: {container_name}"
        except Exception as e:
            return f"Error getting logs: {e}"

    def restart_container(self, container_name: str) -> tuple:
        """Restart a container. Returns (success, message)."""
        if not self.is_connected():
            return False, "Not connected to Docker daemon"

        try:
            container = self.client.containers.get(container_name)
            container.restart(timeout=30)
            logger.info(f"Restarted container: {container_name}")
            return True, f"Container {container_name} restarted successfully"
        except docker.errors.NotFound:
            return False, f"Container not found: {container_name}"
        except Exception as e:
            logger.error(f"Failed to restart container {container_name}: {e}")
            return False, f"Failed to restart: {e}"
