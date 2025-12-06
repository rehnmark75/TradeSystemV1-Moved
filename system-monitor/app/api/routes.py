"""
API routes for the System Monitor service.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# These will be injected by main.py
docker_monitor = None
health_checker = None
metrics_collector = None
alert_manager = None
telegram_notifier = None
email_notifier = None


class AcknowledgeRequest(BaseModel):
    acknowledged_by: str = "admin"


class TestNotificationRequest(BaseModel):
    channel: str = "telegram"  # telegram or email


@router.get("/health")
async def health_check():
    """Health check endpoint for the monitor service itself."""
    return {
        "status": "healthy",
        "service": "system-monitor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docker_connected": docker_monitor.is_connected() if docker_monitor else False,
    }


@router.get("/api/v1/status")
async def get_system_status():
    """Get overall system status summary."""
    if not docker_monitor:
        raise HTTPException(status_code=503, detail="Docker monitor not initialized")

    system_health = docker_monitor.get_system_health()

    # Add active alert count
    if metrics_collector:
        system_health.active_alerts = metrics_collector.get_active_alert_count()

    return {
        "status": system_health.get_overall_status(),
        "status_emoji": system_health.get_status_emoji(),
        "timestamp": system_health.timestamp.isoformat(),
        "health_score": system_health.health_score,
        "total_containers": system_health.total_containers,
        "running_containers": system_health.running_containers,
        "stopped_containers": system_health.stopped_containers,
        "unhealthy_containers": system_health.unhealthy_containers,
        "warning_containers": system_health.warning_containers,
        "active_alerts": system_health.active_alerts,
        "critical_issues": system_health.critical_issues,
        "warnings": system_health.warnings,
    }


@router.get("/api/v1/containers")
async def get_all_containers(include_stopped: bool = Query(True)):
    """Get health information for all containers."""
    if not docker_monitor:
        raise HTTPException(status_code=503, detail="Docker monitor not initialized")

    containers = docker_monitor.get_all_containers(include_stopped=include_stopped)

    return {
        "containers": [
            {
                "name": c.name,
                "container_id": c.container_id,
                "status": c.status.value,
                "status_emoji": c.get_status_emoji(),
                "health_status": c.health_status.value,
                "image": c.image,
                "uptime_seconds": c.uptime_seconds,
                "uptime_human": c.uptime_human,
                "restart_count": c.restart_count,
                "ports": c.ports,
                "metrics": {
                    "cpu_percent": c.metrics.cpu_percent,
                    "memory_percent": c.metrics.memory_percent,
                    "memory_bytes": c.metrics.memory_bytes,
                    "memory_limit": c.metrics.memory_limit,
                    "network_rx_bytes": c.metrics.network_rx_bytes,
                    "network_tx_bytes": c.metrics.network_tx_bytes,
                },
                "description": c.description,
                "is_critical": c.is_critical,
                "warnings": c.warnings,
                "errors": c.errors,
            }
            for c in containers
        ],
        "count": len(containers),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/containers/{container_name}")
async def get_container_detail(container_name: str):
    """Get detailed health information for a specific container."""
    if not docker_monitor:
        raise HTTPException(status_code=503, detail="Docker monitor not initialized")

    container = docker_monitor.get_container_health(container_name)

    if not container:
        raise HTTPException(status_code=404, detail=f"Container '{container_name}' not found")

    # Get metrics history
    metrics_history = []
    if metrics_collector:
        metrics_history = metrics_collector.get_container_metrics_history(container_name, hours=24)

    # Get uptime stats
    uptime_stats = {}
    if metrics_collector:
        uptime_stats = metrics_collector.get_uptime_stats(container_name, days=7)

    return {
        "name": container.name,
        "container_id": container.container_id,
        "status": container.status.value,
        "status_emoji": container.get_status_emoji(),
        "health_status": container.health_status.value,
        "image": container.image,
        "created_at": container.created_at,
        "started_at": container.started_at,
        "uptime_seconds": container.uptime_seconds,
        "uptime_human": container.uptime_human,
        "restart_count": container.restart_count,
        "exit_code": container.exit_code,
        "ports": container.ports,
        "metrics": {
            "cpu_percent": container.metrics.cpu_percent,
            "memory_percent": container.metrics.memory_percent,
            "memory_bytes": container.metrics.memory_bytes,
            "memory_limit": container.metrics.memory_limit,
            "network_rx_bytes": container.metrics.network_rx_bytes,
            "network_tx_bytes": container.metrics.network_tx_bytes,
            "block_read_bytes": container.metrics.block_read_bytes,
            "block_write_bytes": container.metrics.block_write_bytes,
            "pids": container.metrics.pids,
        },
        "health_endpoint": container.health_endpoint,
        "health_check_response_time_ms": container.health_check_response_time_ms,
        "consecutive_failures": container.consecutive_failures,
        "description": container.description,
        "is_critical": container.is_critical,
        "warnings": container.warnings,
        "errors": container.errors,
        "metrics_history": metrics_history,
        "uptime_stats": uptime_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/containers/{container_name}/logs")
async def get_container_logs(
    container_name: str,
    lines: int = Query(100, ge=1, le=1000)
):
    """Get recent logs from a container."""
    if not docker_monitor:
        raise HTTPException(status_code=503, detail="Docker monitor not initialized")

    logs = docker_monitor.get_container_logs(container_name, lines=lines)

    return {
        "container_name": container_name,
        "logs": logs,
        "lines_requested": lines,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/containers/{container_name}/restart")
async def restart_container(container_name: str):
    """Restart a container."""
    if not docker_monitor:
        raise HTTPException(status_code=503, detail="Docker monitor not initialized")

    success, message = docker_monitor.restart_container(container_name)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "container_name": container_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/metrics")
async def get_current_metrics():
    """Get current metrics snapshot for all containers."""
    if not docker_monitor:
        raise HTTPException(status_code=503, detail="Docker monitor not initialized")

    system_health = docker_monitor.get_system_health()

    return {
        "timestamp": system_health.timestamp.isoformat(),
        "containers": {
            c.name: {
                "cpu_percent": c.metrics.cpu_percent,
                "memory_percent": c.metrics.memory_percent,
                "memory_bytes": c.metrics.memory_bytes,
                "network_rx_bytes": c.metrics.network_rx_bytes,
                "network_tx_bytes": c.metrics.network_tx_bytes,
            }
            for c in system_health.containers
        },
    }


@router.get("/api/v1/metrics/history")
async def get_metrics_history(
    container_name: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168)  # Max 1 week
):
    """Get historical metrics."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")

    if container_name:
        history = metrics_collector.get_container_metrics_history(container_name, hours=hours)
        return {
            "container_name": container_name,
            "hours": hours,
            "metrics": history,
            "count": len(history),
        }
    else:
        history = metrics_collector.get_all_metrics_history(hours=hours)
        return {
            "hours": hours,
            "containers": history,
            "container_count": len(history),
        }


@router.get("/api/v1/alerts")
async def get_alerts(
    limit: int = Query(50, ge=1, le=200),
    active_only: bool = Query(False)
):
    """Get recent alerts."""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")

    alerts = alert_manager.get_recent_alerts(limit=limit, active_only=active_only)

    return {
        "alerts": alerts,
        "count": len(alerts),
        "active_only": active_only,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/alerts/active")
async def get_active_alerts():
    """Get currently active (unresolved) alerts."""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")

    alerts = alert_manager.get_recent_alerts(limit=100, active_only=True)

    return {
        "alerts": alerts,
        "count": len(alerts),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, request: AcknowledgeRequest):
    """Acknowledge an alert."""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")

    success = alert_manager.acknowledge_alert(alert_id, request.acknowledged_by)

    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    return {
        "success": True,
        "alert_id": alert_id,
        "acknowledged_by": request.acknowledged_by,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int):
    """Resolve an alert."""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")

    success = alert_manager.resolve_alert(alert_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    return {
        "success": True,
        "alert_id": alert_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/test-notification")
async def test_notification(request: TestNotificationRequest):
    """Test notification channel."""
    if request.channel == "telegram":
        if not telegram_notifier:
            raise HTTPException(status_code=503, detail="Telegram notifier not initialized")

        success, message = await telegram_notifier.test_connection()
        return {
            "channel": "telegram",
            "success": success,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    elif request.channel == "email":
        if not email_notifier:
            raise HTTPException(status_code=503, detail="Email notifier not initialized")

        success, message = await email_notifier.test_connection()
        return {
            "channel": "email",
            "success": success,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unknown channel: {request.channel}")


@router.get("/api/v1/health-checks")
async def get_health_check_results():
    """Get results from all service health checks."""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")

    results = await health_checker.check_all_services()

    return {
        "services": {
            name: {
                "status": info["status"].value,
                "response_time_ms": info["response_time_ms"],
                "error": info["error"],
                "consecutive_failures": info["state"].get("consecutive_failures", 0),
                "last_check": info["state"].get("last_check").isoformat() if info["state"].get("last_check") else None,
            }
            for name, info in results.items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/config")
async def get_monitor_config():
    """Get current monitor configuration (non-sensitive)."""
    from ..config import settings, HEALTH_ENDPOINTS, TCP_CHECK_CONTAINERS

    return {
        "monitor_interval_seconds": settings.monitor_interval,
        "alert_cooldown_seconds": settings.alert_cooldown,
        "metrics_retention_days": settings.metrics_retention_days,
        "thresholds": {
            "cpu_warning": settings.cpu_warning_threshold,
            "cpu_critical": settings.cpu_critical_threshold,
            "memory_warning": settings.memory_warning_threshold,
            "memory_critical": settings.memory_critical_threshold,
            "restart_loop_count": settings.restart_loop_threshold,
            "restart_loop_window_seconds": settings.restart_loop_window,
            "health_check_failures": settings.health_check_failures_threshold,
        },
        "critical_containers": settings.critical_containers,
        "health_endpoints": list(HEALTH_ENDPOINTS.keys()),
        "tcp_check_containers": list(TCP_CHECK_CONTAINERS.keys()),
        "notifications": {
            "telegram_enabled": settings.telegram_enabled and bool(settings.telegram_bot_token),
            "email_enabled": settings.email_enabled and bool(settings.smtp_host),
        },
    }
