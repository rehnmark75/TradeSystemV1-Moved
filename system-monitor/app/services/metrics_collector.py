"""
Metrics collector service - stores and retrieves historical metrics.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from ..config import settings
from ..models import ContainerHealth, SystemHealth

logger = logging.getLogger(__name__)

Base = declarative_base()


class ContainerMetricsRecord(Base):
    """SQLAlchemy model for container metrics."""
    __tablename__ = "container_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    container_name = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    status = Column(String(20), nullable=False)
    health_status = Column(String(20))
    cpu_percent = Column(Float)
    memory_bytes = Column(BigInteger)
    memory_percent = Column(Float)
    network_rx_bytes = Column(BigInteger)
    network_tx_bytes = Column(BigInteger)
    restart_count = Column(Integer)
    uptime_seconds = Column(Integer)


class SystemAlertRecord(Base):
    """SQLAlchemy model for system alerts."""
    __tablename__ = "system_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    container_name = Column(String(100))
    title = Column(String(255), nullable=False)
    message = Column(String, nullable=False)
    details = Column(String)  # JSON string
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    acknowledged_at = Column(DateTime(timezone=True))
    acknowledged_by = Column(String(100))
    resolved_at = Column(DateTime(timezone=True))
    notification_sent = Column(Integer, default=0)  # Boolean as int for compatibility


class MetricsCollector:
    """Service for collecting and storing container metrics."""

    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._initialized = False

    def _get_engine(self):
        """Get or create database engine."""
        if self._engine is None:
            try:
                self._engine = create_engine(
                    settings.database_url,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_recycle=1800,
                )
                self._session_factory = sessionmaker(bind=self._engine)
                logger.info("Database engine created successfully")
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}")
                raise
        return self._engine

    def initialize_tables(self):
        """Create tables if they don't exist."""
        if self._initialized:
            return

        try:
            engine = self._get_engine()
            Base.metadata.create_all(engine)
            self._initialized = True
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise

    def _get_session(self):
        """Get a new database session."""
        if self._session_factory is None:
            self._get_engine()
        return self._session_factory()

    def store_container_metrics(self, container: ContainerHealth):
        """Store metrics for a single container."""
        try:
            session = self._get_session()
            record = ContainerMetricsRecord(
                container_name=container.name,
                timestamp=datetime.now(timezone.utc),
                status=container.status.value,
                health_status=container.health_status.value,
                cpu_percent=container.metrics.cpu_percent,
                memory_bytes=container.metrics.memory_bytes,
                memory_percent=container.metrics.memory_percent,
                network_rx_bytes=container.metrics.network_rx_bytes,
                network_tx_bytes=container.metrics.network_tx_bytes,
                restart_count=container.restart_count,
                uptime_seconds=container.uptime_seconds,
            )
            session.add(record)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Failed to store metrics for {container.name}: {e}")

    def store_system_metrics(self, system_health: SystemHealth):
        """Store metrics for all containers in system health."""
        for container in system_health.containers:
            self.store_container_metrics(container)

    def get_container_metrics_history(
        self,
        container_name: str,
        hours: int = 24
    ) -> List[Dict]:
        """Get historical metrics for a container."""
        try:
            session = self._get_session()
            since = datetime.now(timezone.utc) - timedelta(hours=hours)

            records = session.query(ContainerMetricsRecord).filter(
                ContainerMetricsRecord.container_name == container_name,
                ContainerMetricsRecord.timestamp >= since
            ).order_by(ContainerMetricsRecord.timestamp.asc()).all()

            session.close()

            return [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "status": r.status,
                    "health_status": r.health_status,
                    "cpu_percent": r.cpu_percent,
                    "memory_bytes": r.memory_bytes,
                    "memory_percent": r.memory_percent,
                    "network_rx_bytes": r.network_rx_bytes,
                    "network_tx_bytes": r.network_tx_bytes,
                    "restart_count": r.restart_count,
                    "uptime_seconds": r.uptime_seconds,
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"Failed to get metrics history for {container_name}: {e}")
            return []

    def get_all_metrics_history(self, hours: int = 24) -> Dict[str, List[Dict]]:
        """Get historical metrics for all containers."""
        try:
            session = self._get_session()
            since = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Get distinct container names
            container_names = session.query(
                ContainerMetricsRecord.container_name
            ).distinct().all()

            session.close()

            results = {}
            for (name,) in container_names:
                results[name] = self.get_container_metrics_history(name, hours)

            return results
        except Exception as e:
            logger.error(f"Failed to get all metrics history: {e}")
            return {}

    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        try:
            session = self._get_session()
            cutoff = datetime.now(timezone.utc) - timedelta(days=settings.metrics_retention_days)

            deleted = session.query(ContainerMetricsRecord).filter(
                ContainerMetricsRecord.timestamp < cutoff
            ).delete()

            session.commit()
            session.close()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old metrics records")
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")

    def store_alert(self, alert) -> Optional[int]:
        """Store an alert record. Returns alert ID."""
        try:
            import json
            session = self._get_session()

            record = SystemAlertRecord(
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                container_name=alert.container_name,
                title=alert.title,
                message=alert.message,
                details=json.dumps(alert.details) if alert.details else None,
                created_at=alert.created_at,
                notification_sent=1 if alert.notification_sent else 0,
            )
            session.add(record)
            session.commit()
            alert_id = record.id
            session.close()

            return alert_id
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
            return None

    def get_recent_alerts(self, limit: int = 50, active_only: bool = False) -> List[Dict]:
        """Get recent alerts."""
        try:
            import json
            session = self._get_session()

            query = session.query(SystemAlertRecord)

            if active_only:
                query = query.filter(SystemAlertRecord.resolved_at.is_(None))

            records = query.order_by(
                SystemAlertRecord.created_at.desc()
            ).limit(limit).all()

            session.close()

            return [
                {
                    "id": r.id,
                    "alert_type": r.alert_type,
                    "severity": r.severity,
                    "container_name": r.container_name,
                    "title": r.title,
                    "message": r.message,
                    "details": json.loads(r.details) if r.details else {},
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "acknowledged_at": r.acknowledged_at.isoformat() if r.acknowledged_at else None,
                    "acknowledged_by": r.acknowledged_by,
                    "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
                    "notification_sent": bool(r.notification_sent),
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {e}")
            return []

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str = "admin") -> bool:
        """Acknowledge an alert."""
        try:
            session = self._get_session()

            record = session.query(SystemAlertRecord).filter(
                SystemAlertRecord.id == alert_id
            ).first()

            if record:
                record.acknowledged_at = datetime.now(timezone.utc)
                record.acknowledged_by = acknowledged_by
                session.commit()
                session.close()
                return True

            session.close()
            return False
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False

    def resolve_alert(self, alert_id: int) -> bool:
        """Mark an alert as resolved."""
        try:
            session = self._get_session()

            record = session.query(SystemAlertRecord).filter(
                SystemAlertRecord.id == alert_id
            ).first()

            if record:
                record.resolved_at = datetime.now(timezone.utc)
                session.commit()
                session.close()
                return True

            session.close()
            return False
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False

    def get_active_alert_count(self) -> int:
        """Get count of active (unresolved) alerts."""
        try:
            session = self._get_session()
            count = session.query(SystemAlertRecord).filter(
                SystemAlertRecord.resolved_at.is_(None)
            ).count()
            session.close()
            return count
        except Exception as e:
            logger.error(f"Failed to get active alert count: {e}")
            return 0

    def get_uptime_stats(self, container_name: str, days: int = 7) -> Dict:
        """Calculate uptime statistics for a container."""
        try:
            session = self._get_session()
            since = datetime.now(timezone.utc) - timedelta(days=days)

            records = session.query(ContainerMetricsRecord).filter(
                ContainerMetricsRecord.container_name == container_name,
                ContainerMetricsRecord.timestamp >= since
            ).all()

            session.close()

            if not records:
                return {"uptime_percent": 0.0, "total_checks": 0, "running_checks": 0}

            total = len(records)
            running = sum(1 for r in records if r.status == "running")

            return {
                "uptime_percent": round((running / total) * 100, 2) if total > 0 else 0.0,
                "total_checks": total,
                "running_checks": running,
            }
        except Exception as e:
            logger.error(f"Failed to get uptime stats for {container_name}: {e}")
            return {"uptime_percent": 0.0, "total_checks": 0, "running_checks": 0}
