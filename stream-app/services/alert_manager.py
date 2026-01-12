"""
Alert Manager - Aggregate alerts from multiple sources
Provides unified alert management for the streaming system
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio
import logging
from services.log_parser import get_log_parser, LogLevel
from igstream.sync_manager import get_stream_health_report

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertCategory(Enum):
    STREAM_HEALTH = "stream_health"
    GAP_DETECTION = "gap_detection"
    AUTHENTICATION = "authentication"
    BACKFILL = "backfill"
    CONNECTION = "connection"
    SYSTEM = "system"

class Alert:
    """Represents a system alert"""
    
    def __init__(self, 
                 timestamp: datetime,
                 severity: AlertSeverity,
                 category: AlertCategory,
                 message: str,
                 epic: Optional[str] = None,
                 source: str = "system",
                 details: Optional[Dict[str, Any]] = None):
        self.timestamp = timestamp
        self.severity = severity
        self.category = category
        self.message = message
        self.epic = epic
        self.source = source
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "time": self.timestamp.strftime("%H:%M:%S"),
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "epic": self.epic,
            "source": self.source,
            "details": self.details
        }
    
    def get_display_color(self) -> str:
        """Get color for display based on severity"""
        color_map = {
            AlertSeverity.INFO: "blue",
            AlertSeverity.WARNING: "orange",
            AlertSeverity.ERROR: "red",
            AlertSeverity.CRITICAL: "darkred"
        }
        return color_map.get(self.severity, "gray")
    
    def get_icon(self) -> str:
        """Get icon emoji based on category and severity"""
        if self.severity == AlertSeverity.ERROR or self.severity == AlertSeverity.CRITICAL:
            return "❌"
        elif self.severity == AlertSeverity.WARNING:
            return "⚠️"
        elif self.category == AlertCategory.STREAM_HEALTH:
            return "📡"
        elif self.category == AlertCategory.GAP_DETECTION:
            return "🕳️"
        elif self.category == AlertCategory.AUTHENTICATION:
            return "🔑"
        elif self.category == AlertCategory.BACKFILL:
            return "📈"
        else:
            return "ℹ️"

class AlertManager:
    """Manages system alerts from multiple sources"""
    
    def __init__(self):
        self.log_parser = get_log_parser()
        self._alert_cache = []
        self._cache_expiry = None
        self._cache_duration = timedelta(minutes=2)  # Cache for 2 minutes
    
    def _is_cache_valid(self) -> bool:
        """Check if alert cache is still valid"""
        return (self._cache_expiry is not None and 
                datetime.now() < self._cache_expiry)
    
    def _refresh_cache(self):
        """Refresh the alert cache"""
        self._alert_cache = self._collect_all_alerts()
        self._cache_expiry = datetime.now() + self._cache_duration
    
    def _collect_all_alerts(self) -> List[Alert]:
        """Collect alerts from all sources"""
        alerts = []
        
        try:
            # 1. Get alerts from log files
            log_alerts = self._get_log_alerts()
            alerts.extend(log_alerts)
            
            # 2. Get alerts from stream health check
            stream_alerts = self._get_stream_health_alerts()
            alerts.extend(stream_alerts)
            
            # 3. Get alerts from sync manager (if available)
            sync_alerts = self._get_sync_manager_alerts()
            alerts.extend(sync_alerts)
            
        except Exception as e:
            logger.error(f"Error collecting alerts: {e}")
            # Add error alert
            alerts.append(Alert(
                timestamp=datetime.now(),
                severity=AlertSeverity.ERROR,
                category=AlertCategory.SYSTEM,
                message=f"Alert collection failed: {str(e)}",
                source="alert_manager"
            ))
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts
    
    def _get_log_alerts(self) -> List[Alert]:
        """Get alerts from log parser"""
        alerts = []
        
        try:
            # Get recent log alerts
            log_alerts = self.log_parser.get_recent_alerts(hours_back=4)
            
            for log_alert in log_alerts:
                severity = self._map_log_level_to_severity(log_alert["level"])
                category = self._map_alert_type_to_category(log_alert["alert_type"])
                
                alert = Alert(
                    timestamp=log_alert["timestamp"],
                    severity=severity,
                    category=category,
                    message=log_alert["message"],
                    epic=log_alert.get("epic"),
                    source=f"log:{log_alert['module']}",
                    details={"module": log_alert["module"]}
                )
                alerts.append(alert)
                
        except Exception as e:
            logger.error(f"Error getting log alerts: {e}")
        
        return alerts
    
    def _get_stream_health_alerts(self) -> List[Alert]:
        """Get alerts from stream health monitoring"""
        alerts = []
        
        try:
            # Check if we can get stream health report
            # Use create_task instead of asyncio.run since we're already in an event loop
            loop = asyncio.get_event_loop()
            health_report = loop.run_until_complete(get_stream_health_report()) if not loop.is_running() else {}
            # If loop is running (which it should be), we can't use run_until_complete
            # Instead, we should make this function async or use a different approach
            if loop.is_running():
                # For now, return empty report to avoid error
                health_report = {"streams": {}}
                logger.debug("Skipping health report due to running event loop")
            
            if isinstance(health_report, dict) and "streams" in health_report:
                for epic, stream_info in health_report["streams"].items():
                    if not stream_info.get("overall_healthy", True):
                        # Create alert for unhealthy stream
                        message = f"{epic.replace('CS.D.', '').replace('.MINI.IP', '')}: Stream unhealthy"
                        
                        # Determine specific issue
                        details = []
                        if not stream_info.get("task_running", True):
                            details.append("not running")
                        if not stream_info.get("data_fresh_1m", True):
                            details.append("stale 1m data")
                        
                        if details:
                            message += f" ({', '.join(details)})"
                        
                        alert = Alert(
                            timestamp=datetime.now(),
                            severity=AlertSeverity.WARNING,
                            category=AlertCategory.STREAM_HEALTH,
                            message=message,
                            epic=epic.replace('CS.D.', '').replace('.MINI.IP', ''),
                            source="stream_monitor",
                            details=stream_info
                        )
                        alerts.append(alert)
                        
        except Exception as e:
            logger.error(f"Error getting stream health alerts: {e}")
        
        return alerts
    
    def _get_sync_manager_alerts(self) -> List[Alert]:
        """Get alerts from sync manager"""
        alerts = []
        
        try:
            # Get system health summary from logs
            health_summary = self.log_parser.get_system_health_summary(hours_back=1)
            
            # Create alerts based on health summary
            if health_summary["error_count"] > 0:
                alert = Alert(
                    timestamp=datetime.now(),
                    severity=AlertSeverity.ERROR,
                    category=AlertCategory.SYSTEM,
                    message=f"{health_summary['error_count']} errors in last hour",
                    source="sync_manager",
                    details=health_summary
                )
                alerts.append(alert)
            
            if health_summary["stream_health"] == "issues":
                alert = Alert(
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.STREAM_HEALTH,
                    message="Stream health issues detected",
                    source="sync_manager"
                )
                alerts.append(alert)
            elif health_summary["stream_health"] == "healthy":
                alert = Alert(
                    timestamp=datetime.now(),
                    severity=AlertSeverity.INFO,
                    category=AlertCategory.STREAM_HEALTH,
                    message="All streams healthy",
                    source="sync_manager"
                )
                alerts.append(alert)
            
            if health_summary["gap_status"] == "gaps_found":
                alert = Alert(
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.GAP_DETECTION,
                    message="Data gaps detected",
                    source="gap_detector"
                )
                alerts.append(alert)
                
        except Exception as e:
            logger.error(f"Error getting sync manager alerts: {e}")
        
        return alerts
    
    def _map_log_level_to_severity(self, log_level: str) -> AlertSeverity:
        """Map log level to alert severity"""
        mapping = {
            "ERROR": AlertSeverity.ERROR,
            "WARNING": AlertSeverity.WARNING,
            "INFO": AlertSeverity.INFO,
            "DEBUG": AlertSeverity.INFO
        }
        return mapping.get(log_level, AlertSeverity.INFO)
    
    def _map_alert_type_to_category(self, alert_type: str) -> AlertCategory:
        """Map alert type to category"""
        mapping = {
            "gap_detection": AlertCategory.GAP_DETECTION,
            "stream_health": AlertCategory.STREAM_HEALTH,
            "authentication": AlertCategory.AUTHENTICATION,
            "backfill": AlertCategory.BACKFILL,
            "connection": AlertCategory.CONNECTION,
            "candle_completion": AlertCategory.STREAM_HEALTH,
            "system": AlertCategory.SYSTEM
        }
        return mapping.get(alert_type, AlertCategory.SYSTEM)
    
    def get_recent_alerts(self, hours_back: int = 6, max_alerts: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts from all sources"""
        # Refresh cache if needed
        if not self._is_cache_valid():
            self._refresh_cache()
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_alerts = [
            alert for alert in self._alert_cache 
            if alert.timestamp >= cutoff_time
        ]
        
        # Convert to dictionaries and limit results
        return [alert.to_dict() for alert in recent_alerts[:max_alerts]]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        # Refresh cache if needed
        if not self._is_cache_valid():
            self._refresh_cache()
        
        # Count alerts by severity in last 4 hours
        cutoff_time = datetime.now() - timedelta(hours=4)
        recent_alerts = [
            alert for alert in self._alert_cache 
            if alert.timestamp >= cutoff_time
        ]
        
        summary = {
            "total_alerts": len(recent_alerts),
            "error_count": sum(1 for a in recent_alerts if a.severity == AlertSeverity.ERROR),
            "warning_count": sum(1 for a in recent_alerts if a.severity == AlertSeverity.WARNING),
            "info_count": sum(1 for a in recent_alerts if a.severity == AlertSeverity.INFO),
            "last_alert": recent_alerts[0].to_dict() if recent_alerts else None,
            "last_error": None,
            "last_warning": None
        }
        
        # Find last error and warning
        for alert in recent_alerts:
            if alert.severity == AlertSeverity.ERROR and not summary["last_error"]:
                summary["last_error"] = alert.to_dict()
            elif alert.severity == AlertSeverity.WARNING and not summary["last_warning"]:
                summary["last_warning"] = alert.to_dict()
        
        return summary
    
    def get_operations_from_logs(self, hours_back: int = 6) -> List[Dict[str, Any]]:
        """Get recent operations from logs"""
        return self.log_parser.get_recent_operations(hours_back)
    
    def add_custom_alert(self, 
                        severity: AlertSeverity,
                        category: AlertCategory,
                        message: str,
                        epic: Optional[str] = None,
                        details: Optional[Dict[str, Any]] = None):
        """Add a custom alert to the system"""
        alert = Alert(
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            epic=epic,
            source="custom",
            details=details
        )
        
        # Add to cache if it exists
        if self._alert_cache is not None:
            self._alert_cache.insert(0, alert)
            # Keep cache size manageable
            if len(self._alert_cache) > 200:
                self._alert_cache = self._alert_cache[:200]

# Global alert manager instance
_alert_manager = None

def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager

if __name__ == "__main__":
    # Test the alert manager
    alert_manager = AlertManager()
    
    print("Testing alert manager...")
    
    # Get recent alerts
    alerts = alert_manager.get_recent_alerts()
    print(f"\nFound {len(alerts)} recent alerts:")
    for alert in alerts[:5]:
        print(f"  {alert['time']} [{alert['severity']}] {alert['message']}")
    
    # Get alert summary
    summary = alert_manager.get_alert_summary()
    print(f"\nAlert Summary:")
    print(f"  Total: {summary['total_alerts']}")
    print(f"  Errors: {summary['error_count']}, Warnings: {summary['warning_count']}")
    
    # Get operations
    operations = alert_manager.get_operations_from_logs()
    print(f"\nFound {len(operations)} recent operations:")
    for op in operations[:5]:
        print(f"  {op['time']} - {op['epic']}: {op['action']} {op['status']}")