"""
Operation Tracker - Track real-time operations for system monitoring
Provides in-memory tracking of backfill operations, gap detection, and other system activities
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)

class OperationType(Enum):
    GAP_DETECTION = "gap_detection"
    GAP_FILL = "gap_fill"
    STREAM_RESTART = "stream_restart"
    AUTH_REFRESH = "auth_refresh"
    CONNECTION = "connection"
    CANDLE_COMPLETION = "candle_completion"
    SYSTEM = "system"

class OperationStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    WARNING = "warning"

class Operation:
    """Represents a system operation"""
    
    def __init__(self,
                 operation_type: OperationType,
                 epic: Optional[str] = None,
                 message: str = "",
                 status: OperationStatus = OperationStatus.IN_PROGRESS,
                 details: Optional[Dict[str, Any]] = None):
        self.timestamp = datetime.now()
        self.operation_type = operation_type
        self.epic = epic or "SYSTEM"
        self.message = message
        self.status = status
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operation to dictionary"""
        return {
            "time": self.timestamp.strftime("%H:%M:%S"),
            "timestamp": self.timestamp.isoformat(),
            "epic": self.epic,
            "action": self._get_action_description(),
            "status": self._get_status_icon(),
            "message": self.message,
            "type": self.operation_type.value,
            "details": self.details
        }
    
    def _get_action_description(self) -> str:
        """Get human-readable action description"""
        action_map = {
            OperationType.GAP_DETECTION: "Gap detection",
            OperationType.GAP_FILL: "Gap fill",
            OperationType.STREAM_RESTART: "Stream restart", 
            OperationType.AUTH_REFRESH: "Auth refresh",
            OperationType.CONNECTION: "Connection",
            OperationType.CANDLE_COMPLETION: "Candle completed",
            OperationType.SYSTEM: "System operation"
        }
        return action_map.get(self.operation_type, "Unknown")
    
    def _get_status_icon(self) -> str:
        """Get status icon based on operation status"""
        icon_map = {
            OperationStatus.SUCCESS: "✅",
            OperationStatus.FAILURE: "❌",
            OperationStatus.IN_PROGRESS: "⏳",
            OperationStatus.WARNING: "⚠️"
        }
        return icon_map.get(self.status, "❓")

class OperationTracker:
    """Tracks system operations in memory for real-time monitoring"""
    
    def __init__(self, max_operations: int = 500, retention_hours: int = 24):
        self.max_operations = max_operations
        self.retention_hours = retention_hours
        self.operations: List[Operation] = []
        self.lock = threading.Lock()
        
        # Statistics tracking
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "gap_fills_completed": 0,
            "auth_refreshes": 0
        }
    
    def track_operation(self,
                       operation_type: OperationType,
                       epic: Optional[str] = None,
                       message: str = "",
                       status: OperationStatus = OperationStatus.SUCCESS,
                       details: Optional[Dict[str, Any]] = None) -> Operation:
        """Track a new operation"""
        
        operation = Operation(
            operation_type=operation_type,
            epic=epic,
            message=message,
            status=status,
            details=details
        )
        
        with self.lock:
            # Add to beginning of list (newest first)
            self.operations.insert(0, operation)
            
            # Update statistics
            self.stats["total_operations"] += 1
            if status == OperationStatus.SUCCESS:
                self.stats["successful_operations"] += 1
            elif status == OperationStatus.FAILURE:
                self.stats["failed_operations"] += 1
            
            # Update type-specific stats
            if operation_type == OperationType.GAP_FILL and status == OperationStatus.SUCCESS:
                self.stats["gap_fills_completed"] += 1
            elif operation_type == OperationType.AUTH_REFRESH and status == OperationStatus.SUCCESS:
                self.stats["auth_refreshes"] += 1
            
            # Cleanup old operations
            self._cleanup_old_operations()
            
            # Limit total operations
            if len(self.operations) > self.max_operations:
                self.operations = self.operations[:self.max_operations]
        
        logger.debug(f"Tracked operation: {operation_type.value} for {epic or 'SYSTEM'} - {status.value}")
        return operation
    
    def _cleanup_old_operations(self):
        """Remove operations older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        self.operations = [op for op in self.operations if op.timestamp >= cutoff_time]
    
    def get_recent_operations(self, hours_back: int = 6, max_count: int = 50) -> List[Dict[str, Any]]:
        """Get recent operations"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_ops = [
                op for op in self.operations 
                if op.timestamp >= cutoff_time
            ]
            
            # Limit results
            return [op.to_dict() for op in recent_ops[:max_count]]
    
    def get_operations_by_type(self, operation_type: OperationType, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get operations by type"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            type_ops = [
                op for op in self.operations 
                if op.operation_type == operation_type and op.timestamp >= cutoff_time
            ]
            
            return [op.to_dict() for op in type_ops]
    
    def get_operations_by_epic(self, epic: str, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get operations for a specific epic"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            epic_ops = [
                op for op in self.operations 
                if op.epic == epic and op.timestamp >= cutoff_time
            ]
            
            return [op.to_dict() for op in epic_ops]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get operation statistics"""
        with self.lock:
            # Calculate recent statistics (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_ops = [op for op in self.operations if op.timestamp >= cutoff_time]
            
            recent_stats = {
                "recent_operations": len(recent_ops),
                "recent_successes": sum(1 for op in recent_ops if op.status == OperationStatus.SUCCESS),
                "recent_failures": sum(1 for op in recent_ops if op.status == OperationStatus.FAILURE)
            }
            
            return {
                **self.stats,
                **recent_stats,
                "total_tracked": len(self.operations),
                "last_operation": self.operations[0].to_dict() if self.operations else None
            }
    
    def clear_old_statistics(self):
        """Reset statistics (useful for long-running services)"""
        with self.lock:
            self.stats = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "gap_fills_completed": 0,
                "auth_refreshes": 0
            }

# Global operation tracker instance
_operation_tracker = None

def get_operation_tracker() -> OperationTracker:
    """Get the global operation tracker instance"""
    global _operation_tracker
    if _operation_tracker is None:
        _operation_tracker = OperationTracker()
    return _operation_tracker

def track_gap_detection(epics_checked: int, gaps_found: int) -> Operation:
    """Convenience function to track gap detection"""
    tracker = get_operation_tracker()
    status = OperationStatus.SUCCESS if gaps_found == 0 else OperationStatus.WARNING
    message = f"Checked {epics_checked} epics, found {gaps_found} gaps"
    
    return tracker.track_operation(
        operation_type=OperationType.GAP_DETECTION,
        epic=None,
        message=message,
        status=status,
        details={"epics_checked": epics_checked, "gaps_found": gaps_found}
    )

def track_gap_fill(epic: str, gap_start: str, gap_end: str, candles_filled: int, success: bool) -> Operation:
    """Convenience function to track gap filling"""
    tracker = get_operation_tracker()
    status = OperationStatus.SUCCESS if success else OperationStatus.FAILURE
    message = f"Filled {candles_filled} candles ({gap_start} to {gap_end})"
    
    return tracker.track_operation(
        operation_type=OperationType.GAP_FILL,
        epic=epic,
        message=message,
        status=status,
        details={
            "gap_start": gap_start,
            "gap_end": gap_end,
            "candles_filled": candles_filled
        }
    )

def track_stream_restart(epic: str, reason: str, success: bool) -> Operation:
    """Convenience function to track stream restarts"""
    tracker = get_operation_tracker()
    status = OperationStatus.SUCCESS if success else OperationStatus.FAILURE
    message = f"Stream restart: {reason}"
    
    return tracker.track_operation(
        operation_type=OperationType.STREAM_RESTART,
        epic=epic,
        message=message,
        status=status,
        details={"reason": reason}
    )

def track_auth_refresh(success: bool, error_message: str = None) -> Operation:
    """Convenience function to track authentication refresh"""
    tracker = get_operation_tracker()
    status = OperationStatus.SUCCESS if success else OperationStatus.FAILURE
    message = "Auth refresh successful" if success else f"Auth refresh failed: {error_message}"
    
    return tracker.track_operation(
        operation_type=OperationType.AUTH_REFRESH,
        epic=None,
        message=message,
        status=status,
        details={"error": error_message} if error_message else {}
    )

def track_candle_completion(epic: str, timeframe: int, candle_time: str) -> Operation:
    """Convenience function to track candle completion"""
    tracker = get_operation_tracker()
    message = f"{timeframe}m candle completed at {candle_time}"
    
    return tracker.track_operation(
        operation_type=OperationType.CANDLE_COMPLETION,
        epic=epic,
        message=message,
        status=OperationStatus.SUCCESS,
        details={"timeframe": timeframe, "candle_time": candle_time}
    )

if __name__ == "__main__":
    # Test the operation tracker
    tracker = get_operation_tracker()
    
    print("Testing operation tracker...")
    
    # Track some test operations
    track_gap_detection(epics_checked=9, gaps_found=2)
    track_gap_fill("EURUSD", "10:00", "10:05", candles_filled=1, success=True)
    track_stream_restart("GBPUSD", "stale data", success=True)
    track_auth_refresh(success=True)
    
    # Get recent operations
    recent_ops = tracker.get_recent_operations()
    print(f"\nRecent operations ({len(recent_ops)}):")
    for op in recent_ops:
        print(f"  {op['time']} - {op['epic']}: {op['action']} {op['status']}")
    
    # Get statistics
    stats = tracker.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total: {stats['total_operations']}")
    print(f"  Success: {stats['successful_operations']}")
    print(f"  Gap fills: {stats['gap_fills_completed']}")
    print(f"  Auth refreshes: {stats['auth_refreshes']}")