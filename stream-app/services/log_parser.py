"""
Log Parser Service - Extract alerts and activity from log files
Provides structured data from streaming service logs for real-time monitoring
"""

import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"

class AlertType(Enum):
    GAP_DETECTION = "gap_detection"
    STREAM_HEALTH = "stream_health"
    AUTHENTICATION = "authentication"
    CANDLE_COMPLETION = "candle_completion"
    BACKFILL = "backfill"
    CONNECTION = "connection"
    SYSTEM = "system"

class LogParser:
    """Parser for streaming service logs"""
    
    def __init__(self, log_file_path: str = "/app/logs/fastapi-stream.log"):
        self.log_file_path = log_file_path
        
        # Log patterns for different types of messages
        self.patterns = {
            AlertType.GAP_DETECTION: [
                r"gap detected",
                r"No gaps detected",
                r"Running gap detection",
            ],
            AlertType.STREAM_HEALTH: [
                r"Data.*is stale",
                r"streams healthy",
                r"Stream.*restart",
                r"Stream.*connected",
                r"Stream.*disconnected"
            ],
            AlertType.AUTHENTICATION: [
                r"Auth refresh",
                r"authentication refreshed",
                r"IG Login successful",
                r"Failed to refresh auth"
            ],
            AlertType.CANDLE_COMPLETION: [
                r"candle completed",
                r"🟢 🆕.*candle"
            ],
            AlertType.BACKFILL: [
                r"backfill",
                r"Gap filled",
                r"Auto-backfill"
            ],
            AlertType.CONNECTION: [
                r"Connected to",
                r"Connection.*established",
                r"Connection.*lost"
            ]
        }
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line into structured data"""
        try:
            # Standard log format: TIMESTAMP - LEVEL - MODULE - MESSAGE
            # Example: 2025-09-03 06:44:59,528 - WARNING - igstream.sync_manager - Message
            
            log_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - (\w+) - ([^-]+) - (.+)"
            match = re.match(log_pattern, line.strip())
            
            if not match:
                return None
            
            timestamp_str, level, module, message = match.groups()
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                timestamp = datetime.now()
            
            # Determine alert type
            alert_type = self._categorize_message(message)
            
            # Extract epic if present
            epic = self._extract_epic(message)
            
            return {
                "timestamp": timestamp,
                "level": level.strip(),
                "module": module.strip(),
                "message": message.strip(),
                "alert_type": alert_type.value if alert_type else AlertType.SYSTEM.value,
                "epic": epic
            }
            
        except Exception as e:
            logger.error(f"Error parsing log line: {e}")
            return None
    
    def _categorize_message(self, message: str) -> Optional[AlertType]:
        """Categorize message based on content patterns"""
        message_lower = message.lower()
        
        for alert_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return alert_type
        
        return AlertType.SYSTEM
    
    def _extract_epic(self, message: str) -> Optional[str]:
        """Extract epic name from message if present"""
        # Look for pattern like CS.D.EURUSD.CEEM.IP
        epic_pattern = r"CS\.D\.([A-Z]{6})\.MINI\.IP"
        match = re.search(epic_pattern, message)
        
        if match:
            return match.group(1)  # Return just the currency pair (e.g., EURUSD)
        
        return None
    
    def get_recent_logs(self, hours_back: int = 2, max_entries: int = 100) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        if not os.path.exists(self.log_file_path):
            logger.warning(f"Log file not found: {self.log_file_path}")
            return []
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_logs = []
            
            # Read the log file from the end
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Process lines from newest to oldest
            for line in reversed(lines):
                if len(recent_logs) >= max_entries:
                    break
                
                parsed_line = self.parse_log_line(line)
                if parsed_line and parsed_line["timestamp"] >= cutoff_time:
                    recent_logs.append(parsed_line)
            
            # Sort by timestamp (newest first)
            recent_logs.sort(key=lambda x: x["timestamp"], reverse=True)
            return recent_logs
            
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return []
    
    def get_recent_alerts(self, hours_back: int = 4, max_alerts: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts (WARNING and ERROR level entries)"""
        all_logs = self.get_recent_logs(hours_back, max_entries=500)
        
        # Filter for alerts (WARNING and ERROR)
        alerts = [
            log for log in all_logs
            if log["level"] in ["WARNING", "ERROR"]
        ]
        
        return alerts[:max_alerts]
    
    def get_recent_operations(self, hours_back: int = 6, max_operations: int = 50) -> List[Dict[str, Any]]:
        """Get recent operations (INFO level entries about system operations)"""
        all_logs = self.get_recent_logs(hours_back, max_entries=1000)
        
        # Filter for operational messages
        operation_keywords = [
            "candle completed", "gap filled", "stream restart", "auth refresh",
            "backfill", "connected", "started", "stopped"
        ]
        
        operations = []
        for log in all_logs:
            if log["level"] == "INFO":
                message_lower = log["message"].lower()
                if any(keyword in message_lower for keyword in operation_keywords):
                    # Create operation-specific format
                    operation = {
                        "time": log["timestamp"].strftime("%H:%M:%S"),
                        "epic": log["epic"] or "SYSTEM",
                        "action": self._extract_action(log["message"]),
                        "status": "✅" if "success" in message_lower or "completed" in message_lower 
                                      or "✅" in log["message"] else "ℹ️",
                        "full_message": log["message"]
                    }
                    operations.append(operation)
        
        return operations[:max_operations]
    
    def _extract_action(self, message: str) -> str:
        """Extract a concise action description from log message"""
        message_lower = message.lower()
        
        # Define action mappings
        action_patterns = {
            r"candle completed": "Candle completed",
            r"gap.*detected": "Gap detected", 
            r"gap.*filled": "Gap filled",
            r"stream.*restart": "Stream restart",
            r"auth.*refresh": "Auth refresh",
            r"backfill.*start": "Backfill started",
            r"connected": "Connected",
            r"disconnected": "Disconnected"
        }
        
        for pattern, action in action_patterns.items():
            if re.search(pattern, message_lower):
                return action
        
        # If no specific pattern matches, create generic action
        words = message.split()[:3]  # Take first 3 words
        return " ".join(words)
    
    def get_system_health_summary(self, hours_back: int = 1) -> Dict[str, Any]:
        """Get system health summary from recent logs"""
        recent_logs = self.get_recent_logs(hours_back)
        
        summary = {
            "total_entries": len(recent_logs),
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "last_error": None,
            "last_warning": None,
            "stream_health": "unknown",
            "gap_status": "unknown"
        }
        
        for log in recent_logs:
            level = log["level"]
            if level == "ERROR":
                summary["error_count"] += 1
                if not summary["last_error"]:
                    summary["last_error"] = log
            elif level == "WARNING":
                summary["warning_count"] += 1
                if not summary["last_warning"]:
                    summary["last_warning"] = log
            elif level == "INFO":
                summary["info_count"] += 1
            
            # Check for specific health indicators
            message = log["message"].lower()
            if "streams healthy" in message:
                summary["stream_health"] = "healthy"
            elif "stale" in message or "disconnect" in message:
                summary["stream_health"] = "issues"
            
            if "no gaps detected" in message:
                summary["gap_status"] = "no_gaps"
            elif "gap detected" in message:
                summary["gap_status"] = "gaps_found"
        
        return summary

def get_log_parser() -> LogParser:
    """Get a configured log parser instance"""
    return LogParser()

if __name__ == "__main__":
    # Test the log parser
    parser = LogParser()
    
    print("Testing log parser...")
    
    # Get recent alerts
    alerts = parser.get_recent_alerts(hours_back=2)
    print(f"\nFound {len(alerts)} recent alerts:")
    for alert in alerts[:5]:
        print(f"  {alert['timestamp']} [{alert['level']}] {alert['message'][:60]}...")
    
    # Get recent operations
    operations = parser.get_recent_operations(hours_back=2)
    print(f"\nFound {len(operations)} recent operations:")
    for op in operations[:5]:
        print(f"  {op['time']} - {op['epic']}: {op['action']} {op['status']}")
    
    # Get health summary
    health = parser.get_system_health_summary()
    print(f"\nSystem Health Summary:")
    print(f"  Total entries: {health['total_entries']}")
    print(f"  Errors: {health['error_count']}, Warnings: {health['warning_count']}")
    print(f"  Stream health: {health['stream_health']}")
    print(f"  Gap status: {health['gap_status']}")