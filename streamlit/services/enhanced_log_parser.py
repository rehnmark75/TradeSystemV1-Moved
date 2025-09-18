"""
Enhanced Log Parser Service - Multi-source log intelligence for trading system
Extends existing log parser to handle forex scanner signals, stream health, and system operations
"""

import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"

class LogSource(Enum):
    FOREX_SCANNER = "forex_scanner"
    STREAM_SERVICE = "stream_service"
    TRADE_MONITOR = "trade_monitor"
    FASTAPI_DEV = "fastapi_dev"
    SYSTEM = "system"

class AlertType(Enum):
    # Signal-related alerts
    SIGNAL_DETECTED = "signal_detected"
    SIGNAL_REJECTED = "signal_rejected"
    SIGNAL_VALIDATION = "signal_validation"

    # Stream health alerts
    GAP_DETECTION = "gap_detection"
    STREAM_HEALTH = "stream_health"
    AUTHENTICATION = "authentication"
    CANDLE_COMPLETION = "candle_completion"
    BACKFILL = "backfill"
    CONNECTION = "connection"

    # System alerts
    SYSTEM = "system"
    PERFORMANCE = "performance"
    ERROR = "error"

@dataclass
class SignalContext:
    """Rich context for signal-related log entries"""
    strategy: Optional[str] = None
    epic: Optional[str] = None
    signal_type: Optional[str] = None  # BULL/BEAR
    confidence: Optional[float] = None
    timeframe: Optional[str] = None
    rejection_reason: Optional[str] = None
    validation_details: Optional[Dict[str, Any]] = None

@dataclass
class ParsedLogEntry:
    """Structured representation of a parsed log entry"""
    timestamp: datetime
    level: str
    module: str
    message: str
    alert_type: str
    log_source: str
    epic: Optional[str] = None
    signal_context: Optional[SignalContext] = None
    raw_line: Optional[str] = None

class EnhancedLogParser:
    """Enhanced parser for multiple log sources with signal intelligence"""

    # Log file mappings
    LOG_FILES = {
        LogSource.FOREX_SCANNER: [
            "/logs/worker/forex_scanner.log",
            "/logs/worker/forex_scanner_*.log",  # Date-based logs
            "/logs/worker/trading-signals.log"
        ],
        LogSource.STREAM_SERVICE: [
            "/logs/stream/fastapi-stream.log"
        ],
        LogSource.TRADE_MONITOR: [
            "/logs/dev/trade_monitor.log"
        ],
        LogSource.FASTAPI_DEV: [
            "/logs/dev/fastapi-dev.log"
        ]
    }

    def __init__(self, base_log_dir: str = "/home/hr/Projects/TradeSystemV1"):
        self.base_log_dir = base_log_dir

        # Enhanced patterns for signal detection
        self.signal_patterns = {
            AlertType.SIGNAL_DETECTED: [
                r"💎.*signal.*detected",
                r"🚀.*signal.*found",
                r"✅.*signal.*generated",
                r"🎯.*signal.*confirmed",
                r"signal.*detected.*confidence",
                r"BULL.*signal|BEAR.*signal",
                r"signal.*found.*strategy",
                r"📊.*CS\.D\.[A-Z]{6}\.MINI\.IP.*(BULL|BEAR)",  # New pattern
                r"Scanner detected.*signals",
                r"signals ready",
                r"Scan completed.*signals"
            ],
            AlertType.SIGNAL_REJECTED: [
                r"🚫.*signal.*rejected",
                r"❌.*signal.*rejected",
                r"⚠️.*signal.*rejected",
                r"signal.*rejected.*reason",
                r"rejected.*macd|rejected.*ema|rejected.*validation",
                r"Strong negative.*histogram",
                r"suggested.*but.*shows",
                r"🚫.*REJECTED.*CS\.D\.[A-Z]{6}\.MINI\.IP",  # New pattern
                r"SELL signal REJECTED|BUY signal REJECTED",
                r"Filtered out.*invalid signals"
            ],
            AlertType.SIGNAL_VALIDATION: [
                r"signal.*validation.*failed",
                r"technical.*validation.*failed",
                r"Complete.*validation.*failed",
                r"validation.*passed",
                r"breakout.*validation",
                r"momentum.*confirmation",
                r"Validating.*signals for trading",
                r"Validation complete.*valid.*invalid",
                r"EMA200 validation",
                r"Intelligence.*passed|Intelligence.*failed"
            ]
        }

        # Stream health patterns (from existing parser)
        self.stream_patterns = {
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

        # Combine all patterns
        self.all_patterns = {**self.signal_patterns, **self.stream_patterns}

    def parse_log_line(self, line: str, source: LogSource = LogSource.SYSTEM) -> Optional[ParsedLogEntry]:
        """Parse a single log line with enhanced signal intelligence"""
        try:
            # Handle different log formats
            if source == LogSource.FOREX_SCANNER:
                return self._parse_forex_scanner_line(line)
            else:
                return self._parse_standard_line(line, source)

        except Exception as e:
            logger.error(f"Error parsing log line: {e}")
            return None

    def _parse_forex_scanner_line(self, line: str) -> Optional[ParsedLogEntry]:
        """Parse forex scanner specific log format"""
        # Forex scanner format: 2025-09-18 11:20:24 CEST - INFO - MESSAGE
        # OR: TIMESTAMP - LEVEL - MODULE - MESSAGE

        patterns = [
            # Format: YYYY-MM-DD HH:MM:SS TIMEZONE - LEVEL - MESSAGE
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) [A-Z]+ - (\w+) - (.+)",
            # Format: TIMESTAMP - LEVEL - MODULE - MESSAGE
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d* - (\w+) - ([^-]+) - (.+)",
            # Format: TIMESTAMP, LEVEL - MODULE - MESSAGE
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - (\w+) - ([^-]+) - (.+)"
        ]

        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                if len(match.groups()) == 3:
                    timestamp_str, level, message = match.groups()
                    module = "forex_scanner"
                else:
                    timestamp_str, level, module, message = match.groups()

                # Parse timestamp
                try:
                    # Handle different timestamp formats
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            timestamp = datetime.strptime(timestamp_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        timestamp = datetime.now()
                except ValueError:
                    timestamp = datetime.now()

                # Extract signal intelligence
                alert_type = self._categorize_message(message)
                epic = self._extract_epic(message)
                signal_context = self._extract_signal_context(message, level)

                return ParsedLogEntry(
                    timestamp=timestamp,
                    level=level.strip(),
                    module=module.strip() if len(match.groups()) > 3 else "forex_scanner",
                    message=message.strip(),
                    alert_type=alert_type.value,
                    log_source=LogSource.FOREX_SCANNER.value,
                    epic=epic,
                    signal_context=signal_context,
                    raw_line=line
                )

        return None

    def _parse_standard_line(self, line: str, source: LogSource) -> Optional[ParsedLogEntry]:
        """Parse standard log format (existing stream service format)"""
        # Standard format: TIMESTAMP - LEVEL - MODULE - MESSAGE
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

        # Extract information
        alert_type = self._categorize_message(message)
        epic = self._extract_epic(message)

        return ParsedLogEntry(
            timestamp=timestamp,
            level=level.strip(),
            module=module.strip(),
            message=message.strip(),
            alert_type=alert_type.value,
            log_source=source.value,
            epic=epic,
            raw_line=line
        )

    def _categorize_message(self, message: str) -> AlertType:
        """Enhanced message categorization with signal intelligence"""
        message_lower = message.lower()

        for alert_type, patterns in self.all_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return alert_type

        # Default categorization based on keywords
        if any(word in message_lower for word in ["error", "failed", "exception"]):
            return AlertType.ERROR
        elif any(word in message_lower for word in ["warning", "⚠️", "🚫"]):
            return AlertType.SIGNAL_REJECTED if "signal" in message_lower else AlertType.SYSTEM
        elif any(word in message_lower for word in ["signal", "detected", "found"]):
            return AlertType.SIGNAL_DETECTED

        return AlertType.SYSTEM

    def _extract_epic(self, message: str) -> Optional[str]:
        """Extract currency pair from message with multiple patterns"""
        patterns = [
            # Standard epic format: CS.D.EURUSD.MINI.IP
            r"CS\.D\.([A-Z]{6})\.MINI\.IP",
            # Direct currency pair mention
            r"\b([A-Z]{3}[A-Z]{3})\b",
            # Currency pairs with separators
            r"\b([A-Z]{3}[/\-_][A-Z]{3})\b"
        ]

        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                pair = match.group(1)
                # Normalize separator
                pair = pair.replace("/", "").replace("-", "").replace("_", "")
                if len(pair) == 6:  # Valid currency pair
                    return pair

        return None

    def _extract_signal_context(self, message: str, level: str) -> Optional[SignalContext]:
        """Extract rich signal context from message"""
        context = SignalContext()

        # Extract strategy
        strategy_patterns = [
            r"(EMA|MACD|ZERO_LAG|KAMA|SMC|BOLLINGER|ICHIMOKU)",
            r"Strategy.*?([A-Z_]+)",
            r"🎯\s*([A-Z][a-z]+)\s*Strategy"
        ]

        for pattern in strategy_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                context.strategy = match.group(1).upper()
                break

        # Extract signal type
        if re.search(r"BULL|bull|Buy|BUY", message):
            context.signal_type = "BULL"
        elif re.search(r"BEAR|bear|Sell|SELL", message):
            context.signal_type = "BEAR"

        # Extract confidence - look for patterns like "BEAR (95.0%)" or "95.0%"
        confidence_patterns = [
            r"\((\d+\.?\d*)%\)",  # Pattern like (95.0%)
            r"(\d+\.?\d*)%",      # General percentage
            r"confidence.*?(\d+\.?\d*)"  # confidence: XX
        ]

        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, message)
            if confidence_match:
                conf_str = confidence_match.group(1)
                try:
                    confidence = float(conf_str)
                    context.confidence = confidence / 100 if confidence > 1 else confidence
                    break
                except ValueError:
                    continue

        # Extract timeframe
        timeframe_match = re.search(r"(\d+[mhd]|[A-Z0-9]+m)", message)
        if timeframe_match:
            context.timeframe = timeframe_match.group(1)

        # Extract rejection reason for rejected signals
        if level == "WARNING" and ("rejected" in message.lower() or "REJECTED" in message):
            # Extract reason after "REJECTED" - handle patterns like:
            # "🚫 SELL signal REJECTED CS.D.AUDJPY.MINI.IP: 97.89900 >= 97.82749 (price at/above EMA200)"
            reason_patterns = [
                r"REJECTED.*?:\s*(.+)",  # After colon
                r"rejected[:\-\s]*(.+)",  # General rejected pattern
                r"\(([^)]+)\)$",  # Text in parentheses at end
            ]

            for pattern in reason_patterns:
                reason_match = re.search(pattern, message, re.IGNORECASE)
                if reason_match:
                    context.rejection_reason = reason_match.group(1).strip()
                    break

        # Extract validation details
        if "validation" in message.lower():
            validation = {}

            # Look for specific validation failures
            if "price data" in message.lower():
                validation["price_data"] = "missing"
            if "macd" in message.lower():
                validation["macd_check"] = "failed"
            if "histogram" in message.lower():
                validation["momentum"] = "negative"

            if validation:
                context.validation_details = validation

        # Return context only if we extracted meaningful information
        if any([context.strategy, context.signal_type, context.confidence,
                context.rejection_reason, context.validation_details]):
            return context

        return None

    def get_recent_logs(self, source: Union[LogSource, str] = None,
                       hours_back: int = 2, max_entries: int = 100) -> List[ParsedLogEntry]:
        """Get recent log entries from specified source(s)"""
        if isinstance(source, str):
            try:
                source = LogSource(source)
            except ValueError:
                source = None

        all_logs = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # Determine which sources to parse
        sources_to_parse = [source] if source else list(LogSource)

        for log_source in sources_to_parse:
            if log_source not in self.LOG_FILES:
                continue

            for log_pattern in self.LOG_FILES[log_source]:
                log_path = os.path.join(self.base_log_dir, log_pattern.lstrip('/'))

                # Handle wildcard patterns
                if '*' in log_path:
                    import glob
                    matching_files = glob.glob(log_path)
                    # Sort by modification time, newest first
                    matching_files.sort(key=os.path.getmtime, reverse=True)
                    # Take most recent file for wildcard patterns
                    if matching_files:
                        log_path = matching_files[0]
                    else:
                        continue

                if not os.path.exists(log_path):
                    continue

                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # Process lines from newest to oldest
                    source_logs = []
                    for line in reversed(lines):
                        parsed_line = self.parse_log_line(line.strip(), log_source)
                        if parsed_line and parsed_line.timestamp >= cutoff_time:
                            source_logs.append(parsed_line)

                        # Limit per source to avoid memory issues, but don't break the outer loop
                        if len(source_logs) >= max_entries:
                            break

                    all_logs.extend(source_logs)

                except Exception as e:
                    logger.error(f"Error reading log file {log_path}: {e}")
                    continue

        # Sort by timestamp (newest first)
        all_logs.sort(key=lambda x: x.timestamp, reverse=True)
        return all_logs[:max_entries]

    def get_signal_intelligence(self, hours_back: int = 4, max_signals: int = 50) -> List[ParsedLogEntry]:
        """Get recent signal-related log entries with rich context"""
        all_logs = self.get_recent_logs(LogSource.FOREX_SCANNER, hours_back, max_entries=500)

        # Filter for signal-related entries
        signal_logs = [
            log for log in all_logs
            if log.alert_type in [
                AlertType.SIGNAL_DETECTED.value,
                AlertType.SIGNAL_REJECTED.value,
                AlertType.SIGNAL_VALIDATION.value
            ]
        ]

        return signal_logs[:max_signals]

    def get_system_health_summary(self, hours_back: int = 1) -> Dict[str, Any]:
        """Enhanced system health summary across all sources"""
        recent_logs = self.get_recent_logs(hours_back=hours_back)

        summary = {
            "total_entries": len(recent_logs),
            "by_source": {},
            "by_level": {"ERROR": 0, "WARNING": 0, "INFO": 0, "DEBUG": 0},
            "signal_stats": {
                "signals_detected": 0,
                "signals_rejected": 0,
                "rejection_reasons": {}
            },
            "last_error": None,
            "last_warning": None,
            "last_signal": None,
            "stream_health": "unknown",
            "forex_scanner_health": "unknown"
        }

        scanner_healthy_indicators = 0
        scanner_issue_indicators = 0
        stream_healthy_indicators = 0
        stream_issue_indicators = 0

        for log in recent_logs:
            # Count by source
            source = log.log_source
            summary["by_source"][source] = summary["by_source"].get(source, 0) + 1

            # Count by level
            if log.level in summary["by_level"]:
                summary["by_level"][log.level] += 1

            # Track signal statistics
            if log.alert_type == AlertType.SIGNAL_DETECTED.value:
                summary["signal_stats"]["signals_detected"] += 1
                if not summary["last_signal"]:
                    summary["last_signal"] = log
            elif log.alert_type == AlertType.SIGNAL_REJECTED.value:
                summary["signal_stats"]["signals_rejected"] += 1
                if log.signal_context and log.signal_context.rejection_reason:
                    reason = log.signal_context.rejection_reason
                    summary["signal_stats"]["rejection_reasons"][reason] = \
                        summary["signal_stats"]["rejection_reasons"].get(reason, 0) + 1

            # Track last error/warning
            if log.level == "ERROR" and not summary["last_error"]:
                summary["last_error"] = log
            elif log.level == "WARNING" and not summary["last_warning"]:
                summary["last_warning"] = log

            # Enhanced health indicators
            message = log.message.lower()

            # Stream health indicators - look for actual stream service patterns
            if any(phrase in message for phrase in [
                "streams healthy", "stream.*connected", "backfill.*complete",
                "candle completed", "✅ no gaps detected", "database stats", "🟢.*candle"
            ]):
                stream_healthy_indicators += 1
            elif any(phrase in message for phrase in [
                "stream.*disconnected", "stream.*failed", "data.*stale", "gap.*detected", "connection.*lost"
            ]):
                stream_issue_indicators += 1

            # Forex scanner health indicators - look for actual patterns from logs
            if any(phrase in message for phrase in [
                "scanner.*initialized", "clean scanner initialized", "intelligentforexscanner initialized",
                "✅.*initialized", "database connection established", "scan completed.*signals"
            ]):
                scanner_healthy_indicators += 1
            elif any(phrase in message for phrase in [
                "scanner.*failed", "initialization.*failed", "❌", "database.*failed", "connection.*lost"
            ]):
                scanner_issue_indicators += 1

        # Determine health status based on indicators
        if stream_healthy_indicators > stream_issue_indicators and stream_healthy_indicators > 0:
            summary["stream_health"] = "healthy"
        elif stream_issue_indicators > 0:
            summary["stream_health"] = "issues"

        if scanner_healthy_indicators > scanner_issue_indicators and scanner_healthy_indicators > 0:
            summary["forex_scanner_health"] = "healthy"
        elif scanner_issue_indicators > 0:
            summary["forex_scanner_health"] = "issues"

        # If we have recent successful signal processing, scanner is likely healthy
        if summary["signal_stats"]["signals_detected"] > 0 or summary["signal_stats"]["signals_rejected"] > 0:
            summary["forex_scanner_health"] = "healthy"

        return summary

def get_enhanced_log_parser() -> EnhancedLogParser:
    """Get a configured enhanced log parser instance"""
    return EnhancedLogParser()

if __name__ == "__main__":
    # Test the enhanced log parser
    parser = EnhancedLogParser()

    print("Testing enhanced log parser...")

    # Get recent signal intelligence
    signals = parser.get_signal_intelligence(hours_back=4)
    print(f"\nFound {len(signals)} recent signal events:")
    for signal in signals[:5]:
        context = signal.signal_context
        if context:
            print(f"  {signal.timestamp} [{signal.level}] {context.strategy or 'Unknown'} "
                  f"{context.signal_type or 'Unknown'} - {signal.message[:60]}...")
        else:
            print(f"  {signal.timestamp} [{signal.level}] {signal.message[:60]}...")

    # Get system health
    health = parser.get_system_health_summary()
    print(f"\nSystem Health Summary:")
    print(f"  Total entries: {health['total_entries']}")
    print(f"  By source: {health['by_source']}")
    print(f"  Signal stats: {health['signal_stats']}")
    print(f"  Stream health: {health['stream_health']}")
    print(f"  Scanner health: {health['forex_scanner_health']}")