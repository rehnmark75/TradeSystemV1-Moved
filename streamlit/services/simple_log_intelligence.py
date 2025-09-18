"""
Simple Log Intelligence Service - Lightweight version with minimal dependencies
Works reliably in containerized environments without external dependencies
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class SimpleLogParser:
    """Lightweight log parser with minimal dependencies"""

    def __init__(self, base_log_dir: str = None):
        # Auto-detect if running in container or on host
        if base_log_dir is None:
            if os.path.exists('/logs/worker'):
                self.base_log_dir = ""  # Container path
            else:
                self.base_log_dir = "/home/hr/Projects/TradeSystemV1"  # Host path

        # Log file paths (absolute paths for container compatibility)
        if self.base_log_dir == "":  # Container
            self.log_files = {
                'forex_scanner': [
                    '/logs/worker/forex_scanner_20250918.log',  # Today's file
                    '/logs/worker/forex_scanner_20250917.log',  # Yesterday
                    '/logs/worker/trading-signals.log'
                ],
                'stream_service': [
                    '/logs/stream/fastapi-stream.log'
                ],
                'trade_monitor': [
                    '/logs/dev/trade_monitor.log'
                ]
            }
        else:  # Host
            self.log_files = {
                'forex_scanner': [
                    'logs/worker/forex_scanner_20250918.log',  # Today's file
                    'logs/worker/forex_scanner_20250917.log',  # Yesterday
                    'logs/worker/trading-signals.log'
                ],
                'stream_service': [
                    'logs/stream/fastapi-stream.log'
                ],
                'trade_monitor': [
                    'logs/dev/trade_monitor.log'
                ]
            }

        # Signal detection patterns
        self.signal_patterns = {
            'detected': [
                r'📊.*CS\.D\.[A-Z]{6}\.MINI\.IP.*(BULL|BEAR)',
                r'Scanner detected.*signals',
                r'signals ready',
                r'Scan completed.*signals'
            ],
            'rejected': [
                r'🚫.*REJECTED.*CS\.D\.[A-Z]{6}\.MINI\.IP',
                r'SELL signal REJECTED|BUY signal REJECTED',
                r'Filtered out.*invalid signals'
            ]
        }

        # Health patterns
        self.health_patterns = {
            'scanner_healthy': [
                r'Clean scanner initialized',
                r'IntelligentForexScanner initialized',
                r'✅.*initialized',
                r'Database connection established',
                r'Scan completed.*signals'
            ],
            'stream_healthy': [
                r'candle completed',
                r'✅ No gaps detected',
                r'Database stats',
                r'🟢.*candle'
            ]
        }

    def get_recent_signal_data(self, hours_back: int = 4) -> Dict[str, Any]:
        """Get recent signal data from logs"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        signals_detected = 0
        signals_rejected = 0
        confidences = []
        epics = set()

        # Read forex scanner logs
        for log_file in self.log_files['forex_scanner']:
            if self.base_log_dir == "":  # Container - use absolute paths
                file_path = log_file
            else:  # Host - join with base dir
                file_path = os.path.join(self.base_log_dir, log_file)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Parse timestamp
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if not timestamp_match:
                            continue

                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                            if log_time < cutoff_time:
                                continue
                        except ValueError:
                            continue

                        # Check for signals
                        line_lower = line.lower()

                        # Count detected signals
                        for pattern in self.signal_patterns['detected']:
                            if re.search(pattern, line, re.IGNORECASE):
                                signals_detected += 1

                                # Extract confidence
                                conf_match = re.search(r'\((\d+\.?\d*)%\)', line)
                                if conf_match:
                                    try:
                                        conf = float(conf_match.group(1)) / 100
                                        confidences.append(conf)
                                    except ValueError:
                                        pass

                                # Extract epic
                                epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                                if epic_match:
                                    epics.add(epic_match.group(1))
                                break

                        # Count rejected signals
                        for pattern in self.signal_patterns['rejected']:
                            if re.search(pattern, line, re.IGNORECASE):
                                signals_rejected += 1

                                # Extract epic from rejection
                                epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                                if epic_match:
                                    epics.add(epic_match.group(1))
                                break

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

        # Calculate metrics
        total_signals = signals_detected + signals_rejected
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        success_rate = signals_detected / max(1, total_signals) if total_signals > 0 else 0.0

        return {
            'total_signals': total_signals,
            'signals_detected': signals_detected,
            'signals_rejected': signals_rejected,
            'avg_confidence': avg_confidence,
            'success_rate': success_rate * 0.8,  # Conservative estimate
            'top_epic': list(epics)[0] if epics else None,
            'active_pairs': len(epics)
        }

    def get_system_health(self, hours_back: int = 2) -> Dict[str, Any]:
        """Get system health status"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        scanner_healthy_count = 0
        scanner_error_count = 0
        stream_healthy_count = 0
        stream_error_count = 0

        total_errors = 0
        total_warnings = 0
        last_error = None
        last_warning = None

        # Check forex scanner health
        for log_file in self.log_files['forex_scanner']:
            if self.base_log_dir == "":  # Container - use absolute paths
                file_path = log_file
            else:  # Host - join with base dir
                file_path = os.path.join(self.base_log_dir, log_file)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Parse timestamp
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if not timestamp_match:
                            continue

                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                            if log_time < cutoff_time:
                                continue
                        except ValueError:
                            continue

                        # Check health patterns
                        for pattern in self.health_patterns['scanner_healthy']:
                            if re.search(pattern, line, re.IGNORECASE):
                                scanner_healthy_count += 1
                                break

                        # Count errors and warnings
                        if ' - ERROR - ' in line:
                            total_errors += 1
                            scanner_error_count += 1
                            if not last_error:
                                last_error = {'time': log_time, 'message': line.strip()}
                        elif ' - WARNING - ' in line:
                            total_warnings += 1
                            if not last_warning:
                                last_warning = {'time': log_time, 'message': line.strip()}

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

        # Check stream service health
        for log_file in self.log_files['stream_service']:
            if self.base_log_dir == "":  # Container - use absolute paths
                file_path = log_file
            else:  # Host - join with base dir
                file_path = os.path.join(self.base_log_dir, log_file)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Parse timestamp
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if not timestamp_match:
                            continue

                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                            if log_time < cutoff_time:
                                continue
                        except ValueError:
                            continue

                        # Check health patterns
                        for pattern in self.health_patterns['stream_healthy']:
                            if re.search(pattern, line, re.IGNORECASE):
                                stream_healthy_count += 1
                                break

                        # Count errors
                        if ' - ERROR - ' in line:
                            total_errors += 1
                            stream_error_count += 1

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

        # Determine health status
        scanner_health = "healthy" if scanner_healthy_count > scanner_error_count and scanner_healthy_count > 0 else "unknown"
        stream_health = "healthy" if stream_healthy_count > stream_error_count and stream_healthy_count > 0 else "unknown"

        # Overall status
        if total_errors > 10:
            overall_status = "critical"
        elif total_errors > 0 or total_warnings > 20:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return {
            'overall_status': overall_status,
            'forex_scanner_health': scanner_health,
            'stream_health': stream_health,
            'error_count_24h': total_errors,
            'warning_count_24h': total_warnings,
            'last_error': last_error,
            'last_warning': last_warning,
            'scanner_indicators': scanner_healthy_count,
            'stream_indicators': stream_healthy_count
        }

    def get_recent_activity(self, hours_back: int = 1, max_entries: int = 10) -> List[Dict[str, Any]]:
        """Get recent signal activity"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        activities = []

        # Read recent forex scanner logs (try multiple files)
        for log_file in self.log_files['forex_scanner']:
            if self.base_log_dir == "":  # Container - use absolute paths
                file_path = log_file
            else:  # Host - join with base dir
                file_path = os.path.join(self.base_log_dir, log_file)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Process recent lines from end of file
                for line in reversed(lines[-2000:]):  # Check more lines to catch recent signals
                    # Parse timestamp
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if not timestamp_match:
                        continue

                    try:
                        log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        # Use a more lenient time filter for recent activity
                        if log_time < cutoff_time:
                            continue
                    except ValueError:
                        continue

                    # Check for interesting activity
                    activity = None

                    # Signal detected - look for the signal line with confidence
                    if re.search(r'📊.*CS\.D\.[A-Z]{6}\.MINI\.IP.*(BULL|BEAR).*\(\d+\.?\d*%\)', line):
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        conf_match = re.search(r'\((\d+\.?\d*)%\)', line)
                        signal_match = re.search(r'(BULL|BEAR)', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'signal_detected',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'signal_type': signal_match.group(1) if signal_match else 'Unknown',
                            'confidence': float(conf_match.group(1))/100 if conf_match else None,
                            'message': line.strip()
                        }

                    # Signal rejected
                    elif re.search(r'🚫.*REJECTED.*CS\.D\.[A-Z]{6}\.MINI\.IP', line):
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        reason_match = re.search(r'REJECTED.*?:\s*(.+)', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'signal_rejected',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'reason': reason_match.group(1).strip() if reason_match else 'Validation failed',
                            'message': line.strip()
                        }

                    if activity:
                        activities.append(activity)

                        # Continue collecting from this file
                        if len(activities) >= max_entries * 2:  # Collect more then filter
                            break

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

            # If we found enough activities, stop looking at more files
            if len(activities) >= max_entries:
                break

        # Sort by timestamp (newest first) and return top entries
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        return activities[:max_entries]

def get_simple_log_intelligence() -> SimpleLogParser:
    """Get a simple log intelligence parser"""
    return SimpleLogParser()

if __name__ == "__main__":
    # Test the simple parser
    parser = SimpleLogParser()

    print("=== TESTING SIMPLE LOG INTELLIGENCE ===")

    # Test signal data
    signals = parser.get_recent_signal_data(hours_back=4)
    print(f"\nSignal Data (4h):")
    print(f"  Total: {signals['total_signals']}")
    print(f"  Detected: {signals['signals_detected']}")
    print(f"  Rejected: {signals['signals_rejected']}")
    print(f"  Avg Confidence: {signals['avg_confidence']:.1%}")
    print(f"  Top Epic: {signals['top_epic']}")

    # Test health
    health = parser.get_system_health()
    print(f"\nSystem Health:")
    print(f"  Overall: {health['overall_status']}")
    print(f"  Scanner: {health['forex_scanner_health']}")
    print(f"  Stream: {health['stream_health']}")
    print(f"  Errors: {health['error_count_24h']}")

    # Test activity
    activities = parser.get_recent_activity()
    print(f"\nRecent Activities ({len(activities)}):")
    for activity in activities[:3]:
        print(f"  {activity['timestamp']} - {activity['type']} - {activity.get('epic', 'N/A')}")