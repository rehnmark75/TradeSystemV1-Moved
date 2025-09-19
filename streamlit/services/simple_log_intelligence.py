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

        # Generate today's and yesterday's log file names dynamically
        today = datetime.now().strftime("%Y%m%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

        # Log file paths (absolute paths for container compatibility)
        if self.base_log_dir == "":  # Container
            self.log_files = {
                'forex_scanner': [
                    f'/logs/worker/forex_scanner_{today}.log',  # Today's file
                    f'/logs/worker/forex_scanner_{yesterday}.log',  # Yesterday
                    '/logs/worker/trading-signals.log'
                ],
                'stream_service': [
                    '/logs/stream/fastapi-stream.log'
                ],
                'trade_monitor': [
                    '/logs/dev/trade_monitor.log'
                ],
                'fastapi_dev': [
                    '/logs/dev/fastapi-dev.log'
                ],
                'dev_trade': [
                    '/logs/dev/dev-trade.log'
                ],
                'trade_sync': [
                    '/logs/dev/trade_sync.log'
                ]
            }
        else:  # Host
            self.log_files = {
                'forex_scanner': [
                    f'logs/worker/forex_scanner_{today}.log',  # Today's file
                    f'logs/worker/forex_scanner_{yesterday}.log',  # Yesterday
                    'logs/worker/trading-signals.log'
                ],
                'stream_service': [
                    'logs/stream/fastapi-stream.log'
                ],
                'trade_monitor': [
                    'logs/dev/trade_monitor.log'
                ],
                'fastapi_dev': [
                    'logs/dev/fastapi-dev.log'
                ],
                'dev_trade': [
                    'logs/dev/dev-trade.log'
                ],
                'trade_sync': [
                    'logs/dev/trade_sync.log'
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

        # Trade event patterns
        self.trade_patterns = {
            'trade_opened': [
                r'✅ Trade logged: CS\.D\.[A-Z]{6}\.MINI\.IP.*?(BUY|SELL)',
                r'Place-Order: Parsed EPIC.*Direction: (BUY|SELL)',
                r'No open position for CS\.D\.[A-Z]{6}\.MINI\.IP, placing order'
            ],
            'trade_closed': [
                r'Trade.*closed.*profit',
                r'Position closed.*CS\.D\.[A-Z]{6}\.MINI\.IP',
                r'Deal closed.*CS\.D\.[A-Z]{6}\.MINI\.IP'
            ],
            'trade_monitoring': [
                r'📊 \[PROFIT\] Trade \d+.*(BUY|SELL): entry=.*profit=.*pts',
                r'🔧 \[COMBINED\] Processing trade \d+ CS\.D\.[A-Z]{6}\.MINI\.IP',
                r'📊 \[TRAILING CONFIG\] CS\.D\.[A-Z]{6}\.MINI\.IP'
            ],
            'trade_adjustments': [
                r'\[ADJUST-STOP\] CS\.D\.[A-Z]{6}\.MINI\.IP',
                r'Stop level.*→ New:',
                r'Limit level.*→ New:'
            ],
            'trailing_events': [
                r'🎯 \[BREAK-EVEN TRIGGER\] Trade \d+:',
                r'💰 \[STAGE 2 TRIGGER\] Trade \d+:',
                r'🚀 \[STAGE 3 TRIGGER\] Trade \d+:',
                r'🎉 \[BREAK-EVEN\] Trade \d+',
                r'💎 \[STAGE 2\] Trade \d+:',
                r'🎯 \[STAGE 3\] Trade \d+:',
                r'🎯 \[TRAILING SUCCESS\] Trade \d+',
                r'\[PROGRESSIVE STAGE \d+\] CS\.D\.[A-Z]{6}\.MINI\.IP',
                r'\[PERCENTAGE TRAIL\] CS\.D\.[A-Z]{6}\.MINI\.IP:',
                r'\[INTELLIGENT TRAIL\] CS\.D\.[A-Z]{6}\.MINI\.IP',
                r'📊 \[TRAILING CONFIG\] CS\.D\.[A-Z]{6}\.MINI\.IP',
                r'\[SAFE DISTANCE RESULT\] Trade \d+',
                r'Stage \d+:.*trailing',
                r'trailingStopDistance',
                r'trailingStep'
            ]
        }

        # Health patterns
        self.health_patterns = {
            'scanner_healthy': [
                r'Clean scanner initialized',
                r'IntelligentForexScanner initialized',
                r'✅.*initialized',
                r'Database connection established',
                r'Scan completed.*signals',
                r'Scanner detected.*signals',
                r'Enhanced data for.*bars',
                r'EMA validation passed',
                r'✅.*validation passed'
            ],
            'stream_healthy': [
                r'candle completed',
                r'✅ No gaps detected',
                r'Database stats',
                r'🟢.*candle'
            ],
            'trade_monitor_healthy': [
                r'🔄 === Enhanced Monitoring Cycle',
                r'Processing \d+ active trades',
                r'✅ \[ACTIVE\] Trade \d+ position still active',
                r'Cycle #\d+ complete'
            ],
            'fastapi_healthy': [
                r'HTTP Request:.*200 OK',
                r'Processing trade \d+.*status=tracking',
                r'Successfully.*trade'
            ]
        }

    def get_recent_signal_data(self, hours_back: int = 4) -> Dict[str, Any]:
        """Get recent signal data from logs"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        signals_detected = 0
        signals_rejected = 0
        confidences = []
        epics = set()

        # Read all signal-related logs (scanner + trade events)
        all_signal_sources = self.log_files['forex_scanner'] + self.log_files.get('dev_trade', [])
        for log_file in all_signal_sources:
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

    def get_recent_trade_events(self, hours_back: int = 4) -> Dict[str, Any]:
        """Get recent trade events from dev logs"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        trade_opened = 0
        trade_closed = 0
        trade_monitoring = 0
        trade_adjustments = 0
        active_trades = set()

        # Read trade-related logs
        trade_sources = (self.log_files.get('fastapi_dev', []) +
                        self.log_files.get('dev_trade', []) +
                        self.log_files.get('trade_monitor', []))

        for log_file in trade_sources:
            if self.base_log_dir == "":
                file_path = log_file
            else:
                file_path = os.path.join(self.base_log_dir, log_file)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Parse timestamp (handle different formats)
                        timestamp_match = (re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line) or
                                         re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+', line))
                        if not timestamp_match:
                            continue

                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                            if log_time < cutoff_time:
                                continue
                        except ValueError:
                            continue

                        # Check for trade events
                        for pattern in self.trade_patterns['trade_opened']:
                            if re.search(pattern, line, re.IGNORECASE):
                                trade_opened += 1
                                # Extract epic
                                epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                                if epic_match:
                                    active_trades.add(epic_match.group(1))
                                break

                        for pattern in self.trade_patterns['trade_closed']:
                            if re.search(pattern, line, re.IGNORECASE):
                                trade_closed += 1
                                break

                        for pattern in self.trade_patterns['trade_monitoring']:
                            if re.search(pattern, line, re.IGNORECASE):
                                trade_monitoring += 1
                                # Extract epic and trade ID
                                epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                                if epic_match:
                                    active_trades.add(epic_match.group(1))
                                break

                        for pattern in self.trade_patterns['trade_adjustments']:
                            if re.search(pattern, line, re.IGNORECASE):
                                trade_adjustments += 1
                                break

                        # Count trailing events
                        for pattern in self.trade_patterns.get('trailing_events', []):
                            if re.search(pattern, line, re.IGNORECASE):
                                trade_adjustments += 1  # Include in adjustments count
                                break

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

        return {
            'trade_opened': trade_opened,
            'trade_closed': trade_closed,
            'trade_monitoring': trade_monitoring,
            'trade_adjustments': trade_adjustments,
            'active_trades': len(active_trades),
            'active_trade_pairs': list(active_trades)
        }

    def get_system_health(self, hours_back: int = 2) -> Dict[str, Any]:
        """Get system health status"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        scanner_healthy_count = 0
        scanner_error_count = 0
        stream_healthy_count = 0
        stream_error_count = 0
        trade_monitor_healthy_count = 0
        trade_monitor_error_count = 0
        fastapi_healthy_count = 0
        fastapi_error_count = 0

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

        # Check trade monitor health
        for log_file in self.log_files.get('trade_monitor', []):
            if self.base_log_dir == "":
                file_path = log_file
            else:
                file_path = os.path.join(self.base_log_dir, log_file)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if not timestamp_match:
                            continue

                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                            if log_time < cutoff_time:
                                continue
                        except ValueError:
                            continue

                        for pattern in self.health_patterns['trade_monitor_healthy']:
                            if re.search(pattern, line, re.IGNORECASE):
                                trade_monitor_healthy_count += 1
                                break

                        if ' - ERROR - ' in line or ' | ERROR |' in line:
                            total_errors += 1
                            trade_monitor_error_count += 1

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

        # Check FastAPI dev health
        for log_file in self.log_files.get('fastapi_dev', []):
            if self.base_log_dir == "":
                file_path = log_file
            else:
                file_path = os.path.join(self.base_log_dir, log_file)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if not timestamp_match:
                            continue

                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                            if log_time < cutoff_time:
                                continue
                        except ValueError:
                            continue

                        for pattern in self.health_patterns['fastapi_healthy']:
                            if re.search(pattern, line, re.IGNORECASE):
                                fastapi_healthy_count += 1
                                break

                        if ' - ERROR - ' in line:
                            total_errors += 1
                            fastapi_error_count += 1

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

        # Determine health status with recent activity weighting
        # Check recent activity (last 30 minutes) to prioritize current status
        recent_cutoff = datetime.now() - timedelta(minutes=30)

        # Get recent health indicators for more accurate current status
        recent_scanner_healthy = 0
        recent_scanner_errors = 0

        # Quick check of recent activity for scanner
        for log_file in self.log_files['forex_scanner']:
            if self.base_log_dir == "":
                file_path = log_file
            else:
                file_path = os.path.join(self.base_log_dir, log_file)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read last 100 lines for recent activity
                    lines = f.readlines()
                    for line in lines[-100:]:
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if timestamp_match:
                            try:
                                log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                                if log_time >= recent_cutoff:
                                    # Check for recent healthy patterns
                                    for pattern in self.health_patterns['scanner_healthy']:
                                        if re.search(pattern, line, re.IGNORECASE):
                                            recent_scanner_healthy += 1
                                            break
                                    # Check for recent errors
                                    if ' - ERROR - ' in line:
                                        recent_scanner_errors += 1
                            except ValueError:
                                continue
            except Exception:
                continue

        # Enhanced health logic: If recent activity is healthy, consider service healthy
        # even if historical errors exist
        if recent_scanner_healthy > 0 and recent_scanner_errors == 0:
            scanner_health = "healthy"
        elif scanner_healthy_count > scanner_error_count and scanner_healthy_count > 0:
            scanner_health = "healthy"
        elif recent_scanner_healthy > recent_scanner_errors and recent_scanner_healthy > 0:
            scanner_health = "healthy"
        else:
            scanner_health = "unknown"

        stream_health = "healthy" if stream_healthy_count > stream_error_count and stream_healthy_count > 0 else "unknown"
        trade_monitor_health = "healthy" if trade_monitor_healthy_count > trade_monitor_error_count and trade_monitor_healthy_count > 0 else "unknown"
        fastapi_health = "healthy" if fastapi_healthy_count > fastapi_error_count and fastapi_healthy_count > 0 else "unknown"

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
            'trade_monitor_health': trade_monitor_health,
            'fastapi_health': fastapi_health,
            'error_count_24h': total_errors,
            'warning_count_24h': total_warnings,
            'last_error': last_error,
            'last_warning': last_warning,
            'scanner_indicators': scanner_healthy_count,
            'stream_indicators': stream_healthy_count,
            'trade_monitor_indicators': trade_monitor_healthy_count,
            'fastapi_indicators': fastapi_healthy_count
        }

    def get_recent_activity(self, hours_back: int = 1, max_entries: int = 10) -> List[Dict[str, Any]]:
        """Get recent signal activity"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        activities = []

        # Read recent logs from multiple sources
        all_activity_sources = (self.log_files['forex_scanner'] +
                               self.log_files.get('dev_trade', []) +
                               self.log_files.get('fastapi_dev', []) +
                               self.log_files.get('trade_monitor', []))

        for log_file in all_activity_sources:
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

                    # Trade opened - enhanced extraction
                    elif re.search(r'✅ Trade logged: CS\.D\.[A-Z]{6}\.MINI\.IP.*?(BUY|SELL)', line):
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        direction_match = re.search(r'(BUY|SELL)', line)
                        price_match = re.search(r'(\d+\.?\d*)\s+(BUY|SELL)', line)

                        # Try to extract deal reference from next lines or context
                        deal_ref_match = re.search(r'"dealReference": "([^"]+)"', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trade_opened',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'direction': direction_match.group(1) if direction_match else 'Unknown',
                            'entry_price': float(price_match.group(1)) if price_match else None,
                            'deal_reference': deal_ref_match.group(1) if deal_ref_match else None,
                            'message': line.strip()
                        }

                    # Trade monitoring/profit update - enhanced extraction
                    elif re.search(r'📊 \[PROFIT\] Trade \d+.*(BUY|SELL): entry=.*profit=.*pts', line):
                        # Extract all available data from the profit line
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        direction_match = re.search(r'(BUY|SELL)', line)
                        entry_match = re.search(r'entry=([0-9.]+)', line)
                        current_match = re.search(r'current=([0-9.]+)', line)
                        profit_match = re.search(r'profit=([+-]?\d+)pts', line)
                        trigger_match = re.search(r'trigger=(\d+)pts', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)

                        # Calculate additional metrics
                        entry_price = float(entry_match.group(1)) if entry_match else None
                        current_price = float(current_match.group(1)) if current_match else None
                        profit_pts = int(profit_match.group(1)) if profit_match else 0
                        trigger_pts = int(trigger_match.group(1)) if trigger_match else 0

                        # Calculate percentage move if prices available
                        price_move_pct = None
                        if entry_price and current_price and entry_price > 0:
                            price_move = ((current_price - entry_price) / entry_price) * 100
                            if direction_match and direction_match.group(1) == 'SELL':
                                price_move = -price_move  # Invert for SELL positions
                            price_move_pct = round(price_move, 4)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trade_monitoring',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'direction': direction_match.group(1) if direction_match else 'Unknown',
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'profit_pts': profit_pts,
                            'trigger_pts': trigger_pts,
                            'price_move_pct': price_move_pct,
                            'progress_to_trigger': round((profit_pts / max(1, trigger_pts)) * 100, 1) if trigger_pts > 0 else 0,
                            'message': line.strip()
                        }

                    # Trade adjustments - enhanced extraction
                    elif re.search(r'\[ADJUST-STOP\] CS\.D\.[A-Z]{6}\.MINI\.IP', line):
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        direction_match = re.search(r'Direction: (BUY|SELL)', line)

                        # Extract stop level changes
                        old_stop_match = re.search(r'Old stop level: ([0-9.]+)', line)
                        new_stop_match = re.search(r'New: ([0-9.]+)', line)

                        # Extract limit level changes
                        old_limit_match = re.search(r'Old limit level: ([0-9.]+)', line)
                        new_limit_match = re.search(r'New: ([0-9.]+)', line)

                        # Determine adjustment type
                        adjustment_type = 'stop_loss'
                        if 'limit' in line.lower():
                            adjustment_type = 'take_profit'

                        activity = {
                            'timestamp': log_time,
                            'type': 'trade_adjustment',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'direction': direction_match.group(1) if direction_match else 'Unknown',
                            'adjustment_type': adjustment_type,
                            'old_stop_level': float(old_stop_match.group(1)) if old_stop_match else None,
                            'new_stop_level': float(new_stop_match.group(1)) if new_stop_match else None,
                            'old_limit_level': float(old_limit_match.group(1)) if old_limit_match else None,
                            'new_limit_level': float(new_limit_match.group(1)) if new_limit_match else None,
                            'message': line.strip()
                        }

                    # Trailing events - break-even triggers
                    elif re.search(r'🎯 \[BREAK-EVEN TRIGGER\] Trade \d+:', line):
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        profit_match = re.search(r'Profit (\d+)pts', line)
                        trigger_match = re.search(r'trigger (\d+)pts', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_breakeven_trigger',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'profit_pts': int(profit_match.group(1)) if profit_match else None,
                            'trigger_pts': int(trigger_match.group(1)) if trigger_match else None,
                            'stage': 'stage1_breakeven',
                            'message': line.strip()
                        }

                    # Stage 2 profit lock triggers
                    elif re.search(r'💰 \[STAGE 2 TRIGGER\] Trade \d+:', line):
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        profit_match = re.search(r'Profit (\d+)pts', line)
                        trigger_match = re.search(r'trigger (\d+)pts', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_stage2_trigger',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'profit_pts': int(profit_match.group(1)) if profit_match else None,
                            'trigger_pts': int(trigger_match.group(1)) if trigger_match else None,
                            'stage': 'stage2_profit_lock',
                            'message': line.strip()
                        }

                    # Stage 3 percentage trailing triggers
                    elif re.search(r'🚀 \[STAGE 3 TRIGGER\] Trade \d+:', line):
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        profit_match = re.search(r'Profit (\d+)pts', line)
                        trigger_match = re.search(r'trigger (\d+)pts', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_stage3_trigger',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'profit_pts': int(profit_match.group(1)) if profit_match else None,
                            'trigger_pts': int(trigger_match.group(1)) if trigger_match else None,
                            'stage': 'stage3_percentage_trailing',
                            'message': line.strip()
                        }

                    # Break-even execution
                    elif re.search(r'🎉 \[BREAK-EVEN\] Trade \d+', line):
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        stop_level_match = re.search(r'(\d+\.\d+)', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_breakeven_executed',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'new_stop_level': float(stop_level_match.group(1)) if stop_level_match else None,
                            'stage': 'stage1_breakeven',
                            'message': line.strip()
                        }

                    # Stage 2 profit lock execution
                    elif re.search(r'💎 \[STAGE 2\] Trade \d+:', line):
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        stop_level_match = re.search(r'locked at ([0-9.]+)', line)
                        lock_points_match = re.search(r'\+(\d+)pts', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_stage2_executed',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'new_stop_level': float(stop_level_match.group(1)) if stop_level_match else None,
                            'lock_points': int(lock_points_match.group(1)) if lock_points_match else None,
                            'stage': 'stage2_profit_lock',
                            'message': line.strip()
                        }

                    # Stage 3 percentage trailing execution
                    elif re.search(r'🎯 \[STAGE 3\] Trade \d+:', line):
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        stop_level_match = re.search(r'trailing to ([0-9.]+)', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_stage3_executed',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'new_stop_level': float(stop_level_match.group(1)) if stop_level_match else None,
                            'stage': 'stage3_percentage_trailing',
                            'message': line.strip()
                        }

                    # General trailing success events
                    elif re.search(r'🎯 \[TRAILING SUCCESS\] Trade \d+', line):
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        stop_level_match = re.search(r'Stop moved to ([0-9.]+)', line)
                        adjustment_match = re.search(r'\((\d+) pts\)', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_success',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'new_stop_level': float(stop_level_match.group(1)) if stop_level_match else None,
                            'adjustment_pts': int(adjustment_match.group(1)) if adjustment_match else None,
                            'stage': 'general_trailing',
                            'message': line.strip()
                        }

                    # Progressive stage events
                    elif re.search(r'\[PROGRESSIVE STAGE \d+\] CS\.D\.[A-Z]{6}\.MINI\.IP', line):
                        stage_match = re.search(r'STAGE (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        profit_match = re.search(r'Profit: (\d+)pts', line)
                        trail_match = re.search(r'Trail: ([0-9.]+)', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_progressive_stage',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'stage_number': int(stage_match.group(1)) if stage_match else None,
                            'profit_pts': int(profit_match.group(1)) if profit_match else None,
                            'trail_level': float(trail_match.group(1)) if trail_match else None,
                            'stage': f'progressive_stage_{stage_match.group(1) if stage_match else "unknown"}',
                            'message': line.strip()
                        }

                    # Percentage trail calculations
                    elif re.search(r'\[PERCENTAGE TRAIL\] CS\.D\.[A-Z]{6}\.MINI\.IP:', line):
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        profit_match = re.search(r'Profit ([0-9.]+)pts', line)
                        retracement_match = re.search(r'(\d+)% retracement', line)
                        distance_match = re.search(r'([0-9.]+)pts trail distance', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_percentage_calculation',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'profit_pts': float(profit_match.group(1)) if profit_match else None,
                            'retracement_percentage': int(retracement_match.group(1)) if retracement_match else None,
                            'trail_distance_pts': float(distance_match.group(1)) if distance_match else None,
                            'stage': 'percentage_calculation',
                            'message': line.strip()
                        }

                    # Intelligent trail calculations
                    elif re.search(r'\[INTELLIGENT TRAIL\] CS\.D\.[A-Z]{6}\.MINI\.IP', line):
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        direction_match = re.search(r'(BUY|SELL)', line)
                        current_match = re.search(r'current=([0-9.]+)', line)
                        stop_match = re.search(r'current_stop=([0-9.]+)', line)
                        trail_match = re.search(r'trail_level=([0-9.]+)', line)
                        distance_match = re.search(r'distance_from_current=([0-9.]+)pts', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_intelligent_calculation',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'direction': direction_match.group(1) if direction_match else 'Unknown',
                            'current_price': float(current_match.group(1)) if current_match else None,
                            'current_stop': float(stop_match.group(1)) if stop_match else None,
                            'calculated_trail_level': float(trail_match.group(1)) if trail_match else None,
                            'distance_from_current_pts': float(distance_match.group(1)) if distance_match else None,
                            'stage': 'intelligent_calculation',
                            'message': line.strip()
                        }

                    # Trailing configuration events
                    elif re.search(r'📊 \[TRAILING CONFIG\] CS\.D\.[A-Z]{6}\.MINI\.IP', line):
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_config',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'stage': 'configuration',
                            'message': line.strip()
                        }

                    # Safe distance calculation events
                    elif re.search(r'\[SAFE DISTANCE RESULT\] Trade \d+', line):
                        trade_id_match = re.search(r'Trade (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        distance_match = re.search(r'Returning ([0-9.]+) points', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_safe_distance',
                            'trade_id': trade_id_match.group(1) if trade_id_match else 'Unknown',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'safe_distance_pts': float(distance_match.group(1)) if distance_match else None,
                            'stage': 'distance_calculation',
                            'message': line.strip()
                        }

                    # Stage configuration lines (e.g., "Stage 1: Break-even at +6pts")
                    elif re.search(r'Stage \d+:.*trailing|Stage \d+:.*Break-even|Stage \d+:.*Profit lock', line):
                        stage_match = re.search(r'Stage (\d+)', line)
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_stage_config',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'stage_number': int(stage_match.group(1)) if stage_match else None,
                            'stage': f'stage_{stage_match.group(1) if stage_match else "unknown"}_config',
                            'message': line.strip()
                        }

                    # Trailing stop details (trailingStopDistance, trailingStep)
                    elif re.search(r'trailingStopDistance|trailingStep', line):
                        epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                        trailing_distance_match = re.search(r'trailingStopDistance["\']?\s*:\s*([0-9.]+)', line)
                        trailing_step_match = re.search(r'trailingStep["\']?\s*:\s*([0-9.]+)', line)

                        activity = {
                            'timestamp': log_time,
                            'type': 'trailing_api_details',
                            'epic': epic_match.group(1) if epic_match else 'Unknown',
                            'trailing_stop_distance': float(trailing_distance_match.group(1)) if trailing_distance_match else None,
                            'trailing_step': float(trailing_step_match.group(1)) if trailing_step_match else None,
                            'stage': 'api_details',
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