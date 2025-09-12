# core/monitoring/session_manager.py
"""
Session Manager - Extracted from IntelligentForexScanner  
Handles scanning sessions, performance tracking, and statistics
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class SessionManager:
    """
    Manages scanning sessions, tracks performance, and provides statistics
    Extracted from IntelligentForexScanner for modular session handling
    """
    
    def __init__(self,
                 timezone_manager=None,
                 logger: Optional[logging.Logger] = None):
        
        self.timezone_manager = timezone_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Session state
        self.session_active = False
        self.start_time = None
        self.end_time = None
        
        # Session statistics
        self.scan_count = 0
        self.signal_count = 0
        self.processed_signals = 0
        self.approved_signals = 0
        self.claude_analyzed_signals = 0
        self.notifications_sent = 0
        
        # Performance tracking
        self.scan_durations = []
        self.epic_performance = {}
        self.hourly_stats = {}
        
        self.logger.info("📊 SessionManager initialized")
    
    def start_session(self):
        """Start a new scanning session"""
        if self.session_active:
            self.logger.warning("⚠️ Session already active, stopping current session first")
            self.stop_session()
        
        self.session_active = True
        self.start_time = datetime.now()
        
        if self.timezone_manager:
            local_time = self.timezone_manager.get_current_local_time()
            self.logger.info(f"🚀 Session started at {self.timezone_manager.format_for_display(local_time)}")
        else:
            self.logger.info(f"🚀 Session started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Reset session statistics
        self._reset_session_stats()
    
    def stop_session(self):
        """Stop the current scanning session"""
        if not self.session_active:
            self.logger.warning("⚠️ No active session to stop")
            return
        
        self.session_active = False
        self.end_time = datetime.now()
        
        # Log session summary
        self._log_session_summary()
        
        self.logger.info("🛑 Session stopped")
    
    def update_scan_stats(self, signals: List[Dict]):
        """Update statistics after a scan"""
        if not self.session_active:
            return
        
        self.scan_count += 1
        self.signal_count += len(signals)
        
        # Track scan performance
        scan_start = datetime.now()
        
        # Process signal statistics
        for signal in signals:
            self._update_signal_stats(signal)
        
        # Track scan duration (estimated)
        scan_duration = (datetime.now() - scan_start).total_seconds()
        self.scan_durations.append(scan_duration)
        
        # Update hourly statistics
        self._update_hourly_stats(len(signals))
        
        self.logger.debug(f"📊 Scan #{self.scan_count}: {len(signals)} signals")
    
    def _update_signal_stats(self, signal: Dict):
        """Update statistics for individual signal"""
        epic = signal.get('epic', 'Unknown')
        
        # Track epic performance
        if epic not in self.epic_performance:
            self.epic_performance[epic] = {
                'signal_count': 0,
                'avg_confidence': 0.0,
                'approved_count': 0,
                'claude_analyzed_count': 0
            }
        
        epic_stats = self.epic_performance[epic]
        epic_stats['signal_count'] += 1
        
        # Update confidence average
        confidence = signal.get('confidence_score', 0)
        current_avg = epic_stats['avg_confidence']
        count = epic_stats['signal_count']
        epic_stats['avg_confidence'] = ((current_avg * (count - 1)) + confidence) / count
        
        # Track processing status
        if signal.get('trade_approved'):
            epic_stats['approved_count'] += 1
            self.approved_signals += 1
        
        if signal.get('claude_quality_score') is not None:
            epic_stats['claude_analyzed_count'] += 1
            self.claude_analyzed_signals += 1
        
        if signal.get('notification_sent'):
            self.notifications_sent += 1
        
        self.processed_signals += 1
    
    def _update_hourly_stats(self, signal_count: int):
        """Update hourly statistics"""
        current_hour = datetime.now().strftime('%Y-%m-%d %H:00')
        
        if current_hour not in self.hourly_stats:
            self.hourly_stats[current_hour] = {
                'scans': 0,
                'signals': 0,
                'start_time': datetime.now()
            }
        
        self.hourly_stats[current_hour]['scans'] += 1
        self.hourly_stats[current_hour]['signals'] += signal_count
    
    def _reset_session_stats(self):
        """Reset session statistics"""
        self.scan_count = 0
        self.signal_count = 0
        self.processed_signals = 0
        self.approved_signals = 0
        self.claude_analyzed_signals = 0
        self.notifications_sent = 0
        self.scan_durations.clear()
        self.epic_performance.clear()
        self.hourly_stats.clear()
    
    def _log_session_summary(self):
        """Log comprehensive session summary"""
        if not self.start_time:
            return
        
        duration = self.end_time - self.start_time if self.end_time else datetime.now() - self.start_time
        
        # Get alert statistics from database if available
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                session_alerts = self.db_manager.execute_query(f"""
                    SELECT COUNT(*) as count 
                    FROM alert_history 
                    WHERE alert_timestamp >= '{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}'
                """)
                alert_count = session_alerts.iloc[0]['count'] if len(session_alerts) > 0 else "Unknown"
            else:
                alert_count = "Unknown"
        except:
            alert_count = "Unknown"
        
        self.logger.info("📊 Scanning Session Summary:")
        
        # Time information
        if self.timezone_manager:
            start_display = self.timezone_manager.format_for_display(self.start_time)
            end_display = self.timezone_manager.format_for_display(self.end_time or datetime.now())
            self.logger.info(f"   ⏰ Started: {start_display}")
            self.logger.info(f"   ⏰ Ended: {end_display}")
        else:
            self.logger.info(f"   ⏰ Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"   ⏰ Ended: {(self.end_time or datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.logger.info(f"   ⏱️  Duration: {duration}")
        
        # Scanning statistics
        self.logger.info(f"   🔍 Total scans: {self.scan_count}")
        self.logger.info(f"   🎯 Total signals: {self.signal_count}")
        self.logger.info(f"   📊 Processed signals: {self.processed_signals}")
        self.logger.info(f"   ✅ Approved signals: {self.approved_signals}")
        self.logger.info(f"   💾 Alerts saved to DB: {alert_count}")
        
        # Claude and notification statistics
        if self.claude_analyzed_signals > 0:
            self.logger.info(f"   🤖 Claude analyzed: {self.claude_analyzed_signals}")
        if self.notifications_sent > 0:
            self.logger.info(f"   📢 Notifications sent: {self.notifications_sent}")
        
        # Performance statistics
        if self.scan_durations:
            avg_scan_time = sum(self.scan_durations) / len(self.scan_durations)
            self.logger.info(f"   ⚡ Avg scan time: {avg_scan_time:.2f}s")
        
        # Epic performance summary
        if self.epic_performance:
            top_performers = sorted(
                self.epic_performance.items(),
                key=lambda x: x[1]['signal_count'],
                reverse=True
            )[:3]  # Top 3 performers
            
            self.logger.info(f"   🏆 Top performing epics:")
            for epic, stats in top_performers:
                self.logger.info(f"      {epic}: {stats['signal_count']} signals, {stats['avg_confidence']:.1%} avg confidence")
    
    def get_session_status(self) -> Dict:
        """Get current session status and statistics"""
        current_time = datetime.now()
        
        status = {
            'session_active': self.session_active,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'current_time': current_time.isoformat(),
            'statistics': {
                'scan_count': self.scan_count,
                'signal_count': self.signal_count,
                'processed_signals': self.processed_signals,
                'approved_signals': self.approved_signals,
                'claude_analyzed_signals': self.claude_analyzed_signals,
                'notifications_sent': self.notifications_sent
            },
            'performance': {
                'epic_performance': dict(self.epic_performance),
                'hourly_stats': dict(self.hourly_stats)
            }
        }
        
        # Add duration if session is active
        if self.session_active and self.start_time:
            duration = current_time - self.start_time
            status['session_duration_seconds'] = duration.total_seconds()
            status['session_duration_display'] = str(duration)
        
        # Add scan performance metrics
        if self.scan_durations:
            status['scan_performance'] = {
                'total_scans': len(self.scan_durations),
                'avg_scan_duration': sum(self.scan_durations) / len(self.scan_durations),
                'min_scan_duration': min(self.scan_durations),
                'max_scan_duration': max(self.scan_durations)
            }
        
        return status
    
    def get_epic_statistics(self) -> Dict:
        """Get detailed epic performance statistics"""
        if not self.epic_performance:
            return {'message': 'No epic statistics available'}
        
        # Calculate additional metrics
        epic_stats = {}
        total_signals = sum(stats['signal_count'] for stats in self.epic_performance.values())
        
        for epic, stats in self.epic_performance.items():
            signal_count = stats['signal_count']
            
            epic_stats[epic] = {
                'signal_count': signal_count,
                'signal_percentage': (signal_count / total_signals * 100) if total_signals > 0 else 0,
                'avg_confidence': stats['avg_confidence'],
                'approved_count': stats['approved_count'],
                'approval_rate': (stats['approved_count'] / signal_count * 100) if signal_count > 0 else 0,
                'claude_analyzed_count': stats['claude_analyzed_count'],
                'claude_analysis_rate': (stats['claude_analyzed_count'] / signal_count * 100) if signal_count > 0 else 0
            }
        
        # Sort by signal count
        sorted_epics = sorted(epic_stats.items(), key=lambda x: x[1]['signal_count'], reverse=True)
        
        return {
            'total_epics': len(epic_stats),
            'total_signals': total_signals,
            'epic_details': dict(sorted_epics),
            'summary': {
                'most_active_epic': sorted_epics[0][0] if sorted_epics else None,
                'highest_confidence_epic': max(epic_stats.items(), key=lambda x: x[1]['avg_confidence'])[0] if epic_stats else None,
                'highest_approval_rate_epic': max(epic_stats.items(), key=lambda x: x[1]['approval_rate'])[0] if epic_stats else None
            }
        }
    
    def get_hourly_performance(self) -> Dict:
        """Get hourly performance breakdown"""
        if not self.hourly_stats:
            return {'message': 'No hourly statistics available'}
        
        # Sort by hour
        sorted_hours = sorted(self.hourly_stats.items())
        
        hourly_data = []
        for hour_str, stats in sorted_hours:
            hourly_data.append({
                'hour': hour_str,
                'scans': stats['scans'],
                'signals': stats['signals'],
                'signals_per_scan': stats['signals'] / stats['scans'] if stats['scans'] > 0 else 0,
                'start_time': stats['start_time'].isoformat()
            })
        
        # Calculate trends
        total_hours = len(hourly_data)
        total_scans = sum(hour['scans'] for hour in hourly_data)
        total_signals = sum(hour['signals'] for hour in hourly_data)
        
        return {
            'total_hours': total_hours,
            'total_scans': total_scans,
            'total_signals': total_signals,
            'avg_scans_per_hour': total_scans / total_hours if total_hours > 0 else 0,
            'avg_signals_per_hour': total_signals / total_hours if total_hours > 0 else 0,
            'hourly_breakdown': hourly_data,
            'peak_hour': max(hourly_data, key=lambda x: x['signals'])['hour'] if hourly_data else None
        }
    
    def export_session_data(self) -> Dict:
        """Export comprehensive session data for analysis"""
        return {
            'session_info': {
                'session_active': self.session_active,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'export_time': datetime.now().isoformat()
            },
            'statistics': self.get_session_status()['statistics'],
            'epic_performance': self.get_epic_statistics(),
            'hourly_performance': self.get_hourly_performance(),
            'scan_durations': list(self.scan_durations)
        }
    
    def reset_statistics(self):
        """Reset all statistics (useful for testing)"""
        self._reset_session_stats()
        self.logger.info("📊 Session statistics reset")
    
    def add_manual_stat(self, stat_type: str, value: int = 1):
        """Manually add to statistics (useful for external updates)"""
        if stat_type == 'scan_count':
            self.scan_count += value
        elif stat_type == 'signal_count':
            self.signal_count += value
        elif stat_type == 'processed_signals':
            self.processed_signals += value
        elif stat_type == 'approved_signals':
            self.approved_signals += value
        elif stat_type == 'claude_analyzed_signals':
            self.claude_analyzed_signals += value
        elif stat_type == 'notifications_sent':
            self.notifications_sent += value
        else:
            self.logger.warning(f"⚠️ Unknown stat type: {stat_type}")
            return False
        
        self.logger.debug(f"📊 Manual stat update: {stat_type} += {value}")
        return True
    
    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary"""
        if not self.session_active and not self.start_time:
            return "No session data available"
        
        duration = (datetime.now() - self.start_time) if self.session_active else (self.end_time - self.start_time)
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        
        # Calculate rates
        scans_per_hour = (self.scan_count / duration.total_seconds() * 3600) if duration.total_seconds() > 0 else 0
        signals_per_hour = (self.signal_count / duration.total_seconds() * 3600) if duration.total_seconds() > 0 else 0
        
        # Approval rate
        approval_rate = (self.approved_signals / self.signal_count * 100) if self.signal_count > 0 else 0
        
        # Average scan time
        avg_scan_time = sum(self.scan_durations) / len(self.scan_durations) if self.scan_durations else 0
        
        summary = f"""
📊 Session Performance Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️  Duration: {duration_str}
🔍 Scans: {self.scan_count} ({scans_per_hour:.1f}/hour)
🎯 Signals: {self.signal_count} ({signals_per_hour:.1f}/hour)
✅ Approved: {self.approved_signals} ({approval_rate:.1f}%)
⚡ Avg Scan Time: {avg_scan_time:.2f}s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Add top performing epic
        if self.epic_performance:
            top_epic = max(self.epic_performance.items(), key=lambda x: x[1]['signal_count'])
            summary += f"🏆 Top Epic: {top_epic[0]} ({top_epic[1]['signal_count']} signals)\n"
        
        return summary.strip()
    
    def update_configuration(self, **kwargs):
        """Update session manager configuration"""
        updated_items = []
        
        if 'timezone_manager' in kwargs:
            self.timezone_manager = kwargs['timezone_manager']
            updated_items.append("Timezone manager updated")
        
        if updated_items:
            self.logger.info(f"📝 SessionManager configuration updated:")
            for item in updated_items:
                self.logger.info(f"   {item}")
        
        return updated_items
    
    def validate_session_state(self) -> tuple[bool, list[str]]:
        """Validate current session state"""
        issues = []
        
        if self.session_active and not self.start_time:
            issues.append("Session marked as active but no start time set")
        
        if not self.session_active and self.start_time and not self.end_time:
            issues.append("Session marked as inactive but no end time set")
        
        if self.scan_count < 0:
            issues.append(f"Invalid scan count: {self.scan_count}")
        
        if self.signal_count < 0:
            issues.append(f"Invalid signal count: {self.signal_count}")
        
        if self.approved_signals > self.signal_count:
            issues.append(f"Approved signals ({self.approved_signals}) exceeds total signals ({self.signal_count})")
        
        return len(issues) == 0, issues