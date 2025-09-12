# core/trading/session_manager.py
"""
Session Manager - Extracted from TradingOrchestrator
Handles session lifecycle, performance tracking, and statistics management
"""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class SessionManager:
    """
    Manages trading session lifecycle, performance tracking, and statistics
    Extracted from TradingOrchestrator to provide focused session management
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 scan_interval: int = 60):
        
        self.logger = logger or logging.getLogger(__name__)
        self.scan_interval = scan_interval
        
        # Session state
        self.session_active = False
        self.session_start_time = None
        self.session_end_time = None
        self.scan_count = 0
        
        # Performance statistics
        self.performance_stats = {
            'session_start': None,
            'session_end': None,
            'total_scans': 0,
            'total_signals': 0,
            'total_trades': 0,
            'total_scan_time': 0.0,
            'average_scan_time': 0.0,
            'fastest_scan': float('inf'),
            'slowest_scan': 0.0,
            'signals_per_hour': 0.0,
            'scans_per_hour': 0.0
        }
        
        # Daily tracking
        self.daily_signals = []
        self.scan_times = []
        
        # Session metadata
        self.session_metadata = {
            'session_id': None,
            'start_timestamp': None,
            'configuration': {},
            'system_info': {}
        }
        
        self.logger.info("📊 SessionManager initialized")
        self.logger.info(f"   Scan interval: {self.scan_interval}s")
    
    def start_session(self, session_config: Dict = None) -> str:
        """
        Start a new trading session
        
        Args:
            session_config: Optional session configuration
            
        Returns:
            Session ID
        """
        if self.session_active:
            self.logger.warning("⚠️ Session already active - stopping current session first")
            self.stop_session()
        
        # Generate session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize session
        self.session_active = True
        self.session_start_time = datetime.now()
        self.scan_count = 0
        
        # Reset performance stats for new session
        self.performance_stats = {
            'session_start': self.session_start_time,
            'session_end': None,
            'total_scans': 0,
            'total_signals': 0,
            'total_trades': 0,
            'total_scan_time': 0.0,
            'average_scan_time': 0.0,
            'fastest_scan': float('inf'),
            'slowest_scan': 0.0,
            'signals_per_hour': 0.0,
            'scans_per_hour': 0.0
        }
        
        # Reset daily tracking
        self.daily_signals = []
        self.scan_times = []
        
        # Store session metadata
        self.session_metadata = {
            'session_id': session_id,
            'start_timestamp': self.session_start_time.isoformat(),
            'configuration': session_config or {},
            'system_info': self._get_system_info()
        }
        
        self.logger.info(f"🚀 Trading session started: {session_id}")
        self.logger.info(f"   Start time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"   Scan interval: {self.scan_interval}s")
        
        return session_id
    
    def stop_session(self) -> Dict:
        """
        Stop the current trading session
        
        Returns:
            Session summary statistics
        """
        if not self.session_active:
            self.logger.warning("⚠️ No active session to stop")
            return {}
        
        self.session_end_time = datetime.now()
        self.session_active = False
        
        # Calculate final statistics
        session_duration = self.session_end_time - self.session_start_time
        self.performance_stats['session_end'] = self.session_end_time
        
        # Update session metadata
        self.session_metadata['end_timestamp'] = self.session_end_time.isoformat()
        self.session_metadata['duration_seconds'] = session_duration.total_seconds()
        
        # Log session summary
        self._log_session_summary(session_duration)
        
        session_summary = self.get_session_summary()
        
        self.logger.info(f"🏁 Trading session ended: {self.session_metadata['session_id']}")
        self.logger.info(f"   Duration: {self._format_duration(session_duration)}")
        self.logger.info(f"   Total scans: {self.performance_stats['total_scans']}")
        self.logger.info(f"   Total signals: {self.performance_stats['total_signals']}")
        self.logger.info(f"   Total trades: {self.performance_stats['total_trades']}")
        
        return session_summary
    
    def update_scan_stats(self, signals: List[Dict], scan_duration: float = None, trades_executed: int = 0):
        """
        Update performance statistics after a scan
        
        Args:
            signals: List of signals detected in the scan
            scan_duration: Time taken for the scan in seconds
            trades_executed: Number of trades executed
        """
        if not self.session_active:
            self.logger.warning("⚠️ No active session - stats not updated")
            return
        
        # Update scan count
        self.scan_count += 1
        self.performance_stats['total_scans'] += 1
        
        # Update signal count
        signal_count = len(signals) if signals else 0
        self.performance_stats['total_signals'] += signal_count
        
        # Update trade count
        self.performance_stats['total_trades'] += trades_executed
        
        # Update scan timing statistics
        if scan_duration is not None:
            self.scan_times.append(scan_duration)
            self.performance_stats['total_scan_time'] += scan_duration
            self.performance_stats['average_scan_time'] = (
                self.performance_stats['total_scan_time'] / self.performance_stats['total_scans']
            )
            
            # Track fastest and slowest scans
            if scan_duration < self.performance_stats['fastest_scan']:
                self.performance_stats['fastest_scan'] = scan_duration
            if scan_duration > self.performance_stats['slowest_scan']:
                self.performance_stats['slowest_scan'] = scan_duration
        
        # Add signals to daily tracking
        if signals:
            self.daily_signals.extend(signals)
            
        # Clean old signals (keep only today's)
        self._clean_daily_signals()
        
        # Calculate rates (signals/hour, scans/hour)
        self._calculate_rates()
        
        self.logger.debug(f"📊 Stats updated: Scan #{self.scan_count}, "
                         f"{signal_count} signals, {scan_duration:.1f}s" if scan_duration else "")
    
    def log_scan_summary(self, raw_signals: int, processed_signals: int, scan_duration: float, epic_count: int = None):
        """
        Log a comprehensive scan summary
        
        Args:
            raw_signals: Number of raw signals detected
            processed_signals: Number of signals after processing
            scan_duration: Time taken for the scan
            epic_count: Number of epics scanned
        """
        self.logger.info(f"📊 Sequential scan #{self.scan_count} complete in {scan_duration:.1f}s")
        
        if epic_count:
            self.logger.info(f"📊 Scanned {epic_count} epics")
            
        self.logger.info(f"📊 Raw signals: {raw_signals}")
        self.logger.info(f"✅ Processed signals: {processed_signals}")
        self.logger.info(f"📈 Daily total: {len(self.daily_signals)}")
        
        if processed_signals > 0:
            self.logger.info(f"🚨 Found {processed_signals} qualified signals!")
            
        # Log performance metrics if we have enough data
        if self.scan_count > 1:
            avg_time = self.performance_stats.get('average_scan_time', 0)
            self.logger.debug(f"📊 Avg scan time: {avg_time:.1f}s, "
                            f"Signals/hour: {self.performance_stats.get('signals_per_hour', 0):.1f}")
    
    def get_session_status(self) -> Dict:
        """
        Get current session status
        
        Returns:
            Dictionary containing session status and statistics
        """
        if not self.session_active:
            return {
                'session_active': False,
                'message': 'No active session'
            }
        
        current_time = datetime.now()
        session_duration = current_time - self.session_start_time
        
        return {
            'session_active': True,
            'session_id': self.session_metadata['session_id'],
            'start_time': self.session_start_time.isoformat(),
            'current_time': current_time.isoformat(),
            'duration_seconds': session_duration.total_seconds(),
            'duration_formatted': self._format_duration(session_duration),
            'scan_count': self.scan_count,
            'scan_interval': self.scan_interval,
            'daily_signals': len(self.daily_signals),
            'performance_stats': self.performance_stats.copy(),
            'last_scan_time': self.scan_times[-1] if self.scan_times else None
        }
    
    def get_session_summary(self) -> Dict:
        """
        Get comprehensive session summary
        
        Returns:
            Complete session summary with all statistics
        """
        status = self.get_session_status()
        
        # Add additional summary data
        summary = {
            'session_metadata': self.session_metadata.copy(),
            'session_status': status,
            'performance_summary': {
                'total_duration': status.get('duration_seconds', 0),
                'scans_completed': self.performance_stats['total_scans'],
                'signals_detected': self.performance_stats['total_signals'],
                'trades_executed': self.performance_stats['total_trades'],
                'average_scan_time': self.performance_stats['average_scan_time'],
                'fastest_scan': self.performance_stats['fastest_scan'] if self.performance_stats['fastest_scan'] != float('inf') else 0,
                'slowest_scan': self.performance_stats['slowest_scan'],
                'signals_per_hour': self.performance_stats['signals_per_hour'],
                'scans_per_hour': self.performance_stats['scans_per_hour']
            },
            'daily_signal_count': len(self.daily_signals),
            'scan_time_distribution': self._get_scan_time_distribution()
        }
        
        return summary
    
    def reset_session_stats(self):
        """Reset session statistics (useful for testing)"""
        self.scan_count = 0
        self.daily_signals = []
        self.scan_times = []
        
        # Reset performance stats but keep session times
        session_start = self.performance_stats.get('session_start')
        self.performance_stats = {
            'session_start': session_start,
            'session_end': None,
            'total_scans': 0,
            'total_signals': 0,
            'total_trades': 0,
            'total_scan_time': 0.0,
            'average_scan_time': 0.0,
            'fastest_scan': float('inf'),
            'slowest_scan': 0.0,
            'signals_per_hour': 0.0,
            'scans_per_hour': 0.0
        }
        
        self.logger.info("📊 Session statistics reset")
    
    def _clean_daily_signals(self):
        """Remove signals older than today"""
        today = datetime.now().date()
        self.daily_signals = [
            s for s in self.daily_signals 
            if isinstance(s.get('timestamp'), datetime) and s['timestamp'].date() == today
        ]
    
    def _calculate_rates(self):
        """Calculate signals per hour and scans per hour"""
        if not self.session_active or not self.session_start_time:
            return
            
        session_duration_hours = (datetime.now() - self.session_start_time).total_seconds() / 3600
        
        if session_duration_hours > 0:
            self.performance_stats['signals_per_hour'] = (
                self.performance_stats['total_signals'] / session_duration_hours
            )
            self.performance_stats['scans_per_hour'] = (
                self.performance_stats['total_scans'] / session_duration_hours
            )
    
    def _get_scan_time_distribution(self) -> Dict:
        """Get scan time distribution statistics"""
        if not self.scan_times:
            return {}
            
        scan_times = self.scan_times
        scan_times.sort()
        
        return {
            'count': len(scan_times),
            'mean': sum(scan_times) / len(scan_times),
            'median': scan_times[len(scan_times) // 2],
            'min': min(scan_times),
            'max': max(scan_times),
            'p95': scan_times[int(0.95 * len(scan_times))] if len(scan_times) > 20 else max(scan_times)
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information for session metadata"""
        import platform
        import psutil
        
        try:
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get system info: {e}")
            return {'error': str(e)}
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format duration as human-readable string"""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _log_session_summary(self, session_duration: timedelta):
        """Log detailed session summary"""
        self.logger.info("📊 Session Summary:")
        self.logger.info(f"   Duration: {self._format_duration(session_duration)}")
        self.logger.info(f"   Total scans: {self.performance_stats['total_scans']}")
        self.logger.info(f"   Total signals: {self.performance_stats['total_signals']}")
        self.logger.info(f"   Total trades: {self.performance_stats['total_trades']}")
        
        if self.performance_stats['total_scans'] > 0:
            self.logger.info(f"   Average scan time: {self.performance_stats['average_scan_time']:.1f}s")
            self.logger.info(f"   Signals per hour: {self.performance_stats['signals_per_hour']:.1f}")
            self.logger.info(f"   Scans per hour: {self.performance_stats['scans_per_hour']:.1f}")
            
        if self.performance_stats['fastest_scan'] != float('inf'):
            self.logger.info(f"   Fastest scan: {self.performance_stats['fastest_scan']:.1f}s")
            self.logger.info(f"   Slowest scan: {self.performance_stats['slowest_scan']:.1f}s")
    
    def is_session_active(self) -> bool:
        """Check if session is currently active"""
        return self.session_active
    
    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID"""
        return self.session_metadata.get('session_id') if self.session_active else None
    
    def update_scan_interval(self, new_interval: int):
        """Update the scan interval"""
        old_interval = self.scan_interval
        self.scan_interval = new_interval
        self.logger.info(f"📝 Scan interval updated: {old_interval}s → {new_interval}s")
    


    def update_scan_statistics(self, scan_data: Dict):
        """
        Update session statistics after a scan (COMPATIBILITY METHOD)
        
        Args:
            scan_data: Dictionary containing scan results and metadata
        """
        try:
            if not self.session_active:
                self.logger.warning("⚠️ No active session - scan statistics not updated")
                return
            
            # Extract scan information
            scan_number = scan_data.get('scan_number', 0)
            signals_found = scan_data.get('signals_found', 0)
            scan_duration = scan_data.get('scan_duration', 0)
            timestamp = scan_data.get('timestamp', datetime.now())
            
            # Update scan count
            self.scan_count = max(self.scan_count, scan_number)
            self.performance_stats['total_scans'] = self.scan_count
            
            # Update signal count
            self.performance_stats['total_signals'] += signals_found
            
            # Update scan timing if provided
            if scan_duration > 0:
                self.scan_times.append(scan_duration)
                
                # Keep only recent scan times (last 100)
                if len(self.scan_times) > 100:
                    self.scan_times = self.scan_times[-100:]
                
                # Update timing statistics
                total_scan_time = sum(self.scan_times)
                self.performance_stats['total_scan_time'] = total_scan_time
                self.performance_stats['average_scan_time'] = total_scan_time / len(self.scan_times)
                self.performance_stats['fastest_scan'] = min(self.scan_times)
                self.performance_stats['slowest_scan'] = max(self.scan_times)
            
            # Calculate rates
            current_time = datetime.now()
            session_duration = (current_time - self.session_start_time).total_seconds()
            
            if session_duration > 0:
                self.performance_stats['scans_per_hour'] = (self.scan_count / session_duration) * 3600
                self.performance_stats['signals_per_hour'] = (self.performance_stats['total_signals'] / session_duration) * 3600
            
            # Track daily signals
            if signals_found > 0:
                signal_entry = {
                    'timestamp': timestamp,
                    'count': signals_found,
                    'scan_number': scan_number
                }
                self.daily_signals.append(signal_entry)
            
            # Log progress periodically
            if self.scan_count % 10 == 0:
                self.logger.info(f"📊 Session progress: {self.scan_count} scans, {self.performance_stats['total_signals']} total signals")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating scan statistics: {e}")

    def update_scan_interval(self, new_interval: int):
        """
        Update scan interval (COMPATIBILITY METHOD)
        
        Args:
            new_interval: New scan interval in seconds
        """
        try:
            old_interval = self.scan_interval
            self.scan_interval = new_interval
            
            self.logger.info(f"🔧 Scan interval updated: {old_interval}s → {new_interval}s")
            
            # Update session metadata
            if 'configuration' not in self.session_metadata:
                self.session_metadata['configuration'] = {}
            
            self.session_metadata['configuration']['scan_interval'] = new_interval
            self.session_metadata['configuration']['updated_at'] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating scan interval: {e}")

    def reset_session_stats(self):
        """
        Reset session statistics (COMPATIBILITY METHOD)
        """
        try:
            # Reset performance statistics
            self.performance_stats = {
                'session_start': self.session_start_time,
                'session_end': None,
                'total_scans': 0,
                'total_signals': 0,
                'total_trades': 0,
                'total_scan_time': 0.0,
                'average_scan_time': 0.0,
                'fastest_scan': float('inf'),
                'slowest_scan': 0.0,
                'signals_per_hour': 0.0,
                'scans_per_hour': 0.0
            }
            
            # Reset counters
            self.scan_count = 0
            
            # Clear tracking lists
            self.daily_signals = []
            self.scan_times = []
            
            self.logger.info("🔄 Session statistics reset")
            
        except Exception as e:
            self.logger.error(f"❌ Error resetting session stats: {e}")