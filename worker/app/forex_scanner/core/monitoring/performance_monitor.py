# core/monitoring/performance_monitor.py
"""
Real-time Performance Monitoring System
Continuous monitoring and alerting for trading system performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime, timedelta
import json
import threading
import time
from collections import deque, defaultdict
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.errors

try:
    import config
except ImportError:
    from forex_scanner import config


class PerformanceMonitor:
    """Real-time performance monitoring and alerting system"""
    
    def __init__(self, db_manager, alert_callback: Optional[Callable] = None):
        self.db_manager = db_manager
        self.alert_callback = alert_callback
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.signal_history = deque(maxlen=1000)  # Last 1000 signals
        self.performance_metrics = {}
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 60  # Check every minute
        
        # Performance caching
        self.metric_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Initialize monitoring tables in PostgreSQL
        self._initialize_monitoring_tables()
    
    def _initialize_alert_thresholds(self) -> Dict:
        """Initialize performance alert thresholds"""
        return {
            'win_rate': {
                'warning': 0.45,    # Below 45% win rate
                'critical': 0.35    # Below 35% win rate
            },
            'drawdown': {
                'warning': 0.10,    # 10% drawdown
                'critical': 0.20    # 20% drawdown
            },
            'signal_frequency': {
                'warning': 2,       # Less than 2 signals per day
                'critical': 0.5     # Less than 0.5 signals per day
            },
            'confidence_accuracy': {
                'warning': 0.70,    # Less than 70% accuracy
                'critical': 0.60    # Less than 60% accuracy
            },
            'avg_profit_decline': {
                'warning': 0.20,    # 20% decline in average profit
                'critical': 0.40    # 40% decline in average profit
            }
        }
    
    
    def _initialize_monitoring_tables(self):
        """Initialize PostgreSQL tables for monitoring data"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Performance snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    win_rate DECIMAL(5,4),
                    total_signals INTEGER,
                    avg_profit DECIMAL(10,4),
                    avg_loss DECIMAL(10,4),
                    max_drawdown DECIMAL(5,4),
                    profit_factor DECIMAL(10,4),
                    sharpe_ratio DECIMAL(10,4),
                    signal_frequency DECIMAL(10,4),
                    confidence_accuracy DECIMAL(5,4),
                    active_strategy VARCHAR(50),
                    market_regime VARCHAR(50),
                    net_profit DECIMAL(15,4),
                    total_profit DECIMAL(15,4),
                    total_loss DECIMAL(15,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    alert_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    metric VARCHAR(50) NOT NULL,
                    current_value DECIMAL(15,6),
                    threshold_value DECIMAL(15,6),
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Signal tracking table for detailed analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_tracking (
                    id SERIAL PRIMARY KEY,
                    signal_timestamp TIMESTAMP NOT NULL,
                    epic VARCHAR(50) NOT NULL,
                    signal_type VARCHAR(10) NOT NULL,
                    confidence DECIMAL(5,4),
                    strategy VARCHAR(50),
                    entry_price DECIMAL(10,5),
                    exit_price DECIMAL(10,5),
                    outcome VARCHAR(10), -- 'win', 'loss', 'pending'
                    profit_loss_pips DECIMAL(10,2),
                    duration_minutes INTEGER,
                    stop_loss DECIMAL(10,5),
                    take_profit DECIMAL(10,5),
                    volume_confirmation BOOLEAN,
                    ema_separation_pips DECIMAL(10,2),
                    market_regime VARCHAR(50),
                    session VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(50) NOT NULL,
                    date_tracked DATE NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    winning_signals INTEGER DEFAULT 0,
                    losing_signals INTEGER DEFAULT 0,
                    win_rate DECIMAL(5,4),
                    avg_profit_pips DECIMAL(10,2),
                    avg_loss_pips DECIMAL(10,2),
                    profit_factor DECIMAL(10,4),
                    total_profit_pips DECIMAL(15,2),
                    max_consecutive_wins INTEGER DEFAULT 0,
                    max_consecutive_losses INTEGER DEFAULT 0,
                    largest_win_pips DECIMAL(10,2),
                    largest_loss_pips DECIMAL(10,2),
                    avg_confidence DECIMAL(5,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy_name, date_tracked)
                )
            ''')
            
            # Epic performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS epic_performance (
                    id SERIAL PRIMARY KEY,
                    epic VARCHAR(50) NOT NULL,
                    date_tracked DATE NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    winning_signals INTEGER DEFAULT 0,
                    win_rate DECIMAL(5,4),
                    avg_profit_pips DECIMAL(10,2),
                    avg_loss_pips DECIMAL(10,2),
                    total_profit_pips DECIMAL(15,2),
                    volatility_score DECIMAL(5,4),
                    avg_confidence DECIMAL(5,4),
                    best_session VARCHAR(20),
                    worst_session VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(epic, date_tracked)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_snapshots_timestamp ON performance_snapshots(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_alerts_timestamp ON performance_alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_alerts_resolved ON performance_alerts(resolved)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_tracking_timestamp ON signal_tracking(signal_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_tracking_epic ON signal_tracking(epic)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_tracking_strategy ON signal_tracking(strategy)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_tracking_outcome ON signal_tracking(outcome)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_performance_date ON strategy_performance(date_tracked)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_epic_performance_date ON epic_performance(date_tracked)')
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info("✅ Performance monitoring tables initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring tables: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_active:
            self.logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("🔍 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("⏹️ Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check for alerts
                self._check_performance_alerts()
                
                # Save snapshot
                self._save_performance_snapshot()
                
                # Clean old data
                self._cleanup_old_data()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def record_signal(self, signal: Dict):
        """Record a new signal for performance tracking"""
        timestamp = datetime.now()
        
        signal_record = {
            'timestamp': timestamp.isoformat(),
            'epic': signal.get('epic'),
            'signal_type': signal.get('signal_type'),
            'confidence': signal.get('confidence_score', 0),
            'strategy': signal.get('strategy'),
            'entry_price': signal.get('price'),
            'outcome': None,  # To be updated later
            'profit_loss': None,
            'exit_price': None,
            'duration': None
        }
        
        # Add to in-memory history
        self.signal_history.append(signal_record)
        
        # Save to PostgreSQL
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Determine session and market regime (simplified for now)
            current_hour = timestamp.hour
            if 8 <= current_hour <= 16:
                session = 'london'
            elif 13 <= current_hour <= 21:
                session = 'new_york' if current_hour >= 13 else 'london'
            elif 23 <= current_hour or current_hour <= 8:
                session = 'asian'
            else:
                session = 'unknown'
            
            cursor.execute('''
                INSERT INTO signal_tracking 
                (signal_timestamp, epic, signal_type, confidence, strategy, entry_price, 
                 outcome, volume_confirmation, ema_separation_pips, market_regime, session)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                timestamp,
                signal.get('epic'),
                signal.get('signal_type'),
                signal.get('confidence_score', 0),
                signal.get('strategy'),
                signal.get('price'),
                'pending',  # Initial outcome
                signal.get('volume_confirmation', False),
                signal.get('ema_separation_pips', 0),
                'unknown',  # TODO: Get from market intelligence
                session
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.debug(f"📝 Recorded signal: {signal['epic']} {signal['signal_type']}")
            
        except Exception as e:
            self.logger.error(f"Error recording signal to database: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def update_signal_outcome(self, signal_id: str, outcome: Dict):
        """Update signal outcome with profit/loss information"""
        # Find and update the signal in history
        for signal in reversed(self.signal_history):
            if (signal['epic'] == outcome.get('epic') and 
                signal['timestamp'] == outcome.get('timestamp')):
                
                signal['outcome'] = outcome.get('outcome')  # 'win' or 'loss'
                signal['profit_loss'] = outcome.get('profit_loss_pips')
                signal['exit_price'] = outcome.get('exit_price')
                signal['duration'] = outcome.get('duration_minutes')
                
                self.logger.debug(f"📊 Updated signal outcome: {outcome}")
                break
        
        # Update in PostgreSQL
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE signal_tracking 
                SET outcome = %s, 
                    profit_loss_pips = %s, 
                    exit_price = %s, 
                    duration_minutes = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE epic = %s 
                AND signal_timestamp = %s
                AND outcome = 'pending'
            ''', (
                outcome.get('outcome'),
                outcome.get('profit_loss_pips'),
                outcome.get('exit_price'),
                outcome.get('duration_minutes'),
                outcome.get('epic'),
                outcome.get('timestamp')
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating signal outcome in database: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def _update_performance_metrics(self):
        """Update current performance metrics"""
        if not self.signal_history:
            return
        
        # Get signals with outcomes
        completed_signals = [s for s in self.signal_history if s['outcome'] is not None]
        
        if not completed_signals:
            return
        
        # Calculate metrics
        total_signals = len(completed_signals)
        winning_signals = len([s for s in completed_signals if s['outcome'] == 'win'])
        losing_signals = total_signals - winning_signals
        
        # Win rate
        win_rate = winning_signals / total_signals if total_signals > 0 else 0
        
        # Profit/Loss metrics
        profits = [s['profit_loss'] for s in completed_signals if s['outcome'] == 'win']
        losses = [abs(s['profit_loss']) for s in completed_signals if s['outcome'] == 'loss']
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Drawdown calculation
        cumulative_pnl = []
        running_total = 0
        for signal in completed_signals:
            running_total += signal['profit_loss'] or 0
            cumulative_pnl.append(running_total)
        
        max_drawdown = 0
        if cumulative_pnl:
            peak = cumulative_pnl[0]
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / abs(peak) if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        # Signal frequency (signals per day)
        if completed_signals:
            first_signal_time = datetime.fromisoformat(completed_signals[0]['timestamp'])
            last_signal_time = datetime.fromisoformat(completed_signals[-1]['timestamp'])
            time_span_days = (last_signal_time - first_signal_time).days + 1
            signal_frequency = total_signals / time_span_days if time_span_days > 0 else 0
        else:
            signal_frequency = 0
        
        # Confidence accuracy
        high_confidence_signals = [s for s in completed_signals if s['confidence'] > 0.8]
        if high_confidence_signals:
            high_conf_wins = len([s for s in high_confidence_signals if s['outcome'] == 'win'])
            confidence_accuracy = high_conf_wins / len(high_confidence_signals)
        else:
            confidence_accuracy = 0
        
        # Sharpe ratio (simplified)
        if cumulative_pnl and len(cumulative_pnl) > 1:
            returns = np.diff(cumulative_pnl)
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Update metrics
        self.performance_metrics = {
            'timestamp': datetime.now().isoformat(),
            'win_rate': win_rate,
            'total_signals': total_signals,
            'winning_signals': winning_signals,
            'losing_signals': losing_signals,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'signal_frequency': signal_frequency,
            'confidence_accuracy': confidence_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss
        }
        
        # Cache metrics
        self.metric_cache = {
            'metrics': self.performance_metrics.copy(),
            'timestamp': datetime.now()
        }
    
    def _check_performance_alerts(self):
        """Check for performance alerts and trigger notifications"""
        if not self.performance_metrics:
            return
        
        alerts = []
        
        # Check each metric against thresholds
        for metric, thresholds in self.alert_thresholds.items():
            current_value = self.performance_metrics.get(metric, 0)
            
            # Skip if we don't have this metric
            if current_value is None:
                continue
            
            # Check critical threshold
            if ('critical' in thresholds and 
                self._check_threshold(metric, current_value, thresholds['critical'], 'critical')):
                alerts.append({
                    'severity': 'CRITICAL',
                    'metric': metric,
                    'current_value': current_value,
                    'threshold': thresholds['critical'],
                    'message': f"CRITICAL: {metric} is {current_value:.3f}, below critical threshold of {thresholds['critical']}"
                })
            
            # Check warning threshold
            elif ('warning' in thresholds and 
                  self._check_threshold(metric, current_value, thresholds['warning'], 'warning')):
                alerts.append({
                    'severity': 'WARNING',
                    'metric': metric,
                    'current_value': current_value,
                    'threshold': thresholds['warning'],
                    'message': f"WARNING: {metric} is {current_value:.3f}, below warning threshold of {thresholds['warning']}"
                })
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
    
    def _check_threshold(self, metric: str, current_value: float, threshold: float, severity: str) -> bool:
        """Check if a metric violates a threshold"""
        # Different metrics have different threshold logic
        if metric in ['win_rate', 'signal_frequency', 'confidence_accuracy']:
            return current_value < threshold  # Below threshold is bad
        elif metric in ['drawdown', 'avg_profit_decline']:
            return current_value > threshold  # Above threshold is bad
        
        return False
    
    def _process_alert(self, alert: Dict):
        """Process and record an alert"""
        # Check if this alert was already triggered recently
        if self._is_duplicate_alert(alert):
            return
        
        # Save alert to database
        self._save_alert(alert)
        
        # Log alert
        self.logger.warning(f"🚨 PERFORMANCE ALERT: {alert['message']}")
        
        # Trigger callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _is_duplicate_alert(self, alert: Dict) -> bool:
        """Check if this alert was already triggered recently"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Check for similar alerts in the last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            cursor.execute('''
                SELECT COUNT(*) FROM performance_alerts 
                WHERE metric = %s AND severity = %s AND timestamp > %s AND resolved = FALSE
            ''', (alert['metric'], alert['severity'], one_hour_ago))
            
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return count > 0
            
        except Exception as e:
            self.logger.error(f"Error checking duplicate alerts: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return False
    
    def _save_alert(self, alert: Dict):
        """Save alert to database"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_alerts 
                (timestamp, alert_type, severity, metric, current_value, threshold_value, message)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                datetime.now(),
                'performance',
                alert['severity'],
                alert['metric'],
                alert['current_value'],
                alert['threshold'],
                alert['message']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving alert: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def _save_performance_snapshot(self):
        """Save current performance snapshot to database"""
        if not self.performance_metrics:
            return
        
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            metrics = self.performance_metrics
            
            cursor.execute('''
                INSERT INTO performance_snapshots 
                (timestamp, win_rate, total_signals, avg_profit, avg_loss, max_drawdown, 
                 profit_factor, sharpe_ratio, signal_frequency, confidence_accuracy, 
                 active_strategy, market_regime, net_profit, total_profit, total_loss)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                datetime.fromisoformat(metrics['timestamp']),
                metrics['win_rate'],
                metrics['total_signals'],
                metrics['avg_profit'],
                metrics['avg_loss'],
                metrics['max_drawdown'],
                metrics['profit_factor'],
                metrics['sharpe_ratio'],
                metrics['signal_frequency'],
                metrics['confidence_accuracy'],
                getattr(config, 'ACTIVE_STRATEGY', 'combined'),
                'unknown',  # TODO: Get from market intelligence
                metrics['net_profit'],
                metrics['total_profit'],
                metrics['total_loss']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving performance snapshot: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Keep only last 30 days of snapshots
            thirty_days_ago = datetime.now() - timedelta(days=30)
            cursor.execute('DELETE FROM performance_snapshots WHERE timestamp < %s', (thirty_days_ago,))
            
            # Keep only last 7 days of resolved alerts
            seven_days_ago = datetime.now() - timedelta(days=7)
            cursor.execute('DELETE FROM performance_alerts WHERE timestamp < %s AND resolved = TRUE', (seven_days_ago,))
            
            # Keep only last 90 days of signal tracking
            ninety_days_ago = datetime.now() - timedelta(days=90)
            cursor.execute('DELETE FROM signal_tracking WHERE signal_timestamp < %s', (ninety_days_ago,))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get performance summary for the last N hours"""
        if not self.performance_metrics:
            return {'error': 'No performance data available'}
        
        # Use cached metrics if available and recent
        if (self.metric_cache and 
            (datetime.now() - self.metric_cache['timestamp']).seconds < self.cache_timeout):
            metrics = self.metric_cache['metrics']
        else:
            metrics = self.performance_metrics
        
        # Get recent signals
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_signals = [
            s for s in self.signal_history 
            if datetime.fromisoformat(s['timestamp']) > cutoff_time and s['outcome'] is not None
        ]
        
        # Calculate period-specific metrics
        period_metrics = {}
        if recent_signals:
            period_wins = len([s for s in recent_signals if s['outcome'] == 'win'])
            period_metrics = {
                'period_signals': len(recent_signals),
                'period_win_rate': period_wins / len(recent_signals),
                'period_profit': sum(s['profit_loss'] or 0 for s in recent_signals),
                'signals_per_hour': len(recent_signals) / hours
            }
        
        return {
            'overall_metrics': metrics,
            'period_metrics': period_metrics,
            'recent_signals_count': len(recent_signals),
            'monitoring_active': self.monitoring_active,
            'last_updated': metrics.get('timestamp')
        }
    
    def get_performance_trends(self, days: int = 7) -> Dict:
        """Get performance trends over time"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get snapshots from the last N days
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT * FROM performance_snapshots 
                WHERE timestamp > %s 
                ORDER BY timestamp
            ''', (cutoff_date,))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return {'error': 'No historical data available'}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([dict(row) for row in rows])
            
            # Calculate trends
            trends = {}
            for metric in ['win_rate', 'avg_profit', 'signal_frequency', 'confidence_accuracy']:
                if metric in df.columns and len(df) > 1:
                    values = df[metric].dropna()
                    if len(values) > 1:
                        # Simple linear trend
                        x = np.arange(len(values))
                        slope, intercept = np.polyfit(x, values, 1)
                        trends[metric] = {
                            'slope': slope,
                            'direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                            'current_value': float(values.iloc[-1]) if pd.notna(values.iloc[-1]) else 0,
                            'change_rate': slope * len(values)  # Total change over period
                        }
            
            return {
                'trends': trends,
                'data_points': len(df),
                'period_days': days,
                'latest_snapshot': dict(rows[-1]) if rows else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance trends: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
            return {'error': str(e)}
    
    def get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute('''
                SELECT * FROM performance_alerts 
                WHERE resolved = FALSE 
                ORDER BY timestamp DESC
            ''')
            
            alerts = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return []
    
    def resolve_alert(self, alert_id: int):
        """Mark an alert as resolved"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE performance_alerts 
                SET resolved = TRUE, resolved_at = CURRENT_TIMESTAMP 
                WHERE id = %s
            ''', (alert_id,))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"✅ Resolved alert {alert_id}")
            
        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        summary = self.get_performance_summary(24)
        trends = self.get_performance_trends(7)
        active_alerts = self.get_active_alerts()
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(summary, trends, active_alerts)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'trends': trends,
            'active_alerts': active_alerts,
            'recommendations': recommendations,
            'health_score': self._calculate_system_health_score(summary, active_alerts)
        }
    
    def _generate_performance_recommendations(self, summary: Dict, trends: Dict, alerts: List[Dict]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        overall_metrics = summary.get('overall_metrics', {})
        
        # Win rate recommendations
        win_rate = overall_metrics.get('win_rate', 0)
        if win_rate < 0.5:
            recommendations.append("Consider tightening signal confidence thresholds to improve win rate")
        
        # Signal frequency recommendations
        signal_freq = overall_metrics.get('signal_frequency', 0)
        if signal_freq < 1:
            recommendations.append("Signal frequency is low - consider expanding epic list or adjusting strategy parameters")
        elif signal_freq > 10:
            recommendations.append("Very high signal frequency - consider increasing confidence thresholds to filter noise")
        
        # Drawdown recommendations
        drawdown = overall_metrics.get('max_drawdown', 0)
        if drawdown > 0.15:
            recommendations.append("High drawdown detected - consider reducing position sizes or tightening stop losses")
        
        # Trend-based recommendations
        if trends.get('trends'):
            for metric, trend_data in trends['trends'].items():
                if trend_data['direction'] == 'declining':
                    if metric == 'win_rate':
                        recommendations.append("Win rate is declining - review recent market conditions and strategy effectiveness")
                    elif metric == 'avg_profit':
                        recommendations.append("Average profit is declining - consider adjusting profit targets or market timing")
        
        # Alert-based recommendations
        critical_alerts = [a for a in alerts if a['severity'] == 'CRITICAL']
        if critical_alerts:
            recommendations.append("Critical performance alerts active - immediate strategy review recommended")
        
        return recommendations
    
    def _calculate_system_health_score(self, summary: Dict, active_alerts: List[Dict]) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        
        overall_metrics = summary.get('overall_metrics', {})
        
        # Deduct points for poor performance
        win_rate = overall_metrics.get('win_rate', 0.5)
        if win_rate < 0.5:
            score -= (0.5 - win_rate) * 100  # Up to 50 points deduction
        
        # Deduct for high drawdown
        drawdown = overall_metrics.get('max_drawdown', 0)
        score -= drawdown * 200  # Up to 40 points for 20% drawdown
        
        # Deduct for active alerts
        critical_alerts = len([a for a in active_alerts if a['severity'] == 'CRITICAL'])
        warning_alerts = len([a for a in active_alerts if a['severity'] == 'WARNING'])
        
        score -= critical_alerts * 15  # 15 points per critical alert
        score -= warning_alerts * 5   # 5 points per warning alert
        
        # Bonus for good profit factor
        profit_factor = overall_metrics.get('profit_factor', 1)
        if profit_factor > 1.5:
            score += min(10, (profit_factor - 1) * 10)  # Up to 10 bonus points
        
        return max(0, min(100, score))
    
    def export_performance_data(self, days: int = 30) -> Dict:
        """Export performance data for external analysis"""
        try:
            conn = self.db_manager.get_connection()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get snapshots
            snapshots_df = pd.read_sql_query('''
                SELECT * FROM performance_snapshots 
                WHERE timestamp > %s 
                ORDER BY timestamp
            ''', conn, params=(cutoff_date,))
            
            # Get alerts
            alerts_df = pd.read_sql_query('''
                SELECT * FROM performance_alerts 
                WHERE timestamp > %s 
                ORDER BY timestamp
            ''', conn, params=(cutoff_date,))
            
            # Get signal tracking data
            signals_df = pd.read_sql_query('''
                SELECT * FROM signal_tracking 
                WHERE signal_timestamp > %s 
                ORDER BY signal_timestamp
            ''', conn, params=(cutoff_date,))
            
            # Get strategy performance
            strategy_df = pd.read_sql_query('''
                SELECT * FROM strategy_performance 
                WHERE date_tracked > %s 
                ORDER BY date_tracked
            ''', conn, params=(cutoff_date.date(),))
            
            # Get epic performance
            epic_df = pd.read_sql_query('''
                SELECT * FROM epic_performance 
                WHERE date_tracked > %s 
                ORDER BY date_tracked
            ''', conn, params=(cutoff_date.date(),))
            
            conn.close()
            
            return {
                'snapshots': snapshots_df.to_dict('records') if not snapshots_df.empty else [],
                'alerts': alerts_df.to_dict('records') if not alerts_df.empty else [],
                'signals': signals_df.to_dict('records') if not signals_df.empty else [],
                'strategy_performance': strategy_df.to_dict('records') if not strategy_df.empty else [],
                'epic_performance': epic_df.to_dict('records') if not epic_df.empty else [],
                'export_timestamp': datetime.now().isoformat(),
                'period_days': days
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting performance data: {e}")
            if 'conn' in locals():
                conn.close()
            return {'error': str(e)}
    
    def get_signal_analytics(self, hours: int = 24) -> Dict:
        """Get detailed signal analytics from PostgreSQL"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Get recent signals from database
            cursor.execute('''
                SELECT * FROM signal_tracking 
                WHERE signal_timestamp > %s 
                AND outcome IN ('win', 'loss')
                ORDER BY signal_timestamp DESC
            ''', (cutoff_time,))
            
            recent_signals = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            if not recent_signals:
                return {'error': 'No signals in specified period'}
            
            # Strategy breakdown
            strategy_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'total_profit': 0})
            
            # Epic breakdown
            epic_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'total_profit': 0})
            
            # Session breakdown
            session_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'total_profit': 0})
            
            # Confidence level analysis
            confidence_ranges = {
                'high': [],      # >0.8
                'medium': [],    # 0.6-0.8
                'low': []        # <0.6
            }
            
            for signal in recent_signals:
                strategy = signal.get('strategy', 'unknown')
                epic = signal.get('epic', 'unknown')
                session = signal.get('session', 'unknown')
                confidence = float(signal.get('confidence', 0))
                profit = float(signal.get('profit_loss_pips', 0))
                outcome = signal.get('outcome')
                
                # Strategy stats
                strategy_stats[strategy]['count'] += 1
                if outcome == 'win':
                    strategy_stats[strategy]['wins'] += 1
                strategy_stats[strategy]['total_profit'] += profit
                
                # Epic stats
                epic_stats[epic]['count'] += 1
                if outcome == 'win':
                    epic_stats[epic]['wins'] += 1
                epic_stats[epic]['total_profit'] += profit
                
                # Session stats
                session_stats[session]['count'] += 1
                if outcome == 'win':
                    session_stats[session]['wins'] += 1
                session_stats[session]['total_profit'] += profit
                
                # Confidence analysis
                if confidence > 0.8:
                    confidence_ranges['high'].append(signal)
                elif confidence > 0.6:
                    confidence_ranges['medium'].append(signal)
                else:
                    confidence_ranges['low'].append(signal)
            
            # Calculate win rates for each category
            for category in [strategy_stats, epic_stats, session_stats]:
                for key, stats in category.items():
                    stats['win_rate'] = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            
            confidence_analysis = {}
            for level, signals in confidence_ranges.items():
                if signals:
                    wins = len([s for s in signals if s['outcome'] == 'win'])
                    confidence_analysis[level] = {
                        'count': len(signals),
                        'win_rate': wins / len(signals),
                        'avg_profit': np.mean([float(s['profit_loss_pips'] or 0) for s in signals])
                    }
            
            return {
                'period_hours': hours,
                'total_signals': len(recent_signals),
                'strategy_breakdown': dict(strategy_stats),
                'epic_breakdown': dict(epic_stats),
                'session_breakdown': dict(session_stats),
                'confidence_analysis': confidence_analysis,
                'best_performing_strategy': max(strategy_stats.items(), key=lambda x: x[1]['win_rate'])[0] if strategy_stats else None,
                'best_performing_epic': max(epic_stats.items(), key=lambda x: x[1]['win_rate'])[0] if epic_stats else None,
                'best_performing_session': max(session_stats.items(), key=lambda x: x[1]['win_rate'])[0] if session_stats else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal analytics: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return {'error': str(e)}
    
    def update_daily_strategy_performance(self):
        """Update daily strategy performance aggregations"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Get today's date
            today = datetime.now().date()
            
            # Aggregate strategy performance for today
            cursor.execute('''
                SELECT 
                    strategy,
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as winning_signals,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losing_signals,
                    AVG(CASE WHEN outcome = 'win' THEN profit_loss_pips ELSE NULL END) as avg_profit_pips,
                    AVG(CASE WHEN outcome = 'loss' THEN ABS(profit_loss_pips) ELSE NULL END) as avg_loss_pips,
                    SUM(profit_loss_pips) as total_profit_pips,
                    MAX(profit_loss_pips) as largest_win_pips,
                    MIN(profit_loss_pips) as largest_loss_pips,
                    AVG(confidence) as avg_confidence
                FROM signal_tracking 
                WHERE DATE(signal_timestamp) = %s 
                AND outcome IN ('win', 'loss')
                GROUP BY strategy
            ''', (today,))
            
            strategy_data = cursor.fetchall()
            
            # Update or insert strategy performance records
            for row in strategy_data:
                strategy, total_signals, winning_signals, losing_signals, avg_profit_pips, avg_loss_pips, total_profit_pips, largest_win_pips, largest_loss_pips, avg_confidence = row
                
                win_rate = winning_signals / total_signals if total_signals > 0 else 0
                profit_factor = avg_profit_pips / avg_loss_pips if avg_loss_pips and avg_loss_pips > 0 else 0
                
                cursor.execute('''
                    INSERT INTO strategy_performance 
                    (strategy_name, date_tracked, total_signals, winning_signals, losing_signals, 
                     win_rate, avg_profit_pips, avg_loss_pips, profit_factor, total_profit_pips, 
                     largest_win_pips, largest_loss_pips, avg_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (strategy_name, date_tracked) 
                    DO UPDATE SET
                        total_signals = EXCLUDED.total_signals,
                        winning_signals = EXCLUDED.winning_signals,
                        losing_signals = EXCLUDED.losing_signals,
                        win_rate = EXCLUDED.win_rate,
                        avg_profit_pips = EXCLUDED.avg_profit_pips,
                        avg_loss_pips = EXCLUDED.avg_loss_pips,
                        profit_factor = EXCLUDED.profit_factor,
                        total_profit_pips = EXCLUDED.total_profit_pips,
                        largest_win_pips = EXCLUDED.largest_win_pips,
                        largest_loss_pips = EXCLUDED.largest_loss_pips,
                        avg_confidence = EXCLUDED.avg_confidence
                ''', (
                    strategy, today, total_signals, winning_signals, losing_signals,
                    win_rate, avg_profit_pips, avg_loss_pips, profit_factor, total_profit_pips,
                    largest_win_pips, largest_loss_pips, avg_confidence
                ))
            
            # Aggregate epic performance for today
            cursor.execute('''
                SELECT 
                    epic,
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as winning_signals,
                    AVG(CASE WHEN outcome = 'win' THEN profit_loss_pips ELSE NULL END) as avg_profit_pips,
                    AVG(CASE WHEN outcome = 'loss' THEN ABS(profit_loss_pips) ELSE NULL END) as avg_loss_pips,
                    SUM(profit_loss_pips) as total_profit_pips,
                    AVG(confidence) as avg_confidence,
                    STDDEV(profit_loss_pips) as volatility_score
                FROM signal_tracking 
                WHERE DATE(signal_timestamp) = %s 
                AND outcome IN ('win', 'loss')
                GROUP BY epic
            ''', (today,))
            
            epic_data = cursor.fetchall()
            
            # Update or insert epic performance records
            for row in epic_data:
                epic, total_signals, winning_signals, avg_profit_pips, avg_loss_pips, total_profit_pips, avg_confidence, volatility_score = row
                
                win_rate = winning_signals / total_signals if total_signals > 0 else 0
                
                # Find best and worst sessions for this epic
                cursor.execute('''
                    SELECT session, AVG(profit_loss_pips) as avg_profit
                    FROM signal_tracking 
                    WHERE DATE(signal_timestamp) = %s 
                    AND epic = %s
                    AND outcome IN ('win', 'loss')
                    GROUP BY session
                    ORDER BY avg_profit DESC
                ''', (today, epic))
                
                session_results = cursor.fetchall()
                best_session = session_results[0][0] if session_results else None
                worst_session = session_results[-1][0] if len(session_results) > 1 else None
                
                cursor.execute('''
                    INSERT INTO epic_performance 
                    (epic, date_tracked, total_signals, winning_signals, win_rate, 
                     avg_profit_pips, avg_loss_pips, total_profit_pips, volatility_score, 
                     avg_confidence, best_session, worst_session)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (epic, date_tracked) 
                    DO UPDATE SET
                        total_signals = EXCLUDED.total_signals,
                        winning_signals = EXCLUDED.winning_signals,
                        win_rate = EXCLUDED.win_rate,
                        avg_profit_pips = EXCLUDED.avg_profit_pips,
                        avg_loss_pips = EXCLUDED.avg_loss_pips,
                        total_profit_pips = EXCLUDED.total_profit_pips,
                        volatility_score = EXCLUDED.volatility_score,
                        avg_confidence = EXCLUDED.avg_confidence,
                        best_session = EXCLUDED.best_session,
                        worst_session = EXCLUDED.worst_session
                ''', (
                    epic, today, total_signals, winning_signals, win_rate,
                    avg_profit_pips, avg_loss_pips, total_profit_pips, volatility_score,
                    avg_confidence, best_session, worst_session
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"✅ Updated daily performance aggregations for {today}")
            
        except Exception as e:
            self.logger.error(f"Error updating daily strategy performance: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def get_strategy_comparison(self, days: int = 7) -> Dict:
        """Get strategy performance comparison over time"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cutoff_date = datetime.now().date() - timedelta(days=days)
            
            cursor.execute('''
                SELECT 
                    strategy_name,
                    AVG(win_rate) as avg_win_rate,
                    AVG(avg_profit_pips) as avg_profit_pips,
                    AVG(avg_loss_pips) as avg_loss_pips,
                    AVG(profit_factor) as avg_profit_factor,
                    SUM(total_signals) as total_signals,
                    SUM(total_profit_pips) as total_profit_pips,
                    AVG(avg_confidence) as avg_confidence,
                    COUNT(*) as days_active
                FROM strategy_performance 
                WHERE date_tracked > %s 
                GROUP BY strategy_name
                ORDER BY avg_win_rate DESC
            ''', (cutoff_date,))
            
            strategy_comparison = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            return {
                'period_days': days,
                'strategies': strategy_comparison,
                'best_strategy': strategy_comparison[0] if strategy_comparison else None,
                'total_strategies': len(strategy_comparison)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy comparison: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return {'error': str(e)}
    
    def get_epic_rankings(self, days: int = 7) -> Dict:
        """Get epic performance rankings"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cutoff_date = datetime.now().date() - timedelta(days=days)
            
            cursor.execute('''
                SELECT 
                    epic,
                    AVG(win_rate) as avg_win_rate,
                    AVG(avg_profit_pips) as avg_profit_pips,
                    SUM(total_signals) as total_signals,
                    SUM(total_profit_pips) as total_profit_pips,
                    AVG(volatility_score) as avg_volatility,
                    AVG(avg_confidence) as avg_confidence,
                    COUNT(*) as days_active,
                    STRING_AGG(DISTINCT best_session, ', ') as best_sessions
                FROM epic_performance 
                WHERE date_tracked > %s 
                GROUP BY epic
                ORDER BY avg_win_rate DESC, total_profit_pips DESC
            ''', (cutoff_date,))
            
            epic_rankings = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            return {
                'period_days': days,
                'epics': epic_rankings,
                'best_epic': epic_rankings[0] if epic_rankings else None,
                'total_epics': len(epic_rankings)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting epic rankings: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return {'error': str(e)}
    
    def get_session_analysis(self, days: int = 7) -> Dict:
        """Analyze performance by trading session"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cutoff_time = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT 
                    session,
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as winning_signals,
                    AVG(CASE WHEN outcome = 'win' THEN profit_loss_pips ELSE NULL END) as avg_profit_pips,
                    AVG(CASE WHEN outcome = 'loss' THEN ABS(profit_loss_pips) ELSE NULL END) as avg_loss_pips,
                    SUM(profit_loss_pips) as total_profit_pips,
                    AVG(confidence) as avg_confidence,
                    AVG(duration_minutes) as avg_duration_minutes
                FROM signal_tracking 
                WHERE signal_timestamp > %s 
                AND outcome IN ('win', 'loss')
                GROUP BY session
                ORDER BY total_signals DESC
            ''', (cutoff_time,))
            
            session_data = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict['win_rate'] = (
                    float(row_dict['winning_signals']) / float(row_dict['total_signals']) 
                    if row_dict['total_signals'] > 0 else 0
                )
                session_data.append(row_dict)
            
            cursor.close()
            conn.close()
            
            return {
                'period_days': days,
                'sessions': session_data,
                'best_session': max(session_data, key=lambda x: x['win_rate']) if session_data else None,
                'most_active_session': max(session_data, key=lambda x: x['total_signals']) if session_data else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session analysis: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return {'error': str(e)}