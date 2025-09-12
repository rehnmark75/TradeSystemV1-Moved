# core/trading/performance_tracker.py
"""
Performance Tracker - Extracted from TradingOrchestrator
Tracks trading performance, metrics, and analytics
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json
try:
    import config
except ImportError:
    from forex_scanner import config


class PerformanceTracker:
    """
    Tracks trading performance, metrics, and analytics
    Extracted from TradingOrchestrator to provide focused performance monitoring
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 db_manager=None):
        
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        
        # Performance metrics
        self.session_metrics = {
            'total_signals': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'total_volume': 0.0,
            'session_start': None,
            'session_duration': 0,
            'average_confidence': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        # Strategy performance tracking
        self.strategy_performance = defaultdict(lambda: {
            'signals': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_pnl': 0.0
        })
        
        # Epic performance tracking
        self.epic_performance = defaultdict(lambda: {
            'signals': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'avg_confidence': 0.0
        })
        
        # Time-based performance
        self.hourly_performance = defaultdict(lambda: {
            'signals': 0,
            'trades': 0,
            'pnl': 0.0
        })
        
        # Trade execution metrics
        self.execution_metrics = {
            'total_execution_time': 0.0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'fastest_execution': float('inf'),
            'slowest_execution': 0.0
        }
        
        # Daily tracking
        self.daily_metrics = {}
        self.current_date = datetime.now().date()
        
        # Historical data
        self.trade_history = []
        self.signal_history = []
        
        self.logger.info("📊 PerformanceTracker initialized")
    
    def track_signal_performance(self, signal: Dict, outcome: str = None):
        """
        Track performance of individual signals
        
        Args:
            signal: Signal dictionary
            outcome: Optional outcome ('EXECUTED', 'REJECTED', 'FAILED')
        """
        try:
            current_time = datetime.now()
            
            # Update session metrics
            self.session_metrics['total_signals'] += 1
            
            # Extract signal information
            epic = signal.get('epic', 'UNKNOWN')
            strategy = signal.get('strategy', 'UNKNOWN')
            confidence = float(signal.get('confidence', 0))
            
            # Track strategy performance
            self.strategy_performance[strategy]['signals'] += 1
            
            # Track epic performance
            self.epic_performance[epic]['signals'] += 1
            
            # Update confidence tracking
            if confidence > 0:
                current_avg = self.epic_performance[epic]['avg_confidence']
                current_count = self.epic_performance[epic]['signals']
                new_avg = ((current_avg * (current_count - 1)) + confidence) / current_count
                self.epic_performance[epic]['avg_confidence'] = new_avg
                
                # Update session average confidence
                session_count = self.session_metrics['total_signals']
                session_avg = self.session_metrics['average_confidence']
                new_session_avg = ((session_avg * (session_count - 1)) + confidence) / session_count
                self.session_metrics['average_confidence'] = new_session_avg
            
            # Track hourly performance
            hour = current_time.hour
            self.hourly_performance[hour]['signals'] += 1
            
            # Add to signal history
            signal_record = {
                'timestamp': current_time,
                'epic': epic,
                'strategy': strategy,
                'confidence': confidence,
                'outcome': outcome,
                'direction': signal.get('direction'),
                'price': signal.get('price')
            }
            self.signal_history.append(signal_record)
            
            # Keep only recent history (last 1000 signals)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            self.logger.debug(f"📊 Signal tracked: {epic} {strategy} {confidence}% - {outcome}")
            
        except Exception as e:
            self.logger.error(f"❌ Error tracking signal performance: {e}")
    
    def track_trade_execution(self, trade_result: Dict, execution_time: float = None):
        """
        Track trade execution performance
        
        Args:
            trade_result: Dictionary containing trade execution results
            execution_time: Time taken to execute trade in seconds
        """
        try:
            current_time = datetime.now()
            
            # Extract trade information
            epic = trade_result.get('epic', 'UNKNOWN')
            strategy = trade_result.get('strategy', 'UNKNOWN')
            pnl = float(trade_result.get('pnl', 0.0))
            status = trade_result.get('status', 'UNKNOWN')
            volume = float(trade_result.get('volume', 0.0))
            
            # Update session metrics
            self.session_metrics['total_trades'] += 1
            self.session_metrics['total_pnl'] += pnl
            self.session_metrics['total_volume'] += volume
            
            # Track success/failure
            if status in ['FILLED', 'COMPLETED', 'SUCCESS']:
                self.session_metrics['successful_trades'] += 1
                self.strategy_performance[strategy]['trades'] += 1
                self.epic_performance[epic]['trades'] += 1
                
                # Track win/loss
                if pnl > 0:
                    self.strategy_performance[strategy]['wins'] += 1
                    self.epic_performance[epic]['wins'] += 1
                elif pnl < 0:
                    self.strategy_performance[strategy]['losses'] += 1
                    self.epic_performance[epic]['losses'] += 1
                
                # Update PnL tracking
                self.strategy_performance[strategy]['total_pnl'] += pnl
                self.epic_performance[epic]['total_pnl'] += pnl
                
                # Update best/worst trade
                if pnl > self.session_metrics['best_trade']:
                    self.session_metrics['best_trade'] = pnl
                if pnl < self.session_metrics['worst_trade']:
                    self.session_metrics['worst_trade'] = pnl
                
            else:
                self.session_metrics['failed_trades'] += 1
            
            # Track execution timing
            if execution_time is not None:
                self.execution_metrics['total_execution_time'] += execution_time
                if status in ['FILLED', 'COMPLETED', 'SUCCESS']:
                    self.execution_metrics['successful_executions'] += 1
                else:
                    self.execution_metrics['failed_executions'] += 1
                
                # Update timing statistics
                total_executions = (self.execution_metrics['successful_executions'] + 
                                  self.execution_metrics['failed_executions'])
                if total_executions > 0:
                    self.execution_metrics['average_execution_time'] = (
                        self.execution_metrics['total_execution_time'] / total_executions
                    )
                
                if execution_time < self.execution_metrics['fastest_execution']:
                    self.execution_metrics['fastest_execution'] = execution_time
                if execution_time > self.execution_metrics['slowest_execution']:
                    self.execution_metrics['slowest_execution'] = execution_time
            
            # Track hourly performance
            hour = current_time.hour
            self.hourly_performance[hour]['trades'] += 1
            self.hourly_performance[hour]['pnl'] += pnl
            
            # Add to trade history
            trade_record = {
                'timestamp': current_time,
                'epic': epic,
                'strategy': strategy,
                'pnl': pnl,
                'volume': volume,
                'status': status,
                'execution_time': execution_time
            }
            self.trade_history.append(trade_record)
            
            # Keep only recent history (last 500 trades)
            if len(self.trade_history) > 500:
                self.trade_history = self.trade_history[-500:]
            
            self.logger.info(f"📊 Trade tracked: {epic} ${pnl:.2f} {status}")
            
        except Exception as e:
            self.logger.error(f"❌ Error tracking trade execution: {e}")
    
    def calculate_win_rates(self):
        """Calculate win rates for strategies and epics"""
        try:
            # Strategy win rates
            for strategy, metrics in self.strategy_performance.items():
                total_trades = metrics['wins'] + metrics['losses']
                if total_trades > 0:
                    metrics['win_rate'] = (metrics['wins'] / total_trades) * 100
                    metrics['avg_pnl'] = metrics['total_pnl'] / total_trades
            
            # Epic win rates
            for epic, metrics in self.epic_performance.items():
                total_trades = metrics['wins'] + metrics['losses']
                if total_trades > 0:
                    metrics['win_rate'] = (metrics['wins'] / total_trades) * 100
                    metrics['avg_pnl'] = metrics['total_pnl'] / total_trades
                    
        except Exception as e:
            self.logger.error(f"❌ Error calculating win rates: {e}")
    
    def generate_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            self.calculate_win_rates()
            
            # Calculate session duration
            if self.session_metrics['session_start']:
                duration = datetime.now() - self.session_metrics['session_start']
                self.session_metrics['session_duration'] = duration.total_seconds()
            
            # Calculate overall win rate
            total_wins = self.session_metrics['successful_trades']
            total_trades = self.session_metrics['total_trades']
            overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            
            # Get top performing strategies
            top_strategies = sorted(
                [(k, v) for k, v in self.strategy_performance.items() if v['trades'] > 0],
                key=lambda x: x[1]['win_rate'],
                reverse=True
            )[:5]
            
            # Get top performing epics
            top_epics = sorted(
                [(k, v) for k, v in self.epic_performance.items() if v['trades'] > 0],
                key=lambda x: x[1]['total_pnl'],
                reverse=True
            )[:5]
            
            # Calculate hourly activity
            hourly_summary = {}
            for hour, metrics in self.hourly_performance.items():
                hourly_summary[f"{hour:02d}:00"] = {
                    'signals': metrics['signals'],
                    'trades': metrics['trades'],
                    'pnl': round(metrics['pnl'], 2)
                }
            
            # Risk metrics
            if total_trades > 0:
                avg_pnl_per_trade = self.session_metrics['total_pnl'] / total_trades
                profit_factor = abs(self.session_metrics['best_trade'] / self.session_metrics['worst_trade']) if self.session_metrics['worst_trade'] < 0 else 0
            else:
                avg_pnl_per_trade = 0
                profit_factor = 0
            
            report = {
                'session_overview': {
                    'total_signals': self.session_metrics['total_signals'],
                    'total_trades': self.session_metrics['total_trades'],
                    'win_rate': round(overall_win_rate, 2),
                    'total_pnl': round(self.session_metrics['total_pnl'], 2),
                    'average_confidence': round(self.session_metrics['average_confidence'], 1),
                    'session_duration_hours': round(self.session_metrics['session_duration'] / 3600, 2),
                    'signals_per_hour': round(self.session_metrics['total_signals'] / max(self.session_metrics['session_duration'] / 3600, 1), 2),
                    'trades_per_hour': round(self.session_metrics['total_trades'] / max(self.session_metrics['session_duration'] / 3600, 1), 2)
                },
                'trading_performance': {
                    'successful_trades': self.session_metrics['successful_trades'],
                    'failed_trades': self.session_metrics['failed_trades'],
                    'best_trade': round(self.session_metrics['best_trade'], 2),
                    'worst_trade': round(self.session_metrics['worst_trade'], 2),
                    'average_pnl_per_trade': round(avg_pnl_per_trade, 2),
                    'total_volume': round(self.session_metrics['total_volume'], 4),
                    'profit_factor': round(profit_factor, 2)
                },
                'execution_metrics': {
                    'successful_executions': self.execution_metrics['successful_executions'],
                    'failed_executions': self.execution_metrics['failed_executions'],
                    'average_execution_time': round(self.execution_metrics['average_execution_time'], 3),
                    'fastest_execution': round(self.execution_metrics['fastest_execution'], 3) if self.execution_metrics['fastest_execution'] != float('inf') else 0,
                    'slowest_execution': round(self.execution_metrics['slowest_execution'], 3)
                },
                'strategy_performance': {
                    strategy: {
                        'signals': metrics['signals'],
                        'trades': metrics['trades'],
                        'win_rate': round(metrics['win_rate'], 2),
                        'total_pnl': round(metrics['total_pnl'], 2),
                        'avg_pnl': round(metrics['avg_pnl'], 2)
                    } for strategy, metrics in dict(top_strategies).items()
                },
                'epic_performance': {
                    epic: {
                        'signals': metrics['signals'],
                        'trades': metrics['trades'],
                        'win_rate': round(metrics['win_rate'], 2),
                        'total_pnl': round(metrics['total_pnl'], 2),
                        'avg_confidence': round(metrics['avg_confidence'], 1)
                    } for epic, metrics in dict(top_epics).items()
                },
                'hourly_activity': hourly_summary,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating performance report: {e}")
            return {'error': str(e)}
    
    def track_strategy_performance(self, strategy: str, result: Dict):
        """
        Track performance of specific strategy
        
        Args:
            strategy: Strategy name
            result: Strategy execution result
        """
        try:
            # This method allows tracking strategy-specific metrics
            # beyond what's captured in general trade tracking
            
            performance_data = {
                'strategy': strategy,
                'timestamp': datetime.now(),
                'result': result,
                'metrics': result.get('metrics', {})
            }
            
            # Store in strategy-specific tracking
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    'signals': 0,
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'avg_pnl': 0.0,
                    'detailed_results': []
                }
            
            self.strategy_performance[strategy]['detailed_results'].append(performance_data)
            
            # Keep only recent detailed results
            if len(self.strategy_performance[strategy]['detailed_results']) > 100:
                self.strategy_performance[strategy]['detailed_results'] = \
                    self.strategy_performance[strategy]['detailed_results'][-100:]
            
        except Exception as e:
            self.logger.error(f"❌ Error tracking strategy performance: {e}")
    
    def monitor_execution_latency(self, operation: str, latency: float):
        """
        Monitor execution latency for different operations
        
        Args:
            operation: Operation name (e.g., 'signal_detection', 'order_execution')
            latency: Latency in seconds
        """
        try:
            if not hasattr(self, 'latency_tracking'):
                self.latency_tracking = defaultdict(list)
            
            self.latency_tracking[operation].append({
                'timestamp': datetime.now(),
                'latency': latency
            })
            
            # Keep only recent measurements (last 100)
            if len(self.latency_tracking[operation]) > 100:
                self.latency_tracking[operation] = self.latency_tracking[operation][-100:]
            
            # Log slow operations
            if latency > 5.0:  # Alert for operations taking more than 5 seconds
                self.logger.warning(f"⚠️ Slow {operation}: {latency:.2f}s")
                
        except Exception as e:
            self.logger.error(f"❌ Error monitoring latency: {e}")
    
    def reset_session_metrics(self):
        """Reset session metrics for new session"""
        self.session_metrics = {
            'total_signals': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'total_volume': 0.0,
            'session_start': datetime.now(),
            'session_duration': 0,
            'average_confidence': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        self.strategy_performance.clear()
        self.epic_performance.clear()
        self.hourly_performance.clear()
        
        self.execution_metrics = {
            'total_execution_time': 0.0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'fastest_execution': float('inf'),
            'slowest_execution': 0.0
        }
        
        self.logger.info("🔄 Session metrics reset")
    
    def save_performance_data(self, filename: str = None):
        """Save performance data to file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"performance_report_{timestamp}.json"
            
            report = self.generate_performance_report()
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"📄 Performance data saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving performance data: {e}")
    
    def get_real_time_metrics(self) -> Dict:
        """Get real-time performance metrics for monitoring"""
        try:
            current_time = datetime.now()
            
            # Calculate recent performance (last hour)
            one_hour_ago = current_time - timedelta(hours=1)
            recent_trades = [t for t in self.trade_history if t['timestamp'] > one_hour_ago]
            recent_signals = [s for s in self.signal_history if s['timestamp'] > one_hour_ago]
            
            recent_pnl = sum(t['pnl'] for t in recent_trades)
            recent_wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            recent_win_rate = (recent_wins / len(recent_trades) * 100) if recent_trades else 0
            
            return {
                'timestamp': current_time.isoformat(),
                'session_uptime_minutes': round((current_time - self.session_metrics.get('session_start', current_time)).total_seconds() / 60, 1),
                'total_signals': self.session_metrics['total_signals'],
                'total_trades': self.session_metrics['total_trades'],
                'session_pnl': round(self.session_metrics['total_pnl'], 2),
                'recent_hour': {
                    'signals': len(recent_signals),
                    'trades': len(recent_trades),
                    'pnl': round(recent_pnl, 2),
                    'win_rate': round(recent_win_rate, 1)
                },
                'execution_health': {
                    'avg_execution_time': round(self.execution_metrics['average_execution_time'], 3),
                    'success_rate': round((self.execution_metrics['successful_executions'] / 
                                         max(self.execution_metrics['successful_executions'] + 
                                             self.execution_metrics['failed_executions'], 1)) * 100, 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting real-time metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}