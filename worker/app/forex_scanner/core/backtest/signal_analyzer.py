# core/backtest/signal_analyzer.py
"""
Signal Analyzer
Displays and analyzes signal data
"""

import pandas as pd
from typing import List, Dict, Optional,Union
from datetime import datetime
import logging


class SignalAnalyzer:
    """Analyzes and displays signal data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def display_signal_list(self, signals: List[Dict], timezone_manager=None, max_signals: int = 50):
        """
        Display detailed list of signals with timestamps
        """
        if not signals:
            self.logger.info("📋 No signals to display")
            return
        
        # Import here to avoid circular imports
        from utils.timezone_utils import TimezoneManager
        if timezone_manager is None:
            timezone_manager = TimezoneManager('Europe/Stockholm')
        
        # Sort signals by timestamp (newest first)
        sorted_signals = sorted(signals, key=lambda x: x['timestamp'], reverse=True)
        
        # Limit display
        display_signals = sorted_signals[:max_signals]
        
        self.logger.info(f"📋 SIGNAL LIST (showing {len(display_signals)} of {len(signals)} signals):")
        self.logger.info("=" * 120)
        
        # Header
        header = f"{'#':<3} {'TIMESTAMP':<20} {'PAIR':<8} {'TYPE':<4} {'STRATEGY':<15} {'PRICE':<8} {'CONF':<6} {'PROFIT':<8} {'LOSS':<8} {'R:R':<6}"
        self.logger.info(header)
        self.logger.info("-" * 120)
        
        for i, signal in enumerate(display_signals, 1):
            # Format timestamp
            timestamp = signal.get('timestamp')
            if isinstance(timestamp, str):
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    timestamp_str = str(timestamp)[:16]
            
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()
            
            if isinstance(timestamp, datetime):
                try:
                    local_time = timezone_manager.utc_to_local(timestamp)
                    timestamp_str = local_time.strftime('%Y-%m-%d %H:%M')
                except:
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
            else:
                timestamp_str = str(timestamp)[:16]
            
            # Extract pair from epic
            epic = signal.get('epic', 'Unknown')
            pair = epic.split('.')[-3] if '.' in epic else epic[:6]
            
            # Signal details
            signal_type = signal.get('signal_type', 'UNK')
            strategy = signal.get('strategy', 'unknown')[:13]  # Truncate strategy name
            confidence = signal.get('confidence_score', 0)
            price = signal.get('price', 0)
            profit_pips = signal.get('max_profit_pips', 0)
            loss_pips = signal.get('max_loss_pips', 0)
            
            # Risk:Reward ratio
            if loss_pips > 0:
                rr_ratio = profit_pips / loss_pips
                rr_str = f"{rr_ratio:.2f}"
            else:
                rr_str = "N/A"
            
            # Format row
            row = f"{i:<3} {timestamp_str:<20} {pair:<8} {signal_type:<4} {strategy:<15} {price:<8.5f} {confidence:<6.1%} {profit_pips:<8.1f} {loss_pips:<8.1f} {rr_str:<6}"
            self.logger.info(row)
        
        if len(signals) > max_signals:
            self.logger.info(f"... and {len(signals) - max_signals} more signals")
        
        self.logger.info("=" * 120)
    
    def display_signal_summary_by_pair(self, signals: List[Dict]):
        """Display signal summary grouped by currency pair"""
        if not signals:
            return
        
        # Group signals by pair
        pair_stats = {}
        
        for signal in signals:
            epic = signal.get('epic', 'Unknown')
            pair = epic.split('.')[-3] if '.' in epic else epic[:6]
            
            if pair not in pair_stats:
                pair_stats[pair] = {
                    'total': 0,
                    'bull': 0,
                    'bear': 0,
                    'strategies': {}
                }
            
            stats = pair_stats[pair]
            stats['total'] += 1
            
            if signal.get('signal_type') == 'BULL':
                stats['bull'] += 1
            elif signal.get('signal_type') == 'BEAR':
                stats['bear'] += 1
            
            # Track strategies
            strategy = signal.get('strategy', 'unknown')
            if strategy not in stats['strategies']:
                stats['strategies'][strategy] = 0
            stats['strategies'][strategy] += 1
        
        # Display summary
        self.logger.info("📊 SIGNALS BY CURRENCY PAIR:")
        self.logger.info("=" * 90)
        
        header = f"{'PAIR':<8} {'TOTAL':<6} {'BULL':<5} {'BEAR':<5} {'STRATEGIES':<30}"
        self.logger.info(header)
        self.logger.info("-" * 90)
        
        for pair in sorted(pair_stats.keys()):
            stats = pair_stats[pair]
            strategies_str = ", ".join([f"{k}:{v}" for k, v in stats['strategies'].items()])[:28]
            row = f"{pair:<8} {stats['total']:<6} {stats['bull']:<5} {stats['bear']:<5} {strategies_str:<30}"
            self.logger.info(row)
        
        self.logger.info("=" * 90)
    
    def display_signal_summary_by_strategy(self, signals: List[Dict]):
        """Display signal summary grouped by strategy"""
        if not signals:
            return
        
        # Group signals by strategy
        strategy_stats = {}
        
        for signal in signals:
            strategy = signal.get('strategy', 'unknown')
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total': 0,
                    'bull': 0,
                    'bear': 0,
                    'avg_confidence': 0,
                    'confidence_sum': 0,
                    'performance_signals': 0,
                    'avg_profit': 0,
                    'avg_loss': 0
                }
            
            stats = strategy_stats[strategy]
            stats['total'] += 1
            
            if signal.get('signal_type') == 'BULL':
                stats['bull'] += 1
            elif signal.get('signal_type') == 'BEAR':
                stats['bear'] += 1
            
            # Confidence tracking
            confidence = signal.get('confidence_score', 0)
            stats['confidence_sum'] += confidence
            
            # Performance tracking
            if 'max_profit_pips' in signal:
                stats['performance_signals'] += 1
                stats['avg_profit'] += signal['max_profit_pips']
                stats['avg_loss'] += signal.get('max_loss_pips', 0)
        
        # Calculate averages
        for strategy, stats in strategy_stats.items():
            if stats['total'] > 0:
                stats['avg_confidence'] = stats['confidence_sum'] / stats['total']
            if stats['performance_signals'] > 0:
                stats['avg_profit'] = stats['avg_profit'] / stats['performance_signals']
                stats['avg_loss'] = stats['avg_loss'] / stats['performance_signals']
        
        # Display summary
        self.logger.info("📊 SIGNALS BY STRATEGY:")
        self.logger.info("=" * 100)
        
        header = f"{'STRATEGY':<20} {'TOTAL':<6} {'BULL':<5} {'BEAR':<5} {'AVG_CONF':<9} {'AVG_PROFIT':<11} {'AVG_LOSS':<9}"
        self.logger.info(header)
        self.logger.info("-" * 100)
        
        for strategy in sorted(strategy_stats.keys()):
            stats = strategy_stats[strategy]
            row = (f"{strategy:<20} {stats['total']:<6} {stats['bull']:<5} {stats['bear']:<5} "
                   f"{stats['avg_confidence']:<9.1%} {stats['avg_profit']:<11.1f} {stats['avg_loss']:<9.1f}")
            self.logger.info(row)
        
        self.logger.info("=" * 100)
    
    def analyze_signal_timing(self, signals: List[Dict]):
        """Analyze signal timing patterns"""
        if not signals:
            return
        
        # Import for timestamp handling
        from collections import defaultdict
        
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        
        for signal in signals:
            timestamp = signal.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                
                if hasattr(timestamp, 'hour'):
                    hourly_counts[timestamp.hour] += 1
                if hasattr(timestamp, 'weekday'):
                    daily_counts[timestamp.weekday()] += 1
        
        # Display hourly distribution
        self.logger.info("🕐 SIGNALS BY HOUR:")
        self.logger.info("-" * 50)
        for hour in sorted(hourly_counts.keys()):
            count = hourly_counts[hour]
            bar = "█" * (count // 2) if count > 0 else ""
            self.logger.info(f"{hour:02d}:00  {count:3d} {bar}")
        
        # Display daily distribution
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        self.logger.info("\n📅 SIGNALS BY DAY:")
        self.logger.info("-" * 50)
        for day_idx in sorted(daily_counts.keys()):
            count = daily_counts[day_idx]
            day_name = days[day_idx] if day_idx < len(days) else f"Day{day_idx}"
            bar = "█" * (count // 5) if count > 0 else ""
            self.logger.info(f"{day_name}  {count:3d} {bar}")
    
    def find_best_performing_signals(self, signals: List[Dict], top_n: int = 10) -> List[Dict]:
        """Find the best performing signals by risk/reward ratio"""
        # Filter signals with performance data
        performance_signals = [s for s in signals if 'max_profit_pips' in s and 'max_loss_pips' in s]
        
        if not performance_signals:
            return []
        
        # Sort by risk/reward ratio
        sorted_signals = sorted(
            performance_signals, 
            key=lambda x: x.get('risk_reward_ratio', 0), 
            reverse=True
        )
        
        return sorted_signals[:top_n]
    
    def find_worst_performing_signals(self, signals: List[Dict], bottom_n: int = 10) -> List[Dict]:
        """Find the worst performing signals by risk/reward ratio"""
        # Filter signals with performance data
        performance_signals = [s for s in signals if 'max_profit_pips' in s and 'max_loss_pips' in s]
        
        if not performance_signals:
            return []
        
        # Sort by risk/reward ratio (ascending for worst)
        sorted_signals = sorted(
            performance_signals, 
            key=lambda x: x.get('risk_reward_ratio', 0)
        )
        
        return sorted_signals[:bottom_n]
    
    def export_signals_to_csv(self, signals: List[Dict], filename: str = "signals_export.csv"):
        """Export signals to CSV file"""
        if not signals:
            self.logger.warning("No signals to export")
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(signals)
            
            # Clean up nested dictionaries (like enhanced_data)
            columns_to_keep = [
                'timestamp', 'epic', 'signal_type', 'strategy', 'price', 
                'confidence_score', 'timeframe', 'max_profit_pips', 'max_loss_pips', 
                'risk_reward_ratio', 'ema_9', 'ema_21', 'ema_200'
            ]
            
            # Keep only columns that exist
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df_export = df[available_columns]
            
            # Export to CSV
            df_export.to_csv(filename, index=False)
            self.logger.info(f"📁 Exported {len(signals)} signals to {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export signals: {e}")