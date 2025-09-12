"""
Performance Tracker Module
Tracks signal outcomes, calculates performance metrics, and optimizes trading parameters
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

from core.database import DatabaseManager


class SignalOutcome(Enum):
    """Signal outcome types"""
    PENDING = "pending"
    WIN = "win" 
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    CANCELLED = "cancelled"


@dataclass
class SignalPerformance:
    """Individual signal performance data"""
    signal_id: int
    epic: str
    signal_type: str
    strategy: str
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence_score: float = 0.0
    outcome: SignalOutcome = SignalOutcome.PENDING
    pips_gained: float = 0.0
    profit_loss: float = 0.0
    duration_minutes: Optional[int] = None
    entry_timestamp: Optional[datetime] = None
    exit_timestamp: Optional[datetime] = None


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    strategy_name: str
    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    pending_signals: int = 0
    win_rate: float = 0.0
    total_pips: float = 0.0
    avg_pips_per_trade: float = 0.0
    avg_confidence: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = None


class PerformanceTracker:
    """Main performance tracking and optimization system"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Performance cache
        self._performance_cache = {}
        self._daily_metrics = {}
        
        # Optimization parameters
        self.optimization_enabled = True
        self.min_signals_for_optimization = 20
        
        # Initialize performance tracking
        self._setup_performance_tracking()
    
    def _setup_performance_tracking(self):
        """Setup performance tracking infrastructure"""
        try:
            # Create performance tracking tables if they don't exist
            self._create_performance_tables()
            
            # Load existing performance data
            self._load_performance_cache()
            
            self.logger.info("✅ Performance tracking system initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup performance tracking: {e}")
    
    def _create_performance_tables(self):
        """Create performance tracking database tables"""
        
        # Signal outcomes table
        create_signal_outcomes = """
        CREATE TABLE IF NOT EXISTS signal_outcomes (
            id SERIAL PRIMARY KEY,
            alert_history_id INTEGER REFERENCES alert_history(id),
            epic VARCHAR(50) NOT NULL,
            signal_type VARCHAR(10) NOT NULL,
            strategy VARCHAR(50) NOT NULL,
            entry_price DECIMAL(10,5),
            exit_price DECIMAL(10,5),
            stop_loss DECIMAL(10,5),
            take_profit DECIMAL(10,5),
            confidence_score DECIMAL(4,3),
            outcome VARCHAR(20) DEFAULT 'pending',
            pips_gained DECIMAL(8,2) DEFAULT 0,
            profit_loss DECIMAL(10,2) DEFAULT 0,
            duration_minutes INTEGER,
            entry_timestamp TIMESTAMP,
            exit_timestamp TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # Strategy performance summary table (enhanced)
        create_strategy_performance = """
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id SERIAL PRIMARY KEY,
            strategy_name VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            total_signals INTEGER DEFAULT 0,
            winning_signals INTEGER DEFAULT 0,
            losing_signals INTEGER DEFAULT 0,
            pending_signals INTEGER DEFAULT 0,
            win_rate DECIMAL(5,4) DEFAULT 0,
            total_pips DECIMAL(10,2) DEFAULT 0,
            avg_pips_per_trade DECIMAL(8,2) DEFAULT 0,
            avg_confidence DECIMAL(4,3) DEFAULT 0,
            profit_factor DECIMAL(6,2) DEFAULT 0,
            max_drawdown DECIMAL(8,2) DEFAULT 0,
            sharpe_ratio DECIMAL(6,3) DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(strategy_name, date)
        );
        """
        
        # Performance optimization log
        create_optimization_log = """
        CREATE TABLE IF NOT EXISTS performance_optimization_log (
            id SERIAL PRIMARY KEY,
            optimization_type VARCHAR(50) NOT NULL,
            parameter_name VARCHAR(100) NOT NULL,
            old_value TEXT,
            new_value TEXT,
            reason TEXT,
            performance_improvement DECIMAL(6,3),
            applied_at TIMESTAMP DEFAULT NOW(),
            applied_by VARCHAR(100) DEFAULT 'system'
        );
        """
        
        # Execute table creation
        self.db_manager.execute_query(create_signal_outcomes)
        self.db_manager.execute_query(create_strategy_performance)
        self.db_manager.execute_query(create_optimization_log)
        
        self.logger.info("✅ Performance tracking tables created/verified")
    
    def track_signal(
        self, 
        alert_history_id: int,
        epic: str, 
        signal_type: str, 
        strategy: str,
        entry_price: float,
        confidence_score: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> int:
        """Track a new signal for performance monitoring"""
        
        try:
            query = """
            INSERT INTO signal_outcomes (
                alert_history_id, epic, signal_type, strategy, 
                entry_price, confidence_score, stop_loss, take_profit,
                entry_timestamp, outcome
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            result = self.db_manager.execute_query(
                query,
                (alert_history_id, epic, signal_type, strategy, 
                 entry_price, confidence_score, stop_loss, take_profit,
                 datetime.now(), SignalOutcome.PENDING.value)
            )
            
            signal_outcome_id = result.iloc[0]['id']
            
            self.logger.info(f"📊 Tracking signal {signal_outcome_id}: {epic} {signal_type} ({strategy})")
            
            return signal_outcome_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to track signal: {e}")
            return None
    
    def update_signal_outcome(
        self, 
        signal_outcome_id: int,
        exit_price: float,
        outcome: SignalOutcome,
        exit_timestamp: Optional[datetime] = None
    ) -> bool:
        """Update signal outcome when trade is closed"""
        
        try:
            if exit_timestamp is None:
                exit_timestamp = datetime.now()
            
            # Get signal details
            signal_query = """
            SELECT * FROM signal_outcomes 
            WHERE id = %s
            """
            
            signal_data = self.db_manager.execute_query(signal_query, (signal_outcome_id,))
            
            if signal_data.empty:
                self.logger.error(f"Signal {signal_outcome_id} not found")
                return False
            
            signal = signal_data.iloc[0]
            
            # Calculate performance metrics
            pips_gained = self._calculate_pips(
                signal['epic'],
                signal['entry_price'],
                exit_price,
                signal['signal_type']
            )
            
            # Calculate duration
            duration_minutes = None
            if signal['entry_timestamp']:
                duration = exit_timestamp - signal['entry_timestamp']
                duration_minutes = int(duration.total_seconds() / 60)
            
            # Update signal outcome
            update_query = """
            UPDATE signal_outcomes 
            SET exit_price = %s, outcome = %s, pips_gained = %s,
                duration_minutes = %s, exit_timestamp = %s, updated_at = NOW()
            WHERE id = %s
            """
            
            self.db_manager.execute_query(
                update_query,
                (exit_price, outcome.value, pips_gained, 
                 duration_minutes, exit_timestamp, signal_outcome_id)
            )
            
            # Update strategy performance
            self._update_strategy_performance(signal['strategy'])
            
            # Update daily summary
            self._update_daily_summary()
            
            self.logger.info(f"📊 Updated signal {signal_outcome_id}: {outcome.value} ({pips_gained:.1f} pips)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update signal outcome: {e}")
            return False
    
    def _calculate_pips(self, epic: str, entry_price: float, exit_price: float, signal_type: str) -> float:
        """Calculate pip gain/loss for a trade"""
        
        # Get pip multiplier for the pair
        pip_multipliers = {
            'EURUSD': 10000, 'GBPUSD': 10000, 'AUDUSD': 10000, 'NZDUSD': 10000,
            'USDCAD': 10000, 'USDCHF': 10000,
            'USDJPY': 100, 'EURJPY': 100, 'GBPJPY': 100
        }
        
        # Extract pair from epic
        pair = 'EURUSD'  # Default
        for p in pip_multipliers.keys():
            if p in epic.upper():
                pair = p
                break
        
        multiplier = pip_multipliers.get(pair, 10000)
        
        # Calculate pip difference
        if signal_type.upper() == 'BUY':
            pip_diff = (exit_price - entry_price) * multiplier
        else:  # SELL
            pip_diff = (entry_price - exit_price) * multiplier
        
        return round(pip_diff, 1)
    
    def _update_strategy_performance(self, strategy_name: str):
        """Update strategy performance metrics"""
        
        try:
            today = datetime.now().date()
            
            # Get today's performance for this strategy
            perf_query = """
            SELECT 
                COUNT(*) as total_signals,
                COUNT(CASE WHEN outcome = 'win' THEN 1 END) as winning_signals,
                COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losing_signals,
                COUNT(CASE WHEN outcome = 'pending' THEN 1 END) as pending_signals,
                COALESCE(AVG(CASE WHEN outcome IN ('win', 'loss') THEN pips_gained END), 0) as avg_pips,
                COALESCE(SUM(pips_gained), 0) as total_pips,
                COALESCE(AVG(confidence_score), 0) as avg_confidence
            FROM signal_outcomes 
            WHERE strategy = %s AND DATE(entry_timestamp) = %s
            """
            
            result = self.db_manager.execute_query(perf_query, (strategy_name, today))
            
            if not result.empty:
                row = result.iloc[0]
                
                # Calculate win rate
                closed_trades = row['winning_signals'] + row['losing_signals']
                win_rate = row['winning_signals'] / closed_trades if closed_trades > 0 else 0
                
                # Calculate profit factor
                winning_pips = self.db_manager.execute_query("""
                    SELECT COALESCE(SUM(pips_gained), 0) as winning_pips
                    FROM signal_outcomes 
                    WHERE strategy = %s AND DATE(entry_timestamp) = %s AND outcome = 'win'
                """, (strategy_name, today)).iloc[0]['winning_pips']
                
                losing_pips = self.db_manager.execute_query("""
                    SELECT COALESCE(ABS(SUM(pips_gained)), 0) as losing_pips
                    FROM signal_outcomes 
                    WHERE strategy = %s AND DATE(entry_timestamp) = %s AND outcome = 'loss'
                """, (strategy_name, today)).iloc[0]['losing_pips']
                
                profit_factor = winning_pips / losing_pips if losing_pips > 0 else 0
                
                # Upsert strategy performance
                upsert_query = """
                INSERT INTO strategy_performance (
                    strategy_name, date, total_signals, winning_signals, 
                    losing_signals, pending_signals, win_rate, total_pips,
                    avg_pips_per_trade, avg_confidence, profit_factor, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (strategy_name, date) 
                DO UPDATE SET
                    total_signals = EXCLUDED.total_signals,
                    winning_signals = EXCLUDED.winning_signals,
                    losing_signals = EXCLUDED.losing_signals,
                    pending_signals = EXCLUDED.pending_signals,
                    win_rate = EXCLUDED.win_rate,
                    total_pips = EXCLUDED.total_pips,
                    avg_pips_per_trade = EXCLUDED.avg_pips_per_trade,
                    avg_confidence = EXCLUDED.avg_confidence,
                    profit_factor = EXCLUDED.profit_factor,
                    updated_at = NOW()
                """
                
                self.db_manager.execute_query(
                    upsert_query,
                    (strategy_name, today, row['total_signals'], row['winning_signals'],
                     row['losing_signals'], row['pending_signals'], win_rate, 
                     row['total_pips'], row['avg_pips'], row['avg_confidence'], profit_factor)
                )
                
                self.logger.info(f"📊 Updated {strategy_name} performance: {win_rate:.1%} win rate, {row['total_pips']:.1f} pips")
        
        except Exception as e:
            self.logger.error(f"❌ Failed to update strategy performance: {e}")
    
    def _update_daily_summary(self):
        """Update overall daily performance summary"""
        
        try:
            today = datetime.now().date()
            
            # Calculate overall daily metrics
            summary_query = """
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN outcome = 'win' THEN 1 END) as winning_trades,
                COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losing_trades,
                COALESCE(SUM(pips_gained), 0) as total_pips,
                COALESCE(AVG(confidence_score), 0) as avg_confidence
            FROM signal_outcomes 
            WHERE DATE(entry_timestamp) = %s
            """
            
            result = self.db_manager.execute_query(summary_query, (today,))
            
            if not result.empty:
                row = result.iloc[0]
                
                closed_trades = row['winning_trades'] + row['losing_trades']
                win_rate = row['winning_trades'] / closed_trades if closed_trades > 0 else 0
                avg_pips_per_trade = row['total_pips'] / closed_trades if closed_trades > 0 else 0
                
                # Upsert to strategy_summary table (for dashboard compatibility)
                upsert_summary = """
                INSERT INTO strategy_summary (
                    created_at, total_trades, winning_trades, losing_trades,
                    win_rate, total_pips, avg_confidence, avg_pips_per_trade, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (created_at::date) 
                DO UPDATE SET
                    total_trades = EXCLUDED.total_trades,
                    winning_trades = EXCLUDED.winning_trades,
                    losing_trades = EXCLUDED.losing_trades,
                    win_rate = EXCLUDED.win_rate,
                    total_pips = EXCLUDED.total_pips,
                    avg_confidence = EXCLUDED.avg_confidence,
                    avg_pips_per_trade = EXCLUDED.avg_pips_per_trade,
                    notes = EXCLUDED.notes
                """
                
                self.db_manager.execute_query(
                    upsert_summary,
                    (today, row['total_trades'], row['winning_trades'], row['losing_trades'],
                     win_rate, row['total_pips'], row['avg_confidence'], avg_pips_per_trade,
                     f"Auto-generated daily summary for {today}")
                )
                
                self.logger.info(f"📊 Updated daily summary: {row['total_trades']} trades, {win_rate:.1%} win rate")
        
        except Exception as e:
            self.logger.error(f"❌ Failed to update daily summary: {e}")
    
    def get_strategy_metrics(self, strategy_name: str, days: int = 30) -> StrategyMetrics:
        """Get comprehensive strategy performance metrics"""
        
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                COUNT(*) as total_signals,
                COUNT(CASE WHEN outcome = 'win' THEN 1 END) as winning_signals,
                COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losing_signals,
                COUNT(CASE WHEN outcome = 'pending' THEN 1 END) as pending_signals,
                COALESCE(AVG(CASE WHEN outcome IN ('win', 'loss') THEN pips_gained END), 0) as avg_pips,
                COALESCE(SUM(pips_gained), 0) as total_pips,
                COALESCE(AVG(confidence_score), 0) as avg_confidence
            FROM signal_outcomes 
            WHERE strategy = %s AND DATE(entry_timestamp) BETWEEN %s AND %s
            """
            
            result = self.db_manager.execute_query(query, (strategy_name, start_date, end_date))
            
            if result.empty:
                return StrategyMetrics(strategy_name=strategy_name)
            
            row = result.iloc[0]
            
            # Calculate metrics
            closed_trades = row['winning_signals'] + row['losing_signals']
            win_rate = row['winning_signals'] / closed_trades if closed_trades > 0 else 0
            
            # Calculate profit factor
            winning_pips = self.db_manager.execute_query("""
                SELECT COALESCE(SUM(pips_gained), 0) as winning_pips
                FROM signal_outcomes 
                WHERE strategy = %s AND DATE(entry_timestamp) BETWEEN %s AND %s AND outcome = 'win'
            """, (strategy_name, start_date, end_date)).iloc[0]['winning_pips']
            
            losing_pips = self.db_manager.execute_query("""
                SELECT COALESCE(ABS(SUM(pips_gained)), 0) as losing_pips
                FROM signal_outcomes 
                WHERE strategy = %s AND DATE(entry_timestamp) BETWEEN %s AND %s AND outcome = 'loss'
            """, (strategy_name, start_date, end_date)).iloc[0]['losing_pips']
            
            profit_factor = winning_pips / losing_pips if losing_pips > 0 else 0
            
            return StrategyMetrics(
                strategy_name=strategy_name,
                total_signals=row['total_signals'],
                winning_signals=row['winning_signals'],
                losing_signals=row['losing_signals'],
                pending_signals=row['pending_signals'],
                win_rate=win_rate,
                total_pips=row['total_pips'],
                avg_pips_per_trade=row['avg_pips'],
                avg_confidence=row['avg_confidence'],
                profit_factor=profit_factor,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get strategy metrics: {e}")
            return StrategyMetrics(strategy_name=strategy_name)
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """Analyze performance and suggest parameter optimizations"""
        
        if not self.optimization_enabled:
            return {"message": "Optimization disabled"}
        
        try:
            # Get recent performance data
            recent_performance = self._analyze_recent_performance()
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(recent_performance)
            
            # Log optimization analysis
            self.logger.info(f"🔧 Generated {len(suggestions)} optimization suggestions")
            
            return {
                "timestamp": datetime.now(),
                "performance_analysis": recent_performance,
                "optimization_suggestions": suggestions,
                "auto_apply_enabled": False  # Manual approval required
            }
            
        except Exception as e:
            self.logger.error(f"❌ Optimization analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Analyze recent performance for optimization"""
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Get strategy performance breakdown
        strategy_query = """
        SELECT 
            strategy,
            COUNT(*) as signals,
            COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
            COALESCE(AVG(confidence_score), 0) as avg_confidence,
            COALESCE(SUM(pips_gained), 0) as total_pips
        FROM signal_outcomes 
        WHERE DATE(entry_timestamp) BETWEEN %s AND %s
        GROUP BY strategy
        """
        
        strategy_perf = self.db_manager.execute_query(strategy_query, (start_date, end_date))
        
        # Get confidence score analysis
        confidence_query = """
        SELECT 
            CASE 
                WHEN confidence_score >= 0.8 THEN 'high'
                WHEN confidence_score >= 0.65 THEN 'medium'
                ELSE 'low'
            END as confidence_range,
            COUNT(*) as signals,
            COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
            COALESCE(AVG(pips_gained), 0) as avg_pips
        FROM signal_outcomes 
        WHERE DATE(entry_timestamp) BETWEEN %s AND %s AND outcome IN ('win', 'loss')
        GROUP BY confidence_range
        """
        
        confidence_perf = self.db_manager.execute_query(confidence_query, (start_date, end_date))
        
        return {
            "period": f"{start_date} to {end_date}",
            "strategy_performance": strategy_perf.to_dict('records') if not strategy_perf.empty else [],
            "confidence_analysis": confidence_perf.to_dict('records') if not confidence_perf.empty else []
        }
    
    def _generate_optimization_suggestions(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on performance analysis"""
        
        suggestions = []
        
        # Analyze strategy performance
        strategy_perf = performance_data.get('strategy_performance', [])
        
        for strategy in strategy_perf:
            win_rate = strategy['wins'] / strategy['signals'] if strategy['signals'] > 0 else 0
            
            # Low performing strategy
            if win_rate < 0.5 and strategy['signals'] >= 10:
                suggestions.append({
                    "type": "strategy_adjustment",
                    "parameter": f"{strategy['strategy']}_enabled",
                    "current_value": True,
                    "suggested_value": False,
                    "reason": f"Low win rate ({win_rate:.1%}) for {strategy['strategy']} strategy",
                    "expected_improvement": "Reduce losing trades",
                    "confidence": 0.7
                })
        
        # Analyze confidence thresholds
        confidence_perf = performance_data.get('confidence_analysis', [])
        
        high_conf = next((c for c in confidence_perf if c['confidence_range'] == 'high'), None)
        low_conf = next((c for c in confidence_perf if c['confidence_range'] == 'low'), None)
        
        if high_conf and low_conf:
            high_win_rate = high_conf['wins'] / high_conf['signals'] if high_conf['signals'] > 0 else 0
            low_win_rate = low_conf['wins'] / low_conf['signals'] if low_conf['signals'] > 0 else 0
            
            # If high confidence performs much better, suggest raising threshold
            if high_win_rate - low_win_rate > 0.2:
                suggestions.append({
                    "type": "confidence_threshold",
                    "parameter": "MIN_CONFIDENCE",
                    "current_value": 0.6,  # This should come from config
                    "suggested_value": 0.75,
                    "reason": f"High confidence signals perform {(high_win_rate - low_win_rate):.1%} better",
                    "expected_improvement": f"Improve win rate by filtering low-quality signals",
                    "confidence": 0.8
                })
        
        return suggestions
    
    def _load_performance_cache(self):
        """Load recent performance data into cache"""
        try:
            # Load last 24 hours of performance data
            yesterday = datetime.now() - timedelta(days=1)
            
            cache_query = """
            SELECT strategy, outcome, confidence_score, pips_gained
            FROM signal_outcomes 
            WHERE entry_timestamp >= %s
            """
            
            result = self.db_manager.execute_query(cache_query, (yesterday,))
            
            if not result.empty:
                self._performance_cache = result.to_dict('records')
                self.logger.info(f"📊 Loaded {len(self._performance_cache)} recent signals into cache")
        
        except Exception as e:
            self.logger.error(f"❌ Failed to load performance cache: {e}")


# Integration helper functions

def track_new_signal(
    db_manager: DatabaseManager,
    alert_history_id: int,
    epic: str,
    signal_type: str,
    strategy: str,
    entry_price: float,
    confidence_score: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None
) -> Optional[int]:
    """Helper function to track a new signal"""
    
    tracker = PerformanceTracker(db_manager)
    return tracker.track_signal(
        alert_history_id=alert_history_id,
        epic=epic,
        signal_type=signal_type,
        strategy=strategy,
        entry_price=entry_price,
        confidence_score=confidence_score,
        stop_loss=stop_loss,
        take_profit=take_profit
    )


def update_signal_result(
    db_manager: DatabaseManager,
    signal_outcome_id: int,
    exit_price: float,
    outcome: SignalOutcome,
    exit_timestamp: Optional[datetime] = None
) -> bool:
    """Helper function to update signal outcome"""
    
    tracker = PerformanceTracker(db_manager)
    return tracker.update_signal_outcome(
        signal_outcome_id=signal_outcome_id,
        exit_price=exit_price,
        outcome=outcome,
        exit_timestamp=exit_timestamp
    )


def get_daily_performance_summary(db_manager: DatabaseManager) -> Dict[str, Any]:
    """Get today's performance summary for dashboard"""
    
    tracker = PerformanceTracker(db_manager)
    
    # Get overall metrics
    overall_metrics = tracker.get_strategy_metrics("Overall", days=1)
    
    # Get strategy breakdown
    strategies = ["EMA", "MACD", "Combined"]
    strategy_metrics = {}
    
    for strategy in strategies:
        metrics = tracker.get_strategy_metrics(strategy, days=1)
        strategy_metrics[strategy] = {
            "signals": metrics.total_signals,
            "win_rate": metrics.win_rate,
            "total_pips": metrics.total_pips
        }
    
    return {
        "overall": {
            "total_signals": overall_metrics.total_signals,
            "win_rate": overall_metrics.win_rate,
            "total_pips": overall_metrics.total_pips,
            "avg_confidence": overall_metrics.avg_confidence
        },
        "strategies": strategy_metrics,
        "last_updated": datetime.now()
    }