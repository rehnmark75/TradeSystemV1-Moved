"""
Log Database Correlator - Correlate log entries with alert_history database
Provides enriched signal intelligence by linking logs with stored signal data
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from enhanced_log_parser import EnhancedLogParser, ParsedLogEntry, SignalContext, LogSource

logger = logging.getLogger(__name__)

@dataclass
class EnrichedSignalEntry:
    """Signal log entry enriched with database information"""
    # Log data
    log_entry: ParsedLogEntry

    # Database correlation
    alert_history_match: Optional[Dict[str, Any]] = None
    signal_performance: Optional[Dict[str, Any]] = None
    strategy_stats: Optional[Dict[str, Any]] = None

    # Enriched insights
    success_probability: Optional[float] = None
    similar_signals_count: Optional[int] = None
    average_confidence: Optional[float] = None

@dataclass
class SignalPerformanceMetrics:
    """Performance metrics for signal analysis"""
    total_signals: int = 0
    successful_signals: int = 0
    success_rate: float = 0.0
    average_confidence: float = 0.0
    avg_profit_pips: Optional[float] = None
    best_timeframe: Optional[str] = None
    best_strategy: Optional[str] = None

class LogDatabaseCorrelator:
    """Correlate log entries with alert_history database for enriched intelligence"""

    def __init__(self, database_url: str = None):
        """
        Initialize with database connection

        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/forex"
        )

        self.engine = None
        self.Session = None
        self.logger = logging.getLogger(__name__)

        # Initialize log parser
        self.log_parser = EnhancedLogParser()

        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.database_url)
            self.Session = sessionmaker(bind=self.engine)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self.logger.info("✅ Database correlation service connected")

        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            raise

    def get_enriched_signal_intelligence(self, hours_back: int = 4, max_signals: int = 50) -> List[EnrichedSignalEntry]:
        """Get signal log entries enriched with database correlation"""
        try:
            # Get signal logs from enhanced parser
            signal_logs = self.log_parser.get_signal_intelligence(hours_back, max_signals)

            enriched_signals = []

            for log_entry in signal_logs:
                enriched = self._enrich_signal_entry(log_entry)
                enriched_signals.append(enriched)

            return enriched_signals

        except Exception as e:
            self.logger.error(f"Error getting enriched signal intelligence: {e}")
            return []

    def _enrich_signal_entry(self, log_entry: ParsedLogEntry) -> EnrichedSignalEntry:
        """Enrich a single log entry with database information"""
        enriched = EnrichedSignalEntry(log_entry=log_entry)

        try:
            # Try to find matching alert_history entry
            alert_match = self._find_alert_history_match(log_entry)
            if alert_match:
                enriched.alert_history_match = alert_match

                # Get performance data for this signal
                enriched.signal_performance = self._get_signal_performance(alert_match)

            # Get strategy statistics
            if log_entry.signal_context and log_entry.signal_context.strategy:
                enriched.strategy_stats = self._get_strategy_stats(
                    log_entry.signal_context.strategy,
                    log_entry.epic
                )

            # Calculate success probability based on similar signals
            enriched.success_probability = self._calculate_success_probability(log_entry)

            # Get count of similar signals
            enriched.similar_signals_count = self._count_similar_signals(log_entry)

        except Exception as e:
            self.logger.error(f"Error enriching signal entry: {e}")

        return enriched

    def _find_alert_history_match(self, log_entry: ParsedLogEntry) -> Optional[Dict[str, Any]]:
        """Find matching alert_history entry for log entry"""
        if not log_entry.signal_context or not log_entry.epic:
            return None

        try:
            # Search for alert_history entries within time window
            time_window = timedelta(minutes=5)  # 5-minute window for matching
            start_time = log_entry.timestamp - time_window
            end_time = log_entry.timestamp + time_window

            query = text("""
                SELECT id, epic, signal_type, strategy, confidence_score,
                       timeframe, created_at, signal_trigger, market_regime
                FROM alert_history
                WHERE epic = :epic
                  AND created_at BETWEEN :start_time AND :end_time
                  AND (:strategy IS NULL OR strategy = :strategy)
                ORDER BY created_at DESC
                LIMIT 1
            """)

            params = {
                'epic': log_entry.epic,
                'start_time': start_time,
                'end_time': end_time,
                'strategy': log_entry.signal_context.strategy
            }

            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                row = result.fetchone()

                if row:
                    return dict(row._mapping)

        except Exception as e:
            self.logger.error(f"Error finding alert history match: {e}")

        return None

    def _get_signal_performance(self, alert_history: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get performance data for a specific signal"""
        try:
            alert_id = alert_history['id']

            # This would connect to your trading results/PnL database
            # For now, we'll return basic performance metrics
            query = text("""
                SELECT
                    COUNT(*) as total_similar,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as recent_count
                FROM alert_history
                WHERE epic = :epic
                  AND strategy = :strategy
                  AND signal_type = :signal_type
                  AND created_at > NOW() - INTERVAL '30 days'
            """)

            params = {
                'epic': alert_history['epic'],
                'strategy': alert_history['strategy'],
                'signal_type': alert_history['signal_type']
            }

            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                row = result.fetchone()

                if row:
                    return dict(row._mapping)

        except Exception as e:
            self.logger.error(f"Error getting signal performance: {e}")

        return None

    def _get_strategy_stats(self, strategy: str, epic: str = None) -> Optional[Dict[str, Any]]:
        """Get comprehensive strategy statistics"""
        try:
            query = text("""
                SELECT
                    strategy,
                    COUNT(*) as total_signals,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(DISTINCT epic) as unique_pairs,
                    COUNT(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as signals_24h,
                    COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as signals_7d,
                    MAX(created_at) as last_signal,
                    MIN(confidence_score) as min_confidence,
                    MAX(confidence_score) as max_confidence
                FROM alert_history
                WHERE strategy = :strategy
                  AND (:epic IS NULL OR epic = :epic)
                  AND created_at > NOW() - INTERVAL '30 days'
                GROUP BY strategy
            """)

            params = {
                'strategy': strategy,
                'epic': epic
            }

            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                row = result.fetchone()

                if row:
                    stats = dict(row._mapping)
                    # Convert datetime to string for JSON serialization
                    if stats.get('last_signal'):
                        stats['last_signal'] = stats['last_signal'].isoformat()
                    return stats

        except Exception as e:
            self.logger.error(f"Error getting strategy stats: {e}")

        return None

    def _calculate_success_probability(self, log_entry: ParsedLogEntry) -> Optional[float]:
        """Calculate success probability based on historical data"""
        if not log_entry.signal_context:
            return None

        try:
            context = log_entry.signal_context

            # Build conditions for similar signals
            conditions = []
            params = {}

            if context.strategy:
                conditions.append("strategy = :strategy")
                params['strategy'] = context.strategy

            if context.signal_type:
                conditions.append("signal_type = :signal_type")
                params['signal_type'] = context.signal_type

            if log_entry.epic:
                conditions.append("epic = :epic")
                params['epic'] = log_entry.epic

            if context.confidence and context.confidence > 0:
                # Look for signals within confidence range
                conditions.append("confidence_score BETWEEN :conf_min AND :conf_max")
                params['conf_min'] = max(0, context.confidence - 0.1)
                params['conf_max'] = min(1, context.confidence + 0.1)

            if not conditions:
                return None

            where_clause = " AND ".join(conditions)

            query = text(f"""
                SELECT
                    COUNT(*) as total_signals,
                    AVG(confidence_score) as avg_confidence
                FROM alert_history
                WHERE {where_clause}
                  AND created_at > NOW() - INTERVAL '60 days'
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                row = result.fetchone()

                if row and row.total_signals > 0:
                    # Simple success probability based on historical confidence
                    # This could be enhanced with actual PnL data
                    base_probability = min(0.9, row.avg_confidence * 1.2)
                    return float(base_probability)

        except Exception as e:
            self.logger.error(f"Error calculating success probability: {e}")

        return None

    def _count_similar_signals(self, log_entry: ParsedLogEntry) -> Optional[int]:
        """Count similar signals in recent history"""
        if not log_entry.signal_context:
            return None

        try:
            context = log_entry.signal_context

            query = text("""
                SELECT COUNT(*) as count
                FROM alert_history
                WHERE (:strategy IS NULL OR strategy = :strategy)
                  AND (:epic IS NULL OR epic = :epic)
                  AND (:signal_type IS NULL OR signal_type = :signal_type)
                  AND created_at > NOW() - INTERVAL '7 days'
            """)

            params = {
                'strategy': context.strategy,
                'epic': log_entry.epic,
                'signal_type': context.signal_type
            }

            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                row = result.fetchone()

                if row:
                    return int(row.count)

        except Exception as e:
            self.logger.error(f"Error counting similar signals: {e}")

        return None

    def get_signal_performance_metrics(self, days_back: int = 30) -> SignalPerformanceMetrics:
        """Get comprehensive signal performance metrics"""
        try:
            query = text("""
                SELECT
                    COUNT(*) as total_signals,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(DISTINCT epic) as unique_pairs,
                    MODE() WITHIN GROUP (ORDER BY timeframe) as most_common_timeframe,
                    MODE() WITHIN GROUP (ORDER BY strategy) as most_common_strategy
                FROM alert_history
                WHERE created_at > NOW() - INTERVAL :days_back DAY
            """)

            params = {'days_back': days_back}

            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                row = result.fetchone()

                if row:
                    return SignalPerformanceMetrics(
                        total_signals=int(row.total_signals or 0),
                        average_confidence=float(row.avg_confidence or 0.0),
                        best_timeframe=row.most_common_timeframe,
                        best_strategy=row.most_common_strategy
                    )

        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")

        return SignalPerformanceMetrics()

    def search_signals(self,
                      strategy: str = None,
                      epic: str = None,
                      signal_type: str = None,
                      min_confidence: float = None,
                      max_confidence: float = None,
                      hours_back: int = 24,
                      include_logs: bool = True) -> List[Dict[str, Any]]:
        """Advanced search for signals with optional log correlation"""
        try:
            # Build dynamic query based on search criteria
            conditions = ["created_at > NOW() - INTERVAL :hours_back HOUR"]
            params = {'hours_back': hours_back}

            if strategy:
                conditions.append("strategy = :strategy")
                params['strategy'] = strategy

            if epic:
                conditions.append("epic = :epic")
                params['epic'] = epic

            if signal_type:
                conditions.append("signal_type = :signal_type")
                params['signal_type'] = signal_type

            if min_confidence is not None:
                conditions.append("confidence_score >= :min_confidence")
                params['min_confidence'] = min_confidence

            if max_confidence is not None:
                conditions.append("confidence_score <= :max_confidence")
                params['max_confidence'] = max_confidence

            where_clause = " AND ".join(conditions)

            query = text(f"""
                SELECT *
                FROM alert_history
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 100
            """)

            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                signals = [dict(row._mapping) for row in result.fetchall()]

            # Optionally correlate with logs
            if include_logs and signals:
                log_entries = self.log_parser.get_signal_intelligence(hours_back=hours_back)

                # Simple correlation by timestamp and epic
                for signal in signals:
                    signal['correlated_logs'] = []
                    signal_time = signal['created_at']

                    for log_entry in log_entries:
                        time_diff = abs((log_entry.timestamp - signal_time).total_seconds())
                        if (time_diff <= 300 and  # 5-minute window
                            log_entry.epic == signal['epic']):
                            signal['correlated_logs'].append({
                                'timestamp': log_entry.timestamp.isoformat(),
                                'level': log_entry.level,
                                'message': log_entry.message,
                                'alert_type': log_entry.alert_type
                            })

            return signals

        except Exception as e:
            self.logger.error(f"Error searching signals: {e}")
            return []

def get_log_database_correlator() -> LogDatabaseCorrelator:
    """Get a configured log database correlator instance"""
    return LogDatabaseCorrelator()

if __name__ == "__main__":
    # Test the correlator
    correlator = LogDatabaseCorrelator()

    print("Testing log database correlator...")

    # Get enriched signal intelligence
    enriched_signals = correlator.get_enriched_signal_intelligence(hours_back=4)
    print(f"\nFound {len(enriched_signals)} enriched signal entries:")

    for signal in enriched_signals[:3]:
        log = signal.log_entry
        print(f"\n  Signal: {log.timestamp} - {log.message[:60]}...")
        if signal.alert_history_match:
            print(f"    DB Match: ID {signal.alert_history_match['id']} - "
                  f"Confidence {signal.alert_history_match['confidence_score']}")
        if signal.success_probability:
            print(f"    Success Probability: {signal.success_probability:.1%}")
        if signal.similar_signals_count:
            print(f"    Similar Signals (7d): {signal.similar_signals_count}")

    # Get performance metrics
    metrics = correlator.get_signal_performance_metrics()
    print(f"\nPerformance Metrics (30d):")
    print(f"  Total Signals: {metrics.total_signals}")
    print(f"  Average Confidence: {metrics.average_confidence:.1%}")
    print(f"  Best Strategy: {metrics.best_strategy}")
    print(f"  Best Timeframe: {metrics.best_timeframe}")