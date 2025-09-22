# core/intelligence/market_intelligence_analytics.py
"""
Market Intelligence Analytics Module

Provides comprehensive analytics and reporting capabilities for market intelligence data.
Enables deep analysis of market conditions, regime patterns, and trading performance correlation.

Key Features:
- Historical market regime analysis
- Session-based market condition patterns
- Strategy performance correlation with market intelligence
- Market intelligence trend identification
- Real-time market condition monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from collections import defaultdict

try:
    import config
except ImportError:
    from forex_scanner import config


class MarketIntelligenceAnalytics:
    """
    Advanced analytics for market intelligence data

    Provides comprehensive analysis capabilities for understanding:
    - Market regime patterns and transitions
    - Session-based market behaviors
    - Strategy performance under different market conditions
    - Long-term market intelligence trends
    """

    def __init__(self, db_manager):
        """
        Initialize with injected DatabaseManager

        Args:
            db_manager: DatabaseManager instance for database operations
        """
        if db_manager is None:
            raise ValueError("DatabaseManager is required - cannot be None")

        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def _get_connection(self):
        """Get database connection through injected DatabaseManager"""
        return self.db_manager.get_connection()

    def _execute_query(self, query: str, params: List[Any] = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame"""
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=params or [])
            conn.close()
            return df
        except Exception as e:
            self.logger.error(f"❌ Query execution failed: {e}")
            return pd.DataFrame()

    def get_regime_transition_analysis(self, days: int = 30) -> Dict:
        """
        Analyze market regime transitions over time

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with regime transition patterns and statistics
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            # Get regime data over time
            query = '''
                SELECT
                    scan_timestamp,
                    dominant_regime,
                    regime_confidence,
                    current_session,
                    market_bias,
                    average_trend_strength,
                    LAG(dominant_regime) OVER (ORDER BY scan_timestamp) as prev_regime
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                ORDER BY scan_timestamp
            '''

            df = self._execute_query(query, [cutoff_date])

            if df.empty:
                return {'error': 'No data available'}

            # Calculate regime transitions
            transitions = []
            regime_durations = defaultdict(list)
            current_regime_start = None
            current_regime = None

            for idx, row in df.iterrows():
                regime = row['dominant_regime']
                timestamp = row['scan_timestamp']

                if current_regime != regime:
                    # Regime transition detected
                    if current_regime is not None and current_regime_start is not None:
                        duration = (timestamp - current_regime_start).total_seconds() / 60  # minutes
                        regime_durations[current_regime].append(duration)

                        transitions.append({
                            'from_regime': current_regime,
                            'to_regime': regime,
                            'transition_time': timestamp,
                            'duration_minutes': duration,
                            'session': row['current_session'],
                            'confidence': row['regime_confidence']
                        })

                    current_regime = regime
                    current_regime_start = timestamp

            # Calculate statistics
            regime_stats = {}
            for regime, durations in regime_durations.items():
                regime_stats[regime] = {
                    'count': len(durations),
                    'avg_duration_minutes': np.mean(durations) if durations else 0,
                    'min_duration_minutes': np.min(durations) if durations else 0,
                    'max_duration_minutes': np.max(durations) if durations else 0,
                    'total_time_percentage': sum(durations) / (days * 24 * 60) * 100 if durations else 0
                }

            # Most common transitions
            transition_counts = defaultdict(int)
            for t in transitions:
                key = f"{t['from_regime']} -> {t['to_regime']}"
                transition_counts[key] += 1

            most_common_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                'analysis_period_days': days,
                'total_transitions': len(transitions),
                'regime_statistics': regime_stats,
                'most_common_transitions': most_common_transitions,
                'transitions': transitions,
                'regime_distribution': df['dominant_regime'].value_counts().to_dict()
            }

        except Exception as e:
            self.logger.error(f"❌ Regime transition analysis failed: {e}")
            return {'error': str(e)}

    def get_session_market_patterns(self, days: int = 14) -> Dict:
        """
        Analyze market patterns by trading session

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with session-based market patterns
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            query = '''
                SELECT
                    current_session,
                    session_volatility,
                    dominant_regime,
                    regime_confidence,
                    market_bias,
                    average_trend_strength,
                    average_volatility,
                    risk_sentiment,
                    COUNT(*) as scan_count,
                    AVG(regime_confidence) as avg_confidence,
                    AVG(average_trend_strength) as avg_trend_strength,
                    AVG(average_volatility) as avg_volatility_score
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                GROUP BY current_session, session_volatility, dominant_regime, market_bias, risk_sentiment
                ORDER BY current_session, scan_count DESC
            '''

            df = self._execute_query(query, [cutoff_date])

            if df.empty:
                return {'error': 'No data available'}

            # Organize data by session
            session_patterns = {}
            for session in df['current_session'].unique():
                session_data = df[df['current_session'] == session]

                # Most common regime by session
                regime_distribution = session_data.groupby('dominant_regime')['scan_count'].sum().to_dict()

                # Volatility patterns
                volatility_patterns = session_data.groupby('session_volatility')['scan_count'].sum().to_dict()

                # Market bias patterns
                bias_patterns = session_data.groupby('market_bias')['scan_count'].sum().to_dict()

                # Risk sentiment patterns
                risk_patterns = session_data.groupby('risk_sentiment')['scan_count'].sum().to_dict()

                # Average metrics
                avg_metrics = {
                    'avg_confidence': session_data['avg_confidence'].mean(),
                    'avg_trend_strength': session_data['avg_trend_strength'].mean(),
                    'avg_volatility': session_data['avg_volatility_score'].mean(),
                    'total_scans': session_data['scan_count'].sum()
                }

                session_patterns[session] = {
                    'regime_distribution': regime_distribution,
                    'volatility_patterns': volatility_patterns,
                    'market_bias_patterns': bias_patterns,
                    'risk_sentiment_patterns': risk_patterns,
                    'average_metrics': avg_metrics
                }

            return {
                'analysis_period_days': days,
                'session_patterns': session_patterns,
                'overall_session_distribution': df.groupby('current_session')['scan_count'].sum().to_dict()
            }

        except Exception as e:
            self.logger.error(f"❌ Session pattern analysis failed: {e}")
            return {'error': str(e)}

    def get_market_intelligence_performance_correlation(self, days: int = 30) -> Dict:
        """
        Correlate market intelligence data with signal performance

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with performance correlation analysis
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            # Query to join market intelligence with alert history
            query = '''
                SELECT
                    mi.scan_timestamp,
                    mi.dominant_regime,
                    mi.regime_confidence,
                    mi.current_session,
                    mi.market_bias,
                    mi.risk_sentiment,
                    ah.strategy,
                    ah.signal_type,
                    ah.confidence_score,
                    ah.claude_approved,
                    ah.claude_score,
                    COUNT(*) as signal_count
                FROM market_intelligence_history mi
                LEFT JOIN alert_history ah ON DATE_TRUNC('hour', mi.scan_timestamp) = DATE_TRUNC('hour', ah.alert_timestamp)
                WHERE mi.scan_timestamp > %s
                  AND ah.alert_timestamp IS NOT NULL
                GROUP BY
                    mi.scan_timestamp, mi.dominant_regime, mi.regime_confidence,
                    mi.current_session, mi.market_bias, mi.risk_sentiment,
                    ah.strategy, ah.signal_type, ah.confidence_score,
                    ah.claude_approved, ah.claude_score
                ORDER BY mi.scan_timestamp DESC
            '''

            df = self._execute_query(query, [cutoff_date])

            if df.empty:
                return {'error': 'No correlated data available'}

            # Analyze performance by regime
            regime_performance = {}
            for regime in df['dominant_regime'].unique():
                regime_data = df[df['dominant_regime'] == regime]

                performance = {
                    'total_signals': regime_data['signal_count'].sum(),
                    'avg_signal_confidence': regime_data['confidence_score'].mean(),
                    'avg_regime_confidence': regime_data['regime_confidence'].mean(),
                    'claude_approval_rate': regime_data['claude_approved'].mean() if 'claude_approved' in regime_data.columns else 0,
                    'avg_claude_score': regime_data['claude_score'].mean() if 'claude_score' in regime_data.columns else 0,
                    'strategy_distribution': regime_data.groupby('strategy')['signal_count'].sum().to_dict(),
                    'signal_type_distribution': regime_data.groupby('signal_type')['signal_count'].sum().to_dict()
                }

                regime_performance[regime] = performance

            # Analyze performance by session
            session_performance = {}
            for session in df['current_session'].unique():
                session_data = df[df['current_session'] == session]

                performance = {
                    'total_signals': session_data['signal_count'].sum(),
                    'avg_signal_confidence': session_data['confidence_score'].mean(),
                    'regime_distribution': session_data.groupby('dominant_regime')['signal_count'].sum().to_dict(),
                    'strategy_distribution': session_data.groupby('strategy')['signal_count'].sum().to_dict()
                }

                session_performance[session] = performance

            # Market bias correlation
            bias_performance = {}
            for bias in df['market_bias'].unique():
                bias_data = df[df['market_bias'] == bias]

                performance = {
                    'total_signals': bias_data['signal_count'].sum(),
                    'avg_signal_confidence': bias_data['confidence_score'].mean(),
                    'signal_type_distribution': bias_data.groupby('signal_type')['signal_count'].sum().to_dict()
                }

                bias_performance[bias] = performance

            return {
                'analysis_period_days': days,
                'regime_performance': regime_performance,
                'session_performance': session_performance,
                'market_bias_performance': bias_performance,
                'total_analyzed_signals': df['signal_count'].sum(),
                'correlation_insights': self._generate_correlation_insights(df)
            }

        except Exception as e:
            self.logger.error(f"❌ Performance correlation analysis failed: {e}")
            return {'error': str(e)}

    def _generate_correlation_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate insights from correlation analysis"""
        insights = []

        try:
            # Best performing regime
            regime_performance = df.groupby('dominant_regime').agg({
                'confidence_score': 'mean',
                'signal_count': 'sum'
            }).reset_index()

            if not regime_performance.empty:
                best_regime = regime_performance.loc[regime_performance['confidence_score'].idxmax()]
                insights.append(f"Best performing regime: {best_regime['dominant_regime']} "
                              f"(avg confidence: {best_regime['confidence_score']:.1%})")

            # Most productive session
            session_performance = df.groupby('current_session')['signal_count'].sum()
            if not session_performance.empty:
                best_session = session_performance.idxmax()
                insights.append(f"Most productive session: {best_session} "
                              f"({session_performance[best_session]} signals)")

            # Market bias effectiveness
            bias_performance = df.groupby('market_bias')['confidence_score'].mean()
            if not bias_performance.empty:
                best_bias = bias_performance.idxmax()
                insights.append(f"Highest confidence market bias: {best_bias} "
                              f"({bias_performance[best_bias]:.1%})")

        except Exception as e:
            self.logger.debug(f"Insight generation error: {e}")

        return insights

    def get_market_intelligence_summary_dashboard(self, hours: int = 24) -> Dict:
        """
        Generate a comprehensive dashboard summary of recent market intelligence

        Args:
            hours: Number of hours to analyze for dashboard

        Returns:
            Dictionary with dashboard data for visualization
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Basic statistics
            stats_query = '''
                SELECT
                    COUNT(*) as total_scans,
                    COUNT(DISTINCT dominant_regime) as unique_regimes,
                    COUNT(DISTINCT current_session) as unique_sessions,
                    AVG(regime_confidence) as avg_confidence,
                    MAX(regime_confidence) as max_confidence,
                    MIN(regime_confidence) as min_confidence
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
            '''

            stats_df = self._execute_query(stats_query, [cutoff_time])

            # Current market state (most recent)
            current_state_query = '''
                SELECT
                    dominant_regime,
                    regime_confidence,
                    current_session,
                    market_bias,
                    risk_sentiment,
                    scan_timestamp
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                ORDER BY scan_timestamp DESC
                LIMIT 1
            '''

            current_df = self._execute_query(current_state_query, [cutoff_time])

            # Regime distribution
            regime_dist_query = '''
                SELECT
                    dominant_regime,
                    COUNT(*) as count,
                    AVG(regime_confidence) as avg_confidence
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                GROUP BY dominant_regime
                ORDER BY count DESC
            '''

            regime_df = self._execute_query(regime_dist_query, [cutoff_time])

            # Hourly confidence trend
            trend_query = '''
                SELECT
                    DATE_TRUNC('hour', scan_timestamp) as hour,
                    AVG(regime_confidence) as avg_confidence,
                    COUNT(*) as scan_count,
                    MODE() WITHIN GROUP (ORDER BY dominant_regime) as most_common_regime
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                GROUP BY hour
                ORDER BY hour
            '''

            trend_df = self._execute_query(trend_query, [cutoff_time])

            # Compile dashboard data
            dashboard = {
                'time_period_hours': hours,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'summary_statistics': stats_df.iloc[0].to_dict() if not stats_df.empty else {},
                'current_market_state': current_df.iloc[0].to_dict() if not current_df.empty else {},
                'regime_distribution': regime_df.to_dict('records'),
                'confidence_trend': trend_df.to_dict('records'),
                'health_indicators': self._calculate_health_indicators(cutoff_time)
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"❌ Dashboard generation failed: {e}")
            return {'error': str(e)}

    def _calculate_health_indicators(self, cutoff_time: datetime) -> Dict:
        """Calculate system health indicators for market intelligence"""
        try:
            health_query = '''
                SELECT
                    COUNT(*) as total_records,
                    COUNT(*) / EXTRACT(EPOCH FROM (MAX(scan_timestamp) - MIN(scan_timestamp))) * 3600 as records_per_hour,
                    AVG(CASE WHEN regime_confidence > 0.8 THEN 1 ELSE 0 END) as high_confidence_rate,
                    COUNT(DISTINCT scan_cycle_id) as unique_scan_cycles,
                    AVG(epic_count) as avg_epics_analyzed
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
            '''

            health_df = self._execute_query(health_query, [cutoff_time])

            if health_df.empty:
                return {'status': 'no_data'}

            health_data = health_df.iloc[0].to_dict()

            # Determine overall health status
            records_per_hour = health_data.get('records_per_hour', 0)
            high_conf_rate = health_data.get('high_confidence_rate', 0)

            if records_per_hour > 1 and high_conf_rate > 0.3:
                status = 'healthy'
            elif records_per_hour > 0.5:
                status = 'warning'
            else:
                status = 'critical'

            health_data['status'] = status
            return health_data

        except Exception as e:
            self.logger.error(f"❌ Health indicator calculation failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def export_market_intelligence_data(self, start_date: datetime, end_date: datetime,
                                      format: str = 'csv') -> Optional[str]:
        """
        Export market intelligence data for external analysis

        Args:
            start_date: Start date for export
            end_date: End date for export
            format: Export format ('csv', 'json', 'excel')

        Returns:
            Path to exported file or None if failed
        """
        try:
            query = '''
                SELECT
                    scan_timestamp,
                    scan_cycle_id,
                    epic_count,
                    dominant_regime,
                    regime_confidence,
                    current_session,
                    session_volatility,
                    market_bias,
                    average_trend_strength,
                    average_volatility,
                    risk_sentiment,
                    recommended_strategy,
                    confidence_threshold,
                    position_sizing_recommendation
                FROM market_intelligence_history
                WHERE scan_timestamp BETWEEN %s AND %s
                ORDER BY scan_timestamp
            '''

            df = self._execute_query(query, [start_date, end_date])

            if df.empty:
                self.logger.warning("No data to export")
                return None

            # Generate filename
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"market_intelligence_{start_str}_to_{end_str}_{timestamp}"

            # Export based on format
            import os
            export_dir = "/tmp"  # You may want to make this configurable

            if format.lower() == 'csv':
                filepath = os.path.join(export_dir, f"{filename}.csv")
                df.to_csv(filepath, index=False)

            elif format.lower() == 'json':
                filepath = os.path.join(export_dir, f"{filename}.json")
                df.to_json(filepath, orient='records', date_format='iso', indent=2)

            elif format.lower() == 'excel':
                filepath = os.path.join(export_dir, f"{filename}.xlsx")
                df.to_excel(filepath, index=False, sheet_name='Market Intelligence')

            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None

            self.logger.info(f"✅ Exported {len(df)} records to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}")
            return None


# Convenience functions for quick analysis
def quick_regime_analysis(db_manager, days: int = 7) -> Dict:
    """Quick analysis of recent market regimes"""
    analytics = MarketIntelligenceAnalytics(db_manager)
    return analytics.get_regime_transition_analysis(days)


def quick_session_analysis(db_manager, days: int = 7) -> Dict:
    """Quick analysis of session patterns"""
    analytics = MarketIntelligenceAnalytics(db_manager)
    return analytics.get_session_market_patterns(days)


def quick_dashboard(db_manager, hours: int = 24) -> Dict:
    """Quick dashboard for recent market intelligence"""
    analytics = MarketIntelligenceAnalytics(db_manager)
    return analytics.get_market_intelligence_summary_dashboard(hours)


# Backward compatibility exports
__all__ = ['MarketIntelligenceAnalytics', 'quick_regime_analysis', 'quick_session_analysis', 'quick_dashboard']