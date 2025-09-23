# core/intelligence/market_intelligence_history_manager.py
"""
Market Intelligence History Manager

Manages storage and retrieval of market intelligence data for each scan cycle.
Provides comprehensive market intelligence analytics independent of signal generation.

Key Features:
- Store market intelligence for every scan cycle
- Efficient querying and analytics
- Historical market regime analysis
- Performance optimization with indexed fields
- Strategy performance correlation with market conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import uuid
from psycopg2.extras import RealDictCursor

try:
    from utils.scanner_utils import make_json_serializable
except ImportError:
    from forex_scanner.utils.scanner_utils import make_json_serializable

try:
    import config
except ImportError:
    from forex_scanner import config


class MarketIntelligenceHistoryManager:
    """
    Manages market intelligence history storage and analytics

    Responsibilities:
    - Store market intelligence data for each scan cycle
    - Provide efficient querying capabilities
    - Generate historical analytics and trends
    - Support performance analysis and optimization
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
        self._initialize_table()

    def _get_connection(self):
        """Get database connection through injected DatabaseManager"""
        return self.db_manager.get_connection()

    def _execute_with_connection(self, operation_func, operation_name="database operation"):
        """
        Execute database operation with proper connection management

        Args:
            operation_func: Function that takes (conn, cursor) and performs the operation
            operation_name: Name for logging purposes

        Returns:
            Result from operation_func or None on error
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            result = operation_func(conn, cursor)
            conn.commit()
            return result

        except Exception as e:
            self.logger.error(f"❌ {operation_name} failed: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise

        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def _initialize_table(self):
        """Initialize market intelligence history table if it doesn't exist"""
        def check_and_create_table_operation(conn, cursor):
            # First check if table already exists
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = 'market_intelligence_history'
            """)

            table_exists = cursor.fetchone()[0] > 0

            if table_exists:
                self.logger.info("✅ Market intelligence history table already exists")
                return

            # Table doesn't exist, create it
            self.logger.info("🔧 Creating market intelligence history table...")

            # Read SQL schema from migrations file
            import os
            schema_path = os.path.join(
                os.path.dirname(__file__),
                '../../migrations/create_market_intelligence_history_table_simple.sql'
            )

            try:
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()

                # Execute the schema
                cursor.execute(schema_sql)
                self.logger.info("✅ Market intelligence history table created successfully")

            except FileNotFoundError:
                self.logger.warning(f"⚠️ Schema file not found at {schema_path}, creating minimal table")
                self._create_minimal_table(cursor)

        try:
            self._execute_with_connection(check_and_create_table_operation, "table initialization")
        except Exception as e:
            self.logger.error(f"Failed to initialize market intelligence history table: {e}")
            raise

    def _create_minimal_table(self, cursor):
        """Create minimal table structure if schema file is not available"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_intelligence_history (
                id SERIAL PRIMARY KEY,
                scan_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                scan_cycle_id VARCHAR(64),
                epic_list TEXT[],
                epic_count INTEGER DEFAULT 0,
                dominant_regime VARCHAR(30) NOT NULL,
                regime_confidence DECIMAL(5,4) NOT NULL DEFAULT 0.5,
                regime_scores JSON,
                current_session VARCHAR(20) NOT NULL,
                session_volatility VARCHAR(20),
                market_bias VARCHAR(20),
                intelligence_source VARCHAR(50) DEFAULT 'MarketIntelligenceEngine',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create basic indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_mi_hist_timestamp ON market_intelligence_history(scan_timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_mi_hist_regime ON market_intelligence_history(dominant_regime)')

    def save_market_intelligence(self, intelligence_report: Dict, epic_list: List[str],
                                scan_cycle_id: Optional[str] = None) -> Optional[int]:
        """
        Save comprehensive market intelligence data from a scan cycle

        Args:
            intelligence_report: Market intelligence report from MarketIntelligenceEngine
            epic_list: List of epics analyzed in this scan
            scan_cycle_id: Optional unique identifier for this scan cycle

        Returns:
            Record ID if successful, None if failed
        """
        try:
            if not isinstance(intelligence_report, dict):
                self.logger.error(f"❌ Intelligence report must be a dictionary, got {type(intelligence_report)}")
                return None

            # Generate scan cycle ID if not provided
            if not scan_cycle_id:
                scan_cycle_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            def save_operation(conn, cursor):
                # Extract data from intelligence report
                data = self._extract_intelligence_data(intelligence_report, epic_list, scan_cycle_id)

                # Insert into database
                insert_query = '''
                    INSERT INTO market_intelligence_history (
                        scan_timestamp, scan_cycle_id, epic_list, epic_count,
                        dominant_regime, regime_confidence, regime_scores,
                        current_session, session_volatility, session_characteristics, optimal_timeframes,
                        market_bias, average_trend_strength, average_volatility, directional_consensus,
                        market_efficiency, volatility_percentile,
                        correlation_analysis, currency_strength, risk_sentiment,
                        recommended_strategy, confidence_threshold, position_sizing_recommendation,
                        strategy_adjustments, market_strength_summary, pair_analyses,
                        intelligence_source, successful_pair_analyses, failed_pair_analyses,
                        regime_trending_score, regime_ranging_score, regime_breakout_score,
                        regime_reversal_score, regime_high_vol_score, regime_low_vol_score
                    ) VALUES (
                        %(scan_timestamp)s, %(scan_cycle_id)s, %(epic_list)s, %(epic_count)s,
                        %(dominant_regime)s, %(regime_confidence)s, %(regime_scores)s,
                        %(current_session)s, %(session_volatility)s, %(session_characteristics)s, %(optimal_timeframes)s,
                        %(market_bias)s, %(average_trend_strength)s, %(average_volatility)s, %(directional_consensus)s,
                        %(market_efficiency)s, %(volatility_percentile)s,
                        %(correlation_analysis)s, %(currency_strength)s, %(risk_sentiment)s,
                        %(recommended_strategy)s, %(confidence_threshold)s, %(position_sizing_recommendation)s,
                        %(strategy_adjustments)s, %(market_strength_summary)s, %(pair_analyses)s,
                        %(intelligence_source)s, %(successful_pair_analyses)s, %(failed_pair_analyses)s,
                        %(regime_trending_score)s, %(regime_ranging_score)s, %(regime_breakout_score)s,
                        %(regime_reversal_score)s, %(regime_high_vol_score)s, %(regime_low_vol_score)s
                    ) RETURNING id
                '''

                cursor.execute(insert_query, data)
                record_id = cursor.fetchone()[0]

                regime = data.get('dominant_regime', 'unknown')
                confidence = data.get('regime_confidence', 0.5)
                session = data.get('current_session', 'unknown')
                epic_count = data.get('epic_count', 0)

                # Generate individual epic regime breakdown
                pair_analyses = data.get('pair_analyses', {})
                if isinstance(pair_analyses, str):
                    import json
                    try:
                        pair_analyses = json.loads(pair_analyses)
                    except:
                        pair_analyses = {}

                epic_breakdown = []
                for epic, analysis in pair_analyses.items():
                    if isinstance(analysis, dict) and 'regime_scores' in analysis:
                        regime_scores = analysis['regime_scores']
                        # Determine dominant regime for this epic
                        epic_regime = max(regime_scores, key=regime_scores.get) if regime_scores else 'unknown'
                        epic_confidence = regime_scores.get(epic_regime, 0.5) if regime_scores else 0.5

                        # Extract clean epic name
                        clean_epic = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
                        epic_breakdown.append(f"{clean_epic}({epic_regime[:4]},{epic_confidence:.1%})")

                self.logger.info(f"✅ Saved market intelligence #{record_id}: {regime} regime ({confidence:.1%}) "
                               f"during {session} session - {epic_count} epics analyzed")

                # Log individual epic breakdown if available
                if epic_breakdown:
                    breakdown_str = ", ".join(epic_breakdown)
                    self.logger.info(f"📊 Epic breakdown: {breakdown_str}")

                return record_id

            return self._execute_with_connection(save_operation, "save market intelligence")

        except Exception as e:
            self.logger.error(f"❌ Error saving market intelligence: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return None

    def _extract_intelligence_data(self, intelligence_report: Dict, epic_list: List[str],
                                 scan_cycle_id: str) -> Dict:
        """Extract and format intelligence data for database insertion"""
        try:
            # Basic metadata
            data = {
                'scan_timestamp': datetime.now(timezone.utc),
                'scan_cycle_id': scan_cycle_id,
                'epic_list': epic_list,
                'epic_count': len(epic_list),
                'intelligence_source': 'MarketIntelligenceEngine'
            }

            # Market regime analysis
            market_regime = intelligence_report.get('market_regime', {})
            data.update({
                'dominant_regime': market_regime.get('dominant_regime', 'unknown'),
                'regime_confidence': float(market_regime.get('confidence', 0.5)),
                'regime_scores': self._safe_json_serialize(market_regime.get('regime_scores', {}))
            })

            # Extract individual regime scores for indexing
            regime_scores = market_regime.get('regime_scores', {})
            data.update({
                'regime_trending_score': float(regime_scores.get('trending', 0.5)),
                'regime_ranging_score': float(regime_scores.get('ranging', 0.5)),
                'regime_breakout_score': float(regime_scores.get('breakout', 0.3)),
                'regime_reversal_score': float(regime_scores.get('reversal', 0.3)),
                'regime_high_vol_score': float(regime_scores.get('high_volatility', 0.4)),
                'regime_low_vol_score': float(regime_scores.get('low_volatility', 0.6))
            })

            # Session analysis
            session_analysis = intelligence_report.get('session_analysis', {})
            session_config = session_analysis.get('session_config', {})
            data.update({
                'current_session': session_analysis.get('current_session', 'unknown'),
                'session_volatility': session_config.get('volatility', 'medium'),
                'session_characteristics': self._safe_array(session_config.get('characteristics', [])),
                'optimal_timeframes': self._safe_array(session_analysis.get('optimal_timeframes', ['15m']))
            })

            # Market strength and context
            market_strength = market_regime.get('market_strength', {})
            data.update({
                'market_bias': market_strength.get('market_bias', 'neutral'),
                'average_trend_strength': self._safe_float(market_strength.get('average_trend_strength')),
                'average_volatility': self._safe_float(market_strength.get('average_volatility')),
                'directional_consensus': self._safe_float(market_strength.get('directional_consensus')),
                'market_efficiency': self._safe_float(market_strength.get('market_efficiency')),
                'volatility_percentile': self._safe_float(intelligence_report.get('volatility_percentile', 50.0))
            })

            # Correlation analysis
            correlation_analysis = market_regime.get('correlation_analysis', {})
            data.update({
                'correlation_analysis': self._safe_json_serialize(correlation_analysis),
                'currency_strength': self._safe_json_serialize(correlation_analysis.get('currency_strength', {})),
                'risk_sentiment': correlation_analysis.get('risk_on_off', 'neutral')
            })

            # Strategy recommendations
            recommended_strategy = market_regime.get('recommended_strategy', {})
            trading_recommendations = intelligence_report.get('trading_recommendations', {})
            data.update({
                'recommended_strategy': recommended_strategy.get('strategy', 'conservative'),
                'confidence_threshold': self._safe_float(trading_recommendations.get('confidence_threshold', 0.7)),
                'position_sizing_recommendation': trading_recommendations.get('position_sizing', 'NORMAL'),
                'strategy_adjustments': trading_recommendations.get('strategy_adjustments', 'Standard approach')
            })

            # Complex analysis data
            data.update({
                'market_strength_summary': self._safe_json_serialize(market_strength),
                'pair_analyses': self._safe_json_serialize(market_regime.get('pair_analyses', {}))
            })

            # Analysis metadata and individual epic regimes
            pair_analyses = market_regime.get('pair_analyses', {})
            data.update({
                'successful_pair_analyses': len([a for a in pair_analyses.values() if a.get('current_price')]),
                'failed_pair_analyses': len(pair_analyses) - len([a for a in pair_analyses.values() if a.get('current_price')])
            })

            # Extract individual epic regimes for easier querying
            individual_epic_regimes = {}
            for epic, analysis in pair_analyses.items():
                if isinstance(analysis, dict) and 'regime_scores' in analysis:
                    regime_scores = analysis['regime_scores']
                    epic_regime = max(regime_scores, key=regime_scores.get) if regime_scores else 'unknown'
                    epic_confidence = regime_scores.get(epic_regime, 0.5) if regime_scores else 0.5

                    # Clean epic name for consistent key
                    clean_epic = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
                    individual_epic_regimes[clean_epic] = {
                        'regime': epic_regime,
                        'confidence': float(epic_confidence),
                        'regime_scores': regime_scores
                    }

            data['individual_epic_regimes'] = self._safe_json_serialize(individual_epic_regimes)

            return data

        except Exception as e:
            self.logger.error(f"❌ Error extracting intelligence data: {e}")
            # Return minimal data to avoid complete failure
            return {
                'scan_timestamp': datetime.now(timezone.utc),
                'scan_cycle_id': scan_cycle_id,
                'epic_list': epic_list,
                'epic_count': len(epic_list),
                'dominant_regime': 'unknown',
                'regime_confidence': 0.5,
                'current_session': 'unknown',
                'intelligence_source': 'MarketIntelligenceEngine'
            }

    def _safe_json_serialize(self, obj) -> Optional[str]:
        """Safely serialize object to JSON string"""
        if obj is None:
            return None
        try:
            cleaned_obj = make_json_serializable(obj)
            return json.dumps(cleaned_obj) if cleaned_obj else None
        except (TypeError, ValueError):
            return json.dumps(str(obj)) if obj else None

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            if isinstance(value, str) and value.lower() in ['nan', 'inf', '-inf']:
                return None
            result = float(value)
            return result if not (np.isnan(result) or np.isinf(result)) else None
        except (TypeError, ValueError):
            return None

    def _safe_array(self, value) -> List[str]:
        """Safely convert value to array format for PostgreSQL"""
        if value is None:
            return []

        if isinstance(value, list):
            # Already a list, ensure all items are strings
            return [str(item) for item in value]
        elif isinstance(value, str):
            # Single string, convert to single-item array
            return [value]
        else:
            # Other types, convert to string and wrap in array
            return [str(value)]

    def get_recent_intelligence(self, hours: int = 24, limit: int = 100) -> pd.DataFrame:
        """Get recent market intelligence data"""
        def query_operation(conn, cursor):
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            query = '''
                SELECT
                    scan_timestamp, scan_cycle_id, epic_count,
                    dominant_regime, regime_confidence,
                    current_session, session_volatility, market_bias,
                    average_trend_strength, average_volatility,
                    risk_sentiment, recommended_strategy,
                    regime_trending_score, regime_ranging_score,
                    regime_breakout_score, regime_reversal_score
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                ORDER BY scan_timestamp DESC
                LIMIT %s
            '''

            return pd.read_sql_query(query, conn, params=[cutoff_time, limit])

        try:
            return self._execute_with_connection(query_operation, "get recent intelligence")
        except Exception as e:
            self.logger.error(f"Error getting recent intelligence: {e}")
            return pd.DataFrame()

    def get_regime_analysis(self, days: int = 7) -> Dict:
        """Analyze market regime patterns over specified period"""
        def query_operation(conn, cursor):
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

            # Regime distribution
            cursor.execute('''
                SELECT
                    dominant_regime,
                    COUNT(*) as occurrences,
                    AVG(regime_confidence) as avg_confidence,
                    MIN(regime_confidence) as min_confidence,
                    MAX(regime_confidence) as max_confidence
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                GROUP BY dominant_regime
                ORDER BY occurrences DESC
            ''', [cutoff_time])

            regime_distribution = [dict(row) for row in cursor.fetchall()]

            # Session-regime correlation
            cursor.execute('''
                SELECT
                    current_session,
                    dominant_regime,
                    COUNT(*) as occurrences,
                    AVG(regime_confidence) as avg_confidence
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                GROUP BY current_session, dominant_regime
                ORDER BY current_session, occurrences DESC
            ''', [cutoff_time])

            session_regime = [dict(row) for row in cursor.fetchall()]

            # Confidence trends
            cursor.execute('''
                SELECT
                    DATE_TRUNC('hour', scan_timestamp) as hour,
                    AVG(regime_confidence) as avg_confidence,
                    COUNT(*) as scan_count
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                GROUP BY hour
                ORDER BY hour DESC
                LIMIT 48
            ''', [cutoff_time])

            confidence_trends = [dict(row) for row in cursor.fetchall()]

            return {
                'regime_distribution': regime_distribution,
                'session_regime_correlation': session_regime,
                'confidence_trends': confidence_trends,
                'analysis_period_days': days
            }

        try:
            return self._execute_with_connection(query_operation, "get regime analysis")
        except Exception as e:
            self.logger.error(f"Error getting regime analysis: {e}")
            return {}

    def get_session_analytics(self, days: int = 7) -> Dict:
        """Analyze session-based market intelligence patterns"""
        def query_operation(conn, cursor):
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

            cursor.execute('''
                SELECT
                    current_session,
                    session_volatility,
                    market_bias,
                    COUNT(*) as scan_count,
                    AVG(regime_confidence) as avg_confidence,
                    AVG(average_trend_strength) as avg_trend_strength,
                    AVG(average_volatility) as avg_volatility
                FROM market_intelligence_history
                WHERE scan_timestamp > %s
                GROUP BY current_session, session_volatility, market_bias
                ORDER BY current_session, scan_count DESC
            ''', [cutoff_time])

            return [dict(row) for row in cursor.fetchall()]

        try:
            results = self._execute_with_connection(query_operation, "get session analytics")
            return {'session_analytics': results, 'analysis_period_days': days}
        except Exception as e:
            self.logger.error(f"Error getting session analytics: {e}")
            return {}

    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """Clean up old market intelligence records to manage storage"""
        def cleanup_operation(conn, cursor):
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            cursor.execute('''
                DELETE FROM market_intelligence_history
                WHERE scan_timestamp < %s
            ''', [cutoff_time])

            deleted_count = cursor.rowcount
            self.logger.info(f"🧹 Cleaned up {deleted_count} old market intelligence records (older than {days_to_keep} days)")
            return deleted_count

        try:
            return self._execute_with_connection(cleanup_operation, "cleanup old records")
        except Exception as e:
            self.logger.error(f"Error cleaning up old records: {e}")
            return 0


# Backward compatibility exports
__all__ = ['MarketIntelligenceHistoryManager']