"""
SMC Rejection History Manager

Manages storage and retrieval of SMC Simple strategy rejections for analysis.
Stores rejection data with full market context to enable strategy improvement.

Created: 2025-12-18
"""

import json
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    from utils.scanner_utils import make_json_serializable
except ImportError:
    try:
        from forex_scanner.utils.scanner_utils import make_json_serializable
    except ImportError:
        def make_json_serializable(obj):
            """Fallback serialization"""
            if obj is None:
                return None
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            return obj


class SMCRejectionHistoryManager:
    """
    Manages storage and retrieval of SMC Simple strategy rejections.

    Features:
    - Batch insert support for performance
    - Full market context capture
    - Query methods for dashboard integration
    - Statistics aggregation
    """

    # Rejection stage constants
    STAGE_SESSION = 'SESSION'
    STAGE_COOLDOWN = 'COOLDOWN'
    STAGE_TIER1_EMA = 'TIER1_EMA'
    STAGE_TIER2_SWING = 'TIER2_SWING'
    STAGE_TIER3_PULLBACK = 'TIER3_PULLBACK'
    STAGE_RISK_LIMIT = 'RISK_LIMIT'
    STAGE_RISK_RR = 'RISK_RR'
    STAGE_RISK_TP = 'RISK_TP'
    STAGE_CONFIDENCE = 'CONFIDENCE'
    STAGE_SR_PATH_BLOCKED = 'SR_PATH_BLOCKED'  # S/R blocking path to target
    STAGE_SMC_CONFLICT = 'SMC_CONFLICT'  # SMC data conflicts with signal direction
    # v2.9.0: New rejection stages for volume/confidence filters
    STAGE_CONFIDENCE_CAP = 'CONFIDENCE_CAP'  # Confidence exceeds maximum threshold
    STAGE_VOLUME_LOW = 'VOLUME_LOW'  # Volume ratio below minimum threshold
    STAGE_VOLUME_NO_DATA = 'VOLUME_NO_DATA'  # No volume data available
    # v2.10.0: MACD momentum alignment filter
    STAGE_MACD_MISALIGNED = 'MACD_MISALIGNED'  # Trade direction against MACD momentum

    VALID_STAGES = [
        STAGE_SESSION, STAGE_COOLDOWN, STAGE_TIER1_EMA, STAGE_TIER2_SWING,
        STAGE_TIER3_PULLBACK, STAGE_RISK_LIMIT, STAGE_RISK_RR, STAGE_RISK_TP,
        STAGE_CONFIDENCE, STAGE_SR_PATH_BLOCKED, STAGE_SMC_CONFLICT,
        STAGE_CONFIDENCE_CAP, STAGE_VOLUME_LOW, STAGE_VOLUME_NO_DATA,
        STAGE_MACD_MISALIGNED
    ]

    def __init__(self, db_manager, config=None):
        """
        Initialize with database manager and optional config.

        Args:
            db_manager: DatabaseManager instance for database operations
            config: Strategy config module (optional, will import if not provided)
        """
        if db_manager is None:
            raise ValueError("DatabaseManager is required - cannot be None")

        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

        # Load config
        if config is None:
            try:
                from forex_scanner.configdata.strategies import config_smc_simple as config
            except ImportError:
                try:
                    from configdata.strategies import config_smc_simple as config
                except ImportError:
                    config = None

        self.config = config
        self.enabled = getattr(config, 'REJECTION_TRACKING_ENABLED', True) if config else True
        self.batch_size = getattr(config, 'REJECTION_BATCH_SIZE', 50) if config else 50
        self.log_to_console = getattr(config, 'REJECTION_LOG_TO_CONSOLE', False) if config else False

        # Batch buffer
        self._batch_buffer = []

        self.logger.info(f"SMCRejectionHistoryManager initialized (enabled={self.enabled})")

    def _get_connection(self):
        """Get database connection through injected DatabaseManager."""
        return self.db_manager.get_connection()

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            import numpy as np
            if isinstance(value, (np.integer, np.floating)):
                return float(value)
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value) -> Optional[int]:
        """Safely convert value to int."""
        if value is None:
            return None
        try:
            import numpy as np
            if isinstance(value, (np.integer, np.floating)):
                return int(value)
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_json(self, value) -> Optional[str]:
        """Safely convert dict/list to JSON string."""
        if value is None:
            return None
        try:
            cleaned = make_json_serializable(value)
            return json.dumps(cleaned)
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Failed to serialize to JSON: {e}")
            return None

    def _get_session_name(self, hour: int) -> str:
        """Determine market session from UTC hour."""
        if hour is None:
            return 'unknown'
        if 7 <= hour < 12:
            return 'london'
        elif 12 <= hour < 16:
            return 'overlap'
        elif 16 <= hour < 21:
            return 'new_york'
        else:
            return 'asian'

    def save_rejection(self, rejection_data: Dict[str, Any]) -> bool:
        """
        Save a single rejection with full market context.

        Args:
            rejection_data: Dictionary containing rejection details and market state

        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enabled:
            return False

        # Validate required fields
        if 'rejection_stage' not in rejection_data:
            self.logger.warning("Missing rejection_stage in rejection_data")
            return False

        if rejection_data['rejection_stage'] not in self.VALID_STAGES:
            self.logger.warning(f"Invalid rejection_stage: {rejection_data['rejection_stage']}")
            return False

        # Log to console if enabled
        if self.log_to_console:
            self.logger.info(
                f"REJECTION [{rejection_data.get('rejection_stage')}] "
                f"{rejection_data.get('epic', 'UNKNOWN')}: {rejection_data.get('rejection_reason', '')}"
            )

        # Add to batch buffer
        self._batch_buffer.append(rejection_data)

        # Flush if batch is full
        if len(self._batch_buffer) >= self.batch_size:
            return self._flush_batch()

        return True

    def _flush_batch(self) -> bool:
        """Flush the batch buffer to database."""
        if not self._batch_buffer:
            return True

        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            insert_sql = """
                INSERT INTO smc_simple_rejections (
                    scan_timestamp, epic, pair,
                    rejection_stage, rejection_reason, rejection_details,
                    attempted_direction,
                    current_price, bid_price, ask_price, spread_pips,
                    market_hour, market_session, is_market_hours,
                    ema_4h_value, ema_distance_pips, price_position_vs_ema,
                    atr_15m, atr_5m, atr_percentile,
                    current_volume, volume_sma, volume_ratio,
                    swing_high_level, swing_low_level, swing_lookback_bars,
                    swings_found_count, last_swing_bars_ago,
                    pullback_depth, fib_zone, swing_range_pips,
                    potential_entry, potential_stop_loss, potential_take_profit,
                    potential_risk_pips, potential_reward_pips, potential_rr_ratio,
                    confidence_score, confidence_breakdown,
                    candle_5m_open, candle_5m_high, candle_5m_low, candle_5m_close, candle_5m_volume,
                    candle_15m_open, candle_15m_high, candle_15m_low, candle_15m_close, candle_15m_volume,
                    candle_4h_open, candle_4h_high, candle_4h_low, candle_4h_close, candle_4h_volume,
                    strategy_version, strategy_config_hash, strategy_config,
                    macd_line, macd_signal, macd_histogram, macd_aligned, macd_momentum
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
            """

            for data in self._batch_buffer:
                # Extract market hour and determine session
                market_hour = self._safe_int(data.get('market_hour'))
                market_session = data.get('market_session') or self._get_session_name(market_hour)

                values = (
                    data.get('scan_timestamp', datetime.now(timezone.utc)),
                    data.get('epic', ''),
                    data.get('pair', ''),
                    data.get('rejection_stage'),
                    data.get('rejection_reason', ''),
                    self._safe_json(data.get('rejection_details')),
                    data.get('attempted_direction'),
                    self._safe_float(data.get('current_price')),
                    self._safe_float(data.get('bid_price')),
                    self._safe_float(data.get('ask_price')),
                    self._safe_float(data.get('spread_pips')),
                    market_hour,
                    market_session,
                    data.get('is_market_hours'),
                    self._safe_float(data.get('ema_4h_value')),
                    self._safe_float(data.get('ema_distance_pips')),
                    data.get('price_position_vs_ema'),
                    self._safe_float(data.get('atr_15m')),
                    self._safe_float(data.get('atr_5m')),
                    self._safe_float(data.get('atr_percentile')),
                    self._safe_float(data.get('current_volume')),
                    self._safe_float(data.get('volume_sma')),
                    self._safe_float(data.get('volume_ratio')),
                    self._safe_float(data.get('swing_high_level')),
                    self._safe_float(data.get('swing_low_level')),
                    self._safe_int(data.get('swing_lookback_bars')),
                    self._safe_int(data.get('swings_found_count')),
                    self._safe_int(data.get('last_swing_bars_ago')),
                    self._safe_float(data.get('pullback_depth')),
                    data.get('fib_zone'),
                    self._safe_float(data.get('swing_range_pips')),
                    self._safe_float(data.get('potential_entry')),
                    self._safe_float(data.get('potential_stop_loss')),
                    self._safe_float(data.get('potential_take_profit')),
                    self._safe_float(data.get('potential_risk_pips')),
                    self._safe_float(data.get('potential_reward_pips')),
                    self._safe_float(data.get('potential_rr_ratio')),
                    self._safe_float(data.get('confidence_score')),
                    self._safe_json(data.get('confidence_breakdown')),
                    self._safe_float(data.get('candle_5m_open')),
                    self._safe_float(data.get('candle_5m_high')),
                    self._safe_float(data.get('candle_5m_low')),
                    self._safe_float(data.get('candle_5m_close')),
                    self._safe_float(data.get('candle_5m_volume')),
                    self._safe_float(data.get('candle_15m_open')),
                    self._safe_float(data.get('candle_15m_high')),
                    self._safe_float(data.get('candle_15m_low')),
                    self._safe_float(data.get('candle_15m_close')),
                    self._safe_float(data.get('candle_15m_volume')),
                    self._safe_float(data.get('candle_4h_open')),
                    self._safe_float(data.get('candle_4h_high')),
                    self._safe_float(data.get('candle_4h_low')),
                    self._safe_float(data.get('candle_4h_close')),
                    self._safe_float(data.get('candle_4h_volume')),
                    data.get('strategy_version'),
                    data.get('strategy_config_hash'),
                    self._safe_json(data.get('strategy_config')),
                    # v2.10.0: MACD data
                    self._safe_float(data.get('macd_line')),
                    self._safe_float(data.get('macd_signal')),
                    self._safe_float(data.get('macd_histogram')),
                    data.get('macd_aligned'),
                    data.get('macd_momentum')
                )

                cursor.execute(insert_sql, values)

            conn.commit()
            count = len(self._batch_buffer)
            self._batch_buffer = []
            self.logger.debug(f"Flushed {count} rejections to database")
            return True

        except Exception as e:
            self.logger.error(f"Failed to flush rejection batch: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return False

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

    def flush(self) -> bool:
        """Force flush any pending rejections in the buffer."""
        return self._flush_batch()

    def get_rejection_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get aggregated rejection statistics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with rejection statistics
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_rejections,
                    COUNT(DISTINCT epic) as unique_pairs,
                    COUNT(DISTINCT DATE(scan_timestamp)) as days_with_data,
                    MODE() WITHIN GROUP (ORDER BY rejection_stage) as most_common_stage,
                    ROUND(AVG(confidence_score)::numeric, 4) as avg_confidence,
                    COUNT(CASE WHEN rejection_stage = 'CONFIDENCE' THEN 1 END) as near_misses,
                    ROUND(AVG(atr_percentile)::numeric, 2) as avg_atr_percentile
                FROM smc_simple_rejections
                WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            """, (days,))

            row = cursor.fetchone()

            if row:
                return {
                    'total_rejections': row[0] or 0,
                    'unique_pairs': row[1] or 0,
                    'days_with_data': row[2] or 0,
                    'most_common_stage': row[3] or 'N/A',
                    'avg_confidence': float(row[4]) if row[4] else 0.0,
                    'near_misses': row[5] or 0,
                    'avg_atr_percentile': float(row[6]) if row[6] else 0.0,
                    'avg_rejections_per_day': round(row[0] / max(row[2], 1), 1) if row[0] else 0
                }

            return {
                'total_rejections': 0,
                'unique_pairs': 0,
                'days_with_data': 0,
                'most_common_stage': 'N/A',
                'avg_confidence': 0.0,
                'near_misses': 0,
                'avg_atr_percentile': 0.0,
                'avg_rejections_per_day': 0
            }

        except Exception as e:
            self.logger.error(f"Failed to get rejection stats: {e}")
            return {}

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

    def get_rejections_by_stage(self, stage: str = None, days: int = 30,
                                 limit: int = 100) -> List[Dict]:
        """
        Query rejections filtered by stage.

        Args:
            stage: Rejection stage to filter (None for all)
            days: Number of days to look back
            limit: Maximum results to return

        Returns:
            List of rejection records
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if stage:
                cursor.execute("""
                    SELECT * FROM smc_simple_rejections
                    WHERE rejection_stage = %s
                      AND scan_timestamp >= NOW() - INTERVAL '%s days'
                    ORDER BY scan_timestamp DESC
                    LIMIT %s
                """, (stage, days, limit))
            else:
                cursor.execute("""
                    SELECT * FROM smc_simple_rejections
                    WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
                    ORDER BY scan_timestamp DESC
                    LIMIT %s
                """, (days, limit))

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to get rejections by stage: {e}")
            return []

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

    def get_near_misses(self, min_confidence: float = 0.45, days: int = 30,
                        limit: int = 50) -> List[Dict]:
        """
        Get signals that almost passed (high confidence rejects).

        Args:
            min_confidence: Minimum confidence score to consider
            days: Number of days to look back
            limit: Maximum results to return

        Returns:
            List of near-miss rejection records
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    scan_timestamp, epic, pair, attempted_direction,
                    confidence_score, rejection_reason, potential_rr_ratio,
                    market_session, ema_distance_pips, pullback_depth, fib_zone,
                    potential_risk_pips, potential_reward_pips
                FROM smc_simple_rejections
                WHERE rejection_stage = 'CONFIDENCE'
                  AND confidence_score >= %s
                  AND scan_timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY confidence_score DESC
                LIMIT %s
            """, (min_confidence, days, limit))

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to get near misses: {e}")
            return []

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

    def get_rejection_breakdown(self, days: int = 30) -> Dict[str, Any]:
        """
        Get breakdown of rejections by stage, session, and hour.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with breakdown data for dashboard
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            result = {
                'by_stage': [],
                'by_session': [],
                'by_hour': [],
                'by_pair': []
            }

            # By stage
            cursor.execute("""
                SELECT rejection_stage, COUNT(*) as count,
                       ROUND(AVG(confidence_score)::numeric, 4) as avg_confidence
                FROM smc_simple_rejections
                WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY rejection_stage
                ORDER BY count DESC
            """, (days,))
            result['by_stage'] = [
                {'stage': row[0], 'count': row[1], 'avg_confidence': float(row[2]) if row[2] else 0}
                for row in cursor.fetchall()
            ]

            # By session
            cursor.execute("""
                SELECT market_session, rejection_stage, COUNT(*) as count
                FROM smc_simple_rejections
                WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY market_session, rejection_stage
                ORDER BY market_session, count DESC
            """, (days,))
            result['by_session'] = [
                {'session': row[0], 'stage': row[1], 'count': row[2]}
                for row in cursor.fetchall()
            ]

            # By hour
            cursor.execute("""
                SELECT market_hour, COUNT(*) as count
                FROM smc_simple_rejections
                WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY market_hour
                ORDER BY market_hour
            """, (days,))
            result['by_hour'] = [
                {'hour': row[0], 'count': row[1]}
                for row in cursor.fetchall()
            ]

            # By pair
            cursor.execute("""
                SELECT pair, COUNT(*) as count,
                       MODE() WITHIN GROUP (ORDER BY rejection_stage) as top_stage
                FROM smc_simple_rejections
                WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY pair
                ORDER BY count DESC
            """, (days,))
            result['by_pair'] = [
                {'pair': row[0], 'count': row[1], 'top_stage': row[2]}
                for row in cursor.fetchall()
            ]

            return result

        except Exception as e:
            self.logger.error(f"Failed to get rejection breakdown: {e}")
            return {'by_stage': [], 'by_session': [], 'by_hour': [], 'by_pair': []}

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

    def get_rejections_dataframe(self, days: int = 30, pair: str = None,
                                  session: str = None, stage: str = None) -> pd.DataFrame:
        """
        Get rejections as a pandas DataFrame with optional filters.

        Args:
            days: Number of days to look back
            pair: Filter by pair (optional)
            session: Filter by session (optional)
            stage: Filter by stage (optional)

        Returns:
            DataFrame with rejection data
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()

            # Build query with filters
            query = """
                SELECT * FROM smc_simple_rejections
                WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            """
            params = [days]

            if pair:
                query += " AND pair = %s"
                params.append(pair)

            if session:
                query += " AND market_session = %s"
                params.append(session)

            if stage:
                query += " AND rejection_stage = %s"
                params.append(stage)

            query += " ORDER BY scan_timestamp DESC"

            df = pd.read_sql_query(query, conn, params=params)
            return df

        except Exception as e:
            self.logger.error(f"Failed to get rejections dataframe: {e}")
            return pd.DataFrame()

        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def cleanup_old_rejections(self, days_to_keep: int = 90) -> int:
        """
        Delete rejections older than specified days.

        Args:
            days_to_keep: Number of days of data to retain

        Returns:
            Number of rows deleted
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM smc_simple_rejections
                WHERE scan_timestamp < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))

            deleted = cursor.rowcount
            conn.commit()

            self.logger.info(f"Cleaned up {deleted} old rejection records (keeping {days_to_keep} days)")
            return deleted

        except Exception as e:
            self.logger.error(f"Failed to cleanup old rejections: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return 0

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
