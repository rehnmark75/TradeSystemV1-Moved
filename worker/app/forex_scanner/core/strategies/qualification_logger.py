#!/usr/bin/env python3
"""
Qualification Logger - Persists signal qualification results to database

VERSION: 1.0.0
DATE: 2026-01-17

PURPOSE:
    Persists ScalpSignalQualifier results to the scalp_qualification_log table
    for later analysis and correlation with trade outcomes.

USAGE:
    logger = QualificationLogger()
    logger.log_qualification(signal, score, filter_results, mode, blocked)

TABLE: scalp_qualification_log (in strategy_config database)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None


class QualificationLogger:
    """
    Persists qualification filter results to scalp_qualification_log table.

    Logs all qualification runs when enabled, storing individual filter
    results for analysis and correlation with trade outcomes.
    """

    VERSION = "1.0.0"

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize QualificationLogger.

        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)
        self._db_config = self._get_db_config()

    def _get_db_config(self) -> Dict:
        """Get database configuration for strategy_config database."""
        import os
        return {
            'host': os.environ.get('POSTGRES_HOST', 'postgres'),
            'port': int(os.environ.get('POSTGRES_PORT', 5432)),
            'database': 'strategy_config',
            'user': os.environ.get('POSTGRES_USER', 'postgres'),
            'password': os.environ.get('POSTGRES_PASSWORD', 'postgres')
        }

    def _get_connection(self):
        """Get database connection."""
        if psycopg2 is None:
            self.logger.warning("psycopg2 not available, qualification logging disabled")
            return None

        try:
            return psycopg2.connect(**self._db_config)
        except Exception as e:
            self.logger.error(f"Failed to connect to strategy_config database: {e}")
            return None

    def log_qualification(
        self,
        signal: Dict,
        score: float,
        filter_results: List[Dict],
        mode: str,
        blocked: bool
    ) -> Optional[int]:
        """
        Log qualification results to database.

        Args:
            signal: Signal dictionary with epic, pair, direction, etc.
            score: Overall qualification score (0.0-1.0)
            filter_results: List of per-filter result dictionaries
            mode: 'MONITORING' or 'ACTIVE'
            blocked: Whether signal was blocked (only True in ACTIVE mode)

        Returns:
            Insert ID if successful, None otherwise
        """
        conn = self._get_connection()
        if not conn:
            return None

        try:
            # Build log record from signal and filter results
            log_data = self._build_log_record(signal, score, filter_results, mode, blocked)

            # Insert into database
            insert_id = self._insert_log(conn, log_data)

            if insert_id:
                self.logger.debug(f"Logged qualification result ID={insert_id} for {signal.get('epic')} (score={score:.0%})")

            return insert_id

        except Exception as e:
            self.logger.error(f"Failed to log qualification: {e}")
            return None
        finally:
            conn.close()

    def _build_log_record(
        self,
        signal: Dict,
        score: float,
        filter_results: List[Dict],
        mode: str,
        blocked: bool
    ) -> Dict:
        """
        Build log record from signal and filter results.

        Extracts individual filter results into dedicated columns.
        """
        # Signal identification
        record = {
            'epic': signal.get('epic', 'UNKNOWN'),
            'pair': signal.get('pair', signal.get('epic', 'UNKNOWN')),
            'direction': signal.get('direction', 'UNKNOWN'),
            'signal_timestamp': signal.get('candle_timestamp') or signal.get('timestamp'),

            # Qualification summary
            'qualification_score': score,
            'qualification_mode': mode,
            'signal_blocked': blocked,

            # Entry details from signal
            'entry_price': signal.get('entry_price') or signal.get('price'),
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'entry_type': signal.get('entry_type'),

            # Market context
            'spread_at_signal': signal.get('spread_pips'),
            'atr_at_signal': signal.get('atr'),
            'ema_distance_pips': signal.get('ema_distance_pips'),
        }

        # Extract individual filter results
        filter_data = self._extract_filter_results(filter_results)
        record.update(filter_data)

        return record

    def _extract_filter_results(self, filter_results: List[Dict]) -> Dict:
        """
        Extract individual filter results into database columns.

        Maps filter result dictionaries to their dedicated columns.
        """
        data = {
            # RSI filter
            'rsi_passed': None,
            'rsi_value': None,
            'rsi_prev': None,
            'rsi_reason': None,

            # Two-Pole filter
            'two_pole_passed': None,
            'two_pole_value': None,
            'two_pole_is_green': None,
            'two_pole_is_purple': None,
            'two_pole_reason': None,

            # MACD filter
            'macd_passed': None,
            'macd_histogram': None,
            'macd_histogram_prev': None,
            'macd_reason': None,

            # Consecutive candles filter
            'consecutive_candles_passed': None,
            'consecutive_candles_count': None,
            'consecutive_candles_reason': None,

            # Anti-chop filter
            'anti_chop_passed': None,
            'anti_chop_alternations': None,
            'anti_chop_reason': None,

            # Body dominance filter
            'body_dominance_passed': None,
            'body_dominance_ratio': None,
            'body_dominance_reason': None,

            # Micro-range filter
            'micro_range_passed': None,
            'micro_range_pips': None,
            'micro_range_reason': None,

            # Momentum candle filter
            'momentum_candle_passed': None,
            'momentum_candle_multiplier': None,
            'momentum_candle_reason': None,
        }

        for result in filter_results:
            filter_name = result.get('filter', '')

            if filter_name == 'RSI_MOMENTUM':
                data['rsi_passed'] = result.get('passed')
                data['rsi_value'] = result.get('rsi_value')
                data['rsi_prev'] = result.get('rsi_prev')
                data['rsi_reason'] = result.get('reason')

            elif filter_name == 'TWO_POLE':
                data['two_pole_passed'] = result.get('passed')
                data['two_pole_value'] = result.get('osc_value')
                data['two_pole_is_green'] = result.get('is_green')
                data['two_pole_is_purple'] = result.get('is_purple')
                data['two_pole_reason'] = result.get('reason')

            elif filter_name == 'MACD_DIRECTION':
                data['macd_passed'] = result.get('passed')
                data['macd_histogram'] = result.get('histogram')
                data['macd_histogram_prev'] = result.get('histogram_prev')
                data['macd_reason'] = result.get('reason')

            elif filter_name == 'CONSECUTIVE_CANDLES':
                data['consecutive_candles_passed'] = result.get('passed')
                data['consecutive_candles_count'] = result.get('consecutive_count')
                data['consecutive_candles_reason'] = result.get('reason')

            elif filter_name == 'ANTI_CHOP':
                data['anti_chop_passed'] = result.get('passed')
                data['anti_chop_alternations'] = result.get('alternations')
                data['anti_chop_reason'] = result.get('reason')

            elif filter_name == 'BODY_DOMINANCE':
                data['body_dominance_passed'] = result.get('passed')
                data['body_dominance_ratio'] = result.get('ratio')
                data['body_dominance_reason'] = result.get('reason')

            elif filter_name == 'MICRO_RANGE':
                data['micro_range_passed'] = result.get('passed')
                data['micro_range_pips'] = result.get('range_pips')
                data['micro_range_reason'] = result.get('reason')

            elif filter_name == 'MOMENTUM_CANDLE':
                data['momentum_candle_passed'] = result.get('passed')
                data['momentum_candle_multiplier'] = result.get('multiplier')
                data['momentum_candle_reason'] = result.get('reason')

        return data

    def _insert_log(self, conn, log_data: Dict) -> Optional[int]:
        """
        Insert log record into scalp_qualification_log table.

        Returns insert ID if successful.
        """
        # Build dynamic INSERT query from log_data keys
        columns = list(log_data.keys())
        placeholders = [f'%({col})s' for col in columns]

        query = f"""
            INSERT INTO scalp_qualification_log (
                {', '.join(columns)}
            ) VALUES (
                {', '.join(placeholders)}
            ) RETURNING id
        """

        try:
            with conn.cursor() as cursor:
                cursor.execute(query, log_data)
                result = cursor.fetchone()
                conn.commit()
                return result[0] if result else None
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Insert failed: {e}")
            raise

    def update_trade_outcome(
        self,
        log_id: int,
        trade_outcome: str,
        pnl_pips: float = None,
        trade_id: str = None
    ) -> bool:
        """
        Update qualification log with trade outcome after trade closes.

        Args:
            log_id: ID from log_qualification()
            trade_outcome: 'WIN', 'LOSS', 'EXPIRED', 'CANCELLED'
            pnl_pips: Profit/loss in pips
            trade_id: IG trade reference

        Returns:
            True if update successful
        """
        conn = self._get_connection()
        if not conn:
            return False

        try:
            query = """
                UPDATE scalp_qualification_log
                SET trade_outcome = %s,
                    pnl_pips = %s,
                    trade_id = %s
                WHERE id = %s
            """

            with conn.cursor() as cursor:
                cursor.execute(query, (trade_outcome, pnl_pips, trade_id, log_id))
                conn.commit()

            self.logger.info(f"Updated qualification log {log_id}: {trade_outcome} ({pnl_pips} pips)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update trade outcome: {e}")
            return False
        finally:
            conn.close()

    def link_to_alert(self, log_id: int, alert_id: int) -> bool:
        """
        Link qualification log to alert_history record.

        Useful for joining qualification data with full signal data.
        """
        conn = self._get_connection()
        if not conn:
            return False

        try:
            # Add alert_id column if needed (backwards compatible)
            with conn.cursor() as cursor:
                cursor.execute("""
                    ALTER TABLE scalp_qualification_log
                    ADD COLUMN IF NOT EXISTS alert_id INTEGER
                """)

                cursor.execute("""
                    UPDATE scalp_qualification_log
                    SET alert_id = %s
                    WHERE id = %s
                """, (alert_id, log_id))

                conn.commit()

            return True

        except Exception as e:
            self.logger.error(f"Failed to link alert: {e}")
            return False
        finally:
            conn.close()
