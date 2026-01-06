# core/intelligence/scan_performance_manager.py
"""
Scan Performance Snapshot Manager

Captures per-epic indicator data for EVERY scan cycle, enabling:
- Rejection pattern analysis (why signals were filtered)
- Signal quality correlation (what conditions produce good signals)
- Market condition analysis (what does the market look like when quiet)

Links to:
- market_intelligence_history (via scan_cycle_id)
- alert_history (via signal_id when signals are generated)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

try:
    from utils.scanner_utils import make_json_serializable
except ImportError:
    from forex_scanner.utils.scanner_utils import make_json_serializable


class ScanPerformanceManager:
    """
    Manages scan performance snapshot storage and retrieval.

    Captures indicator data for every epic on every scan, regardless
    of whether a signal was generated.
    """

    def __init__(self, db_manager):
        """
        Initialize with injected DatabaseManager.

        Args:
            db_manager: DatabaseManager instance for database operations
        """
        if db_manager is None:
            raise ValueError("DatabaseManager is required - cannot be None")

        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self._table_verified = False
        self._initialize_table()

    def _get_connection(self):
        """Get database connection through injected DatabaseManager"""
        return self.db_manager.get_connection()

    def _execute_with_connection(self, operation_func, operation_name="database operation"):
        """Execute database operation with proper connection management"""
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            result = operation_func(conn, cursor)
            conn.commit()
            return result

        except Exception as e:
            self.logger.error(f"{operation_name} failed: {e}")
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
        """Initialize scan_performance_snapshot table if it doesn't exist"""
        def check_table_operation(conn, cursor):
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = 'scan_performance_snapshot'
            """)
            return cursor.fetchone()[0] > 0

        try:
            self._table_verified = self._execute_with_connection(
                check_table_operation, "table check"
            )

            if self._table_verified:
                self.logger.debug("scan_performance_snapshot table exists")
            else:
                self.logger.warning(
                    "scan_performance_snapshot table does not exist. "
                    "Run migration: create_scan_performance_snapshot_table.sql"
                )
        except Exception as e:
            self.logger.error(f"Failed to verify table: {e}")
            self._table_verified = False

    def save_scan_snapshot(
        self,
        scan_cycle_id: str,
        scan_timestamp: datetime,
        epic: str,
        indicator_data: Dict[str, Any],
        signal_outcome: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        smc_context: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Save a performance snapshot for one epic from one scan.

        Args:
            scan_cycle_id: Unique identifier linking to market_intelligence_history
            scan_timestamp: When the scan occurred
            epic: The currency pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
            indicator_data: Dict containing all indicator values
            signal_outcome: Dict with signal result info (generated, rejected, reason)
            market_context: Dict with regime, session, volatility info
            smc_context: Dict with SMC analysis (order blocks, FVG, liquidity)

        Returns:
            Record ID if successful, None otherwise
        """
        if not self._table_verified:
            return None

        def save_operation(conn, cursor):
            # Extract pair name from epic
            pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')

            # Signal outcome defaults
            signal_outcome_data = signal_outcome or {}
            signal_generated = signal_outcome_data.get('generated', False)
            signal_type = signal_outcome_data.get('signal_type')
            signal_id = signal_outcome_data.get('signal_id')
            rejection_reason = signal_outcome_data.get('rejection_reason')
            rejection_details = signal_outcome_data.get('rejection_details')
            raw_confidence = signal_outcome_data.get('raw_confidence')
            final_confidence = signal_outcome_data.get('final_confidence')
            confidence_threshold = signal_outcome_data.get('confidence_threshold')

            # Market context defaults
            market_ctx = market_context or {}

            # SMC context defaults
            smc = smc_context or {}

            # Classify indicators
            er = indicator_data.get('efficiency_ratio')
            er_class = self._classify_efficiency_ratio(er)

            adx = indicator_data.get('adx')
            adx_strength = self._classify_adx_strength(adx)

            rsi = indicator_data.get('rsi_14') or indicator_data.get('rsi')
            rsi_zone = self._classify_rsi_zone(rsi)

            vol_state = market_ctx.get('volatility_state') or self._classify_volatility(
                indicator_data.get('atr_percentile')
            )

            bb_position = self._classify_bb_position(
                indicator_data.get('current_price'),
                indicator_data.get('bb_upper'),
                indicator_data.get('bb_middle'),
                indicator_data.get('bb_lower')
            )

            # MACD trend
            macd_hist = indicator_data.get('macd_histogram')
            macd_trend = 'bullish' if macd_hist and macd_hist > 0 else ('bearish' if macd_hist and macd_hist < 0 else 'neutral')

            # Extended indicators (anything not in dedicated columns)
            extended = {
                k: v for k, v in indicator_data.items()
                if k not in self._get_dedicated_indicator_columns()
            }
            extended_json = json.dumps(make_json_serializable(extended)) if extended else None

            cursor.execute("""
                INSERT INTO scan_performance_snapshot (
                    scan_cycle_id, scan_timestamp, epic, pair_name,
                    signal_generated, signal_type, signal_id, rejection_reason, rejection_details,
                    raw_confidence, final_confidence, confidence_threshold,
                    current_price, bid_price, ask_price, spread_pips,
                    ema_9, ema_21, ema_50, ema_200, ema_bias_4h, price_vs_ema50,
                    macd_line, macd_signal, macd_histogram, macd_trend,
                    rsi_14, rsi_zone,
                    efficiency_ratio, er_classification,
                    atr_14, atr_pips, atr_percentile, volatility_state,
                    bb_upper, bb_middle, bb_lower, bb_width, bb_width_percentile, bb_position,
                    adx, plus_di, minus_di, adx_trend_strength,
                    market_regime, regime_confidence, session, session_volatility,
                    near_order_block, ob_type, ob_distance_pips,
                    near_fvg, fvg_type, fvg_distance_pips,
                    liquidity_sweep_detected, liquidity_sweep_type,
                    smart_money_score, smart_money_validated,
                    mtf_alignment, mtf_confluence_score,
                    entry_quality_score, fib_zone_distance,
                    extended_indicators
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s
                ) RETURNING id
            """, (
                scan_cycle_id, scan_timestamp, epic, pair_name,
                signal_generated, signal_type, signal_id, rejection_reason, rejection_details,
                raw_confidence, final_confidence, confidence_threshold,
                indicator_data.get('current_price'), indicator_data.get('bid_price'),
                indicator_data.get('ask_price'), indicator_data.get('spread_pips'),
                indicator_data.get('ema_9'), indicator_data.get('ema_21'),
                indicator_data.get('ema_50'), indicator_data.get('ema_200'),
                indicator_data.get('ema_bias_4h'), indicator_data.get('price_vs_ema50'),
                indicator_data.get('macd_line'), indicator_data.get('macd_signal'),
                indicator_data.get('macd_histogram'), macd_trend,
                rsi, rsi_zone,
                er, er_class,
                indicator_data.get('atr_14') or indicator_data.get('atr'),
                indicator_data.get('atr_pips'), indicator_data.get('atr_percentile'), vol_state,
                indicator_data.get('bb_upper'), indicator_data.get('bb_middle'),
                indicator_data.get('bb_lower'), indicator_data.get('bb_width'),
                indicator_data.get('bb_width_percentile'), bb_position,
                adx, indicator_data.get('plus_di'), indicator_data.get('minus_di'), adx_strength,
                market_ctx.get('market_regime'), market_ctx.get('regime_confidence'),
                market_ctx.get('session'), market_ctx.get('session_volatility'),
                smc.get('near_order_block', False), smc.get('ob_type'),
                smc.get('ob_distance_pips'),
                smc.get('near_fvg', False), smc.get('fvg_type'),
                smc.get('fvg_distance_pips'),
                smc.get('liquidity_sweep_detected', False), smc.get('liquidity_sweep_type'),
                indicator_data.get('smart_money_score'), indicator_data.get('smart_money_validated', False),
                indicator_data.get('mtf_alignment'), indicator_data.get('mtf_confluence_score'),
                indicator_data.get('entry_quality_score'), indicator_data.get('fib_zone_distance'),
                extended_json
            ))

            result = cursor.fetchone()
            return result[0] if result else None

        try:
            record_id = self._execute_with_connection(save_operation, "save scan snapshot")
            return record_id
        except Exception as e:
            self.logger.error(f"Failed to save scan snapshot for {epic}: {e}")
            return None

    def save_batch_snapshots(
        self,
        scan_cycle_id: str,
        scan_timestamp: datetime,
        snapshots: List[Dict[str, Any]]
    ) -> int:
        """
        Save multiple snapshots in a single transaction for efficiency.

        Args:
            scan_cycle_id: Unique identifier for this scan cycle
            scan_timestamp: When the scan occurred
            snapshots: List of dicts, each containing:
                - epic: The currency pair
                - indicator_data: Dict of indicators
                - signal_outcome: Optional signal result info
                - market_context: Optional market context
                - smc_context: Optional SMC analysis

        Returns:
            Number of snapshots saved successfully
        """
        if not self._table_verified or not snapshots:
            return 0

        saved_count = 0
        for snapshot in snapshots:
            try:
                result = self.save_scan_snapshot(
                    scan_cycle_id=scan_cycle_id,
                    scan_timestamp=scan_timestamp,
                    epic=snapshot['epic'],
                    indicator_data=snapshot.get('indicator_data', {}),
                    signal_outcome=snapshot.get('signal_outcome'),
                    market_context=snapshot.get('market_context'),
                    smc_context=snapshot.get('smc_context')
                )
                if result:
                    saved_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to save snapshot for {snapshot.get('epic')}: {e}")
                continue

        self.logger.debug(f"Saved {saved_count}/{len(snapshots)} scan snapshots")
        return saved_count

    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """
        Remove old snapshot records to prevent database bloat.

        Args:
            days_to_keep: Number of days of history to retain

        Returns:
            Number of records deleted
        """
        def cleanup_operation(conn, cursor):
            cursor.execute(
                "SELECT cleanup_old_scan_snapshots(%s)",
                (days_to_keep,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0

        try:
            deleted = self._execute_with_connection(cleanup_operation, "cleanup snapshots")
            if deleted > 0:
                self.logger.info(f"Cleaned up {deleted} old scan snapshots (keeping {days_to_keep} days)")
            return deleted
        except Exception as e:
            self.logger.warning(f"Snapshot cleanup failed: {e}")
            return 0

    def get_rejection_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get rejection statistics for the specified time period.

        Returns breakdown of rejections by reason, regime, and indicators.
        """
        def stats_operation(conn, cursor):
            cursor.execute("""
                SELECT
                    rejection_reason,
                    market_regime,
                    COUNT(*) as count,
                    AVG(raw_confidence) as avg_confidence,
                    AVG(efficiency_ratio) as avg_er,
                    AVG(adx) as avg_adx
                FROM scan_performance_snapshot
                WHERE rejection_reason IS NOT NULL
                  AND scan_timestamp > NOW() - INTERVAL '%s hours'
                GROUP BY rejection_reason, market_regime
                ORDER BY count DESC
            """, (hours,))

            rows = cursor.fetchall()
            return [
                {
                    'rejection_reason': r[0],
                    'market_regime': r[1],
                    'count': r[2],
                    'avg_confidence': float(r[3]) if r[3] else None,
                    'avg_er': float(r[4]) if r[4] else None,
                    'avg_adx': float(r[5]) if r[5] else None
                }
                for r in rows
            ]

        try:
            return self._execute_with_connection(stats_operation, "get rejection stats")
        except Exception:
            return []

    def get_signal_indicator_comparison(self, days: int = 7) -> Dict[str, Any]:
        """
        Compare indicator distributions between signals and non-signals.

        Useful for understanding what conditions produce signals.
        """
        def comparison_operation(conn, cursor):
            cursor.execute("""
                SELECT
                    signal_generated,
                    COUNT(*) as count,
                    AVG(efficiency_ratio) as avg_er,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY efficiency_ratio) as median_er,
                    AVG(adx) as avg_adx,
                    AVG(rsi_14) as avg_rsi,
                    AVG(atr_percentile) as avg_atr_pct
                FROM scan_performance_snapshot
                WHERE scan_timestamp > NOW() - INTERVAL '%s days'
                GROUP BY signal_generated
            """, (days,))

            rows = cursor.fetchall()
            result = {}
            for r in rows:
                key = 'signals' if r[0] else 'no_signals'
                result[key] = {
                    'count': r[1],
                    'avg_er': float(r[2]) if r[2] else None,
                    'median_er': float(r[3]) if r[3] else None,
                    'avg_adx': float(r[4]) if r[4] else None,
                    'avg_rsi': float(r[5]) if r[5] else None,
                    'avg_atr_pct': float(r[6]) if r[6] else None
                }
            return result

        try:
            return self._execute_with_connection(comparison_operation, "get indicator comparison")
        except Exception:
            return {}

    # =========================================================================
    # Classification helpers
    # =========================================================================

    def _classify_efficiency_ratio(self, er: Optional[float]) -> Optional[str]:
        """Classify efficiency ratio into bands"""
        if er is None:
            return None
        if er >= 0.7:
            return 'strong_trend'
        if er >= 0.5:
            return 'good_trend'
        if er >= 0.3:
            return 'weak_trend'
        return 'choppy'

    def _classify_adx_strength(self, adx: Optional[float]) -> Optional[str]:
        """Classify ADX into trend strength categories"""
        if adx is None:
            return None
        if adx >= 50:
            return 'very_strong'
        if adx >= 35:
            return 'strong'
        if adx >= 25:
            return 'moderate'
        if adx >= 15:
            return 'weak'
        return 'no_trend'

    def _classify_rsi_zone(self, rsi: Optional[float]) -> Optional[str]:
        """Classify RSI into zones"""
        if rsi is None:
            return None
        if rsi >= 70:
            return 'overbought'
        if rsi <= 30:
            return 'oversold'
        return 'neutral'

    def _classify_volatility(self, atr_percentile: Optional[float]) -> Optional[str]:
        """Classify volatility state from ATR percentile"""
        if atr_percentile is None:
            return None
        if atr_percentile >= 90:
            return 'extreme'
        if atr_percentile >= 70:
            return 'high'
        if atr_percentile >= 30:
            return 'normal'
        return 'low'

    def _classify_bb_position(
        self,
        price: Optional[float],
        upper: Optional[float],
        middle: Optional[float],
        lower: Optional[float]
    ) -> Optional[str]:
        """Classify price position relative to Bollinger Bands"""
        if price is None or upper is None or lower is None:
            return None
        if price > upper:
            return 'above_upper'
        if price < lower:
            return 'below_lower'
        if middle is not None:
            band_width = upper - lower
            if band_width > 0:
                upper_zone = middle + (band_width * 0.25)
                lower_zone = middle - (band_width * 0.25)
                if price > upper_zone:
                    return 'upper_zone'
                if price < lower_zone:
                    return 'lower_zone'
        return 'middle'

    def _get_dedicated_indicator_columns(self) -> set:
        """Return set of indicator keys that have dedicated columns"""
        return {
            'current_price', 'bid_price', 'ask_price', 'spread_pips',
            'ema_9', 'ema_21', 'ema_50', 'ema_200', 'ema_bias_4h', 'price_vs_ema50',
            'macd_line', 'macd_signal', 'macd_histogram',
            'rsi_14', 'rsi',
            'efficiency_ratio',
            'atr_14', 'atr', 'atr_pips', 'atr_percentile',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_width_percentile',
            'adx', 'plus_di', 'minus_di',
            'smart_money_score', 'smart_money_validated',
            'mtf_alignment', 'mtf_confluence_score',
            'entry_quality_score', 'fib_zone_distance',
            'volatility_state'  # Added Jan 2026 - derived from atr_percentile
        }
