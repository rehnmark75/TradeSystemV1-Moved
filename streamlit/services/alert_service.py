"""
Alert Service

Provides alert management for stock scanner notifications:
- Create alerts for breakouts, watchlist entries, volume spikes
- Store alerts in database (stock_alerts table)
- Check and trigger pending alerts
- Mark alerts as sent/acknowledged

This is the service layer for the alert system. Full implementation
will include email/push notification delivery.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class AlertService:
    """Service for managing stock alerts."""

    ALERT_TYPES = [
        'breakout',           # Price breaks above resistance
        'watchlist_entry',    # Stock enters a watchlist
        'volume_spike',       # Volume > 3x average
        'rs_improvement',     # RS percentile improves significantly
        'signal_triggered',   # Scanner signal generated
        'price_target',       # Price reaches target level
        'stop_hit',           # Price hits stop loss level
    ]

    PRIORITY_LEVELS = ['low', 'normal', 'high', 'critical']

    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or "postgresql://postgres:postgres@postgres:5432/stocks"

    def _get_connection(self):
        """Get database connection."""
        try:
            return psycopg2.connect(self.connection_string)
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None

    def create_alert(
        self,
        ticker: str,
        alert_type: str,
        alert_subtype: str = None,
        trigger_price: float = None,
        trigger_volume: int = None,
        alert_message: str = None,
        alert_data: dict = None,
        priority: str = 'normal'
    ) -> Optional[int]:
        """
        Create a new alert.

        Args:
            ticker: Stock ticker symbol
            alert_type: Type of alert (breakout, watchlist_entry, etc.)
            alert_subtype: Optional subtype (e.g., 'resistance_break')
            trigger_price: Price that triggered the alert
            trigger_volume: Volume that triggered the alert
            alert_message: Human-readable alert message
            alert_data: Additional JSON data
            priority: Alert priority (low, normal, high, critical)

        Returns:
            Alert ID if created successfully, None otherwise
        """
        if alert_type not in self.ALERT_TYPES:
            logger.warning(f"Unknown alert type: {alert_type}")

        if priority not in self.PRIORITY_LEVELS:
            priority = 'normal'

        conn = self._get_connection()
        if not conn:
            return None

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO stock_alerts (
                        ticker, alert_type, alert_subtype,
                        trigger_price, trigger_volume,
                        alert_message, alert_data, priority,
                        status, triggered_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, 'pending', NOW()
                    )
                    RETURNING id
                """, (
                    ticker, alert_type, alert_subtype,
                    trigger_price, trigger_volume,
                    alert_message, psycopg2.extras.Json(alert_data) if alert_data else None,
                    priority
                ))
                alert_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Created alert {alert_id}: {alert_type} for {ticker}")
                return alert_id
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

    def get_pending_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending alerts that need to be sent."""
        conn = self._get_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT *
                    FROM stock_alerts
                    WHERE status = 'pending'
                    ORDER BY
                        CASE priority
                            WHEN 'critical' THEN 1
                            WHEN 'high' THEN 2
                            WHEN 'normal' THEN 3
                            WHEN 'low' THEN 4
                        END,
                        triggered_at DESC
                    LIMIT %s
                """, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get pending alerts: {e}")
            return []
        finally:
            conn.close()

    def get_alerts_for_ticker(self, ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts for a specific ticker."""
        conn = self._get_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT *
                    FROM stock_alerts
                    WHERE ticker = %s
                    ORDER BY triggered_at DESC
                    LIMIT %s
                """, (ticker, limit))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get alerts for {ticker}: {e}")
            return []
        finally:
            conn.close()

    def mark_alert_sent(self, alert_id: int) -> bool:
        """Mark an alert as sent."""
        return self._update_alert_status(alert_id, 'sent')

    def mark_alert_acknowledged(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged by user."""
        return self._update_alert_status(alert_id, 'acknowledged')

    def expire_old_alerts(self, hours: int = 24) -> int:
        """Expire alerts older than specified hours."""
        conn = self._get_connection()
        if not conn:
            return 0

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE stock_alerts
                    SET status = 'expired'
                    WHERE status = 'pending'
                    AND triggered_at < NOW() - INTERVAL '%s hours'
                """, (hours,))
                count = cursor.rowcount
                conn.commit()
                logger.info(f"Expired {count} old alerts")
                return count
        except Exception as e:
            logger.error(f"Failed to expire alerts: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def _update_alert_status(self, alert_id: int, status: str) -> bool:
        """Update alert status."""
        conn = self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE stock_alerts
                    SET status = %s, sent_at = CASE WHEN %s = 'sent' THEN NOW() ELSE sent_at END
                    WHERE id = %s
                """, (status, status, alert_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update alert {alert_id}: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics for dashboard."""
        conn = self._get_connection()
        if not conn:
            return {}

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'pending') as pending_count,
                        COUNT(*) FILTER (WHERE status = 'sent' AND triggered_at > NOW() - INTERVAL '24 hours') as sent_today,
                        COUNT(*) FILTER (WHERE triggered_at > NOW() - INTERVAL '24 hours') as total_today,
                        COUNT(*) FILTER (WHERE priority = 'critical' AND status = 'pending') as critical_pending
                    FROM stock_alerts
                """)
                row = cursor.fetchone()
                return dict(row) if row else {}
        except Exception as e:
            logger.error(f"Failed to get alert stats: {e}")
            return {}
        finally:
            conn.close()


# Factory function
def get_alert_service() -> AlertService:
    """Get alert service instance."""
    return AlertService()


# Alert trigger functions for common scenarios
def create_breakout_alert(ticker: str, price: float, resistance_level: float, volume: int = None) -> Optional[int]:
    """Create a breakout alert."""
    service = get_alert_service()
    return service.create_alert(
        ticker=ticker,
        alert_type='breakout',
        alert_subtype='resistance_break',
        trigger_price=price,
        trigger_volume=volume,
        alert_message=f"{ticker} broke above resistance at ${resistance_level:.2f}",
        alert_data={'resistance_level': resistance_level},
        priority='high'
    )


def create_volume_spike_alert(ticker: str, volume: int, avg_volume: int, price: float) -> Optional[int]:
    """Create a volume spike alert."""
    service = get_alert_service()
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1
    return service.create_alert(
        ticker=ticker,
        alert_type='volume_spike',
        alert_subtype=f'{volume_ratio:.1f}x_volume',
        trigger_price=price,
        trigger_volume=volume,
        alert_message=f"{ticker} volume spike: {volume_ratio:.1f}x average",
        alert_data={'avg_volume': avg_volume, 'volume_ratio': volume_ratio},
        priority='normal' if volume_ratio < 3 else 'high'
    )


def create_signal_alert(ticker: str, scanner_name: str, signal_type: str, entry_price: float, score: int) -> Optional[int]:
    """Create a scanner signal alert."""
    service = get_alert_service()
    priority = 'high' if score >= 80 else 'normal'
    return service.create_alert(
        ticker=ticker,
        alert_type='signal_triggered',
        alert_subtype=scanner_name,
        trigger_price=entry_price,
        alert_message=f"{ticker}: {scanner_name} {signal_type} signal (Score: {score})",
        alert_data={'scanner_name': scanner_name, 'signal_type': signal_type, 'score': score},
        priority=priority
    )
