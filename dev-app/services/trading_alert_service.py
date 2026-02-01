# services/trading_alert_service.py
"""
Trading Alert Service - Escalate critical errors beyond logging

CRITICAL: This service addresses the gap where errors are only logged,
leading to silent accumulation of bad states.

Created: Jan 2026 as part of bulletproof trailing system

Alert Levels:
- LOG: All events (default logging)
- TELEGRAM: Warnings (>5 pip mismatch, 2+ retries, verification failures)
- CRITICAL: Major issues (>10 pip mismatch, exhausted retries, DB failures)
"""

import os
import logging
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("trading_alerts")
logger.setLevel(logging.INFO)


class AlertLevel(Enum):
    """Alert severity levels"""
    LOG = "log"              # Log only
    TELEGRAM = "telegram"    # Send Telegram message
    CRITICAL = "critical"    # Telegram + flag for review


class AlertType(Enum):
    """Types of trading alerts"""
    MISMATCH = "mismatch"                      # DB/IG desync
    VERIFICATION_FAILED = "verification_failed"  # Stop update not applied
    RETRY_EXHAUSTED = "retry_exhausted"        # All retries failed
    DB_COMMIT_FAILED = "db_commit_failed"      # Database write failed
    RECOVERY_FAILED = "recovery_failed"        # Could not recover from failure
    POSITION_NOT_FOUND = "position_not_found"  # Position missing on IG
    TRADE_OPENED = "trade_opened"              # New trade placed


@dataclass
class TradingAlert:
    """Structured alert data"""
    alert_type: AlertType
    level: AlertLevel
    trade_id: int
    deal_id: str
    epic: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "level": self.level.value,
            "trade_id": self.trade_id,
            "deal_id": self.deal_id,
            "epic": self.epic,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at.isoformat()
        }


class TradingAlertService:
    """
    Escalate critical trading errors beyond logging.

    Integrates with Telegram for real-time alerts on serious issues.
    """

    # Thresholds for escalation
    MISMATCH_TELEGRAM_THRESHOLD = 5.0   # pips - send Telegram
    MISMATCH_CRITICAL_THRESHOLD = 10.0  # pips - critical alert
    RETRY_TELEGRAM_THRESHOLD = 2        # attempts - send Telegram

    def __init__(self):
        self.logger = logger
        self._telegram_enabled = os.getenv("TRAILING_ALERT_TELEGRAM_ENABLED", "true").lower() == "true"
        self._telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self._telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self._alert_history: List[TradingAlert] = []
        self._max_history = 100

        # Trade notification toggle (separate from error alerts)
        self._trade_notification_enabled = os.getenv("TRADE_NOTIFICATION_TELEGRAM_ENABLED", "false").lower() == "true"

        # Check if Telegram is properly configured
        if self._telegram_enabled and (not self._telegram_bot_token or not self._telegram_chat_id):
            self.logger.warning(
                "[ALERT SERVICE] Telegram enabled but credentials not configured. "
                "Alerts will be logged only."
            )
            self._telegram_enabled = False

        if self._trade_notification_enabled:
            self.logger.info("[ALERT SERVICE] Trade notifications enabled via Telegram")

    def _can_send_telegram(self) -> bool:
        """Check if Telegram sending is available"""
        return bool(
            self._telegram_enabled and
            self._telegram_bot_token and
            self._telegram_chat_id
        )

    async def _send_telegram(self, message: str) -> bool:
        """Send message to Telegram using HTTP API"""
        if not self._can_send_telegram():
            return False

        try:
            url = f"https://api.telegram.org/bot{self._telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self._telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    self.logger.info("[ALERT SERVICE] Telegram message sent successfully")
                    return True
                else:
                    self.logger.error(
                        f"[ALERT SERVICE] Telegram API error: {response.status_code} - {response.text}"
                    )
                    return False

        except Exception as e:
            self.logger.error(f"[ALERT SERVICE] Failed to send Telegram: {e}")
            return False

    def _format_mismatch_message(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        db_stop: float,
        ig_stop: float,
        mismatch_pips: float,
        trailing_active: bool
    ) -> str:
        """Format mismatch alert for Telegram"""
        severity = "CRITICAL" if mismatch_pips >= self.MISMATCH_CRITICAL_THRESHOLD else "WARNING"
        emoji = "🚨" if severity == "CRITICAL" else "⚠️"

        message = f"{emoji} *TRAILING STOP MISMATCH - {severity}*\n"
        message += "━" * 25 + "\n\n"
        message += f"📊 *Trade:* `{trade_id}` ({deal_id[:8]}...)\n"
        message += f"💱 *Pair:* `{self._format_epic(epic)}`\n"
        message += f"🔄 *Trailing Active:* `{trailing_active}`\n\n"
        message += f"📉 *DB Stop:* `{db_stop}`\n"
        message += f"📈 *IG Stop:* `{ig_stop}`\n"
        message += f"⚡ *Mismatch:* `{mismatch_pips}` pips\n\n"

        if mismatch_pips >= self.MISMATCH_CRITICAL_THRESHOLD:
            message += "🔧 *Action Required:* Manual investigation needed\n"
        else:
            message += "📝 *Note:* Monitoring - no action required yet\n"

        return message

    def _format_verification_message(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        expected_stop: float,
        actual_stop: float,
        mismatch_pips: float
    ) -> str:
        """Format verification failure alert for Telegram"""
        message = "🚫 *STOP UPDATE VERIFICATION FAILED*\n"
        message += "━" * 25 + "\n\n"
        message += f"📊 *Trade:* `{trade_id}` ({deal_id[:8]}...)\n"
        message += f"💱 *Pair:* `{self._format_epic(epic)}`\n\n"
        message += f"📤 *Sent Stop:* `{expected_stop}`\n"
        message += f"📥 *IG Has:* `{actual_stop}`\n"
        message += f"⚡ *Mismatch:* `{mismatch_pips}` pips\n\n"
        message += "🔧 *Action:* IG may have rejected the stop level.\n"
        message += "Check spread, min distance, or market conditions.\n"

        return message

    def _format_retry_exhausted_message(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        operation: str,
        attempts: int,
        last_error: str
    ) -> str:
        """Format retry exhausted alert for Telegram"""
        message = "❌ *API RETRIES EXHAUSTED*\n"
        message += "━" * 25 + "\n\n"
        message += f"📊 *Trade:* `{trade_id}` ({deal_id[:8]}...)\n"
        message += f"💱 *Pair:* `{self._format_epic(epic)}`\n"
        message += f"🔄 *Operation:* `{operation}`\n"
        message += f"🔢 *Attempts:* `{attempts}`\n\n"
        message += f"💥 *Last Error:*\n`{last_error[:200]}`\n\n"
        message += "🔧 *Action:* Check IG API connectivity and rate limits.\n"

        return message

    def _format_db_failure_message(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        error: str,
        ig_value: Optional[float]
    ) -> str:
        """Format DB commit failure alert for Telegram"""
        message = "💾 *DATABASE COMMIT FAILED*\n"
        message += "━" * 25 + "\n\n"
        message += f"📊 *Trade:* `{trade_id}` ({deal_id[:8]}...)\n"
        message += f"💱 *Pair:* `{self._format_epic(epic)}`\n\n"
        message += f"💥 *Error:*\n`{error[:200]}`\n\n"

        if ig_value:
            message += f"📈 *IG Stop (source of truth):* `{ig_value}`\n\n"

        message += "🔧 *Action:* IG was updated but DB write failed.\n"
        message += "Manual DB sync may be required.\n"

        return message

    def _format_epic(self, epic: str) -> str:
        """Extract pair name from epic"""
        # CS.D.EURUSD.CEEM.IP -> EURUSD
        if ".D." in epic:
            parts = epic.split(".")
            if len(parts) >= 3:
                return parts[2]
        return epic

    def _add_to_history(self, alert: TradingAlert):
        """Add alert to history, maintaining max size"""
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

    async def send_mismatch_alert(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        db_stop: float,
        ig_stop: float,
        mismatch_pips: float,
        trailing_active: bool = False
    ) -> TradingAlert:
        """
        Send alert for DB/IG mismatch.

        Escalation:
        - < 5 pips: LOG only
        - 5-10 pips: TELEGRAM
        - > 10 pips: CRITICAL (Telegram + flagged)
        """
        # Determine alert level
        if mismatch_pips >= self.MISMATCH_CRITICAL_THRESHOLD:
            level = AlertLevel.CRITICAL
        elif mismatch_pips >= self.MISMATCH_TELEGRAM_THRESHOLD:
            level = AlertLevel.TELEGRAM
        else:
            level = AlertLevel.LOG

        alert = TradingAlert(
            alert_type=AlertType.MISMATCH,
            level=level,
            trade_id=trade_id,
            deal_id=deal_id,
            epic=epic,
            message=f"DB/IG mismatch: {mismatch_pips} pips",
            details={
                "db_stop": db_stop,
                "ig_stop": ig_stop,
                "mismatch_pips": mismatch_pips,
                "trailing_active": trailing_active
            }
        )

        # Always log
        log_msg = (
            f"[MISMATCH ALERT] Trade {trade_id} {self._format_epic(epic)}: "
            f"DB={db_stop}, IG={ig_stop} ({mismatch_pips} pips) - {level.value}"
        )

        if level == AlertLevel.CRITICAL:
            self.logger.error(log_msg)
        elif level == AlertLevel.TELEGRAM:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)

        # Send Telegram if warranted
        if level in [AlertLevel.TELEGRAM, AlertLevel.CRITICAL]:
            message = self._format_mismatch_message(
                trade_id, deal_id, epic, db_stop, ig_stop, mismatch_pips, trailing_active
            )
            await self._send_telegram(message)

        self._add_to_history(alert)
        return alert

    async def send_verification_failure(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        expected_stop: float,
        actual_stop: Optional[float],
        mismatch_pips: float = 0
    ) -> TradingAlert:
        """
        Send alert when stop update verification fails.
        Always sends Telegram (this is a serious issue).
        """
        alert = TradingAlert(
            alert_type=AlertType.VERIFICATION_FAILED,
            level=AlertLevel.TELEGRAM,
            trade_id=trade_id,
            deal_id=deal_id,
            epic=epic,
            message=f"Stop verification failed: sent {expected_stop}, got {actual_stop}",
            details={
                "expected_stop": expected_stop,
                "actual_stop": actual_stop,
                "mismatch_pips": mismatch_pips
            }
        )

        self.logger.error(
            f"[VERIFICATION ALERT] Trade {trade_id} {self._format_epic(epic)}: "
            f"Sent {expected_stop}, IG has {actual_stop} ({mismatch_pips} pips)"
        )

        message = self._format_verification_message(
            trade_id, deal_id, epic, expected_stop, actual_stop or 0, mismatch_pips
        )
        await self._send_telegram(message)

        self._add_to_history(alert)
        return alert

    async def send_retry_exhausted(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        operation: str,
        attempts: int,
        last_error: str
    ) -> TradingAlert:
        """
        Send alert when all API retries are exhausted.
        Always sends Telegram.
        """
        alert = TradingAlert(
            alert_type=AlertType.RETRY_EXHAUSTED,
            level=AlertLevel.CRITICAL,
            trade_id=trade_id,
            deal_id=deal_id,
            epic=epic,
            message=f"Retries exhausted for {operation} after {attempts} attempts",
            details={
                "operation": operation,
                "attempts": attempts,
                "last_error": last_error
            }
        )

        self.logger.error(
            f"[RETRY ALERT] Trade {trade_id} {self._format_epic(epic)}: "
            f"{operation} failed after {attempts} attempts - {last_error[:100]}"
        )

        message = self._format_retry_exhausted_message(
            trade_id, deal_id, epic, operation, attempts, last_error
        )
        await self._send_telegram(message)

        self._add_to_history(alert)
        return alert

    async def send_db_commit_failed(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        error: str,
        ig_value: Optional[float] = None
    ) -> TradingAlert:
        """
        Send alert when DB commit fails after IG update succeeded.
        This is critical because it causes desync.
        """
        alert = TradingAlert(
            alert_type=AlertType.DB_COMMIT_FAILED,
            level=AlertLevel.CRITICAL,
            trade_id=trade_id,
            deal_id=deal_id,
            epic=epic,
            message=f"DB commit failed: {error[:100]}",
            details={
                "error": error,
                "ig_value": ig_value
            }
        )

        self.logger.error(
            f"[DB COMMIT ALERT] Trade {trade_id} {self._format_epic(epic)}: "
            f"IG updated to {ig_value} but DB commit failed - {error[:100]}"
        )

        message = self._format_db_failure_message(
            trade_id, deal_id, epic, error, ig_value
        )
        await self._send_telegram(message)

        self._add_to_history(alert)
        return alert

    async def send_recovery_failed(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        original_error: str,
        recovery_error: str
    ) -> TradingAlert:
        """
        Send alert when recovery from a failure also fails.
        This requires immediate manual intervention.
        """
        alert = TradingAlert(
            alert_type=AlertType.RECOVERY_FAILED,
            level=AlertLevel.CRITICAL,
            trade_id=trade_id,
            deal_id=deal_id,
            epic=epic,
            message="Recovery failed - manual intervention required",
            details={
                "original_error": original_error,
                "recovery_error": recovery_error
            }
        )

        self.logger.critical(
            f"[RECOVERY FAILED] Trade {trade_id} {self._format_epic(epic)}: "
            f"Original: {original_error[:50]}, Recovery: {recovery_error[:50]}"
        )

        message = "🆘 *RECOVERY FAILED - MANUAL INTERVENTION REQUIRED*\n"
        message += "━" * 25 + "\n\n"
        message += f"📊 *Trade:* `{trade_id}` ({deal_id[:8]}...)\n"
        message += f"💱 *Pair:* `{self._format_epic(epic)}`\n\n"
        message += f"💥 *Original Error:*\n`{original_error[:150]}`\n\n"
        message += f"🔄 *Recovery Error:*\n`{recovery_error[:150]}`\n\n"
        message += "🚨 *IMMEDIATE ACTION REQUIRED*\n"

        await self._send_telegram(message)

        self._add_to_history(alert)
        return alert

    async def send_trade_opened_notification(
        self,
        trade_id: int,
        deal_id: str,
        epic: str,
        direction: str,
        entry_price: float,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
        is_scalp: bool = False,
        order_type: str = "market"
    ) -> Optional[TradingAlert]:
        """
        Send Telegram notification when a new trade is placed.

        Only sends if TRADE_NOTIFICATION_TELEGRAM_ENABLED=true.
        """
        if not self._trade_notification_enabled:
            return None

        if not self._can_send_telegram():
            self.logger.debug("[TRADE NOTIFICATION] Telegram not configured, skipping notification")
            return None

        pair_name = self._format_epic(epic)

        # Direction emoji
        dir_emoji = "🟢" if direction.upper() == "BUY" else "🔴"
        dir_text = "LONG" if direction.upper() == "BUY" else "SHORT"

        # Order type indicator
        order_emoji = "⚡" if order_type == "market" else "📋"
        order_text = "Market" if order_type == "market" else "Limit"

        # Scalp indicator
        scalp_text = " (Scalp)" if is_scalp else ""

        message = f"{dir_emoji} *NEW TRADE OPENED*{scalp_text}\n"
        message += "━" * 25 + "\n\n"
        message += f"💱 *Pair:* `{pair_name}`\n"
        message += f"📊 *Direction:* `{dir_text}`\n"
        message += f"{order_emoji} *Type:* `{order_text}`\n"
        message += f"💰 *Entry:* `{entry_price:.5f}`\n"

        if stop_price:
            message += f"🛑 *Stop Loss:* `{stop_price:.5f}`\n"
        if limit_price:
            message += f"🎯 *Take Profit:* `{limit_price:.5f}`\n"

        message += f"\n🆔 *Trade ID:* `{trade_id}`"

        # Create alert record
        alert = TradingAlert(
            alert_type=AlertType.TRADE_OPENED,
            level=AlertLevel.LOG,  # Not critical, just informational
            trade_id=trade_id,
            deal_id=deal_id,
            epic=epic,
            message=f"Trade opened: {pair_name} {direction}",
            details={
                "entry_price": entry_price,
                "stop_price": stop_price,
                "limit_price": limit_price,
                "is_scalp": is_scalp,
                "order_type": order_type
            }
        )

        await self._send_telegram(message)
        self.logger.info(f"[TRADE NOTIFICATION] Sent notification for {pair_name} {direction}")

        self._add_to_history(alert)
        return alert

    def get_recent_alerts(
        self,
        limit: int = 20,
        level: Optional[AlertLevel] = None,
        alert_type: Optional[AlertType] = None
    ) -> List[TradingAlert]:
        """Get recent alerts, optionally filtered"""
        alerts = self._alert_history

        if level:
            alerts = [a for a in alerts if a.level == level]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return alerts[-limit:]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        last_hour = datetime.utcnow().replace(hour=datetime.utcnow().hour)
        recent = [a for a in self._alert_history if a.created_at >= last_hour]

        return {
            "total_alerts": len(self._alert_history),
            "last_hour": len(recent),
            "by_type": {
                t.value: len([a for a in recent if a.alert_type == t])
                for t in AlertType
            },
            "by_level": {
                l.value: len([a for a in recent if a.level == l])
                for l in AlertLevel
            },
            "critical_count": len([a for a in recent if a.level == AlertLevel.CRITICAL])
        }


# Singleton instance
_alert_service: Optional[TradingAlertService] = None


def get_alert_service() -> TradingAlertService:
    """Get singleton instance of alert service"""
    global _alert_service
    if _alert_service is None:
        _alert_service = TradingAlertService()
    return _alert_service
