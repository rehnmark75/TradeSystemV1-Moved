"""
Telegram notification service for system alerts.
"""
import logging
import asyncio
from typing import Optional
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
from ..config import settings
from ..models import Alert, AlertSeverity

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends notifications to Telegram."""

    def __init__(self):
        self.bot: Optional[Bot] = None
        self.chat_id: Optional[str] = None
        self._initialized = False
        self._initialize()

    def _initialize(self):
        """Initialize Telegram bot."""
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            logger.warning("Telegram credentials not configured - notifications disabled")
            return

        try:
            self.bot = Bot(token=settings.telegram_bot_token)
            self.chat_id = settings.telegram_chat_id
            self._initialized = True
            logger.info("Telegram notifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")

    def is_enabled(self) -> bool:
        """Check if Telegram notifications are enabled."""
        return self._initialized and settings.telegram_enabled

    async def send_message(self, message: str, parse_mode: str = ParseMode.MARKDOWN) -> bool:
        """Send a message to Telegram."""
        if not self.is_enabled():
            logger.debug("Telegram notifications disabled, skipping")
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False

    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert notification to Telegram."""
        if not self.is_enabled():
            return False

        message = self._format_alert_message(alert)
        return await self.send_message(message)

    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert for Telegram with rich formatting."""
        emoji = alert.get_emoji()
        severity_text = alert.severity.value.upper()

        # Header
        message = f"{emoji} *{severity_text}: {self._escape_markdown(alert.title)}*\n"
        message += "━" * 20 + "\n\n"

        # Container info
        if alert.container_name:
            message += f"📦 *Container:* `{alert.container_name}`\n"

        # Timestamp
        timestamp = alert.created_at.strftime("%Y-%m-%d %H:%M:%S")
        message += f"⏰ *Time:* {timestamp}\n\n"

        # Message body
        message += f"{self._escape_markdown(alert.message)}\n"

        # Details
        if alert.details:
            message += "\n*Details:*\n"
            for key, value in alert.details.items():
                # Format key nicely
                formatted_key = key.replace("_", " ").title()
                message += f"  • {formatted_key}: `{value}`\n"

        # Suggested actions based on alert type
        actions = self._get_suggested_actions(alert)
        if actions:
            message += f"\n🔧 *Suggested Actions:*\n{actions}"

        return message

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram Markdown."""
        if not text:
            return ""
        # Escape characters that have special meaning in Markdown
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        return text

    def _get_suggested_actions(self, alert: Alert) -> str:
        """Get suggested actions based on alert type."""
        from ..models import AlertType

        container = alert.container_name or "container"

        actions = {
            AlertType.CONTAINER_DOWN: f"""
  1\\. Check logs: `docker logs {container}`
  2\\. Restart: `docker restart {container}`
  3\\. Check dependencies""",

            AlertType.RESTART_LOOP: f"""
  1\\. Check logs: `docker logs {container}`
  2\\. Check resource limits
  3\\. Review recent changes""",

            AlertType.HEALTH_CHECK_FAILED: f"""
  1\\. Check service logs
  2\\. Verify endpoint: `curl http://localhost/health`
  3\\. Check network connectivity""",

            AlertType.HIGH_CPU: f"""
  1\\. Check processes: `docker top {container}`
  2\\. Review recent workload
  3\\. Consider scaling""",

            AlertType.HIGH_MEMORY: f"""
  1\\. Check memory usage: `docker stats {container}`
  2\\. Look for memory leaks
  3\\. Consider increasing limits""",

            AlertType.CONTAINER_RECOVERED: "",  # No action needed
        }

        return actions.get(alert.alert_type, "")

    async def send_system_status_summary(self, system_health) -> bool:
        """Send a system status summary message."""
        if not self.is_enabled():
            return False

        status_emoji = system_health.get_status_emoji()
        status = system_health.get_overall_status().upper()

        message = f"{status_emoji} *System Status: {status}*\n"
        message += "━" * 25 + "\n\n"

        # Summary stats
        message += f"🟢 Running: {system_health.running_containers}\n"
        message += f"🔴 Stopped: {system_health.stopped_containers}\n"
        message += f"⚠️ Warnings: {system_health.warning_containers}\n"
        message += f"📊 Health Score: {system_health.health_score:.0f}%\n"

        if system_health.active_alerts > 0:
            message += f"🔔 Active Alerts: {system_health.active_alerts}\n"

        # Critical issues
        if system_health.critical_issues:
            message += "\n🚨 *Critical Issues:*\n"
            for issue in system_health.critical_issues[:5]:
                message += f"  • {self._escape_markdown(issue)}\n"

        # Warnings
        if system_health.warnings:
            message += "\n⚠️ *Warnings:*\n"
            for warning in system_health.warnings[:5]:
                message += f"  • {self._escape_markdown(warning)}\n"

        message += f"\n_Last updated: {system_health.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"

        return await self.send_message(message)

    async def test_connection(self) -> tuple:
        """Test Telegram connection. Returns (success, message)."""
        if not self.is_enabled():
            return False, "Telegram not configured or disabled"

        try:
            bot_info = await self.bot.get_me()
            # Use plain text for test message to avoid parsing issues
            test_message = f"✅ Test Connection Successful\n\nBot: @{bot_info.username}\nSystem Monitor is connected!"
            success = await self.send_message(test_message, parse_mode=None)

            if success:
                return True, f"Connected to bot @{bot_info.username}"
            else:
                return False, "Failed to send test message"

        except Exception as e:
            return False, f"Connection failed: {e}"
