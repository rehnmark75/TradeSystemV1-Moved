"""
Alert models for system monitoring.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    CONTAINER_DOWN = "container_down"
    CONTAINER_RESTARTING = "container_restarting"
    CONTAINER_RECOVERED = "container_recovered"
    HEALTH_CHECK_FAILED = "health_check_failed"
    HEALTH_CHECK_RECOVERED = "health_check_recovered"
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    RESTART_LOOP = "restart_loop"
    DISK_SPACE_LOW = "disk_space_low"
    SERVICE_DEGRADED = "service_degraded"
    CUSTOM = "custom"


class Alert(BaseModel):
    """System alert model."""
    id: Optional[str] = None
    alert_type: AlertType
    severity: AlertSeverity
    container_name: Optional[str] = None
    title: str
    message: str
    details: Dict[str, Any] = {}
    created_at: datetime = datetime.now()
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    notification_sent: bool = False
    notification_channels: list = []

    def get_emoji(self) -> str:
        """Get emoji for alert severity."""
        return {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
        }.get(self.severity, "📢")

    def format_telegram_message(self) -> str:
        """Format alert for Telegram notification."""
        emoji = self.get_emoji()
        severity_text = self.severity.value.upper()

        message = f"{emoji} *{severity_text}: {self.title}*\n"
        message += "━" * 20 + "\n"

        if self.container_name:
            message += f"📦 *Container:* `{self.container_name}`\n"

        message += f"⏰ *Time:* {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"\n{self.message}\n"

        if self.details:
            message += "\n*Details:*\n"
            for key, value in self.details.items():
                message += f"  • {key}: `{value}`\n"

        return message

    def format_email_subject(self) -> str:
        """Format alert for email subject line."""
        emoji = self.get_emoji()
        return f"{emoji} [{self.severity.value.upper()}] {self.title}"

    def format_email_body(self) -> str:
        """Format alert for email body (HTML)."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.CRITICAL: "#dc3545",
        }
        color = severity_colors.get(self.severity, "#6c757d")

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0;">
                <h2 style="margin: 0;">{self.get_emoji()} {self.title}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Severity: {self.severity.value.upper()}</p>
            </div>
            <div style="border: 1px solid #ddd; border-top: none; padding: 20px; border-radius: 0 0 5px 5px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Time:</strong></td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{self.created_at.strftime('%Y-%m-%d %H:%M:%S')}</td>
                    </tr>
        """

        if self.container_name:
            html += f"""
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><strong>Container:</strong></td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #eee;"><code>{self.container_name}</code></td>
                    </tr>
            """

        html += f"""
                </table>
                <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                    <p style="margin: 0;">{self.message}</p>
                </div>
        """

        if self.details:
            html += """
                <div style="margin-top: 20px;">
                    <h3 style="margin-bottom: 10px;">Details:</h3>
                    <table style="width: 100%; border-collapse: collapse;">
            """
            for key, value in self.details.items():
                html += f"""
                        <tr>
                            <td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>{key}:</strong></td>
                            <td style="padding: 5px; border-bottom: 1px solid #eee;"><code>{value}</code></td>
                        </tr>
                """
            html += """
                    </table>
                </div>
            """

        html += """
            </div>
            <div style="text-align: center; padding: 15px; color: #6c757d; font-size: 12px;">
                <p>TradeSystemV1 - System Monitor</p>
            </div>
        </div>
        """

        return html
