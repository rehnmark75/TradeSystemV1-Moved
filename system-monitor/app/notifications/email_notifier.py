"""
Email notification service for system alerts.
"""
import logging
import asyncio
from typing import Optional, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib
from ..config import settings
from ..models import Alert, AlertSeverity

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Sends email notifications for critical alerts."""

    def __init__(self):
        self._initialized = False
        self._pending_alerts: List[Alert] = []
        self._batch_task: Optional[asyncio.Task] = None
        self._initialize()

    def _initialize(self):
        """Check email configuration."""
        required = [
            settings.smtp_host,
            settings.smtp_user,
            settings.smtp_password,
            settings.admin_email,
        ]

        if not all(required):
            logger.warning("Email credentials not fully configured - email notifications disabled")
            return

        self._initialized = True
        logger.info("Email notifier initialized")

    def is_enabled(self) -> bool:
        """Check if email notifications are enabled."""
        return self._initialized and settings.email_enabled

    async def send_email(
        self,
        subject: str,
        body_html: str,
        body_text: Optional[str] = None,
        to_email: Optional[str] = None,
    ) -> bool:
        """Send an email."""
        if not self.is_enabled():
            logger.debug("Email notifications disabled, skipping")
            return False

        recipient = to_email or settings.admin_email

        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = settings.smtp_user
            message["To"] = recipient

            # Add plain text version if provided
            if body_text:
                part1 = MIMEText(body_text, "plain")
                message.attach(part1)

            # Add HTML version
            part2 = MIMEText(body_html, "html")
            message.attach(part2)

            # Send email
            await aiosmtplib.send(
                message,
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_user,
                password=settings.smtp_password,
                use_tls=settings.smtp_use_tls,
            )

            logger.info(f"Email sent successfully to {recipient}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert notification via email."""
        if not self.is_enabled():
            return False

        subject = alert.format_email_subject()
        body_html = alert.format_email_body()
        body_text = self._format_plain_text(alert)

        return await self.send_email(subject, body_html, body_text)

    def _format_plain_text(self, alert: Alert) -> str:
        """Format alert as plain text for email."""
        lines = [
            f"[{alert.severity.value.upper()}] {alert.title}",
            "-" * 40,
            "",
            f"Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if alert.container_name:
            lines.append(f"Container: {alert.container_name}")

        lines.extend([
            "",
            alert.message,
            "",
        ])

        if alert.details:
            lines.append("Details:")
            for key, value in alert.details.items():
                lines.append(f"  - {key}: {value}")

        lines.extend([
            "",
            "-" * 40,
            "TradeSystemV1 - System Monitor",
        ])

        return "\n".join(lines)

    async def send_daily_summary(self, system_health, alerts: List[dict]) -> bool:
        """Send a daily summary email."""
        if not self.is_enabled():
            return False

        subject = f"📊 TradeSystemV1 Daily Summary - {system_health.get_overall_status().title()}"

        # Build HTML email
        status_color = {
            "healthy": "#28a745",
            "warning": "#ffc107",
            "degraded": "#fd7e14",
            "critical": "#dc3545",
        }.get(system_health.get_overall_status(), "#6c757d")

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto;">
            <div style="background-color: {status_color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                <h1 style="margin: 0;">TradeSystemV1 Daily Summary</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">
                    Status: {system_health.get_status_emoji()} {system_health.get_overall_status().upper()}
                </p>
            </div>

            <div style="border: 1px solid #ddd; border-top: none; padding: 20px;">
                <h2 style="color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px;">
                    System Overview
                </h2>

                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">
                            <strong>Health Score</strong>
                        </td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">
                            <span style="font-size: 24px; font-weight: bold; color: {status_color};">
                                {system_health.health_score:.0f}%
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">🟢 Running Containers</td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">
                            {system_health.running_containers}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">🔴 Stopped Containers</td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">
                            {system_health.stopped_containers}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">⚠️ Warnings</td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">
                            {system_health.warning_containers}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">🔔 Active Alerts</td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">
                            {system_health.active_alerts}
                        </td>
                    </tr>
                </table>
        """

        # Container status table
        html += """
                <h2 style="color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px;">
                    Container Status
                </h2>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Container</th>
                        <th style="padding: 10px; text-align: center; border-bottom: 2px solid #ddd;">Status</th>
                        <th style="padding: 10px; text-align: right; border-bottom: 2px solid #ddd;">Uptime</th>
                        <th style="padding: 10px; text-align: right; border-bottom: 2px solid #ddd;">CPU</th>
                        <th style="padding: 10px; text-align: right; border-bottom: 2px solid #ddd;">Memory</th>
                    </tr>
        """

        for container in system_health.containers:
            status_emoji = container.get_status_emoji()
            html += f"""
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">
                            <strong>{container.name}</strong>
                            {"⭐" if container.is_critical else ""}
                        </td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #eee;">
                            {status_emoji} {container.status.value}
                        </td>
                        <td style="padding: 10px; text-align: right; border-bottom: 1px solid #eee;">
                            {container.uptime_human}
                        </td>
                        <td style="padding: 10px; text-align: right; border-bottom: 1px solid #eee;">
                            {container.metrics.cpu_percent:.1f}%
                        </td>
                        <td style="padding: 10px; text-align: right; border-bottom: 1px solid #eee;">
                            {container.metrics.memory_percent:.1f}%
                        </td>
                    </tr>
            """

        html += "</table>"

        # Recent alerts
        if alerts:
            html += """
                <h2 style="color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px;">
                    Recent Alerts (Last 24h)
                </h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Time</th>
                        <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Severity</th>
                        <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Alert</th>
                    </tr>
            """

            for alert in alerts[:10]:
                severity_color = {
                    "critical": "#dc3545",
                    "warning": "#ffc107",
                    "info": "#17a2b8",
                }.get(alert.get("severity", ""), "#6c757d")

                html += f"""
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">
                            {alert.get('created_at', 'N/A')[:19]}
                        </td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">
                            <span style="background-color: {severity_color}; color: white; padding: 2px 8px; border-radius: 3px;">
                                {alert.get('severity', 'unknown').upper()}
                            </span>
                        </td>
                        <td style="padding: 10px; border-bottom: 1px solid #eee;">
                            {alert.get('title', 'Unknown alert')}
                        </td>
                    </tr>
                """

            html += "</table>"

        html += """
            </div>
            <div style="text-align: center; padding: 20px; color: #6c757d; font-size: 12px;">
                <p>TradeSystemV1 - System Monitor</p>
                <p>This is an automated message. Please do not reply.</p>
            </div>
        </div>
        """

        return await self.send_email(subject, html)

    async def test_connection(self) -> tuple:
        """Test email connection. Returns (success, message)."""
        if not self.is_enabled():
            return False, "Email not configured or disabled"

        try:
            subject = "✅ TradeSystemV1 - Test Email"
            body_html = """
            <div style="font-family: Arial, sans-serif; padding: 20px;">
                <h2 style="color: #28a745;">Test Email Successful!</h2>
                <p>This is a test email from TradeSystemV1 System Monitor.</p>
                <p>Email notifications are working correctly.</p>
            </div>
            """

            success = await self.send_email(subject, body_html, "Test email from TradeSystemV1 System Monitor.")

            if success:
                return True, f"Test email sent to {settings.admin_email}"
            else:
                return False, "Failed to send test email"

        except Exception as e:
            return False, f"Connection failed: {e}"
