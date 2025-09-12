import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json

try:
    import config
except ImportError:
    # Fallback config for standalone testing
    class Config:
        NOTIFICATIONS = {'file': False, 'email': False, 'webhook': False}
    config = Config()


class NotificationManager:
    """Enhanced notification manager with completely safe formatting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("📢 NotificationManager initialized")
    
    def send_signal_alert(self, signal: Dict[str, Any], message: Optional[str] = None, claude_decision: Optional[Dict] = None, executed: bool = False):
        """
        Enhanced signal alert with completely safe formatting
        
        Args:
            signal: Signal dictionary
            message: Optional custom message
            claude_decision: Optional Claude analysis results
            executed: Whether order was executed
        """
        try:
            # Build comprehensive message if not provided
            if message is None:
                message = self._build_enhanced_signal_message(signal, claude_decision, executed)
            
            # Console notification (always enabled)
            self._send_console_alert(signal, message, claude_decision, executed)
            
            # Additional notification channels
            self._send_optional_notifications(signal, message)
            
        except Exception as e:
            self.logger.error(f"❌ Error sending signal alert: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _build_enhanced_signal_message(self, signal: Dict[str, Any], claude_decision: Optional[Dict] = None, executed: bool = False) -> str:
        """Build comprehensive signal message with all context"""
        
        # Basic signal info - all safely extracted
        epic = str(signal.get('epic', 'Unknown'))
        signal_type = str(signal.get('signal_type', 'Unknown'))
        confidence = float(signal.get('confidence_score', 0))
        strategy = str(signal.get('strategy', 'Unknown'))
        
        # Convert confidence to percentage safely
        confidence_percentage = confidence * 100
        
        # Price information
        if 'price_mid' in signal:
            price_info = f"MID: {signal['price_mid']:.5f}"
            if 'execution_price' in signal:
                price_info += f", EXEC: {signal['execution_price']:.5f}"
        else:
            price_info = f"Price: {signal.get('price', 'N/A')}"
        
        # Claude analysis info
        claude_info = ""
        if claude_decision and isinstance(claude_decision, dict):
            if 'structured' in claude_decision:
                structured = claude_decision['structured']
                score = structured.get('signal_quality_score', 'N/A')
                decision = structured.get('trade_decision', 'N/A')
                claude_info = f"\n🤖 Claude: {score}/10 - {decision}"
            else:
                # Handle simple claude decision format
                score = claude_decision.get('score', 'N/A')
                decision = claude_decision.get('decision', 'N/A')
                claude_info = f"\n🤖 Claude: {score}/10 - {decision}"
        
        # Execution status
        execution_info = "\n💰 TRADE EXECUTED" if executed else ""
        
        # Build comprehensive message with safe formatting
        message = f"""📊 {signal_type} Signal: {epic}
Strategy: {strategy}
Confidence: {confidence_percentage:.1f}%
{price_info}{claude_info}{execution_info}"""
        
        return message
    
    def _send_console_alert(self, signal: Dict[str, Any], message: str, claude_decision: Optional[Dict] = None, executed: bool = False):
        """Send console alert with completely safe formatting"""
        
        # Extract basic info safely with explicit type conversion
        epic = str(signal.get('epic', 'Unknown'))
        signal_type = str(signal.get('signal_type', 'Unknown'))
        confidence = float(signal.get('confidence_score', 0))
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert confidence to percentage safely - this was the missing variable!
        confidence_percent = confidence * 100
        
        # Make message safe for f-string formatting
        safe_message = str(message).replace('{', '{{').replace('}', '}}')
        
        # Status line
        if executed:
            status_line = "EXECUTED ✅"
        elif claude_decision:
            if isinstance(claude_decision, dict):
                if claude_decision.get('structured', {}).get('trade_decision') == 'BUY':
                    status_line = "CLAUDE APPROVED ✅"
                else:
                    status_line = "CLAUDE REVIEWED ⚠️"
            else:
                status_line = "CLAUDE REVIEWED ⚠️"
        else:
            status_line = "DETECTED 🔍"
        
        # Safe message formatting - using variables that are actually defined
        alert_display = f"""
🚨 TRADING SIGNAL - {status_line} 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epic: {epic}
Signal: {signal_type}
Confidence: {confidence_percent:.1f}%
Time: {timestamp}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{safe_message}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        print(alert_display)
        self.logger.info(f"🚨 {signal_type} signal alert displayed for {epic}")
    
    def _send_optional_notifications(self, signal: Dict[str, Any], message: str):
        """Send additional notification channels if configured"""
        
        try:
            # File notification
            if hasattr(config, 'NOTIFICATIONS') and config.NOTIFICATIONS.get('file', False):
                self._send_file_alert(signal, message)
            
            # Email notification
            if hasattr(config, 'NOTIFICATIONS') and config.NOTIFICATIONS.get('email', False):
                self._send_email_alert(signal, message)
            
            # Webhook notification
            if hasattr(config, 'NOTIFICATIONS') and config.NOTIFICATIONS.get('webhook', False):
                self._send_webhook_alert(signal, message)
        except Exception as e:
            self.logger.debug(f"Optional notifications error: {e}")
    
    def _send_file_alert(self, signal: Dict[str, Any], message: str):
        """Enhanced file alert with safe formatting"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            epic = str(signal.get('epic', 'Unknown'))
            signal_type = str(signal.get('signal_type', 'Unknown'))
            confidence = float(signal.get('confidence_score', 0))
            strategy = str(signal.get('strategy', 'Unknown'))
            
            # Get price safely
            if 'price_mid' in signal:
                price = float(signal['price_mid'])
            else:
                price = float(signal.get('price', 0))
            
            # Safe CSV format - keep confidence as decimal
            safe_message = str(message).replace(',', ';').replace('\n', ' ')
            alert_line = f"{timestamp},{epic},{signal_type},{strategy},{confidence:.4f},{price:.5f},{safe_message}\n"
            
            # Append to signals file
            with open('trading_signals.csv', 'a') as f:
                # Write header if file is new
                if f.tell() == 0:
                    header = "timestamp,epic,signal_type,strategy,confidence,price,message\n"
                    f.write(header)
                f.write(alert_line)
            
            self.logger.info(f"📁 Signal saved to file: {epic}")
            
        except Exception as e:
            self.logger.error(f"❌ File alert failed: {e}")
    
    def _send_email_alert(self, signal: Dict[str, Any], message: str):
        """Email alert placeholder"""
        self.logger.debug("📧 Email notifications not implemented")
    
    def _send_webhook_alert(self, signal: Dict[str, Any], message: str):
        """Webhook alert placeholder"""
        self.logger.debug("🔗 Webhook notifications not implemented")
    
    def send_system_notification(self, message: str, level: str = "info"):
        """Send system notification with safe formatting"""
        level_emoji = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        
        # Get emoji safely
        emoji = level_emoji.get(level.lower(), 'ℹ️')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Make message safe - this was causing the message_safe error
        safe_message = str(message).replace('{', '{{').replace('}', '}}')
        level_upper = str(level).upper()
        
        system_message = f"""
{emoji} SYSTEM NOTIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Time: {timestamp}
Level: {level_upper}
Message: {safe_message}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        print(system_message)
        self.logger.info(f"{emoji} System notification: {safe_message}")
    
    def send_scanner_status(self, status: Dict[str, Any]):
        """Send scanner status update with safe formatting"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Extract status values safely
        running = bool(status.get('running', False))
        epic_count = int(status.get('epic_count', 0))
        scan_interval = int(status.get('scan_interval', 0))
        claude_enabled = bool(status.get('claude_enabled', False))
        bid_adjustment = bool(status.get('bid_adjustment', False))
        min_confidence = float(status.get('min_confidence', 0))
        recent_alerts_count = int(status.get('recent_alerts_count', 0))
        last_scan_time = str(status.get('last_scan_time', 'Never'))
        
        # Convert confidence to percentage
        min_confidence_percent = min_confidence * 100
        
        # Calculate uptime info if available
        uptime_info = ""
        if status.get('start_time'):
            uptime_info = f"Start Time: {status['start_time']}"
        
        status_message = f"""
📊 SCANNER STATUS UPDATE
━━━━━━━━━━━━━━━
Running: {running}
Epic Count: {epic_count}
Scan Interval: {scan_interval}s
Claude Enabled: {claude_enabled}
BID Adjustment: {bid_adjustment}
Min Confidence: {min_confidence_percent:.1f}%
{uptime_info}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recent Activity:
Total Alerts: {recent_alerts_count}
Last Scan: {last_scan_time}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        print(status_message)
        self.logger.info("📊 Scanner status update sent")
    
    def send_trade_execution_alert(self, signal: Dict[str, Any], execution_result: Dict[str, Any]):
        """Send trade execution specific alert with safe formatting"""
        
        # Extract values safely
        epic = str(signal.get('epic', 'Unknown'))
        signal_type = str(signal.get('signal_type', 'Unknown'))
        success = bool(execution_result.get('success', False))
        order_id = str(execution_result.get('order_id', 'N/A'))
        
        if success:
            execution_price = str(execution_result.get('execution_price', 'N/A'))
            size = str(execution_result.get('size', 'N/A'))
            
            message = f"""
💰 TRADE EXECUTED SUCCESSFULLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epic: {epic}
Signal: {signal_type}
Order ID: {order_id}
Execution Price: {execution_price}
Size: {size}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        else:
            error_msg = str(execution_result.get('message', 'Unknown error'))
            message = f"""
❌ TRADE EXECUTION FAILED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epic: {epic}
Signal: {signal_type}
Error: {error_msg}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        print(message)
        self.logger.info(f"💰 Trade execution alert sent for {epic}")
