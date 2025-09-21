
#!/usr/bin/env python3
"""
Comprehensive Notification Diagnostic Tool
Identifies why alerts are in database but not visible in logs
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import traceback

# Add the forex_scanner directory to path
sys.path.append('/app/forex_scanner')

def setup_test_logging():
    """Setup detailed logging for diagnostic"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('notification_diagnostic.log')
        ]
    )
    return logging.getLogger(__name__)

def check_database_alerts():
    """Check what's actually in the alert_history table"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CHECKING DATABASE ALERTS")
    
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        
        # Get total count
        total_count = db.execute_query("SELECT COUNT(*) as count FROM alert_history").iloc[0]['count']
        logger.info(f"📊 Total alerts in database: {total_count}")
        
        if total_count > 0:
            # Get recent alerts with full details
            recent_alerts = db.execute_query("""
                SELECT id, epic, signal_type, strategy, confidence_score, 
                       alert_timestamp, alert_message, alert_level
                FROM alert_history 
                ORDER BY alert_timestamp DESC 
                LIMIT 10
            """)
            
            logger.info("📋 Recent alerts:")
            for _, alert in recent_alerts.iterrows():
                timestamp = alert['alert_timestamp']
                logger.info(f"   ID {alert['id']}: {alert['epic']} {alert['signal_type']} "
                          f"({alert['strategy']}) - {alert['confidence_score']:.1%} "
                          f"at {timestamp}")
                if alert['alert_message']:
                    logger.info(f"      Message: {alert['alert_message']}")
            
            # Check alert distribution by timeframe
            logger.info("\\n📅 Alert distribution (last 7 days):")
            daily_counts = db.execute_query("""
                SELECT DATE(alert_timestamp) as date, COUNT(*) as count
                FROM alert_history 
                WHERE alert_timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY DATE(alert_timestamp)
                ORDER BY date DESC
            """)
            
            for _, day in daily_counts.iterrows():
                logger.info(f"   {day['date']}: {day['count']} alerts")
        
        return total_count > 0
        
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        traceback.print_exc()
        return False

def test_notification_manager():
    """Test the NotificationManager independently"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 TESTING NOTIFICATION MANAGER")
    
    try:
        from alerts.notifications import NotificationManager
        
        nm = NotificationManager()
        logger.info("✅ NotificationManager imported successfully")
        
        # Test signal
        test_signal = {
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'signal_type': 'BULL',
            'confidence_score': 0.85,
            'strategy': 'diagnostic_test',
            'price_mid': 1.0850,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("📤 Sending test notification...")
        nm.send_signal_alert(test_signal)
        logger.info("✅ Test notification sent - check if it appeared above")
        
        # Test system notification
        logger.info("📤 Sending test system notification...")
        nm.send_system_notification("Diagnostic test message", "info")
        logger.info("✅ System notification sent")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ NotificationManager test failed: {e}")
        traceback.print_exc()
        return False

def check_current_scanner_implementation():
    """Check how the current scanner handles notifications"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CHECKING CURRENT SCANNER IMPLEMENTATION")
    
    # Check if trade_scan.py exists and examine it
    scanner_files = [
        '/app/forex_scanner/trade_scan.py',
        '/app/forex_scanner/core/scanner.py',
        '/app/trade_scan.py'
    ]
    
    for file_path in scanner_files:
        if os.path.exists(file_path):
            logger.info(f"📄 Found scanner file: {file_path}")
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for notification-related code
                if 'NotificationManager' in content:
                    logger.info("   ✅ NotificationManager is imported")
                else:
                    logger.warning("   ⚠️ NotificationManager NOT found in imports")
                
                if 'send_signal_alert' in content:
                    logger.info("   ✅ send_signal_alert method is called")
                else:
                    logger.warning("   ⚠️ send_signal_alert method NOT called")
                
                if 'notification_manager' in content:
                    logger.info("   ✅ notification_manager instance found")
                else:
                    logger.warning("   ⚠️ notification_manager instance NOT found")
                
                # Look for signal processing logic
                if '_process_signal' in content:
                    logger.info("   ✅ _process_signal method found")
                else:
                    logger.warning("   ⚠️ _process_signal method NOT found")
                    
            except Exception as e:
                logger.error(f"   ❌ Error reading file: {e}")
        else:
            logger.info(f"   ❌ File not found: {file_path}")

def check_logging_configuration():
    """Check logging configuration"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CHECKING LOGGING CONFIGURATION")
    
    # Check current logging level
    root_logger = logging.getLogger()
    logger.info(f"📊 Root logger level: {logging.getLevelName(root_logger.level)}")
    
    # Check handlers
    handlers = root_logger.handlers
    logger.info(f"📊 Number of handlers: {len(handlers)}")
    
    for i, handler in enumerate(handlers):
        logger.info(f"   Handler {i}: {type(handler).__name__}")
        logger.info(f"   Level: {logging.getLevelName(handler.level)}")
        if hasattr(handler, 'stream'):
            logger.info(f"   Stream: {handler.stream}")
    
    # Test different log levels
    logger.info("📤 Testing log levels:")
    logger.debug("   This is a DEBUG message")
    logger.info("   This is an INFO message")
    logger.warning("   This is a WARNING message")
    logger.error("   This is an ERROR message")

def run_complete_diagnostic():
    """Run complete diagnostic suite"""
    logger = setup_test_logging()
    
    print("🔧 NOTIFICATION SYSTEM DIAGNOSTIC")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("Database Alerts", check_database_alerts),
        ("NotificationManager", test_notification_manager),
        ("Scanner Implementation", check_current_scanner_implementation),
        ("Logging Configuration", check_logging_configuration)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\\n🔍 Running {check_name} check...")
        try:
            result = check_func()
            results[check_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            results[check_name] = False
            print(f"   ❌ FAILED: {e}")
    
    # Summary
    print("\\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    
    for check_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
    
    # Recommendations
    print("\\n💡 RECOMMENDATIONS")
    print("=" * 20)
    
    if not results.get("Database Alerts"):
        print("1. ❌ No alerts in database - check signal detection")
    else:
        print("1. ✅ Alerts are being saved to database")
    
    if not results.get("NotificationManager"):
        print("2. ❌ NotificationManager broken - check imports")
    else:
        print("2. ✅ NotificationManager working")
    
    if results.get("Database Alerts") and not results.get("NotificationManager"):
        print("\\n🎯 ROOT CAUSE: NotificationManager not being called properly")
        print("   SOLUTION: Use the EnhancedTradeScanner in the artifact")
    
    if results.get("Database Alerts") and results.get("NotificationManager"):
        print("\\n🎯 ROOT CAUSE: Scanner not calling NotificationManager")
        print("   SOLUTION: Add notification_manager.send_signal_alert() to signal processing")
    
    print("\\n🚀 NEXT STEPS:")
    print("1. Replace your current scanner with EnhancedTradeScanner")
    print("2. Ensure logging level is INFO or DEBUG")
    print("3. Test with: scanner.test_notification_system()")
    print("4. Run live scanning and monitor logs")

if __name__ == "__main__":
    run_complete_diagnostic()