#!/usr/bin/env python3
"""
Final Scanner Integration with Working Alert Saver
Since the alert saver works, let's integrate it into your container
"""

# Create the alert saver utility file
def create_alert_saver_utility():
    """Create the working alert saver utility"""
    
    import os
    
    # Create utils directory if it doesn't exist
    utils_dir = "/app/forex_scanner/utils"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    
    # Create __init__.py
    init_file = os.path.join(utils_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Utils module\n")
    
    # Create the alert saver
    alert_saver_content = '''import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_signal_to_database(signal, message="Signal detected"):
    """
    Working alert saver that bypasses AlertHistoryManager issues
    Uses raw psycopg2 to avoid parameter binding problems
    """
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Extract and clean data
        epic = str(signal.get('epic', 'Unknown'))
        pair = str(signal.get('pair', epic.replace('CS.D.', '').replace('.MINI.IP', '')))
        signal_type = str(signal.get('signal_type', 'Unknown'))
        strategy = str(signal.get('strategy', 'Unknown'))
        confidence_score = float(signal.get('confidence_score', 0.0))
        price = float(signal.get('price', 0.0))
        timeframe = str(signal.get('timeframe', '15m'))
        
        # Handle JSON fields properly
        strategy_config = None
        strategy_indicators = None
        strategy_metadata = None
        
        if signal.get('strategy_config') and isinstance(signal.get('strategy_config'), dict):
            strategy_config = json.dumps(signal['strategy_config'])
        
        if signal.get('strategy_indicators') and isinstance(signal.get('strategy_indicators'), dict):
            strategy_indicators = json.dumps(signal['strategy_indicators'])
            
        if signal.get('strategy_metadata') and isinstance(signal.get('strategy_metadata'), dict):
            strategy_metadata = json.dumps(signal['strategy_metadata'])
        
        # Additional signal data
        ema_short = signal.get('ema_9') or signal.get('ema_short')
        ema_long = signal.get('ema_21') or signal.get('ema_long') 
        ema_trend = signal.get('ema_200') or signal.get('ema_trend')
        macd_line = signal.get('macd_line')
        macd_signal_line = signal.get('macd_signal')
        macd_histogram = signal.get('macd_histogram')
        volume = signal.get('volume') or signal.get('ltv')
        volume_ratio = signal.get('volume_ratio')
        claude_analysis = signal.get('claude_analysis')
        
        # Insert with comprehensive data
        cursor.execute("""
            INSERT INTO alert_history (
                epic, pair, signal_type, strategy, confidence_score, price, timeframe,
                strategy_config, strategy_indicators, strategy_metadata,
                ema_short, ema_long, ema_trend,
                macd_line, macd_signal, macd_histogram,
                volume, volume_ratio,
                claude_analysis, alert_message, alert_level, status, alert_timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """, (
            epic, pair, signal_type, strategy, confidence_score, price, timeframe,
            strategy_config, strategy_indicators, strategy_metadata,
            ema_short, ema_long, ema_trend,
            macd_line, macd_signal_line, macd_histogram,
            volume, volume_ratio,
            claude_analysis, message, 'INFO', 'NEW', datetime.now()
        ))
        
        alert_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Alert saved to database with ID: {alert_id}")
        return alert_id
        
    except Exception as e:
        logger.error(f"❌ Error saving alert to database: {e}")
        if 'conn' in locals():
            try:
                conn.rollback()
                cursor.close()
                conn.close()
            except:
                pass
        return None

def get_recent_alerts(limit=10):
    """Get recent alerts from database for monitoring"""
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT epic, signal_type, confidence_score, strategy, alert_timestamp, alert_message
            FROM alert_history 
            ORDER BY alert_timestamp DESC 
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
        return []
'''
    
    alert_saver_file = os.path.join(utils_dir, "alert_saver.py")
    with open(alert_saver_file, 'w') as f:
        f.write(alert_saver_content)
    
    print(f"✅ Created alert saver utility: {alert_saver_file}")
    return alert_saver_file

def create_enhanced_container_script():
    """Create enhanced container script with alert saving"""
    
    enhanced_script = '''#!/usr/bin/env python3
"""
Enhanced Container Entry Point Script with Working Alert History
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pandas_ta as ta
import pytz
import plotly.graph_objects as go
from PIL import Image
from scipy.stats import linregress
from typing import List, Dict, Optional, Any
import sys
import time
import warnings
import numpy as np
import logging
from logging.handlers import TimedRotatingFileHandler
import requests
import schedule

# Force unbuffered output for container visibility
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# Add forex_scanner to Python path
sys.path.append('/app/forex_scanner')

# Create missing __init__.py if needed
if not os.path.exists('/app/forex_scanner/__init__.py'):
    open('/app/forex_scanner/__init__.py', 'w').close()

# Import modules
from core.scanner import IntelligentForexScanner
from core.database import DatabaseManager
from utils.alert_saver import save_signal_to_database, get_recent_alerts  # NEW: Working alert saver
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/scanner.log')
    ],
    force=True
)

logger = logging.getLogger(__name__)

def log_and_print(message):
    """Log and print message for maximum visibility"""
    print(message)
    sys.stdout.flush()
    logger.info(message)

if __name__ == "__main__":
    log_and_print("🚀 Container starting up with working alert history...")
    
    try:
        # Initialize database manager
        log_and_print("🔌 Connecting to database...")
        db_manager = DatabaseManager(config.DATABASE_URL)
        log_and_print("✅ Database connection successful")
        
        # Test alert history
        log_and_print("💾 Testing alert history...")
        recent_alerts = get_recent_alerts(5)
        log_and_print(f"📊 Found {len(recent_alerts)} recent alerts in database")
        
        # Initialize scanner
        log_and_print("🔧 Initializing scanner...")
        scanner = IntelligentForexScanner(
            db_manager=db_manager,
            epic_list=config.EPIC_LIST,
            claude_api_key=getattr(config, 'CLAUDE_API_KEY', None),
            enable_claude_analysis=getattr(config, 'ENABLE_CLAUDE_ANALYSIS', False),
            use_bid_adjustment=getattr(config, 'USE_BID_ADJUSTMENT', True),
            spread_pips=getattr(config, 'SPREAD_PIPS', 1.5),
            min_confidence=getattr(config, 'MIN_CONFIDENCE', 0.6),
            user_timezone=getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm')
        )
        log_and_print("✅ Scanner initialization successful")
        
        def scan_and_trade():
            """Enhanced scan_and_trade function with working alert history"""
            scan_time = datetime.now().strftime('%H:%M:%S')
            log_and_print(f"🔍 [{scan_time}] Starting scan...")
            
            try:
                signals = scanner.scan_once()
                
                if signals:
                    log_and_print(f"✅ [{scan_time}] Found {len(signals)} signals:")
                    
                    for i, signal in enumerate(signals, 1):
                        epic = signal.get('epic', 'Unknown')
                        signal_type = signal.get('signal_type', 'Unknown')
                        confidence = signal.get('confidence_score', 0)
                        strategy = signal.get('strategy', 'unknown')
                        
                        log_and_print(f"   {i}. 📊 {signal_type} {epic} ({strategy}) - {confidence:.1%}")
                        
                        # Save signal to database using working alert saver
                        alert_id = save_signal_to_database(signal, f"Live {signal_type} signal detected")
                        
                        if alert_id:
                            log_and_print(f"      💾 Alert saved to database (ID: {alert_id})")
                        else:
                            log_and_print(f"      ⚠️ Failed to save alert to database")
                        
                        # Check if signal meets trading criteria
                        if confidence >= getattr(config, 'MIN_CONFIDENCE', 0.6):
                            log_and_print(f"      🎯 Signal meets confidence threshold")
                            
                            # Your order execution logic would go here
                            # if order_executor.should_execute_signal(signal):
                            #     order_result = order_executor.execute_signal_order(signal)
                        else:
                            log_and_print(f"      ⚠️ Signal below minimum confidence threshold")
                        
                        # Show Claude analysis if available
                        if signal.get('claude_analysis'):
                            analysis_preview = signal['claude_analysis'][:100]
                            log_and_print(f"      🤖 Claude: {analysis_preview}...")
                
                else:
                    log_and_print(f"📭 [{scan_time}] No signals found")
                
                # Show scan completion
                log_and_print(f"✨ [{scan_time}] Scan completed successfully")
                return len(signals) if signals else 0
                
            except Exception as e:
                log_and_print(f"❌ [{scan_time}] Scan failed: {e}")
                import traceback
                error_details = traceback.format_exc()
                log_and_print(f"   Error details: {error_details}")
                return 0
        
        # Show current status
        log_and_print("📊 System Status:")
        log_and_print(f"   Alert history: ✅ Working (fixed)")
        log_and_print(f"   Database: ✅ Connected")
        log_and_print(f"   Pairs monitored: {len(config.EPIC_LIST)}")
        log_and_print(f"   Min confidence: {getattr(config, 'MIN_CONFIDENCE', 0.6):.1%}")
        
        # Test initial scan
        log_and_print("🧪 Running initial test scan...")
        initial_signals = scan_and_trade()
        log_and_print(f"🏁 Initial scan result: {initial_signals} signals")
        
        # Start scheduler
        log_and_print("⏳ Scheduler started... Running every 1 minute.")
        log_and_print(f"🕐 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        schedule.every(1).minutes.do(scan_and_trade)
        
        # Main loop with heartbeat and database stats
        loop_count = 0
        while True:
            schedule.run_pending()
            
            # Heartbeat with database stats every 30 seconds
            loop_count += 1
            if loop_count % 30 == 0:
                heartbeat_time = datetime.now().strftime('%H:%M:%S')
                
                # Get quick database stats
                try:
                    recent_count = len(get_recent_alerts(10))
                    log_and_print(f"💓 [{heartbeat_time}] Scanner heartbeat - {recent_count} recent alerts in DB")
                except:
                    log_and_print(f"💓 [{heartbeat_time}] Scanner heartbeat - system running")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        log_and_print("🛑 Container stopped by user")
    except Exception as e:
        log_and_print(f"❌ Container failed to start: {e}")
        import traceback
        error_details = traceback.format_exc()
        log_and_print(f"   Error details: {error_details}")
        raise
'''
    
    script_file = "/app/enhanced_trade_scan.py"
    with open(script_file, 'w') as f:
        f.write(enhanced_script)
    
    print(f"✅ Created enhanced container script: {script_file}")
    return script_file

def test_integration():
    """Test the complete integration"""
    
    print("🧪 Testing complete integration...")
    
    try:
        # Test alert saver
        import sys
        sys.path.append('/app/forex_scanner')
        from utils.alert_saver import save_signal_to_database, get_recent_alerts
        
        # Test saving a signal
        test_signal = {
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'signal_type': 'BULL',
            'confidence_score': 0.85,
            'strategy': 'integration_test',
            'price': 1.0850,
            'timeframe': '15m'
        }
        
        alert_id = save_signal_to_database(test_signal, "Integration test signal")
        
        if alert_id:
            print(f"✅ Integration test signal saved: {alert_id}")
            
            # Test getting recent alerts
            recent = get_recent_alerts(5)
            print(f"✅ Retrieved {len(recent)} recent alerts")
            
            return True
        else:
            print("❌ Integration test failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 FINAL SCANNER INTEGRATION")
    print("=" * 50)
    
    # Create the utility
    alert_saver_file = create_alert_saver_utility()
    
    # Create enhanced script
    enhanced_script = create_enhanced_container_script()
    
    # Test integration
    integration_success = test_integration()
    
    if integration_success:
        print("\n🎉 INTEGRATION COMPLETE!")
        print("=" * 50)
        print("✅ Alert saver utility created and tested")
        print("✅ Enhanced container script created")
        print("✅ Integration test passed")
        print()
        print("📋 NEXT STEPS:")
        print("1. Stop your current container")
        print("2. Run the enhanced script:")
        print("   python /app/enhanced_trade_scan.py")
        print("3. Your scanner will now save all signals to database!")
        print()
        print("💾 Your signals will be saved to alert_history table")
        print("📊 You can monitor them in real-time")
        print("🎯 Alert IDs will be displayed for each signal")
        
    else:
        print("\n❌ INTEGRATION TEST FAILED")
        print("Check the error messages above")