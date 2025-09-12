#!/usr/bin/env python3
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
