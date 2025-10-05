#!/usr/bin/env python3
"""
Debug script to check why signals aren't being saved to alert_history
"""

import sys
sys.path.append('/app/forex_scanner')

def check_alert_history_table():
    """Check if alert_history table exists and has data"""
    print("🔍 Checking alert_history table...")
    
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        
        # Check if table exists
        table_check = db.execute_query("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'alert_history'
        """)
        
        if len(table_check) > 0:
            print("✅ alert_history table exists")
            
            # Check table structure
            columns = db.execute_query("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'alert_history'
                ORDER BY ordinal_position
            """)
            
            print(f"📋 Table has {len(columns)} columns:")
            for _, row in columns.iterrows():
                print(f"   {row['column_name']}: {row['data_type']}")
            
            # Check for any existing data
            count = db.execute_query("SELECT COUNT(*) as count FROM alert_history")
            total_alerts = count.iloc[0]['count']
            print(f"📊 Total alerts in database: {total_alerts}")
            
            if total_alerts > 0:
                recent = db.execute_query("""
                    SELECT epic, signal_type, confidence_score, strategy, alert_timestamp 
                    FROM alert_history 
                    ORDER BY alert_timestamp DESC 
                    LIMIT 5
                """)
                print("📅 Recent alerts:")
                print(recent.to_string(index=False))
            else:
                print("📭 No alerts found in database")
                
        else:
            print("❌ alert_history table does not exist!")
            print("   You may need to run the database setup script")
            
    except Exception as e:
        print(f"❌ Error checking alert_history: {e}")
        import traceback
        traceback.print_exc()

def check_alert_history_manager():
    """Check if AlertHistoryManager is working"""
    print("\n🔧 Testing AlertHistoryManager...")
    
    try:
        from alerts.alert_history import AlertHistoryManager
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        alert_manager = AlertHistoryManager(db)
        
        print("✅ AlertHistoryManager created successfully")
        
        # Test saving a dummy signal
        test_signal = {
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'signal_type': 'BULL',
            'confidence_score': 0.75,
            'strategy': 'test_strategy',
            'timestamp': '2025-06-30 13:00:00',
            'price': 1.0850
        }
        
        print("🧪 Testing alert save...")
        alert_id = alert_manager.save_alert(test_signal, "Test alert")
        
        if alert_id:
            print(f"✅ Test alert saved with ID: {alert_id}")
            
            # Verify it was saved
            verification = db.execute_query(f"""
                SELECT * FROM alert_history WHERE id = {alert_id}
            """)
            
            if len(verification) > 0:
                print("✅ Alert verified in database")
            else:
                print("❌ Alert not found in database after save")
        else:
            print("❌ Failed to save test alert")
            
    except Exception as e:
        print(f"❌ Error testing AlertHistoryManager: {e}")
        import traceback
        traceback.print_exc()

def check_live_scanner_integration():
    """Check if live scanner is using AlertHistoryManager"""
    print("\n🔍 Checking live scanner integration...")
    
    try:
        from core.scanner import IntelligentForexScanner
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        
        # Check if scanner has alert_history attribute
        scanner = IntelligentForexScanner(
            db_manager=db,
            epic_list=['CS.D.EURUSD.CEEM.IP'],
            enable_claude_analysis=False
        )
        
        if hasattr(scanner, 'alert_history'):
            print("✅ Scanner has alert_history manager")
            print(f"   Type: {type(scanner.alert_history)}")
        else:
            print("❌ Scanner missing alert_history manager")
            print("   This is why signals aren't being saved!")
            
        # Check if scanner's _process_signal method saves alerts
        import inspect
        if hasattr(scanner, '_process_signal'):
            source = inspect.getsource(scanner._process_signal)
            if 'alert_history' in source or 'save_alert' in source:
                print("✅ Scanner._process_signal includes alert saving")
            else:
                print("❌ Scanner._process_signal does NOT save alerts")
                print("   This needs to be fixed!")
        else:
            print("⚠️ Scanner has no _process_signal method")
            
    except Exception as e:
        print(f"❌ Error checking scanner integration: {e}")
        import traceback
        traceback.print_exc()

def check_backtest_vs_live_difference():
    """Check difference between backtest and live scanning"""
    print("\n🔄 Checking backtest vs live scanning...")
    
    try:
        # Check backtest path
        from commands.backtest_commands import BacktestCommands
        print("✅ BacktestCommands accessible")
        
        # Check if backtest saves to alert_history
        import inspect
        source = inspect.getsource(BacktestCommands)
        if 'alert_history' in source.lower():
            print("✅ BacktestCommands mentions alert_history")
        else:
            print("❌ BacktestCommands does NOT use alert_history")
            print("   Backtest signals are not saved to database")
            
        # Check live scanner
        from core.scanner import IntelligentForexScanner
        scanner_source = inspect.getsource(IntelligentForexScanner)
        if 'alert_history' in scanner_source.lower():
            print("✅ IntelligentForexScanner mentions alert_history")
        else:
            print("❌ IntelligentForexScanner does NOT use alert_history")
            print("   Live signals are not saved to database")
            
    except Exception as e:
        print(f"❌ Error checking backtest vs live: {e}")

def get_signal_processing_status():
    """Check what happens when scanner processes signals"""
    print("\n🎯 Checking signal processing flow...")
    
    try:
        from core.scanner import IntelligentForexScanner
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        scanner = IntelligentForexScanner(
            db_manager=db,
            epic_list=['CS.D.EURUSD.CEEM.IP'],
            enable_claude_analysis=False
        )
        
        # Run a quick scan
        print("🔍 Running test scan...")
        signals = scanner.scan_once()
        
        print(f"📊 Scan found {len(signals)} signals")
        
        if signals:
            print("✅ Signals found - checking if they're being processed...")
            
            # Check if any alerts were added to database during this scan
            count_before = db.execute_query("SELECT COUNT(*) as count FROM alert_history").iloc[0]['count']
            
            # Process one signal manually to test
            if hasattr(scanner, '_process_signal'):
                print("🔄 Testing signal processing...")
                scanner._process_signal(signals[0])
                
                count_after = db.execute_query("SELECT COUNT(*) as count FROM alert_history").iloc[0]['count']
                
                if count_after > count_before:
                    print(f"✅ Signal was saved! Count: {count_before} → {count_after}")
                else:
                    print(f"❌ Signal was NOT saved. Count unchanged: {count_before}")
            else:
                print("⚠️ Scanner has no _process_signal method")
        else:
            print("📭 No signals found in test scan")
            
    except Exception as e:
        print(f"❌ Error checking signal processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 ALERT HISTORY DEBUG SCRIPT")
    print("=" * 80)
    
    check_alert_history_table()
    check_alert_history_manager() 
    check_live_scanner_integration()
    check_backtest_vs_live_difference()
    get_signal_processing_status()
    
    print("\n" + "=" * 80)
    print("🎯 SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print("If alerts aren't being saved, it's likely because:")
    print("1. Live scanner isn't calling alert_history.save_alert()")
    print("2. Backtest doesn't save to alert_history (by design)")
    print("3. AlertHistoryManager isn't integrated into signal processing")
    print("\nRun this script to identify the exact issue!")