#!/usr/bin/env python3
"""
Quick patch to fix AlertHistoryManager save_alert method
Run this once to fix the JSON/dict issue
"""

def patch_alert_history_manager():
    """Patch the AlertHistoryManager to fix the save_alert method"""
    
    alert_history_file = "/app/forex_scanner/alerts/alert_history.py"
    
    try:
        # Read the current file
        with open(alert_history_file, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if 'json.dumps(signal.get(' in content:
            print("✅ AlertHistoryManager already patched")
            return True
        
        # Find the problematic line and fix it
        # The issue is likely with JSON fields not being converted to strings
        
        # Add import at the top if not present
        if 'import json' not in content:
            content = content.replace('import logging', 'import json\nimport logging')
            print("✅ Added json import")
        
        # The specific fix for the save_alert method
        # Look for the line that's causing issues and replace it
        
        # This is a more targeted fix - we'll add JSON conversion
        json_field_fixes = [
            "strategy_config = json.dumps(signal.get('strategy_config', {})) if signal.get('strategy_config') else None",
            "strategy_indicators = json.dumps(signal.get('strategy_indicators', {})) if signal.get('strategy_indicators') else None", 
            "strategy_metadata = json.dumps(signal.get('strategy_metadata', {})) if signal.get('strategy_metadata') else None",
            "signal_conditions = json.dumps(signal.get('signal_conditions', {})) if signal.get('signal_conditions') else None"
        ]
        
        # Find where to insert the fixes (look for strategy_config usage)
        if 'strategy_config' in content and 'json.dumps' not in content:
            # Insert the JSON conversion code
            insert_point = content.find("strategy_config = signal.get('strategy_config'")
            if insert_point != -1:
                # Replace the problematic lines
                content = content.replace(
                    "strategy_config = signal.get('strategy_config')",
                    "strategy_config = json.dumps(signal.get('strategy_config', {})) if signal.get('strategy_config') else None"
                )
                content = content.replace(
                    "strategy_indicators = signal.get('strategy_indicators')", 
                    "strategy_indicators = json.dumps(signal.get('strategy_indicators', {})) if signal.get('strategy_indicators') else None"
                )
                content = content.replace(
                    "strategy_metadata = signal.get('strategy_metadata')",
                    "strategy_metadata = json.dumps(signal.get('strategy_metadata', {})) if signal.get('strategy_metadata') else None"
                )
                content = content.replace(
                    "signal_conditions = signal.get('signal_conditions')",
                    "signal_conditions = json.dumps(signal.get('signal_conditions', {})) if signal.get('signal_conditions') else None"
                )
                
                print("✅ Applied JSON conversion fixes")
        
        # Write the patched content back
        with open(alert_history_file, 'w') as f:
            f.write(content)
        
        print("✅ AlertHistoryManager patched successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to patch AlertHistoryManager: {e}")
        return False

def test_patched_manager():
    """Test the patched AlertHistoryManager"""
    import sys
    sys.path.append('/app/forex_scanner')
    
    print("\n🧪 Testing patched AlertHistoryManager...")
    
    try:
        # Force reload the module to get the patched version
        import importlib
        import alerts.alert_history
        importlib.reload(alerts.alert_history)
        
        from alerts.alert_history import AlertHistoryManager
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        alert_mgr = AlertHistoryManager(db)
        
        # Test with the same signal that failed before
        test_signal = {
            'epic': 'CS.D.AUDJPY.MINI.IP',
            'signal_type': 'BEAR',
            'confidence_score': 0.95,
            'strategy': 'macd_ema200_m',
            'price': 94.11600,
            'timestamp': '2025-06-30 12:30:00',
            'timeframe': '15m',
            'strategy_config': {'macd_fast': 12, 'macd_slow': 26},  # This caused the original error
            'strategy_indicators': {'ema_200': 94.0, 'macd_line': -0.001},
            'strategy_metadata': {'source': 'backtest_recreation'}
        }
        
        print("💾 Testing alert save with JSON data...")
        alert_id = alert_mgr.save_alert(test_signal, "Patched test alert")
        
        if alert_id:
            print(f"✅ SUCCESS! Alert saved with ID: {alert_id}")
            return True
        else:
            print("❌ Still failing to save alert")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 PATCHING ALERT HISTORY MANAGER")
    print("=" * 50)
    
    success = patch_alert_history_manager()
    
    if success:
        test_success = test_patched_manager()
        
        if test_success:
            print("\n🎉 PATCH SUCCESSFUL!")
            print("Your AlertHistoryManager should now save signals correctly.")
            print("Your live scanner will start saving alerts to the database.")
        else:
            print("\n⚠️ PATCH APPLIED BUT TEST FAILED")
            print("The file was patched but there may be other issues.")
    else:
        print("\n❌ PATCH FAILED")
        print("Manual intervention required.")