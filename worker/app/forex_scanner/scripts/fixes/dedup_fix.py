# direct_alert_fix.py
"""
Direct fix for AlertHistoryManager - manually add the missing code
This approach directly inserts the deduplication metadata extraction code
"""

import os
import re

def find_save_alert_method():
    """Find and display the save_alert method structure"""
    
    alert_history_path = '/app/forex_scanner/alerts/alert_history.py'
    
    try:
        with open(alert_history_path, 'r') as f:
            content = f.read()
        
        print("🔍 FINDING save_alert METHOD STRUCTURE")
        print("=" * 40)
        
        # Find the save_alert method
        save_alert_pattern = r'def save_alert\(self, signal: Dict.*?\n(.*?)cursor\.execute'
        match = re.search(save_alert_pattern, content, re.DOTALL)
        
        if match:
            method_content = match.group(1)
            lines = method_content.split('\n')
            
            print("📋 Current save_alert method structure:")
            for i, line in enumerate(lines[:20], 1):  # Show first 20 lines
                if line.strip():
                    print(f"  {i:2d}: {line}")
            
            if len(lines) > 20:
                print(f"  ... and {len(lines) - 20} more lines")
                
            return content, method_content
        else:
            print("❌ Could not find save_alert method")
            return content, None
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None, None

def apply_direct_fix():
    """Apply the direct fix for deduplication metadata"""
    
    alert_history_path = '/app/forex_scanner/alerts/alert_history.py'
    
    try:
        with open(alert_history_path, 'r') as f:
            content = f.read()
        
        print("\n🔧 APPLYING DIRECT FIX")
        print("-" * 25)
        
        # Look for the _extract_alert_data call and add our code right after it
        if '_extract_alert_data(signal, alert_message, alert_level)' in content:
            print("✅ Found _extract_alert_data call")
            
            # Add deduplication fields right after the _extract_alert_data call
            pattern = r'(alert_data = self\._extract_alert_data\(signal, alert_message, alert_level\))'
            
            replacement = r'''\1
            
            # Add deduplication metadata fields (DIRECT FIX)
            alert_data['signal_hash'] = signal.get('signal_hash')
            alert_data['data_source'] = signal.get('data_source', 'live_scanner')
            alert_data['market_timestamp'] = signal.get('market_timestamp')
            alert_data['cooldown_key'] = signal.get('cooldown_key')
            
            # Handle market_timestamp conversion if needed
            if isinstance(alert_data['market_timestamp'], str):
                try:
                    from datetime import datetime
                    alert_data['market_timestamp'] = datetime.fromisoformat(alert_data['market_timestamp'].replace('Z', '+00:00'))
                except:
                    alert_data['market_timestamp'] = None'''
            
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                print("✅ Added deduplication fields after _extract_alert_data")
            else:
                print("❌ Could not find _extract_alert_data pattern")
                return False
                
        # Alternative: look for any alert_data dictionary and add our fields
        elif 'alert_data = {' in content:
            print("✅ Found alert_data dictionary")
            
            # Find where alert_data is fully populated and add our fields
            pattern = r'(alert_data = \{[^}]+\})'
            
            def add_dedup_fields(match):
                original_dict = match.group(1)
                
                dedup_addition = '''
        
        # Add deduplication metadata fields (DIRECT FIX)
        alert_data['signal_hash'] = signal.get('signal_hash')
        alert_data['data_source'] = signal.get('data_source', 'live_scanner')
        alert_data['market_timestamp'] = signal.get('market_timestamp') 
        alert_data['cooldown_key'] = signal.get('cooldown_key')
        
        # Handle market_timestamp conversion if needed
        if isinstance(alert_data['market_timestamp'], str):
            try:
                from datetime import datetime
                alert_data['market_timestamp'] = datetime.fromisoformat(alert_data['market_timestamp'].replace('Z', '+00:00'))
            except:
                alert_data['market_timestamp'] = None'''
                
                return original_dict + dedup_addition
            
            content = re.sub(pattern, add_dedup_fields, content, flags=re.DOTALL)
            print("✅ Added deduplication fields to alert_data dictionary")
            
        # Last resort: add before cursor.execute
        elif 'cursor.execute' in content and 'INSERT INTO alert_history' in content:
            print("✅ Adding fields before cursor.execute (last resort)")
            
            # Find the cursor.execute line and add our code before it
            pattern = r'(\s+)(cursor\.execute\(.*?INSERT INTO alert_history)'
            
            def add_before_execute(match):
                indent = match.group(1)
                execute_stmt = match.group(2)
                
                dedup_code = f'''{indent}# Add deduplication metadata fields (DIRECT FIX)
{indent}if 'alert_data' not in locals():
{indent}    alert_data = {{}}
{indent}alert_data['signal_hash'] = signal.get('signal_hash')
{indent}alert_data['data_source'] = signal.get('data_source', 'live_scanner')
{indent}alert_data['market_timestamp'] = signal.get('market_timestamp')
{indent}alert_data['cooldown_key'] = signal.get('cooldown_key')
{indent}
{indent}# Handle market_timestamp conversion if needed
{indent}if isinstance(alert_data.get('market_timestamp'), str):
{indent}    try:
{indent}        from datetime import datetime
{indent}        alert_data['market_timestamp'] = datetime.fromisoformat(alert_data['market_timestamp'].replace('Z', '+00:00'))
{indent}    except:
{indent}        alert_data['market_timestamp'] = None
{indent}
{indent}{execute_stmt}'''
                
                return dedup_code
            
            content = re.sub(pattern, add_before_execute, content, flags=re.DOTALL)
            print("✅ Added deduplication fields before cursor.execute")
            
        else:
            print("❌ Could not find suitable location to add deduplication fields")
            return False
        
        # Write the fixed file
        with open(alert_history_path, 'w') as f:
            f.write(content)
        
        print("✅ Direct fix applied successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error applying direct fix: {e}")
        return False

def test_direct_fix():
    """Test the direct fix"""
    
    print("\n🧪 TESTING DIRECT FIX")
    print("-" * 20)
    
    try:
        import sys
        sys.path.insert(0, '/app')
        sys.path.insert(0, '/app/forex_scanner')
        
        from core.database import DatabaseManager
        from alerts.alert_history import AlertHistoryManager
        import config
        from datetime import datetime
        
        # Test signal with deduplication metadata
        test_signal = {
            'epic': 'CS.D.EURUSD.MINI.IP',
            'signal_type': 'BUY',
            'strategy': 'DIRECT_FIX_TEST',
            'price': 1.0950,
            'confidence_score': 0.85,
            'timeframe': '1H',
            'signal_hash': 'direct_fix_hash_12345',
            'cooldown_key': 'CS.D.EURUSD.MINI.IP:BUY:DIRECT_FIX_TEST', 
            'data_source': 'direct_fix_test',
            'market_timestamp': datetime.now().isoformat()
        }
        
        print(f"Testing with signal: {test_signal['epic']} {test_signal['signal_type']}")
        print(f"Expected metadata:")
        print(f"  signal_hash: {test_signal['signal_hash']}")
        print(f"  cooldown_key: {test_signal['cooldown_key']}")
        print(f"  data_source: {test_signal['data_source']}")
        
        # Initialize and test
        db_manager = DatabaseManager(config.DATABASE_URL)
        alert_manager = AlertHistoryManager(db_manager)
        
        alert_id = alert_manager.save_alert(test_signal, "Direct fix test alert", 'INFO')
        
        if alert_id:
            print(f"✅ Alert saved with ID: {alert_id}")
            
            # Verify metadata was saved
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT signal_hash, data_source, cooldown_key, market_timestamp
                FROM alert_history WHERE id = %s
            """, (alert_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                signal_hash, data_source, cooldown_key, market_timestamp = result
                
                print(f"📊 Saved metadata verification:")
                print(f"  signal_hash: {signal_hash} {'✅' if signal_hash else '❌'}")
                print(f"  data_source: {data_source} {'✅' if data_source else '❌'}")
                print(f"  cooldown_key: {cooldown_key} {'✅' if cooldown_key else '❌'}")
                print(f"  market_timestamp: {market_timestamp} {'✅' if market_timestamp else '❌'}")
                
                if signal_hash and cooldown_key and data_source:
                    print("\n🎉 DIRECT FIX SUCCESS!")
                    print("✅ All deduplication metadata is now being saved correctly!")
                    return True
                else:
                    print("\n❌ Some metadata is still missing")
                    return False
            else:
                print("❌ Could not retrieve saved alert")
                return False
        else:
            print("❌ Alert save failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing direct fix: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for direct fix"""
    
    print("🔧 DIRECT ALERT HISTORY MANAGER FIX")
    print("=" * 40)
    print("This will directly add deduplication metadata extraction to save_alert()")
    print()
    
    # Step 1: Analyze current structure
    content, method_content = find_save_alert_method()
    
    if not content:
        print("❌ Could not analyze AlertHistoryManager file")
        return False
    
    # Step 2: Apply direct fix
    if apply_direct_fix():
        print("\n✅ Direct fix applied successfully!")
    else:
        print("\n❌ Direct fix failed")
        return False
    
    # Step 3: Test the fix
    if test_direct_fix():
        print("\n🎉 COMPLETE SUCCESS!")
        print("The AlertHistoryManager is now saving deduplication metadata correctly!")
        
        print("\n📋 WHAT'S FIXED:")
        print("✅ signal_hash - for exact duplicate detection")
        print("✅ cooldown_key - for time-based deduplication")
        print("✅ data_source - for tracking alert origin")
        print("✅ market_timestamp - for accurate timing")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Restart your scanner")
        print("2. Run live scans")
        print("3. Verify deduplication is working in database")
        
        return True
    else:
        print("\n❌ Direct fix test failed")
        print("The fix was applied but may need manual adjustment")
        return False

if __name__ == "__main__":
    main()