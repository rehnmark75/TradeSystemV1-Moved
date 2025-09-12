# fix_extract_alert_data.py
"""
Fix the _extract_alert_data method to include deduplication metadata
This addresses the missing metadata extraction that the previous script couldn't find
"""

import os
import re
import shutil
from datetime import datetime

def analyze_alert_history_file():
    """Analyze the AlertHistoryManager file to understand its structure"""
    
    alert_history_path = '/app/forex_scanner/alerts/alert_history.py'
    
    try:
        with open(alert_history_path, 'r') as f:
            content = f.read()
        
        print("🔍 ANALYZING ALERT HISTORY FILE STRUCTURE")
        print("=" * 45)
        
        # Look for method signatures
        methods = re.findall(r'def (\w+)\(', content)
        print(f"📋 Found methods: {', '.join(methods)}")
        
        # Check for _extract_alert_data specifically
        if '_extract_alert_data' in methods:
            print("✅ Found _extract_alert_data method")
        else:
            print("❌ _extract_alert_data method not found")
            
        # Look for where alert_data is populated
        alert_data_lines = []
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'alert_data[' in line or 'alert_data =' in line:
                alert_data_lines.append((i, line.strip()))
        
        print(f"\n📊 Found {len(alert_data_lines)} lines with alert_data:")
        for line_num, line in alert_data_lines[:10]:  # Show first 10
            print(f"  {line_num:3d}: {line}")
        
        if len(alert_data_lines) > 10:
            print(f"  ... and {len(alert_data_lines) - 10} more lines")
        
        # Look for the save_alert method structure
        save_alert_match = re.search(r'def save_alert\(.*?\n(.*?)cursor\.execute', content, re.DOTALL)
        if save_alert_match:
            save_alert_content = save_alert_match.group(1)
            print(f"\n📋 save_alert method structure (partial):")
            lines = save_alert_content.split('\n')[:15]  # First 15 lines
            for line in lines:
                if line.strip():
                    print(f"  {line}")
        
        return content, methods, alert_data_lines
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return None, [], []

def fix_alert_data_extraction():
    """Fix the alert data extraction to include deduplication metadata"""
    
    alert_history_path = '/app/forex_scanner/alerts/alert_history.py'
    
    try:
        with open(alert_history_path, 'r') as f:
            content = f.read()
        
        print("\n🔧 FIXING ALERT DATA EXTRACTION")
        print("-" * 35)
        
        # Strategy 1: Find where alert_data is created and add our fields
        if 'alert_data = {' in content:
            print("✅ Found alert_data dictionary creation")
            
            # Find the end of the alert_data dictionary and add our fields
            pattern = r'(alert_data = \{[^}]+\})'
            
            def add_dedup_fields(match):
                alert_data_dict = match.group(1)
                
                # Add deduplication fields before the closing brace
                dedup_fields = '''
        
        # Add deduplication metadata fields
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
                
                return alert_data_dict + dedup_fields
            
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, add_dedup_fields, content, flags=re.DOTALL)
                print("✅ Added deduplication fields to alert_data dictionary")
            
        # Strategy 2: Look for individual alert_data assignments and add ours
        elif "alert_data['epic']" in content or 'alert_data["epic"]' in content:
            print("✅ Found individual alert_data assignments")
            
            # Find the last alert_data assignment and add our fields after it
            lines = content.split('\n')
            last_alert_data_line = -1
            
            for i, line in enumerate(lines):
                if 'alert_data[' in line and '=' in line:
                    last_alert_data_line = i
            
            if last_alert_data_line >= 0:
                # Insert our deduplication fields after the last assignment
                dedup_lines = [
                    '',
                    '        # Add deduplication metadata fields',
                    "        alert_data['signal_hash'] = signal.get('signal_hash')",
                    "        alert_data['data_source'] = signal.get('data_source', 'live_scanner')",
                    "        alert_data['market_timestamp'] = signal.get('market_timestamp')",
                    "        alert_data['cooldown_key'] = signal.get('cooldown_key')",
                    '',
                    '        # Handle market_timestamp conversion if needed',
                    "        if isinstance(alert_data['market_timestamp'], str):",
                    '            try:',
                    '                from datetime import datetime',
                    "                alert_data['market_timestamp'] = datetime.fromisoformat(alert_data['market_timestamp'].replace('Z', '+00:00'))",
                    '            except:',
                    "                alert_data['market_timestamp'] = None"
                ]
                
                # Insert the new lines
                for j, new_line in enumerate(dedup_lines):
                    lines.insert(last_alert_data_line + 1 + j, new_line)
                
                content = '\n'.join(lines)
                print(f"✅ Added deduplication fields after line {last_alert_data_line + 1}")
        
        # Strategy 3: Find the cursor.execute line and add our fields before it
        elif 'cursor.execute' in content:
            print("✅ Found cursor.execute - adding fields before database insertion")
            
            pattern = r'(\s+)(cursor\.execute\(.*?INSERT INTO alert_history)'
            
            def add_before_execute(match):
                indent = match.group(1)
                execute_line = match.group(2)
                
                dedup_extraction = f'''{indent}# Add deduplication metadata fields
{indent}alert_data['signal_hash'] = signal.get('signal_hash')
{indent}alert_data['data_source'] = signal.get('data_source', 'live_scanner')
{indent}alert_data['market_timestamp'] = signal.get('market_timestamp')
{indent}alert_data['cooldown_key'] = signal.get('cooldown_key')
{indent}
{indent}# Handle market_timestamp conversion if needed
{indent}if isinstance(alert_data['market_timestamp'], str):
{indent}    try:
{indent}        from datetime import datetime
{indent}        alert_data['market_timestamp'] = datetime.fromisoformat(alert_data['market_timestamp'].replace('Z', '+00:00'))
{indent}    except:
{indent}        alert_data['market_timestamp'] = None
{indent}
{indent}{execute_line}'''
                
                return dedup_extraction
            
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, add_before_execute, content, flags=re.DOTALL)
                print("✅ Added deduplication fields before cursor.execute")
        
        else:
            print("❌ Could not find suitable location to add deduplication fields")
            return False
        
        # Write the fixed file
        with open(alert_history_path, 'w') as f:
            f.write(content)
        
        print("✅ Alert data extraction fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing alert data extraction: {e}")
        return False

def create_minimal_test():
    """Create a minimal test to verify the fix works"""
    
    print("\n🧪 CREATING MINIMAL TEST")
    print("-" * 25)
    
    test_code = '''
# Test the fixed AlertHistoryManager
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
    'strategy': 'MINIMAL_TEST',
    'price': 1.0950,
    'confidence_score': 0.85,
    'timeframe': '1H',
    'signal_hash': 'test_hash_12345',
    'cooldown_key': 'CS.D.EURUSD.MINI.IP:BUY:MINIMAL_TEST',
    'data_source': 'minimal_test',
    'market_timestamp': datetime.now().isoformat()
}

print("Testing AlertHistoryManager with metadata...")
print(f"Signal hash: {test_signal['signal_hash']}")
print(f"Cooldown key: {test_signal['cooldown_key']}")

# Initialize and test
db_manager = DatabaseManager(config.DATABASE_URL)
alert_manager = AlertHistoryManager(db_manager)

alert_id = alert_manager.save_alert(test_signal, "Minimal test alert", 'INFO')

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
        print(f"Saved metadata:")
        print(f"  signal_hash: {signal_hash} {'✅' if signal_hash else '❌'}")
        print(f"  data_source: {data_source} {'✅' if data_source else '❌'}")  
        print(f"  cooldown_key: {cooldown_key} {'✅' if cooldown_key else '❌'}")
        print(f"  market_timestamp: {market_timestamp} {'✅' if market_timestamp else '❌'}")
        
        if signal_hash and cooldown_key and data_source:
            print("🎉 SUCCESS! Deduplication metadata is working!")
        else:
            print("❌ Some metadata is still missing")
    else:
        print("❌ Could not retrieve saved alert")
else:
    print("❌ Alert save failed")
'''
    
    # Write test file
    with open('/app/test_minimal_fix.py', 'w') as f:
        f.write(test_code)
    
    print("✅ Created minimal test: /app/test_minimal_fix.py")
    print("\nTo run the test:")
    print("cd /app && python test_minimal_fix.py")

def main():
    """Main function to fix the alert data extraction"""
    
    print("🔧 FIXING ALERT DATA EXTRACTION")
    print("=" * 35)
    
    # Step 1: Analyze the file structure
    content, methods, alert_data_lines = analyze_alert_history_file()
    
    if not content:
        print("❌ Could not analyze AlertHistoryManager file")
        return False
    
    # Step 2: Fix the alert data extraction
    if fix_alert_data_extraction():
        print("\n✅ Alert data extraction fixed!")
    else:
        print("\n❌ Failed to fix alert data extraction")
        return False
    
    # Step 3: Create test
    create_minimal_test()
    
    print("\n📋 NEXT STEPS:")
    print("1. Run the minimal test: cd /app && python test_minimal_fix.py")
    print("2. If test passes, run the verification script: python verify_dedup_fix.py")
    print("3. Restart your scanner to use the fixed AlertHistoryManager")
    
    return True

if __name__ == "__main__":
    main()