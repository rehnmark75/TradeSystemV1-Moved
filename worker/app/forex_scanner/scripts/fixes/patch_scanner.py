#!/usr/bin/env python3
"""
Quick patch script to fix alert history saving in scanner
This will modify your existing scanner.py to save alerts properly
"""

import os
import shutil
import re
from datetime import datetime

def backup_existing_scanner():
    """Create backup of existing scanner.py"""
    scanner_path = "/app/forex_scanner/core/scanner.py"
    backup_path = f"/app/forex_scanner/core/scanner.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if os.path.exists(scanner_path):
        shutil.copy2(scanner_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
        return True
    else:
        print(f"❌ Scanner file not found: {scanner_path}")
        return False

def patch_scanner_imports():
    """Add necessary imports to scanner.py"""
    scanner_path = "/app/forex_scanner/core/scanner.py"
    
    try:
        with open(scanner_path, 'r') as f:
            content = f.read()
        
        # Check if AlertHistoryManager import exists
        if 'from alerts.alert_history import AlertHistoryManager' not in content:
            # Find a good place to add the import
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from alerts.notifications import NotificationManager'):
                    lines.insert(i + 1, 'from alerts.alert_history import AlertHistoryManager')
                    break
            content = '\n'.join(lines)
            print("✅ Added AlertHistoryManager import")
        
        # Write back
        with open(scanner_path, 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error patching imports: {e}")
        return False

def patch_scanner_init():
    """Add alert_history initialization to __init__ method"""
    scanner_path = "/app/forex_scanner/core/scanner.py"
    
    try:
        with open(scanner_path, 'r') as f:
            content = f.read()
        
        # Check if alert_history is already initialized
        if 'self.alert_history = AlertHistoryManager(db_manager)' not in content:
            # Find the notification_manager initialization
            pattern = r'(self\.notification_manager = NotificationManager\(\))'
            replacement = r'\1\n        \n        # Initialize alert history manager\n        self.alert_history = AlertHistoryManager(db_manager)'
            
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                print("✅ Added alert_history initialization")
            else:
                print("⚠️ Could not find notification_manager initialization pattern")
        
        # Write back
        with open(scanner_path, 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error patching init: {e}")
        return False

def patch_process_signal_method():
    """Add alert saving to _process_signal method"""
    scanner_path = "/app/forex_scanner/core/scanner.py"
    
    try:
        with open(scanner_path, 'r') as f:
            content = f.read()
        
        # Look for the _process_signal method and add alert saving
        if 'alert_id = self._save_signal(signal)' not in content:
            # Find where to insert the save_signal call
            pattern = r'(self\.logger\.info\(f"   Price: \{signal\.get\(\'price\', \'N/A\'\)\}"\))'
            replacement = r'''\1
        
        # Save signal to alert_history table
        alert_id = self._save_signal(signal)'''
            
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                print("✅ Added alert saving to _process_signal method")
            else:
                print("⚠️ Could not find price logging pattern - trying alternative approach")
                # Try alternative pattern
                alt_pattern = r'(self\.logger\.info\(f"📊 Processing \{signal_type\} signal for \{epic\}"\))'
                alt_replacement = r'''\1
        
        # Save signal to alert_history table
        alert_id = self._save_signal(signal)'''
        
                if re.search(alt_pattern, content):
                    content = re.sub(alt_pattern, alt_replacement, content)
                    print("✅ Added alert saving using alternative pattern")
        
        # Write back
        with open(scanner_path, 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error patching _process_signal: {e}")
        return False

def add_save_signal_method():
    """Add the _save_signal method to the scanner class"""
    scanner_path = "/app/forex_scanner/core/scanner.py"
    
    try:
        with open(scanner_path, 'r') as f:
            content = f.read()
        
        # Check if _save_signal method already exists
        if 'def _save_signal(self, signal: Dict)' not in content:
            # Find a good place to add the method (before _execute_order method)
            save_signal_method = '''
    def _save_signal(self, signal: Dict) -> Optional[int]:
        """
        Save signal to alert_history table
        
        Args:
            signal: Signal dictionary to save
            
        Returns:
            Alert ID if successful, None if failed
        """
        try:
            alert_message = f"Scanner signal: {signal['signal_type']} {signal['epic']} @ {signal['confidence_score']:.1%}"
            
            alert_id = self.alert_history.save_alert(
                signal,
                alert_message,
                alert_level='INFO'
            )
            
            if alert_id:
                self.logger.info(f"💾 Signal saved to alert_history (ID: {alert_id})")
                return alert_id
            else:
                self.logger.warning(f"⚠️ Failed to save signal to alert_history")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving signal: {e}")
            return None
'''
            
            # Find the _execute_order method and insert before it
            pattern = r'(\s+def _execute_order\(self, signal: Dict\):)'
            replacement = save_signal_method + r'\1'
            
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                print("✅ Added _save_signal method")
            else:
                # Alternative: add before get_scanner_status method
                alt_pattern = r'(\s+def get_scanner_status\(self\) -> Dict:)'
                if re.search(alt_pattern, content):
                    content = re.sub(alt_pattern, save_signal_method + r'\1', content)
                    print("✅ Added _save_signal method (alternative position)")
                else:
                    print("⚠️ Could not find suitable position for _save_signal method")
                    return False
        
        # Write back
        with open(scanner_path, 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding _save_signal method: {e}")
        return False

def create_backup_alert_saver():
    """Create the backup alert saver utility file"""
    utils_dir = "/app/forex_scanner/utils"
    
    # Ensure utils directory exists
    os.makedirs(utils_dir, exist_ok=True)
    
    backup_saver_content = '''# utils/backup_alert_saver.py
"""
Backup Alert Saver - Direct database insertion for AlertHistoryManager issues
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def save_signal_to_database_direct(signal: Dict, message: str = "Signal detected") -> Optional[int]:
    """Direct database insertion bypassing potential SQLAlchemy issues"""
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Extract basic signal data
        epic = str(signal.get('epic', 'Unknown'))
        pair = str(signal.get('pair', epic.replace('CS.D.', '').replace('.MINI.IP', '')))
        signal_type = str(signal.get('signal_type', 'Unknown'))
        strategy = str(signal.get('strategy', 'ema_strategy'))
        confidence_score = float(signal.get('confidence_score', 0.0))
        price = float(signal.get('price', 0.0))
        timeframe = str(signal.get('timeframe', '15m'))
        
        # Handle JSON fields safely
        strategy_config = json.dumps(signal.get('strategy_config', {})) if signal.get('strategy_config') else None
        strategy_indicators = json.dumps(signal.get('strategy_indicators', {})) if signal.get('strategy_indicators') else None
        strategy_metadata = json.dumps(signal.get('strategy_metadata', {})) if signal.get('strategy_metadata') else None
        
        # Insert essential fields
        cursor.execute("""
            INSERT INTO alert_history (
                epic, pair, signal_type, strategy, confidence_score, price, timeframe,
                strategy_config, strategy_indicators, strategy_metadata,
                alert_message, alert_level, status, alert_timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """, (
            epic, pair, signal_type, strategy, confidence_score, price, timeframe,
            strategy_config, strategy_indicators, strategy_metadata,
            message, 'INFO', 'NEW', datetime.now()
        ))
        
        alert_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Alert saved directly to database (ID: {alert_id})")
        return alert_id
        
    except Exception as e:
        logger.error(f"❌ Direct save failed: {e}")
        if 'conn' in locals():
            try:
                conn.rollback()
                cursor.close()
                conn.close()
            except:
                pass
        return None
'''
    
    backup_saver_file = os.path.join(utils_dir, "backup_alert_saver.py")
    with open(backup_saver_file, 'w') as f:
        f.write(backup_saver_content)
    
    print(f"✅ Created backup alert saver: {backup_saver_file}")
    return backup_saver_file

def add_fallback_option():
    """Add fallback alert saving option to scanner"""
    scanner_path = "/app/forex_scanner/core/scanner.py"
    
    try:
        with open(scanner_path, 'r') as f:
            content = f.read()
        
        # Add backup import
        if 'from utils.backup_alert_saver import save_signal_to_database_direct' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from alerts.alert_history import AlertHistoryManager'):
                    lines.insert(i + 1, 'from utils.backup_alert_saver import save_signal_to_database_direct')
                    break
            content = '\n'.join(lines)
            print("✅ Added backup alert saver import")
        
        # Modify _save_signal method to include fallback
        if 'save_signal_to_database_direct' not in content or 'except Exception as e:' in content:
            # Replace the existing _save_signal method with enhanced version
            old_method_pattern = r'def _save_signal\(self, signal: Dict\) -> Optional\[int\]:.*?except Exception as e:\s*self\.logger\.error\(f"❌ Error saving signal: \{e\}"\)\s*return None'
            
            new_method = '''def _save_signal(self, signal: Dict) -> Optional[int]:
        """
        Save signal to alert_history table with fallback option
        
        Args:
            signal: Signal dictionary to save
            
        Returns:
            Alert ID if successful, None if failed
        """
        try:
            alert_message = f"Scanner signal: {signal['signal_type']} {signal['epic']} @ {signal['confidence_score']:.1%}"
            
            # Try primary method first
            alert_id = self.alert_history.save_alert(
                signal,
                alert_message,
                alert_level='INFO'
            )
            
            if alert_id:
                self.logger.info(f"💾 Signal saved to alert_history (ID: {alert_id})")
                return alert_id
            else:
                self.logger.warning(f"⚠️ Primary save failed, trying backup method...")
                # Try backup method
                backup_id = save_signal_to_database_direct(signal, alert_message)
                if backup_id:
                    self.logger.info(f"💾 Signal saved via backup method (ID: {backup_id})")
                    return backup_id
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Primary save error: {e}")
            # Try backup method as fallback
            try:
                self.logger.info("🔄 Attempting backup save method...")
                backup_id = save_signal_to_database_direct(signal, alert_message)
                if backup_id:
                    self.logger.info(f"💾 Signal saved via backup method (ID: {backup_id})")
                    return backup_id
            except Exception as backup_e:
                self.logger.error(f"❌ Backup save also failed: {backup_e}")
            return None'''
            
            if re.search(old_method_pattern, content, re.DOTALL):
                content = re.sub(old_method_pattern, new_method, content, flags=re.DOTALL)
                print("✅ Enhanced _save_signal method with fallback")
        
        # Write back
        with open(scanner_path, 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding fallback option: {e}")
        return False

def verify_patch():
    """Verify that the patch was applied successfully"""
    scanner_path = "/app/forex_scanner/core/scanner.py"
    
    try:
        with open(scanner_path, 'r') as f:
            content = f.read()
        
        checks = {
            'AlertHistoryManager import': 'from alerts.alert_history import AlertHistoryManager' in content,
            'alert_history initialization': 'self.alert_history = AlertHistoryManager(db_manager)' in content,
            '_save_signal method': 'def _save_signal(self, signal: Dict)' in content,
            'save_signal call': 'alert_id = self._save_signal(signal)' in content,
            'backup import': 'from utils.backup_alert_saver import save_signal_to_database_direct' in content
        }
        
        print("\n🔍 Verification Results:")
        print("=" * 50)
        
        all_good = True
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"{status} {check_name}: {result}")
            if not result:
                all_good = False
        
        if all_good:
            print("\n🎉 All patches applied successfully!")
            print("💾 Alert history saving should now work in your scanner!")
        else:
            print("\n⚠️ Some patches may need manual review")
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error verifying patch: {e}")
        return False

def main():
    """Main patch application function"""
    print("🔧 FOREX SCANNER ALERT HISTORY PATCH")
    print("=" * 50)
    print("This will patch your scanner.py to save alerts to alert_history table")
    
    # Step 1: Backup
    print("\n📋 Step 1: Creating backup...")
    if not backup_existing_scanner():
        return False
    
    # Step 2: Create backup alert saver
    print("\n📋 Step 2: Creating backup alert saver...")
    create_backup_alert_saver()
    
    # Step 3: Patch imports
    print("\n📋 Step 3: Patching imports...")
    if not patch_scanner_imports():
        return False
    
    # Step 4: Patch initialization
    print("\n📋 Step 4: Patching initialization...")
    if not patch_scanner_init():
        return False
    
    # Step 5: Add _save_signal method
    print("\n📋 Step 5: Adding _save_signal method...")
    if not add_save_signal_method():
        return False
    
    # Step 6: Patch _process_signal method
    print("\n📋 Step 6: Patching _process_signal method...")
    if not patch_process_signal_method():
        return False
    
    # Step 7: Add fallback option
    print("\n📋 Step 7: Adding fallback option...")
    add_fallback_option()
    
    # Step 8: Verify
    print("\n📋 Step 8: Verifying patch...")
    success = verify_patch()
    
    if success:
        print("\n🎉 PATCH COMPLETED SUCCESSFULLY!")
        print("\n📝 Next Steps:")
        print("1. Test the scanner: python main.py scan --epic CS.D.EURUSD.CEEM.IP")
        print("2. Check database: SELECT * FROM alert_history ORDER BY alert_timestamp DESC LIMIT 5;")
        print("3. Monitor logs for '💾 Signal saved to alert_history' messages")
    else:
        print("\n⚠️ Patch completed with some issues - manual review recommended")
    
    return success

if __name__ == "__main__":
    main()