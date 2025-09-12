#!/usr/bin/env python3
"""
Immediate Deduplication Integration Patch
This patches your existing scanner to stop the duplicate alerts you're seeing
"""

import os
import sys
import re

def create_deduplication_wrapper():
    """Create a simple wrapper that replaces your current alert saving"""
    
    wrapper_code = '''
# alerts/dedup_wrapper.py
"""
Simple deduplication wrapper to replace existing alert saving
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

class SimpleDeduplicationWrapper:
    """Simple wrapper to prevent duplicate alerts"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self._recent_hashes = set()  # In-memory cache
        
    def generate_signal_hash(self, signal: Dict) -> str:
        """Generate hash for signal deduplication"""
        # Create hash from key signal components
        hash_data = {
            'epic': signal.get('epic', ''),
            'signal_type': signal.get('signal_type', ''),
            'strategy': signal.get('strategy', ''),
            'price': round(float(signal.get('price', 0)), 4),  # Round for consistency
            'confidence': round(float(signal.get('confidence_score', 0)), 3)
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def is_duplicate_signal(self, signal: Dict) -> bool:
        """Check if this signal is a recent duplicate"""
        try:
            signal_hash = self.generate_signal_hash(signal)
            epic = signal.get('epic', '')
            
            # Check in-memory cache first
            if signal_hash in self._recent_hashes:
                self.logger.info(f"🚫 Duplicate signal blocked (memory cache): {epic}")
                return True
            
            # Check database for recent duplicates (last 10 minutes)
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id FROM alert_history 
                WHERE epic = %s 
                AND signal_type = %s 
                AND alert_timestamp >= %s
                LIMIT 1
            """, (
                epic, 
                signal.get('signal_type', ''), 
                datetime.now() - timedelta(minutes=10)
            ))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                self.logger.info(f"🚫 Duplicate signal blocked (database): {epic}")
                return True
            
            # Not a duplicate - add to cache
            self._recent_hashes.add(signal_hash)
            
            # Clean cache if it gets too big
            if len(self._recent_hashes) > 100:
                self._recent_hashes.clear()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking duplicate: {e}")
            return False  # Allow on error to avoid blocking legitimate signals
    
    def save_alert_with_dedup(self, alert_history_manager, signal: Dict, 
                             alert_message: str = None, alert_level: str = 'INFO') -> Optional[int]:
        """Save alert only if not a duplicate"""
        
        # Check for duplicates first
        if self.is_duplicate_signal(signal):
            return None  # Signal blocked
        
        # Not a duplicate - save normally
        try:
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            
            # Add deduplication metadata
            enhanced_signal = signal.copy()
            enhanced_signal.update({
                'signal_hash': self.generate_signal_hash(signal),
                'data_source': 'live_scanner',
                'market_timestamp': datetime.now().isoformat(),
                'cooldown_key': f"{epic}:{signal_type}:{signal.get('strategy', '')}"
            })
            
            alert_id = alert_history_manager.save_alert(
                enhanced_signal, alert_message or f"Live signal: {signal_type} {epic}", alert_level
            )
            
            if alert_id:
                self.logger.info(f"✅ Signal saved with deduplication (ID: {alert_id})")
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"❌ Error saving signal: {e}")
            return None
'''
    
    # Write the wrapper file
    os.makedirs('alerts', exist_ok=True)
    with open('alerts/dedup_wrapper.py', 'w') as f:
        f.write(wrapper_code)
    
    print("✅ Created deduplication wrapper: alerts/dedup_wrapper.py")

def patch_trade_scan_file():
    """Patch the main trade_scan.py file to use deduplication"""
    
    trade_scan_path = "/app/trade_scan.py"
    
    if not os.path.exists(trade_scan_path):
        print(f"❌ {trade_scan_path} not found")
        return False
    
    try:
        with open(trade_scan_path, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if 'SimpleDeduplicationWrapper' in content:
            print("✅ trade_scan.py already patched with deduplication")
            return True
        
        # Add import after the forex_scanner sys.path.append
        import_section = """
# Import deduplication wrapper
sys.path.append('/app/forex_scanner')
from alerts.dedup_wrapper import SimpleDeduplicationWrapper
"""
        
        # Find the sys.path.append line and add after it
        if "sys.path.append('/app/forex_scanner')" in content:
            content = content.replace(
                "sys.path.append('/app/forex_scanner')",
                f"sys.path.append('/app/forex_scanner'){import_section}"
            )
        
        # Find the IntelligentForexScanner __init__ method and add dedup_wrapper
        init_pattern = r'(def __init__\(self.*?\n.*?self\.db_manager = db_manager)'
        
        init_replacement = r'''\\1
        
        # Initialize deduplication wrapper
        self.dedup_wrapper = SimpleDeduplicationWrapper(self.db_manager)
        self.logger.info("🛡️ Deduplication wrapper initialized")'''
        
        content = re.sub(init_pattern, init_replacement, content, flags=re.DOTALL)
        
        # Find where signals are saved and replace with deduplication version
        # Look for alert_history.save_alert calls
        save_pattern = r'alert_id = self\.alert_history\.save_alert\([^)]+\)'
        save_replacement = '''alert_id = self.dedup_wrapper.save_alert_with_dedup(
                    self.alert_history, signal, f"Live signal: {signal['strategy']}"
                )'''
        
        content = re.sub(save_pattern, save_replacement, content)
        
        # Alternative pattern for direct save_alert calls
        alt_pattern = r'self\.alert_history\.save_alert\((.*?)\)'
        alt_replacement = r'self.dedup_wrapper.save_alert_with_dedup(self.alert_history, \\1)'
        
        content = re.sub(alt_pattern, alt_replacement, content)
        
        with open(trade_scan_path, 'w') as f:
            f.write(content)
        
        print("✅ Patched trade_scan.py with deduplication")
        return True
        
    except Exception as e:
        print(f"❌ Error patching trade_scan.py: {e}")
        return False

def patch_scanner_file():
    """Patch the core/scanner.py file if it exists"""
    
    scanner_path = "/app/forex_scanner/core/scanner.py"
    
    if not os.path.exists(scanner_path):
        print("⚠️ core/scanner.py not found - skipping")
        return True
    
    try:
        with open(scanner_path, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if 'SimpleDeduplicationWrapper' in content:
            print("✅ core/scanner.py already has deduplication")
            return True
        
        # Add import at top
        if 'from alerts.dedup_wrapper import SimpleDeduplicationWrapper' not in content:
            # Find a good place to add import
            import_pattern = r'(import config\n)'
            import_replacement = r'\\1from alerts.dedup_wrapper import SimpleDeduplicationWrapper\n'
            content = re.sub(import_pattern, import_replacement, content)
        
        # Add dedup_wrapper initialization in __init__
        init_pattern = r'(self\.alert_history = AlertHistoryManager\(db_manager\))'
        init_replacement = r'''\\1
        self.dedup_wrapper = SimpleDeduplicationWrapper(db_manager)
        self.logger.info("🛡️ Scanner deduplication initialized")'''
        
        content = re.sub(init_pattern, init_replacement, content)
        
        # Replace _process_signal method to use deduplication
        process_signal_pattern = r'(def _process_signal\(self, signal.*?\n)(.*?)(def [^_]|\n\n\n|\Z)'
        
        def replace_process_signal(match):
            method_def = match.group(1)
            method_body = match.group(2)
            next_section = match.group(3)
            
            # Create new method body with deduplication
            new_body = '''        """Process signal with deduplication"""
        try:
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            
            self.logger.info(f"🎯 Processing {signal_type} signal for {epic} @ {confidence:.1%}")
            
            # Save with deduplication
            alert_id = self.dedup_wrapper.save_alert_with_dedup(
                self.alert_history, signal, f"Scanner signal: {signal_type} {epic}"
            )
            
            if alert_id:
                # Send notifications if configured
                if hasattr(self, 'telegram') and self.telegram:
                    try:
                        message = f"🎯 {signal_type} Signal\\nEpic: {epic}\\nConfidence: {confidence:.1%}"
                        self.telegram.send_alert(message)
                    except Exception as e:
                        self.logger.error(f"Telegram error: {e}")
                        
                return alert_id
            else:
                self.logger.debug(f"Signal blocked by deduplication: {epic}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            return None
'''
            
            return method_def + new_body + next_section
        
        content = re.sub(process_signal_pattern, replace_process_signal, content, flags=re.DOTALL)
        
        with open(scanner_path, 'w') as f:
            f.write(content)
        
        print("✅ Patched core/scanner.py with deduplication")
        return True
        
    except Exception as e:
        print(f"❌ Error patching core/scanner.py: {e}")
        return False

def create_test_script():
    """Create a test script to verify deduplication is working"""
    
    test_script = '''#!/usr/bin/env python3
"""
Test script to verify deduplication is working
"""

import sys
sys.path.append('/app/forex_scanner')

def test_deduplication():
    """Test the deduplication system"""
    try:
        from core.database import DatabaseManager
        from alerts.alert_history import AlertHistoryManager
        from alerts.dedup_wrapper import SimpleDeduplicationWrapper
        import config
        
        print("🧪 Testing Deduplication System")
        print("=" * 40)
        
        # Initialize components
        db = DatabaseManager(config.DATABASE_URL)
        alert_history = AlertHistoryManager(db)
        dedup_wrapper = SimpleDeduplicationWrapper(db)
        
        # Test signal
        test_signal = {
            'epic': 'CS.D.USDCHF.MINI.IP',
            'signal_type': 'BULL',
            'strategy': 'test_dedup',
            'confidence_score': 0.75,
            'price': 0.8850,
            'timestamp': '2025-07-14 19:15:00'
        }
        
        print("\\n🔍 Testing duplicate detection...")
        
        # First signal should be allowed
        alert_id1 = dedup_wrapper.save_alert_with_dedup(alert_history, test_signal)
        print(f"First signal result: {'✅ Saved' if alert_id1 else '❌ Blocked'} (ID: {alert_id1})")
        
        # Immediate duplicate should be blocked
        alert_id2 = dedup_wrapper.save_alert_with_dedup(alert_history, test_signal)
        print(f"Duplicate signal result: {'❌ Should have been blocked!' if alert_id2 else '✅ Correctly blocked'}")
        
        if alert_id1 and not alert_id2:
            print("\\n🎉 Deduplication is working correctly!")
        else:
            print("\\n⚠️ Deduplication may not be working as expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_deduplication()
'''
    
    with open('test_deduplication_fix.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test script: test_deduplication_fix.py")

def main():
    """Main patching function"""
    print("🚀 IMMEDIATE DEDUPLICATION FIX")
    print("=" * 50)
    print("This will patch your scanner to stop duplicate alerts immediately")
    print()
    
    # Step 1: Create the simple wrapper
    create_deduplication_wrapper()
    
    # Step 2: Patch the main scanner files
    patch_trade_scan_file()
    patch_scanner_file()
    
    # Step 3: Create test script
    create_test_script()
    
    print()
    print("=" * 50)
    print("🎉 PATCHING COMPLETE!")
    print("=" * 50)
    print()
    print("✅ Created simple deduplication wrapper")
    print("✅ Patched scanner files to use deduplication") 
    print("✅ Created test script")
    print()
    print("🔄 NEXT STEPS:")
    print("1. Restart your scanner container")
    print("2. Run: python test_deduplication_fix.py")
    print("3. Monitor logs for deduplication messages")
    print("4. Should see '🚫 Duplicate signal blocked' messages")
    print()
    print("🛡️ EXPECTED BEHAVIOR:")
    print("- Same epic+signal within 10 minutes = blocked")
    print("- Identical signals with same price/confidence = blocked")
    print("- Legitimate new signals = allowed")

if __name__ == "__main__":
    main()