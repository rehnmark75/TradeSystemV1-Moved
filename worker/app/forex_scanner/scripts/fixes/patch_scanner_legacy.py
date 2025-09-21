#!/usr/bin/env python3
"""
Compatibility patch for refactored scanner
Adds missing legacy methods for backward compatibility
"""

import os
import sys

def patch_scanner_file():
    """Add legacy compatibility methods to the refactored scanner"""
    
    scanner_path = '/app/forex_scanner/core/scanner.py'
    
    if not os.path.exists(scanner_path):
        print(f"❌ Scanner file not found: {scanner_path}")
        return False
    
    print("🔧 Patching scanner for legacy compatibility...")
    
    with open(scanner_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if '_scan_single_epic' in content and 'Legacy compatibility methods' in content:
        print("   ✅ Scanner already patched")
        return True
    
    # Find the location to insert legacy methods
    insert_point = "    # Testing and diagnostics methods"
    
    if insert_point not in content:
        print("   ❌ Could not find insertion point in scanner file")
        return False
    
    # Legacy methods to add
    legacy_methods = '''
    # Legacy compatibility methods (delegate to components)
    def _scan_single_epic(self, epic: str, enable_multi_tf: bool = False) -> Optional[Dict]:
        """Legacy method - delegates to scanner controller"""
        try:
            return self.scanner_controller._scan_single_epic(epic)
        except Exception as e:
            self.logger.error(f"❌ Error in _scan_single_epic for {epic}: {e}")
            return None
    
    def _scan_without_intelligence(self) -> List[Dict]:
        """Legacy method - delegates to scanner controller"""
        try:
            return self.scanner_controller._detect_raw_signals()
        except Exception as e:
            self.logger.error(f"❌ Error in _scan_without_intelligence: {e}")
            return []
    
    def _scan_with_backtest_intelligence(self) -> List[Dict]:
        """Legacy method - delegates to scanner controller with backtest intelligence"""
        try:
            raw_signals = self.scanner_controller._detect_raw_signals()
            return self.intelligence_manager.apply_backtest_intelligence_filtering(raw_signals)
        except Exception as e:
            self.logger.error(f"❌ Error in _scan_with_backtest_intelligence: {e}")
            return []
    
    def _scan_with_configurable_intelligence(self) -> List[Dict]:
        """Legacy method - delegates to scanner controller with configurable intelligence"""
        try:
            raw_signals = self.scanner_controller._detect_raw_signals()
            return self.intelligence_manager.apply_intelligence_filtering(raw_signals)
        except Exception as e:
            self.logger.error(f"❌ Error in _scan_with_configurable_intelligence: {e}")
            return []
    
    def _process_signal(self, signal: Dict):
        """Legacy method - delegates to signal processor"""
        try:
            return self.signal_processor.process_signal(signal)
        except Exception as e:
            self.logger.error(f"❌ Error in _process_signal: {e}")
            return signal
    
    def _apply_deduplication_filtering(self, signals: List[Dict]) -> List[Dict]:
        """Legacy method - delegates to scanner controller"""
        try:
            return self.scanner_controller._apply_deduplication_filtering(signals)
        except Exception as e:
            self.logger.error(f"❌ Error in _apply_deduplication_filtering: {e}")
            return signals
    
    def _log_signal_summary(self, signal: Dict, index: int, total: int):
        """Legacy method - delegates to signal processor"""
        try:
            return self.signal_processor.log_signal_summary(signal, index, total)
        except Exception as e:
            self.logger.error(f"❌ Error in _log_signal_summary: {e}")

    '''
    
    # Insert the legacy methods
    content = content.replace(insert_point, legacy_methods + "\n    " + insert_point)
    
    # Write the patched file
    with open(scanner_path, 'w') as f:
        f.write(content)
    
    print("   ✅ Scanner patched with legacy compatibility methods")
    return True

def patch_scanner_controller():
    """Ensure scanner controller has public scan_single_epic method"""
    
    controller_path = '/app/forex_scanner/core/scanning/scanner_controller.py'
    
    if not os.path.exists(controller_path):
        print(f"   ⚠️ ScannerController file not found: {controller_path}")
        return False
    
    print("🔧 Patching scanner controller...")
    
    with open(controller_path, 'r') as f:
        content = f.read()
    
    # Check if public method already exists
    if 'def scan_single_epic(self, epic: str' in content:
        print("   ✅ ScannerController already has public scan_single_epic method")
        return True
    
    # Add public method before private method
    if 'def _scan_single_epic(self, epic: str' in content:
        public_method = '''
    def scan_single_epic(self, epic: str, enable_multi_tf: bool = False) -> Optional[Dict]:
        """Public method to scan a single epic"""
        return self._scan_single_epic(epic)
    '''
        
        content = content.replace(
            'def _scan_single_epic(self, epic: str',
            public_method + '\n    def _scan_single_epic(self, epic: str'
        )
        
        with open(controller_path, 'w') as f:
            f.write(content)
        
        print("   ✅ Added public scan_single_epic method to ScannerController")
        return True
    else:
        print("   ❌ Could not find _scan_single_epic method in ScannerController")
        return False

def create_fallback_scanner():
    """Create a fallback scanner that works with existing code"""
    
    fallback_content = '''# core/scanner_fallback.py
"""
Fallback scanner for immediate compatibility
Use this if the refactored version has issues
"""

import sys
import os
sys.path.insert(0, '/app/forex_scanner')

import logging
from typing import Dict, List, Optional
from datetime import datetime
import config

class IntelligentForexScanner:
    """Fallback scanner with minimal implementation"""
    
    def __init__(self, db_manager, epic_list=None, **kwargs):
        self.db_manager = db_manager
        self.epic_list = epic_list or getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.CEEM.IP'])
        self.intelligence_mode = kwargs.get('intelligence_mode', 'backtest_consistent')
        self.scan_interval = kwargs.get('scan_interval', 60)
        self.min_confidence = kwargs.get('min_confidence', 0.7)
        self.running = False
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize basic components
        from core.signal_detector import SignalDetector
        self.signal_detector = SignalDetector(db_manager, kwargs.get('user_timezone', 'Europe/Stockholm'))
        
        self.logger.info("🔧 Fallback scanner initialized")
    
    def _scan_single_epic(self, epic: str, enable_multi_tf: bool = False) -> Optional[Dict]:
        """Scan a single epic"""
        try:
            pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            
            signal = self.signal_detector.detect_signals_mid_prices(
                epic=epic,
                pair=pair_name,
                timeframe=getattr(config, 'DEFAULT_TIMEFRAME', '5m')
            )
            
            if signal:
                self.logger.info(f"📊 Signal detected for {epic}: {signal.get('signal_type')} at {signal.get('confidence_score', 0):.1%}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning {epic}: {e}")
            return None
    
    def scan_once(self) -> List[Dict]:
        """Scan all epics once"""
        signals = []
        
        for epic in self.epic_list:
            signal = self._scan_single_epic(epic)
            if signal and signal.get('confidence_score', 0) >= self.min_confidence:
                signals.append(signal)
        
        self.logger.info(f"📊 Scan completed: {len(signals)} signals found")
        return signals
    
    def start_continuous_scan(self):
        """Start continuous scanning"""
        import time
        
        self.running = True
        self.logger.info("🚀 Starting continuous scan")
        
        try:
            while self.running:
                signals = self.scan_once()
                
                for signal in signals:
                    self._process_signal(signal)
                
                time.sleep(self.scan_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Scanner stopped by user")
        finally:
            self.running = False
    
    def stop(self):
        """Stop the scanner"""
        self.running = False
    
    def _process_signal(self, signal: Dict):
        """Basic signal processing"""
        epic = signal.get('epic')
        signal_type = signal.get('signal_type')
        confidence = signal.get('confidence_score', 0)
        
        self.logger.info(f"📊 Processing {signal_type} signal for {epic} ({confidence:.1%})")
        
        # Basic processing - just log for now
        signal['processed'] = True
        signal['processing_timestamp'] = datetime.now().isoformat()
        
        return signal
    
    def get_scanner_status(self) -> Dict:
        """Get scanner status"""
        return {
            'running': self.running,
            'epic_count': len(self.epic_list),
            'intelligence_mode': self.intelligence_mode,
            'fallback_mode': True
        }
    
    def update_configuration(self, **kwargs):
        """Update configuration"""
        updated = []
        
        if 'epic_list' in kwargs:
            self.epic_list = kwargs['epic_list']
            updated.append("epic_list")
        
        if 'min_confidence' in kwargs:
            self.min_confidence = kwargs['min_confidence']
            updated.append("min_confidence")
        
        self.logger.info(f"📝 Configuration updated: {', '.join(updated)}")
        return updated

# Keep the existing simple ForexScanner as well
class ForexScanner:
    """Simple forex scanner - unchanged"""
    pass
'''
    
    fallback_path = '/app/forex_scanner/core/scanner_fallback.py'
    
    with open(fallback_path, 'w') as f:
        f.write(fallback_content)
    
    print(f"✅ Created fallback scanner: {fallback_path}")
    print("   Use this if the refactored version has issues")

def main():
    """Main patching function"""
    
    print("🔧 Applying compatibility patches...")
    print("=" * 50)
    
    # Step 1: Patch main scanner
    scanner_patched = patch_scanner_file()
    
    # Step 2: Patch scanner controller
    controller_patched = patch_scanner_controller()
    
    # Step 3: Create fallback scanner
    create_fallback_scanner()
    
    print("\n" + "=" * 50)
    
    if scanner_patched and controller_patched:
        print("✅ All patches applied successfully!")
        print("\nThe refactored scanner should now work with existing code.")
    else:
        print("⚠️ Some patches failed. You can use the fallback scanner:")
        print("   Change imports from 'core.scanner' to 'core.scanner_fallback'")

if __name__ == "__main__":
    main()