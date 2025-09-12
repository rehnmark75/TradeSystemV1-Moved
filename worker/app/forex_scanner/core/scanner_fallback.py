# core/scanner_fallback.py
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
try:
    import config
except ImportError:
    from forex_scanner import config

class IntelligentForexScanner:
    """Fallback scanner with minimal implementation"""
    
    def __init__(self, db_manager, epic_list=None, **kwargs):
        self.db_manager = db_manager
        self.epic_list = epic_list or getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.MINI.IP'])
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
