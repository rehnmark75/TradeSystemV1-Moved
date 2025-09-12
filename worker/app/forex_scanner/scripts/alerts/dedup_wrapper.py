
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
