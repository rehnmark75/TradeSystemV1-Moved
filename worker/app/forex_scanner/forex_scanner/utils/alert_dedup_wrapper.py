# utils/alert_dedup_wrapper.py
"""
Simple Alert Deduplication Wrapper
Ensures all alert saves include deduplication metadata
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Optional
import logging

class AlertSaver:
    """Wrapper that ensures alerts are saved with deduplication metadata"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
    def generate_signal_hash(self, signal: Dict) -> str:
        """Generate hash for signal deduplication"""
        hash_data = {
            'epic': signal.get('epic', ''),
            'signal_type': signal.get('signal_type', ''),
            'strategy': signal.get('strategy', ''),
            'price': round(float(signal.get('price', 0)), 4),
            'confidence_score': round(float(signal.get('confidence_score', 0)), 3)
        }
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def save_alert_with_dedup(self, signal: Dict, message: str = None) -> Optional[int]:
        """Save alert with deduplication metadata"""
        try:
            # Generate deduplication metadata
            signal_hash = self.generate_signal_hash(signal)
            cooldown_key = f"{signal.get('epic', '')}:{signal.get('signal_type', '')}:{signal.get('strategy', '')}"
            
            # Enhanced signal with metadata
            enhanced_signal = signal.copy()
            enhanced_signal.update({
                'signal_hash': signal_hash,
                'data_source': 'live_scanner',
                'market_timestamp': signal.get('timestamp', datetime.now().isoformat()),
                'cooldown_key': cooldown_key
            })
            
            # Save to database
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alert_history (
                    epic, pair, signal_type, strategy, confidence_score, price,
                    alert_message, alert_level, alert_timestamp,
                    signal_hash, data_source, market_timestamp, cooldown_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                enhanced_signal.get('epic'),
                enhanced_signal.get('pair', enhanced_signal.get('epic')),
                enhanced_signal.get('signal_type'),
                enhanced_signal.get('strategy'),
                enhanced_signal.get('confidence_score'),
                enhanced_signal.get('price'),
                message or f"Signal: {enhanced_signal.get('signal_type')} {enhanced_signal.get('epic')}",
                'INFO',
                datetime.now(),
                signal_hash,
                'live_scanner',
                enhanced_signal.get('market_timestamp'),
                cooldown_key
            ))
            
            alert_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"💾 Alert saved with deduplication (ID: {alert_id})")
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Error saving alert with deduplication: {e}")
            return None

# Global function for easy use
def save_signal_with_dedup(signal: Dict, message: str = None) -> Optional[int]:
    """Convenience function to save signal with deduplication"""
    from core.database import DatabaseManager
    import config
    
    db_manager = DatabaseManager(config.DATABASE_URL)
    saver = AlertSaver(db_manager)
    return saver.save_alert_with_dedup(signal, message)
