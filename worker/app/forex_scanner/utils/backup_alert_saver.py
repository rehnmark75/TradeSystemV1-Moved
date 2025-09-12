# utils/backup_alert_saver.py
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
