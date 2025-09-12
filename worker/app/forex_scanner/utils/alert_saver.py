"""
Reliable fallback signal saver that bypasses AlertHistoryManager issues
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def save_signal_to_database_direct(signal: Dict, message: str = "Signal detected") -> Optional[int]:
    """
    Direct database save bypassing AlertHistoryManager
    Handles JSON fields properly for PostgreSQL
    """
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Extract and clean signal data
        epic = str(signal.get('epic', 'Unknown'))
        pair = str(signal.get('pair', epic.replace('CS.D.', '').replace('.MINI.IP', '')))
        signal_type = str(signal.get('signal_type', 'Unknown'))
        strategy = str(signal.get('strategy', 'Unknown'))
        confidence_score = float(signal.get('confidence_score', 0.0))
        price = float(signal.get('price', 0.0))
        timeframe = str(signal.get('timeframe', '15m'))
        
        # Handle JSON fields properly
        def safe_json(value):
            if value is None:
                return None
            if isinstance(value, dict):
                return json.dumps(value)
            return str(value)
        
        strategy_config = safe_json(signal.get('strategy_config'))
        strategy_indicators = safe_json(signal.get('strategy_indicators'))
        strategy_metadata = safe_json(signal.get('strategy_metadata'))
        signal_conditions = safe_json(signal.get('signal_conditions'))
        
        # Handle deduplication metadata
        signal_hash = signal.get('signal_hash')
        data_source = signal.get('data_source', 'live_scanner')
        cooldown_key = signal.get('cooldown_key')
        market_timestamp = signal.get('market_timestamp')
        
        # Convert market_timestamp if it's a string
        if isinstance(market_timestamp, str):
            try:
                market_timestamp = datetime.fromisoformat(market_timestamp.replace('Z', '+00:00'))
            except:
                market_timestamp = None
        
        # Insert with minimal required fields
        cursor.execute("""
            INSERT INTO alert_history (
                epic, pair, signal_type, strategy, confidence_score, price, timeframe,
                strategy_config, strategy_indicators, strategy_metadata, signal_conditions,
                signal_hash, data_source, cooldown_key, market_timestamp,
                alert_message, alert_level, status, alert_timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """, (
            epic, pair, signal_type, strategy, confidence_score, price, timeframe,
            strategy_config, strategy_indicators, strategy_metadata, signal_conditions,
            signal_hash, data_source, cooldown_key, market_timestamp,
            message, 'INFO', 'NEW', datetime.now()
        ))
        
        alert_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Signal saved directly to database (ID: {alert_id})")
        return alert_id
        
    except Exception as e:
        logger.error(f"❌ Direct database save failed: {e}")
        if 'conn' in locals():
            try:
                conn.rollback()
                cursor.close()
                conn.close()
            except:
                pass
        return None

def get_recent_alerts(limit: int = 10) -> list:
    """Get recent alerts from database"""
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, epic, signal_type, confidence_score, strategy, alert_timestamp, alert_message
            FROM alert_history 
            ORDER BY alert_timestamp DESC 
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
        return []
