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
        
        # Extract and clean signal data with proper null handling
        epic = signal.get('epic')
        if not epic or epic is None:
            raise ValueError("Epic cannot be null - required field")
        epic = str(epic)
        
        pair = signal.get('pair')
        if not pair or pair is None:
            # Try to derive from epic
            if epic and 'CS.D.' in str(epic):
                pair = str(epic).replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
            else:
                pair = 'UNKNOWN'
        pair = str(pair)
        
        signal_type = signal.get('signal_type')
        if not signal_type or signal_type is None:
            raise ValueError("Signal type cannot be null - required field")
        signal_type = str(signal_type)
        
        strategy = signal.get('strategy')
        if not strategy or strategy is None:
            raise ValueError("Strategy cannot be null - required field")
        strategy = str(strategy)
        
        confidence_score = signal.get('confidence_score')
        if confidence_score is None:
            raise ValueError("Confidence score cannot be null - required field")
        confidence_score = float(confidence_score)
        
        price = signal.get('price')
        if price is None:
            raise ValueError("Price cannot be null - required field")
        price = float(price)
        
        timeframe = signal.get('timeframe')
        if not timeframe or timeframe is None:
            timeframe = '15m'  # Default fallback
        timeframe = str(timeframe)
        
        # Handle JSON fields properly
        def safe_json(value):
            if value is None:
                return None
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)
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
