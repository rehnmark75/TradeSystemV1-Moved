# core/alert_deduplication.py
"""
Enhanced Alert Deduplication System
Prevents repeated alerts from hammering by using alert_history table
with multiple deduplication strategies and configurable cooldown periods
"""

import hashlib
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

try:
    import config
except ImportError:
    from forex_scanner import config


@dataclass
class AlertCooldownConfig:
    """Configuration for alert cooldown periods - now reads from config.py"""
    # ✅ NOW READS FROM CONFIG.PY INSTEAD OF HARDCODED VALUES
    epic_signal_cooldown_minutes: int = getattr(config, 'ALERT_COOLDOWN_MINUTES', 5)
    strategy_cooldown_minutes: int = getattr(config, 'STRATEGY_COOLDOWN_MINUTES', 3)
    global_cooldown_seconds: int = getattr(config, 'GLOBAL_COOLDOWN_SECONDS', 30)
    max_alerts_per_hour: int = getattr(config, 'MAX_ALERTS_PER_HOUR', 50)
    max_alerts_per_epic_hour: int = getattr(config, 'MAX_ALERTS_PER_EPIC_HOUR', 6)


class AlertDeduplicationManager:
    """
    Manages alert deduplication using multiple strategies:
    1. Signal hash-based deduplication (exact duplicates)
    2. Time-based cooldowns (prevent alert flooding)
    3. Epic+signal type combinations (prevent repeated same signals)
    4. Global rate limiting (prevent system overload)
    """
    
    def __init__(self, db_manager, config_override: AlertCooldownConfig = None):
        self.db_manager = db_manager
        
        # ✅ ENHANCED CONFIG HANDLING - Use config.py values by default
        if config_override:
            self.config = config_override
        else:
            # Create config from config.py values
            self.config = AlertCooldownConfig(
                epic_signal_cooldown_minutes=getattr(config, 'ALERT_COOLDOWN_MINUTES', 5),
                strategy_cooldown_minutes=getattr(config, 'STRATEGY_COOLDOWN_MINUTES', 3),
                global_cooldown_seconds=getattr(config, 'GLOBAL_COOLDOWN_SECONDS', 30),
                max_alerts_per_hour=getattr(config, 'MAX_ALERTS_PER_HOUR', 50),
                max_alerts_per_epic_hour=getattr(config, 'MAX_ALERTS_PER_EPIC_HOUR', 6)
            )
            
        self.logger = logging.getLogger(__name__)
        
        # ✅ LOG THE ACTUAL CONFIGURATION BEING USED
        self.logger.info("🛡️ Alert deduplication manager initialized")
        self.logger.info(f"   Epic cooldown: {self.config.epic_signal_cooldown_minutes} minutes")
        self.logger.info(f"   Strategy cooldown: {self.config.strategy_cooldown_minutes} minutes") 
        self.logger.info(f"   Global cooldown: {self.config.global_cooldown_seconds} seconds")
        self.logger.info(f"   Max alerts/hour: {self.config.max_alerts_per_hour}")
        self.logger.info(f"   Max alerts/epic/hour: {self.config.max_alerts_per_epic_hour}")
        
        # Initialize enhanced alert_history table if needed
        self._ensure_enhanced_table_structure()
        
        # In-memory caches for performance
        self._recent_signals_cache = {}  # {epic: {signal_type: last_timestamp}}
        self._signal_hash_cache = set()  # Set of recent signal hashes
        self._hourly_alert_count = 0
        self._last_count_reset = datetime.now()
    
    def _ensure_enhanced_table_structure(self):
        """Ensure alert_history table has enhanced columns for deduplication"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Add enhanced columns if they don't exist
            enhancement_queries = [
                "ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS signal_hash VARCHAR(32)",
                "ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS data_source VARCHAR(20) DEFAULT 'live_scanner'",
                "ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS market_timestamp TIMESTAMP",
                "ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS cooldown_key VARCHAR(100)",
            ]
            
            for query in enhancement_queries:
                cursor.execute(query)
            
            # Add enhanced indexes for performance
            index_queries = [
                "CREATE INDEX IF NOT EXISTS idx_alert_history_signal_hash ON alert_history(signal_hash)",
                "CREATE INDEX IF NOT EXISTS idx_alert_history_cooldown_key ON alert_history(cooldown_key)",
                "CREATE INDEX IF NOT EXISTS idx_alert_history_epic_signal_time ON alert_history(epic, signal_type, alert_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_alert_history_market_timestamp ON alert_history(market_timestamp)",
            ]
            
            for query in index_queries:
                cursor.execute(query)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info("✅ Enhanced alert_history table structure verified")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to enhance alert_history table: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
    
    def generate_signal_hash(self, signal: Dict) -> str:
        """
        Generate a unique hash for the signal to detect exact duplicates
        """
        # Create a normalized signal for hashing
        hash_data = {
            'epic': signal.get('epic', ''),
            'signal_type': signal.get('signal_type', ''),
            'strategy': signal.get('strategy', ''),
            'price': round(float(signal.get('price', 0)), 5),  # Round to 5 decimals
            'confidence_score': round(float(signal.get('confidence_score', 0)), 4),
            'timeframe': signal.get('timeframe', ''),
            # Include key technical indicators in hash
            'ema_short': round(float(signal.get('ema_short', 0)), 5) if signal.get('ema_short') else None,
            'ema_long': round(float(signal.get('ema_long', 0)), 5) if signal.get('ema_long') else None,
            'macd_line': round(float(signal.get('macd_line', 0)), 6) if signal.get('macd_line') else None,
        }
        
        # Convert to deterministic JSON string
        hash_string = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        
        # Generate MD5 hash
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def generate_cooldown_key(self, signal: Dict) -> str:
        """Generate a cooldown key for time-based deduplication"""
        epic = signal.get('epic', '')
        signal_type = signal.get('signal_type', '')
        strategy = signal.get('strategy', '')
        
        return f"{epic}:{signal_type}:{strategy}"
    
    def _reset_hourly_counters_if_needed(self):
        """Reset hourly counters if an hour has passed"""
        now = datetime.now()
        if (now - self._last_count_reset).total_seconds() >= 3600:  # 1 hour
            self._hourly_alert_count = 0
            self._last_count_reset = now
            self.logger.info("🔄 Hourly alert counters reset")
    
    def _check_global_rate_limits(self) -> Tuple[bool, str]:
        """Check global rate limiting"""
        self._reset_hourly_counters_if_needed()
        
        if self._hourly_alert_count >= self.config.max_alerts_per_hour:
            return False, f"Global hourly limit reached ({self.config.max_alerts_per_hour})"
        
        return True, ""
    
    def _check_epic_rate_limits(self, epic: str) -> Tuple[bool, str]:
        """Check per-epic rate limiting"""
        try:
            # Count alerts for this epic in the last hour
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM alert_history 
                WHERE epic = %s 
                AND alert_timestamp >= %s
            """, (epic, datetime.now() - timedelta(hours=1)))
            
            result = cursor.fetchone()
            epic_count = result[0] if result else 0
            
            cursor.close()
            conn.close()
            
            if epic_count >= self.config.max_alerts_per_epic_hour:
                return False, f"Epic hourly limit reached for {epic} ({self.config.max_alerts_per_epic_hour})"
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Error checking epic rate limits: {e}")
            return True, ""  # Allow on error to avoid blocking legitimate alerts
    
    def _check_signal_hash_duplicate(self, signal_hash: str) -> Tuple[bool, str]:
        """Check if this exact signal hash was recently seen"""
        if signal_hash in self._signal_hash_cache:
            return False, "Exact duplicate signal detected (hash match)"
        
        # Check database for recent hash matches (last 15 minutes)
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id FROM alert_history 
                WHERE signal_hash = %s 
                AND alert_timestamp >= %s
                LIMIT 1
            """, (signal_hash, datetime.now() - timedelta(minutes=15)))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return False, "Exact duplicate found in recent history"
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Error checking signal hash: {e}")
            return True, ""  # Allow on error
    
    def _check_cooldown_periods(self, cooldown_key: str) -> Tuple[bool, str]:
        """Check time-based cooldown periods"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Check for recent alerts with same cooldown key
            cursor.execute("""
                SELECT alert_timestamp, epic, signal_type, strategy
                FROM alert_history 
                WHERE cooldown_key = %s 
                AND alert_timestamp >= %s
                ORDER BY alert_timestamp DESC
                LIMIT 1
            """, (cooldown_key, datetime.now() - timedelta(minutes=self.config.epic_signal_cooldown_minutes)))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                last_alert_time = result[0]
                time_diff = datetime.now() - last_alert_time
                remaining_cooldown = self.config.epic_signal_cooldown_minutes * 60 - time_diff.total_seconds()
                
                if remaining_cooldown > 0:
                    return False, f"Cooldown active: {remaining_cooldown:.0f}s remaining"
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Error checking cooldown periods: {e}")
            return True, ""  # Allow on error
    
    def should_allow_alert(self, signal: Dict) -> Tuple[bool, str, Dict]:
        """
        Main method to check if an alert should be allowed
        
        Args:
            signal: Signal dictionary to check
            
        Returns:
            Tuple of (allow: bool, reason: str, metadata: Dict)
        """
        epic = signal.get('epic', 'Unknown')
        signal_type = signal.get('signal_type', 'Unknown')
        strategy = signal.get('strategy', 'Unknown')
        
        # Generate deduplication keys
        signal_hash = self.generate_signal_hash(signal)
        cooldown_key = self.generate_cooldown_key(signal)
        
        metadata = {
            'signal_hash': signal_hash,
            'cooldown_key': cooldown_key,
            'check_timestamp': datetime.now().isoformat()
        }
        
        # Run all deduplication checks
        checks = [
            ("Global Rate Limit", self._check_global_rate_limits()),
            ("Epic Rate Limit", self._check_epic_rate_limits(epic)),
            ("Signal Hash Duplicate", self._check_signal_hash_duplicate(signal_hash)),
            ("Cooldown Period", self._check_cooldown_periods(cooldown_key)),
        ]
        
        # Log check results
        self.logger.debug(f"🔍 Deduplication checks for {epic} {signal_type}:")
        
        for check_name, (passed, reason) in checks:
            if not passed:
                self.logger.info(f"🚫 Alert blocked: {check_name} - {reason}")
                metadata['blocked_by'] = check_name
                metadata['block_reason'] = reason
                return False, f"{check_name}: {reason}", metadata
            else:
                self.logger.debug(f"✅ {check_name}: Passed")
        
        # All checks passed - update caches
        self._signal_hash_cache.add(signal_hash)
        self._hourly_alert_count += 1
        
        # Update in-memory cache
        if epic not in self._recent_signals_cache:
            self._recent_signals_cache[epic] = {}
        self._recent_signals_cache[epic][signal_type] = datetime.now()
        
        # Clean up old cache entries (keep cache size manageable)
        if len(self._signal_hash_cache) > 1000:
            # Remove oldest 20% of entries (simplified cleanup)
            hash_list = list(self._signal_hash_cache)
            self._signal_hash_cache = set(hash_list[-800:])
        
        self.logger.info(f"✅ Alert approved: {epic} {signal_type} ({strategy})")
        metadata['approved'] = True
        return True, "Alert approved", metadata
    
    def save_alert_with_deduplication(self, alert_history_manager, signal: Dict, 
                                     alert_message: str = None, alert_level: str = 'INFO') -> Optional[int]:
        """
        Save alert only if it passes deduplication checks
        
        Args:
            alert_history_manager: AlertHistoryManager instance
            signal: Signal dictionary
            alert_message: Alert message
            alert_level: Alert level
            
        Returns:
            Alert ID if saved, None if blocked or failed
        """
        # Check if alert should be allowed
        allow, reason, metadata = self.should_allow_alert(signal)
        
        if not allow:
            self.logger.info(f"🚫 Alert blocked and not saved: {reason}")
            return None
        
        # Add deduplication metadata to signal
        enhanced_signal = signal.copy()
        enhanced_signal.update({
            'signal_hash': metadata['signal_hash'],
            'cooldown_key': metadata['cooldown_key'],
            'data_source': 'live_scanner',
            'market_timestamp': signal.get('timestamp', datetime.now().isoformat())
        })
        
        # Save the alert
        try:
            alert_id = alert_history_manager.save_alert(enhanced_signal, alert_message, alert_level)
            
            if alert_id:
                self.logger.info(f"💾 Alert saved with deduplication metadata (ID: {alert_id})")
                return alert_id
            else:
                self.logger.error("❌ Failed to save alert to database")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving alert: {e}")
            return None
    
    def get_deduplication_stats(self) -> Dict:
        """Get current deduplication statistics"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Get recent statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_alerts_24h,
                    COUNT(DISTINCT epic) as unique_epics_24h,
                    COUNT(DISTINCT signal_hash) as unique_signals_24h,
                    AVG(confidence_score) as avg_confidence_24h
                FROM alert_history 
                WHERE alert_timestamp >= %s
            """, (datetime.now() - timedelta(hours=24),))
            
            stats = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return {
                'total_alerts_24h': stats[0] if stats else 0,
                'unique_epics_24h': stats[1] if stats else 0, 
                'unique_signals_24h': stats[2] if stats else 0,
                'avg_confidence_24h': float(stats[3]) if stats and stats[3] else 0,
                'cache_size': len(self._signal_hash_cache),
                'hourly_alert_count': self._hourly_alert_count,
                'config': {
                    'epic_signal_cooldown_minutes': self.config.epic_signal_cooldown_minutes,
                    'max_alerts_per_hour': self.config.max_alerts_per_hour,
                    'max_alerts_per_epic_hour': self.config.max_alerts_per_epic_hour
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting deduplication stats: {e}")
            return {}


# Utility functions for easy integration

def create_deduplication_manager(db_manager, 
                                epic_cooldown_minutes: int = None,
                                max_alerts_per_hour: int = None) -> AlertDeduplicationManager:
    """Factory function to create deduplication manager with custom config"""
    # ✅ ENHANCED: Use config.py values as defaults instead of hardcoded values
    epic_cooldown = epic_cooldown_minutes or getattr(config, 'ALERT_COOLDOWN_MINUTES', 5)
    max_alerts = max_alerts_per_hour or getattr(config, 'MAX_ALERTS_PER_HOUR', 50)
    
    custom_config = AlertCooldownConfig(
        epic_signal_cooldown_minutes=epic_cooldown,
        max_alerts_per_hour=max_alerts,
        strategy_cooldown_minutes=getattr(config, 'STRATEGY_COOLDOWN_MINUTES', 3),
        global_cooldown_seconds=getattr(config, 'GLOBAL_COOLDOWN_SECONDS', 30),
        max_alerts_per_epic_hour=getattr(config, 'MAX_ALERTS_PER_EPIC_HOUR', 6)
    )
    
    return AlertDeduplicationManager(db_manager, custom_config)


def save_alert_if_allowed(db_manager, alert_history_manager, signal: Dict, 
                         alert_message: str = None) -> Optional[int]:
    """
    Convenience function to save alert with deduplication in one call
    """
    dedup_manager = AlertDeduplicationManager(db_manager)
    return dedup_manager.save_alert_with_deduplication(
        alert_history_manager, signal, alert_message
    )