# core/alert_deduplication.py
"""
Enhanced Alert Deduplication System
Prevents repeated alerts from hammering by using alert_history table
with multiple deduplication strategies and configurable cooldown periods

CRITICAL: Database-driven configuration - NO FALLBACK to config.py
All settings must come from scanner_global_config table.
"""

import hashlib
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Import scanner config service for database-driven settings - REQUIRED, NO FALLBACK
try:
    from forex_scanner.services.scanner_config_service import get_scanner_config, ScannerConfig
    SCANNER_CONFIG_AVAILABLE = True
except ImportError:
    SCANNER_CONFIG_AVAILABLE = False


@dataclass
class AlertCooldownConfig:
    """Configuration for alert cooldown periods - now reads from database"""
    epic_signal_cooldown_minutes: int = 5
    strategy_cooldown_minutes: int = 3
    global_cooldown_seconds: int = 30
    max_alerts_per_hour: int = 50
    max_alerts_per_epic_hour: int = 6


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
        self.logger = logging.getLogger(__name__)

        # ✅ CRITICAL: Database-driven configuration - NO FALLBACK to config.py
        if config_override:
            self.config = config_override
            self._scanner_cfg = None
        else:
            # REQUIRE database configuration - no fallback allowed
            if not SCANNER_CONFIG_AVAILABLE:
                raise RuntimeError(
                    "❌ CRITICAL: Scanner config service not available - database is REQUIRED, no fallback allowed"
                )

            try:
                self._scanner_cfg = get_scanner_config()
            except Exception as e:
                raise RuntimeError(
                    f"❌ CRITICAL: Failed to load scanner config from database: {e} - no fallback allowed"
                )

            if not self._scanner_cfg:
                raise RuntimeError(
                    "❌ CRITICAL: Scanner config returned None - database is REQUIRED, no fallback allowed"
                )

            # Build config from database - NO FALLBACK
            self.config = AlertCooldownConfig(
                epic_signal_cooldown_minutes=self._scanner_cfg.alert_cooldown_minutes,
                strategy_cooldown_minutes=self._scanner_cfg.strategy_cooldown_minutes,
                global_cooldown_seconds=self._scanner_cfg.global_cooldown_seconds,
                max_alerts_per_hour=self._scanner_cfg.max_alerts_per_hour,
                max_alerts_per_epic_hour=self._scanner_cfg.max_alerts_per_epic_hour
            )
            self.logger.info("[CONFIG:DB] ✅ Deduplication config loaded from database (NO FALLBACK)")

        # ✅ LOG THE ACTUAL CONFIGURATION BEING USED
        self.logger.info("🛡️ Alert deduplication manager initialized")
        self.logger.info(f"   Epic cooldown: {self.config.epic_signal_cooldown_minutes} minutes")
        self.logger.info(f"   Strategy cooldown: {self.config.strategy_cooldown_minutes} minutes")
        self.logger.info(f"   Global cooldown: {self.config.global_cooldown_seconds} seconds")
        self.logger.info(f"   Max alerts/hour: {self.config.max_alerts_per_hour}")
        self.logger.info(f"   Max alerts/epic/hour: {self.config.max_alerts_per_epic_hour}")

        # Initialize enhanced alert_history table if needed
        self._ensure_enhanced_table_structure()

        # In-memory caches for performance - values from database (NO FALLBACK)
        self._recent_signals_cache = {}  # {epic: {signal_type: last_timestamp}}
        self._signal_hash_cache = {}  # Dict of {hash: timestamp} for time-aware expiry
        self._cooldown_key_cache = {}  # {cooldown_key: timestamp} for race condition prevention

        # Cache settings from database - if config_override was used, use defaults
        if self._scanner_cfg:
            self._cache_expiry_minutes = self._scanner_cfg.signal_hash_cache_expiry_minutes
            self._max_cache_size = self._scanner_cfg.max_signal_hash_cache_size
            self._enable_hash_check = self._scanner_cfg.enable_signal_hash_check
            self._enable_time_hash = self._scanner_cfg.enable_time_based_hash_components
        else:
            # Using config_override - use sensible defaults
            self._cache_expiry_minutes = 15
            self._max_cache_size = 1000
            self._enable_hash_check = True
            self._enable_time_hash = True

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
        Enhanced with time components to prevent false positives
        """
        # Get current time and create time bucket (15-minute intervals)
        current_time = datetime.now()
        time_bucket = current_time.strftime('%Y-%m-%d-%H') + f":{(current_time.minute // 15) * 15:02d}"

        # Determine market session
        hour = current_time.hour
        if 0 <= hour < 8:
            market_session = 'SYDNEY_TOKYO'
        elif 8 <= hour < 16:
            market_session = 'LONDON'
        elif 16 <= hour < 24:
            market_session = 'NEW_YORK'
        else:
            market_session = 'TRANSITION'

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
            # NEW: Add time-based components (configurable)
            **({
                'time_bucket': time_bucket,  # 15-minute time window
                'market_session': market_session,  # Market session context
            } if self._enable_time_hash else {})
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
    
    def _cleanup_expired_cache(self):
        """Remove expired entries from in-memory hash cache"""
        now = datetime.now()
        expiry_threshold = now - timedelta(minutes=self._cache_expiry_minutes)

        # Count expired entries before cleanup
        expired_count = sum(1 for timestamp in self._signal_hash_cache.values()
                           if timestamp < expiry_threshold)

        if expired_count > 0:
            # Remove expired entries
            self._signal_hash_cache = {
                hash_key: timestamp for hash_key, timestamp in self._signal_hash_cache.items()
                if timestamp >= expiry_threshold
            }
            self.logger.debug(f"🧹 Cleaned {expired_count} expired cache entries, {len(self._signal_hash_cache)} remain")

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
        """Check if this exact signal hash was recently seen (with time-aware cache)"""
        # First cleanup expired cache entries
        self._cleanup_expired_cache()

        # Check in-memory cache (now time-aware)
        if signal_hash in self._signal_hash_cache:
            cache_timestamp = self._signal_hash_cache[signal_hash]
            time_diff = datetime.now() - cache_timestamp
            self.logger.debug(f"📅 Hash found in cache from {time_diff.total_seconds():.0f}s ago")
            return False, "Exact duplicate signal detected (hash match)"

        # Check database for recent hash matches (last 15 minutes)
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, alert_timestamp FROM alert_history
                WHERE signal_hash = %s
                AND alert_timestamp >= %s
                LIMIT 1
            """, (signal_hash, datetime.now() - timedelta(minutes=15)))

            result = cursor.fetchone()
            cursor.close()
            conn.close()

            if result:
                db_timestamp = result[1]
                time_diff = datetime.now() - db_timestamp
                self.logger.debug(f"📊 Hash found in database from {time_diff.total_seconds():.0f}s ago")
                return False, "Exact duplicate found in recent history"

            return True, ""

        except Exception as e:
            self.logger.error(f"Error checking signal hash: {e}")
            return True, ""  # Allow on error
    
    def _check_cooldown_periods(self, cooldown_key: str) -> Tuple[bool, str]:
        """Check time-based cooldown periods"""
        try:
            now = datetime.now()
            cooldown_seconds = self.config.epic_signal_cooldown_minutes * 60

            # FIRST: Check in-memory cache to prevent race conditions
            # This catches duplicate signals that arrive within milliseconds of each other
            if cooldown_key in self._cooldown_key_cache:
                cache_timestamp = self._cooldown_key_cache[cooldown_key]
                time_diff = (now - cache_timestamp).total_seconds()
                remaining = cooldown_seconds - time_diff

                if remaining > 0:
                    self.logger.debug(f"🚫 Cooldown blocked by in-memory cache: {cooldown_key}")
                    return False, f"Cooldown active (cache): {remaining:.0f}s remaining"

            # SECOND: Check database for recent alerts with same cooldown key
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT alert_timestamp, epic, signal_type, strategy
                FROM alert_history
                WHERE cooldown_key = %s
                AND alert_timestamp >= %s
                ORDER BY alert_timestamp DESC
                LIMIT 1
            """, (cooldown_key, now - timedelta(minutes=self.config.epic_signal_cooldown_minutes)))

            result = cursor.fetchone()
            cursor.close()
            conn.close()

            if result:
                last_alert_time = result[0]
                time_diff = now - last_alert_time
                remaining_cooldown = cooldown_seconds - time_diff.total_seconds()

                if remaining_cooldown > 0:
                    # Update cache from database to prevent future race conditions
                    self._cooldown_key_cache[cooldown_key] = last_alert_time
                    return False, f"Cooldown active: {remaining_cooldown:.0f}s remaining"

            return True, ""

        except Exception as e:
            self.logger.error(f"Error checking cooldown periods: {e}")
            return True, ""  # Allow on error

    def _check_trade_cooldown(self, epic: str) -> Tuple[bool, str]:
        """
        Check if epic is in trade cooldown based on actual trade_log entries.
        This prevents wasting Claude API calls on signals that will be rejected at execution.

        Checks:
        1. ACTIVE TRADES - Block if there's an open position (tracking, pending, break_even, trailing)
        2. Recent trade OPENINGS (prevents back-to-back entries within cooldown period)
        3. Recent trade CLOSURES (allows market to settle after close)
        """
        try:
            # Get cooldown settings from database (NO FALLBACK)
            if self._scanner_cfg:
                trade_cooldown_minutes = self._scanner_cfg.trade_cooldown_minutes
                trade_cooldown_enabled = self._scanner_cfg.trade_cooldown_enabled
            else:
                # Using config_override - use sensible defaults
                trade_cooldown_minutes = 30
                trade_cooldown_enabled = True

            # Skip check if cooldown is disabled
            if not trade_cooldown_enabled:
                return True, ""

            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            current_time = datetime.utcnow()
            cooldown_threshold = current_time - timedelta(minutes=trade_cooldown_minutes)

            # CRITICAL: Check for ACTIVE/OPEN trades first (any status that means position is open)
            cursor.execute("""
                SELECT timestamp, status, direction
                FROM trade_log
                WHERE symbol = %s
                  AND status IN ('pending', 'tracking', 'break_even', 'trailing')
                ORDER BY timestamp DESC
                LIMIT 1
            """, (epic,))

            active_trade = cursor.fetchone()

            if active_trade:
                opening_time = active_trade[0]
                status = active_trade[1]
                direction = active_trade[2]

                # Handle timezone
                if opening_time and opening_time.tzinfo is None:
                    opening_time = opening_time.replace(tzinfo=None)

                time_since_open = "unknown"
                if opening_time and opening_time <= current_time:
                    minutes_open = int((current_time - opening_time).total_seconds() / 60)
                    time_since_open = f"{minutes_open}min"

                cursor.close()
                conn.close()
                return False, f"Active {direction} trade ({status}) open for {time_since_open}"

            # Check for recent trade OPENINGS within cooldown period
            cursor.execute("""
                SELECT timestamp, status, direction
                FROM trade_log
                WHERE symbol = %s
                  AND timestamp IS NOT NULL
                  AND timestamp >= %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (epic, cooldown_threshold))

            recent_opened = cursor.fetchone()

            if recent_opened:
                opening_time = recent_opened[0]
                status = recent_opened[1]

                # Handle timezone-naive timestamps
                if opening_time.tzinfo is None:
                    opening_time = opening_time.replace(tzinfo=None)

                # Check if timestamp is valid (not in the future)
                if opening_time <= current_time:
                    time_elapsed = (current_time - opening_time).total_seconds() / 60
                    remaining = int(trade_cooldown_minutes - time_elapsed)

                    if remaining > 0:
                        cursor.close()
                        conn.close()
                        return False, f"Trade opened {int(time_elapsed)}min ago, {remaining}min cooldown remaining"

            # Check for recent trade CLOSURES
            cursor.execute("""
                SELECT closed_at, status, direction
                FROM trade_log
                WHERE symbol = %s
                  AND status IN ('closed', 'expired')
                  AND closed_at IS NOT NULL
                  AND closed_at >= %s
                ORDER BY closed_at DESC
                LIMIT 1
            """, (epic, cooldown_threshold))

            recent_closed = cursor.fetchone()

            if recent_closed:
                close_time = recent_closed[0]

                # Handle timezone-naive timestamps
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=None)

                # Check if timestamp is valid (not in the future)
                if close_time <= current_time:
                    time_elapsed = (current_time - close_time).total_seconds() / 60
                    remaining = int(trade_cooldown_minutes - time_elapsed)

                    if remaining > 0:
                        cursor.close()
                        conn.close()
                        return False, f"Trade closed {int(time_elapsed)}min ago, {remaining}min cooldown remaining"

            cursor.close()
            conn.close()
            return True, ""

        except Exception as e:
            self.logger.error(f"Error checking trade cooldown for {epic}: {e}")
            return True, ""  # Allow on error (fail-open for safety)

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

        # Enhanced logging for hash generation
        self.logger.debug(f"🔍 Dedup check for {epic} {signal_type} ({strategy}):")
        self.logger.debug(f"   Generated hash: {signal_hash[:8]}...")
        self.logger.debug(f"   Cooldown key: {cooldown_key}")
        self.logger.debug(f"   Cache size: {len(self._signal_hash_cache)} entries")

        metadata = {
            'signal_hash': signal_hash,
            'cooldown_key': cooldown_key,
            'check_timestamp': datetime.now().isoformat(),
            'cache_size': len(self._signal_hash_cache)
        }
        
        # Run all deduplication checks
        checks = [
            ("Global Rate Limit", self._check_global_rate_limits()),
            ("Epic Rate Limit", self._check_epic_rate_limits(epic)),
            ("Cooldown Period", self._check_cooldown_periods(cooldown_key)),
            ("Trade Cooldown", self._check_trade_cooldown(epic)),  # Check actual trade_log
        ]

        # Only add hash check if enabled (master switch)
        if self._enable_hash_check:
            checks.insert(2, ("Signal Hash Duplicate", self._check_signal_hash_duplicate(signal_hash)))
        
        # Log check results
        self.logger.debug(f"🔍 Deduplication checks for {epic} {signal_type}:")
        
        for check_name, (passed, reason) in checks:
            if not passed:
                self.logger.info(f"🚫 Alert blocked: {check_name} - {reason}")
                self.logger.debug(f"   Epic: {epic}, Strategy: {strategy}, Signal: {signal_type}")
                self.logger.debug(f"   Hash: {signal_hash[:12]}..., Cache entries: {len(self._signal_hash_cache)}")
                metadata['blocked_by'] = check_name
                metadata['block_reason'] = reason
                return False, f"{check_name}: {reason}", metadata
            else:
                self.logger.debug(f"✅ {check_name}: Passed")
        
        # All checks passed - update caches
        current_time = datetime.now()
        self._signal_hash_cache[signal_hash] = current_time
        self._cooldown_key_cache[cooldown_key] = current_time  # Prevent race condition duplicates
        self._hourly_alert_count += 1

        # Update in-memory cache
        if epic not in self._recent_signals_cache:
            self._recent_signals_cache[epic] = {}
        self._recent_signals_cache[epic][signal_type] = current_time

        # Clean up old cache entries (time-based cleanup)
        if len(self._signal_hash_cache) > self._max_cache_size:
            # Force cleanup if cache gets too large
            self._cleanup_expired_cache()
            # If still too large after expiry cleanup, remove oldest entries
            if len(self._signal_hash_cache) > self._max_cache_size:
                sorted_items = sorted(self._signal_hash_cache.items(), key=lambda x: x[1])
                # Keep newest 80% of max size
                keep_size = int(self._max_cache_size * 0.8)
                self._signal_hash_cache = dict(sorted_items[-keep_size:])
                self.logger.debug(f"🧹 Performed size-based cache cleanup, kept {keep_size} newest entries")
        
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
        """Get current deduplication statistics with enhanced cache info"""
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

            # Calculate cache age statistics
            now = datetime.now()
            cache_ages = [(now - timestamp).total_seconds() for timestamp in self._signal_hash_cache.values()]
            avg_cache_age = sum(cache_ages) / len(cache_ages) if cache_ages else 0
            max_cache_age = max(cache_ages) if cache_ages else 0

            return {
                'total_alerts_24h': stats[0] if stats else 0,
                'unique_epics_24h': stats[1] if stats else 0,
                'unique_signals_24h': stats[2] if stats else 0,
                'avg_confidence_24h': float(stats[3]) if stats and stats[3] else 0,
                'cache_size': len(self._signal_hash_cache),
                'hourly_alert_count': self._hourly_alert_count,
                'cache_stats': {
                    'avg_age_seconds': avg_cache_age,
                    'max_age_seconds': max_cache_age,
                    'expiry_threshold_seconds': self._cache_expiry_minutes * 60
                },
                'config': {
                    'epic_signal_cooldown_minutes': self.config.epic_signal_cooldown_minutes,
                    'cache_expiry_minutes': self._cache_expiry_minutes,
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
    """
    Factory function to create deduplication manager.

    If custom config values are provided, they override database defaults.
    If no custom values provided, database configuration is used (NO FALLBACK to config.py).
    """
    if epic_cooldown_minutes is not None or max_alerts_per_hour is not None:
        # Custom config provided - load base from database then override
        if not SCANNER_CONFIG_AVAILABLE:
            raise RuntimeError(
                "❌ CRITICAL: Scanner config service not available - database is REQUIRED, no fallback allowed"
            )
        scanner_cfg = get_scanner_config()
        if not scanner_cfg:
            raise RuntimeError(
                "❌ CRITICAL: Scanner config returned None - database is REQUIRED, no fallback allowed"
            )

        custom_config = AlertCooldownConfig(
            epic_signal_cooldown_minutes=epic_cooldown_minutes if epic_cooldown_minutes is not None else scanner_cfg.alert_cooldown_minutes,
            max_alerts_per_hour=max_alerts_per_hour if max_alerts_per_hour is not None else scanner_cfg.max_alerts_per_hour,
            strategy_cooldown_minutes=scanner_cfg.strategy_cooldown_minutes,
            global_cooldown_seconds=scanner_cfg.global_cooldown_seconds,
            max_alerts_per_epic_hour=scanner_cfg.max_alerts_per_epic_hour
        )
        return AlertDeduplicationManager(db_manager, custom_config)
    else:
        # No custom config - use database defaults (NO FALLBACK)
        return AlertDeduplicationManager(db_manager)


def save_alert_if_allowed(db_manager, alert_history_manager, signal: Dict, 
                         alert_message: str = None) -> Optional[int]:
    """
    Convenience function to save alert with deduplication in one call
    """
    dedup_manager = AlertDeduplicationManager(db_manager)
    return dedup_manager.save_alert_with_deduplication(
        alert_history_manager, signal, alert_message
    )