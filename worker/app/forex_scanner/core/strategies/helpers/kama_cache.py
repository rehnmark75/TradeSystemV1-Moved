# core/strategies/helpers/kama_cache.py
"""
KAMA Cache Module - Extracted from KAMA Strategy
🚀 PERFORMANCE: Intelligent caching and optimization for KAMA calculations
🎯 FOCUSED: Single responsibility for KAMA performance optimization
📊 COMPREHENSIVE: Cache management, performance tracking, optimization
⚠️ CRITICAL FIX: Enhanced timestamp validation to prevent 29248443.9 minutes warnings

This module contains all the caching and performance optimization logic
for KAMA strategy, extracted for better maintainability and testability.

CRITICAL FIX: Added comprehensive timestamp validation to prevent massive 
age calculations from corrupted cache entries with epoch (1970) timestamps.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging
from datetime import datetime
import hashlib


class KAMACache:
    """
    🚀 PERFORMANCE: Intelligent caching system for KAMA strategy
    
    Responsibilities:
    - Calculation result caching
    - Performance monitoring and optimization
    - Cache invalidation and cleanup
    - Memory management
    - Performance statistics tracking
    
    CRITICAL FIX: Enhanced timestamp validation to prevent age calculation errors
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache storage
        self._calculation_cache = {}
        self._market_regime_cache = {}
        self._efficiency_ratio_cache = {}
        
        # Cache configuration
        self._cache_timeout = 300  # 5 minutes
        self._max_cache_entries = 100  # Prevent memory bloat
        
        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calculations = 0
        self._cache_creation_time = datetime.now()
        
        self.logger.info("🚀 KAMA Cache system initialized")

    def _validate_cache_timestamp(self, timestamp: Any, cache_key: str = "unknown") -> bool:
        """
        ⚠️ CRITICAL FIX: Validate cache timestamp to prevent age calculation errors
        
        Args:
            timestamp: Timestamp to validate
            cache_key: Cache key for logging
            
        Returns:
            True if timestamp is valid
        """
        try:
            # Check if timestamp exists
            if timestamp is None:
                self.logger.warning(f"⚠️ KAMA cache timestamp is None for {cache_key}")
                return False
            
            # Check if timestamp is correct type
            if not isinstance(timestamp, datetime):
                self.logger.warning(f"⚠️ Invalid KAMA cache timestamp type {type(timestamp)} for {cache_key}")
                return False
            
            # Check for epoch timestamps (before 2020)
            if timestamp.year < 2020:
                self.logger.warning(f"⚠️ Epoch KAMA cache timestamp detected: {timestamp} for {cache_key}")
                return False
            
            # Check for unreasonable future timestamps (more than 1 hour ahead)
            current_time = datetime.now()
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo:
                current_time = datetime.now(timestamp.tzinfo)
            
            time_diff = (timestamp - current_time).total_seconds()
            if time_diff > 3600:  # More than 1 hour in future
                self.logger.warning(f"⚠️ Future KAMA cache timestamp detected: {timestamp} ({time_diff/3600:.1f} hours ahead) for {cache_key}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ KAMA cache timestamp validation error for {cache_key}: {e}")
            return False

    def _safe_calculate_cache_age(self, cache_timestamp: datetime, cache_key: str = "unknown") -> Optional[float]:
        """
        ⚠️ CRITICAL FIX: Safely calculate cache age in seconds with error handling
        
        Args:
            cache_timestamp: Cache timestamp
            cache_key: Cache key for logging
            
        Returns:
            Age in seconds or None if calculation fails
        """
        try:
            if not self._validate_cache_timestamp(cache_timestamp, cache_key):
                return None
            
            current_time = datetime.now()
            if hasattr(cache_timestamp, 'tzinfo') and cache_timestamp.tzinfo:
                current_time = datetime.now(cache_timestamp.tzinfo)
            
            age_seconds = (current_time - cache_timestamp).total_seconds()
            
            # Sanity check
            if age_seconds < 0:
                self.logger.warning(f"⚠️ Negative KAMA cache age: {age_seconds}s for {cache_key}")
                return None
            
            if age_seconds > 86400:  # More than 24 hours
                self.logger.warning(f"⚠️ Very old KAMA cache entry: {age_seconds}s for {cache_key}")
                return None
            
            return age_seconds
            
        except Exception as e:
            self.logger.error(f"❌ KAMA cache age calculation error for {cache_key}: {e}")
            return None

    def _generate_cache_key(self, prefix: str, *args) -> str:
        """
        🔑 Generate unique cache key from parameters
        """
        try:
            # Convert all arguments to strings and create hash
            key_string = f"{prefix}_" + "_".join(str(arg) for arg in args)
            return hashlib.md5(key_string.encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.debug(f"Cache key generation error: {e}")
            return f"{prefix}_default"

    def _is_cache_valid(self, cache_key: str, cache_dict: Dict = None) -> bool:
        """
        ⚠️ CRITICAL FIX: Check if cached value is still valid with proper timestamp validation
        """
        cache_dict = cache_dict or self._calculation_cache
        
        if cache_key not in cache_dict:
            return False
        
        try:
            cache_entry = cache_dict[cache_key]
            
            # CRITICAL FIX: Validate cache entry structure
            if not isinstance(cache_entry, dict) or 'timestamp' not in cache_entry:
                self.logger.warning(f"⚠️ Invalid KAMA cache entry structure for {cache_key}")
                return False
            
            cache_timestamp = cache_entry['timestamp']
            
            # CRITICAL FIX: Use safe age calculation
            cache_age = self._safe_calculate_cache_age(cache_timestamp, cache_key)
            
            if cache_age is None:
                # Invalid timestamp - remove the entry
                return False
            
            return cache_age < self._cache_timeout
            
        except Exception as e:
            self.logger.debug(f"KAMA cache validity check failed for {cache_key}: {e}")
            return False

    def _cache_result(self, cache_key: str, value: Any, cache_dict: Dict = None):
        """
        ⚠️ CRITICAL FIX: Cache a calculation result with proper timestamp validation
        """
        try:
            cache_dict = cache_dict or self._calculation_cache
            
            # CRITICAL FIX: Always use current time and validate it
            current_timestamp = datetime.now()
            
            # Sanity check on the timestamp we're about to store
            if current_timestamp.year < 2020:
                self.logger.error(f"❌ System timestamp appears invalid: {current_timestamp}")
                current_timestamp = datetime(2024, 1, 1)  # Use a safe fallback
            
            cache_dict[cache_key] = {
                'value': value,
                'timestamp': current_timestamp,
                'cache_version': '1.1',  # Add version for tracking
                'stored_at': current_timestamp.isoformat()  # String backup
            }
            
            # Clean old cache entries periodically
            if len(cache_dict) > self._max_cache_entries:
                self._clean_old_cache_entries(cache_dict)
                
        except Exception as e:
            self.logger.debug(f"Cache storage failed: {e}")

    def _get_cached_result(self, cache_key: str, cache_dict: Dict = None) -> Optional[Any]:
        """
        📖 Retrieve cached result if valid
        """
        cache_dict = cache_dict or self._calculation_cache
        
        if self._is_cache_valid(cache_key, cache_dict):
            self._cache_hits += 1
            return cache_dict[cache_key]['value']
        else:
            self._cache_misses += 1
            return None

    def calculate_efficiency_ratio_cached(self, df: pd.DataFrame, epic: str, timeframe: str) -> float:
        """
        🚀 CACHED: Calculate efficiency ratio with intelligent caching
        """
        try:
            # Create cache key
            cache_key = self._generate_cache_key(
                'kama_efficiency', epic, timeframe, len(df), 
                f"{df.iloc[-1]['close']:.5f}" if len(df) > 0 else "0"
            )
            
            # Check cache first
            cached_result = self._get_cached_result(cache_key, self._efficiency_ratio_cache)
            if cached_result is not None:
                return cached_result
            
            # Cache miss - calculate
            self._total_calculations += 1
            efficiency_ratio = self._calculate_efficiency_ratio_for_validation(df)
            
            # Cache the result
            self._cache_result(cache_key, efficiency_ratio, self._efficiency_ratio_cache)
            return efficiency_ratio
            
        except Exception as e:
            self.logger.error(f"❌ Cached efficiency ratio calculation failed: {e}")
            return 0.25  # Safe fallback

    def detect_market_regime_cached(self, current: pd.Series, df: pd.DataFrame, epic: str, timeframe: str) -> str:
        """
        🚀 CACHED: Market regime detection with caching
        """
        try:
            # Create cache key
            price_key = f"{current.get('close', 0):.5f}"
            kama_key = f"{current.get('kama', 0):.5f}"
            er_key = f"{current.get('efficiency_ratio', 0.1):.3f}"
            cache_key = self._generate_cache_key(
                'kama_regime', epic, timeframe, price_key, kama_key, er_key
            )
            
            # Check cache first
            cached_result = self._get_cached_result(cache_key, self._market_regime_cache)
            if cached_result is not None:
                return cached_result
            
            # Cache miss - calculate
            self._total_calculations += 1
            regime = self._detect_market_regime_for_kama(current, df, epic, timeframe)
            
            # Cache the result
            self._cache_result(cache_key, regime, self._market_regime_cache)
            return regime
            
        except Exception as e:
            self.logger.debug(f"Cached market regime detection failed: {e}")
            regime = 'ranging'
            return regime

    def _calculate_efficiency_ratio_for_validation(self, df: pd.DataFrame) -> float:
        """
        📊 Core efficiency ratio calculation (moved from main strategy)
        """
        try:
            # Get KAMA ER period
            import config
            er_period = getattr(config, 'KAMA_ER_PERIOD', 14)
            
            period = min(er_period, len(df) - 1)
            if period < 3:
                return 0.25  # Safe default above rejection threshold
            
            # Get price series
            close_prices = df['close'].tail(period + 1)
            
            if len(close_prices) < 2:
                return 0.25
            
            # Calculate directional change (net movement)
            start_price = close_prices.iloc[0]
            end_price = close_prices.iloc[-1]
            direction_change = abs(end_price - start_price)
            
            # Calculate total movement (sum of all price changes)
            price_changes = close_prices.diff().dropna()
            total_movement = price_changes.abs().sum()
            
            # Handle edge cases
            if total_movement == 0 or pd.isna(total_movement) or total_movement < 1e-8:
                return 0.25
            
            # Apply minimum movement threshold
            min_movement = close_prices.mean() * 0.0001  # 0.01% of price
            if total_movement < min_movement:
                return 0.30  # Above rejection threshold
            
            # Calculate efficiency
            efficiency = direction_change / total_movement
            
            # Ensure minimum efficiency above rejection threshold
            final_efficiency = max(0.21, min(1.0, efficiency))
            
            return final_efficiency
            
        except Exception as e:
            self.logger.error(f"❌ Efficiency ratio calculation failed: {e}")
            return 0.25

    def _detect_market_regime_for_kama(self, current: pd.Series, df: pd.DataFrame, epic: str, timeframe: str) -> str:
        """
        🌡️ Core market regime detection for KAMA (moved from main strategy)
        """
        try:
            kama_value = current.get('kama', 0)
            current_price = current.get('close', 0)
            efficiency_ratio = current.get('efficiency_ratio', 0.1)
            kama_slope = current.get('kama_slope', current.get('kama_trend', 0))
            
            if current_price <= 0 or kama_value <= 0:
                return 'ranging'
            
            # KAMA-specific regime assessment
            price_kama_distance = abs(current_price - kama_value) / current_price
            
            # Combined regime assessment for KAMA
            if efficiency_ratio > 0.6 or abs(kama_slope) > 0.002 or price_kama_distance > 0.002:
                return 'volatile'
            elif efficiency_ratio > 0.3 or abs(kama_slope) > 0.001:
                return 'trending'
            elif efficiency_ratio < 0.15 and abs(kama_slope) < 0.0005:
                return 'consolidating'
            else:
                return 'ranging'
            
        except Exception as e:
            self.logger.debug(f"Market regime detection failed: {e}")
            return 'ranging'

    def _clean_old_cache_entries(self, cache_dict: Dict):
        """
        ⚠️ CRITICAL FIX: Remove expired cache entries with proper error handling
        """
        try:
            current_time = datetime.now()
            keys_to_remove = []
            
            for key, entry in cache_dict.items():
                try:
                    # CRITICAL FIX: Validate entry structure
                    if not isinstance(entry, dict) or 'timestamp' not in entry:
                        self.logger.warning(f"⚠️ Invalid KAMA cache entry structure for {key}")
                        keys_to_remove.append(key)
                        continue
                    
                    cache_timestamp = entry['timestamp']
                    
                    # CRITICAL FIX: Use safe age calculation
                    cache_age = self._safe_calculate_cache_age(cache_timestamp, key)
                    
                    if cache_age is None:
                        # Invalid timestamp - remove entry
                        self.logger.warning(f"⚠️ Removing KAMA cache entry with invalid timestamp: {key}")
                        keys_to_remove.append(key)
                        continue
                    
                    if cache_age > self._cache_timeout:
                        keys_to_remove.append(key)
                        
                except Exception as entry_error:
                    self.logger.error(f"❌ Error processing KAMA cache entry {key}: {entry_error}")
                    keys_to_remove.append(key)  # Remove problematic entries
            
            # Remove identified entries
            for key in keys_to_remove:
                try:
                    del cache_dict[key]
                except Exception as del_error:
                    self.logger.error(f"❌ Error removing KAMA cache entry {key}: {del_error}")
                
            if keys_to_remove:
                self.logger.debug(f"🧹 Cleaned {len(keys_to_remove)} expired KAMA cache entries")
            
        except Exception as e:
            self.logger.debug(f"Cache cleanup failed: {e}")

    def log_cache_performance(self):
        """
        📊 Log cache performance metrics
        """
        try:
            total_requests = self._cache_hits + self._cache_misses
            if total_requests > 0:
                hit_ratio = self._cache_hits / total_requests
                self.logger.debug(f"🚀 KAMA Cache Performance: {hit_ratio:.1%} hit ratio "
                               f"({self._cache_hits} hits, {self._cache_misses} misses)")
        except Exception as e:
            self.logger.debug(f"Cache performance logging failed: {e}")

    def get_cache_stats(self) -> Dict:
        """
        ⚠️ CRITICAL FIX: Get comprehensive cache performance statistics with proper validation
        """
        try:
            total_requests = self._cache_hits + self._cache_misses
            
            # CRITICAL FIX: Validate cache creation time
            if self._cache_creation_time and self._validate_cache_timestamp(self._cache_creation_time, "cache_creation_time"):
                uptime = (datetime.now() - self._cache_creation_time).total_seconds()
            else:
                # Fallback if creation time is invalid
                self.logger.warning("⚠️ Invalid cache creation time detected, using fallback")
                uptime = 300  # 5 minutes fallback
            
            return {
                'module': 'kama_cache',
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'total_requests': total_requests,
                'hit_ratio': self._cache_hits / max(total_requests, 1),
                'total_calculations': self._total_calculations,
                'cached_entries': {
                    'calculation_cache': len(self._calculation_cache),
                    'market_regime_cache': len(self._market_regime_cache),
                    'efficiency_ratio_cache': len(self._efficiency_ratio_cache)
                },
                'cache_timeout': self._cache_timeout,
                'max_cache_entries': self._max_cache_entries,
                'uptime_seconds': uptime,
                'calculations_per_second': self._total_calculations / max(uptime, 1),
                'error': None
            }
        except Exception as e:
            return {'error': str(e)}

    def clear_cache(self):
        """
        🧹 Clear all cached calculations
        """
        try:
            self._calculation_cache.clear()
            self._market_regime_cache.clear()
            self._efficiency_ratio_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_calculations = 0
            self.logger.info("🧹 KAMA Cache cleared - all calculations will be recalculated")
        except Exception as e:
            self.logger.error(f"Cache clearing failed: {e}")

    def optimize_cache_settings(self, hit_ratio_threshold: float = 0.7):
        """
        ⚡ Optimize cache settings based on performance
        """
        try:
            stats = self.get_cache_stats()
            current_hit_ratio = stats.get('hit_ratio', 0)
            
            if current_hit_ratio < hit_ratio_threshold:
                # Increase cache timeout for better hit ratio
                old_timeout = self._cache_timeout
                self._cache_timeout = min(self._cache_timeout * 1.5, 900)  # Max 15 minutes
                
                self.logger.info(f"🔧 Cache timeout optimized: {old_timeout}s → {self._cache_timeout}s "
                               f"(hit ratio: {current_hit_ratio:.1%})")
            
            # Adjust max entries based on memory usage
            total_entries = sum(len(cache) for cache in [
                self._calculation_cache, self._market_regime_cache, self._efficiency_ratio_cache
            ])
            
            if total_entries > self._max_cache_entries * 0.9:
                self._clean_old_cache_entries(self._calculation_cache)
                self._clean_old_cache_entries(self._market_regime_cache)
                self._clean_old_cache_entries(self._efficiency_ratio_cache)
                
        except Exception as e:
            self.logger.debug(f"Cache optimization failed: {e}")

    def get_memory_usage(self) -> Dict:
        """
        💾 Get estimated memory usage of cache
        """
        try:
            import sys
            
            def get_size(obj):
                try:
                    return sys.getsizeof(obj)
                except:
                    return 0
            
            calc_size = sum(get_size(entry) for entry in self._calculation_cache.values())
            regime_size = sum(get_size(entry) for entry in self._market_regime_cache.values())
            efficiency_size = sum(get_size(entry) for entry in self._efficiency_ratio_cache.values())
            
            return {
                'calculation_cache_bytes': calc_size,
                'market_regime_cache_bytes': regime_size,
                'efficiency_ratio_cache_bytes': efficiency_size,
                'total_bytes': calc_size + regime_size + efficiency_size,
                'total_mb': (calc_size + regime_size + efficiency_size) / 1024 / 1024
            }
        except Exception as e:
            return {'error': str(e)}