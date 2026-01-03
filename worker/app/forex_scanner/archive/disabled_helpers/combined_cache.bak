# core/strategies/helpers/combined_cache.py
"""
Combined Strategy Cache - MODULAR HELPER
🔥 PERFORMANCE OPTIMIZED: Intelligent caching for combined strategy operations
🏗️ MODULAR: Focused on caching and performance optimization
🎯 MAINTAINABLE: Single responsibility - caching only
⚡ EFFICIENT: Smart cache management with automatic cleanup
⚠️ CRITICAL FIX: Enhanced timestamp validation to prevent 29248443.9 minutes warnings

CRITICAL FIX: Added comprehensive timestamp validation to prevent massive 
age calculations from corrupted cache entries with epoch (1970) timestamps.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import hashlib
import json
import time
from collections import OrderedDict


class CombinedCache:
    """
    🔥 CACHE: Performance caching for combined strategy operations
    
    Handles:
    - Strategy result caching
    - Signal combination caching
    - Market analysis caching
    - Performance optimization
    - Automatic cache cleanup
    
    CRITICAL FIX: Enhanced timestamp validation to prevent age calculation errors
    """
    
    def __init__(self, logger: logging.Logger, max_cache_size: int = 1000):
        self.logger = logger
        self.max_cache_size = max_cache_size
        
        # Main cache storage using OrderedDict for LRU behavior
        self._cache = OrderedDict()
        
        # Cache metadata
        self._cache_metadata = {}
        
        # Cache statistics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'cache_size': 0,
            'last_cleanup': datetime.now(),
            'total_cached_items': 0,
            'cache_memory_estimate': 0
        }
        
        # Cache configuration
        self.cache_config = {
            'max_age_minutes': 30,  # Cache items expire after 30 minutes
            'cleanup_threshold': 0.8,  # Cleanup when 80% full
            'enable_compression': True,  # Compress large cache items
            'max_item_size_kb': 100,  # Maximum size per cache item
            'strategy_result_ttl': 15,  # Strategy results TTL in minutes
            'market_analysis_ttl': 60,  # Market analysis TTL in minutes
        }
        
        self.logger.debug("✅ CombinedCache initialized")

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
                self.logger.warning(f"⚠️ Combined cache timestamp is None for {cache_key}")
                return False
            
            # Check if timestamp is correct type
            if not isinstance(timestamp, datetime):
                self.logger.warning(f"⚠️ Invalid combined cache timestamp type {type(timestamp)} for {cache_key}")
                return False
            
            # Check for epoch timestamps (before 2020)
            if timestamp.year < 2020:
                self.logger.warning(f"⚠️ Epoch combined cache timestamp detected: {timestamp} for {cache_key}")
                return False
            
            # Check for unreasonable future timestamps (more than 1 hour ahead)
            current_time = datetime.now()
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo:
                current_time = datetime.now(timestamp.tzinfo)
            
            time_diff = (timestamp - current_time).total_seconds()
            if time_diff > 3600:  # More than 1 hour in future
                self.logger.warning(f"⚠️ Future combined cache timestamp detected: {timestamp} ({time_diff/3600:.1f} hours ahead) for {cache_key}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Combined cache timestamp validation error for {cache_key}: {e}")
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
                self.logger.warning(f"⚠️ Negative combined cache age: {age_seconds}s for {cache_key}")
                return None
            
            if age_seconds > 86400:  # More than 24 hours
                self.logger.warning(f"⚠️ Very old combined cache entry: {age_seconds}s for {cache_key}")
                return None
            
            return age_seconds
            
        except Exception as e:
            self.logger.error(f"❌ Combined cache age calculation error for {cache_key}: {e}")
            return None

    def cache_result(self, key: str, result: Any, ttl_minutes: Optional[int] = None) -> bool:
        """
        ⚠️ CRITICAL FIX: Cache a result with proper timestamp validation
        
        Args:
            key: Cache key
            result: Result to cache
            ttl_minutes: Time to live in minutes (optional)
            
        Returns:
            True if cached successfully
        """
        try:
            # Generate cache key hash for consistent storage
            cache_key = self._generate_cache_key(key)
            
            # CRITICAL FIX: Always use current time and validate it
            current_timestamp = datetime.now()
            
            # Sanity check on the timestamp we're about to store
            if current_timestamp.year < 2020:
                self.logger.error(f"❌ System timestamp appears invalid: {current_timestamp}")
                current_timestamp = datetime(2024, 1, 1)  # Use a safe fallback
            
            # Prepare cache item
            cache_item = {
                'data': result,
                'timestamp': current_timestamp,
                'ttl_minutes': ttl_minutes or self.cache_config['max_age_minutes'],
                'access_count': 0,
                'size_estimate': self._estimate_size(result),
                'cache_version': '1.1',  # Add version for tracking
                'stored_at': current_timestamp.isoformat()  # String backup
            }
            
            # Check if item is too large
            if cache_item['size_estimate'] > self.cache_config['max_item_size_kb'] * 1024:
                self.logger.warning(f"⚠️ Cache item too large: {cache_item['size_estimate']} bytes")
                return False
            
            # Add to cache (OrderedDict maintains insertion order for LRU)
            self._cache[cache_key] = cache_item
            self._cache_metadata[cache_key] = {
                'original_key': key,
                'cached_at': current_timestamp,
                'category': self._determine_cache_category(key)
            }
            
            # Update statistics
            self.cache_stats['total_cached_items'] += 1
            self.cache_stats['cache_size'] = len(self._cache)
            self.cache_stats['cache_memory_estimate'] += cache_item['size_estimate']
            
            # Trigger cleanup if needed
            if len(self._cache) > self.max_cache_size * self.cache_config['cleanup_threshold']:
                self._cleanup_cache()
            
            self.logger.debug(f"✅ Cached item: {key} (size: {cache_item['size_estimate']} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Cache storage failed for {key}: {e}")
            return False

    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        ⚠️ CRITICAL FIX: Retrieve a cached result with proper timestamp validation
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found/expired
        """
        try:
            cache_key = self._generate_cache_key(key)
            
            if cache_key not in self._cache:
                self.cache_stats['misses'] += 1
                return None
            
            cache_item = self._cache[cache_key]
            
            # CRITICAL FIX: Validate cache item structure
            if not isinstance(cache_item, dict) or 'timestamp' not in cache_item:
                self.logger.warning(f"⚠️ Invalid combined cache item structure for {key}")
                if cache_key in self._cache:
                    del self._cache[cache_key]
                if cache_key in self._cache_metadata:
                    del self._cache_metadata[cache_key]
                self.cache_stats['misses'] += 1
                return None
            
            cache_timestamp = cache_item['timestamp']
            
            # CRITICAL FIX: Use safe age calculation
            age_seconds = self._safe_calculate_cache_age(cache_timestamp, cache_key)
            
            if age_seconds is None:
                # Invalid timestamp - remove the entry
                self.logger.warning(f"⚠️ Removing combined cache entry with invalid timestamp: {key}")
                if cache_key in self._cache:
                    del self._cache[cache_key]
                if cache_key in self._cache_metadata:
                    del self._cache_metadata[cache_key]
                self.cache_stats['misses'] += 1
                return None
            
            # Check if expired
            age_minutes = age_seconds / 60
            ttl_minutes = cache_item.get('ttl_minutes', self.cache_config['max_age_minutes'])
            
            if age_minutes > ttl_minutes:
                # Remove expired item
                del self._cache[cache_key]
                if cache_key in self._cache_metadata:
                    del self._cache_metadata[cache_key]
                self.cache_stats['misses'] += 1
                self.cache_stats['cache_size'] = len(self._cache)
                return None
            
            # Update access statistics
            cache_item['access_count'] += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            
            # Update statistics
            self.cache_stats['hits'] += 1
            
            self.logger.debug(f"✅ Cache hit: {key}")
            return cache_item['data']
            
        except Exception as e:
            self.logger.error(f"❌ Cache retrieval failed for {key}: {e}")
            self.cache_stats['misses'] += 1
            return None

    def cache_strategy_results(self, epic: str, timeframe: str, strategy_results: Dict[str, Any]) -> bool:
        """
        Cache results from multiple strategies
        
        Args:
            epic: Trading epic
            timeframe: Timeframe
            strategy_results: Dictionary of strategy results
            
        Returns:
            True if cached successfully
        """
        try:
            cache_key = f"strategy_results_{epic}_{timeframe}_{int(time.time() // 300)}"  # 5-minute buckets
            
            return self.cache_result(
                cache_key, 
                strategy_results, 
                ttl_minutes=self.cache_config['strategy_result_ttl']
            )
            
        except Exception as e:
            self.logger.error(f"❌ Strategy results caching failed: {e}")
            return False

    def get_cached_strategy_results(self, epic: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Get cached strategy results
        
        Args:
            epic: Trading epic
            timeframe: Timeframe
            
        Returns:
            Cached strategy results or None
        """
        try:
            cache_key = f"strategy_results_{epic}_{timeframe}_{int(time.time() // 300)}"  # 5-minute buckets
            return self.get_cached_result(cache_key)
            
        except Exception as e:
            self.logger.error(f"❌ Strategy results retrieval failed: {e}")
            return None

    def cache_market_analysis(self, epic: str, analysis_data: Dict[str, Any]) -> bool:
        """
        Cache market analysis results
        
        Args:
            epic: Trading epic
            analysis_data: Market analysis data
            
        Returns:
            True if cached successfully
        """
        try:
            cache_key = f"market_analysis_{epic}_{int(time.time() // 1800)}"  # 30-minute buckets
            
            return self.cache_result(
                cache_key,
                analysis_data,
                ttl_minutes=self.cache_config['market_analysis_ttl']
            )
            
        except Exception as e:
            self.logger.error(f"❌ Market analysis caching failed: {e}")
            return False

    def get_cached_market_analysis(self, epic: str) -> Optional[Dict[str, Any]]:
        """
        Get cached market analysis
        
        Args:
            epic: Trading epic
            
        Returns:
            Cached market analysis or None
        """
        try:
            cache_key = f"market_analysis_{epic}_{int(time.time() // 1800)}"  # 30-minute buckets
            return self.get_cached_result(cache_key)
            
        except Exception as e:
            self.logger.error(f"❌ Market analysis retrieval failed: {e}")
            return None

    def cache_combination_result(self, combination_key: str, combined_signal: Dict[str, Any]) -> bool:
        """
        Cache combined signal result
        
        Args:
            combination_key: Unique key for this combination
            combined_signal: Combined signal data
            
        Returns:
            True if cached successfully
        """
        try:
            cache_key = f"combination_{combination_key}"
            
            return self.cache_result(
                cache_key,
                combined_signal,
                ttl_minutes=5  # Short TTL for signal combinations
            )
            
        except Exception as e:
            self.logger.error(f"❌ Combination result caching failed: {e}")
            return False

    def get_cached_combination(self, combination_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached combination result
        
        Args:
            combination_key: Unique key for this combination
            
        Returns:
            Cached combination result or None
        """
        try:
            cache_key = f"combination_{combination_key}"
            return self.get_cached_result(cache_key)
            
        except Exception as e:
            self.logger.error(f"❌ Combination result retrieval failed: {e}")
            return None

    def _generate_cache_key(self, key: str) -> str:
        """Generate consistent cache key hash"""
        try:
            # Create hash of the key for consistent storage
            return hashlib.md5(key.encode('utf-8')).hexdigest()
        except Exception:
            # Fallback to original key if hashing fails
            return key

    def _estimate_size(self, obj: Any) -> int:
        """Estimate the size of an object in bytes"""
        try:
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, dict):
                return len(json.dumps(obj, default=str).encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            else:
                return len(str(obj).encode('utf-8'))
        except Exception:
            return 1024  # Default estimate

    def _determine_cache_category(self, key: str) -> str:
        """Determine the category of a cache key"""
        if 'strategy_results' in key:
            return 'strategy_results'
        elif 'market_analysis' in key:
            return 'market_analysis'
        elif 'combination' in key:
            return 'signal_combination'
        else:
            return 'general'

    def _cleanup_cache(self) -> None:
        """⚠️ CRITICAL FIX: Clean up expired and least recently used cache items with proper validation"""
        try:
            cleanup_start = time.time()
            items_removed = 0
            memory_freed = 0
            
            # Remove expired and invalid items first
            current_time = datetime.now()
            expired_keys = []
            
            for cache_key, cache_item in self._cache.items():
                try:
                    # CRITICAL FIX: Validate cache item structure
                    if not isinstance(cache_item, dict) or 'timestamp' not in cache_item:
                        self.logger.warning(f"⚠️ Invalid combined cache item structure during cleanup: {cache_key}")
                        expired_keys.append(cache_key)
                        continue
                    
                    cache_timestamp = cache_item['timestamp']
                    
                    # CRITICAL FIX: Use safe age calculation
                    age_seconds = self._safe_calculate_cache_age(cache_timestamp, cache_key)
                    
                    if age_seconds is None:
                        # Invalid timestamp - remove entry
                        self.logger.warning(f"⚠️ Removing combined cache entry with invalid timestamp during cleanup: {cache_key}")
                        expired_keys.append(cache_key)
                        continue
                    
                    age_minutes = age_seconds / 60
                    ttl_minutes = cache_item.get('ttl_minutes', self.cache_config['max_age_minutes'])
                    
                    if age_minutes > ttl_minutes:
                        expired_keys.append(cache_key)
                        
                except Exception as item_error:
                    self.logger.error(f"❌ Error processing combined cache item {cache_key} during cleanup: {item_error}")
                    expired_keys.append(cache_key)  # Remove problematic entries
            
            # Remove expired/invalid entries
            for key in expired_keys:
                try:
                    cache_item = self._cache.get(key, {})
                    memory_freed += cache_item.get('size_estimate', 0)
                    if key in self._cache:
                        del self._cache[key]
                    if key in self._cache_metadata:
                        del self._cache_metadata[key]
                    items_removed += 1
                except Exception as del_error:
                    self.logger.error(f"❌ Error removing combined cache entry {key}: {del_error}")
            
            # If still over capacity, remove LRU items
            while len(self._cache) > self.max_cache_size:
                try:
                    # OrderedDict pops from the beginning (least recently used)
                    cache_key, cache_item = self._cache.popitem(last=False)
                    memory_freed += cache_item.get('size_estimate', 0)
                    if cache_key in self._cache_metadata:
                        del self._cache_metadata[cache_key]
                    items_removed += 1
                except Exception as lru_error:
                    self.logger.error(f"❌ Error removing LRU cache entry: {lru_error}")
                    break  # Exit loop if we can't remove items
            
            # Update statistics
            self.cache_stats['cache_size'] = len(self._cache)
            self.cache_stats['cache_memory_estimate'] -= memory_freed
            self.cache_stats['last_cleanup'] = current_time
            
            cleanup_time = time.time() - cleanup_start
            
            self.logger.debug(f"🧹 Combined cache cleanup: {items_removed} items removed, "
                            f"{memory_freed} bytes freed, {cleanup_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Combined cache cleanup failed: {e}")

    def clear_cache(self) -> None:
        """Clear all cache entries"""
        try:
            cache_size_before = len(self._cache)
            memory_before = self.cache_stats['cache_memory_estimate']
            
            self._cache.clear()
            self._cache_metadata.clear()
            
            # Reset statistics
            self.cache_stats['cache_size'] = 0
            self.cache_stats['cache_memory_estimate'] = 0
            
            self.logger.info(f"🧹 Cache cleared: {cache_size_before} items, {memory_before} bytes")
            
        except Exception as e:
            self.logger.error(f"❌ Cache clearing failed: {e}")

    def clear_category(self, category: str) -> None:
        """Clear cache entries by category"""
        try:
            keys_to_remove = []
            
            for cache_key, metadata in self._cache_metadata.items():
                if metadata.get('category') == category:
                    keys_to_remove.append(cache_key)
            
            items_removed = 0
            memory_freed = 0
            
            for key in keys_to_remove:
                if key in self._cache:
                    cache_item = self._cache[key]
                    memory_freed += cache_item.get('size_estimate', 0)
                    del self._cache[key]
                    items_removed += 1
                
                if key in self._cache_metadata:
                    del self._cache_metadata[key]
            
            # Update statistics
            self.cache_stats['cache_size'] = len(self._cache)
            self.cache_stats['cache_memory_estimate'] -= memory_freed
            
            self.logger.info(f"🧹 Cleared {category} cache: {items_removed} items, {memory_freed} bytes")
            
        except Exception as e:
            self.logger.error(f"❌ Category cache clearing failed: {e}")

    def optimize_cache_settings(self) -> None:
        """Optimize cache settings based on usage patterns"""
        try:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            
            if total_requests > 0:
                hit_rate = self.cache_stats['hits'] / total_requests
                
                # Adjust cache size based on hit rate
                if hit_rate < 0.5:  # Low hit rate
                    # Increase cache size or reduce TTL
                    self.max_cache_size = min(2000, int(self.max_cache_size * 1.2))
                    self.cache_config['max_age_minutes'] = max(15, int(self.cache_config['max_age_minutes'] * 0.8))
                elif hit_rate > 0.8:  # High hit rate
                    # Can afford to reduce cache size or increase TTL
                    self.cache_config['max_age_minutes'] = min(60, int(self.cache_config['max_age_minutes'] * 1.1))
                
                self.logger.info(f"🧠 Cache optimized: hit_rate={hit_rate:.2%}, size={self.max_cache_size}, ttl={self.cache_config['max_age_minutes']}min")
            
        except Exception as e:
            self.logger.error(f"❌ Cache optimization failed: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            # Category breakdown
            category_stats = {}
            for metadata in self._cache_metadata.values():
                category = metadata.get('category', 'unknown')
                if category not in category_stats:
                    category_stats[category] = 0
                category_stats[category] += 1
            
            # Memory usage
            memory_mb = self.cache_stats['cache_memory_estimate'] / (1024 * 1024)
            
            return {
                'cache_size': self.cache_stats['cache_size'],
                'max_cache_size': self.max_cache_size,
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'],
                'hit_rate_percent': hit_rate,
                'total_cached_items': self.cache_stats['total_cached_items'],
                'memory_usage_mb': memory_mb,
                'last_cleanup': self.cache_stats['last_cleanup'].isoformat(),
                'category_breakdown': category_stats,
                'cache_config': self.cache_config,
                'performance_rating': 'excellent' if hit_rate > 80 else 'good' if hit_rate > 60 else 'needs_optimization',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Cache stats collection failed: {e}")
            return {'error': str(e)}

    def export_cache_diagnostics(self) -> Dict[str, Any]:
        """⚠️ CRITICAL FIX: Export detailed cache diagnostics with proper validation"""
        try:
            diagnostics = {
                'cache_health': {
                    'cache_size': len(self._cache),
                    'metadata_size': len(self._cache_metadata),
                    'sizes_match': len(self._cache) == len(self._cache_metadata)
                },
                'cache_distribution': {},
                'memory_analysis': {
                    'total_memory_estimate': self.cache_stats['cache_memory_estimate'],
                    'average_item_size': 0,
                    'largest_items': []
                },
                'access_patterns': {
                    'total_hits': self.cache_stats['hits'],
                    'total_misses': self.cache_stats['misses'],
                    'most_accessed_items': []
                },
                'age_analysis': {
                    'oldest_item_age_minutes': 0,
                    'newest_item_age_minutes': 0,
                    'average_age_minutes': 0,
                    'invalid_timestamps': 0
                }
            }
            
            # Analyze cache distribution by category
            for metadata in self._cache_metadata.values():
                category = metadata.get('category', 'unknown')
                if category not in diagnostics['cache_distribution']:
                    diagnostics['cache_distribution'][category] = 0
                diagnostics['cache_distribution'][category] += 1
            
            # Memory analysis
            if self._cache:
                valid_items = []
                for key, item in self._cache.items():
                    if isinstance(item, dict) and 'size_estimate' in item:
                        valid_items.append((key, item))
                
                if valid_items:
                    item_sizes = [item['size_estimate'] for _, item in valid_items]
                    diagnostics['memory_analysis']['average_item_size'] = sum(item_sizes) / len(item_sizes)
                    
                    # Find largest items
                    largest_items = sorted(valid_items, key=lambda x: x[1]['size_estimate'], reverse=True)[:5]
                    diagnostics['memory_analysis']['largest_items'] = [
                        {
                            'key': self._cache_metadata.get(key, {}).get('original_key', 'unknown'),
                            'size_bytes': item['size_estimate'],
                            'category': self._cache_metadata.get(key, {}).get('category', 'unknown')
                        }
                        for key, item in largest_items
                    ]
            
            # Access pattern analysis
            if self._cache:
                valid_access_items = []
                for key, item in self._cache.items():
                    if isinstance(item, dict) and 'access_count' in item:
                        valid_access_items.append((key, item['access_count']))
                
                if valid_access_items:
                    most_accessed = sorted(valid_access_items, key=lambda x: x[1], reverse=True)[:5]
                    diagnostics['access_patterns']['most_accessed_items'] = [
                        {
                            'key': self._cache_metadata.get(key, {}).get('original_key', 'unknown'),
                            'access_count': count,
                            'category': self._cache_metadata.get(key, {}).get('category', 'unknown')
                        }
                        for key, count in most_accessed
                    ]
            
            # CRITICAL FIX: Age analysis with proper validation
            if self._cache:
                current_time = datetime.now()
                valid_ages = []
                invalid_timestamps = 0
                
                for key, item in self._cache.items():
                    try:
                        if not isinstance(item, dict) or 'timestamp' not in item:
                            invalid_timestamps += 1
                            continue
                        
                        timestamp = item['timestamp']
                        age_seconds = self._safe_calculate_cache_age(timestamp, key)
                        
                        if age_seconds is None:
                            invalid_timestamps += 1
                        else:
                            valid_ages.append(age_seconds / 60)  # Convert to minutes
                            
                    except Exception as age_error:
                        self.logger.debug(f"Error calculating age for cache item {key}: {age_error}")
                        invalid_timestamps += 1
                
                if valid_ages:
                    diagnostics['age_analysis']['oldest_item_age_minutes'] = max(valid_ages)
                    diagnostics['age_analysis']['newest_item_age_minutes'] = min(valid_ages)
                    diagnostics['age_analysis']['average_age_minutes'] = sum(valid_ages) / len(valid_ages)
                
                diagnostics['age_analysis']['invalid_timestamps'] = invalid_timestamps
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"❌ Cache diagnostics export failed: {e}")
            return {'error': str(e)}

    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics information"""
        return {
            'module_name': 'CombinedCache',
            'initialization_successful': True,
            'cache_operational': True,
            'max_cache_size': self.max_cache_size,
            'current_cache_size': len(self._cache),
            'cache_config': self.cache_config,
            'cache_stats': self.get_cache_stats(),
            'cache_diagnostics': self.export_cache_diagnostics(),
            'timestamp': datetime.now().isoformat()
        }