# core/memory_cache.py
"""
In-Memory Data Cache for Forex Scanner
Loads historical candle data into memory for ultra-fast backtest performance
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import threading
import gc
from dataclasses import dataclass

try:
    from .database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_memory_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    load_time_seconds: float = 0.0
    data_range_start: Optional[datetime] = None
    data_range_end: Optional[datetime] = None
    total_rows: int = 0
    epics_cached: int = 0


class InMemoryForexCache:
    """
    High-performance in-memory cache for forex candle data
    Optimized for backtest performance with 16GB+ memory systems
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

        # Cache storage: epic -> timeframe -> DataFrame
        self.cache: Dict[str, Dict[int, pd.DataFrame]] = {}

        # Cache metadata
        self.stats = CacheStats()
        self.is_loaded = False
        self.load_lock = threading.Lock()

        # Cache configuration
        self.max_memory_mb = 2048  # 2GB limit for safety on 16GB system
        self.compression_enabled = True

        self.logger.info("🧠 InMemoryForexCache initialized")

    def load_all_data(self, force_reload: bool = False) -> bool:
        """
        Load entire ig_candles table into memory
        Returns True if successful, False otherwise
        """
        with self.load_lock:
            if self.is_loaded and not force_reload:
                self.logger.info("📊 Cache already loaded, skipping")
                return True

            try:
                start_time = datetime.now()
                self.logger.info("🚀 Loading all forex data into memory...")

                # Clear existing cache
                self.cache.clear()
                gc.collect()

                # Load all data in one query
                query = """
                SELECT start_time, epic, timeframe, open, high, low, close, volume, ltv
                FROM ig_candles
                ORDER BY epic, timeframe, start_time
                """

                df_all = self.db_manager.execute_query(query)

                if df_all.empty:
                    self.logger.warning("❌ No data found in ig_candles table")
                    return False

                # Convert start_time to datetime if it's not already
                df_all['start_time'] = pd.to_datetime(df_all['start_time'])

                # Group by epic and timeframe for efficient storage
                self.logger.info(f"📈 Organizing {len(df_all):,} rows into cache structure...")

                grouped = df_all.groupby(['epic', 'timeframe'])

                for (epic, timeframe), group_df in grouped:
                    # Initialize epic cache if needed
                    if epic not in self.cache:
                        self.cache[epic] = {}

                    # Set start_time as index for fast time-based lookups
                    indexed_df = group_df.set_index('start_time').sort_index()

                    # Drop epic and timeframe columns since they're in the keys
                    indexed_df = indexed_df.drop(['epic', 'timeframe'], axis=1)

                    # Apply compression if enabled
                    if self.compression_enabled:
                        # Convert to more memory-efficient dtypes
                        indexed_df['open'] = indexed_df['open'].astype('float32')
                        indexed_df['high'] = indexed_df['high'].astype('float32')
                        indexed_df['low'] = indexed_df['low'].astype('float32')
                        indexed_df['close'] = indexed_df['close'].astype('float32')
                        indexed_df['volume'] = indexed_df['volume'].astype('int32')
                        indexed_df['ltv'] = indexed_df['ltv'].astype('int32')

                    self.cache[epic][timeframe] = indexed_df

                    self.logger.debug(f"✅ Cached {epic} {timeframe}min: {len(indexed_df):,} rows")

                # Calculate statistics
                end_time = datetime.now()
                self.stats.load_time_seconds = (end_time - start_time).total_seconds()
                self.stats.total_rows = len(df_all)
                self.stats.epics_cached = len(self.cache)
                self.stats.data_range_start = df_all['start_time'].min()
                self.stats.data_range_end = df_all['start_time'].max()
                self.stats.total_memory_mb = self._calculate_memory_usage()

                self.is_loaded = True

                self.logger.info(f"✅ Cache loaded successfully!")
                self.logger.info(f"   📊 {self.stats.total_rows:,} rows across {self.stats.epics_cached} epics")
                self.logger.info(f"   🕒 Load time: {self.stats.load_time_seconds:.1f} seconds")
                self.logger.info(f"   💾 Memory usage: {self.stats.total_memory_mb:.1f} MB")
                self.logger.info(f"   📅 Data range: {self.stats.data_range_start} to {self.stats.data_range_end}")

                return True

            except Exception as e:
                self.logger.error(f"❌ Failed to load cache: {e}")
                self.cache.clear()
                self.is_loaded = False
                return False

    def get_historical_data(
        self,
        epic: str,
        timeframe: int,
        start_time: datetime,
        end_time: datetime,
        required_bars: int = None
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data from cache (ultra-fast)

        Args:
            epic: Trading instrument epic
            timeframe: Timeframe in minutes (5, 15, 60, etc.)
            start_time: Start datetime
            end_time: End datetime
            required_bars: Minimum number of bars needed

        Returns:
            DataFrame with OHLCV data or None if not available
        """
        if not self.is_loaded:
            self.logger.warning("🚫 Cache not loaded, falling back to database")
            self.stats.cache_misses += 1
            return None

        if epic not in self.cache or timeframe not in self.cache[epic]:
            self.logger.debug(f"🚫 {epic} {timeframe}min not in cache")
            self.stats.cache_misses += 1
            return None

        try:
            # Get data from cache
            cached_df = self.cache[epic][timeframe]

            # Filter by time range
            mask = (cached_df.index >= start_time) & (cached_df.index <= end_time)
            result_df = cached_df.loc[mask].copy()

            # Check if we have enough bars
            if required_bars and len(result_df) < required_bars:
                # Try to get more data by expanding the time range backwards
                extended_start = start_time - timedelta(days=30)  # Go back 30 days
                mask = (cached_df.index >= extended_start) & (cached_df.index <= end_time)
                result_df = cached_df.loc[mask].copy()

                if len(result_df) < required_bars:
                    self.logger.warning(f"⚠️ Insufficient data: got {len(result_df)}, need {required_bars}")
                    self.stats.cache_misses += 1
                    return None

            # Reset index to have start_time as a column (to match database format)
            result_df = result_df.reset_index()

            self.stats.cache_hits += 1
            self.logger.debug(f"✅ Cache hit: {epic} {timeframe}min, {len(result_df)} rows")

            return result_df

        except Exception as e:
            self.logger.error(f"❌ Cache retrieval error: {e}")
            self.stats.cache_misses += 1
            return None

    def get_cache_stats(self) -> Dict:
        """Get detailed cache statistics"""
        hit_rate = 0.0
        total_requests = self.stats.cache_hits + self.stats.cache_misses
        if total_requests > 0:
            hit_rate = (self.stats.cache_hits / total_requests) * 100

        return {
            'loaded': self.is_loaded,
            'memory_usage_mb': self.stats.total_memory_mb,
            'total_rows': self.stats.total_rows,
            'epics_cached': self.stats.epics_cached,
            'data_range': {
                'start': self.stats.data_range_start.isoformat() if self.stats.data_range_start else None,
                'end': self.stats.data_range_end.isoformat() if self.stats.data_range_end else None
            },
            'performance': {
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'hit_rate_percent': hit_rate,
                'load_time_seconds': self.stats.load_time_seconds
            },
            'epic_breakdown': {
                epic: list(timeframes.keys())
                for epic, timeframes in self.cache.items()
            } if self.is_loaded else {}
        }

    def _calculate_memory_usage(self) -> float:
        """Calculate approximate memory usage in MB"""
        total_bytes = 0

        for epic, timeframes in self.cache.items():
            for timeframe, df in timeframes.items():
                # Estimate DataFrame memory usage
                total_bytes += df.memory_usage(deep=True).sum()

        return total_bytes / (1024 * 1024)  # Convert to MB

    def clear_cache(self):
        """Clear all cached data"""
        with self.load_lock:
            self.cache.clear()
            self.is_loaded = False
            self.stats = CacheStats()
            gc.collect()
            self.logger.info("🗑️ Cache cleared")

    def warmup_cache(self):
        """Warm up cache by loading all data"""
        if not self.is_loaded:
            self.logger.info("🔥 Warming up cache...")
            return self.load_all_data()
        return True


# Global cache instance (singleton pattern)
_global_cache: Optional[InMemoryForexCache] = None


def get_forex_cache(db_manager: DatabaseManager = None) -> InMemoryForexCache:
    """Get or create the global forex cache instance"""
    global _global_cache

    if _global_cache is None and db_manager is not None:
        _global_cache = InMemoryForexCache(db_manager)

    return _global_cache


def initialize_cache(db_manager: DatabaseManager, auto_load: bool = True) -> InMemoryForexCache:
    """Initialize the global cache"""
    global _global_cache

    _global_cache = InMemoryForexCache(db_manager)

    if auto_load:
        _global_cache.load_all_data()

    return _global_cache