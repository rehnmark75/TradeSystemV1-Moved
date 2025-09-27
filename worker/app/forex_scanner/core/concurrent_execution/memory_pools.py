# core/concurrent_execution/memory_pools.py
"""
Memory Pools - High-performance memory management for concurrent backtest execution
Implements zero-allocation hot paths and efficient memory recycling
"""

import gc
import sys
import threading
import time
import mmap
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict

import numpy as np
import pandas as pd

T = TypeVar('T')


class PoolType(Enum):
    """Types of memory pools"""
    SMALL_OBJECTS = "small_objects"        # < 1KB objects
    MEDIUM_OBJECTS = "medium_objects"      # 1KB - 1MB objects
    LARGE_OBJECTS = "large_objects"        # > 1MB objects
    PANDAS_DATAFRAMES = "pandas_dataframes"
    NUMPY_ARRAYS = "numpy_arrays"
    CACHED_INDICATORS = "cached_indicators"


@dataclass
class PoolStats:
    """Statistics for a memory pool"""
    pool_type: PoolType
    allocated_objects: int = 0
    available_objects: int = 0
    total_allocated_mb: float = 0.0
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def efficiency_percent(self) -> float:
        total_objects = self.allocated_objects + self.available_objects
        if total_objects == 0:
            return 100.0
        return (self.available_objects / total_objects) * 100.0


class MemoryPool(Generic[T]):
    """
    High-performance memory pool with object recycling

    Features:
    - Zero-allocation object reuse
    - Type-safe object management
    - Thread-safe operations
    - Automatic garbage collection
    - Memory usage monitoring
    """

    def __init__(self,
                 pool_type: PoolType,
                 object_factory: Callable[[], T],
                 object_reset: Optional[Callable[[T], None]] = None,
                 max_objects: int = 1000,
                 preallocation_size: int = 100,
                 auto_gc_threshold: int = 500):

        self.pool_type = pool_type
        self.object_factory = object_factory
        self.object_reset = object_reset or (lambda x: None)
        self.max_objects = max_objects
        self.preallocation_size = preallocation_size
        self.auto_gc_threshold = auto_gc_threshold

        # Pool storage
        self.available_objects: deque[T] = deque()
        self.allocated_objects: Dict[id, T] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = PoolStats(pool_type)

        # Logging
        self.logger = logging.getLogger(f"MemoryPool.{pool_type.value}")

    def preallocate(self):
        """Pre-allocate objects for zero-allocation hot paths"""
        with self.lock:
            for _ in range(self.preallocation_size):
                if len(self.available_objects) < self.max_objects:
                    obj = self.object_factory()
                    self.available_objects.append(obj)
                    self.stats.allocated_objects += 1

            self.logger.info(f"🚀 Pre-allocated {len(self.available_objects)} objects")

    def get(self) -> Optional[T]:
        """Get an object from the pool (zero allocation if available)"""
        with self.lock:
            self.stats.total_requests += 1

            if self.available_objects:
                # Cache hit - reuse existing object
                obj = self.available_objects.popleft()
                self.allocated_objects[id(obj)] = obj
                self.stats.available_objects = len(self.available_objects)
                self.stats.cache_hits += 1
                return obj

            # Cache miss - create new object if under limit
            if len(self.allocated_objects) < self.max_objects:
                obj = self.object_factory()
                self.allocated_objects[id(obj)] = obj
                self.stats.allocated_objects += 1
                self.stats.cache_misses += 1
                return obj

            # Pool exhausted
            self.stats.cache_misses += 1
            return None

    def return_object(self, obj: T):
        """Return an object to the pool for reuse"""
        with self.lock:
            obj_id = id(obj)

            if obj_id not in self.allocated_objects:
                return  # Not from this pool

            # Reset object state
            try:
                self.object_reset(obj)
            except Exception as e:
                self.logger.warning(f"⚠️ Error resetting object: {e}")
                # Don't return potentially corrupted object
                del self.allocated_objects[obj_id]
                return

            # Return to available pool
            self.available_objects.append(obj)
            del self.allocated_objects[obj_id]
            self.stats.available_objects = len(self.available_objects)

            # Trigger garbage collection if needed
            if len(self.available_objects) > self.auto_gc_threshold:
                self._auto_garbage_collect()

    def _auto_garbage_collect(self):
        """Automatically reduce pool size if too many objects are cached"""
        excess_objects = len(self.available_objects) - (self.preallocation_size * 2)
        if excess_objects > 0:
            for _ in range(excess_objects):
                if self.available_objects:
                    self.available_objects.popleft()

            self.stats.available_objects = len(self.available_objects)
            self.logger.info(f"🗑️ Auto-GC removed {excess_objects} excess objects")

    def cleanup(self):
        """Clean up all objects in the pool"""
        with self.lock:
            self.available_objects.clear()
            self.allocated_objects.clear()
            self.stats = PoolStats(self.pool_type)

            # Force garbage collection
            gc.collect()

            self.logger.info(f"🧹 Pool cleaned up")

    def get_stats(self) -> PoolStats:
        """Get current pool statistics"""
        with self.lock:
            self.stats.available_objects = len(self.available_objects)
            return self.stats


class CachedDataBuffer:
    """
    High-performance data buffer with caching for technical indicators

    Features:
    - Memory-mapped storage for large datasets
    - LRU cache for frequently accessed data
    - Zero-copy operations where possible
    - Efficient numpy array management
    """

    def __init__(self,
                 buffer_size_mb: int = 256,
                 max_cached_arrays: int = 100,
                 use_memory_mapping: bool = True):

        self.buffer_size_mb = buffer_size_mb
        self.max_cached_arrays = max_cached_arrays
        self.use_memory_mapping = use_memory_mapping

        # Cache storage
        self.cached_arrays: Dict[str, np.ndarray] = {}
        self.access_order: deque[str] = deque()
        self.cache_sizes: Dict[str, int] = {}

        # Memory mapping
        self.memory_mapped_files: Dict[str, mmap.mmap] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.total_memory_bytes = 0

        self.logger = logging.getLogger("CachedDataBuffer")

    def get_array(self, key: str, shape: tuple, dtype=np.float64) -> Optional[np.ndarray]:
        """Get a cached array or create a new one"""
        with self.lock:
            if key in self.cached_arrays:
                # Cache hit - move to front of LRU
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hit_count += 1
                return self.cached_arrays[key]

            # Cache miss - create new array
            self.miss_count += 1

            try:
                if self.use_memory_mapping:
                    array = self._create_memory_mapped_array(key, shape, dtype)
                else:
                    array = np.zeros(shape, dtype=dtype)

                # Add to cache
                self._add_to_cache(key, array)
                return array

            except Exception as e:
                self.logger.error(f"❌ Error creating array {key}: {e}")
                return None

    def store_array(self, key: str, array: np.ndarray):
        """Store an array in the cache"""
        with self.lock:
            self._add_to_cache(key, array.copy())

    def _create_memory_mapped_array(self, key: str, shape: tuple, dtype) -> np.ndarray:
        """Create a memory-mapped numpy array"""
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize

        # Create memory-mapped file
        mm = mmap.mmap(-1, size_bytes)
        self.memory_mapped_files[key] = mm

        # Create numpy array view
        array = np.frombuffer(mm, dtype=dtype).reshape(shape)
        return array

    def _add_to_cache(self, key: str, array: np.ndarray):
        """Add array to cache with LRU eviction"""
        array_size = array.nbytes

        # Check if we need to evict
        while (len(self.cached_arrays) >= self.max_cached_arrays or
               self.total_memory_bytes + array_size > self.buffer_size_mb * 1024 * 1024):

            if not self.access_order:
                break

            # Evict least recently used
            lru_key = self.access_order.popleft()
            self._evict_array(lru_key)

        # Add new array
        self.cached_arrays[key] = array
        self.cache_sizes[key] = array_size
        self.access_order.append(key)
        self.total_memory_bytes += array_size

    def _evict_array(self, key: str):
        """Evict an array from cache"""
        if key in self.cached_arrays:
            array_size = self.cache_sizes.get(key, 0)

            del self.cached_arrays[key]
            del self.cache_sizes[key]
            self.total_memory_bytes -= array_size

            # Clean up memory mapping if used
            if key in self.memory_mapped_files:
                self.memory_mapped_files[key].close()
                del self.memory_mapped_files[key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1)

        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "cached_arrays": len(self.cached_arrays),
            "total_memory_mb": self.total_memory_bytes / (1024 * 1024),
            "memory_mapped_arrays": len(self.memory_mapped_files)
        }

    def cleanup(self):
        """Clean up all cached data"""
        with self.lock:
            # Close memory-mapped files
            for mm in self.memory_mapped_files.values():
                mm.close()

            self.cached_arrays.clear()
            self.access_order.clear()
            self.cache_sizes.clear()
            self.memory_mapped_files.clear()
            self.total_memory_bytes = 0

            gc.collect()
            self.logger.info("🧹 Cache cleaned up")


class DataFramePool(MemoryPool[pd.DataFrame]):
    """Specialized memory pool for pandas DataFrames"""

    def __init__(self, max_objects: int = 100):
        def create_dataframe() -> pd.DataFrame:
            return pd.DataFrame()

        def reset_dataframe(df: pd.DataFrame):
            df.drop(df.index, inplace=True)
            df.drop(df.columns, inplace=True)

        super().__init__(
            PoolType.PANDAS_DATAFRAMES,
            create_dataframe,
            reset_dataframe,
            max_objects
        )


class NumpyArrayPool(MemoryPool[np.ndarray]):
    """Specialized memory pool for numpy arrays"""

    def __init__(self, shape: tuple, dtype=np.float64, max_objects: int = 100):
        self.array_shape = shape
        self.array_dtype = dtype

        def create_array() -> np.ndarray:
            return np.zeros(shape, dtype=dtype)

        def reset_array(arr: np.ndarray):
            arr.fill(0)

        super().__init__(
            PoolType.NUMPY_ARRAYS,
            create_array,
            reset_array,
            max_objects
        )


class MemoryPoolManager:
    """
    Central manager for all memory pools

    Features:
    - Pool lifecycle management
    - Global memory monitoring
    - Automatic optimization
    - Performance metrics
    """

    def __init__(self, config: 'ExecutionConfig'):
        self.config = config
        self.pools: Dict[str, MemoryPool] = {}
        self.data_buffers: Dict[str, CachedDataBuffer] = {}

        # Global statistics
        self.total_memory_allocated_mb = 0.0
        self.pool_creation_count = 0

        # Thread safety
        self.lock = threading.Lock()

        self.logger = logging.getLogger("MemoryPoolManager")

        # Initialize default pools
        self._initialize_default_pools()

    def _initialize_default_pools(self):
        """Initialize commonly used memory pools"""
        try:
            # DataFrame pool for market data
            self.create_dataframe_pool("market_data", max_objects=50)

            # Array pools for technical indicators
            self.create_numpy_array_pool("price_arrays", (1000,), np.float64, 20)
            self.create_numpy_array_pool("indicator_arrays", (1000,), np.float64, 50)

            # Data buffer for caching
            self.create_data_buffer("indicator_cache", buffer_size_mb=128)

            self.logger.info("🏊 Default memory pools initialized")

        except Exception as e:
            self.logger.error(f"❌ Error initializing default pools: {e}")

    def create_dataframe_pool(self, name: str, max_objects: int = 100) -> DataFramePool:
        """Create a pandas DataFrame pool"""
        with self.lock:
            pool = DataFramePool(max_objects)
            pool.preallocate()
            self.pools[name] = pool
            self.pool_creation_count += 1

            self.logger.info(f"📊 Created DataFrame pool '{name}' with {max_objects} max objects")
            return pool

    def create_numpy_array_pool(self,
                               name: str,
                               shape: tuple,
                               dtype=np.float64,
                               max_objects: int = 100) -> NumpyArrayPool:
        """Create a numpy array pool"""
        with self.lock:
            pool = NumpyArrayPool(shape, dtype, max_objects)
            pool.preallocate()
            self.pools[name] = pool
            self.pool_creation_count += 1

            self.logger.info(f"🔢 Created numpy array pool '{name}' with shape {shape}")
            return pool

    def create_data_buffer(self, name: str, buffer_size_mb: int = 256) -> CachedDataBuffer:
        """Create a cached data buffer"""
        with self.lock:
            buffer = CachedDataBuffer(buffer_size_mb)
            self.data_buffers[name] = buffer

            self.logger.info(f"💾 Created data buffer '{name}' with {buffer_size_mb}MB capacity")
            return buffer

    def get_pool(self, name: str = "default") -> Optional[MemoryPool]:
        """Get a memory pool by name"""
        return self.pools.get(name)

    def get_data_buffer(self, name: str = "default") -> Optional[CachedDataBuffer]:
        """Get a data buffer by name"""
        return self.data_buffers.get(name)

    def initialize_pools(self):
        """Pre-initialize all pools for optimal performance"""
        with self.lock:
            for pool in self.pools.values():
                if hasattr(pool, 'preallocate'):
                    pool.preallocate()

            self.logger.info(f"⚡ Initialized {len(self.pools)} memory pools")

    def cleanup(self):
        """Clean up all pools and free memory"""
        with self.lock:
            for pool in self.pools.values():
                pool.cleanup()

            for buffer in self.data_buffers.values():
                buffer.cleanup()

            self.pools.clear()
            self.data_buffers.clear()

            # Force garbage collection
            gc.collect()

            self.logger.info("🧹 All memory pools cleaned up")

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global memory pool statistics"""
        with self.lock:
            total_allocated = 0
            total_available = 0
            total_hit_rate = 0
            pool_stats = {}

            for name, pool in self.pools.items():
                stats = pool.get_stats()
                pool_stats[name] = {
                    "type": stats.pool_type.value,
                    "allocated": stats.allocated_objects,
                    "available": stats.available_objects,
                    "hit_rate": stats.hit_rate,
                    "efficiency": stats.efficiency_percent
                }

                total_allocated += stats.allocated_objects
                total_available += stats.available_objects
                total_hit_rate += stats.hit_rate

            avg_hit_rate = total_hit_rate / max(len(self.pools), 1)

            # Data buffer stats
            buffer_stats = {}
            for name, buffer in self.data_buffers.items():
                buffer_stats[name] = buffer.get_cache_stats()

            return {
                "pools": pool_stats,
                "data_buffers": buffer_stats,
                "summary": {
                    "total_pools": len(self.pools),
                    "total_data_buffers": len(self.data_buffers),
                    "total_allocated_objects": total_allocated,
                    "total_available_objects": total_available,
                    "average_hit_rate": avg_hit_rate,
                    "pool_creation_count": self.pool_creation_count
                }
            }

    def optimize_pools(self):
        """Optimize pool performance based on usage patterns"""
        with self.lock:
            for name, pool in self.pools.items():
                stats = pool.get_stats()

                # Auto-resize pools based on usage
                if stats.hit_rate < 0.5 and stats.allocated_objects < pool.max_objects // 2:
                    # Poor hit rate and low usage - consider shrinking
                    pool._auto_garbage_collect()

                elif stats.hit_rate > 0.9 and stats.available_objects < 10:
                    # High hit rate but low availability - consider growing
                    pool.preallocate()

            self.logger.info("⚡ Memory pools optimized based on usage patterns")


# Factory functions for easy pool creation

def create_market_data_pools(config: 'ExecutionConfig') -> MemoryPoolManager:
    """Create memory pools optimized for market data processing"""
    manager = MemoryPoolManager(config)

    # Market data specific pools
    manager.create_dataframe_pool("ohlcv_data", max_objects=100)
    manager.create_dataframe_pool("tick_data", max_objects=50)

    # Technical indicator arrays
    manager.create_numpy_array_pool("sma_arrays", (500,), np.float64, 30)
    manager.create_numpy_array_pool("ema_arrays", (500,), np.float64, 30)
    manager.create_numpy_array_pool("rsi_arrays", (500,), np.float64, 20)
    manager.create_numpy_array_pool("macd_arrays", (500,), np.float64, 20)

    # Large data buffer for caching computed indicators
    manager.create_data_buffer("technical_indicators", buffer_size_mb=512)

    return manager


def create_backtest_pools(config: 'ExecutionConfig') -> MemoryPoolManager:
    """Create memory pools optimized for backtest execution"""
    manager = MemoryPoolManager(config)

    # Backtest specific pools
    manager.create_dataframe_pool("backtest_results", max_objects=200)
    manager.create_dataframe_pool("signal_history", max_objects=100)

    # Large arrays for historical data
    manager.create_numpy_array_pool("historical_prices", (10000,), np.float64, 20)
    manager.create_numpy_array_pool("signal_arrays", (10000,), np.bool_, 30)

    # Massive cache for pre-computed data
    manager.create_data_buffer("historical_cache", buffer_size_mb=1024)

    return manager