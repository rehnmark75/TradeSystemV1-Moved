# core/database/async_database_manager.py
"""
AsyncDatabaseManager - Enhanced database manager with async capabilities
Provides connection pooling, async operations, and optimized query execution
"""

import asyncio
import asyncpg
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import json
from urllib.parse import urlparse
from dataclasses import dataclass
import pandas as pd

# Connection pool statistics
@dataclass
class ConnectionPoolStats:
    """Statistics for connection pool monitoring"""
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_tasks: int
    total_queries_executed: int
    average_query_time_ms: float
    peak_connections: int
    pool_created_at: datetime
    last_reset_at: Optional[datetime] = None


class AsyncDatabaseManager:
    """
    Enhanced async database manager optimized for high-frequency trading data
    Features: Connection pooling, async operations, query optimization, monitoring
    """

    def __init__(self, database_url: str, **pool_kwargs):
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)

        # Connection pool configuration
        self.pool_config = {
            'min_size': pool_kwargs.get('min_connections', 5),
            'max_size': pool_kwargs.get('max_connections', 20),
            'max_queries': pool_kwargs.get('max_queries_per_connection', 50000),
            'max_inactive_connection_lifetime': pool_kwargs.get('max_inactive_lifetime', 3600.0),
            'command_timeout': pool_kwargs.get('command_timeout', 60.0),
            'server_settings': {
                'application_name': 'TradeSystemV1_AsyncDB',
                'jit': 'off',  # Disable JIT for consistent query performance
                'shared_preload_libraries': 'pg_stat_statements'
            }
        }

        # Connection pool and stats
        self._pool = None
        self._pool_stats = None
        self._is_initialized = False

        # Query performance tracking
        self._query_stats = {
            'total_queries': 0,
            'total_query_time': 0.0,
            'slow_queries': [],
            'query_cache_hits': 0,
            'query_cache_misses': 0
        }

        # Transaction support
        self._transaction_stack = {}  # task_id -> connection

    async def initialize(self):
        """Initialize the async database connection pool"""
        if self._is_initialized:
            return

        try:
            self.logger.info("🔌 Initializing async database connection pool...")

            # Parse database URL for asyncpg
            parsed_url = urlparse(self.database_url)

            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=parsed_url.hostname,
                port=parsed_url.port or 5432,
                database=parsed_url.path[1:],  # Remove leading '/'
                user=parsed_url.username,
                password=parsed_url.password,
                **self.pool_config
            )

            # Initialize pool statistics
            self._pool_stats = ConnectionPoolStats(
                total_connections=self.pool_config['max_size'],
                active_connections=0,
                idle_connections=self.pool_config['min_size'],
                waiting_tasks=0,
                total_queries_executed=0,
                average_query_time_ms=0.0,
                peak_connections=self.pool_config['min_size'],
                pool_created_at=datetime.now(timezone.utc)
            )

            # Test connection and setup
            await self._setup_database_optimizations()

            self._is_initialized = True
            self.logger.info(f"✅ Async database pool initialized: "
                           f"min={self.pool_config['min_size']}, "
                           f"max={self.pool_config['max_size']}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database pool: {e}")
            raise

    async def _setup_database_optimizations(self):
        """Setup database-level optimizations"""
        try:
            async with self._pool.acquire() as conn:
                # Enable query plan caching
                await conn.execute("SET plan_cache_mode = 'force_generic_plan'")

                # Optimize for OLAP workloads (backtesting queries)
                await conn.execute("SET work_mem = '256MB'")
                await conn.execute("SET effective_cache_size = '4GB'")
                await conn.execute("SET random_page_cost = 1.1")
                await conn.execute("SET seq_page_cost = 1.0")

                # Enable parallel query execution for large datasets
                await conn.execute("SET max_parallel_workers_per_gather = 4")
                await conn.execute("SET max_parallel_workers = 8")
                await conn.execute("SET parallel_tuple_cost = 0.1")

                # Financial data specific optimizations
                await conn.execute("SET timezone = 'UTC'")
                await conn.execute("SET datestyle = 'ISO'")

                self.logger.info("✅ Database optimizations applied")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not apply all database optimizations: {e}")

    async def close(self):
        """Close the connection pool"""
        if self._pool:
            await self._pool.close()
            self._is_initialized = False
            self.logger.info("🔌 Database connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool"""
        if not self._is_initialized:
            await self.initialize()

        async with self._pool.acquire() as conn:
            try:
                # Update pool stats
                self._update_pool_stats('acquire')
                yield conn
            finally:
                self._update_pool_stats('release')

    async def execute_query(self, query: str, params: Optional[Tuple] = None,
                          fetch_mode: str = 'all') -> Union[List[Dict], Dict, None]:
        """
        Execute a query with performance tracking

        Args:
            query: SQL query string
            params: Query parameters
            fetch_mode: 'all', 'one', 'none' for different return types
        """
        start_time = time.time()

        try:
            async with self.get_connection() as conn:
                # Execute query
                if fetch_mode == 'all':
                    rows = await conn.fetch(query, *(params or ()))
                    result = [dict(row) for row in rows]
                elif fetch_mode == 'one':
                    row = await conn.fetchrow(query, *(params or ()))
                    result = dict(row) if row else None
                elif fetch_mode == 'none':
                    await conn.execute(query, *(params or ()))
                    result = None
                else:
                    raise ValueError(f"Invalid fetch_mode: {fetch_mode}")

                # Track query performance
                execution_time = (time.time() - start_time) * 1000  # ms
                self._track_query_performance(query, execution_time, params)

                return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"❌ Query failed ({execution_time:.2f}ms): {str(e)[:100]}")
            self.logger.error(f"Query: {query[:200]}...")
            raise

    async def execute_batch(self, query: str, params_list: List[Tuple],
                          batch_size: int = 1000) -> int:
        """
        Execute a batch of queries efficiently

        Args:
            query: SQL query with placeholders
            params_list: List of parameter tuples
            batch_size: Number of records per batch

        Returns:
            Number of records processed
        """
        if not params_list:
            return 0

        start_time = time.time()
        total_processed = 0

        try:
            async with self.get_connection() as conn:
                # Process in batches to avoid memory issues
                for i in range(0, len(params_list), batch_size):
                    batch_params = params_list[i:i + batch_size]

                    # Use executemany for efficient batch processing
                    await conn.executemany(query, batch_params)
                    total_processed += len(batch_params)

                    # Log progress for large batches
                    if len(params_list) > 10000 and i > 0 and i % 10000 == 0:
                        elapsed = time.time() - start_time
                        rate = total_processed / elapsed
                        self.logger.info(f"📊 Batch progress: {total_processed:,}/{len(params_list):,} "
                                       f"({rate:.0f} records/sec)")

                execution_time = (time.time() - start_time) * 1000
                self._track_query_performance(f"BATCH: {query}", execution_time,
                                            f"{len(params_list)} records")

                self.logger.info(f"✅ Batch completed: {total_processed:,} records "
                               f"in {execution_time:.1f}ms "
                               f"({total_processed / max(execution_time/1000, 0.001):.0f} records/sec)")

                return total_processed

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"❌ Batch execution failed ({execution_time:.2f}ms): {e}")
            raise

    async def fetch_dataframe(self, query: str, params: Optional[Tuple] = None,
                            index_col: Optional[str] = None) -> pd.DataFrame:
        """
        Execute query and return results as pandas DataFrame
        Optimized for time-series financial data
        """
        start_time = time.time()

        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(query, *(params or ()))

                if not rows:
                    return pd.DataFrame()

                # Convert to DataFrame
                data = [dict(row) for row in rows]
                df = pd.DataFrame(data)

                # Set index if specified
                if index_col and index_col in df.columns:
                    df = df.set_index(index_col)

                    # Auto-detect and convert datetime indexes to UTC
                    if df.index.dtype.name == 'object':
                        try:
                            df.index = pd.to_datetime(df.index, utc=True)
                        except:
                            pass

                execution_time = (time.time() - start_time) * 1000
                self._track_query_performance(f"DATAFRAME: {query}", execution_time, params)

                self.logger.debug(f"📊 DataFrame query: {len(df)} rows in {execution_time:.1f}ms")

                return df

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"❌ DataFrame query failed ({execution_time:.2f}ms): {e}")
            raise

    @asynccontextmanager
    async def transaction(self):
        """
        Async transaction context manager
        Supports nested transactions using savepoints
        """
        if not self._is_initialized:
            await self.initialize()

        task_id = id(asyncio.current_task())

        # Check if we're already in a transaction
        if task_id in self._transaction_stack:
            # Nested transaction - use savepoint
            conn = self._transaction_stack[task_id]
            savepoint_name = f"sp_{len(self._transaction_stack)}"

            try:
                await conn.execute(f"SAVEPOINT {savepoint_name}")
                yield conn
                await conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            except Exception:
                await conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                raise
        else:
            # New transaction
            async with self._pool.acquire() as conn:
                self._transaction_stack[task_id] = conn

                try:
                    async with conn.transaction():
                        yield conn
                finally:
                    del self._transaction_stack[task_id]

    def _track_query_performance(self, query: str, execution_time_ms: float, params: Any):
        """Track query performance metrics"""
        self._query_stats['total_queries'] += 1
        self._query_stats['total_query_time'] += execution_time_ms

        # Track slow queries (> 1 second)
        if execution_time_ms > 1000:
            slow_query = {
                'query': query[:500],  # Truncate for storage
                'execution_time_ms': execution_time_ms,
                'params': str(params)[:200] if params else None,
                'timestamp': datetime.now(timezone.utc)
            }

            self._query_stats['slow_queries'].append(slow_query)

            # Keep only last 100 slow queries
            if len(self._query_stats['slow_queries']) > 100:
                self._query_stats['slow_queries'] = self._query_stats['slow_queries'][-100:]

            self.logger.warning(f"🐌 Slow query detected: {execution_time_ms:.1f}ms")

    def _update_pool_stats(self, action: str):
        """Update connection pool statistics"""
        if not self._pool_stats:
            return

        if action == 'acquire':
            self._pool_stats.active_connections += 1
            self._pool_stats.idle_connections = max(0, self._pool_stats.idle_connections - 1)
            self._pool_stats.peak_connections = max(
                self._pool_stats.peak_connections,
                self._pool_stats.active_connections
            )
        elif action == 'release':
            self._pool_stats.active_connections = max(0, self._pool_stats.active_connections - 1)
            self._pool_stats.idle_connections += 1

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if not self._pool_stats:
            return {}

        avg_query_time = (
            self._query_stats['total_query_time'] /
            max(self._query_stats['total_queries'], 1)
        )

        return {
            'pool_size': {
                'min_size': self.pool_config['min_size'],
                'max_size': self.pool_config['max_size'],
                'active_connections': self._pool_stats.active_connections,
                'idle_connections': self._pool_stats.idle_connections,
                'peak_connections': self._pool_stats.peak_connections
            },
            'query_performance': {
                'total_queries': self._query_stats['total_queries'],
                'average_query_time_ms': avg_query_time,
                'slow_queries_count': len(self._query_stats['slow_queries']),
                'cache_hit_rate': (
                    self._query_stats['query_cache_hits'] /
                    max(self._query_stats['total_queries'], 1)
                ) * 100
            },
            'pool_created_at': self._pool_stats.pool_created_at,
            'uptime_seconds': (
                datetime.now(timezone.utc) - self._pool_stats.pool_created_at
            ).total_seconds()
        }

    async def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """Analyze query execution plan and performance"""
        try:
            async with self.get_connection() as conn:
                # Get query execution plan
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                result = await conn.fetchval(explain_query)

                return {
                    'query': query[:500],
                    'execution_plan': result[0] if result else None,
                    'analysis_timestamp': datetime.now(timezone.utc)
                }

        except Exception as e:
            self.logger.error(f"Error analyzing query performance: {e}")
            return {'error': str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        start_time = time.time()

        try:
            async with self.get_connection() as conn:
                # Test basic connectivity
                await conn.execute("SELECT 1")

                # Check database size and activity
                db_stats = await conn.fetchrow("""
                    SELECT
                        pg_database_size(current_database()) as db_size_bytes,
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                        (SELECT count(*) FROM pg_stat_activity) as total_connections
                """)

                response_time_ms = (time.time() - start_time) * 1000

                return {
                    'status': 'healthy',
                    'response_time_ms': response_time_ms,
                    'database_size_mb': db_stats['db_size_bytes'] / 1024 / 1024,
                    'active_db_connections': db_stats['active_connections'],
                    'total_db_connections': db_stats['total_connections'],
                    'pool_stats': self.get_pool_stats(),
                    'timestamp': datetime.now(timezone.utc)
                }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc)
            }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


class DatabaseConnectionManager:
    """Singleton manager for async database connections"""

    _instance = None
    _db_manager = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_manager(cls, database_url: str = None, **kwargs) -> AsyncDatabaseManager:
        """Get or create the async database manager"""
        if cls._db_manager is None:
            if database_url is None:
                raise ValueError("database_url required for first initialization")

            cls._db_manager = AsyncDatabaseManager(database_url, **kwargs)
            await cls._db_manager.initialize()

        return cls._db_manager

    @classmethod
    async def close_manager(cls):
        """Close the database manager"""
        if cls._db_manager:
            await cls._db_manager.close()
            cls._db_manager = None


# Factory functions for easy usage
async def create_async_db_manager(database_url: str, **kwargs) -> AsyncDatabaseManager:
    """Create and initialize AsyncDatabaseManager"""
    manager = AsyncDatabaseManager(database_url, **kwargs)
    await manager.initialize()
    return manager


def get_optimal_pool_config(workload_type: str = 'mixed') -> Dict[str, Any]:
    """Get optimal connection pool configuration for different workloads"""

    configs = {
        'backtest_heavy': {
            'min_connections': 8,
            'max_connections': 32,
            'max_queries_per_connection': 10000,
            'max_inactive_lifetime': 1800.0,
            'command_timeout': 300.0
        },
        'realtime_trading': {
            'min_connections': 3,
            'max_connections': 10,
            'max_queries_per_connection': 50000,
            'max_inactive_lifetime': 3600.0,
            'command_timeout': 30.0
        },
        'mixed': {
            'min_connections': 5,
            'max_connections': 20,
            'max_queries_per_connection': 25000,
            'max_inactive_lifetime': 3600.0,
            'command_timeout': 60.0
        }
    }

    return configs.get(workload_type, configs['mixed'])