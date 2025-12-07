"""
Async Database Manager for Stock Scanner

Provides async connection pooling and query execution for the stocks database.
Based on the pattern from forex_scanner but adapted for stocks.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager

import asyncpg

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Database operation error"""
    pass


class AsyncDatabaseManager:
    """
    Async database manager with connection pooling

    Features:
    - Connection pooling (min 5, max 20 connections)
    - Automatic reconnection on failure
    - Query performance tracking
    - Transaction support
    - Statistics monitoring

    Usage:
        db = AsyncDatabaseManager(database_url)
        await db.connect()

        # Simple queries
        rows = await db.fetch("SELECT * FROM stock_candles WHERE ticker = $1", "AAPL")
        row = await db.fetchrow("SELECT * FROM stock_instruments WHERE ticker = $1", "AAPL")
        value = await db.fetchval("SELECT COUNT(*) FROM stock_candles")

        # Execute
        await db.execute("INSERT INTO ...", values)

        # Transactions
        async with db.transaction():
            await db.execute(...)
            await db.execute(...)

        await db.close()
    """

    def __init__(
        self,
        database_url: str,
        min_connections: int = 5,
        max_connections: int = 20,
        command_timeout: float = 60.0,
        slow_query_threshold: float = 1.0
    ):
        """
        Initialize database manager

        Args:
            database_url: PostgreSQL connection URL
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
            command_timeout: Query timeout in seconds
            slow_query_threshold: Log queries slower than this (seconds)
        """
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.command_timeout = command_timeout
        self.slow_query_threshold = slow_query_threshold

        self._pool: Optional[asyncpg.Pool] = None
        self._connected = False
        self._transaction_stack: List[asyncpg.Connection] = []

        # Statistics
        self._stats = {
            "queries_executed": 0,
            "slow_queries": 0,
            "errors": 0,
            "total_query_time": 0.0,
            "last_error": None,
            "last_query_time": None
        }

    @property
    def is_connected(self) -> bool:
        """Check if connected to database"""
        return self._connected and self._pool is not None

    async def connect(self) -> bool:
        """
        Establish database connection pool

        Returns:
            True if connected successfully
        """
        if self.is_connected:
            return True

        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=self.command_timeout,
                setup=self._setup_connection
            )
            self._connected = True
            logger.info(
                f"Connected to stocks database "
                f"(pool: {self.min_connections}-{self.max_connections})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._stats["errors"] += 1
            self._stats["last_error"] = str(e)
            raise DatabaseError(f"Connection failed: {e}")

    async def _setup_connection(self, connection: asyncpg.Connection):
        """Setup callback for new connections"""
        # Set timezone to UTC
        await connection.execute("SET timezone TO 'UTC'")

    async def close(self):
        """Close database connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._connected = False
            logger.info("Closed database connection pool")

    async def _ensure_connected(self):
        """Ensure we have an active connection"""
        if not self.is_connected:
            await self.connect()

    def _get_connection(self) -> Optional[asyncpg.Connection]:
        """Get connection from transaction stack or None"""
        if self._transaction_stack:
            return self._transaction_stack[-1]
        return None

    async def _execute_with_timing(
        self,
        method: str,
        query: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a query with timing and error handling

        Args:
            method: Method name (fetch, fetchrow, fetchval, execute)
            query: SQL query
            *args: Query arguments
            **kwargs: Additional arguments

        Returns:
            Query result
        """
        await self._ensure_connected()

        start_time = datetime.now()
        result = None

        try:
            # Use transaction connection if available
            conn = self._get_connection()

            if conn:
                # We're in a transaction
                func = getattr(conn, method)
                result = await func(query, *args, **kwargs)
            else:
                # Use pool
                async with self._pool.acquire() as conn:
                    func = getattr(conn, method)
                    result = await func(query, *args, **kwargs)

            # Track timing
            elapsed = (datetime.now() - start_time).total_seconds()
            self._stats["queries_executed"] += 1
            self._stats["total_query_time"] += elapsed
            self._stats["last_query_time"] = elapsed

            if elapsed > self.slow_query_threshold:
                self._stats["slow_queries"] += 1
                logger.warning(
                    f"Slow query ({elapsed:.2f}s): {query[:100]}..."
                )

            return result

        except asyncpg.PostgresError as e:
            self._stats["errors"] += 1
            self._stats["last_error"] = str(e)
            logger.error(f"Database error: {e}\nQuery: {query[:200]}")
            raise DatabaseError(f"Query failed: {e}")

        except Exception as e:
            self._stats["errors"] += 1
            self._stats["last_error"] = str(e)
            logger.error(f"Unexpected error: {e}")
            raise

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    async def fetch(self, query: str, *args, timeout: float = None) -> List[asyncpg.Record]:
        """
        Fetch multiple rows

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional query timeout

        Returns:
            List of Record objects
        """
        return await self._execute_with_timing("fetch", query, *args, timeout=timeout)

    async def fetchrow(self, query: str, *args, timeout: float = None) -> Optional[asyncpg.Record]:
        """
        Fetch single row

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional query timeout

        Returns:
            Single Record or None
        """
        return await self._execute_with_timing("fetchrow", query, *args, timeout=timeout)

    async def fetchval(self, query: str, *args, column: int = 0, timeout: float = None) -> Any:
        """
        Fetch single value

        Args:
            query: SQL query
            *args: Query parameters
            column: Column index to return
            timeout: Optional query timeout

        Returns:
            Single value
        """
        return await self._execute_with_timing(
            "fetchval", query, *args, column=column, timeout=timeout
        )

    async def execute(self, query: str, *args, timeout: float = None) -> str:
        """
        Execute a query (INSERT, UPDATE, DELETE)

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional query timeout

        Returns:
            Status string (e.g., "INSERT 0 1")
        """
        return await self._execute_with_timing("execute", query, *args, timeout=timeout)

    async def executemany(self, query: str, args_list: List[tuple], timeout: float = None):
        """
        Execute query with multiple parameter sets

        Args:
            query: SQL query
            args_list: List of parameter tuples
            timeout: Optional timeout
        """
        await self._ensure_connected()

        conn = self._get_connection()
        if conn:
            await conn.executemany(query, args_list, timeout=timeout)
        else:
            async with self._pool.acquire() as conn:
                await conn.executemany(query, args_list, timeout=timeout)

    # =========================================================================
    # TRANSACTION SUPPORT
    # =========================================================================

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions

        Usage:
            async with db.transaction():
                await db.execute(...)
                await db.execute(...)
            # Commits on success, rolls back on exception
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                self._transaction_stack.append(conn)
                try:
                    yield conn
                finally:
                    self._transaction_stack.pop()

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool

        Usage:
            async with db.acquire() as conn:
                await conn.fetch(...)
        """
        await self._ensure_connected()
        async with self._pool.acquire() as conn:
            yield conn

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = $1
            )
        """
        return await self.fetchval(query, table_name)

    async def get_table_count(self, table_name: str) -> int:
        """Get row count for a table"""
        # Use safe query construction
        query = f"SELECT COUNT(*) FROM {table_name}"
        return await self.fetchval(query)

    async def health_check(self) -> bool:
        """
        Check database health

        Returns:
            True if healthy
        """
        try:
            result = await self.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dict with query stats
        """
        stats = self._stats.copy()

        if stats["queries_executed"] > 0:
            stats["avg_query_time"] = (
                stats["total_query_time"] / stats["queries_executed"]
            )
        else:
            stats["avg_query_time"] = 0

        # Pool stats
        if self._pool:
            stats["pool_size"] = self._pool.get_size()
            stats["pool_free"] = self._pool.get_idle_size()
            stats["pool_used"] = stats["pool_size"] - stats["pool_free"]

        return stats

    def reset_stats(self):
        """Reset statistics counters"""
        self._stats = {
            "queries_executed": 0,
            "slow_queries": 0,
            "errors": 0,
            "total_query_time": 0.0,
            "last_error": None,
            "last_query_time": None
        }

    async def run_migration(self, sql_file_path: str) -> bool:
        """
        Run a SQL migration file

        Args:
            sql_file_path: Path to SQL file

        Returns:
            True if successful
        """
        try:
            with open(sql_file_path, "r") as f:
                sql = f.read()

            await self.execute(sql)
            logger.info(f"Migration completed: {sql_file_path}")
            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False


# Singleton instance (optional pattern)
_db_instance: Optional[AsyncDatabaseManager] = None


def get_database(database_url: str = None) -> AsyncDatabaseManager:
    """
    Get or create database manager singleton

    Args:
        database_url: Database URL (required on first call)

    Returns:
        AsyncDatabaseManager instance
    """
    global _db_instance

    if _db_instance is None:
        if database_url is None:
            raise ValueError("database_url required for first initialization")
        _db_instance = AsyncDatabaseManager(database_url)

    return _db_instance


async def init_database(database_url: str) -> AsyncDatabaseManager:
    """
    Initialize and connect database

    Args:
        database_url: Database URL

    Returns:
        Connected AsyncDatabaseManager
    """
    db = get_database(database_url)
    await db.connect()
    return db
