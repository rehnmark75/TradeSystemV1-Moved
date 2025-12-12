"""
Centralized database connection management for Streamlit.
Provides connection pooling and singleton engines via @st.cache_resource.

This module eliminates the performance issue of creating new database connections
on every page load. Instead, connections are pooled and reused across the session.
"""

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from psycopg2 import pool as psycopg2_pool
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _get_connection_string(db_type: str = "trading") -> str:
    """
    Get database connection string from secrets or environment.

    Args:
        db_type: One of "trading", "config", or "stocks"

    Returns:
        PostgreSQL connection string
    """
    # Try Streamlit secrets first
    if hasattr(st, 'secrets') and hasattr(st.secrets, 'database'):
        try:
            if db_type == "trading":
                return st.secrets.database.trading_connection_string
            elif db_type == "config":
                return st.secrets.database.config_connection_string
            elif db_type == "stocks":
                return st.secrets.database.stocks_connection_string
        except (AttributeError, KeyError):
            pass

        # Fallback to generic connection string
        try:
            return st.secrets.database.connection_string
        except (AttributeError, KeyError):
            pass

    # Fallback to environment variables
    if db_type == "trading":
        return os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
    elif db_type == "config":
        return os.getenv("CONFIG_DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex_config")
    elif db_type == "stocks":
        return os.getenv("STOCKS_DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/stocks")

    return os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")


@st.cache_resource
def get_sqlalchemy_engine(db_type: str = "trading"):
    """
    Get a cached SQLAlchemy engine with connection pooling.

    This is a singleton per database type, created once per Streamlit session.
    Uses QueuePool with sensible defaults for web applications.

    Args:
        db_type: One of "trading", "config", or "stocks"

    Returns:
        SQLAlchemy Engine with connection pooling
    """
    conn_str = _get_connection_string(db_type)

    engine = create_engine(
        conn_str,
        poolclass=QueuePool,
        pool_size=5,           # Maintain 5 connections
        max_overflow=10,       # Allow up to 10 additional connections under load
        pool_pre_ping=True,    # Verify connections before use
        pool_recycle=3600,     # Recycle connections after 1 hour
        pool_timeout=30,       # Wait up to 30s for a connection
        echo=False             # Set to True for SQL logging during debug
    )

    logger.info(f"Created SQLAlchemy engine pool for {db_type} database")
    return engine


@st.cache_resource
def get_psycopg2_pool(db_type: str = "trading", minconn: int = 2, maxconn: int = 10):
    """
    Get a cached psycopg2 connection pool.

    For code that uses raw psycopg2 instead of SQLAlchemy.

    Args:
        db_type: One of "trading", "config", or "stocks"
        minconn: Minimum connections to maintain
        maxconn: Maximum connections allowed

    Returns:
        ThreadedConnectionPool instance
    """
    conn_str = _get_connection_string(db_type)

    connection_pool = psycopg2_pool.ThreadedConnectionPool(
        minconn=minconn,
        maxconn=maxconn,
        dsn=conn_str
    )

    logger.info(f"Created psycopg2 connection pool for {db_type} database (min={minconn}, max={maxconn})")
    return connection_pool


class PooledConnection:
    """
    Wrapper around psycopg2 connection that returns to pool on close().

    This allows existing code that calls conn.close() to work correctly
    with connection pooling - the connection is returned to the pool
    instead of being destroyed.
    """

    def __init__(self, conn, pool, db_type):
        self._conn = conn
        self._pool = pool
        self._db_type = db_type
        self._closed = False

    def __getattr__(self, name):
        """Proxy all attributes to the underlying connection."""
        return getattr(self._conn, name)

    def close(self):
        """Return connection to pool instead of closing it."""
        if not self._closed:
            self._pool.putconn(self._conn)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._conn.rollback()
        self.close()
        return False


def get_psycopg2_connection(db_type: str = "trading"):
    """
    Get a connection from the psycopg2 pool.

    The returned connection is wrapped so that calling close() will
    return it to the pool instead of destroying it.

    Args:
        db_type: One of "trading", "config", or "stocks"

    Returns:
        PooledConnection wrapping a psycopg2 connection
    """
    pool = get_psycopg2_pool(db_type)
    conn = pool.getconn()
    return PooledConnection(conn, pool, db_type)


def return_psycopg2_connection(conn, db_type: str = "trading"):
    """
    Return a connection to the psycopg2 pool.

    Note: If using get_psycopg2_connection(), just call conn.close() instead
    as the PooledConnection wrapper handles returning to the pool automatically.

    Args:
        conn: The connection to return (can be PooledConnection or raw connection)
        db_type: The database type the connection belongs to
    """
    if isinstance(conn, PooledConnection):
        conn.close()  # PooledConnection.close() returns to pool
    else:
        pool = get_psycopg2_pool(db_type)
        pool.putconn(conn)


class DatabaseContextManager:
    """
    Context manager for psycopg2 connections with automatic return to pool.

    Usage:
        with DatabaseContextManager() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT ...")
                results = cursor.fetchall()

    The connection is automatically returned to the pool when exiting the context.
    If an exception occurs, the transaction is rolled back.
    """

    def __init__(self, db_type: str = "trading"):
        """
        Initialize context manager.

        Args:
            db_type: One of "trading", "config", or "stocks"
        """
        self.db_type = db_type
        self.conn = None

    def __enter__(self):
        """Acquire connection from pool."""
        self.conn = get_psycopg2_connection(self.db_type)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Return connection to pool, rolling back on exception."""
        if self.conn:
            if exc_type is not None:
                self.conn.rollback()
            return_psycopg2_connection(self.conn, self.db_type)
        return False  # Don't suppress exceptions


def get_connection_string(db_type: str = "trading") -> str:
    """
    Public accessor for connection string.

    Useful for cases where you need the raw string but prefer
    to use this module's configuration logic.

    Args:
        db_type: One of "trading", "config", or "stocks"

    Returns:
        PostgreSQL connection string
    """
    return _get_connection_string(db_type)
