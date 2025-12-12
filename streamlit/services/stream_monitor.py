"""
Stream Monitor Service - Database and API monitoring utilities
Provides functions to monitor IG streaming health through database queries and API calls
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
import logging

# Import centralized database utilities
from services.db_utils import get_sqlalchemy_engine

logger = logging.getLogger(__name__)

class StreamMonitor:
    """Monitor streaming services and database health"""

    def __init__(self):
        self.stream_api_base = "http://fastapi-stream:8000"  # Internal Docker port
        self.main_api_base = "http://fastapi-dev:8000"       # Internal Docker port
        self.db_engine = None
        self._init_database()

    def _init_database(self):
        """Initialize database connection using shared connection pool."""
        try:
            # Use centralized cached engine with connection pooling
            self.db_engine = get_sqlalchemy_engine("trading")

            # Test the connection and log which database we're connected to
            with self.db_engine.connect() as test_conn:
                db_name = test_conn.execute(text("SELECT current_database()")).fetchone()[0]
                logger.info(f"StreamMonitor connected to database: {db_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def safe_api_call(self, url: str, timeout: int = 5) -> Dict[str, Any]:
        """Safely make API call with error handling"""
        try:
            logger.debug(f"Making API call to: {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            # Try with localhost as fallback
            if "fastapi-stream:8000" in url:
                fallback_url = url.replace("fastapi-stream:8000", "localhost:8000")
                try:
                    logger.debug(f"Trying fallback URL: {fallback_url}")
                    response = requests.get(fallback_url, timeout=timeout)
                    response.raise_for_status()
                    return response.json()
                except Exception as fallback_e:
                    logger.error(f"Fallback also failed: {fallback_e}")
            return {"error": "Service unavailable", "status": "offline"}
        except requests.exceptions.Timeout:
            return {"error": "Request timeout", "status": "timeout"}
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP {e.response.status_code}", "status": "error"}
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            return {"error": str(e), "status": "unknown"}
    
    def get_backfill_status(self) -> Dict[str, Any]:
        """Get backfill service status"""
        return self.safe_api_call(f"{self.stream_api_base}/backfill/status")
    
    def get_backfill_gaps(self) -> Dict[str, Any]:
        """Get current data gaps"""
        return self.safe_api_call(f"{self.stream_api_base}/backfill/gaps")
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Get streaming service status"""
        return self.safe_api_call(f"{self.stream_api_base}/stream/status")
    
    def get_candle_health_from_api(self, epic: str, timeframe: int = 5) -> Dict[str, Any]:
        """Get recent candle data from API to check health"""
        url = f"{self.stream_api_base}/stream/candles/{epic}?timeframe={timeframe}"
        return self.safe_api_call(url)
    
    def get_latest_candle_from_api(self, epic: str, timeframe: int = 5) -> Dict[str, Any]:
        """Get the latest candle for an epic from API"""
        url = f"{self.stream_api_base}/stream/candle/latest/{epic}?timeframe={timeframe}"
        return self.safe_api_call(url)
    
    def get_candle_health_from_db(self, epic: str, timeframe: int = 5, hours_back: int = 2) -> Dict[str, Any]:
        """Get candle health directly from database"""
        if not self.db_engine:
            return {"error": "Database not available"}
        
        try:
            query = text("""
                SELECT 
                    start_time,
                    epic,
                    timeframe,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    created_at
                FROM ig_candles 
                WHERE epic = :epic 
                    AND timeframe = :timeframe 
                    AND start_time >= :start_time
                ORDER BY start_time DESC
                LIMIT 50
            """)
            
            start_time = datetime.now() - timedelta(hours=hours_back)
            current_time = datetime.now()
            logger.info(f"Querying candles for {epic}, timeframe={timeframe}")
            logger.info(f"Current time: {current_time}")  
            logger.info(f"Looking for data since: {start_time}")
            logger.info(f"Hours back: {hours_back}")
            
            with self.db_engine.connect() as conn:
                # Debug: Check which database we're connected to
                try:
                    db_check = conn.execute(text("SELECT current_database()"))
                    current_db = db_check.fetchone()[0]
                    logger.info(f"StreamMonitor connected to database: {current_db}")
                    
                    # List available tables
                    tables_query = text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                    tables_result = conn.execute(tables_query)
                    tables = [row[0] for row in tables_result]
                    logger.info(f"Available tables: {tables}")
                except Exception as db_debug_e:
                    logger.error(f"Database debug failed: {db_debug_e}")
                
                # Debug: Test a simple count query first
                try:
                    count_query = text("SELECT COUNT(*) FROM ig_candles WHERE epic = :epic AND timeframe = :timeframe")
                    count_result = conn.execute(count_query, {"epic": epic, "timeframe": timeframe})
                    total_count = count_result.fetchone()[0]
                    logger.info(f"Total candles for {epic} timeframe {timeframe}: {total_count}")
                    
                    # Test with time filter
                    count_with_time = text("SELECT COUNT(*) FROM ig_candles WHERE epic = :epic AND timeframe = :timeframe AND start_time >= :start_time")
                    count_time_result = conn.execute(count_with_time, {"epic": epic, "timeframe": timeframe, "start_time": start_time})
                    time_count = count_time_result.fetchone()[0]
                    logger.info(f"Candles for {epic} since {start_time}: {time_count}")
                except Exception as debug_e:
                    logger.error(f"Debug queries failed: {debug_e}")
                
                result = conn.execute(query, {
                    "epic": epic,
                    "timeframe": timeframe,
                    "start_time": start_time
                })
                
                candles = []
                for row in result:
                    candles.append({
                        "time": row.start_time,
                        "epic": row.epic,
                        "timeframe": row.timeframe,
                        "open": row.open,
                        "high": row.high,
                        "low": row.low,
                        "close": row.close,
                        "volume": row.volume,
                        "created_at": row.created_at
                    })
                
                logger.info(f"Found {len(candles)} candles for {epic}")
                if candles:
                    logger.info(f"Latest candle time: {candles[0]['time']}")
                
                return {
                    "epic": epic,
                    "timeframe": timeframe,
                    "candle_count": len(candles),
                    "candles": candles,
                    "latest_candle": candles[0] if candles else None
                }
                
        except Exception as e:
            return {"error": f"Database query failed: {str(e)}"}
    
    def get_data_gaps_from_db(self, epic: str, timeframe: int = 5, hours_back: int = 24) -> Dict[str, Any]:
        """Detect data gaps directly from database"""
        if not self.db_engine:
            return {"error": "Database not available"}
        
        try:
            # Add debugging to see which database we're connected to
            with self.db_engine.connect() as debug_conn:
                db_name = debug_conn.execute(text("SELECT current_database()")).fetchone()[0]
                logger.info(f"Gap detection connecting to database: {db_name}")
                
                # Check if table exists
                table_check = debug_conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'ig_candles'
                    )
                """)).fetchone()[0]
                logger.info(f"ig_candles table exists: {table_check}")
        except Exception as debug_e:
            logger.error(f"Debug check failed: {debug_e}")
        
        try:
            query = text("""
                SELECT 
                    start_time,
                    LAG(start_time) OVER (ORDER BY start_time) as prev_time,
                    EXTRACT(EPOCH FROM (start_time - LAG(start_time) OVER (ORDER BY start_time))) / 60 as gap_minutes
                FROM ig_candles 
                WHERE epic = :epic 
                    AND timeframe = :timeframe 
                    AND start_time >= :start_time
                ORDER BY start_time DESC
            """)
            
            start_time = datetime.now() - timedelta(hours=hours_back)
            expected_gap_minutes = timeframe
            
            with self.db_engine.connect() as conn:
                result = conn.execute(query, {
                    "epic": epic,
                    "timeframe": timeframe, 
                    "start_time": start_time
                })
                
                gaps = []
                for row in result:
                    if row.gap_minutes and row.gap_minutes > expected_gap_minutes * 1.5:  # Allow 50% tolerance
                        gaps.append({
                            "start_time": row.prev_time,
                            "end_time": row.start_time,
                            "gap_minutes": row.gap_minutes,
                            "missing_candles": int(row.gap_minutes / expected_gap_minutes) - 1
                        })
                
                return {
                    "epic": epic,
                    "timeframe": timeframe,
                    "total_gaps": len(gaps),
                    "gaps": gaps[:10]  # Return up to 10 most recent gaps
                }
                
        except Exception as e:
            return {"error": f"Gap detection failed: {str(e)}"}
    
    def get_database_health(self) -> Dict[str, Any]:
        """Check overall database health"""
        if not self.db_engine:
            return {"error": "Database not available", "status": "offline"}
        
        try:
            # Check connection and get basic stats
            with self.db_engine.connect() as conn:
                # First check what database we're connected to
                db_check = conn.execute(text("SELECT current_database()"))
                current_db = db_check.fetchone()[0]
                
                # Check if tables exist and get row counts
                tables_query = text("""
                    SELECT 
                        schemaname,
                        relname as tablename,
                        n_live_tup as row_count,
                        last_vacuum,
                        last_analyze
                    FROM pg_stat_user_tables 
                    WHERE relname IN ('ig_candles', 'candles', 'trade_logs')
                    ORDER BY relname
                """)
                
                result = conn.execute(tables_query)
                table_stats = []
                
                for row in result:
                    table_stats.append({
                        "schema": row.schemaname,
                        "table": row.tablename,
                        "row_count": row.row_count,
                        "last_vacuum": row.last_vacuum,
                        "last_analyze": row.last_analyze
                    })
                
                # Try to get candle stats with error handling
                candle_stats = {"error": "No candle data available"}
                try:
                    candle_stats_query = text("""
                        SELECT 
                            COUNT(*) as total_candles,
                            COUNT(DISTINCT epic) as unique_epics,
                            MIN(start_time) as oldest_candle,
                            MAX(start_time) as newest_candle,
                            COUNT(CASE WHEN start_time >= NOW() - INTERVAL '1 hour' THEN 1 END) as recent_candles
                        FROM ig_candles
                    """)
                    
                    candle_result = conn.execute(candle_stats_query)
                    row = candle_result.fetchone()
                    if row:
                        candle_stats = {
                            "total_candles": row.total_candles,
                            "unique_epics": row.unique_epics,
                            "oldest_candle": row.oldest_candle,
                            "newest_candle": row.newest_candle,
                            "recent_candles": row.recent_candles
                        }
                except Exception as e:
                    candle_stats = {"error": f"Candle query failed: {str(e)}"}
                
                return {
                    "status": "healthy",
                    "connection": "active",
                    "current_database": current_db,
                    "table_stats": table_stats,
                    "candle_stats": candle_stats,
                    "last_check": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "error": f"Database health check failed: {str(e)}",
                "status": "error"
            }
    
    def get_epic_streaming_health(self, epics: List[str], timeframe: int = 5) -> List[Dict[str, Any]]:
        """Get comprehensive streaming health for multiple epics"""
        health_results = []
        
        for epic in epics:
            # Get latest candle from database
            db_health = self.get_candle_health_from_db(epic, timeframe, hours_back=1)
            
            # Get API health
            api_health = self.get_latest_candle_from_api(epic, timeframe)
            
            # Calculate health metrics
            status = "unknown"
            latest_time = None
            age_minutes = None
            
            if "error" not in db_health and db_health.get("latest_candle"):
                latest_candle = db_health["latest_candle"]
                latest_time = latest_candle["time"]
                
                if isinstance(latest_time, str):
                    latest_dt = datetime.fromisoformat(latest_time)
                else:
                    latest_dt = latest_time
                
                age_minutes = (datetime.now() - latest_dt).total_seconds() / 60
                
                # Determine health status
                if age_minutes <= timeframe * 2:
                    status = "healthy"
                elif age_minutes <= timeframe * 5:
                    status = "stale"
                else:
                    status = "very_stale"
            else:
                status = "error"
            
            health_results.append({
                "epic": epic,
                "status": status,
                "latest_time": latest_time,
                "age_minutes": age_minutes,
                "candle_count": db_health.get("candle_count", 0),
                "db_error": db_health.get("error"),
                "api_error": api_health.get("error")
            })
        
        return health_results