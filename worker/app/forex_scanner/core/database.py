# Complete fixed version of forex_scanner/core/database.py

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Optional
import pandas as pd


class DatabaseManager:
    """Database connection and query manager"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.Session = None
        self.logger = logging.getLogger(__name__)
        
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.database_url)
            self.Session = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info("✅ Database connection established")
            
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def get_engine(self):
        """Get SQLAlchemy engine"""
        return self.engine
    
    def get_connection(self):
        """
        Get raw database connection for psycopg2 operations
        This method is needed for AlertHistoryManager compatibility
        """
        try:
            # Always use direct psycopg2 connection to ensure context manager support
            return self._create_psycopg2_connection()
        except Exception as e:
            self.logger.error(f"❌ Failed to get raw connection: {e}")
            raise
    
    def _create_psycopg2_connection(self):
        """Fallback method to create psycopg2 connection directly"""
        import psycopg2
        from urllib.parse import urlparse
        
        try:
            # Parse the database URL to get connection parameters
            parsed = urlparse(self.database_url)
            
            # Create psycopg2 connection using parsed URL components
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:] if parsed.path else None,  # Remove leading '/'
                user=parsed.username,
                password=parsed.password
            )
            
            return conn
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create psycopg2 connection: {e}")
            raise
    
    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        try:
            with self.engine.connect() as conn:
                if isinstance(query, str):
                    # Convert string to text() object
                    query_obj = text(query)
                    query_str = query.strip().upper()
                else:
                    # Already a text() object
                    query_obj = query
                    query_str = str(query).strip().upper()

                result = conn.execute(query_obj, params or {})

                # Check if this is a non-SELECT query that doesn't return rows
                if ((query_str.startswith('UPDATE') or
                     query_str.startswith('DELETE') or
                     (query_str.startswith('INSERT') and 'RETURNING' not in query_str)) and
                    'VOID' not in query_str):
                    # For UPDATE/DELETE queries or INSERT without RETURNING, return empty DataFrame
                    # but commit the transaction
                    conn.commit()
                    return pd.DataFrame()
                else:
                    # For SELECT queries, INSERT with RETURNING, or function calls, return the data as DataFrame
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    return df

        except Exception as e:
            self.logger.error(f"❌ Query execution failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error(f"❌ Connection test failed: {e}")
            return False