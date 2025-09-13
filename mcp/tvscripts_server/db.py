"""
Database operations for TradingView Scripts MCP Server

Handles SQLite database operations including:
- Script metadata storage
- Full-text search with FTS5
- Script body storage and retrieval
- Performance tracking
"""

import sqlite3
import hashlib
import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DB:
    """Database manager for TradingView scripts with FTS5 search capabilities"""
    
    def __init__(self, path: str = "tv_scripts.db"):
        """Initialize database connection and create tables if needed"""
        self.db_path = Path(path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        
    def _create_tables(self):
        """Create database tables and FTS5 virtual table"""
        cursor = self.conn.cursor()
        
        # Main scripts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scripts(
                id INTEGER PRIMARY KEY,
                slug TEXT UNIQUE NOT NULL,
                title TEXT,
                author TEXT,
                tags TEXT,
                open_source INTEGER NOT NULL,
                url TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                script_type TEXT,
                description TEXT,
                likes_count INTEGER DEFAULT 0,
                uses_count INTEGER DEFAULT 0
            )
        """)
        
        # Script bodies table (separate for performance)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS script_bodies(
                script_id INTEGER PRIMARY KEY,
                code TEXT,
                code_sha256 TEXT,
                normalized_code TEXT,
                extracted_indicators TEXT,
                extracted_parameters TEXT,
                FOREIGN KEY(script_id) REFERENCES scripts(id)
            )
        """)
        
        # FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS scripts_fts USING fts5(
                title, tags, description, code, author,
                content='',
                tokenize='porter'
            )
        """)
        
        # Strategy import tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_imports(
                id INTEGER PRIMARY KEY,
                script_id INTEGER,
                import_timestamp INTEGER,
                config_name TEXT,
                generated_config TEXT,
                performance_score REAL,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY(script_id) REFERENCES scripts(id)
            )
        """)
        
        # Optimization results for imported strategies
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS imported_strategy_performance(
                id INTEGER PRIMARY KEY,
                import_id INTEGER,
                epic TEXT,
                optimization_score REAL,
                win_rate REAL,
                profit_factor REAL,
                net_pips REAL,
                total_signals INTEGER,
                created_at INTEGER,
                FOREIGN KEY(import_id) REFERENCES strategy_imports(id)
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_slug ON scripts(slug)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_open_source ON scripts(open_source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_type ON scripts(script_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_imports_status ON strategy_imports(status)")
        
        self.conn.commit()
        logger.info("Database tables created/verified successfully")
    
    def search(self, query: str, limit: int = 20, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Full-text search across scripts with optional filters
        
        Args:
            query: Search query string
            limit: Maximum results to return
            filters: Optional filters (open_source_only, script_type, etc.)
        
        Returns:
            List of script dictionaries
        """
        try:
            # Base FTS5 search
            base_query = """
                SELECT s.slug, s.title, s.author, s.tags, s.open_source, s.url, 
                       s.description, s.likes_count, s.uses_count, s.script_type,
                       rank
                FROM scripts_fts 
                JOIN scripts s ON s.rowid = scripts_fts.rowid
                WHERE scripts_fts MATCH ?
            """
            
            params = [query]
            
            # Add filters
            if filters:
                if filters.get('open_source_only', True):
                    base_query += " AND s.open_source = 1"
                
                if filters.get('script_type'):
                    base_query += " AND s.script_type = ?"
                    params.append(filters['script_type'])
                
                if filters.get('min_likes'):
                    base_query += " AND s.likes_count >= ?"
                    params.append(filters['min_likes'])
            
            base_query += " ORDER BY rank LIMIT ?"
            params.append(limit)
            
            cursor = self.conn.execute(base_query, params)
            results = [dict(row) for row in cursor.fetchall()]
            
            logger.info(f"Search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def get(self, slug: str) -> Optional[Dict]:
        """
        Get script by slug with full details including code
        
        Args:
            slug: Script slug identifier
            
        Returns:
            Script dictionary with code if available
        """
        try:
            query = """
                SELECT s.slug, s.title, s.author, s.tags, s.open_source, s.url,
                       s.description, s.likes_count, s.uses_count, s.script_type,
                       s.created_at, s.updated_at,
                       b.code, b.normalized_code, b.extracted_indicators, b.extracted_parameters
                FROM scripts s 
                LEFT JOIN script_bodies b ON b.script_id = s.id 
                WHERE s.slug = ?
            """
            
            cursor = self.conn.execute(query, (slug,))
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                logger.info(f"Retrieved script: {slug}")
                return result
            else:
                logger.warning(f"Script not found: {slug}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get script {slug}: {e}")
            return None
    
    def save_script(self, metadata: Dict, code: Optional[str] = None) -> bool:
        """
        Save or update script metadata and code
        
        Args:
            metadata: Script metadata dictionary
            code: Pine Script code (if available)
            
        Returns:
            Success boolean
        """
        try:
            timestamp = int(time.time())
            
            # Insert or update script metadata
            cursor = self.conn.execute("""
                INSERT INTO scripts(
                    slug, title, author, tags, open_source, url, description,
                    likes_count, uses_count, script_type, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    title=excluded.title,
                    author=excluded.author,
                    tags=excluded.tags,
                    open_source=excluded.open_source,
                    url=excluded.url,
                    description=excluded.description,
                    likes_count=excluded.likes_count,
                    uses_count=excluded.uses_count,
                    script_type=excluded.script_type,
                    updated_at=excluded.updated_at
            """, (
                metadata.get('slug', ''),
                metadata.get('title', ''),
                metadata.get('author', ''),
                metadata.get('tags', ''),
                int(metadata.get('open_source', False)),
                metadata.get('url', ''),
                metadata.get('description', ''),
                metadata.get('likes_count', 0),
                metadata.get('uses_count', 0),
                metadata.get('script_type', ''),
                timestamp,
                timestamp
            ))
            
            # Get script ID
            script_id = cursor.lastrowid
            if not script_id:
                # Get existing ID
                cursor = self.conn.execute("SELECT id FROM scripts WHERE slug = ?", (metadata['slug'],))
                row = cursor.fetchone()
                script_id = row['id'] if row else None
            
            # Save script body if provided
            if code and script_id:
                # Normalize and extract indicators (placeholder for now)
                normalized_code = self._normalize_pine_code(code)
                code_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO script_bodies(
                        script_id, code, code_sha256, normalized_code
                    ) VALUES(?, ?, ?, ?)
                """, (script_id, code, code_hash, normalized_code))
            
            # Update FTS5 table
            if script_id:
                self.conn.execute("""
                    INSERT OR REPLACE INTO scripts_fts(rowid, title, tags, description, code, author)
                    VALUES(?, ?, ?, ?, ?, ?)
                """, (
                    script_id,
                    metadata.get('title', ''),
                    metadata.get('tags', ''),
                    metadata.get('description', ''),
                    code or '',
                    metadata.get('author', '')
                ))
            
            self.conn.commit()
            logger.info(f"Saved script: {metadata.get('slug', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save script: {e}")
            self.conn.rollback()
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            # Total scripts
            cursor = self.conn.execute("SELECT COUNT(*) as total FROM scripts")
            stats['total_scripts'] = cursor.fetchone()['total']
            
            # Open source scripts
            cursor = self.conn.execute("SELECT COUNT(*) as open_source FROM scripts WHERE open_source = 1")
            stats['open_source_scripts'] = cursor.fetchone()['open_source']
            
            # Scripts with code
            cursor = self.conn.execute("SELECT COUNT(*) as with_code FROM script_bodies WHERE code IS NOT NULL")
            stats['scripts_with_code'] = cursor.fetchone()['with_code']
            
            # Script types
            cursor = self.conn.execute("""
                SELECT script_type, COUNT(*) as count 
                FROM scripts 
                WHERE script_type IS NOT NULL 
                GROUP BY script_type
            """)
            stats['script_types'] = {row['script_type']: row['count'] for row in cursor.fetchall()}
            
            # Strategy imports
            cursor = self.conn.execute("SELECT COUNT(*) as imports FROM strategy_imports")
            stats['strategy_imports'] = cursor.fetchone()['imports']
            
            logger.info("Retrieved database statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def _normalize_pine_code(self, code: str) -> str:
        """Normalize Pine Script code for better analysis"""
        # Basic normalization - can be enhanced later
        lines = code.split('\n')
        normalized_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('//'):
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")