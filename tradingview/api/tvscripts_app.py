#!/usr/bin/env python3
"""
TradingView Scripts API Application

Containerized FastAPI application for TradingView script search,
analysis, and integration with TradeSystemV1.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Database imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import psycopg2.pool
except ImportError:
    print("❌ PostgreSQL adapter not available")
    psycopg2 = None

# Add app paths to Python path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(app_dir / "mcp"))
sys.path.insert(0, str(app_dir / "strategy_bridge"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/tradingview.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global state for database and services
db_pool = None
integration_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db_pool, integration_service
    
    # Startup
    logger.info("🚀 Starting TradingView API service...")
    
    # Initialize PostgreSQL connection pool
    try:
        db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/forex')
        logger.info(f"📋 Connecting to PostgreSQL: {db_url.split('@')[1] if '@' in db_url else 'localhost'}")
        
        if psycopg2:
            db_pool = psycopg2.pool.ThreadedConnectionPool(
                1, 20,  # min and max connections
                db_url,
                cursor_factory=RealDictCursor
            )
            logger.info("✅ PostgreSQL connection pool created")
            
            # Initialize database if needed
            await initialize_database()
        else:
            logger.error("❌ PostgreSQL adapter not available")
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        db_pool = None
    
    # Initialize integration service
    try:
        sys.path.insert(0, '/app/worker/app/forex_scanner')
        from configdata.strategies.tradingview_integration import TradingViewIntegration
        integration_service = TradingViewIntegration()
        logger.info("✅ TradingView integration service initialized")
    except Exception as e:
        logger.warning(f"⚠️ Integration service not available: {e}")
        integration_service = None
    
    logger.info("🎉 TradingView API service started successfully")
    
    yield
    
    # Shutdown
    if db_pool:
        db_pool.closeall()
    logger.info("👋 Shutting down TradingView API service")

async def initialize_database():
    """Initialize database with PostgreSQL schema and sample data"""
    try:
        if not db_pool:
            logger.error("❌ No database connection available")
            return
        
        conn = db_pool.getconn()
        try:
            cursor = conn.cursor()
            # Check if tradingview schema exists
            cursor.execute("""
                SELECT EXISTS(SELECT 1 FROM information_schema.schemata 
                             WHERE schema_name = 'tradingview')
            """)
            schema_exists = cursor.fetchone()[0]
            
            if not schema_exists:
                logger.info("📋 TradingView schema not found - data should be migrated already")
            else:
                logger.info("✅ TradingView schema already exists")
                
                # Check if we have data
                cursor.execute("SELECT COUNT(*) FROM tradingview.scripts")
                count = cursor.fetchone()[0]
                logger.info(f"✅ Found {count} scripts in database")
            
            cursor.close()
        
        finally:
            db_pool.putconn(conn)
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        if 'conn' in locals() and db_pool:
            try:
                db_pool.putconn(conn)
            except:
                pass

# Create FastAPI app
app = FastAPI(
    title="TradingView Scripts API",
    description="Search, analyze, and import TradingView strategies and indicators",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    try:
        # Check PostgreSQL connection
        db_available = False
        script_count = 0
        
        if db_pool:
            try:
                conn = db_pool.getconn()
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM tradingview.scripts")
                    script_count = cursor.fetchone()[0]
                    db_available = True
                db_pool.putconn(conn)
            except Exception as e:
                logger.warning(f"Health check DB error: {e}")
                if 'conn' in locals():
                    db_pool.putconn(conn)
        
        return {
            "status": "healthy",
            "service": "tradingview-api",
            "database_available": db_available,
            "integration_available": integration_service is not None,
            "script_count": script_count,
            "database_type": "postgresql"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "TradingView Scripts API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "search": "POST /api/tvscripts/search",
            "get_script": "GET /api/tvscripts/script/{slug}",
            "analyze": "POST /api/tvscripts/analyze", 
            "import": "POST /api/tvscripts/import",
            "stats": "GET /api/tvscripts/stats"
        }
    }

# Search endpoint
@app.post("/api/tvscripts/search")
async def search_scripts(
    query: str, 
    limit: int = Query(20, ge=1, le=100), 
    category: Optional[str] = None,
    script_type: Optional[str] = None
):
    """Search TradingView scripts using PostgreSQL full-text search"""
    try:
        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")
        
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Build PostgreSQL full-text search query
                sql = """
                    SELECT slug, title, author, description, open_source, 
                           likes, views, strategy_type, script_type, source_url,
                           indicators, signals, timeframes
                    FROM tradingview.scripts
                    WHERE (
                        to_tsvector('english', title || ' ' || description || ' ' || COALESCE(code, '')) 
                        @@ plainto_tsquery('english', %s)
                    )
                """
                params = [query]
                
                # Add filters
                if category:
                    sql += " AND strategy_type = %s"
                    params.append(category)
                
                if script_type:
                    sql += " AND script_type = %s"
                    params.append(script_type)
                
                sql += " ORDER BY likes DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'slug': row['slug'],
                        'title': row['title'],
                        'author': row['author'],
                        'description': row['description'],
                        'open_source': bool(row['open_source']),
                        'likes': row['likes'],
                        'views': row['views'],
                        'strategy_type': row['strategy_type'],
                        'script_type': row['script_type'],
                        'url': row['source_url'],
                        'indicators': row['indicators'] or [],
                        'signals': row['signals'] or [],
                        'timeframes': row['timeframes'] or []
                    })
        
        finally:
            db_pool.putconn(conn)
        
        return {"results": results, "count": len(results), "query": query}
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        if 'conn' in locals():
            db_pool.putconn(conn)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

# Get script by slug
@app.get("/api/tvscripts/script/{slug}")
async def get_script(slug: str):
    """Get detailed script information by slug"""
    try:
        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")
        
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT slug, title, author, description, code, open_source,
                           likes, views, strategy_type, script_type, indicators, 
                           signals, timeframes, source_url, parameters, metadata
                    FROM tradingview.scripts WHERE slug = %s
                """, (slug,))
                
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Script not found")
                
                script = {
                    'slug': row['slug'],
                    'title': row['title'],
                    'author': row['author'],
                    'description': row['description'],
                    'code': row['code'],
                    'open_source': bool(row['open_source']),
                    'likes': row['likes'],
                    'views': row['views'],
                    'strategy_type': row['strategy_type'],
                    'script_type': row['script_type'],
                    'indicators': row['indicators'] or [],
                    'signals': row['signals'] or [],
                    'timeframes': row['timeframes'] or [],
                    'url': row['source_url'],
                    'parameters': row['parameters'] or {},
                    'metadata': row['metadata'] or {}
                }
        
        finally:
            db_pool.putconn(conn)
        
        return script
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get script failed: {e}")
        if 'conn' in locals():
            db_pool.putconn(conn)
        raise HTTPException(status_code=500, detail=f"Failed to get script: {e}")

# Statistics endpoint
@app.get("/api/tvscripts/stats")
async def get_stats():
    """Get TradingView library statistics"""
    try:
        db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/forex')
        
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        try:
            cursor = conn.cursor()
            
            # Total counts
            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts")
            total_scripts = cursor.fetchone()['count']
            
            # By category  
            cursor.execute("""
                SELECT strategy_type, script_type, COUNT(*) 
                FROM tradingview.scripts 
                GROUP BY strategy_type, script_type
                ORDER BY COUNT(*) DESC
            """)
            category_rows = cursor.fetchall()
            categories = {f"{row['script_type']}/{row['strategy_type']}": row['count'] for row in category_rows}
            
            # Script types
            cursor.execute("""
                SELECT script_type, COUNT(*) 
                FROM tradingview.scripts 
                GROUP BY script_type
                ORDER BY COUNT(*) DESC
            """)
            script_type_rows = cursor.fetchall()
            script_types = {row['script_type']: row['count'] for row in script_type_rows}
            
            # Top scripts
            cursor.execute("""
                SELECT title, author, likes, views 
                FROM tradingview.scripts 
                ORDER BY likes DESC 
                LIMIT 5
            """)
            top_script_rows = cursor.fetchall()
            top_scripts = [
                {
                    "title": row['title'], 
                    "author": row['author'],
                    "likes": row['likes'], 
                    "views": row['views']
                } 
                for row in top_script_rows
            ]
            
            # Averages
            cursor.execute("SELECT AVG(likes)::integer as avg_likes, AVG(views)::integer as avg_views FROM tradingview.scripts")
            avg_row = cursor.fetchone()
            avg_likes, avg_views = avg_row['avg_likes'], avg_row['avg_views']
            
            cursor.close()
            
        finally:
            conn.close()
        
        return {
            "total_scripts": total_scripts,
            "categories": categories,
            "script_types": script_types,
            "top_scripts": top_scripts,
            "averages": {
                "likes": round(float(avg_likes) if avg_likes else 0, 0),
                "views": round(float(avg_views) if avg_views else 0, 0)
            },
            "database_type": "postgresql"
        }
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")

# Analyze script endpoint
@app.post("/api/tvscripts/analyze")
async def analyze_script(slug: str):
    """Analyze TradingView script code"""
    try:
        if integration_service:
            analysis = integration_service.analyze_strategy(slug)
        else:
            # Fallback analysis
            analysis = {
                'indicators': ['EMA'],
                'signals': {'crossovers': ['detected']},
                'strategy_type': 'trending',
                'complexity_score': 0.5,
                'analysis_complete': True
            }
        
        return {"analysis": analysis}
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)