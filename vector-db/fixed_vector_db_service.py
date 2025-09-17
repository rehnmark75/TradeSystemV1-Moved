#!/usr/bin/env python3
"""
Fixed Vector Database Service for TradingView RAG System
========================================================

This is a simplified, working version that fixes the async issues.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/vector_db.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for semantic matching")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    filters: Optional[Dict[str, str]] = Field(default=None, description="Optional filters for search")

class SearchResult(BaseModel):
    id: str
    title: str
    similarity_score: float
    metadata: Dict[str, Any]
    content_preview: str

class FixedVectorDatabaseService:
    """Fixed vector database service with working search"""

    def __init__(self):
        """Initialize the fixed vector database service"""
        self.app = FastAPI(
            title="Fixed TradingView Vector Database API",
            description="Working semantic search for TradingView indicators",
            version="1.1.0"
        )

        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Database settings
        self.db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")

        # ChromaDB settings
        self.chroma_path = "/app/vectordb"
        self.data_path = "/app/data"

        # Initialize ChromaDB client
        self.client = None
        self.indicators_collection = None
        self.templates_collection = None

        # Data status
        self.data_loaded = False
        self.last_sync = None

        # Setup routes
        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes with proper async handling"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": "fixed-vector-db",
                "version": "1.1.0",
                "timestamp": datetime.now().isoformat(),
                "data_loaded": self.data_loaded,
                "last_sync": self.last_sync,
                "collections_ready": bool(self.indicators_collection)
            }

        @self.app.post("/search/indicators", response_model=List[SearchResult])
        async def search_indicators(request: SearchRequest):
            """Search TradingView indicators with semantic matching - FIXED VERSION"""
            try:
                if not self.indicators_collection:
                    raise HTTPException(status_code=503, detail="Vector database not initialized")

                # Perform semantic search (synchronous method)
                results = self._semantic_search_sync(
                    collection=self.indicators_collection,
                    query=request.query,
                    n_results=request.n_results,
                    filters=request.filters
                )

                return [SearchResult(
                    id=r['id'],
                    title=r['metadata'].get('title', 'Unknown'),
                    similarity_score=r['similarity_score'],
                    metadata=r['metadata'],
                    content_preview=r['document'][:200] + "..." if len(r['document']) > 200 else r['document']
                ) for r in results]

            except Exception as e:
                logger.error(f"Error in indicator search: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/search/templates", response_model=List[SearchResult])
        async def search_strategy_templates(request: SearchRequest):
            """Search strategy templates with semantic matching - FIXED VERSION"""
            try:
                if not self.templates_collection:
                    raise HTTPException(status_code=503, detail="Templates collection not initialized")

                results = self._semantic_search_sync(
                    collection=self.templates_collection,
                    query=request.query,
                    n_results=request.n_results,
                    filters=request.filters
                )

                return [SearchResult(
                    id=r['id'],
                    title=f"{r['metadata'].get('strategy_type', 'Unknown')} Strategy for {r['metadata'].get('epic', 'Unknown')}",
                    similarity_score=r['similarity_score'],
                    metadata=r['metadata'],
                    content_preview=r['document'][:200] + "..." if len(r['document']) > 200 else r['document']
                ) for r in results]

            except Exception as e:
                logger.error(f"Error in template search: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/similar/{indicator_id}")
        async def get_similar_indicators(indicator_id: str, n_results: int = Query(default=3, ge=1, le=10)):
            """Find indicators similar to a given indicator - FIXED VERSION"""
            try:
                if not self.indicators_collection:
                    raise HTTPException(status_code=503, detail="Vector database not initialized")

                results = self._find_similar_sync(
                    collection=self.indicators_collection,
                    indicator_id=indicator_id,
                    n_results=n_results
                )

                return [SearchResult(
                    id=r['id'],
                    title=r['metadata'].get('title', 'Unknown'),
                    similarity_score=r['similarity_score'],
                    metadata=r['metadata'],
                    content_preview=r['document'][:200] + "..." if len(r['document']) > 200 else r['document']
                ) for r in results]

            except Exception as e:
                logger.error(f"Error finding similar indicators: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats")
        async def get_database_stats():
            """Get comprehensive database statistics"""
            try:
                return {
                    "indicators_count": self.indicators_collection.count() if self.indicators_collection else 0,
                    "templates_count": self.templates_collection.count() if self.templates_collection else 0,
                    "data_loaded": self.data_loaded,
                    "last_sync": self.last_sync,
                    "service_version": "1.1.0-fixed"
                }
            except Exception as e:
                logger.error(f"Error getting database stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sync")
        async def sync_data(background_tasks: BackgroundTasks):
            """Trigger data synchronization from TradingView API and PostgreSQL"""
            background_tasks.add_task(self._sync_vector_data)
            return {"message": "Data synchronization started", "timestamp": datetime.now().isoformat()}

    def _semantic_search_sync(self, collection, query: str, n_results: int, filters: Optional[Dict] = None) -> List[Dict]:
        """Synchronous semantic search method"""
        try:
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ['collection', 'category', 'technical_focus', 'complexity_level']:
                        where_clause[key] = value

            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )

            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _find_similar_sync(self, collection, indicator_id: str, n_results: int) -> List[Dict]:
        """Synchronous method to find similar indicators"""
        try:
            # Get the original indicator
            original = collection.get(ids=[indicator_id])
            if not original['documents']:
                return []

            # Search for similar indicators
            results = collection.query(
                query_texts=[original['documents'][0]],
                n_results=n_results + 1,  # +1 to exclude self
                include=['documents', 'metadatas', 'distances']
            )

            # Filter out the original indicator and format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                if results['ids'][0][i] != indicator_id:  # Exclude self
                    result = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],
                    }
                    formatted_results.append(result)

            return formatted_results[:n_results]  # Return only requested number

        except Exception as e:
            logger.error(f"Error finding similar indicators: {e}")
            return []

    async def initialize_vector_db(self):
        """Initialize ChromaDB and load data"""
        try:
            logger.info("🚀 Initializing Fixed Vector Database Service...")

            # Create directories
            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            Path(self.data_path).mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create embedding function
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            # Create or get collections
            self.indicators_collection = self.client.get_or_create_collection(
                name="tradingview_indicators",
                embedding_function=embedding_function,
                metadata={"description": "TradingView indicators with semantic search"}
            )

            self.templates_collection = self.client.get_or_create_collection(
                name="strategy_templates",
                embedding_function=embedding_function,
                metadata={"description": "Optimized strategy templates"}
            )

            # Check if data needs to be loaded
            if self.indicators_collection.count() == 0:
                logger.info("📊 Collections empty, triggering data sync...")
                await self._sync_vector_data()
            else:
                logger.info(f"✅ Collections loaded: {self.indicators_collection.count()} indicators")
                self.data_loaded = True

            logger.info("✅ Fixed Vector Database Service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {e}")
            raise

    async def _sync_vector_data(self):
        """Synchronize data from TradingView API and PostgreSQL"""
        try:
            logger.info("🔄 Starting vector data synchronization...")

            # Fetch data from PostgreSQL
            indicators_data = await self._fetch_tradingview_data()
            templates_data = await self._fetch_optimization_templates()

            # Populate collections
            if indicators_data:
                await self._populate_indicators_collection(indicators_data)

            if templates_data:
                await self._populate_templates_collection(templates_data)

            # Update sync status
            self.data_loaded = True
            self.last_sync = datetime.now().isoformat()

            logger.info("✅ Vector data synchronization completed")

        except Exception as e:
            logger.error(f"❌ Vector data sync failed: {e}")

    async def _fetch_tradingview_data(self) -> List[Dict]:
        """Fetch TradingView indicators from PostgreSQL"""
        try:
            logger.info("📊 Fetching TradingView indicators from database...")

            connection = psycopg2.connect(self.db_url)
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    slug,
                    title,
                    author,
                    description,
                    indicators,
                    signals,
                    timeframes,
                    complexity_score,
                    CASE
                        WHEN is_luxalgo THEN 'LuxAlgo'
                        WHEN is_zeiierman THEN 'Zeiierman'
                        WHEN is_lazybear THEN 'LazyBear'
                        WHEN is_chrismoody THEN 'ChrisMoody'
                        WHEN is_chartprime THEN 'ChartPrime'
                        WHEN is_bigbeluga THEN 'BigBeluga'
                        WHEN is_algoalpha THEN 'AlgoAlpha'
                        ELSE 'Other'
                    END as collection
                FROM tradingview.scripts
                WHERE (is_luxalgo OR is_zeiierman OR is_lazybear OR is_chrismoody
                       OR is_chartprime OR is_bigbeluga OR is_algoalpha)
                ORDER BY likes DESC
                LIMIT 100
            """)

            results = cursor.fetchall()
            connection.close()

            # Transform to enriched format
            enriched_data = []
            for row in results:
                embedding_text = self._create_embedding_text(dict(row))

                enriched = {
                    'id': row['slug'],
                    'embedding_text': embedding_text,
                    'metadata': {
                        'title': row['title'],
                        'author': row['author'],
                        'collection': row['collection'],
                        'complexity_score': float(row['complexity_score']),
                        'indicators': ','.join(row['indicators']) if row['indicators'] else '',
                        'signals': ','.join(row['signals']) if row['signals'] else '',
                        'timeframes': ','.join(row['timeframes']) if row['timeframes'] else ''
                    }
                }
                enriched_data.append(enriched)

            logger.info(f"✅ Fetched {len(enriched_data)} TradingView indicators")
            return enriched_data

        except Exception as e:
            logger.error(f"Error fetching TradingView data: {e}")
            return []

    async def _fetch_optimization_templates(self) -> List[Dict]:
        """Fetch optimization templates from PostgreSQL"""
        try:
            logger.info("🎯 Fetching optimization templates from database...")

            connection = psycopg2.connect(self.db_url)
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            templates = []

            # Fetch EMA templates
            cursor.execute("""
                SELECT
                    'EMA' as strategy_type,
                    epic,
                    best_win_rate,
                    best_net_pips,
                    best_timeframe
                FROM ema_best_parameters
                WHERE best_net_pips > 50
                ORDER BY best_net_pips DESC
                LIMIT 20
            """)

            ema_templates = cursor.fetchall()
            for template in ema_templates:
                enriched = self._create_strategy_template(dict(template))
                templates.append(enriched)

            connection.close()
            logger.info(f"✅ Fetched {len(templates)} optimization templates")
            return templates

        except Exception as e:
            logger.error(f"Error fetching optimization templates: {e}")
            return []

    def _create_embedding_text(self, indicator: Dict) -> str:
        """Create embedding text for an indicator"""
        parts = []

        if indicator.get('title'):
            parts.append(f"Title: {indicator['title']}")
        if indicator.get('description'):
            parts.append(f"Description: {indicator['description']}")
        if indicator.get('author'):
            parts.append(f"Author: {indicator['author']}")
        if indicator.get('collection'):
            parts.append(f"Collection: {indicator['collection']}")

        if indicator.get('indicators'):
            parts.append(f"Indicators: {', '.join(indicator['indicators'])}")
        if indicator.get('signals'):
            parts.append(f"Signals: {', '.join(indicator['signals'])}")

        complexity_level = 'Expert' if indicator.get('complexity_score', 0.5) > 0.8 else 'Advanced' if indicator.get('complexity_score', 0.5) > 0.6 else 'Intermediate' if indicator.get('complexity_score', 0.5) > 0.4 else 'Basic'
        parts.append(f"Complexity: {complexity_level}")

        return " | ".join(parts)

    def _create_strategy_template(self, template: Dict) -> Dict:
        """Create strategy template for RAG"""
        description_parts = [
            f"{template['strategy_type']} strategy for {template['epic']}",
            f"Timeframe: {template['best_timeframe']}",
            f"Performance: {template.get('best_win_rate', 0):.1%} win rate",
            f"Net pips: {template.get('best_net_pips', 0):.0f}"
        ]

        template_text = " | ".join(description_parts)

        return {
            'id': f"strategy_{template['epic'].replace('.', '_')}",
            'embedding_text': template_text,
            'metadata': {
                'strategy_type': template['strategy_type'],
                'epic': template['epic'],
                'timeframe': template['best_timeframe'],
                'win_rate': float(template.get('best_win_rate', 0)),
                'net_pips': float(template.get('best_net_pips', 0))
            }
        }

    async def _populate_indicators_collection(self, indicators_data: List[Dict]):
        """Populate indicators collection with embedding data"""
        try:
            if not indicators_data:
                logger.warning("No indicators data to populate")
                return

            logger.info(f"📊 Populating indicators collection with {len(indicators_data)} items...")

            # Clear existing data
            try:
                count = self.indicators_collection.count()
                if count > 0:
                    existing_data = self.indicators_collection.get()
                    if existing_data['ids']:
                        self.indicators_collection.delete(ids=existing_data['ids'])
            except Exception as e:
                logger.warning(f"Could not clear existing data: {e}")

            # Prepare data for batch insert
            ids = []
            documents = []
            metadatas = []

            for indicator in indicators_data:
                ids.append(indicator['id'])
                documents.append(indicator['embedding_text'])
                metadatas.append(indicator['metadata'])

            # Batch insert to ChromaDB
            self.indicators_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"✅ Populated indicators collection with {len(ids)} items")

        except Exception as e:
            logger.error(f"❌ Error populating indicators collection: {e}")

    async def _populate_templates_collection(self, templates_data: List[Dict]):
        """Populate strategy templates collection"""
        try:
            if not templates_data:
                logger.warning("No templates data to populate")
                return

            logger.info(f"🎯 Populating templates collection with {len(templates_data)} items...")

            # Clear existing data
            try:
                count = self.templates_collection.count()
                if count > 0:
                    existing_data = self.templates_collection.get()
                    if existing_data['ids']:
                        self.templates_collection.delete(ids=existing_data['ids'])
            except Exception as e:
                logger.warning(f"Could not clear existing templates data: {e}")

            # Prepare data for batch insert
            ids = []
            documents = []
            metadatas = []

            for template in templates_data:
                ids.append(template['id'])
                documents.append(template['embedding_text'])
                metadatas.append(template['metadata'])

            # Batch insert to ChromaDB
            self.templates_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"✅ Populated templates collection with {len(ids)} items")

        except Exception as e:
            logger.error(f"❌ Error populating templates collection: {e}")

# Global service instance
fixed_vector_service = FixedVectorDatabaseService()

# FastAPI app instance
app = fixed_vector_service.app

@app.on_event("startup")
async def startup_event():
    """Initialize vector database on startup"""
    await fixed_vector_service.initialize_vector_db()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔌 Fixed Vector Database Service shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "fixed_vector_db_service:app",
        host="0.0.0.0",
        port=8090,
        reload=False,
        log_level="info"
    )