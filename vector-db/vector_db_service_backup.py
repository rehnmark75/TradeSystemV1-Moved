#!/usr/bin/env python3
"""
Vector Database Service for TradingView RAG System

This service provides a FastAPI-based REST API for semantic search and retrieval
of TradingView indicators and strategy templates using ChromaDB.
"""

import os
import json
import logging
import asyncio
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
import requests
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

class MemoryEvent(BaseModel):
    timestamp: datetime
    event_type: str  # search, composition, similarity, pattern_detected
    query: str
    results_count: int
    user_context: Optional[Dict[str, Any]] = None
    patterns_identified: Optional[List[str]] = None

class PatternInsight(BaseModel):
    pattern_type: str
    confidence: float
    description: str
    recommendations: List[str]
    historical_performance: Optional[Dict[str, float]] = None

class StrategyCompositionRequest(BaseModel):
    description: str = Field(..., description="Natural language description of desired strategy")
    market_condition: Optional[str] = Field(default="trending", description="Market condition: trending, ranging, volatile")
    trading_style: Optional[str] = Field(default="day_trading", description="Trading style: scalping, day_trading, swing_trading")
    complexity_level: Optional[str] = Field(default="intermediate", description="Complexity: beginner, intermediate, advanced")

class StrategyComposition(BaseModel):
    primary_indicators: List[SearchResult]
    confirmation_indicators: List[SearchResult]
    filter_indicators: List[SearchResult]
    strategy_template: Optional[SearchResult]
    reasoning: str

class VectorDatabaseService:
    """Main vector database service class"""

    def __init__(self):
        """Initialize the vector database service"""
        self.app = FastAPI(
            title="TradingView Vector Database API",
            description="Semantic search and strategy composition for TradingView indicators",
            version="1.0.0"
        )

        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Memory and pattern tracking
        self.memory_events: List[MemoryEvent] = []
        self.pattern_cache: Dict[str, PatternInsight] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.search_history: List[Dict[str, Any]] = []
        self.composition_patterns: Dict[str, int] = {}  # Track popular combinations

        # Database settings
        self.db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
        self.tradingview_api = os.getenv("TRADINGVIEW_API_URL", "http://tradingview:8080/api/tvscripts")

        # ChromaDB settings
        self.chroma_path = "/app/vectordb"
        self.data_path = "/app/data"

        # Initialize ChromaDB client
        self.client = None
        self.indicators_collection = None
        self.templates_collection = None
        self.taxonomy_collection = None

        # Data status
        self.data_loaded = False
        self.last_sync = None

        # Setup routes
        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "data_loaded": self.data_loaded,
                "last_sync": self.last_sync,
                "collections_ready": bool(self.indicators_collection and self.templates_collection)
            }

        @self.app.post("/search/indicators", response_model=List[SearchResult])
        async def search_indicators(request: SearchRequest):
            """Search TradingView indicators with semantic matching"""
            try:
                if not self.indicators_collection:
                    raise HTTPException(status_code=503, detail="Vector database not initialized")

                results = self.semantic_search_indicators(
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
            """Search strategy templates with semantic matching"""
            try:
                if not self.templates_collection:
                    raise HTTPException(status_code=503, detail="Templates collection not initialized")

                results = self.search_strategy_templates(
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

        @self.app.post("/compose/strategy", response_model=StrategyComposition)
        async def compose_strategy(request: StrategyCompositionRequest):
            """AI-powered strategy composition based on natural language description"""
            try:
                composition = self.compose_trading_strategy(
                    description=request.description,
                    market_condition=request.market_condition,
                    trading_style=request.trading_style,
                    complexity_level=request.complexity_level
                )

                return composition

            except Exception as e:
                logger.error(f"Error in strategy composition: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/similar/{indicator_id}")
        async def get_similar_indicators(indicator_id: str, n_results: int = Query(default=3, ge=1, le=10)):
            """Find indicators similar to a given indicator"""
            try:
                if not self.indicators_collection:
                    raise HTTPException(status_code=503, detail="Vector database not initialized")

                results = self.get_similar_indicators(indicator_id, n_results)

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

        @self.app.post("/sync")
        async def sync_data(background_tasks: BackgroundTasks):
            """Trigger data synchronization from TradingView API and PostgreSQL"""
            background_tasks.add_task(self.sync_vector_data)
            return {"message": "Data synchronization started", "timestamp": datetime.now().isoformat()}

        @self.app.get("/stats")
        async def get_database_stats():
            """Get comprehensive database statistics"""
            try:
                stats = self.get_database_stats()
                return stats
            except Exception as e:
                logger.error(f"Error getting database stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def initialize_vector_db(self):
        """Initialize ChromaDB and load data"""
        try:
            logger.info("🚀 Initializing Vector Database Service...")

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

            self.taxonomy_collection = self.client.get_or_create_collection(
                name="market_taxonomy",
                embedding_function=embedding_function,
                metadata={"description": "Market context taxonomy"}
            )

            # Check if data needs to be loaded
            if self.indicators_collection.count() == 0:
                logger.info("📊 Collections empty, triggering data sync...")
                await self.sync_vector_data()
            else:
                logger.info(f"✅ Collections loaded: {self.indicators_collection.count()} indicators")
                self.data_loaded = True

            logger.info("✅ Vector Database Service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {e}")
            raise

    async def sync_vector_data(self):
        """Synchronize data from TradingView API and PostgreSQL"""
        try:
            logger.info("🔄 Starting vector data synchronization...")

            # Fetch data from TradingView API
            indicators_data = await self.fetch_tradingview_data()

            # Fetch optimization templates from PostgreSQL
            templates_data = await self.fetch_optimization_templates()

            # Populate collections
            if indicators_data:
                await self.populate_indicators_collection(indicators_data)

            if templates_data:
                await self.populate_templates_collection(templates_data)

            # Update sync status
            self.data_loaded = True
            self.last_sync = datetime.now().isoformat()

            logger.info("✅ Vector data synchronization completed")

        except Exception as e:
            logger.error(f"❌ Vector data sync failed: {e}")
            raise

    async def fetch_tradingview_data(self) -> List[Dict]:
        """Fetch TradingView indicators from PostgreSQL"""
        try:
            logger.info("📊 Fetching TradingView indicators from database...")

            connection = psycopg2.connect(self.db_url)
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            # Query to get all enriched indicator data
            cursor.execute("""
                SELECT
                    slug,
                    title,
                    author,
                    description,
                    strategy_type,
                    indicators,
                    signals,
                    timeframes,
                    complexity_score,
                    open_source,
                    likes,
                    views,
                    CASE
                        WHEN is_luxalgo THEN 'LuxAlgo'
                        WHEN is_zeiierman THEN 'Zeiierman'
                        WHEN is_lazybear THEN 'LazyBear'
                        WHEN is_chrismoody THEN 'ChrisMoody'
                        WHEN is_chartprime THEN 'ChartPrime'
                        WHEN is_bigbeluga THEN 'BigBeluga'
                        WHEN is_algoalpha THEN 'AlgoAlpha'
                        ELSE 'Other'
                    END as collection,
                    COALESCE(
                        luxalgo_category,
                        zeiierman_category,
                        lazybear_category,
                        chrismoody_category,
                        chartprime_category,
                        bigbeluga_category,
                        algoalpha_category
                    ) as category,
                    COALESCE(
                        technical_focus,
                        innovation_type,
                        tool_type,
                        whale_tracking_type,
                        algorithm_type
                    ) as technical_focus
                FROM tradingview.scripts
                WHERE (is_luxalgo OR is_zeiierman OR is_lazybear OR is_chrismoody
                       OR is_chartprime OR is_bigbeluga OR is_algoalpha)
                ORDER BY likes DESC, complexity_score DESC
            """)

            results = cursor.fetchall()

            # Transform to enriched format
            enriched_data = []
            for row in results:
                # Create embedding text
                embedding_parts = [
                    f"Title: {row['title']}",
                    f"Description: {row['description'] or ''}",
                    f"Author: {row['author']} ({row['collection']} collection)",
                    f"Category: {row['category'] or ''}",
                    f"Technical Focus: {row['technical_focus'] or ''}",
                ]

                if row['indicators']:
                    embedding_parts.append(f"Indicators: {', '.join(row['indicators'])}")
                if row['signals']:
                    embedding_parts.append(f"Signals: {', '.join(row['signals'])}")
                if row['timeframes']:
                    embedding_parts.append(f"Timeframes: {', '.join(row['timeframes'])}")

                complexity_level = self._classify_complexity(row['complexity_score'])
                embedding_parts.append(f"Complexity: {complexity_level}")

                market_context = self._infer_market_context(dict(row))
                if market_context:
                    embedding_parts.append(f"Best for: {market_context}")

                embedding_text = " | ".join(embedding_parts)

                enriched = {
                    'id': row['slug'],
                    'title': row['title'],
                    'author': row['author'],
                    'collection': row['collection'],
                    'category': row['category'],
                    'technical_focus': row['technical_focus'],
                    'embedding_text': embedding_text,
                    'metadata': {
                        'title': row['title'],
                        'author': row['author'],
                        'collection': row['collection'],
                        'category': row['category'] or '',
                        'technical_focus': row['technical_focus'] or '',
                        'complexity_level': complexity_level,
                        'complexity_score': float(row['complexity_score']),
                        'strategy_type': row['strategy_type'],
                        'timeframes': ','.join(row['timeframes']) if row['timeframes'] else '',
                        'indicators': ','.join(row['indicators']) if row['indicators'] else '',
                        'signals': ','.join(row['signals']) if row['signals'] else '',
                        'market_context': market_context,
                        'likes': row['likes'],
                        'views': row['views'],
                        'open_source': row['open_source']
                    }
                }
                enriched_data.append(enriched)

            connection.close()
            logger.info(f"✅ Fetched {len(enriched_data)} TradingView indicators")
            return enriched_data

        except Exception as e:
            logger.error(f"Error fetching TradingView data: {e}")
            return []

    async def fetch_optimization_templates(self) -> List[Dict]:
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
                    best_ema_config as config_preset,
                    best_confidence_threshold,
                    best_timeframe,
                    optimal_stop_loss_pips,
                    optimal_take_profit_pips,
                    best_win_rate,
                    best_profit_factor,
                    best_net_pips
                FROM ema_best_parameters
                WHERE best_net_pips > 50
                ORDER BY best_net_pips DESC
                LIMIT 20
            """)

            ema_templates = cursor.fetchall()
            for template in ema_templates:
                enriched = self._create_strategy_template(dict(template))
                templates.append(enriched)

            # Fetch MACD templates
            cursor.execute("""
                SELECT
                    'MACD' as strategy_type,
                    epic,
                    best_fast_ema,
                    best_slow_ema,
                    best_signal_ema,
                    best_confidence_threshold,
                    best_timeframe,
                    optimal_stop_loss_pips,
                    optimal_take_profit_pips,
                    best_win_rate,
                    best_composite_score as performance_score
                FROM macd_best_parameters
                WHERE best_composite_score > 1.0
                ORDER BY best_composite_score DESC
                LIMIT 15
            """)

            macd_templates = cursor.fetchall()
            for template in macd_templates:
                enriched = self._create_macd_template(dict(template))
                templates.append(enriched)

            connection.close()
            logger.info(f"✅ Fetched {len(templates)} optimization templates")
            return templates

        except Exception as e:
            logger.error(f"Error fetching optimization templates: {e}")
            return []

    def semantic_search_indicators(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Perform semantic search on indicators"""
        try:
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ['collection', 'category', 'technical_focus', 'complexity_level']:
                        where_clause[key] = value

            results = self.indicators_collection.query(
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

    def search_strategy_templates(self, query: str, n_results: int = 3, filters: Optional[Dict] = None) -> List[Dict]:
        """Search strategy templates with semantic understanding"""
        try:
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ['strategy_type', 'epic_type', 'timeframe']:
                        where_clause[key] = value

            results = self.templates_collection.query(
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
                    'similarity_score': 1 - results['distances'][0][i],
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Error searching strategy templates: {e}")
            return []

    def get_similar_indicators(self, indicator_id: str, n_results: int = 3) -> List[Dict]:
        """Find indicators similar to a given indicator"""
        try:
            # Get the original indicator
            original = self.indicators_collection.get(ids=[indicator_id])
            if not original['documents']:
                return []

            # Search for similar indicators
            results = self.indicators_collection.query(
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
            logger.error(f"❌ Error finding similar indicators: {e}")
            return []

    def compose_trading_strategy(self, description: str, market_condition: str,
                               trading_style: str, complexity_level: str) -> StrategyComposition:
        """AI-powered strategy composition"""
        # This would implement the intelligent strategy composition logic
        return StrategyComposition(
            primary_indicators=[],
            confirmation_indicators=[],
            filter_indicators=[],
            strategy_template=None,
            reasoning="Strategy composition not yet implemented"
        )

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            return {
                "indicators_count": self.indicators_collection.count() if self.indicators_collection else 0,
                "templates_count": self.templates_collection.count() if self.templates_collection else 0,
                "taxonomy_count": self.taxonomy_collection.count() if self.taxonomy_collection else 0,
                "data_loaded": self.data_loaded,
                "last_sync": self.last_sync
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    async def populate_indicators_collection(self, indicators_data: List[Dict]):
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
                    # Get all existing IDs and delete them
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
            raise

    async def populate_templates_collection(self, templates_data: List[Dict]):
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
                    # Get all existing IDs and delete them
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
            raise

    # Memory and Pattern Recognition Methods
    def add_memory_event(self, event_type: str, query: str, results_count: int,
                        user_context: Optional[Dict] = None, patterns: Optional[List[str]] = None):
        """Add a memory event for tracking user interactions and patterns"""
        event = MemoryEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            query=query,
            results_count=results_count,
            user_context=user_context,
            patterns_identified=patterns
        )
        self.memory_events.append(event)

        # Keep only last 1000 events to prevent memory bloat
        if len(self.memory_events) > 1000:
            self.memory_events = self.memory_events[-1000:]

        # Update search history
        self.search_history.append({
            'timestamp': event.timestamp.isoformat(),
            'query': query,
            'type': event_type,
            'results': results_count
        })

        logger.info(f"📝 Memory event added: {event_type} - '{query}' ({results_count} results)")

    def detect_search_patterns(self) -> List[PatternInsight]:
        """Analyze search history to detect patterns and provide insights"""
        patterns = []

        if len(self.memory_events) < 5:
            return patterns

        # Analyze recent searches for trends
        recent_events = self.memory_events[-20:]  # Last 20 events

        # Pattern 1: Frequent strategy type searches
        strategy_queries = [e for e in recent_events if 'strategy' in e.query.lower()]
        if len(strategy_queries) >= 3:
            strategy_types = [self._extract_strategy_keywords(e.query) for e in strategy_queries]
            most_common = max(set(strategy_types), key=strategy_types.count) if strategy_types else None

            if most_common:
                patterns.append(PatternInsight(
                    pattern_type="frequent_strategy_search",
                    confidence=0.8,
                    description=f"User frequently searches for {most_common} strategies",
                    recommendations=[
                        f"Consider bookmarking top {most_common} indicators",
                        f"Explore advanced {most_common} configurations",
                        f"Review historical performance of {most_common} strategies"
                    ]
                ))

        # Pattern 2: Market condition focus
        market_keywords = ['trending', 'ranging', 'volatile', 'breakout', 'reversal']
        market_mentions = {}
        for event in recent_events:
            for keyword in market_keywords:
                if keyword in event.query.lower():
                    market_mentions[keyword] = market_mentions.get(keyword, 0) + 1

        if market_mentions:
            top_market = max(market_mentions, key=market_mentions.get)
            patterns.append(PatternInsight(
                pattern_type="market_condition_focus",
                confidence=0.7,
                description=f"User shows interest in {top_market} market conditions",
                recommendations=[
                    f"Suggest indicators optimized for {top_market} markets",
                    f"Show strategy templates with strong {top_market} performance",
                    f"Alert about {top_market} market opportunities"
                ]
            ))

        # Pattern 3: Complexity preference
        complexity_searches = [e for e in recent_events if any(word in e.query.lower()
                             for word in ['simple', 'advanced', 'complex', 'basic', 'expert'])]
        if complexity_searches:
            patterns.append(PatternInsight(
                pattern_type="complexity_preference",
                confidence=0.6,
                description="User has specific complexity preferences",
                recommendations=[
                    "Filter results by preferred complexity level",
                    "Suggest learning path for skill progression",
                    "Recommend appropriate strategy templates"
                ]
            ))

        return patterns

    def get_personalized_recommendations(self, query: str) -> List[str]:
        """Generate personalized recommendations based on memory and patterns"""
        recommendations = []

        # Analyze user's search history for preferences
        recent_queries = [e.query for e in self.memory_events[-10:]]

        # Find common themes
        common_words = {}
        for q in recent_queries:
            words = q.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    common_words[word] = common_words.get(word, 0) + 1

        # Generate recommendations based on patterns
        if common_words:
            top_interest = max(common_words, key=common_words.get)
            recommendations.extend([
                f"Explore more {top_interest}-related indicators",
                f"Check strategy templates featuring {top_interest}",
                f"Review similar indicators to your {top_interest} searches"
            ])

        # Add contextual recommendations based on current query
        if 'momentum' in query.lower():
            recommendations.append("Consider combining with trend confirmation indicators")
        elif 'trend' in query.lower():
            recommendations.append("Add momentum indicators for entry timing")
        elif 'volume' in query.lower():
            recommendations.append("Combine with price action indicators for validation")

        return recommendations[:5]  # Limit to top 5 recommendations

    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences based on interactions"""
        self.user_preferences.update(preferences)
        logger.info(f"👤 User preferences updated: {preferences}")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of memory events and patterns"""
        if not self.memory_events:
            return {"message": "No memory events recorded yet"}

        total_events = len(self.memory_events)
        recent_events = self.memory_events[-10:] if total_events >= 10 else self.memory_events

        # Count event types
        event_types = {}
        for event in self.memory_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

        # Get popular queries
        query_counts = {}
        for event in self.memory_events:
            query_counts[event.query] = query_counts.get(event.query, 0) + 1

        popular_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_events": total_events,
            "event_types": event_types,
            "popular_queries": popular_queries,
            "recent_activity": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.event_type,
                    "query": e.query,
                    "results": e.results_count
                } for e in recent_events
            ],
            "detected_patterns": len(self.pattern_cache),
            "user_preferences": self.user_preferences
        }

    def _extract_strategy_keywords(self, query: str) -> str:
        """Extract strategy-related keywords from query"""
        strategy_types = ['ema', 'macd', 'rsi', 'bollinger', 'momentum', 'trend', 'oscillator']
        query_lower = query.lower()

        for strategy in strategy_types:
            if strategy in query_lower:
                return strategy

        return "general"

    async def semantic_search_indicators(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Perform semantic search on indicators collection (async wrapper)"""
        if not hasattr(self, 'indicators_collection') or self.indicators_collection is None:
            raise Exception("Indicators collection not initialized")

        try:
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ['collection', 'category', 'technical_focus', 'complexity_level']:
                        where_clause[key] = value

            # Perform search
            results = self.indicators_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'relevance': 'high' if (1 - results['distances'][0][i]) > 0.8 else 'medium' if (1 - results['distances'][0][i]) > 0.6 else 'low'
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Error in semantic search: {e}")
            return []

    async def search_strategy_templates(self, query: str, n_results: int = 3, filters: Optional[Dict] = None) -> List[Dict]:
        """Search strategy templates with semantic understanding (async wrapper)"""
        if not hasattr(self, 'templates_collection') or self.templates_collection is None:
            raise Exception("Templates collection not initialized")

        try:
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ['strategy_type', 'epic_type', 'timeframe']:
                        where_clause[key] = value

            results = self.templates_collection.query(
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
                    'similarity_score': 1 - results['distances'][0][i],
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Error searching strategy templates: {e}")
            return []

    async def get_similar_indicators(self, indicator_id: str, n_results: int = 3) -> List[Dict]:
        """Find indicators similar to a given indicator (async wrapper)"""
        if not hasattr(self, 'indicators_collection') or self.indicators_collection is None:
            raise Exception("Indicators collection not initialized")

        try:
            # Get the original indicator
            original = self.indicators_collection.get(ids=[indicator_id])
            if not original['documents']:
                return []

            # Search for similar indicators
            results = self.indicators_collection.query(
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
            logger.error(f"❌ Error finding similar indicators: {e}")
            return []

    def _classify_complexity(self, score: float) -> str:
        """Classify complexity score into human-readable levels"""
        if score >= 0.9:
            return "Expert"
        elif score >= 0.8:
            return "Advanced"
        elif score >= 0.6:
            return "Intermediate"
        else:
            return "Basic"

    def _infer_market_context(self, indicator: Dict) -> str:
        """Infer market context and best use cases from indicator metadata"""
        context_hints = []

        # Analyze category for market context
        category = indicator.get('category', '') or ''
        category = category.lower()
        if 'trend' in category:
            context_hints.append("trending markets")
        elif 'oscillator' in category or 'momentum' in category:
            context_hints.append("ranging markets")
        elif 'volatility' in category or 'breakout' in category:
            context_hints.append("volatile markets")
        elif 'volume' in category:
            context_hints.append("high volume periods")

        # Analyze technical focus
        focus = indicator.get('technical_focus', '') or ''
        focus = focus.lower()
        if 'scalping' in focus:
            context_hints.append("scalping strategies")
        elif 'institutional' in focus or 'whale' in focus:
            context_hints.append("institutional analysis")

        # Analyze complexity for user recommendations
        complexity = indicator.get('complexity_score', 0.5)
        if complexity >= 0.85:
            context_hints.append("experienced traders")
        elif complexity <= 0.6:
            context_hints.append("beginner-friendly")

        return ", ".join(context_hints) if context_hints else "general trading"

    def _create_strategy_template(self, template: Dict) -> Dict:
        """Create enriched strategy template for RAG"""
        epic_type = self._classify_epic_type(template['epic'])

        description_parts = [
            f"{template['strategy_type']} strategy for {template['epic']}",
            f"Configuration: {template['config_preset']}",
            f"Timeframe: {template['best_timeframe']}",
            f"Confidence threshold: {template['best_confidence_threshold']:.1%}",
            f"Risk management: {template['optimal_stop_loss_pips']:.0f} SL / {template['optimal_take_profit_pips']:.0f} TP",
            f"Performance: {template.get('best_win_rate', 0):.1%} win rate",
            f"Best for: {epic_type} pairs"
        ]

        template_text = " | ".join(description_parts)

        return {
            'id': f"ema_{template['epic'].replace('.', '_')}",
            'embedding_text': template_text,
            'metadata': {
                'strategy_type': template['strategy_type'],
                'epic': template['epic'],
                'epic_type': epic_type,
                'config_preset': template['config_preset'],
                'confidence_threshold': float(template['best_confidence_threshold']),
                'timeframe': template['best_timeframe'],
                'stop_loss_pips': float(template['optimal_stop_loss_pips']),
                'take_profit_pips': float(template['optimal_take_profit_pips']),
                'win_rate': float(template.get('best_win_rate', 0)),
                'profit_factor': float(template.get('best_profit_factor', 0)),
                'net_pips': float(template.get('best_net_pips', 0))
            }
        }

    def _create_macd_template(self, template: Dict) -> Dict:
        """Create MACD-specific strategy template"""
        epic_type = self._classify_epic_type(template['epic'])

        description_parts = [
            f"MACD strategy for {template['epic']}",
            f"Parameters: {template['best_fast_ema']}/{template['best_slow_ema']}/{template['best_signal_ema']}",
            f"Timeframe: {template['best_timeframe']}",
            f"Confidence: {template['best_confidence_threshold']:.1%}",
            f"Risk: {template['optimal_stop_loss_pips']:.0f} SL / {template['optimal_take_profit_pips']:.0f} TP",
            f"Performance score: {template['performance_score']:.3f}",
            f"Optimized for: {epic_type} pairs"
        ]

        template_text = " | ".join(description_parts)

        return {
            'id': f"macd_{template['epic'].replace('.', '_')}",
            'embedding_text': template_text,
            'metadata': {
                'strategy_type': 'MACD',
                'epic': template['epic'],
                'epic_type': epic_type,
                'fast_ema': int(template['best_fast_ema']),
                'slow_ema': int(template['best_slow_ema']),
                'signal_ema': int(template['best_signal_ema']),
                'confidence_threshold': float(template['best_confidence_threshold']),
                'timeframe': template['best_timeframe'],
                'stop_loss_pips': float(template['optimal_stop_loss_pips']),
                'take_profit_pips': float(template['optimal_take_profit_pips']),
                'win_rate': float(template.get('best_win_rate', 0)),
                'performance_score': float(template['performance_score'])
            }
        }

    def _classify_epic_type(self, epic: str) -> str:
        """Classify epic into major/minor/exotic currency types"""
        if 'EURUSD' in epic or 'GBPUSD' in epic or 'USDJPY' in epic or 'USDCHF' in epic:
            return "Major"
        elif 'EURJPY' in epic or 'GBPJPY' in epic or 'AUDUSD' in epic or 'NZDUSD' in epic:
            return "Minor"
        else:
            return "Exotic"

# Global service instance
vector_service = VectorDatabaseService()

# Memory-Enhanced API Endpoints
@vector_service.app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vector-db", "timestamp": datetime.now().isoformat()}

@vector_service.app.post("/search/indicators", response_model=List[SearchResult])
async def search_indicators(request: SearchRequest):
    """Search indicators with memory tracking"""
    try:
        # Perform semantic search
        results = await vector_service.semantic_search_indicators(
            query=request.query,
            n_results=request.n_results,
            filters=request.filters
        )

        # Track memory event
        vector_service.add_memory_event(
            event_type="search_indicators",
            query=request.query,
            results_count=len(results),
            user_context={"filters": request.filters}
        )

        # Format results
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=result['id'],
                title=result['metadata'].get('title', 'Unknown'),
                similarity_score=result['similarity_score'],
                metadata=result['metadata'],
                content_preview=result['document'][:200] + "..." if len(result['document']) > 200 else result['document']
            ))

        return search_results

    except Exception as e:
        logger.error(f"❌ Error in indicator search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_service.app.post("/search/templates", response_model=List[SearchResult])
async def search_templates(request: SearchRequest):
    """Search strategy templates with memory tracking"""
    try:
        results = await vector_service.search_strategy_templates(
            query=request.query,
            n_results=request.n_results,
            filters=request.filters
        )

        # Track memory event
        vector_service.add_memory_event(
            event_type="search_templates",
            query=request.query,
            results_count=len(results),
            user_context={"filters": request.filters}
        )

        # Format results
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=result['id'],
                title=f"{result['metadata'].get('strategy_type', 'Unknown')} - {result['metadata'].get('epic', 'Unknown')}",
                similarity_score=result['similarity_score'],
                metadata=result['metadata'],
                content_preview=result['document'][:200] + "..." if len(result['document']) > 200 else result['document']
            ))

        return search_results

    except Exception as e:
        logger.error(f"❌ Error in template search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_service.app.get("/similar/{indicator_id}", response_model=List[SearchResult])
async def find_similar_indicators(indicator_id: str, n_results: int = Query(default=3, ge=1, le=10)):
    """Find similar indicators with memory tracking"""
    try:
        results = await vector_service.get_similar_indicators(indicator_id, n_results)

        # Track memory event
        vector_service.add_memory_event(
            event_type="similarity_search",
            query=f"similar_to:{indicator_id}",
            results_count=len(results),
            user_context={"reference_indicator": indicator_id}
        )

        # Format results
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=result['id'],
                title=result['metadata'].get('title', 'Unknown'),
                similarity_score=result['similarity_score'],
                metadata=result['metadata'],
                content_preview=result['document'][:200] + "..." if len(result['document']) > 200 else result['document']
            ))

        return search_results

    except Exception as e:
        logger.error(f"❌ Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_service.app.post("/compose/strategy", response_model=StrategyComposition)
async def compose_strategy(request: StrategyCompositionRequest):
    """AI-powered strategy composition with memory and pattern recognition"""
    try:
        # Generate personalized recommendations
        recommendations = vector_service.get_personalized_recommendations(request.description)

        # Search for primary indicators
        primary_results = await vector_service.semantic_search_indicators(
            query=f"{request.description} primary trend momentum",
            n_results=3,
            filters={"complexity_level": request.complexity_level} if request.complexity_level != "intermediate" else None
        )

        # Search for confirmation indicators
        confirmation_results = await vector_service.semantic_search_indicators(
            query=f"{request.description} confirmation signal",
            n_results=2,
            filters={"complexity_level": request.complexity_level} if request.complexity_level != "intermediate" else None
        )

        # Search for filter indicators
        filter_results = await vector_service.semantic_search_indicators(
            query=f"{request.description} filter volatility risk",
            n_results=2,
            filters={"complexity_level": request.complexity_level} if request.complexity_level != "intermediate" else None
        )

        # Find relevant strategy template
        template_results = await vector_service.search_strategy_templates(
            query=f"{request.description} {request.market_condition} {request.trading_style}",
            n_results=1
        )

        # Detect patterns and update composition tracking
        patterns = vector_service.detect_search_patterns()
        composition_key = f"{request.market_condition}_{request.trading_style}_{request.complexity_level}"
        vector_service.composition_patterns[composition_key] = vector_service.composition_patterns.get(composition_key, 0) + 1

        # Track comprehensive memory event
        vector_service.add_memory_event(
            event_type="strategy_composition",
            query=request.description,
            results_count=len(primary_results) + len(confirmation_results) + len(filter_results),
            user_context={
                "market_condition": request.market_condition,
                "trading_style": request.trading_style,
                "complexity_level": request.complexity_level
            },
            patterns=[p.pattern_type for p in patterns]
        )

        # Build reasoning with pattern insights
        reasoning_parts = [
            f"Strategy composition for {request.market_condition} markets using {request.trading_style} approach.",
            f"Selected {len(primary_results)} primary indicators for trend identification and momentum analysis.",
            f"Added {len(confirmation_results)} confirmation indicators to validate signals.",
            f"Included {len(filter_results)} filter indicators for risk management."
        ]

        if patterns:
            reasoning_parts.append(f"Based on your search patterns, we've optimized for {patterns[0].description.lower()}.")

        if recommendations:
            reasoning_parts.append(f"Personalized recommendation: {recommendations[0]}")

        # Convert results to SearchResult format
        def convert_to_search_result(results):
            return [SearchResult(
                id=r['id'],
                title=r['metadata'].get('title', r['metadata'].get('strategy_type', 'Unknown')),
                similarity_score=r['similarity_score'],
                metadata=r['metadata'],
                content_preview=r['document'][:200] + "..." if len(r['document']) > 200 else r['document']
            ) for r in results]

        composition = StrategyComposition(
            primary_indicators=convert_to_search_result(primary_results),
            confirmation_indicators=convert_to_search_result(confirmation_results),
            filter_indicators=convert_to_search_result(filter_results),
            strategy_template=convert_to_search_result(template_results)[0] if template_results else None,
            reasoning=" ".join(reasoning_parts)
        )

        return composition

    except Exception as e:
        logger.error(f"❌ Error in strategy composition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_service.app.get("/memory/summary")
async def get_memory_summary():
    """Get memory and pattern analysis summary"""
    try:
        summary = vector_service.get_memory_summary()
        patterns = vector_service.detect_search_patterns()

        return {
            "memory_summary": summary,
            "detected_patterns": [
                {
                    "type": p.pattern_type,
                    "confidence": p.confidence,
                    "description": p.description,
                    "recommendations": p.recommendations
                } for p in patterns
            ],
            "composition_trends": vector_service.composition_patterns
        }

    except Exception as e:
        logger.error(f"❌ Error getting memory summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_service.app.post("/memory/preferences")
async def update_preferences(preferences: Dict[str, Any]):
    """Update user preferences"""
    try:
        vector_service.update_user_preferences(preferences)
        return {"status": "success", "message": "Preferences updated", "preferences": preferences}

    except Exception as e:
        logger.error(f"❌ Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_service.app.get("/patterns/insights", response_model=List[PatternInsight])
async def get_pattern_insights():
    """Get detailed pattern insights and recommendations"""
    try:
        patterns = vector_service.detect_search_patterns()
        return patterns

    except Exception as e:
        logger.error(f"❌ Error getting pattern insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_service.app.get("/recommendations/{query}")
async def get_personalized_recommendations_endpoint(query: str):
    """Get personalized recommendations for a query"""
    try:
        recommendations = vector_service.get_personalized_recommendations(query)
        return {"query": query, "recommendations": recommendations}

    except Exception as e:
        logger.error(f"❌ Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI app instance
app = vector_service.app

@app.on_event("startup")
async def startup_event():
    """Initialize vector database on startup"""
    await vector_service.initialize_vector_db()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔌 Vector Database Service shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "vector_db_service:app",
        host="0.0.0.0",
        port=8090,
        reload=False,
        log_level="info"
    )