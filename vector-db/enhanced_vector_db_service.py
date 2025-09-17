#!/usr/bin/env python3
"""
Enhanced Vector Database Service with Advanced RAG Capabilities
==============================================================

This is the upgraded version of the vector database service that integrates:
- Enhanced Pine Script parsing and semantic analysis
- Multi-modal embeddings with performance data
- Intelligent query processing with concept normalization
- Performance-aware recommendations
- Advanced strategy composition engine
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
import psycopg2
from psycopg2.extras import RealDictCursor

# Import our enhanced components
from enhanced_rag_embeddings import (
    EnhancedRAGEmbeddings,
    EnhancedEmbeddingInput,
    TradingConceptNormalizer
)
from intelligent_query_processor import (
    IntelligentQueryProcessor,
    ProcessedQuery,
    QueryIntent
)
from performance_aware_rag import (
    PerformanceAwareRAG,
    PerformanceWeightedRecommendation,
    MarketRegime
)
from strategy_composition_engine import (
    StrategyArchitect,
    IndicatorCompatibilityAnalyzer,
    StrategyComposition
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/enhanced_vector_db.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enhanced Pydantic models
class EnhancedSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results")
    filters: Optional[Dict[str, str]] = Field(default=None, description="Filters")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="User context")
    include_performance: bool = Field(default=True, description="Include performance weighting")

class EnhancedSearchResult(BaseModel):
    id: str
    title: str
    similarity_score: float
    performance_weight: float
    final_score: float
    metadata: Dict[str, Any]
    content_preview: str
    recommendation_reason: Optional[str] = None
    risk_assessment: Optional[str] = None
    performance_context: Optional[Dict[str, Any]] = None

class StrategyCompositionRequest(BaseModel):
    description: str = Field(..., description="Strategy requirements description")
    market_condition: Optional[str] = Field(default="trending", description="Market condition")
    trading_style: Optional[str] = Field(default="day_trading", description="Trading style")
    risk_tolerance: Optional[str] = Field(default="medium", description="Risk tolerance")
    complexity_preference: Optional[str] = Field(default="intermediate", description="Complexity")
    available_indicators: Optional[List[str]] = Field(default=None, description="Available indicators")

class QueryAnalysisResponse(BaseModel):
    original_query: str
    normalized_query: str
    intents: List[Dict[str, Any]]
    expanded_terms: List[str]
    suggested_filters: Dict[str, Any]
    query_complexity: str
    processing_suggestions: List[str]

class MarketRegimeResponse(BaseModel):
    regime_type: str
    volatility_level: str
    trend_strength: float
    confidence: float
    recommendations: List[str]

class EnhancedVectorDatabaseService:
    """Enhanced vector database service with advanced RAG capabilities"""

    def __init__(self):
        """Initialize the enhanced vector database service"""
        self.app = FastAPI(
            title="Enhanced TradingView Vector Database API",
            description="Advanced RAG system for trading indicators and strategies",
            version="2.0.0"
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

        # Initialize enhanced components
        self.query_processor = IntelligentQueryProcessor()
        self.embeddings_engine = EnhancedRAGEmbeddings(self.db_url)
        self.performance_rag = PerformanceAwareRAG(self.db_url)

        # Strategy composition components
        self.compatibility_analyzer = IndicatorCompatibilityAnalyzer()
        self.strategy_architect = StrategyArchitect(self.compatibility_analyzer)

        # ChromaDB client
        self.client = None
        self.indicators_collection = None
        self.enhanced_collection = None

        # Data status
        self.enhanced_data_loaded = False
        self.last_enhancement_sync = None

        # Setup routes
        self.setup_enhanced_routes()

    def setup_enhanced_routes(self):
        """Setup enhanced FastAPI routes"""

        @self.app.get("/health")
        async def enhanced_health_check():
            """Enhanced health check endpoint"""
            return {
                "status": "healthy",
                "service": "enhanced-vector-db",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "enhanced_features": {
                    "pine_script_parsing": True,
                    "multi_modal_embeddings": True,
                    "intelligent_query_processing": True,
                    "performance_aware_recommendations": True,
                    "strategy_composition": True
                },
                "data_status": {
                    "enhanced_data_loaded": self.enhanced_data_loaded,
                    "last_enhancement_sync": self.last_enhancement_sync
                }
            }

        @self.app.post("/search/enhanced", response_model=List[EnhancedSearchResult])
        async def enhanced_search(request: EnhancedSearchRequest):
            """Enhanced semantic search with intelligence and performance weighting"""
            try:
                # Process query with intelligence
                processed_query = self.query_processor.process_query(
                    request.query,
                    request.user_context
                )

                # Create search variants for better recall
                search_variants = self.query_processor.create_search_variants(processed_query)

                # Perform multi-variant search
                all_results = []
                for variant in search_variants[:3]:  # Use top 3 variants
                    variant_results = await self._semantic_search_enhanced(
                        variant,
                        request.n_results,
                        request.filters
                    )
                    all_results.extend(variant_results)

                # Remove duplicates and limit results
                unique_results = self._deduplicate_results(all_results)[:request.n_results * 2]

                # Apply performance weighting if requested
                if request.include_performance and unique_results:
                    performance_weighted = self.performance_rag.get_performance_weighted_recommendations(
                        request.query,
                        unique_results,
                        user_context=request.user_context
                    )

                    # Convert to enhanced search results
                    enhanced_results = []
                    for rec in performance_weighted[:request.n_results]:
                        enhanced_results.append(EnhancedSearchResult(
                            id=rec.indicator_id,
                            title=rec.title,
                            similarity_score=rec.base_score,
                            performance_weight=rec.performance_score,
                            final_score=rec.final_score,
                            metadata={},  # Would be populated from original result
                            content_preview="",  # Would be populated
                            recommendation_reason=rec.recommendation_reason,
                            risk_assessment=rec.risk_assessment,
                            performance_context=rec.performance_context
                        ))

                    return enhanced_results

                else:
                    # Return basic enhanced results
                    enhanced_results = []
                    for result in unique_results[:request.n_results]:
                        enhanced_results.append(EnhancedSearchResult(
                            id=result.get('id', ''),
                            title=result.get('metadata', {}).get('title', 'Unknown'),
                            similarity_score=result.get('similarity_score', 0.0),
                            performance_weight=0.5,  # Default
                            final_score=result.get('similarity_score', 0.0),
                            metadata=result.get('metadata', {}),
                            content_preview=result.get('document', '')[:200] + "...",
                            recommendation_reason="Matches your search criteria",
                            risk_assessment="Standard risk profile"
                        ))

                    return enhanced_results

            except Exception as e:
                logger.error(f"Enhanced search failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/query/analyze", response_model=QueryAnalysisResponse)
        async def analyze_query(query: str):
            """Analyze and process a natural language query"""
            try:
                processed = self.query_processor.process_query(query)

                return QueryAnalysisResponse(
                    original_query=processed.original_query,
                    normalized_query=processed.normalized_query,
                    intents=[{
                        "type": intent.intent_type,
                        "confidence": intent.confidence,
                        "parameters": intent.parameters,
                        "entities": intent.entities
                    } for intent in processed.intents],
                    expanded_terms=processed.expanded_terms,
                    suggested_filters=processed.filters,
                    query_complexity=processed.complexity_level,
                    processing_suggestions=self._generate_processing_suggestions(processed)
                )

            except Exception as e:
                logger.error(f"Query analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/market/regime", response_model=MarketRegimeResponse)
        async def get_market_regime(epic: str = "CS.D.EURUSD.MINI.IP"):
            """Get current market regime analysis"""
            try:
                regime = self.performance_rag.regime_detector.detect_current_regime(epic)

                recommendations = self._generate_regime_recommendations(regime)

                return MarketRegimeResponse(
                    regime_type=regime.regime_type,
                    volatility_level=regime.volatility_level,
                    trend_strength=regime.trend_strength,
                    confidence=regime.confidence,
                    recommendations=recommendations
                )

            except Exception as e:
                logger.error(f"Market regime detection failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/strategy/compose")
        async def compose_strategy(request: StrategyCompositionRequest):
            """Compose a trading strategy using AI and compatibility analysis"""
            try:
                # Get available indicators from the database
                if request.available_indicators:
                    available_indicators = request.available_indicators
                else:
                    available_indicators = await self._get_available_indicators()

                # Define strategy requirements
                requirements = {
                    'market_condition': request.market_condition,
                    'trading_style': request.trading_style,
                    'risk_tolerance': request.risk_tolerance,
                    'complexity': request.complexity_preference
                }

                # Design the strategy
                strategy = self.strategy_architect.design_strategy(
                    requirements,
                    available_indicators
                )

                return {
                    "strategy_id": strategy.strategy_id,
                    "name": strategy.name,
                    "description": strategy.description,
                    "compatibility_score": strategy.compatibility_score,
                    "risk_score": strategy.risk_score,
                    "complexity_score": strategy.complexity_score,
                    "components": [
                        {
                            "indicator": comp.indicator.name,
                            "role": comp.role.value,
                            "weight": comp.weight,
                            "parameters": comp.parameters,
                            "entry_conditions": comp.entry_conditions,
                            "exit_conditions": comp.exit_conditions
                        }
                        for comp in strategy.components
                    ],
                    "expected_performance": strategy.expected_performance,
                    "market_suitability": strategy.market_suitability,
                    "conflicts_detected": strategy.conflicts_detected,
                    "optimization_suggestions": strategy.optimization_suggestions,
                    "backtest_recommendations": strategy.backtest_recommendations
                }

            except Exception as e:
                logger.error(f"Strategy composition failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/indicators/compatibility")
        async def analyze_compatibility(indicators: List[str]):
            """Analyze compatibility between multiple indicators"""
            try:
                analysis = self.compatibility_analyzer.analyze_indicator_set(indicators)

                return {
                    "indicators": indicators,
                    "overall_compatibility": analysis['overall_compatibility'],
                    "category_balance": analysis['category_balance'],
                    "conflicts": analysis['conflicts'],
                    "synergies": analysis['synergies'],
                    "recommendations": analysis['recommendations']
                }

            except Exception as e:
                logger.error(f"Compatibility analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/data/enhance")
        async def enhance_database(background_tasks: BackgroundTasks):
            """Trigger enhanced data processing and embedding generation"""
            background_tasks.add_task(self._enhance_vector_database)
            return {
                "message": "Enhanced data processing started",
                "timestamp": datetime.now().isoformat()
            }

        @self.app.get("/stats/enhanced")
        async def get_enhanced_stats():
            """Get enhanced database statistics"""
            try:
                stats = await self._get_enhanced_database_stats()
                return stats
            except Exception as e:
                logger.error(f"Error getting enhanced stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def initialize_enhanced_vector_db(self):
        """Initialize enhanced vector database"""
        try:
            logger.info("🚀 Initializing Enhanced Vector Database Service...")

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

            # Create enhanced collection
            self.enhanced_collection = self.client.get_or_create_collection(
                name="enhanced_tradingview_indicators",
                embedding_function=embedding_function,
                metadata={"description": "Enhanced TradingView indicators with multi-modal embeddings"}
            )

            # Check if enhanced data needs to be loaded
            if self.enhanced_collection.count() == 0:
                logger.info("📊 Enhanced collection empty, triggering data enhancement...")
                await self._enhance_vector_database()
            else:
                logger.info(f"✅ Enhanced collection loaded: {self.enhanced_collection.count()} indicators")
                self.enhanced_data_loaded = True

            logger.info("✅ Enhanced Vector Database Service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced vector database: {e}")
            raise

    async def _semantic_search_enhanced(self, query: str, n_results: int, filters: Optional[Dict] = None) -> List[Dict]:
        """Perform enhanced semantic search"""
        if not self.enhanced_collection:
            return []

        try:
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ['collection', 'category', 'complexity_level', 'trading_style']:
                        where_clause[key] = value

            results = self.enhanced_collection.query(
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
            logger.error(f"Enhanced semantic search failed: {e}")
            return []

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on ID"""
        seen_ids = set()
        unique_results = []

        for result in results:
            result_id = result.get('id', '')
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)

        return unique_results

    def _generate_processing_suggestions(self, processed_query: ProcessedQuery) -> List[str]:
        """Generate suggestions for improving query processing"""
        suggestions = []

        if processed_query.complexity_level == 'basic':
            suggestions.append("Try adding more specific terms for better results")
        elif processed_query.complexity_level == 'advanced':
            suggestions.append("Consider breaking down complex queries into simpler parts")

        if not processed_query.filters:
            suggestions.append("Add filters like timeframe or trading style for more targeted results")

        if len(processed_query.expanded_terms) < 3:
            suggestions.append("Try using more descriptive terms related to your trading needs")

        return suggestions

    def _generate_regime_recommendations(self, regime) -> List[str]:
        """Generate recommendations based on market regime"""
        recommendations = []

        if regime.regime_type == "trending":
            recommendations.extend([
                "Consider trend-following indicators like EMA and MACD",
                "Use momentum indicators for entry timing",
                "Avoid mean-reversion strategies"
            ])
        elif regime.regime_type == "ranging":
            recommendations.extend([
                "Focus on oscillators like RSI and Stochastic",
                "Use support/resistance levels",
                "Consider mean-reversion strategies"
            ])
        elif regime.regime_type == "volatile":
            recommendations.extend([
                "Use volatility indicators like Bollinger Bands",
                "Implement wider stop losses",
                "Consider breakout strategies"
            ])

        return recommendations

    async def _get_available_indicators(self) -> List[str]:
        """Get list of available indicators from database"""
        try:
            connection = psycopg2.connect(self.db_url)
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT DISTINCT unnest(indicators) as indicator
                FROM tradingview.scripts
                WHERE indicators IS NOT NULL
                ORDER BY indicator
            """)

            results = cursor.fetchall()
            connection.close()

            return [row['indicator'] for row in results if row['indicator']]

        except Exception as e:
            logger.error(f"Error getting available indicators: {e}")
            return ['moving_average', 'rsi', 'macd', 'bollinger_bands', 'atr']  # Fallback

    async def _enhance_vector_database(self):
        """Enhance the vector database with advanced processing"""
        try:
            logger.info("🔄 Starting enhanced vector database processing...")

            # Fetch TradingView scripts from database
            scripts_data = await self._fetch_enhanced_scripts_data()

            if scripts_data:
                # Process with enhanced embeddings
                enhanced_embeddings = []

                for script in scripts_data:
                    # Create enhanced embedding input
                    embedding_input = EnhancedEmbeddingInput(
                        script_id=script['slug'],
                        title=script['title'],
                        author=script['author'],
                        description=script.get('description', ''),
                        code=script.get('code'),
                        collection=script.get('collection', ''),
                        category=script.get('category', ''),
                        complexity_score=script.get('complexity_score', 0.5),
                        indicators=script.get('indicators', []),
                        signals=script.get('signals', []),
                        timeframes=script.get('timeframes', [])
                    )

                    # Generate enhanced embedding
                    try:
                        embedding_result = self.embeddings_engine.create_enhanced_embedding(embedding_input)
                        enhanced_embeddings.append(embedding_result)
                    except Exception as e:
                        logger.warning(f"Failed to create embedding for {script['slug']}: {e}")

                # Populate enhanced collection
                if enhanced_embeddings:
                    await self._populate_enhanced_collection(enhanced_embeddings)

                # Update status
                self.enhanced_data_loaded = True
                self.last_enhancement_sync = datetime.now().isoformat()

                logger.info("✅ Enhanced vector database processing completed")

        except Exception as e:
            logger.error(f"❌ Enhanced vector database processing failed: {e}")
            raise

    async def _fetch_enhanced_scripts_data(self) -> List[Dict]:
        """Fetch TradingView scripts data for enhanced processing"""
        try:
            connection = psycopg2.connect(self.db_url)
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    slug, title, author, description, code,
                    indicators, signals, timeframes,
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
                    END as collection,
                    COALESCE(
                        luxalgo_category,
                        zeiierman_category,
                        lazybear_category,
                        chrismoody_category,
                        chartprime_category,
                        bigbeluga_category,
                        algoalpha_category
                    ) as category
                FROM tradingview.scripts
                WHERE (is_luxalgo OR is_zeiierman OR is_lazybear OR is_chrismoody
                       OR is_chartprime OR is_bigbeluga OR is_algoalpha)
                ORDER BY likes DESC
                LIMIT 100
            """)

            results = cursor.fetchall()
            connection.close()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error fetching enhanced scripts data: {e}")
            return []

    async def _populate_enhanced_collection(self, embeddings: List):
        """Populate the enhanced collection with processed embeddings"""
        try:
            logger.info(f"📊 Populating enhanced collection with {len(embeddings)} embeddings...")

            # Clear existing data
            try:
                count = self.enhanced_collection.count()
                if count > 0:
                    existing_data = self.enhanced_collection.get()
                    if existing_data['ids']:
                        self.enhanced_collection.delete(ids=existing_data['ids'])
            except Exception as e:
                logger.warning(f"Could not clear existing enhanced data: {e}")

            # Prepare data for batch insert
            ids = []
            documents = []
            metadatas = []

            for embedding in embeddings:
                ids.append(embedding.script_id)
                documents.append(embedding.embedding_text)

                # Combine metadata with semantic features
                metadata = embedding.semantic_features.copy()
                metadata.update({
                    'performance_weight': embedding.performance_weight,
                    'market_context': embedding.market_context
                })
                metadatas.append(metadata)

            # Batch insert to ChromaDB
            self.enhanced_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"✅ Populated enhanced collection with {len(ids)} items")

        except Exception as e:
            logger.error(f"❌ Error populating enhanced collection: {e}")
            raise

    async def _get_enhanced_database_stats(self) -> Dict[str, Any]:
        """Get enhanced database statistics"""
        stats = {
            "enhanced_collection_count": self.enhanced_collection.count() if self.enhanced_collection else 0,
            "enhanced_data_loaded": self.enhanced_data_loaded,
            "last_enhancement_sync": self.last_enhancement_sync,
            "capabilities": {
                "pine_script_parsing": True,
                "multi_modal_embeddings": True,
                "intelligent_query_processing": True,
                "performance_aware_recommendations": True,
                "strategy_composition": True,
                "market_regime_detection": True
            }
        }

        return stats

# Global service instance
enhanced_vector_service = EnhancedVectorDatabaseService()

# FastAPI app instance
app = enhanced_vector_service.app

@app.on_event("startup")
async def startup_event():
    """Initialize enhanced vector database on startup"""
    await enhanced_vector_service.initialize_enhanced_vector_db()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔌 Enhanced Vector Database Service shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_vector_db_service:app",
        host="0.0.0.0",
        port=8090,
        reload=False,
        log_level="info"
    )