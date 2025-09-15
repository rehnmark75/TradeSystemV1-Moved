#!/usr/bin/env python3
"""
RAG Vector Database Setup for TradingView Strategy Building

This script implements a ChromaDB-based vector database for semantic search
and retrieval of TradingView indicators and strategy templates, enabling
AI-powered strategy composition and intelligent indicator combination suggestions.
"""

import os
import sys
import json
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGVectorDatabase:
    """ChromaDB-based vector database for TradingView indicators and strategy templates"""

    def __init__(self, persist_directory: str = "/app/rag_vectordb"):
        """Initialize the vector database"""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"✅ Initialized embedding model: all-MiniLM-L6-v2")

        # Create embedding function for ChromaDB
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Collections
        self.indicators_collection = None
        self.templates_collection = None
        self.taxonomy_collection = None

        # Data containers
        self.enriched_indicators = []
        self.strategy_templates = []
        self.market_taxonomy = {}

    def load_rag_data(self, data_directory: str = "/app/rag_data"):
        """Load prepared RAG data from files"""
        logger.info("📂 Loading RAG data for vector database...")

        data_dir = Path(data_directory)

        try:
            # Load enriched indicators
            indicators_file = data_dir / 'enriched_indicators.json'
            with open(indicators_file, 'r') as f:
                self.enriched_indicators = json.load(f)

            # Load strategy templates
            templates_file = data_dir / 'strategy_templates.json'
            with open(templates_file, 'r') as f:
                self.strategy_templates = json.load(f)

            # Load market taxonomy
            taxonomy_file = data_dir / 'market_taxonomy.json'
            with open(taxonomy_file, 'r') as f:
                self.market_taxonomy = json.load(f)

            logger.info(f"✅ Loaded {len(self.enriched_indicators)} indicators, {len(self.strategy_templates)} templates")

        except Exception as e:
            logger.error(f"❌ Error loading RAG data: {e}")
            raise

    def create_collections(self):
        """Create ChromaDB collections for different data types"""
        logger.info("🏗️ Creating ChromaDB collections...")

        try:
            # Create indicators collection
            self.indicators_collection = self.client.get_or_create_collection(
                name="tradingview_indicators",
                embedding_function=self.embedding_function,
                metadata={"description": "TradingView indicators with semantic search capabilities"}
            )

            # Create strategy templates collection
            self.templates_collection = self.client.get_or_create_collection(
                name="strategy_templates",
                embedding_function=self.embedding_function,
                metadata={"description": "Optimized strategy templates from backtesting results"}
            )

            # Create taxonomy collection
            self.taxonomy_collection = self.client.get_or_create_collection(
                name="market_taxonomy",
                embedding_function=self.embedding_function,
                metadata={"description": "Market context and trading methodology taxonomy"}
            )

            logger.info("✅ Collections created successfully")

        except Exception as e:
            logger.error(f"❌ Error creating collections: {e}")
            raise

    def populate_indicators_collection(self):
        """Populate indicators collection with embedding data"""
        logger.info("📊 Populating indicators collection...")

        if not self.enriched_indicators:
            logger.warning("No enriched indicators to populate")
            return

        try:
            # Prepare data for batch insert
            ids = []
            documents = []
            metadatas = []

            for indicator in self.enriched_indicators:
                ids.append(indicator['id'])
                documents.append(indicator['embedding_text'])

                # Create metadata with all relevant information
                metadata = {
                    'title': indicator['title'],
                    'author': indicator['author'],
                    'collection': indicator['collection'],
                    'category': indicator['category'],
                    'technical_focus': indicator['technical_focus'],
                    'complexity_level': indicator['metadata']['complexity_level'],
                    'complexity_score': indicator['metadata']['complexity_score'],
                    'strategy_type': indicator['metadata']['strategy_type'],
                    'timeframes': ','.join(indicator['metadata']['timeframes']),
                    'indicators': ','.join(indicator['metadata']['indicators']),
                    'signals': ','.join(indicator['metadata']['signals']),
                    'market_context': indicator['metadata']['market_context'],
                    'likes': indicator['metadata']['popularity']['likes'],
                    'views': indicator['metadata']['popularity']['views'],
                    'open_source': indicator['metadata']['open_source']
                }
                metadatas.append(metadata)

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

    def populate_templates_collection(self):
        """Populate strategy templates collection"""
        logger.info("🎯 Populating strategy templates collection...")

        if not self.strategy_templates:
            logger.warning("No strategy templates to populate")
            return

        try:
            # Prepare data for batch insert
            ids = []
            documents = []
            metadatas = []

            for template in self.strategy_templates:
                ids.append(template['id'])
                documents.append(template['embedding_text'])

                # Create metadata for strategy template
                metadata = {
                    'strategy_type': template['strategy_type'],
                    'epic': template['epic'],
                    'epic_type': template['epic_type'],
                    'timeframe': template['market_context']['timeframe'],
                    'win_rate': template['performance'].get('win_rate', 0),
                    'profit_factor': template['performance'].get('profit_factor', 0),
                    'net_pips': template['performance'].get('net_pips', 0),
                    'performance_score': template['performance'].get('performance_score', 0)
                }

                # Add strategy-specific parameters
                if template['strategy_type'] == 'EMA':
                    metadata.update({
                        'config_preset': template['parameters']['config_preset'],
                        'confidence_threshold': template['parameters']['confidence_threshold'],
                        'stop_loss_pips': template['parameters']['stop_loss_pips'],
                        'take_profit_pips': template['parameters']['take_profit_pips']
                    })
                elif template['strategy_type'] == 'MACD':
                    metadata.update({
                        'fast_ema': template['parameters']['fast_ema'],
                        'slow_ema': template['parameters']['slow_ema'],
                        'signal_ema': template['parameters']['signal_ema'],
                        'confidence_threshold': template['parameters']['confidence_threshold']
                    })

                metadatas.append(metadata)

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

    def populate_taxonomy_collection(self):
        """Populate market taxonomy collection"""
        logger.info("🏗️ Populating market taxonomy collection...")

        try:
            ids = []
            documents = []
            metadatas = []

            # Process market regimes
            for regime_name, regime_data in self.market_taxonomy['market_regimes'].items():
                id_name = f"market_regime_{regime_name}"
                ids.append(id_name)

                # Create embedding text for market regime
                doc_text = f"Market Regime: {regime_name} | Description: {regime_data['description']} | Best Indicators: {', '.join(regime_data['best_indicators'])} | Avoid: {', '.join(regime_data['avoid_indicators'])} | Timeframes: {', '.join(regime_data['optimal_timeframes'])} | Risk: {regime_data['risk_characteristics']}"
                documents.append(doc_text)

                metadata = {
                    'type': 'market_regime',
                    'name': regime_name,
                    'description': regime_data['description'],
                    'best_indicators': ','.join(regime_data['best_indicators']),
                    'avoid_indicators': ','.join(regime_data['avoid_indicators']),
                    'optimal_timeframes': ','.join(regime_data['optimal_timeframes']),
                    'risk_characteristics': regime_data['risk_characteristics']
                }
                metadatas.append(metadata)

            # Process trading styles
            for style_name, style_data in self.market_taxonomy['trading_styles'].items():
                id_name = f"trading_style_{style_name}"
                ids.append(id_name)

                doc_text = f"Trading Style: {style_name} | Timeframes: {', '.join(style_data['timeframes'])} | Preferred Indicators: {', '.join(style_data['preferred_indicators'])} | Risk Management: {style_data['risk_management']} | Sessions: {', '.join(style_data['session_preference'])}"
                documents.append(doc_text)

                metadata = {
                    'type': 'trading_style',
                    'name': style_name,
                    'timeframes': ','.join(style_data['timeframes']),
                    'preferred_indicators': ','.join(style_data['preferred_indicators']),
                    'risk_management': style_data['risk_management'],
                    'session_preference': ','.join(style_data['session_preference'])
                }
                metadatas.append(metadata)

            # Process indicator combinations
            for combo_name, combo_data in self.market_taxonomy['indicator_combinations'].items():
                id_name = f"indicator_combo_{combo_name}"
                ids.append(id_name)

                doc_text = f"Indicator Combination: {combo_name} | Primary: {', '.join(combo_data['primary'])} | Confirmation: {', '.join(combo_data['confirmation'])} | Filter: {', '.join(combo_data['filter'])} | Use Case: {combo_data['use_case']}"
                documents.append(doc_text)

                metadata = {
                    'type': 'indicator_combination',
                    'name': combo_name,
                    'primary': ','.join(combo_data['primary']),
                    'confirmation': ','.join(combo_data['confirmation']),
                    'filter': ','.join(combo_data['filter']),
                    'use_case': combo_data['use_case']
                }
                metadatas.append(metadata)

            # Batch insert to ChromaDB
            self.taxonomy_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"✅ Populated taxonomy collection with {len(ids)} items")

        except Exception as e:
            logger.error(f"❌ Error populating taxonomy collection: {e}")
            raise

    def semantic_search_indicators(self, query: str, n_results: int = 5,
                                 filters: Optional[Dict] = None) -> List[Dict]:
        """Perform semantic search on indicators collection"""
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

    def search_strategy_templates(self, query: str, n_results: int = 3,
                                filters: Optional[Dict] = None) -> List[Dict]:
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

    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        try:
            stats = {
                'indicators': {
                    'total_count': self.indicators_collection.count(),
                    'collections': {},
                    'complexity_levels': {},
                    'categories': {}
                },
                'templates': {
                    'total_count': self.templates_collection.count(),
                    'strategy_types': {},
                    'epic_types': {}
                },
                'taxonomy': {
                    'total_count': self.taxonomy_collection.count(),
                    'types': {}
                }
            }

            # Get detailed statistics (this would require querying all data)
            # For now, return basic counts
            return stats

        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}

    def setup_vector_database(self):
        """Execute complete vector database setup pipeline"""
        logger.info("🚀 Starting Vector Database Setup Pipeline")
        logger.info("=" * 50)

        try:
            # Step 1: Load RAG data
            self.load_rag_data()

            # Step 2: Create collections
            self.create_collections()

            # Step 3: Populate collections
            self.populate_indicators_collection()
            self.populate_templates_collection()
            self.populate_taxonomy_collection()

            # Step 4: Verify setup
            stats = self.get_database_stats()

            logger.info("🎉 Vector Database Setup Complete!")
            logger.info(f"📊 Final Statistics:")
            logger.info(f"   Indicators: {stats['indicators']['total_count']}")
            logger.info(f"   Templates: {stats['templates']['total_count']}")
            logger.info(f"   Taxonomy: {stats['taxonomy']['total_count']}")
            logger.info(f"   Database Path: {self.persist_directory}")

            return True

        except Exception as e:
            logger.error(f"❌ Vector database setup failed: {e}")
            return False

def main():
    """Main execution function"""
    print("🗃️ RAG Vector Database Setup for TradingView Strategy Building")
    print("=" * 65)

    vector_db = RAGVectorDatabase()
    success = vector_db.setup_vector_database()

    if success:
        print("\n✅ Vector database setup completed successfully!")
        print("🔗 Next steps:")
        print("   1. Implement semantic search API endpoints")
        print("   2. Build strategy composition engine")
        print("   3. Create natural language interface")
        print("   4. Test semantic search capabilities")
    else:
        print("\n❌ Vector database setup failed. Check logs for details.")
        return 1

    return 0

if __name__ == "__main__":
    main()