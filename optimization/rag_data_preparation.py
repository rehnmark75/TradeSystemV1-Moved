#!/usr/bin/env python3
"""
RAG Data Preparation for TradingView Strategy Building

This script prepares the TradingView indicator data and optimization results
for RAG (Retrieval Augmented Generation) implementation, enabling AI-powered
strategy composition and intelligent indicator combination suggestions.
"""

import os
import sys
import json
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGDataPreparation:
    """Prepare TradingView and optimization data for RAG implementation"""

    def __init__(self):
        """Initialize the RAG data preparation service"""
        # Database connection settings
        self.db_host = 'postgres'
        self.db_port = 5432
        self.db_name = 'forex'
        self.db_user = 'postgres'
        self.db_pass = 'postgres'

        # Output directories
        self.output_dir = Path('/app/rag_data')
        self.output_dir.mkdir(exist_ok=True)

        # Data containers
        self.enriched_indicators = []
        self.strategy_templates = []
        self.market_taxonomy = {}

    def connect_db(self) -> Optional[psycopg2.extensions.connection]:
        """Connect to PostgreSQL database"""
        try:
            # Use connection string format to avoid parameter type issues
            conn_string = f"host='{self.db_host}' port='{self.db_port}' dbname='{self.db_name}' user='{self.db_user}' password='{self.db_pass}'"
            connection = psycopg2.connect(conn_string)
            logger.info(f"✅ Database connection successful")
            return connection
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            logger.error(f"Connection string: host={self.db_host}, port={self.db_port}, db={self.db_name}, user={self.db_user}")
            return None

    def extract_indicator_data(self) -> List[Dict]:
        """Extract and enrich TradingView indicator data for RAG"""
        logger.info("📊 Extracting TradingView indicator data for RAG preparation...")

        connection = self.connect_db()
        if not connection:
            return []

        try:
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            # Comprehensive query to get all indicator data with enrichment
            logger.info("📋 Executing indicator extraction query...")
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

                    -- Collection and category information
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

                    -- Unified category
                    COALESCE(
                        luxalgo_category,
                        zeiierman_category,
                        lazybear_category,
                        chrismoody_category,
                        chartprime_category,
                        bigbeluga_category,
                        algoalpha_category
                    ) as category,

                    -- Technical focus classification
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

            # Enrich each indicator for RAG
            enriched_data = []
            for row in results:
                enriched = self.create_embedding_content(dict(row))
                enriched_data.append(enriched)

            logger.info(f"✅ Extracted and enriched {len(enriched_data)} indicators")
            return enriched_data

        except Exception as e:
            logger.error(f"❌ Error extracting indicator data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        finally:
            connection.close()

    def create_embedding_content(self, indicator: Dict) -> Dict:
        """Create embedding-ready content by combining all relevant metadata"""

        # Create comprehensive text content for embedding
        embedding_content = []

        # Title and description
        embedding_content.append(f"Title: {indicator['title']}")
        embedding_content.append(f"Description: {indicator['description']}")

        # Author and collection context
        embedding_content.append(f"Author: {indicator['author']} ({indicator['collection']} collection)")

        # Technical classification
        if indicator['category']:
            embedding_content.append(f"Category: {indicator['category']}")
        if indicator['technical_focus']:
            embedding_content.append(f"Technical Focus: {indicator['technical_focus']}")

        # Indicators and signals
        if indicator['indicators']:
            indicators_text = ", ".join(indicator['indicators'])
            embedding_content.append(f"Indicators: {indicators_text}")

        if indicator['signals']:
            signals_text = ", ".join(indicator['signals'])
            embedding_content.append(f"Signals: {signals_text}")

        # Timeframes and complexity
        if indicator['timeframes']:
            timeframes_text = ", ".join(indicator['timeframes'])
            embedding_content.append(f"Timeframes: {timeframes_text}")

        embedding_content.append(f"Complexity: {self.classify_complexity(indicator['complexity_score'])}")

        # Market applicability
        market_context = self.infer_market_context(indicator)
        if market_context:
            embedding_content.append(f"Best for: {market_context}")

        # Combine into single embedding text
        full_embedding_text = " | ".join(embedding_content)

        # Create enriched indicator object
        enriched = {
            'id': indicator['slug'],
            'title': indicator['title'],
            'author': indicator['author'],
            'collection': indicator['collection'],
            'category': indicator['category'],
            'technical_focus': indicator['technical_focus'],
            'embedding_text': full_embedding_text,
            'metadata': {
                'strategy_type': indicator['strategy_type'],
                'indicators': indicator['indicators'] or [],
                'signals': indicator['signals'] or [],
                'timeframes': indicator['timeframes'] or [],
                'complexity_score': indicator['complexity_score'],
                'complexity_level': self.classify_complexity(indicator['complexity_score']),
                'open_source': indicator['open_source'],
                'popularity': {
                    'likes': indicator['likes'],
                    'views': indicator['views']
                },
                'market_context': self.infer_market_context(indicator)
            }
        }

        return enriched

    def classify_complexity(self, score: float) -> str:
        """Classify complexity score into human-readable levels"""
        if score >= 0.9:
            return "Expert"
        elif score >= 0.8:
            return "Advanced"
        elif score >= 0.6:
            return "Intermediate"
        else:
            return "Basic"

    def infer_market_context(self, indicator: Dict) -> str:
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

    def extract_strategy_templates(self) -> List[Dict]:
        """Extract successful strategy templates from optimization results"""
        logger.info("🎯 Extracting strategy templates from optimization results...")

        connection = self.connect_db()
        if not connection:
            return []

        try:
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            templates = []

            # Extract EMA strategy templates
            logger.info("📋 Executing EMA templates extraction query...")
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
                WHERE best_net_pips > 50  -- Only profitable templates
                ORDER BY best_net_pips DESC
                LIMIT 20
            """)

            ema_templates = cursor.fetchall()
            for template in ema_templates:
                enriched_template = self.create_strategy_template(dict(template))
                templates.append(enriched_template)

            # Extract MACD strategy templates
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
                WHERE best_composite_score > 1.0  -- Only good performing templates
                ORDER BY best_composite_score DESC
                LIMIT 15
            """)

            macd_templates = cursor.fetchall()
            for template in macd_templates:
                enriched_template = self.create_macd_template(dict(template))
                templates.append(enriched_template)

            logger.info(f"✅ Extracted {len(templates)} strategy templates")
            return templates

        except Exception as e:
            logger.error(f"❌ Error extracting strategy templates: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        finally:
            connection.close()

    def create_strategy_template(self, template: Dict) -> Dict:
        """Create enriched strategy template for RAG"""

        # Determine epic characteristics
        epic_type = self.classify_epic_type(template['epic'])
        market_session = self.infer_session_preference(template['epic'])

        # Create template description for embedding
        description_parts = [
            f"{template['strategy_type']} strategy for {template['epic']}",
            f"Configuration: {template['config_preset']}",
            f"Timeframe: {template['best_timeframe']}",
            f"Confidence threshold: {template['best_confidence_threshold']:.1%}",
            f"Risk management: {template['optimal_stop_loss_pips']:.0f} SL / {template['optimal_take_profit_pips']:.0f} TP",
            f"Performance: {template.get('best_win_rate', 0):.1%} win rate",
            f"Best for: {epic_type} pairs, {market_session} session"
        ]

        template_text = " | ".join(description_parts)

        enriched = {
            'id': f"{template['strategy_type'].lower()}_{template['epic'].replace('.', '_')}",
            'strategy_type': template['strategy_type'],
            'epic': template['epic'],
            'epic_type': epic_type,
            'embedding_text': template_text,
            'parameters': {
                'config_preset': template['config_preset'],
                'confidence_threshold': template['best_confidence_threshold'],
                'timeframe': template['best_timeframe'],
                'stop_loss_pips': template['optimal_stop_loss_pips'],
                'take_profit_pips': template['optimal_take_profit_pips']
            },
            'performance': {
                'win_rate': template.get('best_win_rate', 0),
                'profit_factor': template.get('best_profit_factor', 0),
                'net_pips': template.get('best_net_pips', 0)
            },
            'market_context': {
                'epic_type': epic_type,
                'session_preference': market_session,
                'timeframe': template['best_timeframe']
            }
        }

        return enriched

    def create_macd_template(self, template: Dict) -> Dict:
        """Create MACD-specific strategy template"""

        epic_type = self.classify_epic_type(template['epic'])

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

        enriched = {
            'id': f"macd_{template['epic'].replace('.', '_')}",
            'strategy_type': 'MACD',
            'epic': template['epic'],
            'epic_type': epic_type,
            'embedding_text': template_text,
            'parameters': {
                'fast_ema': template['best_fast_ema'],
                'slow_ema': template['best_slow_ema'],
                'signal_ema': template['best_signal_ema'],
                'confidence_threshold': template['best_confidence_threshold'],
                'timeframe': template['best_timeframe'],
                'stop_loss_pips': template['optimal_stop_loss_pips'],
                'take_profit_pips': template['optimal_take_profit_pips']
            },
            'performance': {
                'win_rate': template.get('best_win_rate', 0),
                'performance_score': template['performance_score']
            },
            'market_context': {
                'epic_type': epic_type,
                'timeframe': template['best_timeframe']
            }
        }

        return enriched

    def classify_epic_type(self, epic: str) -> str:
        """Classify epic into major/minor/exotic currency types"""
        if 'EURUSD' in epic or 'GBPUSD' in epic or 'USDJPY' in epic or 'USDCHF' in epic:
            return "Major"
        elif 'EURJPY' in epic or 'GBPJPY' in epic or 'AUDUSD' in epic or 'NZDUSD' in epic:
            return "Minor"
        else:
            return "Exotic"

    def infer_session_preference(self, epic: str) -> str:
        """Infer best trading session based on epic"""
        if 'EUR' in epic or 'GBP' in epic:
            return "London"
        elif 'JPY' in epic or 'AUD' in epic or 'NZD' in epic:
            return "Asian"
        elif 'USD' in epic:
            return "New York"
        else:
            return "London"  # Default

    def build_market_taxonomy(self) -> Dict:
        """Build comprehensive market context taxonomy for strategy building"""
        logger.info("🏗️ Building market taxonomy for intelligent strategy composition...")

        taxonomy = {
            'market_regimes': {
                'trending': {
                    'description': 'Strong directional movement with clear trend',
                    'best_indicators': ['trend_following', 'moving_average', 'trend_channel'],
                    'avoid_indicators': ['oscillator', 'reversal_indicator'],
                    'optimal_timeframes': ['1h', '4h', '1d'],
                    'risk_characteristics': 'Medium volatility, clear direction'
                },
                'ranging': {
                    'description': 'Sideways movement within defined boundaries',
                    'best_indicators': ['oscillator', 'support_resistance', 'mean_reversion'],
                    'avoid_indicators': ['trend_following', 'breakout'],
                    'optimal_timeframes': ['5m', '15m', '1h'],
                    'risk_characteristics': 'Low volatility, range-bound'
                },
                'volatile': {
                    'description': 'High volatility with rapid price movements',
                    'best_indicators': ['volatility', 'breakout', 'momentum'],
                    'avoid_indicators': ['slow_trend', 'conservative'],
                    'optimal_timeframes': ['1m', '5m', '15m'],
                    'risk_characteristics': 'High volatility, requires wider stops'
                }
            },
            'trading_styles': {
                'scalping': {
                    'timeframes': ['1m', '5m'],
                    'preferred_indicators': ['momentum', 'volume', 'scalping'],
                    'risk_management': 'Tight stops, quick exits',
                    'session_preference': ['london', 'new_york']
                },
                'day_trading': {
                    'timeframes': ['5m', '15m', '1h'],
                    'preferred_indicators': ['trend', 'momentum', 'support_resistance'],
                    'risk_management': 'Medium stops, intraday focus',
                    'session_preference': ['london', 'new_york']
                },
                'swing_trading': {
                    'timeframes': ['1h', '4h', '1d'],
                    'preferred_indicators': ['trend', 'divergence', 'pattern_recognition'],
                    'risk_management': 'Wide stops, multi-day holds',
                    'session_preference': ['any']
                }
            },
            'indicator_combinations': {
                'trend_momentum': {
                    'primary': ['trend_following', 'moving_average'],
                    'confirmation': ['momentum', 'volume'],
                    'filter': ['support_resistance'],
                    'use_case': 'Strong trending markets with momentum confirmation'
                },
                'mean_reversion': {
                    'primary': ['oscillator', 'reversal_indicator'],
                    'confirmation': ['support_resistance', 'volume'],
                    'filter': ['trend'],
                    'use_case': 'Range-bound markets with clear levels'
                },
                'breakout_system': {
                    'primary': ['volatility', 'breakout'],
                    'confirmation': ['volume', 'momentum'],
                    'filter': ['support_resistance'],
                    'use_case': 'Low volatility compression followed by expansion'
                }
            },
            'complexity_guidelines': {
                'beginner': {
                    'max_indicators': 2,
                    'preferred_complexity': 'Basic',
                    'recommended_collections': ['ChrisMoody', 'LazyBear'],
                    'avoid_collections': ['Zeiierman', 'AlgoAlpha']
                },
                'intermediate': {
                    'max_indicators': 3,
                    'preferred_complexity': 'Intermediate',
                    'recommended_collections': ['LuxAlgo', 'ChartPrime', 'LazyBear'],
                    'avoid_collections': ['Zeiierman neural networks']
                },
                'advanced': {
                    'max_indicators': 5,
                    'preferred_complexity': 'Advanced',
                    'recommended_collections': ['Zeiierman', 'BigBeluga', 'AlgoAlpha'],
                    'special_features': ['AI algorithms', 'institutional analysis']
                }
            }
        }

        return taxonomy

    def save_rag_data(self):
        """Save all prepared data for RAG implementation"""
        logger.info("💾 Saving RAG-prepared data...")

        try:
            # Save enriched indicators
            indicators_file = self.output_dir / 'enriched_indicators.json'
            with open(indicators_file, 'w') as f:
                json.dump(self.enriched_indicators, f, indent=2, default=str)

            # Save strategy templates
            templates_file = self.output_dir / 'strategy_templates.json'
            with open(templates_file, 'w') as f:
                json.dump(self.strategy_templates, f, indent=2, default=str)

            # Save market taxonomy
            taxonomy_file = self.output_dir / 'market_taxonomy.json'
            with open(taxonomy_file, 'w') as f:
                json.dump(self.market_taxonomy, f, indent=2, default=str)

            # Create summary report
            summary = {
                'preparation_timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'total_indicators': len(self.enriched_indicators),
                    'total_templates': len(self.strategy_templates),
                    'collections_covered': list(set([ind['collection'] for ind in self.enriched_indicators])),
                    'strategy_types_covered': list(set([tmp['strategy_type'] for tmp in self.strategy_templates])),
                    'complexity_distribution': self.analyze_complexity_distribution(),
                    'timeframe_coverage': self.analyze_timeframe_coverage()
                },
                'rag_readiness': {
                    'embedding_texts_prepared': True,
                    'metadata_enriched': True,
                    'taxonomy_built': True,
                    'templates_extracted': True,
                    'next_steps': [
                        'Generate embeddings using sentence-transformers',
                        'Setup vector database (ChromaDB/Pinecone)',
                        'Implement semantic search',
                        'Build strategy composition engine'
                    ]
                }
            }

            summary_file = self.output_dir / 'rag_preparation_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"✅ RAG data saved to {self.output_dir}")
            logger.info(f"📊 Summary: {len(self.enriched_indicators)} indicators, {len(self.strategy_templates)} templates")

        except Exception as e:
            logger.error(f"❌ Error saving RAG data: {e}")

    def analyze_complexity_distribution(self) -> Dict:
        """Analyze complexity distribution for summary"""
        distribution = {'Basic': 0, 'Intermediate': 0, 'Advanced': 0, 'Expert': 0}
        for indicator in self.enriched_indicators:
            level = indicator['metadata']['complexity_level']
            distribution[level] += 1
        return distribution

    def analyze_timeframe_coverage(self) -> List[str]:
        """Analyze timeframe coverage for summary"""
        all_timeframes = set()
        for indicator in self.enriched_indicators:
            timeframes = indicator['metadata']['timeframes']
            all_timeframes.update(timeframes)
        return sorted(list(all_timeframes))

    def run_preparation(self):
        """Execute complete RAG data preparation pipeline"""
        logger.info("🚀 Starting RAG Data Preparation Pipeline")
        logger.info("=" * 50)

        # Phase 1: Extract and enrich indicator data
        self.enriched_indicators = self.extract_indicator_data()

        # Phase 2: Extract strategy templates
        self.strategy_templates = self.extract_strategy_templates()

        # Phase 3: Build market taxonomy
        self.market_taxonomy = self.build_market_taxonomy()

        # Phase 4: Save all data
        self.save_rag_data()

        logger.info("🎉 RAG Data Preparation Complete!")
        logger.info(f"📊 Results:")
        logger.info(f"   Indicators prepared: {len(self.enriched_indicators)}")
        logger.info(f"   Strategy templates: {len(self.strategy_templates)}")
        logger.info(f"   Market taxonomy: Complete")
        logger.info(f"   Output directory: {self.output_dir}")

def main():
    """Main execution function"""
    print("🤖 RAG Data Preparation for TradingView Strategy Building")
    print("=" * 60)

    preparation_service = RAGDataPreparation()
    preparation_service.run_preparation()

    print("\n✅ RAG data preparation completed successfully!")
    print("🔗 Next steps:")
    print("   1. Generate embeddings using sentence-transformers")
    print("   2. Setup vector database (ChromaDB or Pinecone)")
    print("   3. Implement semantic search capabilities")
    print("   4. Build AI-powered strategy composition engine")

if __name__ == "__main__":
    main()