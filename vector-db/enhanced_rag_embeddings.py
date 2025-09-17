#!/usr/bin/env python3
"""
Enhanced Multi-Modal RAG Embeddings for Trading System
======================================================

This module implements advanced embedding strategies that combine:
- Pine Script code structure and semantics
- Trading concept understanding
- Performance metrics integration
- Market context awareness
"""

import json
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import re

# Enhanced imports for embeddings
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

@dataclass
class EnhancedEmbeddingInput:
    """Input data for creating enhanced embeddings"""
    # Basic script data
    script_id: str
    title: str
    author: str
    description: str
    code: Optional[str] = None

    # TradingView metadata
    collection: str = ""
    category: str = ""
    complexity_score: float = 0.5
    indicators: List[str] = None
    signals: List[str] = None
    timeframes: List[str] = None

    # Performance data (from optimization tables)
    performance_metrics: Dict[str, float] = None
    best_timeframes: List[str] = None
    profitable_pairs: List[str] = None

    # Market context
    market_conditions: List[str] = None
    trading_style: List[str] = None

    def __post_init__(self):
        if self.indicators is None:
            self.indicators = []
        if self.signals is None:
            self.signals = []
        if self.timeframes is None:
            self.timeframes = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.best_timeframes is None:
            self.best_timeframes = []
        if self.profitable_pairs is None:
            self.profitable_pairs = []
        if self.market_conditions is None:
            self.market_conditions = []
        if self.trading_style is None:
            self.trading_style = []

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    script_id: str
    embedding_vector: List[float]
    embedding_text: str
    semantic_features: Dict[str, Any]
    performance_weight: float
    market_context: Dict[str, Any]

class TradingConceptNormalizer:
    """Normalizes trading concepts for better semantic understanding"""

    def __init__(self):
        self.concept_mappings = {
            # Moving averages
            'moving_average': ['sma', 'ema', 'wma', 'vwma', 'moving average', 'ma', 'average'],
            'exponential_moving_average': ['ema', 'exponential ma', 'exponential moving average'],
            'simple_moving_average': ['sma', 'simple ma', 'simple moving average'],

            # Oscillators
            'rsi': ['rsi', 'relative strength index', 'relative strength'],
            'macd': ['macd', 'moving average convergence divergence', 'mac-d'],
            'stochastic': ['stoch', 'stochastic', 'stochastic oscillator'],
            'bollinger_bands': ['bb', 'bollinger', 'bollinger bands', 'bands'],

            # Trend indicators
            'adx': ['adx', 'average directional index', 'directional movement'],
            'supertrend': ['supertrend', 'super trend', 'trend following'],
            'parabolic_sar': ['psar', 'parabolic sar', 'sar', 'stop and reverse'],

            # Volume indicators
            'volume_profile': ['vp', 'volume profile', 'poc', 'point of control'],
            'vwap': ['vwap', 'volume weighted average price', 'volume weighted'],
            'on_balance_volume': ['obv', 'on balance volume', 'volume accumulation'],

            # Smart money concepts
            'order_blocks': ['ob', 'order block', 'institutional level', 'supply demand'],
            'fair_value_gap': ['fvg', 'fair value gap', 'imbalance', 'inefficiency'],
            'liquidity_sweep': ['sweep', 'liquidity grab', 'stop hunt', 'raid'],
            'break_of_structure': ['bos', 'break of structure', 'structure break'],
            'change_of_character': ['choch', 'change of character', 'trend change'],

            # Market conditions
            'trending_market': ['trend', 'trending', 'directional', 'momentum'],
            'ranging_market': ['range', 'ranging', 'sideways', 'consolidation', 'choppy'],
            'volatile_market': ['volatile', 'volatility', 'expansion', 'breakout'],

            # Trading styles
            'scalping': ['scalp', 'scalping', 'short term', 'quick trades'],
            'day_trading': ['day trade', 'day trading', 'intraday', 'daily'],
            'swing_trading': ['swing', 'swing trading', 'position', 'multi-day'],

            # Signal types
            'buy_signal': ['buy', 'long', 'bullish', 'up', 'call'],
            'sell_signal': ['sell', 'short', 'bearish', 'down', 'put'],
            'reversal': ['reversal', 'reverse', 'turn', 'pivot', 'divergence'],
            'continuation': ['continuation', 'follow through', 'momentum'],
            'breakout': ['breakout', 'break', 'breakthrough', 'expansion'],
        }

        # Create reverse mapping for quick lookup
        self.term_to_concept = {}
        for concept, terms in self.concept_mappings.items():
            for term in terms:
                self.term_to_concept[term.lower()] = concept

    def normalize_text(self, text: str) -> str:
        """Normalize text by replacing terms with standardized concepts"""
        if not text:
            return ""

        text_lower = text.lower()
        normalized_terms = []

        # Split into words and normalize
        words = re.findall(r'\b\w+\b', text_lower)

        for word in words:
            if word in self.term_to_concept:
                normalized_terms.append(self.term_to_concept[word])
            else:
                normalized_terms.append(word)

        return ' '.join(normalized_terms)

    def extract_concepts(self, text: str) -> List[str]:
        """Extract normalized trading concepts from text"""
        if not text:
            return []

        text_lower = text.lower()
        concepts = []

        # Check for each concept
        for concept, terms in self.concept_mappings.items():
            for term in terms:
                if term in text_lower:
                    concepts.append(concept)
                    break

        return list(set(concepts))

class PerformanceIntegrator:
    """Integrates performance data from optimization tables"""

    def __init__(self, db_url: str):
        self.db_url = db_url

    def get_strategy_performance(self, strategy_type: str, epic: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for a strategy type"""
        try:
            connection = psycopg2.connect(self.db_url)
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            performance_data = {}

            if strategy_type.upper() == 'EMA':
                # Get EMA performance data
                query = """
                    SELECT
                        epic,
                        best_win_rate,
                        best_profit_factor,
                        best_net_pips,
                        best_timeframe,
                        optimal_stop_loss_pips,
                        optimal_take_profit_pips
                    FROM ema_best_parameters
                    WHERE best_net_pips > 50
                """
                if epic:
                    query += " AND epic = %s"
                    cursor.execute(query, (epic,))
                else:
                    query += " ORDER BY best_net_pips DESC LIMIT 10"
                    cursor.execute(query)

                results = cursor.fetchall()
                performance_data['ema_performance'] = [dict(row) for row in results]

            elif strategy_type.upper() == 'MACD':
                # Get MACD performance data
                query = """
                    SELECT
                        epic,
                        best_win_rate,
                        best_composite_score,
                        best_timeframe,
                        optimal_stop_loss_pips,
                        optimal_take_profit_pips
                    FROM macd_best_parameters
                    WHERE best_composite_score > 1.0
                """
                if epic:
                    query += " AND epic = %s"
                    cursor.execute(query, (epic,))
                else:
                    query += " ORDER BY best_composite_score DESC LIMIT 10"
                    cursor.execute(query)

                results = cursor.fetchall()
                performance_data['macd_performance'] = [dict(row) for row in results]

            connection.close()
            return performance_data

        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return {}

    def calculate_performance_weight(self, performance_data: Dict[str, Any]) -> float:
        """Calculate a performance weight for ranking indicators"""
        if not performance_data:
            return 0.5  # Neutral weight

        weights = []

        # Process EMA performance
        if 'ema_performance' in performance_data:
            for perf in performance_data['ema_performance']:
                win_rate = perf.get('best_win_rate', 0.5)
                profit_factor = perf.get('best_profit_factor', 1.0)
                net_pips = perf.get('best_net_pips', 0)

                # Normalize and combine metrics
                weight = (win_rate * 0.4) + (min(profit_factor / 2.0, 1.0) * 0.3) + (min(net_pips / 500, 1.0) * 0.3)
                weights.append(weight)

        # Process MACD performance
        if 'macd_performance' in performance_data:
            for perf in performance_data['macd_performance']:
                win_rate = perf.get('best_win_rate', 0.5)
                composite_score = perf.get('best_composite_score', 1.0)

                # Normalize and combine metrics
                weight = (win_rate * 0.6) + (min(composite_score / 3.0, 1.0) * 0.4)
                weights.append(weight)

        return np.mean(weights) if weights else 0.5

class EnhancedRAGEmbeddings:
    """Advanced embedding system for trading indicators and strategies"""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.concept_normalizer = TradingConceptNormalizer()
        self.performance_integrator = PerformanceIntegrator(db_url)

        # Initialize embedding models
        self.code_model = SentenceTransformer('microsoft/codebert-base')
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.trading_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        logger.info("Initialized enhanced RAG embeddings with multi-modal models")

    def create_enhanced_embedding(self, input_data: EnhancedEmbeddingInput) -> EmbeddingResult:
        """Create multi-modal embedding from input data"""

        # 1. Create base text embedding
        text_embedding = self._create_text_embedding(input_data)

        # 2. Create code embedding (if available)
        code_embedding = self._create_code_embedding(input_data)

        # 3. Create concept embedding
        concept_embedding = self._create_concept_embedding(input_data)

        # 4. Create performance embedding
        performance_embedding = self._create_performance_embedding(input_data)

        # 5. Combine embeddings with weights
        combined_embedding = self._combine_embeddings([
            (text_embedding, 0.3),
            (code_embedding, 0.25),
            (concept_embedding, 0.25),
            (performance_embedding, 0.2)
        ])

        # 6. Generate semantic features
        semantic_features = self._extract_semantic_features(input_data)

        # 7. Calculate performance weight
        performance_weight = self.performance_integrator.calculate_performance_weight(
            input_data.performance_metrics
        )

        # 8. Create enhanced embedding text
        embedding_text = self._create_enhanced_embedding_text(input_data, semantic_features)

        # 9. Generate market context
        market_context = self._generate_market_context(input_data)

        return EmbeddingResult(
            script_id=input_data.script_id,
            embedding_vector=combined_embedding.tolist(),
            embedding_text=embedding_text,
            semantic_features=semantic_features,
            performance_weight=performance_weight,
            market_context=market_context
        )

    def _create_text_embedding(self, input_data: EnhancedEmbeddingInput) -> np.ndarray:
        """Create embedding from text description"""
        text_parts = []

        if input_data.title:
            text_parts.append(input_data.title)
        if input_data.description:
            text_parts.append(input_data.description)
        if input_data.category:
            text_parts.append(f"Category: {input_data.category}")

        # Normalize trading concepts in text
        text = " ".join(text_parts)
        normalized_text = self.concept_normalizer.normalize_text(text)

        return self.text_model.encode(normalized_text)

    def _create_code_embedding(self, input_data: EnhancedEmbeddingInput) -> np.ndarray:
        """Create embedding from Pine Script code"""
        if not input_data.code:
            # Return zero vector if no code
            return np.zeros(768)  # CodeBERT dimension

        # Clean and prepare code for embedding
        code_lines = input_data.code.split('\n')

        # Filter relevant code lines (skip comments and empty lines)
        relevant_lines = []
        for line in code_lines:
            line = line.strip()
            if line and not line.startswith('//') and not line.startswith('//@'):
                relevant_lines.append(line)

        # Limit code length for embedding
        code_text = ' '.join(relevant_lines[:50])  # First 50 relevant lines

        return self.code_model.encode(code_text)

    def _create_concept_embedding(self, input_data: EnhancedEmbeddingInput) -> np.ndarray:
        """Create embedding from trading concepts"""
        concept_parts = []

        # Add indicators
        if input_data.indicators:
            normalized_indicators = [self.concept_normalizer.normalize_text(ind) for ind in input_data.indicators]
            concept_parts.extend(normalized_indicators)

        # Add signals
        if input_data.signals:
            normalized_signals = [self.concept_normalizer.normalize_text(sig) for sig in input_data.signals]
            concept_parts.extend(normalized_signals)

        # Add market conditions
        if input_data.market_conditions:
            concept_parts.extend(input_data.market_conditions)

        # Add trading style
        if input_data.trading_style:
            concept_parts.extend(input_data.trading_style)

        # Create concept text
        concept_text = " ".join(concept_parts) if concept_parts else "general trading indicator"

        return self.trading_model.encode(concept_text)

    def _create_performance_embedding(self, input_data: EnhancedEmbeddingInput) -> np.ndarray:
        """Create embedding from performance characteristics"""
        performance_parts = []

        # Complexity-based features
        complexity_level = self._classify_complexity(input_data.complexity_score)
        performance_parts.append(f"complexity_{complexity_level}")

        # Collection-based features
        if input_data.collection:
            performance_parts.append(f"collection_{input_data.collection.lower()}")

        # Timeframe preferences
        if input_data.best_timeframes:
            performance_parts.extend([f"timeframe_{tf}" for tf in input_data.best_timeframes])
        elif input_data.timeframes:
            performance_parts.extend([f"timeframe_{tf}" for tf in input_data.timeframes[:3]])

        # Performance metrics
        if input_data.performance_metrics:
            for metric, value in input_data.performance_metrics.items():
                if value > 0:
                    performance_parts.append(f"performance_{metric}")

        # Profitable pairs
        if input_data.profitable_pairs:
            performance_parts.extend([f"profitable_on_{pair}" for pair in input_data.profitable_pairs[:3]])

        performance_text = " ".join(performance_parts) if performance_parts else "standard_performance"

        return self.trading_model.encode(performance_text)

    def _combine_embeddings(self, weighted_embeddings: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Combine multiple embeddings with weights"""
        # Ensure all embeddings have the same dimension
        max_dim = max(emb.shape[0] for emb, _ in weighted_embeddings)

        combined = np.zeros(max_dim)
        total_weight = 0

        for embedding, weight in weighted_embeddings:
            if embedding.shape[0] < max_dim:
                # Pad smaller embeddings
                padded = np.pad(embedding, (0, max_dim - embedding.shape[0]), mode='constant')
            else:
                padded = embedding[:max_dim]

            combined += padded * weight
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            combined /= total_weight

        return combined

    def _extract_semantic_features(self, input_data: EnhancedEmbeddingInput) -> Dict[str, Any]:
        """Extract semantic features for metadata"""
        features = {
            # Basic classification
            'collection': input_data.collection,
            'category': input_data.category,
            'complexity_level': self._classify_complexity(input_data.complexity_score),
            'author': input_data.author,

            # Technical features
            'indicators_count': len(input_data.indicators),
            'signals_count': len(input_data.signals),
            'has_code': bool(input_data.code),

            # Extracted concepts
            'extracted_concepts': self.concept_normalizer.extract_concepts(
                f"{input_data.title} {input_data.description}"
            ),

            # Performance features
            'has_performance_data': bool(input_data.performance_metrics),
            'performance_weight': self.performance_integrator.calculate_performance_weight(
                input_data.performance_metrics
            ),

            # Market applicability
            'best_timeframes': input_data.best_timeframes or input_data.timeframes[:3],
            'market_conditions': input_data.market_conditions,
            'trading_styles': input_data.trading_style,

            # Utility scores
            'versatility_score': self._calculate_versatility_score(input_data),
            'reliability_score': self._calculate_reliability_score(input_data),
        }

        return features

    def _create_enhanced_embedding_text(self, input_data: EnhancedEmbeddingInput,
                                      semantic_features: Dict[str, Any]) -> str:
        """Create comprehensive embedding text for search"""
        parts = []

        # Title and basic info
        if input_data.title:
            parts.append(f"Title: {input_data.title}")

        # Author and collection info
        author_info = f"Author: {input_data.author}"
        if input_data.collection:
            author_info += f" ({input_data.collection} collection)"
        parts.append(author_info)

        # Description with normalized concepts
        if input_data.description:
            normalized_desc = self.concept_normalizer.normalize_text(input_data.description)
            parts.append(f"Description: {normalized_desc}")

        # Category and complexity
        if input_data.category:
            parts.append(f"Category: {input_data.category}")
        parts.append(f"Complexity: {semantic_features['complexity_level']}")

        # Technical indicators used
        if input_data.indicators:
            normalized_indicators = [self.concept_normalizer.normalize_text(ind) for ind in input_data.indicators]
            parts.append(f"Technical Indicators: {', '.join(normalized_indicators)}")

        # Signal types
        if input_data.signals:
            normalized_signals = [self.concept_normalizer.normalize_text(sig) for sig in input_data.signals]
            parts.append(f"Signals: {', '.join(normalized_signals)}")

        # Market conditions and trading style
        if input_data.market_conditions:
            parts.append(f"Best for: {', '.join(input_data.market_conditions)} markets")

        if input_data.trading_style:
            parts.append(f"Trading Style: {', '.join(input_data.trading_style)}")

        # Performance characteristics
        if input_data.performance_metrics:
            parts.append("Performance: Optimized and backtested")

        if input_data.best_timeframes:
            parts.append(f"Recommended Timeframes: {', '.join(input_data.best_timeframes[:3])}")
        elif input_data.timeframes:
            parts.append(f"Timeframes: {', '.join(input_data.timeframes[:3])}")

        # Versatility and reliability
        if semantic_features['versatility_score'] > 0.7:
            parts.append("Highly versatile across different market conditions")

        if semantic_features['reliability_score'] > 0.7:
            parts.append("Reliable performance with proven track record")

        # Extracted concepts
        if semantic_features['extracted_concepts']:
            unique_concepts = list(set(semantic_features['extracted_concepts']))
            parts.append(f"Trading Concepts: {', '.join(unique_concepts[:5])}")

        return " | ".join(parts)

    def _generate_market_context(self, input_data: EnhancedEmbeddingInput) -> Dict[str, Any]:
        """Generate market context for the indicator"""
        context = {
            'optimal_conditions': input_data.market_conditions or ['general'],
            'timeframe_suitability': input_data.best_timeframes or input_data.timeframes,
            'trading_session_preference': self._infer_session_preference(input_data),
            'volatility_preference': self._infer_volatility_preference(input_data),
            'trend_dependency': self._infer_trend_dependency(input_data),
            'volume_dependency': self._infer_volume_dependency(input_data)
        }

        return context

    def _classify_complexity(self, score: float) -> str:
        """Classify complexity score into levels"""
        if score >= 0.8:
            return "Expert"
        elif score >= 0.6:
            return "Advanced"
        elif score >= 0.4:
            return "Intermediate"
        else:
            return "Basic"

    def _calculate_versatility_score(self, input_data: EnhancedEmbeddingInput) -> float:
        """Calculate versatility score based on multiple factors"""
        score = 0.0

        # Multiple timeframes support
        timeframes = input_data.best_timeframes or input_data.timeframes
        if len(timeframes) >= 3:
            score += 0.3

        # Multiple market conditions
        if len(input_data.market_conditions) >= 2:
            score += 0.3

        # Multiple trading styles
        if len(input_data.trading_style) >= 2:
            score += 0.2

        # Configuration flexibility (many indicators suggest flexibility)
        if len(input_data.indicators) >= 3:
            score += 0.2

        return min(score, 1.0)

    def _calculate_reliability_score(self, input_data: EnhancedEmbeddingInput) -> float:
        """Calculate reliability score based on performance and track record"""
        score = 0.0

        # Performance data availability
        if input_data.performance_metrics:
            score += 0.4

        # Known author/collection
        if input_data.collection in ['LuxAlgo', 'Zeiierman', 'LazyBear']:
            score += 0.3

        # Code availability (open source verification)
        if input_data.code:
            score += 0.2

        # Complexity suggests sophistication
        if input_data.complexity_score > 0.6:
            score += 0.1

        return min(score, 1.0)

    def _infer_session_preference(self, input_data: EnhancedEmbeddingInput) -> List[str]:
        """Infer preferred trading sessions"""
        # Simple heuristics based on timeframes and style
        timeframes = input_data.best_timeframes or input_data.timeframes

        if any(tf in ['1m', '5m', '15m'] for tf in timeframes):
            return ['london', 'newyork']  # High liquidity sessions
        elif any(tf in ['1h', '4h'] for tf in timeframes):
            return ['london', 'newyork', 'asia']  # All major sessions
        else:
            return ['any']  # Daily+ timeframes work in any session

    def _infer_volatility_preference(self, input_data: EnhancedEmbeddingInput) -> str:
        """Infer volatility preference"""
        if 'volatile' in input_data.market_conditions:
            return 'high'
        elif 'ranging' in input_data.market_conditions:
            return 'low'
        else:
            return 'medium'

    def _infer_trend_dependency(self, input_data: EnhancedEmbeddingInput) -> str:
        """Infer trend dependency"""
        concepts = self.concept_normalizer.extract_concepts(
            f"{input_data.title} {input_data.description}"
        )

        if any(concept in concepts for concept in ['moving_average', 'supertrend', 'adx']):
            return 'high'
        elif 'ranging' in input_data.market_conditions:
            return 'low'
        else:
            return 'medium'

    def _infer_volume_dependency(self, input_data: EnhancedEmbeddingInput) -> str:
        """Infer volume dependency"""
        concepts = self.concept_normalizer.extract_concepts(
            f"{input_data.title} {input_data.description}"
        )

        if any(concept in concepts for concept in ['vwap', 'volume_profile', 'on_balance_volume']):
            return 'high'
        else:
            return 'low'

# Test function
def test_enhanced_embeddings():
    """Test the enhanced embedding system"""

    # Mock data for testing
    test_input = EnhancedEmbeddingInput(
        script_id="test_rsi_div",
        title="Enhanced RSI with Divergence Detection",
        author="TradingPro",
        description="Advanced RSI oscillator with automatic divergence detection for reversal signals",
        collection="Premium",
        category="oscillator",
        complexity_score=0.75,
        indicators=['RSI', 'Divergence', 'Oscillator'],
        signals=['Bullish Divergence', 'Bearish Divergence', 'Overbought', 'Oversold'],
        timeframes=['15m', '1h', '4h'],
        performance_metrics={'win_rate': 0.68, 'profit_factor': 1.85},
        market_conditions=['trending', 'volatile'],
        trading_style=['day_trading', 'swing_trading']
    )

    # Create embeddings instance (mock DB URL)
    embeddings = EnhancedRAGEmbeddings("postgresql://test")

    # Generate embedding
    result = embeddings.create_enhanced_embedding(test_input)

    print("=== Enhanced Embedding Result ===")
    print(f"Script ID: {result.script_id}")
    print(f"Embedding dimension: {len(result.embedding_vector)}")
    print(f"Performance weight: {result.performance_weight:.3f}")
    print(f"\nEmbedding text:\n{result.embedding_text}")
    print(f"\nSemantic features:")
    for key, value in result.semantic_features.items():
        print(f"  {key}: {value}")
    print(f"\nMarket context:")
    for key, value in result.market_context.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_enhanced_embeddings()