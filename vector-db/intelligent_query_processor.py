#!/usr/bin/env python3
"""
Intelligent Query Processor for RAG Trading System
=================================================

This module provides advanced query processing capabilities that:
- Normalizes trading terminology and concepts
- Expands queries with synonyms and related terms
- Adds contextual filtering based on market conditions
- Translates natural language to technical parameters
- Provides multi-intent query understanding
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime
import spacy
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class QueryIntent:
    """Represents a parsed query intent"""
    intent_type: str  # search, recommend, compare, analyze, compose
    confidence: float
    parameters: Dict[str, Any]
    entities: List[str]

@dataclass
class ProcessedQuery:
    """Result of query processing"""
    original_query: str
    normalized_query: str
    expanded_terms: List[str]
    intents: List[QueryIntent]
    filters: Dict[str, Any]
    context: Dict[str, Any]
    query_type: str  # single_intent, multi_intent, conversational
    complexity_level: str  # basic, intermediate, advanced

class TradingTermNormalizer:
    """Normalizes and expands trading terminology"""

    def __init__(self):
        self.trading_synonyms = {
            # Moving averages
            'moving_average': ['ma', 'moving average', 'average', 'mean'],
            'exponential_moving_average': ['ema', 'exponential ma', 'exp ma', 'exponential average'],
            'simple_moving_average': ['sma', 'simple ma', 'simple average'],
            'weighted_moving_average': ['wma', 'weighted ma', 'weighted average'],
            'hull_moving_average': ['hma', 'hull ma', 'hull average'],

            # Oscillators
            'relative_strength_index': ['rsi', 'relative strength', 'strength index'],
            'stochastic': ['stoch', 'stochastic oscillator', '%k', '%d'],
            'macd': ['macd', 'moving average convergence divergence', 'mac-d'],
            'commodity_channel_index': ['cci', 'commodity channel', 'channel index'],
            'williams_percent_r': ['williams %r', 'williams', '%r'],

            # Trend indicators
            'average_directional_index': ['adx', 'directional index', 'directional movement'],
            'parabolic_sar': ['psar', 'sar', 'parabolic', 'stop and reverse'],
            'supertrend': ['super trend', 'supertrend', 'trend indicator'],
            'aroon': ['aroon up', 'aroon down', 'aroon oscillator'],

            # Volume indicators
            'volume_weighted_average_price': ['vwap', 'volume weighted', 'weighted price'],
            'on_balance_volume': ['obv', 'balance volume', 'volume accumulation'],
            'chaikin_money_flow': ['cmf', 'chaikin', 'money flow'],
            'volume_profile': ['vp', 'volume profile', 'poc', 'value area'],

            # Volatility indicators
            'bollinger_bands': ['bb', 'bollinger', 'bands', 'volatility bands'],
            'keltner_channels': ['kc', 'keltner', 'channels'],
            'average_true_range': ['atr', 'true range', 'volatility'],

            # Smart Money Concepts
            'order_blocks': ['ob', 'order block', 'institutional level', 'supply zone', 'demand zone'],
            'fair_value_gap': ['fvg', 'fair value', 'imbalance', 'inefficiency'],
            'break_of_structure': ['bos', 'break of structure', 'structure break'],
            'change_of_character': ['choch', 'change of character', 'character change'],
            'liquidity_sweep': ['sweep', 'liquidity grab', 'stop hunt', 'raid'],
            'inducement': ['inducement', 'false break', 'liquidity grab'],

            # Market Structure
            'support_resistance': ['support', 'resistance', 'sr', 'key level'],
            'trend_line': ['trendline', 'trend line', 'line'],
            'fibonacci': ['fib', 'fibonacci', 'retracement', 'extension'],
            'pivot_points': ['pivot', 'pivot points', 'daily pivot'],

            # Chart Patterns
            'head_and_shoulders': ['h&s', 'head and shoulders', 'shoulder pattern'],
            'double_top': ['double top', 'twin peaks'],
            'double_bottom': ['double bottom', 'twin troughs'],
            'triangle': ['triangle', 'ascending triangle', 'descending triangle'],
            'flag': ['flag', 'bull flag', 'bear flag'],
            'pennant': ['pennant', 'symmetrical triangle'],

            # Market Conditions
            'trending': ['trend', 'trending', 'directional', 'momentum'],
            'ranging': ['range', 'ranging', 'sideways', 'consolidation', 'choppy'],
            'volatile': ['volatile', 'volatility', 'expansion'],
            'breakout': ['breakout', 'break', 'breakthrough'],
            'reversal': ['reversal', 'reverse', 'turn', 'pivot'],

            # Trading Styles
            'scalping': ['scalp', 'scalping', 'quick trades'],
            'day_trading': ['day trade', 'day trading', 'intraday'],
            'swing_trading': ['swing', 'swing trading', 'position'],
            'position_trading': ['position', 'long term', 'buy and hold'],

            # Signal Types
            'buy_signal': ['buy', 'long', 'bullish', 'up', 'call'],
            'sell_signal': ['sell', 'short', 'bearish', 'down', 'put'],
            'entry_signal': ['entry', 'enter', 'signal'],
            'exit_signal': ['exit', 'close', 'take profit', 'stop loss'],

            # Time frames
            'minute_chart': ['1m', '5m', '15m', 'minute', 'short term'],
            'hourly_chart': ['1h', '4h', 'hourly', 'hour'],
            'daily_chart': ['1d', 'daily', 'day', 'eod'],
            'weekly_chart': ['1w', 'weekly', 'week'],
            'monthly_chart': ['1mo', 'monthly', 'month'],

            # Currency pairs
            'major_pairs': ['eurusd', 'gbpusd', 'usdjpy', 'usdchf', 'majors'],
            'minor_pairs': ['eurjpy', 'gbpjpy', 'audusd', 'nzdusd', 'minors'],
            'exotic_pairs': ['usdtry', 'usdzar', 'exotics']
        }

        # Create reverse mapping for quick lookup
        self.term_to_concept = {}
        for concept, terms in self.trading_synonyms.items():
            for term in terms:
                self.term_to_concept[term.lower()] = concept

        # Related concepts for query expansion
        self.related_concepts = {
            'trend': ['momentum', 'direction', 'supertrend', 'moving_average'],
            'momentum': ['rsi', 'macd', 'stochastic', 'oscillator'],
            'volatility': ['bollinger_bands', 'atr', 'keltner_channels'],
            'volume': ['vwap', 'obv', 'volume_profile'],
            'reversal': ['divergence', 'pivot', 'support_resistance'],
            'breakout': ['volatility', 'volume', 'trend'],
            'scalping': ['minute_chart', 'quick_trades', 'momentum'],
            'swing_trading': ['daily_chart', 'trend', 'support_resistance']
        }

        # Context modifiers
        self.context_modifiers = {
            'timeframe': ['short term', 'long term', 'intraday', 'daily', 'weekly'],
            'market_condition': ['trending', 'ranging', 'volatile', 'quiet'],
            'trading_style': ['aggressive', 'conservative', 'scalping', 'swing'],
            'risk_level': ['high risk', 'low risk', 'safe', 'risky'],
            'experience': ['beginner', 'intermediate', 'advanced', 'expert', 'professional']
        }

class IntentClassifier:
    """Classifies query intents and extracts parameters"""

    def __init__(self):
        self.intent_patterns = {
            'search': [
                r'find|search|look for|show me|list',
                r'what.*indicators?',
                r'which.*strategies?',
                r'get.*for'
            ],
            'recommend': [
                r'recommend|suggest|advise|best.*for',
                r'what.*should.*use',
                r'help.*choose',
                r'good.*for'
            ],
            'compare': [
                r'compare|vs|versus|difference|better',
                r'which.*better',
                r'pros.*cons',
                r'advantages.*disadvantages'
            ],
            'analyze': [
                r'analyze|analysis|performance|backtest',
                r'how.*perform',
                r'statistics|stats|metrics',
                r'results.*of'
            ],
            'compose': [
                r'create.*strategy|build.*system',
                r'combine.*indicators',
                r'setup.*for',
                r'strategy.*for'
            ],
            'explain': [
                r'what.*is|how.*works?|explain',
                r'definition|meaning',
                r'understand.*how',
                r'tell.*about'
            ]
        }

        self.parameter_patterns = {
            'timeframe': r'(?:on\s+)?(\d+[mhwd]|minute|hour|daily|weekly)',
            'pair': r'(EUR/USD|GBP/USD|USD/JPY|[A-Z]{6}|[A-Z]{3}/[A-Z]{3})',
            'market_condition': r'(trending|ranging|volatile|breakout|reversal)',
            'trading_style': r'(scalping|day.trading|swing.trading|position)',
            'complexity': r'(simple|basic|advanced|complex|professional)',
            'performance': r'(profitable|winning|best.performing|high.win.rate)'
        }

    def classify_intent(self, query: str) -> List[QueryIntent]:
        """Classify the intent of a query"""
        query_lower = query.lower()
        intents = []

        # Check each intent type
        for intent_type, patterns in self.intent_patterns.items():
            confidence = 0.0
            matched_patterns = []

            for pattern in patterns:
                if re.search(pattern, query_lower):
                    confidence += 0.3
                    matched_patterns.append(pattern)

            if confidence > 0:
                # Extract parameters
                parameters = self._extract_parameters(query_lower)

                # Extract entities
                entities = self._extract_entities(query)

                intent = QueryIntent(
                    intent_type=intent_type,
                    confidence=min(confidence, 1.0),
                    parameters=parameters,
                    entities=entities
                )
                intents.append(intent)

        # Sort by confidence
        intents.sort(key=lambda x: x.confidence, reverse=True)

        # If no intents found, default to search
        if not intents:
            intents.append(QueryIntent(
                intent_type='search',
                confidence=0.5,
                parameters={},
                entities=[]
            ))

        return intents

    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from query"""
        parameters = {}

        for param_type, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                parameters[param_type] = matches[0] if len(matches) == 1 else matches

        return parameters

    def _extract_entities(self, query: str) -> List[str]:
        """Extract trading entities from query"""
        entities = []

        # Simple entity extraction using patterns
        trading_terms = [
            'RSI', 'MACD', 'EMA', 'SMA', 'Bollinger', 'VWAP', 'Stochastic',
            'ADX', 'ATR', 'Fibonacci', 'Support', 'Resistance', 'Trend',
            'Momentum', 'Volume', 'Volatility', 'Breakout', 'Reversal'
        ]

        query_upper = query.upper()
        for term in trading_terms:
            if term in query_upper:
                entities.append(term)

        return entities

class QueryExpander:
    """Expands queries with related terms and concepts"""

    def __init__(self, normalizer: TradingTermNormalizer):
        self.normalizer = normalizer

    def expand_query(self, normalized_query: str, intents: List[QueryIntent]) -> List[str]:
        """Expand query with related terms"""
        expanded_terms = [normalized_query]

        # Extract key concepts
        words = normalized_query.lower().split()
        concepts = []

        for word in words:
            if word in self.normalizer.term_to_concept:
                concept = self.normalizer.term_to_concept[word]
                concepts.append(concept)

        # Add related concepts
        for concept in concepts:
            if concept in self.normalizer.related_concepts:
                related = self.normalizer.related_concepts[concept]
                expanded_terms.extend(related[:3])  # Limit to top 3 related

        # Intent-specific expansion
        if intents:
            primary_intent = intents[0]
            if primary_intent.intent_type == 'recommend':
                expanded_terms.extend(['best', 'optimal', 'recommended'])
            elif primary_intent.intent_type == 'analyze':
                expanded_terms.extend(['performance', 'backtest', 'statistics'])
            elif primary_intent.intent_type == 'compare':
                expanded_terms.extend(['versus', 'comparison', 'difference'])

        return list(set(expanded_terms))  # Remove duplicates

class ContextualFilter:
    """Applies contextual filters based on query understanding"""

    def __init__(self):
        self.filter_mappings = {
            'beginner': {'complexity_level': 'Basic'},
            'advanced': {'complexity_level': 'Advanced'},
            'expert': {'complexity_level': 'Expert'},
            'scalping': {'trading_style': 'scalping', 'timeframes': ['1m', '5m', '15m']},
            'day_trading': {'trading_style': 'day_trading', 'timeframes': ['15m', '1h', '4h']},
            'swing_trading': {'trading_style': 'swing_trading', 'timeframes': ['4h', '1d']},
            'trending': {'market_conditions': 'trending'},
            'ranging': {'market_conditions': 'ranging'},
            'volatile': {'market_conditions': 'volatile'},
            'luxalgo': {'collection': 'LuxAlgo'},
            'zeiierman': {'collection': 'Zeiierman'},
            'lazybear': {'collection': 'LazyBear'}
        }

    def generate_filters(self, processed_query: ProcessedQuery) -> Dict[str, Any]:
        """Generate filters based on query context"""
        filters = {}

        # Extract filters from query text
        query_lower = processed_query.normalized_query.lower()

        for keyword, filter_dict in self.filter_mappings.items():
            if keyword in query_lower:
                filters.update(filter_dict)

        # Extract filters from intent parameters
        for intent in processed_query.intents:
            if intent.parameters:
                filters.update(intent.parameters)

        return filters

class IntelligentQueryProcessor:
    """Main query processor that orchestrates all components"""

    def __init__(self):
        self.normalizer = TradingTermNormalizer()
        self.intent_classifier = IntentClassifier()
        self.query_expander = QueryExpander(self.normalizer)
        self.contextual_filter = ContextualFilter()

        logger.info("Initialized intelligent query processor")

    def process_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> ProcessedQuery:
        """Process a natural language query into structured format"""

        # Step 1: Normalize trading terms
        normalized_query = self._normalize_query(query)

        # Step 2: Classify intents
        intents = self.intent_classifier.classify_intent(query)

        # Step 3: Expand with related terms
        expanded_terms = self.query_expander.expand_query(normalized_query, intents)

        # Step 4: Determine query type and complexity
        query_type = self._determine_query_type(intents)
        complexity_level = self._determine_complexity(query, intents)

        # Step 5: Create processed query object
        processed_query = ProcessedQuery(
            original_query=query,
            normalized_query=normalized_query,
            expanded_terms=expanded_terms,
            intents=intents,
            filters={},
            context=user_context or {},
            query_type=query_type,
            complexity_level=complexity_level
        )

        # Step 6: Generate contextual filters
        processed_query.filters = self.contextual_filter.generate_filters(processed_query)

        # Step 7: Add user context
        if user_context:
            self._apply_user_context(processed_query, user_context)

        return processed_query

    def _normalize_query(self, query: str) -> str:
        """Normalize trading terms in the query"""
        words = re.findall(r'\b\w+\b', query.lower())
        normalized_words = []

        for word in words:
            if word in self.normalizer.term_to_concept:
                concept = self.normalizer.term_to_concept[word]
                normalized_words.append(concept)
            else:
                normalized_words.append(word)

        return ' '.join(normalized_words)

    def _determine_query_type(self, intents: List[QueryIntent]) -> str:
        """Determine the type of query"""
        if len(intents) == 1:
            return 'single_intent'
        elif len(intents) > 1:
            return 'multi_intent'
        else:
            return 'conversational'

    def _determine_complexity(self, query: str, intents: List[QueryIntent]) -> str:
        """Determine query complexity"""
        complexity_indicators = {
            'basic': ['simple', 'easy', 'beginner', 'basic'],
            'intermediate': ['strategy', 'indicator', 'analysis'],
            'advanced': ['backtest', 'optimization', 'performance', 'combine', 'sophisticated']
        }

        query_lower = query.lower()

        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return level

        # Default based on intent complexity
        if intents and intents[0].intent_type in ['analyze', 'compose']:
            return 'advanced'
        elif intents and intents[0].intent_type in ['recommend', 'compare']:
            return 'intermediate'
        else:
            return 'basic'

    def _apply_user_context(self, processed_query: ProcessedQuery, user_context: Dict[str, Any]):
        """Apply user context to enhance query processing"""

        # Apply user preferences
        if 'preferred_complexity' in user_context:
            processed_query.filters['complexity_level'] = user_context['preferred_complexity']

        if 'preferred_timeframes' in user_context:
            processed_query.filters['timeframes'] = user_context['preferred_timeframes']

        if 'trading_style' in user_context:
            processed_query.filters['trading_style'] = user_context['trading_style']

        # Apply trading session context
        if 'current_session' in user_context:
            session = user_context['current_session']
            if session in ['london', 'newyork']:
                # High volatility sessions
                processed_query.expanded_terms.extend(['breakout', 'momentum'])
            elif session == 'asia':
                # Lower volatility session
                processed_query.expanded_terms.extend(['ranging', 'oscillator'])

    def create_search_variants(self, processed_query: ProcessedQuery) -> List[str]:
        """Create multiple search variants for better recall"""
        variants = []

        # Original normalized query
        variants.append(processed_query.normalized_query)

        # Intent-specific variants
        for intent in processed_query.intents[:2]:  # Top 2 intents
            if intent.intent_type == 'recommend':
                variants.append(f"best {processed_query.normalized_query}")
                variants.append(f"recommended {processed_query.normalized_query}")
            elif intent.intent_type == 'analyze':
                variants.append(f"{processed_query.normalized_query} performance")
                variants.append(f"{processed_query.normalized_query} analysis")

        # Expanded term combinations
        if len(processed_query.expanded_terms) > 1:
            # Combine key terms
            key_terms = processed_query.expanded_terms[:3]
            variants.append(' '.join(key_terms))

        # Context-specific variants
        if processed_query.filters:
            for key, value in processed_query.filters.items():
                if isinstance(value, str):
                    variants.append(f"{processed_query.normalized_query} {value}")

        return list(set(variants))  # Remove duplicates

# Test function
def test_query_processor():
    """Test the intelligent query processor"""
    processor = IntelligentQueryProcessor()

    test_queries = [
        "Find the best RSI indicator for scalping EURUSD",
        "Recommend momentum indicators for trending markets",
        "Compare EMA vs SMA for day trading",
        "What indicators work well for ranging markets?",
        "Show me advanced LuxAlgo indicators with high win rate",
        "Create a strategy combining trend and momentum for 4h timeframe"
    ]

    for query in test_queries:
        print(f"\n=== Processing Query: '{query}' ===")

        processed = processor.process_query(query)

        print(f"Normalized: {processed.normalized_query}")
        print(f"Query Type: {processed.query_type}")
        print(f"Complexity: {processed.complexity_level}")

        print("Intents:")
        for intent in processed.intents:
            print(f"  - {intent.intent_type} (confidence: {intent.confidence:.2f})")
            if intent.parameters:
                print(f"    Parameters: {intent.parameters}")
            if intent.entities:
                print(f"    Entities: {intent.entities}")

        print(f"Expanded Terms: {processed.expanded_terms[:5]}")
        print(f"Filters: {processed.filters}")

        # Test search variants
        variants = processor.create_search_variants(processed)
        print(f"Search Variants: {variants[:3]}")

if __name__ == "__main__":
    test_query_processor()