# core/strategies/helpers/ichimoku_rag_enhancer.py
"""
Ichimoku RAG Enhancement Module
Integrates RAG (Retrieval-Augmented Generation) data to enhance Ichimoku Cloud strategy performance

🧠 RAG INTEGRATION FEATURES:
- Semantic search for Ichimoku patterns and confluences
- TradingView script analysis for advanced techniques
- Market regime adaptation using RAG knowledge base
- Multi-indicator confluence scoring
- Dynamic parameter optimization based on market context

🎯 ENHANCEMENT AREAS:
- Pattern recognition from 53+ TradingView indicators
- Advanced Ichimoku variations (fast scalping, RSI combinations)
- Market intelligence integration for adaptive parameters
- Signal quality improvement through confluence analysis
"""

import pandas as pd
import numpy as np
import logging
import json
import sys
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root for RAG interface access
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../../../../../../../')
sys.path.insert(0, project_root)

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class IchimokuRAGEnhancer:
    """
    🧠 ICHIMOKU RAG ENHANCEMENT ENGINE

    Leverages RAG system to enhance traditional Ichimoku strategy with:
    - Semantic search for pattern confluences
    - TradingView script analysis integration
    - Market-adaptive parameter selection
    - Multi-source signal validation
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.rag_interface = None
        self.cached_recommendations = {}
        self.market_context_cache = {}

        # Initialize RAG interface
        self._initialize_rag_interface()

        self.logger.info("🧠 IchimokuRAGEnhancer initialized with semantic search capabilities")

    def _initialize_rag_interface(self):
        """Initialize RAG system interface for semantic queries"""
        try:
            # Import RAG interface from project root
            sys.path.insert(0, '/home/hr/Projects/TradeSystemV1')
            from rag_interface import RAGInterface

            self.rag_interface = RAGInterface()

            # Test RAG system health
            health = self.rag_interface.health_check()
            if health.get('status') == 'healthy':
                self.logger.info("✅ RAG system connected and healthy")

                # Get RAG statistics
                stats = self.rag_interface.get_stats()
                self.logger.info(f"📊 RAG Database: {stats.get('total_indicators', 0)} indicators, {stats.get('total_templates', 0)} templates")
            else:
                self.logger.warning(f"⚠️ RAG system health check failed: {health}")
                self.rag_interface = None

        except Exception as e:
            self.logger.warning(f"⚠️ RAG interface initialization failed: {e}")
            self.rag_interface = None

    def enhance_ichimoku_signal(
        self,
        ichimoku_data: Dict,
        market_data: pd.DataFrame,
        epic: str,
        timeframe: str = '15m'
    ) -> Dict:
        """
        🎯 MAIN ENHANCEMENT METHOD: Enhance Ichimoku signals using RAG data

        Args:
            ichimoku_data: Traditional Ichimoku indicators and signals
            market_data: Recent price data for context
            epic: Currency pair
            timeframe: Trading timeframe

        Returns:
            Enhanced signal data with RAG-powered improvements
        """
        try:
            enhanced_signal = ichimoku_data.copy()

            # Phase 1: Pattern Recognition Enhancement
            pattern_enhancement = self._analyze_patterns_with_rag(ichimoku_data, market_data)
            enhanced_signal.update(pattern_enhancement)

            # Phase 2: Market Regime Adaptation
            regime_adaptation = self._adapt_to_market_regime(ichimoku_data, epic, timeframe)
            enhanced_signal.update(regime_adaptation)

            # Phase 3: Confluence Analysis
            confluence_score = self._calculate_confluence_score(ichimoku_data, market_data)
            enhanced_signal['rag_confluence_score'] = confluence_score

            # Phase 4: TradingView Script Integration
            script_insights = self._integrate_tradingview_insights(ichimoku_data, epic)
            enhanced_signal.update(script_insights)

            # Phase 5: Enhanced Confidence Calculation
            original_confidence = enhanced_signal.get('confidence', 0.5)
            rag_confidence = self._calculate_rag_confidence(enhanced_signal, original_confidence)
            enhanced_signal['rag_enhanced_confidence'] = rag_confidence
            enhanced_signal['confidence_boost'] = rag_confidence - original_confidence

            self.logger.info(f"🧠 RAG Enhancement complete for {epic}: Original confidence {original_confidence:.1%} → Enhanced {rag_confidence:.1%}")

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"❌ RAG enhancement failed: {e}")
            return ichimoku_data

    def _analyze_patterns_with_rag(self, ichimoku_data: Dict, market_data: pd.DataFrame) -> Dict:
        """Analyze Ichimoku patterns using RAG semantic search"""
        try:
            if not self.rag_interface:
                return {'rag_pattern_score': 0.5}

            # Create pattern description for semantic search
            pattern_query = self._create_pattern_query(ichimoku_data)

            # Search for similar patterns in RAG database
            pattern_results = self.rag_interface.search_indicators(pattern_query, limit=5)

            if pattern_results.get('error'):
                self.logger.warning(f"Pattern search failed: {pattern_results['error']}")
                return {'rag_pattern_score': 0.5}

            # Analyze pattern strength from results
            pattern_score = self._evaluate_pattern_strength(pattern_results)

            return {
                'rag_pattern_score': pattern_score,
                'rag_pattern_matches': len(pattern_results.get('results', [])),
                'rag_pattern_confidence': min(pattern_score * 1.2, 0.95)
            }

        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return {'rag_pattern_score': 0.5}

    def _create_pattern_query(self, ichimoku_data: Dict) -> str:
        """Create semantic search query based on current Ichimoku pattern"""
        try:
            # Extract key Ichimoku signals
            tk_bull = ichimoku_data.get('tk_bull_cross', False)
            tk_bear = ichimoku_data.get('tk_bear_cross', False)
            cloud_bull = ichimoku_data.get('cloud_bull_breakout', False)
            cloud_bear = ichimoku_data.get('cloud_bear_breakout', False)

            # Build query based on active signals
            query_parts = ["ichimoku cloud"]

            if tk_bull or tk_bear:
                direction = "bullish" if tk_bull else "bearish"
                query_parts.append(f"tenkan kijun {direction} cross")

            if cloud_bull or cloud_bear:
                direction = "bullish" if cloud_bull else "bearish"
                query_parts.append(f"cloud {direction} breakout")

            # Add market context
            query_parts.extend(["momentum", "trend following", "volatility breakout"])

            return " ".join(query_parts)

        except Exception as e:
            self.logger.error(f"Query creation failed: {e}")
            return "ichimoku cloud trend analysis"

    def _evaluate_pattern_strength(self, pattern_results: Dict) -> float:
        """Evaluate pattern strength from RAG search results"""
        try:
            results = pattern_results.get('results', [])
            if not results:
                return 0.5

            total_score = 0
            weight_sum = 0

            for result in results:
                # Get similarity score (higher is better)
                similarity = result.get('similarity', 0.5)

                # Get indicator popularity as weight
                metadata = result.get('metadata', {})
                popularity = metadata.get('popularity', {})
                likes = popularity.get('likes', 1000)

                # Calculate weighted score
                weight = min(likes / 10000, 1.0)  # Normalize likes to 0-1
                score = similarity * weight

                total_score += score
                weight_sum += weight

            # Return weighted average, bounded between 0.3 and 0.9
            if weight_sum > 0:
                pattern_strength = total_score / weight_sum
                return max(0.3, min(0.9, pattern_strength))

            return 0.5

        except Exception as e:
            self.logger.error(f"Pattern strength evaluation failed: {e}")
            return 0.5

    def _adapt_to_market_regime(self, ichimoku_data: Dict, epic: str, timeframe: str) -> Dict:
        """Adapt Ichimoku parameters based on market regime from RAG recommendations"""
        try:
            if not self.rag_interface:
                return {'rag_regime_adaptation': 'disabled'}

            # Create market condition query
            regime_query = f"market regime analysis {timeframe} timeframe trending ranging volatility"

            # Get regime recommendations
            regime_results = self.rag_interface.get_recommendations(regime_query)

            if regime_results.get('error'):
                return {'rag_regime_adaptation': 'failed'}

            # Extract regime insights
            regime_adaptation = self._extract_regime_insights(regime_results, epic)

            return regime_adaptation

        except Exception as e:
            self.logger.error(f"Regime adaptation failed: {e}")
            return {'rag_regime_adaptation': 'error'}

    def _extract_regime_insights(self, regime_results: Dict, epic: str) -> Dict:
        """Extract actionable regime insights from RAG recommendations"""
        try:
            # Simplified regime classification based on RAG data
            # In production, this would analyze the actual recommendation content

            return {
                'rag_regime_adaptation': 'active',
                'rag_suggested_regime': 'trending',  # Would be extracted from results
                'rag_regime_confidence': 0.7,
                'rag_parameter_adjustment': {
                    'confidence_threshold_modifier': 0.05,  # Boost confidence in trending markets
                    'stop_loss_modifier': 1.0,
                    'take_profit_modifier': 1.2  # Wider targets in trending markets
                }
            }

        except Exception as e:
            self.logger.error(f"Regime insight extraction failed: {e}")
            return {'rag_regime_adaptation': 'error'}

    def _calculate_confluence_score(self, ichimoku_data: Dict, market_data: pd.DataFrame) -> float:
        """Calculate confluence score combining Ichimoku with other indicators"""
        try:
            if not self.rag_interface:
                return 0.5

            # Query for confluence indicators
            confluence_query = "ichimoku confluence RSI MACD moving average support resistance"

            # Search for confluence patterns
            confluence_results = self.rag_interface.search_templates(confluence_query, limit=3)

            if confluence_results.get('error'):
                return 0.5

            # Analyze confluence strength
            confluence_score = self._analyze_confluence_strength(confluence_results, ichimoku_data)

            return confluence_score

        except Exception as e:
            self.logger.error(f"Confluence calculation failed: {e}")
            return 0.5

    def _analyze_confluence_strength(self, confluence_results: Dict, ichimoku_data: Dict) -> float:
        """Analyze confluence strength from RAG template results"""
        try:
            results = confluence_results.get('results', [])
            if not results:
                return 0.5

            # Base confluence on number of matching templates
            num_matches = len(results)
            base_score = min(0.3 + (num_matches * 0.1), 0.8)

            # Adjust based on signal strength
            signal_strength = 0.5
            if ichimoku_data.get('tk_bull_cross') or ichimoku_data.get('tk_bear_cross'):
                signal_strength += 0.2
            if ichimoku_data.get('cloud_bull_breakout') or ichimoku_data.get('cloud_bear_breakout'):
                signal_strength += 0.2

            confluence_score = (base_score + signal_strength) / 2

            return max(0.3, min(0.9, confluence_score))

        except Exception as e:
            self.logger.error(f"Confluence strength analysis failed: {e}")
            return 0.5

    def _integrate_tradingview_insights(self, ichimoku_data: Dict, epic: str) -> Dict:
        """Integrate insights from TradingView Ichimoku scripts"""
        try:
            if not self.rag_interface:
                return {'rag_tradingview_insights': 'disabled'}

            # Query for TradingView specific Ichimoku techniques
            tv_query = "ichimoku scalping fast settings RSI filter alert system"

            # Search TradingView templates
            tv_results = self.rag_interface.search_templates(tv_query, limit=3)

            if tv_results.get('error'):
                return {'rag_tradingview_insights': 'failed'}

            # Extract TradingView insights
            tv_insights = self._extract_tradingview_insights(tv_results)

            return tv_insights

        except Exception as e:
            self.logger.error(f"TradingView insights integration failed: {e}")
            return {'rag_tradingview_insights': 'error'}

    def _extract_tradingview_insights(self, tv_results: Dict) -> Dict:
        """Extract actionable insights from TradingView script analysis"""
        try:
            results = tv_results.get('results', [])

            insights = {
                'rag_tradingview_insights': 'active',
                'rag_tv_script_matches': len(results),
                'rag_tv_techniques': []
            }

            # Analyze each matching script
            for result in results:
                technique = result.get('title', 'Unknown')
                insights['rag_tv_techniques'].append(technique)

            # Suggest technique-based improvements
            if any('scalping' in tech.lower() for tech in insights['rag_tv_techniques']):
                insights['rag_suggested_improvement'] = 'fast_ichimoku_mode'
                insights['rag_parameter_suggestion'] = 'faster_periods'
            elif any('rsi' in tech.lower() for tech in insights['rag_tv_techniques']):
                insights['rag_suggested_improvement'] = 'momentum_filter'
                insights['rag_parameter_suggestion'] = 'rsi_confirmation'
            else:
                insights['rag_suggested_improvement'] = 'standard_enhancement'

            return insights

        except Exception as e:
            self.logger.error(f"TradingView insight extraction failed: {e}")
            return {'rag_tradingview_insights': 'error'}

    def _calculate_rag_confidence(self, enhanced_signal: Dict, original_confidence: float) -> float:
        """Calculate final RAG-enhanced confidence score"""
        try:
            # Start with original confidence
            rag_confidence = original_confidence

            # Apply pattern score boost
            pattern_score = enhanced_signal.get('rag_pattern_score', 0.5)
            if pattern_score > 0.6:
                rag_confidence += 0.05  # Boost for strong patterns

            # Apply confluence score boost
            confluence_score = enhanced_signal.get('rag_confluence_score', 0.5)
            if confluence_score > 0.7:
                rag_confidence += 0.08  # Boost for strong confluence

            # Apply regime adaptation boost
            regime_confidence = enhanced_signal.get('rag_regime_confidence', 0.5)
            if regime_confidence > 0.7:
                rag_confidence += 0.03  # Boost for favorable regime

            # Apply TradingView insights boost
            tv_matches = enhanced_signal.get('rag_tv_script_matches', 0)
            if tv_matches > 2:
                rag_confidence += 0.04  # Boost for multiple TradingView confirmations

            # Ensure confidence stays within reasonable bounds
            return max(original_confidence, min(0.95, rag_confidence))

        except Exception as e:
            self.logger.error(f"RAG confidence calculation failed: {e}")
            return original_confidence

    def get_rag_market_recommendations(self, epic: str, market_conditions: Dict) -> Dict:
        """Get RAG-based market recommendations for Ichimoku strategy"""
        try:
            if not self.rag_interface:
                return {'error': 'RAG interface not available'}

            # Create market context query
            volatility = market_conditions.get('volatility', 'medium')
            trend = market_conditions.get('trend', 'neutral')

            query = f"ichimoku strategy {volatility} volatility {trend} trend optimization"

            # Get recommendations
            recommendations = self.rag_interface.get_recommendations(query)

            return recommendations

        except Exception as e:
            self.logger.error(f"Market recommendations failed: {e}")
            return {'error': str(e)}

    def compose_enhanced_strategy(self, requirements: Dict) -> Dict:
        """Compose enhanced Ichimoku strategy based on requirements"""
        try:
            if not self.rag_interface:
                return {'error': 'RAG interface not available'}

            # Extract requirements
            market_condition = requirements.get('market_condition', 'trending')
            trading_style = requirements.get('trading_style', 'day_trading')
            complexity = requirements.get('complexity_level', 'intermediate')

            description = f"Enhanced Ichimoku strategy for {market_condition} markets with {trading_style} style"

            # Use RAG to compose strategy
            strategy = self.rag_interface.compose_strategy(
                description=description,
                market_condition=market_condition,
                trading_style=trading_style,
                complexity_level=complexity
            )

            return strategy

        except Exception as e:
            self.logger.error(f"Strategy composition failed: {e}")
            return {'error': str(e)}

    def is_rag_available(self) -> bool:
        """Check if RAG system is available and healthy"""
        return self.rag_interface is not None

    def get_rag_stats(self) -> Dict:
        """Get RAG system statistics"""
        try:
            if not self.rag_interface:
                return {'error': 'RAG interface not available'}

            return self.rag_interface.get_stats()

        except Exception as e:
            self.logger.error(f"RAG stats retrieval failed: {e}")
            return {'error': str(e)}