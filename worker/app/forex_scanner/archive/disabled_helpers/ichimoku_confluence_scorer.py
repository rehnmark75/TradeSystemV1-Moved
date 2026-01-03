# core/strategies/helpers/ichimoku_confluence_scorer.py
"""
Ichimoku Confluence Scoring System
Advanced confluence analysis using RAG-sourced indicators and TradingView techniques

🔗 CONFLUENCE ANALYSIS FEATURES:
- Multi-indicator confluence detection using RAG database
- TradingView script pattern matching for confirmation signals
- Weighted scoring based on indicator reliability and popularity
- Market regime-aware confluence adjustments
- Session-specific confluence patterns

📊 CONFLUENCE INDICATORS:
- Traditional: RSI, MACD, Bollinger Bands, Moving Averages
- Advanced: Stochastic, Williams %R, CCI, Momentum indicators
- Volume: Volume Profile, OBV, Accumulation/Distribution
- Support/Resistance: Pivot Points, Fibonacci levels, Key levels
- Pattern: Candlestick patterns, Chart patterns, Price action
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


@dataclass
class ConfluenceIndicator:
    """Represents a confluence indicator with its signal and weight"""
    name: str
    indicator_type: str  # 'momentum', 'trend', 'volume', 'support_resistance', 'pattern'
    signal_direction: str  # 'BULL', 'BEAR', 'NEUTRAL'
    strength: float  # 0.0 to 1.0
    weight: float  # Importance weight
    source: str  # 'rag_search', 'tradingview_script', 'calculated', 'manual'
    timeframe_compatibility: List[str]
    market_regime_preference: List[str]
    confidence: float = 0.7


@dataclass
class ConfluenceScore:
    """Confluence scoring results"""
    total_score: float
    bull_score: float
    bear_score: float
    neutral_score: float
    indicator_count: int
    high_confidence_indicators: int
    weighted_strength: float
    regime_adjustment: float
    session_adjustment: float
    confluence_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'


class IchimokuConfluenceScorer:
    """
    🔗 ADVANCED CONFLUENCE SCORING ENGINE

    Comprehensive confluence analysis system that:
    - Leverages RAG database for indicator discovery
    - Analyzes TradingView scripts for pattern confirmation
    - Applies market regime and session-specific weighting
    - Provides detailed confluence breakdown and scoring
    """

    def __init__(self, rag_enhancer=None, tradingview_parser=None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.rag_enhancer = rag_enhancer
        self.tradingview_parser = tradingview_parser

        # Confluence indicator weights by type
        self.indicator_weights = {
            'momentum': 0.25,      # RSI, MACD, Stochastic
            'trend': 0.30,         # Moving averages, trend lines
            'volume': 0.15,        # Volume indicators
            'support_resistance': 0.20,  # Key levels, S/R
            'pattern': 0.10        # Candlestick and chart patterns
        }

        # Market regime multipliers
        self.regime_multipliers = {
            'trending': {'trend': 1.3, 'momentum': 1.1, 'volume': 0.9, 'support_resistance': 0.8, 'pattern': 1.0},
            'ranging': {'trend': 0.7, 'momentum': 1.2, 'volume': 1.0, 'support_resistance': 1.4, 'pattern': 1.2},
            'breakout': {'trend': 1.2, 'momentum': 1.3, 'volume': 1.5, 'support_resistance': 1.2, 'pattern': 0.9},
            'high_volatility': {'trend': 0.9, 'momentum': 0.8, 'volume': 1.1, 'support_resistance': 1.0, 'pattern': 0.8},
            'low_volatility': {'trend': 1.1, 'momentum': 1.2, 'volume': 0.9, 'support_resistance': 1.1, 'pattern': 1.1}
        }

        # Session preference multipliers
        self.session_multipliers = {
            'asian': {'momentum': 1.1, 'support_resistance': 1.2},
            'london': {'trend': 1.2, 'volume': 1.1},
            'new_york': {'momentum': 1.1, 'pattern': 1.1}
        }

        self.logger.info("🔗 Ichimoku Confluence Scorer initialized")

    def calculate_confluence_score(
        self,
        ichimoku_signal: Dict,
        market_data: pd.DataFrame,
        epic: str,
        market_conditions: Dict = None
    ) -> ConfluenceScore:
        """
        🎯 CALCULATE CONFLUENCE SCORE: Comprehensive confluence analysis

        Args:
            ichimoku_signal: Base Ichimoku signal data
            market_data: Historical market data for analysis
            epic: Currency pair
            market_conditions: Current market conditions

        Returns:
            Comprehensive confluence scoring results
        """
        try:
            self.logger.info(f"🔗 Calculating confluence score for {epic} Ichimoku signal...")

            # Discover confluence indicators from multiple sources
            confluence_indicators = self._discover_confluence_indicators(
                ichimoku_signal, market_data, epic, market_conditions
            )

            self.logger.info(f"🔗 Found {len(confluence_indicators)} confluence indicators")

            # Calculate raw confluence scores
            raw_scores = self._calculate_raw_scores(confluence_indicators, ichimoku_signal)

            # Apply market regime adjustments
            regime_adjusted_scores = self._apply_regime_adjustments(
                raw_scores, confluence_indicators, market_conditions
            )

            # Apply session adjustments
            final_scores = self._apply_session_adjustments(
                regime_adjusted_scores, confluence_indicators, market_conditions
            )

            # Generate final confluence score
            confluence_score = self._generate_confluence_score(
                final_scores, confluence_indicators, market_conditions
            )

            self.logger.info(f"🔗 Confluence analysis complete for {epic}: "
                           f"Level={confluence_score.confluence_level}, "
                           f"Score={confluence_score.total_score:.3f}, "
                           f"Indicators={confluence_score.indicator_count}")

            return confluence_score

        except Exception as e:
            self.logger.error(f"Confluence score calculation failed: {e}")
            return self._get_fallback_confluence_score()

    def _discover_confluence_indicators(
        self,
        ichimoku_signal: Dict,
        market_data: pd.DataFrame,
        epic: str,
        market_conditions: Dict = None
    ) -> List[ConfluenceIndicator]:
        """Discover confluence indicators from multiple sources"""
        indicators = []

        try:
            # Source 1: RAG-based indicator discovery
            rag_indicators = self._discover_rag_indicators(ichimoku_signal, epic, market_conditions)
            indicators.extend(rag_indicators)

            # Source 2: TradingView script analysis
            tv_indicators = self._discover_tradingview_indicators(ichimoku_signal, epic, market_conditions)
            indicators.extend(tv_indicators)

            # Source 3: Calculated technical indicators
            calculated_indicators = self._calculate_technical_indicators(ichimoku_signal, market_data)
            indicators.extend(calculated_indicators)

            # Source 4: Pattern recognition indicators
            pattern_indicators = self._discover_pattern_indicators(ichimoku_signal, market_data)
            indicators.extend(pattern_indicators)

            # Remove duplicates and normalize
            indicators = self._normalize_indicators(indicators)

            return indicators

        except Exception as e:
            self.logger.error(f"Indicator discovery failed: {e}")
            return []

    def _discover_rag_indicators(
        self,
        ichimoku_signal: Dict,
        epic: str,
        market_conditions: Dict = None
    ) -> List[ConfluenceIndicator]:
        """Discover indicators using RAG semantic search"""
        indicators = []

        try:
            if not self.rag_enhancer or not self.rag_enhancer.rag_interface:
                return indicators

            # Create semantic search queries for different indicator types
            signal_type = ichimoku_signal.get('signal_type', 'UNKNOWN')
            queries = self._create_rag_queries(signal_type, epic, market_conditions)

            for query_type, query in queries.items():
                try:
                    # Search RAG database
                    search_results = self.rag_enhancer.rag_interface.search_indicators(query, limit=3)

                    if not search_results.get('error'):
                        # Convert search results to confluence indicators
                        query_indicators = self._convert_rag_results_to_indicators(
                            search_results, query_type, signal_type
                        )
                        indicators.extend(query_indicators)

                except Exception as e:
                    self.logger.warning(f"RAG query failed for {query_type}: {e}")
                    continue

            return indicators

        except Exception as e:
            self.logger.error(f"RAG indicator discovery failed: {e}")
            return []

    def _create_rag_queries(self, signal_type: str, epic: str, market_conditions: Dict = None) -> Dict[str, str]:
        """Create semantic search queries for different indicator types"""
        try:
            direction = "bullish" if signal_type == 'BULL' else "bearish" if signal_type == 'BEAR' else "neutral"
            regime = market_conditions.get('regime', 'trending') if market_conditions else 'trending'

            queries = {
                'momentum': f"{direction} momentum indicators RSI MACD stochastic {regime} markets {epic}",
                'trend': f"{direction} trend following moving average EMA SMA trend line {regime}",
                'volume': f"volume analysis OBV accumulation distribution volume profile {direction}",
                'support_resistance': f"support resistance levels pivot points fibonacci {direction} breakout",
                'pattern': f"{direction} candlestick patterns chart patterns price action {regime}"
            }

            return queries

        except Exception:
            return {
                'momentum': f"momentum indicators RSI MACD",
                'trend': f"trend indicators moving average",
                'volume': f"volume indicators",
                'support_resistance': f"support resistance levels",
                'pattern': f"chart patterns"
            }

    def _convert_rag_results_to_indicators(
        self,
        search_results: Dict,
        query_type: str,
        signal_type: str
    ) -> List[ConfluenceIndicator]:
        """Convert RAG search results to confluence indicators"""
        indicators = []

        try:
            results = search_results.get('results', [])

            for result in results:
                # Extract indicator information
                title = result.get('title', 'Unknown Indicator')
                similarity = result.get('similarity', 0.5)
                metadata = result.get('metadata', {})

                # Calculate indicator properties
                popularity = metadata.get('popularity', {})
                likes = popularity.get('likes', 1000)
                weight = min(likes / 10000, 1.0)  # Normalize to 0-1

                # Determine signal strength from similarity and metadata
                strength = min(similarity * 1.2, 1.0)

                # Create confluence indicator
                indicator = ConfluenceIndicator(
                    name=title,
                    indicator_type=query_type,
                    signal_direction=signal_type,
                    strength=strength,
                    weight=weight,
                    source='rag_search',
                    timeframe_compatibility=metadata.get('timeframes', ['15m', '1h', '4h']),
                    market_regime_preference=metadata.get('market_context', ['trending']).split(),
                    confidence=similarity
                )

                indicators.append(indicator)

            return indicators

        except Exception as e:
            self.logger.error(f"RAG result conversion failed: {e}")
            return []

    def _discover_tradingview_indicators(
        self,
        ichimoku_signal: Dict,
        epic: str,
        market_conditions: Dict = None
    ) -> List[ConfluenceIndicator]:
        """Discover indicators from TradingView script analysis"""
        indicators = []

        try:
            if not self.tradingview_parser:
                return indicators

            # Get TradingView variations and insights
            market_type = market_conditions.get('regime', 'trending') if market_conditions else 'trending'
            timeframe = market_conditions.get('timeframe', '15m') if market_conditions else '15m'

            variations = self.tradingview_parser.get_best_variations_for_market(market_type, timeframe)

            for variation in variations[:3]:  # Top 3 variations
                # Convert variation to confluence indicator
                signal_type = ichimoku_signal.get('signal_type', 'UNKNOWN')

                # Analyze filters for confluence signals
                for filter_name in variation.additional_filters:
                    indicator_type = self._classify_filter_type(filter_name)

                    indicator = ConfluenceIndicator(
                        name=f"{variation.name} - {filter_name}",
                        indicator_type=indicator_type,
                        signal_direction=signal_type,
                        strength=variation.confidence_score,
                        weight=0.8,  # High weight for TradingView variations
                        source='tradingview_script',
                        timeframe_compatibility=[timeframe],
                        market_regime_preference=variation.market_conditions,
                        confidence=variation.confidence_score
                    )

                    indicators.append(indicator)

            return indicators

        except Exception as e:
            self.logger.error(f"TradingView indicator discovery failed: {e}")
            return []

    def _classify_filter_type(self, filter_name: str) -> str:
        """Classify filter type based on name"""
        filter_name_lower = filter_name.lower()

        if any(keyword in filter_name_lower for keyword in ['rsi', 'momentum', 'stochastic', 'macd']):
            return 'momentum'
        elif any(keyword in filter_name_lower for keyword in ['volume', 'obv', 'accumulation']):
            return 'volume'
        elif any(keyword in filter_name_lower for keyword in ['support', 'resistance', 'level', 'pivot']):
            return 'support_resistance'
        elif any(keyword in filter_name_lower for keyword in ['pattern', 'candlestick']):
            return 'pattern'
        else:
            return 'trend'

    def _calculate_technical_indicators(
        self,
        ichimoku_signal: Dict,
        market_data: pd.DataFrame
    ) -> List[ConfluenceIndicator]:
        """Calculate technical indicators for confluence analysis"""
        indicators = []

        try:
            if len(market_data) < 50:
                return indicators

            latest_data = market_data.iloc[-1]
            signal_type = ichimoku_signal.get('signal_type', 'UNKNOWN')

            # RSI confluence
            rsi_indicator = self._calculate_rsi_confluence(market_data, signal_type)
            if rsi_indicator:
                indicators.append(rsi_indicator)

            # Moving Average confluence
            ma_indicator = self._calculate_ma_confluence(market_data, signal_type)
            if ma_indicator:
                indicators.append(ma_indicator)

            # Volume confluence
            volume_indicator = self._calculate_volume_confluence(market_data, signal_type)
            if volume_indicator:
                indicators.append(volume_indicator)

            return indicators

        except Exception as e:
            self.logger.error(f"Technical indicator calculation failed: {e}")
            return []

    def _calculate_rsi_confluence(self, market_data: pd.DataFrame, signal_type: str) -> Optional[ConfluenceIndicator]:
        """Calculate RSI confluence indicator"""
        try:
            if len(market_data) < 20:
                return None

            # Simple RSI calculation
            closes = market_data['close'].values
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gains = pd.Series(gains).rolling(window=14).mean().iloc[-1]
            avg_losses = pd.Series(losses).rolling(window=14).mean().iloc[-1]

            if avg_losses == 0:
                rsi = 100
            else:
                rs = avg_gains / avg_losses
                rsi = 100 - (100 / (1 + rs))

            # Determine confluence strength
            if signal_type == 'BULL':
                if rsi > 30 and rsi < 70:  # Good bullish zone
                    strength = min((rsi - 30) / 40, 1.0)
                else:
                    strength = 0.2  # Weak confluence
            elif signal_type == 'BEAR':
                if rsi > 30 and rsi < 70:  # Good bearish zone
                    strength = min((70 - rsi) / 40, 1.0)
                else:
                    strength = 0.2  # Weak confluence
            else:
                strength = 0.5

            return ConfluenceIndicator(
                name="RSI Confluence",
                indicator_type="momentum",
                signal_direction=signal_type,
                strength=strength,
                weight=0.7,
                source="calculated",
                timeframe_compatibility=['15m', '1h', '4h'],
                market_regime_preference=['trending', 'ranging'],
                confidence=0.8
            )

        except Exception:
            return None

    def _calculate_ma_confluence(self, market_data: pd.DataFrame, signal_type: str) -> Optional[ConfluenceIndicator]:
        """Calculate Moving Average confluence indicator"""
        try:
            if len(market_data) < 50:
                return None

            # Calculate EMAs
            ema_20 = market_data['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = market_data['close'].ewm(span=50).mean().iloc[-1]
            current_price = market_data['close'].iloc[-1]

            # Determine confluence
            if signal_type == 'BULL':
                if current_price > ema_20 > ema_50:
                    strength = 0.9  # Strong bullish alignment
                elif current_price > ema_20:
                    strength = 0.6  # Moderate bullish
                else:
                    strength = 0.2  # Weak
            elif signal_type == 'BEAR':
                if current_price < ema_20 < ema_50:
                    strength = 0.9  # Strong bearish alignment
                elif current_price < ema_20:
                    strength = 0.6  # Moderate bearish
                else:
                    strength = 0.2  # Weak
            else:
                strength = 0.5

            return ConfluenceIndicator(
                name="EMA Confluence",
                indicator_type="trend",
                signal_direction=signal_type,
                strength=strength,
                weight=0.8,
                source="calculated",
                timeframe_compatibility=['15m', '1h', '4h', '1d'],
                market_regime_preference=['trending'],
                confidence=0.8
            )

        except Exception:
            return None

    def _calculate_volume_confluence(self, market_data: pd.DataFrame, signal_type: str) -> Optional[ConfluenceIndicator]:
        """Calculate Volume confluence indicator"""
        try:
            if len(market_data) < 20 or 'ltv' not in market_data.columns:
                return None

            # Compare recent volume to average
            recent_volume = market_data['ltv'].iloc[-5:].mean()
            avg_volume = market_data['ltv'].iloc[-20:].mean()

            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # Volume confluence based on direction
            if volume_ratio > 1.2:  # Above average volume
                strength = min((volume_ratio - 1.0) / 1.0, 1.0)
            else:
                strength = 0.3  # Low volume confluence

            return ConfluenceIndicator(
                name="Volume Confluence",
                indicator_type="volume",
                signal_direction=signal_type,
                strength=strength,
                weight=0.6,
                source="calculated",
                timeframe_compatibility=['15m', '1h', '4h'],
                market_regime_preference=['breakout', 'trending'],
                confidence=0.7
            )

        except Exception:
            return None

    def _discover_pattern_indicators(
        self,
        ichimoku_signal: Dict,
        market_data: pd.DataFrame
    ) -> List[ConfluenceIndicator]:
        """Discover pattern-based confluence indicators"""
        indicators = []

        try:
            if len(market_data) < 10:
                return indicators

            signal_type = ichimoku_signal.get('signal_type', 'UNKNOWN')

            # Simple candlestick pattern detection
            pattern_indicator = self._detect_candlestick_patterns(market_data, signal_type)
            if pattern_indicator:
                indicators.append(pattern_indicator)

            return indicators

        except Exception as e:
            self.logger.error(f"Pattern indicator discovery failed: {e}")
            return []

    def _detect_candlestick_patterns(self, market_data: pd.DataFrame, signal_type: str) -> Optional[ConfluenceIndicator]:
        """Simple candlestick pattern detection"""
        try:
            if len(market_data) < 3:
                return None

            latest = market_data.iloc[-1]
            prev = market_data.iloc[-2]

            # Simple hammer/doji detection
            body_size = abs(latest['close'] - latest['open'])
            total_size = latest['high'] - latest['low']

            if total_size == 0:
                return None

            body_ratio = body_size / total_size

            # Doji pattern (small body)
            if body_ratio < 0.1:
                strength = 0.6
                pattern_name = "Doji Pattern"
            # Hammer pattern (small body, long lower wick)
            elif (latest['low'] < min(latest['open'], latest['close']) and
                  (min(latest['open'], latest['close']) - latest['low']) > 2 * body_size):
                strength = 0.7
                pattern_name = "Hammer Pattern"
            else:
                return None

            return ConfluenceIndicator(
                name=pattern_name,
                indicator_type="pattern",
                signal_direction=signal_type,
                strength=strength,
                weight=0.5,
                source="calculated",
                timeframe_compatibility=['15m', '1h', '4h'],
                market_regime_preference=['ranging', 'reversal'],
                confidence=0.6
            )

        except Exception:
            return None

    def _normalize_indicators(self, indicators: List[ConfluenceIndicator]) -> List[ConfluenceIndicator]:
        """Remove duplicates and normalize indicator list"""
        try:
            # Remove exact duplicates
            seen_names = set()
            normalized = []

            for indicator in indicators:
                if indicator.name not in seen_names:
                    normalized.append(indicator)
                    seen_names.add(indicator.name)

            # Sort by confidence and strength
            normalized.sort(key=lambda x: (x.confidence * x.strength), reverse=True)

            # Limit to top 10 indicators
            return normalized[:10]

        except Exception:
            return indicators

    def _calculate_raw_scores(
        self,
        indicators: List[ConfluenceIndicator],
        ichimoku_signal: Dict
    ) -> Dict[str, float]:
        """Calculate raw confluence scores by direction"""
        try:
            bull_score = 0.0
            bear_score = 0.0
            neutral_score = 0.0
            total_weight = 0.0

            for indicator in indicators:
                # Apply indicator type weight
                type_weight = self.indicator_weights.get(indicator.indicator_type, 0.2)
                indicator_contribution = indicator.strength * indicator.weight * type_weight

                if indicator.signal_direction == 'BULL':
                    bull_score += indicator_contribution
                elif indicator.signal_direction == 'BEAR':
                    bear_score += indicator_contribution
                else:
                    neutral_score += indicator_contribution

                total_weight += indicator.weight * type_weight

            # Normalize scores
            if total_weight > 0:
                bull_score /= total_weight
                bear_score /= total_weight
                neutral_score /= total_weight

            return {
                'bull_score': min(bull_score, 1.0),
                'bear_score': min(bear_score, 1.0),
                'neutral_score': min(neutral_score, 1.0),
                'total_weight': total_weight
            }

        except Exception as e:
            self.logger.error(f"Raw score calculation failed: {e}")
            return {'bull_score': 0.5, 'bear_score': 0.5, 'neutral_score': 0.5, 'total_weight': 1.0}

    def _apply_regime_adjustments(
        self,
        raw_scores: Dict,
        indicators: List[ConfluenceIndicator],
        market_conditions: Dict = None
    ) -> Dict[str, float]:
        """Apply market regime adjustments to scores"""
        try:
            if not market_conditions:
                return raw_scores

            regime = market_conditions.get('regime', 'trending')
            regime_multipliers = self.regime_multipliers.get(regime, {})

            adjusted_scores = raw_scores.copy()
            adjustment_factor = 0.0

            # Calculate weighted adjustment based on indicator types
            for indicator in indicators:
                type_multiplier = regime_multipliers.get(indicator.indicator_type, 1.0)
                type_weight = self.indicator_weights.get(indicator.indicator_type, 0.2)
                adjustment_factor += (type_multiplier - 1.0) * type_weight

            # Apply adjustment (limit to ±20%)
            adjustment_factor = max(-0.2, min(0.2, adjustment_factor))

            for key in ['bull_score', 'bear_score', 'neutral_score']:
                if key in adjusted_scores:
                    adjusted_scores[key] *= (1.0 + adjustment_factor)
                    adjusted_scores[key] = max(0.0, min(1.0, adjusted_scores[key]))

            adjusted_scores['regime_adjustment'] = adjustment_factor

            return adjusted_scores

        except Exception as e:
            self.logger.error(f"Regime adjustment failed: {e}")
            return raw_scores

    def _apply_session_adjustments(
        self,
        regime_adjusted_scores: Dict,
        indicators: List[ConfluenceIndicator],
        market_conditions: Dict = None
    ) -> Dict[str, float]:
        """Apply trading session adjustments to scores"""
        try:
            if not market_conditions:
                return regime_adjusted_scores

            session = market_conditions.get('trading_session', 'london')
            session_multipliers = self.session_multipliers.get(session, {})

            adjusted_scores = regime_adjusted_scores.copy()
            session_adjustment = 0.0

            # Calculate session-based adjustment
            for indicator in indicators:
                session_multiplier = session_multipliers.get(indicator.indicator_type, 1.0)
                type_weight = self.indicator_weights.get(indicator.indicator_type, 0.2)
                session_adjustment += (session_multiplier - 1.0) * type_weight

            # Apply session adjustment (limit to ±10%)
            session_adjustment = max(-0.1, min(0.1, session_adjustment))

            for key in ['bull_score', 'bear_score', 'neutral_score']:
                if key in adjusted_scores:
                    adjusted_scores[key] *= (1.0 + session_adjustment)
                    adjusted_scores[key] = max(0.0, min(1.0, adjusted_scores[key]))

            adjusted_scores['session_adjustment'] = session_adjustment

            return adjusted_scores

        except Exception as e:
            self.logger.error(f"Session adjustment failed: {e}")
            return regime_adjusted_scores

    def _generate_confluence_score(
        self,
        final_scores: Dict,
        indicators: List[ConfluenceIndicator],
        market_conditions: Dict = None
    ) -> ConfluenceScore:
        """Generate final confluence score object"""
        try:
            bull_score = final_scores.get('bull_score', 0.5)
            bear_score = final_scores.get('bear_score', 0.5)
            neutral_score = final_scores.get('neutral_score', 0.5)

            # Calculate total score
            total_score = max(bull_score, bear_score, neutral_score)

            # Calculate weighted strength
            weighted_strength = sum(
                indicator.strength * indicator.weight for indicator in indicators
            ) / len(indicators) if indicators else 0.5

            # Count high confidence indicators
            high_confidence_count = sum(
                1 for indicator in indicators if indicator.confidence > 0.7
            )

            # Determine confluence level
            if total_score >= 0.8 and high_confidence_count >= 3:
                confluence_level = 'VERY_HIGH'
            elif total_score >= 0.7 and high_confidence_count >= 2:
                confluence_level = 'HIGH'
            elif total_score >= 0.5:
                confluence_level = 'MEDIUM'
            else:
                confluence_level = 'LOW'

            return ConfluenceScore(
                total_score=total_score,
                bull_score=bull_score,
                bear_score=bear_score,
                neutral_score=neutral_score,
                indicator_count=len(indicators),
                high_confidence_indicators=high_confidence_count,
                weighted_strength=weighted_strength,
                regime_adjustment=final_scores.get('regime_adjustment', 0.0),
                session_adjustment=final_scores.get('session_adjustment', 0.0),
                confluence_level=confluence_level
            )

        except Exception as e:
            self.logger.error(f"Confluence score generation failed: {e}")
            return self._get_fallback_confluence_score()

    def _get_fallback_confluence_score(self) -> ConfluenceScore:
        """Fallback confluence score when calculation fails"""
        return ConfluenceScore(
            total_score=0.5,
            bull_score=0.5,
            bear_score=0.5,
            neutral_score=0.5,
            indicator_count=0,
            high_confidence_indicators=0,
            weighted_strength=0.5,
            regime_adjustment=0.0,
            session_adjustment=0.0,
            confluence_level='MEDIUM'
        )

    def get_confluence_breakdown(self, confluence_score: ConfluenceScore, indicators: List[ConfluenceIndicator]) -> Dict:
        """Get detailed breakdown of confluence analysis"""
        try:
            breakdown = {
                'overall_assessment': {
                    'confluence_level': confluence_score.confluence_level,
                    'total_score': confluence_score.total_score,
                    'directional_bias': self._determine_directional_bias(confluence_score),
                    'confidence_rating': self._calculate_confidence_rating(confluence_score)
                },
                'directional_scores': {
                    'bull_score': confluence_score.bull_score,
                    'bear_score': confluence_score.bear_score,
                    'neutral_score': confluence_score.neutral_score
                },
                'indicator_analysis': {
                    'total_indicators': confluence_score.indicator_count,
                    'high_confidence_indicators': confluence_score.high_confidence_indicators,
                    'weighted_strength': confluence_score.weighted_strength,
                    'indicator_breakdown': self._create_indicator_breakdown(indicators)
                },
                'adjustments': {
                    'regime_adjustment': confluence_score.regime_adjustment,
                    'session_adjustment': confluence_score.session_adjustment,
                    'net_adjustment': confluence_score.regime_adjustment + confluence_score.session_adjustment
                }
            }

            return breakdown

        except Exception as e:
            self.logger.error(f"Confluence breakdown generation failed: {e}")
            return {'error': str(e)}

    def _determine_directional_bias(self, confluence_score: ConfluenceScore) -> str:
        """Determine overall directional bias from confluence scores"""
        if confluence_score.bull_score > confluence_score.bear_score + 0.1:
            return 'BULLISH'
        elif confluence_score.bear_score > confluence_score.bull_score + 0.1:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _calculate_confidence_rating(self, confluence_score: ConfluenceScore) -> str:
        """Calculate overall confidence rating"""
        confidence_score = (
            confluence_score.total_score * 0.4 +
            confluence_score.weighted_strength * 0.3 +
            (confluence_score.high_confidence_indicators / max(confluence_score.indicator_count, 1)) * 0.3
        )

        if confidence_score >= 0.8:
            return 'VERY_HIGH'
        elif confidence_score >= 0.7:
            return 'HIGH'
        elif confidence_score >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _create_indicator_breakdown(self, indicators: List[ConfluenceIndicator]) -> Dict:
        """Create detailed breakdown of indicators by type"""
        breakdown = {}

        try:
            for indicator_type in self.indicator_weights.keys():
                type_indicators = [ind for ind in indicators if ind.indicator_type == indicator_type]

                if type_indicators:
                    breakdown[indicator_type] = {
                        'count': len(type_indicators),
                        'average_strength': sum(ind.strength for ind in type_indicators) / len(type_indicators),
                        'average_confidence': sum(ind.confidence for ind in type_indicators) / len(type_indicators),
                        'indicators': [
                            {
                                'name': ind.name,
                                'signal_direction': ind.signal_direction,
                                'strength': ind.strength,
                                'source': ind.source
                            }
                            for ind in type_indicators
                        ]
                    }

            return breakdown

        except Exception as e:
            self.logger.error(f"Indicator breakdown creation failed: {e}")
            return {}

    def is_confluence_favorable(self, confluence_score: ConfluenceScore, ichimoku_signal_type: str) -> bool:
        """Check if confluence is favorable for the Ichimoku signal"""
        try:
            if confluence_score.confluence_level in ['HIGH', 'VERY_HIGH']:
                if ichimoku_signal_type == 'BULL':
                    return confluence_score.bull_score > confluence_score.bear_score + 0.15
                elif ichimoku_signal_type == 'BEAR':
                    return confluence_score.bear_score > confluence_score.bull_score + 0.15

            return False

        except Exception:
            return False