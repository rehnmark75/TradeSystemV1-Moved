# core/strategies/helpers/ichimoku_mtf_rag_validator.py
"""
Ichimoku Multi-Timeframe RAG Validator
Advanced multi-timeframe validation using RAG templates and TradingView strategies

🕐 MULTI-TIMEFRAME FEATURES:
- RAG-powered timeframe correlation analysis
- TradingView strategy template matching across timeframes
- Hierarchical timeframe validation (daily → hourly → intraday)
- Dynamic timeframe weight adjustment based on market conditions
- Pattern recognition across timeframe clusters

📊 VALIDATION LOGIC:
- Higher timeframes provide trend context and bias
- Lower timeframes provide precise entry timing
- RAG templates suggest optimal timeframe combinations
- Market regime determines timeframe hierarchy importance
- Session analysis for timeframe relevance
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class TimeframeHierarchy(Enum):
    """Timeframe hierarchy levels"""
    MONTHLY = "1M"
    WEEKLY = "1w"
    DAILY = "1d"
    FOUR_HOUR = "4h"
    HOURLY = "1h"
    THIRTY_MIN = "30m"
    FIFTEEN_MIN = "15m"
    FIVE_MIN = "5m"
    ONE_MIN = "1m"


@dataclass
class TimeframeAnalysis:
    """Analysis results for a specific timeframe"""
    timeframe: str
    ichimoku_bias: str  # 'BULL', 'BEAR', 'NEUTRAL'
    trend_strength: float  # 0.0 to 1.0
    cloud_position: str  # 'ABOVE', 'BELOW', 'INSIDE'
    tk_relationship: str  # 'BULL_CROSS', 'BEAR_CROSS', 'ALIGNED', 'NEUTRAL'
    chikou_clear: bool
    confidence: float
    rag_template_matches: List[str]
    market_regime_compatibility: float


@dataclass
class MTFValidationResult:
    """Multi-timeframe validation results"""
    primary_timeframe: str
    validation_passed: bool
    overall_bias: str
    confidence_score: float
    timeframe_agreement_score: float
    rag_template_consensus: float
    higher_tf_support: bool
    lower_tf_confirmation: bool
    conflicting_timeframes: List[str]
    supporting_timeframes: List[str]
    validation_details: Dict[str, TimeframeAnalysis]


class IchimokuMTFRAGValidator:
    """
    🕐 MULTI-TIMEFRAME RAG VALIDATION ENGINE

    Advanced validation system that:
    - Analyzes Ichimoku signals across multiple timeframes
    - Uses RAG templates to identify optimal timeframe combinations
    - Provides hierarchical validation with market regime awareness
    - Leverages TradingView strategies for timeframe correlation patterns
    """

    def __init__(self, data_fetcher=None, rag_enhancer=None, tradingview_parser=None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher
        self.rag_enhancer = rag_enhancer
        self.tradingview_parser = tradingview_parser

        # Timeframe hierarchy mapping
        self.timeframe_hierarchy = {
            '1M': 0, '1w': 1, '1d': 2, '4h': 3, '1h': 4,
            '30m': 5, '15m': 6, '5m': 7, '1m': 8
        }

        # Timeframe weight multipliers based on market conditions
        self.regime_timeframe_weights = {
            'trending': {
                '1d': 1.4, '4h': 1.3, '1h': 1.1, '15m': 1.0, '5m': 0.8
            },
            'ranging': {
                '1d': 1.0, '4h': 1.1, '1h': 1.3, '15m': 1.2, '5m': 1.0
            },
            'breakout': {
                '1d': 1.3, '4h': 1.4, '1h': 1.2, '15m': 1.1, '5m': 0.9
            },
            'high_volatility': {
                '1d': 1.2, '4h': 1.3, '1h': 1.0, '15m': 0.9, '5m': 0.7
            },
            'low_volatility': {
                '1d': 1.1, '4h': 1.0, '1h': 1.1, '15m': 1.2, '5m': 1.1
            }
        }

        # Standard timeframe combinations for different trading styles
        self.timeframe_combinations = {
            'swing_trading': ['1d', '4h', '1h'],
            'day_trading': ['4h', '1h', '15m'],
            'scalping': ['1h', '15m', '5m'],
            'position_trading': ['1w', '1d', '4h']
        }

        self.logger.info("🕐 Ichimoku MTF RAG Validator initialized")

    def validate_mtf_signal(
        self,
        primary_signal: Dict,
        epic: str,
        primary_timeframe: str,
        market_conditions: Dict = None,
        trading_style: str = 'day_trading'
    ) -> MTFValidationResult:
        """
        🎯 VALIDATE MTF SIGNAL: Comprehensive multi-timeframe validation

        Args:
            primary_signal: Primary Ichimoku signal to validate
            epic: Currency pair
            primary_timeframe: Primary timeframe for the signal
            market_conditions: Current market conditions
            trading_style: Trading style to determine timeframe combination

        Returns:
            Comprehensive multi-timeframe validation results
        """
        try:
            self.logger.info(f"🕐 Starting MTF validation for {epic} {primary_timeframe} signal...")

            # Get appropriate timeframe combination
            timeframes = self._get_timeframe_combination(primary_timeframe, trading_style, market_conditions)

            # Analyze each timeframe
            timeframe_analyses = {}
            for tf in timeframes:
                try:
                    analysis = self._analyze_timeframe(epic, tf, primary_signal, market_conditions)
                    if analysis:
                        timeframe_analyses[tf] = analysis
                except Exception as e:
                    self.logger.warning(f"Failed to analyze timeframe {tf}: {e}")
                    continue

            # Calculate validation result
            validation_result = self._calculate_mtf_validation(
                primary_signal, primary_timeframe, timeframe_analyses, market_conditions
            )

            self.logger.info(f"🕐 MTF validation complete for {epic}: "
                           f"Passed={validation_result.validation_passed}, "
                           f"Bias={validation_result.overall_bias}, "
                           f"Confidence={validation_result.confidence_score:.2f}")

            return validation_result

        except Exception as e:
            self.logger.error(f"MTF validation failed: {e}")
            return self._get_fallback_validation_result(primary_timeframe)

    def _get_timeframe_combination(
        self,
        primary_timeframe: str,
        trading_style: str,
        market_conditions: Dict = None
    ) -> List[str]:
        """Get optimal timeframe combination for analysis"""
        try:
            # Start with base combination
            base_combination = self.timeframe_combinations.get(trading_style, ['4h', '1h', '15m'])

            # Ensure primary timeframe is included
            if primary_timeframe not in base_combination:
                base_combination.append(primary_timeframe)

            # Get RAG recommendations for timeframe combination
            if self.rag_enhancer and self.rag_enhancer.rag_interface:
                rag_timeframes = self._get_rag_timeframe_recommendations(
                    trading_style, market_conditions
                )
                if rag_timeframes:
                    # Merge RAG recommendations with base combination
                    base_combination.extend(rag_timeframes)

            # Remove duplicates and sort by hierarchy
            unique_timeframes = list(set(base_combination))
            unique_timeframes.sort(key=lambda x: self.timeframe_hierarchy.get(x, 10))

            # Limit to 5 timeframes for performance
            return unique_timeframes[:5]

        except Exception as e:
            self.logger.error(f"Timeframe combination selection failed: {e}")
            return ['4h', '1h', '15m']

    def _get_rag_timeframe_recommendations(
        self,
        trading_style: str,
        market_conditions: Dict = None
    ) -> List[str]:
        """Get timeframe recommendations from RAG system"""
        try:
            if not self.rag_enhancer or not self.rag_enhancer.rag_interface:
                return []

            # Create search query for timeframe strategies
            regime = market_conditions.get('regime', 'trending') if market_conditions else 'trending'
            query = f"ichimoku {trading_style} multi timeframe {regime} markets strategy"

            # Search for relevant strategies
            search_results = self.rag_enhancer.rag_interface.search_templates(query, limit=3)

            if search_results.get('error'):
                return []

            # Extract timeframes from search results
            recommended_timeframes = []
            for result in search_results.get('results', []):
                metadata = result.get('metadata', {})
                timeframes = metadata.get('timeframes', [])
                recommended_timeframes.extend(timeframes)

            # Filter valid timeframes
            valid_timeframes = [
                tf for tf in recommended_timeframes
                if tf in self.timeframe_hierarchy
            ]

            return list(set(valid_timeframes))

        except Exception as e:
            self.logger.warning(f"RAG timeframe recommendations failed: {e}")
            return []

    def _analyze_timeframe(
        self,
        epic: str,
        timeframe: str,
        primary_signal: Dict,
        market_conditions: Dict = None
    ) -> Optional[TimeframeAnalysis]:
        """Analyze Ichimoku signals for a specific timeframe"""
        try:
            if not self.data_fetcher:
                return None

            # Fetch data for the timeframe
            # Note: This is a simplified example - in production, you'd fetch actual market data
            # market_data = self.data_fetcher.get_historical_data(epic, timeframe, periods=100)

            # For now, simulate timeframe analysis based on primary signal
            analysis = self._simulate_timeframe_analysis(timeframe, primary_signal, market_conditions)

            # Get RAG template matches for this timeframe
            rag_matches = self._get_rag_template_matches(timeframe, primary_signal, market_conditions)

            analysis.rag_template_matches = rag_matches
            analysis.market_regime_compatibility = self._calculate_regime_compatibility(
                timeframe, market_conditions
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Timeframe analysis failed for {timeframe}: {e}")
            return None

    def _simulate_timeframe_analysis(
        self,
        timeframe: str,
        primary_signal: Dict,
        market_conditions: Dict = None
    ) -> TimeframeAnalysis:
        """Simulate timeframe analysis (placeholder for actual implementation)"""
        try:
            # Get timeframe hierarchy level
            tf_level = self.timeframe_hierarchy.get(timeframe, 5)
            primary_bias = primary_signal.get('signal_type', 'NEUTRAL')

            # Simulate bias correlation based on timeframe hierarchy
            # Higher timeframes tend to be more stable
            if tf_level <= 2:  # Daily and above
                bias_stability = 0.9
            elif tf_level <= 4:  # Hourly
                bias_stability = 0.7
            else:  # Intraday
                bias_stability = 0.5

            # Simulate bias agreement
            import random
            random.seed(hash(timeframe + str(tf_level)))  # Deterministic for testing

            if random.random() < bias_stability:
                ichimoku_bias = primary_bias
                confidence = 0.7 + random.random() * 0.2
            else:
                ichimoku_bias = 'NEUTRAL'
                confidence = 0.4 + random.random() * 0.3

            # Simulate other properties
            trend_strength = confidence * (0.8 + random.random() * 0.2)
            cloud_position = 'ABOVE' if ichimoku_bias == 'BULL' else 'BELOW' if ichimoku_bias == 'BEAR' else 'INSIDE'
            tk_relationship = f"{ichimoku_bias}_CROSS" if ichimoku_bias != 'NEUTRAL' else 'NEUTRAL'
            chikou_clear = confidence > 0.6

            return TimeframeAnalysis(
                timeframe=timeframe,
                ichimoku_bias=ichimoku_bias,
                trend_strength=trend_strength,
                cloud_position=cloud_position,
                tk_relationship=tk_relationship,
                chikou_clear=chikou_clear,
                confidence=confidence,
                rag_template_matches=[],  # Will be filled by caller
                market_regime_compatibility=0.8  # Will be calculated by caller
            )

        except Exception as e:
            self.logger.error(f"Timeframe analysis simulation failed: {e}")
            return TimeframeAnalysis(
                timeframe=timeframe,
                ichimoku_bias='NEUTRAL',
                trend_strength=0.5,
                cloud_position='INSIDE',
                tk_relationship='NEUTRAL',
                chikou_clear=False,
                confidence=0.5,
                rag_template_matches=[],
                market_regime_compatibility=0.5
            )

    def _get_rag_template_matches(
        self,
        timeframe: str,
        primary_signal: Dict,
        market_conditions: Dict = None
    ) -> List[str]:
        """Get RAG template matches for specific timeframe"""
        try:
            if not self.rag_enhancer or not self.rag_enhancer.rag_interface:
                return []

            signal_type = primary_signal.get('signal_type', 'UNKNOWN')
            direction = "bullish" if signal_type == 'BULL' else "bearish" if signal_type == 'BEAR' else "neutral"

            # Create timeframe-specific query
            query = f"ichimoku {direction} {timeframe} timeframe strategy template"

            if market_conditions:
                regime = market_conditions.get('regime', 'trending')
                query += f" {regime} market"

            # Search for templates
            search_results = self.rag_enhancer.rag_interface.search_templates(query, limit=3)

            if search_results.get('error'):
                return []

            # Extract template names
            template_matches = []
            for result in search_results.get('results', []):
                title = result.get('title', 'Unknown Template')
                template_matches.append(title)

            return template_matches

        except Exception as e:
            self.logger.warning(f"RAG template matching failed: {e}")
            return []

    def _calculate_regime_compatibility(
        self,
        timeframe: str,
        market_conditions: Dict = None
    ) -> float:
        """Calculate how well this timeframe fits the current market regime"""
        try:
            if not market_conditions:
                return 0.8

            regime = market_conditions.get('regime', 'trending')
            regime_weights = self.regime_timeframe_weights.get(regime, {})

            weight = regime_weights.get(timeframe, 1.0)

            # Normalize to 0-1 range
            return min(1.0, weight / 1.4)

        except Exception:
            return 0.8

    def _calculate_mtf_validation(
        self,
        primary_signal: Dict,
        primary_timeframe: str,
        timeframe_analyses: Dict[str, TimeframeAnalysis],
        market_conditions: Dict = None
    ) -> MTFValidationResult:
        """Calculate comprehensive multi-timeframe validation result"""
        try:
            primary_bias = primary_signal.get('signal_type', 'UNKNOWN')

            # Calculate timeframe agreement
            agreement_data = self._calculate_timeframe_agreement(
                primary_bias, timeframe_analyses, market_conditions
            )

            # Calculate RAG template consensus
            template_consensus = self._calculate_rag_template_consensus(timeframe_analyses)

            # Determine higher/lower timeframe support
            higher_tf_support, lower_tf_confirmation = self._analyze_timeframe_hierarchy_support(
                primary_timeframe, primary_bias, timeframe_analyses
            )

            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                agreement_data, template_consensus, higher_tf_support, lower_tf_confirmation
            )

            # Determine validation result
            validation_passed = self._determine_validation_passed(
                agreement_data['agreement_score'], confidence_score, higher_tf_support
            )

            # Determine overall bias
            overall_bias = self._determine_overall_bias(primary_bias, agreement_data)

            return MTFValidationResult(
                primary_timeframe=primary_timeframe,
                validation_passed=validation_passed,
                overall_bias=overall_bias,
                confidence_score=confidence_score,
                timeframe_agreement_score=agreement_data['agreement_score'],
                rag_template_consensus=template_consensus,
                higher_tf_support=higher_tf_support,
                lower_tf_confirmation=lower_tf_confirmation,
                conflicting_timeframes=agreement_data['conflicting_timeframes'],
                supporting_timeframes=agreement_data['supporting_timeframes'],
                validation_details=timeframe_analyses
            )

        except Exception as e:
            self.logger.error(f"MTF validation calculation failed: {e}")
            return self._get_fallback_validation_result(primary_timeframe)

    def _calculate_timeframe_agreement(
        self,
        primary_bias: str,
        timeframe_analyses: Dict[str, TimeframeAnalysis],
        market_conditions: Dict = None
    ) -> Dict:
        """Calculate agreement across timeframes"""
        try:
            supporting_timeframes = []
            conflicting_timeframes = []
            neutral_timeframes = []

            regime = market_conditions.get('regime', 'trending') if market_conditions else 'trending'
            regime_weights = self.regime_timeframe_weights.get(regime, {})

            total_weight = 0.0
            agreement_weight = 0.0

            for tf, analysis in timeframe_analyses.items():
                tf_weight = regime_weights.get(tf, 1.0)
                total_weight += tf_weight

                if analysis.ichimoku_bias == primary_bias:
                    supporting_timeframes.append(tf)
                    agreement_weight += tf_weight * analysis.confidence
                elif analysis.ichimoku_bias == 'NEUTRAL':
                    neutral_timeframes.append(tf)
                    agreement_weight += tf_weight * 0.5  # Neutral is half support
                else:
                    conflicting_timeframes.append(tf)

            # Calculate agreement score
            agreement_score = agreement_weight / total_weight if total_weight > 0 else 0.5

            return {
                'agreement_score': agreement_score,
                'supporting_timeframes': supporting_timeframes,
                'conflicting_timeframes': conflicting_timeframes,
                'neutral_timeframes': neutral_timeframes
            }

        except Exception as e:
            self.logger.error(f"Timeframe agreement calculation failed: {e}")
            return {
                'agreement_score': 0.5,
                'supporting_timeframes': [],
                'conflicting_timeframes': [],
                'neutral_timeframes': []
            }

    def _calculate_rag_template_consensus(self, timeframe_analyses: Dict[str, TimeframeAnalysis]) -> float:
        """Calculate consensus from RAG template matches"""
        try:
            total_templates = 0
            consensus_templates = 0

            # Count template matches across timeframes
            all_templates = {}
            for analysis in timeframe_analyses.values():
                for template in analysis.rag_template_matches:
                    all_templates[template] = all_templates.get(template, 0) + 1
                    total_templates += 1

            # Calculate consensus (templates appearing in multiple timeframes)
            for template, count in all_templates.items():
                if count > 1:  # Template appears in multiple timeframes
                    consensus_templates += count

            if total_templates == 0:
                return 0.5

            return consensus_templates / total_templates

        except Exception:
            return 0.5

    def _analyze_timeframe_hierarchy_support(
        self,
        primary_timeframe: str,
        primary_bias: str,
        timeframe_analyses: Dict[str, TimeframeAnalysis]
    ) -> Tuple[bool, bool]:
        """Analyze support from higher and lower timeframes"""
        try:
            primary_level = self.timeframe_hierarchy.get(primary_timeframe, 5)

            higher_tf_support = False
            lower_tf_confirmation = False

            # Check higher timeframes (lower hierarchy numbers)
            for tf, analysis in timeframe_analyses.items():
                tf_level = self.timeframe_hierarchy.get(tf, 5)

                if tf_level < primary_level:  # Higher timeframe
                    if analysis.ichimoku_bias == primary_bias and analysis.confidence > 0.6:
                        higher_tf_support = True

                elif tf_level > primary_level:  # Lower timeframe
                    if analysis.ichimoku_bias == primary_bias and analysis.confidence > 0.6:
                        lower_tf_confirmation = True

            return higher_tf_support, lower_tf_confirmation

        except Exception:
            return False, False

    def _calculate_overall_confidence(
        self,
        agreement_data: Dict,
        template_consensus: float,
        higher_tf_support: bool,
        lower_tf_confirmation: bool
    ) -> float:
        """Calculate overall validation confidence"""
        try:
            # Base confidence from timeframe agreement
            confidence = agreement_data['agreement_score'] * 0.4

            # Template consensus contribution
            confidence += template_consensus * 0.2

            # Higher timeframe support bonus
            if higher_tf_support:
                confidence += 0.2

            # Lower timeframe confirmation bonus
            if lower_tf_confirmation:
                confidence += 0.1

            # Penalty for conflicting timeframes
            conflicting_count = len(agreement_data.get('conflicting_timeframes', []))
            total_count = len(agreement_data.get('supporting_timeframes', [])) + conflicting_count
            if total_count > 0:
                conflict_penalty = (conflicting_count / total_count) * 0.1
                confidence -= conflict_penalty

            return max(0.0, min(1.0, confidence))

        except Exception:
            return 0.5

    def _determine_validation_passed(
        self,
        agreement_score: float,
        confidence_score: float,
        higher_tf_support: bool
    ) -> bool:
        """Determine if MTF validation passes"""
        try:
            # Base validation criteria
            if agreement_score >= 0.7 and confidence_score >= 0.6:
                return True

            # Alternative validation with higher timeframe support
            if higher_tf_support and agreement_score >= 0.6 and confidence_score >= 0.5:
                return True

            return False

        except Exception:
            return False

    def _determine_overall_bias(self, primary_bias: str, agreement_data: Dict) -> str:
        """Determine overall bias from MTF analysis"""
        try:
            agreement_score = agreement_data.get('agreement_score', 0.5)

            if agreement_score >= 0.6:
                return primary_bias
            else:
                return 'NEUTRAL'

        except Exception:
            return 'NEUTRAL'

    def _get_fallback_validation_result(self, primary_timeframe: str) -> MTFValidationResult:
        """Fallback validation result when analysis fails"""
        return MTFValidationResult(
            primary_timeframe=primary_timeframe,
            validation_passed=False,
            overall_bias='NEUTRAL',
            confidence_score=0.5,
            timeframe_agreement_score=0.5,
            rag_template_consensus=0.5,
            higher_tf_support=False,
            lower_tf_confirmation=False,
            conflicting_timeframes=[],
            supporting_timeframes=[],
            validation_details={}
        )

    def get_mtf_summary(self, validation_result: MTFValidationResult) -> Dict:
        """Get summary of MTF validation results"""
        try:
            summary = {
                'validation_status': 'PASSED' if validation_result.validation_passed else 'FAILED',
                'primary_timeframe': validation_result.primary_timeframe,
                'overall_bias': validation_result.overall_bias,
                'confidence_level': self._classify_confidence_level(validation_result.confidence_score),
                'timeframe_analysis': {
                    'total_timeframes': len(validation_result.validation_details),
                    'supporting_timeframes': len(validation_result.supporting_timeframes),
                    'conflicting_timeframes': len(validation_result.conflicting_timeframes),
                    'agreement_percentage': validation_result.timeframe_agreement_score * 100
                },
                'rag_integration': {
                    'template_consensus': validation_result.rag_template_consensus,
                    'template_consensus_level': self._classify_consensus_level(validation_result.rag_template_consensus)
                },
                'hierarchy_support': {
                    'higher_timeframe_support': validation_result.higher_tf_support,
                    'lower_timeframe_confirmation': validation_result.lower_tf_confirmation
                }
            }

            return summary

        except Exception as e:
            self.logger.error(f"MTF summary generation failed: {e}")
            return {'error': str(e)}

    def _classify_confidence_level(self, confidence_score: float) -> str:
        """Classify confidence score into levels"""
        if confidence_score >= 0.8:
            return 'VERY_HIGH'
        elif confidence_score >= 0.7:
            return 'HIGH'
        elif confidence_score >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _classify_consensus_level(self, consensus_score: float) -> str:
        """Classify consensus score into levels"""
        if consensus_score >= 0.8:
            return 'STRONG'
        elif consensus_score >= 0.6:
            return 'MODERATE'
        else:
            return 'WEAK'

    def is_mtf_validation_available(self) -> bool:
        """Check if MTF validation is available"""
        return self.data_fetcher is not None

    def get_supported_timeframe_combinations(self) -> Dict:
        """Get supported timeframe combinations"""
        return {
            'combinations': self.timeframe_combinations,
            'hierarchy': self.timeframe_hierarchy,
            'regime_weights': self.regime_timeframe_weights
        }