#!/usr/bin/env python3
"""
Advanced Strategy Composition Engine
===================================

This module provides intelligent strategy composition capabilities:
- Automatic indicator compatibility analysis
- Multi-layered strategy architecture (primary, confirmation, filter)
- Risk management integration
- Performance optimization suggestions
- Market regime adaptation
- Conflict detection and resolution
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from itertools import combinations
import json

logger = logging.getLogger(__name__)

class IndicatorRole(Enum):
    """Roles that indicators can play in a strategy"""
    PRIMARY = "primary"
    CONFIRMATION = "confirmation"
    FILTER = "filter"
    EXIT = "exit"
    RISK_MANAGEMENT = "risk_management"

class CompatibilityLevel(Enum):
    """Compatibility levels between indicators"""
    EXCELLENT = "excellent"  # Highly complementary
    GOOD = "good"           # Compatible with synergy
    NEUTRAL = "neutral"     # Compatible but no special synergy
    POOR = "poor"          # Some conflicts but usable
    INCOMPATIBLE = "incompatible"  # Significant conflicts

@dataclass
class IndicatorProfile:
    """Profile of an indicator for composition analysis"""
    indicator_id: str
    name: str
    category: str  # trend, momentum, volatility, volume, oscillator
    signal_type: str  # continuous, discrete, binary
    timeframe_sensitivity: str  # low, medium, high
    market_regime_preference: List[str]  # trending, ranging, volatile
    computational_lag: int  # Number of periods for calculation
    signal_frequency: str  # high, medium, low
    complexity_level: str  # basic, intermediate, advanced
    overlapping_functions: Set[str]  # Functions this indicator provides
    dependencies: List[str]  # Other indicators this depends on
    conflicts: List[str]  # Indicators this conflicts with

@dataclass
class StrategyComponent:
    """A component of a trading strategy"""
    indicator: IndicatorProfile
    role: IndicatorRole
    weight: float
    parameters: Dict[str, Any]
    entry_conditions: List[str]
    exit_conditions: List[str]

@dataclass
class StrategyComposition:
    """Complete strategy composition"""
    strategy_id: str
    name: str
    description: str
    components: List[StrategyComponent]
    compatibility_score: float
    risk_score: float
    complexity_score: float
    expected_performance: Dict[str, float]
    market_suitability: Dict[str, float]
    conflicts_detected: List[str]
    optimization_suggestions: List[str]
    backtest_recommendations: Dict[str, Any]

class IndicatorCompatibilityAnalyzer:
    """Analyzes compatibility between different indicators"""

    def __init__(self):
        # Define indicator categories and their characteristics
        self.indicator_characteristics = {
            'moving_average': {
                'category': 'trend',
                'signal_type': 'continuous',
                'lag': 'medium',
                'frequency': 'medium',
                'functions': {'trend_following', 'support_resistance'},
                'best_with': ['momentum', 'volume'],
                'conflicts_with': []
            },
            'rsi': {
                'category': 'momentum',
                'signal_type': 'continuous',
                'lag': 'low',
                'frequency': 'high',
                'functions': {'overbought_oversold', 'divergence'},
                'best_with': ['trend', 'volume'],
                'conflicts_with': ['stochastic']  # Similar oscillators
            },
            'macd': {
                'category': 'momentum',
                'signal_type': 'continuous',
                'lag': 'medium',
                'frequency': 'medium',
                'functions': {'momentum', 'trend_change', 'divergence'},
                'best_with': ['trend', 'support_resistance'],
                'conflicts_with': []
            },
            'bollinger_bands': {
                'category': 'volatility',
                'signal_type': 'continuous',
                'lag': 'medium',
                'frequency': 'medium',
                'functions': {'volatility', 'mean_reversion', 'breakout'},
                'best_with': ['momentum', 'volume'],
                'conflicts_with': []
            },
            'stochastic': {
                'category': 'momentum',
                'signal_type': 'continuous',
                'lag': 'low',
                'frequency': 'high',
                'functions': {'overbought_oversold', 'momentum'},
                'best_with': ['trend', 'volatility'],
                'conflicts_with': ['rsi']  # Similar function
            },
            'atr': {
                'category': 'volatility',
                'signal_type': 'continuous',
                'lag': 'low',
                'frequency': 'low',
                'functions': {'volatility_measurement'},
                'best_with': ['trend', 'momentum'],
                'conflicts_with': []
            },
            'vwap': {
                'category': 'volume',
                'signal_type': 'continuous',
                'lag': 'low',
                'frequency': 'medium',
                'functions': {'volume_weighted_price', 'institutional_level'},
                'best_with': ['momentum', 'trend'],
                'conflicts_with': []
            },
            'adx': {
                'category': 'trend',
                'signal_type': 'continuous',
                'lag': 'high',
                'frequency': 'low',
                'functions': {'trend_strength'},
                'best_with': ['momentum', 'volatility'],
                'conflicts_with': []
            }
        }

        # Compatibility matrix
        self.compatibility_matrix = self._build_compatibility_matrix()

        # Role assignment preferences
        self.role_preferences = {
            IndicatorRole.PRIMARY: ['trend', 'momentum'],
            IndicatorRole.CONFIRMATION: ['momentum', 'volume', 'volatility'],
            IndicatorRole.FILTER: ['volatility', 'volume', 'trend_strength'],
            IndicatorRole.EXIT: ['momentum', 'support_resistance'],
            IndicatorRole.RISK_MANAGEMENT: ['volatility', 'volume']
        }

    def _build_compatibility_matrix(self) -> Dict[Tuple[str, str], CompatibilityLevel]:
        """Build compatibility matrix between indicators"""
        matrix = {}

        indicators = list(self.indicator_characteristics.keys())

        for ind1, ind2 in combinations(indicators, 2):
            compatibility = self._calculate_compatibility(ind1, ind2)
            matrix[(ind1, ind2)] = compatibility
            matrix[(ind2, ind1)] = compatibility  # Symmetric

        return matrix

    def _calculate_compatibility(self, ind1: str, ind2: str) -> CompatibilityLevel:
        """Calculate compatibility between two indicators"""
        char1 = self.indicator_characteristics[ind1]
        char2 = self.indicator_characteristics[ind2]

        # Check for explicit conflicts
        if ind2 in char1.get('conflicts_with', []) or ind1 in char2.get('conflicts_with', []):
            return CompatibilityLevel.INCOMPATIBLE

        # Check for explicit synergies
        if char2['category'] in char1.get('best_with', []) or char1['category'] in char2.get('best_with', []):
            return CompatibilityLevel.EXCELLENT

        # Check for function overlap (potential redundancy)
        functions1 = char1.get('functions', set())
        functions2 = char2.get('functions', set())

        if isinstance(functions1, list):
            functions1 = set(functions1)
        if isinstance(functions2, list):
            functions2 = set(functions2)

        overlap = len(functions1.intersection(functions2))
        total_functions = len(functions1.union(functions2))

        if overlap / total_functions > 0.7:  # High overlap
            return CompatibilityLevel.POOR
        elif overlap / total_functions > 0.3:  # Medium overlap
            return CompatibilityLevel.NEUTRAL
        else:  # Low overlap - complementary
            return CompatibilityLevel.GOOD

    def analyze_indicator_set(self, indicators: List[str]) -> Dict[str, Any]:
        """Analyze compatibility of a set of indicators"""
        if len(indicators) < 2:
            return {"overall_compatibility": "insufficient_data"}

        compatibility_scores = []
        conflicts = []
        synergies = []

        # Analyze pairwise compatibility
        for ind1, ind2 in combinations(indicators, 2):
            if (ind1, ind2) in self.compatibility_matrix:
                compatibility = self.compatibility_matrix[(ind1, ind2)]

                if compatibility == CompatibilityLevel.INCOMPATIBLE:
                    conflicts.append(f"{ind1} conflicts with {ind2}")
                    compatibility_scores.append(0.0)
                elif compatibility == CompatibilityLevel.POOR:
                    compatibility_scores.append(0.3)
                elif compatibility == CompatibilityLevel.NEUTRAL:
                    compatibility_scores.append(0.6)
                elif compatibility == CompatibilityLevel.GOOD:
                    compatibility_scores.append(0.8)
                elif compatibility == CompatibilityLevel.EXCELLENT:
                    synergies.append(f"{ind1} synergizes well with {ind2}")
                    compatibility_scores.append(1.0)

        # Calculate overall compatibility
        if compatibility_scores:
            overall_score = np.mean(compatibility_scores)
        else:
            overall_score = 0.5

        # Check for category balance
        categories = [self.indicator_characteristics[ind]['category'] for ind in indicators]
        category_balance = len(set(categories)) / len(categories)

        return {
            "overall_compatibility": overall_score,
            "category_balance": category_balance,
            "conflicts": conflicts,
            "synergies": synergies,
            "recommendations": self._generate_compatibility_recommendations(indicators, overall_score)
        }

    def _generate_compatibility_recommendations(self, indicators: List[str], score: float) -> List[str]:
        """Generate recommendations to improve compatibility"""
        recommendations = []

        if score < 0.4:
            recommendations.append("Consider removing conflicting indicators")
            recommendations.append("Look for more complementary alternatives")

        # Check for missing categories
        categories = set(self.indicator_characteristics[ind]['category'] for ind in indicators)
        if 'volume' not in categories:
            recommendations.append("Consider adding a volume indicator for confirmation")
        if 'volatility' not in categories:
            recommendations.append("Consider adding a volatility indicator for risk management")

        return recommendations

class StrategyArchitect:
    """Designs multi-layered trading strategy architecture"""

    def __init__(self, compatibility_analyzer: IndicatorCompatibilityAnalyzer):
        self.compatibility_analyzer = compatibility_analyzer

        # Strategy templates for different market conditions
        self.strategy_templates = {
            'trending_momentum': {
                'primary': ['moving_average', 'macd'],
                'confirmation': ['rsi', 'adx'],
                'filter': ['atr', 'vwap'],
                'description': 'Trend-following strategy with momentum confirmation'
            },
            'mean_reversion': {
                'primary': ['bollinger_bands', 'rsi'],
                'confirmation': ['stochastic', 'vwap'],
                'filter': ['atr'],
                'description': 'Mean reversion strategy for ranging markets'
            },
            'breakout_volatility': {
                'primary': ['bollinger_bands', 'atr'],
                'confirmation': ['volume', 'macd'],
                'filter': ['adx'],
                'description': 'Volatility breakout strategy'
            },
            'smart_money': {
                'primary': ['order_blocks', 'fair_value_gap'],
                'confirmation': ['volume', 'market_structure'],
                'filter': ['session_time'],
                'description': 'Smart money concepts strategy'
            }
        }

    def design_strategy(self, requirements: Dict[str, Any],
                       available_indicators: List[str]) -> StrategyComposition:
        """Design a strategy based on requirements and available indicators"""

        # Extract requirements
        market_condition = requirements.get('market_condition', 'trending')
        trading_style = requirements.get('trading_style', 'day_trading')
        risk_tolerance = requirements.get('risk_tolerance', 'medium')
        complexity_preference = requirements.get('complexity', 'intermediate')

        # Select appropriate template
        template = self._select_template(market_condition, trading_style)

        # Map available indicators to strategy roles
        role_assignments = self._assign_roles(available_indicators, template)

        # Analyze compatibility
        compatibility_analysis = self.compatibility_analyzer.analyze_indicator_set(
            [ind for role_inds in role_assignments.values() for ind in role_inds]
        )

        # Create strategy components
        components = self._create_components(role_assignments, requirements)

        # Calculate strategy scores
        complexity_score = self._calculate_complexity_score(components)
        risk_score = self._calculate_risk_score(components, risk_tolerance)

        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(
            components, compatibility_analysis, requirements
        )

        # Create strategy composition
        strategy = StrategyComposition(
            strategy_id=f"strategy_{market_condition}_{trading_style}",
            name=f"{template['description']} - {trading_style.replace('_', ' ').title()}",
            description=self._generate_strategy_description(components, requirements),
            components=components,
            compatibility_score=compatibility_analysis['overall_compatibility'],
            risk_score=risk_score,
            complexity_score=complexity_score,
            expected_performance=self._estimate_performance(components, requirements),
            market_suitability=self._assess_market_suitability(components),
            conflicts_detected=compatibility_analysis['conflicts'],
            optimization_suggestions=suggestions,
            backtest_recommendations=self._generate_backtest_recommendations(components)
        )

        return strategy

    def _select_template(self, market_condition: str, trading_style: str) -> Dict[str, Any]:
        """Select appropriate strategy template"""

        # Simple template selection logic
        if market_condition == 'trending':
            if trading_style in ['scalping', 'day_trading']:
                return self.strategy_templates['trending_momentum']
            else:
                return self.strategy_templates['trending_momentum']
        elif market_condition == 'ranging':
            return self.strategy_templates['mean_reversion']
        elif market_condition == 'volatile':
            return self.strategy_templates['breakout_volatility']
        else:
            return self.strategy_templates['trending_momentum']  # Default

    def _assign_roles(self, available_indicators: List[str],
                     template: Dict[str, Any]) -> Dict[IndicatorRole, List[str]]:
        """Assign roles to available indicators based on template"""

        role_assignments = {role: [] for role in IndicatorRole}

        # Get indicator characteristics
        characteristics = self.compatibility_analyzer.indicator_characteristics

        # Assign primary indicators
        for indicator in available_indicators:
            if indicator in characteristics:
                char = characteristics[indicator]
                category = char['category']

                # Assign based on template preferences and indicator characteristics
                if category in template.get('primary', []):
                    role_assignments[IndicatorRole.PRIMARY].append(indicator)
                elif category in template.get('confirmation', []):
                    role_assignments[IndicatorRole.CONFIRMATION].append(indicator)
                elif category in template.get('filter', []):
                    role_assignments[IndicatorRole.FILTER].append(indicator)

        # Ensure we have at least one primary indicator
        if not role_assignments[IndicatorRole.PRIMARY] and available_indicators:
            # Use the first available indicator as primary
            role_assignments[IndicatorRole.PRIMARY].append(available_indicators[0])

        return role_assignments

    def _create_components(self, role_assignments: Dict[IndicatorRole, List[str]],
                          requirements: Dict[str, Any]) -> List[StrategyComponent]:
        """Create strategy components from role assignments"""

        components = []
        characteristics = self.compatibility_analyzer.indicator_characteristics

        for role, indicators in role_assignments.items():
            for i, indicator in enumerate(indicators[:2]):  # Limit to 2 per role
                if indicator in characteristics:
                    # Calculate weight based on role and position
                    weight = self._calculate_component_weight(role, i, len(indicators))

                    # Generate entry/exit conditions
                    entry_conditions = self._generate_entry_conditions(indicator, role)
                    exit_conditions = self._generate_exit_conditions(indicator, role)

                    # Generate parameters
                    parameters = self._generate_parameters(indicator, requirements)

                    component = StrategyComponent(
                        indicator=IndicatorProfile(
                            indicator_id=indicator,
                            name=indicator.replace('_', ' ').title(),
                            category=characteristics[indicator]['category'],
                            signal_type=characteristics[indicator]['signal_type'],
                            timeframe_sensitivity='medium',
                            market_regime_preference=[requirements.get('market_condition', 'trending')],
                            computational_lag=10,  # Default
                            signal_frequency=characteristics[indicator]['frequency'],
                            complexity_level='intermediate',
                            overlapping_functions=characteristics[indicator]['functions'],
                            dependencies=[],
                            conflicts=characteristics[indicator].get('conflicts_with', [])
                        ),
                        role=role,
                        weight=weight,
                        parameters=parameters,
                        entry_conditions=entry_conditions,
                        exit_conditions=exit_conditions
                    )
                    components.append(component)

        return components

    def _calculate_component_weight(self, role: IndicatorRole, position: int, total_in_role: int) -> float:
        """Calculate weight for a component based on its role and position"""
        base_weights = {
            IndicatorRole.PRIMARY: 0.4,
            IndicatorRole.CONFIRMATION: 0.3,
            IndicatorRole.FILTER: 0.2,
            IndicatorRole.EXIT: 0.1,
            IndicatorRole.RISK_MANAGEMENT: 0.1
        }

        base_weight = base_weights.get(role, 0.1)

        # Reduce weight if multiple indicators in same role
        if total_in_role > 1:
            position_factor = (total_in_role - position) / total_in_role
            return base_weight * position_factor
        else:
            return base_weight

    def _generate_entry_conditions(self, indicator: str, role: IndicatorRole) -> List[str]:
        """Generate entry conditions for an indicator in a specific role"""
        conditions = []

        if role == IndicatorRole.PRIMARY:
            if indicator == 'moving_average':
                conditions = ['Price above EMA', 'EMA slope positive']
            elif indicator == 'rsi':
                conditions = ['RSI > 50', 'RSI rising']
            elif indicator == 'macd':
                conditions = ['MACD line above signal', 'MACD histogram positive']

        elif role == IndicatorRole.CONFIRMATION:
            if indicator == 'rsi':
                conditions = ['RSI not overbought', 'RSI momentum positive']
            elif indicator == 'volume':
                conditions = ['Volume above average', 'Volume increasing']

        elif role == IndicatorRole.FILTER:
            if indicator == 'atr':
                conditions = ['ATR above threshold', 'Volatility suitable']
            elif indicator == 'adx':
                conditions = ['ADX > 25', 'Trend strength sufficient']

        return conditions or ['Signal positive']

    def _generate_exit_conditions(self, indicator: str, role: IndicatorRole) -> List[str]:
        """Generate exit conditions for an indicator"""
        if indicator == 'rsi':
            return ['RSI overbought', 'RSI divergence']
        elif indicator == 'moving_average':
            return ['Price below EMA', 'EMA slope negative']
        elif indicator == 'macd':
            return ['MACD below signal', 'MACD histogram negative']
        else:
            return ['Signal reversal', 'Stop loss hit']

    def _generate_parameters(self, indicator: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimal parameters for an indicator"""
        # Default parameters that could be optimized based on requirements
        default_params = {
            'moving_average': {'period': 21, 'type': 'EMA'},
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2},
            'atr': {'period': 14},
            'adx': {'period': 14}
        }

        params = default_params.get(indicator, {})

        # Adjust based on trading style
        trading_style = requirements.get('trading_style', 'day_trading')
        if trading_style == 'scalping':
            # Shorter periods for scalping
            if 'period' in params:
                params['period'] = max(params['period'] // 2, 5)
        elif trading_style == 'swing_trading':
            # Longer periods for swing trading
            if 'period' in params:
                params['period'] = params['period'] * 2

        return params

    def _calculate_complexity_score(self, components: List[StrategyComponent]) -> float:
        """Calculate overall complexity score of the strategy"""
        if not components:
            return 0.0

        complexity_factors = []

        # Number of components
        complexity_factors.append(min(len(components) / 5, 1.0))

        # Unique roles
        roles = set(comp.role for comp in components)
        complexity_factors.append(len(roles) / len(IndicatorRole))

        # Individual component complexity
        for comp in components:
            if comp.indicator.complexity_level == 'basic':
                complexity_factors.append(0.3)
            elif comp.indicator.complexity_level == 'intermediate':
                complexity_factors.append(0.6)
            elif comp.indicator.complexity_level == 'advanced':
                complexity_factors.append(0.9)

        return np.mean(complexity_factors)

    def _calculate_risk_score(self, components: List[StrategyComponent], risk_tolerance: str) -> float:
        """Calculate risk score based on components and tolerance"""
        base_risk = 0.5

        # Adjust for number of components (more components = potentially more risk)
        component_risk = min(len(components) / 10, 0.3)

        # Adjust for role distribution
        roles = [comp.role for comp in components]
        if IndicatorRole.RISK_MANAGEMENT not in roles:
            base_risk += 0.2
        if IndicatorRole.FILTER not in roles:
            base_risk += 0.1

        # Adjust for user risk tolerance
        tolerance_adjustment = {
            'low': -0.2,
            'medium': 0.0,
            'high': 0.2
        }.get(risk_tolerance, 0.0)

        return min(base_risk + component_risk + tolerance_adjustment, 1.0)

    def _estimate_performance(self, components: List[StrategyComponent],
                            requirements: Dict[str, Any]) -> Dict[str, float]:
        """Estimate expected performance metrics"""
        # Simple heuristic-based performance estimation
        base_performance = {
            'expected_win_rate': 0.55,
            'expected_profit_factor': 1.3,
            'expected_drawdown': 0.15,
            'expected_sharpe_ratio': 0.8
        }

        # Adjust based on components
        for comp in components:
            if comp.role == IndicatorRole.PRIMARY:
                if comp.indicator.category == 'trend':
                    base_performance['expected_win_rate'] += 0.05
                elif comp.indicator.category == 'momentum':
                    base_performance['expected_profit_factor'] += 0.1

            elif comp.role == IndicatorRole.FILTER:
                base_performance['expected_drawdown'] -= 0.03

        return base_performance

    def _assess_market_suitability(self, components: List[StrategyComponent]) -> Dict[str, float]:
        """Assess strategy suitability for different market conditions"""
        suitability = {
            'trending': 0.5,
            'ranging': 0.5,
            'volatile': 0.5
        }

        for comp in components:
            if comp.indicator.category == 'trend':
                suitability['trending'] += 0.2
                suitability['ranging'] -= 0.1
            elif comp.indicator.category == 'momentum':
                suitability['trending'] += 0.1
                suitability['volatile'] += 0.1
            elif comp.indicator.category == 'volatility':
                suitability['volatile'] += 0.2
                suitability['ranging'] += 0.1

        # Normalize
        for key in suitability:
            suitability[key] = max(0.0, min(1.0, suitability[key]))

        return suitability

    def _generate_optimization_suggestions(self, components: List[StrategyComponent],
                                         compatibility_analysis: Dict[str, Any],
                                         requirements: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []

        # Compatibility suggestions
        if compatibility_analysis['overall_compatibility'] < 0.6:
            suggestions.append("Consider reviewing indicator combinations for better compatibility")

        # Role balance suggestions
        roles = [comp.role for comp in components]
        if IndicatorRole.FILTER not in roles:
            suggestions.append("Add a filter indicator to reduce false signals")
        if IndicatorRole.RISK_MANAGEMENT not in roles:
            suggestions.append("Consider adding risk management indicators")

        # Complexity suggestions
        complexity = self._calculate_complexity_score(components)
        if complexity > 0.8:
            suggestions.append("Strategy may be overly complex - consider simplifying")
        elif complexity < 0.3:
            suggestions.append("Strategy may be too simple - consider adding confirmation")

        return suggestions

    def _generate_backtest_recommendations(self, components: List[StrategyComponent]) -> Dict[str, Any]:
        """Generate backtest recommendations"""
        return {
            'minimum_data_period': '6 months',
            'recommended_timeframes': ['1h', '4h', '1d'],
            'test_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
            'out_of_sample_ratio': 0.3,
            'walk_forward_periods': 12,
            'optimization_parameters': [comp.indicator.indicator_id for comp in components[:3]]
        }

    def _generate_strategy_description(self, components: List[StrategyComponent],
                                     requirements: Dict[str, Any]) -> str:
        """Generate human-readable strategy description"""
        primary_indicators = [comp.indicator.name for comp in components if comp.role == IndicatorRole.PRIMARY]
        confirmation_indicators = [comp.indicator.name for comp in components if comp.role == IndicatorRole.CONFIRMATION]

        description = f"This strategy uses {', '.join(primary_indicators)} as primary signals"

        if confirmation_indicators:
            description += f", with {', '.join(confirmation_indicators)} for confirmation"

        market_condition = requirements.get('market_condition', 'general')
        description += f". Designed for {market_condition} market conditions"

        trading_style = requirements.get('trading_style', 'day_trading')
        description += f" and {trading_style.replace('_', ' ')} style."

        return description

# Test function
def test_strategy_composition():
    """Test the strategy composition engine"""

    # Initialize components
    compatibility_analyzer = IndicatorCompatibilityAnalyzer()
    strategy_architect = StrategyArchitect(compatibility_analyzer)

    # Test compatibility analysis
    indicators = ['moving_average', 'rsi', 'macd', 'atr']
    compatibility = compatibility_analyzer.analyze_indicator_set(indicators)

    print("=== Compatibility Analysis ===")
    print(f"Overall compatibility: {compatibility['overall_compatibility']:.3f}")
    print(f"Category balance: {compatibility['category_balance']:.3f}")
    print(f"Conflicts: {compatibility['conflicts']}")
    print(f"Synergies: {compatibility['synergies']}")

    # Test strategy design
    requirements = {
        'market_condition': 'trending',
        'trading_style': 'day_trading',
        'risk_tolerance': 'medium',
        'complexity': 'intermediate'
    }

    available_indicators = ['moving_average', 'rsi', 'macd', 'bollinger_bands', 'atr', 'vwap']

    strategy = strategy_architect.design_strategy(requirements, available_indicators)

    print(f"\n=== Strategy Composition ===")
    print(f"Name: {strategy.name}")
    print(f"Description: {strategy.description}")
    print(f"Compatibility Score: {strategy.compatibility_score:.3f}")
    print(f"Risk Score: {strategy.risk_score:.3f}")
    print(f"Complexity Score: {strategy.complexity_score:.3f}")

    print(f"\nComponents:")
    for comp in strategy.components:
        print(f"  - {comp.indicator.name} ({comp.role.value}): weight {comp.weight:.2f}")
        print(f"    Entry: {comp.entry_conditions}")
        print(f"    Parameters: {comp.parameters}")

    print(f"\nExpected Performance:")
    for metric, value in strategy.expected_performance.items():
        print(f"  {metric}: {value:.3f}")

    print(f"\nOptimization Suggestions:")
    for suggestion in strategy.optimization_suggestions:
        print(f"  - {suggestion}")

if __name__ == "__main__":
    test_strategy_composition()