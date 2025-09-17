#!/usr/bin/env python3
"""
Enhanced Pine Script Parser for RAG System
==========================================

This module provides advanced parsing capabilities for Pine Script code to extract:
- Function signatures and parameters
- Variable declarations and assignments
- Input/output definitions
- Technical indicator usage patterns
- Strategy logic patterns
- Dependencies and imports
"""

import re
import json
import ast
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PineScriptElement:
    """Represents a parsed element from Pine Script"""
    element_type: str  # input, variable, function, strategy, etc.
    name: str
    value: Optional[str] = None
    parameters: Dict[str, Any] = None
    description: Optional[str] = None
    line_number: int = 0

@dataclass
class PineScriptAnalysis:
    """Complete analysis of a Pine Script"""
    version: str = "5"
    script_type: str = "indicator"  # indicator, strategy, library
    title: str = ""
    shorttitle: str = ""
    overlay: bool = False

    # Core elements
    inputs: List[PineScriptElement] = None
    variables: List[PineScriptElement] = None
    functions: List[PineScriptElement] = None
    plots: List[PineScriptElement] = None

    # Semantic analysis
    indicators_used: List[str] = None
    math_operations: List[str] = None
    conditional_logic: List[str] = None
    timeframe_functions: List[str] = None

    # Complexity metrics
    complexity_score: float = 0.0
    line_count: int = 0
    function_count: int = 0
    input_count: int = 0

    # Trading concepts
    trading_concepts: List[str] = None
    market_conditions: List[str] = None
    signal_types: List[str] = None

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = []
        if self.variables is None:
            self.variables = []
        if self.functions is None:
            self.functions = []
        if self.plots is None:
            self.plots = []
        if self.indicators_used is None:
            self.indicators_used = []
        if self.math_operations is None:
            self.math_operations = []
        if self.conditional_logic is None:
            self.conditional_logic = []
        if self.timeframe_functions is None:
            self.timeframe_functions = []
        if self.trading_concepts is None:
            self.trading_concepts = []
        if self.market_conditions is None:
            self.market_conditions = []
        if self.signal_types is None:
            self.signal_types = []

class EnhancedPineParser:
    """Enhanced Pine Script parser for semantic analysis"""

    def __init__(self):
        self.trading_concepts = {
            # Trend concepts
            'trend': ['trend', 'trending', 'uptrend', 'downtrend', 'trend_direction', 'trend_strength'],
            'momentum': ['momentum', 'velocity', 'acceleration', 'rate_of_change', 'roc'],
            'volatility': ['volatility', 'volatility_index', 'atr', 'true_range', 'vix'],
            'volume': ['volume', 'vol', 'volume_profile', 'volume_weighted', 'vwap'],

            # Pattern concepts
            'reversal': ['reversal', 'reverse', 'pivot', 'turning_point', 'divergence'],
            'breakout': ['breakout', 'breakout_level', 'resistance_break', 'support_break'],
            'confluence': ['confluence', 'confirmation', 'multiple_signals', 'convergence'],

            # Market structure
            'support_resistance': ['support', 'resistance', 'level', 'zone', 'horizontal'],
            'channels': ['channel', 'trendline', 'parallel', 'envelope', 'bands'],
            'fibonacci': ['fibonacci', 'fib', 'retracement', 'extension', 'golden_ratio'],

            # Smart money concepts
            'order_blocks': ['order_block', 'ob', 'institutional_level', 'imbalance'],
            'liquidity': ['liquidity', 'sweep', 'raid', 'stop_hunt', 'wick'],
            'market_structure': ['bos', 'choch', 'structure', 'swing_high', 'swing_low'],

            # Time-based concepts
            'session': ['session', 'asia', 'london', 'newyork', 'timezone'],
            'timeframe': ['timeframe', 'htf', 'ltf', 'multi_timeframe', 'mtf']
        }

        self.pine_indicators = {
            'moving_averages': ['sma', 'ema', 'wma', 'vwma', 'hma', 'alma', 'smma', 'rma'],
            'oscillators': ['rsi', 'stoch', 'macd', 'cci', 'williams', 'mfi', 'tsi', 'uo'],
            'trend_indicators': ['adx', 'parabolic_sar', 'ichimoku', 'supertrend', 'aroon'],
            'volume_indicators': ['obv', 'ad', 'cmf', 'fi', 'nvi', 'pvi', 'vwap'],
            'volatility_indicators': ['bb', 'kc', 'dc', 'atr', 'stddev'],
            'support_resistance': ['pivot', 'fibonacci', 'support', 'resistance']
        }

        self.market_conditions = {
            'trending': ['trending', 'trend', 'directional', 'momentum'],
            'ranging': ['ranging', 'sideways', 'consolidation', 'choppy'],
            'volatile': ['volatile', 'high_volatility', 'expansion', 'breakout'],
            'low_volatility': ['low_volatility', 'contraction', 'squeeze']
        }

    def parse_pine_script(self, code: str) -> PineScriptAnalysis:
        """Main parsing function that analyzes Pine Script code"""
        if not code or not code.strip():
            return PineScriptAnalysis()

        analysis = PineScriptAnalysis()

        # Clean and prepare code
        lines = code.split('\n')
        analysis.line_count = len([l for l in lines if l.strip() and not l.strip().startswith('//')])

        # Parse header and metadata
        self._parse_header(lines, analysis)

        # Parse core elements
        self._parse_inputs(lines, analysis)
        self._parse_variables(lines, analysis)
        self._parse_functions(lines, analysis)
        self._parse_plots(lines, analysis)

        # Semantic analysis
        self._analyze_indicators(lines, analysis)
        self._analyze_trading_concepts(lines, analysis)
        self._analyze_complexity(analysis)

        return analysis

    def _parse_header(self, lines: List[str], analysis: PineScriptAnalysis):
        """Parse script header and metadata"""
        for i, line in enumerate(lines):
            line = line.strip()

            # Version detection
            if line.startswith('//@version='):
                analysis.version = line.split('=')[1].strip()

            # Script type and title
            elif line.startswith('indicator(') or line.startswith('strategy('):
                analysis.script_type = 'indicator' if line.startswith('indicator(') else 'strategy'

                # Extract title and parameters
                title_match = re.search(r'["\']([^"\']+)["\']', line)
                if title_match:
                    analysis.title = title_match.group(1)

                # Extract shorttitle
                shorttitle_match = re.search(r'shorttitle\s*=\s*["\']([^"\']+)["\']', line)
                if shorttitle_match:
                    analysis.shorttitle = shorttitle_match.group(1)

                # Extract overlay
                if 'overlay=true' in line or 'overlay = true' in line:
                    analysis.overlay = True

    def _parse_inputs(self, lines: List[str], analysis: PineScriptAnalysis):
        """Parse input definitions"""
        for i, line in enumerate(lines):
            line = line.strip()

            # Match input patterns
            input_patterns = [
                r'(\w+)\s*=\s*input\.(\w+)\s*\(([^)]+)\)',
                r'(\w+)\s*=\s*input\s*\(([^)]+)\)'
            ]

            for pattern in input_patterns:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) == 3:
                        name, input_type, params = match.groups()
                    else:
                        name, params = match.groups()
                        input_type = 'auto'

                    # Parse parameters
                    param_dict = self._parse_parameters(params)

                    element = PineScriptElement(
                        element_type='input',
                        name=name,
                        parameters=param_dict,
                        line_number=i + 1
                    )
                    analysis.inputs.append(element)

    def _parse_variables(self, lines: List[str], analysis: PineScriptAnalysis):
        """Parse variable declarations"""
        for i, line in enumerate(lines):
            line = line.strip()

            # Variable assignment patterns
            var_patterns = [
                r'var\s+(\w+)\s*=\s*(.+)',
                r'(\w+)\s*:=\s*(.+)',
                r'(\w+)\s*=\s*(?!input)(.+)'
            ]

            for pattern in var_patterns:
                match = re.search(pattern, line)
                if match and not line.startswith('//'):
                    name, value = match.groups()

                    # Skip function definitions and certain keywords
                    if any(keyword in line for keyword in ['function', 'method', 'if', 'for', 'while']):
                        continue

                    element = PineScriptElement(
                        element_type='variable',
                        name=name,
                        value=value.strip(),
                        line_number=i + 1
                    )
                    analysis.variables.append(element)

    def _parse_functions(self, lines: List[str], analysis: PineScriptAnalysis):
        """Parse function definitions"""
        for i, line in enumerate(lines):
            line = line.strip()

            # Function definition patterns
            func_patterns = [
                r'(\w+)\s*\([^)]*\)\s*=>',
                r'method\s+(\w+)\s*\([^)]*\)',
                r'function\s+(\w+)\s*\([^)]*\)'
            ]

            for pattern in func_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)

                    element = PineScriptElement(
                        element_type='function',
                        name=name,
                        value=line,
                        line_number=i + 1
                    )
                    analysis.functions.append(element)

    def _parse_plots(self, lines: List[str], analysis: PineScriptAnalysis):
        """Parse plot statements"""
        for i, line in enumerate(lines):
            line = line.strip()

            # Plot patterns
            plot_patterns = [
                r'plot\s*\(([^)]+)\)',
                r'plotshape\s*\(([^)]+)\)',
                r'plotchar\s*\(([^)]+)\)',
                r'bgcolor\s*\(([^)]+)\)',
                r'fill\s*\(([^)]+)\)'
            ]

            for pattern in plot_patterns:
                match = re.search(pattern, line)
                if match:
                    params = match.group(1)
                    param_dict = self._parse_parameters(params)

                    element = PineScriptElement(
                        element_type='plot',
                        name=f"plot_{i}",
                        parameters=param_dict,
                        line_number=i + 1
                    )
                    analysis.plots.append(element)

    def _analyze_indicators(self, lines: List[str], analysis: PineScriptAnalysis):
        """Analyze technical indicators used in the script"""
        code_text = ' '.join(lines).lower()

        for category, indicators in self.pine_indicators.items():
            for indicator in indicators:
                if indicator in code_text:
                    analysis.indicators_used.append(indicator)

    def _analyze_trading_concepts(self, lines: List[str], analysis: PineScriptAnalysis):
        """Analyze trading concepts present in the script"""
        code_text = ' '.join(lines).lower()

        # Analyze trading concepts
        for concept_category, concepts in self.trading_concepts.items():
            for concept in concepts:
                if concept in code_text:
                    analysis.trading_concepts.append(concept_category)

        # Analyze market conditions
        for condition, keywords in self.market_conditions.items():
            for keyword in keywords:
                if keyword in code_text:
                    analysis.market_conditions.append(condition)

        # Deduplicate
        analysis.trading_concepts = list(set(analysis.trading_concepts))
        analysis.market_conditions = list(set(analysis.market_conditions))

    def _analyze_complexity(self, analysis: PineScriptAnalysis):
        """Calculate complexity score based on various factors"""
        score = 0.0

        # Base complexity from line count
        score += min(analysis.line_count / 100, 0.3)

        # Input complexity
        analysis.input_count = len(analysis.inputs)
        score += min(analysis.input_count / 20, 0.2)

        # Function complexity
        analysis.function_count = len(analysis.functions)
        score += min(analysis.function_count / 10, 0.2)

        # Indicator usage complexity
        score += min(len(analysis.indicators_used) / 15, 0.2)

        # Trading concept complexity
        score += min(len(analysis.trading_concepts) / 10, 0.1)

        analysis.complexity_score = min(score, 1.0)

    def _parse_parameters(self, param_string: str) -> Dict[str, Any]:
        """Parse parameter string into dictionary"""
        params = {}

        # Simple parameter parsing - can be enhanced
        param_parts = param_string.split(',')
        for part in param_parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                params[key.strip()] = value.strip()
            else:
                # Positional parameter
                params[f'param_{len(params)}'] = part

        return params

    def extract_semantic_features(self, analysis: PineScriptAnalysis) -> Dict[str, Any]:
        """Extract semantic features for embedding"""
        features = {
            # Basic metadata
            'script_type': analysis.script_type,
            'version': analysis.version,
            'overlay': analysis.overlay,

            # Complexity metrics
            'complexity_score': analysis.complexity_score,
            'line_count': analysis.line_count,
            'function_count': analysis.function_count,
            'input_count': analysis.input_count,

            # Technical features
            'indicators_used': analysis.indicators_used,
            'trading_concepts': analysis.trading_concepts,
            'market_conditions': analysis.market_conditions,

            # Structural features
            'has_custom_functions': len(analysis.functions) > 0,
            'has_multiple_plots': len(analysis.plots) > 1,
            'configurable': len(analysis.inputs) > 3,

            # Semantic categories
            'primary_category': self._determine_primary_category(analysis),
            'signal_types': self._determine_signal_types(analysis),
            'best_for_timeframes': self._determine_timeframes(analysis),
            'best_for_markets': self._determine_market_types(analysis)
        }

        return features

    def _determine_primary_category(self, analysis: PineScriptAnalysis) -> str:
        """Determine the primary category of the indicator"""
        if 'oscillators' in analysis.indicators_used:
            return 'oscillator'
        elif 'trend_indicators' in analysis.indicators_used:
            return 'trend'
        elif 'volume_indicators' in analysis.indicators_used:
            return 'volume'
        elif 'volatility_indicators' in analysis.indicators_used:
            return 'volatility'
        elif 'order_blocks' in analysis.trading_concepts:
            return 'smart_money'
        else:
            return 'general'

    def _determine_signal_types(self, analysis: PineScriptAnalysis) -> List[str]:
        """Determine what types of signals this indicator provides"""
        signals = []

        if 'reversal' in analysis.trading_concepts:
            signals.append('reversal')
        if 'breakout' in analysis.trading_concepts:
            signals.append('breakout')
        if 'trend' in analysis.trading_concepts:
            signals.append('trend_following')
        if 'confluence' in analysis.trading_concepts:
            signals.append('confirmation')

        return signals

    def _determine_timeframes(self, analysis: PineScriptAnalysis) -> List[str]:
        """Determine best timeframes for this indicator"""
        timeframes = []

        # Heuristic based on complexity and type
        if analysis.complexity_score > 0.8:
            timeframes.extend(['1d', '4h', '1h'])
        elif analysis.complexity_score > 0.6:
            timeframes.extend(['4h', '1h', '15m'])
        else:
            timeframes.extend(['1h', '15m', '5m', '1m'])

        return timeframes

    def _determine_market_types(self, analysis: PineScriptAnalysis) -> List[str]:
        """Determine best market types for this indicator"""
        markets = []

        if 'trending' in analysis.market_conditions:
            markets.append('trending')
        if 'ranging' in analysis.market_conditions:
            markets.append('ranging')
        if 'volatile' in analysis.market_conditions:
            markets.append('volatile')

        # Default if nothing specific detected
        if not markets:
            markets.append('general')

        return markets

    def create_embedding_text(self, analysis: PineScriptAnalysis, metadata: Dict[str, Any]) -> str:
        """Create rich embedding text for semantic search"""
        parts = []

        # Title and description
        if analysis.title:
            parts.append(f"Title: {analysis.title}")
        if metadata.get('description'):
            parts.append(f"Description: {metadata['description']}")

        # Technical classification
        parts.append(f"Type: {analysis.script_type}")
        if analysis.overlay:
            parts.append("Overlay: True (displays on main chart)")

        # Indicators used
        if analysis.indicators_used:
            parts.append(f"Technical Indicators: {', '.join(analysis.indicators_used)}")

        # Trading concepts
        if analysis.trading_concepts:
            parts.append(f"Trading Concepts: {', '.join(analysis.trading_concepts)}")

        # Market conditions
        if analysis.market_conditions:
            parts.append(f"Best for: {', '.join(analysis.market_conditions)} markets")

        # Complexity and usage
        complexity_level = 'Expert' if analysis.complexity_score > 0.8 else 'Advanced' if analysis.complexity_score > 0.6 else 'Intermediate' if analysis.complexity_score > 0.4 else 'Basic'
        parts.append(f"Complexity: {complexity_level}")

        # Features
        features = self.extract_semantic_features(analysis)
        if features['configurable']:
            parts.append("Highly configurable with multiple parameters")
        if features['has_custom_functions']:
            parts.append("Contains custom functions and advanced logic")

        # Signal types
        if features['signal_types']:
            parts.append(f"Signals: {', '.join(features['signal_types'])}")

        # Timeframes
        if features['best_for_timeframes']:
            parts.append(f"Recommended timeframes: {', '.join(features['best_for_timeframes'][:3])}")

        return " | ".join(parts)


def test_pine_parser():
    """Test the Pine Script parser with sample code"""
    sample_code = '''
    //@version=5
    indicator("Enhanced RSI with Divergence", shorttitle="RSI+Div", overlay=false)

    // Input parameters
    length = input.int(14, "RSI Length", minval=1)
    source = input(close, "Source")
    overbought = input.int(70, "Overbought Level")
    oversold = input.int(30, "Oversold Level")

    // Calculate RSI
    rsi = ta.rsi(source, length)

    // Divergence detection
    bullish_div = ta.divergence(rsi, low, ltf_lookback=5, rtf_lookback=5, detect_type=ta.divergence_type.bullish)
    bearish_div = ta.divergence(rsi, high, ltf_lookback=5, rtf_lookback=5, detect_type=ta.divergence_type.bearish)

    // Plot RSI
    plot(rsi, "RSI", color=color.blue)
    hline(overbought, "Overbought", color=color.red)
    hline(oversold, "Oversold", color=color.green)

    // Plot divergences
    plotshape(bullish_div, "Bullish Divergence", shape.triangleup, location.bottom, color.green, size=size.small)
    plotshape(bearish_div, "Bearish Divergence", shape.triangledown, location.top, color.red, size=size.small)
    '''

    parser = EnhancedPineParser()
    analysis = parser.parse_pine_script(sample_code)

    print("=== Pine Script Analysis ===")
    print(f"Title: {analysis.title}")
    print(f"Type: {analysis.script_type}")
    print(f"Complexity: {analysis.complexity_score:.2f}")
    print(f"Inputs: {len(analysis.inputs)}")
    print(f"Functions: {len(analysis.functions)}")
    print(f"Plots: {len(analysis.plots)}")
    print(f"Indicators Used: {analysis.indicators_used}")
    print(f"Trading Concepts: {analysis.trading_concepts}")

    # Test embedding text creation
    features = parser.extract_semantic_features(analysis)
    embedding_text = parser.create_embedding_text(analysis, {"description": "RSI oscillator with divergence detection"})

    print(f"\n=== Embedding Text ===")
    print(embedding_text)

    print(f"\n=== Semantic Features ===")
    print(json.dumps(features, indent=2))

if __name__ == "__main__":
    test_pine_parser()