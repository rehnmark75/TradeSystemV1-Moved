# core/strategies/helpers/tradingview_script_parser.py
"""
TradingView Script Parser for Ichimoku Enhancement
Analyzes TradingView Ichimoku scripts and extracts actionable trading techniques

🎯 PARSING CAPABILITIES:
- Pine Script code analysis for Ichimoku variations
- Parameter extraction (periods, thresholds, filters)
- Signal generation logic identification
- Market regime specific techniques
- Alert and notification patterns

📊 EXTRACTED INSIGHTS:
- Fast/slow Ichimoku period combinations
- RSI/momentum filter integration patterns
- Cloud thickness and breakout logic
- Multi-timeframe validation techniques
- Scalping vs swing trading adaptations
"""

import re
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


@dataclass
class IchimokuVariation:
    """Represents an Ichimoku variation extracted from TradingView scripts"""
    name: str
    tenkan_period: int
    kijun_period: int
    senkou_b_period: int
    chikou_shift: int
    technique_type: str  # 'classic', 'fast', 'scalping', 'hybrid'
    additional_filters: List[str]
    market_conditions: List[str]
    confidence_score: float
    source_script: str


@dataclass
class TradingViewInsight:
    """Actionable insight extracted from TradingView scripts"""
    insight_type: str  # 'parameter_optimization', 'signal_enhancement', 'filter_addition'
    description: str
    implementation_code: str
    confidence: float
    market_applicability: List[str]


class TradingViewScriptParser:
    """
    📊 TRADINGVIEW SCRIPT ANALYSIS ENGINE

    Parses TradingView Ichimoku scripts and extracts:
    - Parameter variations and optimizations
    - Signal enhancement techniques
    - Filter and confirmation patterns
    - Market-specific adaptations
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_config = self._get_db_config()
        self.ichimoku_variations = []
        self.extracted_insights = []

        # Pine Script patterns for Ichimoku analysis
        self.pine_patterns = {
            'tenkan_period': r'tenkan[_\s]*(?:period|len|length)[^=]*=\s*(\d+)',
            'kijun_period': r'kijun[_\s]*(?:period|len|length)[^=]*=\s*(\d+)',
            'senkou_b_period': r'senkou[_\s]*b[_\s]*(?:period|len|length)[^=]*=\s*(\d+)',
            'chikou_shift': r'chikou[_\s]*(?:shift|lag|displacement)[^=]*=\s*(\d+)',
            'rsi_filter': r'rsi[_\s]*(?:filter|confirmation|threshold)',
            'momentum_filter': r'momentum[_\s]*(?:filter|confirmation)',
            'cloud_thickness': r'cloud[_\s]*(?:thickness|width|distance)',
            'tk_cross': r'(?:tenkan|tk)[_\s]*(?:cross|crossover|crossunder)',
            'cloud_breakout': r'cloud[_\s]*(?:breakout|break|penetration)',
            'alert_condition': r'alert\s*\(',
            'strategy_entry': r'strategy\.(?:entry|long|short)',
            'stop_loss': r'(?:stop|sl)[_\s]*(?:loss|pips)',
            'take_profit': r'(?:take|tp)[_\s]*(?:profit|pips)'
        }

        self.logger.info("📊 TradingView Script Parser initialized")

    def _get_db_config(self) -> Dict:
        """Get database configuration for TradingView script access"""
        return {
            'host': getattr(config, 'DB_HOST', 'postgres'),
            'database': getattr(config, 'DB_NAME', 'forex'),
            'user': getattr(config, 'DB_USER', 'postgres'),
            'password': getattr(config, 'DB_PASSWORD', 'postgres'),
            'port': getattr(config, 'DB_PORT', 5432)
        }

    def parse_ichimoku_scripts(self) -> List[IchimokuVariation]:
        """
        Parse all Ichimoku-related TradingView scripts from database

        Returns:
            List of extracted Ichimoku variations and techniques
        """
        try:
            self.logger.info("🔍 Starting TradingView Ichimoku script analysis...")

            # Connect to database and fetch Ichimoku scripts
            scripts = self._fetch_ichimoku_scripts()
            self.logger.info(f"📊 Found {len(scripts)} Ichimoku scripts to analyze")

            variations = []
            insights = []

            # Analyze each script
            for script in scripts:
                try:
                    # Extract Ichimoku variation
                    variation = self._analyze_script_parameters(script)
                    if variation:
                        variations.append(variation)

                    # Extract trading insights
                    script_insights = self._extract_trading_insights(script)
                    insights.extend(script_insights)

                except Exception as e:
                    self.logger.warning(f"Failed to analyze script {script.get('slug', 'unknown')}: {e}")
                    continue

            self.ichimoku_variations = variations
            self.extracted_insights = insights

            self.logger.info(f"✅ Script analysis complete: {len(variations)} variations, {len(insights)} insights")
            return variations

        except Exception as e:
            self.logger.error(f"❌ Script parsing failed: {e}")
            return []

    def _fetch_ichimoku_scripts(self) -> List[Dict]:
        """Fetch Ichimoku-related scripts from PostgreSQL database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Query for Ichimoku scripts
            query = """
                SELECT id, slug, title, author, description, code,
                       script_type, strategy_type, indicators, signals,
                       timeframes, likes, views, metadata
                FROM tradingview.scripts
                WHERE
                    LOWER(title) LIKE '%ichimoku%' OR
                    LOWER(description) LIKE '%ichimoku%' OR
                    'Ichimoku' = ANY(indicators) OR
                    LOWER(code) LIKE '%ichimoku%' OR
                    LOWER(code) LIKE '%tenkan%' OR
                    LOWER(code) LIKE '%kijun%' OR
                    LOWER(code) LIKE '%senkou%' OR
                    LOWER(code) LIKE '%chikou%' OR
                    LOWER(code) LIKE '%kumo%'
                ORDER BY likes DESC, views DESC
                LIMIT 50
            """

            cursor.execute(query)
            scripts = cursor.fetchall()

            cursor.close()
            conn.close()

            # Convert to list of dictionaries
            return [dict(script) for script in scripts]

        except Exception as e:
            self.logger.error(f"Database query failed: {e}")
            return []

    def _analyze_script_parameters(self, script: Dict) -> Optional[IchimokuVariation]:
        """Analyze a single script to extract Ichimoku parameter variation"""
        try:
            code = script.get('code', '')
            title = script.get('title', '')

            if not code:
                return None

            # Extract Ichimoku periods using regex patterns
            tenkan = self._extract_parameter(code, 'tenkan_period', default=9)
            kijun = self._extract_parameter(code, 'kijun_period', default=26)
            senkou_b = self._extract_parameter(code, 'senkou_b_period', default=52)
            chikou = self._extract_parameter(code, 'chikou_shift', default=26)

            # Determine technique type
            technique_type = self._classify_technique(code, title, tenkan, kijun, senkou_b)

            # Extract additional filters
            filters = self._extract_filters(code)

            # Determine market conditions
            market_conditions = self._extract_market_conditions(script)

            # Calculate confidence score based on popularity and code quality
            confidence = self._calculate_script_confidence(script, code)

            return IchimokuVariation(
                name=title,
                tenkan_period=tenkan,
                kijun_period=kijun,
                senkou_b_period=senkou_b,
                chikou_shift=chikou,
                technique_type=technique_type,
                additional_filters=filters,
                market_conditions=market_conditions,
                confidence_score=confidence,
                source_script=script.get('slug', '')
            )

        except Exception as e:
            self.logger.error(f"Parameter analysis failed for {script.get('slug', 'unknown')}: {e}")
            return None

    def _extract_parameter(self, code: str, param_type: str, default: int) -> int:
        """Extract a specific parameter from Pine Script code"""
        try:
            pattern = self.pine_patterns.get(param_type)
            if not pattern:
                return default

            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                # Return the first valid integer found
                for match in matches:
                    try:
                        value = int(match)
                        if 1 <= value <= 200:  # Reasonable bounds
                            return value
                    except ValueError:
                        continue

            return default

        except Exception:
            return default

    def _classify_technique(self, code: str, title: str, tenkan: int, kijun: int, senkou_b: int) -> str:
        """Classify the Ichimoku technique type based on parameters and content"""
        try:
            # Check for scalping indicators (fast periods)
            if tenkan <= 5 and kijun <= 15:
                return 'scalping'

            # Check for fast trading (faster than traditional)
            if tenkan < 9 or kijun < 26:
                return 'fast'

            # Check for hybrid approaches (additional indicators)
            if any(indicator in code.lower() for indicator in ['rsi', 'macd', 'stochastic', 'bollinger']):
                return 'hybrid'

            # Check title/description for type hints
            title_lower = title.lower()
            if 'scalp' in title_lower:
                return 'scalping'
            elif 'fast' in title_lower or 'quick' in title_lower:
                return 'fast'
            elif 'combined' in title_lower or 'hybrid' in title_lower:
                return 'hybrid'

            # Default to classic if using traditional periods
            return 'classic'

        except Exception:
            return 'classic'

    def _extract_filters(self, code: str) -> List[str]:
        """Extract additional filters and confirmations from the script"""
        filters = []

        try:
            # Check for common filter patterns
            filter_checks = {
                'rsi_filter': 'RSI filter',
                'momentum_filter': 'Momentum confirmation',
                'alert_condition': 'Alert system',
                'stop_loss': 'Stop loss logic',
                'take_profit': 'Take profit logic'
            }

            for pattern_name, filter_name in filter_checks.items():
                pattern = self.pine_patterns.get(pattern_name)
                if pattern and re.search(pattern, code, re.IGNORECASE):
                    filters.append(filter_name)

            # Check for volume filters
            if 'volume' in code.lower():
                filters.append('Volume filter')

            # Check for session filters
            if any(session in code.lower() for session in ['session', 'asian', 'london', 'new_york']):
                filters.append('Session filter')

            return filters

        except Exception:
            return []

    def _extract_market_conditions(self, script: Dict) -> List[str]:
        """Extract applicable market conditions from script metadata"""
        try:
            conditions = []

            # Check timeframes
            timeframes = script.get('timeframes', [])
            if timeframes:
                if any(tf in ['1m', '5m', '15m'] for tf in timeframes):
                    conditions.append('scalping')
                if any(tf in ['1h', '4h'] for tf in timeframes):
                    conditions.append('intraday')
                if any(tf in ['1d', '1w'] for tf in timeframes):
                    conditions.append('swing_trading')

            # Check strategy type
            strategy_type = script.get('strategy_type', '')
            if strategy_type:
                conditions.append(strategy_type)

            # Check description for market condition hints
            description = script.get('description', '').lower()
            if 'trending' in description:
                conditions.append('trending_markets')
            if 'ranging' in description or 'consolidation' in description:
                conditions.append('ranging_markets')
            if 'volatile' in description or 'breakout' in description:
                conditions.append('high_volatility')

            return list(set(conditions))  # Remove duplicates

        except Exception:
            return []

    def _calculate_script_confidence(self, script: Dict, code: str) -> float:
        """Calculate confidence score for the script based on various factors"""
        try:
            confidence = 0.5  # Base confidence

            # Popularity factor (up to 0.2 boost)
            likes = script.get('likes', 0)
            views = script.get('views', 0)
            popularity_score = min((likes / 10000) + (views / 100000), 0.2)
            confidence += popularity_score

            # Code quality factor (up to 0.15 boost)
            code_length = len(code)
            if code_length > 1000:  # Substantial code
                confidence += 0.1
            if '// ' in code or '/*' in code:  # Has comments
                confidence += 0.05

            # Open source factor
            if script.get('open_source', False):
                confidence += 0.05

            # Author reputation (simplified)
            author = script.get('author', '').lower()
            if author in ['tradingview', 'lazybear', 'luxalgo', 'chrismoody']:
                confidence += 0.1

            return min(confidence, 0.95)  # Cap at 95%

        except Exception:
            return 0.5

    def _extract_trading_insights(self, script: Dict) -> List[TradingViewInsight]:
        """Extract actionable trading insights from the script"""
        insights = []

        try:
            code = script.get('code', '')
            title = script.get('title', '')

            # Extract parameter optimization insights
            if 'input' in code.lower():
                insights.append(TradingViewInsight(
                    insight_type='parameter_optimization',
                    description=f'Configurable parameters found in {title}',
                    implementation_code=self._extract_parameter_code(code),
                    confidence=0.7,
                    market_applicability=['all_markets']
                ))

            # Extract signal enhancement techniques
            if any(pattern in code.lower() for pattern in ['crossover', 'crossunder', 'alert']):
                insights.append(TradingViewInsight(
                    insight_type='signal_enhancement',
                    description=f'Signal generation logic from {title}',
                    implementation_code=self._extract_signal_code(code),
                    confidence=0.8,
                    market_applicability=self._extract_market_conditions(script)
                ))

            # Extract filter techniques
            filters = self._extract_filters(code)
            if filters:
                insights.append(TradingViewInsight(
                    insight_type='filter_addition',
                    description=f'Filter techniques: {", ".join(filters)}',
                    implementation_code=self._extract_filter_code(code),
                    confidence=0.6,
                    market_applicability=self._extract_market_conditions(script)
                ))

            return insights

        except Exception as e:
            self.logger.error(f"Insight extraction failed: {e}")
            return []

    def _extract_parameter_code(self, code: str) -> str:
        """Extract parameter definition code segments"""
        try:
            # Find input parameter definitions
            lines = code.split('\n')
            param_lines = [line for line in lines if 'input' in line.lower() and ('tenkan' in line.lower() or 'kijun' in line.lower() or 'ichimoku' in line.lower())]
            return '\n'.join(param_lines[:5])  # Return first 5 relevant lines
        except Exception:
            return '// Parameter extraction failed'

    def _extract_signal_code(self, code: str) -> str:
        """Extract signal generation code segments"""
        try:
            # Find signal generation logic
            lines = code.split('\n')
            signal_lines = [line for line in lines if any(keyword in line.lower() for keyword in ['crossover', 'crossunder', 'plotshape', 'alert'])]
            return '\n'.join(signal_lines[:5])  # Return first 5 relevant lines
        except Exception:
            return '// Signal extraction failed'

    def _extract_filter_code(self, code: str) -> str:
        """Extract filter/confirmation code segments"""
        try:
            # Find filter logic
            lines = code.split('\n')
            filter_lines = [line for line in lines if any(keyword in line.lower() for keyword in ['rsi', 'filter', 'confirmation', 'and', 'or'])]
            return '\n'.join(filter_lines[:5])  # Return first 5 relevant lines
        except Exception:
            return '// Filter extraction failed'

    def get_best_variations_for_market(self, market_type: str, timeframe: str) -> List[IchimokuVariation]:
        """Get best Ichimoku variations for specific market conditions"""
        try:
            if not self.ichimoku_variations:
                self.parse_ichimoku_scripts()

            # Filter variations by market type and timeframe
            suitable_variations = []

            for variation in self.ichimoku_variations:
                # Check if variation is suitable for market type
                if market_type in variation.market_conditions or 'all_markets' in variation.market_conditions:
                    # Check timeframe compatibility
                    if timeframe in ['1m', '5m'] and variation.technique_type in ['scalping', 'fast']:
                        suitable_variations.append(variation)
                    elif timeframe in ['15m', '1h'] and variation.technique_type in ['fast', 'classic', 'hybrid']:
                        suitable_variations.append(variation)
                    elif timeframe in ['4h', '1d'] and variation.technique_type in ['classic', 'hybrid']:
                        suitable_variations.append(variation)

            # Sort by confidence score
            suitable_variations.sort(key=lambda x: x.confidence_score, reverse=True)

            return suitable_variations[:5]  # Return top 5

        except Exception as e:
            self.logger.error(f"Variation filtering failed: {e}")
            return []

    def get_insights_by_type(self, insight_type: str) -> List[TradingViewInsight]:
        """Get insights filtered by type"""
        try:
            if not self.extracted_insights:
                self.parse_ichimoku_scripts()

            return [insight for insight in self.extracted_insights if insight.insight_type == insight_type]

        except Exception as e:
            self.logger.error(f"Insight filtering failed: {e}")
            return []

    def generate_enhanced_parameters(self, epic: str, market_conditions: Dict) -> Dict:
        """Generate enhanced Ichimoku parameters based on TradingView analysis"""
        try:
            market_type = market_conditions.get('regime', 'trending')
            timeframe = market_conditions.get('timeframe', '15m')
            volatility = market_conditions.get('volatility', 'medium')

            # Get suitable variations
            variations = self.get_best_variations_for_market(market_type, timeframe)

            if not variations:
                return {'error': 'No suitable variations found'}

            # Select best variation based on confidence and market fit
            best_variation = variations[0]

            # Generate enhanced parameters
            enhanced_params = {
                'source': 'tradingview_analysis',
                'variation_name': best_variation.name,
                'technique_type': best_variation.technique_type,
                'confidence': best_variation.confidence_score,
                'parameters': {
                    'tenkan_period': best_variation.tenkan_period,
                    'kijun_period': best_variation.kijun_period,
                    'senkou_b_period': best_variation.senkou_b_period,
                    'chikou_shift': best_variation.chikou_shift
                },
                'filters': best_variation.additional_filters,
                'market_conditions': best_variation.market_conditions,
                'recommended_adjustments': self._generate_adjustments(best_variation, volatility)
            }

            return enhanced_params

        except Exception as e:
            self.logger.error(f"Enhanced parameter generation failed: {e}")
            return {'error': str(e)}

    def _generate_adjustments(self, variation: IchimokuVariation, volatility: str) -> Dict:
        """Generate parameter adjustments based on volatility"""
        adjustments = {}

        try:
            # Adjust for volatility
            if volatility == 'high':
                adjustments['confidence_threshold'] = 0.05  # Increase threshold in high volatility
                adjustments['stop_loss_modifier'] = 1.2  # Wider stops
                adjustments['take_profit_modifier'] = 1.5  # Wider targets
            elif volatility == 'low':
                adjustments['confidence_threshold'] = -0.03  # Lower threshold in low volatility
                adjustments['stop_loss_modifier'] = 0.8  # Tighter stops
                adjustments['take_profit_modifier'] = 0.9  # Tighter targets
            else:
                adjustments['confidence_threshold'] = 0.0  # No adjustment
                adjustments['stop_loss_modifier'] = 1.0
                adjustments['take_profit_modifier'] = 1.0

            # Technique-specific adjustments
            if variation.technique_type == 'scalping':
                adjustments['quick_exits'] = True
                adjustments['reduced_confirmation'] = True
            elif variation.technique_type == 'hybrid':
                adjustments['additional_confirmations'] = True
                adjustments['confluence_required'] = True

            return adjustments

        except Exception:
            return {}

    def get_statistics(self) -> Dict:
        """Get parsing statistics and summary"""
        try:
            if not self.ichimoku_variations:
                self.parse_ichimoku_scripts()

            stats = {
                'total_variations': len(self.ichimoku_variations),
                'total_insights': len(self.extracted_insights),
                'technique_distribution': {},
                'filter_frequency': {},
                'average_confidence': 0.0
            }

            # Technique distribution
            for variation in self.ichimoku_variations:
                technique = variation.technique_type
                stats['technique_distribution'][technique] = stats['technique_distribution'].get(technique, 0) + 1

            # Filter frequency
            for variation in self.ichimoku_variations:
                for filter_name in variation.additional_filters:
                    stats['filter_frequency'][filter_name] = stats['filter_frequency'].get(filter_name, 0) + 1

            # Average confidence
            if self.ichimoku_variations:
                total_confidence = sum(v.confidence_score for v in self.ichimoku_variations)
                stats['average_confidence'] = total_confidence / len(self.ichimoku_variations)

            return stats

        except Exception as e:
            self.logger.error(f"Statistics generation failed: {e}")
            return {'error': str(e)}