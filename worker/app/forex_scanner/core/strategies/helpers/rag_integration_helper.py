# core/strategies/helpers/rag_integration_helper.py
"""
RAG Integration Helper for RAG Intelligence Strategy
==================================================

Enhanced RAG system integration with fallback mechanisms, caching,
and intelligent query optimization for trading strategy selection.

Key Features:
- Robust RAG system connectivity with health monitoring
- Intelligent query building based on market conditions
- Response caching and optimization
- Fallback strategy selection when RAG unavailable
- Performance tracking and success rate analysis
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib

# TradingView API RAG Interface
class RAGInterface:
    """RAG Interface that connects to TradingView API service"""

    def __init__(self, base_url: str = "http://tradingview:8080"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def health_check(self) -> Dict[str, Any]:
        """Check if TradingView service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else {"status": "unhealthy"}
        except Exception:
            return {"status": "unavailable"}

    def search_indicators(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search TradingView indicators/scripts"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/tvscripts/search",
                params={"query": query, "limit": limit},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return {"indicators": data.get("results", []), "count": data.get("count", 0)}
            return {"error": f"API returned {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def search_templates(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search TradingView templates (alias for search_indicators)"""
        return self.search_indicators(query, limit)

    def get_recommendations(self, query: str) -> Dict[str, Any]:
        """Get script recommendations (alias for search_indicators)"""
        return self.search_indicators(query, 3)

    def compose_strategy(self, market_condition: str, indicators: List[str], **kwargs) -> Dict[str, Any]:
        """Compose strategy by finding best matching scripts"""
        query = f"{market_condition} {' '.join(indicators)}"
        return self.search_indicators(query, 1)

    def get_stats(self) -> Dict[str, Any]:
        """Get TradingView service stats"""
        try:
            response = self.session.get(f"{self.base_url}/api/tvscripts/stats", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception:
            return {}


@dataclass
class RAGQuery:
    """RAG query structure"""
    query_type: str  # 'strategy', 'indicator', 'template'
    query_text: str
    parameters: Dict[str, Any]
    market_context: Dict[str, Any]
    priority: int  # 1-5, higher is more important
    timestamp: datetime


@dataclass
class RAGResponse:
    """RAG response structure"""
    query_id: str
    response_type: str
    content: Dict[str, Any]
    confidence_score: float
    processing_time: float
    success: bool
    error_message: Optional[str]
    timestamp: datetime


@dataclass
class StrategyCode:
    """Enhanced strategy code structure"""
    code_id: str
    code_type: str
    description: str
    parameters: Dict[str, Any]
    market_suitability: List[str]
    complexity_level: str
    confidence_score: float
    source_indicators: List[str]
    backtest_performance: Optional[Dict]
    created_timestamp: datetime

    # TradingView specific fields
    pine_script: Optional[str] = None
    source_script_id: Optional[str] = None
    source_title: Optional[str] = None
    market_regime: Optional[str] = None


class RAGIntegrationHelper:
    """
    Enhanced RAG system integration helper with robust error handling,
    caching, and intelligent strategy selection capabilities.
    """

    def __init__(self,
                 rag_base_url: str = "http://tradingview:8080",
                 cache_duration_minutes: int = 10,
                 logger: Optional[logging.Logger] = None):

        self.rag_base_url = rag_base_url
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.logger = logger or logging.getLogger(__name__)

        # RAG interface
        self.rag_interface = None
        self.rag_available = False
        self.last_health_check = None

        # Caching system
        self.query_cache = {}
        self.strategy_cache = {}

        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'cached_responses': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'health_check_failures': 0,
            'fallback_activations': 0
        }

        # Fallback strategies
        self.fallback_strategies = self._initialize_fallback_strategies()

        # Initialize RAG connection
        self._initialize_rag_connection()

        self.logger.info(f"RAG Integration Helper initialized (RAG available: {self.rag_available})")

    def _initialize_rag_connection(self):
        """Initialize connection to RAG system with health check"""
        try:
            if RAGInterface is None:
                self.logger.warning("RAG Interface not available - using fallback mode")
                return

            self.rag_interface = RAGInterface(base_url=self.rag_base_url)

            # Perform health check
            health_result = self._perform_health_check()
            if health_result:
                self.rag_available = True
                self.logger.info("✅ RAG system connected and healthy")
            else:
                self.logger.warning("⚠️ RAG system unhealthy - using fallback mode")

        except Exception as e:
            self.logger.warning(f"RAG system initialization failed: {e}")
            self.rag_available = False

    def _perform_health_check(self) -> bool:
        """Perform RAG system health check"""
        try:
            if not self.rag_interface:
                return False

            health = self.rag_interface.health_check()
            self.last_health_check = datetime.utcnow()

            if health.get('status') == 'healthy':
                return True
            else:
                self.stats['health_check_failures'] += 1
                self.logger.warning(f"RAG health check failed: {health}")
                return False

        except Exception as e:
            self.stats['health_check_failures'] += 1
            self.logger.error(f"RAG health check error: {e}")
            return False

    def get_optimal_strategy_code(self,
                                 market_condition: Dict[str, Any],
                                 trading_context: Dict[str, Any]) -> StrategyCode:
        """
        Get optimal strategy code based on market conditions

        Args:
            market_condition: Current market regime and analysis
            trading_context: Trading session, timeframe, etc.

        Returns:
            StrategyCode object with optimal strategy
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(market_condition, trading_context)

            # Check cache first
            if self._is_cache_valid(cache_key):
                self.stats['cached_responses'] += 1
                cached_strategy = self.strategy_cache[cache_key]
                self.logger.debug(f"🎯 Using cached strategy: {cached_strategy.code_id}")
                return cached_strategy

            # Try RAG system if available
            if self.rag_available and self._should_use_rag():
                strategy_code = self._get_rag_strategy_code(market_condition, trading_context)
                if strategy_code:
                    # Cache successful result
                    self.strategy_cache[cache_key] = strategy_code
                    self.strategy_cache[f"{cache_key}_timestamp"] = datetime.utcnow()
                    return strategy_code

            # Fall back to predefined strategies
            self.stats['fallback_activations'] += 1
            self.logger.info("🔄 Using fallback strategy selection")
            fallback_strategy = self._get_fallback_strategy_code(market_condition, trading_context)

            # Cache fallback result (shorter duration)
            fallback_cache_key = f"{cache_key}_fallback"
            self.strategy_cache[fallback_cache_key] = fallback_strategy
            self.strategy_cache[f"{fallback_cache_key}_timestamp"] = datetime.utcnow()

            return fallback_strategy

        except Exception as e:
            self.logger.error(f"Strategy code selection failed: {e}")
            return self._get_emergency_fallback_strategy()

    def _get_rag_strategy_code(self,
                              market_condition: Dict[str, Any],
                              trading_context: Dict[str, Any]) -> Optional[StrategyCode]:
        """Get strategy code from TradingView scripts database"""
        try:
            self.stats['total_queries'] += 1
            start_time = time.time()

            # Query TradingView database for best matching script
            script_data = self._query_tradingview_scripts(market_condition, trading_context)

            processing_time = time.time() - start_time
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + processing_time) /
                self.stats['total_queries']
            )

            if not script_data:
                self.stats['failed_queries'] += 1
                self.logger.warning("No suitable TradingView scripts found")
                return None

            # Create strategy code from selected TradingView script
            strategy_code = self._create_strategy_from_script(script_data, market_condition)

            if strategy_code:
                self.stats['successful_queries'] += 1
                self.logger.info(f"📈 Selected TradingView script: '{strategy_code.code_id}' "
                               f"({strategy_code.confidence_score:.1%} confidence)")

            return strategy_code

        except Exception as e:
            self.stats['failed_queries'] += 1
            self.logger.error(f"TradingView script selection failed: {e}")
            return None

    def _query_tradingview_scripts(self,
                                  market_condition: Dict[str, Any],
                                  trading_context: Dict[str, Any]) -> Optional[Dict]:
        """Query TradingView service for best matching scripts"""
        try:
            regime = market_condition.get('regime', 'ranging')
            volatility = market_condition.get('volatility', 'medium')
            session = trading_context.get('session', 'london')

            # Build query for TradingView service
            query_params = {
                'market_regime': regime,
                'volatility': volatility,
                'session': session,
                'limit': 5
            }

            # Use the existing RAG interface to search for TradingView scripts
            if not self.rag_interface:
                return None

            # Build search query description
            query_description = f"{regime} {volatility} volatility {session} session trading strategy"

            # Search for indicators/strategies that match the market conditions
            response = self.rag_interface.search_indicators(query_description, limit=5)

            if response and 'error' not in response:
                # Convert RAG response to script format
                indicators = response.get('indicators', [])
                if indicators:
                    self.logger.info(f"🎯 Found {len(indicators)} matching TradingView indicators")
                    self.logger.info(f"📈 Selected: {indicators[0].get('title', 'Unknown')}")
                    return self._convert_rag_to_script_format(indicators[0], regime, volatility, session)
                else:
                    self.logger.warning("No matching TradingView indicators found")
                    return None
            else:
                self.logger.error(f"RAG search failed: {response.get('error', 'Unknown error')}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to query TradingView scripts: {e}")
            return None

    def _convert_rag_to_script_format(self,
                                    indicator_data: Dict,
                                    regime: str,
                                    volatility: str,
                                    session: str) -> Dict:
        """Convert TradingView API indicator data to script format"""
        try:
            # Extract data from TradingView API response
            slug = indicator_data.get('slug', f"rag_{regime}_{int(time.time())}")
            title = indicator_data.get('title', 'TradingView Selected Strategy')
            description = indicator_data.get('description', 'TradingView strategy selected for current market conditions')
            author = indicator_data.get('author', 'TradingView')

            # TradingView API doesn't include actual code, so we create synthetic trading logic
            # based on the script metadata (indicators, signals, etc.)
            indicators = indicator_data.get('indicators', [])
            signals = indicator_data.get('signals', [])
            timeframes = indicator_data.get('timeframes', ['15m', '1h'])

            # Generate synthetic Pine Script logic based on script metadata
            synthetic_code = self._generate_synthetic_trading_logic(
                title, indicators, signals, regime, volatility
            )

            # Create script data format compatible with our strategy
            script_data = {
                'id': slug,
                'title': title,
                'description': f"{description} (optimized for {regime} markets, {volatility} volatility, {session} session)",
                'author': author,
                'code': synthetic_code,
                'indicators': indicators,
                'signals': signals,
                'timeframes': timeframes,
                'strategy_type': self._infer_strategy_type_from_metadata(indicators, signals),
                'rating': min(0.95, indicator_data.get('likes', 1000) / 30000.0),  # Normalize likes to rating
                'market_regime': regime,
                'volatility': volatility,
                'session': session,
                'source': 'tradingview_api',
                'open_source': indicator_data.get('open_source', True),
                'likes': indicator_data.get('likes', 0),
                'views': indicator_data.get('views', 0)
            }

            return script_data

        except Exception as e:
            self.logger.error(f"Failed to convert TradingView data to script format: {e}")
            return None

    def _generate_synthetic_trading_logic(self,
                                      title: str,
                                      indicators: List[str],
                                      signals: List[str],
                                      regime: str,
                                      volatility: str) -> str:
        """Generate synthetic trading logic based on TradingView script metadata"""

        # Create a trading logic description based on the script's indicators and signals
        logic_parts = []

        # Add title context
        logic_parts.append(f"// {title} - Adapted for {regime} markets")

        # Add indicator logic
        if indicators:
            logic_parts.append("// Key Indicators:")
            for indicator in indicators[:3]:  # Use top 3 indicators
                logic_parts.append(f"// - {indicator}")

        # Add signal logic
        if signals:
            logic_parts.append("// Trading Signals:")
            for signal in signals[:3]:  # Use top 3 signals
                logic_parts.append(f"// - {signal}")

        # Add regime-specific logic
        if regime == 'trending_up':
            logic_parts.append("// Trending Up Market: Focus on momentum and trend continuation")
            entry_logic = "long_signal = close > ema_fast and volume > volume_avg"
        elif regime == 'trending_down':
            logic_parts.append("// Trending Down Market: Focus on short opportunities")
            entry_logic = "short_signal = close < ema_fast and volume > volume_avg"
        elif regime == 'ranging':
            logic_parts.append("// Ranging Market: Focus on mean reversion")
            entry_logic = "range_signal = (close > upper_band) or (close < lower_band)"
        else:  # breakout
            logic_parts.append("// Breakout Market: Focus on momentum breakouts")
            entry_logic = "breakout_signal = volume > volume_threshold and price_change > breakout_threshold"

        logic_parts.append(f"entry_condition = {entry_logic}")

        return "\n".join(logic_parts)

    def _infer_strategy_type_from_metadata(self, indicators: List[str], signals: List[str]) -> str:
        """Infer strategy type from TradingView script metadata"""
        combined_text = " ".join(indicators + signals).lower()

        if any(keyword in combined_text for keyword in ['trend', 'ema', 'moving average', 'ma']):
            return 'trend_following'
        elif any(keyword in combined_text for keyword in ['rsi', 'oscillator', 'stochastic', 'momentum']):
            return 'oscillator'
        elif any(keyword in combined_text for keyword in ['bollinger', 'band', 'channel', 'range']):
            return 'mean_reversion'
        elif any(keyword in combined_text for keyword in ['breakout', 'break', 'volume', 'volatility']):
            return 'breakout'
        elif any(keyword in combined_text for keyword in ['wave', 'elliott', 'fibonacci']):
            return 'pattern_recognition'
        else:
            return 'hybrid'

    def _infer_strategy_type(self, code: str, description: str) -> str:
        """Infer strategy type from code and description (legacy method)"""
        combined_text = f"{code} {description}".lower()

        if any(keyword in combined_text for keyword in ['ema', 'moving average', 'ma']):
            return 'trend_following'
        elif any(keyword in combined_text for keyword in ['rsi', 'oscillator', 'stochastic']):
            return 'oscillator'
        elif any(keyword in combined_text for keyword in ['bollinger', 'bands', 'channel']):
            return 'mean_reversion'
        elif any(keyword in combined_text for keyword in ['breakout', 'volatility', 'atr']):
            return 'breakout'
        else:
            return 'mixed'

    def _create_strategy_from_script(self,
                                   script_data: Dict,
                                   market_condition: Dict[str, Any]) -> StrategyCode:
        """Create executable strategy code from TradingView script"""
        try:
            # Extract key information from the script
            title = script_data.get('title', 'Unknown Strategy')
            description = script_data.get('description', 'TradingView strategy')
            pine_code = script_data.get('code', '')
            script_id = script_data.get('id', 'unknown')

            # Parse Pine Script parameters
            parameters = self._extract_pine_parameters(pine_code)

            # Calculate confidence based on script rating and market match
            base_confidence = script_data.get('rating', 0.7)  # Default rating
            regime_match = self._calculate_regime_match(script_data, market_condition)
            confidence_score = min(base_confidence * regime_match, 1.0)

            # Create strategy code object
            strategy_code = StrategyCode(
                code_id=f"tv_{script_id}_{int(time.time())}",
                code_type='tradingview_pine',
                description=f"{title}: {description[:100]}...",
                parameters=parameters,
                market_suitability=[market_condition.get('regime', 'ranging')],
                complexity_level='intermediate',
                confidence_score=confidence_score,
                source_indicators=[],
                backtest_performance=None,
                created_timestamp=datetime.utcnow(),
                pine_script=pine_code,
                source_script_id=script_id,
                source_title=title,
                market_regime=market_condition.get('regime', 'ranging')
            )

            self.logger.info(f"✅ Created strategy from TradingView script: {title}")
            return strategy_code

        except Exception as e:
            self.logger.error(f"Failed to create strategy from script: {e}")
            return None

    def _extract_pine_parameters(self, pine_code: str) -> Dict[str, Any]:
        """Extract configurable parameters from Pine Script code"""
        parameters = {}
        try:
            # Look for input() declarations in Pine Script
            import re
            input_pattern = r'(\w+)\s*=\s*input(?:\.(\w+))?\s*\(\s*([^,\)]+)(?:,\s*(?:title\s*=\s*)?["\']([^"\']*)["\'])?\s*\)'
            matches = re.findall(input_pattern, pine_code)

            for match in matches:
                var_name, input_type, default_value, title = match
                parameters[var_name] = {
                    'type': input_type or 'float',
                    'default': default_value.strip().strip('"\''),
                    'title': title or var_name
                }

            if not parameters:
                # Add some common default parameters
                parameters = {
                    'length': {'type': 'int', 'default': '14', 'title': 'Period Length'},
                    'multiplier': {'type': 'float', 'default': '2.0', 'title': 'Multiplier'}
                }

        except Exception as e:
            self.logger.warning(f"Parameter extraction failed: {e}")
            parameters = {'length': {'type': 'int', 'default': '14', 'title': 'Length'}}

        return parameters

    def _calculate_regime_match(self, script_data: Dict, market_condition: Dict) -> float:
        """Calculate how well the script matches current market regime"""
        try:
            regime = market_condition.get('regime', 'ranging')
            script_type = script_data.get('strategy_type', '').lower()
            script_description = script_data.get('description', '').lower()

            # Score based on strategy type matching
            match_score = 0.7  # Base score

            if regime == 'trending_up' or regime == 'trending_down':
                if any(word in script_type or word in script_description for word in
                       ['trend', 'momentum', 'breakout', 'moving average', 'ema', 'sma']):
                    match_score = 0.9
                elif any(word in script_type or word in script_description for word in
                        ['oscillator', 'rsi', 'mean reversion']):
                    match_score = 0.5

            elif regime == 'ranging':
                if any(word in script_type or word in script_description for word in
                       ['oscillator', 'rsi', 'mean reversion', 'bollinger', 'support', 'resistance']):
                    match_score = 0.9
                elif any(word in script_type or word in script_description for word in
                        ['trend', 'momentum', 'breakout']):
                    match_score = 0.6

            elif regime == 'breakout':
                if any(word in script_type or word in script_description for word in
                       ['breakout', 'volatility', 'channel', 'bollinger']):
                    match_score = 0.95

            return match_score

        except Exception:
            return 0.7

    def _build_rag_queries(self,
                          market_condition: Dict[str, Any],
                          trading_context: Dict[str, Any]) -> Dict[str, RAGQuery]:
        """Build comprehensive RAG queries based on market conditions"""
        try:
            regime = market_condition.get('regime', 'ranging')
            volatility = market_condition.get('volatility', 'medium')
            session = trading_context.get('session', 'london')
            timeframe = trading_context.get('timeframe', '15m')

            # Strategy composition query
            strategy_description = self._build_strategy_description(regime, volatility, session)
            strategy_query = RAGQuery(
                query_type='strategy',
                query_text=strategy_description,
                parameters={
                    'market_condition': regime,
                    'trading_style': self._determine_trading_style(regime, session),
                    'complexity_level': self._determine_complexity_level(volatility, timeframe)
                },
                market_context=market_condition,
                priority=5,
                timestamp=datetime.utcnow()
            )

            # Indicator search query
            indicator_query_text = self._build_indicator_query(regime, volatility)
            indicator_query = RAGQuery(
                query_type='indicator',
                query_text=indicator_query_text,
                parameters={'limit': 5},
                market_context=market_condition,
                priority=4,
                timestamp=datetime.utcnow()
            )

            # Template search query
            template_query_text = self._build_template_query(regime, session)
            template_query = RAGQuery(
                query_type='template',
                query_text=template_query_text,
                parameters={'limit': 3},
                market_context=market_condition,
                priority=3,
                timestamp=datetime.utcnow()
            )

            return {
                'strategy': strategy_query,
                'indicators': indicator_query,
                'templates': template_query
            }

        except Exception as e:
            self.logger.error(f"RAG query building failed: {e}")
            return {}

    def _execute_rag_query(self, query: RAGQuery) -> Optional[Dict]:
        """Execute a single RAG query with error handling"""
        try:
            if not self.rag_interface:
                return None

            if query.query_type == 'strategy':
                response = self.rag_interface.compose_strategy(
                    description=query.query_text,
                    market_condition=query.parameters.get('market_condition', 'ranging'),
                    trading_style=query.parameters.get('trading_style', 'day_trading'),
                    complexity_level=query.parameters.get('complexity_level', 'intermediate')
                )
            elif query.query_type == 'indicator':
                response = self.rag_interface.search_indicators(
                    query=query.query_text,
                    limit=query.parameters.get('limit', 5)
                )
            elif query.query_type == 'template':
                response = self.rag_interface.search_templates(
                    query=query.query_text,
                    limit=query.parameters.get('limit', 3)
                )
            else:
                self.logger.error(f"Unknown query type: {query.query_type}")
                return None

            # Check for errors in response
            if 'error' in response:
                self.logger.warning(f"RAG query error: {response['error']}")
                return None

            return response

        except Exception as e:
            self.logger.error(f"RAG query execution failed: {e}")
            return None

    def _synthesize_strategy_code(self,
                                 strategy_response: Dict,
                                 indicator_response: Dict,
                                 template_response: Optional[Dict],
                                 market_condition: Dict,
                                 trading_context: Dict) -> StrategyCode:
        """Synthesize RAG responses into unified strategy code"""
        try:
            # Generate unique strategy ID
            strategy_id = self._generate_strategy_id(strategy_response, market_condition)

            # Extract strategy description and parameters
            strategy_description = strategy_response.get('description', 'RAG-generated strategy')
            base_parameters = strategy_response.get('parameters', {})

            # Integrate indicators
            indicators = indicator_response.get('results', [])
            indicator_names = [ind.get('name', f'indicator_{i}') for i, ind in enumerate(indicators[:3])]

            # Extract template insights if available
            template_insights = {}
            if template_response and 'results' in template_response:
                templates = template_response['results']
                if templates:
                    template_insights = templates[0].get('parameters', {})

            # Merge parameters from all sources
            combined_parameters = self._merge_strategy_parameters(
                base_parameters,
                template_insights,
                market_condition,
                trading_context
            )

            # Calculate confidence score
            confidence_score = self._calculate_strategy_confidence(
                strategy_response,
                indicator_response,
                template_response,
                market_condition
            )

            # Determine market suitability
            market_suitability = self._determine_market_suitability(
                market_condition, combined_parameters
            )

            # Create strategy code
            strategy_code = StrategyCode(
                code_id=strategy_id,
                code_type='rag_composite',
                description=strategy_description,
                parameters=combined_parameters,
                market_suitability=market_suitability,
                complexity_level=trading_context.get('complexity', 'intermediate'),
                confidence_score=confidence_score,
                source_indicators=indicator_names,
                backtest_performance=None,  # Would be populated from historical data
                created_timestamp=datetime.utcnow()
            )

            return strategy_code

        except Exception as e:
            self.logger.error(f"Strategy synthesis failed: {e}")
            return self._get_emergency_fallback_strategy()

    def _get_fallback_strategy_code(self,
                                   market_condition: Dict[str, Any],
                                   trading_context: Dict[str, Any]) -> StrategyCode:
        """Get fallback strategy when RAG is unavailable"""
        try:
            regime = market_condition.get('regime', 'ranging')

            # Select appropriate fallback strategy
            if regime in ['trending_up', 'trending_down']:
                fallback = self.fallback_strategies['trend_following']
            elif regime == 'ranging':
                fallback = self.fallback_strategies['mean_reversion']
            elif regime == 'breakout':
                fallback = self.fallback_strategies['breakout']
            else:
                fallback = self.fallback_strategies['adaptive']

            # Customize parameters based on market condition
            customized_parameters = self._customize_fallback_parameters(
                fallback['parameters'].copy(),
                market_condition,
                trading_context
            )

            strategy_code = StrategyCode(
                code_id=f"fallback_{regime}_{datetime.utcnow().strftime('%H%M%S')}",
                code_type='fallback',
                description=fallback['description'],
                parameters=customized_parameters,
                market_suitability=[regime],
                complexity_level='intermediate',
                confidence_score=0.6,  # Lower confidence for fallback
                source_indicators=fallback['indicators'],
                backtest_performance=None,
                created_timestamp=datetime.utcnow()
            )

            self.logger.info(f"🔄 Fallback strategy selected: {strategy_code.code_id}")
            return strategy_code

        except Exception as e:
            self.logger.error(f"Fallback strategy selection failed: {e}")
            return self._get_emergency_fallback_strategy()

    def _initialize_fallback_strategies(self) -> Dict[str, Dict]:
        """Initialize predefined fallback strategies"""
        return {
            'trend_following': {
                'description': 'EMA-based trend following strategy',
                'parameters': {
                    'fast_ema': 12,
                    'slow_ema': 26,
                    'trend_ema': 50,
                    'stop_loss_pips': 20,
                    'take_profit_pips': 40,
                    'risk_reward_ratio': 2.0
                },
                'indicators': ['EMA', 'MACD', 'RSI']
            },
            'mean_reversion': {
                'description': 'Bollinger Bands mean reversion strategy',
                'parameters': {
                    'bb_period': 20,
                    'bb_std_dev': 2.0,
                    'rsi_period': 14,
                    'rsi_oversold': 20,
                    'rsi_overbought': 80,
                    'stop_loss_pips': 15,
                    'take_profit_pips': 25
                },
                'indicators': ['BollingerBands', 'RSI', 'StochasticOscillator']
            },
            'breakout': {
                'description': 'ATR-based breakout strategy',
                'parameters': {
                    'atr_period': 14,
                    'atr_multiplier': 2.0,
                    'breakout_lookback': 20,
                    'volume_threshold': 1.2,
                    'stop_loss_pips': 25,
                    'take_profit_pips': 60
                },
                'indicators': ['ATR', 'VolumeOscillator', 'DonchianChannels']
            },
            'adaptive': {
                'description': 'Adaptive multi-indicator strategy',
                'parameters': {
                    'primary_ema': 21,
                    'secondary_ema': 50,
                    'atr_period': 14,
                    'rsi_period': 14,
                    'stop_loss_pips': 18,
                    'take_profit_pips': 36
                },
                'indicators': ['EMA', 'ATR', 'RSI', 'MACD']
            }
        }

    # Helper methods

    def _generate_cache_key(self, market_condition: Dict, trading_context: Dict) -> str:
        """Generate cache key from market conditions and context"""
        key_data = {
            'regime': market_condition.get('regime', 'unknown'),
            'volatility': market_condition.get('volatility', 'medium'),
            'session': trading_context.get('session', 'unknown'),
            'timeframe': trading_context.get('timeframe', '15m')
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        if cache_key not in self.strategy_cache:
            return False

        timestamp_key = f"{cache_key}_timestamp"
        if timestamp_key not in self.strategy_cache:
            return False

        cache_time = self.strategy_cache[timestamp_key]
        return datetime.utcnow() - cache_time < self.cache_duration

    def _should_use_rag(self) -> bool:
        """Determine if RAG system should be used"""
        if not self.rag_available:
            return False

        # Check if health check is needed
        if (not self.last_health_check or
            datetime.utcnow() - self.last_health_check > timedelta(minutes=5)):
            self.rag_available = self._perform_health_check()

        return self.rag_available

    def _generate_strategy_id(self, strategy_response: Dict, market_condition: Dict) -> str:
        """Generate unique strategy ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        regime = market_condition.get('regime', 'unknown')[:4]
        hash_part = hashlib.md5(json.dumps(strategy_response).encode()).hexdigest()[:8]
        return f"rag_{regime}_{timestamp}_{hash_part}"

    def _build_strategy_description(self, regime: str, volatility: str, session: str) -> str:
        """Build strategy description for RAG query"""
        descriptions = {
            'trending_up': f'bullish momentum trend-following strategy for {volatility} volatility {session} session',
            'trending_down': f'bearish momentum trend-following strategy for {volatility} volatility {session} session',
            'ranging': f'mean reversion oscillator strategy for {volatility} volatility sideways markets',
            'breakout': f'volatility breakout strategy for explosive {volatility} volatility moves'
        }
        return descriptions.get(regime, 'adaptive multi-timeframe trading strategy')

    def _determine_trading_style(self, regime: str, session: str) -> str:
        """Determine trading style based on regime and session"""
        if session == 'asian':
            return 'range_trading'
        elif regime in ['trending_up', 'trending_down']:
            return 'swing_trading'
        elif regime == 'breakout':
            return 'momentum_trading'
        else:
            return 'day_trading'

    def _determine_complexity_level(self, volatility: str, timeframe: str) -> str:
        """Determine complexity level based on conditions"""
        if volatility in ['low', 'medium_low']:
            return 'beginner'
        elif volatility == 'high' and timeframe in ['5m']:
            return 'advanced'
        else:
            return 'intermediate'

    def _build_indicator_query(self, regime: str, volatility: str) -> str:
        """Build indicator search query"""
        queries = {
            'trending_up': f'bullish trend indicators momentum oscillators {volatility} volatility',
            'trending_down': f'bearish trend indicators momentum oscillators {volatility} volatility',
            'ranging': f'mean reversion indicators oscillators bollinger bands RSI {volatility} volatility',
            'breakout': f'volatility indicators breakout systems ATR donchian channels {volatility} volatility'
        }
        return queries.get(regime, 'technical indicators multi-timeframe analysis')

    def _build_template_query(self, regime: str, session: str) -> str:
        """Build template search query"""
        queries = {
            'trending_up': f'bullish trend continuation templates {session} session',
            'trending_down': f'bearish trend continuation templates {session} session',
            'ranging': f'range trading templates mean reversion {session} session',
            'breakout': f'breakout trading templates volatility expansion {session} session'
        }
        return queries.get(regime, 'adaptive trading templates')

    def _merge_strategy_parameters(self,
                                  base_params: Dict,
                                  template_params: Dict,
                                  market_condition: Dict,
                                  trading_context: Dict) -> Dict:
        """Merge parameters from multiple sources"""
        merged = base_params.copy()
        merged.update(template_params)

        # Add market-specific adjustments
        volatility = market_condition.get('volatility', 'medium')
        if volatility in ['high', 'peak']:
            # Increase stop loss and take profit for high volatility
            merged['stop_loss_pips'] = merged.get('stop_loss_pips', 20) * 1.3
            merged['take_profit_pips'] = merged.get('take_profit_pips', 40) * 1.3
        elif volatility == 'low':
            # Decrease for low volatility
            merged['stop_loss_pips'] = merged.get('stop_loss_pips', 20) * 0.7
            merged['take_profit_pips'] = merged.get('take_profit_pips', 40) * 0.7

        return merged

    def _calculate_strategy_confidence(self,
                                      strategy_response: Dict,
                                      indicator_response: Dict,
                                      template_response: Optional[Dict],
                                      market_condition: Dict) -> float:
        """Calculate confidence score for strategy"""
        confidence = 0.6  # Base confidence

        # RAG response quality
        if strategy_response.get('confidence', 0) > 0.8:
            confidence += 0.1

        if indicator_response.get('results'):
            confidence += 0.1

        if template_response and template_response.get('results'):
            confidence += 0.05

        # Market condition confidence
        regime_confidence = market_condition.get('confidence', 0.5)
        confidence += (regime_confidence - 0.5) * 0.2

        return min(0.95, max(0.4, confidence))

    def _determine_market_suitability(self, market_condition: Dict, parameters: Dict) -> List[str]:
        """Determine which market conditions strategy is suitable for"""
        regime = market_condition.get('regime', 'ranging')
        base_suitability = [regime]

        # Add related regimes based on parameters
        if 'trend_ema' in parameters:
            base_suitability.extend(['trending_up', 'trending_down'])
        if 'bb_std_dev' in parameters:
            base_suitability.append('ranging')
        if 'atr_multiplier' in parameters:
            base_suitability.append('breakout')

        return list(set(base_suitability))

    def _customize_fallback_parameters(self,
                                      base_params: Dict,
                                      market_condition: Dict,
                                      trading_context: Dict) -> Dict:
        """Customize fallback parameters based on market conditions"""
        params = base_params.copy()

        # Adjust for volatility
        volatility = market_condition.get('volatility', 'medium')
        if volatility in ['high', 'peak']:
            if 'stop_loss_pips' in params:
                params['stop_loss_pips'] *= 1.4
            if 'take_profit_pips' in params:
                params['take_profit_pips'] *= 1.4
        elif volatility == 'low':
            if 'stop_loss_pips' in params:
                params['stop_loss_pips'] *= 0.6
            if 'take_profit_pips' in params:
                params['take_profit_pips'] *= 0.6

        return params

    def _get_emergency_fallback_strategy(self) -> StrategyCode:
        """Get emergency fallback strategy for critical failures"""
        return StrategyCode(
            code_id=f"emergency_{datetime.utcnow().strftime('%H%M%S')}",
            code_type='emergency_fallback',
            description='Conservative EMA crossover strategy',
            parameters={
                'fast_ema': 12,
                'slow_ema': 26,
                'stop_loss_pips': 15,
                'take_profit_pips': 30,
                'risk_reward_ratio': 2.0
            },
            market_suitability=['ranging'],
            complexity_level='beginner',
            confidence_score=0.4,
            source_indicators=['EMA'],
            backtest_performance=None,
            created_timestamp=datetime.utcnow()
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get RAG integration performance statistics"""
        return {
            **self.stats,
            'rag_available': self.rag_available,
            'cache_size': len(self.strategy_cache),
            'fallback_strategies_available': len(self.fallback_strategies),
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None
        }