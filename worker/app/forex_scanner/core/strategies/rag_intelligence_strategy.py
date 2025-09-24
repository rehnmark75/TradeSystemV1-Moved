# core/strategies/rag_intelligence_strategy.py
"""
RAG-Enhanced Market Intelligence Strategy Implementation
=====================================================

Features:
- Market Intelligence Data Integration: Analyzes last 24 hours of market conditions
- RAG System Integration: Dynamically selects optimal TradingView code
- Adaptive Signal Detection: Adjusts strategy based on market regime
- Session-Aware Optimization: Optimizes for current trading session
- Database-Driven Parameters: Uses intelligence data for parameter tuning

The strategy follows the existing BaseStrategy pattern while adding:
1. Market regime analysis using intelligence database
2. Dynamic TradingView code selection via RAG
3. Adaptive parameter optimization
4. Session-aware signal filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging
from datetime import datetime, timedelta
import json
import requests
from dataclasses import dataclass

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
from .helpers.market_intelligence_analyzer import MarketIntelligenceAnalyzer
from .helpers.rag_integration_helper import RAGIntegrationHelper

try:
    from configdata import config
    from configdata.strategies.config_rag_intelligence_strategy import RAGIntelligenceConfig
except ImportError:
    try:
        from forex_scanner.configdata import config
        from forex_scanner.configdata.strategies.config_rag_intelligence_strategy import RAGIntelligenceConfig
    except ImportError:
        # Create minimal fallback config
        class MinimalConfig:
            MIN_CONFIDENCE = 0.6
            RAG_CACHE_DURATION_MINUTES = 10
        config = MinimalConfig()

        class RAGIntelligenceConfig:
            def __init__(self):
                self.STRATEGY_NAME = "RAG Intelligence"
                self.MIN_CONFIDENCE = 0.6
                self.RAG_CACHE_DURATION_MINUTES = 10
                self.MARKET_REGIMES = {
                    'trending_up': {'weight': 1.0},
                    'trending_down': {'weight': 1.0},
                    'ranging': {'weight': 0.8},
                    'breakout': {'weight': 0.9}
                }
                self.RAG_QUERY_TEMPLATES = {}

# RAG Interface import with better error handling
RAGInterface = None
try:
    # Try different import paths
    import sys
    import os

    # Method 1: Direct import
    try:
        from rag_interface import RAGInterface
    except ImportError:
        # Method 2: Add project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        sys.path.append(project_root)
        try:
            from rag_interface import RAGInterface
        except ImportError:
            # Method 3: Try absolute path
            try:
                sys.path.append('/home/hr/Projects/TradeSystemV1')
                from rag_interface import RAGInterface
            except ImportError:
                RAGInterface = None
except Exception:
    RAGInterface = None


@dataclass
class MarketCondition:
    """Market condition data structure"""
    regime: str  # trending_up, trending_down, ranging, breakout
    confidence: float
    session: str  # asian, london, new_york, overlap
    volatility: str  # low, medium, high, peak
    dominant_timeframe: str
    success_factors: List[str]
    timestamp: datetime


@dataclass
class RAGStrategyCode:
    """RAG-selected strategy code structure"""
    code_type: str  # indicator, template, composite
    code_content: str
    parameters: Dict[str, Any]
    market_suitability: str
    confidence_score: float
    source_id: str


class RAGIntelligenceStrategy(BaseStrategy):
    """
    RAG-Enhanced Market Intelligence Strategy

    This strategy combines:
    1. Market Intelligence Analysis: Real-time market regime detection
    2. RAG Code Selection: Dynamic TradingView strategy selection
    3. Adaptive Parameters: Intelligence-driven parameter optimization
    4. Session Awareness: Trading session optimization
    """

    def __init__(self,
                 epic: str = None,
                 data_fetcher=None,
                 backtest_mode: bool = False,
                 market_analysis_hours: int = 24,
                 rag_base_url: str = "http://localhost:8090"):

        # Initialize parent class
        super().__init__('rag_intelligence')

        # Core attributes
        self.epic = epic
        self.data_fetcher = data_fetcher
        self.backtest_mode = backtest_mode
        self.market_analysis_hours = market_analysis_hours

        # Components initialization
        self.price_adjuster = PriceAdjuster()

        # Configuration
        self.config = RAGIntelligenceConfig()

        # Market Intelligence Analyzer
        self.intelligence_analyzer = None
        if data_fetcher and hasattr(data_fetcher, 'db_manager'):
            try:
                self.intelligence_analyzer = MarketIntelligenceAnalyzer(
                    db_manager=data_fetcher.db_manager,
                    logger=self.logger
                )
                self.logger.info("✅ Market Intelligence Analyzer initialized")
            except Exception as e:
                self.logger.warning(f"Market Intelligence Analyzer initialization failed: {e}")
        else:
            self.logger.warning("No database manager available - using simplified intelligence")

        # RAG System Integration using enhanced helper
        self.rag_helper = RAGIntegrationHelper(
            rag_base_url=rag_base_url,
            cache_duration_minutes=self.config.RAG_CACHE_DURATION_MINUTES,
            logger=self.logger
        )

        # Market Intelligence Components
        self.current_market_condition = None
        self.selected_strategy_code = None
        self.intelligence_cache = {}
        self.cache_duration = timedelta(minutes=5)  # Cache intelligence data for 5 minutes

        # Performance tracking
        self.stats = {
            'total_signals': 0,
            'rag_selections': 0,
            'intelligence_queries': 0,
            'regime_changes': 0,
            'cache_hits': 0
        }

        # Strategy parameters (will be dynamically updated)
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.6)  # Higher default for intelligence strategy
        self.min_bars = 100  # Need more data for intelligence analysis

        self.logger.info(f"RAG Intelligence Strategy initialized for {epic or 'multi-epic'}")

    def analyze_market_conditions(self, epic: str) -> MarketCondition:
        """
        Analyze market conditions using intelligence database

        Args:
            epic: Trading instrument to analyze

        Returns:
            MarketCondition object with current market analysis
        """
        cache_key = f"market_condition_{epic}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            self.stats['cache_hits'] += 1
            return self.intelligence_cache[cache_key]

        try:
            self.stats['intelligence_queries'] += 1

            # Use market intelligence analyzer if available
            if self.intelligence_analyzer:
                analysis = self.intelligence_analyzer.analyze_market_conditions(
                    epic, self.market_analysis_hours
                )

                # Extract regime analysis
                regime_analysis = analysis.get('regime_analysis')
                session_analysis = analysis.get('session_analysis')
                success_patterns = analysis.get('success_patterns', [])

                # Create market condition from analysis
                market_condition = MarketCondition(
                    regime=regime_analysis.regime,
                    confidence=regime_analysis.confidence,
                    session=session_analysis.current_session,
                    volatility=session_analysis.volatility_forecast,
                    dominant_timeframe=session_analysis.recommended_timeframes[0] if session_analysis.recommended_timeframes else '15m',
                    success_factors=[p.pattern_type for p in success_patterns[:3]],  # Top 3 patterns
                    timestamp=datetime.utcnow()
                )

            else:
                # Fallback to simplified analysis
                self.logger.warning("Using simplified market analysis (no intelligence analyzer)")
                market_condition = self._get_fallback_market_condition()

            # Cache the result
            self.intelligence_cache[cache_key] = market_condition
            self.intelligence_cache[f"{cache_key}_timestamp"] = datetime.utcnow()

            # Log regime changes
            if (self.current_market_condition and
                self.current_market_condition.regime != market_condition.regime):
                self.stats['regime_changes'] += 1
                self.logger.info(f"📊 Regime change detected: {self.current_market_condition.regime} -> {market_condition.regime}")

            self.current_market_condition = market_condition
            return market_condition

        except Exception as e:
            self.logger.error(f"Market condition analysis failed: {e}")
            return self._get_fallback_market_condition()

    def select_optimal_code(self, market_condition: MarketCondition) -> RAGStrategyCode:
        """
        Use RAG system to select optimal TradingView code for current market conditions

        Args:
            market_condition: Current market condition analysis

        Returns:
            RAGStrategyCode object with selected strategy code and parameters
        """
        try:
            self.stats['rag_selections'] += 1

            # Prepare market condition and trading context for RAG helper
            market_condition_dict = {
                'regime': market_condition.regime,
                'confidence': market_condition.confidence,
                'volatility': market_condition.volatility,
                'session': market_condition.session
            }

            trading_context = {
                'session': market_condition.session,
                'timeframe': market_condition.dominant_timeframe,
                'success_factors': market_condition.success_factors,
                'complexity': 'intermediate'
            }

            # Use RAG helper to get optimal strategy code
            strategy_code_enhanced = self.rag_helper.get_optimal_strategy_code(
                market_condition_dict,
                trading_context
            )

            # Convert enhanced strategy code to our RAGStrategyCode format
            strategy_code = RAGStrategyCode(
                code_type=strategy_code_enhanced.code_type,
                code_content=strategy_code_enhanced.description,
                parameters=strategy_code_enhanced.parameters,
                market_suitability=market_condition.regime,
                confidence_score=strategy_code_enhanced.confidence_score,
                source_id=strategy_code_enhanced.code_id
            )

            self.selected_strategy_code = strategy_code
            self.logger.info(f"🤖 RAG selected {strategy_code.code_type} strategy '{strategy_code.source_id}' "
                           f"with {strategy_code.confidence_score:.1%} confidence")

            return strategy_code

        except Exception as e:
            self.logger.error(f"RAG code selection failed: {e}")
            return self._get_fallback_strategy_code(market_condition)

    def detect_signal(self,
                     df: pd.DataFrame,
                     epic: str,
                     spread_pips: float = 1.5,
                     timeframe: str = '5m') -> Optional[Dict]:
        """
        Detect trading signals using RAG-selected strategy and market intelligence

        Args:
            df: OHLCV data
            epic: Trading instrument
            spread_pips: Spread cost
            timeframe: Analysis timeframe

        Returns:
            Signal dictionary or None if no signal
        """
        try:
            self.stats['total_signals'] += 1

            # Ensure we have enough data
            if len(df) < self.min_bars:
                self.logger.debug(f"Insufficient data: {len(df)} bars < {self.min_bars} required")
                return None

            # Step 1: Analyze current market conditions
            market_condition = self.analyze_market_conditions(epic)

            # Step 2: Select optimal strategy code via RAG
            strategy_code = self.select_optimal_code(market_condition)

            # Step 3: Execute strategy code to detect signals
            raw_signal = self._execute_strategy_code(df, strategy_code, market_condition, timeframe)

            if not raw_signal:
                return None

            # Step 4: Apply intelligence-based filtering
            filtered_signal = self._apply_intelligence_filtering(
                raw_signal,
                market_condition,
                df,
                epic,
                spread_pips
            )

            if not filtered_signal:
                self.logger.debug("Signal filtered out by intelligence system")
                return None

            # Step 5: Enhance signal with intelligence context
            enhanced_signal = self._enhance_signal_with_intelligence(
                filtered_signal,
                market_condition,
                strategy_code
            )

            self.logger.info(f"🎯 Intelligence signal generated: {enhanced_signal['direction']} @ {enhanced_signal['confidence']:.1%}")

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"Signal detection failed: {e}")
            return None

    def _execute_strategy_code(self,
                              df: pd.DataFrame,
                              strategy_code: RAGStrategyCode,
                              market_condition: MarketCondition,
                              timeframe: str) -> Optional[Dict]:
        """
        Execute the RAG-selected TradingView Pine Script to generate raw signals
        """
        try:
            # Check if we have actual Pine Script code
            if hasattr(strategy_code, 'pine_script') and strategy_code.pine_script:
                return self._execute_pine_script(df, strategy_code, market_condition)
            else:
                # Fallback to simplified logic if no Pine Script available
                self.logger.warning("No Pine Script code available, using fallback logic")
                return self._execute_fallback_logic(df, strategy_code, market_condition)

        except Exception as e:
            self.logger.error(f"Strategy code execution failed: {e}")
            return None

    def _execute_trend_following_logic(self,
                                     df: pd.DataFrame,
                                     direction: str,
                                     strategy_code: RAGStrategyCode) -> Optional[Dict]:
        """Execute trend-following strategy logic"""
        try:
            # Calculate EMAs based on strategy parameters
            fast_ema = strategy_code.parameters.get('fast_ema', 12)
            slow_ema = strategy_code.parameters.get('slow_ema', 26)

            df['ema_fast'] = df['close'].ewm(span=fast_ema).mean()
            df['ema_slow'] = df['close'].ewm(span=slow_ema).mean()

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Trend following logic
            if direction == 'BUY':
                # Check for bullish EMA crossover or continuation
                if (latest['ema_fast'] > latest['ema_slow'] and
                    latest['close'] > latest['ema_fast']):

                    # Additional momentum check
                    if latest['ema_fast'] > prev['ema_fast']:
                        return {
                            'direction': 'BUY',
                            'entry_price': latest['close'],
                            'confidence': 0.7,
                            'signal_type': 'trend_following',
                            'timestamp': latest['timestamp'] if 'timestamp' in latest else datetime.utcnow()
                        }

            elif direction == 'SELL':
                # Check for bearish EMA crossover or continuation
                if (latest['ema_fast'] < latest['ema_slow'] and
                    latest['close'] < latest['ema_fast']):

                    # Additional momentum check
                    if latest['ema_fast'] < prev['ema_fast']:
                        return {
                            'direction': 'SELL',
                            'entry_price': latest['close'],
                            'confidence': 0.7,
                            'signal_type': 'trend_following',
                            'timestamp': latest['timestamp'] if 'timestamp' in latest else datetime.utcnow()
                        }

            return None

        except Exception as e:
            self.logger.error(f"Trend following logic failed: {e}")
            return None

    def _execute_mean_reversion_logic(self,
                                    df: pd.DataFrame,
                                    strategy_code: RAGStrategyCode) -> Optional[Dict]:
        """Execute mean reversion strategy logic for ranging markets"""
        try:
            # Calculate Bollinger Bands for mean reversion
            period = strategy_code.parameters.get('bb_period', 20)
            std_dev = strategy_code.parameters.get('bb_std', 2.0)

            df['sma'] = df['close'].rolling(window=period).mean()
            df['std'] = df['close'].rolling(window=period).std()
            df['bb_upper'] = df['sma'] + (df['std'] * std_dev)
            df['bb_lower'] = df['sma'] - (df['std'] * std_dev)

            latest = df.iloc[-1]

            # Mean reversion signals
            if latest['close'] <= latest['bb_lower']:
                return {
                    'direction': 'BUY',
                    'entry_price': latest['close'],
                    'confidence': 0.65,
                    'signal_type': 'mean_reversion',
                    'timestamp': latest['timestamp'] if 'timestamp' in latest else datetime.utcnow()
                }

            elif latest['close'] >= latest['bb_upper']:
                return {
                    'direction': 'SELL',
                    'entry_price': latest['close'],
                    'confidence': 0.65,
                    'signal_type': 'mean_reversion',
                    'timestamp': latest['timestamp'] if 'timestamp' in latest else datetime.utcnow()
                }

            return None

        except Exception as e:
            self.logger.error(f"Mean reversion logic failed: {e}")
            return None

    def _execute_breakout_logic(self,
                              df: pd.DataFrame,
                              strategy_code: RAGStrategyCode) -> Optional[Dict]:
        """Execute breakout strategy logic for volatile markets"""
        try:
            # Calculate ATR for volatility-based breakouts
            lookback = strategy_code.parameters.get('atr_period', 14)
            multiplier = strategy_code.parameters.get('atr_multiplier', 2.0)

            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_close'] = abs(df['low'] - df['close'].shift(1))
            df['atr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1).rolling(lookback).mean()

            # Recent high/low for breakout detection
            recent_high = df['high'].rolling(lookback).max().iloc[-1]
            recent_low = df['low'].rolling(lookback).min().iloc[-1]

            latest = df.iloc[-1]

            # Breakout signals
            if latest['close'] > recent_high:
                return {
                    'direction': 'BUY',
                    'entry_price': latest['close'],
                    'confidence': 0.75,
                    'signal_type': 'breakout',
                    'timestamp': latest['timestamp'] if 'timestamp' in latest else datetime.utcnow()
                }

            elif latest['close'] < recent_low:
                return {
                    'direction': 'SELL',
                    'entry_price': latest['close'],
                    'confidence': 0.75,
                    'signal_type': 'breakout',
                    'timestamp': latest['timestamp'] if 'timestamp' in latest else datetime.utcnow()
                }

            return None

        except Exception as e:
            self.logger.error(f"Breakout logic failed: {e}")
            return None

    def _apply_intelligence_filtering(self,
                                    signal: Dict,
                                    market_condition: MarketCondition,
                                    df: pd.DataFrame,
                                    epic: str,
                                    spread_pips: float) -> Optional[Dict]:
        """Apply intelligence-based signal filtering"""
        try:
            # Session filtering
            if not self._is_favorable_session(market_condition):
                self.logger.debug(f"Signal filtered: unfavorable session {market_condition.session}")
                return None

            # Regime alignment check
            if not self._check_regime_alignment(signal, market_condition):
                self.logger.debug(f"Signal filtered: poor regime alignment")
                return None

            # Volatility check
            if not self._check_volatility_suitability(market_condition, df):
                self.logger.debug(f"Signal filtered: unsuitable volatility")
                return None

            # Confidence boost from intelligence
            confidence_boost = self._calculate_intelligence_confidence_boost(
                signal, market_condition
            )

            signal['confidence'] = min(0.95, signal['confidence'] + confidence_boost)

            # Final confidence check
            if signal['confidence'] < self.min_confidence:
                self.logger.debug(f"Signal filtered: low confidence {signal['confidence']:.1%} < {self.min_confidence:.1%}")
                return None

            return signal

        except Exception as e:
            self.logger.error(f"Intelligence filtering failed: {e}")
            return signal  # Return original signal if filtering fails

    def _enhance_signal_with_intelligence(self,
                                         signal: Dict,
                                         market_condition: MarketCondition,
                                         strategy_code: RAGStrategyCode) -> Dict:
        """Enhance signal with intelligence context and metadata"""
        try:
            # Add intelligence context
            signal['intelligence_context'] = {
                'market_regime': market_condition.regime,
                'regime_confidence': market_condition.confidence,
                'trading_session': market_condition.session,
                'volatility_level': market_condition.volatility,
                'success_factors': market_condition.success_factors,
                'rag_strategy_type': strategy_code.code_type,
                'rag_confidence': strategy_code.confidence_score
            }

            # Add strategy metadata
            signal['strategy'] = 'rag_intelligence'
            signal['strategy_version'] = '1.0'
            signal['analysis_timestamp'] = datetime.utcnow()

            # Calculate optimal position sizing based on intelligence
            signal['position_size_multiplier'] = self._calculate_position_size_multiplier(
                market_condition, signal['confidence']
            )

            # Set intelligent stop loss and take profit
            stop_take_levels = self._calculate_intelligent_levels(
                signal, market_condition, strategy_code
            )
            signal.update(stop_take_levels)

            return signal

        except Exception as e:
            self.logger.error(f"Signal enhancement failed: {e}")
            return signal

    # Helper Methods

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.intelligence_cache:
            return False
        timestamp_key = f"{cache_key}_timestamp"
        if timestamp_key not in self.intelligence_cache:
            return False
        cache_time = self.intelligence_cache[timestamp_key]
        return datetime.utcnow() - cache_time < self.cache_duration

    def _query_intelligence_data(self, epic: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Query market intelligence data from database"""
        # Simplified implementation - would query actual intelligence tables
        # For now, return empty DataFrame to trigger fallback
        return pd.DataFrame()

    def _determine_dominant_regime(self, data: pd.DataFrame) -> str:
        """Determine dominant market regime from intelligence data"""
        # Simplified regime detection
        regimes = ['trending_up', 'trending_down', 'ranging', 'breakout']
        return np.random.choice(regimes)  # Placeholder

    def _calculate_regime_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence in regime classification"""
        return np.random.uniform(0.6, 0.9)  # Placeholder

    def _analyze_current_session(self) -> Dict:
        """Analyze current trading session"""
        current_hour = datetime.utcnow().hour

        if 22 <= current_hour or current_hour < 8:
            return {'session': 'asian', 'volatility': 'low'}
        elif 8 <= current_hour < 13:
            return {'session': 'london', 'volatility': 'high'}
        elif 13 <= current_hour < 16:
            return {'session': 'overlap', 'volatility': 'peak'}
        else:
            return {'session': 'new_york', 'volatility': 'high'}

    def _extract_success_factors(self, data: pd.DataFrame, regime: str) -> List[str]:
        """Extract success factors from historical data"""
        return [
            'strong_momentum',
            'session_alignment',
            'volume_confirmation'
        ]  # Placeholder

    def _determine_optimal_timeframe(self, data: pd.DataFrame) -> str:
        """Determine optimal timeframe for current conditions"""
        return '15m'  # Placeholder

    def _get_fallback_market_condition(self) -> MarketCondition:
        """Get fallback market condition when analysis fails"""
        return MarketCondition(
            regime='ranging',
            confidence=0.5,
            session='unknown',
            volatility='medium',
            dominant_timeframe='15m',
            success_factors=['basic_analysis'],
            timestamp=datetime.utcnow()
        )

    def _get_fallback_strategy_code(self, market_condition: MarketCondition) -> RAGStrategyCode:
        """Get fallback strategy when RAG is unavailable"""
        return RAGStrategyCode(
            code_type='fallback',
            code_content='simple_ema_crossover',
            parameters={'fast_ema': 12, 'slow_ema': 26},
            market_suitability=market_condition.regime,
            confidence_score=0.6,
            source_id='fallback_001'
        )

    def _build_strategy_description(self, market_condition: MarketCondition) -> str:
        """Build strategy description for RAG query"""
        descriptions = {
            'trending_up': 'momentum trend-following strategy for bullish markets',
            'trending_down': 'momentum trend-following strategy for bearish markets',
            'ranging': 'mean reversion strategy for sideways markets',
            'breakout': 'volatility breakout strategy for explosive moves'
        }
        return descriptions.get(market_condition.regime, 'adaptive trading strategy')

    def _determine_trading_style(self, market_condition: MarketCondition) -> str:
        """Determine trading style based on market conditions"""
        if market_condition.session == 'asian':
            return 'range_trading'
        elif market_condition.volatility in ['high', 'peak']:
            return 'swing_trading'
        else:
            return 'day_trading'

    def _build_indicator_query(self, market_condition: MarketCondition) -> str:
        """Build indicator search query for RAG"""
        queries = {
            'trending_up': 'momentum oscillators trend-following indicators',
            'trending_down': 'bearish momentum indicators trend reversal',
            'ranging': 'mean reversion oscillators bollinger bands',
            'breakout': 'volatility indicators ATR breakout systems'
        }
        return queries.get(market_condition.regime, 'technical indicators')

    def _combine_rag_responses(self,
                              strategy_response: Dict,
                              indicators_response: Dict,
                              market_condition: MarketCondition) -> RAGStrategyCode:
        """Combine RAG responses into unified strategy code"""
        # Simplified combination - in reality would parse actual responses
        return RAGStrategyCode(
            code_type='composite',
            code_content='rag_enhanced_strategy',
            parameters={
                'fast_ema': 12,
                'slow_ema': 26,
                'bb_period': 20,
                'bb_std': 2.0,
                'atr_period': 14,
                'atr_multiplier': 2.0
            },
            market_suitability=market_condition.regime,
            confidence_score=0.8,
            source_id='rag_composite_001'
        )

    def _is_favorable_session(self, market_condition: MarketCondition) -> bool:
        """Check if current session is favorable for trading"""
        # Avoid trading during low-volatility periods
        return market_condition.volatility != 'minimal'

    def _check_regime_alignment(self, signal: Dict, market_condition: MarketCondition) -> bool:
        """Check if signal aligns with market regime"""
        regime = market_condition.regime
        direction = signal['direction']

        # Strong alignment for trending markets
        if regime == 'trending_up' and direction == 'BUY':
            return True
        if regime == 'trending_down' and direction == 'SELL':
            return True

        # Allow both directions for ranging and breakout
        if regime in ['ranging', 'breakout']:
            return True

        # Poor alignment
        return market_condition.confidence < 0.7  # Allow if low confidence in regime

    def _check_volatility_suitability(self, market_condition: MarketCondition, df: pd.DataFrame) -> bool:
        """Check if volatility is suitable for strategy"""
        # Simplified check - ensure some minimum volatility
        if len(df) < 20:
            return True

        recent_volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        return recent_volatility > 0.001  # Minimum volatility threshold

    def _calculate_intelligence_confidence_boost(self,
                                               signal: Dict,
                                               market_condition: MarketCondition) -> float:
        """Calculate confidence boost from intelligence factors"""
        boost = 0.0

        # Regime confidence boost
        if market_condition.confidence > 0.8:
            boost += 0.1

        # Session boost
        if market_condition.volatility in ['high', 'peak']:
            boost += 0.05

        # Success factors boost
        if len(market_condition.success_factors) > 2:
            boost += 0.05

        return min(0.2, boost)  # Cap at 20% boost

    def _calculate_position_size_multiplier(self,
                                          market_condition: MarketCondition,
                                          confidence: float) -> float:
        """Calculate position size multiplier based on intelligence"""
        base_multiplier = 1.0

        # Increase size in high-confidence, favorable conditions
        if confidence > 0.8 and market_condition.confidence > 0.8:
            base_multiplier *= 1.2

        # Reduce size in uncertain conditions
        if market_condition.volatility == 'peak':
            base_multiplier *= 0.8

        return min(1.5, max(0.5, base_multiplier))

    def _calculate_intelligent_levels(self,
                                    signal: Dict,
                                    market_condition: MarketCondition,
                                    strategy_code: RAGStrategyCode) -> Dict:
        """Calculate intelligent stop loss and take profit levels"""
        # Base levels (would be more sophisticated in practice)
        base_sl_pips = 20
        base_tp_pips = 40

        # Adjust based on volatility
        volatility_multiplier = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3,
            'peak': 1.5
        }.get(market_condition.volatility, 1.0)

        sl_pips = base_sl_pips * volatility_multiplier
        tp_pips = base_tp_pips * volatility_multiplier

        return {
            'stop_loss_pips': round(sl_pips, 1),
            'take_profit_pips': round(tp_pips, 1),
            'risk_reward_ratio': round(tp_pips / sl_pips, 2)
        }

    def _execute_pine_script(self,
                           df: pd.DataFrame,
                           strategy_code: RAGStrategyCode,
                           market_condition: MarketCondition) -> Optional[Dict]:
        """Execute TradingView Pine Script logic"""
        try:
            pine_code = strategy_code.pine_script
            parameters = strategy_code.parameters

            self.logger.info(f"📈 Executing Pine Script from: {strategy_code.source_title}")

            # Parse and execute Pine Script indicators/signals
            signal = self._parse_pine_script_signals(df, pine_code, parameters)

            if signal:
                self.logger.info(f"🎯 Pine Script generated {signal['direction']} signal "
                               f"(confidence: {signal['confidence']:.1%})")

            return signal

        except Exception as e:
            self.logger.error(f"Pine Script execution failed: {e}")
            return self._execute_fallback_logic(df, strategy_code, market_condition)

    def _parse_pine_script_signals(self,
                                 df: pd.DataFrame,
                                 pine_code: str,
                                 parameters: Dict) -> Optional[Dict]:
        """Parse Pine Script and extract trading signals"""
        try:
            latest_bar = df.iloc[-1]

            # Look for common Pine Script patterns and translate to signals
            if any(pattern in pine_code.lower() for pattern in ['strategy.entry', 'strategy.long', 'buy']):
                signal = self._evaluate_pine_entry_conditions(df, pine_code, parameters, 'BUY')
            elif any(pattern in pine_code.lower() for pattern in ['strategy.short', 'sell']):
                signal = self._evaluate_pine_entry_conditions(df, pine_code, parameters, 'SELL')
            elif any(pattern in pine_code.lower() for pattern in ['longcondition', 'shortcondition']):
                signal = self._evaluate_pine_conditions(df, pine_code, parameters)
            else:
                # Default evaluation based on common indicators
                signal = self._evaluate_default_pine_logic(df, pine_code, parameters)

            return signal

        except Exception as e:
            self.logger.error(f"Pine Script parsing failed: {e}")
            return None

    def _evaluate_pine_entry_conditions(self,
                                       df: pd.DataFrame,
                                       pine_code: str,
                                       parameters: Dict,
                                       direction: str) -> Optional[Dict]:
        """Evaluate Pine Script entry conditions"""
        try:
            latest_bar = df.iloc[-1]
            length = int(parameters.get('length', {}).get('default', '14'))

            # Calculate indicators commonly used in Pine Scripts
            df['rsi'] = self._calculate_rsi(df, length)
            df['ema'] = df['close'].ewm(span=length).mean()

            confidence = 0.7  # Base confidence

            # RSI-based signals
            if 'rsi' in pine_code.lower():
                if direction == 'BUY' and latest_bar['rsi'] < 30:
                    confidence += 0.1
                elif direction == 'SELL' and latest_bar['rsi'] > 70:
                    confidence += 0.1

            # EMA-based signals
            if 'ema' in pine_code.lower():
                if direction == 'BUY' and latest_bar['close'] > latest_bar['ema']:
                    confidence += 0.1
                elif direction == 'SELL' and latest_bar['close'] < latest_bar['ema']:
                    confidence += 0.1

            # Only generate signal if confidence is high enough
            if confidence >= 0.7:
                return {
                    'direction': direction,
                    'entry_price': float(latest_bar['close']),
                    'confidence': min(confidence, 1.0),
                    'stop_loss': self._calculate_pine_stop_loss(latest_bar, direction, parameters),
                    'take_profit': self._calculate_pine_take_profit(latest_bar, direction, parameters),
                    'source': 'pine_script'
                }

            return None

        except Exception as e:
            self.logger.error(f"Pine entry condition evaluation failed: {e}")
            return None

    def _evaluate_pine_conditions(self,
                                df: pd.DataFrame,
                                pine_code: str,
                                parameters: Dict) -> Optional[Dict]:
        """Evaluate general Pine Script conditions"""
        try:
            latest_bar = df.iloc[-1]
            prev_bar = df.iloc[-2] if len(df) > 1 else latest_bar
            length = int(parameters.get('length', {}).get('default', '14'))

            # Look for crossover patterns
            if 'crossover' in pine_code.lower() or 'cross(' in pine_code.lower():
                df['ema_fast'] = df['close'].ewm(span=length//2).mean()
                df['ema_slow'] = df['close'].ewm(span=length).mean()

                # Check for EMA crossover
                if (latest_bar['ema_fast'] > latest_bar['ema_slow'] and
                    prev_bar['ema_fast'] <= prev_bar['ema_slow']):
                    direction = 'BUY'
                elif (latest_bar['ema_fast'] < latest_bar['ema_slow'] and
                      prev_bar['ema_fast'] >= prev_bar['ema_slow']):
                    direction = 'SELL'
                else:
                    return None

                return {
                    'direction': direction,
                    'entry_price': float(latest_bar['close']),
                    'confidence': 0.75,
                    'stop_loss': self._calculate_pine_stop_loss(latest_bar, direction, parameters),
                    'take_profit': self._calculate_pine_take_profit(latest_bar, direction, parameters),
                    'source': 'pine_crossover'
                }

            return None

        except Exception as e:
            self.logger.error(f"Pine condition evaluation failed: {e}")
            return None

    def _evaluate_default_pine_logic(self,
                                   df: pd.DataFrame,
                                   pine_code: str,
                                   parameters: Dict) -> Optional[Dict]:
        """Default Pine Script evaluation when no specific patterns are found"""
        try:
            latest_bar = df.iloc[-1]
            length = int(parameters.get('length', {}).get('default', '14'))

            # Calculate basic indicators
            df['rsi'] = self._calculate_rsi(df, length)
            df['ema'] = df['close'].ewm(span=length).mean()

            # Simple momentum-based signal
            rsi_val = latest_bar['rsi']
            ema_val = latest_bar['ema']
            close_val = latest_bar['close']

            if rsi_val < 30 and close_val > ema_val:
                direction = 'BUY'
                confidence = 0.7
            elif rsi_val > 70 and close_val < ema_val:
                direction = 'SELL'
                confidence = 0.7
            else:
                return None

            return {
                'direction': direction,
                'entry_price': float(close_val),
                'confidence': confidence,
                'stop_loss': self._calculate_pine_stop_loss(latest_bar, direction, parameters),
                'take_profit': self._calculate_pine_take_profit(latest_bar, direction, parameters),
                'source': 'pine_default'
            }

        except Exception as e:
            self.logger.error(f"Default Pine logic failed: {e}")
            return None

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception:
            return pd.Series([50] * len(df), index=df.index)

    def _calculate_pine_stop_loss(self, bar: pd.Series, direction: str, parameters: Dict) -> float:
        """Calculate stop loss based on Pine Script parameters"""
        try:
            multiplier = float(parameters.get('multiplier', {}).get('default', '2.0'))
            base_price = float(bar['close'])

            if direction == 'BUY':
                return base_price * (1 - 0.02 * multiplier)
            else:
                return base_price * (1 + 0.02 * multiplier)
        except Exception:
            return float(bar['close']) * (0.98 if direction == 'BUY' else 1.02)

    def _calculate_pine_take_profit(self, bar: pd.Series, direction: str, parameters: Dict) -> float:
        """Calculate take profit based on Pine Script parameters"""
        try:
            multiplier = float(parameters.get('multiplier', {}).get('default', '2.0'))
            base_price = float(bar['close'])

            if direction == 'BUY':
                return base_price * (1 + 0.03 * multiplier)
            else:
                return base_price * (1 - 0.03 * multiplier)
        except Exception:
            return float(bar['close']) * (1.03 if direction == 'BUY' else 0.97)

    def _execute_fallback_logic(self,
                              df: pd.DataFrame,
                              strategy_code: RAGStrategyCode,
                              market_condition: MarketCondition) -> Optional[Dict]:
        """Fallback logic when Pine Script is not available"""
        if market_condition.regime == 'trending_up':
            return self._execute_trend_following_logic(df, 'BUY', strategy_code)
        elif market_condition.regime == 'trending_down':
            return self._execute_trend_following_logic(df, 'SELL', strategy_code)
        elif market_condition.regime == 'ranging':
            return self._execute_mean_reversion_logic(df, strategy_code)
        elif market_condition.regime == 'breakout':
            return self._execute_breakout_logic(df, strategy_code)
        else:
            return None

    def get_strategy_stats(self) -> Dict:
        """Get strategy performance statistics"""
        # Get RAG helper stats
        rag_stats = self.rag_helper.get_performance_stats() if self.rag_helper else {}

        return {
            **self.stats,
            'current_regime': self.current_market_condition.regime if self.current_market_condition else 'unknown',
            'rag_available': rag_stats.get('rag_available', False),
            'rag_stats': rag_stats,
            'cache_efficiency': self.stats['cache_hits'] / max(1, self.stats['intelligence_queries']),
            'strategy_name': self.name,
            'intelligence_analyzer_available': self.intelligence_analyzer is not None,
            'selected_strategy_code': self.selected_strategy_code.source_id if self.selected_strategy_code else None
        }

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators for the RAG Intelligence Strategy.

        Since this strategy primarily uses market intelligence data and RAG-selected code,
        it needs minimal traditional indicators. The actual indicators used are determined
        dynamically by the selected RAG strategy code.

        Returns:
            List of basic indicators needed for market analysis
        """
        return [
            'close',  # Essential for all analysis
            'high',   # For volatility and range analysis
            'low',    # For volatility and range analysis
            'volume', # For market intelligence analysis
            'ema_12', # Fast EMA for trend analysis
            'ema_26'  # Slow EMA for trend analysis
        ]