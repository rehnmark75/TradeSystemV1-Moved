#!/usr/bin/env python3
"""
Performance-Aware RAG System
============================

This module integrates live optimization data with the RAG system to provide:
- Performance-weighted recommendations
- Market regime-aware suggestions
- Real-time strategy effectiveness analysis
- Historical performance context
- Risk-adjusted indicator rankings
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for indicators/strategies"""
    win_rate: float
    profit_factor: float
    net_pips: float
    max_drawdown: float
    avg_trade_duration: float
    total_trades: int
    risk_adjusted_return: float
    consistency_score: float

@dataclass
class MarketRegime:
    """Current market regime information"""
    regime_type: str  # trending, ranging, volatile, transitional
    volatility_level: str  # low, medium, high
    trend_strength: float  # 0-1
    market_phase: str  # accumulation, markup, distribution, markdown
    session_activity: str  # asia, london, newyork, overlap
    confidence: float

@dataclass
class PerformanceWeightedRecommendation:
    """Recommendation with performance context"""
    indicator_id: str
    title: str
    base_score: float
    performance_score: float
    regime_suitability: float
    final_score: float
    recommendation_reason: str
    performance_context: Dict[str, Any]
    risk_assessment: str
    optimal_parameters: Dict[str, Any] = None

class MarketRegimeDetector:
    """Detects current market regime from recent data"""

    def __init__(self, db_url: str):
        self.db_url = db_url

    def detect_current_regime(self, epic: str = "CS.D.EURUSD.MINI.IP",
                            lookback_hours: int = 168) -> MarketRegime:
        """Detect current market regime for given epic"""
        try:
            connection = psycopg2.connect(self.db_url)
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            # Get recent price data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)

            cursor.execute("""
                SELECT
                    high, low, close, volume,
                    created_at
                FROM ohlc_data
                WHERE epic = %s
                AND created_at >= %s
                AND created_at <= %s
                ORDER BY created_at ASC
            """, (epic, start_time, end_time))

            data = cursor.fetchall()
            connection.close()

            if len(data) < 50:  # Need sufficient data
                return self._default_regime()

            # Convert to DataFrame for analysis
            df = pd.DataFrame([dict(row) for row in data])

            # Calculate regime indicators
            regime_type = self._detect_trend_regime(df)
            volatility_level = self._detect_volatility_level(df)
            trend_strength = self._calculate_trend_strength(df)
            market_phase = self._detect_market_phase(df)
            session_activity = self._detect_session_activity()

            return MarketRegime(
                regime_type=regime_type,
                volatility_level=volatility_level,
                trend_strength=trend_strength,
                market_phase=market_phase,
                session_activity=session_activity,
                confidence=0.8  # Could be improved with more sophisticated methods
            )

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return self._default_regime()

    def _detect_trend_regime(self, df: pd.DataFrame) -> str:
        """Detect if market is trending or ranging"""
        if len(df) < 20:
            return "ranging"

        # Calculate moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else df['close'].rolling(len(df)//2).mean()

        # Calculate ADX-like trend strength
        price_changes = df['close'].pct_change().abs()
        avg_change = price_changes.rolling(14).mean().iloc[-1]

        # Check trend consistency
        recent_closes = df['close'].tail(20).values
        trend_consistency = self._calculate_trend_consistency(recent_closes)

        if trend_consistency > 0.7 and avg_change > 0.01:
            return "trending"
        elif avg_change > 0.015:
            return "volatile"
        else:
            return "ranging"

    def _detect_volatility_level(self, df: pd.DataFrame) -> str:
        """Detect volatility level"""
        if len(df) < 14:
            return "medium"

        # Calculate True Range and ATR
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['prev_close']).abs()
        df['tr3'] = (df['low'] - df['prev_close']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()

        # Get recent ATR and compare to historical
        recent_atr = df['atr'].tail(5).mean()
        historical_atr = df['atr'].mean()

        if recent_atr > historical_atr * 1.5:
            return "high"
        elif recent_atr < historical_atr * 0.7:
            return "low"
        else:
            return "medium"

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        if len(df) < 20:
            return 0.5

        # Linear regression slope
        x = np.arange(len(df))
        y = df['close'].values
        slope = np.polyfit(x, y, 1)[0]

        # Normalize slope
        price_range = df['close'].max() - df['close'].min()
        normalized_slope = abs(slope) / (price_range / len(df))

        return min(normalized_slope * 10, 1.0)  # Scale and cap at 1.0

    def _detect_market_phase(self, df: pd.DataFrame) -> str:
        """Detect market phase using volume and price action"""
        if len(df) < 20:
            return "transitional"

        # Simple heuristic based on recent price and volume
        recent_data = df.tail(10)
        price_trend = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        volume_trend = recent_data['volume'].mean() if 'volume' in df.columns else 1000

        if price_trend > 0 and volume_trend > df['volume'].mean():
            return "markup"
        elif price_trend < 0 and volume_trend > df['volume'].mean():
            return "markdown"
        elif volume_trend < df['volume'].mean():
            return "accumulation"
        else:
            return "distribution"

    def _detect_session_activity(self) -> str:
        """Detect current trading session"""
        current_hour = datetime.now().hour

        if 0 <= current_hour < 7:
            return "asia"
        elif 7 <= current_hour < 16:
            return "london"
        elif 16 <= current_hour < 21:
            return "newyork"
        else:
            return "overlap"

    def _calculate_trend_consistency(self, prices: np.ndarray) -> float:
        """Calculate how consistent the trend is"""
        if len(prices) < 5:
            return 0.5

        # Calculate directional consistency
        changes = np.diff(prices)
        positive_changes = np.sum(changes > 0)
        negative_changes = np.sum(changes < 0)
        total_changes = len(changes)

        # Consistency is the proportion of changes in the dominant direction
        dominant_direction = max(positive_changes, negative_changes)
        return dominant_direction / total_changes if total_changes > 0 else 0.5

    def _default_regime(self) -> MarketRegime:
        """Return default regime when detection fails"""
        return MarketRegime(
            regime_type="ranging",
            volatility_level="medium",
            trend_strength=0.5,
            market_phase="transitional",
            session_activity="london",
            confidence=0.3
        )

class PerformanceAnalyzer:
    """Analyzes historical performance of indicators and strategies"""

    def __init__(self, db_url: str):
        self.db_url = db_url

    def get_indicator_performance(self, strategy_type: str,
                                epic: Optional[str] = None,
                                timeframe: Optional[str] = None) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics for indicators"""
        try:
            connection = psycopg2.connect(self.db_url)
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            performance_data = {}

            if strategy_type.upper() == 'EMA':
                performance_data.update(self._get_ema_performance(cursor, epic, timeframe))
            elif strategy_type.upper() == 'MACD':
                performance_data.update(self._get_macd_performance(cursor, epic, timeframe))
            elif strategy_type.upper() == 'SMC':
                performance_data.update(self._get_smc_performance(cursor, epic, timeframe))

            connection.close()
            return performance_data

        except Exception as e:
            logger.error(f"Error getting indicator performance: {e}")
            return {}

    def _get_ema_performance(self, cursor, epic: Optional[str], timeframe: Optional[str]) -> Dict[str, PerformanceMetrics]:
        """Get EMA strategy performance"""
        query = """
            SELECT
                epic,
                best_timeframe,
                best_win_rate,
                best_profit_factor,
                best_net_pips,
                best_drawdown,
                best_avg_trade_duration,
                best_total_trades,
                best_ema_config
            FROM ema_best_parameters
            WHERE best_net_pips > 0
        """

        params = []
        if epic:
            query += " AND epic = %s"
            params.append(epic)
        if timeframe:
            query += " AND best_timeframe = %s"
            params.append(timeframe)

        query += " ORDER BY best_net_pips DESC LIMIT 20"

        cursor.execute(query, params)
        results = cursor.fetchall()

        performance_data = {}
        for row in results:
            key = f"ema_{row['epic']}_{row['best_timeframe']}"

            # Calculate risk-adjusted return and consistency
            risk_adjusted_return = self._calculate_risk_adjusted_return(
                row['best_net_pips'], row['best_drawdown']
            )
            consistency_score = self._calculate_consistency_score(
                row['best_win_rate'], row['best_profit_factor']
            )

            performance_data[key] = PerformanceMetrics(
                win_rate=row['best_win_rate'],
                profit_factor=row['best_profit_factor'],
                net_pips=row['best_net_pips'],
                max_drawdown=row['best_drawdown'],
                avg_trade_duration=row['best_avg_trade_duration'],
                total_trades=row['best_total_trades'],
                risk_adjusted_return=risk_adjusted_return,
                consistency_score=consistency_score
            )

        return performance_data

    def _get_macd_performance(self, cursor, epic: Optional[str], timeframe: Optional[str]) -> Dict[str, PerformanceMetrics]:
        """Get MACD strategy performance"""
        query = """
            SELECT
                epic,
                best_timeframe,
                best_win_rate,
                best_composite_score,
                optimal_stop_loss_pips,
                optimal_take_profit_pips,
                best_fast_ema,
                best_slow_ema,
                best_signal_ema
            FROM macd_best_parameters
            WHERE best_composite_score > 1.0
        """

        params = []
        if epic:
            query += " AND epic = %s"
            params.append(epic)
        if timeframe:
            query += " AND best_timeframe = %s"
            params.append(timeframe)

        query += " ORDER BY best_composite_score DESC LIMIT 20"

        cursor.execute(query, params)
        results = cursor.fetchall()

        performance_data = {}
        for row in results:
            key = f"macd_{row['epic']}_{row['best_timeframe']}"

            # Estimate metrics from available data
            profit_factor = row['best_composite_score']
            net_pips = (row['optimal_take_profit_pips'] - row['optimal_stop_loss_pips']) * row['best_win_rate']

            risk_adjusted_return = self._calculate_risk_adjusted_return(net_pips, row['optimal_stop_loss_pips'])
            consistency_score = self._calculate_consistency_score(row['best_win_rate'], profit_factor)

            performance_data[key] = PerformanceMetrics(
                win_rate=row['best_win_rate'],
                profit_factor=profit_factor,
                net_pips=net_pips,
                max_drawdown=row['optimal_stop_loss_pips'],  # Approximation
                avg_trade_duration=24.0,  # Default assumption
                total_trades=100,  # Default assumption
                risk_adjusted_return=risk_adjusted_return,
                consistency_score=consistency_score
            )

        return performance_data

    def _get_smc_performance(self, cursor, epic: Optional[str], timeframe: Optional[str]) -> Dict[str, PerformanceMetrics]:
        """Get Smart Money Concepts performance"""
        # Check if SMC tables exist
        try:
            query = """
                SELECT
                    epic,
                    best_timeframe,
                    best_win_rate,
                    best_composite_score,
                    optimal_stop_loss_pips,
                    optimal_take_profit_pips
                FROM smc_best_parameters
                WHERE best_composite_score > 1.0
            """

            params = []
            if epic:
                query += " AND epic = %s"
                params.append(epic)
            if timeframe:
                query += " AND best_timeframe = %s"
                params.append(timeframe)

            query += " ORDER BY best_composite_score DESC LIMIT 10"

            cursor.execute(query, params)
            results = cursor.fetchall()

            performance_data = {}
            for row in results:
                key = f"smc_{row['epic']}_{row['best_timeframe']}"

                profit_factor = row['best_composite_score']
                net_pips = (row['optimal_take_profit_pips'] - row['optimal_stop_loss_pips']) * row['best_win_rate']

                risk_adjusted_return = self._calculate_risk_adjusted_return(net_pips, row['optimal_stop_loss_pips'])
                consistency_score = self._calculate_consistency_score(row['best_win_rate'], profit_factor)

                performance_data[key] = PerformanceMetrics(
                    win_rate=row['best_win_rate'],
                    profit_factor=profit_factor,
                    net_pips=net_pips,
                    max_drawdown=row['optimal_stop_loss_pips'],
                    avg_trade_duration=12.0,  # SMC typically shorter duration
                    total_trades=150,
                    risk_adjusted_return=risk_adjusted_return,
                    consistency_score=consistency_score
                )

            return performance_data

        except Exception:
            return {}  # SMC tables may not exist

    def _calculate_risk_adjusted_return(self, net_pips: float, max_drawdown: float) -> float:
        """Calculate risk-adjusted return (simplified Sharpe-like ratio)"""
        if max_drawdown == 0:
            return net_pips
        return net_pips / max_drawdown

    def _calculate_consistency_score(self, win_rate: float, profit_factor: float) -> float:
        """Calculate consistency score combining win rate and profit factor"""
        # Penalize extreme win rates (too high or too low might indicate overfitting)
        win_rate_score = 1.0 - abs(win_rate - 0.6) * 2  # Optimal around 60%
        profit_factor_score = min(profit_factor / 2.0, 1.0)  # Cap at 2.0 PF

        return (win_rate_score * 0.6 + profit_factor_score * 0.4)

class PerformanceAwareRAG:
    """Main class integrating performance data with RAG recommendations"""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.regime_detector = MarketRegimeDetector(db_url)
        self.performance_analyzer = PerformanceAnalyzer(db_url)

        # Regime-specific indicator preferences
        self.regime_preferences = {
            'trending': {
                'preferred_types': ['trend', 'momentum'],
                'preferred_indicators': ['moving_average', 'macd', 'adx', 'supertrend'],
                'weight_multiplier': 1.2
            },
            'ranging': {
                'preferred_types': ['oscillator', 'mean_reversion'],
                'preferred_indicators': ['rsi', 'stochastic', 'bollinger_bands'],
                'weight_multiplier': 1.3
            },
            'volatile': {
                'preferred_types': ['volatility', 'breakout'],
                'preferred_indicators': ['bollinger_bands', 'atr', 'keltner_channels'],
                'weight_multiplier': 1.1
            }
        }

    def get_performance_weighted_recommendations(self,
                                               query: str,
                                               base_results: List[Dict[str, Any]],
                                               epic: str = "CS.D.EURUSD.MINI.IP",
                                               user_context: Optional[Dict[str, Any]] = None) -> List[PerformanceWeightedRecommendation]:
        """Get recommendations weighted by performance and market regime"""

        # Detect current market regime
        current_regime = self.regime_detector.detect_current_regime(epic)
        logger.info(f"Detected market regime: {current_regime.regime_type} (confidence: {current_regime.confidence:.2f})")

        # Get performance data for relevant strategies
        performance_data = self._gather_performance_data(base_results)

        # Generate weighted recommendations
        recommendations = []
        for result in base_results:
            recommendation = self._create_weighted_recommendation(
                result, current_regime, performance_data, user_context
            )
            recommendations.append(recommendation)

        # Sort by final score
        recommendations.sort(key=lambda x: x.final_score, reverse=True)

        return recommendations

    def _gather_performance_data(self, base_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, PerformanceMetrics]]:
        """Gather performance data for all relevant indicators"""
        performance_data = {}

        # Identify strategy types from results
        strategy_types = set()
        for result in base_results:
            metadata = result.get('metadata', {})

            # Try to identify strategy type from various fields
            if 'strategy_type' in metadata:
                strategy_types.add(metadata['strategy_type'].upper())

            # Infer from indicators used
            indicators = metadata.get('indicators', '').lower()
            if 'ema' in indicators or 'sma' in indicators:
                strategy_types.add('EMA')
            if 'macd' in indicators:
                strategy_types.add('MACD')
            if 'order_block' in indicators or 'liquidity' in indicators:
                strategy_types.add('SMC')

        # Get performance data for each strategy type
        for strategy_type in strategy_types:
            performance_data[strategy_type] = self.performance_analyzer.get_indicator_performance(strategy_type)

        return performance_data

    def _create_weighted_recommendation(self,
                                      result: Dict[str, Any],
                                      current_regime: MarketRegime,
                                      performance_data: Dict[str, Dict[str, PerformanceMetrics]],
                                      user_context: Optional[Dict[str, Any]]) -> PerformanceWeightedRecommendation:
        """Create a performance-weighted recommendation"""

        # Base score from similarity
        base_score = result.get('similarity_score', 0.5)

        # Calculate performance score
        performance_score = self._calculate_performance_score(result, performance_data)

        # Calculate regime suitability
        regime_suitability = self._calculate_regime_suitability(result, current_regime)

        # Apply user context weights
        user_weight = self._calculate_user_context_weight(result, user_context)

        # Calculate final weighted score
        final_score = (
            base_score * 0.3 +
            performance_score * 0.4 +
            regime_suitability * 0.2 +
            user_weight * 0.1
        )

        # Generate recommendation reason
        reason = self._generate_recommendation_reason(result, current_regime, performance_score)

        # Assess risk
        risk_assessment = self._assess_risk(result, performance_data)

        # Get optimal parameters
        optimal_parameters = self._get_optimal_parameters(result, performance_data)

        return PerformanceWeightedRecommendation(
            indicator_id=result.get('id', ''),
            title=result.get('metadata', {}).get('title', 'Unknown'),
            base_score=base_score,
            performance_score=performance_score,
            regime_suitability=regime_suitability,
            final_score=final_score,
            recommendation_reason=reason,
            performance_context=self._create_performance_context(result, performance_data),
            risk_assessment=risk_assessment,
            optimal_parameters=optimal_parameters
        )

    def _calculate_performance_score(self, result: Dict[str, Any],
                                   performance_data: Dict[str, Dict[str, PerformanceMetrics]]) -> float:
        """Calculate performance score for an indicator"""
        metadata = result.get('metadata', {})

        # Try to find matching performance data
        best_performance = None

        for strategy_type, performances in performance_data.items():
            for perf_key, perf_metrics in performances.items():
                if strategy_type.lower() in metadata.get('indicators', '').lower():
                    if best_performance is None or perf_metrics.risk_adjusted_return > best_performance.risk_adjusted_return:
                        best_performance = perf_metrics

        if best_performance:
            # Normalize performance metrics to 0-1 scale
            win_rate_score = best_performance.win_rate
            profit_factor_score = min(best_performance.profit_factor / 3.0, 1.0)
            consistency_score = best_performance.consistency_score

            return (win_rate_score * 0.4 + profit_factor_score * 0.3 + consistency_score * 0.3)
        else:
            return 0.5  # Default when no performance data

    def _calculate_regime_suitability(self, result: Dict[str, Any], current_regime: MarketRegime) -> float:
        """Calculate how suitable an indicator is for current market regime"""
        metadata = result.get('metadata', {})

        # Get regime preferences
        regime_prefs = self.regime_preferences.get(current_regime.regime_type, {})
        preferred_indicators = regime_prefs.get('preferred_indicators', [])

        # Check if indicator matches regime preferences
        suitability_score = 0.5  # Base score

        # Check indicator types
        indicators = metadata.get('indicators', '').lower()
        category = metadata.get('category', '').lower()

        for preferred in preferred_indicators:
            if preferred in indicators or preferred in category:
                suitability_score += 0.2
                break

        # Bonus for regime-specific market conditions
        market_context = metadata.get('market_context', '')
        if current_regime.regime_type in market_context:
            suitability_score += 0.15

        # Apply confidence weight
        suitability_score *= current_regime.confidence

        return min(suitability_score, 1.0)

    def _calculate_user_context_weight(self, result: Dict[str, Any],
                                     user_context: Optional[Dict[str, Any]]) -> float:
        """Calculate weight based on user context and preferences"""
        if not user_context:
            return 0.5

        weight = 0.5
        metadata = result.get('metadata', {})

        # User experience level
        user_level = user_context.get('experience_level', 'intermediate')
        indicator_complexity = metadata.get('complexity_level', 'intermediate')

        if user_level == indicator_complexity:
            weight += 0.2
        elif abs(['basic', 'intermediate', 'advanced', 'expert'].index(user_level) -
                ['basic', 'intermediate', 'advanced', 'expert'].index(indicator_complexity)) == 1:
            weight += 0.1

        # Preferred timeframes
        user_timeframes = user_context.get('preferred_timeframes', [])
        indicator_timeframes = metadata.get('best_for_timeframes', [])

        if any(tf in indicator_timeframes for tf in user_timeframes):
            weight += 0.15

        # Trading style match
        user_style = user_context.get('trading_style', '')
        if user_style in metadata.get('trading_styles', []):
            weight += 0.15

        return min(weight, 1.0)

    def _generate_recommendation_reason(self, result: Dict[str, Any],
                                      current_regime: MarketRegime,
                                      performance_score: float) -> str:
        """Generate human-readable recommendation reason"""
        metadata = result.get('metadata', {})
        title = metadata.get('title', 'This indicator')

        reasons = []

        # Performance reason
        if performance_score > 0.7:
            reasons.append("has strong historical performance")
        elif performance_score > 0.5:
            reasons.append("shows consistent performance")

        # Regime reason
        regime_prefs = self.regime_preferences.get(current_regime.regime_type, {})
        if any(ind in metadata.get('indicators', '').lower() for ind in regime_prefs.get('preferred_indicators', [])):
            reasons.append(f"is well-suited for {current_regime.regime_type} markets")

        # Complexity reason
        complexity = metadata.get('complexity_level', 'intermediate')
        if complexity == 'basic':
            reasons.append("is easy to understand and implement")
        elif complexity == 'advanced':
            reasons.append("provides sophisticated analysis capabilities")

        if not reasons:
            reasons.append("matches your search criteria")

        return f"{title} is recommended because it " + " and ".join(reasons) + "."

    def _assess_risk(self, result: Dict[str, Any],
                   performance_data: Dict[str, Dict[str, PerformanceMetrics]]) -> str:
        """Assess risk level of using this indicator"""
        metadata = result.get('metadata', {})

        # Check if we have performance data
        has_performance = any(
            strategy_type.lower() in metadata.get('indicators', '').lower()
            for strategy_type in performance_data.keys()
        )

        if has_performance:
            # Find relevant performance metrics
            for strategy_type, performances in performance_data.items():
                for perf_metrics in performances.values():
                    if perf_metrics.max_drawdown > 100:  # High drawdown
                        return "High Risk - Significant drawdown potential"
                    elif perf_metrics.win_rate < 0.4:  # Low win rate
                        return "Medium Risk - Lower win rate, requires good risk management"
                    elif perf_metrics.consistency_score > 0.7:
                        return "Low Risk - Consistent performance with controlled drawdown"

        # Default risk assessment based on complexity
        complexity = metadata.get('complexity_level', 'intermediate')
        if complexity == 'expert':
            return "Medium Risk - Advanced indicator requiring experience"
        elif complexity == 'basic':
            return "Low Risk - Simple and straightforward to use"
        else:
            return "Medium Risk - Standard risk profile"

    def _get_optimal_parameters(self, result: Dict[str, Any],
                              performance_data: Dict[str, Dict[str, PerformanceMetrics]]) -> Dict[str, Any]:
        """Get optimal parameters based on performance data"""
        # This would extract optimal parameters from the performance data
        # For now, return basic recommendations
        return {
            "timeframe": "1h",
            "stop_loss": "30 pips",
            "take_profit": "60 pips",
            "risk_per_trade": "1%"
        }

    def _create_performance_context(self, result: Dict[str, Any],
                                  performance_data: Dict[str, Dict[str, PerformanceMetrics]]) -> Dict[str, Any]:
        """Create performance context for the recommendation"""
        context = {
            "has_backtest_data": False,
            "performance_summary": "No historical data available"
        }

        # Find relevant performance data
        metadata = result.get('metadata', {})
        for strategy_type, performances in performance_data.items():
            if strategy_type.lower() in metadata.get('indicators', '').lower():
                if performances:
                    # Get best performance
                    best_perf = max(performances.values(), key=lambda x: x.risk_adjusted_return)
                    context = {
                        "has_backtest_data": True,
                        "win_rate": f"{best_perf.win_rate:.1%}",
                        "profit_factor": f"{best_perf.profit_factor:.2f}",
                        "net_pips": f"{best_perf.net_pips:.0f} pips",
                        "risk_adjusted_return": f"{best_perf.risk_adjusted_return:.2f}",
                        "performance_summary": f"Historical win rate of {best_perf.win_rate:.1%} with {best_perf.profit_factor:.2f} profit factor"
                    }
                break

        return context

# Test function
def test_performance_aware_rag():
    """Test the performance-aware RAG system"""

    # Mock database URL (would be real in production)
    db_url = "postgresql://postgres:password@localhost:5432/forex"

    try:
        rag = PerformanceAwareRAG(db_url)

        # Mock base results from semantic search
        base_results = [
            {
                'id': 'rsi_enhanced',
                'similarity_score': 0.85,
                'metadata': {
                    'title': 'Enhanced RSI with Divergence',
                    'indicators': 'rsi oscillator',
                    'category': 'oscillator',
                    'complexity_level': 'intermediate',
                    'best_for_timeframes': ['1h', '4h'],
                    'trading_styles': ['day_trading']
                }
            },
            {
                'id': 'ema_crossover',
                'similarity_score': 0.78,
                'metadata': {
                    'title': 'EMA Crossover Strategy',
                    'indicators': 'ema moving_average',
                    'category': 'trend',
                    'complexity_level': 'basic',
                    'best_for_timeframes': ['1h', '4h'],
                    'trading_styles': ['swing_trading']
                }
            }
        ]

        user_context = {
            'experience_level': 'intermediate',
            'preferred_timeframes': ['1h', '4h'],
            'trading_style': 'day_trading'
        }

        # Get performance-weighted recommendations
        recommendations = rag.get_performance_weighted_recommendations(
            query="best indicators for current market",
            base_results=base_results,
            user_context=user_context
        )

        print("=== Performance-Weighted Recommendations ===")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"\n{i}. {rec.title}")
            print(f"   Final Score: {rec.final_score:.3f}")
            print(f"   Performance Score: {rec.performance_score:.3f}")
            print(f"   Regime Suitability: {rec.regime_suitability:.3f}")
            print(f"   Risk Assessment: {rec.risk_assessment}")
            print(f"   Reason: {rec.recommendation_reason}")
            if rec.performance_context.get('has_backtest_data'):
                print(f"   Performance: {rec.performance_context['performance_summary']}")

    except Exception as e:
        print(f"Test failed (expected with mock DB): {e}")
        print("This is normal when testing without a real database connection.")

if __name__ == "__main__":
    test_performance_aware_rag()