# core/strategies/helpers/market_intelligence_analyzer.py
"""
Market Intelligence Data Analyzer for RAG Intelligence Strategy
============================================================

This module provides comprehensive market intelligence analysis by querying
and analyzing historical market data from the database. It identifies market regimes,
trading sessions, volatility patterns, and success factors to inform trading decisions.

Key Features:
- Market regime detection using multiple timeframes
- Trading session analysis with volatility mapping
- Success factor extraction from historical patterns
- Performance-based parameter optimization
- Real-time market condition monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from ..detection.price_adjuster import PriceAdjuster
    from ...database import DatabaseManager
except ImportError:
    try:
        from forex_scanner.core.detection.price_adjuster import PriceAdjuster
        from forex_scanner.core.database import DatabaseManager
    except ImportError:
        # Create minimal fallback classes
        class PriceAdjuster:
            def __init__(self): pass
            def adjust_prices(self, *args, **kwargs): return args[0] if args else None

        class DatabaseManager:
            def __init__(self, url): pass
            def get_connection(self):
                raise ImportError("Database not available")


@dataclass
class RegimeAnalysis:
    """Market regime analysis results"""
    regime: str
    confidence: float
    strength: float
    duration_hours: int
    trend_direction: str
    volatility_level: str
    support_resistance: Dict[str, float]


@dataclass
class SessionAnalysis:
    """Trading session analysis results"""
    current_session: str
    session_strength: float
    optimal_pairs: List[str]
    volatility_forecast: str
    liquidity_level: str
    recommended_timeframes: List[str]


@dataclass
class SuccessPattern:
    """Success pattern identified from historical data"""
    pattern_type: str
    success_rate: float
    avg_profit: float
    conditions: Dict[str, Any]
    frequency: int
    timeframe: str


class MarketIntelligenceAnalyzer:
    """
    Comprehensive market intelligence analyzer that provides
    real-time market condition assessment and historical pattern analysis.
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 logger: Optional[logging.Logger] = None):
        self.db_manager = db_manager
        self.logger = logger or logging.getLogger(__name__)
        self.price_adjuster = PriceAdjuster()

        # Analysis cache
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)

        # Analysis windows
        self.short_window = 24  # hours for regime analysis
        self.medium_window = 72  # hours for trend analysis
        self.long_window = 168  # hours for pattern analysis

        # Technical indicator periods
        self.ema_periods = [5, 13, 21, 50, 200]
        self.volatility_period = 20
        self.atr_period = 14

        self.logger.info("Market Intelligence Analyzer initialized")

    def _fetch_intelligence_history(self, hours: int = 24) -> pd.DataFrame:
        """
        Fetch historical market intelligence data for enhanced analysis

        Args:
            hours: Hours of intelligence history to fetch

        Returns:
            DataFrame with historical intelligence data
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Query market intelligence history
            query = """
                SELECT scan_timestamp, scan_cycle_id, epic_list, epic_count,
                       dominant_regime, regime_confidence, regime_scores,
                       current_session, session_volatility, market_bias,
                       intelligence_source, created_at
                FROM market_intelligence_history
                WHERE scan_timestamp >= %s
                    AND scan_timestamp <= %s
                ORDER BY scan_timestamp DESC
            """

            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[start_time, end_time]
                )

            if not df.empty:
                # Parse JSON regime_scores if present
                if 'regime_scores' in df.columns:
                    df['regime_scores'] = df['regime_scores'].apply(
                        lambda x: json.loads(x) if isinstance(x, str) else x
                    )

                self.logger.info(f"✅ Fetched {len(df)} intelligence history records")
            else:
                self.logger.warning("📊 No intelligence history data available")

            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch intelligence history: {e}")
            return pd.DataFrame()

    def analyze_market_conditions(self,
                                epic: str,
                                analysis_hours: int = 24) -> Dict[str, Any]:
        """
        Comprehensive market condition analysis

        Args:
            epic: Trading instrument
            analysis_hours: Hours of historical data to analyze

        Returns:
            Complete market intelligence analysis
        """
        try:
            cache_key = f"market_conditions_{epic}_{analysis_hours}"

            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]

            self.logger.info(f"📊 Analyzing market conditions for {epic} ({analysis_hours}h)")

            # Get market data
            market_data = self._fetch_market_data(epic, analysis_hours)

            # Get intelligence history for enhanced analysis
            intelligence_history = self._fetch_intelligence_history(analysis_hours)

            if market_data.empty:
                self.logger.warning(f"No market data available for {epic}")
                return self._get_fallback_analysis()

            # Perform comprehensive analysis
            analysis = {
                'epic': epic,
                'analysis_timestamp': datetime.utcnow(),
                'data_period_hours': analysis_hours,
                'total_bars': len(market_data),

                # Core analyses
                'regime_analysis': self._analyze_market_regime(market_data, intelligence_history),
                'session_analysis': self._analyze_trading_session(market_data, intelligence_history),
                'volatility_analysis': self._analyze_volatility_patterns(market_data),
                'trend_analysis': self._analyze_trend_strength(market_data),
                'support_resistance': self._identify_key_levels(market_data),

                # Intelligence history insights
                'intelligence_insights': self._analyze_intelligence_history(intelligence_history),

                # Pattern recognition
                'success_patterns': self._identify_success_patterns(epic, market_data),
                'failure_patterns': self._identify_failure_patterns(epic, market_data),

                # Performance metrics
                'recent_performance': self._analyze_recent_performance(epic, market_data),
                'optimal_parameters': self._suggest_optimal_parameters(epic, market_data),

                # Forward-looking
                'market_forecast': self._generate_market_forecast(market_data),
                'recommended_approach': self._recommend_trading_approach(market_data)
            }

            # Cache results
            self.cache[cache_key] = analysis
            self.cache[f"{cache_key}_timestamp"] = datetime.utcnow()

            self.logger.info(f"✅ Market analysis completed: {analysis['regime_analysis']['regime']} "
                           f"regime with {analysis['regime_analysis']['confidence']:.1%} confidence")

            return analysis

        except Exception as e:
            self.logger.error(f"Market condition analysis failed: {e}")
            return self._get_fallback_analysis()

    def _fetch_market_data(self, epic: str, hours: int) -> pd.DataFrame:
        """
        Fetch market data from database with technical indicators

        Args:
            epic: Trading instrument
            hours: Hours of historical data

        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        try:
            # Calculate timeframe (prefer 15m for intelligence analysis)
            timeframe_minutes = 15
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Query database
            query = """
                SELECT start_time, epic, timeframe,
                       open, high, low, close, ltv as volume
                FROM ig_candles
                WHERE epic = %s
                    AND timeframe = %s
                    AND start_time >= %s
                    AND start_time <= %s
                ORDER BY start_time ASC
            """

            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[epic, timeframe_minutes, start_time, end_time]
                )

            if df.empty:
                self.logger.warning(f"No data found for {epic} in last {hours}h")
                return df

            # Add technical indicators
            df = self._add_technical_indicators(df)

            self.logger.info(f"Fetched {len(df)} bars for {epic} ({hours}h)")
            return df

        except Exception as e:
            self.logger.error(f"Data fetch failed for {epic}: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to market data"""
        try:
            if len(df) < 20:
                return df

            # EMAs for trend analysis
            for period in self.ema_periods:
                if len(df) >= period:
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

            # ATR for volatility
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_close'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=self.atr_period).mean()

            # Volatility measures
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(self.volatility_period).std()
            df['volatility_percentile'] = df['volatility'].rolling(100).rank(pct=True)

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            return df

        except Exception as e:
            self.logger.error(f"Technical indicator calculation failed: {e}")
            return df

    def _analyze_market_regime(self, df: pd.DataFrame, intelligence_history: pd.DataFrame = None) -> RegimeAnalysis:
        """
        Analyze current market regime using multiple criteria

        Args:
            df: Market data with technical indicators

        Returns:
            RegimeAnalysis object with regime classification
        """
        try:
            if len(df) < 50:
                return RegimeAnalysis(
                    regime='ranging', confidence=0.5, strength=0.5,
                    duration_hours=1, trend_direction='neutral',
                    volatility_level='medium', support_resistance={}
                )

            latest = df.iloc[-1]
            recent = df.tail(24)  # Last 24 bars for regime analysis

            # EMA alignment analysis
            ema_alignment = self._calculate_ema_alignment(latest)
            trend_strength = self._calculate_trend_strength(recent)
            volatility_regime = self._classify_volatility_regime(recent)

            # Regime classification logic
            regime, confidence = self._classify_regime(
                ema_alignment, trend_strength, volatility_regime, recent
            )

            # Calculate additional metrics
            regime_duration = self._estimate_regime_duration(df)
            trend_direction = self._determine_trend_direction(recent)
            support_resistance = self._calculate_key_levels(df.tail(48))

            return RegimeAnalysis(
                regime=regime,
                confidence=confidence,
                strength=trend_strength,
                duration_hours=regime_duration,
                trend_direction=trend_direction,
                volatility_level=volatility_regime,
                support_resistance=support_resistance
            )

        except Exception as e:
            self.logger.error(f"Regime analysis failed: {e}")
            return RegimeAnalysis(
                regime='ranging', confidence=0.5, strength=0.5,
                duration_hours=1, trend_direction='neutral',
                volatility_level='medium', support_resistance={}
            )

    def _calculate_ema_alignment(self, latest_bar: pd.Series) -> float:
        """Calculate EMA alignment score (0.0 to 1.0)"""
        try:
            emas = []
            for period in [5, 13, 21, 50, 200]:
                ema_col = f'ema_{period}'
                if ema_col in latest_bar and pd.notna(latest_bar[ema_col]):
                    emas.append(latest_bar[ema_col])

            if len(emas) < 3:
                return 0.5

            # Check alignment
            bullish_alignment = all(emas[i] > emas[i+1] for i in range(len(emas)-1))
            bearish_alignment = all(emas[i] < emas[i+1] for i in range(len(emas)-1))

            if bullish_alignment:
                return 1.0
            elif bearish_alignment:
                return 0.0
            else:
                # Calculate partial alignment
                aligned_pairs = sum(1 for i in range(len(emas)-1)
                                  if (emas[i] > emas[i+1]) == (emas[0] > emas[-1]))
                return aligned_pairs / (len(emas) - 1) * 0.8

        except Exception:
            return 0.5

    def _calculate_trend_strength(self, recent_data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            if len(recent_data) < 10:
                return 0.5

            # ADX-like calculation
            high = recent_data['high']
            low = recent_data['low']
            close = recent_data['close']

            plus_dm = (high.diff().where(high.diff() > 0, 0)).rolling(14).mean()
            minus_dm = ((-low.diff()).where(low.diff() < 0, 0)).rolling(14).mean()
            atr = recent_data['atr'].iloc[-1] if 'atr' in recent_data.columns else 1

            if atr > 0:
                plus_di = 100 * plus_dm.iloc[-1] / atr
                minus_di = 100 * minus_dm.iloc[-1] / atr
                dx = abs(plus_di - minus_di) / (plus_di + minus_di)
                return min(1.0, dx)
            else:
                return 0.5

        except Exception:
            return 0.5

    def _classify_volatility_regime(self, recent_data: pd.DataFrame) -> str:
        """Classify current volatility regime"""
        try:
            if 'volatility_percentile' not in recent_data.columns:
                return 'medium'

            current_vol_percentile = recent_data['volatility_percentile'].iloc[-1]

            if current_vol_percentile < 0.2:
                return 'low'
            elif current_vol_percentile < 0.4:
                return 'medium_low'
            elif current_vol_percentile < 0.6:
                return 'medium'
            elif current_vol_percentile < 0.8:
                return 'medium_high'
            else:
                return 'high'

        except Exception:
            return 'medium'

    def _classify_regime(self,
                        ema_alignment: float,
                        trend_strength: float,
                        volatility_regime: str,
                        recent_data: pd.DataFrame) -> Tuple[str, float]:
        """Main regime classification logic"""
        try:
            # Strong trending conditions
            if trend_strength > 0.7 and ema_alignment > 0.8:
                return 'trending_up', 0.85
            elif trend_strength > 0.7 and ema_alignment < 0.2:
                return 'trending_down', 0.85

            # Breakout conditions (high volatility + moderate trend)
            elif volatility_regime in ['high', 'medium_high'] and trend_strength > 0.5:
                if ema_alignment > 0.6:
                    return 'breakout', 0.75
                elif ema_alignment < 0.4:
                    return 'breakout', 0.75

            # Ranging conditions (low trend strength)
            elif trend_strength < 0.4:
                confidence = 0.8 if volatility_regime in ['low', 'medium_low'] else 0.6
                return 'ranging', confidence

            # Default to ranging with moderate confidence
            else:
                return 'ranging', 0.5

        except Exception:
            return 'ranging', 0.5

    def _analyze_trading_session(self, df: pd.DataFrame, intelligence_history: pd.DataFrame = None) -> SessionAnalysis:
        """Analyze current trading session characteristics"""
        try:
            current_hour = datetime.utcnow().hour

            # Determine current session
            if 22 <= current_hour or current_hour < 8:
                session = 'asian'
                strength = 0.6
                volatility = 'low'
                liquidity = 'medium'
            elif 8 <= current_hour < 13:
                session = 'london'
                strength = 0.9
                volatility = 'high'
                liquidity = 'high'
            elif 13 <= current_hour < 16:
                session = 'overlap'
                strength = 1.0
                volatility = 'peak'
                liquidity = 'peak'
            else:
                session = 'new_york'
                strength = 0.8
                volatility = 'high'
                liquidity = 'high'

            # Session-specific optimal pairs
            optimal_pairs = {
                'asian': ['USDJPY', 'AUDUSD', 'NZDUSD'],
                'london': ['EURUSD', 'GBPUSD', 'EURGBP'],
                'overlap': ['EURUSD', 'GBPUSD', 'USDCAD'],
                'new_york': ['EURUSD', 'USDCAD', 'USDCHF']
            }.get(session, ['EURUSD'])

            # Recommended timeframes
            recommended_timeframes = {
                'asian': ['15m', '1h'],
                'london': ['5m', '15m', '1h'],
                'overlap': ['5m', '15m'],
                'new_york': ['15m', '1h']
            }.get(session, ['15m'])

            return SessionAnalysis(
                current_session=session,
                session_strength=strength,
                optimal_pairs=optimal_pairs,
                volatility_forecast=volatility,
                liquidity_level=liquidity,
                recommended_timeframes=recommended_timeframes
            )

        except Exception as e:
            self.logger.error(f"Session analysis failed: {e}")
            return SessionAnalysis(
                current_session='unknown',
                session_strength=0.5,
                optimal_pairs=['EURUSD'],
                volatility_forecast='medium',
                liquidity_level='medium',
                recommended_timeframes=['15m']
            )

    def _identify_success_patterns(self, epic: str, df: pd.DataFrame) -> List[SuccessPattern]:
        """Identify successful trading patterns from recent data"""
        try:
            patterns = []

            # EMA crossover success pattern
            ema_pattern = self._analyze_ema_crossover_pattern(df)
            if ema_pattern:
                patterns.append(ema_pattern)

            # Breakout success pattern
            breakout_pattern = self._analyze_breakout_pattern(df)
            if breakout_pattern:
                patterns.append(breakout_pattern)

            # Mean reversion pattern
            mean_reversion_pattern = self._analyze_mean_reversion_pattern(df)
            if mean_reversion_pattern:
                patterns.append(mean_reversion_pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Success pattern identification failed: {e}")
            return []

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached analysis is still valid"""
        if cache_key not in self.cache:
            return False

        timestamp_key = f"{cache_key}_timestamp"
        if timestamp_key not in self.cache:
            return False

        cache_time = self.cache[timestamp_key]
        return datetime.utcnow() - cache_time < self.cache_duration

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Get fallback analysis when primary analysis fails"""
        return {
            'epic': 'unknown',
            'analysis_timestamp': datetime.utcnow(),
            'data_period_hours': 24,
            'total_bars': 0,
            'regime_analysis': RegimeAnalysis(
                regime='ranging', confidence=0.5, strength=0.5,
                duration_hours=1, trend_direction='neutral',
                volatility_level='medium', support_resistance={}
            ),
            'session_analysis': self._analyze_trading_session(pd.DataFrame()),
            'success_patterns': [],
            'recent_performance': {'win_rate': 0.5, 'avg_profit': 0.0},
            'optimal_parameters': {'confidence_threshold': 0.6},
            'market_forecast': {'direction': 'neutral', 'confidence': 0.5},
            'recommended_approach': 'conservative'
        }

    # Placeholder methods for additional analysis features
    def _analyze_volatility_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility patterns - placeholder"""
        return {'current_percentile': 0.5, 'trend': 'stable'}

    def _analyze_trend_strength(self, df: pd.DataFrame) -> Dict:
        """Analyze overall trend strength - placeholder"""
        return {'strength': 0.5, 'direction': 'neutral'}

    def _identify_key_levels(self, df: pd.DataFrame) -> Dict:
        """Identify key support/resistance levels - placeholder"""
        if df.empty:
            return {}
        latest_close = df['close'].iloc[-1]
        return {
            'support': latest_close * 0.995,
            'resistance': latest_close * 1.005
        }

    def _identify_failure_patterns(self, epic: str, df: pd.DataFrame) -> List:
        """Identify failure patterns - placeholder"""
        return []

    def _analyze_recent_performance(self, epic: str, df: pd.DataFrame) -> Dict:
        """Analyze recent performance metrics - placeholder"""
        return {'win_rate': 0.6, 'avg_profit': 25.5}

    def _suggest_optimal_parameters(self, epic: str, df: pd.DataFrame) -> Dict:
        """Suggest optimal parameters - placeholder"""
        return {'confidence_threshold': 0.65, 'stop_loss_pips': 20}

    def _generate_market_forecast(self, df: pd.DataFrame) -> Dict:
        """Generate market forecast - placeholder"""
        return {'direction': 'neutral', 'confidence': 0.5}

    def _recommend_trading_approach(self, df: pd.DataFrame) -> str:
        """Recommend trading approach - placeholder"""
        return 'balanced'

    # Additional placeholder methods
    def _estimate_regime_duration(self, df: pd.DataFrame) -> int:
        return 6  # hours

    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        return 'neutral'

    def _calculate_key_levels(self, df: pd.DataFrame) -> Dict:
        return {}

    def _analyze_ema_crossover_pattern(self, df: pd.DataFrame) -> Optional[SuccessPattern]:
        return None

    def _analyze_breakout_pattern(self, df: pd.DataFrame) -> Optional[SuccessPattern]:
        return None

    def _analyze_mean_reversion_pattern(self, df: pd.DataFrame) -> Optional[SuccessPattern]:
        return None

    def _analyze_intelligence_history(self, intelligence_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze historical market intelligence data for enhanced insights

        Args:
            intelligence_df: DataFrame with historical intelligence data

        Returns:
            Dictionary with intelligence insights
        """
        if intelligence_df.empty:
            self.logger.warning("📊 No intelligence history available for analysis")
            return {
                'historical_regime_pattern': 'unknown',
                'regime_stability': 0.5,
                'session_patterns': {},
                'recent_bias': 'neutral',
                'confidence_trend': 'stable'
            }

        try:
            # Analyze regime patterns
            regime_counts = intelligence_df['dominant_regime'].value_counts()
            most_common_regime = regime_counts.index[0] if not regime_counts.empty else 'ranging'
            regime_stability = regime_counts.iloc[0] / len(intelligence_df) if not regime_counts.empty else 0.5

            # Analyze session patterns
            session_patterns = {}
            if 'current_session' in intelligence_df.columns:
                for session in intelligence_df['current_session'].unique():
                    session_data = intelligence_df[intelligence_df['current_session'] == session]
                    if not session_data.empty:
                        session_patterns[session] = {
                            'dominant_regime': session_data['dominant_regime'].mode().iloc[0] if not session_data['dominant_regime'].mode().empty else 'ranging',
                            'avg_confidence': float(session_data['regime_confidence'].mean()) if 'regime_confidence' in session_data.columns else 0.5,
                            'volatility_pattern': session_data['session_volatility'].mode().iloc[0] if 'session_volatility' in session_data.columns and not session_data['session_volatility'].mode().empty else 'medium'
                        }

            # Analyze market bias trends
            recent_bias = 'neutral'
            if 'market_bias' in intelligence_df.columns and not intelligence_df['market_bias'].empty:
                bias_counts = intelligence_df['market_bias'].value_counts()
                recent_bias = bias_counts.index[0] if not bias_counts.empty else 'neutral'

            # Analyze confidence trends
            confidence_trend = 'stable'
            if 'regime_confidence' in intelligence_df.columns and len(intelligence_df) >= 5:
                recent_confidence = intelligence_df.head(5)['regime_confidence'].mean()
                older_confidence = intelligence_df.tail(5)['regime_confidence'].mean()

                if recent_confidence > older_confidence * 1.1:
                    confidence_trend = 'increasing'
                elif recent_confidence < older_confidence * 0.9:
                    confidence_trend = 'decreasing'

            insights = {
                'historical_regime_pattern': most_common_regime,
                'regime_stability': float(regime_stability),
                'session_patterns': session_patterns,
                'recent_bias': recent_bias,
                'confidence_trend': confidence_trend,
                'intelligence_records_analyzed': len(intelligence_df),
                'time_span_hours': int((intelligence_df['scan_timestamp'].max() - intelligence_df['scan_timestamp'].min()).total_seconds() / 3600) if len(intelligence_df) > 1 else 0
            }

            self.logger.info(f"✅ Intelligence history analysis: {most_common_regime} regime ({regime_stability:.1%} stability)")
            return insights

        except Exception as e:
            self.logger.error(f"Intelligence history analysis failed: {e}")
            return {
                'historical_regime_pattern': 'unknown',
                'regime_stability': 0.5,
                'session_patterns': {},
                'recent_bias': 'neutral',
                'confidence_trend': 'stable',
                'error': str(e)
            }