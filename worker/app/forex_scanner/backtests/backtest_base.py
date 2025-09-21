# ============================================================================
# backtests/backtest_base.py - Enhanced Unified Backtest Framework
# ============================================================================

import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from core.database import DatabaseManager
    from core.data_fetcher import DataFetcher
    from core.backtest.performance_analyzer import PerformanceAnalyzer
    from core.backtest.signal_analyzer import SignalAnalyzer
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner.core.backtest.performance_analyzer import PerformanceAnalyzer
    from forex_scanner.core.backtest.signal_analyzer import SignalAnalyzer

# Import optimization service
try:
    from optimization.optimal_parameter_service import OptimalParameterService
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.optimization.optimal_parameter_service import OptimalParameterService
        OPTIMIZATION_AVAILABLE = True
    except ImportError:
        OPTIMIZATION_AVAILABLE = False
        logging.getLogger(__name__).warning("Optimization service not available - using fallback parameters")

# Import unified parameter manager
try:
    from .parameter_manager import ParameterManager, ParameterSet
except ImportError:
    try:
        from parameter_manager import ParameterManager, ParameterSet
    except ImportError:
        ParameterManager = None
        ParameterSet = None
        logging.getLogger(__name__).warning("ParameterManager not available - using basic parameter handling")

try:
    import config
except ImportError:
    from forex_scanner import config


# ============================================================================
# Standardized Formats and Data Classes
# ============================================================================

class SignalType(Enum):
    """Standardized signal types"""
    BUY = "BUY"
    SELL = "SELL"
    BULL = "BULL"
    BEAR = "BEAR"
    LONG = "LONG"
    SHORT = "SHORT"


class MarketRegime(Enum):
    """Market regime types for intelligent backtesting"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class TradingSession(Enum):
    """Trading session types"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "london_ny_overlap"
    OVERNIGHT = "overnight"


@dataclass
class MarketConditions:
    """Market conditions and intelligence data"""
    regime: MarketRegime = MarketRegime.UNKNOWN
    volatility_percentile: float = 0.5
    session: TradingSession = TradingSession.OVERNIGHT
    major_events: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    trend_strength: float = 0.0
    support_resistance_levels: Dict[str, float] = field(default_factory=dict)
    volume_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardSignal:
    """Standardized signal format for all strategies"""
    # Core signal data
    signal_type: SignalType
    strategy: str
    epic: str
    price: float
    confidence: float  # 0.0-1.0
    timestamp: str  # UTC timestamp

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    # Market intelligence
    market_conditions: Optional[MarketConditions] = None

    # Performance metrics (calculated post-signal)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    # Smart Money analysis (if available)
    smart_money_analysis: Dict[str, Any] = field(default_factory=dict)

    # Additional metadata
    timeframe: str = "15m"
    spread_pips: float = 0.0
    session_info: Dict[str, Any] = field(default_factory=dict)
    technical_indicators: Dict[str, Any] = field(default_factory=dict)
    entry_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        result = {
            'signal_type': self.signal_type.value if isinstance(self.signal_type, SignalType) else str(self.signal_type),
            'strategy': self.strategy,
            'epic': self.epic,
            'price': self.price,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': self.risk_reward_ratio,
            'timeframe': self.timeframe,
            'spread_pips': self.spread_pips,
            'entry_reason': self.entry_reason,
            'performance_metrics': self.performance_metrics,
            'smart_money_analysis': self.smart_money_analysis,
            'session_info': self.session_info,
            'technical_indicators': self.technical_indicators
        }

        # Add market conditions if available
        if self.market_conditions:
            result['market_conditions'] = {
                'regime': self.market_conditions.regime.value,
                'volatility_percentile': self.market_conditions.volatility_percentile,
                'session': self.market_conditions.session.value,
                'sentiment_score': self.market_conditions.sentiment_score,
                'trend_strength': self.market_conditions.trend_strength,
                'major_events': self.market_conditions.major_events,
                'support_resistance_levels': self.market_conditions.support_resistance_levels,
                'volume_profile': self.market_conditions.volume_profile
            }

        return result


@dataclass
class EpicResult:
    """Results for a single epic"""
    epic: str
    signals: List[StandardSignal]
    performance_metrics: Dict[str, Any]
    market_conditions_summary: MarketConditions
    data_quality: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class StandardBacktestResult:
    """Standardized backtest result format"""
    strategy_name: str
    epic_results: Dict[str, EpicResult]  # epic -> EpicResult
    total_signals: int
    overall_performance: Dict[str, Any]
    market_intelligence_summary: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

    # API compatibility fields
    @property
    def epic(self) -> str:
        """Main epic for API compatibility"""
        if len(self.epic_results) == 1:
            return list(self.epic_results.keys())[0]
        return "ALL_EPICS"

    @property
    def signals(self) -> List[Dict[str, Any]]:
        """All signals as dict list for API compatibility"""
        all_signals = []
        for epic_result in self.epic_results.values():
            for signal in epic_result.signals:
                all_signals.append(signal.to_dict())
        return all_signals

    @property
    def performance_metrics(self) -> Dict[str, Any]:
        """Overall performance metrics for API compatibility"""
        return self.overall_performance

    @property
    def execution_time(self) -> float:
        """Execution time for API compatibility"""
        return self.execution_metadata.get('execution_time', 0.0)

    @property
    def timeframe(self) -> str:
        """Timeframe for API compatibility"""
        return self.execution_metadata.get('timeframe', '15m')


class BacktestBase(ABC):
    """Enhanced base class for all strategy backtests with standardized formats and market intelligence"""

    def __init__(self, strategy_name: str, use_optimal_parameters: bool = True, enable_caching: bool = True, **kwargs):
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(f"backtest_{strategy_name}")
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager, config.USER_TIMEZONE)
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()

        # Performance optimization settings
        self.enable_caching = enable_caching
        self.result_cache = {} if enable_caching else None
        self.data_cache = {} if enable_caching else None
        self.max_cache_size = 50  # Maximum cached results

        # Database optimization integration
        self.use_optimal_parameters = use_optimal_parameters and OPTIMIZATION_AVAILABLE
        self.optimal_service = OptimalParameterService() if self.use_optimal_parameters else None

        # Enhanced parameter management
        self.parameter_manager = ParameterManager(use_optimal_parameters=use_optimal_parameters) if ParameterManager else None

        # Market intelligence initialization
        self.market_intelligence_enabled = True
        try:
            from core.market_intelligence import MarketIntelligenceEngine
            self.market_intelligence = MarketIntelligenceEngine()
        except ImportError:
            self.market_intelligence_enabled = False
            self.logger.warning("⚠️ Market intelligence not available")

        # Smart Money integration
        self.smart_money_enabled = False
        try:
            from core.smart_money_integration import SmartMoneyIntegration
            self.smart_money_integration = SmartMoneyIntegration(
                database_manager=self.db_manager,
                data_fetcher=self.data_fetcher
            )
            self.smart_money_enabled = True
        except ImportError:
            self.logger.info("ℹ️ Smart Money integration not available (optional)")

        # Performance tracking
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'execution_times': [],
            'total_signals_processed': 0,
            'total_epics_processed': 0,
            'total_backtests_run': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'total_signals_generated': 0
        }

        # Log initialization status
        features = []
        if self.parameter_manager:
            features.append("🎯 ParameterManager")
        elif self.use_optimal_parameters:
            features.append("🎯 Database Optimization")

        if self.market_intelligence_enabled:
            features.append("🧠 Market Intelligence")

        if self.smart_money_enabled:
            features.append("💰 Smart Money")

        if self.enable_caching:
            features.append("⚡ Performance Caching")

        self.logger.info(f"🚀 Enhanced {strategy_name} initialized: {', '.join(features)}")
    
    @abstractmethod
    def initialize_strategy(self, epic: str = None):
        """Initialize the specific strategy with optional epic for optimal parameters"""
        pass

    @abstractmethod
    def run_strategy_backtest(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[StandardSignal]:
        """Run backtest for this specific strategy - returns standardized signals"""
        pass

    def get_market_conditions(self, timestamp: datetime, data: pd.DataFrame) -> MarketConditions:
        """Get market conditions for a given timestamp"""
        if not self.market_intelligence_enabled:
            return MarketConditions()  # Return default conditions

        try:
            # Detect market regime
            regime = self.market_intelligence.detect_regime(data)

            # Get trading session
            session = self._get_trading_session(timestamp)

            # Calculate volatility percentile
            volatility_percentile = self._calculate_volatility_percentile(data)

            # Get trend strength
            trend_strength = self._calculate_trend_strength(data)

            return MarketConditions(
                regime=regime,
                volatility_percentile=volatility_percentile,
                session=session,
                trend_strength=trend_strength,
                major_events=[],  # Could be enhanced with news API
                sentiment_score=0.0,  # Could be enhanced with sentiment analysis
                support_resistance_levels={},  # Could be enhanced with level detection
                volume_profile={}  # Could be enhanced with volume analysis
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Market conditions detection failed: {e}")
            return MarketConditions()

    def _get_trading_session(self, timestamp: datetime) -> TradingSession:
        """Determine trading session from timestamp"""
        hour = timestamp.hour

        # Asian session: 22:00 - 08:00 UTC (previous day)
        if hour >= 22 or hour < 8:
            return TradingSession.ASIAN

        # London session: 08:00 - 16:00 UTC
        elif 8 <= hour < 16:
            # London-NY overlap: 13:00 - 16:00 UTC
            if 13 <= hour < 16:
                return TradingSession.OVERLAP_LONDON_NY
            return TradingSession.LONDON

        # New York session: 13:00 - 22:00 UTC
        elif 13 <= hour < 22:
            return TradingSession.NEW_YORK

        else:
            return TradingSession.OVERNIGHT

    def _calculate_volatility_percentile(self, data: pd.DataFrame, window: int = 100) -> float:
        """Calculate volatility percentile"""
        try:
            if len(data) < window:
                return 0.5  # Default to median

            # Calculate ATR-based volatility
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()

            if len(atr) < window:
                return 0.5

            # Calculate percentile of current ATR
            current_atr = atr.iloc[-1]
            atr_window = atr.tail(window)
            percentile = (atr_window < current_atr).sum() / len(atr_window)

            return min(max(percentile, 0.0), 1.0)
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility calculation failed: {e}")
            return 0.5

    def _calculate_trend_strength(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate trend strength (0.0 = no trend, 1.0 = strong trend)"""
        try:
            if len(data) < window:
                return 0.0

            # Use EMA slope and price position relative to EMA
            ema = data['close'].ewm(span=window).mean()
            ema_slope = (ema.iloc[-1] - ema.iloc[-window]) / window
            price_position = (data['close'].iloc[-1] - ema.iloc[-1]) / ema.iloc[-1]

            # Combine slope and position for trend strength
            trend_strength = abs(ema_slope) * 1000 + abs(price_position) * 2

            return min(trend_strength, 1.0)
        except Exception as e:
            self.logger.warning(f"⚠️ Trend strength calculation failed: {e}")
            return 0.0

    def standardize_signal(self, raw_signal: Dict[str, Any], epic: str, timeframe: str,
                          market_conditions: Optional[MarketConditions] = None) -> StandardSignal:
        """Convert raw signal dict to StandardSignal format"""
        try:
            # Extract and standardize signal type
            signal_type_raw = raw_signal.get('signal_type', '')
            signal_type = self._parse_signal_type(signal_type_raw)

            # Standardize confidence (ensure 0.0-1.0 range)
            confidence = raw_signal.get('confidence', raw_signal.get('confidence_score', 0.0))
            if confidence > 1.0:
                confidence = confidence / 100.0

            # Create StandardSignal
            signal = StandardSignal(
                signal_type=signal_type,
                strategy=self.strategy_name,
                epic=epic,
                price=float(raw_signal.get('price', 0.0)),
                confidence=float(confidence),
                timestamp=str(raw_signal.get('timestamp', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))),
                stop_loss=raw_signal.get('stop_loss'),
                take_profit=raw_signal.get('take_profit'),
                risk_reward_ratio=raw_signal.get('risk_reward_ratio'),
                market_conditions=market_conditions,
                timeframe=timeframe,
                spread_pips=float(raw_signal.get('spread_pips', 0.0)),
                entry_reason=raw_signal.get('entry_reason', ''),
                performance_metrics=raw_signal.get('performance_metrics', {}),
                smart_money_analysis=raw_signal.get('smart_money_analysis', {}),
                session_info=raw_signal.get('session_info', {}),
                technical_indicators=raw_signal.get('technical_indicators', {})
            )

            return signal

        except Exception as e:
            self.logger.error(f"❌ Signal standardization failed: {e}")
            # Return minimal valid signal
            return StandardSignal(
                signal_type=SignalType.BUY,
                strategy=self.strategy_name,
                epic=epic,
                price=0.0,
                confidence=0.0,
                timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                timeframe=timeframe
            )

    def _parse_signal_type(self, signal_type_str: str) -> SignalType:
        """Parse signal type string to SignalType enum"""
        # Handle None or empty signal type
        if not signal_type_str:
            return SignalType.BUY

        signal_type_str = str(signal_type_str).upper()

        # Direct matches
        for signal_type in SignalType:
            if signal_type.value == signal_type_str:
                return signal_type

        # Fuzzy matches
        if 'BUY' in signal_type_str or 'BULL' in signal_type_str or 'LONG' in signal_type_str:
            return SignalType.BUY
        elif 'SELL' in signal_type_str or 'BEAR' in signal_type_str or 'SHORT' in signal_type_str:
            return SignalType.SELL

        # Default fallback
        return SignalType.BUY

    def enhance_signal_with_smart_money(self, signal: StandardSignal) -> StandardSignal:
        """Enhance signal with Smart Money analysis if available"""
        if not self.smart_money_enabled:
            return signal

        try:
            enhanced_signal_dict = self.smart_money_integration.enhance_signal_with_smart_money(
                signal=signal.to_dict(),
                epic=signal.epic,
                timeframe=signal.timeframe
            )

            if enhanced_signal_dict and 'smart_money_analysis' in enhanced_signal_dict:
                signal.smart_money_analysis = enhanced_signal_dict['smart_money_analysis']

            return signal

        except Exception as e:
            self.logger.warning(f"⚠️ Smart Money enhancement failed: {e}")
            return signal
    
    def get_parameters(
        self,
        epic: Optional[str] = None,
        user_parameters: Optional[Dict[str, Any]] = None,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Union[ParameterSet, None]:
        """Get unified parameter set using ParameterManager"""
        if not self.parameter_manager:
            return self.get_optimal_parameters(epic)  # Fallback to legacy method

        try:
            param_set = self.parameter_manager.get_parameters(
                strategy_name=self.strategy_name,
                epic=epic,
                user_parameters=user_parameters,
                market_conditions=market_conditions
            )

            # Log parameter summary
            summary = self.parameter_manager.get_parameter_info_summary(param_set)
            self.logger.info(f"📋 {summary}")

            return param_set

        except Exception as e:
            self.logger.error(f"❌ Failed to get parameters: {e}")
            return self.get_optimal_parameters(epic)  # Fallback

    def get_optimal_parameters(self, epic: str):
        """Legacy method for optimal parameters (backward compatibility)"""
        if not self.use_optimal_parameters or not self.optimal_service:
            return None

        try:
            params = self.optimal_service.get_epic_parameters(epic)
            self.logger.info(f"✅ Using optimal parameters for {epic}:")
            self.logger.info(f"   Config: {getattr(params, 'ema_config', 'N/A')}")
            self.logger.info(f"   Confidence: {params.confidence_threshold:.1%}")
            self.logger.info(f"   Timeframe: {params.timeframe}")
            self.logger.info(f"   SL/TP: {params.stop_loss_pips:.0f}/{params.take_profit_pips:.0f} pips")
            self.logger.info(f"   R:R: {params.risk_reward_ratio:.1f}")
            self.logger.info(f"   Performance: {params.performance_score:.3f}")
            return params
        except Exception as e:
            self.logger.warning(f"❌ Failed to get optimal parameters for {epic}: {e}")
            return None

    def validate_parameters(self, param_set: ParameterSet) -> bool:
        """Validate parameter set using ParameterManager"""
        if not self.parameter_manager or not param_set:
            return True  # Skip validation if manager not available

        errors = self.parameter_manager.validate_parameter_set(param_set)
        if errors:
            self.logger.warning(f"⚠️ Parameter validation errors:")
            for param_name, error_msg in errors.items():
                self.logger.warning(f"   {param_name}: {error_msg}")
            return False

        return True
    
    def get_cache_key(self, epic_list, days: int, timeframe: str, kwargs: dict = None) -> str:
        """Generate cache key for backtest results"""
        if not self.enable_caching:
            return None

        # Handle both single epic (str) and list of epics
        if isinstance(epic_list, str):
            epic_key = epic_list
        elif isinstance(epic_list, list):
            epic_key = "|".join(sorted(epic_list)) if epic_list else "ALL_EPICS"
        else:
            epic_key = "ALL_EPICS"

        # Create deterministic cache key
        key_parts = [
            self.strategy_name,
            epic_key,
            str(days),
            timeframe
        ]

        # Add significant parameters to cache key
        if kwargs:
            for param_name in sorted(kwargs.keys()):
                if param_name not in ['show_signals', 'user_parameters']:
                    key_parts.append(f"{param_name}:{kwargs[param_name]}")

        return "_".join(key_parts)

    def get_cached_result(self, cache_key: str) -> Optional[StandardBacktestResult]:
        """Get cached backtest result if available"""
        if not self.enable_caching or not cache_key or not self.result_cache:
            return None

        cached_result = self.result_cache.get(cache_key)
        if cached_result:
            self.performance_stats['cache_hits'] += 1
            self.logger.info(f"📋 Using cached result for {cache_key}")
            return cached_result

        self.performance_stats['cache_misses'] += 1
        return None

    def cache_result(self, cache_key: str, result: StandardBacktestResult):
        """Cache backtest result for future use"""
        if not self.enable_caching or not cache_key or not self.result_cache:
            return

        # Manage cache size
        if len(self.result_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = list(self.result_cache.keys())[0]
            del self.result_cache[oldest_key]

        self.result_cache[cache_key] = result
        self.logger.info(f"💾 Cached result for {cache_key}")

    def clear_cache(self):
        """Clear all cached results"""
        if self.result_cache:
            self.result_cache.clear()
        if self.data_cache:
            self.data_cache.clear()
        if self.parameter_manager:
            self.parameter_manager.clear_cache()
        if self.market_intelligence_enabled and hasattr(self.market_intelligence, 'clear_cache'):
            self.market_intelligence.clear_cache()

        self.logger.info("🗑️ All caches cleared")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance optimization summary"""
        stats = self.performance_stats.copy()

        if stats['cache_hits'] + stats['cache_misses'] > 0:
            cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            stats['cache_hit_rate'] = cache_hit_rate

        if stats['execution_times']:
            stats['avg_execution_time'] = sum(stats['execution_times']) / len(stats['execution_times'])
            stats['min_execution_time'] = min(stats['execution_times'])
            stats['max_execution_time'] = max(stats['execution_times'])

        stats['features_enabled'] = {
            'caching': self.enable_caching,
            'parameter_optimization': self.use_optimal_parameters,
            'market_intelligence': self.market_intelligence_enabled,
            'smart_money': self.smart_money_enabled
        }

        return stats

    def update_performance_stats(self, result: StandardBacktestResult):
        """Update performance statistics after backtest execution"""
        try:
            self.performance_stats['total_backtests_run'] += 1

            if result.success:
                self.performance_stats['successful_backtests'] += 1
            else:
                self.performance_stats['failed_backtests'] += 1

            # Track execution time if available
            if result.execution_metadata and 'execution_time' in result.execution_metadata:
                execution_time = result.execution_metadata['execution_time']
                self.performance_stats['execution_times'].append(execution_time)

            # Track signal counts
            self.performance_stats['total_signals_generated'] += result.total_signals

        except Exception as e:
            self.logger.warning(f"⚠️ Performance stats update failed: {e}")

    def run_backtest(
        self,
        epic: str = None,
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        **kwargs
    ) -> StandardBacktestResult:
        """Enhanced backtest execution with standardized results and market intelligence"""
        start_time = datetime.now()
        epic_list = [epic] if epic else config.EPIC_LIST

        # Check cache if enabled
        if self.enable_caching:
            cache_key = self.get_cache_key(epic_list, days, timeframe, kwargs)
            cached_result = self.get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"🚀 Using cached result for {self.strategy_name}")
                return cached_result

        self.logger.info(f"🧪 Running {self.strategy_name} backtest")
        self.logger.info(f"   Epic(s): {epic_list}")
        self.logger.info(f"   Days: {days}, Timeframe: {timeframe}")
        self.logger.info(f"   Database optimization: {'✅ ENABLED' if self.use_optimal_parameters else '❌ DISABLED'}")
        self.logger.info(f"   Market intelligence: {'✅ ENABLED' if self.market_intelligence_enabled else '❌ DISABLED'}")
        self.logger.info(f"   Smart Money analysis: {'✅ ENABLED' if self.smart_money_enabled else '❌ DISABLED'}")

        epic_results = {}
        all_signals = []
        execution_errors = []

        try:
            for current_epic in epic_list:
                self.logger.info(f"\n📊 Processing {current_epic}")

                try:
                    # Initialize strategy with epic for optimal parameters
                    strategy = self.initialize_strategy(current_epic)

                    # Get data - extract pair from epic (e.g., CS.D.EURUSD.MINI.IP -> EURUSD)
                    pair = current_epic.split('.')[2] if '.' in current_epic else current_epic
                    df = self.data_fetcher.get_enhanced_data(
                        epic=current_epic,
                        pair=pair,
                        timeframe=timeframe,
                        lookback_hours=days * 24
                    )

                    if df is None or df.empty:
                        error_msg = f"No data available for {current_epic}"
                        self.logger.warning(f"❌ {error_msg}")
                        epic_results[current_epic] = EpicResult(
                            epic=current_epic,
                            signals=[],
                            performance_metrics={},
                            market_conditions_summary=MarketConditions(),
                            error_message=error_msg
                        )
                        continue

                    # Get market conditions for the data period
                    market_conditions = self.get_market_conditions(
                        timestamp=datetime.now(),  # Could be enhanced with data timestamp
                        data=df
                    )

                    # Run strategy backtest (now returns StandardSignal list)
                    signals = self.run_strategy_backtest(
                        df, current_epic, config.SPREAD_PIPS, timeframe
                    )

                    # Enhance signals with market intelligence and Smart Money analysis
                    enhanced_signals = []
                    for signal in signals:
                        # Ensure signal has market conditions
                        if signal.market_conditions is None:
                            signal.market_conditions = market_conditions

                        # Apply Smart Money enhancement
                        enhanced_signal = self.enhance_signal_with_smart_money(signal)
                        enhanced_signals.append(enhanced_signal)

                    # Calculate performance metrics for this epic
                    epic_performance = self._calculate_epic_performance(enhanced_signals)

                    # Create epic result
                    epic_results[current_epic] = EpicResult(
                        epic=current_epic,
                        signals=enhanced_signals,
                        performance_metrics=epic_performance,
                        market_conditions_summary=market_conditions,
                        data_quality={
                            'data_points': len(df),
                            'time_range': f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "No data",
                            'missing_data_percentage': 0.0  # Could be enhanced
                        }
                    )

                    all_signals.extend(enhanced_signals)
                    self.logger.info(f"   ✅ Found {len(enhanced_signals)} signals for {current_epic}")

                except Exception as epic_error:
                    error_msg = f"Failed to process {current_epic}: {str(epic_error)}"
                    self.logger.error(f"❌ {error_msg}")
                    execution_errors.append(error_msg)

                    epic_results[current_epic] = EpicResult(
                        epic=current_epic,
                        signals=[],
                        performance_metrics={},
                        market_conditions_summary=MarketConditions(),
                        error_message=error_msg
                    )

            # Calculate overall performance
            overall_performance = self._calculate_overall_performance(all_signals)

            # Create market intelligence summary
            market_intelligence_summary = self._create_market_intelligence_summary(epic_results)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Create execution metadata
            execution_metadata = {
                'execution_time': execution_time,
                'timeframe': timeframe,
                'days': days,
                'total_epics_processed': len(epic_list),
                'successful_epics': len([r for r in epic_results.values() if not r.error_message]),
                'failed_epics': len([r for r in epic_results.values() if r.error_message]),
                'database_optimization_used': self.use_optimal_parameters,
                'market_intelligence_used': self.market_intelligence_enabled,
                'smart_money_used': self.smart_money_enabled,
                'errors': execution_errors
            }

            # Create standardized result
            result = StandardBacktestResult(
                strategy_name=self.strategy_name,
                epic_results=epic_results,
                total_signals=len(all_signals),
                overall_performance=overall_performance,
                market_intelligence_summary=market_intelligence_summary,
                execution_metadata=execution_metadata,
                success=len(all_signals) > 0 and len(execution_errors) == 0
            )

            # Display results if requested
            if show_signals and all_signals:
                self._display_enhanced_signals(all_signals)

            self._display_enhanced_performance(result)

            self.logger.info(f"✅ Backtest completed in {execution_time:.2f}s")
            self.logger.info(f"   Total signals: {len(all_signals)}")
            self.logger.info(f"   Successful epics: {execution_metadata['successful_epics']}/{len(epic_list)}")

            # Store result in cache if enabled
            if self.enable_caching:
                cache_key = self.get_cache_key(epic_list, days, timeframe, kwargs)
                self.cache_result(cache_key, result)
                self.logger.debug(f"📦 Cached result for future use")

            # Update performance statistics
            self.update_performance_stats(result)

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Backtest execution failed: {str(e)}"
            self.logger.error(f"❌ {error_message}")

            # Return error result
            return StandardBacktestResult(
                strategy_name=self.strategy_name,
                epic_results=epic_results,
                total_signals=0,
                overall_performance={},
                market_intelligence_summary={},
                execution_metadata={
                    'execution_time': execution_time,
                    'timeframe': timeframe,
                    'days': days,
                    'errors': [error_message]
                },
                success=False,
                error_message=error_message
            )
    
    def _calculate_epic_performance(self, signals: List[StandardSignal]) -> Dict[str, Any]:
        """Calculate performance metrics for a single epic"""
        if not signals:
            return {
                'total_signals': 0,
                'avg_confidence': 0.0,
                'win_rate': 0.0,
                'avg_profit_pips': 0.0,
                'avg_loss_pips': 0.0,
                'risk_reward_ratio': 0.0,
                'best_signal_confidence': 0.0,
                'signal_types': {}
            }

        try:
            total_signals = len(signals)

            # Confidence analysis
            confidences = [s.confidence for s in signals if s.confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            best_confidence = max(confidences) if confidences else 0.0

            # Signal type distribution
            signal_types = {}
            for signal in signals:
                signal_type = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1

            # Performance metrics (if available)
            profitable_signals = []
            losing_signals = []

            for signal in signals:
                perf_metrics = signal.performance_metrics
                if perf_metrics and 'max_profit_pips' in perf_metrics and 'max_loss_pips' in perf_metrics:
                    if perf_metrics.get('is_winner', False):
                        profitable_signals.append(perf_metrics['max_profit_pips'])
                    elif perf_metrics.get('is_loser', False):
                        losing_signals.append(perf_metrics['max_loss_pips'])

            win_rate = len(profitable_signals) / (len(profitable_signals) + len(losing_signals)) if (len(profitable_signals) + len(losing_signals)) > 0 else 0.0
            avg_profit = sum(profitable_signals) / len(profitable_signals) if profitable_signals else 0.0
            avg_loss = sum(losing_signals) / len(losing_signals) if losing_signals else 0.0
            risk_reward = avg_profit / avg_loss if avg_loss > 0 else float('inf')

            return {
                'total_signals': total_signals,
                'avg_confidence': avg_confidence,
                'win_rate': win_rate,
                'avg_profit_pips': avg_profit,
                'avg_loss_pips': avg_loss,
                'risk_reward_ratio': risk_reward,
                'best_signal_confidence': best_confidence,
                'signal_types': signal_types,
                'profitable_signals': len(profitable_signals),
                'losing_signals': len(losing_signals)
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Performance calculation failed: {e}")
            return {'total_signals': len(signals), 'error': str(e)}

    def _calculate_overall_performance(self, all_signals: List[StandardSignal]) -> Dict[str, Any]:
        """Calculate overall performance across all epics"""
        if not all_signals:
            return {}

        # Use the same logic as epic performance but across all signals
        overall_perf = self._calculate_epic_performance(all_signals)

        # Add additional overall metrics
        overall_perf['total_epics_with_signals'] = len(set(s.epic for s in all_signals))

        # Market intelligence summary
        market_regimes = {}
        trading_sessions = {}

        for signal in all_signals:
            if signal.market_conditions:
                regime = signal.market_conditions.regime.value if hasattr(signal.market_conditions.regime, 'value') else str(signal.market_conditions.regime)
                session = signal.market_conditions.session.value if hasattr(signal.market_conditions.session, 'value') else str(signal.market_conditions.session)

                market_regimes[regime] = market_regimes.get(regime, 0) + 1
                trading_sessions[session] = trading_sessions.get(session, 0) + 1

        overall_perf['market_regime_distribution'] = market_regimes
        overall_perf['trading_session_distribution'] = trading_sessions

        return overall_perf

    def _create_market_intelligence_summary(self, epic_results: Dict[str, EpicResult]) -> Dict[str, Any]:
        """Create market intelligence summary from epic results"""
        summary = {
            'market_conditions_detected': self.market_intelligence_enabled,
            'smart_money_analysis_applied': self.smart_money_enabled,
            'regime_analysis': {},
            'volatility_analysis': {},
            'session_analysis': {}
        }

        if not self.market_intelligence_enabled:
            return summary

        # Aggregate market conditions across epics
        regimes = []
        volatilities = []
        sessions = []

        for epic_result in epic_results.values():
            if epic_result.market_conditions_summary:
                conditions = epic_result.market_conditions_summary
                regimes.append(conditions.regime.value if hasattr(conditions.regime, 'value') else str(conditions.regime))
                volatilities.append(conditions.volatility_percentile)
                sessions.append(conditions.session.value if hasattr(conditions.session, 'value') else str(conditions.session))

        # Create regime distribution
        regime_dist = {}
        for regime in regimes:
            regime_dist[regime] = regime_dist.get(regime, 0) + 1

        # Create session distribution
        session_dist = {}
        for session in sessions:
            session_dist[session] = session_dist.get(session, 0) + 1

        summary['regime_analysis'] = {
            'distribution': regime_dist,
            'most_common_regime': max(regime_dist, key=regime_dist.get) if regime_dist else 'unknown'
        }

        summary['volatility_analysis'] = {
            'average_volatility_percentile': sum(volatilities) / len(volatilities) if volatilities else 0.5,
            'min_volatility': min(volatilities) if volatilities else 0.0,
            'max_volatility': max(volatilities) if volatilities else 1.0
        }

        summary['session_analysis'] = {
            'distribution': session_dist,
            'most_active_session': max(session_dist, key=session_dist.get) if session_dist else 'unknown'
        }

        return summary

    def _display_enhanced_signals(self, signals: List[StandardSignal]):
        """Display standardized signals with enhanced information"""
        self.logger.info(f"\n🎯 ENHANCED SIGNAL ANALYSIS:")
        self.logger.info("=" * 120)
        self.logger.info("#   TIMESTAMP            EPIC     TYPE CONF   PRICE    S/L      T/P      R:R    REGIME    SESSION")
        self.logger.info("-" * 120)

        display_signals = signals[:20]  # Show max 20 signals

        for i, signal in enumerate(display_signals, 1):
            epic_short = signal.epic.split('.')[2] if '.' in signal.epic else signal.epic[:8]
            signal_type = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)

            regime = 'N/A'
            session = 'N/A'
            if signal.market_conditions:
                regime = signal.market_conditions.regime.value if hasattr(signal.market_conditions.regime, 'value') else str(signal.market_conditions.regime)
                session = signal.market_conditions.session.value if hasattr(signal.market_conditions.session, 'value') else str(signal.market_conditions.session)

            row = (f"{i:<3} {signal.timestamp[:19]:<20} {epic_short:<8} {signal_type:<4} "
                   f"{signal.confidence:<6.1%} {signal.price:<8.5f} "
                   f"{signal.stop_loss or 0:<8.5f} {signal.take_profit or 0:<8.5f} "
                   f"{signal.risk_reward_ratio or 0:<6.2f} {regime[:8]:<9} {session[:7]}")

            self.logger.info(row)

        self.logger.info("=" * 120)

        if len(signals) > 20:
            self.logger.info(f"📝 Showing latest 20 of {len(signals)} total signals")
        else:
            self.logger.info(f"📝 Showing all {len(signals)} signals")

    def _display_enhanced_performance(self, result: StandardBacktestResult):
        """Display enhanced performance metrics"""
        self.logger.info(f"\n📈 ENHANCED {result.strategy_name.upper()} PERFORMANCE:")
        self.logger.info("=" * 60)

        # Overall metrics
        overall = result.overall_performance
        self.logger.info(f"📊 Overall Results:")
        self.logger.info(f"   Total Signals: {overall.get('total_signals', 0)}")
        self.logger.info(f"   Average Confidence: {overall.get('avg_confidence', 0):.1%}")
        self.logger.info(f"   Win Rate: {overall.get('win_rate', 0):.1%}")
        self.logger.info(f"   Risk/Reward Ratio: {overall.get('risk_reward_ratio', 0):.2f}")

        # Epic breakdown
        successful_epics = [epic for epic, result in result.epic_results.items() if not result.error_message]
        failed_epics = [epic for epic, result in result.epic_results.items() if result.error_message]

        self.logger.info(f"\n📈 Epic Results:")
        self.logger.info(f"   Successful: {len(successful_epics)}/{len(result.epic_results)}")
        self.logger.info(f"   Failed: {len(failed_epics)}")

        for epic, epic_result in result.epic_results.items():
            if not epic_result.error_message:
                signal_count = len(epic_result.signals)
                avg_conf = epic_result.performance_metrics.get('avg_confidence', 0)
                self.logger.info(f"     {epic}: {signal_count} signals (avg conf: {avg_conf:.1%})")
            else:
                self.logger.info(f"     {epic}: ❌ {epic_result.error_message}")

        # Market intelligence summary
        if result.market_intelligence_summary and result.market_intelligence_summary.get('market_conditions_detected'):
            self.logger.info(f"\n🧠 Market Intelligence:")
            regime_analysis = result.market_intelligence_summary.get('regime_analysis', {})
            volatility_analysis = result.market_intelligence_summary.get('volatility_analysis', {})

            if regime_analysis:
                most_common = regime_analysis.get('most_common_regime', 'unknown')
                self.logger.info(f"   Most Common Regime: {most_common}")

            if volatility_analysis:
                avg_vol = volatility_analysis.get('average_volatility_percentile', 0.5)
                self.logger.info(f"   Average Volatility: {avg_vol:.1%} percentile")

        # Execution metadata
        exec_meta = result.execution_metadata
        self.logger.info(f"\n⚡ Execution Summary:")
        self.logger.info(f"   Execution Time: {exec_meta.get('execution_time', 0):.2f}s")
        self.logger.info(f"   Database Optimization: {'✅' if exec_meta.get('database_optimization_used') else '❌'}")
        self.logger.info(f"   Market Intelligence: {'✅' if exec_meta.get('market_intelligence_used') else '❌'}")
        self.logger.info(f"   Smart Money Analysis: {'✅' if exec_meta.get('smart_money_used') else '❌'}")

    def _display_performance(self, metrics: Dict):
        """Legacy display method for backward compatibility"""
        self.logger.info(f"📈 {self.strategy_name} Performance:")
        self.logger.info(f"   Signals: {metrics.get('total_signals', 0)}")
        self.logger.info(f"   Avg Confidence: {metrics.get('avg_confidence', 0):.1%}")
        self.logger.info(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        self.logger.info(f"   Avg Profit: {metrics.get('avg_profit_pips', 0):.1f} pips")
