# core/backtest/unified_backtest_engine.py
"""
Unified Backtest Engine - Central orchestrator for all strategy backtesting

This engine consolidates functionality from all individual backtest files,
providing a single interface for running backtests on any strategy with
consistent output, parameter testing, and performance analysis.

Features:
- Multi-strategy support with dynamic discovery
- Parameter optimization and systematic testing
- Smart Money Concepts integration
- Database-driven optimal parameters
- Consistent performance analysis and reporting
- Individual signal validation
- Multi-timeframe analysis support
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import traceback

try:
    import sys
    sys.path.append('..')
    from core.database import DatabaseManager
    from core.data_fetcher import DataFetcher
    from performance_analyzer import PerformanceAnalyzer
    from signal_analyzer import SignalAnalyzer
    from core.signal_detector import SignalDetector
    import config
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner.backtests.performance_analyzer import PerformanceAnalyzer
    from forex_scanner.backtests.signal_analyzer import SignalAnalyzer
    from forex_scanner.core.signal_detector import SignalDetector
    from forex_scanner import config

# Smart Money Integration (optional)
try:
    from core.smart_money_readonly_analyzer import SmartMoneyReadOnlyAnalyzer
    from core.smart_money_integration import SmartMoneyIntegration
    SMART_MONEY_AVAILABLE = True
except ImportError:
    SMART_MONEY_AVAILABLE = False

# Optimization service (optional)
try:
    from optimization.optimal_parameter_service import OptimalParameterService
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.optimization.optimal_parameter_service import OptimalParameterService
        OPTIMIZATION_AVAILABLE = True
    except ImportError:
        OPTIMIZATION_AVAILABLE = False


class BacktestMode(Enum):
    """Backtest execution modes"""
    SINGLE_STRATEGY = "single"
    MULTI_STRATEGY = "multi"
    PARAMETER_SWEEP = "sweep"
    VALIDATION = "validation"
    COMPARISON = "comparison"


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    mode: BacktestMode
    strategies: List[str]
    epics: List[str]
    timeframes: List[str]
    days: int
    show_signals: bool = False
    use_optimal_parameters: bool = True
    enable_smart_money: bool = False
    parameter_ranges: Optional[Dict[str, Any]] = None
    export_format: Optional[str] = None
    output_file: Optional[str] = None
    max_signals_display: int = 20
    detailed_analysis: bool = False


@dataclass
class BacktestResult:
    """Result from a single backtest run"""
    strategy: str
    epic: str
    timeframe: str
    signals: List[Dict]
    performance: Dict
    execution_time: float
    parameters_used: Optional[Dict] = None
    smart_money_stats: Optional[Dict] = None
    error: Optional[str] = None


class UnifiedBacktestEngine:
    """
    Central engine for all strategy backtesting operations

    Consolidates functionality from individual backtest files into a unified system
    that provides consistent interfaces, parameter testing, and reporting.
    """

    def __init__(self):
        self.logger = logging.getLogger('unified_backtest')
        self.setup_logging()

        # Core components
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager, 'UTC')  # Use UTC for consistency
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        self.signal_detector = SignalDetector(self.db_manager, 'UTC')

        # Optimization service
        self.optimization_service = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.optimization_service = OptimalParameterService()
                self.logger.info("🎯 Database optimization service available")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize optimization service: {e}")

        # Smart Money components
        self.smart_money_integration = None
        self.smart_money_analyzer = None
        self.smart_money_enabled = False

        # Strategy registry (will be populated by strategy registry module)
        self.available_strategies = {}

        # Performance tracking
        self.session_stats = {
            'strategies_tested': 0,
            'total_signals_found': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'execution_time': 0.0
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def initialize_smart_money(self, enable: bool = False) -> bool:
        """Initialize Smart Money analysis components"""
        if not enable or not SMART_MONEY_AVAILABLE:
            self.smart_money_enabled = False
            return False

        try:
            self.smart_money_integration = SmartMoneyIntegration(
                self.db_manager,
                self.data_fetcher
            )
            self.smart_money_analyzer = SmartMoneyReadOnlyAnalyzer(self.data_fetcher)
            self.smart_money_enabled = True
            self.logger.info("🧠 Smart Money analysis initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Smart Money initialization failed: {e}")
            self.smart_money_enabled = False
            return False

    def register_strategy(self, name: str, strategy_class: Any, backtest_class: Any = None):
        """Register a strategy for backtesting"""
        self.available_strategies[name] = {
            'strategy_class': strategy_class,
            'backtest_class': backtest_class,
            'name': name
        }
        self.logger.debug(f"📋 Registered strategy: {name}")

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.available_strategies.keys())

    def run_backtest(self, config: BacktestConfig) -> List[BacktestResult]:
        """
        Main entry point for running backtests

        Args:
            config: Backtest configuration

        Returns:
            List of backtest results
        """
        start_time = datetime.now()

        self.logger.info("🧪 UNIFIED BACKTEST ENGINE")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Mode: {config.mode.value}")
        self.logger.info(f"🎯 Strategies: {config.strategies}")
        self.logger.info(f"📈 Epics: {config.epics}")
        self.logger.info(f"⏰ Timeframes: {config.timeframes}")
        self.logger.info(f"📅 Days: {config.days}")
        self.logger.info(f"🎯 Database optimization: {'✅' if config.use_optimal_parameters else '❌'}")
        self.logger.info(f"🧠 Smart Money: {'✅' if config.smart_money.enabled else '❌'}")

        # Initialize Smart Money if enabled
        if config.smart_money.enabled:
            self.initialize_smart_money(True)

        try:
            # Route to appropriate execution method based on mode
            if config.mode == BacktestMode.SINGLE_STRATEGY:
                results = self._run_single_strategy_backtest(config)
            elif config.mode == BacktestMode.MULTI_STRATEGY:
                results = self._run_multi_strategy_backtest(config)
            elif config.mode == BacktestMode.PARAMETER_SWEEP:
                results = self._run_parameter_sweep_backtest(config)
            elif config.mode == BacktestMode.VALIDATION:
                results = self._run_validation_backtest(config)
            elif config.mode == BacktestMode.COMPARISON:
                results = self._run_comparison_backtest(config)
            else:
                raise ValueError(f"Unknown backtest mode: {config.mode}")

            # Update session statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.session_stats['execution_time'] = execution_time
            self.session_stats['successful_runs'] += len([r for r in results if r.error is None])
            self.session_stats['failed_runs'] += len([r for r in results if r.error is not None])
            self.session_stats['total_signals_found'] += sum(len(r.signals) for r in results if r.signals)

            self.logger.info(f"\n✅ Backtest completed in {execution_time:.1f}s")
            self.logger.info(f"📊 Total results: {len(results)}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            traceback.print_exc()
            return []

    def _run_single_strategy_backtest(self, config: BacktestConfig) -> List[BacktestResult]:
        """Run backtest for a single strategy"""
        results = []
        strategy_name = config.strategies[0]

        self.logger.info(f"\n📈 Running single strategy backtest: {strategy_name}")

        for epic in config.epics:
            for timeframe in config.timeframes:
                result = self._execute_strategy_backtest(
                    strategy_name, epic, timeframe, config
                )
                results.append(result)

        return results

    def _run_multi_strategy_backtest(self, config: BacktestConfig) -> List[BacktestResult]:
        """Run backtest for multiple strategies"""
        results = []

        self.logger.info(f"\n🎯 Running multi-strategy backtest: {len(config.strategies)} strategies")

        for strategy_name in config.strategies:
            self.logger.info(f"\n📊 Testing strategy: {strategy_name}")

            for epic in config.epics:
                for timeframe in config.timeframes:
                    result = self._execute_strategy_backtest(
                        strategy_name, epic, timeframe, config
                    )
                    results.append(result)

        return results

    def _run_parameter_sweep_backtest(self, config: BacktestConfig) -> List[BacktestResult]:
        """Run parameter sweep backtest"""
        # This will be implemented with the parameter manager
        self.logger.warning("⚠️ Parameter sweep not yet implemented")
        return []

    def _run_validation_backtest(self, config: BacktestConfig) -> List[BacktestResult]:
        """Run validation backtest for specific signals"""
        # This will be implemented for signal validation
        self.logger.warning("⚠️ Validation mode not yet implemented")
        return []

    def _run_comparison_backtest(self, config: BacktestConfig) -> List[BacktestResult]:
        """Run comparison backtest"""
        return self._run_multi_strategy_backtest(config)

    def _execute_strategy_backtest(
        self,
        strategy_name: str,
        epic: str,
        timeframe: str,
        config: BacktestConfig
    ) -> BacktestResult:
        """
        Execute backtest for a single strategy/epic/timeframe combination

        This method consolidates the core backtest logic from individual files
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"   📊 {strategy_name} - {epic} ({timeframe})")

            # Get strategy instance (bypass registry for EMA to match old backtest exactly)
            if strategy_name == 'ema':
                from core.strategies.ema_strategy import EMAStrategy
                strategy = EMAStrategy(data_fetcher=self.data_fetcher, backtest_mode=True)
            else:
                strategy = self._get_strategy_instance(
                    strategy_name, epic, timeframe, config.use_optimal_parameters
                )

            if not strategy:
                return BacktestResult(
                    strategy=strategy_name,
                    epic=epic,
                    timeframe=timeframe,
                    signals=[],
                    performance={},
                    execution_time=0.0,
                    error=f"Failed to initialize strategy {strategy_name}"
                )

            # Extract pair from epic
            pair = self._extract_pair_from_epic(epic)

            # Get enhanced data
            df = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=config.days * 24,
                ema_strategy=strategy if hasattr(strategy, 'get_ema_periods') else None
            )

            if df is None or df.empty:
                return BacktestResult(
                    strategy=strategy_name,
                    epic=epic,
                    timeframe=timeframe,
                    signals=[],
                    performance={},
                    execution_time=0.0,
                    error="No data available"
                )

            # Run strategy-specific backtest
            signals = self._run_strategy_backtest(
                strategy, strategy_name, df, epic, timeframe, config
            )

            # Analyze performance
            performance = self.performance_analyzer.analyze_performance(signals)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"      ✅ {len(signals)} signals found in {execution_time:.1f}s")

            return BacktestResult(
                strategy=strategy_name,
                epic=epic,
                timeframe=timeframe,
                signals=signals,
                performance=performance,
                execution_time=execution_time,
                parameters_used=self._get_strategy_parameters(strategy),
                smart_money_stats=self._get_smart_money_stats() if self.smart_money_enabled else None
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Strategy backtest failed: {str(e)}"
            self.logger.error(f"      ❌ {error_msg}")

            return BacktestResult(
                strategy=strategy_name,
                epic=epic,
                timeframe=timeframe,
                signals=[],
                performance={},
                execution_time=execution_time,
                error=error_msg
            )

    def _get_strategy_instance(
        self,
        strategy_name: str,
        epic: str,
        timeframe: str,
        use_optimal_parameters: bool
    ):
        """Get an instance of the specified strategy"""
        try:
            # This is a simplified version - will be enhanced with strategy registry
            if strategy_name == 'ema':
                from core.strategies.ema_strategy import EMAStrategy
                # Use same initialization as old backtest_ema.py for compatibility
                return EMAStrategy(
                    data_fetcher=self.data_fetcher,
                    backtest_mode=True
                )
            elif strategy_name == 'macd':
                from core.strategies.macd_strategy import MACDStrategy
                return MACDStrategy(
                    data_fetcher=self.data_fetcher,
                    backtest_mode=True,
                    epic=epic,
                    timeframe=timeframe,
                    use_optimized_parameters=use_optimal_parameters
                )
            elif strategy_name == 'kama':
                from core.strategies.kama_strategy import KAMAStrategy
                return KAMAStrategy()
            elif strategy_name == 'combined':
                from core.strategies.combined_strategy import CombinedStrategy
                return CombinedStrategy()
            elif strategy_name == 'bb_supertrend':
                from core.strategies.bb_supertrend_strategy import BollingerSupertrendStrategy
                bb_config = getattr(config, 'DEFAULT_BB_SUPERTREND_CONFIG', 'default')
                return BollingerSupertrendStrategy(config_name=bb_config)
            else:
                self.logger.error(f"❌ Unknown strategy: {strategy_name}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {strategy_name} strategy: {e}")
            return None

    def _run_strategy_backtest(
        self,
        strategy,
        strategy_name: str,
        df: pd.DataFrame,
        epic: str,
        timeframe: str,
        config: BacktestConfig
    ) -> List[Dict]:
        """
        Run backtest for a specific strategy instance

        This consolidates the core backtest logic from individual files
        """
        signals = []
        min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)

        # Use EXACT same loop as working old backtest (from test_exact_old_method.py)
        for i in range(min_bars, len(df)):
            try:
                # Get data up to current point (simulate real-time)
                current_data = df.iloc[:i+1].copy()

                # Use EXACT same method as working old backtest
                if strategy.enable_mtf_analysis and strategy.mtf_analyzer:
                    signal = strategy.detect_signal_with_mtf(
                        current_data, epic, config.SPREAD_PIPS, timeframe
                    )
                else:
                    # Get current timestamp (old backtest method)
                    current_timestamp = df.index[i] if i < len(df) else df.index[-1]

                    signal = strategy.detect_signal(
                        current_data, epic, config.SPREAD_PIPS, timeframe,
                        evaluation_time=current_timestamp
                    )

                if signal:
                    # Apply Smart Money enhancement if enabled
                    if self.smart_money_enabled and self.smart_money_integration:
                        try:
                            enhanced_signal = self.smart_money_integration.enhance_signal_with_smart_money(
                                signal=signal,
                                epic=epic,
                                timeframe=timeframe
                            )
                            if enhanced_signal:
                                signal = enhanced_signal
                        except Exception as e:
                            self.logger.debug(f"Smart Money enhancement failed: {e}")

                    # Add backtest metadata
                    signal['backtest_timestamp'] = current_timestamp
                    signal['backtest_index'] = i
                    signal['strategy'] = strategy_name

                    # Add performance metrics
                    enhanced_signal = self._add_performance_metrics(signal, df, i)
                    signals.append(enhanced_signal)

            except Exception as e:
                self.logger.debug(f"⚠️ Error processing candle {i}: {e}")
                continue

        # Sort by timestamp (newest first)
        try:
            signals.sort(key=lambda x: self._get_sortable_timestamp(x), reverse=True)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not sort signals: {e}")

        return signals

    def _add_performance_metrics(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """
        Add performance metrics by looking ahead

        This consolidates the performance calculation logic from individual files
        """
        try:
            enhanced_signal = signal.copy()

            entry_price = signal.get('price', df.iloc[signal_idx]['close'])
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()

            # Look ahead for performance (up to 96 bars for 15m = 24 hours)
            max_lookback = min(96, len(df) - signal_idx - 1)

            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]

                # Trading parameters (configurable in future)
                target_pips = 15
                initial_stop_pips = 10
                breakeven_trigger = 8
                stop_to_profit_trigger = 15
                stop_to_profit_level = 10
                trailing_ratio = 0.5

                # Simulate trade with trailing stop logic
                trade_result = self._simulate_trade_with_trailing_stop(
                    signal_type, entry_price, future_data,
                    target_pips, initial_stop_pips, breakeven_trigger,
                    stop_to_profit_trigger, stop_to_profit_level, trailing_ratio
                )

                enhanced_signal.update(trade_result)
            else:
                enhanced_signal.update({
                    'max_profit_pips': 0.0,
                    'max_loss_pips': 0.0,
                    'is_winner': False,
                    'is_loser': False,
                    'trade_outcome': 'NO_DATA'
                })

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"❌ Error adding performance metrics: {e}")
            enhanced_signal = signal.copy()
            enhanced_signal.update({
                'max_profit_pips': 0.0,
                'max_loss_pips': 0.0,
                'is_winner': False,
                'is_loser': False,
                'trade_outcome': 'ERROR',
                'error': str(e)
            })
            return enhanced_signal

    def _simulate_trade_with_trailing_stop(
        self,
        signal_type: str,
        entry_price: float,
        future_data: pd.DataFrame,
        target_pips: float,
        initial_stop_pips: float,
        breakeven_trigger: float,
        stop_to_profit_trigger: float,
        stop_to_profit_level: float,
        trailing_ratio: float
    ) -> Dict:
        """Simulate trade execution with advanced trailing stop logic"""

        # Initialize trade tracking
        trade_closed = False
        exit_pnl = 0.0
        exit_bar = None
        exit_reason = "TIMEOUT"

        # Trailing stop state
        current_stop_pips = initial_stop_pips
        best_profit_pips = 0.0
        stop_moved_to_breakeven = False
        stop_moved_to_profit = False

        # Simulate trade bar by bar
        for bar_idx, (_, bar) in enumerate(future_data.iterrows()):
            if trade_closed:
                break

            high_price = bar['high']
            low_price = bar['low']

            if signal_type in ['BUY', 'BULL', 'LONG']:
                # Long trade logic
                current_profit_pips = (high_price - entry_price) * 10000
                current_loss_pips = (entry_price - low_price) * 10000

                # Update best profit and trailing stop
                if current_profit_pips > best_profit_pips:
                    best_profit_pips = current_profit_pips
                    current_stop_pips = self._update_trailing_stop(
                        best_profit_pips, current_stop_pips,
                        breakeven_trigger, stop_to_profit_trigger,
                        stop_to_profit_level, trailing_ratio,
                        stop_moved_to_breakeven, stop_moved_to_profit
                    )

                # Check exit conditions
                trade_closed, exit_pnl, exit_reason = self._check_exit_conditions(
                    current_profit_pips, current_loss_pips, current_stop_pips,
                    target_pips, bar_idx
                )

            elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                # Short trade logic
                current_profit_pips = (entry_price - low_price) * 10000
                current_loss_pips = (high_price - entry_price) * 10000

                # Update best profit and trailing stop
                if current_profit_pips > best_profit_pips:
                    best_profit_pips = current_profit_pips
                    current_stop_pips = self._update_trailing_stop(
                        best_profit_pips, current_stop_pips,
                        breakeven_trigger, stop_to_profit_trigger,
                        stop_to_profit_level, trailing_ratio,
                        stop_moved_to_breakeven, stop_moved_to_profit
                    )

                # Check exit conditions
                trade_closed, exit_pnl, exit_reason = self._check_exit_conditions(
                    current_profit_pips, current_loss_pips, current_stop_pips,
                    target_pips, bar_idx
                )

        # Determine final outcome
        if trade_closed:
            if exit_reason == "PROFIT_TARGET":
                trade_outcome = "WIN"
                is_winner = True
                is_loser = False
                final_profit = exit_pnl
                final_loss = 0
            elif exit_pnl > 0:
                trade_outcome = "WIN"
                is_winner = True
                is_loser = False
                final_profit = exit_pnl
                final_loss = 0
            else:
                trade_outcome = "LOSE"
                is_winner = False
                is_loser = True
                final_profit = 0
                final_loss = abs(exit_pnl)
        else:
            # Handle timeout case
            if len(future_data) > 0:
                final_price = future_data.iloc[-1]['close']
                if signal_type in ['BUY', 'BULL', 'LONG']:
                    final_exit_pnl = (final_price - entry_price) * 10000
                else:
                    final_exit_pnl = (entry_price - final_price) * 10000

                if final_exit_pnl > 5.0:
                    trade_outcome = "WIN_TIMEOUT"
                    is_winner = True
                    is_loser = False
                    final_profit = round(final_exit_pnl, 1)
                    final_loss = 0
                elif final_exit_pnl < -3.0:
                    trade_outcome = "LOSE_TIMEOUT"
                    is_winner = False
                    is_loser = True
                    final_profit = 0
                    final_loss = round(abs(final_exit_pnl), 1)
                else:
                    trade_outcome = "BREAKEVEN_TIMEOUT"
                    is_winner = False
                    is_loser = False
                    final_profit = max(final_exit_pnl, 0)
                    final_loss = max(-final_exit_pnl, 0)
            else:
                trade_outcome = "NO_DATA"
                is_winner = False
                is_loser = False
                final_profit = 0
                final_loss = 0

        return {
            'max_profit_pips': round(final_profit, 1),
            'max_loss_pips': round(final_loss, 1),
            'profit_loss_ratio': round(final_profit / final_loss, 2) if final_loss > 0 else float('inf'),
            'entry_price': entry_price,
            'is_winner': is_winner,
            'is_loser': is_loser,
            'trade_outcome': trade_outcome,
            'exit_reason': exit_reason,
            'exit_bar': exit_bar,
            'exit_pnl': exit_pnl,
            'target_pips': target_pips,
            'initial_stop_pips': initial_stop_pips,
            'trailing_stop_used': stop_moved_to_profit or stop_moved_to_breakeven,
            'best_profit_achieved': best_profit_pips
        }

    def _update_trailing_stop(
        self, best_profit: float, current_stop: float,
        breakeven_trigger: float, profit_trigger: float, profit_level: float,
        trailing_ratio: float, moved_to_breakeven: bool, moved_to_profit: bool
    ) -> float:
        """Update trailing stop based on profit progression"""

        if best_profit >= breakeven_trigger and not moved_to_breakeven:
            return 0  # Move to breakeven
        elif best_profit >= profit_trigger and not moved_to_profit:
            return -profit_level  # Move to profit protection
        elif best_profit > profit_trigger and moved_to_profit:
            excess_profit = best_profit - profit_trigger
            trailing_adjustment = excess_profit * trailing_ratio
            return -(profit_level + trailing_adjustment)

        return current_stop

    def _check_exit_conditions(
        self, profit_pips: float, loss_pips: float, stop_pips: float,
        target_pips: float, bar_idx: int
    ) -> Tuple[bool, float, str]:
        """Check if trade should be closed"""

        # Check profit target
        if profit_pips >= target_pips:
            return True, target_pips, "PROFIT_TARGET"

        # Check stop loss conditions
        if stop_pips > 0:  # Traditional stop loss
            if loss_pips >= stop_pips:
                return True, -stop_pips, "STOP_LOSS"
        else:  # Profit protection stop
            profit_protection_level = abs(stop_pips)
            if profit_pips <= profit_protection_level or loss_pips > 0:
                exit_pnl = profit_protection_level if profit_pips >= profit_protection_level else -loss_pips
                return True, exit_pnl, "TRAILING_STOP"

        return False, 0.0, "CONTINUE"

    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic"""
        try:
            # Handle both CEEM (EURUSD only) and MINI (all other pairs) formats
            if '.D.' in epic:
                parts = epic.split('.D.')
                if len(parts) > 1:
                    # Check for CEEM format (EURUSD only)
                    if '.CEEM.IP' in epic:
                        pair_part = parts[1].split('.CEEM.IP')[0]
                        return pair_part
                    # Check for MINI format (all other pairs)
                    elif '.MINI.IP' in epic:
                        pair_part = parts[1].split('.MINI.IP')[0]
                        return pair_part

            # Fallback to config
            pair_info = getattr(config, 'PAIR_INFO', {})
            if epic in pair_info:
                return pair_info[epic].get('pair', 'EURUSD')

            self.logger.warning(f"⚠️ Could not extract pair from {epic}, using EURUSD")
            return 'EURUSD'

        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting pair from {epic}: {e}, using EURUSD")
            return 'EURUSD'

    def _get_proper_timestamp(self, df_row, row_index: int) -> str:
        """Get proper timestamp from data row"""
        try:
            # Try different timestamp sources
            for col in ['datetime_utc', 'start_time', 'timestamp', 'datetime']:
                if col in df_row and df_row[col] is not None:
                    candidate = df_row[col]
                    if isinstance(candidate, str) and candidate != 'Unknown':
                        if 'UTC' not in candidate:
                            return f"{candidate} UTC"
                        return candidate
                    elif hasattr(candidate, 'strftime'):
                        return candidate.strftime('%Y-%m-%d %H:%M:%S UTC')

            # Fallback
            base_time = datetime(2025, 8, 3, 0, 0, 0)
            estimated_time = base_time + timedelta(minutes=15 * row_index)
            return estimated_time.strftime('%Y-%m-%d %H:%M:%S UTC')

        except Exception:
            fallback_time = datetime.utcnow() - timedelta(minutes=15 * (1000 - row_index))
            return fallback_time.strftime('%Y-%m-%d %H:%M:%S UTC')

    def _get_sortable_timestamp(self, signal: Dict) -> pd.Timestamp:
        """Get timestamp for sorting"""
        try:
            timestamp_str = signal.get('backtest_timestamp', signal.get('timestamp', ''))
            if timestamp_str and timestamp_str != 'Unknown':
                return pd.to_datetime(timestamp_str)

            # Fallback using index
            index = signal.get('backtest_index', 0)
            base_time = pd.Timestamp('2025-08-04 00:00:00')
            return base_time + pd.Timedelta(minutes=15 * index)

        except Exception:
            return pd.Timestamp('1900-01-01')

    def _get_strategy_parameters(self, strategy) -> Optional[Dict]:
        """Extract parameters from strategy instance"""
        try:
            if hasattr(strategy, 'get_parameters'):
                return strategy.get_parameters()
            elif hasattr(strategy, 'config'):
                return strategy.config
            else:
                return None
        except Exception:
            return None

    def _get_smart_money_stats(self) -> Optional[Dict]:
        """Get Smart Money analysis statistics"""
        if not self.smart_money_enabled:
            return None

        try:
            # This would be enhanced with actual stats from Smart Money integration
            return {
                'enabled': True,
                'signals_enhanced': 0,
                'analysis_time': 0.0
            }
        except Exception:
            return None

    def get_session_statistics(self) -> Dict:
        """Get statistics for the current session"""
        return self.session_stats.copy()

    def validate_signal(
        self,
        strategy_name: str,
        epic: str,
        timestamp: str,
        timeframe: str = '15m',
        show_calculations: bool = True,
        show_raw_data: bool = False
    ) -> Dict:
        """
        Validate a specific signal with detailed analysis

        This consolidates the signal validation functionality from individual files
        """
        self.logger.info("🔍 SIGNAL VALIDATION")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Strategy: {strategy_name}")
        self.logger.info(f"📊 Epic: {epic}")
        self.logger.info(f"⏰ Timestamp: {timestamp}")
        self.logger.info(f"📈 Timeframe: {timeframe}")

        try:
            # Get strategy instance
            strategy = self._get_strategy_instance(strategy_name, epic, timeframe, True)
            if not strategy:
                return {'error': f'Failed to initialize strategy {strategy_name}'}

            # Extract pair and get data
            pair = self._extract_pair_from_epic(epic)

            # Parse target timestamp
            target_time = pd.to_datetime(timestamp)
            if target_time.tz is not None:
                target_time = target_time.tz_localize(None)

            # Get extended data around the target timestamp
            df = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=120  # 5 days
            )

            if df is None or df.empty:
                return {'error': 'No data available'}

            # Find closest data point
            timestamp_cols = ['datetime_utc', 'start_time', 'timestamp', 'datetime']
            df_with_time = None

            for col in timestamp_cols:
                if col in df.columns:
                    try:
                        df['datetime_parsed'] = pd.to_datetime(df[col], errors='coerce')
                        if df['datetime_parsed'].dt.tz is not None:
                            df['datetime_parsed'] = df['datetime_parsed'].dt.tz_localize(None)

                        df_with_time = df.dropna(subset=['datetime_parsed'])
                        if not df_with_time.empty:
                            break
                    except Exception:
                        continue

            if df_with_time is None or df_with_time.empty:
                return {'error': 'No valid timestamps found in data'}

            # Find closest timestamp
            time_diffs = abs(df_with_time['datetime_parsed'] - target_time)
            closest_idx = time_diffs.idxmin()
            closest_time = df_with_time.loc[closest_idx, 'datetime_parsed']

            # Get context data
            data_idx = df_with_time.index.get_loc(closest_idx)
            min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)

            context_start = max(0, data_idx - min_bars)
            context_end = min(len(df), data_idx + 10)

            validation_data = df.iloc[context_start:context_end + 1].copy()
            signal_row_idx = data_idx - context_start

            # Attempt signal detection
            signal_data_slice = validation_data.iloc[:signal_row_idx + 1].copy()

            detected_signal = None
            if hasattr(strategy, 'detect_signal'):
                try:
                    detected_signal = strategy.detect_signal(
                        signal_data_slice, epic, config.SPREAD_PIPS, timeframe
                    )
                except Exception as e:
                    self.logger.error(f"Signal detection failed: {e}")

            return {
                'strategy': strategy_name,
                'epic': epic,
                'timestamp': timestamp,
                'closest_time': str(closest_time),
                'time_difference': str(abs(closest_time - target_time)),
                'data_points': len(validation_data),
                'signal_detected': detected_signal is not None,
                'signal_details': detected_signal if detected_signal else None,
                'validation_successful': True
            }

        except Exception as e:
            error_msg = f"Signal validation failed: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}