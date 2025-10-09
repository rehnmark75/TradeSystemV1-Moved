#!/usr/bin/env python3
"""
Advanced Momentum Strategy Backtest with Minimal Lag Analysis
============================================================
Run: python backtest_momentum.py --epic CS.D.EURUSD.CEEM.IP --days 7 --timeframe 15m

FEATURES:
- Momentum oscillator with minimal lag (inspired by AlgoAlpha AI Momentum Predictor)
- Velocity-based momentum confirmation (inspired by Zeiierman Quantitative Oscillator)
- Volume-weighted momentum analysis (inspired by BigBeluga Whale Movement Tracker)
- Multi-timeframe validation capabilities
- UTC timestamp consistency for all operations
- Advanced trailing stop system with breakeven and profit protection
- Comprehensive trade outcome analysis with exit reason tracking
- Database optimization support for dynamic parameters
"""

import sys
import os
import argparse
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

sys.path.insert(0, project_root)

from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from core.strategies.momentum_strategy import MomentumStrategy
from core.backtest.performance_analyzer import PerformanceAnalyzer
from core.backtest.signal_analyzer import SignalAnalyzer

from configdata import config as strategy_config
try:
    import config
except ImportError:
    from forex_scanner import config


class MomentumBacktest:
    """Advanced Momentum Strategy Backtesting with TradingView-Inspired Indicators"""

    def __init__(self):
        self.logger = logging.getLogger('momentum_backtest')
        self.setup_logging()

        # Initialize components
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        # Use UTC for all timestamps in backtest for consistency
        self.data_fetcher = DataFetcher(self.db_manager, 'UTC')
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        self.strategy = None

        # Performance tracking
        self.momentum_stats = {
            'signals_analyzed': 0,
            'signals_generated': 0,
            'analysis_failures': 0,
            'avg_analysis_time': 0
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def _ensure_utc_timestamp(self, timestamp):
        """
        Ensure timestamp is in UTC format (no timezone conversion)
        All timestamps will remain in UTC for consistency

        Args:
            timestamp: Timestamp in various formats

        Returns:
            Formatted UTC timestamp string
        """
        try:
            # If it's already a string
            if isinstance(timestamp, str):
                # Ensure UTC suffix if not present
                if 'UTC' not in timestamp:
                    return f"{timestamp} UTC"
                return timestamp

            # If it's a pandas Timestamp or datetime, format it
            if hasattr(timestamp, 'strftime'):
                return timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')

            # Fallback
            return f"{str(timestamp)} UTC"

        except Exception as e:
            self.logger.debug(f"Timestamp conversion error: {e}")
            return f"{str(timestamp)} UTC"

    def initialize_momentum_strategy(self, momentum_config: str = None, epic: str = None, use_optimal_parameters: bool = True):
        """Initialize momentum strategy with database optimization support"""

        # Check if we should use optimal parameters from database
        use_optimal = use_optimal_parameters and epic is not None

        if use_optimal:
            try:
                from optimization.optimal_parameter_service import OptimalParameterService
                param_service = OptimalParameterService()
                optimal_params = param_service.get_epic_parameters(epic, strategy='momentum')

                if optimal_params:
                    self.logger.info(f"🎯 Using DATABASE OPTIMAL parameters for {epic}:")
                    self.logger.info(f"   Momentum Config: {optimal_params.momentum_config}")
                    self.logger.info(f"   Confidence: {optimal_params.confidence_threshold:.1%}")
                    self.logger.info(f"   SL/TP: {optimal_params.stop_loss_pips:.0f}/{optimal_params.take_profit_pips:.0f} pips")
                    self.logger.info(f"   Performance Score: {optimal_params.performance_score:.3f}")

                    # Initialize strategy with optimal parameters enabled
                    self.strategy = MomentumStrategy(
                        data_fetcher=self.data_fetcher,
                        backtest_mode=True,
                        use_optimal_parameters=True
                    )

                    # Set the epic so strategy can load its optimal parameters
                    if hasattr(self.strategy, '_epic'):
                        self.strategy._epic = epic

                    self.logger.info("✅ Momentum Strategy initialized with DATABASE OPTIMIZATION")
                    return None

            except Exception as e:
                self.logger.warning(f"❌ Failed to load optimal parameters for {epic}: {e}")
                self.logger.warning("   Falling back to static configuration...")

        # Fallback to static configuration
        original_config = None
        if momentum_config and hasattr(strategy_config, 'ACTIVE_MOMENTUM_CONFIG'):
            original_config = strategy_config.ACTIVE_MOMENTUM_CONFIG
            self.logger.info(f"🔧 Using static momentum config: {momentum_config}")

        # Initialize with static configuration
        self.strategy = MomentumStrategy(data_fetcher=self.data_fetcher, backtest_mode=True)
        self.logger.info("✅ Momentum Strategy initialized with STATIC CONFIGURATION")
        self.logger.info("🔥 BACKTEST MODE ENABLED: Time-based cooldowns disabled for historical analysis")

        # Get momentum configuration details for display
        momentum_configs = getattr(strategy_config, 'MOMENTUM_STRATEGY_CONFIG', {})
        active_config = momentum_config or getattr(strategy_config, 'ACTIVE_MOMENTUM_CONFIG', 'default')

        if active_config in momentum_configs:
            periods = momentum_configs[active_config]
            self.logger.info(f"   📊 Momentum Periods: fast={periods.get('fast_period')}, slow={periods.get('slow_period')}, signal={periods.get('signal_period')}")
            self.logger.info(f"   📊 Features: velocity={periods.get('velocity_period')}, volume={periods.get('volume_period')}")

        return original_config

    def run_live_scanner_simulation(
            self,
            epic: str = None,
            days: int = 1,
            timeframe: str = None,
            show_signals: bool = True
        ) -> bool:
            """
            Simulate the exact live scanner behavior using the REAL signal detection path
            """

            timeframe = timeframe or getattr(config, 'DEFAULT_TIMEFRAME', '15m')
            epic_list = [epic] if epic else config.EPIC_LIST

            self.logger.info("🔄 LIVE SCANNER SIMULATION MODE")
            self.logger.info("=" * 50)
            self.logger.info(f"📊 Epic(s): {epic_list}")
            self.logger.info(f"⏰ Timeframe: {timeframe}")
            self.logger.info(f"📅 Days: {days}")

            try:
                # CRITICAL FIX: Use the EXACT same signal detection path as live scanner
                from core.signal_detector import SignalDetector
                signal_detector = SignalDetector(self.db_manager, 'UTC')

                self.logger.info("✅ Using real SignalDetector (matches live scanner exactly)")

                all_signals = []

                for current_epic in epic_list:
                    self.logger.info(f"\n📈 Simulating live scanner for {current_epic}")

                    # Get pair info (same as live scanner)
                    pair_info = config.PAIR_INFO.get(current_epic, {'pair': 'EURUSD', 'pip_multiplier': 10000})
                    pair_name = pair_info['pair']

                    self.logger.info(f"   📊 Pair: {pair_name}")
                    self.logger.info(f"   📊 Timeframe: {timeframe}")
                    self.logger.info(f"   📊 BID adjustment: {config.USE_BID_ADJUSTMENT}")
                    self.logger.info(f"   📊 Spread pips: {config.SPREAD_PIPS}")

                    # CRITICAL FIX: Use EXACT live scanner detection method
                    if config.USE_BID_ADJUSTMENT:
                        signal = signal_detector.detect_signals_bid_adjusted(
                            current_epic, pair_name, config.SPREAD_PIPS, timeframe
                        )
                    else:
                        signal = signal_detector.detect_signals_mid_prices(
                            current_epic, pair_name, timeframe
                        )

                    if signal:
                        self.logger.info(f"   🎯 Live scanner simulation found signal!")
                        self.logger.info(f"      Type: {signal.get('signal_type')}")
                        self.logger.info(f"      Strategy: {signal.get('strategy')}")
                        self.logger.info(f"      Confidence: {signal.get('confidence', signal.get('confidence_score', 0)):.1%}")
                        self.logger.info(f"      Price: {signal.get('price', 0):.5f}")

                        signal['simulation_mode'] = 'live_scanner'
                        # Use UTC for consistency
                        signal['detected_at'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

                        all_signals.append(signal)
                    else:
                        self.logger.info(f"   ❌ Live scanner simulation found no signals")

                # Display results
                if all_signals:
                    self.logger.info(f"\n✅ LIVE SCANNER SIMULATION RESULTS:")
                    self.logger.info(f"   📊 Total signals: {len(all_signals)}")

                    if show_signals:
                        self._display_simulation_signals(all_signals)

                    return True
                else:
                    self.logger.warning("❌ Live scanner simulation found no signals")
                    return False

            except Exception as e:
                self.logger.error(f"❌ Live scanner simulation failed: {e}")
                import traceback
                traceback.print_exc()
                return False

    def run_backtest(
        self,
        epic: str = None,
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        momentum_config: str = None,
        min_confidence: float = None,
        use_optimal_parameters: bool = True
    ) -> bool:
        """Run advanced momentum strategy backtest with database optimization support"""

        epic_list = [epic] if epic else config.EPIC_LIST

        self.logger.info("🚀 ADVANCED MOMENTUM STRATEGY BACKTEST")
        self.logger.info("=" * 50)
        self.logger.info(f"📊 Epic(s): {epic_list}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"📅 Days: {days}")
        self.logger.info(f"🎯 Show signals: {show_signals}")
        self.logger.info(f"🎯 Database optimization: {'✅ ENABLED' if use_optimal_parameters else '❌ DISABLED'}")
        self.logger.info("🔥 Minimal Lag Momentum Analysis with TradingView-Inspired Indicators")

        if min_confidence:
            original_min_conf = getattr(config, 'MIN_CONFIDENCE', 0.7)
            config.MIN_CONFIDENCE = min_confidence
            self.logger.info(f"🎚️ Min confidence: {min_confidence:.1%} (was {original_min_conf:.1%})")

        try:
            all_signals = []
            epic_results = {}

            for current_epic in epic_list:
                self.logger.info(f"\n📈 Processing {current_epic}")

                # Initialize strategy per epic for optimal parameters
                original_config = self.initialize_momentum_strategy(
                    momentum_config=momentum_config,
                    epic=current_epic,
                    use_optimal_parameters=use_optimal_parameters
                )

                # Extract pair from epic
                pair = self._extract_pair_from_epic(current_epic)

                # CRITICAL FIX: Use same data fetching as live scanner
                df = self.data_fetcher.get_enhanced_data(
                    epic=current_epic,
                    pair=pair,
                    timeframe=timeframe,
                    lookback_hours=days * 24
                )

                if df is None:
                    self.logger.warning(f"❌ Failed to fetch data for {current_epic}")
                    epic_results[current_epic] = {'signals': 0, 'error': 'Data fetch failed'}
                    continue

                if df.empty:
                    self.logger.warning(f"❌ No data available for {current_epic}")
                    epic_results[current_epic] = {'signals': 0, 'error': 'No data'}
                    continue

                self.logger.info(f"   📊 Data points: {len(df)}")

                # Show data range info
                if len(df) > 0:
                    first_row = df.iloc[0]
                    last_row = df.iloc[-1]

                    start_time = self._get_proper_timestamp(first_row, 0)
                    end_time = self._get_proper_timestamp(last_row, len(df)-1)

                    self.logger.info(f"   📅 Data range: {start_time} to {end_time}")

                # Run momentum backtest using the strategy method
                signals = self._run_momentum_backtest(df, current_epic, timeframe)

                all_signals.extend(signals)
                epic_results[current_epic] = {'signals': len(signals)}

                self.logger.info(f"   🎯 Momentum signals found: {len(signals)}")

            # Display results
            self._display_epic_results(epic_results)

            if all_signals:
                self.logger.info(f"\n✅ TOTAL MOMENTUM SIGNALS: {len(all_signals)}")

                if show_signals:
                    self._display_signals(all_signals)

                self._analyze_performance(all_signals)

                if original_config:
                    strategy_config.ACTIVE_MOMENTUM_CONFIG = original_config

                self.log_performance_summary()
                return True
            else:
                self.logger.warning("❌ No momentum signals found in backtest period")
                return False

        except Exception as e:
            self.logger.error(f"❌ Momentum backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _run_momentum_backtest(self, df: pd.DataFrame, epic: str, timeframe: str) -> List[Dict]:
        """Run momentum backtest using the actual strategy detect_signal method"""
        signals = []

        # Use same minimum bars as live scanner
        min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)

        for i in range(min_bars, len(df)):
            try:
                # Get data up to current point (simulate real-time)
                current_data = df.iloc[:i+1].copy()

                # Get the current market timestamp for this iteration
                current_timestamp = self._get_proper_timestamp(df.iloc[i], i)

                # Use momentum strategy detection
                signal = self.strategy.detect_signal(
                    current_data, epic, config.SPREAD_PIPS, timeframe,
                    evaluation_time=current_timestamp
                )

                if signal:
                    # Standardize confidence for display
                    confidence_value = signal.get('confidence', signal.get('confidence_score', 0))
                    if confidence_value is not None:
                        signal['confidence'] = confidence_value
                        signal['confidence_score'] = confidence_value

                    # Log the signal detection with market timestamp
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"🎯 MOMENTUM SIGNAL DETECTED at market time: {current_timestamp}")
                    self.logger.info(f"   Type: {signal.get('signal_type', 'Unknown')}")
                    self.logger.info(f"   Confidence: {signal.get('confidence', 0):.1%}")
                    self.logger.info(f"   Price: {signal.get('price', 0):.5f}")
                    self.logger.info(f"   Fast Momentum: {signal.get('momentum_fast', 0):.6f}")
                    self.logger.info(f"   Slow Momentum: {signal.get('momentum_slow', 0):.6f}")
                    self.logger.info(f"   Velocity: {signal.get('velocity_momentum', 0):.6f}")
                    self.logger.info(f"   Trigger: {signal.get('trigger_reason', 'momentum_crossover')}")
                    self.logger.info(f"{'='*60}")

                    # Add backtest metadata
                    timestamp_value = current_timestamp
                    signal['backtest_timestamp'] = timestamp_value
                    signal['backtest_index'] = i
                    signal['candle_data'] = {
                        'open': float(df.iloc[i]['open']),
                        'high': float(df.iloc[i]['high']),
                        'low': float(df.iloc[i]['low']),
                        'close': float(df.iloc[i]['close']),
                        'timestamp': timestamp_value
                    }

                    # Set multiple timestamp fields for compatibility
                    signal['timestamp'] = timestamp_value
                    signal['market_timestamp'] = timestamp_value

                    # Add performance metrics
                    enhanced_signal = self._add_performance_metrics(signal, df, i)
                    signals.append(enhanced_signal)

                    self.logger.debug(f"📊 Momentum signal at {signal['backtest_timestamp']}: "
                                    f"{signal.get('signal_type')} (conf: {confidence_value:.1%})")

            except Exception as e:
                self.logger.debug(f"⚠️ Error processing candle {i}: {e}")
                continue

        # Sort by timestamp (newest first) for consistency with live scanner
        sorted_signals = sorted(signals, key=lambda x: self._get_sortable_timestamp(x), reverse=True)

        self.logger.debug(f"📊 Generated {len(signals)} momentum signals")

        return sorted_signals

    def _get_proper_timestamp(self, df_row, row_index: int) -> str:
        """Get proper timestamp from data row (ensures UTC)"""
        try:
            # Try different timestamp sources
            for col in ['datetime_utc', 'start_time', 'timestamp', 'datetime']:
                if col in df_row and df_row[col] is not None:
                    candidate = df_row[col]
                    if isinstance(candidate, str) and candidate != 'Unknown':
                        # Ensure UTC suffix if not present
                        if 'UTC' not in candidate:
                            return f"{candidate} UTC"
                        return candidate
                    elif hasattr(candidate, 'strftime'):
                        return candidate.strftime('%Y-%m-%d %H:%M:%S UTC')

            # Try index if available
            if hasattr(df_row, 'name') and df_row.name is not None:
                if hasattr(df_row.name, 'strftime'):
                    return df_row.name.strftime('%Y-%m-%d %H:%M:%S UTC')

            # Fallback - use UTC time
            base_time = datetime(2025, 8, 3, 0, 0, 0)  # UTC base time
            estimated_time = base_time + timedelta(minutes=15 * row_index)
            return estimated_time.strftime('%Y-%m-%d %H:%M:%S UTC')

        except Exception:
            # Use UTC for fallback
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

    def _add_performance_metrics(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """Add performance metrics by looking ahead with trailing stop simulation"""
        try:
            enhanced_signal = signal.copy()

            entry_price = signal.get('price', df.iloc[signal_idx]['close'])
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()

            # Look ahead for performance (up to 96 bars for 15m = 24 hours)
            max_lookback = min(96, len(df) - signal_idx - 1)

            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]

                # Trailing Stop Configuration (matches other strategies)
                target_pips = 15  # Profit target
                initial_stop_pips = 10  # Initial stop loss
                breakeven_trigger = 8    # Move to breakeven at 8 pips profit
                stop_to_profit_trigger = 15  # Move stop to 10 pips profit when trade hits 15 pips
                stop_to_profit_level = 10    # Stop level when above trigger
                trailing_start = 15      # Start trailing after this profit level
                trailing_ratio = 0.5     # Trail 1 pip for every 2 pips profit (1:2 ratio)

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
                        # Long trade: profit on price going up, loss on price going down
                        current_profit_pips = (high_price - entry_price) * 10000
                        current_loss_pips = (entry_price - low_price) * 10000

                        # Update best profit achieved
                        if current_profit_pips > best_profit_pips:
                            best_profit_pips = current_profit_pips

                            # TRAILING STOP LOGIC
                            # 1. Move to breakeven at 8 pips profit
                            if best_profit_pips >= breakeven_trigger and not stop_moved_to_breakeven:
                                current_stop_pips = 0  # Breakeven (no loss)
                                stop_moved_to_breakeven = True

                            # 2. Move stop to 10 pips profit when trade hits 15 pips
                            elif best_profit_pips >= stop_to_profit_trigger and not stop_moved_to_profit:
                                current_stop_pips = -stop_to_profit_level  # Negative = profit protection
                                stop_moved_to_profit = True

                            # 3. Start trailing: 1 pip for every 2 pips above 15 pip level
                            elif best_profit_pips > trailing_start and stop_moved_to_profit:
                                excess_profit = best_profit_pips - trailing_start
                                trailing_adjustment = excess_profit * trailing_ratio
                                current_stop_pips = -(stop_to_profit_level + trailing_adjustment)

                        # Check exit conditions
                        if current_stop_pips > 0:  # Traditional stop loss
                            if current_loss_pips >= current_stop_pips:
                                exit_pnl = -current_stop_pips
                                exit_reason = "STOP_LOSS"
                                trade_closed = True
                                exit_bar = bar_idx
                        else:  # Profit protection stop
                            profit_protection_level = abs(current_stop_pips)
                            if current_profit_pips <= profit_protection_level or current_loss_pips > 0:
                                exit_pnl = profit_protection_level if current_profit_pips >= profit_protection_level else -current_loss_pips
                                exit_reason = "TRAILING_STOP"
                                trade_closed = True
                                exit_bar = bar_idx

                        # Check profit target
                        if current_profit_pips >= target_pips:
                            exit_pnl = target_pips
                            exit_reason = "PROFIT_TARGET"
                            trade_closed = True
                            exit_bar = bar_idx

                    elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                        # Short trade logic (similar structure but inverted)
                        current_profit_pips = (entry_price - low_price) * 10000
                        current_loss_pips = (high_price - entry_price) * 10000

                        # Update best profit achieved and apply same trailing logic
                        if current_profit_pips > best_profit_pips:
                            best_profit_pips = current_profit_pips

                            if best_profit_pips >= breakeven_trigger and not stop_moved_to_breakeven:
                                current_stop_pips = 0
                                stop_moved_to_breakeven = True
                            elif best_profit_pips >= stop_to_profit_trigger and not stop_moved_to_profit:
                                current_stop_pips = -stop_to_profit_level
                                stop_moved_to_profit = True
                            elif best_profit_pips > trailing_start and stop_moved_to_profit:
                                excess_profit = best_profit_pips - trailing_start
                                trailing_adjustment = excess_profit * trailing_ratio
                                current_stop_pips = -(stop_to_profit_level + trailing_adjustment)

                        # Check exit conditions
                        if current_stop_pips > 0:
                            if current_loss_pips >= current_stop_pips:
                                exit_pnl = -current_stop_pips
                                exit_reason = "STOP_LOSS"
                                trade_closed = True
                                exit_bar = bar_idx
                        else:
                            profit_protection_level = abs(current_stop_pips)
                            if current_profit_pips <= profit_protection_level or current_loss_pips > 0:
                                exit_pnl = profit_protection_level if current_profit_pips >= profit_protection_level else -current_loss_pips
                                exit_reason = "TRAILING_STOP"
                                trade_closed = True
                                exit_bar = bar_idx

                        # Check profit target
                        if current_profit_pips >= target_pips:
                            exit_pnl = target_pips
                            exit_reason = "PROFIT_TARGET"
                            trade_closed = True
                            exit_bar = bar_idx

                # Determine final trade outcome
                if trade_closed:
                    if exit_reason == "PROFIT_TARGET":
                        trade_outcome = "WIN"
                        is_winner = True
                        is_loser = False
                        final_profit = exit_pnl
                        final_loss = 0
                    elif exit_reason in ["STOP_LOSS", "TRAILING_STOP"]:
                        if exit_pnl > 0:
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
                    # Trade timeout - use realistic exit at current market price
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

                enhanced_signal.update({
                    'max_profit_pips': round(final_profit, 1),
                    'max_loss_pips': round(final_loss, 1),
                    'profit_loss_ratio': round(final_profit / final_loss, 2) if final_loss > 0 else float('inf'),
                    'lookback_bars': max_lookback,
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
                    'best_profit_achieved': best_profit_pips,
                })
            else:
                enhanced_signal.update({
                    'max_profit_pips': 0.0,
                    'max_loss_pips': 0.0,
                    'is_winner': False,
                    'is_loser': False,
                    'trade_outcome': 'NO_DATA',
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

    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic"""
        try:
            if '.D.' in epic and '.MINI.IP' in epic:
                parts = epic.split('.D.')
                if len(parts) > 1:
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

    def _display_epic_results(self, epic_results: Dict):
        """Display results by epic"""
        self.logger.info("\n📊 RESULTS BY EPIC:")
        self.logger.info("-" * 30)

        for epic, result in epic_results.items():
            if 'error' in result:
                self.logger.info(f"   {epic}: ❌ {result['error']}")
            else:
                self.logger.info(f"   {epic}: {result['signals']} signals")

    def _display_signals(self, signals: List[Dict]):
        """Display individual signals"""
        self.logger.info("\n🎯 INDIVIDUAL MOMENTUM SIGNALS:")
        self.logger.info("=" * 130)
        self.logger.info("#   TIMESTAMP            PAIR     TYPE STRATEGY        PRICE    CONF   PROFIT   LOSS     R:R    FAST     SLOW")
        self.logger.info("-" * 130)

        display_signals = signals[:20]  # Show max 20 signals

        for i, signal in enumerate(display_signals, 1):
            timestamp_str = signal.get('backtest_timestamp', signal.get('timestamp', 'Unknown'))

            epic = signal.get('epic', 'Unknown')
            if 'CS.D.' in epic and '.MINI.IP' in epic:
                pair = epic.split('.D.')[1].split('.MINI.IP')[0]
            else:
                pair = epic[-6:] if len(epic) >= 6 else epic

            signal_type = signal.get('signal_type', 'UNK')
            if signal_type in ['BUY', 'BULL', 'LONG']:
                type_display = 'BUY'
            elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                type_display = 'SELL'
            else:
                type_display = signal_type or 'UNK'

            confidence = signal.get('confidence', signal.get('confidence_score', 0))
            if confidence > 1:
                confidence = confidence / 100.0

            price = signal.get('price', 0)
            max_profit = signal.get('max_profit_pips', 0)
            max_loss = signal.get('max_loss_pips', 0)
            risk_reward = signal.get('profit_loss_ratio', max_profit / max_loss if max_loss > 0 else float('inf'))

            # Momentum-specific data
            fast_momentum = signal.get('momentum_fast', 0)
            slow_momentum = signal.get('momentum_slow', 0)

            row = f"{i:<3} {timestamp_str:<20} {pair:<8} {type_display:<4} {'momentum':<15} {price:<8.5f} {confidence:<6.1%} {max_profit:<8.1f} {max_loss:<8.1f} {risk_reward:<6.2f} {fast_momentum:<8.6f} {slow_momentum:<8.6f}"
            self.logger.info(row)

        self.logger.info("=" * 130)

        if len(signals) > 20:
            self.logger.info(f"📝 Showing latest 20 of {len(signals)} total signals (newest first)")
        else:
            self.logger.info(f"📝 Showing all {len(signals)} signals (newest first)")

    def _analyze_performance(self, signals: List[Dict]):
        """Analyze performance metrics"""
        try:
            total_signals = len(signals)

            # Signal type categorization
            bull_signals = sum(1 for s in signals if s.get('signal_type', '').upper() in ['BUY', 'BULL', 'LONG'])
            bear_signals = sum(1 for s in signals if s.get('signal_type', '').upper() in ['SELL', 'BEAR', 'SHORT'])

            # Confidence analysis
            confidences = []
            for s in signals:
                conf = s.get('confidence', s.get('confidence_score', 0))
                if conf is not None:
                    if conf > 1:
                        conf = conf / 100.0
                    confidences.append(conf)

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Performance metrics
            profit_signals = [s for s in signals if 'max_profit_pips' in s and 'max_loss_pips' in s]
            valid_performance_signals = [s for s in profit_signals if s.get('max_profit_pips', 0) > 0 or s.get('max_loss_pips', 0) > 0]

            self.logger.info("\n📈 MOMENTUM STRATEGY PERFORMANCE:")
            self.logger.info("=" * 60)
            self.logger.info(f"   📊 Total Signals: {total_signals}")
            self.logger.info(f"   🎯 Average Confidence: {avg_confidence:.1%}")
            self.logger.info(f"   📈 Bull Signals: {bull_signals}")
            self.logger.info(f"   📉 Bear Signals: {bear_signals}")

            if valid_performance_signals:
                profits = [s['max_profit_pips'] for s in valid_performance_signals]
                losses = [s['max_loss_pips'] for s in valid_performance_signals]

                avg_profit = sum(profits) / len(profits)
                avg_loss = sum(losses) / len(losses)

                # Win/loss calculation using actual trade outcomes
                winners = [s for s in valid_performance_signals if s.get('trade_outcome') == 'WIN']
                losers = [s for s in valid_performance_signals if s.get('trade_outcome') == 'LOSE']
                neutral = [s for s in valid_performance_signals if s.get('trade_outcome') in ['NEUTRAL', 'TIMEOUT']]

                # Categorize by exit reason
                profit_target_exits = [s for s in winners if s.get('exit_reason') == 'PROFIT_TARGET']
                trailing_stop_exits = [s for s in winners + losers if s.get('exit_reason') == 'TRAILING_STOP']
                stop_loss_exits = [s for s in losers if s.get('exit_reason') == 'STOP_LOSS']

                closed_trades = len(winners) + len(losers)
                win_rate = len(winners) / closed_trades if closed_trades > 0 else 0

                self.logger.info(f"   💰 Average Profit: {avg_profit:.1f} pips")
                self.logger.info(f"   📉 Average Loss: {avg_loss:.1f} pips")
                self.logger.info(f"   🏆 Win Rate: {win_rate:.1%}")
                self.logger.info(f"   📊 Trade Outcomes:")
                self.logger.info(f"      ✅ Winners: {len(winners)} (profitable exits)")
                self.logger.info(f"      ❌ Losers: {len(losers)} (loss exits)")
                self.logger.info(f"      ⚪ Neutral/Timeout: {len(neutral)} (no clear outcome)")
                self.logger.info(f"   🎯 Exit Breakdown:")
                self.logger.info(f"      🏁 Profit Target: {len(profit_target_exits)} trades")
                self.logger.info(f"      📈 Trailing Stop: {len(trailing_stop_exits)} trades")
                self.logger.info(f"      🛑 Stop Loss: {len(stop_loss_exits)} trades")

                # Show trailing stop effectiveness
                trailing_wins = [s for s in trailing_stop_exits if s.get('trade_outcome') == 'WIN']
                if trailing_stop_exits:
                    trailing_effectiveness = len(trailing_wins) / len(trailing_stop_exits)
                    self.logger.info(f"   🔄 Trailing Stop Effectiveness: {trailing_effectiveness:.1%} ({len(trailing_wins)}/{len(trailing_stop_exits)})")

                # Momentum-specific analysis
                self.logger.info(f"\n🚀 MOMENTUM-SPECIFIC ANALYSIS:")
                momentum_signals = [s for s in signals if 'momentum_fast' in s and 'momentum_slow' in s]
                if momentum_signals:
                    avg_fast_momentum = sum(abs(s.get('momentum_fast', 0)) for s in momentum_signals) / len(momentum_signals)
                    avg_slow_momentum = sum(abs(s.get('momentum_slow', 0)) for s in momentum_signals) / len(momentum_signals)
                    avg_divergence = sum(abs(s.get('momentum_divergence', 0)) for s in momentum_signals) / len(momentum_signals)

                    self.logger.info(f"   📊 Average Fast Momentum: {avg_fast_momentum:.6f}")
                    self.logger.info(f"   📊 Average Slow Momentum: {avg_slow_momentum:.6f}")
                    self.logger.info(f"   📊 Average Momentum Divergence: {avg_divergence:.6f}")

                    # Velocity analysis
                    velocity_signals = [s for s in signals if 'velocity_momentum' in s]
                    if velocity_signals:
                        avg_velocity = sum(abs(s.get('velocity_momentum', 0)) for s in velocity_signals) / len(velocity_signals)
                        self.logger.info(f"   🚀 Average Velocity Momentum: {avg_velocity:.6f}")

            else:
                self.logger.info(f"   💰 Average Profit: 0.0 pips (no valid data)")
                self.logger.info(f"   📉 Average Loss: 0.0 pips (no valid data)")
                self.logger.info(f"   🏆 Win Rate: 0.0% (no valid data)")

        except Exception as e:
            self.logger.error(f"⚠️ Performance analysis failed: {e}")

    def _display_simulation_signals(self, signals: List[Dict]):
        """Display live scanner simulation signals"""
        self.logger.info("\n🔄 LIVE SCANNER SIMULATION SIGNALS:")
        self.logger.info("=" * 80)
        self.logger.info("EPIC                    TYPE  CONF   PRICE     TIMESTAMP")
        self.logger.info("-" * 80)

        for signal in signals:
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence', signal.get('confidence_score', 0))
            price = signal.get('price', 0)
            timestamp = signal.get('timestamp', 'Unknown')

            self.logger.info(f"{epic:<24} {signal_type:<4} {confidence:<6.1%} {price:<9.5f} {timestamp}")

        self.logger.info("=" * 80)

    def log_performance_summary(self):
        """Log momentum strategy performance summary"""
        if hasattr(self, 'momentum_stats') and self.momentum_stats.get('signals_analyzed', 0) > 0:
            stats = self.momentum_stats
            self.logger.info("\n🚀 MOMENTUM STRATEGY PERFORMANCE SUMMARY:")
            self.logger.info("=" * 60)
            self.logger.info(f"   📊 Signals analyzed: {stats.get('signals_analyzed', 0)}")
            self.logger.info(f"   ✅ Signals generated: {stats.get('signals_generated', 0)}")
            self.logger.info(f"   ❌ Analysis failures: {stats.get('analysis_failures', 0)}")
            analyzed = stats.get('signals_analyzed', 0)
            generated = stats.get('signals_generated', 0)
            if analyzed > 0:
                success_rate = generated / analyzed
                self.logger.info(f"   🎯 Success rate: {success_rate:.1%}")
            avg_time = stats.get('avg_analysis_time', 0.0)
            if avg_time > 0:
                self.logger.info(f"   ⏱️  Avg analysis time: {avg_time:.2f}s")
            self.logger.info("=" * 60)


def main():
    """Main execution with enhanced implementation"""
    parser = argparse.ArgumentParser(description='Advanced Momentum Strategy Backtest with TradingView-Inspired Indicators')

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--simulate-live', action='store_true',
                           help='Simulate live scanner behavior using real signal detector')
    mode_group.add_argument('--backtest', action='store_true', default=True,
                           help='Run backtest using momentum strategy (default)')

    # Arguments
    parser.add_argument('--epic', help='Epic to test')
    parser.add_argument('--days', type=int, default=7, help='Days to test')
    parser.add_argument('--timeframe', help='Timeframe (default: from config)')
    parser.add_argument('--show-signals', action='store_true', help='Show individual signals')
    parser.add_argument('--momentum-config', help='Momentum configuration')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold')
    parser.add_argument('--no-optimal-params', action='store_true', help='Disable database optimization (use static parameters)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    backtest = MomentumBacktest()

    if args.simulate_live:
        success = backtest.run_live_scanner_simulation(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=True
        )
    else:
        success = backtest.run_backtest(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=args.show_signals,
            momentum_config=args.momentum_config,
            min_confidence=args.min_confidence,
            use_optimal_parameters=not args.no_optimal_params
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()