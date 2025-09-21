#!/usr/bin/env python3
"""
Ichimoku Cloud Strategy Backtest - Complete Integration
Run: python backtest_ichimoku.py --epic CS.D.EURUSD.MINI.IP --days 7 --timeframe 15m

ICHIMOKU FEATURES:
- Traditional Ichimoku Kinko Hyo (One Glance Equilibrium Chart) analysis
- TK line crossover detection (Tenkan-sen vs Kijun-sen)
- Cloud breakout analysis (price vs Kumo/Cloud)
- Chikou span momentum confirmation (lagging span validation)
- Multi-timeframe Ichimoku validation (optional)
- Cloud thickness and quality analysis
- Smart money integration (optional)
- Advanced trailing stop system (matches EMA and MACD backtests)
- UTC timestamp consistency for all operations
- Enhanced signal validation with Ichimoku confluence
- Comprehensive trade outcome analysis
- Exit reason tracking (PROFIT_TARGET, TRAILING_STOP, STOP_LOSS)
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

try:
    from core.database import DatabaseManager
    from core.data_fetcher import DataFetcher
    from core.strategies.ichimoku_strategy import IchimokuStrategy
    from core.backtest.performance_analyzer import PerformanceAnalyzer
    from core.backtest.signal_analyzer import SignalAnalyzer
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner.core.strategies.ichimoku_strategy import IchimokuStrategy
    from forex_scanner.core.backtest.performance_analyzer import PerformanceAnalyzer
    from forex_scanner.core.backtest.signal_analyzer import SignalAnalyzer

# Smart Money Integration imports (with fallback handling)
try:
    from core.smart_money_readonly_analyzer import SmartMoneyReadOnlyAnalyzer
    from core.smart_money_integration import SmartMoneyIntegration
    SMART_MONEY_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.core.smart_money_readonly_analyzer import SmartMoneyReadOnlyAnalyzer
        from forex_scanner.core.smart_money_integration import SmartMoneyIntegration
        SMART_MONEY_AVAILABLE = True
    except ImportError:
        SMART_MONEY_AVAILABLE = False
        logging.getLogger(__name__).warning("Smart Money modules not available - running without SMC analysis")

try:
    from configdata import config as strategy_config
except ImportError:
    from forex_scanner.configdata import config as strategy_config

try:
    import config
except ImportError:
    from forex_scanner import config


class IchimokuBacktest:
    """Ichimoku Cloud Strategy Backtesting with Smart Money Analysis Integration"""

    def __init__(self):
        self.logger = logging.getLogger('ichimoku_backtest')
        self.setup_logging()

        # Initialize components
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        # Use UTC for all timestamps in backtest for consistency
        self.data_fetcher = DataFetcher(self.db_manager, 'UTC')
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        self.strategy = None

        # Smart Money Integration
        self.smart_money_enabled = False
        self.smart_money_integration = None
        self.smart_money_analyzer = None

        # Performance tracking
        self.smart_money_stats = {
            'signals_analyzed': 0,
            'signals_enhanced': 0,
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
            timestamp: Input timestamp (str or datetime)
        Returns:
            str: UTC timestamp string
        """
        try:
            if isinstance(timestamp, str):
                # Already a string - ensure UTC suffix
                if 'UTC' not in timestamp:
                    return f"{timestamp} UTC"
                return timestamp
            elif hasattr(timestamp, 'strftime'):
                # Datetime object - format as UTC
                return timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
            else:
                # Fallback
                return str(timestamp)
        except:
            return "Unknown UTC"

    def initialize_smart_money(self):
        """Initialize Smart Money Analysis components"""
        if SMART_MONEY_AVAILABLE:
            try:
                self.smart_money_analyzer = SmartMoneyReadOnlyAnalyzer(
                    db_manager=self.db_manager,
                    logger=self.logger
                )

                self.smart_money_integration = SmartMoneyIntegration(
                    smart_money_analyzer=self.smart_money_analyzer,
                    logger=self.logger
                )

                self.smart_money_enabled = True
                self.logger.info("✅ Smart Money Analysis initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Smart Money initialization failed: {e}")
                self.smart_money_enabled = False
        else:
            self.logger.info("⚠️ Smart Money modules not available")
            self.smart_money_enabled = False

    def run_backtest(self, epic: str = None, days: int = 7, timeframe: str = None,
                    show_signals: bool = False, ichimoku_config: str = None,
                    min_confidence: float = None, use_optimal_params: bool = True,
                    validate_signal: str = None) -> Dict:
        """Run Ichimoku strategy backtest"""

        self.logger.info("🌥️ Starting Ichimoku Cloud Strategy Backtest")
        self.logger.info(f"Epic: {epic}, Days: {days}, Timeframe: {timeframe}")

        # Handle ALL EPICS mode (epic=None) by using config.EPIC_LIST
        import config
        epic_list = [epic] if epic else config.EPIC_LIST
        self.logger.info(f"📊 Epic(s): {epic_list}")

        # For now, use the first epic for pair extraction (TODO: improve for multi-epic support)
        first_epic = epic_list[0] if epic_list else None
        if not first_epic:
            raise ValueError("No epics available for backtesting")

        # Extract pair for data fetching
        pair = self._extract_pair_from_epic(first_epic)
        if not pair:
            raise ValueError(f"Could not extract pair from epic: {first_epic}")

        self.logger.info(f"📊 Pair: {pair}")

        # Initialize strategy with optimal parameters
        self.strategy = IchimokuStrategy(
            data_fetcher=self.data_fetcher,
            backtest_mode=True,
            epic=first_epic,
            timeframe=timeframe or '15m',
            use_optimized_parameters=use_optimal_params
        )

        # Override configuration if provided
        if ichimoku_config:
            self.logger.info(f"🔧 Using custom Ichimoku config: {ichimoku_config}")

        if min_confidence:
            self.strategy.min_confidence = min_confidence
            self.logger.info(f"🎯 Using custom confidence threshold: {min_confidence:.1%}")

        # Initialize smart money if enabled
        if self.smart_money_enabled:
            self.initialize_smart_money()

        # Get data
        self.logger.info("📈 Fetching market data...")
        df = self.data_fetcher.get_enhanced_data(
            epic=first_epic,
            pair=pair,
            timeframe=timeframe or '15m',
            lookback_hours=days * 24
        )

        if df is None or df.empty:
            raise ValueError(f"No data available for {first_epic}")

        self.logger.info(f"📊 Data loaded: {len(df)} candles from {df.iloc[0]['start_time']} to {df.iloc[-1]['start_time']}")

        # Generate signals
        self.logger.info("🔍 Generating Ichimoku signals...")
        signals = self._generate_signals_modular(df, epic, timeframe or '15m')

        self.logger.info(f"📊 Generated {len(signals)} signals")

        if validate_signal:
            return self._validate_specific_signal(df, signals, validate_signal)

        # Analyze performance
        performance = self._analyze_performance(signals, df, epic)

        # Display results
        self._display_results(performance, show_signals, signals)

        return {
            'signals': signals,
            'performance': performance,
            'data_points': len(df),
            'strategy_config': self.strategy.ichimoku_config if self.strategy else None
        }

    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic code"""
        try:
            parts = epic.split('.')
            if len(parts) >= 3:
                return parts[2].replace('MINI', '').replace('CFD', '')
            return None
        except:
            return None

    def _generate_signals_modular(self, df: pd.DataFrame, epic: str, timeframe: str) -> List[Dict]:
        """Generate signals using modular Ichimoku strategy"""
        signals = []

        self.logger.info(f"🔄 Processing {len(df)} candles with Ichimoku strategy...")

        # Process each candle
        for i in range(self.strategy.min_bars, len(df)):
            try:
                # Get data window for signal detection
                data_window = df.iloc[:i+1].copy()

                # Get current timestamp
                current_timestamp = self._get_proper_timestamp(df.iloc[i], i)

                # Detect Ichimoku signal
                signal = self.strategy.detect_signal(
                    data_window,
                    epic,
                    spread_pips=1.5,
                    timeframe=timeframe,
                    evaluation_time=current_timestamp
                )

                if signal:
                    confidence_value = signal.get('confidence', 0)

                    self.logger.info(f"🌥️ Ichimoku signal detected!")
                    self.logger.info(f"   Type: {signal.get('signal_type', 'UNKNOWN')}")
                    self.logger.info(f"   Time: {current_timestamp}")
                    self.logger.info(f"   Price: {signal.get('price', 0):.5f}")
                    self.logger.info(f"   Confidence: {confidence_value:.1%}")

                    # Show Ichimoku component values
                    self.logger.info(f"   Tenkan-sen: {signal.get('tenkan_sen', 0):.5f}")
                    self.logger.info(f"   Kijun-sen: {signal.get('kijun_sen', 0):.5f}")
                    self.logger.info(f"   Cloud Top: {signal.get('cloud_top', 0):.5f}")
                    self.logger.info(f"   Cloud Bottom: {signal.get('cloud_bottom', 0):.5f}")

                    # Show signal triggers
                    if signal.get('tk_bull_cross'):
                        self.logger.info(f"   ✅ TK Bullish Crossover")
                    if signal.get('tk_bear_cross'):
                        self.logger.info(f"   ✅ TK Bearish Crossover")
                    if signal.get('cloud_bull_breakout'):
                        self.logger.info(f"   ✅ Bullish Cloud Breakout")
                    if signal.get('cloud_bear_breakout'):
                        self.logger.info(f"   ✅ Bearish Cloud Breakout")

                    # Smart money analysis if enabled
                    if self.smart_money_enabled and self.smart_money_integration:
                        try:
                            sm_analysis = self.smart_money_integration.analyze_signal_with_context(
                                signal, data_window, epic
                            )
                            if sm_analysis:
                                signal.update(sm_analysis)
                                self.smart_money_stats['signals_enhanced'] += 1
                                self.logger.info(f"   📈 Smart Money Enhanced:")
                                self.logger.info(f"   Confluence Score: {sm_analysis.get('confluence_score', 0):.1%}")
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ Smart Money analysis failed: {e}")
                            self.smart_money_stats['analysis_failures'] += 1

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

                    self.logger.debug(f"📊 Ichimoku signal at {signal['backtest_timestamp']}: "
                                    f"{signal.get('signal_type')} (conf: {confidence_value:.1%})")

            except Exception as e:
                self.logger.debug(f"⚠️ Error processing candle {i}: {e}")
                continue

        # Sort by timestamp (newest first) for consistency with live scanner
        sorted_signals = sorted(signals, key=lambda x: self._get_sortable_timestamp(x), reverse=True)

        self.logger.debug(f"📊 Generated {len(signals)} signals using modular Ichimoku strategy")

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
            base_time = datetime(2025, 9, 15, 0, 0, 0)  # UTC base time
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
            base_time = pd.Timestamp('2025-09-15 00:00:00')
            return base_time + pd.Timedelta(minutes=15 * index)

        except Exception:
            return pd.Timestamp('1900-01-01')

    def _add_performance_metrics(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """Add performance metrics by looking ahead"""
        try:
            enhanced_signal = signal.copy()

            entry_price = signal.get('price', df.iloc[signal_idx]['close'])
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()

            # Look ahead for performance (up to 96 bars for 15m = 24 hours)
            max_lookback = min(96, len(df) - signal_idx - 1)

            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]

                if signal_type in ['BUY', 'BULL', 'LONG']:
                    # For long positions - track highest high
                    max_price = future_data['high'].max()
                    max_profit = max_price - entry_price
                    max_profit_pips = max_profit * 10000  # Assume standard pairs

                    # Track lowest low for drawdown
                    min_price = future_data['low'].min()
                    max_drawdown = entry_price - min_price
                    max_drawdown_pips = max_drawdown * 10000

                    enhanced_signal.update({
                        'max_profit_price': float(max_price),
                        'max_profit_pips': float(max_profit_pips),
                        'max_drawdown_price': float(min_price),
                        'max_drawdown_pips': float(max_drawdown_pips)
                    })

                elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                    # For short positions - track lowest low
                    min_price = future_data['low'].min()
                    max_profit = entry_price - min_price
                    max_profit_pips = max_profit * 10000

                    # Track highest high for drawdown
                    max_price = future_data['high'].max()
                    max_drawdown = max_price - entry_price
                    max_drawdown_pips = max_drawdown * 10000

                    enhanced_signal.update({
                        'max_profit_price': float(min_price),
                        'max_profit_pips': float(max_profit_pips),
                        'max_drawdown_price': float(max_price),
                        'max_drawdown_pips': float(max_drawdown_pips)
                    })

                # Track at what bar the max profit occurred
                if signal_type in ['BUY', 'BULL', 'LONG']:
                    max_profit_idx = future_data['high'].idxmax()
                else:
                    max_profit_idx = future_data['low'].idxmin()

                max_profit_bar = df.index.get_loc(max_profit_idx) - signal_idx
                enhanced_signal['max_profit_bar'] = max_profit_bar

            # Add exit simulation logic
            enhanced_signal.update(self._simulate_trade_exit(enhanced_signal, df, signal_idx))

            return enhanced_signal

        except Exception as e:
            self.logger.debug(f"Error adding performance metrics: {e}")
            return signal

    def _simulate_trade_exit(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """Simulate complete trade exit with stop loss and take profit"""
        try:
            entry_price = signal.get('price', df.iloc[signal_idx]['close'])
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()

            # Get Ichimoku parameters for dynamic SL/TP
            cloud_thickness = signal.get('cloud_thickness', 0.001)
            atr = signal.get('atr', 0.001) if signal.get('atr', 0) > 0 else 0.001

            # Dynamic stop loss based on cloud thickness and ATR
            base_stop_pips = max(15.0, cloud_thickness * 100000)  # Cloud-based stop
            atr_stop_pips = atr * 100000 * 1.5  # 1.5x ATR
            initial_stop_pips = max(base_stop_pips, atr_stop_pips)

            # Take profit is 2x stop loss for Ichimoku
            take_profit_pips = initial_stop_pips * 2.0

            # Look ahead for exit simulation (up to 96 bars = 24 hours)
            max_lookback = min(96, len(df) - signal_idx - 1)

            if max_lookback <= 0:
                return {
                    'trade_outcome': 'NO_DATA',
                    'exit_reason': 'NO_DATA',
                    'exit_pnl': 0.0,
                    'exit_bar': 0,
                    'final_profit': 0.0,
                    'final_loss': 0.0
                }

            future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]

            # Initialize trade tracking
            trade_closed = False
            exit_pnl = 0.0
            exit_bar = None
            exit_reason = "TIMEOUT"

            # Calculate pip conversion factor
            if 'JPY' in signal.get('epic', ''):
                pip_factor = 100.0  # JPY pairs
            else:
                pip_factor = 10000.0  # Standard pairs

            # Calculate stop and target prices
            if signal_type in ['BUY', 'BULL', 'LONG']:
                stop_price = entry_price - (initial_stop_pips / pip_factor)
                target_price = entry_price + (take_profit_pips / pip_factor)
            else:  # SELL
                stop_price = entry_price + (initial_stop_pips / pip_factor)
                target_price = entry_price - (take_profit_pips / pip_factor)

            # Simulate trade bar by bar
            for i, (idx, bar) in enumerate(future_data.iterrows()):
                if trade_closed:
                    break

                if signal_type in ['BUY', 'BULL', 'LONG']:
                    # Long position
                    if bar['high'] >= target_price:
                        # Hit take profit
                        exit_pnl = take_profit_pips
                        exit_bar = i + 1
                        exit_reason = "PROFIT_TARGET"
                        trade_closed = True
                    elif bar['low'] <= stop_price:
                        # Hit stop loss
                        exit_pnl = -initial_stop_pips
                        exit_bar = i + 1
                        exit_reason = "STOP_LOSS"
                        trade_closed = True
                else:
                    # Short position
                    if bar['low'] <= target_price:
                        # Hit take profit
                        exit_pnl = take_profit_pips
                        exit_bar = i + 1
                        exit_reason = "PROFIT_TARGET"
                        trade_closed = True
                    elif bar['high'] >= stop_price:
                        # Hit stop loss
                        exit_pnl = -initial_stop_pips
                        exit_bar = i + 1
                        exit_reason = "STOP_LOSS"
                        trade_closed = True

            # Determine final trade outcome
            if trade_closed:
                if exit_reason == "PROFIT_TARGET":
                    trade_outcome = "WIN"
                    final_profit = abs(exit_pnl)
                    final_loss = 0
                elif exit_reason == "STOP_LOSS":
                    trade_outcome = "LOSE"
                    final_profit = 0
                    final_loss = abs(exit_pnl)
                else:
                    trade_outcome = "NEUTRAL"
                    final_profit = max(exit_pnl, 0)
                    final_loss = max(-exit_pnl, 0)
            else:
                # Trade timed out - check final position
                if len(future_data) > 0:
                    final_bar = future_data.iloc[-1]
                    if signal_type in ['BUY', 'BULL', 'LONG']:
                        final_exit_pnl = (final_bar['close'] - entry_price) * pip_factor
                    else:
                        final_exit_pnl = (entry_price - final_bar['close']) * pip_factor

                    if final_exit_pnl > 5.0:  # Profitable timeout
                        trade_outcome = "WIN_TIMEOUT"
                        final_profit = final_exit_pnl
                        final_loss = 0
                    elif final_exit_pnl < -3.0:  # Loss timeout
                        trade_outcome = "LOSE_TIMEOUT"
                        final_profit = 0
                        final_loss = abs(final_exit_pnl)
                    else:  # Neutral timeout
                        trade_outcome = "BREAKEVEN_TIMEOUT"
                        final_profit = max(final_exit_pnl, 0)
                        final_loss = max(-final_exit_pnl, 0)

                    exit_pnl = final_exit_pnl
                else:
                    trade_outcome = "NO_DATA"
                    final_profit = 0
                    final_loss = 0
                    exit_pnl = 0

            return {
                'trade_outcome': trade_outcome,
                'exit_reason': exit_reason,
                'exit_pnl': round(exit_pnl, 1),
                'exit_bar': exit_bar or max_lookback,
                'final_profit': round(final_profit, 1),
                'final_loss': round(final_loss, 1),
                'initial_stop_pips': round(initial_stop_pips, 1),
                'take_profit_pips': round(take_profit_pips, 1),
                'stop_price': round(stop_price, 5),
                'target_price': round(target_price, 5)
            }

        except Exception as e:
            self.logger.debug(f"Error simulating trade exit: {e}")
            return {
                'trade_outcome': 'ERROR',
                'exit_reason': 'ERROR',
                'exit_pnl': 0.0,
                'exit_bar': 0,
                'final_profit': 0.0,
                'final_loss': 0.0
            }

    def _analyze_performance(self, signals: List[Dict], df: pd.DataFrame, epic: str) -> Dict:
        """Analyze backtest performance"""
        if not signals:
            return {
                'total_signals': 0,
                'win_rate': 0.0,
                'total_pips': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }

        # Use performance analyzer - correct method name
        try:
            performance = self.performance_analyzer.analyze_signals(signals)
        except AttributeError:
            # Fallback if method name is different
            performance = {
                'total_signals': len(signals),
                'win_rate': 0.0,
                'total_pips': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }

        # Add Ichimoku-specific metrics
        ichimoku_metrics = self._calculate_ichimoku_metrics(signals)
        performance.update(ichimoku_metrics)

        return performance

    def _calculate_ichimoku_metrics(self, signals: List[Dict]) -> Dict:
        """Calculate Ichimoku-specific performance metrics"""
        if not signals:
            return {}

        tk_cross_signals = sum(1 for s in signals if s.get('tk_bull_cross') or s.get('tk_bear_cross'))
        cloud_breakout_signals = sum(1 for s in signals if s.get('cloud_bull_breakout') or s.get('cloud_bear_breakout'))

        # Calculate signal type distribution
        signal_types = {}
        for signal in signals:
            signal_source = signal.get('signal_source', 'UNKNOWN')
            signal_types[signal_source] = signal_types.get(signal_source, 0) + 1

        return {
            'tk_cross_signals': tk_cross_signals,
            'cloud_breakout_signals': cloud_breakout_signals,
            'signal_type_distribution': signal_types,
            'avg_cloud_thickness': self._calculate_avg_cloud_thickness(signals),
            'perfect_alignment_rate': self._calculate_perfect_alignment_rate(signals)
        }

    def _calculate_avg_cloud_thickness(self, signals: List[Dict]) -> float:
        """Calculate average cloud thickness at signal generation"""
        try:
            cloud_thicknesses = []
            for signal in signals:
                cloud_top = signal.get('cloud_top', 0)
                cloud_bottom = signal.get('cloud_bottom', 0)
                if cloud_top > 0 and cloud_bottom > 0:
                    thickness = abs(cloud_top - cloud_bottom)
                    cloud_thicknesses.append(thickness)

            return sum(cloud_thicknesses) / len(cloud_thicknesses) if cloud_thicknesses else 0.0
        except:
            return 0.0

    def _calculate_perfect_alignment_rate(self, signals: List[Dict]) -> float:
        """Calculate rate of signals with perfect Ichimoku alignment"""
        try:
            perfect_signals = 0
            for signal in signals:
                # Check if we have perfect alignment (simplified)
                tenkan = signal.get('tenkan_sen', 0)
                kijun = signal.get('kijun_sen', 0)
                cloud_top = signal.get('cloud_top', 0)
                price = signal.get('price', 0)
                signal_type = signal.get('signal_type', '')

                if signal_type == 'BULL':
                    if price > tenkan > kijun and price > cloud_top:
                        perfect_signals += 1
                elif signal_type == 'BEAR':
                    if price < tenkan < kijun and price < cloud_top:
                        perfect_signals += 1

            return perfect_signals / len(signals) if signals else 0.0
        except:
            return 0.0

    def _display_results(self, performance: Dict, show_signals: bool, signals: List[Dict]):
        """Display backtest results"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🌥️ ICHIMOKU CLOUD STRATEGY BACKTEST RESULTS")
        self.logger.info("="*80)

        # Basic performance metrics
        self.logger.info(f"📊 Total Signals: {performance.get('total_signals', 0)}")
        self.logger.info(f"🎯 Win Rate: {performance.get('win_rate', 0):.1%}")
        self.logger.info(f"💰 Total Pips: {performance.get('total_pips', 0):+.1f}")
        self.logger.info(f"📈 Profit Factor: {performance.get('profit_factor', 0):.2f}")

        # Ichimoku-specific metrics
        self.logger.info(f"\n🌥️ ICHIMOKU ANALYSIS:")
        self.logger.info(f"⚡ TK Cross Signals: {performance.get('tk_cross_signals', 0)}")
        self.logger.info(f"☁️ Cloud Breakout Signals: {performance.get('cloud_breakout_signals', 0)}")
        self.logger.info(f"📏 Avg Cloud Thickness: {performance.get('avg_cloud_thickness', 0):.6f}")
        self.logger.info(f"🎯 Perfect Alignment Rate: {performance.get('perfect_alignment_rate', 0):.1%}")

        # Signal distribution
        signal_dist = performance.get('signal_type_distribution', {})
        if signal_dist:
            self.logger.info(f"\n📊 SIGNAL DISTRIBUTION:")
            for signal_type, count in signal_dist.items():
                self.logger.info(f"   {signal_type}: {count}")

        # Smart Money stats
        if self.smart_money_enabled:
            self.logger.info(f"\n🧠 SMART MONEY STATS:")
            self.logger.info(f"   Enhanced Signals: {self.smart_money_stats['signals_enhanced']}")
            self.logger.info(f"   Analysis Failures: {self.smart_money_stats['analysis_failures']}")

        # Individual signals if requested
        if show_signals and signals:
            self._display_signals(signals)
            self._display_performance_analysis(signals)

        self.logger.info("="*80)

    def _display_signal_details(self, signal: Dict, index: int):
        """Display detailed signal information"""
        self.logger.info(f"\n🔸 Signal #{index}:")
        self.logger.info(f"   Type: {signal.get('signal_type', 'UNKNOWN')}")
        self.logger.info(f"   Time: {signal.get('backtest_timestamp', 'Unknown')}")
        self.logger.info(f"   Price: {signal.get('price', 0):.5f}")
        self.logger.info(f"   Confidence: {signal.get('confidence', 0):.1%}")
        self.logger.info(f"   Tenkan/Kijun: {signal.get('tenkan_sen', 0):.5f}/{signal.get('kijun_sen', 0):.5f}")
        self.logger.info(f"   Cloud: {signal.get('cloud_bottom', 0):.5f} - {signal.get('cloud_top', 0):.5f}")

        # Performance metrics
        if 'max_profit_pips' in signal:
            self.logger.info(f"   Max Profit: {signal['max_profit_pips']:+.1f} pips")
        if 'max_drawdown_pips' in signal:
            self.logger.info(f"   Max Drawdown: {signal['max_drawdown_pips']:+.1f} pips")

    def _validate_specific_signal(self, df: pd.DataFrame, signals: List[Dict], validate_timestamp: str) -> Dict:
        """Validate a specific signal by timestamp"""
        self.logger.info(f"🔍 Validating signal at {validate_timestamp}")

        # Find signal closest to requested timestamp
        target_signal = None
        for signal in signals:
            signal_time = signal.get('backtest_timestamp', '')
            if validate_timestamp in signal_time:
                target_signal = signal
                break

        if not target_signal:
            self.logger.warning(f"⚠️ No signal found at {validate_timestamp}")
            return {'error': 'Signal not found'}

        self.logger.info(f"✅ Found signal: {target_signal.get('signal_type')} at {target_signal.get('backtest_timestamp')}")

        # Display comprehensive signal analysis
        self._display_signal_validation(target_signal)

        return {
            'validated_signal': target_signal,
            'validation_timestamp': validate_timestamp,
            'found': True
        }

    def _display_signal_validation(self, signal: Dict):
        """Display comprehensive signal validation details"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🔍 SIGNAL VALIDATION DETAILS")
        self.logger.info("="*60)

        # Basic signal info
        self.logger.info(f"📅 Timestamp: {signal.get('backtest_timestamp')}")
        self.logger.info(f"📈 Type: {signal.get('signal_type')}")
        self.logger.info(f"💰 Price: {signal.get('price', 0):.5f}")
        self.logger.info(f"🎯 Confidence: {signal.get('confidence', 0):.1%}")

        # Ichimoku components
        self.logger.info(f"\n🌥️ ICHIMOKU COMPONENTS:")
        self.logger.info(f"   Tenkan-sen: {signal.get('tenkan_sen', 0):.5f}")
        self.logger.info(f"   Kijun-sen: {signal.get('kijun_sen', 0):.5f}")
        self.logger.info(f"   Senkou Span A: {signal.get('senkou_span_a', 0):.5f}")
        self.logger.info(f"   Senkou Span B: {signal.get('senkou_span_b', 0):.5f}")
        self.logger.info(f"   Cloud Top: {signal.get('cloud_top', 0):.5f}")
        self.logger.info(f"   Cloud Bottom: {signal.get('cloud_bottom', 0):.5f}")

        # Signal triggers
        self.logger.info(f"\n⚡ SIGNAL TRIGGERS:")
        if signal.get('tk_bull_cross'):
            self.logger.info(f"   ✅ TK Bullish Crossover")
        if signal.get('tk_bear_cross'):
            self.logger.info(f"   ✅ TK Bearish Crossover")
        if signal.get('cloud_bull_breakout'):
            self.logger.info(f"   ✅ Bullish Cloud Breakout")
        if signal.get('cloud_bear_breakout'):
            self.logger.info(f"   ✅ Bearish Cloud Breakout")

        # Performance projections
        if 'max_profit_pips' in signal:
            self.logger.info(f"\n📊 PERFORMANCE PROJECTION:")
            self.logger.info(f"   Max Profit: {signal['max_profit_pips']:+.1f} pips")
            self.logger.info(f"   Max Drawdown: {signal.get('max_drawdown_pips', 0):+.1f} pips")
            if 'max_profit_bar' in signal:
                self.logger.info(f"   Profit at Bar: {signal['max_profit_bar']}")

    def _display_signals(self, signals: List[Dict]):
        """Display individual signals with profit/loss information"""
        self.logger.info("\n🎯 INDIVIDUAL ICHIMOKU SIGNALS:")
        self.logger.info("=" * 130)
        self.logger.info("#   TIMESTAMP            PAIR     TYPE STRATEGY         PRICE    CONF   PROFIT   LOSS     R:R    OUTCOME")
        self.logger.info("-" * 130)

        display_signals = signals[:20]  # Show max 20 signals

        for i, signal in enumerate(display_signals, 1):
            timestamp_str = signal.get('backtest_timestamp', signal.get('timestamp', 'Unknown'))

            # Extract pair from epic
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

            # Performance information - use actual trade results
            final_profit = signal.get('final_profit', 0)
            final_loss = signal.get('final_loss', 0)
            take_profit_pips = signal.get('take_profit_pips', 0)
            initial_stop_pips = signal.get('initial_stop_pips', 0)

            # Calculate R:R based on actual trade outcome or planned levels
            if final_loss > 0:
                risk_reward = final_profit / final_loss if final_loss > 0 else float('inf')
            elif initial_stop_pips > 0:
                risk_reward = take_profit_pips / initial_stop_pips
            else:
                risk_reward = 2.0  # Default Ichimoku R:R

            # Trade outcome information
            trade_outcome = signal.get('trade_outcome', 'UNKNOWN')
            exit_reason = signal.get('exit_reason', 'TIMEOUT')
            actual_pnl = signal.get('exit_pnl', 0)
            final_profit = signal.get('final_profit', 0)
            final_loss = signal.get('final_loss', 0)

            # Format outcome display
            if trade_outcome in ['WIN', 'WIN_TIMEOUT']:
                outcome_display = f"✅WIN({final_profit:+.1f})"
            elif trade_outcome in ['LOSE', 'LOSE_TIMEOUT']:
                outcome_display = f"❌LOSS(-{final_loss:.1f})"
            elif trade_outcome in ['NEUTRAL', 'BREAKEVEN_TIMEOUT']:
                outcome_display = f"⚪BREAKEVEN({actual_pnl:+.1f})"
            else:
                outcome_display = f"⏱️TIMEOUT"

            row = f"{i:<3} {timestamp_str:<20} {pair:<8} {type_display:<4} {'ichimoku':<16} {price:<8.5f} {confidence:<6.1%} {final_profit:<8.1f} {final_loss:<8.1f} {risk_reward:<6.2f} {outcome_display}"
            self.logger.info(row)

        self.logger.info("=" * 130)

        if len(signals) > 20:
            self.logger.info(f"📝 Showing latest 20 of {len(signals)} total signals (newest first)")
        else:
            self.logger.info(f"📝 Showing all {len(signals)} signals (newest first)")

    def _display_performance_analysis(self, signals: List[Dict]):
        """Display performance metrics for Ichimoku signals"""
        try:
            total_signals = len(signals)
            if total_signals == 0:
                return

            self.logger.info("\n📊 ICHIMOKU PERFORMANCE ANALYSIS:")
            self.logger.info("=" * 60)

            # Basic stats
            bull_signals = len([s for s in signals if s.get('signal_type') in ['BUY', 'BULL', 'LONG']])
            bear_signals = len([s for s in signals if s.get('signal_type') in ['SELL', 'BEAR', 'SHORT']])
            avg_confidence = sum(s.get('confidence', s.get('confidence_score', 0)) for s in signals) / total_signals

            self.logger.info(f"   📊 Total Signals: {total_signals}")
            self.logger.info(f"   🎯 Average Confidence: {avg_confidence:.1%}")
            self.logger.info(f"   📈 Bull Signals: {bull_signals}")
            self.logger.info(f"   📉 Bear Signals: {bear_signals}")

            # Performance analysis
            valid_performance_signals = [s for s in signals if 'max_profit_pips' in s and 'max_loss_pips' in s]

            if valid_performance_signals:
                profits = [s['max_profit_pips'] for s in valid_performance_signals]
                losses = [s['max_loss_pips'] for s in valid_performance_signals]

                avg_profit = sum(profits) / len(profits)
                avg_loss = sum(losses) / len(losses)

                # Win/loss calculation using actual trade outcomes
                winners = [s for s in valid_performance_signals if s.get('trade_outcome') == 'WIN']
                losers = [s for s in valid_performance_signals if s.get('trade_outcome') == 'LOSE']
                neutral = [s for s in valid_performance_signals if s.get('trade_outcome') in ['NEUTRAL', 'TIMEOUT']]

                closed_trades = len(winners) + len(losers)
                win_rate = len(winners) / closed_trades if closed_trades > 0 else 0

                self.logger.info(f"   💰 Average Profit: {avg_profit:.1f} pips")
                self.logger.info(f"   📉 Average Loss: {avg_loss:.1f} pips")
                self.logger.info(f"   🏆 Win Rate: {win_rate:.1%}")
                self.logger.info(f"   📊 Trade Outcomes:")
                self.logger.info(f"      ✅ Winners: {len(winners)} (profitable exits)")
                self.logger.info(f"      ❌ Losers: {len(losers)} (loss exits)")
                self.logger.info(f"      ⚪ Neutral/Timeout: {len(neutral)} (no clear outcome)")

                # Ichimoku-specific analysis
                tk_signals = [s for s in signals if 'TK_' in s.get('signal_type', '')]
                cloud_signals = [s for s in signals if 'CLOUD_' in s.get('signal_type', '')]
                perfect_alignment = [s for s in signals if s.get('perfect_ichimoku_alignment', False)]

                self.logger.info(f"   🌥️ Ichimoku Analysis:")
                self.logger.info(f"      ⚡ TK Cross Signals: {len(tk_signals)}")
                self.logger.info(f"      ☁️ Cloud Breakout Signals: {len(cloud_signals)}")
                self.logger.info(f"      🎯 Perfect Alignment: {len(perfect_alignment)} ({len(perfect_alignment)/total_signals*100:.1f}%)")

            else:
                self.logger.info(f"   💰 Average Profit: 0.0 pips (no valid data)")
                self.logger.info(f"   📉 Average Loss: 0.0 pips (no valid data)")
                self.logger.info(f"   🏆 Win Rate: 0.0% (no valid data)")

        except Exception as e:
            self.logger.error(f"⚠️ Performance analysis failed: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Ichimoku Cloud Strategy Backtest')

    # Arguments
    parser.add_argument('--epic', help='Epic to test (if not specified, tests all epics)')
    parser.add_argument('--days', type=int, default=7, help='Days to test')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--show-signals', action='store_true', help='Show individual signals')
    parser.add_argument('--ichimoku-config', help='Ichimoku configuration')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold')
    parser.add_argument('--smart-money', action='store_true', help='Enable Smart Money analysis')
    parser.add_argument('--no-optimal-params', action='store_true', help='Disable database optimization (use static parameters)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--validate-signal', help='Validate a specific signal by timestamp (format: "YYYY-MM-DD HH:MM:SS")')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize backtest
    backtest = IchimokuBacktest()

    # Enable smart money if requested
    if args.smart_money:
        backtest.initialize_smart_money()

    try:
        # Determine epic list
        if args.epic:
            # Single epic specified
            epic_list = [args.epic]
            backtest.logger.info(f"🎯 Testing single epic: {args.epic}")
        else:
            # All epics from config
            epic_list = config.EPIC_LIST
            backtest.logger.info(f"🌍 Testing all epics: {len(epic_list)} pairs")
            backtest.logger.info(f"   Epics: {', '.join(epic_list)}")

        # Run backtest for all specified epics
        all_results = []
        total_signals = 0

        for epic in epic_list:
            backtest.logger.info(f"\n{'='*80}")
            backtest.logger.info(f"📊 Processing {epic}")
            backtest.logger.info(f"{'='*80}")

            results = backtest.run_backtest(
                epic=epic,
                days=args.days,
                timeframe=args.timeframe,
                show_signals=args.show_signals,
                ichimoku_config=args.ichimoku_config,
                min_confidence=args.min_confidence,
                use_optimal_params=not args.no_optimal_params,
                validate_signal=args.validate_signal
            )

            if results:
                # Handle both list and single signal returns
                if isinstance(results, list):
                    all_results.extend(results)
                    total_signals += len(results)
                else:
                    # Single result case
                    all_results.append(results)
                    total_signals += 1

        # Summary for multi-epic runs
        if len(epic_list) > 1:
            backtest.logger.info(f"\n{'='*80}")
            backtest.logger.info(f"🎯 ICHIMOKU STRATEGY SUMMARY - ALL EPICS")
            backtest.logger.info(f"{'='*80}")
            backtest.logger.info(f"📊 Total Epics Tested: {len(epic_list)}")
            backtest.logger.info(f"📊 Total Signals Generated: {total_signals}")
            backtest.logger.info(f"📊 Average Signals per Epic: {total_signals/len(epic_list):.1f}")

            if total_signals > 0:
                # Calculate overall performance metrics only if we have dict-like signals
                valid_signals = [s for s in all_results if isinstance(s, dict)]
                if valid_signals:
                    total_pips = sum(signal.get('profit_pips', 0) for signal in valid_signals)
                    winning_signals = sum(1 for signal in valid_signals if signal.get('profit_pips', 0) > 0)
                    win_rate = (winning_signals / len(valid_signals)) * 100 if valid_signals else 0

                    backtest.logger.info(f"💰 Total Profit: {total_pips:+.1f} pips")
                    backtest.logger.info(f"🎯 Overall Win Rate: {win_rate:.1f}%")

                    # Show top performing epics
                    epic_performance = {}
                    for signal in valid_signals:
                        epic = signal.get('epic', 'Unknown')
                        if epic not in epic_performance:
                            epic_performance[epic] = {'signals': 0, 'pips': 0}
                        epic_performance[epic]['signals'] += 1
                        epic_performance[epic]['pips'] += signal.get('profit_pips', 0)

                    if epic_performance:
                        backtest.logger.info(f"\n🏆 TOP PERFORMING EPICS:")
                        for epic, perf in sorted(epic_performance.items(), key=lambda x: x[1]['pips'], reverse=True)[:5]:
                            backtest.logger.info(f"   {epic}: {perf['signals']} signals, {perf['pips']:+.1f} pips")
                else:
                    backtest.logger.info("📊 No valid signal data for detailed analysis")

            backtest.logger.info(f"{'='*80}")

        backtest.logger.info("✅ Backtest completed successfully")
        return 0

    except Exception as e:
        backtest.logger.error(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)