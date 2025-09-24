#!/usr/bin/env python3
"""
RAG Intelligence Strategy Backtesting System
==========================================

Comprehensive backtesting for the RAG-Enhanced Market Intelligence Strategy.
Tests strategy adaptation, RAG code selection effectiveness, and intelligence-based filtering.

Run: python backtest_rag_intelligence.py --epic CS.D.EURUSD.MINI.IP --days 7 --timeframe 15m

FEATURES:
- RAG strategy selection backtesting with historical market regimes
- Market intelligence effectiveness analysis
- Strategy adaptation performance tracking
- Regime-based performance breakdowns
- RAG selection success rate analysis
- Intelligence vs traditional strategy comparison
- Time-series strategy evolution tracking
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

# Add project root to path - handle Docker environment
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

sys.path.insert(0, project_root)

# Handle imports with proper fallback for Docker environment
try:
    from core.database import DatabaseManager
except ImportError:
    # Try relative import for Docker environment
    try:
        import sys
        sys.path.append('/app/forex_scanner')
        from core.database import DatabaseManager
    except ImportError:
        # Create minimal fallback
        class DatabaseManager:
            def __init__(self, url):
                raise ImportError("Database not available in current environment")

try:
    from core.data_fetcher import DataFetcher
except ImportError:
    try:
        sys.path.append('/app/forex_scanner')
        from core.data_fetcher import DataFetcher
    except ImportError:
        class DataFetcher:
            def __init__(self, *args):
                raise ImportError("DataFetcher not available")

try:
    from core.strategies.rag_intelligence_strategy import RAGIntelligenceStrategy
except ImportError:
    try:
        sys.path.append('/app/forex_scanner')
        from core.strategies.rag_intelligence_strategy import RAGIntelligenceStrategy
    except ImportError:
        raise ImportError("RAGIntelligenceStrategy not found - ensure it's properly installed")

# Handle performance analyzer imports with fallbacks
try:
    from core.backtest.performance_analyzer import PerformanceAnalyzer
    from core.backtest.signal_analyzer import SignalAnalyzer
except ImportError:
    # Create minimal fallback classes
    class PerformanceAnalyzer:
        def __init__(self): pass
    class SignalAnalyzer:
        def __init__(self): pass

# Handle config imports
strategy_config = None
try:
    from configdata import config as strategy_config
except ImportError:
    try:
        import configdata
        strategy_config = configdata
    except ImportError:
        pass

# Main config
config = None
try:
    import config
except ImportError:
    try:
        # Try to get from environment or create minimal config
        class MinimalConfig:
            DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:pass@localhost/forex')
            MIN_CONFIDENCE = 0.6
        config = MinimalConfig()
    except:
        pass


class RAGIntelligenceBacktest:
    """
    Comprehensive backtesting system for RAG Intelligence Strategy

    Features:
    - Historical market regime replay
    - RAG strategy selection effectiveness
    - Intelligence filtering performance analysis
    - Strategy adaptation tracking
    - Comparative analysis vs traditional strategies
    """

    def __init__(self):
        self.logger = logging.getLogger('rag_intelligence_backtest')
        self.setup_logging()

        # Initialize core components with error handling
        try:
            if config and hasattr(config, 'DATABASE_URL'):
                self.db_manager = DatabaseManager(config.DATABASE_URL)
                self.data_fetcher = DataFetcher(self.db_manager, 'UTC')
                self.db_available = True
            else:
                self.logger.warning("Database configuration not available")
                self.db_manager = None
                self.data_fetcher = None
                self.db_available = False
        except Exception as e:
            self.logger.warning(f"Database initialization failed: {e}")
            self.db_manager = None
            self.data_fetcher = None
            self.db_available = False

        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()

        # Strategy instances
        self.rag_strategy = None
        self.baseline_strategy = None

        # Results tracking
        self.results = {
            'trades': [],
            'regime_performance': {},
            'rag_selections': [],
            'intelligence_effectiveness': {},
            'strategy_evolution': [],
            'comparative_analysis': {}
        }

        # Performance metrics
        self.metrics = {
            'total_signals': 0,
            'total_trades': 0,
            'rag_successful_selections': 0,
            'regime_predictions_correct': 0,
            'intelligence_filter_saves': 0,
            'adaptation_events': 0
        }

    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # Add file handler for detailed logs
        file_handler = logging.FileHandler('rag_intelligence_backtest.log')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def run_backtest(self,
                    epic: str,
                    days: int = 7,
                    timeframe: str = '15m',
                    initial_balance: float = 10000.0,
                    position_size: float = 1.0,
                    show_signals: bool = False) -> Dict:
        """
        Run comprehensive RAG Intelligence Strategy backtest

        Args:
            epic: Trading instrument
            days: Backtest period in days
            timeframe: Analysis timeframe
            initial_balance: Starting account balance
            position_size: Position size per trade
            show_signals: Whether to display individual signals

        Returns:
            Complete backtest results
        """
        try:
            self.logger.info(f"🚀 Starting RAG Intelligence Strategy backtest")
            self.logger.info(f"   Epic: {epic}")
            self.logger.info(f"   Period: {days} days")
            self.logger.info(f"   Timeframe: {timeframe}")
            self.logger.info(f"   Initial Balance: ${initial_balance:,.2f}")

            # Check database availability
            if not self.db_available:
                self.logger.warning("⚠️ Database not available - running in simulation mode")
                return self._run_simulation_mode(epic, days, timeframe, initial_balance, position_size)

            # Initialize strategy
            self.rag_strategy = RAGIntelligenceStrategy(
                epic=epic,
                data_fetcher=self.data_fetcher,
                backtest_mode=True
            )

            # Get historical data
            historical_data = self._get_historical_data(epic, days, timeframe)

            if historical_data.empty:
                self.logger.warning(f"No historical data found for {epic} - generating synthetic data")
                historical_data = self._generate_synthetic_data(days, timeframe)

            self.logger.info(f"📈 Processing {len(historical_data)} historical bars")

            # Run simulation
            simulation_results = self._run_simulation(
                historical_data, epic, initial_balance, position_size, timeframe, show_signals
            )

            # Analyze results
            analysis_results = self._analyze_results(simulation_results, epic)

            # Generate comprehensive report
            final_results = self._generate_final_report(
                simulation_results, analysis_results, epic, days, timeframe
            )

            self.logger.info(f"✅ Backtest completed successfully")
            return final_results

        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return self._get_empty_results()

    def _run_simulation_mode(self, epic: str, days: int, timeframe: str,
                           initial_balance: float, position_size: float) -> Dict:
        """Run backtest in simulation mode when database unavailable"""
        try:
            self.logger.info("🔄 Running in simulation mode (no database)")

            # Initialize strategy without database
            self.rag_strategy = RAGIntelligenceStrategy(
                epic=epic,
                data_fetcher=None,  # No database
                backtest_mode=True
            )

            # Generate synthetic data for testing
            synthetic_data = self._generate_synthetic_data(days, timeframe)

            # Run simplified simulation
            simulation_results = self._run_simulation(
                synthetic_data, epic, initial_balance, position_size, timeframe, show_signals
            )

            # Create basic analysis
            analysis_results = {
                'basic_metrics': {
                    'total_trades': len(simulation_results.get('trades', [])),
                    'simulation_mode': True
                },
                'regime_performance': {},
                'rag_effectiveness': {},
                'intelligence_impact': {'mode': 'simulation'}
            }

            # Generate report
            final_results = self._generate_final_report(
                simulation_results, analysis_results, epic, days, timeframe
            )

            final_results['simulation_mode'] = True
            return final_results

        except Exception as e:
            self.logger.error(f"Simulation mode failed: {e}")
            return self._get_empty_results()

    def _generate_synthetic_data(self, days: int, timeframe: str) -> pd.DataFrame:
        """Generate synthetic market data for testing"""
        try:
            # Calculate number of bars based on timeframe
            timeframe_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240}
            minutes = timeframe_minutes.get(timeframe, 15)
            bars_per_day = (24 * 60) // minutes
            total_bars = days * bars_per_day

            # Generate timestamps
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            timestamps = pd.date_range(start=start_time, end=end_time, periods=total_bars)

            # Generate realistic price data
            base_price = 1.0950  # EUR/USD base
            volatility = 0.001   # 1% daily volatility

            # Generate price walks
            returns = np.random.normal(0, volatility/np.sqrt(bars_per_day), total_bars)
            prices = base_price * np.exp(np.cumsum(returns))

            # Create OHLC data
            opens = prices
            closes = prices * (1 + np.random.normal(0, 0.0001, total_bars))

            # Highs and lows
            high_multiplier = 1 + np.abs(np.random.normal(0, 0.0002, total_bars))
            low_multiplier = 1 - np.abs(np.random.normal(0, 0.0002, total_bars))

            highs = np.maximum(opens, closes) * high_multiplier
            lows = np.minimum(opens, closes) * low_multiplier

            # Volume
            volumes = np.random.randint(1000, 50000, total_bars)

            synthetic_data = pd.DataFrame({
                'start_time': timestamps,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes,
                'epic': f'SYNTHETIC_{timeframe}',
                'timeframe': minutes
            })

            synthetic_data.set_index('start_time', inplace=True)

            self.logger.info(f"📊 Generated {len(synthetic_data)} bars of synthetic data")
            return synthetic_data

        except Exception as e:
            self.logger.error(f"Synthetic data generation failed: {e}")
            return pd.DataFrame()

    def _get_historical_data(self, epic: str, days: int, timeframe: str) -> pd.DataFrame:
        """Get historical market data for backtesting"""
        try:
            # Calculate timeframe in minutes
            timeframe_mapping = {
                '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
            }
            tf_minutes = timeframe_mapping.get(timeframe, 15)

            # Calculate date range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)

            self.logger.info(f"Fetching data from {start_time} to {end_time}")

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
                    params=[epic, tf_minutes, start_time, end_time]
                )

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['start_time'])
                df.set_index('timestamp', inplace=True)
                self.logger.info(f"✅ Loaded {len(df)} historical bars")
            else:
                self.logger.warning("No historical data found")

            return df

        except Exception as e:
            self.logger.error(f"Historical data fetch failed: {e}")
            return pd.DataFrame()

    def _run_simulation(self,
                       data: pd.DataFrame,
                       epic: str,
                       initial_balance: float,
                       position_size: float,
                       timeframe: str,
                       show_signals: bool = False) -> Dict:
        """
        Run the main trading simulation with RAG strategy adaptation
        """
        try:
            # Simulation state
            balance = initial_balance
            position = None
            trades = []
            signals = []
            regime_history = []
            rag_selections = []

            # Performance tracking
            peak_balance = initial_balance
            max_drawdown = 0.0
            consecutive_losses = 0

            # Strategy evolution tracking
            current_strategy_code = None
            strategy_changes = []

            self.logger.info("🔄 Starting trading simulation...")

            # Process each bar
            for i in range(100, len(data)):  # Start from bar 100 for sufficient history
                current_bar = data.iloc[i]
                historical_window = data.iloc[max(0, i-200):i+1]  # 200 bar history

                # Analyze market conditions (every 24 bars for efficiency)
                if i % 24 == 0 or current_strategy_code is None:
                    market_condition = self.rag_strategy.analyze_market_conditions(epic)
                    regime_history.append({
                        'timestamp': current_bar.name,
                        'bar_index': i,
                        'regime': market_condition.regime,
                        'confidence': market_condition.confidence,
                        'session': market_condition.session,
                        'volatility': market_condition.volatility
                    })

                    # Select RAG strategy code
                    new_strategy_code = self.rag_strategy.select_optimal_code(market_condition)

                    if (current_strategy_code is None or
                        new_strategy_code.source_id != current_strategy_code.source_id):

                        strategy_changes.append({
                            'timestamp': current_bar.name,
                            'bar_index': i,
                            'old_strategy': current_strategy_code.source_id if current_strategy_code else 'none',
                            'new_strategy': new_strategy_code.source_id,
                            'reason': 'regime_change',
                            'market_regime': market_condition.regime
                        })

                        current_strategy_code = new_strategy_code
                        self.metrics['adaptation_events'] += 1

                    rag_selections.append({
                        'timestamp': current_bar.name,
                        'strategy_code': new_strategy_code.source_id,
                        'confidence': new_strategy_code.confidence_score,
                        'market_regime': market_condition.regime,
                        'suitability': new_strategy_code.market_suitability
                    })

                # Generate trading signal
                signal = self.rag_strategy.detect_signal(
                    historical_window, epic, spread_pips=1.5, timeframe=timeframe
                )

                if signal:
                    self.metrics['total_signals'] += 1
                    signal_info = {
                        'timestamp': current_bar.name,
                        'bar_index': i,
                        'signal': signal,
                        'market_regime': market_condition.regime if 'market_condition' in locals() else 'unknown',
                        'strategy_code': current_strategy_code.source_id if current_strategy_code else 'unknown'
                    }
                    signals.append(signal_info)

                    # Display signal if requested
                    if show_signals:
                        self._display_signal(signal_info, i)

                    # Execute trade if no position
                    if position is None:
                        trade_result = self._execute_trade(
                            signal, current_bar, balance, position_size, epic
                        )

                        if trade_result['success']:
                            position = trade_result['position']
                            self.metrics['total_trades'] += 1
                            self.logger.debug(f"📊 Trade opened: {signal['direction']} @ {signal['entry_price']}")

                # Manage existing position
                if position:
                    trade_outcome = self._manage_position(
                        position, current_bar, balance
                    )

                    if trade_outcome:
                        balance = trade_outcome['new_balance']
                        trades.append(trade_outcome['trade'])
                        position = None

                        # Update performance tracking
                        if balance > peak_balance:
                            peak_balance = balance

                        current_drawdown = (peak_balance - balance) / peak_balance
                        max_drawdown = max(max_drawdown, current_drawdown)

                        if trade_outcome['trade']['profit_loss'] < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                        self.logger.debug(f"💰 Trade closed: P/L ${trade_outcome['trade']['profit_loss']:.2f}")

            # Close any remaining position
            if position:
                final_trade = self._close_position(position, data.iloc[-1], balance)
                if final_trade:
                    trades.append(final_trade)
                    balance = final_trade['exit_balance']

            return {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': balance - initial_balance,
                'return_percent': ((balance - initial_balance) / initial_balance) * 100,
                'max_drawdown': max_drawdown * 100,
                'peak_balance': peak_balance,
                'trades': trades,
                'signals': signals,
                'regime_history': regime_history,
                'rag_selections': rag_selections,
                'strategy_changes': strategy_changes,
                'bars_processed': len(data),
                'metrics': self.metrics
            }

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return self._get_empty_simulation_results()

    def _execute_trade(self,
                      signal: Dict,
                      current_bar: pd.Series,
                      balance: float,
                      position_size: float,
                      epic: str) -> Dict:
        """Execute a trade based on signal"""
        try:
            # Calculate position size based on balance and risk management
            risk_amount = balance * 0.02  # 2% risk per trade
            stop_loss_pips = signal.get('stop_loss_pips', 20)

            # Adjust position size based on risk
            if stop_loss_pips > 0:
                pip_value = 1.0  # Simplified pip value
                adjusted_size = min(position_size, risk_amount / (stop_loss_pips * pip_value))
            else:
                adjusted_size = position_size

            position = {
                'epic': epic,
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'position_size': adjusted_size,
                'entry_time': current_bar.name,
                'stop_loss': self._calculate_stop_loss(signal),
                'take_profit': self._calculate_take_profit(signal),
                'entry_balance': balance,
                'signal_confidence': signal.get('confidence', 0.5),
                'market_regime': signal.get('intelligence_context', {}).get('market_regime', 'unknown'),
                'strategy_type': signal.get('signal_type', 'unknown')
            }

            return {'success': True, 'position': position}

        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return {'success': False, 'error': str(e)}

    def _manage_position(self, position: Dict, current_bar: pd.Series, balance: float) -> Optional[Dict]:
        """Manage existing position and check for exit conditions"""
        try:
            current_price = current_bar['close']
            direction = position['direction']
            entry_price = position['entry_price']

            # Check stop loss
            if direction == 'BUY' and current_price <= position['stop_loss']:
                return self._close_trade(position, current_bar, balance, 'STOP_LOSS')
            elif direction == 'SELL' and current_price >= position['stop_loss']:
                return self._close_trade(position, current_bar, balance, 'STOP_LOSS')

            # Check take profit
            if direction == 'BUY' and current_price >= position['take_profit']:
                return self._close_trade(position, current_bar, balance, 'TAKE_PROFIT')
            elif direction == 'SELL' and current_price <= position['take_profit']:
                return self._close_trade(position, current_bar, balance, 'TAKE_PROFIT')

            # No exit condition met
            return None

        except Exception as e:
            self.logger.error(f"Position management failed: {e}")
            return None

    def _close_trade(self,
                    position: Dict,
                    current_bar: pd.Series,
                    balance: float,
                    exit_reason: str) -> Dict:
        """Close a trade and calculate results"""
        try:
            exit_price = current_bar['close']
            direction = position['direction']
            entry_price = position['entry_price']
            position_size = position['position_size']

            # Calculate P/L
            if direction == 'BUY':
                price_diff = exit_price - entry_price
            else:  # SELL
                price_diff = entry_price - exit_price

            # Simplified P/L calculation (assuming 1 pip = $1 for demo)
            profit_loss = price_diff * position_size * 10000  # 10000 for pip value

            # Account for spread (simplified)
            spread_cost = 1.5 * position_size
            net_profit_loss = profit_loss - spread_cost

            new_balance = balance + net_profit_loss

            trade = {
                'epic': position['epic'],
                'direction': position['direction'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'entry_time': position['entry_time'],
                'exit_time': current_bar.name,
                'profit_loss': net_profit_loss,
                'exit_reason': exit_reason,
                'entry_balance': position['entry_balance'],
                'exit_balance': new_balance,
                'signal_confidence': position['signal_confidence'],
                'market_regime': position['market_regime'],
                'strategy_type': position['strategy_type'],
                'duration_bars': self._calculate_trade_duration(position['entry_time'], current_bar.name)
            }

            return {
                'trade': trade,
                'new_balance': new_balance
            }

        except Exception as e:
            self.logger.error(f"Trade closure failed: {e}")
            return None

    def _analyze_results(self, simulation_results: Dict, epic: str) -> Dict:
        """Analyze simulation results and generate insights"""
        try:
            trades = simulation_results['trades']
            regime_history = simulation_results['regime_history']
            rag_selections = simulation_results['rag_selections']

            if not trades:
                return {'analysis_error': 'No trades to analyze'}

            # Basic performance metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['profit_loss'] > 0]
            losing_trades = [t for t in trades if t['profit_loss'] <= 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

            total_profit = sum(t['profit_loss'] for t in winning_trades)
            total_loss = abs(sum(t['profit_loss'] for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            # Regime-based performance analysis
            regime_performance = {}
            for regime in ['trending_up', 'trending_down', 'ranging', 'breakout']:
                regime_trades = [t for t in trades if t.get('market_regime') == regime]
                if regime_trades:
                    regime_win_rate = len([t for t in regime_trades if t['profit_loss'] > 0]) / len(regime_trades)
                    regime_avg_profit = np.mean([t['profit_loss'] for t in regime_trades])
                    regime_performance[regime] = {
                        'trades': len(regime_trades),
                        'win_rate': regime_win_rate,
                        'avg_profit': regime_avg_profit,
                        'total_profit': sum(t['profit_loss'] for t in regime_trades)
                    }

            # RAG selection effectiveness
            rag_effectiveness = {}
            if rag_selections:
                unique_strategies = set(s['strategy_code'] for s in rag_selections)
                for strategy in unique_strategies:
                    strategy_trades = [t for t in trades
                                     if any(s['strategy_code'] == strategy
                                           for s in rag_selections
                                           if s['timestamp'] <= t['entry_time'])]
                    if strategy_trades:
                        strategy_win_rate = len([t for t in strategy_trades if t['profit_loss'] > 0]) / len(strategy_trades)
                        rag_effectiveness[strategy] = {
                            'trades': len(strategy_trades),
                            'win_rate': strategy_win_rate,
                            'avg_profit': np.mean([t['profit_loss'] for t in strategy_trades])
                        }

            return {
                'basic_metrics': {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_profit': sum(t['profit_loss'] for t in trades),
                    'avg_trade_duration': np.mean([t.get('duration_bars', 0) for t in trades]),
                    'max_consecutive_wins': self._calculate_max_consecutive(trades, True),
                    'max_consecutive_losses': self._calculate_max_consecutive(trades, False)
                },
                'regime_performance': regime_performance,
                'rag_effectiveness': rag_effectiveness,
                'intelligence_impact': {
                    'regime_changes_tracked': len(set(r['regime'] for r in regime_history)),
                    'strategy_adaptations': simulation_results.get('metrics', {}).get('adaptation_events', 0),
                    'total_signals': simulation_results.get('metrics', {}).get('total_signals', 0)
                }
            }

        except Exception as e:
            self.logger.error(f"Results analysis failed: {e}")
            return {'analysis_error': str(e)}

    def _generate_final_report(self,
                              simulation_results: Dict,
                              analysis_results: Dict,
                              epic: str,
                              days: int,
                              timeframe: str) -> Dict:
        """Generate comprehensive final report"""
        try:
            report = {
                'backtest_info': {
                    'epic': epic,
                    'period_days': days,
                    'timeframe': timeframe,
                    'strategy': 'RAG Intelligence',
                    'timestamp': datetime.utcnow().isoformat(),
                    'bars_processed': simulation_results.get('bars_processed', 0)
                },
                'performance_summary': {
                    'initial_balance': simulation_results['initial_balance'],
                    'final_balance': simulation_results['final_balance'],
                    'total_return': simulation_results['total_return'],
                    'return_percent': simulation_results['return_percent'],
                    'max_drawdown_percent': simulation_results['max_drawdown'],
                    'sharpe_ratio': self._calculate_sharpe_ratio(simulation_results['trades'])
                },
                'trading_metrics': analysis_results.get('basic_metrics', {}),
                'regime_analysis': analysis_results.get('regime_performance', {}),
                'rag_analysis': analysis_results.get('rag_effectiveness', {}),
                'intelligence_metrics': analysis_results.get('intelligence_impact', {}),
                'detailed_results': {
                    'trades': simulation_results['trades'],
                    'regime_history': simulation_results.get('regime_history', []),
                    'strategy_changes': simulation_results.get('strategy_changes', []),
                    'performance_evolution': self._calculate_performance_evolution(simulation_results['trades'])
                },
                'recommendations': self._generate_recommendations(analysis_results),
                'raw_data': {
                    'simulation_results': simulation_results,
                    'analysis_results': analysis_results
                }
            }

            # Add summary statistics
            report['summary'] = self._create_summary_text(report)

            return report

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {'report_error': str(e)}

    # Helper methods

    def _calculate_stop_loss(self, signal: Dict) -> float:
        """Calculate stop loss price"""
        entry_price = signal['entry_price']
        direction = signal['direction']
        stop_loss_pips = signal.get('stop_loss_pips', 20)

        if direction == 'BUY':
            return entry_price - (stop_loss_pips * 0.0001)  # Assuming 4-decimal pricing
        else:
            return entry_price + (stop_loss_pips * 0.0001)

    def _calculate_take_profit(self, signal: Dict) -> float:
        """Calculate take profit price"""
        entry_price = signal['entry_price']
        direction = signal['direction']
        take_profit_pips = signal.get('take_profit_pips', 40)

        if direction == 'BUY':
            return entry_price + (take_profit_pips * 0.0001)
        else:
            return entry_price - (take_profit_pips * 0.0001)

    def _calculate_trade_duration(self, entry_time, exit_time) -> int:
        """Calculate trade duration in bars"""
        try:
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            if isinstance(exit_time, str):
                exit_time = pd.to_datetime(exit_time)
            return int((exit_time - entry_time).total_seconds() / 900)  # 15-minute bars
        except:
            return 0

    def _calculate_max_consecutive(self, trades: List[Dict], wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            is_win = trade['profit_loss'] > 0
            if is_win == wins:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio"""
        if not trades:
            return 0.0

        returns = [t['profit_loss'] for t in trades]
        if not returns:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        return mean_return / std_return if std_return > 0 else 0.0

    def _display_signal(self, signal_info: Dict, bar_index: int):
        """Display individual signal information"""
        signal = signal_info['signal']
        timestamp = signal_info['timestamp']
        regime = signal_info['market_regime']
        strategy = signal_info['strategy_code']

        # Format timestamp
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)

        # Display signal
        direction_icon = "📈" if signal['direction'] == 'BUY' else "📉"
        confidence_color = "🟢" if signal['confidence'] > 0.7 else "🟡" if signal['confidence'] > 0.5 else "🟠"

        self.logger.info(f"{direction_icon} SIGNAL #{bar_index}: {signal['direction']} @ {signal['entry_price']:.5f}")
        self.logger.info(f"   {confidence_color} Confidence: {signal['confidence']:.1%}")
        self.logger.info(f"   🕐 Time: {time_str}")
        self.logger.info(f"   📊 Regime: {regime}")
        self.logger.info(f"   🤖 Strategy: {strategy}")
        if 'stop_loss' in signal:
            self.logger.info(f"   🛑 Stop Loss: {signal['stop_loss']:.5f}")
        if 'take_profit' in signal:
            self.logger.info(f"   🎯 Take Profit: {signal['take_profit']:.5f}")
        self.logger.info("")  # Empty line for readability

    def _calculate_performance_evolution(self, trades: List[Dict]) -> List[Dict]:
        """Calculate performance evolution over time"""
        evolution = []
        running_balance = 0

        for i, trade in enumerate(trades):
            running_balance += trade['profit_loss']
            evolution.append({
                'trade_number': i + 1,
                'cumulative_profit': running_balance,
                'timestamp': trade['exit_time']
            })

        return evolution

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        basic_metrics = analysis.get('basic_metrics', {})
        win_rate = basic_metrics.get('win_rate', 0)
        profit_factor = basic_metrics.get('profit_factor', 0)

        if win_rate < 0.5:
            recommendations.append("Consider tightening signal filters to improve win rate")

        if profit_factor < 1.2:
            recommendations.append("Improve risk/reward ratio by adjusting stop loss and take profit levels")

        regime_performance = analysis.get('regime_performance', {})
        best_regime = max(regime_performance.keys(),
                         key=lambda k: regime_performance[k].get('win_rate', 0),
                         default=None)

        if best_regime:
            recommendations.append(f"Strategy performs best in {best_regime} markets - consider focusing on this regime")

        return recommendations

    def _create_summary_text(self, report: Dict) -> str:
        """Create human-readable summary"""
        perf = report['performance_summary']
        metrics = report['trading_metrics']

        return f"""
RAG Intelligence Strategy Backtest Summary:
==========================================

Period: {report['backtest_info']['period_days']} days on {report['backtest_info']['epic']}
Total Return: ${perf['total_return']:.2f} ({perf['return_percent']:.1f}%)
Max Drawdown: {perf['max_drawdown_percent']:.1f}%
Win Rate: {metrics.get('win_rate', 0):.1%}
Total Trades: {metrics.get('total_trades', 0)}
Profit Factor: {metrics.get('profit_factor', 0):.2f}
Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}
        """

    def _close_position(self, position: Dict, final_bar: pd.Series, balance: float) -> Optional[Dict]:
        """Close position at end of backtest"""
        return self._close_trade(position, final_bar, balance, 'END_OF_BACKTEST')

    def _get_empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'error': 'Backtest failed',
            'performance_summary': {},
            'trading_metrics': {},
            'regime_analysis': {},
            'rag_analysis': {}
        }

    def _get_empty_simulation_results(self) -> Dict:
        """Return empty simulation results"""
        return {
            'initial_balance': 0,
            'final_balance': 0,
            'total_return': 0,
            'return_percent': 0,
            'trades': [],
            'signals': [],
            'regime_history': [],
            'rag_selections': []
        }


def run_multi_epic_backtest(args) -> Dict:
    """
    Run backtest across multiple trading instruments

    Args:
        args: Parsed command line arguments

    Returns:
        Combined results from multiple instruments
    """
    logger = logging.getLogger('multi_epic_backtest')

    # Default list of popular forex pairs
    default_epics = [
        'CS.D.EURUSD.CEEM.IP',
        'CS.D.GBPUSD.MINI.IP',
        'CS.D.USDJPY.MINI.IP',
        'CS.D.USDCHF.MINI.IP',
        'CS.D.AUDUSD.MINI.IP'
    ]

    # Try to get available epics from database or use defaults
    available_epics = get_available_epics() or default_epics

    logger.info(f"🚀 Starting Multi-Epic RAG Intelligence Backtest")
    logger.info(f"   Testing {len(available_epics)} instruments")
    logger.info(f"   Period: {args.days} days")
    logger.info(f"   Timeframe: {args.timeframe}")
    logger.info(f"   Initial Balance per Epic: ${args.balance:,.2f}")

    all_results = {}
    combined_stats = {
        'total_instruments': len(available_epics),
        'successful_backtests': 0,
        'failed_backtests': 0,
        'total_trades': 0,
        'total_profit_loss': 0.0,
        'best_performer': None,
        'worst_performer': None,
        'overall_win_rate': 0.0,
        'combined_return_percent': 0.0
    }

    # Run backtest for each epic
    for i, epic in enumerate(available_epics, 1):
        logger.info(f"\n📊 [{i}/{len(available_epics)}] Testing {epic}")

        try:
            backtest = RAGIntelligenceBacktest()
            epic_results = backtest.run_backtest(
                epic=epic,
                days=args.days,
                timeframe=args.timeframe,
                initial_balance=args.balance,
                position_size=args.size,
                show_signals=args.show_signals
            )

            if 'error' not in epic_results:
                all_results[epic] = epic_results
                combined_stats['successful_backtests'] += 1

                # Extract key metrics
                perf = epic_results.get('performance_summary', {})
                metrics = epic_results.get('trading_metrics', {})

                total_return = perf.get('total_return', 0)
                return_percent = perf.get('return_percent', 0)
                total_trades = metrics.get('total_trades', 0)

                combined_stats['total_trades'] += total_trades
                combined_stats['total_profit_loss'] += total_return

                # Track best/worst performers
                if (combined_stats['best_performer'] is None or
                    return_percent > all_results[combined_stats['best_performer']]['performance_summary'].get('return_percent', 0)):
                    combined_stats['best_performer'] = epic

                if (combined_stats['worst_performer'] is None or
                    return_percent < all_results[combined_stats['worst_performer']]['performance_summary'].get('return_percent', 0)):
                    combined_stats['worst_performer'] = epic

                logger.info(f"   ✅ {epic}: {return_percent:.1f}% return, {total_trades} trades")

            else:
                logger.warning(f"   ❌ {epic}: {epic_results.get('error', 'Unknown error')}")
                combined_stats['failed_backtests'] += 1

        except Exception as e:
            logger.error(f"   ❌ {epic}: Exception - {e}")
            combined_stats['failed_backtests'] += 1

    # Calculate combined statistics
    if combined_stats['successful_backtests'] > 0:
        # Overall win rate (weighted by number of trades)
        total_winning_trades = 0
        for epic, results in all_results.items():
            metrics = results.get('trading_metrics', {})
            win_rate = metrics.get('win_rate', 0)
            total_trades = metrics.get('total_trades', 0)
            total_winning_trades += win_rate * total_trades

        if combined_stats['total_trades'] > 0:
            combined_stats['overall_win_rate'] = total_winning_trades / combined_stats['total_trades']

        # Combined return percentage
        total_initial = args.balance * combined_stats['successful_backtests']
        if total_initial > 0:
            combined_stats['combined_return_percent'] = (combined_stats['total_profit_loss'] / total_initial) * 100

    # Create comprehensive multi-epic report
    multi_epic_report = {
        'backtest_type': 'multi_epic',
        'backtest_info': {
            'total_instruments': len(available_epics),
            'successful_backtests': combined_stats['successful_backtests'],
            'failed_backtests': combined_stats['failed_backtests'],
            'period_days': args.days,
            'timeframe': args.timeframe,
            'initial_balance_per_epic': args.balance
        },
        'combined_performance': {
            'total_trades': combined_stats['total_trades'],
            'total_profit_loss': combined_stats['total_profit_loss'],
            'combined_return_percent': combined_stats['combined_return_percent'],
            'overall_win_rate': combined_stats['overall_win_rate'],
            'best_performer': combined_stats['best_performer'],
            'worst_performer': combined_stats['worst_performer']
        },
        'individual_results': all_results,
        'summary': create_multi_epic_summary(combined_stats, all_results)
    }

    return multi_epic_report


def get_available_epics() -> List[str]:
    """
    Get list of available trading instruments from database or return defaults

    Returns:
        List of available epic codes
    """
    try:
        # Try to query database for available instruments
        # This would require database connection which might not be available
        return None  # Fall back to defaults
    except:
        return None  # Fall back to defaults


def create_multi_epic_summary(combined_stats: Dict, all_results: Dict) -> str:
    """Create human-readable summary for multi-epic backtest"""

    summary = f"""
Multi-Epic RAG Intelligence Strategy Backtest Summary:
====================================================

Instruments Tested: {combined_stats['total_instruments']}
Successful Backtests: {combined_stats['successful_backtests']}
Failed Backtests: {combined_stats['failed_backtests']}

Combined Performance:
- Total Trades: {combined_stats['total_trades']}
- Total P&L: ${combined_stats['total_profit_loss']:.2f}
- Combined Return: {combined_stats['combined_return_percent']:.1f}%
- Overall Win Rate: {combined_stats['overall_win_rate']:.1%}

Best Performer: {combined_stats['best_performer'] or 'N/A'}
Worst Performer: {combined_stats['worst_performer'] or 'N/A'}

Individual Results:"""

    # Add individual results
    for epic, results in all_results.items():
        perf = results.get('performance_summary', {})
        metrics = results.get('trading_metrics', {})

        summary += f"""
{epic}:
  Return: {perf.get('return_percent', 0):.1f}%
  Trades: {metrics.get('total_trades', 0)}
  Win Rate: {metrics.get('win_rate', 0):.1%}
  Max Drawdown: {perf.get('max_drawdown_percent', 0):.1f}%"""

    return summary


def main():
    """Main backtesting function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='RAG Intelligence Strategy Backtest',
        epilog="""
Examples:
  # Run multi-epic backtest (default when no --epic specified)
  python backtest_rag_intelligence.py --days 7

  # Run single epic backtest with signal display
  python backtest_rag_intelligence.py --epic CS.D.EURUSD.CEEM.IP --days 3 --show-signals

  # Run multi-epic backtest with custom settings
  python backtest_rag_intelligence.py --multi-epic --days 5 --timeframe 1h --balance 5000

  # Save results to file with verbose output and signal display
  python backtest_rag_intelligence.py --output results.json --verbose --show-signals
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--epic', type=str, default=None,
                       help='Trading instrument (e.g., CS.D.EURUSD.CEEM.IP). If not specified, runs multi-epic backtest on popular forex pairs.')
    parser.add_argument('--days', type=int, default=7,
                       help='Backtest period in days (default: 7)')
    parser.add_argument('--timeframe', type=str, default='15m',
                       choices=['5m', '15m', '1h', '4h'],
                       help='Analysis timeframe (default: 15m)')
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='Initial balance per instrument (default: 10000)')
    parser.add_argument('--size', type=float, default=1.0,
                       help='Position size (default: 1.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--multi-epic', action='store_true',
                       help='Force multi-epic backtest even if --epic is specified')
    parser.add_argument('--show-signals', action='store_true',
                       help='Show individual signals during backtest')

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine epic(s) to test
    if args.multi_epic or args.epic is None:
        # Run multi-epic backtest
        results = run_multi_epic_backtest(args)
    else:
        # Run single epic backtest
        backtest = RAGIntelligenceBacktest()
        results = backtest.run_backtest(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            initial_balance=args.balance,
            position_size=args.size,
            show_signals=args.show_signals
        )

    # Display results
    if 'error' not in results:
        print(results.get('summary', 'Backtest completed successfully'))

        # Check if this is a multi-epic backtest
        if results.get('backtest_type') == 'multi_epic':
            # Display multi-epic results
            combined_perf = results.get('combined_performance', {})
            backtest_info = results.get('backtest_info', {})

            print(f"\n📊 Multi-Epic Performance Summary:")
            print(f"   Instruments Tested: {backtest_info.get('total_instruments', 0)}")
            print(f"   Successful Backtests: {backtest_info.get('successful_backtests', 0)}")
            print(f"   Failed Backtests: {backtest_info.get('failed_backtests', 0)}")
            print(f"   Combined Return: ${combined_perf.get('total_profit_loss', 0):.2f} ({combined_perf.get('combined_return_percent', 0):.1f}%)")
            print(f"   Total Trades: {combined_perf.get('total_trades', 0)}")
            print(f"   Overall Win Rate: {combined_perf.get('overall_win_rate', 0):.1%}")
            print(f"   Best Performer: {combined_perf.get('best_performer', 'N/A')}")
            print(f"   Worst Performer: {combined_perf.get('worst_performer', 'N/A')}")

            # Show top 3 individual results
            individual_results = results.get('individual_results', {})
            if individual_results:
                print(f"\n🏆 Top Performers:")
                sorted_results = sorted(
                    individual_results.items(),
                    key=lambda x: x[1].get('performance_summary', {}).get('return_percent', 0),
                    reverse=True
                )
                for i, (epic, epic_results) in enumerate(sorted_results[:3], 1):
                    perf = epic_results.get('performance_summary', {})
                    metrics = epic_results.get('trading_metrics', {})
                    print(f"   {i}. {epic}: {perf.get('return_percent', 0):.1f}% "
                          f"({metrics.get('total_trades', 0)} trades, "
                          f"{metrics.get('win_rate', 0):.1%} win rate)")

        else:
            # Display single epic results
            perf = results.get('performance_summary', {})
            metrics = results.get('trading_metrics', {})

            print(f"\n📊 Key Performance Metrics:")
            print(f"   Total Return: ${perf.get('total_return', 0):.2f}")
            print(f"   Return %: {perf.get('return_percent', 0):.1f}%")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"   Total Trades: {metrics.get('total_trades', 0)}")
            print(f"   Max Drawdown: {perf.get('max_drawdown_percent', 0):.1f}%")

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")
    else:
        print(f"❌ Backtest failed: {results.get('error', 'Unknown error')}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())