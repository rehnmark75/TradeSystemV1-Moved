# core/backtest_scanner.py
"""
BacktestScanner - Extends IntelligentForexScanner for historical backtesting
Operates on historical data from ig_candles table with same signal detection logic
"""

import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Iterator, Tuple
import pandas as pd

try:
    import config
    from core.scanner import IntelligentForexScanner
    from core.database import DatabaseManager
    from core.trading.backtest_order_logger import BacktestOrderLogger
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.scanner import IntelligentForexScanner
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.trading.backtest_order_logger import BacktestOrderLogger


class BacktestScanner(IntelligentForexScanner):
    """
    Historical backtesting scanner that extends live scanner functionality
    Uses same signal detection logic but operates on historical data
    """

    def __init__(self,
                 backtest_config: Dict,
                 db_manager: DatabaseManager = None,
                 **kwargs):

        # Set up backtest-specific configuration
        self.backtest_config = backtest_config
        self.backtest_mode = True

        # Extract backtest parameters
        self.start_date = backtest_config['start_date']
        self.end_date = backtest_config['end_date']
        self.execution_id = backtest_config['execution_id']
        self.strategy_name = backtest_config['strategy_name']
        self.timeframe = backtest_config.get('timeframe', '15m')

        # Initialize parent with backtest-specific settings
        backtest_kwargs = kwargs.copy()
        backtest_kwargs.update({
            'intelligence_mode': 'backtest_consistent',
            'epic_list': backtest_config.get('epics', config.EPIC_LIST),
            'scan_interval': 0,  # No continuous scanning in backtest mode
            'db_manager': db_manager
        })

        super().__init__(**backtest_kwargs)

        # Override scanner metadata
        self.scanner_version = 'backtest_v1.0_integrated_pipeline'

        # Backtest-specific components
        self.order_logger = BacktestOrderLogger(
            self.db_manager,
            self.execution_id,
            logger=self.logger
        )

        # CRITICAL FIX: Override signal detector to use BacktestDataFetcher instead of live DataFetcher
        # This ensures backtest uses historical data while live scanner uses real-time data
        self._override_signal_detector_for_backtest()

        # Time iteration state
        self.current_backtest_time = self.start_date
        self.time_increment = self._parse_timeframe_to_timedelta(self.timeframe)

        # Statistics
        self.backtest_stats = {
            'time_periods_processed': 0,
            'total_signals_detected': 0,
            'signals_logged': 0,
            'data_quality_issues': 0,
            'processing_errors': 0
        }

        self.logger.info(f"🧪 BacktestScanner initialized:")
        self.logger.info(f"   Execution ID: {self.execution_id}")
        self.logger.info(f"   Strategy: {self.strategy_name}")
        self.logger.info(f"   Period: {self.start_date} to {self.end_date}")
        self.logger.info(f"   Epics: {len(self.epic_list)} pairs")
        self.logger.info(f"   Timeframe: {self.timeframe}")

    def _override_signal_detector_for_backtest(self):
        """
        CRITICAL: Replace the live DataFetcher with BacktestDataFetcher in the signal detector
        This ensures backtest uses historical data instead of real-time data
        """
        try:
            from .signal_detector import SignalDetector
            from .backtest_data_fetcher import BacktestDataFetcher

            # Create a new signal detector but replace its data_fetcher with BacktestDataFetcher
            backtest_data_fetcher = BacktestDataFetcher(self.db_manager, getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm'))

            # Replace the data_fetcher in the existing signal_detector
            self.signal_detector.data_fetcher = backtest_data_fetcher

            # Also need to replace the data_fetcher in the EMA strategy AND enable backtest mode
            if hasattr(self.signal_detector, 'ema_strategy') and self.signal_detector.ema_strategy:
                self.signal_detector.ema_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.ema_strategy.backtest_mode = True
                # CRITICAL FIX: Enable optimal parameters for consistent signal detection and validation
                self.signal_detector.ema_strategy.use_optimal_parameters = True
                self.logger.info("✅ EMA strategy configured for backtest mode - will use optimal parameters and process all alert timestamps")

            self.logger.info("✅ Signal detector updated to use BacktestDataFetcher for historical data")

        except Exception as e:
            self.logger.error(f"❌ Failed to override signal detector for backtest: {e}")
            raise

    def run_historical_backtest(self) -> Dict:
        """
        Run complete historical backtest for the specified period
        Returns comprehensive results including signals and performance metrics
        """
        self.logger.info(f"🚀 Starting historical backtest execution {self.execution_id}")

        try:
            with self.order_logger:
                # Initialize execution in database
                self._initialize_backtest_execution()

                # Main backtest loop
                results = self._execute_backtest_loop()

                # Generate final report
                final_report = self._generate_backtest_report(results)

                self.logger.info(f"✅ Backtest completed successfully")
                return final_report

        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            self.backtest_stats['processing_errors'] += 1
            raise

    def _initialize_backtest_execution(self):
        """Initialize or update backtest execution record"""
        try:
            query = """
            UPDATE backtest_executions
            SET strategy_name = :strategy_name,
                data_start_date = :data_start_date,
                data_end_date = :data_end_date,
                epics_tested = CAST(:epics_tested AS text[]),
                timeframes = CAST(:timeframes AS text[]),
                config_snapshot = CAST(:config_snapshot AS jsonb),
                updated_at = NOW()
            WHERE id = :execution_id
            """

            config_snapshot = {
                'timeframe': self.timeframe,
                'min_confidence': self.min_confidence,
                'epic_list': self.epic_list,
                'spread_pips': self.spread_pips,
                'use_signal_processor': getattr(self, 'use_signal_processor', False),
                'enable_smart_money': getattr(self, 'enable_smart_money', False),
                'scanner_version': self.scanner_version
            }

            # Format arrays as PostgreSQL strings
            epics_pg_array = '{' + ','.join(f'"{epic}"' for epic in self.epic_list) + '}'
            timeframes_pg_array = '{' + f'"{self.timeframe}"' + '}'

            params = {
                'strategy_name': self.strategy_name,
                'data_start_date': self.start_date,
                'data_end_date': self.end_date,
                'epics_tested': epics_pg_array,
                'timeframes': timeframes_pg_array,
                'config_snapshot': json.dumps(config_snapshot),
                'execution_id': int(self.execution_id)
            }

            # Handle UPDATE query exception
            try:
                self.db_manager.execute_query(query, params)
            except Exception as update_error:
                if "This result object does not return rows" in str(update_error):
                    # UPDATE query succeeded but DatabaseManager can't create DataFrame - this is expected
                    pass
                else:
                    raise update_error
            self.logger.info("✅ Backtest execution initialized in database")

        except Exception as e:
            self.logger.error(f"Error initializing backtest execution: {e}")
            raise

    def _execute_backtest_loop(self) -> Dict:
        """Execute main backtest time iteration loop"""
        results = {
            'signals_by_epic': {},
            'time_periods_processed': 0,
            'total_signals': 0,
            'start_time': datetime.now(),
            'processing_errors': []
        }

        # Create time iterator
        time_iterator = self._create_time_iterator()

        try:
            for current_time in time_iterator:
                try:
                    # Process this time period
                    period_results = self._process_time_period(current_time)

                    # Update results
                    results['time_periods_processed'] += 1
                    self.backtest_stats['time_periods_processed'] += 1

                    if period_results['signals']:
                        for signal in period_results['signals']:
                            epic = signal.get('epic', 'unknown')
                            if epic not in results['signals_by_epic']:
                                results['signals_by_epic'][epic] = []
                            results['signals_by_epic'][epic].append(signal)

                        results['total_signals'] += len(period_results['signals'])
                        self.backtest_stats['total_signals_detected'] += len(period_results['signals'])

                    # Update execution stats periodically
                    if results['time_periods_processed'] % 100 == 0:
                        self._update_execution_progress(results)

                        # Log progress
                        elapsed = (datetime.now() - results['start_time']).total_seconds()
                        rate = results['time_periods_processed'] / max(elapsed, 1)
                        self.logger.info(f"📊 Progress: {results['time_periods_processed']} periods, "
                                       f"{results['total_signals']} signals, "
                                       f"{rate:.1f} periods/sec")

                except Exception as e:
                    self.logger.error(f"Error processing time period {current_time}: {e}")
                    results['processing_errors'].append({
                        'timestamp': current_time,
                        'error': str(e)
                    })
                    self.backtest_stats['processing_errors'] += 1

        except KeyboardInterrupt:
            self.logger.warning("⚠️ Backtest interrupted by user")
            raise

        return results

    def _create_time_iterator(self) -> Iterator[datetime]:
        """Create iterator for backtest time periods"""
        current_time = self.start_date

        while current_time <= self.end_date:
            yield current_time
            current_time += self.time_increment

    def _process_time_period(self, current_time: datetime) -> Dict:
        """Process a single time period for all epics"""
        period_results = {
            'timestamp': current_time,
            'signals': [],
            'data_quality_score': 1.0,
            'epics_processed': 0
        }

        try:
            # Set the current backtest time (important for data fetcher)
            self.current_backtest_time = current_time

            # CRITICAL FIX: Sync timestamp with data fetcher for time-aware data filtering
            if hasattr(self.signal_detector, 'data_fetcher') and hasattr(self.signal_detector.data_fetcher, 'current_backtest_time'):
                self.signal_detector.data_fetcher.current_backtest_time = current_time

            # Override scan_once to use historical data at this timestamp
            signals = self._scan_historical_timepoint(current_time)

            if signals:
                # Process signals through the same pipeline as live scanner
                processed_signals = self._process_backtest_signals(signals, current_time)
                period_results['signals'] = processed_signals

                # Log signals to database
                for signal in processed_signals:
                    success, message, order_data = self.order_logger.place_order(signal)
                    if success:
                        self.backtest_stats['signals_logged'] += 1

        except Exception as e:
            self.logger.error(f"Error processing time period {current_time}: {e}")
            raise

        return period_results

    def _scan_historical_timepoint(self, timestamp: datetime) -> List[Dict]:
        """Scan all epics at a specific historical timestamp"""
        signals = []

        for epic in self.epic_list:
            try:
                epic_signals = self._detect_signals_for_epic_at_time(epic, timestamp)
                if epic_signals:
                    if isinstance(epic_signals, list):
                        signals.extend(epic_signals)
                    else:
                        signals.append(epic_signals)

            except Exception as e:
                self.logger.error(f"Error detecting signals for {epic} at {timestamp}: {e}")
                continue

        return signals

    def _detect_signals_for_epic_at_time(self, epic: str, timestamp: datetime) -> Optional[Dict]:
        """
        Detect signals for a specific epic at a specific timestamp
        Uses the same logic as parent class but with historical data
        Supports strategy filtering for backtests
        """
        try:
            pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')

            # Check if specific strategy is requested
            strategy_name = self.strategy_name.upper()
            self.logger.debug(f"🎯 Backtest strategy filter: '{strategy_name}' for {epic}")

            # Strategy method mapping
            strategy_methods = {
                'EMA_CROSSOVER': 'detect_signals_mid_prices',
                'EMA': 'detect_signals_mid_prices',
                'MACD': 'detect_macd_ema_signals',
                'MACD_EMA': 'detect_macd_ema_signals',
                'KAMA': 'detect_kama_signals',
                'BOLLINGER_SUPERTREND': 'detect_bb_supertrend_signals',
                'BB_SUPERTREND': 'detect_bb_supertrend_signals',
                'BB': 'detect_bb_supertrend_signals',
                'ZERO_LAG': 'detect_zero_lag_signals',
                'ZEROLAG': 'detect_zero_lag_signals',  # Fixed: Accept both forms
                'ZL': 'detect_zero_lag_signals',
                'MOMENTUM': 'detect_momentum_signals',
                'SMC_FAST': 'detect_smc_signals',
                'SMC': 'detect_smc_signals',
                'ICHIMOKU': 'detect_ichimoku_signals',
                'ICHIMOKU_CLOUD': 'detect_ichimoku_signals',
                'MEAN_REVERSION': 'detect_mean_reversion_signals',
                'MEANREV': 'detect_mean_reversion_signals',  # Fixed: Accept short form
                'RANGING_MARKET': 'detect_ranging_market_signals',
                'RANGING': 'detect_ranging_market_signals'
            }

            # Use specific strategy if requested, otherwise use all strategies
            if strategy_name in strategy_methods:
                method_name = strategy_methods[strategy_name]
                if hasattr(self.signal_detector, method_name):
                    self.logger.debug(f"🎯 Running {strategy_name} strategy only for {epic}")
                    method = getattr(self.signal_detector, method_name)
                    signals = method(epic, pair_name, self.timeframe)

                    # Some methods return a single signal dict, others return lists
                    if signals and not isinstance(signals, list):
                        signals = [signals]

                    return signals[0] if signals else None
                else:
                    self.logger.warning(f"⚠️ Strategy method {method_name} not found, falling back to all strategies")

            # Default: use all strategies (original behavior)
            if hasattr(self.signal_detector, 'detect_signals_all_strategies'):
                self.logger.debug(f"🔄 Running all strategies for {epic}")
                signals = self.signal_detector.detect_signals_all_strategies(
                    epic, pair_name, self.spread_pips, self.timeframe
                )
            else:
                # Fallback to single strategy
                signals = self.signal_detector.detect_signals_mid_prices(
                    epic, pair_name, self.timeframe
                )

            return signals

        except Exception as e:
            self.logger.error(f"Error detecting signals for {epic} at {timestamp}: {e}")
            return None

    def _process_backtest_signals(self, signals: List[Dict], timestamp: datetime) -> List[Dict]:
        """
        Process signals through the same pipeline as live scanner
        But add backtest-specific metadata
        """
        processed_signals = []

        for signal in signals:
            try:
                # Add backtest metadata
                signal['backtest_execution_id'] = self.execution_id
                signal['backtest_timestamp'] = timestamp
                signal['backtest_mode'] = True
                signal['signal_timestamp'] = timestamp

                # Process through parent's signal preparation
                processed_signal = self._prepare_signal(signal)
                processed_signal['scanner_version'] = self.scanner_version

                processed_signals.append(processed_signal)

            except Exception as e:
                self.logger.error(f"Error processing backtest signal: {e}")
                continue

        return processed_signals

    def _update_execution_progress(self, results: Dict):
        """Update backtest execution progress in database"""
        try:
            self.order_logger.update_execution_stats(
                completed_combinations=results['time_periods_processed'],
                total_candles_processed=results['time_periods_processed'] * len(self.epic_list)
            )
        except Exception as e:
            self.logger.error(f"Error updating execution progress: {e}")

    def _generate_backtest_report(self, results: Dict) -> Dict:
        """Generate comprehensive backtest report"""
        end_time = datetime.now()
        duration = (end_time - results['start_time']).total_seconds()

        report = {
            'execution_id': self.execution_id,
            'strategy_name': self.strategy_name,
            'backtest_period': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'timeframe': self.timeframe
            },
            'execution_stats': {
                'duration_seconds': duration,
                'time_periods_processed': results['time_periods_processed'],
                'total_signals_detected': results['total_signals'],
                'signals_logged': self.backtest_stats['signals_logged'],
                'processing_rate_per_second': results['time_periods_processed'] / max(duration, 1),
                'processing_errors': len(results['processing_errors'])
            },
            'signal_summary': self._generate_signal_summary(results['signals_by_epic']),
            'performance_summary': self._get_performance_summary(),
            'data_quality': {
                'overall_score': 1.0 - (self.backtest_stats['data_quality_issues'] / max(results['time_periods_processed'], 1))
            },
            'backtest_stats': self.backtest_stats.copy()
        }

        # Log summary
        self.logger.info(f"📊 Backtest Summary:")
        self.logger.info(f"   Duration: {duration:.1f}s")
        self.logger.info(f"   Periods processed: {results['time_periods_processed']}")
        self.logger.info(f"   Signals detected: {results['total_signals']}")
        self.logger.info(f"   Signals logged: {self.backtest_stats['signals_logged']}")
        self.logger.info(f"   Processing rate: {report['execution_stats']['processing_rate_per_second']:.1f} periods/sec")

        return report

    def _generate_signal_summary(self, signals_by_epic: Dict) -> Dict:
        """Generate signal summary by epic"""
        summary = {}

        for epic, signals in signals_by_epic.items():
            bull_signals = len([s for s in signals if s.get('signal_type') in ['BULL', 'BUY']])
            bear_signals = len([s for s in signals if s.get('signal_type') in ['BEAR', 'SELL']])
            avg_confidence = sum(s.get('confidence_score', 0) for s in signals) / max(len(signals), 1)

            summary[epic] = {
                'total_signals': len(signals),
                'bull_signals': bull_signals,
                'bear_signals': bear_signals,
                'avg_confidence': avg_confidence
            }

        return summary

    def _get_performance_summary(self) -> Dict:
        """Get performance summary from database"""
        try:
            summary_params = {'execution_id': int(self.execution_id)}
            try:
                result_df = self.db_manager.execute_query(
                    "SELECT * FROM get_backtest_summary(:execution_id)",
                    summary_params
                )
                result = result_df.iloc[0].to_dict() if not result_df.empty else None

                if result:
                    return {
                        'total_signals': result.get('total_signals', 0),
                        'total_validated_signals': result.get('total_validated_signals', 0),
                        'avg_win_rate': float(result.get('avg_win_rate', 0)) if result.get('avg_win_rate') else 0,
                        'total_pips': float(result.get('total_pips', 0)) if result.get('total_pips') else 0,
                        'avg_profit_factor': float(result.get('avg_profit_factor', 0)) if result.get('avg_profit_factor') else 0,
                        'data_quality': float(result.get('data_quality', 0)) if result.get('data_quality') else 0
                    }
                else:
                    return {}

            except Exception as db_error:
                if "This result object does not return rows" in str(db_error):
                    # This should not happen with a SELECT query, but handle it gracefully
                    self.logger.warning("Database query completed but result object issue encountered")
                    return {}
                else:
                    raise db_error

        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

    def _parse_timeframe_to_timedelta(self, timeframe: str) -> timedelta:
        """Convert timeframe string to timedelta"""
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            return timedelta(minutes=minutes)
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            return timedelta(hours=hours)
        elif timeframe.endswith('d'):
            days = int(timeframe[:-1])
            return timedelta(days=days)
        else:
            # Default to 15 minutes
            return timedelta(minutes=15)

    def get_backtest_statistics(self) -> Dict:
        """Get current backtest statistics"""
        return {
            'backtest_stats': self.backtest_stats.copy(),
            'execution_id': self.execution_id,
            'current_time': self.current_backtest_time,
            'progress': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'current_time': self.current_backtest_time
            },
            'order_logger_stats': self.order_logger.get_statistics()
        }


# Factory function for creating backtest scanner
def create_backtest_scanner(backtest_config: Dict,
                          db_manager: DatabaseManager = None,
                          **kwargs) -> BacktestScanner:
    """Create BacktestScanner instance"""
    return BacktestScanner(backtest_config, db_manager, **kwargs)