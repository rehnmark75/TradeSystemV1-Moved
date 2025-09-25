# core/backtest/historical_scanner_engine.py
"""
Historical Scanner Engine - Full Pipeline Backtesting with Real ig_candles Data
===============================================================================

This module provides backtesting that uses the complete production scanner pipeline
but operates on historical data from the ig_candles table. It steps through time
chronologically and logs trade decisions instead of executing them.

ARCHITECTURE:
- Uses real IntelligentForexScanner.scan_once() pipeline
- Queries actual ig_candles data with timestamp constraints
- Steps through time to simulate live scanning
- Logs complete trade decisions with full context
- Prevents lookahead bias by constraining data availability

BENEFITS:
- Identical logic to production trading
- Real historical data from ig_candles table
- Complete validation pipeline testing
- Realistic performance predictions
- Full decision-making process capture
"""

import logging
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
import time

try:
    from core.database import DatabaseManager
    from core.scanner import IntelligentForexScanner
    from core.trading.trading_orchestrator import TradingOrchestrator
    from core.data_fetcher import DataFetcher
    import config
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.scanner import IntelligentForexScanner
    from forex_scanner.core.trading.trading_orchestrator import TradingOrchestrator
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner import config


class HistoricalScannerEngine:
    """
    Historical Scanner Engine that runs the complete production scanner pipeline
    on historical data from the ig_candles table
    """

    def __init__(self,
                 db_manager: DatabaseManager = None,
                 scan_interval: int = None,
                 epic_list: List[str] = None,
                 user_timezone: str = 'Europe/Stockholm',
                 strategy_filter: str = None,
                 **kwargs):

        self.logger = logging.getLogger(__name__)

        # Database and configuration
        self.db_manager = db_manager or DatabaseManager(config.DATABASE_URL)
        self.scan_interval = scan_interval or getattr(config, 'SCAN_INTERVAL', 900)  # 15 minutes default
        self.epic_list = epic_list or getattr(config, 'EPIC_LIST', [])
        self.user_timezone = user_timezone
        self.strategy_filter = strategy_filter

        # Apply strategy filtering if specified
        if strategy_filter and strategy_filter != 'all':
            self._apply_strategy_filter(strategy_filter)

        # Verify config state right before scanner creation
        self.logger.debug("🔧 Pre-scanner config verification:")
        key_configs = ['EMA_STRATEGY', 'MOMENTUM_STRATEGY', 'MACD_EMA_STRATEGY', 'SMC_STRATEGY', 'ICHIMOKU_CLOUD_STRATEGY']
        for cfg in key_configs:
            if hasattr(config, cfg):
                self.logger.debug(f"   {cfg} = {getattr(config, cfg)}")

        # Initialize scanner with historical mode indicators
        self.scanner = IntelligentForexScanner(
            db_manager=self.db_manager,
            epic_list=self.epic_list,
            scan_interval=self.scan_interval,
            user_timezone=user_timezone,
            intelligence_mode='backtest_consistent',  # Use backtest-optimized mode
            enable_market_intelligence=False,  # Disable market intelligence storage during backtesting
            disable_alert_history=True  # Disable ALL database writes during backtesting
        )

        # Initialize trading orchestrator for decision logging
        self.trading_orchestrator = None
        self._initialize_trading_orchestrator()

        # Historical scanning state
        self.historical_mode = True
        self.current_historical_timestamp = None
        self.backtest_trades = []
        self.scan_statistics = {
            'total_scans': 0,
            'signals_detected': 0,
            'trades_approved': 0,
            'trades_rejected': 0,
            'processing_time': 0
        }

        self.logger.info(f"🕐 HistoricalScannerEngine initialized")
        self.logger.info(f"   Epic list: {len(self.epic_list)} pairs")
        self.logger.info(f"   Scan interval: {self.scan_interval}s ({self.scan_interval/60}min)")
        self.logger.info(f"   Historical mode: {self.historical_mode}")

    def _initialize_trading_orchestrator(self):
        """Initialize trading orchestrator for decision logging"""
        try:
            # Check if TradingOrchestrator is available
            if TradingOrchestrator:
                self.trading_orchestrator = TradingOrchestrator(
                    db_manager=self.db_manager,
                    enable_trading=False,  # Disable actual trading
                    enable_market_intelligence=False,  # Disable market intelligence storage during backtesting
                    enable_claude_analysis=getattr(config, 'ENABLE_CLAUDE_ANALYSIS', False),
                    user_timezone=self.user_timezone,
                    # CRITICAL: Disable ALL database writes during backtesting
                    disable_alert_history=True  # Disable alert_history saves during backtesting
                )
                self.logger.info("✅ TradingOrchestrator initialized for decision logging")
            else:
                self.logger.warning("⚠️ TradingOrchestrator not available - trade decisions won't be logged")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not initialize TradingOrchestrator: {e}")
            self.trading_orchestrator = None

    def get_available_historical_timestamps(self,
                                          start_date: datetime,
                                          end_date: datetime,
                                          timeframe: str = '15m') -> List[datetime]:
        """
        Get available timestamps from ig_candles table for the specified period

        Args:
            start_date: Start of historical period
            end_date: End of historical period
            timeframe: Timeframe to scan (15m, 1h, etc.)

        Returns:
            List of available timestamps for scanning
        """
        try:
            # Query distinct timestamps from preferred_forex_prices table (primary data source)
            # Note: Using start_time column and integer timeframe values
            timeframe_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}.get(timeframe, 15)

            # Use DataFetcher to get available timestamps (handles timeframe synthesis automatically)
            timestamps = []

            if self.epic_list:
                sample_epic = self.epic_list[0]
                pair = sample_epic.split('.')[2] if '.' in sample_epic else sample_epic

                try:
                    # Create temporary DataFetcher
                    from core.data_fetcher import DataFetcher
                except ImportError:
                    from forex_scanner.core.data_fetcher import DataFetcher

                temp_data_fetcher = DataFetcher(self.db_manager)
                temp_data_fetcher.historical_max_timestamp = end_date

                # Get data using DataFetcher
                df = temp_data_fetcher.get_enhanced_data(
                    epic=sample_epic,
                    pair=pair,
                    timeframe=timeframe,
                    lookback_hours=int((end_date - start_date).total_seconds() / 3600) + 24
                )

                if df is not None and not df.empty:
                    try:
                        # Skip DataFetcher if timestamps are clearly wrong (1970 epoch issue)
                        try:
                            min_timestamp = df.index.min()
                            # Handle both integer and datetime timestamps
                            if hasattr(min_timestamp, 'year') and min_timestamp.year < 2020:
                                timestamp_invalid = True
                            elif isinstance(min_timestamp, (int, float)) and min_timestamp < 1577836800:  # 2020-01-01 in Unix timestamp
                                timestamp_invalid = True
                            else:
                                timestamp_invalid = False
                        except:
                            timestamp_invalid = True

                        if timestamp_invalid:
                            self.logger.warning(f"⚠️ DataFetcher returned invalid timestamps, falling back to direct query")
                            timestamps = []
                        else:
                            # Convert index to datetime if needed
                            if not isinstance(df.index, pd.DatetimeIndex):
                                df.index = pd.to_datetime(df.index, unit='s', utc=True)

                            # Make start/end dates timezone-aware to match DataFrame index
                            if df.index.tz is not None:
                                if start_date.tzinfo is None:
                                    start_date = start_date.replace(tzinfo=df.index.tz)
                                if end_date.tzinfo is None:
                                    end_date = end_date.replace(tzinfo=df.index.tz)

                            # Filter timestamps to our date range
                            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
                            timestamps = df_filtered.index.to_list()
                            self.logger.info(f"✅ Using DataFetcher: Found {len(timestamps)} {timeframe} timestamps")
                    except Exception as e:
                        self.logger.error(f"❌ Error processing DataFetcher timestamps: {e}")
                        timestamps = []
                else:
                    self.logger.warning(f"⚠️ DataFetcher returned no data for {sample_epic}")

            # Fallback to direct database query if DataFetcher failed
            if not timestamps:
                self.logger.info(f"🔄 Falling back to direct database query with timeframe synthesis")
                query = """
                    SELECT DISTINCT start_time
                    FROM ig_candles
                    WHERE start_time BETWEEN %s AND %s
                    AND epic = ANY(%s)
                    AND timeframe = %s
                    ORDER BY start_time ASC
                """

                with self.db_manager.get_connection() as conn:
                    with conn.cursor() as cursor:
                        # Try requested timeframe first
                        cursor.execute(query, [start_date, end_date, self.epic_list, timeframe_minutes])
                        results = cursor.fetchall()
                        timestamps = [row[0] for row in results]

                        # If no results, try 5m and synthesize 15m timestamps
                        if not timestamps and timeframe_minutes >= 15:
                            cursor.execute(query, [start_date, end_date, self.epic_list, 5])
                            results_5m = cursor.fetchall()
                            if results_5m:
                                timestamps_5m = [row[0] for row in results_5m]
                                # Simple synthesis: take every 3rd timestamp for 15m from 5m
                                timestamps = [ts for i, ts in enumerate(timestamps_5m) if i % 3 == 0]
                                self.logger.info(f"✅ Synthesized {len(timestamps)} {timeframe} timestamps from {len(timestamps_5m)} 5m timestamps")

            # Filter by scan interval to simulate realistic scanning
            if timestamps:
                filtered_timestamps = []
                last_scan_time = None

                for ts in timestamps:
                    if last_scan_time is None or (ts - last_scan_time).total_seconds() >= self.scan_interval:
                        filtered_timestamps.append(ts)
                        last_scan_time = ts

                self.logger.info(f"📅 Found {len(filtered_timestamps)} scan timestamps ({len(timestamps)} total)")
                self.logger.info(f"🕐 Using timeframe: {timeframe} ({timeframe_minutes} minutes)")
                return filtered_timestamps

            # If still no results, provide diagnostic information
            if not timestamps:
                self.logger.warning(f"⚠️ No historical timestamps found")
                self.logger.info(f"🔍 Diagnostics:")
                self.logger.info(f"   Epics searched: {self.epic_list}")
                self.logger.info(f"   Date range: {start_date} to {end_date}")
                self.logger.info(f"   Timeframe: {timeframe} ({timeframe_minutes} minutes)")

                # Try to get any available data for these epics
                with self.db_manager.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT epic, MIN(start_time) as earliest, MAX(start_time) as latest, COUNT(*) as count
                            FROM ig_candles
                            WHERE epic = ANY(%s) AND timeframe = %s
                            GROUP BY epic
                        """, [self.epic_list, timeframe_minutes])

                        diagnostic_results = cursor.fetchall()
                        if diagnostic_results:
                            self.logger.info(f"   Available data in ig_candles:")
                            for epic, earliest, latest, count in diagnostic_results:
                                self.logger.info(f"     {epic}: {count:,} records from {earliest} to {latest}")
                        else:
                            self.logger.info(f"   No data found in ig_candles for these epics/timeframe")

            return []

        except Exception as e:
            self.logger.error(f"❌ Error getting historical timestamps: {e}")
            import traceback
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return []

    def run_historical_backtest(self,
                              start_date: datetime,
                              end_date: datetime,
                              timeframe: str = '15m') -> Dict:
        """
        Run complete historical backtest using the full scanner pipeline

        Args:
            start_date: Start of backtest period
            end_date: End of backtest period
            timeframe: Timeframe to analyze

        Returns:
            Dictionary with backtest results and statistics
        """
        self.logger.info(f"🚀 Starting Historical Scanner Pipeline Backtest")
        self.logger.info(f"📅 Period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"🎯 Epics: {len(self.epic_list)}")

        # Reset statistics
        self.scan_statistics = {
            'total_scans': 0,
            'signals_detected': 0,
            'trades_approved': 0,
            'trades_rejected': 0,
            'processing_time': 0,
            'start_time': datetime.now(),
            'period_start': start_date,
            'period_end': end_date
        }
        self.backtest_trades = []

        # Get available historical timestamps
        timestamps = self.get_available_historical_timestamps(start_date, end_date, timeframe)

        if not timestamps:
            self.logger.error("❌ No historical data available for the specified period")
            return {'success': False, 'error': 'No historical data available'}

        self.logger.info(f"⏳ Processing {len(timestamps)} historical scan points...")

        # Step through each timestamp chronologically
        for i, current_timestamp in enumerate(timestamps):
            scan_start = time.time()

            try:
                # Set current historical context
                self.current_historical_timestamp = current_timestamp

                # Constrain data fetcher to only see data up to current timestamp
                self._set_historical_constraint(current_timestamp)

                # Run the complete scanner pipeline at this timestamp
                signals = self._run_scanner_at_timestamp(current_timestamp)

                # Process signals through trading orchestrator
                if signals:
                    self._process_historical_signals(signals, current_timestamp)

                self.scan_statistics['total_scans'] += 1
                self.scan_statistics['signals_detected'] += len(signals) if signals else 0

                # Log progress periodically
                if (i + 1) % 50 == 0 or i == len(timestamps) - 1:
                    progress = ((i + 1) / len(timestamps)) * 100
                    self.logger.info(f"📊 Progress: {progress:.1f}% ({i+1}/{len(timestamps)}) - "
                                   f"{self.scan_statistics['signals_detected']} signals detected")

            except Exception as e:
                self.logger.error(f"❌ Error processing timestamp {current_timestamp}: {e}")
                continue

            scan_duration = time.time() - scan_start
            self.scan_statistics['processing_time'] += scan_duration

        # Generate final results
        results = self._generate_backtest_results()

        # Restore original strategy configuration
        self._restore_strategy_config()

        return results

    def _set_historical_constraint(self, max_timestamp: datetime):
        """
        Set historical constraint on data fetcher to prevent lookahead bias

        Args:
            max_timestamp: Maximum timestamp for data availability
        """
        # Modify the scanner's data fetcher to use historical constraint
        if hasattr(self.scanner.signal_detector, 'data_fetcher'):
            self.scanner.signal_detector.data_fetcher.historical_max_timestamp = max_timestamp

        # Also set constraint on any other data fetchers
        if hasattr(self.scanner, 'data_fetcher'):
            self.scanner.data_fetcher.historical_max_timestamp = max_timestamp

    def _run_scanner_at_timestamp(self, timestamp: datetime) -> List[Dict]:
        """
        Run the complete scanner pipeline at a specific historical timestamp

        Args:
            timestamp: Current historical timestamp

        Returns:
            List of signals detected at this timestamp
        """
        try:
            # Log current scan context
            self.logger.debug(f"🔍 Scanning at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

            # Add timeout and detailed logging for the scanner call
            self.logger.debug(f"   📞 Calling scanner.scan_once('historical')...")
            scan_start_time = time.time()

            # Run the complete scanner pipeline
            signals = self.scanner.scan_once('historical')

            scan_duration = time.time() - scan_start_time
            self.logger.debug(f"   ✅ Scanner completed in {scan_duration:.1f}s")

            if signals:
                self.logger.debug(f"   📊 {len(signals)} signals detected")

                # Add historical context to each signal
                for signal in signals:
                    signal['historical_timestamp'] = timestamp
                    signal['backtest_mode'] = True
                    signal['scan_type'] = 'historical'

            return signals

        except Exception as e:
            self.logger.error(f"❌ Error running scanner at {timestamp}: {e}")
            return []

    def _process_historical_signals(self, signals: List[Dict], timestamp: datetime):
        """
        Process signals through trading orchestrator for decision logging

        Args:
            signals: Detected signals
            timestamp: Current historical timestamp
        """
        for signal in signals:
            try:
                # Process signal through trading orchestrator if available
                if self.trading_orchestrator:
                    # This will log the trade decision instead of executing
                    trade_decision = self.trading_orchestrator.process_signal(
                        signal,
                        historical_mode=True
                    )

                    if trade_decision:
                        trade_decision['historical_timestamp'] = timestamp
                        self.backtest_trades.append(trade_decision)

                        if trade_decision.get('action') in ['BUY', 'SELL']:
                            self.scan_statistics['trades_approved'] += 1
                        else:
                            self.scan_statistics['trades_rejected'] += 1
                else:
                    # Fallback: log signal without full orchestrator processing
                    trade_decision = {
                        'historical_timestamp': timestamp,
                        'signal': signal,
                        'action': 'LOGGED_ONLY',
                        'reason': 'TradingOrchestrator not available'
                    }
                    self.backtest_trades.append(trade_decision)

            except Exception as e:
                self.logger.error(f"❌ Error processing signal: {e}")
                continue

    def _generate_backtest_results(self) -> Dict:
        """Generate comprehensive backtest results"""

        total_duration = (datetime.now() - self.scan_statistics['start_time']).total_seconds()

        results = {
            'success': True,
            'statistics': {
                **self.scan_statistics,
                'total_duration_seconds': total_duration,
                'avg_scan_time': self.scan_statistics['processing_time'] / max(1, self.scan_statistics['total_scans']),
                'signals_per_scan': self.scan_statistics['signals_detected'] / max(1, self.scan_statistics['total_scans']),
                'approval_rate': self.scan_statistics['trades_approved'] / max(1, self.scan_statistics['trades_approved'] + self.scan_statistics['trades_rejected'])
            },
            'trades': self.backtest_trades,
            'trade_summary': self._summarize_trades()
        }

        # Log summary
        self.logger.info(f"✅ Historical Backtest Complete!")
        self.logger.info(f"📊 Results Summary:")
        self.logger.info(f"   Total scans: {self.scan_statistics['total_scans']}")
        self.logger.info(f"   Signals detected: {self.scan_statistics['signals_detected']}")
        self.logger.info(f"   Trades approved: {self.scan_statistics['trades_approved']}")
        self.logger.info(f"   Trades rejected: {self.scan_statistics['trades_rejected']}")
        self.logger.info(f"   Approval rate: {results['statistics']['approval_rate']:.1%}")
        self.logger.info(f"   Duration: {total_duration:.1f}s")

        return results

    def _summarize_trades(self) -> Dict:
        """Summarize trade decisions by epic and action"""

        summary = {
            'by_epic': {},
            'by_action': {},
            'by_strategy': {},
            'total_trades': len(self.backtest_trades)
        }

        for trade in self.backtest_trades:
            # By epic
            epic = trade.get('signal', {}).get('epic', 'Unknown')
            if epic not in summary['by_epic']:
                summary['by_epic'][epic] = {'BUY': 0, 'SELL': 0, 'REJECT': 0}

            action = trade.get('action', 'REJECT')
            if action in ['BUY', 'SELL']:
                summary['by_epic'][epic][action] += 1
            else:
                summary['by_epic'][epic]['REJECT'] += 1

            # By action
            if action not in summary['by_action']:
                summary['by_action'][action] = 0
            summary['by_action'][action] += 1

            # By strategy
            strategy = trade.get('signal', {}).get('strategy', 'Unknown')
            if strategy not in summary['by_strategy']:
                summary['by_strategy'][strategy] = 0
            summary['by_strategy'][strategy] += 1

        return summary

    def _apply_strategy_filter(self, strategy_filter: str):
        """
        Apply strategy filtering by temporarily modifying config to enable only the specified strategy

        Args:
            strategy_filter: Strategy to enable ('momentum', 'macd', 'ema', 'zero_lag', 'combined', 'smc', 'ichimoku', 'mean_reversion', 'ranging_market')
        """
        # Strategy mapping to config variables
        strategy_config_map = {
            'momentum': ['MOMENTUM_STRATEGY'],  # Corrected: momentum_strategy not momentum_bias
            'macd': ['MACD_EMA_STRATEGY'],
            'ema': ['EMA_STRATEGY'],
            'zero_lag': ['ZERO_LAG_STRATEGY'],
            'combined': ['COMBINED_STRATEGY'],
            'smc': ['SMC_STRATEGY'],  # Smart Money Concepts strategy
            'ichimoku': ['ICHIMOKU_CLOUD_STRATEGY'],  # Ichimoku Kinko Hyo strategy
            'mean_reversion': ['MEAN_REVERSION_STRATEGY'],  # Multi-oscillator mean reversion strategy
            'ranging_market': ['RANGING_MARKET_STRATEGY']  # Multi-oscillator ranging market strategy
        }

        # Get the list of config variables to enable
        strategies_to_enable = strategy_config_map.get(strategy_filter, [])

        if not strategies_to_enable:
            self.logger.warning(f"⚠️ Unknown strategy filter: {strategy_filter}")
            return

        # Store original values for restoration later
        self.original_strategy_config = {}

        # List of all known strategy config variables
        all_strategy_configs = [
            'MOMENTUM_BIAS_STRATEGY',  # Keep this one too
            'MOMENTUM_STRATEGY',       # The actual momentum strategy
            'MACD_EMA_STRATEGY',
            'EMA_STRATEGY',
            'ZERO_LAG_STRATEGY',
            'KAMA_STRATEGY',
            'BOLLINGER_SUPERTREND_STRATEGY',
            'COMBINED_STRATEGY',
            'SMC_STRATEGY',
            'ICHIMOKU_CLOUD_STRATEGY',
            'MEAN_REVERSION_STRATEGY',
            'RANGING_MARKET_STRATEGY'
        ]

        # Disable all strategies first, then enable the filtered one
        for strategy_config in all_strategy_configs:
            if hasattr(config, strategy_config):
                # Store original value
                self.original_strategy_config[strategy_config] = getattr(config, strategy_config)

                # Disable by default
                setattr(config, strategy_config, False)

        # Enable only the filtered strategy
        for strategy_config in strategies_to_enable:
            if hasattr(config, strategy_config):
                setattr(config, strategy_config, True)
                self.logger.info(f"✅ Strategy filter applied: {strategy_filter} -> {strategy_config} = True")

        # Debug: Log final config state
        self.logger.debug("🔍 Final strategy config state:")
        for strategy_config in all_strategy_configs:
            if hasattr(config, strategy_config):
                value = getattr(config, strategy_config)
                self.logger.debug(f"   {strategy_config} = {value}")
            else:
                self.logger.debug(f"   {strategy_config} = NOT FOUND")

    def _restore_strategy_config(self):
        """Restore original strategy configuration"""
        if hasattr(self, 'original_strategy_config'):
            for strategy_config, original_value in self.original_strategy_config.items():
                setattr(config, strategy_config, original_value)
            self.logger.debug("🔄 Original strategy configuration restored")

    def save_results_to_database(self, results: Dict) -> bool:
        """
        Save backtest results to database for analysis

        Args:
            results: Backtest results dictionary

        Returns:
            Success status
        """
        try:
            # This will be implemented once we create the backtest_trades table
            self.logger.info("💾 Database saving will be implemented with backtest_trades table")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error saving results to database: {e}")
            return False