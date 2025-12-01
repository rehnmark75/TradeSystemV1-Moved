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
    from core.trading.trailing_stop_simulator import TrailingStopSimulator
    from configdata.strategies import config_momentum_strategy
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.scanner import IntelligentForexScanner
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.trading.backtest_order_logger import BacktestOrderLogger
    from forex_scanner.core.trading.trailing_stop_simulator import TrailingStopSimulator
    try:
        from forex_scanner.configdata.strategies import config_momentum_strategy
    except ImportError:
        config_momentum_strategy = None


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
        self.pipeline_mode = backtest_config.get('pipeline_mode', False)

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

        # Log pipeline mode
        mode_desc = "Full Pipeline (with validation)" if self.pipeline_mode else "Basic Strategy Testing"
        self.logger.info(f"🔧 Backtest Mode: {mode_desc}")

        if self.pipeline_mode:
            self.logger.info("   ✅ Trade validation enabled")
            self.logger.info("   ✅ Market intelligence enabled")
            self.logger.info("   ✅ Signal filtering enabled")
        else:
            self.logger.info("   ⚡ Fast strategy testing mode")
            self.logger.info("   ⚡ Minimal validation for parameter optimization")

        # Backtest-specific components
        self.order_logger = BacktestOrderLogger(
            self.db_manager,
            self.execution_id,
            logger=self.logger
        )

        # ============================================================================
        # STANDARD TRAILING STOP COMPONENT - Uses pair-specific config from config_trailing_stops.py
        # ============================================================================
        # The trailing stop is now a STANDARD component that uses pair-specific
        # Progressive 3-Stage configuration. Each pair has its own optimized settings.
        # This ensures consistency between backtest and live trading systems.

        # Import the factory function for creating per-epic simulators
        try:
            from core.trading.trailing_stop_simulator import create_trailing_stop_simulator
        except ImportError:
            from forex_scanner.core.trading.trailing_stop_simulator import create_trailing_stop_simulator

        # Cache for per-epic trailing stop simulators
        self._trailing_stop_simulators = {}
        self._create_trailing_stop_simulator = create_trailing_stop_simulator

        # 🔥 SCALPING-SPECIFIC CONFIGURATION: Still uses fixed SL/TP (no trailing)
        self._use_scalping_mode = 'SCALPING' in self.strategy_name.upper()
        if self._use_scalping_mode:
            try:
                from configdata.strategies.config_scalping_strategy import SCALPING_MODE, SCALPING_STRATEGY_CONFIG
                scalping_config = SCALPING_STRATEGY_CONFIG.get(SCALPING_MODE, {})
                self._scalping_target_pips = scalping_config.get('target_pips', 8.0)
                self._scalping_stop_pips = scalping_config.get('stop_loss_pips', 4.0)
                self._scalping_max_bars = scalping_config.get('max_bars', 10000)
                self.logger.info(f"📊 Scalping Exit Rules: PURE FIXED SL/TP - Target={self._scalping_target_pips} pips, Stop={self._scalping_stop_pips} pips")
                self.logger.info(f"   ⚠️ NO trailing, NO breakeven, NO time exits - trades MUST hit SL or TP")
            except ImportError:
                self._scalping_target_pips = 8.0
                self._scalping_stop_pips = 6.0
                self._scalping_max_bars = 10000
                self.logger.warning(f"⚠️ Could not load scalping config, using fallback")
        else:
            # Standard strategies use Progressive 3-Stage trailing from config_trailing_stops.py
            self.logger.info(f"📊 Trailing Stop: Using pair-specific Progressive 3-Stage system from config_trailing_stops.py")

        # Backward compatibility: keep a default simulator for legacy code
        self.trailing_stop_simulator = TrailingStopSimulator(
            target_pips=30.0,
            initial_stop_pips=20.0,
            max_bars=96,
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

        # CRITICAL VALIDATION: Log lookback configuration alignment
        self._validate_lookback_alignment()

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

            # CRITICAL FIX: Also configure MACD strategy for backtest mode
            if hasattr(self.signal_detector, 'macd_strategy') and self.signal_detector.macd_strategy:
                self.signal_detector.macd_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.macd_strategy.backtest_mode = True
                self.signal_detector.macd_strategy.use_optimized_parameters = True
                self.logger.info("✅ MACD strategy configured for backtest mode - will use optimal parameters and backtest filtering")

            # CRITICAL FIX: Configure EMA strategy for pipeline mode
            if hasattr(self.signal_detector, 'ema_strategy') and self.signal_detector.ema_strategy:
                self.signal_detector.ema_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.ema_strategy.backtest_mode = True
                # Set enhanced validation based on pipeline mode
                self.signal_detector.ema_strategy.enhanced_validation = self.pipeline_mode and getattr(config, 'EMA_ENHANCED_VALIDATION', True)
                # Disable breakout validator in basic mode for performance
                if not self.pipeline_mode:
                    self.signal_detector.ema_strategy.breakout_validator = None
                if self.signal_detector.ema_strategy.enhanced_validation:
                    self.logger.info("✅ EMA strategy configured for PIPELINE mode - enhanced validation enabled")
                else:
                    self.logger.info("✅ EMA strategy configured for BASIC mode - enhanced validation disabled for fast testing")

            # CRITICAL FIX: Configure SMC strategy for pipeline mode
            if hasattr(self.signal_detector, 'smc_strategy') and self.signal_detector.smc_strategy:
                self.signal_detector.smc_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.smc_strategy.backtest_mode = True
                # CRITICAL: Reinitialize MTF analyzer with backtest data_fetcher
                if hasattr(self.signal_detector.smc_strategy, 'reinitialize_mtf_analyzer'):
                    self.signal_detector.smc_strategy.reinitialize_mtf_analyzer(backtest_data_fetcher)
                # Set enhanced validation based on pipeline mode
                self.signal_detector.smc_strategy.enhanced_validation = self.pipeline_mode and getattr(config, 'SMC_ENHANCED_VALIDATION', True)
                if self.signal_detector.smc_strategy.enhanced_validation:
                    self.logger.info("✅ SMC strategy configured for PIPELINE mode - enhanced validation enabled")
                else:
                    self.logger.info("✅ SMC strategy configured for BASIC mode - enhanced validation disabled for fast testing")

            # CRITICAL FIX: Configure Ichimoku strategy for pipeline mode
            if hasattr(self.signal_detector, 'ichimoku_strategy') and self.signal_detector.ichimoku_strategy:
                self.signal_detector.ichimoku_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.ichimoku_strategy.backtest_mode = True
                # CRITICAL: Disable MTF analysis in backtest mode for performance and reliability
                self.signal_detector.ichimoku_strategy.enable_mtf_analysis = False
                # Set enhanced validation based on pipeline mode
                self.signal_detector.ichimoku_strategy.enhanced_validation = self.pipeline_mode and getattr(config, 'ICHIMOKU_ENHANCED_VALIDATION', True)
                # Disable expensive RAG features in basic mode
                if not self.pipeline_mode:
                    self.signal_detector.ichimoku_strategy.rag_enabled = False
                    self.signal_detector.ichimoku_strategy.market_intelligence_adapter = None
                if self.signal_detector.ichimoku_strategy.enhanced_validation:
                    self.logger.info("✅ Ichimoku strategy configured for PIPELINE mode - enhanced validation and RAG enabled, MTF disabled")
                else:
                    self.logger.info("✅ Ichimoku strategy configured for BASIC mode - enhanced validation, RAG, and MTF disabled for fast testing")

            # CRITICAL FIX: Configure Mean Reversion strategy for pipeline mode
            if hasattr(self.signal_detector, 'mean_reversion_strategy') and self.signal_detector.mean_reversion_strategy:
                self.signal_detector.mean_reversion_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.mean_reversion_strategy.backtest_mode = True
                # Set enhanced validation based on pipeline mode
                self.signal_detector.mean_reversion_strategy.enhanced_validation = self.pipeline_mode and getattr(config, 'MEAN_REVERSION_ENHANCED_VALIDATION', True)
                if self.signal_detector.mean_reversion_strategy.enhanced_validation:
                    self.logger.info("✅ Mean Reversion strategy configured for PIPELINE mode - enhanced validation enabled")
                else:
                    self.logger.info("✅ Mean Reversion strategy configured for BASIC mode - enhanced validation disabled for fast testing")

            # CRITICAL FIX: Configure Ranging Market strategy for pipeline mode
            if hasattr(self.signal_detector, 'ranging_market_strategy') and self.signal_detector.ranging_market_strategy:
                self.signal_detector.ranging_market_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.ranging_market_strategy.backtest_mode = True
                # Set enhanced validation based on pipeline mode
                self.signal_detector.ranging_market_strategy.enhanced_validation = self.pipeline_mode and getattr(config, 'RANGING_MARKET_ENHANCED_VALIDATION', True)
                if self.signal_detector.ranging_market_strategy.enhanced_validation:
                    self.logger.info("✅ Ranging Market strategy configured for PIPELINE mode - enhanced validation enabled")
                else:
                    self.logger.info("✅ Ranging Market strategy configured for BASIC mode - enhanced validation disabled for fast testing")

            # CRITICAL FIX: Configure Zero Lag strategy for pipeline mode
            if hasattr(self.signal_detector, 'zero_lag_strategy') and self.signal_detector.zero_lag_strategy:
                self.signal_detector.zero_lag_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.zero_lag_strategy.backtest_mode = True
                # Set enhanced validation based on pipeline mode
                self.signal_detector.zero_lag_strategy.enhanced_validation = self.pipeline_mode and getattr(config, 'ZERO_LAG_ENHANCED_VALIDATION', True)
                if self.signal_detector.zero_lag_strategy.enhanced_validation:
                    self.logger.info("✅ Zero Lag strategy configured for PIPELINE mode - enhanced validation enabled")
                else:
                    self.logger.info("✅ Zero Lag strategy configured for BASIC mode - enhanced validation disabled for fast testing")

            # CRITICAL FIX: Configure Momentum strategy for pipeline mode
            if hasattr(self.signal_detector, 'momentum_strategy') and self.signal_detector.momentum_strategy:
                self.signal_detector.momentum_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.momentum_strategy.backtest_mode = True
                # NOTE: pipeline_mode only controls TradeValidator, not strategy behavior
                self.logger.info("✅ Momentum strategy configured for backtest mode")

            # ADDED: Configure BB+Supertrend strategy for pipeline mode
            if hasattr(self.signal_detector, 'bb_supertrend_strategy') and self.signal_detector.bb_supertrend_strategy:
                self.signal_detector.bb_supertrend_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.bb_supertrend_strategy.backtest_mode = True
                # Set enhanced validation based on pipeline mode
                self.signal_detector.bb_supertrend_strategy.enhanced_validation = self.pipeline_mode and getattr(config, 'BB_SUPERTREND_ENHANCED_VALIDATION', True)
                if self.signal_detector.bb_supertrend_strategy.enhanced_validation:
                    self.logger.info("✅ BB+Supertrend strategy configured for PIPELINE mode - enhanced validation enabled")
                else:
                    self.logger.info("✅ BB+Supertrend strategy configured for BASIC mode - enhanced validation disabled for fast testing")

            # ADDED: Configure KAMA strategy for pipeline mode
            if hasattr(self.signal_detector, 'kama_strategy') and self.signal_detector.kama_strategy:
                self.signal_detector.kama_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.kama_strategy.backtest_mode = True
                # Set enhanced validation based on pipeline mode
                self.signal_detector.kama_strategy.enhanced_validation = self.pipeline_mode and getattr(config, 'KAMA_ENHANCED_VALIDATION', True)
                if self.signal_detector.kama_strategy.enhanced_validation:
                    self.logger.info("✅ KAMA strategy configured for PIPELINE mode - enhanced validation enabled")
                else:
                    self.logger.info("✅ KAMA strategy configured for BASIC mode - enhanced validation disabled for fast testing")

            # ADDED: Configure Scalping strategy for pipeline mode
            if hasattr(self.signal_detector, 'scalping_strategy') and self.signal_detector.scalping_strategy:
                self.signal_detector.scalping_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.scalping_strategy.backtest_mode = True
                # Set enhanced validation based on pipeline mode
                self.signal_detector.scalping_strategy.enhanced_validation = self.pipeline_mode and getattr(config, 'SCALPING_ENHANCED_VALIDATION', True)
                if self.signal_detector.scalping_strategy.enhanced_validation:
                    self.logger.info("✅ Scalping strategy configured for PIPELINE mode - enhanced validation enabled")
                else:
                    self.logger.info("✅ Scalping strategy configured for BASIC mode - enhanced validation disabled for fast testing")

            # CRITICAL FIX: Also configure any existing MACD strategies in cache for backtest mode
            if hasattr(self.signal_detector, 'macd_strategies_cache') and self.signal_detector.macd_strategies_cache:
                for epic, strategy in self.signal_detector.macd_strategies_cache.items():
                    if strategy:
                        strategy.data_fetcher = backtest_data_fetcher
                        strategy.backtest_mode = True
                        strategy.use_optimized_parameters = True
                self.logger.info(f"✅ {len(self.signal_detector.macd_strategies_cache)} cached MACD strategies configured for backtest mode")

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
        """
        Create iterator for backtest time periods

        CRITICAL: Skips weekends (Saturday/Sunday) when forex market is closed
        This prevents false signals from being generated on non-trading days
        """
        current_time = self.start_date

        while current_time <= self.end_date:
            # Skip weekends: Saturday (5) and Sunday (6)
            # Forex market is closed from Friday 22:00 UTC to Sunday 22:00 UTC
            day_of_week = current_time.weekday()

            if day_of_week == 5:  # Saturday
                self.logger.debug(f"⏭️  Skipping Saturday: {current_time}")
                current_time += self.time_increment
                continue
            elif day_of_week == 6:  # Sunday
                self.logger.debug(f"⏭️  Skipping Sunday: {current_time}")
                current_time += self.time_increment
                continue

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

        # 🔧 CRITICAL FIX: Set backtest time BEFORE fetching data
        # This ensures BacktestDataFetcher only returns data UP TO this timestamp
        if hasattr(self.signal_detector, 'data_fetcher') and hasattr(self.signal_detector.data_fetcher, 'set_backtest_time'):
            self.signal_detector.data_fetcher.set_backtest_time(timestamp)
            self.logger.debug(f"⏰ Backtest time set to: {timestamp}")

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
                'SMC_STRUCTURE': 'detect_smc_structure_signals',
                'SMC_PURE': 'detect_smc_structure_signals',
                'SMC_SIMPLE': 'detect_smc_simple_signals',
                'SMC_EMA': 'detect_smc_simple_signals',  # Alternative name
                'ICHIMOKU': 'detect_ichimoku_signals',
                'ICHIMOKU_CLOUD': 'detect_ichimoku_signals',
                'MEAN_REVERSION': 'detect_mean_reversion_signals',
                'MEANREV': 'detect_mean_reversion_signals',  # Fixed: Accept short form
                'RANGING_MARKET': 'detect_ranging_market_signals',
                'RANGING': 'detect_ranging_market_signals',
                'SCALPING': 'detect_scalping_signals',  # ADDED: Scalping strategy support
                'SCALP': 'detect_scalping_signals',  # ADDED: Short form
                'VOLUME_PROFILE': 'detect_volume_profile_signals',  # ADDED: Volume Profile strategy
                'VP': 'detect_volume_profile_signals'  # ADDED: Short form
            }

            # Use specific strategy if requested, otherwise use all strategies
            if strategy_name in strategy_methods:
                method_name = strategy_methods[strategy_name]
                if hasattr(self.signal_detector, method_name):
                    self.logger.debug(f"🎯 Running {strategy_name} strategy only for {epic}")
                    method = getattr(self.signal_detector, method_name)

                    # Different strategies have different signatures
                    # EMA strategies don't take spread_pips, others do
                    if method_name in ['detect_signals_mid_prices']:
                        signals = method(epic, pair_name, self.timeframe)
                    else:
                        # 🔒 HARD-CODED: MACD and SMC strategies always use 1H timeframe
                        strategy_timeframe = '1h' if strategy_name in ['MACD', 'MACD_EMA', 'SMC_STRUCTURE', 'SMC_PURE', 'SMC_SIMPLE', 'SMC_EMA'] else self.timeframe
                        signals = method(epic, pair_name, self.spread_pips, strategy_timeframe)

                    # Some methods return a single signal dict, others return lists
                    if signals and not isinstance(signals, list):
                        signals = [signals]

                    signal = signals[0] if signals else None

                    # Pipeline mode: Run through full validation and processing
                    if signal and self.pipeline_mode:
                        self.logger.debug(f"🔄 PIPELINE MODE ACTIVE: Processing {strategy_name} signal for {epic} through full validation pipeline")
                        signal = self._apply_full_pipeline(signal, epic, timestamp)
                    elif signal:
                        # Basic mode: Mark as validated by default (no validator used)
                        signal['validation_passed'] = True
                        signal['validation_message'] = 'Basic mode - no validation applied'
                        self.logger.debug(f"🚀 BASIC MODE: Skipping trade validator for {strategy_name} signal on {epic}")

                    return signal
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

            # Pipeline mode: Run through full validation and processing for fallback signals
            if signals and self.pipeline_mode:
                self.logger.info(f"🔄 PIPELINE MODE ACTIVE: Processing fallback signals for {epic} through full validation pipeline")
                # For multiple signals, process each one
                if isinstance(signals, list):
                    signals = [self._apply_full_pipeline(signal, epic, timestamp) for signal in signals if signal]
                    signals = [s for s in signals if s]  # Remove None results
                else:
                    signals = self._apply_full_pipeline(signals, epic, timestamp)
            elif signals:
                self.logger.info(f"🚀 BASIC MODE: Skipping trade validator and market intelligence for fallback signals on {epic}")

            return signals

        except Exception as e:
            self.logger.error(f"Error detecting signals for {epic} at {timestamp}: {e}")
            return None

    def _apply_full_pipeline(self, signal: Dict, epic: str, timestamp: datetime) -> Optional[Dict]:
        """
        Apply full signal validation and processing pipeline

        This includes:
        - Trade validation (same as live trading)
        - Market intelligence analysis
        - Signal filtering and enhancement

        Returns None if signal doesn't pass validation
        """
        try:
            # Import components needed for full pipeline
            try:
                from .trading.trade_validator import TradeValidator
            except ImportError:
                from forex_scanner.core.trading.trade_validator import TradeValidator

            # Create TradeValidator if not exists
            if not hasattr(self, '_trade_validator'):
                self._trade_validator = TradeValidator(
                    logger=self.logger,
                    db_manager=self.db_manager,
                    backtest_mode=True
                )

            # Basic signal validation
            if not signal or not isinstance(signal, dict):
                return None

            # Ensure signal has required fields for validation
            if 'epic' not in signal:
                signal['epic'] = epic
            if 'timestamp' not in signal:
                signal['timestamp'] = timestamp.isoformat()

            # Apply trade validation (same logic as live trading)
            self.logger.debug(f"🔍 Pipeline: Validating signal for {epic}")
            is_valid, validation_message = self._trade_validator.validate_signal_for_trading(signal)

            # Add pipeline metadata for both accepted and rejected signals
            signal['pipeline_processed'] = True
            signal['validation_passed'] = is_valid
            signal['validation_message'] = validation_message or ('Passed full pipeline validation' if is_valid else 'Unknown rejection reason')

            if not is_valid:
                self.logger.debug(f"❌ Pipeline: Signal rejected - {validation_message}")
                # Return rejected signal with metadata for collection instead of None
                signal['rejected'] = True
                signal['rejection_reason'] = validation_message
                return signal

            self.logger.debug(f"✅ Pipeline: Signal validated for {epic}")
            return signal

        except Exception as e:
            self.logger.warning(f"⚠️ Pipeline processing failed for {epic}: {e}")
            # In case of pipeline failure, return original signal with metadata
            signal['pipeline_processed'] = False
            signal['pipeline_error'] = str(e)
            return signal

    def _process_backtest_signals(self, signals: List[Dict], timestamp: datetime) -> List[Dict]:
        """
        Process signals through the same pipeline as live scanner
        But add backtest-specific metadata and simulate trade outcomes
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

                # Add trailing stop simulation for realistic trade outcomes
                enhanced_signal = self._add_trade_simulation(processed_signal, timestamp)

                processed_signals.append(enhanced_signal)

            except Exception as e:
                self.logger.error(f"Error processing backtest signal: {e}")
                continue

        return processed_signals

    def _get_trailing_stop_simulator(self, epic: str):
        """
        Get or create a per-epic trailing stop simulator with pair-specific config.

        This ensures each pair uses its optimized Progressive 3-Stage trailing stop
        settings from config_trailing_stops.py, matching the live trading system.

        Args:
            epic: Trading pair (e.g., 'CS.D.EURUSD.MINI.IP')

        Returns:
            TrailingStopSimulator configured for the specific pair
        """
        # Check cache first
        if epic in self._trailing_stop_simulators:
            return self._trailing_stop_simulators[epic]

        # Create new simulator with pair-specific config
        if self._use_scalping_mode:
            # Scalping uses fixed SL/TP, no trailing
            simulator = TrailingStopSimulator(
                epic=epic,
                target_pips=self._scalping_target_pips,
                initial_stop_pips=self._scalping_stop_pips,
                max_bars=self._scalping_max_bars,
                use_fixed_sl_tp=True,
                logger=self.logger
            )
            self.logger.debug(f"📊 Created SCALPING simulator for {epic}: SL={self._scalping_stop_pips}, TP={self._scalping_target_pips}")
        else:
            # Standard strategies use Progressive 3-Stage from config_trailing_stops.py
            # The factory function auto-loads pair config
            simulator = self._create_trailing_stop_simulator(epic=epic, logger=self.logger)
            self.logger.debug(f"📊 Created Progressive 3-Stage simulator for {epic}")

        # Cache for reuse
        self._trailing_stop_simulators[epic] = simulator
        return simulator

    def _add_trade_simulation(self, signal: Dict, signal_timestamp: datetime) -> Dict:
        """
        Add trade simulation with trailing stop logic
        Fetches future price data and simulates trade execution

        IMPORTANT: Uses pair-specific trailing stop configuration from config_trailing_stops.py
        This ensures backtest results match live trading behavior.
        """
        try:
            epic = signal.get('epic')
            if not epic:
                self.logger.warning("Signal missing epic, skipping trade simulation")
                return signal

            # Get per-epic trailing stop simulator with pair-specific config
            trailing_simulator = self._get_trailing_stop_simulator(epic)

            # Fetch future price data for simulation
            future_df = self._fetch_future_price_data(epic, signal_timestamp, max_bars=trailing_simulator.max_bars)

            if future_df is None or len(future_df) == 0:
                self.logger.debug(f"No future data available for {epic} at {signal_timestamp}, skipping simulation")
                signal['trade_result'] = 'NO_FUTURE_DATA'
                signal['is_winner'] = False
                signal['is_loser'] = False
                signal['max_profit_pips'] = 0
                signal['max_loss_pips'] = 0
                return signal

            # Simulate trade with pair-specific trailing stop
            self.logger.debug(f"Simulating trade for {epic} with {len(future_df)} future bars (3-stage trailing)")
            enhanced_signal = trailing_simulator.simulate_trade(
                signal=signal,
                df=future_df,
                signal_idx=0  # Signal is at the start of the future data
            )

            self.logger.debug(f"Simulation complete: profit={enhanced_signal.get('max_profit_pips', 0):.1f}, "
                            f"loss={enhanced_signal.get('max_loss_pips', 0):.1f}, "
                            f"result={enhanced_signal.get('trade_result', 'UNKNOWN')}, "
                            f"stage={enhanced_signal.get('stage_reached', 0)}")

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"❌ Error adding trade simulation: {e}", exc_info=True)
            # Return original signal without simulation data
            return signal

    def _fetch_future_price_data(self, epic: str, signal_timestamp: datetime, max_bars: int = 96) -> Optional[pd.DataFrame]:
        """
        Fetch future price data for trade simulation using BacktestDataFetcher (in-memory cache)

        Args:
            epic: Epic code
            signal_timestamp: Signal timestamp (start of future data)
            max_bars: Maximum number of bars to fetch

        Returns:
            DataFrame with OHLC price data or None if no data available
        """
        try:
            # Use BacktestDataFetcher to get data from in-memory cache (same source as signal detection)
            data_fetcher = self.signal_detector.data_fetcher

            # Extract pair from epic
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')

            # Calculate how much data we need (from NOW back to signal time, plus future bars)
            time_increment = self._parse_timeframe_to_timedelta(self.timeframe)
            end_timestamp = signal_timestamp + (time_increment * max_bars)
            if end_timestamp > self.end_date:
                end_timestamp = self.end_date

            # Calculate lookback from NOW (not from signal time!) to ensure we get enough data
            now = datetime.now(timezone.utc)

            # Ensure signal_timestamp is timezone-aware for math operations
            signal_ts_aware = signal_timestamp
            if isinstance(signal_timestamp, datetime) and signal_timestamp.tzinfo is None:
                signal_ts_aware = signal_timestamp.replace(tzinfo=timezone.utc)

            time_from_now_to_signal = now - signal_ts_aware
            time_from_signal_to_end = end_timestamp - signal_timestamp

            # We need data from (now - time_from_now_to_signal - time_from_signal_to_end)
            total_lookback = time_from_now_to_signal + time_from_signal_to_end + timedelta(hours=1)  # Add buffer
            lookback_hours = int(total_lookback.total_seconds() / 3600)

            # CRITICAL: Temporarily clear current_backtest_time to get unfiltered data
            # The BacktestDataFetcher filters data to current_backtest_time for realistic signals,
            # but for trailing stop simulation we need access to future bars
            original_backtest_time = getattr(data_fetcher, 'current_backtest_time', None)

            try:
                # Clear the time filter to get full historical data including "future" bars
                data_fetcher.current_backtest_time = None

                # Get full data range that includes our signal time AND future bars
                full_df = data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe=self.timeframe,
                    lookback_hours=lookback_hours,
                    ema_strategy=None  # Don't need strategy enhancements for simulation
                )
            finally:
                # Restore original backtest time
                if original_backtest_time is not None:
                    data_fetcher.current_backtest_time = original_backtest_time

            if full_df is None or full_df.empty:
                return None

            # Find the timestamp column
            timestamp_col = None
            for col in ['start_time', 'datetime_utc', 'timestamp', 'datetime']:
                if col in full_df.columns:
                    timestamp_col = col
                    break

            if timestamp_col is None:
                return None

            # Ensure timestamp column is datetime type
            full_df[timestamp_col] = pd.to_datetime(full_df[timestamp_col])

            # Ensure signal_timestamp is timezone-aware (match the DataFrame)
            if isinstance(signal_timestamp, datetime):
                if signal_timestamp.tzinfo is None and full_df[timestamp_col].dt.tz is not None:
                    # Make signal_timestamp timezone-aware (UTC)
                    signal_timestamp = signal_timestamp.replace(tzinfo=timezone.utc)
                elif signal_timestamp.tzinfo is not None and full_df[timestamp_col].dt.tz is None:
                    # Make DataFrame timezone-aware
                    full_df[timestamp_col] = full_df[timestamp_col].dt.tz_localize('UTC')

            # Filter for future data only (after signal timestamp)
            future_df = full_df[full_df[timestamp_col] > signal_timestamp].copy()

            # Limit to max_bars
            if len(future_df) > max_bars:
                future_df = future_df.head(max_bars)

            if len(future_df) == 0:
                return None

            # Rename timestamp column to 'timestamp' for consistency
            if timestamp_col != 'timestamp':
                future_df = future_df.rename(columns={timestamp_col: 'timestamp'})

            return future_df

        except Exception as e:
            self.logger.error(f"Error fetching future price data for {epic}: {e}")
            return None

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

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes (integer) for database queries"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            # Default to 15 minutes
            return 15

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

    def _validate_lookback_alignment(self):
        """
        CRITICAL VALIDATION: Verify backtest uses same lookback as live scanner
        This ensures optimization results are reliable
        """
        try:
            # Get optimal lookback hours using the same method as DataFetcher
            optimal_lookback = self._get_optimal_lookback_hours_from_config()

            # Calculate total backtest period in hours
            backtest_duration_hours = int((self.end_date - self.start_date).total_seconds() / 3600)

            self.logger.info("=" * 80)
            self.logger.info("📊 LOOKBACK ALIGNMENT VALIDATION")
            self.logger.info("=" * 80)
            self.logger.info(f"Timeframe: {self.timeframe}")
            self.logger.info(f"Optimal lookback (live scanner uses): {optimal_lookback}h")
            self.logger.info(f"Backtest period duration: {backtest_duration_hours}h")

            # Calculate expected number of bars for validation
            timeframe_minutes = self._timeframe_to_minutes(self.timeframe)
            expected_bars_per_scan = int((optimal_lookback * 60) / timeframe_minutes)

            self.logger.info(f"Expected bars per scan: {expected_bars_per_scan}")
            self.logger.info(f"Minimum bars required: {config.MIN_BARS_FOR_SIGNAL}")

            if expected_bars_per_scan < config.MIN_BARS_FOR_SIGNAL:
                self.logger.warning(
                    f"⚠️ WARNING: Lookback may be insufficient! "
                    f"Expected {expected_bars_per_scan} < MIN_BARS {config.MIN_BARS_FOR_SIGNAL}"
                )
            else:
                self.logger.info(f"✅ Lookback validation PASSED")

            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"❌ Error validating lookback alignment: {e}")

    def _get_optimal_lookback_hours_from_config(self) -> int:
        """
        Get optimal lookback hours using same logic as DataFetcher
        This is used for validation only
        """
        # Base lookback hours by timeframe (MUST MATCH DataFetcher._get_optimal_lookback_hours)
        base_lookback = {
            '1m': 24,    # 1 day for 1m (1440 bars)
            '5m': 48,    # 2 days for 5m (576 bars)
            '15m': 168,  # 1 week for 15m (672 bars)
            '1h': 720,   # 1 month for 1h (720 bars)
            '1d': 8760   # 1 year for 1d (365 bars)
        }.get(self.timeframe, 48)

        return base_lookback

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes (integer)"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            return 15  # Default to 15 minutes


# Factory function for creating backtest scanner
def create_backtest_scanner(backtest_config: Dict,
                          db_manager: DatabaseManager = None,
                          **kwargs) -> BacktestScanner:
    """Create BacktestScanner instance"""
    return BacktestScanner(backtest_config, db_manager, **kwargs)