# core/backtest_scanner.py
"""
BacktestScanner - Standalone historical backtesting scanner.

Composes SignalDetectionEngine instead of inheriting from IntelligentForexScanner.
Both scanners share the same detection logic through that engine; alert dedup,
order management, Claude integration and market-intelligence *capture* remain
exclusively in IntelligentForexScanner (live path).
"""

import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Iterator, Tuple
import pandas as pd

try:
    import config
    from core.database import DatabaseManager
    from core.signal_detector import SignalDetector
    from core.trading.backtest_order_logger import BacktestOrderLogger
    from core.trading.trailing_stop_simulator import TrailingStopSimulator
    from core.backtesting.backtest_trailing_engine import BacktestTrailingEngine, BACKTEST_USE_LEGACY_SIMULATOR
    from core.backtest_candles_manager import BacktestCandlesManager
    from core.scanning.signal_detection_engine import SignalDetectionEngine
    from services.smc_simple_config_service import get_smc_simple_config
    from services.scanner_config_service import get_scanner_config
    try:
        from configdata.strategies import config_momentum_strategy
    except ImportError:
        config_momentum_strategy = None
    try:
        from services.strategy_router_service import get_strategy_router, StrategyRouterService
    except ImportError:
        get_strategy_router = None
        StrategyRouterService = None
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.signal_detector import SignalDetector
    from forex_scanner.core.trading.backtest_order_logger import BacktestOrderLogger
    from forex_scanner.core.trading.trailing_stop_simulator import TrailingStopSimulator
    from forex_scanner.core.backtesting.backtest_trailing_engine import BacktestTrailingEngine, BACKTEST_USE_LEGACY_SIMULATOR
    from forex_scanner.core.backtest_candles_manager import BacktestCandlesManager
    from forex_scanner.core.scanning.signal_detection_engine import SignalDetectionEngine
    from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
    from forex_scanner.services.scanner_config_service import get_scanner_config
    try:
        from forex_scanner.configdata.strategies import config_momentum_strategy
    except ImportError:
        config_momentum_strategy = None
    try:
        from forex_scanner.services.strategy_router_service import get_strategy_router, StrategyRouterService
    except ImportError:
        get_strategy_router = None
        StrategyRouterService = None


class BacktestScanner:
    """
    Historical backtesting scanner.

    Composes a SignalDetectionEngine (shared with the live scanner path) instead
    of inheriting from IntelligentForexScanner.  Alert dedup, order placement,
    Claude AI analysis and market-intelligence capture are NOT present here —
    they belong exclusively to the live scanner.
    """

    def __init__(self,
                 backtest_config: Dict,
                 db_manager: DatabaseManager = None,
                 config_override: dict = None,
                 use_historical_intelligence: bool = True,
                 **kwargs):

        # Set up backtest-specific configuration
        self.backtest_config = backtest_config
        self.backtest_mode = True

        # Historical intelligence replay support (Phase 3)
        self._use_historical_intelligence = use_historical_intelligence
        self._historical_intelligence_cache = {}  # Cache for loaded intelligence
        self._intelligence_history_manager = None  # Lazy-loaded

        # Store config override for backtest parameter isolation
        # Must be set before _initialize_signal_detector is called
        self._config_override = config_override

        # Extract backtest parameters
        self.start_date = backtest_config['start_date']
        self.end_date = backtest_config['end_date']
        self.execution_id = backtest_config['execution_id']
        self.strategy_name = backtest_config['strategy_name']
        self.timeframe = backtest_config.get('timeframe', '15m')
        self.pipeline_mode = backtest_config.get('pipeline_mode', False)

        # Multi-strategy configuration (v3.0.0) - initialized after parent __init__ for logger access
        self._multi_strategy_config = backtest_config.get('multi_strategy_config', None)
        self._strategy_router = None

        # CRITICAL FIX: Skip signal logging when orchestrator handles logging
        # When True, scanner returns signals but doesn't log to DB (orchestrator does it)
        self.skip_signal_logging = backtest_config.get('skip_signal_logging', False)

        # ── Attributes previously provided by IntelligentForexScanner.__init__ ──
        # Set up logging first (parent did this)
        self.logger = logging.getLogger(__name__)

        # Load scanner + SMC config from DB (same path the live scanner uses)
        self._scanner_cfg = get_scanner_config()
        self._smc_cfg = get_smc_simple_config()

        # Core configuration
        self.db_manager = db_manager
        requested_epics = backtest_config.get('epics', None)
        base_epics = requested_epics or self._smc_cfg.get_effective_enabled_pairs()

        # Inject XAU_GOLD epics only for XAU/GOLD backtests. Single-epic or
        # SMC_SIMPLE FX tests must remain scoped to the requested pair.
        try:
            from services.xau_gold_config_service import get_xau_gold_config
        except ImportError:
            try:
                from forex_scanner.services.xau_gold_config_service import get_xau_gold_config
            except ImportError:
                get_xau_gold_config = None

        gold_epics: List[str] = []
        strategy_upper = (self.strategy_name or '').upper()
        include_xau_epics = (
            requested_epics is None
            and ('XAU' in strategy_upper or 'GOLD' in strategy_upper)
        )
        if include_xau_epics and get_xau_gold_config is not None:
            try:
                xau_cfg = get_xau_gold_config()
                gold_epics = [e for e in xau_cfg.enabled_pairs if e not in base_epics]
            except Exception:
                pass

        self.epic_list = base_epics + gold_epics
        self.min_confidence = (
            kwargs.get('min_confidence')
            or (self._scanner_cfg.min_confidence if self._scanner_cfg else 0.55)
        )
        self.spread_pips = kwargs.get('spread_pips', 1.5)
        self.scan_interval = 0  # No continuous scanning in backtest mode
        self.user_timezone = kwargs.get('user_timezone', 'Europe/Stockholm')
        self.intelligence_mode = 'backtest_consistent'
        self.scanner_version = 'backtest_v1.0_integrated_pipeline'

        # Stub attributes referenced by generic code that queries these
        self.enable_market_intelligence = False
        self.market_intelligence_engine = None
        self.market_intelligence_history = None
        self.enable_scan_performance = False
        self.scan_performance_manager = None
        self.enable_deduplication = False
        self.deduplication_manager = None
        self.signal_processor = None
        self.use_signal_processor = False
        self.enable_smart_money = False
        self.running = False
        self.last_signals = {}
        self.stats = {
            'scans_completed': 0,
            'signals_detected': 0,
            'signals_processed': 0,
            'errors': 0,
        }

        # Initialise the SignalDetector with config_override for parameter isolation
        self.signal_detector = self._initialize_signal_detector(db_manager, self.user_timezone)

        # Compose the shared detection engine
        self.signal_engine = SignalDetectionEngine(
            db_manager=self.db_manager,
            signal_detector=self.signal_detector,
            epic_list=self.epic_list,
            spread_pips=self.spread_pips,
            logger=self.logger,
        )

        self.logger.info("[CONFIG:DB] ✅ BacktestScanner config loaded from database (NO FALLBACK)")
        self.logger.info("🧠 Market Intelligence CAPTURE: DISABLED (backtest mode)")
        self.logger.info("📊 Scan Performance CAPTURE: DISABLED (backtest mode)")

        # Initialize multi-strategy router (v3.0.0) - now that logger is available
        if self._multi_strategy_config and self._multi_strategy_config.get('enabled'):
            try:
                self._strategy_router = get_strategy_router(is_backtest=True) if get_strategy_router else None
                if self._strategy_router:
                    self.logger.info("🎯 Multi-Strategy Mode: ENABLED")
                    enabled_strategies = self._multi_strategy_config.get('enable_strategies')
                    if enabled_strategies:
                        self.logger.info(f"   Strategies: {enabled_strategies}")
                    regime_filter = self._multi_strategy_config.get('regime_filter')
                    if regime_filter:
                        self.logger.info(f"   Regime filter: {regime_filter}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize strategy router: {e}")
                self._strategy_router = None

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
        # TRAILING STOP ENGINE — single source of truth (live config via mount)
        # ============================================================================
        # BacktestTrailingEngine reads PAIR_TRAILING_CONFIGS / SCALP_TRAILING_CONFIGS
        # from dev-app/config.py (mounted as trailing_config_live.py).  This is the
        # exact same dict the live fastapi-dev container uses, so backtest and live
        # trailing behaviour are always in sync.
        #
        # Set env BACKTEST_USE_LEGACY_SIMULATOR=1 to A/B compare against old engine.

        # Cache for per-epic trailing engines
        self._trailing_stop_simulators = {}

        # Scalp flag — when True the engine selects SCALP_TRAILING_CONFIGS and runs
        # full progressive trailing (old code used fixed SL/TP, which was wrong).
        # bt.py --scalp puts scalp_mode_enabled=True into config_override (not backtest_config).
        # Live SMC_SIMPLE orders also infer this from the DB config in order_executor.py;
        # keep backtests aligned so --timeframe 5m matches demo/live exit handling.
        smc_config_scalp_enabled = False
        if 'SMC_SIMPLE' in self.strategy_name.upper() or self.strategy_name.upper() == 'SMC':
            try:
                smc_config_scalp_enabled = bool(getattr(get_smc_simple_config(), 'scalp_mode_enabled', False))
            except Exception as exc:
                self.logger.debug(f"SMC scalp mode config lookup failed; using explicit override only: {exc}")

        self._use_scalping_mode = (
            'SCALPING' in self.strategy_name.upper()
            or bool(self._config_override and self._config_override.get('scalp_mode_enabled'))
            or smc_config_scalp_enabled
        )

        if BACKTEST_USE_LEGACY_SIMULATOR:
            self.logger.warning("⚠️ [LEGACY] BACKTEST_USE_LEGACY_SIMULATOR=1 — using old TrailingStopSimulator for A/B comparison")
            try:
                from core.trading.trailing_stop_simulator import create_trailing_stop_simulator
            except ImportError:
                from forex_scanner.core.trading.trailing_stop_simulator import create_trailing_stop_simulator
            self._create_trailing_stop_simulator = create_trailing_stop_simulator
            self._use_atr_trailing = bool(self._config_override and self._config_override.get('use_atr_trailing'))
        else:
            self._create_trailing_stop_simulator = None  # unused when new engine active
            self._use_atr_trailing = False

        scalp_label = "SCALP (progressive trailing, SCALP_TRAILING_CONFIGS)" if self._use_scalping_mode else "STANDARD (PAIR_TRAILING_CONFIGS)"
        self.logger.info(f"📊 Trailing Engine: BacktestTrailingEngine — {scalp_label}")
        if BACKTEST_USE_LEGACY_SIMULATOR:
            pass  # logged above
        else:
            self.logger.info(f"   spread={getattr(self, '_spread_pips', 1.5)} pip | slippage=0.5 pip applied at entry")

        # Backward-compat default engine (used by any legacy code that reads .trailing_stop_simulator)
        self.trailing_stop_simulator = BacktestTrailingEngine(
            epic=None,
            is_scalp_trade=self._use_scalping_mode,
            config_override=self._config_override,
            max_bars=96,
            logger=self.logger,
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

        # Log config override status
        if self._config_override:
            self.logger.info(f"   🧪 Config Override: {len(self._config_override)} parameters")
            for key, value in self._config_override.items():
                self.logger.info(f"      - {key}: {value}")
        else:
            self.logger.info(f"   Config Override: None (using database config)")

        # Log historical intelligence mode
        if self._use_historical_intelligence:
            self.logger.info(f"   📚 Historical Intelligence: ENABLED (will replay stored market intelligence)")
        else:
            self.logger.info(f"   📚 Historical Intelligence: DISABLED (will recalculate from data)")

        self.logger.info(f"🧪 BacktestScanner initialized:")
        self.logger.info(f"   Execution ID: {self.execution_id}")
        self.logger.info(f"   Strategy: {self.strategy_name}")
        self.logger.info(f"   Period: {self.start_date} to {self.end_date}")
        self.logger.info(f"   Epics: {len(self.epic_list)} pairs")
        self.logger.info(f"   Timeframe: {self.timeframe}")

    def _initialize_signal_detector(self, db_manager, user_timezone):
        """
        Override parent's signal detector initialization to pass config_override
        for backtest parameter isolation.
        """
        try:
            from .signal_detector import SignalDetector
        except ImportError:
            from forex_scanner.core.signal_detector import SignalDetector

        try:
            if db_manager:
                return SignalDetector(db_manager, user_timezone, config_override=self._config_override)
            else:
                # Try to create with temporary db manager
                try:
                    from .database import DatabaseManager
                except ImportError:
                    from forex_scanner.core.database import DatabaseManager
                temp_db = DatabaseManager(getattr(config, 'DATABASE_URL', ''))
                return SignalDetector(temp_db, user_timezone, config_override=self._config_override)
        except Exception as e:
            self.logger.warning(f"⚠️ Signal detector init warning: {e}")
            # Return basic signal detector without db but with config override
            return SignalDetector(None, user_timezone, config_override=self._config_override)

    def _override_signal_detector_for_backtest(self):
        """
        CRITICAL: Replace the live DataFetcher with BacktestDataFetcher in the signal detector
        This ensures backtest uses historical data instead of real-time data
        """
        try:
            from .signal_detector import SignalDetector
            from .backtest_data_fetcher import BacktestDataFetcher

            # CRITICAL: Force-initialize the requested strategy for backtest
            # This ensures the strategy is available even if config flag is False
            strategy_name = self.strategy_name.upper()
            if strategy_name and strategy_name not in ['EMA', 'EMA_CROSSOVER', 'ALL', '']:
                success, message = self.signal_detector.force_initialize_strategy(strategy_name)
                if success:
                    self.logger.info(f"✅ BacktestScanner: Strategy '{strategy_name}' force-initialized: {message}")
                else:
                    self.logger.warning(f"⚠️ BacktestScanner: Could not force-init '{strategy_name}': {message}")

            # Create a new signal detector but replace its data_fetcher with BacktestDataFetcher
            # PERFORMANCE OPTIMIZATION (Jan 2026): Pass start/end dates to load only needed data into cache
            backtest_data_fetcher = BacktestDataFetcher(
                self.db_manager,
                getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm'),
                start_date=self.start_date,
                end_date=self.end_date,
                epics=self.epic_list  # Only load data for tested epics (from parent class)
            )

            # Set initial backtest time with BOTH start_date and end_date
            # This ensures data filtering works correctly during backtest iterations
            backtest_data_fetcher.set_backtest_time(self.end_date, end_date=self.end_date, start_date=self.start_date)
            self.logger.info(f"⏰ BacktestDataFetcher initialized with period: {self.start_date} to {self.end_date}")

            # Replace the data_fetcher in the existing signal_detector
            self.signal_detector.data_fetcher = backtest_data_fetcher

            # CRITICAL FIX: Configure SMC Simple strategy for backtest mode
            # This is the only active strategy after January 2026 cleanup
            # Set the backtest flag on signal_detector so lazy-loaded strategies use in-memory cooldowns
            self.signal_detector._is_backtest_mode = True
            self.logger.info("🧪 Signal detector configured for backtest mode")

            # If strategy is already initialized, configure it directly
            if hasattr(self.signal_detector, 'smc_simple_strategy') and self.signal_detector.smc_simple_strategy:
                self.signal_detector.smc_simple_strategy.data_fetcher = backtest_data_fetcher
                self.signal_detector.smc_simple_strategy._backtest_mode = True  # Note: uses _backtest_mode
                # Reset cooldowns for fresh backtest
                if hasattr(self.signal_detector.smc_simple_strategy, 'reset_cooldowns'):
                    self.signal_detector.smc_simple_strategy.reset_cooldowns()
                self.logger.info("✅ SMC Simple strategy configured for backtest mode")

            self.logger.info("✅ Signal detector updated to use BacktestDataFetcher for historical data")

        except Exception as e:
            self.logger.error(f"❌ Failed to override signal detector for backtest: {e}")
            raise

    def _ensure_backtest_candles_current(self):
        """
        Ensure the ig_candles_backtest table has up-to-date pre-computed candles.

        First run: Populates the table with resampled 5m, 15m, 4h candles (slower)
        Subsequent runs: Validates freshness, only updates if stale (fast)

        This eliminates runtime resampling during backtest iterations.
        """
        try:
            manager = BacktestCandlesManager(self.db_manager)

            # Check and update backtest candles for our epics and period
            result = manager.ensure_data_current(
                epics=self.epic_list,
                start_date=self.start_date,
                end_date=self.end_date,
                max_staleness_hours=1  # Update if more than 1 hour behind source
            )

            if result['updated']:
                self.logger.info(
                    f"📊 Backtest candles updated: {result['total_rows_inserted']:,} rows "
                    f"in {result['time_taken_seconds']:.1f}s"
                )
            else:
                self.logger.info("⚡ Backtest candles table is current - using pre-computed data")

        except Exception as e:
            # Don't fail the backtest if candle validation fails - just warn
            # The backtest can still run using runtime resampling (slower)
            self.logger.warning(f"⚠️ Backtest candles validation failed: {e}")
            self.logger.warning("   Continuing with runtime resampling (may be slower)")

    def run_historical_backtest(self) -> Dict:
        """
        Run complete historical backtest for the specified period
        Returns comprehensive results including signals and performance metrics
        """
        self.logger.info(f"🚀 Starting historical backtest execution {self.execution_id}")

        # STEP 0: Ensure backtest candles table is up-to-date
        # First run populates data (slower), subsequent runs skip if fresh
        self._ensure_backtest_candles_current()

        try:
            with self.order_logger:
                # Initialize execution in database
                self._initialize_backtest_execution()

                # Preload historical intelligence for the backtest period (Phase 3)
                if self._use_historical_intelligence:
                    intel_count = self._preload_historical_intelligence()
                    if intel_count == 0:
                        self.logger.warning("⚠️ No historical intelligence found - will recalculate as needed")

                # Main backtest loop
                results = self._execute_backtest_loop()

                # Generate final report
                final_report = self._generate_backtest_report(results)

                # Log regime routing distribution if multi-strategy was active
                if hasattr(self, '_regime_counts') and self._regime_counts:
                    self.logger.info("📊 Multi-Strategy Regime Routing Summary:")
                    for regime_strategy, count in sorted(self._regime_counts.items(), key=lambda x: -x[1]):
                        self.logger.info(f"   {regime_strategy}: {count} evaluations")

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

                # Log signals to database (skip if orchestrator handles logging)
                if not self.skip_signal_logging:
                    for signal in processed_signals:
                        success, message, order_data = self.order_logger.place_order(signal)
                        if success:
                            self.backtest_stats['signals_logged'] += 1
                else:
                    self.logger.debug(f"📝 Skipping scanner signal logging - orchestrator will handle {len(processed_signals)} signals")

        except Exception as e:
            self.logger.error(f"Error processing time period {current_time}: {e}")
            raise

        return period_results

    def _scan_historical_timepoint(self, timestamp: datetime) -> List[Dict]:
        """Scan all epics at a specific historical timestamp"""
        signals = []

        # 🔧 CRITICAL FIX: Set backtest time BEFORE fetching data
        # This ensures BacktestDataFetcher only returns data UP TO this timestamp
        # Pass start_date and end_date so cache is populated for entire backtest period (Jan 2026 fix)
        if hasattr(self.signal_detector, 'data_fetcher') and hasattr(self.signal_detector.data_fetcher, 'set_backtest_time'):
            self.signal_detector.data_fetcher.set_backtest_time(timestamp, end_date=self.end_date, start_date=self.start_date)
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
            pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '')

            # Check if specific strategy is requested
            strategy_name = self.strategy_name.upper()
            self.logger.debug(f"🎯 Backtest strategy filter: '{strategy_name}' for {epic}")

            # Multi-strategy routing mode (v3.0.0)
            routed_strategy = None
            confidence_modifier = 1.0
            routing_context = None
            if self._strategy_router and self._multi_strategy_config:
                routed_result = self._get_routed_strategy(epic, timestamp)
                if routed_result:
                    routed_strategy, confidence_modifier, routing_context = routed_result

                    # Apply regime filter if specified
                    regime_filter = self._multi_strategy_config.get('regime_filter')
                    if regime_filter and routing_context.get('regime') != regime_filter:
                        self.logger.debug(
                            f"⏭️ Skipping {epic} at {timestamp}: regime '{routing_context.get('regime')}' "
                            f"doesn't match filter '{regime_filter}'"
                        )
                        return None

                    # Apply session filter if specified
                    session_filter = self._multi_strategy_config.get('session_filter')
                    if session_filter and routing_context.get('session') != session_filter:
                        self.logger.debug(
                            f"⏭️ Skipping {epic} at {timestamp}: session '{routing_context.get('session')}' "
                            f"doesn't match filter '{session_filter}'"
                        )
                        return None

                    # Override strategy_name with routed strategy
                    if routed_strategy:
                        # Track regime distribution for summary
                        if not hasattr(self, '_regime_counts'):
                            self._regime_counts = {}
                        regime_key = f"{routing_context.get('regime')}→{routed_strategy}"
                        self._regime_counts[regime_key] = self._regime_counts.get(regime_key, 0) + 1

                        # Log non-default routing at INFO level
                        self.logger.debug(
                            f"🔀 Routing: {epic} at {timestamp} → {routed_strategy} "
                            f"(regime: {routing_context.get('regime')}, ADX: {routing_context.get('adx_value')})"
                        )
                        strategy_name = routed_strategy
                elif self._multi_strategy_config.get('enabled'):
                    # Router returned None, skip this signal in multi-strategy mode
                    self.logger.debug(f"⏭️ No suitable strategy found for {epic} at {timestamp}")
                    return None

            # Run all strategies if none specified
            if strategy_name in ['ALL', 'ALL_STRATEGIES', ''] or not strategy_name:
                self.logger.debug(f"🔄 Running all strategies for {epic}")
                signals = self.signal_detector.detect_signals_all_strategies(
                    epic, pair_name, self.spread_pips, self.timeframe
                )
                if signals and self.pipeline_mode:
                    self.logger.info(f"🔄 PIPELINE MODE: Processing all-strategies signals for {epic}")
                    if isinstance(signals, list):
                        signals = [self._apply_full_pipeline(s, epic, timestamp) for s in signals if s]
                        signals = [s for s in signals if s]
                    else:
                        signals = self._apply_full_pipeline(signals, epic, timestamp)
                return signals

            # Single strategy: SMC_SIMPLE always uses 1h timeframe
            strategy_timeframe = '1h' if strategy_name in ['SMC_SIMPLE', 'SMC_EMA'] else self.timeframe

            self.logger.debug(f"🎯 Running {strategy_name} strategy only for {epic}")
            signal = self.signal_detector._detect_single_strategy(
                strategy_name, epic, pair_name, self.spread_pips, strategy_timeframe,
                current_timestamp=timestamp,
            )

            if signal and self.pipeline_mode:
                self.logger.debug(f"🔄 PIPELINE MODE: Processing {strategy_name} signal for {epic}")
                signal = self._apply_full_pipeline(signal, epic, timestamp)
            elif signal:
                signal['validation_passed'] = True
                signal['validation_message'] = 'Basic mode - no validation applied'

            if signal and routed_strategy:
                signal['routed_strategy'] = routed_strategy
                signal['routing_confidence_modifier'] = confidence_modifier
                if routing_context:
                    signal['routing_regime'] = routing_context.get('regime')
                    signal['routing_session'] = routing_context.get('session')

            return signal

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

    def _get_intelligence_history_manager(self):
        """Lazy-load the MarketIntelligenceHistoryManager"""
        if self._intelligence_history_manager is None:
            try:
                from .intelligence.market_intelligence_history_manager import MarketIntelligenceHistoryManager
            except ImportError:
                from forex_scanner.core.intelligence.market_intelligence_history_manager import MarketIntelligenceHistoryManager

            self._intelligence_history_manager = MarketIntelligenceHistoryManager(self.db_manager)

        return self._intelligence_history_manager

    def _get_historical_intelligence(self, timestamp: datetime, tolerance_minutes: int = 5) -> Optional[Dict]:
        """
        Get historical market intelligence for a given timestamp.

        This method replays stored market intelligence during backtesting to ensure
        backtest results match what live trading would have done.

        Args:
            timestamp: The backtest timestamp to lookup
            tolerance_minutes: Max time difference to accept (default 5 min)

        Returns:
            Dict with intelligence data or None if not found/disabled
        """
        if not self._use_historical_intelligence:
            return None

        # Check cache first (for efficiency during batch processing)
        cache_key = timestamp.strftime('%Y%m%d%H%M')
        if cache_key in self._historical_intelligence_cache:
            cached = self._historical_intelligence_cache[cache_key]
            self.logger.debug(f"📚 Using cached intelligence for {timestamp}")
            return cached

        try:
            manager = self._get_intelligence_history_manager()
            intelligence = manager.get_intelligence_for_timestamp(
                timestamp,
                tolerance_minutes=tolerance_minutes
            )

            # Cache the result (even if None, to avoid repeated queries)
            self._historical_intelligence_cache[cache_key] = intelligence

            if intelligence:
                self.logger.debug(
                    f"📚 HISTORICAL intelligence for {timestamp}: "
                    f"{intelligence.get('dominant_regime')} ({intelligence.get('regime_confidence', 0):.1%})"
                )
            else:
                self.logger.debug(f"⚠️ No stored intelligence for {timestamp}, will recalculate")

            return intelligence

        except Exception as e:
            self.logger.warning(f"⚠️ Error fetching historical intelligence: {e}")
            return None

    def _preload_historical_intelligence(self) -> int:
        """
        Preload all historical intelligence for the backtest period.

        This is more efficient than querying one at a time during the backtest loop.

        Returns:
            Number of intelligence records loaded
        """
        if not self._use_historical_intelligence:
            return 0

        try:
            manager = self._get_intelligence_history_manager()
            records = manager.get_intelligence_for_period(
                self.start_date,
                self.end_date
            )

            # Populate cache with all records
            for record in records:
                ts = record.get('scan_timestamp')
                if ts:
                    cache_key = ts.strftime('%Y%m%d%H%M')
                    self._historical_intelligence_cache[cache_key] = record

            self.logger.info(f"📚 Preloaded {len(records)} historical intelligence records for backtest")
            return len(records)

        except Exception as e:
            self.logger.warning(f"⚠️ Error preloading historical intelligence: {e}")
            return 0

    def _get_routed_strategy(
        self,
        epic: str,
        timestamp: datetime
    ) -> Optional[Tuple[str, float, Dict]]:
        """
        Get the routed strategy for a given epic and timestamp using multi-strategy routing.

        Uses historical intelligence to determine market regime, then routes to
        the most appropriate strategy based on regime-strategy mapping rules.

        Args:
            epic: The trading pair epic (e.g., 'CS.D.EURUSD.MINI.IP')
            timestamp: The backtest timestamp

        Returns:
            Tuple of (strategy_name, confidence_modifier, routing_context) or None
            routing_context contains: regime, session, volatility_state, adx_value
        """
        if not self._strategy_router:
            return None

        try:
            # Always calculate regime from live price data (ADX/ATR)
            # Historical intelligence data has known corruption (always 'trending')
            regime, adx_value, volatility_state = self._calculate_regime_from_data(epic, timestamp)
            session = None

            # Determine session from timestamp
            session = self._get_session_from_timestamp(timestamp)

            # Check enabled strategies filter
            enabled_strategies = self._multi_strategy_config.get('enable_strategies')

            # Get routed strategy from router
            routed_strategy, confidence_modifier = self._strategy_router.get_strategy_for_regime(
                regime=regime,
                epic=epic,
                session=session,
                volatility_state=volatility_state,
                adx_value=adx_value
            )

            # Filter by enabled strategies if specified
            if routed_strategy and enabled_strategies:
                if routed_strategy not in enabled_strategies:
                    self.logger.debug(
                        f"⏭️ Routed strategy {routed_strategy} not in enabled list {enabled_strategies}"
                    )
                    # Try to find a fallback from enabled strategies
                    for fallback in enabled_strategies:
                        alt_strategy, alt_conf = self._strategy_router.get_strategy_for_regime(
                            regime=regime,
                            epic=epic,
                            session=session,
                            volatility_state=volatility_state,
                            adx_value=adx_value
                        )
                        if alt_strategy == fallback:
                            routed_strategy = alt_strategy
                            confidence_modifier = alt_conf
                            break
                    else:
                        # No enabled strategy matches this regime - use first enabled as fallback
                        routed_strategy = enabled_strategies[0]
                        confidence_modifier = 0.8  # Reduced confidence for fallback

            routing_context = {
                'regime': regime,
                'session': session,
                'volatility_state': volatility_state,
                'adx_value': adx_value,
                'routed_from_intelligence': False
            }

            return (routed_strategy, confidence_modifier, routing_context)

        except Exception as e:
            self.logger.warning(f"⚠️ Error in strategy routing for {epic}: {e}")
            return None

    def _calculate_regime_from_data(
        self,
        epic: str,
        timestamp: datetime
    ) -> Tuple[str, Optional[float], str]:
        """
        Calculate market regime directly from price data (ADX/ATR).

        Used as fallback when historical intelligence is unavailable.
        ADX thresholds (standard interpretation):
            < 20: ranging / low_volatility
            20-25: low_volatility (trend developing)
            25-50: trending
            > 50: breakout / high_volatility

        Returns:
            Tuple of (regime, adx_value, volatility_state)
        """
        try:
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
            df = self.signal_detector.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='1h',
                lookback_hours=72
            )

            if df is None or df.empty or len(df) < 20:
                return ('trending', None, 'normal')

            # Get data up to current backtest time for accurate regime detection
            # The data_fetcher handles time windowing via current_backtest_time
            df_ts = df

            if df_ts.empty or len(df_ts) < 14:
                return ('trending', None, 'normal')

            # Get ADX from the latest available bar
            adx_value = None
            if 'adx' in df_ts.columns:
                raw_adx = df_ts['adx'].iloc[-1]
                if raw_adx is not None and not pd.isna(raw_adx):
                    adx_value = float(raw_adx)

            # Get ATR for volatility assessment
            atr_value = None
            if 'atr' in df_ts.columns:
                atr_value = df_ts['atr'].iloc[-1]
                atr_mean = df_ts['atr'].rolling(20).mean().iloc[-1]
            elif 'ATR' in df_ts.columns:
                atr_value = df_ts['ATR'].iloc[-1]
                atr_mean = df_ts['ATR'].rolling(20).mean().iloc[-1]
            else:
                atr_mean = None

            # Determine regime from ADX
            regime = 'trending'  # Default
            if adx_value is not None and not pd.isna(adx_value):
                if adx_value < 20:
                    regime = 'ranging'
                elif adx_value < 25:
                    regime = 'low_volatility'
                elif adx_value > 50:
                    regime = 'breakout'
                else:
                    regime = 'trending'

            # Determine volatility state from ATR
            volatility_state = 'normal'
            if atr_value is not None and atr_mean is not None and atr_mean > 0:
                atr_ratio = atr_value / atr_mean
                if atr_ratio > 1.5:
                    volatility_state = 'high'
                    if regime == 'trending':
                        regime = 'high_volatility'
                elif atr_ratio < 0.7:
                    volatility_state = 'low'
                    if regime == 'trending' and adx_value and adx_value < 25:
                        regime = 'low_volatility'

            # --- Enhanced regime validation: efficiency ratio + weekly consistency ---
            if regime == 'trending':
                # Check 1: Efficiency ratio from KAMA
                er_value = None
                for col in ('efficiency_ratio', 'kama_er', 'kama_10_er', 'kama_14_er'):
                    if col in df_ts.columns:
                        val = df_ts[col].iloc[-1]
                        if val is not None and not pd.isna(val):
                            er_value = float(val)
                            break

                if er_value is not None and er_value < 0.25:
                    self.logger.debug(
                        f"📉 [BT] [{epic}] Regime downgraded trending→ranging: "
                        f"efficiency_ratio={er_value:.3f} < 0.25"
                    )
                    regime = 'ranging'
                else:
                    # Check 2: Weekly directional consistency
                    weekly_result = self._check_weekly_consistency_from_db(epic, timestamp)
                    if weekly_result.get('is_oscillating', False):
                        self.logger.debug(
                            f"📉 [BT] [{epic}] Regime downgraded trending→ranging: "
                            f"weekly oscillation ({weekly_result.get('pattern', '')})"
                        )
                        regime = 'ranging'

            return (regime, adx_value, volatility_state)

        except Exception as e:
            self.logger.debug(f"⚠️ Error calculating regime from data for {epic}: {e}")
            return ('trending', None, 'normal')

    def _check_weekly_consistency_from_db(self, epic: str, as_of: datetime) -> dict:
        """Check weekly directional consistency using pre-synthesized 1h candles.

        Queries ig_candles_backtest (timeframe=60) up to backtest timestamp,
        aggregates to weekly OHLC in SQL.

        Returns dict with 'is_oscillating', 'pattern', 'alternations'.
        """
        default = {'is_oscillating': False, 'pattern': '', 'alternations': 0}
        try:
            from sqlalchemy import text
            query = text("""
                SELECT
                    date_trunc('week', start_time) as week,
                    (array_agg(open ORDER BY start_time ASC))[1] as week_open,
                    (array_agg(close ORDER BY start_time DESC))[1] as week_close
                FROM ig_candles_backtest
                WHERE epic = :epic AND timeframe = 60
                  AND start_time >= :since AND start_time < :as_of
                GROUP BY 1
                HAVING COUNT(*) > 20
                ORDER BY 1
            """)

            since = as_of - pd.Timedelta(weeks=5)
            with self.signal_detector.data_fetcher.db_manager.engine.connect() as conn:
                result = conn.execute(query, {'epic': epic, 'since': since, 'as_of': as_of}).fetchall()

            if len(result) < 3:
                return default

            # Exclude current partial week
            weeks = result[:-1] if len(result) > 3 else result
            if len(weeks) < 3:
                return default

            weeks = weeks[-4:]

            directions = []
            for row in weeks:
                directions.append(1 if float(row[2]) > float(row[1]) else -1)

            alternations = 0
            for i in range(1, len(directions)):
                if directions[i] != directions[i - 1]:
                    alternations += 1

            pattern = '→'.join(['B' if d == 1 else 'S' for d in directions])
            is_oscillating = alternations >= min(len(directions) - 1, 3)

            return {
                'is_oscillating': is_oscillating,
                'pattern': pattern,
                'alternations': alternations
            }
        except Exception:
            return default

    def _get_session_from_timestamp(self, timestamp: datetime) -> str:
        """
        Determine trading session from timestamp.

        Args:
            timestamp: The timestamp to evaluate

        Returns:
            Session name: 'asian', 'london', 'new_york', or 'overlap'
        """
        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        hour = timestamp.hour

        # Session definitions (UTC)
        # Asian: 00:00 - 08:00 UTC
        # London: 08:00 - 16:00 UTC
        # New York: 13:00 - 21:00 UTC
        # Overlap (London/NY): 13:00 - 16:00 UTC

        if 13 <= hour < 16:
            return 'overlap'
        elif 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'london'
        elif 13 <= hour < 21:
            return 'new_york'
        else:
            return 'asian'  # Late night defaults to Asian

    def _format_historical_intelligence_for_signal(self, intelligence: Dict) -> Dict:
        """
        Format historical intelligence data for use in signal enrichment.

        Converts the database format to the format expected by signal processing.

        Args:
            intelligence: Raw intelligence from database

        Returns:
            Formatted intelligence dict for signal processing
        """
        if not intelligence:
            return {}

        return {
            'market_regime': {
                'dominant_regime': intelligence.get('dominant_regime', 'unknown'),
                'confidence': intelligence.get('regime_confidence', 0.5),
                'regime_scores': intelligence.get('regime_scores', {}),
                'market_strength': {
                    'market_bias': intelligence.get('market_bias', 'neutral'),
                    'average_trend_strength': intelligence.get('average_trend_strength'),
                    'average_volatility': intelligence.get('average_volatility'),
                },
                'pair_analyses': intelligence.get('pair_analyses', {}),
            },
            'session_analysis': {
                'current_session': intelligence.get('current_session', 'unknown'),
                'session_config': {
                    'volatility': intelligence.get('session_volatility', 'medium'),
                },
            },
            'trading_recommendations': {
                'recommended_strategy': intelligence.get('recommended_strategy', 'conservative'),
                'confidence_threshold': intelligence.get('confidence_threshold', 0.7),
                'position_sizing': intelligence.get('position_sizing_recommendation', 'NORMAL'),
            },
            'intelligence_source': 'historical_replay',
            'scan_timestamp': intelligence.get('scan_timestamp'),
        }

    def _process_backtest_signals(self, signals: List[Dict], timestamp: datetime) -> List[Dict]:
        """
        Process signals through the same pipeline as live scanner
        But add backtest-specific metadata and simulate trade outcomes
        """
        processed_signals = []

        # Get historical intelligence for this timestamp (if enabled)
        historical_intelligence = self._get_historical_intelligence(timestamp)
        formatted_intelligence = self._format_historical_intelligence_for_signal(historical_intelligence)

        # Check for high volatility regime (used for per-pair filtering below)
        # This filter can only run here since intelligence isn't available during strategy filtering
        is_high_volatility_regime = False
        if formatted_intelligence:
            market_regime_data = formatted_intelligence.get('market_regime', {})
            global_regime = market_regime_data.get('dominant_regime', 'unknown')
            is_high_volatility_regime = (global_regime == 'high_volatility')

        # Get SMC config for per-pair regime filtering
        smc_config = None
        if is_high_volatility_regime:
            try:
                smc_config = get_smc_simple_config()
            except Exception as e:
                self.logger.warning(f"Could not load SMC config for regime filtering: {e}")

        for signal in signals:
            try:
                epic = signal.get('epic', '')

                # Per-pair high volatility regime filter (v2.38.0)
                # Some pairs (like EURUSD) perform poorly in high_volatility regime and should be blocked
                # Other pairs (like USDJPY) perform BETTER in high_volatility and should NOT be blocked
                if is_high_volatility_regime and smc_config:
                    # Check if this specific pair should block high_volatility regime
                    # Also support override from command line for testing
                    block_this_pair = smc_config.get_pair_scalp_block_global_high_volatility(epic)
                    if self._config_override:
                        # Command-line override takes precedence
                        override_value = self._config_override.get('scalp_block_global_high_volatility')
                        if override_value is not None:
                            block_this_pair = override_value

                    if block_this_pair:
                        self.logger.debug(f"🚫 Blocking {epic} signal due to high_volatility regime at {timestamp}")
                        continue

                # Add backtest metadata
                signal['backtest_execution_id'] = self.execution_id
                signal['backtest_timestamp'] = timestamp
                signal['backtest_mode'] = True
                signal['signal_timestamp'] = timestamp

                # Add historical market intelligence if available
                if formatted_intelligence:
                    signal['market_intelligence'] = formatted_intelligence
                    signal['intelligence_source'] = 'historical_replay'
                    signal['market_regime'] = formatted_intelligence.get('market_regime', {}).get('dominant_regime', 'unknown')
                    signal['regime_confidence'] = formatted_intelligence.get('market_regime', {}).get('confidence', 0.5)
                    signal['session'] = formatted_intelligence.get('session_analysis', {}).get('current_session', 'unknown')
                else:
                    signal['intelligence_source'] = 'not_available'

                # Process through parent's signal preparation
                processed_signal = self._prepare_signal(signal)
                processed_signal['scanner_version'] = self.scanner_version

                # Rejected signals (pipeline validation failed) must not be simulated or
                # fed into the adaptive-state window — they are real rejects, not trades.
                if processed_signal.get('rejected'):
                    processed_signal.setdefault('trade_result', 'REJECTED')
                    processed_signal.setdefault('is_winner', False)
                    processed_signal.setdefault('is_loser', False)
                    processed_signals.append(processed_signal)
                    continue

                # Add trailing stop simulation for realistic trade outcomes
                enhanced_signal = self._add_trade_simulation(processed_signal, timestamp)

                processed_signals.append(enhanced_signal)

            except Exception as e:
                self.logger.error(f"Error processing backtest signal: {e}")
                continue

        return processed_signals

    def _get_trailing_stop_simulator(self, epic: str, strategy: str = 'DEFAULT'):
        """
        Get or create a per-epic trailing engine with pair-specific live config.

        Returns BacktestTrailingEngine (new) or TrailingStopSimulator (legacy A/B mode).
        Cache is keyed by (epic, strategy) so per-strategy DB overrides are respected.
        """
        strategy = (strategy or 'DEFAULT').upper()
        cache_key = (epic, strategy)
        if cache_key in self._trailing_stop_simulators:
            return self._trailing_stop_simulators[cache_key]

        if BACKTEST_USE_LEGACY_SIMULATOR:
            # Legacy path: old simulator for A/B comparison runs
            if self._use_scalping_mode:
                simulator = TrailingStopSimulator(
                    epic=epic,
                    target_pips=getattr(self, '_scalping_target_pips', 8.0),
                    initial_stop_pips=getattr(self, '_scalping_stop_pips', 6.0),
                    max_bars=getattr(self, '_scalping_max_bars', 10000),
                    use_fixed_sl_tp=True,
                    strategy=strategy,
                    logger=self.logger,
                )
            else:
                simulator = self._create_trailing_stop_simulator(
                    epic=epic,
                    use_atr_trailing=self._use_atr_trailing,
                    strategy=strategy,
                    logger=self.logger,
                )
        elif strategy == 'SMC_MOMENTUM':
            # SMC_MOMENTUM uses fixed SL/TP (ATR-based TP, no trailing stages).
            # Scalp trailing configs would exit winners at 10-14 pips, far below the
            # 40-70 pip ATR TP — use fixed-SL/TP simulator so only TP/SL matter.
            simulator = TrailingStopSimulator(
                epic=epic,
                target_pips=50.0,       # placeholder — overridden per-signal via reward_pips
                initial_stop_pips=10.0, # placeholder — overridden per-signal via risk_pips
                max_bars=400,
                use_fixed_sl_tp=True,
                strategy=strategy,
                logger=self.logger,
            )
            self.logger.debug(f"📊 TrailingStopSimulator (fixed SL/TP) ready for {epic} (SMC_MOMENTUM)")
        else:
            # New engine: uses live PAIR_TRAILING_CONFIGS / SCALP_TRAILING_CONFIGS.
            # Scalp mode uses SCALP_TRAILING_CONFIGS with full progressive trailing
            # (old code used fixed SL/TP which produced meaningless scalp results).
            simulator = BacktestTrailingEngine(
                epic=epic,
                is_scalp_trade=self._use_scalping_mode,
                config_override=self._config_override,
                max_bars=200,
                strategy=strategy,
                logger=self.logger,
                minutes_per_bar=self._timeframe_to_minutes(self.timeframe),
            )
            self.logger.debug(f"📊 BacktestTrailingEngine ready for {epic} (scalp={self._use_scalping_mode})")

        self._trailing_stop_simulators[cache_key] = simulator
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

            trailing_cfg: dict = {}
            try:
                try:
                    from forex_scanner.services.trailing_config_service import get_trailing_config_service
                except ImportError:
                    from services.trailing_config_service import get_trailing_config_service  # type: ignore
                trailing_cfg = get_trailing_config_service().get_config(epic, is_scalp=True) or {}
            except Exception:
                pass
            if not trailing_cfg:
                try:
                    from config_trailing_stops import get_scalp_trailing_config
                except ImportError:
                    from forex_scanner.config_trailing_stops import get_scalp_trailing_config
                trailing_cfg = get_scalp_trailing_config(epic)

            # Use signal's SL/TP if already set, otherwise use trailing config defaults
            if not signal.get('risk_pips') or signal.get('risk_pips') == 0:
                signal['risk_pips'] = trailing_cfg.get('early_breakeven_trigger_points', 8) + 2
            if not signal.get('reward_pips') or signal.get('reward_pips') == 0:
                signal['reward_pips'] = trailing_cfg.get('stage1_trigger_points', 12)

            self.logger.debug(f"📊 ATR Mode: SL={signal.get('risk_pips')} pips, TP={signal.get('reward_pips')} pips")

            # Get per-epic trailing stop simulator with per-strategy + pair config
            signal_strategy = (
                signal.get('strategy')
                or signal.get('strategy_name')
                or 'DEFAULT'
            )
            trailing_simulator = self._get_trailing_stop_simulator(epic, strategy=signal_strategy)

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
            self.logger.debug(f"📊 Simulation: {epic} with {len(future_df)} {self.timeframe} bars")
            enhanced_signal = trailing_simulator.simulate_trade(
                signal=signal,
                df=future_df,
                signal_idx=0
            )

            self.logger.debug(f"Simulation complete: profit={enhanced_signal.get('max_profit_pips', 0):.1f}, "
                            f"loss={enhanced_signal.get('max_loss_pips', 0):.1f}, "
                            f"result={enhanced_signal.get('trade_result', 'UNKNOWN')}, "
                            f"stage={enhanced_signal.get('stage_reached', 0)}")

            # v3.4.0: Adjust cooldown for unfilled limit orders
            # In live trading, expired limit orders only trigger 30min cooldown instead of full cooldown
            # This replicates that behavior in backtest to match live trading more accurately
            trade_outcome = enhanced_signal.get('trade_outcome', '')
            if trade_outcome == 'LIMIT_NOT_FILLED':
                pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
                # Get strategy from signal_detector to adjust cooldown
                if hasattr(self.signal_detector, 'smc_simple_strategy') and self.signal_detector.smc_simple_strategy:
                    self.signal_detector.smc_simple_strategy.adjust_cooldown_for_unfilled_order(pair, signal_timestamp)
                    self.logger.debug(f"🔄 Cooldown adjusted for unfilled limit order on {pair}")

            # v2.46.0: Feed trade outcome back to strategy's rolling performance window
            # Enables the confidence gate to adapt based on recent backtest outcomes
            is_winner = enhanced_signal.get('is_winner', False)
            is_loser = enhanced_signal.get('is_loser', False)
            if (is_winner or is_loser) and hasattr(self.signal_detector, 'smc_simple_strategy') and self.signal_detector.smc_simple_strategy:
                self.signal_detector.smc_simple_strategy.record_backtest_outcome(epic, bool(is_winner))

            # Feed outcome back to RANGE_FADE strategy for same-session post-loss blocking
            if is_winner or is_loser:
                range_fade_strategy = self.signal_detector._strategies.get('RANGE_FADE')
                if range_fade_strategy is not None and hasattr(range_fade_strategy, 'mark_signal_outcome'):
                    signal_hour = signal_timestamp.hour if signal_timestamp else 0
                    range_fade_strategy.mark_signal_outcome(epic, bool(is_loser), signal_hour)

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"❌ Error adding trade simulation: {e}", exc_info=True)
            # Return original signal without simulation data
            return signal

    def _fetch_future_price_data(self, epic: str, signal_timestamp: datetime, max_bars: int = 96, timeframe_override: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch future price data for trade simulation using BacktestDataFetcher (in-memory cache)

        Args:
            epic: Epic code
            signal_timestamp: Signal timestamp (start of future data)
            max_bars: Maximum number of bars to fetch
            timeframe_override: Optional timeframe to use instead of self.timeframe (e.g., '1m' for VSL mode)

        Returns:
            DataFrame with OHLC price data or None if no data available
        """
        try:
            # Use BacktestDataFetcher to get data from in-memory cache (same source as signal detection)
            data_fetcher = self.signal_detector.data_fetcher

            # Extract pair from epic
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')

            # Use override timeframe if provided (e.g., '1m' for VSL simulation)
            simulation_tf = timeframe_override or self.timeframe

            # Calculate how much data we need (from NOW back to signal time, plus future bars)
            time_increment = self._parse_timeframe_to_timedelta(simulation_tf)
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
                # Use simulation_tf (may be '1m' for VSL mode, or default timeframe otherwise)
                full_df = data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe=simulation_tf,
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

            # Include signal candle at index 0 so BacktestTrailingEngine's signal_idx+1
            # slice starts at the first post-entry bar (not the second).
            future_df = full_df[full_df[timestamp_col] >= signal_timestamp].copy()

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

    def _convert_timestamp_safe(self, timestamp_value):
        """Backtest-mode timestamp normaliser. Mirrors the live scanner helper
        but without the live-scan stats counter."""
        if timestamp_value is None:
            return None
        try:
            if isinstance(timestamp_value, datetime):
                if timestamp_value.tzinfo is None:
                    return timestamp_value.replace(tzinfo=timezone.utc)
                return timestamp_value
            if isinstance(timestamp_value, str):
                if 'T' in timestamp_value:
                    return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                return datetime.fromisoformat(timestamp_value).replace(tzinfo=timezone.utc)
            if isinstance(timestamp_value, (int, float)):
                if 1577836800 <= timestamp_value <= 1893456000:
                    return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
                return None
            if hasattr(timestamp_value, 'to_pydatetime'):
                dt = timestamp_value.to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
        except Exception:
            return None
        return None

    def _prepare_signal(self, signal: Dict) -> Dict:
        """Backtest-mode signal preparation. Was inherited from
        IntelligentForexScanner; replicated here after C3 decomposition."""
        clean_signal = signal.copy()
        clean_signal['scanner_timestamp'] = datetime.now().isoformat()
        clean_signal['scanner_version'] = 'backtest_scanner_v1'
        clean_signal['scanner_validated'] = True
        for field in ('market_timestamp', 'timestamp'):
            if field in clean_signal:
                clean_signal[field] = self._convert_timestamp_safe(clean_signal[field])
        clean_signal['processing_pipeline'] = {
            'raw_detection': True,
            'confidence_filtered': True,
            'dedup_filtered': False,
            'smart_money_enhanced': bool(signal.get('smart_money_validated')),
            'signal_processor_used': False,
            'ready_for_execution': True,
        }
        return clean_signal

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
            min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)
            self.logger.info(f"Minimum bars required: {min_bars}")

            if expected_bars_per_scan < min_bars:
                self.logger.warning(
                    f"⚠️ WARNING: Lookback may be insufficient! "
                    f"Expected {expected_bars_per_scan} < MIN_BARS {min_bars}"
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
                          config_override: dict = None,
                          use_historical_intelligence: bool = True,
                          **kwargs) -> BacktestScanner:
    """
    Create BacktestScanner instance

    Args:
        backtest_config: Backtest configuration dict
        db_manager: Database manager instance
        config_override: Parameter overrides for backtest isolation (Phase 1)
        use_historical_intelligence: Whether to replay stored intelligence (Phase 3)
        **kwargs: Additional arguments passed to BacktestScanner

    Returns:
        Configured BacktestScanner instance
    """
    return BacktestScanner(
        backtest_config,
        db_manager,
        config_override=config_override,
        use_historical_intelligence=use_historical_intelligence,
        **kwargs
    )
