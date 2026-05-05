# core/signal_detector.py
"""
Signal Detection Coordinator
Lightweight coordinator that delegates to specialized strategy modules

NOTE: After January 2026 cleanup, only SMC Simple strategy is active.
Legacy strategies have been archived to forex_scanner/archive/disabled_strategies/
"""

import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

try:
    from .database import DatabaseManager
    from .data_fetcher import DataFetcher
    from .backtest.performance_analyzer import PerformanceAnalyzer
    from .backtest.signal_analyzer import SignalAnalyzer
    from .detection.price_adjuster import PriceAdjuster
    from .smart_money_integration import add_smart_money_to_signal, add_smart_money_to_signals
    # NOTE: LargeCandleFilter removed (Jan 2026) - database columns dropped
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner.core.backtest.performance_analyzer import PerformanceAnalyzer
    from forex_scanner.core.backtest.signal_analyzer import SignalAnalyzer
    from forex_scanner.core.detection.price_adjuster import PriceAdjuster
    from forex_scanner.core.smart_money_integration import add_smart_money_to_signal, add_smart_money_to_signals
    # NOTE: LargeCandleFilter removed (Jan 2026) - database columns dropped

# Database-driven configuration services (NOT legacy config files)
try:
    from services.scanner_config_service import get_scanner_config
    from services.smc_simple_config_service import get_smc_simple_config
    from services.xau_gold_config_service import XAUGoldConfigService
    from services.range_fade_config_service import RangeFadeConfigService
    from services.smc_momentum_config_service import SMCMomentumConfigService
    from services.impulse_fade_config_service import ImpulseFadeConfigService
except ImportError:
    from forex_scanner.services.scanner_config_service import get_scanner_config
    from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
    from forex_scanner.services.xau_gold_config_service import XAUGoldConfigService
    from forex_scanner.services.range_fade_config_service import RangeFadeConfigService
    from forex_scanner.services.smc_momentum_config_service import SMCMomentumConfigService
    from forex_scanner.services.impulse_fade_config_service import ImpulseFadeConfigService


class SignalDetector:
    """
    Lightweight signal detection coordinator

    After January 2026 cleanup, only SMC Simple strategy is active.
    Legacy strategies are archived in forex_scanner/archive/disabled_strategies/
    """

    def __init__(self, db_manager: DatabaseManager, user_timezone: str = 'Europe/Stockholm', config_override: dict = None):
        self.db_manager = db_manager
        self.data_fetcher = DataFetcher(db_manager, user_timezone)
        self.price_adjuster = PriceAdjuster()
        self.logger = logging.getLogger(__name__)

        # Store config override for backtest parameter isolation
        self._config_override = config_override

        # Backtest mode flag - set by backtest_scanner when configuring strategies
        # This ensures lazy-loaded strategies use in-memory cooldowns instead of database
        self._is_backtest_mode = False

        # NOTE: Large candle filter removed (Jan 2026 cleanup)
        # The filter was initialized but never called in the signal flow.
        # Database columns for large candle filter were dropped.
        # See migration: remove_safety_filter_columns.sql
        self.large_candle_filter = None

        # Initialize SMC Simple Strategy (the only active strategy)
        # NOTE: SMC Simple is always enabled after January 2026 cleanup
        # Configuration is loaded from database via get_smc_simple_config()
        try:
            # SMC Simple strategy uses lazy loading for consistency
            self.smc_simple_enabled = True
            self.smc_simple_strategy = None  # Will be lazy-loaded on first use
            self.logger.info("✅ SMC Simple strategy enabled (3-tier EMA, lazy-load, database config)")
        except Exception as e:
            self.logger.error(f"❌ Failed to enable SMC Simple strategy: {e}")
            self.smc_simple_enabled = False
            self.smc_simple_strategy = None

        # Initialize analysis components
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()

        # Multi-strategy support (v3.0.0)
        # Strategy instances are cached in _strategies dict; use _get_strategy() to access.
        self._strategies: Dict[str, Any] = {}
        self._range_fade_profile = None

        # MEAN_REVERSION runs concurrently (not through the router) so it can
        # fire alongside whichever strategy the router picks. Per-pair
        # enable flags + hard ADX gates inside the strategy do the actual
        # filtering. Enable flag is driven from scanner_global_config.
        self.mean_reversion_enabled = True

        # RANGE_FADE runs concurrently (monitor-only by default).
        self.range_fade_enabled = True

        # SMC_MOMENTUM runs concurrently (liquidity sweep + rejection wick).
        # Per-pair is_enabled flags in smc_momentum_pair_overrides gate actual execution.
        self.smc_momentum_enabled = True

        # IMPULSE_FADE runs concurrently (late-US session large-body fade).
        # Per-pair is_enabled flags in impulse_fade_pair_overrides gate execution.
        self.impulse_fade_enabled = True

        # Multi-strategy routing (v3.0.0)
        # Routes signals to different strategies based on ADX-derived market regime
        self._strategy_router = None
        if not self._is_backtest_mode:
            try:
                from services.strategy_router_service import get_strategy_router
                self._strategy_router = get_strategy_router(is_backtest=False)
                if self._strategy_router and self._strategy_router.is_multi_strategy_enabled():
                    self.logger.info("🎯 Multi-strategy routing: ENABLED (live mode)")
                else:
                    self._strategy_router = None
            except ImportError:
                try:
                    from forex_scanner.services.strategy_router_service import get_strategy_router
                    self._strategy_router = get_strategy_router(is_backtest=False)
                    if self._strategy_router and self._strategy_router.is_multi_strategy_enabled():
                        self.logger.info("🎯 Multi-strategy routing: ENABLED (live mode)")
                    else:
                        self._strategy_router = None
                except ImportError:
                    pass
            except Exception as e:
                self.logger.warning(f"⚠️ Multi-strategy routing init failed: {e}")

        self.logger.info("📊 SignalDetector initialized (SMC Simple only - legacy strategies archived)")

    # =========================================================================
    # STRATEGY HELPER
    # =========================================================================

    def _get_strategy(self, name: str, **extra_override) -> Optional[Any]:
        """Return a cached strategy instance, creating it on first call.

        Uses StrategyRegistry to look up the registered class, then
        instantiates it with the SignalDetector's own db_manager / logger /
        config_override so every strategy gets consistent dependencies.
        The instance is stored in self._strategies[name] so subsequent calls
        are free.

        Args:
            name: Canonical strategy name (e.g. 'MEAN_REVERSION').
            **extra_override: Extra key/value pairs merged into config_override
                              for this strategy (e.g. erf_profile='5m').

        Returns:
            Strategy instance, or None if the class is not registered.
        """
        key = name.upper()
        if key in self._strategies:
            return self._strategies[key]

        try:
            from .strategies.strategy_registry import StrategyRegistry
        except (ImportError, ValueError):
            from forex_scanner.core.strategies.strategy_registry import StrategyRegistry

        registry = StrategyRegistry.get_instance()
        strategy_class = registry._strategies.get(key)
        if strategy_class is None:
            self.logger.warning(f"⚠️ Strategy not registered in StrategyRegistry: {key}")
            return None

        override = dict(self._config_override or {})
        override.update(extra_override)

        try:
            instance = strategy_class(
                config=None,
                db_manager=self.db_manager,
                logger=self.logger,
                config_override=override if override else None,
            )
            instance.data_fetcher = self.data_fetcher
            override_note = f" with {len(override)} overrides" if override else ""
            self.logger.info(f"✅ {key} strategy loaded via StrategyRegistry{override_note}")
            self._strategies[key] = instance
            return instance
        except Exception as e:
            self.logger.error(f"❌ Error instantiating {key} via StrategyRegistry: {e}")
            return None

    # =========================================================================
    # BACKTEST FORCE-INITIALIZATION METHODS
    # =========================================================================

    def force_initialize_strategy(self, strategy_name: str) -> Tuple[bool, str]:
        """
        Force-initialize a specific strategy for backtesting, regardless of config flags.

        After January 2026 cleanup, only SMC Simple strategy is available.
        Other strategy names are kept for backward compatibility but will fail.

        Args:
            strategy_name: Strategy name (e.g., 'SMC_SIMPLE', 'SMC')

        Returns:
            Tuple of (success: bool, message: str)
        """
        strategy_name = strategy_name.upper()

        # Canonical name aliases
        aliases = {
            'SMC_EMA': 'SMC_SIMPLE',
            'SMC': 'SMC_SIMPLE',
            'MR': 'MEAN_REVERSION',
            'XAU': 'XAU_GOLD',
            'GOLD': 'XAU_GOLD',
        }
        canonical = aliases.get(strategy_name, strategy_name)

        known = {'SMC_SIMPLE', 'MEAN_REVERSION', 'RANGE_FADE', 'XAU_GOLD', 'SMC_MOMENTUM', 'IMPULSE_FADE'}
        if canonical not in known:
            return False, f"Unknown or archived strategy: {strategy_name}. Available: SMC_SIMPLE, MEAN_REVERSION, RANGE_FADE, XAU_GOLD, SMC_MOMENTUM, IMPULSE_FADE."

        if canonical == 'SMC_SIMPLE':
            # SMC Simple has its own lazy-load path; just arm the flag.
            self.smc_simple_enabled = True
            self.smc_simple_strategy = None
            self.logger.info("🔧 Force-initialized SMC Simple strategy (lazy-load)")
            return True, "SMC Simple strategy force-initialized"

        if canonical == 'RANGE_FADE':
            self._range_fade_profile = "5m"

        # For all other strategies, warm the cache via _get_strategy.
        # Even if instantiation fails here, the detect_* method will retry.
        try:
            extra = {"erf_profile": "5m"} if canonical == 'RANGE_FADE' else {}
            self._get_strategy(canonical, **extra)
            self.logger.info(f"🔧 Force-initialized {canonical} strategy via StrategyRegistry")
            return True, f"{canonical} strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init {canonical}: {e}"

    def _get_default_timeframe(self, timeframe: str = None) -> str:
        """Get default timeframe from database config if not specified"""
        if timeframe is None:
            scanner_config = get_scanner_config()
            return scanner_config.default_timeframe or '15m'
        return timeframe

    def _filter_incomplete_candles(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Filter out incomplete candles from the end of the DataFrame.

        In live trading, the last candle may still be forming (incomplete).
        This causes a timing mismatch with backtesting where all candles are complete.

        By filtering out incomplete candles, we ensure:
        1. Live and backtest use the same complete-candle data
        2. No "1 candle late" timing issues
        3. Strategy decisions are based on confirmed price action

        Args:
            df: DataFrame with candle data (may have 'is_complete' column)
            timeframe: Timeframe string for logging (e.g., '5m', '15m', '4h')

        Returns:
            DataFrame with incomplete candles removed from the end
        """
        if df is None or len(df) == 0:
            return df

        # Check if we're in backtest mode
        is_backtest = self._is_backtest_mode or (
            hasattr(self.data_fetcher, 'current_backtest_time') and
            self.data_fetcher.current_backtest_time is not None
        )

        if is_backtest:
            # BACKTEST FIX: Simulate incomplete candle filtering to match live behavior.
            # In live mode, the last candle is dropped if it hasn't closed yet.
            # In backtest, historical data contains completed candles, but a candle
            # whose period hasn't elapsed at current_backtest_time would still be
            # "forming" in a real scenario. Drop it to match live behavior.
            backtest_time = getattr(self.data_fetcher, 'current_backtest_time', None)
            if backtest_time is not None:
                try:
                    tf_minutes = {
                        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                        '1h': 60, '4h': 240, '1d': 1440
                    }.get(timeframe.lower(), 15)

                    if 'start_time' in df.columns:
                        last_candle_start = pd.to_datetime(df['start_time'].iloc[-1])
                        candle_close_time = last_candle_start + pd.Timedelta(minutes=tf_minutes)

                        bt_time = pd.Timestamp(backtest_time)
                        if last_candle_start.tz is not None and bt_time.tz is None:
                            bt_time = bt_time.tz_localize(last_candle_start.tz)
                        elif last_candle_start.tz is None and bt_time.tz is not None:
                            bt_time = bt_time.tz_localize(None)

                        if bt_time < candle_close_time:
                            self.logger.debug(
                                f"🕐 [{timeframe}] Backtest: filtering incomplete candle "
                                f"(start={last_candle_start}, closes={candle_close_time}, bt_time={bt_time})"
                            )
                            return df.iloc[:-1].copy()
                except Exception as e:
                    self.logger.debug(f"Backtest incomplete candle check failed: {e}")
            return df

        # Check if is_complete column exists
        if 'is_complete' not in df.columns:
            # No completeness tracking - use timestamp-based check
            # Check if last candle's period hasn't closed yet
            try:
                # Get timeframe in minutes
                tf_minutes = {
                    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                    '1h': 60, '4h': 240, '1d': 1440
                }.get(timeframe.lower(), 15)

                # Get last candle timestamp
                if 'start_time' in df.columns:
                    last_candle_start = pd.to_datetime(df['start_time'].iloc[-1])
                elif df.index.name == 'start_time' or isinstance(df.index, pd.DatetimeIndex):
                    last_candle_start = df.index[-1]
                else:
                    # Can't determine - return as-is
                    return df

                # Make timezone-aware comparison
                now = pd.Timestamp.now(tz='UTC')
                if last_candle_start.tz is None:
                    last_candle_start = last_candle_start.tz_localize('UTC')
                else:
                    last_candle_start = last_candle_start.tz_convert('UTC')

                # Calculate when this candle should close
                candle_close_time = last_candle_start + pd.Timedelta(minutes=tf_minutes)

                if now < candle_close_time:
                    # Last candle is still forming - exclude it
                    self.logger.debug(f"🕐 [{timeframe}] Filtering incomplete candle (closes at {candle_close_time})")
                    return df.iloc[:-1].copy()

            except Exception as e:
                self.logger.debug(f"Could not check candle completeness by timestamp: {e}")
                return df
        else:
            # Use is_complete column
            if not df['is_complete'].iloc[-1]:
                self.logger.debug(f"🕐 [{timeframe}] Filtering incomplete candle (is_complete=False)")
                return df.iloc[:-1].copy()

        return df

    # =========================================================================
    # SIGNAL DETECTION METHODS
    # =========================================================================

    def detect_smc_simple_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect SMC Simple 3-tier signals (50 EMA based)

        Uses multi-timeframe analysis:
        - Bias: 4H 50 EMA for directional bias
        - Trigger: 15m swing break with body-close confirmation
        - Entry: 5m pullback to Fibonacci zone
        """
        # Load timeframes from DATABASE config (not legacy config files)
        smc_config = get_smc_simple_config()

        # Check if scalp mode is enabled - use faster timeframes
        scalp_mode = getattr(smc_config, 'scalp_mode_enabled', False)
        if scalp_mode:
            # Use per-pair HTF override if available (e.g., USDCHF uses 15m instead of 1h)
            htf_tf = smc_config.get_pair_scalp_htf_timeframe(epic) or getattr(smc_config, 'scalp_htf_timeframe', '1h')
            trigger_tf = getattr(smc_config, 'scalp_trigger_timeframe', '5m')
            entry_tf = getattr(smc_config, 'scalp_entry_timeframe', '1m')
            self.logger.info(f"🔍 [SMC_SIMPLE] SCALP MODE: Using htf_tf={htf_tf}, trigger_tf={trigger_tf}, entry_tf={entry_tf}")
        else:
            htf_tf = smc_config.htf_timeframe or '4h'
            trigger_tf = smc_config.trigger_timeframe or '15m'
            entry_tf = smc_config.entry_timeframe or '5m'
            self.logger.debug(f"🔍 [SMC_SIMPLE] Using htf_tf={htf_tf}, trigger_tf={trigger_tf}, entry_tf={entry_tf}")

        try:
            # Initialize strategy if not already done (lazy loading)
            if not hasattr(self, 'smc_simple_strategy') or self.smc_simple_strategy is None:
                from .strategies.smc_simple_strategy import create_smc_simple_strategy

                # Strategy loads all config from database via smc_simple_config_service
                # Pass config_override for backtest parameter isolation
                self.smc_simple_strategy = create_smc_simple_strategy(
                    None,
                    logger=self.logger,
                    db_manager=self.db_manager,
                    config_override=self._config_override
                )
                mode_str = "BACKTEST MODE with overrides" if self._config_override else "LIVE MODE"
                self.logger.info(f"✅ SMC Simple strategy initialized ({mode_str}, htf={htf_tf}, trigger={trigger_tf}, entry={entry_tf})")

                # CRITICAL FIX: Set backtest mode on lazy-loaded strategy
                # This ensures cooldowns use in-memory tracking instead of database
                if self._is_backtest_mode:
                    self.smc_simple_strategy._backtest_mode = True
                    self.logger.info("   🧪 Strategy configured for backtest mode (in-memory cooldowns)")

            # Check if data_fetcher is in backtest mode (needed for lookback calculations)
            is_backtest = hasattr(self.data_fetcher, 'current_backtest_time') and self.data_fetcher.current_backtest_time is not None

            # CRITICAL FIX: Reset cooldowns at start of new backtest to prevent stale cooldowns
            if is_backtest and hasattr(self.smc_simple_strategy, 'reset_cooldowns'):
                current_backtest_id = id(self.data_fetcher)
                if not hasattr(self, '_smc_simple_backtest_id') or self._smc_simple_backtest_id != current_backtest_id:
                    self._smc_simple_backtest_id = current_backtest_id
                    self.smc_simple_strategy.reset_cooldowns()

            # Dynamic minimum bars based on actual EMA period configured for this pair
            # Scalp mode: per-pair EMA (10-30) or global default (20)
            # Swing mode: 50 EMA (fixed)
            # Calculate required bars FIRST, then determine lookback hours needed
            if scalp_mode:
                # Get per-pair scalp EMA period, fallback to global
                pair_ema = smc_config.get_pair_scalp_ema_period(epic)
                ema_period = pair_ema if pair_ema is not None else getattr(smc_config, 'scalp_ema_period', 20)
                # Minimum bars: EMA period + 10 bar buffer (handles weekend opens gracefully)
                min_htf_bars = ema_period + 10

                # DYNAMIC LOOKBACK: Calculate hours needed to get min_htf_bars
                # For 1h timeframe, we need min_htf_bars hours of TRADING time
                # But markets are closed weekends (~60h), so multiply by 2.5x for safety
                # Minimum 168 hours (7 days) to always cover at least one full week
                if is_backtest:
                    htf_lookback = 72  # Backtest has continuous data
                else:
                    required_hours = min_htf_bars * 1  # 1h per bar for 1h timeframe
                    # Add weekend buffer: 2.5x multiplier accounts for ~60% market uptime
                    htf_lookback = max(int(required_hours * 2.5), 168)  # Minimum 7 days
                    self.logger.debug(f"📊 Dynamic lookback for {epic}: {min_htf_bars} bars needed → {htf_lookback} hours (7-day minimum)")
            else:
                ema_period = 50  # Swing mode uses 50 EMA
                min_htf_bars = 60
                htf_lookback = 400 if is_backtest else 400  # ~17 days for 4H swing

            # Get HTF data for EMA bias with dynamically calculated lookback
            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=htf_tf,
                lookback_hours=htf_lookback
            )
            # Filter incomplete candles (live mode only) to align timing with backtest
            df_4h = self._filter_incomplete_candles(df_4h, htf_tf)

            if df_4h is None or len(df_4h) < min_htf_bars:
                self.logger.info(f"⚠️ Insufficient {htf_tf} data for {epic} (got {len(df_4h) if df_4h is not None else 0} bars, need {min_htf_bars} for {ema_period} EMA)")
                return None

            # Get trigger timeframe data for swing break detection
            # Scalp mode: 5m trigger needs fewer bars
            # Swing mode: 15m needs 30+ bars for swing detection
            if scalp_mode:
                trigger_lookback = 24 if is_backtest else 12  # ~24h for 5m scalp
            elif trigger_tf == '15m':
                trigger_lookback = 72 if is_backtest else 30
            else:
                trigger_lookback = 100 if is_backtest else 100

            df_trigger = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=trigger_tf,
                lookback_hours=trigger_lookback
            )
            # Filter incomplete candles (live mode only) to align timing with backtest
            df_trigger = self._filter_incomplete_candles(df_trigger, trigger_tf)

            if df_trigger is None or len(df_trigger) < 30:
                self.logger.info(f"⚠️ Insufficient {trigger_tf} data for {epic} (got {len(df_trigger) if df_trigger is not None else 0} bars, need 30)")
                return None

            # Get entry timeframe data for pullback entry
            # Scalp mode: 1m entry needs fewer hours
            # Swing mode: 5m needs ~50 bars for pullback analysis
            df_entry = None
            entry_lookback = None
            if scalp_mode and entry_tf == '1m':
                entry_lookback = 4 if is_backtest else 2  # ~4h for 1m scalp = 240 bars
            elif entry_tf in ['15m', '5m']:
                if entry_tf == '5m':
                    entry_lookback = 24 if is_backtest else 25
                else:
                    entry_lookback = 48 if is_backtest else 50

            # Fetch entry data if lookback was set
            if entry_lookback is not None:
                df_entry = self.data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe=entry_tf,
                    lookback_hours=entry_lookback
                )
                # Filter incomplete candles (live mode only) to align timing with backtest
                df_entry = self._filter_incomplete_candles(df_entry, entry_tf)

            df_entry_len = len(df_entry) if df_entry is not None else 0
            self.logger.info(f"🔍 [SMC_SIMPLE] Passing to strategy: {htf_tf}({len(df_4h)} bars), {trigger_tf}({len(df_trigger)} bars), {entry_tf}({df_entry_len} bars)")

            # Detect signal
            signal = self.smc_simple_strategy.detect_signal(
                df_trigger=df_trigger,
                df_4h=df_4h,
                epic=epic,
                pair=pair,
                df_entry=df_entry
            )

            if signal:
                self.logger.info(f"✅ [SMC_SIMPLE] Signal detected for {epic}: {signal['signal']} @ {signal['entry_price']:.5f}")
                signal = self._add_market_context(signal, df_trigger)

            # Flush any pending rejections to database
            if hasattr(self, 'smc_simple_strategy') and self.smc_simple_strategy is not None:
                self.smc_simple_strategy.flush_rejections()

            return signal

        except Exception as e:
            self.logger.error(f"❌ [SMC_SIMPLE] Error detecting signals for {epic}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def _is_gold_epic(self, epic: str) -> bool:
        """Return True if epic should be routed to XAU_GOLD strategy.

        String match is a necessary pre-filter (prevents EURUSD routing to
        XAU_GOLD when DB is unavailable).  The DB check lets ops disable gold
        without a code change.
        """
        if not epic:
            return False
        e = epic.upper()
        if 'GOLD' not in e and 'XAU' not in e:
            return False
        try:
            return XAUGoldConfigService.get_instance().is_pair_enabled(epic)
        except Exception:
            return True

    def detect_xau_gold_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """Detect XAU_GOLD strategy signals (gold-specific 3-tier SMC)."""
        try:
            # Lazy-init via strategy registry (keeps pattern consistent)
            if not hasattr(self, 'xau_gold_strategy') or self.xau_gold_strategy is None:
                try:
                    from .strategies.xau_gold_strategy import create_xau_gold_strategy
                except ImportError:
                    from forex_scanner.core.strategies.xau_gold_strategy import create_xau_gold_strategy
                self.xau_gold_strategy = create_xau_gold_strategy(
                    db_manager=self.db_manager,
                    logger=self.logger,
                    config_override=self._config_override,
                )
                mode = "BACKTEST with overrides" if self._config_override else "LIVE"
                self.logger.info(f"✅ XAU_GOLD strategy initialized (lazy-load, {mode})")

            # Timeframes from strategy's own config
            cfg = self.xau_gold_strategy.config
            htf_tf = cfg.htf_timeframe
            trigger_tf = cfg.trigger_timeframe
            entry_tf = cfg.entry_timeframe

            is_backtest = (
                hasattr(self.data_fetcher, 'current_backtest_time')
                and self.data_fetcher.current_backtest_time is not None
            )

            # Reset cooldowns at start of new backtest
            if is_backtest:
                current_id = id(self.data_fetcher)
                if getattr(self, '_xau_gold_backtest_id', None) != current_id:
                    self._xau_gold_backtest_id = current_id
                    self.xau_gold_strategy.reset_cooldowns()

            # Lookbacks: 4H needs ~200 bars for EMA50/EMA100 + atr percentile
            htf_lookback = 4 * 220  # ~220 4H bars = ~880 hours (~37 days)
            trigger_lookback = 200  # ~200h for 1H BOS + MACD (also drives bos_search_bars=48 window)
            entry_lookback = 200    # ~200h for 15m (800 bars) — give pullback time to develop

            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic, pair=pair, timeframe=htf_tf, lookback_hours=htf_lookback
            )
            df_4h = self._filter_incomplete_candles(df_4h, htf_tf)
            if df_4h is None or len(df_4h) < 80:
                self.logger.debug(f"⚠️ [XAU_GOLD] Insufficient {htf_tf} data for {epic}: {0 if df_4h is None else len(df_4h)} bars")
                return None

            df_trigger = self.data_fetcher.get_enhanced_data(
                epic=epic, pair=pair, timeframe=trigger_tf, lookback_hours=trigger_lookback
            )
            df_trigger = self._filter_incomplete_candles(df_trigger, trigger_tf)
            if df_trigger is None or len(df_trigger) < 60:
                self.logger.info(f"⚠️ [XAU_GOLD] Insufficient {trigger_tf} data for {epic}: {0 if df_trigger is None else len(df_trigger)} bars")
                return None

            df_entry = self.data_fetcher.get_enhanced_data(
                epic=epic, pair=pair, timeframe=entry_tf, lookback_hours=entry_lookback
            )
            df_entry = self._filter_incomplete_candles(df_entry, entry_tf)
            if df_entry is None or len(df_entry) < 30:
                self.logger.info(f"⚠️ [XAU_GOLD] Insufficient {entry_tf} data for {epic}: {0 if df_entry is None else len(df_entry)} bars")
                return None

            signal = self.xau_gold_strategy.detect_signal(
                df_trigger=df_trigger,
                df_4h=df_4h,
                df_entry=df_entry,
                epic=epic,
                pair=pair,
            )

            if signal:
                signal = self._add_market_context(signal, df_trigger)

            self.xau_gold_strategy.flush_rejections()
            return signal

        except Exception as e:
            self.logger.error(f"❌ [XAU_GOLD] Error detecting signals for {epic}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def detect_smc_momentum_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None,
        current_timestamp=None,
    ) -> Optional[Dict]:
        """Detect SMC_MOMENTUM (liquidity sweep + rejection wick) signals."""
        try:
            if not hasattr(self, '_smc_momentum_strategy') or self._smc_momentum_strategy is None:
                try:
                    from .strategies.smc_momentum_strategy import SMCMomentumStrategy
                except ImportError:
                    from forex_scanner.core.strategies.smc_momentum_strategy import SMCMomentumStrategy
                self._smc_momentum_strategy = SMCMomentumStrategy(
                    config=None,
                    db_manager=self.db_manager,
                    logger=self.logger,
                    config_override=self._config_override,
                )
                self.logger.info("✅ SMC_MOMENTUM strategy initialized (lazy-load)")

            is_backtest = (
                hasattr(self.data_fetcher, 'current_backtest_time')
                and self.data_fetcher.current_backtest_time is not None
            )
            sim_time = None
            if is_backtest:
                current_id = id(self.data_fetcher)
                if getattr(self, '_smc_momentum_backtest_id', None) != current_id:
                    self._smc_momentum_backtest_id = current_id
                    self._smc_momentum_strategy.reset_cooldowns()
                sim_time = self.data_fetcher.current_backtest_time

            # Data requirements: 4H EMA50 bias, 15m sweep detection, 1H ATR
            # 4H needs 880h lookback to get ~220 4H bars (EMA50 needs 55 trading bars;
            # weekends compress ~10 calendar days to ~42 trading bars — not enough)
            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic, pair=pair, timeframe='4h',
                lookback_hours=4 * 220,  # ~880h = ~37 days = ~220 4H bars
            )
            df_4h = self._filter_incomplete_candles(df_4h, '4h')
            if df_4h is None or len(df_4h) < 55:
                self.logger.debug(f"⚠️ [SMC_MOMENTUM] Insufficient 4H data for {epic}: {0 if df_4h is None else len(df_4h)} bars")
                return None

            df_15m = self.data_fetcher.get_enhanced_data(
                epic=epic, pair=pair, timeframe='15m',
                lookback_hours=48,  # ~192 15m bars
            )
            df_15m = self._filter_incomplete_candles(df_15m, '15m')
            if df_15m is None or len(df_15m) < 30:
                self.logger.debug(f"⚠️ [SMC_MOMENTUM] Insufficient 15m data for {epic}: {0 if df_15m is None else len(df_15m)} bars")
                return None

            df_1h = self.data_fetcher.get_enhanced_data(
                epic=epic, pair=pair, timeframe='1h',
                lookback_hours=72,  # ~72 1H bars for ATR(14)
            )
            df_1h = self._filter_incomplete_candles(df_1h, '1h')

            signal = self._smc_momentum_strategy.detect_signal(
                df_trigger=df_15m,
                df_4h=df_4h,
                df_1h=df_1h,
                epic=epic,
                pair=pair,
                current_time=sim_time,
            )

            if signal:
                signal = self._add_market_context(signal, df_15m)

            self._smc_momentum_strategy.flush_rejections()
            return signal

        except Exception as e:
            self.logger.error(f"❌ [SMC_MOMENTUM] Error for {epic}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def detect_signals_all_strategies(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> List[Dict]:
        """
        Detect signals using all enabled strategies.

        After January 2026 cleanup, only SMC Simple strategy is active.
        """
        all_signals = []

        try:
            self.logger.debug(f"🔍 Running strategy detection for {epic}")

            individual_results = {}

            # Determine which strategy to use via regime-based routing
            routing_result = self._get_routed_strategy(epic, pair)
            routed_strategy = routing_result['strategy']

            if routed_strategy == 'MEAN_REVERSION' and self._strategy_router:
                try:
                    self.logger.debug(f"🔍 [MEAN_REVERSION] Starting detection for {epic}")
                    signal = self.detect_mean_reversion_signals(epic, pair, spread_pips, timeframe, routing_context=routing_result)
                    individual_results['mean_reversion'] = signal
                    if signal:
                        all_signals.append(signal)
                        self.logger.info(f"✅ [MEAN_REVERSION] Signal detected for {epic}: {signal.get('signal')} @ {signal.get('entry_price', 0):.5f}")
                    else:
                        self.logger.debug(f"📊 [MEAN_REVERSION] No signal for {epic}")
                except Exception as e:
                    self.logger.error(f"❌ [MEAN_REVERSION] Error for {epic}: {e}")
                    individual_results['mean_reversion'] = None

            elif self._is_gold_epic(epic):
                try:
                    self.logger.debug(f"🔍 [XAU_GOLD] Starting detection for {epic}")
                    signal = self.detect_xau_gold_signals(epic, pair, spread_pips, timeframe)
                    individual_results['xau_gold'] = signal
                    if signal:
                        all_signals.append(signal)
                        self.logger.info(f"✅ [XAU_GOLD] Signal detected for {epic}: {signal.get('signal')} @ {signal.get('entry_price', 0):.2f}")
                    else:
                        self.logger.debug(f"📊 [XAU_GOLD] No signal for {epic}")
                except Exception as e:
                    self.logger.error(f"❌ [XAU_GOLD] Error for {epic}: {e}")
                    individual_results['xau_gold'] = None

            elif routed_strategy == 'SKIPPED':
                # Negative-routing rule matched in _get_routed_strategy: the
                # selected strategy was vetoed by regime_skip_overrides.
                # Skip is already logged + telemetry captured. Do not fall
                # through to SMC_SIMPLE — that would defeat the purpose.
                self.logger.debug(
                    f"⛔ [{epic}] regime-skip in effect — no FX strategy fires this tick"
                )

            else:
                # Default: SMC_SIMPLE (always enabled after January 2026 cleanup)
                if self.smc_simple_enabled:
                    try:
                        self.logger.debug(f"🔍 [SMC SIMPLE] Starting detection for {epic}")
                        smc_simple_signal = self.detect_smc_simple_signals(epic, pair, spread_pips, timeframe)
                        individual_results['smc_simple'] = smc_simple_signal
                        if smc_simple_signal:
                            all_signals.append(smc_simple_signal)
                            self.logger.info(f"✅ [SMC SIMPLE] Signal detected for {epic}: {smc_simple_signal.get('signal')} @ {smc_simple_signal.get('entry_price', 0):.5f}")
                        else:
                            self.logger.debug(f"📊 [SMC SIMPLE] No signal for {epic}")
                    except Exception as e:
                        self.logger.error(f"❌ [SMC SIMPLE] Error for {epic}: {e}")
                        individual_results['smc_simple'] = None

            # MEAN_REVERSION runs concurrently (not through the router) so
            # we can observe its monitor-only performance alongside whichever
            # strategy the router picked. Internal hard ADX gates filter out
            # trending conditions; per-pair enable flags drop the pairs that
            # underperformed in the 90d standalone eval.
            if (self.mean_reversion_enabled
                    and routed_strategy != 'MEAN_REVERSION'
                    and not self._is_gold_epic(epic)):
                try:
                    self.logger.debug(f"🔍 [MEAN_REVERSION] Starting detection for {epic}")
                    mr_signal = self.detect_mean_reversion_signals(
                        epic, pair, spread_pips, timeframe,
                        routing_context=routing_result,
                    )
                    individual_results['mean_reversion'] = mr_signal
                    if mr_signal:
                        all_signals.append(mr_signal)
                        self.logger.info(
                            f"✅ [MEAN_REVERSION] Signal detected for {epic}: "
                            f"{mr_signal.get('signal')} @ {mr_signal.get('entry_price', 0):.5f}"
                        )
                    else:
                        self.logger.debug(f"📊 [MEAN_REVERSION] No signal for {epic}")
                except Exception as e:
                    self.logger.error(f"❌ [MEAN_REVERSION] Error for {epic}: {e}")
                    individual_results['mean_reversion'] = None

            if (self.smc_momentum_enabled
                    and not self._is_gold_epic(epic)
                    and SMCMomentumConfigService.get_instance().get_config().is_pair_enabled(epic)):
                try:
                    self.logger.debug(f"🔍 [SMC_MOMENTUM] Starting detection for {epic}")
                    mom_signal = self.detect_smc_momentum_signals(epic, pair, spread_pips, timeframe)
                    individual_results['smc_momentum'] = mom_signal
                    if mom_signal:
                        all_signals.append(mom_signal)
                        self.logger.info(
                            f"✅ [SMC_MOMENTUM] Signal detected for {epic}: "
                            f"{mom_signal.get('signal')} @ {mom_signal.get('entry_price', 0):.5f}"
                        )
                    else:
                        self.logger.debug(f"📊 [SMC_MOMENTUM] No signal for {epic}")
                except Exception as e:
                    self.logger.error(f"❌ [SMC_MOMENTUM] Error for {epic}: {e}")
                    individual_results['smc_momentum'] = None

            if self.range_fade_enabled and RangeFadeConfigService.get_instance().get_config().is_pair_enabled(epic):
                try:
                    self.logger.debug(f"🔍 [RANGE_FADE] Starting detection for {epic}")
                    erf_signal = self.detect_range_fade_signals(
                        epic, pair, spread_pips, timeframe,
                        current_timestamp=None,
                        routing_context=routing_result,
                    )
                    individual_results['range_fade'] = erf_signal
                    if erf_signal:
                        all_signals.append(erf_signal)
                        self.logger.info(
                            f"✅ [RANGE_FADE] Signal detected for {epic}: "
                            f"{erf_signal.get('signal')} @ {erf_signal.get('entry_price', 0):.5f}"
                        )
                    else:
                        self.logger.debug(f"📊 [RANGE_FADE] No signal for {epic}")
                except Exception as e:
                    self.logger.error(f"❌ [RANGE_FADE] Error for {epic}: {e}")
                    individual_results['range_fade'] = None

            if (self.impulse_fade_enabled
                    and not self._is_gold_epic(epic)
                    and ImpulseFadeConfigService.get_instance().get_config().is_pair_enabled(epic)):
                try:
                    self.logger.debug(f"🔍 [IMPULSE_FADE] Starting detection for {epic}")
                    if_signal = self.detect_impulse_fade_signals(epic, pair, spread_pips, timeframe)
                    individual_results['impulse_fade'] = if_signal
                    if if_signal:
                        all_signals.append(if_signal)
                        self.logger.info(
                            f"✅ [IMPULSE_FADE] Signal detected for {epic}: "
                            f"{if_signal.get('signal')} @ {if_signal.get('entry_price', 0):.5f}"
                        )
                    else:
                        self.logger.debug(f"📊 [IMPULSE_FADE] No signal for {epic}")
                except Exception as e:
                    self.logger.error(f"❌ [IMPULSE_FADE] Error for {epic}: {e}")
                    individual_results['impulse_fade'] = None

            # Propagate regime info from routing to signals that don't set these fields
            if all_signals and routing_result.get('regime'):
                for sig in all_signals:
                    if not sig.get('market_regime_detected'):
                        sig['market_regime_detected'] = routing_result['regime']
                        sig['regime_confidence'] = routing_result['regime_confidence']
                        sig['adx_value'] = routing_result.get('adx_value')
                        sig['volatility_state'] = routing_result.get('volatility_state')
                    if not sig.get('market_regime'):
                        sig['market_regime'] = routing_result['regime']

            # Add smart money analysis to all signals (if enabled).
            # SignalProcessor.process_signal() (live path) detects the
            # 'smart_money_score' marker set here and skips re-analysis to avoid
            # the doubled data-fetch cost. BacktestScanner does not go through
            # SignalProcessor, so this remains the only place backtests pick it up.
            if all_signals:
                try:
                    all_signals = add_smart_money_to_signals(all_signals, self.data_fetcher, self.db_manager)
                    self.logger.debug(f"✅ Smart money analysis added to {len(all_signals)} signals")
                except Exception as e:
                    self.logger.warning(f"⚠️ Smart money analysis failed: {e}")

            # Results logging
            if all_signals:
                strategy_names = [s.get('strategy', 'unknown') for s in all_signals]
                self.logger.info(f"🎯 {epic}: {len(all_signals)} signals from strategies: {', '.join(strategy_names)}")

                for i, signal in enumerate(all_signals, 1):
                    strategy = signal.get('strategy', 'unknown')
                    signal_type = signal.get('signal_type', 'unknown')
                    confidence = signal.get('confidence_score', 0)
                    self.logger.info(f"   📈 Signal {i}: {strategy} - {signal_type} ({confidence:.1%})")
            else:
                self.logger.debug(f"📊 {epic}: No signals from any strategy")

            return all_signals

        except Exception as e:
            self.logger.error(f"❌ Error in detect_signals_all_strategies for {epic}: {e}")
            return all_signals

    _STRATEGY_ALIASES: Dict[str, str] = {
        'SMC_EMA': 'SMC_SIMPLE',
        'MR': 'MEAN_REVERSION',
        'MEANREV': 'MEAN_REVERSION',
        'XAU': 'XAU_GOLD',
        'GOLD': 'XAU_GOLD',
        'SWEEP': 'SMC_MOMENTUM',
    }

    def _detect_single_strategy(
        self,
        strategy_name: str,
        epic: str,
        pair: str,
        spread_pips: float,
        timeframe: str,
        current_timestamp=None,
    ) -> Optional[Dict]:
        """Run one named strategy and return its signal (or None).

        Canonical dispatch used by BacktestScanner to avoid duplicating the
        strategy→method mapping in two places.  Unknown names are logged and
        return None.
        """
        name = self._STRATEGY_ALIASES.get(strategy_name, strategy_name)
        try:
            if name == 'SMC_SIMPLE':
                signal = self.detect_smc_simple_signals(epic, pair, spread_pips, timeframe)
            elif name == 'XAU_GOLD':
                signal = self.detect_xau_gold_signals(epic, pair, spread_pips, timeframe)
            elif name == 'RANGE_FADE':
                signal = self.detect_range_fade_signals(epic, pair, spread_pips, timeframe, current_timestamp=current_timestamp)
            elif name == 'MEAN_REVERSION':
                signal = self.detect_mean_reversion_signals(epic, pair, spread_pips, timeframe, current_timestamp=current_timestamp)
            elif name == 'SMC_MOMENTUM':
                signal = self.detect_smc_momentum_signals(epic, pair, spread_pips, timeframe, current_timestamp=current_timestamp)
            elif name == 'IMPULSE_FADE':
                signal = self.detect_impulse_fade_signals(epic, pair, spread_pips, timeframe, current_timestamp=current_timestamp)
            else:
                self.logger.error(f"❌ Unknown strategy '{strategy_name}' in _detect_single_strategy")
                return None

            if signal and isinstance(signal, list):
                signal = signal[0]
            return signal
        except Exception as e:
            self.logger.error(f"❌ [{name}] Error for {epic}: {e}")
            return None

    def detect_signals(self, epic: str, pair: str, spread_pips: float = 1.5, timeframe: str = None) -> List[Dict]:
        """
        Main signal detection entry point.
        Delegates to detect_signals_all_strategies.
        """
        return self.detect_signals_all_strategies(epic, pair, spread_pips, timeframe)

    # =========================================================================
    # MARKET CONTEXT AND ENHANCEMENT METHODS
    # =========================================================================

    def _add_market_context(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """ENHANCED: Add comprehensive market context including complete technical indicators"""

        # First, add complete technical indicators
        signal = self._add_complete_technical_indicators(signal, df)

        # Then add existing market context
        if df is None or df.empty:
            return signal

        try:
            latest = df.iloc[-1]

            # Add available market context
            context_fields = [
                'volume_ratio_20', 'distance_to_support_pips', 'distance_to_resistance_pips',
                'trend_alignment', 'consolidation_range_pips', 'bars_since_breakout',
                'rejection_wicks_count', 'consecutive_green_candles', 'consecutive_red_candles'
            ]

            for field in context_fields:
                if field in latest.index:
                    signal[field] = latest[field]

            # Add volume confirmation flag
            if 'volume' in signal and 'volume_sma_20' in latest.index:
                signal['volume_confirmation'] = signal['volume'] > latest['volume_sma_20'] * 1.2

            # Add recent price action summary
            if len(df) >= 5:
                recent_data = df.tail(5)
                signal['recent_price_action'] = {
                    'bars_count': len(recent_data),
                    'high_range': float(recent_data['high'].max()),
                    'low_range': float(recent_data['low'].min()),
                    'avg_volume': float(recent_data.get('ltv', recent_data.get('volume', pd.Series([0]))).mean()),
                    'price_trend': 'bullish' if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else 'bearish'
                }

            self.logger.debug(f"✅ Enhanced signal with complete market context for {signal.get('epic')}")
            return signal

        except Exception as e:
            self.logger.error(f"❌ Error adding enhanced market context: {e}")
            return signal

    def _add_complete_technical_indicators(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """
        Add complete technical indicators from DataFrame to signal.
        This ensures Claude gets ALL available technical data for comprehensive analysis.
        """
        if df is None or df.empty:
            return signal

        try:
            latest = df.iloc[-1]

            # 1. PRICE DATA
            signal.update({
                'current_price': float(latest['close']),
                'open_price': float(latest['open']),
                'high_price': float(latest['high']),
                'low_price': float(latest['low']),
                'close_price': float(latest['close'])
            })

            # 2. EMA INDICATORS
            ema_indicators = {}
            for col in df.columns:
                if col.startswith('ema_') and col.replace('ema_', '').isdigit():
                    try:
                        period = int(col.replace('ema_', ''))
                        ema_indicators[col] = float(latest[col])

                        if period == 9:
                            signal['ema_9'] = float(latest[col])
                            signal['ema_short'] = float(latest[col])
                        elif period == 21:
                            signal['ema_21'] = float(latest[col])
                            signal['ema_long'] = float(latest[col])
                        elif period == 200:
                            signal['ema_200'] = float(latest[col])
                            signal['ema_trend'] = float(latest[col])

                    except (ValueError, KeyError):
                        continue

            if ema_indicators:
                signal.update(ema_indicators)

            # 3. MACD INDICATORS
            macd_indicators = {}
            macd_mappings = {
                'macd_line': ['macd_line', 'macd', 'macd_12_26_9'],
                'macd_signal': ['macd_signal', 'macd_signal_line', 'macd_signal_12_26_9'],
                'macd_histogram': ['macd_histogram', 'macd_hist', 'macd_histogram_12_26_9']
            }

            for standard_name, possible_cols in macd_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            macd_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 4. KAMA INDICATORS
            kama_indicators = {}
            kama_mappings = {
                'kama_value': ['kama_value', 'kama', 'kama_10', 'kama_14'],
                'efficiency_ratio': ['efficiency_ratio', 'kama_er', 'kama_10_er', 'kama_14_er'],
                'kama_trend': ['kama_trend', 'kama_slope', 'kama_10_trend', 'kama_14_trend']
            }

            for standard_name, possible_cols in kama_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            kama_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 5. ADX AND DIRECTIONAL INDICATORS
            adx_indicators = {}
            adx_mappings = {
                'adx': ['adx', 'adx_14'],
                'plus_di': ['plus_di', 'di_plus', 'plus_di_14', '+di'],
                'minus_di': ['minus_di', 'di_minus', 'minus_di_14', '-di'],
                'dx': ['dx', 'dx_14']
            }

            for standard_name, possible_cols in adx_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            adx_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 6. OTHER TECHNICAL INDICATORS
            other_indicators = {}
            other_mappings = {
                'rsi': ['rsi', 'rsi_14'],
                'atr': ['atr', 'atr_14'],
                'bb_upper': ['bb_upper', 'bollinger_upper', 'bb_upper_20_2'],
                'bb_middle': ['bb_middle', 'bollinger_middle', 'bb_middle_20_2'],
                'bb_lower': ['bb_lower', 'bollinger_lower', 'bb_lower_20_2']
            }

            for standard_name, possible_cols in other_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            other_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 7. VOLUME DATA
            volume_fields = ['ltv', 'volume', 'volume_sma_20', 'volume_ratio_20']
            for field in volume_fields:
                if field in df.columns:
                    try:
                        if field == 'ltv' or field == 'volume':
                            signal['volume'] = float(latest[field])
                        else:
                            signal[field] = float(latest[field])
                    except (ValueError, KeyError):
                        continue

            # 8. SWING POINT DATA
            swing_data = {}
            swing_fields = {
                'swing_high': ['swing_high', 'recent_swing_high', 'swing_high_price'],
                'swing_low': ['swing_low', 'recent_swing_low', 'swing_low_price'],
                'distance_to_swing_high_pips': ['distance_to_swing_high_pips', 'swing_high_distance_pips'],
                'distance_to_swing_low_pips': ['distance_to_swing_low_pips', 'swing_low_distance_pips'],
                'nearest_swing_type': ['nearest_swing_type', 'swing_type'],
                'swing_strength': ['swing_strength', 'swing_level_strength']
            }

            for standard_name, possible_cols in swing_fields.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            value = latest[col]
                            if isinstance(value, (int, float)):
                                swing_data[standard_name] = float(value)
                            else:
                                swing_data[standard_name] = str(value)
                            signal[standard_name] = swing_data[standard_name]
                            break
                        except (ValueError, KeyError):
                            continue

            # 9. SUPPORT/RESISTANCE DATA
            sr_data = {}
            sr_fields = {
                'nearest_support': ['nearest_support', 'support_level', 'support_price'],
                'nearest_resistance': ['nearest_resistance', 'resistance_level', 'resistance_price'],
                'distance_to_support_pips': ['distance_to_support_pips', 'support_distance_pips'],
                'distance_to_resistance_pips': ['distance_to_resistance_pips', 'resistance_distance_pips'],
                'support_strength': ['support_strength', 'support_level_strength'],
                'resistance_strength': ['resistance_strength', 'resistance_level_strength'],
                'level_flip_detected': ['level_flip_detected', 'sr_flip_detected'],
                'cluster_risk_level': ['cluster_risk_level', 'sr_cluster_risk']
            }

            for standard_name, possible_cols in sr_fields.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            value = latest[col]
                            if isinstance(value, bool):
                                sr_data[standard_name] = bool(value)
                            elif isinstance(value, (int, float)):
                                sr_data[standard_name] = float(value)
                            else:
                                sr_data[standard_name] = str(value)
                            signal[standard_name] = sr_data[standard_name]
                            break
                        except (ValueError, KeyError):
                            continue

            # 10. ADDITIONAL CONTEXT DATA
            context_fields = [
                'volume_confirmation', 'trend_alignment', 'market_session',
                'consolidation_range_pips', 'bars_since_breakout'
            ]
            for field in context_fields:
                if field in df.columns:
                    try:
                        signal[field] = latest[field]
                    except (ValueError, KeyError):
                        continue

            # 11. CREATE COMPREHENSIVE STRATEGY_INDICATORS JSON
            all_indicators = {}
            all_indicators.update(ema_indicators)
            all_indicators.update(macd_indicators)
            all_indicators.update(kama_indicators)
            all_indicators.update(adx_indicators)
            all_indicators.update(other_indicators)

            # Helper function to clean NaN values
            def clean_nan_values(obj):
                """Recursively replace NaN with None in dict/list structures"""
                if isinstance(obj, dict):
                    return {k: clean_nan_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan_values(item) for item in obj]
                elif isinstance(obj, float):
                    import math
                    return None if math.isnan(obj) or math.isinf(obj) else obj
                else:
                    return obj

            if all_indicators:
                existing_strategy_indicators = signal.get('strategy_indicators', {})
                strategy_name = signal.get('strategy', 'unknown')

                if existing_strategy_indicators:
                    self.logger.info(f"🔍 [{strategy_name}] Found existing strategy_indicators with keys: {list(existing_strategy_indicators.keys())}")
                else:
                    self.logger.info(f"🔍 [{strategy_name}] No existing strategy_indicators found - will use dataframe analysis")

                dataframe_indicators = clean_nan_values({
                    'ema_data': ema_indicators,
                    'macd_data': macd_indicators,
                    'kama_data': kama_indicators,
                    'adx_data': adx_indicators,
                    'swing_data': swing_data,
                    'sr_data': sr_data,
                    'other_indicators': other_indicators,
                    'indicator_count': len(all_indicators),
                    'data_source': 'complete_dataframe_analysis'
                })

                if existing_strategy_indicators:
                    merged_indicators = existing_strategy_indicators.copy()
                    merged_indicators['dataframe_analysis'] = dataframe_indicators
                    signal['strategy_indicators'] = merged_indicators
                    self.logger.info(f"✅ [{strategy_name}] Preserved strategy indicators + added {len(all_indicators)} dataframe indicators")
                else:
                    signal['strategy_indicators'] = dataframe_indicators
                    self.logger.debug(f"📊 Enhanced signal with {len(all_indicators)} indicators + swing/SR data")

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error adding complete technical indicators: {e}")
            return signal

    # =========================================================================
    # MULTI-STRATEGY ROUTING (v3.0.0)
    # =========================================================================

    def _get_routed_strategy(self, epic: str, pair: str) -> Dict:
        """
        Determine which strategy to use based on ADX-derived market regime.

        Uses the same regime detection logic as backtest_scanner._calculate_regime_from_data():
        - ADX < 20: ranging → MEAN_REVERSION / SMC_SIMPLE fallback
        - ADX 20-25: low_volatility → MEAN_REVERSION / SMC_SIMPLE fallback
        - ADX 25-50: trending → SMC_SIMPLE
        - ADX > 50: breakout → SMC_SIMPLE

        Falls back to SMC_SIMPLE if router is disabled or any error occurs.

        Args:
            epic: Trading pair epic (e.g., 'CS.D.EURUSD.MINI.IP')
            pair: Pair name (e.g., 'EURUSD')

        Returns:
            Dict with 'strategy', 'regime', 'regime_confidence', 'adx_value', 'volatility_state'
        """
        default_result = {
            'strategy': 'SMC_SIMPLE',
            'regime': None,
            'regime_confidence': None,
            'adx_value': None,
            'volatility_state': None,
        }
        if not self._strategy_router:
            return default_result

        try:
            # Fetch 1h data for regime detection (same as backtest)
            df = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='1h',
                lookback_hours=72
            )

            if df is None or df.empty or len(df) < 20:
                return default_result

            # Get ADX from latest bar
            adx_value = None
            if 'adx' in df.columns:
                raw_adx = df['adx'].iloc[-1]
                if raw_adx is not None and not pd.isna(raw_adx):
                    adx_value = float(raw_adx)

            # Get ATR for volatility assessment
            atr_value = None
            atr_mean = None
            for col in ('atr', 'ATR'):
                if col in df.columns:
                    atr_value = df[col].iloc[-1]
                    atr_mean = df[col].rolling(20).mean().iloc[-1]
                    break

            # Determine regime from ADX (same thresholds as backtest)
            regime = 'trending'
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
            regime_downgraded = False
            if regime == 'trending':
                # Check 1: Efficiency ratio — is price actually going somewhere?
                er_value = self._get_efficiency_ratio_from_df(df)
                if er_value is not None and er_value < 0.25:
                    self.logger.info(
                        f"📉 [{epic}] Regime downgraded trending→ranging: "
                        f"efficiency_ratio={er_value:.3f} < 0.25 (price going nowhere)"
                    )
                    regime = 'ranging'
                    regime_downgraded = True

                # Check 2: Weekly directional consistency — are weeks alternating?
                if not regime_downgraded:
                    weekly_result = self._check_weekly_directional_consistency(epic, pair)
                    if weekly_result.get('is_oscillating', False):
                        self.logger.info(
                            f"📉 [{epic}] Regime downgraded trending→ranging: "
                            f"weekly oscillation detected ({weekly_result.get('pattern', '')})"
                        )
                        regime = 'ranging'
                        regime_downgraded = True

            # Determine session
            session = self._get_current_session()

            # Route via the strategy router service
            strategy_name, confidence_modifier = self._strategy_router.get_strategy_for_regime(
                regime=regime,
                epic=epic,
                session=session,
                volatility_state=volatility_state,
                adx_value=adx_value
            )

            if strategy_name and strategy_name != 'SMC_SIMPLE':
                self.logger.info(
                    f"🎯 [{epic}] Regime={regime} ADX={f'{adx_value:.1f}' if adx_value else 'N/A'} "
                    f"→ {strategy_name} (conf={confidence_modifier:.2f})"
                    f"{' [downgraded]' if regime_downgraded else ''}"
                )
            else:
                strategy_name = 'SMC_SIMPLE'
                self.logger.debug(
                    f"🎯 [{epic}] Regime={regime} ADX={f'{adx_value:.1f}' if adx_value else 'N/A'} → SMC_SIMPLE"
                    f"{' [downgraded]' if regime_downgraded else ''}"
                )

            # Negative-routing: silently block the selected strategy if it's
            # in regime_skip_overrides for this (epic, regime) combo. Used to
            # plug catastrophic-loss windows (e.g. SMC_SIMPLE in 'breakout'
            # regime on EURJPY/USDJPY at PF 0.03 / 0.10 per 90d data).
            try:
                skip_reason = self._strategy_router.should_skip_strategy(
                    epic=epic, regime=regime, strategy=strategy_name,
                )
            except Exception:
                skip_reason = None
            if skip_reason:
                self.logger.info(
                    f"⛔ [{epic}] {strategy_name} blocked in regime={regime}: {skip_reason}"
                )
                try:
                    self._strategy_router.record_skip(
                        epic=epic,
                        regime=regime,
                        blocked_strategy=strategy_name,
                        routing_confidence=float(confidence_modifier or 0.0),
                        signal_summary={
                            'adx_value': adx_value,
                            'volatility_state': volatility_state,
                            'session': session,
                            'regime_downgraded': regime_downgraded,
                        },
                    )
                except Exception:
                    pass
                strategy_name = 'SKIPPED'

            # Determine regime confidence based on classification
            regime_confidence = 0.5  # default for trending
            if regime == 'high_volatility':
                regime_confidence = 0.85
            elif regime == 'breakout':
                regime_confidence = 0.75
            elif regime == 'ranging':
                regime_confidence = 0.85
            elif regime == 'low_volatility':
                regime_confidence = 0.6

            return {
                'strategy': strategy_name,
                'regime': regime,
                'regime_confidence': regime_confidence,
                'adx_value': adx_value,
                'volatility_state': volatility_state,
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Regime routing error for {epic}, defaulting to SMC_SIMPLE: {e}")
            return default_result

    def _get_current_session(self) -> str:
        """
        Determine current trading session from UTC time.

        Returns:
            Session name: 'asian', 'london', 'new_york', or 'overlap'
        """
        from datetime import timezone
        hour = datetime.now(timezone.utc).hour

        if 13 <= hour < 16:
            return 'overlap'
        elif 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'london'
        elif 16 <= hour < 21:
            return 'new_york'
        else:
            return 'asian'

    def _get_efficiency_ratio_from_df(self, df) -> float:
        """Extract efficiency ratio from DataFrame (KAMA-computed column)."""
        try:
            for col in ('efficiency_ratio', 'kama_er', 'kama_10_er', 'kama_14_er'):
                if col in df.columns:
                    val = df[col].iloc[-1]
                    if val is not None and not pd.isna(val):
                        return float(val)
            return None
        except Exception:
            return None

    def _check_weekly_directional_consistency(self, epic: str, pair: str) -> dict:
        """Check if recent weeks show alternating direction (oscillation vs trend).

        Queries pre-synthesized 1h candles from ig_candles_backtest, aggregates
        to weekly OHLC in SQL. Avoids data_batch_size limits.

        Returns:
            dict with 'is_oscillating' (bool), 'pattern' (str), 'alternations' (int)
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
                  AND start_time >= NOW() - INTERVAL '5 weeks'
                GROUP BY 1
                HAVING COUNT(*) > 20
                ORDER BY 1
            """)

            with self.data_fetcher.db_manager.engine.connect() as conn:
                result = conn.execute(query, {'epic': epic}).fetchall()

            if len(result) < 3:
                return default

            # Exclude current partial week (last row)
            weeks = result[:-1] if len(result) > 3 else result
            if len(weeks) < 3:
                return default

            # Take last 4 complete weeks
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

        except Exception as e:
            self.logger.debug(f"⚠️ Weekly consistency check error for {epic}: {e}")
            return default

    # =========================================================================
    # MULTI-STRATEGY DETECTION METHODS (v3.0.0)
    # =========================================================================

    def detect_mean_reversion_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = '15m',
        current_timestamp: datetime = None,
        routing_context: Dict = None,
    ) -> Optional[Dict]:
        """Detect signals using the Mean Reversion strategy.

        BB + RSI extremes + hard ADX gates (15m primary + 1h HTF). Lazy-loads
        on first use, fetches primary + confirmation timeframes via DataFetcher,
        delegates to the strategy's detect_signal() and enhances the result with
        technical indicators before returning.
        """
        try:
            mean_reversion_strategy = self._get_strategy(
                'MEAN_REVERSION',
                config_override=self._config_override,
            )
            if mean_reversion_strategy is None:
                return None

            # Use the per-pair configured primary timeframe (may differ from the
            # scan cadence timeframe — e.g. USDCHF touch-entry runs on 5m).
            primary_tf = mean_reversion_strategy.get_config().get_pair_primary_timeframe(epic)
            df_trigger = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=primary_tf,
                lookback_hours=48,
            )
            if df_trigger is None or df_trigger.empty:
                self.logger.debug(f"[MEAN_REVERSION] No {timeframe} data for {epic}")
                return None

            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='1h',
                lookback_hours=72,
            )

            signal = mean_reversion_strategy.detect_signal(
                df_trigger=df_trigger,
                df_4h=df_4h,
                epic=epic,
                pair=pair,
                spread_pips=spread_pips,
                current_timestamp=current_timestamp,
                routing_context=routing_context,
            )

            if signal:
                signal = self._add_complete_technical_indicators(signal, df_trigger)

            if hasattr(mean_reversion_strategy, 'flush_rejections'):
                mean_reversion_strategy.flush_rejections()

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error detecting mean reversion signals for {epic}: {e}")
            return None

    def detect_range_fade_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = '5m',
        current_timestamp: datetime = None,
        routing_context: Dict = None,
    ) -> Optional[Dict]:
        """Detect signals using the range-fade strategy family (5m only)."""
        try:
            range_fade_strategy = self._get_strategy('RANGE_FADE', erf_profile='5m')
            if range_fade_strategy is None:
                return None

            df_trigger = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=240,
            )
            if df_trigger is None or df_trigger.empty:
                self.logger.debug(f"[RANGE_FADE] No {timeframe} data for {epic}")
                return None

            df_1h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='1h',
                lookback_hours=336,
            )

            signal = range_fade_strategy.detect_signal(
                df_trigger=df_trigger,
                df_4h=df_1h,
                epic=epic,
                pair=pair,
                spread_pips=spread_pips,
                current_timestamp=current_timestamp,
                routing_context=routing_context,
            )

            if signal:
                signal = self._add_complete_technical_indicators(signal, df_trigger)

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error detecting range-fade signals for {epic}: {e}")
            return None

    def detect_impulse_fade_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = '5m',
        current_timestamp: datetime = None,
    ) -> Optional[Dict]:
        """Detect signals using the Impulse Fade strategy (5m only, late-US session)."""
        try:
            impulse_fade_strategy = self._get_strategy(
                'IMPULSE_FADE',
                config_override=self._config_override,
            )
            if impulse_fade_strategy is None:
                return None

            df_5m = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='5m',
                lookback_hours=48,
            )
            if df_5m is None or df_5m.empty:
                self.logger.debug(f"[IMPULSE_FADE] No 5m data for {epic}")
                return None

            signal = impulse_fade_strategy.detect_signal(
                df_trigger=df_5m,
                epic=epic,
                pair=pair,
                spread_pips=spread_pips,
                current_timestamp=current_timestamp,
            )

            if signal:
                signal = self._add_complete_technical_indicators(signal, df_5m)

            if hasattr(impulse_fade_strategy, 'flush_rejections'):
                impulse_fade_strategy.flush_rejections()

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error detecting impulse-fade signals for {epic}: {e}")
            return None


