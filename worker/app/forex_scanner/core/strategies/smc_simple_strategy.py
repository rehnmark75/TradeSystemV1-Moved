#!/usr/bin/env python3
"""
SMC Simple Strategy - 3-Tier EMA-Based Trend Following

VERSION: 2.5.0
DATE: 2025-12-23
STATUS: Pair-Specific Blocking - USDCHF filtered on weak setups

v2.5.0 CHANGES (Pair-Specific Blocking):
    - NEW: Pair-specific blocking conditions for consistently losing pairs
    - USDCHF: Block when EMA distance >60 pips (0% win rate historically)
    - USDCHF: Require volume confirmation (no vol = 0% win rate)
    - USDCHF: Block momentum entries without volume confirmation
    - USDCHF: Higher confidence threshold (60% vs 50% default)
    - Analysis: USDCHF was worst pair (-$4,917, 26.6% WR) - now filtered

v2.4.0 CHANGES (ATR-Based SL Cap):
    - FIX: Fixed 55 pip cap was still 7x ATR on low-volatility pairs
    - ANALYSIS: Trade 1594 (GBPUSD) had 48 pip SL with only 7.4 pip ATR
    - SOLUTION: Dynamic cap = min(ATR × 3, 30 pips absolute)
    - BENEFIT: SL now proportional to volatility (22 pips for GBPUSD vs 48)
    - IMPACT: Win rate needed drops from 70%+ to ~50% for breakeven

v2.2.0 CHANGES (Confidence Scoring Redesign):
    - FIX: swing_break_quality was in config but NOT implemented - NOW IMPLEMENTED
    - FIX: pullback was over-weighted at 40% (25% + 15% fib) - NOW 20%
    - FIX: volume was binary (15%/5%) - NOW gradient based on spike magnitude
    - FIX: EMA used fixed 30 pips - NOW ATR-normalized for cross-pair fairness
    - NEW: Balanced 5-component scoring (each 20% weight)
    - NEW: Swing break quality scores: body close %, break strength, recency
    - Expected: +2-4% win rate improvement, better tier alignment

v2.1.1 CHANGES (Volume Data Fix):
    - FIX: Use 'ltv' column instead of 'volume' for volume confirmation
    - Analysis: IG provides actual data in 'ltv' (Last Traded Volume), 'volume' is always 0
    - Impact: +10% confidence boost when volume spike detected (0.05 → 0.15)

v2.1.0 CHANGES (R:R Root Cause Fixes):
    - FIX: Reduced SL_ATR_MULTIPLIER 1.2→1.0 (tighter stops = better R:R)
    - FIX: Reduced SL_BUFFER_PIPS 8→6 (less buffer = better R:R)
    - FIX: Reduced pair-specific SL buffers by ~25%
    - FIX: Increased R:R weight in confidence scoring 10%→15%
    - NEW: Dynamic swing lookback based on ATR volatility
    - Analysis: 451 R:R rejections (0.01-0.56) were due to inflated SL distances

v2.0.0 CHANGES (Phase 3 - Limit Orders):
    - NEW: Limit order support with intelligent price offsets
    - v2.2.0: CHANGED to stop-entry style (momentum confirmation)
    - BUY orders placed ABOVE market (enter when price breaks up)
    - SELL orders placed BELOW market (enter when price breaks down)
    - Max offset: 3 pips, 6-minute auto-expiry

v1.8.0 CHANGES (Phase 2):
    - NEW: Momentum continuation entry mode (price beyond break = valid entry)
    - NEW: ATR-based swing validation (adapts to pair volatility)
    - NEW: Momentum quality filter (strong breakout candles only)
    - Improved entry type tracking (pullback vs momentum)

v1.7.0 CHANGES (Phase 1):
    - FIX: SL calculation now uses opposite_swing (not swing_level)
    - Wider Fib zones: 23.6%-70% (was 38%-62%)
    - Lower R:R minimum: 1.5 (was 2.5)
    - Lower confidence threshold: 60% (was 80%)

Strategy Architecture:
    TIER 1: 4H 50 EMA for directional bias (institutional standard)
    TIER 2: 15m swing break with body-close confirmation
    TIER 3: 5m pullback OR momentum continuation entry

Entry Types:
    - PULLBACK: Price retraces 23.6%-70% of swing (classic Fib entry)
    - MOMENTUM: Price continues beyond break point (trend continuation)

Target Performance (v2.1.0):
    - Win Rate: 50%+ (improved with better R:R filtering)
    - Profit Factor: 1.5+
    - More signals passing R:R filter due to tighter SL
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta, timezone


class SMCSimpleStrategy:
    """
    Simplified SMC strategy using 3-tier EMA-based approach

    Entry Requirements:
    1. TIER 1: Price on correct side of 4H 50 EMA (trend bias)
    2. TIER 2: 1H candle BODY closes beyond swing high/low (momentum)
    3. TIER 3: 15m price pulls back to Fibonacci zone (entry timing)
    """

    def __init__(self, config, logger=None, db_manager=None):
        """Initialize SMC Simple Strategy

        Args:
            config: Main config module
            logger: Logger instance (optional)
            db_manager: DatabaseManager for rejection tracking (optional)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._db_manager = db_manager  # Store for later initialization

        # Load configuration (sets self.rejection_tracking_enabled)
        self._load_config()

        # State tracking
        self.pair_cooldowns = {}  # {pair: last_signal_time}
        self.recent_signals = {}  # {pair: [(timestamp, price), ...]}
        self.pending_entries = {}  # {pair: pending_entry_data}

        # v3.0.0: Adaptive cooldown state tracking
        self.pair_consecutive_losses = {}  # {pair: int} - consecutive loss count
        self.pair_last_session = {}  # {pair: session_name} - track session changes
        self._trade_outcome_cache = {}  # {pair: {'outcome': dict, 'cached_at': datetime}}

        # Rejection tracking (v2.2.0) - initialized after config is loaded
        self.rejection_manager = None
        if self._db_manager is not None and self.rejection_tracking_enabled:
            try:
                from forex_scanner.alerts.smc_rejection_history import SMCRejectionHistoryManager
                self.rejection_manager = SMCRejectionHistoryManager(db_manager)
                self.logger.info("   Rejection tracking: ENABLED")
            except Exception as e:
                self.logger.warning(f"   Rejection tracking: DISABLED (failed to init: {e})")

        self.logger.info("=" * 60)
        self.logger.info("✅ SMC Simple Strategy v1.0.0 initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"   TIER 1: {self.ema_period} EMA on {self.htf_timeframe}")
        self.logger.info(f"   TIER 2: Swing break on {self.trigger_tf}")
        self.logger.info(f"   TIER 3: Pullback entry on {self.entry_tf}")
        self.logger.info(f"   Min R:R: {self.min_rr_ratio}")
        self.logger.info(f"   Fib Zone: {self.fib_min*100:.1f}% - {self.fib_max*100:.1f}%")
        self.logger.info("=" * 60)

    def _load_config(self):
        """Load configuration from database service (with in-memory cache fallback)"""
        # Try database-driven config first
        try:
            from forex_scanner.services.smc_simple_config_service import (
                get_smc_simple_config_service,
                SMCSimpleConfig
            )

            service = get_smc_simple_config_service()
            config = service.get_config()

            # Store service reference for per-pair lookups
            self._config_service = service
            self._using_database_config = True

            # Map config object to instance attributes
            self.strategy_version = config.version
            self.strategy_name = config.strategy_name

            # TIER 1: HTF Settings
            self.htf_timeframe = config.htf_timeframe
            self.ema_period = config.ema_period
            self.ema_buffer_pips = config.ema_buffer_pips
            self.require_close_beyond_ema = config.require_close_beyond_ema
            self.min_distance_from_ema = config.min_distance_from_ema_pips

            # TIER 2: Trigger Settings
            self.trigger_tf = config.trigger_timeframe
            self.swing_lookback = config.swing_lookback_bars
            self.swing_strength = config.swing_strength_bars
            self.require_body_close = config.require_body_close_break
            self.wick_tolerance_pips = config.wick_tolerance_pips

            # Volume confirmation
            self.volume_enabled = config.volume_confirmation_enabled
            self.volume_sma_period = config.volume_sma_period
            self.volume_multiplier = config.volume_spike_multiplier

            # TIER 3: Entry Settings
            self.entry_tf = config.entry_timeframe
            self.pullback_enabled = config.pullback_enabled
            self.fib_min = config.fib_pullback_min
            self.fib_max = config.fib_pullback_max
            self.fib_optimal = (config.fib_optimal_zone_min, config.fib_optimal_zone_max)
            self.max_pullback_wait = config.max_pullback_wait_bars
            self.pullback_confirm_bars = config.pullback_confirmation_bars

            # Risk Management
            self.min_rr_ratio = config.min_rr_ratio
            self.optimal_rr = config.optimal_rr_ratio
            self.max_rr_ratio = config.max_rr_ratio
            self.sl_buffer_pips = config.sl_buffer_pips
            self.min_tp_pips = config.min_tp_pips
            self.use_swing_target = config.use_swing_target

            # Session Filter
            self.session_filter_enabled = config.session_filter_enabled
            self.block_asian = config.block_asian_session

            # Signal Limits
            self.cooldown_hours = config.signal_cooldown_hours

            # v3.0.0: Adaptive Cooldown Configuration
            self.adaptive_cooldown_enabled = config.adaptive_cooldown_enabled
            self.base_cooldown_hours = config.base_cooldown_hours
            self.cooldown_after_win_multiplier = config.cooldown_after_win_multiplier
            self.cooldown_after_loss_multiplier = config.cooldown_after_loss_multiplier
            self.consecutive_loss_penalty_hours = config.consecutive_loss_penalty_hours
            self.max_consecutive_losses_before_block = config.max_consecutive_losses_before_block
            self.consecutive_loss_block_hours = config.consecutive_loss_block_hours
            self.win_rate_lookback_trades = config.win_rate_lookback_trades
            self.high_win_rate_threshold = config.high_win_rate_threshold
            self.low_win_rate_threshold = config.low_win_rate_threshold
            self.critical_win_rate_threshold = config.critical_win_rate_threshold
            self.high_win_rate_cooldown_reduction = config.high_win_rate_cooldown_reduction
            self.low_win_rate_cooldown_increase = config.low_win_rate_cooldown_increase
            self.high_volatility_atr_multiplier = config.high_volatility_atr_multiplier
            self.volatility_cooldown_adjustment = config.volatility_cooldown_adjustment
            self.strong_trend_cooldown_reduction = config.strong_trend_cooldown_reduction
            self.session_change_reset_cooldown = config.session_change_reset_cooldown
            self.min_cooldown_hours = config.min_cooldown_hours
            self.max_cooldown_hours = config.max_cooldown_hours

            # Confidence thresholds
            self.min_confidence = config.min_confidence_threshold

            # v2.9.0: Data-driven filters
            self.max_confidence = config.max_confidence_threshold
            self.min_volume_ratio = config.min_volume_ratio
            self.volume_filter_enabled = config.volume_filter_enabled
            self.allow_no_volume_data = config.allow_no_volume_data

            # v1.8.0 Phase 2: Momentum Continuation Mode
            self.momentum_mode_enabled = config.momentum_mode_enabled
            self.momentum_min_depth = config.momentum_min_depth
            self.momentum_max_depth = config.momentum_max_depth
            self.momentum_confidence_penalty = config.momentum_confidence_penalty

            # v1.8.0 Phase 2: ATR-based Swing Validation
            self.use_atr_swing_validation = config.use_atr_swing_validation
            self.atr_period = config.atr_period
            self.min_swing_atr_multiplier = config.min_swing_atr_multiplier
            self.fallback_min_swing_pips = config.fallback_min_swing_pips

            # v1.8.0 Phase 2: Momentum Quality Filter
            self.momentum_quality_enabled = config.momentum_quality_enabled
            self.min_breakout_atr_ratio = config.min_breakout_atr_ratio
            self.min_body_percentage = config.min_body_percentage

            # v1.9.0: Pair-specific SL buffers and confidence floors (now from DB)
            self.pair_sl_buffers = {}  # Loaded from per-pair overrides in DB
            self.pair_min_confidence = {}  # Loaded from per-pair overrides in DB
            self.sl_atr_multiplier = config.sl_atr_multiplier
            self.use_atr_stop = config.use_atr_stop

            # v2.11.0: Dynamic confidence thresholds
            self.volume_adjusted_confidence_enabled = config.volume_adjusted_confidence_enabled
            self.high_volume_threshold = config.high_volume_threshold
            self.pair_high_volume_confidence = {}  # From DB per-pair overrides
            self.atr_adjusted_confidence_enabled = config.atr_adjusted_confidence_enabled
            self.low_atr_threshold = config.low_atr_threshold
            self.high_atr_threshold = config.high_atr_threshold
            self.pair_low_atr_confidence = {}  # From DB per-pair overrides
            self.pair_high_atr_confidence = {}  # From DB per-pair overrides
            self.ema_distance_adjusted_confidence_enabled = config.ema_distance_adjusted_confidence_enabled
            self.near_ema_threshold_pips = config.near_ema_threshold_pips
            self.far_ema_threshold_pips = config.far_ema_threshold_pips
            self.pair_near_ema_confidence = {}  # From DB per-pair overrides
            self.pair_far_ema_confidence = {}  # From DB per-pair overrides

            # v2.0.0: Limit Order Configuration
            self.limit_order_enabled = config.limit_order_enabled
            self.limit_expiry_minutes = config.limit_expiry_minutes
            self.pullback_offset_atr_factor = config.pullback_offset_atr_factor
            self.pullback_offset_min_pips = config.pullback_offset_min_pips
            self.pullback_offset_max_pips = config.pullback_offset_max_pips
            self.momentum_offset_pips = config.momentum_offset_pips
            self.min_risk_after_offset_pips = config.min_risk_after_offset_pips
            self.max_risk_after_offset_pips = config.max_risk_after_offset_pips

            # v2.4.0: ATR-based SL cap
            self.max_sl_atr_multiplier = config.max_sl_atr_multiplier
            self.max_sl_absolute_pips = config.max_sl_absolute_pips

            # v2.1.0: Dynamic Swing Lookback Configuration
            self.use_dynamic_swing_lookback = config.use_dynamic_swing_lookback
            self.swing_lookback_atr_low = config.swing_lookback_atr_low
            self.swing_lookback_atr_high = config.swing_lookback_atr_high
            self.swing_lookback_min = config.swing_lookback_min
            self.swing_lookback_max = config.swing_lookback_max

            # Debug
            self.debug_logging = config.enable_debug_logging

            # v2.2.0: Rejection Tracking
            self.rejection_tracking_enabled = config.rejection_tracking_enabled
            self.rejection_log_to_console = config.rejection_log_to_console

            # v2.6.0: Pair-specific parameter overrides (now from DB)
            self.pair_parameter_overrides = {}  # Loaded from DB per-pair overrides

            # Store config for helper functions
            self._db_config = config

            self.logger.info(f"✅ SMC Simple config v{config.version} loaded from DATABASE (source: {config.source})")
            return

        except Exception as e:
            self.logger.warning(f"Database config unavailable: {e}, falling back to file config")
            self._using_database_config = False
            self._config_service = None
            self._db_config = None

        # Fallback to file-based config
        self._load_config_from_file()

    def _load_config_from_file(self):
        """Load configuration from file (fallback when database unavailable)"""
        # Import config module
        try:
            from configdata.strategies import config_smc_simple as smc_config
        except ImportError:
            try:
                from forex_scanner.configdata.strategies import config_smc_simple as smc_config
            except ImportError:
                from ..configdata.strategies import config_smc_simple as smc_config

        self._using_database_config = False
        self._config_service = None
        self._db_config = None

        # Strategy metadata
        self.strategy_version = getattr(smc_config, 'STRATEGY_VERSION', '1.0.0')
        self.strategy_name = getattr(smc_config, 'STRATEGY_NAME', 'SMC_SIMPLE')

        # TIER 1: HTF Settings
        self.htf_timeframe = getattr(smc_config, 'HTF_TIMEFRAME', '4h')
        self.ema_period = getattr(smc_config, 'EMA_PERIOD', 50)
        self.ema_buffer_pips = getattr(smc_config, 'EMA_BUFFER_PIPS', 5)
        self.require_close_beyond_ema = getattr(smc_config, 'REQUIRE_CLOSE_BEYOND_EMA', True)
        self.min_distance_from_ema = getattr(smc_config, 'MIN_DISTANCE_FROM_EMA_PIPS', 10)

        # TIER 2: Trigger Settings
        self.trigger_tf = getattr(smc_config, 'TRIGGER_TIMEFRAME', '1h')
        self.swing_lookback = getattr(smc_config, 'SWING_LOOKBACK_BARS', 20)
        self.swing_strength = getattr(smc_config, 'SWING_STRENGTH_BARS', 3)
        self.require_body_close = getattr(smc_config, 'REQUIRE_BODY_CLOSE_BREAK', True)
        self.wick_tolerance_pips = getattr(smc_config, 'WICK_TOLERANCE_PIPS', 2)

        # Volume confirmation
        self.volume_enabled = getattr(smc_config, 'VOLUME_CONFIRMATION_ENABLED', True)
        self.volume_sma_period = getattr(smc_config, 'VOLUME_SMA_PERIOD', 20)
        self.volume_multiplier = getattr(smc_config, 'VOLUME_SPIKE_MULTIPLIER', 1.3)

        # TIER 3: Entry Settings
        self.entry_tf = getattr(smc_config, 'ENTRY_TIMEFRAME', '15m')
        self.pullback_enabled = getattr(smc_config, 'PULLBACK_ENABLED', True)
        self.fib_min = getattr(smc_config, 'FIB_PULLBACK_MIN', 0.236)
        self.fib_max = getattr(smc_config, 'FIB_PULLBACK_MAX', 0.500)
        self.fib_optimal = getattr(smc_config, 'FIB_OPTIMAL_ZONE', (0.382, 0.500))
        self.max_pullback_wait = getattr(smc_config, 'MAX_PULLBACK_WAIT_BARS', 12)
        self.pullback_confirm_bars = getattr(smc_config, 'PULLBACK_CONFIRMATION_BARS', 2)

        # Risk Management
        self.min_rr_ratio = getattr(smc_config, 'MIN_RR_RATIO', 1.5)
        self.optimal_rr = getattr(smc_config, 'OPTIMAL_RR_RATIO', 2.0)
        self.max_rr_ratio = getattr(smc_config, 'MAX_RR_RATIO', 4.0)
        self.sl_buffer_pips = getattr(smc_config, 'SL_BUFFER_PIPS', 8)
        self.min_tp_pips = getattr(smc_config, 'MIN_TP_PIPS', 5)
        self.use_swing_target = getattr(smc_config, 'USE_SWING_TARGET', True)

        # Session Filter
        self.session_filter_enabled = getattr(smc_config, 'SESSION_FILTER_ENABLED', True)
        self.block_asian = getattr(smc_config, 'BLOCK_ASIAN_SESSION', True)

        # Signal Limits
        # NOTE: MAX_SIGNALS_PER_PAIR_PER_DAY removed in v3.0.0 - was never enforced
        # Adaptive cooldown now manages signal frequency dynamically
        self.cooldown_hours = getattr(smc_config, 'SIGNAL_COOLDOWN_HOURS', 4)

        # v3.0.0: Adaptive Cooldown Configuration
        self.adaptive_cooldown_enabled = getattr(smc_config, 'ADAPTIVE_COOLDOWN_ENABLED', False)
        self.base_cooldown_hours = getattr(smc_config, 'BASE_COOLDOWN_HOURS', 2.0)
        self.cooldown_after_win_multiplier = getattr(smc_config, 'COOLDOWN_AFTER_WIN_MULTIPLIER', 0.5)
        self.cooldown_after_loss_multiplier = getattr(smc_config, 'COOLDOWN_AFTER_LOSS_MULTIPLIER', 1.5)
        self.consecutive_loss_penalty_hours = getattr(smc_config, 'CONSECUTIVE_LOSS_PENALTY_HOURS', 1.0)
        self.max_consecutive_losses_before_block = getattr(smc_config, 'MAX_CONSECUTIVE_LOSSES_BEFORE_BLOCK', 3)
        self.consecutive_loss_block_hours = getattr(smc_config, 'CONSECUTIVE_LOSS_BLOCK_HOURS', 8.0)
        self.win_rate_lookback_trades = getattr(smc_config, 'WIN_RATE_LOOKBACK_TRADES', 20)
        self.high_win_rate_threshold = getattr(smc_config, 'HIGH_WIN_RATE_THRESHOLD', 0.60)
        self.low_win_rate_threshold = getattr(smc_config, 'LOW_WIN_RATE_THRESHOLD', 0.40)
        self.critical_win_rate_threshold = getattr(smc_config, 'CRITICAL_WIN_RATE_THRESHOLD', 0.30)
        self.high_win_rate_cooldown_reduction = getattr(smc_config, 'HIGH_WIN_RATE_COOLDOWN_REDUCTION', 0.25)
        self.low_win_rate_cooldown_increase = getattr(smc_config, 'LOW_WIN_RATE_COOLDOWN_INCREASE', 0.50)
        self.high_volatility_atr_multiplier = getattr(smc_config, 'HIGH_VOLATILITY_ATR_MULTIPLIER', 1.5)
        self.volatility_cooldown_adjustment = getattr(smc_config, 'VOLATILITY_COOLDOWN_ADJUSTMENT', 0.30)
        self.strong_trend_cooldown_reduction = getattr(smc_config, 'STRONG_TREND_COOLDOWN_REDUCTION', 0.30)
        self.session_change_reset_cooldown = getattr(smc_config, 'SESSION_CHANGE_RESET_COOLDOWN', True)
        self.min_cooldown_hours = getattr(smc_config, 'MIN_COOLDOWN_HOURS', 1.0)
        self.max_cooldown_hours = getattr(smc_config, 'MAX_COOLDOWN_HOURS', 12.0)

        # Confidence thresholds
        self.min_confidence = getattr(smc_config, 'MIN_CONFIDENCE_THRESHOLD', 0.50)

        # v2.9.0: Data-driven filters (from 85-trade analysis Dec 2025)
        self.max_confidence = getattr(smc_config, 'MAX_CONFIDENCE_THRESHOLD', 0.75)
        self.min_volume_ratio = getattr(smc_config, 'MIN_VOLUME_RATIO', 0.50)
        self.volume_filter_enabled = getattr(smc_config, 'VOLUME_FILTER_ENABLED', True)
        self.allow_no_volume_data = getattr(smc_config, 'ALLOW_NO_VOLUME_DATA', True)

        # v1.8.0 Phase 2: Momentum Continuation Mode
        self.momentum_mode_enabled = getattr(smc_config, 'MOMENTUM_MODE_ENABLED', True)
        self.momentum_min_depth = getattr(smc_config, 'MOMENTUM_MIN_DEPTH', -0.20)
        self.momentum_max_depth = getattr(smc_config, 'MOMENTUM_MAX_DEPTH', 0.0)
        self.momentum_confidence_penalty = getattr(smc_config, 'MOMENTUM_CONFIDENCE_PENALTY', 0.05)

        # v1.8.0 Phase 2: ATR-based Swing Validation
        self.use_atr_swing_validation = getattr(smc_config, 'USE_ATR_SWING_VALIDATION', True)
        self.atr_period = getattr(smc_config, 'ATR_PERIOD', 14)
        self.min_swing_atr_multiplier = getattr(smc_config, 'MIN_SWING_ATR_MULTIPLIER', 0.25)
        self.fallback_min_swing_pips = getattr(smc_config, 'FALLBACK_MIN_SWING_PIPS', 5)

        # v1.8.0 Phase 2: Momentum Quality Filter
        self.momentum_quality_enabled = getattr(smc_config, 'MOMENTUM_QUALITY_ENABLED', True)
        self.min_breakout_atr_ratio = getattr(smc_config, 'MIN_BREAKOUT_ATR_RATIO', 0.5)
        self.min_body_percentage = getattr(smc_config, 'MIN_BODY_PERCENTAGE', 0.60)

        # v1.9.0: Pair-specific SL buffers and confidence floors
        self.pair_sl_buffers = getattr(smc_config, 'PAIR_SL_BUFFERS', {})
        self.pair_min_confidence = getattr(smc_config, 'PAIR_MIN_CONFIDENCE', {})
        self.sl_atr_multiplier = getattr(smc_config, 'SL_ATR_MULTIPLIER', 1.2)
        self.use_atr_stop = getattr(smc_config, 'USE_ATR_STOP', True)

        # v2.11.0: Volume-adjusted confidence thresholds (per-pair)
        self.volume_adjusted_confidence_enabled = getattr(smc_config, 'VOLUME_ADJUSTED_CONFIDENCE_ENABLED', True)
        self.high_volume_threshold = getattr(smc_config, 'HIGH_VOLUME_THRESHOLD', 0.70)
        self.pair_high_volume_confidence = getattr(smc_config, 'PAIR_HIGH_VOLUME_CONFIDENCE', {})

        # v2.11.0: ATR-adjusted confidence thresholds (per-pair)
        self.atr_adjusted_confidence_enabled = getattr(smc_config, 'ATR_ADJUSTED_CONFIDENCE_ENABLED', True)
        self.low_atr_threshold = getattr(smc_config, 'LOW_ATR_THRESHOLD', 0.0004)
        self.high_atr_threshold = getattr(smc_config, 'HIGH_ATR_THRESHOLD', 0.0008)
        self.pair_low_atr_confidence = getattr(smc_config, 'PAIR_LOW_ATR_CONFIDENCE', {})
        self.pair_high_atr_confidence = getattr(smc_config, 'PAIR_HIGH_ATR_CONFIDENCE', {})

        # v2.11.0: EMA distance-adjusted confidence thresholds (per-pair)
        self.ema_distance_adjusted_confidence_enabled = getattr(smc_config, 'EMA_DISTANCE_ADJUSTED_CONFIDENCE_ENABLED', True)
        self.near_ema_threshold_pips = getattr(smc_config, 'NEAR_EMA_THRESHOLD_PIPS', 20)
        self.far_ema_threshold_pips = getattr(smc_config, 'FAR_EMA_THRESHOLD_PIPS', 30)
        self.pair_near_ema_confidence = getattr(smc_config, 'PAIR_NEAR_EMA_CONFIDENCE', {})
        self.pair_far_ema_confidence = getattr(smc_config, 'PAIR_FAR_EMA_CONFIDENCE', {})

        # v2.0.0: Limit Order Configuration
        self.limit_order_enabled = getattr(smc_config, 'LIMIT_ORDER_ENABLED', True)
        self.limit_expiry_minutes = getattr(smc_config, 'LIMIT_EXPIRY_MINUTES', 6)
        self.pullback_offset_atr_factor = getattr(smc_config, 'PULLBACK_OFFSET_ATR_FACTOR', 0.3)
        self.pullback_offset_min_pips = getattr(smc_config, 'PULLBACK_OFFSET_MIN_PIPS', 3.0)
        self.pullback_offset_max_pips = getattr(smc_config, 'PULLBACK_OFFSET_MAX_PIPS', 8.0)
        self.momentum_offset_pips = getattr(smc_config, 'MOMENTUM_OFFSET_PIPS', 4.0)
        self.min_risk_after_offset_pips = getattr(smc_config, 'MIN_RISK_AFTER_OFFSET_PIPS', 5.0)
        self.max_risk_after_offset_pips = getattr(smc_config, 'MAX_RISK_AFTER_OFFSET_PIPS', 40.0)

        # v2.4.0: ATR-based SL cap (dynamic max risk based on volatility)
        self.max_sl_atr_multiplier = getattr(smc_config, 'MAX_SL_ATR_MULTIPLIER', 3.0)
        self.max_sl_absolute_pips = getattr(smc_config, 'MAX_SL_ABSOLUTE_PIPS', 30.0)

        # v2.1.0: Dynamic Swing Lookback Configuration
        self.use_dynamic_swing_lookback = getattr(smc_config, 'USE_DYNAMIC_SWING_LOOKBACK', True)
        self.swing_lookback_atr_low = getattr(smc_config, 'SWING_LOOKBACK_ATR_LOW', 8)
        self.swing_lookback_atr_high = getattr(smc_config, 'SWING_LOOKBACK_ATR_HIGH', 15)
        self.swing_lookback_min = getattr(smc_config, 'SWING_LOOKBACK_MIN', 15)
        self.swing_lookback_max = getattr(smc_config, 'SWING_LOOKBACK_MAX', 30)

        # Debug
        self.debug_logging = getattr(smc_config, 'ENABLE_DEBUG_LOGGING', True)

        # v2.2.0: Rejection Tracking
        self.rejection_tracking_enabled = getattr(smc_config, 'REJECTION_TRACKING_ENABLED', False)
        self.rejection_log_to_console = getattr(smc_config, 'REJECTION_LOG_TO_CONSOLE', False)

        # v2.6.0: Pair-specific parameter overrides
        self.pair_parameter_overrides = getattr(smc_config, 'PAIR_PARAMETER_OVERRIDES', {})

    def _get_pair_param(self, epic: str, param_name: str, default_value):
        """
        Get parameter with pair-specific override if configured.
        Uses database service if available, otherwise falls back to file config.

        Args:
            epic: The trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
            param_name: The parameter name to look up (e.g., 'MIN_BODY_PERCENTAGE')
            default_value: The default value to return if no override exists

        Returns:
            The overridden value if configured, otherwise the default value
        """
        # Use database config service if available
        if hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
            return self._db_config.get_for_pair(epic, param_name, default_value)

        # Fallback to file-based overrides
        config = self.pair_parameter_overrides.get(epic)
        if config and config.get('enabled', False):
            overrides = config.get('overrides', {})
            if param_name in overrides:
                return overrides[param_name]
        return default_value

    def detect_signal(
        self,
        df_trigger: pd.DataFrame,
        df_4h: pd.DataFrame,
        epic: str,
        pair: str,
        df_entry: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Detect trading signal using 3-tier approach

        Args:
            df_trigger: Trigger timeframe data (15m or 1h based on config)
            df_4h: 4H timeframe OHLCV data (bias timeframe)
            epic: IG Markets epic code
            pair: Currency pair name
            df_entry: Optional entry timeframe data (5m or 15m based on config)

        Returns:
            Signal dict or None if no valid signal
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🔍 SMC SIMPLE Strategy v{self.strategy_version} - Signal Detection")
        self.logger.info(f"   Pair: {pair} ({epic})")
        self.logger.info(f"   Timeframes: 4H (bias) → {self.trigger_tf} (trigger) → {self.entry_tf} (entry)")
        self.logger.info(f"   df_entry received: {'Yes, ' + str(len(df_entry)) + ' bars' if df_entry is not None else 'NO - None'}")
        self.logger.info(f"{'='*70}")

        # Get candle timestamp for session check
        # Try 'start_time' column first (common in backtest data), then fall back to index
        if 'start_time' in df_trigger.columns:
            candle_timestamp = df_trigger['start_time'].iloc[-1]
        elif 'timestamp' in df_trigger.columns:
            candle_timestamp = df_trigger['timestamp'].iloc[-1]
        else:
            candle_timestamp = df_trigger.index[-1]

        # Get pip value
        pip_value = 0.01 if 'JPY' in pair else 0.0001

        try:
            # ================================================================
            # PRE-FILTER: Session Check
            # ================================================================
            if self.session_filter_enabled:
                session_valid, session_reason = self._check_session(candle_timestamp, epic)
                if not session_valid:
                    self.logger.info(f"\n🕐 SESSION FILTER: {session_reason}")
                    # Track rejection
                    self._track_rejection(
                        stage='SESSION',
                        reason=session_reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value)
                    )
                    return None
                self.logger.info(f"\n🕐 SESSION: {session_reason}")

            # ================================================================
            # PRE-FILTER: Cooldown Check
            # ================================================================
            # Convert Timestamp to datetime for cooldown comparison (handles backtest properly)
            # Handle various timestamp types: pd.Timestamp, datetime, numpy.int64 (nanoseconds)
            if hasattr(candle_timestamp, 'to_pydatetime'):
                candle_dt = candle_timestamp.to_pydatetime()
            elif isinstance(candle_timestamp, (int, np.integer)):
                # numpy.int64 nanosecond timestamp - convert to datetime
                candle_dt = pd.Timestamp(candle_timestamp).to_pydatetime()
            else:
                candle_dt = candle_timestamp
            cooldown_valid, cooldown_reason = self._check_cooldown(pair, candle_dt)
            if not cooldown_valid:
                self.logger.info(f"\n⏱️  COOLDOWN: {cooldown_reason}")
                # Track rejection
                self._track_rejection(
                    stage='COOLDOWN',
                    reason=cooldown_reason,
                    epic=epic,
                    pair=pair,
                    candle_timestamp=candle_timestamp,
                    context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value)
                )
                return None

            # ================================================================
            # TIER 1: 4H EMA Directional Bias
            # ================================================================
            self.logger.info(f"\n📊 TIER 1: Checking 4H {self.ema_period} EMA Bias")

            ema_result = self._check_ema_bias(df_4h, pip_value)

            if not ema_result['valid']:
                self.logger.info(f"   ❌ {ema_result['reason']}")
                # Track rejection
                self._track_rejection(
                    stage='TIER1_EMA',
                    reason=ema_result['reason'],
                    epic=epic,
                    pair=pair,
                    candle_timestamp=candle_timestamp,
                    context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result)
                )
                return None

            direction = ema_result['direction']  # 'BULL' or 'BEAR'
            ema_value = ema_result['ema_value']
            ema_distance = ema_result['distance_pips']

            self.logger.info(f"   ✅ Direction: {direction}")
            self.logger.info(f"   ✅ 50 EMA: {ema_value:.5f}")
            self.logger.info(f"   ✅ Distance: {ema_distance:.1f} pips from EMA")

            # ================================================================
            # TIER 2: Swing Break Confirmation (15m or 1H based on config)
            # ================================================================
            self.logger.info(f"\n📈 TIER 2: Checking {self.trigger_tf} Swing Break")

            swing_result = self._check_swing_break(df_trigger, direction, pip_value, epic)

            if not swing_result['valid']:
                self.logger.info(f"   ❌ {swing_result['reason']}")
                # Track rejection
                self._track_rejection(
                    stage='TIER2_SWING',
                    reason=swing_result['reason'],
                    epic=epic,
                    pair=pair,
                    candle_timestamp=candle_timestamp,
                    direction=direction,
                    context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, direction=direction)
                )
                return None

            swing_level = swing_result['swing_level']
            opposite_swing = swing_result['opposite_swing']  # v1.6.0: For Fib calculation
            break_candle = swing_result['break_candle']
            volume_confirmed = swing_result['volume_confirmed']

            self.logger.info(f"   ✅ Swing {'High' if direction == 'BEAR' else 'Low'} Break: {swing_level:.5f}")
            self.logger.info(f"   ✅ Opposite Swing: {opposite_swing:.5f}")  # v1.6.0
            self.logger.info(f"   ✅ Body Close Confirmed: Yes")
            if self.volume_enabled:
                self.logger.info(f"   {'✅' if volume_confirmed else '⚠️ '} Volume Spike: {'Yes' if volume_confirmed else 'No (optional)'}")

            # ================================================================
            # TIER 3: Pullback Entry Zone (5m or 15m based on config)
            # ================================================================
            self.logger.info(f"\n🎯 TIER 3: Checking {self.entry_tf} Pullback Zone")

            if df_entry is None or len(df_entry) < 10:
                self.logger.info(f"   ⚠️  No {self.entry_tf} data available, using {self.trigger_tf} for entry")
                entry_df = df_trigger
            else:
                entry_df = df_entry

            pullback_result = self._check_pullback_zone(
                entry_df,
                direction,
                swing_level,
                opposite_swing,  # v1.6.0: Pass opposite swing for correct Fib calc
                break_candle,
                pip_value,
                epic  # v2.6.0: Pass epic for pair-specific overrides
            )

            if not pullback_result['valid']:
                self.logger.info(f"   ❌ {pullback_result['reason']}")
                # Track rejection
                self._track_rejection(
                    stage='TIER3_PULLBACK',
                    reason=pullback_result['reason'],
                    epic=epic,
                    pair=pair,
                    candle_timestamp=candle_timestamp,
                    direction=direction,
                    context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, direction=direction)
                )
                return None

            market_price = pullback_result['entry_price']  # Current close (market price)
            pullback_depth = pullback_result['pullback_depth']
            in_optimal_zone = pullback_result['in_optimal_zone']
            entry_type = pullback_result.get('entry_type', 'PULLBACK')  # v1.8.0

            # v1.8.0: Log entry type
            if entry_type == 'MOMENTUM':
                self.logger.info(f"   ✅ Entry Type: MOMENTUM (continuation)")
                self.logger.info(f"   ✅ Beyond Break: {pullback_depth*100:.1f}%")
            else:
                self.logger.info(f"   ✅ Entry Type: PULLBACK (retracement)")
                self.logger.info(f"   ✅ Pullback Depth: {pullback_depth*100:.1f}%")
            self.logger.info(f"   {'✅' if in_optimal_zone else '⚠️ '} Optimal Zone: {'Yes' if in_optimal_zone else 'No'}")

            # ================================================================
            # v2.0.0: Calculate Limit Entry Price with Offset
            # ================================================================
            self.logger.info(f"\n📊 LIMIT ORDER: Calculating Entry Offset")
            entry_price, limit_offset_pips = self._calculate_limit_entry(
                market_price, direction, entry_type, pip_value, entry_df
            )

            # Determine order type based on config
            order_type = 'limit' if self.limit_order_enabled and limit_offset_pips > 0 else 'market'
            self.logger.info(f"   📋 Order Type: {order_type.upper()}")

            # ================================================================
            # STEP 4: Calculate Stop Loss and Take Profit
            # ================================================================
            self.logger.info(f"\n🛑 STEP 4: Calculating SL/TP")

            # v1.9.0: Get pair-specific SL buffer or use default
            pair_sl_buffer = self.pair_sl_buffers.get(pair, self.pair_sl_buffers.get(epic, self.sl_buffer_pips))
            buffer_sl_distance = pair_sl_buffer * pip_value

            # v1.9.0: Calculate ATR-based SL distance if enabled
            atr_sl_distance = 0
            if self.use_atr_stop:
                atr = self._calculate_atr(df_trigger)
                if atr > 0:
                    atr_sl_distance = atr * self.sl_atr_multiplier
                    self.logger.info(f"   ATR: {atr:.5f}, ATR-based SL: {atr_sl_distance/pip_value:.1f} pips")

            # v1.9.0: Use MAXIMUM of buffer OR ATR-based (whichever gives more room)
            sl_distance = max(buffer_sl_distance, atr_sl_distance)
            self.logger.info(f"   SL Distance: {sl_distance/pip_value:.1f} pips (buffer: {pair_sl_buffer}, ATR: {atr_sl_distance/pip_value:.1f})")

            # v1.7.0 FIX: Stop loss beyond OPPOSITE swing (not the broken swing)
            # For BULL: SL below the swing LOW (opposite_swing) - the level we DON'T want price to break
            # For BEAR: SL above the swing HIGH (opposite_swing) - the level we DON'T want price to break
            # Previous bug: Used swing_level (the breakout level) which gave unrealistic 0.6 pip stops
            #
            # v2.4.0 FIX: ATR-BASED DYNAMIC SL CAP
            # Problem: Fixed 55 pip cap (v2.3.0) was still 7x ATR on low-volatility pairs
            # Analysis: Trade 1594 (GBPUSD) had 48 pip SL with only 7.4 pip ATR = massive risk
            # Solution: Dynamic cap = min(ATR * multiplier, absolute_max)
            # Example: GBPUSD with 7.4 pip ATR → cap = min(7.4 * 3, 30) = 22 pips
            #
            # v2.3.0 (superseded): Fixed cap at max_risk_after_offset_pips (55 pips)
            # v2.4.0: Now uses ATR-proportional cap for better risk management

            # Calculate dynamic max risk based on ATR
            atr_pips = atr / pip_value if atr > 0 else 0
            if atr_pips > 0:
                atr_max_risk_pips = atr_pips * self.max_sl_atr_multiplier
                dynamic_max_risk_pips = min(atr_max_risk_pips, self.max_sl_absolute_pips)
                self.logger.info(f"   Dynamic SL cap: {dynamic_max_risk_pips:.1f} pips (ATR×{self.max_sl_atr_multiplier}={atr_max_risk_pips:.1f}, abs_max={self.max_sl_absolute_pips})")
            else:
                # Fallback to legacy fixed cap if ATR unavailable
                dynamic_max_risk_pips = self.max_risk_after_offset_pips
                self.logger.info(f"   Dynamic SL cap: {dynamic_max_risk_pips:.1f} pips (fallback, no ATR)")

            if direction == 'BULL':
                structural_stop = opposite_swing - sl_distance
                max_risk_stop = entry_price - (dynamic_max_risk_pips * pip_value)
                stop_loss = max(structural_stop, max_risk_stop)  # Higher value = tighter stop
                if stop_loss != structural_stop:
                    self.logger.info(f"   ⚠️ Structural SL capped: {(entry_price - structural_stop)/pip_value:.1f} → {dynamic_max_risk_pips:.1f} pips")
            else:
                structural_stop = opposite_swing + sl_distance
                max_risk_stop = entry_price + (dynamic_max_risk_pips * pip_value)
                stop_loss = min(structural_stop, max_risk_stop)  # Lower value = tighter stop
                if stop_loss != structural_stop:
                    self.logger.info(f"   ⚠️ Structural SL capped: {(structural_stop - entry_price)/pip_value:.1f} → {dynamic_max_risk_pips:.1f} pips")

            risk_pips = abs(entry_price - stop_loss) / pip_value

            # v2.0.0: Risk sanity check for limit orders with offset
            if order_type == 'limit':
                if risk_pips < self.min_risk_after_offset_pips:
                    reason = f"Risk too small after offset ({risk_pips:.1f} < {self.min_risk_after_offset_pips} pips)"
                    self.logger.info(f"   ❌ {reason}")
                    # Track rejection
                    risk_context = {
                        'potential_entry': entry_price,
                        'potential_stop_loss': stop_loss,
                        'potential_risk_pips': risk_pips,
                    }
                    self._track_rejection(
                        stage='RISK_LIMIT',
                        reason=reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        direction=direction,
                        context={**self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, direction=direction), **risk_context}
                    )
                    return None
                if risk_pips > self.max_risk_after_offset_pips:
                    reason = f"Risk too large after offset ({risk_pips:.1f} > {self.max_risk_after_offset_pips} pips)"
                    self.logger.info(f"   ❌ {reason}")
                    # Track rejection
                    risk_context = {
                        'potential_entry': entry_price,
                        'potential_stop_loss': stop_loss,
                        'potential_risk_pips': risk_pips,
                    }
                    self._track_rejection(
                        stage='RISK_LIMIT',
                        reason=reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        direction=direction,
                        context={**self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, direction=direction), **risk_context}
                    )
                    return None

            # Find take profit (next swing structure)
            tp_result = self._calculate_take_profit(
                df_trigger, direction, entry_price, risk_pips, pip_value
            )

            take_profit = tp_result['take_profit']
            reward_pips = tp_result['reward_pips']
            rr_ratio = tp_result['rr_ratio']

            self.logger.info(f"   Stop Loss: {stop_loss:.5f} ({risk_pips:.1f} pips)")
            self.logger.info(f"   Take Profit: {take_profit:.5f} ({reward_pips:.1f} pips)")
            self.logger.info(f"   R:R Ratio: {rr_ratio:.2f}")

            # Validate R:R
            if rr_ratio < self.min_rr_ratio:
                reason = f"R:R too low ({rr_ratio:.2f} < {self.min_rr_ratio})"
                self.logger.info(f"   ❌ {reason}")
                # Track rejection
                risk_result = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_pips': risk_pips,
                    'reward_pips': reward_pips,
                    'rr_ratio': rr_ratio,
                }
                self._track_rejection(
                    stage='RISK_RR',
                    reason=reason,
                    epic=epic,
                    pair=pair,
                    candle_timestamp=candle_timestamp,
                    direction=direction,
                    context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, risk_result=risk_result, direction=direction)
                )
                return None

            self.logger.info(f"   ✅ R:R meets minimum ({rr_ratio:.2f} >= {self.min_rr_ratio})")

            # Validate minimum TP
            if reward_pips < self.min_tp_pips:
                reason = f"TP too small ({reward_pips:.1f} < {self.min_tp_pips} pips)"
                self.logger.info(f"   ❌ {reason}")
                # Track rejection
                risk_result = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_pips': risk_pips,
                    'reward_pips': reward_pips,
                    'rr_ratio': rr_ratio,
                }
                self._track_rejection(
                    stage='RISK_TP',
                    reason=reason,
                    epic=epic,
                    pair=pair,
                    candle_timestamp=candle_timestamp,
                    direction=direction,
                    context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, risk_result=risk_result, direction=direction)
                )
                return None

            self.logger.info(f"   ✅ TP meets minimum ({reward_pips:.1f} >= {self.min_tp_pips} pips)")

            # ================================================================
            # STEP 5: Calculate Confidence Score
            # ================================================================
            # v2.2.0: Calculate ATR and volume ratio BEFORE confidence for new scoring
            atr = self._calculate_atr(df_trigger)

            # Volume ratio (current volume vs SMA) - needed for gradient volume scoring
            volume_ratio = 1.0
            if 'ltv' in df_trigger.columns:
                vol_col = 'ltv'
            elif 'volume' in df_trigger.columns:
                vol_col = 'volume'
            else:
                vol_col = None

            if vol_col:
                current_vol = df_trigger[vol_col].iloc[-1]
                vol_sma = df_trigger[vol_col].iloc[-self.volume_sma_period:].mean()
                if vol_sma > 0:
                    volume_ratio = current_vol / vol_sma

            # v2.2.0: Enhanced confidence with new parameters
            confidence = self._calculate_confidence(
                ema_distance=ema_distance,
                volume_confirmed=volume_confirmed,
                in_optimal_zone=in_optimal_zone,
                rr_ratio=rr_ratio,
                pullback_depth=pullback_depth,
                # v2.2.0: New parameters for improved scoring
                break_candle=break_candle,
                swing_level=swing_level,
                direction=direction,
                volume_ratio=volume_ratio,
                atr=atr
            )

            # v1.8.0: Apply confidence penalty for momentum entries
            if entry_type == 'MOMENTUM':
                confidence -= self.momentum_confidence_penalty
                self.logger.info(f"   ⚠️  Momentum entry penalty: -{self.momentum_confidence_penalty*100:.0f}%")

            # v2.11.0: Dynamic confidence threshold based on EMA distance, ATR, and Volume
            # Priority: EMA distance > ATR > Volume > Pair-specific > Default
            # EMA distance is the strongest predictor of CONFIDENCE rejection outcomes
            # v2.12.0: Added direction-aware confidence threshold support
            if hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
                pair_min_confidence = self._db_config.get_min_confidence_directional(epic, direction)
            else:
                pair_min_confidence = self.pair_min_confidence.get(pair, self.pair_min_confidence.get(epic, self.min_confidence))
            adjustment_type = "pair-specific"

            # v2.11.0: EMA distance-adjusted confidence threshold (HIGHEST PRIORITY)
            # Near EMA (trend-aligned) = lower threshold, Far from EMA (extended) = higher threshold
            if self.ema_distance_adjusted_confidence_enabled and ema_distance is not None:
                ema_distance_pips = ema_distance / pip_value
                if ema_distance_pips < self.near_ema_threshold_pips:
                    # Near EMA - trend aligned, high win rate
                    near_ema_confidence = self.pair_near_ema_confidence.get(
                        pair, self.pair_near_ema_confidence.get(epic)
                    )
                    if near_ema_confidence is not None:
                        pair_min_confidence = near_ema_confidence
                        adjustment_type = "near-EMA"
                        self.logger.info(f"   📍 Near-EMA threshold: {near_ema_confidence*100:.0f}% (EMA dist={ema_distance_pips:.1f} < {self.near_ema_threshold_pips})")
                elif ema_distance_pips >= self.far_ema_threshold_pips:
                    # Far from EMA - overextended, low win rate
                    far_ema_confidence = self.pair_far_ema_confidence.get(
                        pair, self.pair_far_ema_confidence.get(epic)
                    )
                    if far_ema_confidence is not None:
                        pair_min_confidence = far_ema_confidence
                        adjustment_type = "far-EMA"
                        self.logger.info(f"   🚀 Far-EMA threshold: {far_ema_confidence*100:.0f}% (EMA dist={ema_distance_pips:.1f} >= {self.far_ema_threshold_pips})")

            # v2.11.0: ATR-adjusted confidence threshold (only if EMA didn't adjust)
            # In calm markets (low ATR), lower the threshold - price is more predictable
            # In volatile markets (high ATR), raise the threshold - need higher confidence
            if adjustment_type == "pair-specific" and self.atr_adjusted_confidence_enabled and atr is not None:
                if atr < self.low_atr_threshold:
                    # Calm market - check for lower threshold
                    low_atr_confidence = self.pair_low_atr_confidence.get(
                        pair, self.pair_low_atr_confidence.get(epic)
                    )
                    if low_atr_confidence is not None:
                        pair_min_confidence = low_atr_confidence
                        adjustment_type = "low-ATR"
                        self.logger.info(f"   📉 Low-ATR threshold: {low_atr_confidence*100:.0f}% (ATR={atr:.5f} < {self.low_atr_threshold})")
                elif atr >= self.high_atr_threshold:
                    # Volatile market - check for higher threshold
                    high_atr_confidence = self.pair_high_atr_confidence.get(
                        pair, self.pair_high_atr_confidence.get(epic)
                    )
                    if high_atr_confidence is not None:
                        pair_min_confidence = high_atr_confidence
                        adjustment_type = "high-ATR"
                        self.logger.info(f"   📈 High-ATR threshold: {high_atr_confidence*100:.0f}% (ATR={atr:.5f} >= {self.high_atr_threshold})")

            # v2.11.0: Volume-adjusted confidence threshold (only if EMA and ATR didn't adjust)
            # When volume is high, use a lower confidence threshold for pairs that perform well
            if (adjustment_type == "pair-specific" and
                self.volume_adjusted_confidence_enabled and
                volume_ratio is not None and
                volume_ratio >= self.high_volume_threshold):
                # Check if this pair has a volume-adjusted threshold
                high_vol_confidence = self.pair_high_volume_confidence.get(
                    pair, self.pair_high_volume_confidence.get(epic)
                )
                if high_vol_confidence is not None:
                    pair_min_confidence = high_vol_confidence
                    adjustment_type = "high-volume"
                    self.logger.info(f"   📊 Volume-adjusted threshold: {high_vol_confidence*100:.0f}% (vol={volume_ratio:.2f} >= {self.high_volume_threshold})")

            # v2.9.0: Use round(2) for both values to avoid floating-point precision issues
            # (e.g., 0.4799 displaying as 48% but failing 48% threshold)
            if round(confidence, 2) < round(pair_min_confidence, 2):
                reason = f"Confidence too low ({confidence*100:.0f}% < {pair_min_confidence*100:.0f}%)"
                self.logger.info(f"\n❌ {reason} ({adjustment_type})")
                # Track rejection with confidence breakdown - v2.2.0: Updated breakdown
                fib_accuracy = 1.0 - min(abs(pullback_depth - 0.382) / 0.382, 1.0)
                confidence_breakdown = {
                    'total': confidence,
                    'ema_alignment': min(ema_distance / (atr * 3) if atr > 0 else ema_distance / 30, 1.0) * 0.20,
                    'swing_break_quality': 0.10,  # Default, actual is calculated in method
                    'volume_strength': (0.5 + min((volume_ratio - 1.0) / 1.0, 1.0) * 0.5) * 0.20 if volume_confirmed and volume_ratio > 1.0 else 0.04,
                    'pullback_quality': (1.0 if in_optimal_zone and fib_accuracy > 0.7 else 0.8 if in_optimal_zone else 0.5) * 0.20,
                    'rr_quality': min(rr_ratio / 3.0, 1.0) * 0.20,
                    'momentum_penalty': self.momentum_confidence_penalty if entry_type == 'MOMENTUM' else 0.0,
                }
                risk_result = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_pips': risk_pips,
                    'reward_pips': reward_pips,
                    'rr_ratio': rr_ratio,
                }
                context = self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, risk_result=risk_result, direction=direction)
                context['confidence_score'] = confidence
                context['confidence_breakdown'] = confidence_breakdown
                self._track_rejection(
                    stage='CONFIDENCE',
                    reason=reason,
                    epic=epic,
                    pair=pair,
                    candle_timestamp=candle_timestamp,
                    direction=direction,
                    context=context
                )
                return None

            # ================================================================
            # v2.9.0: CONFIDENCE CAP CHECK (Paradox: higher confidence = worse)
            # ================================================================
            # Analysis of 85 trades (Dec 2025) showed confidence > 0.75 had only 42% WR
            # v2.11.1: Use database service method if available (supports per-pair override)
            if self._using_database_config and self._db_config:
                pair_max_confidence = self._db_config.get_pair_max_confidence(epic)
            else:
                pair_max_confidence = self._get_pair_param(epic, 'MAX_CONFIDENCE_THRESHOLD', self.max_confidence)
            if round(confidence, 4) > pair_max_confidence:
                reason = f"Confidence too high ({confidence*100:.0f}% > {pair_max_confidence*100:.0f}% cap)"
                self.logger.info(f"\n❌ {reason} (paradox: high confidence = worse outcomes)")
                # Build context for rejection tracking
                fib_accuracy = 1.0 - min(abs(pullback_depth - 0.382) / 0.382, 1.0)
                confidence_breakdown = {
                    'total': confidence,
                    'ema_alignment': min(ema_distance / (atr * 3) if atr > 0 else ema_distance / 30, 1.0) * 0.20,
                    'swing_break_quality': 0.10,
                    'volume_strength': (0.5 + min((volume_ratio - 1.0) / 1.0, 1.0) * 0.5) * 0.20 if volume_confirmed and volume_ratio > 1.0 else 0.04,
                    'pullback_quality': (1.0 if in_optimal_zone and fib_accuracy > 0.7 else 0.8 if in_optimal_zone else 0.5) * 0.20,
                    'rr_quality': min(rr_ratio / 3.0, 1.0) * 0.20,
                }
                risk_result = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_pips': risk_pips,
                    'reward_pips': reward_pips,
                    'rr_ratio': rr_ratio,
                }
                context = self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, risk_result=risk_result, direction=direction)
                context['confidence_score'] = confidence
                context['confidence_breakdown'] = confidence_breakdown
                self._track_rejection(
                    stage='CONFIDENCE_CAP',
                    reason=reason,
                    epic=epic,
                    pair=pair,
                    candle_timestamp=candle_timestamp,
                    direction=direction,
                    context=context
                )
                return None

            # ================================================================
            # v2.9.0: VOLUME RATIO FILTER (Volume >= 0.50 has 70%+ WR)
            # v2.12.0: Direction-aware volume ratio threshold
            # ================================================================
            if self.volume_filter_enabled:
                # v2.12.0: Get direction-aware volume ratio threshold
                if hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
                    pair_min_volume = self._db_config.get_min_volume_ratio_directional(epic, direction)
                else:
                    pair_min_volume = self._get_pair_param(epic, 'MIN_VOLUME_RATIO', self.min_volume_ratio)
                if volume_ratio is not None and volume_ratio > 0:
                    if volume_ratio < pair_min_volume:
                        reason = f"Volume ratio too low ({volume_ratio:.2f} < {pair_min_volume:.2f} min)"
                        self.logger.info(f"\n❌ {reason}")
                        # Build context for rejection tracking
                        fib_accuracy = 1.0 - min(abs(pullback_depth - 0.382) / 0.382, 1.0)
                        confidence_breakdown = {
                            'total': confidence,
                            'volume_ratio': volume_ratio,
                            'min_volume_required': pair_min_volume,
                        }
                        risk_result = {
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'risk_pips': risk_pips,
                            'reward_pips': reward_pips,
                            'rr_ratio': rr_ratio,
                        }
                        context = self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, risk_result=risk_result, direction=direction)
                        context['confidence_score'] = confidence
                        context['volume_ratio'] = volume_ratio
                        self._track_rejection(
                            stage='VOLUME_LOW',
                            reason=reason,
                            epic=epic,
                            pair=pair,
                            candle_timestamp=candle_timestamp,
                            direction=direction,
                            context=context
                        )
                        return None
                elif not self.allow_no_volume_data:
                    reason = "No volume data available (required by config)"
                    self.logger.info(f"\n❌ {reason}")
                    context = self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result)
                    self._track_rejection(
                        stage='VOLUME_NO_DATA',
                        reason=reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        direction=direction,
                        context=context
                    )
                    return None

            self.logger.info(f"\n📊 Confidence: {confidence*100:.0f}%")

            # ================================================================
            # STEP 6: Calculate Technical Indicators for Analytics
            # ================================================================
            # Note: ATR and volume_ratio already calculated before confidence scoring (v2.2.0)

            # MACD indicators (if available in dataframe, otherwise calculate)
            macd_line = 0.0
            macd_signal = 0.0
            macd_histogram = 0.0
            if 'macd_line' in df_trigger.columns:
                macd_line = float(df_trigger['macd_line'].iloc[-1]) if pd.notna(df_trigger['macd_line'].iloc[-1]) else 0.0
                macd_signal = float(df_trigger['macd_signal'].iloc[-1]) if 'macd_signal' in df_trigger.columns and pd.notna(df_trigger['macd_signal'].iloc[-1]) else 0.0
                macd_histogram = float(df_trigger['macd_histogram'].iloc[-1]) if 'macd_histogram' in df_trigger.columns and pd.notna(df_trigger['macd_histogram'].iloc[-1]) else 0.0
            elif len(df_trigger) >= 26:
                # Calculate MACD if not in dataframe
                close = df_trigger['close']
                ema_12 = close.ewm(span=12, adjust=False).mean()
                ema_26 = close.ewm(span=26, adjust=False).mean()
                macd_line = float(ema_12.iloc[-1] - ema_26.iloc[-1])
                signal_line = (ema_12 - ema_26).ewm(span=9, adjust=False).mean()
                macd_signal = float(signal_line.iloc[-1])
                macd_histogram = macd_line - macd_signal

            # ================================================================
            # v2.10.0: MACD MOMENTUM ALIGNMENT FILTER
            # ================================================================
            # Check if MACD momentum direction aligns with trade direction
            # Analysis: BULL + bullish MACD = 71% WR, misaligned = 38% WR
            try:
                # Use database config service if available
                if self._config_service and hasattr(self._config_service, 'is_macd_filter_enabled'):
                    macd_filter_enabled = self._config_service.is_macd_filter_enabled(epic)
                else:
                    macd_filter_enabled = getattr(self, 'macd_alignment_filter_enabled', True)

                if macd_filter_enabled:
                    # Check MACD alignment
                    macd_momentum = 'bullish' if macd_line > macd_signal else 'bearish'
                    macd_aligned = (direction == 'BULL' and macd_momentum == 'bullish') or \
                                   (direction == 'BEAR' and macd_momentum == 'bearish')
                    macd_reason = f"{macd_momentum} MACD vs {direction} direction"

                    if not macd_aligned:
                        reason = f"MACD momentum misaligned: {direction} trade vs {macd_reason}"
                        self.logger.info(f"\n❌ {reason}")
                        # Build context for rejection tracking
                        risk_result = {
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'risk_pips': risk_pips,
                            'reward_pips': reward_pips,
                            'rr_ratio': rr_ratio,
                        }
                        context = self._collect_market_context(
                            df_trigger, df_4h, df_entry, pip_value,
                            ema_result=ema_result, swing_result=swing_result,
                            pullback_result=pullback_result, risk_result=risk_result,
                            direction=direction
                        )
                        context['confidence_score'] = confidence
                        context['macd_line'] = macd_line
                        context['macd_signal'] = macd_signal
                        context['macd_histogram'] = macd_histogram
                        context['macd_aligned'] = macd_aligned
                        context['macd_momentum'] = macd_momentum
                        self._track_rejection(
                            stage='MACD_MISALIGNED',
                            reason=reason,
                            epic=epic,
                            pair=pair,
                            candle_timestamp=candle_timestamp,
                            direction=direction,
                            context=context
                        )
                        return None
                    else:
                        self.logger.info(f"   ✅ MACD aligned: {macd_reason}")
            except ImportError:
                pass  # Config not available, skip MACD filter

            # ================================================================
            # v2.5.0: PAIR-SPECIFIC BLOCKING CHECK
            # ================================================================
            # Check if this signal should be blocked based on pair-specific conditions
            # This catches weak setups that historically underperform for specific pairs
            # Note: Blocking conditions are now handled via database config (blocking_conditions JSONB)
            # The blocking is checked earlier in signal validation if enabled in pair overrides

            # ================================================================
            # BUILD SIGNAL
            # ================================================================
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✅ VALID SMC SIMPLE SIGNAL DETECTED")
            self.logger.info(f"{'='*70}")

            signal = {
                'strategy': 'SMC_SIMPLE',
                'signal_type': direction,
                'signal': direction,
                'confidence_score': round(confidence, 2),
                'epic': epic,
                'pair': pair,
                'timeframe': '15m',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'partial_tp': None,  # Not using partial TP in simple version
                'partial_percent': None,
                'risk_pips': round(risk_pips, 1),
                'reward_pips': round(reward_pips, 1),
                'rr_ratio': round(rr_ratio, 2),
                'timestamp': candle_dt,  # Use candle timestamp for backtest compatibility

                # Tier-specific data
                'ema_value': ema_value,
                'ema_distance_pips': ema_distance,
                'swing_level': swing_level,
                'pullback_depth': pullback_depth,
                'volume_confirmed': volume_confirmed,
                'in_optimal_zone': in_optimal_zone,
                'entry_type': entry_type,  # v1.8.0: PULLBACK or MOMENTUM

                # v2.0.0: Limit order fields
                'order_type': order_type,  # 'limit' or 'market'
                'market_price': market_price,  # Current market price (before offset)
                'limit_offset_pips': round(limit_offset_pips, 1),  # Offset from market price
                'limit_expiry_minutes': self.limit_expiry_minutes if order_type == 'limit' else None,

                # Technical indicators for analytics (alert_history compatibility)
                'atr': round(atr, 6) if atr else 0.0,
                'volume_ratio': round(volume_ratio, 2),
                'macd_line': round(macd_line, 6),
                'macd_signal': round(macd_signal, 6),
                'macd_histogram': round(macd_histogram, 6),

                # Strategy indicators (for alert_history compatibility)
                'strategy_indicators': {
                    'tier1_ema': {
                        'timeframe': self.htf_timeframe,
                        'ema_period': self.ema_period,
                        'ema_value': ema_value,
                        'distance_pips': ema_distance,
                        'direction': direction
                    },
                    'tier2_swing': {
                        'timeframe': self.trigger_tf,
                        'swing_level': swing_level,
                        'body_close_confirmed': True,
                        'volume_confirmed': volume_confirmed
                    },
                    'tier3_entry': {
                        'timeframe': self.entry_tf,
                        'entry_price': entry_price,
                        'market_price': market_price,
                        'pullback_depth': pullback_depth,
                        'fib_zone': f"{self.fib_min*100:.1f}%-{self.fib_max*100:.1f}%",
                        'in_optimal_zone': in_optimal_zone,
                        'entry_type': entry_type,  # v1.8.0: PULLBACK or MOMENTUM
                        'order_type': order_type,  # v2.0.0: 'limit' or 'market'
                        'limit_offset_pips': limit_offset_pips if order_type == 'limit' else 0.0,
                        'limit_expiry_minutes': self.limit_expiry_minutes if order_type == 'limit' else None
                    },
                    'risk_management': {
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_pips': risk_pips,
                        'reward_pips': reward_pips,
                        'rr_ratio': rr_ratio
                    },
                    'confidence_breakdown': {
                        'total': confidence,
                        'ema_alignment': min(ema_distance / 30, 1.0) * 0.25,
                        'volume_bonus': 0.15 if volume_confirmed else 0.05,
                        'pullback_quality': (1.0 if in_optimal_zone else 0.6) * 0.25,
                        'rr_quality': min(rr_ratio / 3.0, 1.0) * 0.20,
                        'fib_accuracy': (1.0 - abs(pullback_depth - 0.382) / 0.382) * 0.15
                    },
                    'indicator_count': 4,
                    'data_source': 'smc_simple_3tier'
                },

                # Description
                'description': self._build_description(
                    direction, ema_distance, pullback_depth, rr_ratio
                ),

                # Analytics fields - same structure as rejection tracking for comparative analysis
                'market_structure_analysis': {
                    'trend_direction': direction,
                    'ema_bias': {
                        'timeframe': self.htf_timeframe,
                        'ema_period': self.ema_period,
                        'ema_value': ema_value,
                        'distance_pips': ema_distance,
                        'price_position': 'above' if direction == 'BULL' else 'below'
                    },
                    'swing_structure': {
                        'timeframe': self.trigger_tf,
                        'swing_level': swing_level,
                        'opposite_swing': opposite_swing,
                        'break_confirmed': True,
                        'break_type': 'swing_high' if direction == 'BEAR' else 'swing_low'
                    }
                },
                'order_flow_analysis': {
                    'volume_confirmed': volume_confirmed,
                    'volume_ratio': round(volume_ratio, 2),
                    'atr': round(atr, 6) if atr else 0.0,
                    'entry_type': entry_type,  # PULLBACK or MOMENTUM
                    'order_type': order_type   # limit or market
                },
                'confluence_details': {
                    'tier1_ema_aligned': True,
                    'tier2_swing_break': True,
                    'tier3_pullback_valid': True,
                    'in_optimal_fib_zone': in_optimal_zone,
                    'fib_zone': f"{self.fib_min*100:.1f}%-{self.fib_max*100:.1f}%",
                    'pullback_depth_pct': round(pullback_depth * 100, 1),
                    'volume_spike': volume_confirmed,
                    'rr_acceptable': rr_ratio >= self.min_rr_ratio,
                    'confluence_count': sum([
                        True,  # EMA aligned
                        True,  # Swing break
                        True,  # Pullback valid
                        in_optimal_zone,
                        volume_confirmed,
                        rr_ratio >= 2.0
                    ])
                },
                'signal_conditions': {
                    'session': self._get_current_session(candle_dt) if hasattr(self, '_get_current_session') else 'unknown',
                    'spread_pips': entry_df['spread'].iloc[-1] if 'spread' in entry_df.columns else None,
                    'current_price': market_price,
                    'limit_offset_pips': round(limit_offset_pips, 1) if order_type == 'limit' else 0.0,
                    'risk_pips': round(risk_pips, 1),
                    'reward_pips': round(reward_pips, 1),
                    'rr_ratio': round(rr_ratio, 2),
                    'confidence': round(confidence, 2),
                    'atr_pips': round(atr / pip_value, 1) if atr else 0.0
                }
            }

            self.logger.info(f"\n📋 Signal Summary:")
            self.logger.info(f"   Direction: {signal['signal']}")
            self.logger.info(f"   Order Type: {signal['order_type'].upper()}")
            if signal['order_type'] == 'limit':
                self.logger.info(f"   Market Price: {signal['market_price']:.5f}")
                self.logger.info(f"   Limit Entry: {signal['entry_price']:.5f} ({signal['limit_offset_pips']:.1f} pips offset)")
                self.logger.info(f"   Expiry: {signal['limit_expiry_minutes']} minutes")
            else:
                self.logger.info(f"   Entry: {signal['entry_price']:.5f}")
            self.logger.info(f"   Stop Loss: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
            self.logger.info(f"   Take Profit: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
            self.logger.info(f"   R:R Ratio: {signal['rr_ratio']:.2f}")
            self.logger.info(f"\n   {signal['description']}")
            self.logger.info(f"{'='*70}\n")

            # Update cooldown (use candle timestamp for backtest compatibility)
            self._update_cooldown(pair, candle_dt)

            # ================================================================
            # PERFORMANCE METRICS: Calculate enhanced metrics for analysis
            # ================================================================
            try:
                signal = self._add_performance_metrics(
                    signal=signal,
                    df_entry=entry_df,
                    df_trigger=df_trigger,
                    df_4h=df_4h,
                    epic=epic
                )
            except Exception as metrics_error:
                # Don't fail the signal if metrics calculation fails
                self.logger.warning(f"⚠️ Performance metrics calculation failed: {metrics_error}")

            # ================================================================
            # SMC DATA: Detect FVGs and Order Blocks for chart visualization
            # ================================================================
            try:
                signal = self._add_smc_chart_data(
                    signal=signal,
                    df_trigger=df_trigger,
                    df_entry=entry_df,
                    epic=epic,
                    pip_value=pip_value
                )
            except Exception as smc_error:
                # Don't fail the signal if SMC data detection fails
                self.logger.warning(f"⚠️ SMC chart data detection failed: {smc_error}")

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error detecting SMC Simple signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _check_ema_bias(self, df_4h: pd.DataFrame, pip_value: float) -> Dict:
        """
        TIER 1: Check 4H 50 EMA directional bias

        Returns:
            Dict with: valid, direction, ema_value, distance_pips, reason
        """
        if len(df_4h) < self.ema_period + 1:
            return {
                'valid': False,
                'reason': f"Insufficient 4H data ({len(df_4h)} < {self.ema_period + 1} bars)"
            }

        # Calculate 50 EMA
        ema = df_4h['close'].ewm(span=self.ema_period, adjust=False).mean()
        ema_value = ema.iloc[-1]
        current_close = df_4h['close'].iloc[-1]

        # Calculate distance in pips
        distance_pips = abs(current_close - ema_value) / pip_value

        # Check if price is beyond EMA buffer zone
        buffer = self.ema_buffer_pips * pip_value

        if current_close > ema_value + buffer:
            direction = 'BULL'
        elif current_close < ema_value - buffer:
            direction = 'BEAR'
        else:
            return {
                'valid': False,
                'reason': f"Price in EMA buffer zone ({distance_pips:.1f} pips, need >{self.ema_buffer_pips})"
            }

        # Check minimum distance from EMA
        if distance_pips < self.min_distance_from_ema:
            return {
                'valid': False,
                'reason': f"Too close to EMA ({distance_pips:.1f} pips < {self.min_distance_from_ema})"
            }

        # Verify candle CLOSED beyond EMA (not just wicked)
        if self.require_close_beyond_ema:
            if direction == 'BULL' and current_close <= ema_value:
                return {
                    'valid': False,
                    'reason': "Candle did not CLOSE above EMA"
                }
            elif direction == 'BEAR' and current_close >= ema_value:
                return {
                    'valid': False,
                    'reason': "Candle did not CLOSE below EMA"
                }

        return {
            'valid': True,
            'direction': direction,
            'ema_value': ema_value,
            'distance_pips': distance_pips,
            'reason': f"{direction} bias confirmed"
        }

    def _get_dynamic_swing_lookback(self, df: pd.DataFrame, pip_value: float) -> int:
        """
        v2.1.0: Calculate dynamic swing lookback based on current ATR volatility.

        In quiet markets (low ATR): Use shorter lookback to find tighter swings
        In volatile markets (high ATR): Use longer lookback to find stronger swings

        This improves R:R by finding better-spaced swing structures that match
        current market conditions.

        Returns:
            int: Number of bars to look back for swing detection
        """
        if not self.use_dynamic_swing_lookback:
            return self.swing_lookback

        # Calculate current ATR in pips
        atr = self._calculate_atr(df)
        atr_pips = atr / pip_value if atr > 0 else 10  # Default to 10 pips if ATR unavailable

        # Interpolate lookback based on ATR
        if atr_pips <= self.swing_lookback_atr_low:
            # Low volatility: use minimum lookback (tighter swings)
            lookback = self.swing_lookback_min
        elif atr_pips >= self.swing_lookback_atr_high:
            # High volatility: use maximum lookback (stronger swings)
            lookback = self.swing_lookback_max
        else:
            # Linear interpolation between min and max
            atr_range = self.swing_lookback_atr_high - self.swing_lookback_atr_low
            lookback_range = self.swing_lookback_max - self.swing_lookback_min
            atr_ratio = (atr_pips - self.swing_lookback_atr_low) / atr_range
            lookback = int(self.swing_lookback_min + (atr_ratio * lookback_range))

        if self.debug_logging:
            self.logger.debug(f"   v2.1.0 Dynamic lookback: ATR={atr_pips:.1f} pips → {lookback} bars")

        return lookback

    def _check_swing_break(self, df: pd.DataFrame, direction: str, pip_value: float, epic: str = None) -> Dict:
        """
        TIER 2: Check swing break with body-close confirmation on trigger timeframe

        IMPROVED: Now checks for RECENT breaks within lookback, not just current candle
        This allows detecting entries after a break has occurred and price is continuing

        v2.1.0: Uses dynamic swing lookback based on ATR volatility

        Returns:
            Dict with: valid, swing_level, break_candle, volume_confirmed, reason
        """
        # v2.1.0: Get dynamic lookback based on volatility
        effective_lookback = self._get_dynamic_swing_lookback(df, pip_value)

        if len(df) < effective_lookback + 1:
            return {
                'valid': False,
                'reason': f"Insufficient {self.trigger_tf} data ({len(df)} < {effective_lookback + 1} bars)"
            }

        # Find swing points
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        # v2.1.1: Prefer 'ltv' (Last Traded Volume) over 'volume' - IG provides actual data in ltv
        if 'ltv' in df.columns:
            volumes = df['ltv'].values
        elif 'volume' in df.columns:
            volumes = df['volume'].values
        else:
            volumes = None

        swing_highs = []
        swing_lows = []

        # Find swings up to (but not including) the last swing_strength bars
        # so we have time to confirm them
        for i in range(self.swing_strength, len(df) - self.swing_strength - 1):
            # Check for swing high
            is_swing_high = True
            for j in range(1, self.swing_strength + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append((i, highs[i]))

            # Check for swing low
            is_swing_low = True
            for j in range(1, self.swing_strength + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append((i, lows[i]))

        # Check for break based on direction
        current_idx = len(df) - 1
        current_close = closes[-1]
        current_open = opens[-1]

        # v1.3.0: Expanded lookback window - check ANY swing break in full lookback
        # This allows for pullback entries even when price has retraced from break
        # v2.1.0: Use dynamic effective_lookback based on volatility
        break_check_bars = effective_lookback

        if direction == 'BULL':
            # For bullish, need ANY swing high to have been broken within lookback
            if not swing_highs:
                return {'valid': False, 'reason': "No swing highs found"}

            # Get swing highs within lookback, sorted by time (oldest first)
            # v2.1.0: Use effective_lookback instead of fixed self.swing_lookback
            recent_highs = sorted(
                [sh for sh in swing_highs if current_idx - sh[0] <= effective_lookback],
                key=lambda x: x[0]
            )
            if not recent_highs:
                return {'valid': False, 'reason': "No recent swing highs in lookback"}

            # v1.3.0: FLEXIBLE SWING BREAK DETECTION
            # Check EACH swing high in lookback - if ANY was broken, we have a valid setup
            # This allows entries on pullbacks after a break, not just immediate continuations
            break_found = False
            best_swing_level = None
            best_swing_idx = None
            best_break_idx = None

            for swing_idx, swing_level in recent_highs:
                # Check all candles AFTER this swing for a break
                for check_idx in range(swing_idx + 1, current_idx + 1):
                    if highs[check_idx] > swing_level:
                        # Found a break of this swing!
                        if not break_found or swing_level > best_swing_level:
                            # Prefer higher swing levels (stronger breaks)
                            break_found = True
                            best_swing_level = swing_level
                            best_swing_idx = swing_idx
                            best_break_idx = check_idx
                        break  # Found break for this swing, check next swing

            if not break_found:
                # Fallback: check if current price is above the highest recent swing
                highest_swing = max(recent_highs, key=lambda x: x[1])
                swing_level = highest_swing[1]
                current_high = highs[-1]
                gap_pips = (swing_level - current_high) / pip_value
                return {
                    'valid': False,
                    'reason': f"BULL: High {current_high:.5f} below swing {swing_level:.5f} (need +{gap_pips:.1f} pips)"
                }

            swing_level = best_swing_level
            swing_idx = best_swing_idx
            break_candle_idx = best_break_idx

            # Use the break candle data for reference
            break_open = opens[break_candle_idx]
            break_close = closes[break_candle_idx]
            break_high = highs[break_candle_idx]
            break_low = lows[break_candle_idx]

        else:  # BEAR
            # For bearish, need ANY swing low to have been broken within lookback
            if not swing_lows:
                return {'valid': False, 'reason': "No swing lows found"}

            # Get swing lows within lookback, sorted by time (oldest first)
            # v2.1.0: Use effective_lookback instead of fixed self.swing_lookback
            recent_lows = sorted(
                [sl for sl in swing_lows if current_idx - sl[0] <= effective_lookback],
                key=lambda x: x[0]
            )
            if not recent_lows:
                return {'valid': False, 'reason': "No recent swing lows in lookback"}

            # v1.3.0: FLEXIBLE SWING BREAK DETECTION
            # Check EACH swing low in lookback - if ANY was broken, we have a valid setup
            break_found = False
            best_swing_level = None
            best_swing_idx = None
            best_break_idx = None

            for swing_idx, swing_level in recent_lows:
                # Check all candles AFTER this swing for a break
                for check_idx in range(swing_idx + 1, current_idx + 1):
                    if lows[check_idx] < swing_level:
                        # Found a break of this swing!
                        if not break_found or swing_level < best_swing_level:
                            # Prefer lower swing levels (stronger breaks)
                            break_found = True
                            best_swing_level = swing_level
                            best_swing_idx = swing_idx
                            best_break_idx = check_idx
                        break  # Found break for this swing, check next swing

            if not break_found:
                # Fallback: check if current price is below the lowest recent swing
                lowest_swing = min(recent_lows, key=lambda x: x[1])
                swing_level = lowest_swing[1]
                current_low = lows[-1]
                gap_pips = (current_low - swing_level) / pip_value
                return {
                    'valid': False,
                    'reason': f"BEAR: Low {current_low:.5f} above swing {swing_level:.5f} (need -{gap_pips:.1f} pips)"
                }

            swing_level = best_swing_level
            swing_idx = best_swing_idx
            break_candle_idx = best_break_idx

            # Use the break candle data for reference
            break_open = opens[break_candle_idx]
            break_close = closes[break_candle_idx]
            break_high = highs[break_candle_idx]
            break_low = lows[break_candle_idx]

        # Volume confirmation (optional) - check volume on break candle
        volume_confirmed = False
        if self.volume_enabled and volumes is not None:
            vol_sma = np.mean(volumes[max(0, break_candle_idx-self.volume_sma_period):break_candle_idx])
            break_vol = volumes[break_candle_idx]
            if vol_sma > 0:
                volume_confirmed = break_vol > vol_sma * self.volume_multiplier

        # v1.8.0: Momentum quality filter - ensure strong breakout candle
        # v2.6.0: Uses pair-specific overrides for EURUSD
        momentum_quality_passed = True
        if self.momentum_quality_enabled:
            # Calculate ATR for context
            atr = self._calculate_atr(df)
            if atr > 0:
                # Check 1: Breakout candle range should be significant (> 50% of ATR)
                # v2.6.0: Use pair-specific MIN_BREAKOUT_ATR_RATIO if configured
                pair_min_atr_ratio = self._get_pair_param(epic, 'MIN_BREAKOUT_ATR_RATIO', self.min_breakout_atr_ratio)
                break_range = break_high - break_low
                if break_range < atr * pair_min_atr_ratio:
                    return {
                        'valid': False,
                        'reason': f"Weak breakout candle (range {break_range/atr*100:.0f}% of ATR < {pair_min_atr_ratio*100:.0f}%)"
                    }

                # Check 2: Body should be majority of candle (> 60% = strong momentum)
                # v2.6.0: Use pair-specific MIN_BODY_PERCENTAGE if configured
                pair_min_body = self._get_pair_param(epic, 'MIN_BODY_PERCENTAGE', self.min_body_percentage)
                break_body = abs(break_close - break_open)
                body_percentage = break_body / break_range if break_range > 0 else 0
                if body_percentage < pair_min_body:
                    return {
                        'valid': False,
                        'reason': f"Weak breakout body ({body_percentage*100:.0f}% < {pair_min_body*100:.0f}%)"
                    }

        # v1.6.0: Find the opposite swing for Fib calculation
        # For BULL: Need the swing LOW before the broken swing HIGH
        # For BEAR: Need the swing HIGH before the broken swing LOW
        # v2.1.0: Use effective_lookback for consistency
        opposite_swing = None
        if direction == 'BULL':
            # Find swing lows BEFORE the broken swing high
            prior_lows = [sl for sl in swing_lows if sl[0] < best_swing_idx]
            if prior_lows:
                # Get the most recent swing low before the broken high
                opposite_swing = max(prior_lows, key=lambda x: x[0])[1]
            else:
                # Fallback: use the lowest low in the range before the swing
                opposite_swing = min(lows[max(0, best_swing_idx - effective_lookback):best_swing_idx])
        else:  # BEAR
            # Find swing highs BEFORE the broken swing low
            prior_highs = [sh for sh in swing_highs if sh[0] < best_swing_idx]
            if prior_highs:
                # Get the most recent swing high before the broken low
                opposite_swing = max(prior_highs, key=lambda x: x[0])[1]
            else:
                # Fallback: use the highest high in the range before the swing
                opposite_swing = max(highs[max(0, best_swing_idx - effective_lookback):best_swing_idx])

        return {
            'valid': True,
            'swing_level': swing_level,
            'opposite_swing': opposite_swing,  # v1.6.0: For accurate Fib calculation
            'break_candle': {
                'open': break_open,
                'close': break_close,
                'high': break_high,
                'low': break_low,
                'bars_ago': current_idx - break_candle_idx
            },
            'volume_confirmed': volume_confirmed,
            'reason': f"Swing break confirmed ({current_idx - break_candle_idx} bars ago)"
        }

    def _check_pullback_zone(
        self,
        df: pd.DataFrame,
        direction: str,
        swing_level: float,
        opposite_swing: float,
        break_candle: Dict,
        pip_value: float,
        epic: str = None
    ) -> Dict:
        """
        TIER 3: Check if price is in valid entry zone (pullback OR momentum continuation)

        v1.8.0 ENHANCEMENTS:
        - NEW: Momentum continuation mode (price beyond break = valid entry)
        - NEW: ATR-based swing validation (adapts to pair volatility)
        - Entry types: PULLBACK (23.6%-70% Fib) or MOMENTUM (-20% to 0%)

        The Fib retracement is measured using swing data from Tier 2 (15m):
        - For BULL: swing_level = broken swing HIGH, opposite_swing = swing LOW before it
        - For BEAR: swing_level = broken swing LOW, opposite_swing = swing HIGH before it

        Pullback depth interpretation:
        - Negative = price beyond break (momentum continuation)
        - 0% = price at swing_level (the break point)
        - 100% = price at opposite_swing (full retracement)
        - 38.2%-61.8% = golden zone (optimal pullback entry)

        Returns:
            Dict with: valid, entry_price, pullback_depth, in_optimal_zone, entry_type, reason
        """
        current_close = df['close'].iloc[-1]
        entry_price = current_close

        # v1.6.0: Use swing data from Tier 2 for consistent Fib calculation
        # swing_level = the swing that was broken (HIGH for BULL, LOW for BEAR)
        # opposite_swing = the swing on the other side (LOW for BULL, HIGH for BEAR)

        if direction == 'BULL':
            # For BULL: Fib from swing_low (0%) to swing_high (100%)
            # We want price to retrace FROM swing_high TOWARD swing_low
            fib_high = swing_level       # The broken swing high (100% on Fib)
            fib_low = opposite_swing     # The swing low before it (0% on Fib)

            # Calculate range in price terms
            swing_range = fib_high - fib_low

            # Calculate pullback depth
            # Negative = beyond break (momentum), 0% = at break, 100% = at swing low
            pullback_depth = (fib_high - current_close) / swing_range if swing_range > 0 else 0

        else:  # BEAR
            # For BEAR: Fib from swing_high (0%) to swing_low (100%)
            # We want price to retrace FROM swing_low TOWARD swing_high
            fib_low = swing_level        # The broken swing low (100% on Fib)
            fib_high = opposite_swing    # The swing high before it (0% on Fib)

            # Calculate range in price terms
            swing_range = fib_high - fib_low

            # Calculate pullback depth
            # Negative = beyond break (momentum), 0% = at break, 100% = at swing high
            pullback_depth = (current_close - fib_low) / swing_range if swing_range > 0 else 0

        range_pips = swing_range / pip_value if swing_range > 0 else 0

        # v1.8.0: ATR-based swing validation (replaces fixed 10-pip minimum)
        if self.use_atr_swing_validation:
            atr = self._calculate_atr(df)
            if atr > 0:
                min_swing_range = atr * self.min_swing_atr_multiplier
                min_swing_pips = min_swing_range / pip_value
            else:
                # Fallback if ATR unavailable
                min_swing_pips = self.fallback_min_swing_pips
                min_swing_range = min_swing_pips * pip_value

            if swing_range < min_swing_range:
                return {
                    'valid': False,
                    'reason': f"Swing range too small ({range_pips:.1f} pips < {min_swing_pips:.1f} ATR-based min)"
                }
        else:
            # Legacy fixed 10-pip minimum
            if range_pips < 10:
                return {
                    'valid': False,
                    'reason': f"Swing range too small ({range_pips:.1f} pips < 10 pips)"
                }

        # Log debug info for troubleshooting
        if self.debug_logging:
            self.logger.debug(f"   Fib calc: high={fib_high:.5f}, low={fib_low:.5f}, range={range_pips:.1f} pips")
            self.logger.debug(f"   Current: {current_close:.5f}, depth={pullback_depth*100:.1f}%")

        # v1.8.0: Determine entry type and validate zone
        # v2.6.0: Uses pair-specific overrides for EURUSD
        # v2.12.0: Direction-aware overrides (BULL/BEAR specific thresholds)
        entry_type = 'PULLBACK'  # Default

        # v2.12.0: Get direction-aware momentum min depth
        # Priority: direction-specific -> pair parameter_overrides -> global
        if hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
            pair_momentum_min = self._db_config.get_momentum_min_depth(epic, direction)
        else:
            pair_momentum_min = self._get_pair_param(epic, 'MOMENTUM_MIN_DEPTH', self.momentum_min_depth)

        # Check for momentum continuation entry (price beyond break point)
        if pullback_depth < 0:
            if self.momentum_mode_enabled:
                # v1.8.0: Allow momentum entries within configured range
                # v2.12.0: Use direction-aware momentum_min_depth
                if pair_momentum_min <= pullback_depth <= self.momentum_max_depth:
                    entry_type = 'MOMENTUM'
                    # Momentum entries are not in "optimal" Fib zone
                    return {
                        'valid': True,
                        'entry_price': entry_price,
                        'pullback_depth': pullback_depth,
                        'in_optimal_zone': False,
                        'entry_type': entry_type,
                        'swing_range_pips': range_pips,
                        'reason': f"Momentum continuation ({pullback_depth*100:.1f}% beyond break)"
                    }
                elif pullback_depth < pair_momentum_min:
                    return {
                        'valid': False,
                        'reason': f"Price too far beyond break ({pullback_depth*100:.1f}% < {pair_momentum_min*100:.0f}%)"
                    }
            else:
                # Momentum mode disabled - reject prices beyond break
                return {
                    'valid': False,
                    'reason': f"Price beyond break point ({pullback_depth*100:.1f}% - momentum mode disabled)"
                }

        # v1.6.0: Sanity check - pullback depth should not exceed 150% (trend reversal)
        if pullback_depth > 1.5:
            return {
                'valid': False,
                'reason': f"Pullback exceeded swing ({pullback_depth*100:.1f}% - trend reversal)"
            }

        # v2.12.0: Get direction-aware Fib thresholds
        # Priority: direction-specific -> pair parameter_overrides -> global
        if hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
            pair_fib_min = self._db_config.get_fib_pullback_min(epic, direction)
            pair_fib_max = self._db_config.get_fib_pullback_max(epic, direction)
        else:
            pair_fib_min = self._get_pair_param(epic, 'FIB_PULLBACK_MIN', self.fib_min)
            pair_fib_max = self._get_pair_param(epic, 'FIB_PULLBACK_MAX', self.fib_max)

        # Check if in Fib pullback zone
        if pullback_depth < pair_fib_min:
            return {
                'valid': False,
                'reason': f"Insufficient pullback ({pullback_depth*100:.1f}% < {pair_fib_min*100:.1f}%)"
            }
        if pullback_depth > pair_fib_max:
            return {
                'valid': False,
                'reason': f"Pullback too deep ({pullback_depth*100:.1f}% > {pair_fib_max*100:.1f}%)"
            }

        # Check if in optimal zone (golden zone)
        in_optimal = self.fib_optimal[0] <= pullback_depth <= self.fib_optimal[1]

        return {
            'valid': True,
            'entry_price': entry_price,
            'pullback_depth': pullback_depth,
            'in_optimal_zone': in_optimal,
            'entry_type': entry_type,
            'swing_range_pips': range_pips,
            'reason': f"In Fib zone ({pullback_depth*100:.1f}%)"
        }

    def _calculate_take_profit(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        risk_pips: float,
        pip_value: float
    ) -> Dict:
        """Calculate take profit based on next swing structure

        IMPROVED v1.0.8:
        - Ensures minimum TP distance based on risk
        - Uses R:R-based fallback when swing targets are too close
        - Properly filters out swing levels that are too close to entry
        """

        # Find next swing level in trade direction
        highs = df['high'].values
        lows = df['low'].values

        # Calculate minimum TP distance (at least min_rr_ratio * risk)
        min_tp_distance = risk_pips * self.min_rr_ratio * pip_value

        take_profit = None

        if self.use_swing_target:
            if direction == 'BULL':
                # Find next resistance (recent swing high above entry)
                # Must be at least min_tp_distance away from entry
                swing_targets = []
                for i in range(self.swing_strength, len(df) - self.swing_strength):
                    is_swing_high = True
                    for j in range(1, self.swing_strength + 1):
                        if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                            is_swing_high = False
                            break
                    if is_swing_high:
                        distance = highs[i] - entry_price
                        # Only include if above entry AND far enough away
                        if distance >= min_tp_distance:
                            swing_targets.append(highs[i])

                if swing_targets:
                    take_profit = min(swing_targets)  # Nearest valid target

            else:  # BEAR
                # Find next support (recent swing low below entry)
                # Must be at least min_tp_distance away from entry
                swing_targets = []
                for i in range(self.swing_strength, len(df) - self.swing_strength):
                    is_swing_low = True
                    for j in range(1, self.swing_strength + 1):
                        if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                            is_swing_low = False
                            break
                    if is_swing_low:
                        distance = entry_price - lows[i]
                        # Only include if below entry AND far enough away
                        if distance >= min_tp_distance:
                            swing_targets.append(lows[i])

                if swing_targets:
                    take_profit = max(swing_targets)  # Nearest valid target

        # Fallback to R:R-based TP if no valid swing target found
        if take_profit is None:
            if direction == 'BULL':
                take_profit = entry_price + (risk_pips * self.optimal_rr * pip_value)
            else:
                take_profit = entry_price - (risk_pips * self.optimal_rr * pip_value)

        reward_pips = abs(take_profit - entry_price) / pip_value
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0

        # Cap R:R if too high (likely unrealistic)
        if rr_ratio > self.max_rr_ratio:
            reward_pips = risk_pips * self.max_rr_ratio
            if direction == 'BULL':
                take_profit = entry_price + (reward_pips * pip_value)
            else:
                take_profit = entry_price - (reward_pips * pip_value)
            rr_ratio = self.max_rr_ratio

        return {
            'take_profit': take_profit,
            'reward_pips': reward_pips,
            'rr_ratio': rr_ratio
        }

    def _calculate_confidence(
        self,
        ema_distance: float,
        volume_confirmed: bool,
        in_optimal_zone: bool,
        rr_ratio: float,
        pullback_depth: float,
        # v2.2.0: New parameters for improved scoring
        break_candle: Dict = None,
        swing_level: float = None,
        direction: str = None,
        volume_ratio: float = 1.0,
        atr: float = None
    ) -> float:
        """
        Calculate confidence score (0.0 to 1.0)

        v2.2.0: Redesigned scoring with proper tier alignment

        Scoring (5 components, each 20%):
        - EMA alignment: 20% (ATR-normalized distance from EMA)
        - Swing break quality: 20% (body close %, break strength, recency)
        - Volume strength: 20% (gradient based on spike magnitude)
        - Pullback quality: 20% (combined zone + Fib accuracy)
        - R:R ratio quality: 20% (scales toward 3:1)

        Previous issues fixed:
        - swing_break_quality was in config but NOT implemented (now added)
        - pullback was over-weighted at 40% (now 20%)
        - volume was binary (now gradient)
        - EMA used fixed 30 pips (now ATR-normalized)
        """
        # ================================================================
        # 1. EMA ALIGNMENT (20% weight) - ATR-normalized
        # ================================================================
        # v2.2.0: Normalize by ATR for fair cross-pair comparison
        # 3+ ATR from EMA = perfect score
        if atr and atr > 0:
            ema_in_atr = ema_distance / (atr * 3)  # 3 ATR = max
            ema_score = min(ema_in_atr, 1.0) * 0.20
        else:
            # Fallback to pip-based (legacy)
            ema_score = min(ema_distance / 30, 1.0) * 0.20

        # ================================================================
        # 2. SWING BREAK QUALITY (20% weight) - NEW in v2.2.0
        # ================================================================
        swing_break_score = 0.10  # Default to 50% if no break_candle data

        if break_candle and swing_level and direction:
            # Factor A: Body close percentage (strong momentum = full body candle)
            break_range = break_candle.get('high', 0) - break_candle.get('low', 0)
            break_body = abs(break_candle.get('close', 0) - break_candle.get('open', 0))
            body_pct = break_body / break_range if break_range > 0 else 0.5
            body_score = min(body_pct / 0.7, 1.0)  # 70%+ body = perfect

            # Factor B: Break strength (how far beyond swing level)
            if direction == 'BULL':
                break_beyond = break_candle.get('close', 0) - swing_level
            else:
                break_beyond = swing_level - break_candle.get('close', 0)

            # Normalize break strength by ATR if available
            if atr and atr > 0:
                break_strength = min(break_beyond / atr, 1.0) if break_beyond > 0 else 0.3
            else:
                break_strength = 0.5 if break_beyond > 0 else 0.3

            # Factor C: Recency (more recent breaks are better)
            bars_ago = break_candle.get('bars_ago', 10)
            recency_score = max(0.3, 1.0 - (bars_ago / 20))  # 0 bars = 1.0, 20+ bars = 0.3

            # Combined swing break quality (weighted average)
            swing_break_score = (body_score * 0.4 + break_strength * 0.4 + recency_score * 0.2) * 0.20

        # ================================================================
        # 3. VOLUME STRENGTH (20% weight) - Gradient scoring
        # ================================================================
        # v2.2.0: Scale based on volume spike magnitude, not binary
        if volume_confirmed and volume_ratio > 1.0:
            # Scale from 1.0x to 2.0x spike (beyond 2x = max score)
            volume_magnitude = min((volume_ratio - 1.0) / 1.0, 1.0)  # 2.0x = 100%
            volume_score = (0.5 + volume_magnitude * 0.5) * 0.20  # 50-100% of weight
        elif volume_confirmed:
            volume_score = 0.10  # 50% of weight (confirmed but no ratio data)
        else:
            volume_score = 0.04  # 20% of weight (no confirmation)

        # ================================================================
        # 4. PULLBACK QUALITY (20% weight) - Combined zone + Fib accuracy
        # ================================================================
        # v2.2.0: Single combined score instead of separate 25% + 15%
        # Perfect: In optimal zone (38.2-61.8%) AND close to 38.2%
        # Good: In optimal zone OR close to 38.2%
        # Acceptable: In valid zone (23.6-70%)
        # Poor: Edge of zone

        fib_accuracy = 1.0 - min(abs(pullback_depth - 0.382) / 0.382, 1.0)

        if in_optimal_zone and fib_accuracy > 0.7:
            # Perfect: optimal zone + close to golden ratio
            pullback_score = 1.0 * 0.20
        elif in_optimal_zone:
            # Good: in optimal zone
            pullback_score = 0.8 * 0.20
        elif self.fib_min <= pullback_depth <= self.fib_max:
            # Acceptable: in valid zone, scale by Fib accuracy
            pullback_score = (0.4 + fib_accuracy * 0.3) * 0.20
        else:
            # Poor: outside zone (shouldn't happen - would fail tier 3)
            pullback_score = 0.2 * 0.20

        # ================================================================
        # 5. R:R RATIO QUALITY (20% weight)
        # ================================================================
        # Scale toward 3:1 (beyond 3:1 caps at max score)
        rr_score = min(rr_ratio / 3.0, 1.0) * 0.20

        # ================================================================
        # TOTAL CONFIDENCE
        # ================================================================
        confidence = ema_score + swing_break_score + volume_score + pullback_score + rr_score

        return min(confidence, 1.0)

    def _calculate_atr(self, df: pd.DataFrame, period: int = None) -> float:
        """
        Calculate Average True Range (ATR) for volatility-based thresholds

        v1.8.0: Used for ATR-based swing validation and momentum quality filtering

        Args:
            df: OHLCV DataFrame with high, low, close columns
            period: ATR period (default: self.atr_period)

        Returns:
            Current ATR value, or 0.0 if insufficient data
        """
        if period is None:
            period = self.atr_period

        if len(df) < period + 1:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Calculate True Range components
        tr1 = high[1:] - low[1:]  # High - Low
        tr2 = np.abs(high[1:] - close[:-1])  # |High - Previous Close|
        tr3 = np.abs(low[1:] - close[:-1])  # |Low - Previous Close|

        # True Range is the maximum of the three
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)

        # Calculate ATR using exponential moving average (more responsive)
        if len(true_range) < period:
            return 0.0

        # Simple initial ATR
        atr = np.mean(true_range[-period:])

        return atr

    def _calculate_limit_entry(
        self,
        current_close: float,
        direction: str,
        entry_type: str,
        pip_value: float,
        df: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Calculate limit entry price with offset for momentum confirmation.

        v2.2.0: Stop-entry style - confirm price is moving in intended direction:
        - BUY orders placed ABOVE current price (enter when price breaks up)
        - SELL orders placed BELOW current price (enter when price breaks down)
        - Max offset: 3 pips (user request)

        Args:
            current_close: Current market price
            direction: Trade direction ('BULL' or 'BEAR')
            entry_type: Entry type ('PULLBACK' or 'MOMENTUM')
            pip_value: Pip value for the pair (e.g., 0.0001 for EURUSD)
            df: DataFrame for ATR calculation

        Returns:
            Tuple of (limit_entry_price, offset_pips)
        """
        if not self.limit_order_enabled:
            # Limit orders disabled - return current price (market order behavior)
            return current_close, 0.0

        if entry_type == 'PULLBACK':
            # ATR-based offset for pullback entries (adapts to volatility)
            atr = self._calculate_atr(df)
            if atr > 0:
                atr_pips = atr / pip_value
                # Calculate offset as percentage of ATR, clamped to min/max
                offset_pips = atr_pips * self.pullback_offset_atr_factor
                offset_pips = min(max(offset_pips, self.pullback_offset_min_pips), self.pullback_offset_max_pips)
            else:
                # Fallback if ATR unavailable
                offset_pips = self.pullback_offset_min_pips
            self.logger.info(f"   📉 Limit offset (PULLBACK): {offset_pips:.1f} pips (ATR-based)")
        else:
            # Fixed offset for momentum entries (trend is strong)
            offset_pips = self.momentum_offset_pips
            self.logger.info(f"   📉 Limit offset (MOMENTUM): {offset_pips:.1f} pips (fixed)")

        # Calculate offset in price terms
        offset = offset_pips * pip_value

        # Apply offset based on direction (stop-entry style: confirm direction continuation)
        if direction == 'BULL':
            # BUY: Place limit order ABOVE current price (enter when price breaks up)
            limit_entry = current_close + offset
        else:
            # SELL: Place limit order BELOW current price (enter when price breaks down)
            limit_entry = current_close - offset

        self.logger.info(f"   📍 Market price: {current_close:.5f}")
        self.logger.info(f"   📍 Limit entry: {limit_entry:.5f} ({offset_pips:.1f} pips momentum confirmation)")

        return limit_entry, offset_pips

    def _check_session(self, timestamp, epic: str = None) -> Tuple[bool, str]:
        """Check if current time is in allowed trading session

        Session hours are loaded from scanner_global_config database (NO FALLBACK):
        - Asian: session_asian_start_hour to session_asian_end_hour (default 21:00-07:00 UTC)
        - London: session_london_start_hour to session_london_end_hour (default 07:00-16:00 UTC)
        - New York: session_newyork_start_hour to session_newyork_end_hour (default 12:00-21:00 UTC)
        - Overlap: session_overlap_start_hour to session_overlap_end_hour (default 12:00-16:00 UTC)

        Args:
            timestamp: Candle timestamp
            epic: Trading pair epic for pair-specific session overrides (v2.8.0)
        """
        if isinstance(timestamp, pd.Timestamp):
            hour = timestamp.hour
        else:
            hour = datetime.now().hour

        # Load session hours from scanner_global_config database (NO FALLBACK)
        try:
            from forex_scanner.services.scanner_config_service import get_scanner_config
            scanner_cfg = get_scanner_config()
            asian_start = scanner_cfg.session_asian_start_hour
            asian_end = scanner_cfg.session_asian_end_hour
            london_start = scanner_cfg.session_london_start_hour
            overlap_start = scanner_cfg.session_overlap_start_hour
            overlap_end = scanner_cfg.session_overlap_end_hour
            newyork_end = scanner_cfg.session_newyork_end_hour
            block_asian = scanner_cfg.block_asian_session
        except Exception as e:
            # Database required - fail explicitly
            self.logger.error(f"❌ Cannot load session hours from database: {e}")
            raise RuntimeError(f"Session hours database config required: {e}")

        # Asian session check (crosses midnight, e.g., 21:00-07:00)
        in_asian_session = hour >= asian_start or hour < asian_end

        if in_asian_session:
            # v2.8.0: Check pair-specific override for Asian session using database config
            if epic and self._config_service:
                if self._config_service.is_asian_session_allowed(epic):
                    pair_name = epic.split('.')[2] if '.' in epic else epic
                    return True, f"Asian session ALLOWED for {pair_name} (hour={hour})"

            # Use database config for block_asian
            if block_asian:
                return False, f"Asian session blocked (hour={hour})"

        # Active trading sessions (use database config values)
        if london_start <= hour < newyork_end:
            if overlap_start <= hour < overlap_end:
                return True, f"London/NY overlap (hour={hour})"
            elif london_start <= hour < overlap_start:
                return True, f"London session (hour={hour})"
            else:
                return True, f"New York session (hour={hour})"
        else:
            return False, f"Outside trading hours (hour={hour})"

    def _check_cooldown(self, pair: str, current_time: datetime = None) -> Tuple[bool, str]:
        """Check if pair is in cooldown period using adaptive or static logic

        v3.0.0: Now uses adaptive cooldown based on trade outcomes, win rates,
        and market context when ADAPTIVE_COOLDOWN_ENABLED is True.

        Args:
            pair: Currency pair name
            current_time: Current candle timestamp (for backtest mode). If None, uses datetime.now()
        """
        # Use provided time for backtest, or real time for live trading
        check_time = current_time if current_time is not None else datetime.now()

        # No previous signal = no cooldown
        if pair not in self.pair_cooldowns:
            return True, "No cooldown (first signal)"

        last_signal = self.pair_cooldowns[pair]

        # Handle timezone-aware vs naive datetime comparison
        if hasattr(last_signal, 'tzinfo') and last_signal.tzinfo is not None:
            if hasattr(check_time, 'tzinfo') and check_time.tzinfo is None:
                last_signal = last_signal.replace(tzinfo=None)
        elif hasattr(check_time, 'tzinfo') and check_time.tzinfo is not None:
            check_time = check_time.replace(tzinfo=None)

        hours_since = (check_time - last_signal).total_seconds() / 3600

        # Calculate effective cooldown (adaptive or static)
        if self.adaptive_cooldown_enabled:
            effective_cooldown, cooldown_breakdown = self._calculate_adaptive_cooldown(pair, check_time)
            cooldown_type = "adaptive"
        else:
            effective_cooldown = self.cooldown_hours
            cooldown_breakdown = "static"
            cooldown_type = "static"

        if hours_since < effective_cooldown:
            return False, f"In {cooldown_type} cooldown ({hours_since:.1f}h < {effective_cooldown:.1f}h) [{cooldown_breakdown}]"

        return True, f"Cooldown expired ({hours_since:.1f}h >= {effective_cooldown:.1f}h {cooldown_type})"

    def _calculate_adaptive_cooldown(self, pair: str, current_time: datetime) -> Tuple[float, str]:
        """Calculate dynamic cooldown based on trade outcomes and market context

        v3.0.0: Implements intelligent cooldown that adapts to:
        - Last trade outcome (win/loss)
        - Consecutive losses on pair
        - Rolling win rate
        - Session changes

        Args:
            pair: Currency pair name (e.g., 'EURUSD')
            current_time: Current timestamp for calculations

        Returns:
            Tuple of (cooldown_hours, breakdown_string)
        """
        breakdown_parts = []
        cooldown = self.base_cooldown_hours
        breakdown_parts.append(f"base={cooldown:.1f}h")

        # Get the epic for database queries (pair might be 'EURUSD', need 'CS.D.EURUSD.CEEM.IP')
        epic = self._get_epic_for_pair(pair)

        # 1. Check consecutive losses - if blocked, return early
        consecutive_losses = self.pair_consecutive_losses.get(pair, 0)
        if consecutive_losses >= self.max_consecutive_losses_before_block:
            return self.consecutive_loss_block_hours, f"BLOCKED ({consecutive_losses} consecutive losses)"

        # 2. Last trade outcome adjustment
        last_outcome = self._get_last_trade_outcome(epic)
        if last_outcome:
            if last_outcome.get('profitable'):
                cooldown *= self.cooldown_after_win_multiplier
                breakdown_parts.append(f"win×{self.cooldown_after_win_multiplier}")
            else:
                cooldown *= self.cooldown_after_loss_multiplier
                breakdown_parts.append(f"loss×{self.cooldown_after_loss_multiplier}")

        # 3. Consecutive loss penalty (additive)
        if consecutive_losses > 0:
            penalty = consecutive_losses * self.consecutive_loss_penalty_hours
            cooldown += penalty
            breakdown_parts.append(f"+{penalty:.1f}h ({consecutive_losses} consec)")

        # 4. Win rate adjustment
        win_rate = self._get_pair_win_rate(epic)
        if win_rate is not None:
            if win_rate >= self.high_win_rate_threshold:
                reduction = 1.0 - self.high_win_rate_cooldown_reduction
                cooldown *= reduction
                breakdown_parts.append(f"highWR({win_rate*100:.0f}%)×{reduction:.2f}")
            elif win_rate < self.low_win_rate_threshold:
                increase = 1.0 + self.low_win_rate_cooldown_increase
                cooldown *= increase
                breakdown_parts.append(f"lowWR({win_rate*100:.0f}%)×{increase:.2f}")

        # 5. Session change check - reset if new session
        if self.session_change_reset_cooldown:
            current_session = self._get_current_session(current_time)
            last_session = self.pair_last_session.get(pair)
            if last_session and current_session != last_session:
                # Session changed - reduce cooldown significantly
                cooldown = min(cooldown, self.min_cooldown_hours)
                breakdown_parts.append(f"session_change({last_session}→{current_session})")

        # 6. Clamp to bounds
        original_cooldown = cooldown
        cooldown = max(self.min_cooldown_hours, min(self.max_cooldown_hours, cooldown))
        if cooldown != original_cooldown:
            breakdown_parts.append(f"clamped({original_cooldown:.1f}→{cooldown:.1f})")

        return cooldown, " | ".join(breakdown_parts)

    def _get_epic_for_pair(self, pair: str) -> str:
        """Convert pair name to epic format for database queries

        Args:
            pair: Short pair name (e.g., 'EURUSD') or full epic

        Returns:
            Full epic code (e.g., 'CS.D.EURUSD.CEEM.IP')
        """
        # If already an epic, return as-is
        if pair.startswith('CS.D.'):
            return pair

        # Map common pair names to epics
        pair_to_epic = {
            'EURUSD': 'CS.D.EURUSD.CEEM.IP',
            'GBPUSD': 'CS.D.GBPUSD.MINI.IP',
            'USDJPY': 'CS.D.USDJPY.MINI.IP',
            'USDCHF': 'CS.D.USDCHF.MINI.IP',
            'AUDUSD': 'CS.D.AUDUSD.MINI.IP',
            'USDCAD': 'CS.D.USDCAD.MINI.IP',
            'NZDUSD': 'CS.D.NZDUSD.MINI.IP',
            'EURJPY': 'CS.D.EURJPY.MINI.IP',
            'GBPJPY': 'CS.D.GBPJPY.MINI.IP',
            'AUDJPY': 'CS.D.AUDJPY.MINI.IP',
        }
        return pair_to_epic.get(pair.upper(), f'CS.D.{pair.upper()}.MINI.IP')

    def _get_last_trade_outcome(self, epic: str) -> Optional[Dict]:
        """Query database for most recent closed trade on this pair

        Args:
            epic: IG Markets epic code

        Returns:
            Dict with 'profitable', 'pnl', 'closed_at' or None if no trades found
        """
        # Check cache first (avoid repeated DB queries within short period)
        cache_key = epic
        cache_entry = self._trade_outcome_cache.get(cache_key)
        if cache_entry:
            cache_age = (datetime.now() - cache_entry['cached_at']).total_seconds()
            if cache_age < 300:  # 5 minute cache
                return cache_entry['outcome']

        # No db_manager = no database queries possible
        if self._db_manager is None:
            return None

        try:
            conn = self._db_manager.get_connection()
            cursor = conn.cursor()

            # Query for most recent closed trade on this symbol
            query = """
                SELECT
                    symbol,
                    profit_loss,
                    (profit_loss > 0) as profitable,
                    closed_at,
                    pips_gained
                FROM trade_log
                WHERE symbol = %s
                  AND status = 'closed'
                  AND closed_at IS NOT NULL
                ORDER BY closed_at DESC
                LIMIT 1
            """
            cursor.execute(query, (epic,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row:
                outcome = {
                    'symbol': row[0],
                    'pnl': float(row[1]) if row[1] else 0.0,
                    'profitable': bool(row[2]),
                    'closed_at': row[3],
                    'pips_gained': float(row[4]) if row[4] else 0.0
                }
                # Cache the result
                self._trade_outcome_cache[cache_key] = {
                    'outcome': outcome,
                    'cached_at': datetime.now()
                }
                self.logger.debug(f"📊 Last trade outcome for {epic}: {'WIN' if outcome['profitable'] else 'LOSS'} ({outcome['pnl']:.2f})")
                return outcome

            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get last trade outcome for {epic}: {e}")
            return None

    def _get_pair_win_rate(self, epic: str) -> Optional[float]:
        """Calculate rolling win rate for pair over last N trades

        Args:
            epic: IG Markets epic code

        Returns:
            Win rate as float (0.0 to 1.0) or None if insufficient data
        """
        if self._db_manager is None:
            return None

        try:
            conn = self._db_manager.get_connection()
            cursor = conn.cursor()

            # Get last N closed trades for this symbol
            query = """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades
                FROM (
                    SELECT profit_loss
                    FROM trade_log
                    WHERE symbol = %s
                      AND status = 'closed'
                      AND closed_at IS NOT NULL
                      AND profit_loss IS NOT NULL
                    ORDER BY closed_at DESC
                    LIMIT %s
                ) recent_trades
            """
            cursor.execute(query, (epic, self.win_rate_lookback_trades))
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row and row[0] and row[0] >= 5:  # Require at least 5 trades for meaningful win rate
                total = int(row[0])
                wins = int(row[1]) if row[1] else 0
                win_rate = wins / total
                self.logger.debug(f"📊 Win rate for {epic}: {win_rate*100:.1f}% ({wins}/{total})")
                return win_rate

            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get win rate for {epic}: {e}")
            return None

    def _get_current_session(self, current_time: datetime) -> str:
        """Determine current trading session

        Session boundaries are loaded from scanner_global_config database (NO FALLBACK).

        Args:
            current_time: Current timestamp

        Returns:
            Session name: 'asian', 'london', 'new_york', 'overlap'
        """
        hour = current_time.hour

        # Load session hours from scanner_global_config database (NO FALLBACK)
        try:
            from forex_scanner.services.scanner_config_service import get_scanner_config
            scanner_cfg = get_scanner_config()
            asian_start = scanner_cfg.session_asian_start_hour
            asian_end = scanner_cfg.session_asian_end_hour
            london_start = scanner_cfg.session_london_start_hour
            overlap_start = scanner_cfg.session_overlap_start_hour
            overlap_end = scanner_cfg.session_overlap_end_hour
        except Exception as e:
            # Database required - fail explicitly
            self.logger.error(f"❌ Cannot load session hours from database: {e}")
            raise RuntimeError(f"Session hours database config required: {e}")

        # Session definitions (from database)
        if asian_start <= hour or hour < asian_end:
            return 'asian'
        elif london_start <= hour < overlap_start:
            return 'london'
        elif overlap_start <= hour < overlap_end:
            return 'overlap'  # London-NY overlap
        else:
            return 'new_york'

    def _update_cooldown(self, pair: str, signal_time: datetime = None):
        """Update cooldown after signal generation

        Args:
            pair: Currency pair name
            signal_time: Signal timestamp (for backtest mode). If None, uses datetime.now()
        """
        effective_time = signal_time if signal_time is not None else datetime.now()
        self.pair_cooldowns[pair] = effective_time

        # Track current session for session change detection
        if self.adaptive_cooldown_enabled:
            self.pair_last_session[pair] = self._get_current_session(effective_time)

    def update_trade_outcome(self, pair: str, profitable: bool):
        """Update consecutive loss tracking after a trade closes

        Call this method when a trade closes to update the adaptive cooldown state.

        Args:
            pair: Currency pair name
            profitable: Whether the trade was profitable
        """
        if profitable:
            # Reset consecutive losses on win
            self.pair_consecutive_losses[pair] = 0
            self.logger.info(f"🟢 {pair}: Consecutive losses reset after WIN")
        else:
            # Increment consecutive losses
            current = self.pair_consecutive_losses.get(pair, 0)
            self.pair_consecutive_losses[pair] = current + 1
            self.logger.warning(f"🔴 {pair}: Consecutive losses now {current + 1}")

            if self.pair_consecutive_losses[pair] >= self.max_consecutive_losses_before_block:
                self.logger.error(f"⛔ {pair}: BLOCKED for {self.consecutive_loss_block_hours}h after {self.pair_consecutive_losses[pair]} consecutive losses")

        # Clear outcome cache to force fresh query
        epic = self._get_epic_for_pair(pair)
        if epic in self._trade_outcome_cache:
            del self._trade_outcome_cache[epic]

    def reset_cooldowns(self):
        """Reset all cooldowns - call this at the start of each backtest to ensure fresh state"""
        self.pair_cooldowns = {}
        self.pair_consecutive_losses = {}
        self.pair_last_session = {}
        self._trade_outcome_cache = {}
        if self.logger:
            self.logger.info("🔄 SMC Simple strategy cooldowns reset for new backtest (v3.0.0 adaptive state cleared)")

    def _build_description(
        self,
        direction: str,
        ema_distance: float,
        pullback_depth: float,
        rr_ratio: float
    ) -> str:
        """Build human-readable signal description"""
        parts = []

        # Direction and EMA
        parts.append(f"{direction} trend")
        parts.append(f"{ema_distance:.0f} pips from 50 EMA")

        # Pullback
        parts.append(f"{pullback_depth*100:.0f}% pullback")

        # R:R
        parts.append(f"{rr_ratio:.1f}R")

        return ", ".join(parts).capitalize()

    # =========================================================================
    # REJECTION TRACKING (v2.2.0)
    # =========================================================================

    def _track_rejection(
        self,
        stage: str,
        reason: str,
        epic: str,
        pair: str,
        candle_timestamp: Any,
        direction: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> None:
        """
        Track a rejection for later analysis.

        Args:
            stage: Rejection stage (SESSION, COOLDOWN, TIER1_EMA, etc.)
            reason: Human-readable rejection reason
            epic: IG Markets epic code
            pair: Currency pair name
            candle_timestamp: Timestamp of the candle being analyzed
            direction: Attempted direction (BULL/BEAR) if known
            context: Additional context data (prices, indicators, etc.)
        """
        if self.rejection_manager is None:
            return

        try:
            # Convert timestamp to datetime
            if hasattr(candle_timestamp, 'to_pydatetime'):
                scan_ts = candle_timestamp.to_pydatetime()
            elif isinstance(candle_timestamp, (int, np.integer)):
                scan_ts = pd.Timestamp(candle_timestamp).to_pydatetime()
            else:
                scan_ts = candle_timestamp if candle_timestamp else datetime.now(timezone.utc)

            # Ensure timezone-aware
            if scan_ts.tzinfo is None:
                scan_ts = scan_ts.replace(tzinfo=timezone.utc)

            # Build rejection data
            rejection_data = {
                'scan_timestamp': scan_ts,
                'epic': epic,
                'pair': pair,
                'rejection_stage': stage,
                'rejection_reason': reason,
                'attempted_direction': direction,
                'strategy_version': self.strategy_version,
            }

            # Add context if provided
            if context:
                rejection_data.update(context)

            # Determine market hour and session
            if 'market_hour' not in rejection_data:
                rejection_data['market_hour'] = scan_ts.hour

            # Save rejection
            self.rejection_manager.save_rejection(rejection_data)

        except Exception as e:
            if self.debug_logging:
                self.logger.debug(f"Failed to track rejection: {e}")

    def _collect_market_context(
        self,
        df_trigger: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        df_entry: Optional[pd.DataFrame] = None,
        pip_value: float = 0.0001,
        ema_result: Optional[Dict] = None,
        swing_result: Optional[Dict] = None,
        pullback_result: Optional[Dict] = None,
        risk_result: Optional[Dict] = None,
        direction: Optional[str] = None
    ) -> Dict:
        """
        Collect market context at the point of rejection.

        Returns a dictionary with all available market state data.
        """
        context = {}

        # MACD indicators (calculate from df_trigger if available)
        if df_trigger is not None and len(df_trigger) > 0:
            macd_line = 0.0
            macd_signal = 0.0
            macd_histogram = 0.0
            if 'macd_line' in df_trigger.columns:
                macd_line = float(df_trigger['macd_line'].iloc[-1]) if pd.notna(df_trigger['macd_line'].iloc[-1]) else 0.0
                macd_signal = float(df_trigger['macd_signal'].iloc[-1]) if 'macd_signal' in df_trigger.columns and pd.notna(df_trigger['macd_signal'].iloc[-1]) else 0.0
                macd_histogram = float(df_trigger['macd_histogram'].iloc[-1]) if 'macd_histogram' in df_trigger.columns and pd.notna(df_trigger['macd_histogram'].iloc[-1]) else 0.0
            elif len(df_trigger) >= 26:
                # Calculate MACD if not in dataframe
                close = df_trigger['close']
                ema_12 = close.ewm(span=12, adjust=False).mean()
                ema_26 = close.ewm(span=26, adjust=False).mean()
                macd_line = float(ema_12.iloc[-1] - ema_26.iloc[-1])
                signal_line = (ema_12 - ema_26).ewm(span=9, adjust=False).mean()
                macd_signal = float(signal_line.iloc[-1])
                macd_histogram = macd_line - macd_signal

            if macd_line != 0.0 or macd_signal != 0.0:
                context['macd_line'] = macd_line
                context['macd_signal'] = macd_signal
                context['macd_histogram'] = macd_histogram
                macd_momentum = 'bullish' if macd_line > macd_signal else 'bearish'
                context['macd_momentum'] = macd_momentum
                # Check alignment with trade direction if available
                if direction:
                    is_aligned = (direction == 'BULL' and macd_momentum == 'bullish') or \
                                 (direction == 'BEAR' and macd_momentum == 'bearish')
                    context['macd_aligned'] = is_aligned

        # Get current price from entry df or trigger df
        current_df = df_entry if df_entry is not None and len(df_entry) > 0 else df_trigger
        if current_df is not None and len(current_df) > 0:
            context['current_price'] = float(current_df['close'].iloc[-1])
            if 'bid' in current_df.columns:
                context['bid_price'] = float(current_df['bid'].iloc[-1])
            if 'ask' in current_df.columns:
                context['ask_price'] = float(current_df['ask'].iloc[-1])
            if 'spread' in current_df.columns:
                context['spread_pips'] = float(current_df['spread'].iloc[-1])

        # EMA context (Tier 1)
        if ema_result:
            context['ema_4h_value'] = ema_result.get('ema_value')
            context['ema_distance_pips'] = ema_result.get('distance_pips')
            direction = ema_result.get('direction')
            if direction:
                if ema_result.get('valid'):
                    context['price_position_vs_ema'] = 'above' if direction == 'BULL' else 'below'
                else:
                    context['price_position_vs_ema'] = 'in_buffer'

        # ATR from trigger timeframe
        if df_trigger is not None and len(df_trigger) > 0:
            atr = self._calculate_atr(df_trigger)
            if atr > 0:
                context['atr_15m'] = atr

        # ATR from entry timeframe
        if df_entry is not None and len(df_entry) > 0:
            atr_entry = self._calculate_atr(df_entry)
            if atr_entry > 0:
                context['atr_5m'] = atr_entry

        # Volume context
        if df_trigger is not None and len(df_trigger) > 0:
            vol_col = 'ltv' if 'ltv' in df_trigger.columns else 'volume' if 'volume' in df_trigger.columns else None
            if vol_col and df_trigger[vol_col].iloc[-1] > 0:
                context['current_volume'] = float(df_trigger[vol_col].iloc[-1])
                vol_sma = df_trigger[vol_col].iloc[-min(self.volume_sma_period, len(df_trigger)):].mean()
                if vol_sma > 0:
                    context['volume_sma'] = float(vol_sma)
                    context['volume_ratio'] = float(df_trigger[vol_col].iloc[-1] / vol_sma)

        # Swing context (Tier 2)
        if swing_result:
            if swing_result.get('swing_level'):
                # Determine which is high/low based on direction
                if ema_result and ema_result.get('direction') == 'BULL':
                    context['swing_high_level'] = swing_result.get('swing_level')
                    context['swing_low_level'] = swing_result.get('opposite_swing')
                else:
                    context['swing_low_level'] = swing_result.get('swing_level')
                    context['swing_high_level'] = swing_result.get('opposite_swing')
            context['swings_found_count'] = swing_result.get('swings_found', 0)

        # Pullback/Entry context (Tier 3)
        if pullback_result:
            context['pullback_depth'] = pullback_result.get('pullback_depth')
            context['swing_range_pips'] = pullback_result.get('swing_range_pips')
            in_optimal = pullback_result.get('in_optimal_zone', False)
            depth = pullback_result.get('pullback_depth', 0)
            if depth < 0:
                context['fib_zone'] = 'beyond_break'
            elif in_optimal:
                context['fib_zone'] = 'optimal'
            elif self.fib_min <= depth <= self.fib_max:
                context['fib_zone'] = 'valid'
            else:
                context['fib_zone'] = 'outside'

        # Risk/Reward context
        if risk_result:
            context['potential_entry'] = risk_result.get('entry_price')
            context['potential_stop_loss'] = risk_result.get('stop_loss')
            context['potential_take_profit'] = risk_result.get('take_profit')
            context['potential_risk_pips'] = risk_result.get('risk_pips')
            context['potential_reward_pips'] = risk_result.get('reward_pips')
            context['potential_rr_ratio'] = risk_result.get('rr_ratio')

        # OHLCV snapshots
        if df_entry is not None and len(df_entry) > 0:
            context['candle_5m_open'] = float(df_entry['open'].iloc[-1])
            context['candle_5m_high'] = float(df_entry['high'].iloc[-1])
            context['candle_5m_low'] = float(df_entry['low'].iloc[-1])
            context['candle_5m_close'] = float(df_entry['close'].iloc[-1])
            vol_col = 'ltv' if 'ltv' in df_entry.columns else 'volume' if 'volume' in df_entry.columns else None
            if vol_col:
                context['candle_5m_volume'] = float(df_entry[vol_col].iloc[-1])

        if df_trigger is not None and len(df_trigger) > 0:
            context['candle_15m_open'] = float(df_trigger['open'].iloc[-1])
            context['candle_15m_high'] = float(df_trigger['high'].iloc[-1])
            context['candle_15m_low'] = float(df_trigger['low'].iloc[-1])
            context['candle_15m_close'] = float(df_trigger['close'].iloc[-1])
            vol_col = 'ltv' if 'ltv' in df_trigger.columns else 'volume' if 'volume' in df_trigger.columns else None
            if vol_col:
                context['candle_15m_volume'] = float(df_trigger[vol_col].iloc[-1])

        if df_4h is not None and len(df_4h) > 0:
            context['candle_4h_open'] = float(df_4h['open'].iloc[-1])
            context['candle_4h_high'] = float(df_4h['high'].iloc[-1])
            context['candle_4h_low'] = float(df_4h['low'].iloc[-1])
            context['candle_4h_close'] = float(df_4h['close'].iloc[-1])
            vol_col = 'ltv' if 'ltv' in df_4h.columns else 'volume' if 'volume' in df_4h.columns else None
            if vol_col:
                context['candle_4h_volume'] = float(df_4h[vol_col].iloc[-1])

        # ================================================================
        # PERFORMANCE METRICS for rejection analysis
        # ================================================================
        context = self._add_performance_metrics_to_rejection(
            context, df_trigger, df_4h, df_entry
        )

        return context

    def _add_performance_metrics(
        self,
        signal: Dict,
        df_entry: Optional[pd.DataFrame],
        df_trigger: pd.DataFrame,
        df_4h: pd.DataFrame,
        epic: str
    ) -> Dict:
        """
        Add enhanced performance metrics to signal for analysis.

        This method calculates additional metrics (Efficiency Ratio, Market Regime,
        Entry Quality, etc.) and adds them to the signal dict without affecting
        the core signal detection logic.

        Args:
            signal: The signal dict to enhance
            df_entry: Entry timeframe data (5m)
            df_trigger: Trigger timeframe data (15m)
            df_4h: 4H timeframe data
            epic: Trading pair epic code

        Returns:
            Enhanced signal dict with performance_metrics added
        """
        try:
            # Lazy import to avoid circular dependencies
            from forex_scanner.core.strategies.helpers.smc_performance_metrics import (
                get_performance_metrics_calculator
            )

            calculator = get_performance_metrics_calculator(self.logger)

            # Calculate full metrics
            metrics = calculator.calculate_metrics(
                df_5m=df_entry,
                df_15m=df_trigger,
                df_4h=df_4h,
                signal_data=signal,
                epic=epic
            )

            # Add metrics to signal as nested dict
            metrics_dict = metrics.to_json_safe()
            signal['performance_metrics'] = metrics_dict

            # Also add key metrics as top-level fields for easier database storage
            signal['efficiency_ratio'] = metrics.efficiency_ratio
            signal['market_regime_detected'] = metrics.market_regime
            signal['regime_confidence'] = metrics.regime_confidence
            signal['bb_width_percentile'] = metrics.bb_width_percentile
            signal['atr_percentile'] = metrics.atr_percentile
            signal['volatility_state'] = metrics.volatility_state
            signal['entry_quality_score'] = metrics.entry_quality_score
            signal['distance_from_optimal_fib'] = metrics.distance_from_optimal_fib
            signal['entry_candle_momentum'] = metrics.entry_candle_momentum
            signal['mtf_confluence_score'] = metrics.mtf_confluence_score
            signal['htf_candle_position'] = metrics.htf_candle_position
            signal['all_timeframes_aligned'] = metrics.all_timeframes_aligned
            signal['volume_at_swing_break'] = metrics.volume_at_swing_break
            signal['volume_trend'] = metrics.volume_trend
            signal['volume_quality_score'] = metrics.volume_quality_score
            signal['adx_value'] = metrics.adx_value
            signal['adx_plus_di'] = metrics.adx_plus_di
            signal['adx_minus_di'] = metrics.adx_minus_di
            signal['adx_trend_strength'] = metrics.adx_trend_strength

            # Extract extended indicators from DataFrames (non-invasive)
            self._extract_extended_indicators(signal, df_entry, df_trigger, df_4h)

            if self.debug_logging:
                er_str = f"{metrics.efficiency_ratio:.3f}" if metrics.efficiency_ratio is not None else "N/A"
                eq_str = f"{metrics.entry_quality_score:.2f}" if metrics.entry_quality_score is not None else "N/A"
                self.logger.debug(
                    f"📊 Performance metrics added: ER={er_str}, "
                    f"Regime={metrics.market_regime}, EntryQuality={eq_str}"
                )

        except ImportError as e:
            self.logger.debug(f"Performance metrics module not available: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to add performance metrics: {e}")

        return signal

    def _add_performance_metrics_to_rejection(
        self,
        context: Dict,
        df_trigger: Optional[pd.DataFrame],
        df_4h: Optional[pd.DataFrame],
        df_entry: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Add quick performance metrics to rejection context.

        This is a lighter-weight version for rejections where we may not
        have all data available.

        Args:
            context: Existing rejection context dict
            df_trigger: Trigger timeframe data
            df_4h: 4H timeframe data
            df_entry: Optional entry timeframe data

        Returns:
            Context dict with performance metrics added
        """
        try:
            from forex_scanner.core.strategies.helpers.smc_performance_metrics import (
                get_performance_metrics_calculator
            )

            calculator = get_performance_metrics_calculator(self.logger)

            # Use the most available dataframe
            df_primary = df_entry if df_entry is not None and len(df_entry) > 0 else df_trigger

            if df_primary is not None and len(df_primary) >= 20:
                quick_metrics = calculator.calculate_quick_metrics(df_primary)

                context['efficiency_ratio'] = quick_metrics.get('efficiency_ratio')
                context['market_regime_detected'] = quick_metrics.get('market_regime')
                context['atr_percentile'] = quick_metrics.get('atr_percentile')
                context['volatility_state'] = quick_metrics.get('volatility_state')

                # Also calculate BB percentile if possible
                if df_4h is not None:
                    full_metrics = calculator.calculate_metrics(
                        df_5m=df_entry,
                        df_15m=df_trigger,
                        df_4h=df_4h
                    )
                    context['bb_width_percentile'] = full_metrics.bb_width_percentile
                    context['adx_value'] = full_metrics.adx_value
                    context['adx_trend_strength'] = full_metrics.adx_trend_strength
                    context['performance_metrics'] = full_metrics.to_json_safe()

        except ImportError:
            pass  # Module not available
        except Exception as e:
            if self.debug_logging:
                self.logger.debug(f"Failed to add rejection metrics: {e}")

        return context

    def _extract_extended_indicators(
        self,
        signal: Dict,
        df_entry: Optional[pd.DataFrame],
        df_trigger: Optional[pd.DataFrame],
        df_4h: Optional[pd.DataFrame]
    ) -> None:
        """
        Extract extended indicator values from DataFrames for trade analysis.

        This method is NON-INVASIVE - it only reads from DataFrames and adds
        to the signal dict. It cannot break the strategy as it's wrapped in
        try/except and only runs after signal detection is complete.

        Extracts: KAMA, Bollinger Bands, Stochastic, Supertrend, EMAs, RSI zones
        """
        try:
            # Use trigger timeframe as primary source
            df = df_trigger if df_trigger is not None and len(df_trigger) > 0 else df_entry
            if df is None or len(df) == 0:
                return

            latest = df.iloc[-1]
            price = signal.get('entry_price', latest.get('close', 0))

            # Get pip size for calculations
            epic = signal.get('epic', '')
            pip_size = 0.01 if 'JPY' in epic else 0.0001

            # === KAMA Indicators ===
            # KAMA columns are named kama_{period}, kama_{period}_er, etc.
            for period in [10, 21]:
                kama_col = f'kama_{period}'
                if kama_col in df.columns:
                    signal['kama_value'] = self._safe_float(latest.get(kama_col))
                    signal['kama_er'] = self._safe_float(latest.get(f'{kama_col}_er'))
                    signal['kama_trend'] = latest.get(f'{kama_col}_trend')
                    signal['kama_signal'] = latest.get(f'{kama_col}_signal')
                    break

            # === Bollinger Bands ===
            if 'bb_upper' in df.columns:
                bb_upper = self._safe_float(latest.get('bb_upper'))
                bb_lower = self._safe_float(latest.get('bb_lower'))
                signal['bb_upper'] = bb_upper
                signal['bb_lower'] = bb_lower

                # Calculate middle and width
                if bb_upper and bb_lower:
                    signal['bb_middle'] = (bb_upper + bb_lower) / 2
                    signal['bb_width'] = bb_upper - bb_lower

                    # %B indicator: (price - lower) / (upper - lower)
                    bb_range = bb_upper - bb_lower
                    if bb_range > 0:
                        signal['bb_percent_b'] = (price - bb_lower) / bb_range

                    # Price position relative to bands
                    if price > bb_upper:
                        signal['price_vs_bb'] = 'above_upper'
                    elif price < bb_lower:
                        signal['price_vs_bb'] = 'below_lower'
                    else:
                        signal['price_vs_bb'] = 'in_band'

            # === Stochastic ===
            if 'stoch_k' in df.columns:
                stoch_k = self._safe_float(latest.get('stoch_k'))
                stoch_d = self._safe_float(latest.get('stoch_d'))
                signal['stoch_k'] = stoch_k
                signal['stoch_d'] = stoch_d

                if stoch_k is not None:
                    if stoch_k > 80:
                        signal['stoch_zone'] = 'overbought'
                    elif stoch_k < 20:
                        signal['stoch_zone'] = 'oversold'
                    else:
                        signal['stoch_zone'] = 'neutral'

            # === Supertrend ===
            if 'supertrend' in df.columns:
                signal['supertrend_value'] = self._safe_float(latest.get('supertrend'))
                signal['supertrend_direction'] = int(latest.get('supertrend_direction', 0)) if pd.notna(latest.get('supertrend_direction')) else None

            # === RSI Zone ===
            rsi = self._safe_float(latest.get('rsi'))
            if rsi is not None:
                signal['rsi'] = rsi  # Also ensure RSI is in signal
                if rsi > 70:
                    signal['rsi_zone'] = 'overbought'
                elif rsi < 30:
                    signal['rsi_zone'] = 'oversold'
                else:
                    signal['rsi_zone'] = 'neutral'

            # === EMA Stack ===
            ema_values = {}
            for ema_period in [9, 21, 50, 200]:
                col_name = f'ema_{ema_period}'
                if col_name in df.columns:
                    val = self._safe_float(latest.get(col_name))
                    signal[col_name] = val
                    if val is not None:
                        ema_values[ema_period] = val

            # Price vs EMA 200
            if 200 in ema_values:
                signal['price_vs_ema_200'] = 'above' if price > ema_values[200] else 'below'

            # EMA stack order (bullish = 9 > 21 > 50 > 200)
            if len(ema_values) >= 3:
                sorted_emas = sorted(ema_values.items(), key=lambda x: x[1], reverse=True)
                sorted_periods = [p for p, v in sorted_emas]

                # Check if shorter EMAs are above longer ones (bullish)
                if sorted_periods == sorted(sorted_periods):  # Ascending order of periods
                    signal['ema_stack_order'] = 'bullish'
                elif sorted_periods == sorted(sorted_periods, reverse=True):  # Descending
                    signal['ema_stack_order'] = 'bearish'
                else:
                    signal['ema_stack_order'] = 'mixed'

            # === Candle Analysis ===
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                o, h, l, c = latest['open'], latest['high'], latest['low'], latest['close']

                body = abs(c - o)
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l

                signal['candle_body_pips'] = body / pip_size
                signal['candle_upper_wick_pips'] = upper_wick / pip_size
                signal['candle_lower_wick_pips'] = lower_wick / pip_size

                # Candle type
                if body < (h - l) * 0.1:
                    signal['candle_type'] = 'doji'
                elif c > o:
                    signal['candle_type'] = 'bullish'
                else:
                    signal['candle_type'] = 'bearish'

        except Exception as e:
            # Never let this break the strategy
            if self.debug_logging:
                self.logger.debug(f"Extended indicator extraction failed (non-critical): {e}")

    def _add_smc_chart_data(
        self,
        signal: Dict,
        df_trigger: pd.DataFrame,
        df_entry: pd.DataFrame,
        epic: str,
        pip_value: float
    ) -> Dict:
        """
        Add FVG and Order Block data to signal for chart visualization.

        This data is used by the chart generator to draw institutional
        price zones that provide visual context for Claude's analysis.

        Args:
            signal: Signal dict to enhance
            df_trigger: 15m trigger timeframe data
            df_entry: 5m entry timeframe data
            epic: Currency pair epic
            pip_value: Pip value for the pair

        Returns:
            Enhanced signal with smc_data
        """
        try:
            from forex_scanner.core.strategies.helpers.smc_fair_value_gaps import SMCFairValueGaps
            from forex_scanner.core.strategies.helpers.smc_order_blocks import SMCOrderBlocks

            smc_data = {
                'fvg_data': {'active_fvgs': []},
                'order_block_data': {'active_order_blocks': []}
            }

            # Configuration for FVG/OB detection
            config = {
                'epic': epic,
                'pair': epic.split('.')[2] if '.' in epic else epic,
                'pip_value': pip_value,
                'fvg_min_size': 3,  # Minimum 3 pips for FVG
                'fvg_max_age': 30,  # Max 30 bars old
                'order_block_length': 3,
                'order_block_volume_factor': 1.3,
                'bos_threshold': pip_value * 5,
                'max_order_blocks': 3
            }

            # Detect FVGs on trigger timeframe (15m)
            try:
                fvg_detector = SMCFairValueGaps(logger=self.logger)
                df_with_fvgs = fvg_detector.detect_fair_value_gaps(df_trigger.copy(), config)

                # Get active FVGs
                active_fvgs = []
                for fvg in fvg_detector.fair_value_gaps:
                    if fvg.status.value in ['active', 'partially_filled']:
                        active_fvgs.append({
                            'high': fvg.high_price,
                            'low': fvg.low_price,
                            'type': fvg.gap_type.value,
                            'start_index': fvg.start_index,
                            'size_pips': fvg.gap_size_pips,
                            'significance': fvg.significance,
                            'fill_percentage': fvg.fill_percentage
                        })

                smc_data['fvg_data']['active_fvgs'] = active_fvgs[:5]  # Limit to 5
                self.logger.info(f"📊 [CHART] Detected {len(active_fvgs)} active FVGs for chart visualization")

            except Exception as fvg_error:
                self.logger.warning(f"⚠️ [CHART] FVG detection failed: {fvg_error}")

            # Detect Order Blocks on trigger timeframe (15m)
            try:
                ob_detector = SMCOrderBlocks(logger=self.logger)
                df_with_obs = ob_detector.detect_order_blocks(df_trigger.copy(), config)

                # Get active Order Blocks
                active_obs = []
                for ob in ob_detector.order_blocks:
                    if ob.still_valid:
                        active_obs.append({
                            'high': ob.high_price,
                            'low': ob.low_price,
                            'type': ob.block_type.value,
                            'start_index': ob.start_index,
                            'end_index': ob.end_index,
                            'strength': ob.strength.value,
                            'confidence': ob.confidence,
                            'has_fvg_support': ob.has_fvg_support
                        })

                smc_data['order_block_data']['active_order_blocks'] = active_obs[:3]  # Limit to 3
                self.logger.info(f"📊 [CHART] Detected {len(active_obs)} active Order Blocks for chart visualization")

            except Exception as ob_error:
                self.logger.warning(f"⚠️ [CHART] Order Block detection failed: {ob_error}")

            # Add to signal
            signal['smc_data'] = smc_data

        except ImportError as ie:
            # FVG/OB helpers not available - not critical
            self.logger.debug(f"SMC helpers not available: {ie}")
            signal['smc_data'] = {
                'fvg_data': {'active_fvgs': []},
                'order_block_data': {'active_order_blocks': []}
            }
        except Exception as e:
            self.logger.debug(f"SMC chart data extraction failed: {e}")
            signal['smc_data'] = {
                'fvg_data': {'active_fvgs': []},
                'order_block_data': {'active_order_blocks': []}
            }

        return signal

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float, returning None on failure."""
        if value is None or (hasattr(value, '__len__') and len(value) == 0):
            return None
        try:
            import numpy as np
            if isinstance(value, (np.integer, np.floating)):
                result = float(value)
            else:
                result = float(value)
            # Handle NaN
            if result != result:  # NaN check
                return None
            return result
        except (TypeError, ValueError):
            return None

    def flush_rejections(self) -> bool:
        """Flush any pending rejections to database. Call this at end of scan cycle."""
        if self.rejection_manager is not None:
            return self.rejection_manager.flush()
        return True


def create_smc_simple_strategy(config, logger=None, db_manager=None):
    """Factory function to create SMC Simple Strategy instance"""
    return SMCSimpleStrategy(config, logger, db_manager)
