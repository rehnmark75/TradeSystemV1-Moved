#!/usr/bin/env python3
"""
SMC Simple Strategy - 3-Tier EMA-Based Trend Following

VERSION: 2.35.1
DATE: 2026-01-29
STATUS: Scalp Mode for High-Frequency Trading

v2.35.1 CHANGES (Enable S/R Validation for Scalp Mode):
    - FIX: Removed skip_sr_validation bypass for scalp trades
    - ANALYSIS: Trades <6 pips from opposing S/R had 27% win rate, -$895 total loss
    - ANALYSIS: Trades >10 pips from opposing S/R had 46% win rate, +$441 total profit
    - IMPACT: Scalp trades now use 6 pip S/R tolerance filter (same as swing)
    - DATA: 120 losing trades would have been filtered, saving ~$895
    - FUTURE: Per-pair S/R thresholds to be added for fine-tuning

v2.35.0 CHANGES (HTF Bias Score System - Professional Redesign):
    - REDESIGN: Replaced binary HTF alignment filter with continuous bias score (0.0-1.0)
    - PHILOSOPHY: Professional algo trader approach - continuous measurement vs binary gates
    - NEW: htf_bias_calculator.py helper for score calculation
    - NEW: Three-component scoring: candle body (40%), EMA slope (30%), MACD momentum (30%)
    - NEW: Per-pair modes: 'active' (filter), 'monitor' (log only), 'disabled'
    - NEW: Confidence multiplier: bias score adjusts confidence (0.7x-1.3x)
    - CONFIG: htf_bias_enabled (default: true) - master toggle
    - CONFIG: htf_bias_min_threshold (default: 0.400) - minimum score for active mode
    - CONFIG: htf_bias_confidence_multiplier_enabled (default: true)
    - DATABASE: htf_bias_mode per pair (EURUSD/USDCAD/USDJPY = monitor, others = active)
    - ALERT: Added htf_bias_score, htf_bias_mode, htf_bias_details to signal/alert_history
    - SCORE INTERPRETATION:
        0.0-0.3: Strong counter-trend (significantly misaligned)
        0.3-0.5: Weak counter-trend (slightly misaligned)
        0.5-0.7: Neutral (no strong bias)
        0.7-0.9: Aligned (favorable trend)
        0.9-1.0: Strong alignment (optimal conditions)
    - DEPRECATED: scalp_reversal_* parameters (reversal override system removed)
    - IMPACT: Reduces 8 parameters to 2, cleaner code, better signal quality measurement

v2.30.0 CHANGES (Per-Pair ATR-Optimized Scalp SL/TP):
    - NEW: Per-pair scalp SL/TP values based on historical ATR analysis
    - ANALYSIS: Calculated optimal SL = ATR × 1.8, TP = SL × 2.5 for each pair
    - DATABASE: Populated smc_simple_pair_overrides.scalp_sl_pips and scalp_tp_pips
    - RANGE: SL 5.4-15.5 pips, TP 13.5-38.8 pips (pair-specific, not one-size-fits-all)
    - EXAMPLES: EURUSD SL=10.2 TP=25.5, USDJPY SL=15.2 TP=38.0, AUDUSD SL=5.4 TP=13.5
    - CONFIG SERVICE: Added get_pair_scalp_sl() and get_pair_scalp_tp() methods
    - STRATEGY: Checks per-pair scalp SL/TP first, falls back to global if not set
    - IMPACT: Each pair now uses SL/TP optimized for its volatility characteristics
    - R:R: Consistent 1:2.5 risk:reward ratio across all pairs

v2.29.0 CHANGES (Disable Confidence Cap in Scalp Mode):
    - NEW: Confidence cap check is skipped when scalp_mode_enabled=True
    - REASONING: High confidence in scalp mode indicates strong momentum confirmation
    - REASONING: The confidence paradox (high confidence = worse) was observed in regular trading, not scalp
    - IMPACT: Allows high confidence scalp trades to execute without artificial cap
    - CHANGE: Modified line 2735 to skip cap check when self.scalp_mode_enabled

v2.28.0 CHANGES (Scalp Reversal Override - Counter-Trend Opportunity):
    - NEW: When HTF alignment fails, check for strict reversal conditions instead of rejecting
    - NEW: scalp_reversal_enabled - allow counter-trend entries with strict gating
    - NEW: Requires runway >= 15 pips (distance to swing level for profit room)
    - NEW: Requires entry momentum >= 0.60 OR RSI in extreme zone (overbought/oversold)
    - NEW: Blocks reversal in breakout regimes and high volatility states
    - NEW: entry_type = 'REVERSAL' for counter-trend scalp trades
    - IMPACT: Captures reversal opportunities when HTF shows temporary misalignment
    - CONFIG: scalp_reversal_enabled (default: true)
    - CONFIG: scalp_reversal_min_runway_pips (default: 15.0)
    - CONFIG: scalp_reversal_min_entry_momentum (default: 0.60)
    - CONFIG: scalp_reversal_block_regimes (default: ['breakout'])
    - CONFIG: scalp_reversal_block_volatility_states (default: ['high'])
    - CONFIG: scalp_reversal_allow_rsi_extremes (default: true)

v2.24.0 CHANGES (Alternative TIER 3 Entries - Pattern & Divergence):
    - NEW: pattern_as_entry_enabled - allow patterns as standalone TIER 3 entry
    - NEW: divergence_as_entry_enabled - allow RSI divergence as standalone TIER 3 entry
    - NEW: When pullback zone fails, check for pattern/divergence as alternative entry
    - NEW: Pattern entry requires strength >= 80% (configurable via pattern_entry_min_strength)
    - NEW: Divergence entry requires strength >= 50% (configurable via divergence_entry_min_strength)
    - NEW: entry_type now supports 'PATTERN' and 'DIVERGENCE' in addition to PULLBACK/MOMENTUM
    - NEW: Scalp filter allows PATTERN and DIVERGENCE entries alongside MOMENTUM
    - IMPACT: More entry opportunities when HTF structure is aligned but pullback zone fails
    - CONFIG: pattern_as_entry_enabled (default: false), divergence_as_entry_enabled (default: false)

v2.20.0 CHANGES (Scalp Mode - High Frequency Trading):
    - NEW: scalp_mode_enabled toggle for high-frequency 5 pip TP trading
    - NEW: Faster timeframes: 1H/5m/1m instead of 4H/15m/5m
    - NEW: Spread filter blocks entries when spread > 1 pip (critical for scalp profitability)
    - NEW: Relaxed filters: Disable EMA slope, swing proximity, volume in scalp mode
    - NEW: 15-minute cooldown instead of 3 hours
    - NEW: Lower confidence threshold (30% vs 48%) for more entries
    - CONFIG: scalp_mode_enabled (default: false), scalp_tp_pips (5), scalp_sl_pips (5)
    - CONFIG: scalp_max_spread_pips (1.0), scalp_require_tight_spread (true)
    - CONFIG: All scalp settings stored in database smc_simple_global_config table
    - Expected improvement: 10-20+ signals/day vs current 0.07 signals/day

v2.19.0 CHANGES (Status-Based Cooldowns - Expiry Optimization):
    - NEW: Order status tracking (pending, placed, filled, expired, rejected)
    - NEW: Status-based cooldowns: filled=4h, expired=30min, rejected=15min
    - NEW: Consecutive expiry detection blocks after 3+ expiries (spam prevention)
    - IMPACT: Expired limit orders no longer waste 4h cooldown
    - IMPACT: Quicker retry after failed orders
    - Expected improvement: +50% more trade opportunities per day

v2.18.0 CHANGES (Extension Filter - Entry Timing Improvement):
    - NEW: ATR-based extension filter prevents chasing extended moves
    - ANALYSIS: Trades with >15% extension showed 33-38% win rate vs 53% for pullbacks
    - ROOT CAUSE: Momentum entries allowed up to 20% extension, often exhausted
    - FIX: Reject momentum entries beyond 0.5 ATR from swing break point
    - FIX: Reject momentum entries if swing break happened >8 bars ago (staleness)
    - CONFIG: max_extension_atr (default: 0.50 = ~10-15 pips)
    - CONFIG: max_extension_atr_enabled (default: true)
    - CONFIG: momentum_staleness_enabled (default: true)
    - CONFIG: max_momentum_staleness_bars (default: 8 = ~2 hours on 15m)
    - FIX: EURUSD BEAR had momentum_min_depth=-1.0 (100% extension allowed!) - now -0.10
    - FIX: Global momentum_min_depth tightened from -0.20 to -0.15
    - Expected improvement: Win rate 37% → 50%+ on momentum entries

v2.16.0 CHANGES (EMA Slope Validation - Counter-Trend Prevention):
    - NEW: ATR-based EMA slope validation prevents counter-trend trades
    - ANALYSIS: GBPUSD backtest showed 78% of signals were wrong-direction BULL during downtrend
    - ROOT CAUSE: Strategy only checked if price > EMA, not if EMA was rising or falling
    - FIX: Price above FALLING EMA = bearish retest (rejected as BULL signal)
    - FIX: Price below RISING EMA = bullish retest (rejected as BEAR signal)
    - CONFIG: ema_slope_validation_enabled (default: true)
    - CONFIG: ema_slope_lookback_bars (default: 5 = 20 hours on 4H)
    - CONFIG: ema_slope_min_atr_multiplier (default: 0.5 = EMA must move 0.5x ATR)
    - Expected improvement: Win rate 22% → 48%+ on trending pairs

v2.15.0 CHANGES (Swing Proximity Validation):
    - NEW: TIER 4 - Swing Proximity Validation prevents entries too close to swing levels
    - FIX: Trade log analysis showed 65% of losing trades were at wrong swing levels
    - BUY signals now require minimum distance from swing HIGH (resistance)
    - SELL signals now require minimum distance from swing LOW (support)
    - Default: 12 pips minimum distance (configurable via database)
    - Expected improvement: Win rate 20% → 50%+ based on filtered trade analysis
    - FIX: Corrected log message for swing break direction (was inverted)

v2.14.1 CHANGES (Cooldown Persistence Fix):
    - FIX: Database-loaded cooldowns were keyed by epic but checked by pair name
    - BUG: Cooldowns from DB were never enforced due to key mismatch
    - SOLUTION: Extract pair name from epic when loading from database
    - IMPACT: Cooldowns now properly persist across container restarts

v2.14.0 CHANGES (Volume Confirmation Fix):
    - FIX: volume_confirmed was using incomplete candle when break happened on current candle
    - Same issue as volume_ratio fixed in v2.13.0 - incomplete candles have artificially low volume
    - SOLUTION: If break_candle_idx == current_idx, use previous complete candle for volume check
    - IMPACT: More accurate volume confirmation, fewer false negatives

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

# HTF Bias Score Calculator (v2.35.0)
from forex_scanner.core.strategies.helpers.htf_bias_calculator import (
    get_htf_bias_calculator, HTFBiasCalculator
)


# ============================================================================
# TRIGGER TYPE CONSTANTS (v2.23.0)
# Used to track signal entry mechanisms for performance analysis
# ============================================================================

# Existing swing-based triggers (baseline)
TRIGGER_SWING_PULLBACK = 'SWING_PULLBACK'     # Swing break + Fib pullback (23.6%-70%)
TRIGGER_SWING_MOMENTUM = 'SWING_MOMENTUM'     # Swing break + momentum continuation (-20% to 0%)
TRIGGER_SWING_OPTIMAL = 'SWING_OPTIMAL'       # Swing break + optimal Fib zone (38.2%-61.8%)

# Price action pattern triggers
TRIGGER_PIN_BAR = 'PIN_BAR'                   # Pin bar rejection pattern
TRIGGER_ENGULFING = 'ENGULFING'               # Engulfing pattern
TRIGGER_INSIDE_BAR = 'INSIDE_BAR'             # Inside bar breakout
TRIGGER_HAMMER = 'HAMMER'                     # Hammer/shooting star

# Pattern suffixes for combined triggers (e.g., SWING_PULLBACK+PIN)
PATTERN_SUFFIX_PIN = '+PIN'
PATTERN_SUFFIX_ENG = '+ENG'
PATTERN_SUFFIX_INS = '+INS'
PATTERN_SUFFIX_DIV = '+DIV'


class SMCSimpleStrategy:
    """
    Simplified SMC strategy using 3-tier EMA-based approach

    Entry Requirements:
    1. TIER 1: Price on correct side of 4H 50 EMA (trend bias)
    2. TIER 2: 1H candle BODY closes beyond swing high/low (momentum)
    3. TIER 3: 15m price pulls back to Fibonacci zone (entry timing)
    """

    def __init__(self, config, logger=None, db_manager=None, config_override: dict = None):
        """Initialize SMC Simple Strategy

        Args:
            config: Main config module
            logger: Logger instance (optional)
            db_manager: DatabaseManager for rejection tracking (optional)
            config_override: Dict of parameter overrides for backtesting (optional)
                            e.g., {'fixed_stop_loss_pips': 12, 'min_confidence': 0.55}
                            When None = LIVE TRADING (unchanged behavior)
                            When dict = BACKTEST MODE (overrides applied after base config)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._db_manager = db_manager  # Store for later initialization

        # CRITICAL: Backtest mode detection for parameter isolation
        self._backtest_mode = config_override is not None
        self._config_override = config_override

        # Load configuration (sets self.rejection_tracking_enabled)
        self._load_config()

        # State tracking
        self.pair_cooldowns = {}  # {pair: last_signal_time} - ONLY used in backtest mode
        self.recent_signals = {}  # {pair: [(timestamp, price), ...]}
        self.pending_entries = {}  # {pair: pending_entry_data}

        # v3.0.0: Adaptive cooldown state tracking
        self.pair_consecutive_losses = {}  # {pair: int} - consecutive loss count
        self.pair_last_session = {}  # {pair: session_name} - track session changes
        self._trade_outcome_cache = {}  # {pair: {'outcome': dict, 'cached_at': datetime}}

        # v3.2.0: Cooldowns now read directly from alert_history database (single source of truth)
        # No startup load needed - each cooldown check queries the database directly
        # This eliminates state sync issues and simplifies the code

        # Rejection tracking (v2.2.0) - initialized after config is loaded
        # DISABLED in backtest mode to avoid polluting DB and improve performance
        self.rejection_manager = None
        if self._db_manager is not None and self.rejection_tracking_enabled and not self._backtest_mode:
            try:
                from forex_scanner.alerts.smc_rejection_history import SMCRejectionHistoryManager
                self.rejection_manager = SMCRejectionHistoryManager(db_manager)
                self.logger.info("   Rejection tracking: ENABLED")
            except Exception as e:
                self.logger.warning(f"   Rejection tracking: DISABLED (failed to init: {e})")
        elif self._backtest_mode:
            self.logger.info("   Rejection tracking: DISABLED (backtest mode)")

        # v2.31.1: In-memory filter stats for backtest summary (Jan 2026)
        # Tracks filter rejections during backtest to show which filters are most aggressive
        self._filter_stats = {
            'signals_detected': 0,
            'signals_passed': 0,
            'filter_rejections': {}  # {filter_name: count}
        }

        # v2.15.0: Swing Proximity Validator initialization
        self.swing_proximity_validator = None
        if self.swing_proximity_enabled:
            try:
                from forex_scanner.core.strategies.helpers.swing_proximity_validator import SwingProximityValidator
                self.swing_proximity_validator = SwingProximityValidator(
                    config={
                        'enabled': self.swing_proximity_enabled,
                        'min_distance_pips': self.swing_proximity_min_distance_pips,
                        'strict_mode': self.swing_proximity_strict_mode,
                        'resistance_buffer': self.swing_proximity_resistance_buffer,
                        'support_buffer': self.swing_proximity_support_buffer,
                        'lookback_swings': self.swing_proximity_lookback_swings,
                    },
                    logger=self.logger
                )
                self.logger.info(f"   Swing Proximity: ENABLED (min distance: {self.swing_proximity_min_distance_pips} pips)")
            except Exception as e:
                self.logger.warning(f"   Swing Proximity: DISABLED (failed to init: {e})")
                self.swing_proximity_enabled = False
        else:
            self.logger.info("   Swing Proximity: DISABLED (config)")

        # v2.21.0: Scalp Signal Qualifier initialization
        self._signal_qualifier = None
        if self.scalp_mode_enabled:
            try:
                from forex_scanner.core.strategies.scalp_signal_qualifier import ScalpSignalQualifier
                self._signal_qualifier = ScalpSignalQualifier(
                    config=self.config,
                    logger=self.logger,
                    db_config=self._db_config if hasattr(self, '_db_config') else None
                )
                # Apply backtest overrides to qualifier if provided
                if self._config_override:
                    if 'scalp_qualification_enabled' in self._config_override:
                        self._signal_qualifier.enabled = self._config_override['scalp_qualification_enabled']
                    if 'scalp_qualification_mode' in self._config_override:
                        self._signal_qualifier.mode = self._config_override['scalp_qualification_mode']
                    if 'scalp_min_qualification_score' in self._config_override:
                        self._signal_qualifier.min_score = self._config_override['scalp_min_qualification_score']
                    if 'scalp_rsi_filter_enabled' in self._config_override:
                        self._signal_qualifier.rsi_filter_enabled = self._config_override['scalp_rsi_filter_enabled']
                    if 'scalp_two_pole_filter_enabled' in self._config_override:
                        self._signal_qualifier.two_pole_filter_enabled = self._config_override['scalp_two_pole_filter_enabled']
                    if 'scalp_macd_filter_enabled' in self._config_override:
                        self._signal_qualifier.macd_filter_enabled = self._config_override['scalp_macd_filter_enabled']
                    # Micro-regime filter overrides (v2.21.1)
                    if 'scalp_micro_regime_enabled' in self._config_override:
                        self._signal_qualifier.micro_regime_enabled = self._config_override['scalp_micro_regime_enabled']
                    if 'scalp_consecutive_candles_enabled' in self._config_override:
                        self._signal_qualifier.consecutive_candles_enabled = self._config_override['scalp_consecutive_candles_enabled']
                    if 'scalp_anti_chop_enabled' in self._config_override:
                        self._signal_qualifier.anti_chop_enabled = self._config_override['scalp_anti_chop_enabled']
                    if 'scalp_body_dominance_enabled' in self._config_override:
                        self._signal_qualifier.body_dominance_enabled = self._config_override['scalp_body_dominance_enabled']
                    if 'scalp_micro_range_enabled' in self._config_override:
                        self._signal_qualifier.micro_range_enabled = self._config_override['scalp_micro_range_enabled']
                    if 'scalp_momentum_candle_enabled' in self._config_override:
                        self._signal_qualifier.momentum_candle_enabled = self._config_override['scalp_momentum_candle_enabled']
                    # Micro-regime threshold overrides
                    if 'scalp_consecutive_candles_min' in self._config_override:
                        self._signal_qualifier.consecutive_candles_min = self._config_override['scalp_consecutive_candles_min']
                    if 'scalp_anti_chop_lookback' in self._config_override:
                        self._signal_qualifier.anti_chop_lookback = self._config_override['scalp_anti_chop_lookback']
                    if 'scalp_anti_chop_max_alternations' in self._config_override:
                        self._signal_qualifier.anti_chop_max_alternations = self._config_override['scalp_anti_chop_max_alternations']
                    if 'scalp_body_dominance_ratio' in self._config_override:
                        self._signal_qualifier.body_dominance_ratio = self._config_override['scalp_body_dominance_ratio']
                    if 'scalp_micro_range_min_pips' in self._config_override:
                        self._signal_qualifier.micro_range_min_pips = self._config_override['scalp_micro_range_min_pips']
                    if 'scalp_momentum_candle_multiplier' in self._config_override:
                        self._signal_qualifier.momentum_candle_multiplier = self._config_override['scalp_momentum_candle_multiplier']
                micro_info = f", micro-regime={self._signal_qualifier.micro_regime_enabled}" if self._signal_qualifier.micro_regime_enabled else ""
                self.logger.info(f"   Signal Qualifier: {self._signal_qualifier.mode} mode (enabled={self._signal_qualifier.enabled}{micro_info})")
            except Exception as e:
                self.logger.warning(f"   Signal Qualifier: DISABLED (failed to init: {e})")

        self.logger.info("=" * 60)
        self.logger.info("✅ SMC Simple Strategy v2.21.0 initialized")
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

            # v2.16.0: EMA Slope Validation (prevents counter-trend trades)
            self.ema_slope_validation_enabled = config.ema_slope_validation_enabled
            self.ema_slope_lookback_bars = config.ema_slope_lookback_bars
            self.ema_slope_min_atr_multiplier = config.ema_slope_min_atr_multiplier

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

            # v2.18.0: ATR-based Extension Filter (prevents chasing extended moves)
            self.max_extension_atr = getattr(config, 'max_extension_atr', 0.50)
            self.max_extension_atr_enabled = getattr(config, 'max_extension_atr_enabled', True)

            # v2.18.0: Momentum Staleness Filter (rejects old swing breaks)
            self.momentum_staleness_enabled = getattr(config, 'momentum_staleness_enabled', True)
            self.max_momentum_staleness_bars = getattr(config, 'max_momentum_staleness_bars', 8)

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

            # v2.15.0: Fixed SL/TP override (per-pair configurable)
            self.fixed_sl_tp_override_enabled = config.fixed_sl_tp_override_enabled
            self.fixed_stop_loss_pips = config.fixed_stop_loss_pips
            self.fixed_take_profit_pips = config.fixed_take_profit_pips

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

            # v2.15.0: Swing Proximity Validation
            self.swing_proximity_enabled = config.swing_proximity_enabled
            self.swing_proximity_min_distance_pips = config.swing_proximity_min_distance_pips
            self.swing_proximity_strict_mode = config.swing_proximity_strict_mode
            self.swing_proximity_resistance_buffer = config.swing_proximity_resistance_buffer
            self.swing_proximity_support_buffer = config.swing_proximity_support_buffer
            self.swing_proximity_lookback_swings = config.swing_proximity_lookback_swings

            # v2.23.0: Pattern Confirmation (price action patterns for signal enhancement)
            self.pattern_confirmation_enabled = getattr(config, 'pattern_confirmation_enabled', False)
            self.pattern_confirmation_mode = getattr(config, 'pattern_confirmation_mode', 'MONITORING')
            self.pattern_min_strength = getattr(config, 'pattern_min_strength', 0.70)
            self.pattern_confidence_boost = getattr(config, 'pattern_confidence_boost', 0.05)
            self.pattern_pin_bar_enabled = getattr(config, 'pattern_pin_bar_enabled', True)
            self.pattern_engulfing_enabled = getattr(config, 'pattern_engulfing_enabled', True)
            self.pattern_hammer_shooter_enabled = getattr(config, 'pattern_hammer_shooter_enabled', True)
            self.pattern_inside_bar_enabled = getattr(config, 'pattern_inside_bar_enabled', True)
            # v2.24.0: Pattern as alternative TIER 3 entry (not just enhancement)
            self.pattern_as_entry_enabled = getattr(config, 'pattern_as_entry_enabled', False)
            self.pattern_entry_min_strength = getattr(config, 'pattern_entry_min_strength', 0.80)

            # v2.23.0: RSI Divergence Detection
            self.rsi_divergence_enabled = getattr(config, 'rsi_divergence_enabled', False)
            self.rsi_divergence_mode = getattr(config, 'rsi_divergence_mode', 'MONITORING')
            self.rsi_divergence_lookback = getattr(config, 'rsi_divergence_lookback', 20)
            self.rsi_divergence_min_strength = getattr(config, 'rsi_divergence_min_strength', 0.30)
            self.rsi_divergence_confidence_boost = getattr(config, 'rsi_divergence_confidence_boost', 0.08)
            # v2.24.0: RSI Divergence as alternative TIER 3 entry (not just enhancement)
            self.divergence_as_entry_enabled = getattr(config, 'divergence_as_entry_enabled', False)
            self.divergence_entry_min_strength = getattr(config, 'divergence_entry_min_strength', 0.50)

            self.logger.info(f"✅ SMC Simple config v{config.version} loaded from DATABASE (source: {config.source})")

            # SCALP MODE: Load scalp configuration first (before overrides)
            self._load_scalp_mode_config(config)

            # Apply backtest overrides if in backtest mode
            # IMPORTANT: This must happen AFTER _load_scalp_mode_config() so that
            # CLI overrides (--scalp-ema, --scalp-swing-lookback, etc.) take precedence
            # over database values
            if self._backtest_mode:
                self._apply_config_overrides()

            # Apply scalp mode configuration if enabled
            if self.scalp_mode_enabled:
                self._configure_scalp_mode()

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

        # v2.16.0: EMA Slope Validation (prevents counter-trend trades)
        self.ema_slope_validation_enabled = getattr(smc_config, 'EMA_SLOPE_VALIDATION_ENABLED', True)
        self.ema_slope_lookback_bars = getattr(smc_config, 'EMA_SLOPE_LOOKBACK_BARS', 5)
        self.ema_slope_min_atr_multiplier = getattr(smc_config, 'EMA_SLOPE_MIN_ATR_MULTIPLIER', 0.5)

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

        # v2.18.0: ATR-based Extension Filter (prevents chasing extended moves)
        self.max_extension_atr = getattr(smc_config, 'MAX_EXTENSION_ATR', 0.50)
        self.max_extension_atr_enabled = getattr(smc_config, 'MAX_EXTENSION_ATR_ENABLED', True)

        # v2.18.0: Momentum Staleness Filter (rejects old swing breaks)
        self.momentum_staleness_enabled = getattr(smc_config, 'MOMENTUM_STALENESS_ENABLED', True)
        self.max_momentum_staleness_bars = getattr(smc_config, 'MAX_MOMENTUM_STALENESS_BARS', 8)

        # v1.9.0: Pair-specific SL buffers and confidence floors
        self.pair_sl_buffers = getattr(smc_config, 'PAIR_SL_BUFFERS', {})
        self.pair_min_confidence = getattr(smc_config, 'PAIR_MIN_CONFIDENCE', {})
        self.sl_atr_multiplier = getattr(smc_config, 'SL_ATR_MULTIPLIER', 1.2)
        self.use_atr_stop = getattr(smc_config, 'USE_ATR_STOP', True)

        # v2.11.0: Volume-adjusted confidence thresholds (per-pair)
        self.volume_adjusted_confidence_enabled = getattr(smc_config, 'VOLUME_ADJUSTED_CONFIDENCE_ENABLED', True)
        self.high_volume_threshold = getattr(smc_config, 'HIGH_VOLUME_THRESHOLD', 0.70)
        self.high_volume_confidence = getattr(smc_config, 'HIGH_VOLUME_CONFIDENCE', 0.45)  # Global default for backtest overrides
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

        # v2.15.0: Fixed SL/TP override (per-pair configurable)
        self.fixed_sl_tp_override_enabled = getattr(smc_config, 'FIXED_SL_TP_OVERRIDE_ENABLED', False)
        self.fixed_stop_loss_pips = getattr(smc_config, 'FIXED_STOP_LOSS_PIPS', 9.0)
        self.fixed_take_profit_pips = getattr(smc_config, 'FIXED_TAKE_PROFIT_PIPS', 15.0)

        # v2.15.0: Swing Proximity Validation
        self.swing_proximity_enabled = getattr(smc_config, 'SWING_PROXIMITY_ENABLED', True)
        self.swing_proximity_min_distance_pips = getattr(smc_config, 'SWING_PROXIMITY_MIN_DISTANCE_PIPS', 12)
        self.swing_proximity_strict_mode = getattr(smc_config, 'SWING_PROXIMITY_STRICT_MODE', True)
        self.swing_proximity_resistance_buffer = getattr(smc_config, 'SWING_PROXIMITY_RESISTANCE_BUFFER', 1.0)
        self.swing_proximity_support_buffer = getattr(smc_config, 'SWING_PROXIMITY_SUPPORT_BUFFER', 1.0)
        self.swing_proximity_lookback_swings = getattr(smc_config, 'SWING_PROXIMITY_LOOKBACK_SWINGS', 5)

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

        # SCALP MODE: Disabled by default in file config (database-driven feature)
        self.scalp_mode_enabled = False
        self.scalp_tp_pips = 5.0
        self.scalp_sl_pips = 5.0
        self.scalp_max_spread_pips = 1.0
        self.scalp_require_tight_spread = True
        self.scalp_reversal_enabled = True
        self.scalp_reversal_min_runway_pips = 15.0
        self.scalp_reversal_min_entry_momentum = 0.60
        self.scalp_reversal_block_regimes = ['breakout']
        self.scalp_reversal_block_volatility_states = ['high']
        self.scalp_reversal_allow_rsi_extremes = True

        self.logger.info("✅ SMC Simple config loaded from FILE (fallback mode)")

        # Apply backtest overrides if in backtest mode
        if self._backtest_mode:
            self._apply_config_overrides()

    def _apply_config_overrides(self):
        """Apply in-memory overrides for backtesting.

        This method is called after base config is loaded (from DB or file).
        It applies parameter overrides provided via config_override dict,
        allowing backtests to test different parameters without affecting live trading.
        """
        if not self._config_override:
            return

        self.logger.info("=" * 60)
        self.logger.info("🧪 BACKTEST MODE: Applying parameter overrides")
        self.logger.info("=" * 60)

        # Comprehensive parameter mapping: override key -> instance attribute
        override_mapping = {
            # SL/TP Parameters
            'fixed_stop_loss_pips': 'fixed_stop_loss_pips',  # Fixed SL in pips
            'fixed_take_profit_pips': 'fixed_take_profit_pips',  # Fixed TP in pips
            'fixed_sl_tp_override_enabled': 'fixed_sl_tp_override_enabled',  # Enable fixed SL/TP
            'min_rr_ratio': 'min_rr_ratio',
            'optimal_rr_ratio': 'optimal_rr',
            'max_rr_ratio': 'max_rr_ratio',
            'sl_buffer_pips': 'sl_buffer_pips',
            'sl_atr_multiplier': 'sl_atr_multiplier',
            'max_sl_atr_multiplier': 'max_sl_atr_multiplier',
            'max_sl_absolute_pips': 'max_sl_absolute_pips',

            # Confidence Thresholds
            'min_confidence': 'min_confidence',
            'max_confidence': 'max_confidence',
            'min_volume_ratio': 'min_volume_ratio',

            # Fibonacci Zones (Entry Timing)
            'fib_min': 'fib_min',
            'fib_max': 'fib_max',
            'fib_pullback_min': 'fib_min',  # Alias
            'fib_pullback_max': 'fib_max',  # Alias

            # TIER 1: HTF Settings
            'ema_period': 'ema_period',
            'ema_buffer_pips': 'ema_buffer_pips',
            'min_distance_from_ema_pips': 'min_distance_from_ema',
            'require_close_beyond_ema': 'require_close_beyond_ema',
            'htf_timeframe': 'htf_timeframe',

            # TIER 2: Trigger Settings
            'swing_lookback': 'swing_lookback',
            'swing_lookback_bars': 'swing_lookback',  # Alias
            'swing_strength': 'swing_strength',
            'swing_strength_bars': 'swing_strength',  # Alias
            'trigger_tf': 'trigger_tf',
            'trigger_timeframe': 'trigger_tf',  # Alias

            # TIER 3: Entry Settings
            'entry_tf': 'entry_tf',
            'entry_timeframe': 'entry_tf',  # Alias
            'momentum_min_depth': 'momentum_min_depth',
            'momentum_max_depth': 'momentum_max_depth',
            'max_pullback_wait_bars': 'max_pullback_wait',
            'pullback_confirmation_bars': 'pullback_confirm_bars',

            # Volume Settings
            'volume_confirmation_enabled': 'volume_enabled',
            'volume_enabled': 'volume_enabled',  # Alias
            'volume_sma_period': 'volume_sma_period',
            'volume_spike_multiplier': 'volume_multiplier',
            'volume_multiplier': 'volume_multiplier',  # Alias
            'volume_filter_enabled': 'volume_filter_enabled',
            'min_volume_ratio': 'min_volume_ratio',
            'allow_no_volume_data': 'allow_no_volume_data',
            'volume_adjusted_confidence_enabled': 'volume_adjusted_confidence_enabled',
            'high_volume_threshold': 'high_volume_threshold',

            # Session/Filter Settings
            'session_filter_enabled': 'session_filter_enabled',
            'block_asian_session': 'block_asian',
            'allow_asian_session': 'block_asian',  # Inverted logic handled below

            # MACD Filter
            'macd_filter_enabled': 'macd_alignment_filter_enabled',  # Alias for optimization
            'macd_alignment_filter_enabled': 'macd_alignment_filter_enabled',

            # Cooldown Settings
            'signal_cooldown_hours': 'cooldown_hours',
            'cooldown_minutes': 'cooldown_hours',  # Will be converted
            'adaptive_cooldown_enabled': 'adaptive_cooldown_enabled',
            'base_cooldown_hours': 'base_cooldown_hours',

            # Momentum Quality
            'momentum_mode_enabled': 'momentum_mode_enabled',
            'momentum_quality_enabled': 'momentum_quality_enabled',
            'min_breakout_atr_ratio': 'min_breakout_atr_ratio',
            'min_body_percentage': 'min_body_percentage',

            # ATR Settings
            'atr_period': 'atr_period',
            'use_atr_stop': 'use_atr_stop',
            'use_atr_swing_validation': 'use_atr_swing_validation',
            'min_swing_atr_multiplier': 'min_swing_atr_multiplier',

            # Swing Proximity Settings (CRITICAL: These were missing - added Jan 2026)
            'swing_proximity_enabled': 'swing_proximity_enabled',
            'swing_proximity_min_distance_pips': 'swing_proximity_min_distance_pips',
            'swing_proximity_strict_mode': 'swing_proximity_strict_mode',
            'swing_proximity_resistance_buffer': 'swing_proximity_resistance_buffer',
            'swing_proximity_support_buffer': 'swing_proximity_support_buffer',
            'swing_proximity_lookback_swings': 'swing_proximity_lookback_swings',

            # Per-Pair Override Parameters (from SMC Config Tab)
            'smc_conflict_tolerance': 'smc_conflict_tolerance',
            'high_volume_confidence': 'high_volume_confidence',

            # Limit Order Settings
            'limit_order_enabled': 'limit_order_enabled',
            'limit_expiry_minutes': 'limit_expiry_minutes',
            'momentum_offset_pips': 'momentum_offset_pips',
            'scalp_limit_offset_pips': 'scalp_limit_offset_pips',  # Scalp mode offset override
            'pullback_offset_min_pips': 'pullback_offset_min_pips',
            'pullback_offset_max_pips': 'pullback_offset_max_pips',

            # Scalp Mode Master Toggle and Tier Settings (for parameter testing)
            'scalp_mode_enabled': 'scalp_mode_enabled',  # CRITICAL: Master toggle must be in mapping!
            'scalp_htf_timeframe': 'scalp_htf_timeframe',
            'scalp_trigger_timeframe': 'scalp_trigger_timeframe',
            'scalp_entry_timeframe': 'scalp_entry_timeframe',
            'scalp_ema_period': 'scalp_ema_period',
            'scalp_swing_lookback_bars': 'scalp_swing_lookback_bars',
            'scalp_swing_break_tolerance_pips': 'scalp_swing_break_tolerance_pips',
            'scalp_tp_pips': 'scalp_tp_pips',
            'scalp_sl_pips': 'scalp_sl_pips',
            'scalp_min_confidence': 'scalp_min_confidence',
            'scalp_cooldown_minutes': 'scalp_cooldown_minutes',

            # v2.22.0: Scalp Entry Filters (for backtest parameter testing)
            'scalp_momentum_only_filter': 'scalp_momentum_only_filter',
            'scalp_require_htf_alignment': 'scalp_require_htf_alignment',
            'scalp_entry_rsi_buy_max': 'scalp_entry_rsi_buy_max',
            'scalp_entry_rsi_sell_min': 'scalp_entry_rsi_sell_min',
            'scalp_min_ema_distance_pips': 'scalp_min_ema_distance_pips',

            # Scalp filter disable toggles (control which filters scalp mode disables)
            'scalp_disable_swing_proximity': 'scalp_disable_swing_proximity',
            'scalp_disable_ema_slope_validation': 'scalp_disable_ema_slope_validation',
            'scalp_disable_volume_filter': 'scalp_disable_volume_filter',
            'scalp_disable_macd_filter': 'scalp_disable_macd_filter',

            # Scalp reversal override (counter-trend)
            'scalp_reversal_enabled': 'scalp_reversal_enabled',
            'scalp_reversal_min_runway_pips': 'scalp_reversal_min_runway_pips',
            'scalp_reversal_min_entry_momentum': 'scalp_reversal_min_entry_momentum',
            'scalp_reversal_block_regimes': 'scalp_reversal_block_regimes',
            'scalp_reversal_block_volatility_states': 'scalp_reversal_block_volatility_states',
            'scalp_reversal_allow_rsi_extremes': 'scalp_reversal_allow_rsi_extremes',

            # v2.25.0: Scalp Rejection Candle Confirmation
            'scalp_require_rejection_candle': 'scalp_require_rejection_candle',
            'scalp_rejection_min_strength': 'scalp_rejection_min_strength',
            'scalp_use_market_on_rejection': 'scalp_use_market_on_rejection',

            # v2.25.1: Scalp Entry Candle Alignment (simpler alternative to rejection candle)
            'scalp_require_entry_candle_alignment': 'scalp_require_entry_candle_alignment',
            'scalp_use_market_on_entry_alignment': 'scalp_use_market_on_entry_alignment',

            # v2.35.0: HTF Bias Score System (replaces binary HTF alignment)
            'htf_bias_enabled': 'htf_bias_enabled',
            'htf_bias_min_threshold': 'htf_bias_min_threshold',
            'htf_bias_confidence_multiplier_enabled': 'htf_bias_confidence_multiplier_enabled',

            # v2.23.0: Pattern Confirmation (test ACTIVE vs MONITORING mode)
            'pattern_confirmation_enabled': 'pattern_confirmation_enabled',
            'pattern_confirmation_mode': 'pattern_confirmation_mode',
            'pattern_min_strength': 'pattern_min_strength',
            'pattern_confidence_boost': 'pattern_confidence_boost',
            'pattern_pin_bar_enabled': 'pattern_pin_bar_enabled',
            'pattern_engulfing_enabled': 'pattern_engulfing_enabled',
            'pattern_hammer_shooter_enabled': 'pattern_hammer_shooter_enabled',
            'pattern_inside_bar_enabled': 'pattern_inside_bar_enabled',
            # v2.24.0: Pattern as alternative entry
            'pattern_as_entry_enabled': 'pattern_as_entry_enabled',
            'pattern_entry_min_strength': 'pattern_entry_min_strength',

            # v2.23.0: RSI Divergence Detection (test ACTIVE vs MONITORING mode)
            'rsi_divergence_enabled': 'rsi_divergence_enabled',
            'rsi_divergence_mode': 'rsi_divergence_mode',
            'rsi_divergence_lookback': 'rsi_divergence_lookback',
            'rsi_divergence_min_strength': 'rsi_divergence_min_strength',
            'rsi_divergence_confidence_boost': 'rsi_divergence_confidence_boost',
            # v2.24.0: Divergence as alternative entry
            'divergence_as_entry_enabled': 'divergence_as_entry_enabled',
            'divergence_entry_min_strength': 'divergence_entry_min_strength',

            # Debug
            'enable_debug_logging': 'debug_logging',
        }

        overrides_applied = 0
        for param, attr_name in override_mapping.items():
            if param in self._config_override:
                value = self._config_override[param]

                # Special handling for inverted logic
                if param == 'allow_asian_session':
                    value = not value  # Invert for block_asian

                # Special handling for cooldown_minutes -> cooldown_hours conversion
                if param == 'cooldown_minutes':
                    value = value / 60.0  # Convert minutes to hours

                old_value = getattr(self, attr_name, None)
                setattr(self, attr_name, value)
                self.logger.info(f"   [OVERRIDE] {param}: {old_value} → {value}")
                overrides_applied += 1

        # CRITICAL (Jan 2026): Disable dynamic features when testing their base parameters
        # Otherwise the dynamic calculation will override the test value
        if 'swing_lookback_bars' in self._config_override or 'swing_lookback' in self._config_override:
            self.use_dynamic_swing_lookback = False
            self.logger.info(f"   [AUTO-DISABLE] use_dynamic_swing_lookback: True → False (testing swing_lookback)")

        if 'min_confidence' in self._config_override:
            # Disable all dynamic confidence adjustments to test the raw threshold
            self.ema_distance_adjusted_confidence_enabled = False
            self.atr_adjusted_confidence_enabled = False
            self.volume_adjusted_confidence_enabled = False
            self.logger.info(f"   [AUTO-DISABLE] Dynamic confidence adjustments disabled (testing min_confidence)")

        if 'fixed_stop_loss_pips' in self._config_override or 'fixed_take_profit_pips' in self._config_override:
            # Auto-enable fixed SL/TP when testing these parameters
            self.fixed_sl_tp_override_enabled = True
            self.logger.info(f"   [AUTO-ENABLE] fixed_sl_tp_override_enabled: True (testing fixed SL/TP)")

        # CRITICAL (Jan 2026): If scalp_mode_enabled was set via override, trigger scalp mode configuration
        # This ensures scalp mode works in backtest parameter variation testing
        if 'scalp_mode_enabled' in self._config_override and self.scalp_mode_enabled:
            self.logger.info(f"   [AUTO-CONFIGURE] Scalp mode enabled via override - applying scalp tier settings")
            self._configure_scalp_mode()

        self.logger.info(f"   Total overrides applied: {overrides_applied}")
        self.logger.info("=" * 60)

    def _load_scalp_mode_config(self, config):
        """Load scalp mode configuration from database config object.

        This method loads all scalp-related parameters from the config.
        The actual mode activation happens in _configure_scalp_mode() if enabled.
        """
        # Master toggle
        self.scalp_mode_enabled = getattr(config, 'scalp_mode_enabled', False)

        # Scalp SL/TP (1:1 R:R default)
        self.scalp_tp_pips = getattr(config, 'scalp_tp_pips', 5.0)
        self.scalp_sl_pips = getattr(config, 'scalp_sl_pips', 5.0)

        # Spread filter (critical for scalping)
        self.scalp_max_spread_pips = getattr(config, 'scalp_max_spread_pips', 1.0)
        self.scalp_require_tight_spread = getattr(config, 'scalp_require_tight_spread', True)

        # Scalp timeframes (faster than swing mode)
        self.scalp_htf_timeframe = getattr(config, 'scalp_htf_timeframe', '1h')
        self.scalp_trigger_timeframe = getattr(config, 'scalp_trigger_timeframe', '5m')
        self.scalp_entry_timeframe = getattr(config, 'scalp_entry_timeframe', '1m')

        # Scalp EMA settings
        self.scalp_ema_period = getattr(config, 'scalp_ema_period', 20)

        # Scalp confidence (lower to allow more entries)
        self.scalp_min_confidence = getattr(config, 'scalp_min_confidence', 0.30)

        # Scalp filter disables
        self.scalp_disable_ema_slope_validation = getattr(config, 'scalp_disable_ema_slope_validation', True)
        self.scalp_disable_swing_proximity = getattr(config, 'scalp_disable_swing_proximity', True)
        self.scalp_disable_volume_filter = getattr(config, 'scalp_disable_volume_filter', True)
        self.scalp_disable_macd_filter = getattr(config, 'scalp_disable_macd_filter', True)

        # Scalp entry logic
        self.scalp_use_momentum_only = getattr(config, 'scalp_use_momentum_only', True)
        self.scalp_momentum_min_depth = getattr(config, 'scalp_momentum_min_depth', -0.30)
        self.scalp_fib_pullback_min = getattr(config, 'scalp_fib_pullback_min', 0.0)
        self.scalp_fib_pullback_max = getattr(config, 'scalp_fib_pullback_max', 1.0)

        # Scalp reversal override (counter-trend) settings
        self.scalp_reversal_enabled = getattr(config, 'scalp_reversal_enabled', True)
        self.scalp_reversal_min_runway_pips = getattr(config, 'scalp_reversal_min_runway_pips', 15.0)
        self.scalp_reversal_min_entry_momentum = getattr(config, 'scalp_reversal_min_entry_momentum', 0.60)
        self.scalp_reversal_block_regimes = getattr(config, 'scalp_reversal_block_regimes', ['breakout'])
        self.scalp_reversal_block_volatility_states = getattr(config, 'scalp_reversal_block_volatility_states', ['high'])
        self.scalp_reversal_allow_rsi_extremes = getattr(config, 'scalp_reversal_allow_rsi_extremes', True)

        # Scalp cooldown (much shorter)
        self.scalp_cooldown_minutes = getattr(config, 'scalp_cooldown_minutes', 15)

        # Scalp swing detection (12 bars optimized default, 5 was too short)
        self.scalp_swing_lookback_bars = getattr(config, 'scalp_swing_lookback_bars', 12)
        self.scalp_range_position_threshold = getattr(config, 'scalp_range_position_threshold', 0.80)

        # Scalp market orders (faster fills, spread filter is the safeguard)
        self.scalp_use_market_orders = getattr(config, 'scalp_use_market_orders', True)

        # v2.22.0: Scalp entry filters (based on Jan 2026 trade analysis)
        # These filters block non-winning entry patterns
        self.scalp_momentum_only_filter = getattr(config, 'scalp_momentum_only_filter', False)
        self.scalp_require_htf_alignment = getattr(config, 'scalp_require_htf_alignment', False)
        self.scalp_entry_rsi_buy_max = getattr(config, 'scalp_entry_rsi_buy_max', 100.0)
        self.scalp_entry_rsi_sell_min = getattr(config, 'scalp_entry_rsi_sell_min', 0.0)
        self.scalp_min_ema_distance_pips = getattr(config, 'scalp_min_ema_distance_pips', 0.0)

        # v2.35.0: HTF Bias Score System (replaces binary HTF alignment)
        # Professional approach: continuous bias measurement instead of binary filter
        self.htf_bias_enabled = getattr(config, 'htf_bias_enabled', True)
        self.htf_bias_min_threshold = getattr(config, 'htf_bias_min_threshold', 0.400)
        self.htf_bias_confidence_multiplier_enabled = getattr(config, 'htf_bias_confidence_multiplier_enabled', True)
        # Initialize HTF bias calculator
        self._htf_bias_calculator = get_htf_bias_calculator(self.logger)

        # v2.25.0: Scalp rejection candle confirmation
        # Require entry-TF rejection candle before scalp entry
        self.scalp_require_rejection_candle = getattr(config, 'scalp_require_rejection_candle', False)
        self.scalp_rejection_min_strength = getattr(config, 'scalp_rejection_min_strength', 0.70)
        self.scalp_use_market_on_rejection = getattr(config, 'scalp_use_market_on_rejection', True)

        # v2.25.1: Scalp entry candle alignment (simpler alternative)
        # Require entry candle color matches direction (green=BUY, red=SELL)
        self.scalp_require_entry_candle_alignment = getattr(config, 'scalp_require_entry_candle_alignment', False)

    def _configure_scalp_mode(self):
        """Configure strategy for high-frequency scalping mode.

        When scalp_mode_enabled=True, this method overrides the standard parameters
        with scalp-optimized values for faster entries and 5 pip TP targets.

        Key changes:
        - Faster timeframes: 1H/5m/1m instead of 4H/15m/5m
        - Smaller targets: 5 pip TP/SL (1:1 R:R)
        - Relaxed filters: Disable EMA slope, volume, swing proximity
        - Spread gate: Only trade when spread < 1 pip
        - Shorter cooldown: 15 minutes instead of 3 hours

        Testing HTF alignment fix in isolation first (swing proximity disabled)
        """
        self.logger.info("=" * 60)
        self.logger.info("SCALP MODE ENABLED - High Frequency Configuration")
        self.logger.info("=" * 60)

        # Override timeframes for faster signals
        self.htf_timeframe = self.scalp_htf_timeframe
        self.trigger_tf = self.scalp_trigger_timeframe
        self.entry_tf = self.scalp_entry_timeframe

        # Override EMA for faster reaction
        self.ema_period = self.scalp_ema_period

        # Relax EMA buffer for scalp mode (smaller moves on faster timeframes)
        # Reduce buffer from 2.5 pips to 1.0 pip for more entries
        self.ema_buffer_pips = 1.0

        # v2.31.0: Override min EMA distance for scalp mode (use database-configured value)
        # Bug fix: Previously used global min_distance_from_ema=3.0 even in scalp mode,
        # causing backtest to reject signals that live scanner would accept
        self.min_distance_from_ema = self.scalp_min_ema_distance_pips

        # Override SL/TP for scalping (5 pip targets)
        self.fixed_stop_loss_pips = self.scalp_sl_pips
        self.fixed_take_profit_pips = self.scalp_tp_pips
        self.fixed_sl_tp_override_enabled = True

        # Lower confidence threshold for more entries
        self.min_confidence = self.scalp_min_confidence

        # Disable restrictive filters (including swing proximity)
        if self.scalp_disable_ema_slope_validation:
            self.ema_slope_validation_enabled = False

        # Re-enabled: Testing HTF alignment fix in isolation first
        if self.scalp_disable_swing_proximity:
            self.swing_proximity_enabled = False

        if self.scalp_disable_volume_filter:
            self.volume_filter_enabled = False
        if self.scalp_disable_macd_filter:
            # MACD filter is already controlled elsewhere
            pass

        # Widen Fib zones for scalp (any pullback valid)
        self.fib_min = self.scalp_fib_pullback_min
        self.fib_max = self.scalp_fib_pullback_max

        # Momentum settings for scalp
        self.momentum_min_depth = self.scalp_momentum_min_depth

        # Shorter cooldown (convert minutes to hours)
        self.cooldown_hours = self.scalp_cooldown_minutes / 60.0

        # Scalp swing detection settings
        # Use self.scalp_swing_lookback_bars which comes from:
        # 1. CLI override via --scalp-swing-lookback (highest priority)
        # 2. Database config (if no CLI override)
        # Default optimized value is 12 bars (60 min on 5m TF for better swing detection)
        # Note: Original default of 5 was too short, causing 70% of swing rejections
        self.swing_lookback = self.scalp_swing_lookback_bars
        self.use_dynamic_swing_lookback = False  # Use fixed lookback in scalp mode

        # Relax swing break tolerance for scalp mode (allow near-breaks)
        # Use overridden value if provided, otherwise default to 0.5 pips
        # Many rejections were within 0.5-1.5 pips of confirmation
        tolerance_override = getattr(self, 'scalp_swing_break_tolerance_pips', None)
        if tolerance_override is None:
            self.scalp_swing_break_tolerance_pips = 0.5

        # Disable dynamic confidence adjustments in scalp mode
        self.ema_distance_adjusted_confidence_enabled = False
        self.atr_adjusted_confidence_enabled = False
        self.volume_adjusted_confidence_enabled = False

        # Disable low-confidence extra offset in scalp mode (keep 1 pip offset)
        self.scalp_disable_low_confidence_offset = True

        # Use market orders for faster fills (spread filter is the safeguard)
        if self.scalp_use_market_orders:
            self.limit_order_enabled = False
            self.logger.info("   Order Type: MARKET (faster fills, spread-filtered)")
        else:
            # Limit orders with reduced offset for scalping
            # Check for scalp_limit_offset_pips override from backtest CLI (--scalp-offset)
            scalp_offset = getattr(self, 'scalp_limit_offset_pips', None)
            if scalp_offset is None:
                scalp_offset = 1.0  # Default scalp offset is 1 pip
            self.momentum_offset_pips = scalp_offset
            self.pullback_offset_min_pips = scalp_offset
            self.pullback_offset_max_pips = scalp_offset
            self.logger.info(f"   Order Type: LIMIT ({scalp_offset} pip offset for scalp)")

        # Log configuration
        self.logger.info(f"   HTF Timeframe: {self.htf_timeframe}")
        self.logger.info(f"   Trigger Timeframe: {self.trigger_tf}")
        self.logger.info(f"   Entry Timeframe: {self.entry_tf}")
        self.logger.info(f"   EMA Period: {self.ema_period}")
        self.logger.info(f"   Stop Loss: {self.fixed_stop_loss_pips} pips")
        self.logger.info(f"   Take Profit: {self.fixed_take_profit_pips} pips")
        self.logger.info(f"   Min Confidence: {self.min_confidence*100:.0f}%")
        self.logger.info(f"   Cooldown: {self.cooldown_hours*60:.0f} minutes")
        self.logger.info(f"   Max Spread: {self.scalp_max_spread_pips} pips")
        self.logger.info(f"   EMA Buffer: {self.ema_buffer_pips} pips (relaxed for scalp)")
        self.logger.info(f"   Swing Lookback: {self.swing_lookback} bars (increased for better detection)")
        self.logger.info(f"   Swing Break Tolerance: {self.scalp_swing_break_tolerance_pips} pips (allows near-breaks)")
        self.logger.info("   Filters DISABLED: EMA slope, Volume, Swing Proximity")
        self.logger.info("=" * 60)

    def _get_pair_scalp_config(self, epic: str) -> dict:
        """Get effective scalp configuration for a specific pair.

        Priority order (highest to lowest):
        1. Backtest CLI overrides (--scalp-ema, --scalp-swing-lookback, etc.)
        2. Per-pair database overrides (smc_simple_pair_overrides table)
        3. Global scalp defaults (smc_simple_global_config table)

        Args:
            epic: IG Markets epic code (e.g., 'CS.D.EURUSD.CEEM.IP')

        Returns:
            Dict with effective scalp settings for the pair
        """
        # Start with current instance values (already includes CLI overrides if any)
        config = {
            'ema_period': self.scalp_ema_period,
            'swing_lookback_bars': self.scalp_swing_lookback_bars,
            'limit_offset_pips': getattr(self, 'scalp_limit_offset_pips', 1.0),
            'htf_timeframe': self.scalp_htf_timeframe,
            'trigger_timeframe': self.scalp_trigger_timeframe,
            'entry_timeframe': self.scalp_entry_timeframe,
            'min_confidence': self.scalp_min_confidence,
            'cooldown_minutes': self.scalp_cooldown_minutes,
            'swing_break_tolerance_pips': getattr(self, 'scalp_swing_break_tolerance_pips', 0.5),
        }

        # Check for per-pair overrides from database
        # Priority: CLI override (--scalp-ema etc.) > Per-pair DB override > Global default
        # Only skip DB lookup if CLI explicitly set that specific parameter
        if self._db_config:
            cli_overrides = self._config_override or {}

            # EMA period: CLI --scalp-ema > per-pair DB > global
            if 'scalp_ema_period' not in cli_overrides:
                pair_ema = self._db_config.get_pair_scalp_ema_period(epic)
                if pair_ema is not None:
                    config['ema_period'] = pair_ema

            # Swing lookback: CLI --scalp-swing-lookback > per-pair DB > global
            if 'scalp_swing_lookback_bars' not in cli_overrides:
                pair_swing = self._db_config.get_pair_scalp_swing_lookback(epic)
                if pair_swing is not None:
                    config['swing_lookback_bars'] = pair_swing

            # Limit offset: CLI --scalp-offset > per-pair DB > global
            if 'scalp_limit_offset_pips' not in cli_overrides:
                pair_offset = self._db_config.get_pair_scalp_limit_offset(epic)
                if pair_offset is not None:
                    config['limit_offset_pips'] = pair_offset

            # HTF timeframe: CLI --scalp-htf > per-pair DB > global
            if 'scalp_htf_timeframe' not in cli_overrides:
                pair_htf = self._db_config.get_pair_scalp_htf_timeframe(epic)
                if pair_htf is not None:
                    config['htf_timeframe'] = pair_htf

            # Trigger timeframe: per-pair DB > global (no CLI override for this)
            pair_trigger = self._db_config.get_pair_scalp_trigger_timeframe(epic)
            if pair_trigger is not None:
                config['trigger_timeframe'] = pair_trigger

            # Entry timeframe: per-pair DB > global (no CLI override for this)
            pair_entry = self._db_config.get_pair_scalp_entry_timeframe(epic)
            if pair_entry is not None:
                config['entry_timeframe'] = pair_entry

            # Min confidence: per-pair DB > global
            pair_confidence = self._db_config.get_pair_scalp_min_confidence(epic)
            if pair_confidence is not None:
                config['min_confidence'] = pair_confidence

            # Cooldown: per-pair DB > global
            pair_cooldown = self._db_config.get_pair_scalp_cooldown_minutes(epic)
            if pair_cooldown is not None:
                config['cooldown_minutes'] = pair_cooldown

            # Swing break tolerance: per-pair DB > global
            pair_tolerance = self._db_config.get_pair_scalp_swing_break_tolerance(epic)
            if pair_tolerance is not None:
                config['swing_break_tolerance_pips'] = pair_tolerance

        return config

    def _apply_pair_scalp_config(self, epic: str) -> dict:
        """Apply per-pair scalp configuration and return the effective settings.

        This method is called during signal generation to apply pair-specific
        scalp settings. It modifies the strategy instance variables for the
        current pair and returns the effective configuration.

        Args:
            epic: IG Markets epic code

        Returns:
            Dict with the applied scalp settings
        """
        pair_config = self._get_pair_scalp_config(epic)

        # Apply settings to instance
        self.ema_period = pair_config['ema_period']
        self.swing_lookback = pair_config['swing_lookback_bars']
        self.htf_timeframe = pair_config['htf_timeframe']
        self.trigger_tf = pair_config['trigger_timeframe']
        self.entry_tf = pair_config['entry_timeframe']
        self.min_confidence = pair_config['min_confidence']
        self.cooldown_hours = pair_config['cooldown_minutes'] / 60.0
        self.scalp_swing_break_tolerance_pips = pair_config['swing_break_tolerance_pips']

        # v2.22.0: Apply per-pair limit offset for scalp mode
        # This is used by _calculate_limit_entry when limit orders are enabled
        limit_offset = pair_config.get('limit_offset_pips', 1.0)
        self.momentum_offset_pips = limit_offset
        self.pullback_offset_min_pips = limit_offset
        self.pullback_offset_max_pips = limit_offset

        # v2.31.0: Apply per-pair reversal override settings (GBPUSD/NZDUSD optimization)
        self.scalp_reversal_enabled = pair_config.get('reversal_enabled', self.scalp_reversal_enabled)
        self.scalp_reversal_min_runway_pips = pair_config.get('reversal_min_runway_pips', self.scalp_reversal_min_runway_pips)

        # v2.32.1: Per-pair HTF alignment override
        # Allows disabling the HTF alignment filter for specific pairs via parameter_overrides
        if self._db_config:
            htf_align = self._db_config.get_for_pair(epic, 'scalp_require_htf_alignment')
            if htf_align is not None:
                self.scalp_require_htf_alignment = bool(htf_align)

        # Log if using per-pair overrides
        pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
        self.logger.debug(f"📊 {pair} Scalp Config: EMA={pair_config['ema_period']}, "
                         f"Swing={pair_config['swing_lookback_bars']}, "
                         f"HTF={pair_config['htf_timeframe']}, "
                         f"Reversal={self.scalp_reversal_enabled}@{self.scalp_reversal_min_runway_pips}p, "
                         f"Offset={limit_offset} pips")

        return pair_config

    def _apply_pair_scalp_filters(
        self,
        epic: str,
        signal: Dict,
        candle_timestamp: Any
    ) -> Tuple[bool, str]:
        """
        Apply per-pair scalp filters based on Jan 2026 NZDUSD trade analysis.

        These filters are configurable per-pair via parameter_overrides JSONB.
        Currently only NZDUSD has filters configured - other pairs pass through.

        Filters:
        1. Efficiency Ratio - rejects choppy markets (ER < threshold)
        2. Trending Regime - requires market_regime_detected == 'trending'
        3. Session Hours - restricts to high-liquidity hours
        4. MACD Alignment - requires MACD histogram to align with direction
        5. EMA Stack Alignment - requires EMA stack to match direction

        Args:
            epic: IG Markets epic code
            signal: Signal dict with performance metrics already added
            candle_timestamp: Current candle timestamp for session check

        Returns:
            Tuple of (passed: bool, rejection_reason: str)
        """
        if not self._using_database_config or not self._db_config:
            return True, ""

        direction = signal.get('signal_type', signal.get('direction', ''))

        # v2.35.5: Backtest override support for scalp filters
        # These overrides allow testing filters in backtest without affecting live trading
        override = self._config_override if hasattr(self, '_config_override') and self._config_override else {}

        # Filter 1: Efficiency Ratio
        min_er = self._db_config.get_pair_scalp_min_efficiency_ratio(epic)
        if min_er is not None:
            current_er = signal.get('efficiency_ratio', 0.0)
            if current_er is not None and current_er < min_er:
                self._track_filter_rejection('efficiency_ratio')
                return False, f"efficiency_ratio {current_er:.3f} < {min_er:.3f}"

        # Filter 2: Trending Regime
        if self._db_config.get_pair_scalp_require_trending_regime(epic):
            regime = signal.get('market_regime_detected', 'unknown')
            if regime != 'trending':
                self._track_filter_rejection('trending_regime')
                return False, f"market_regime={regime} (requires trending)"

        # Filter 3: Session Hours (UTC)
        start_hour = self._db_config.get_pair_scalp_session_start_hour(epic)
        end_hour = self._db_config.get_pair_scalp_session_end_hour(epic)
        if start_hour is not None and end_hour is not None:
            # Extract hour from candle timestamp
            if hasattr(candle_timestamp, 'hour'):
                current_hour = candle_timestamp.hour
            elif hasattr(candle_timestamp, 'to_pydatetime'):
                current_hour = candle_timestamp.to_pydatetime().hour
            else:
                current_hour = datetime.utcnow().hour

            if current_hour < start_hour or current_hour >= end_hour:
                self._track_filter_rejection('session_hours')
                return False, f"hour={current_hour} outside session {start_hour}-{end_hour} UTC"

        # Filter 4: MACD Alignment
        if self._db_config.get_pair_scalp_require_macd_alignment(epic):
            macd_hist = signal.get('macd_histogram', 0.0)
            if macd_hist is not None:
                if direction == 'BULL' and macd_hist < -0.0001:
                    self._track_filter_rejection('macd_alignment')
                    return False, f"MACD histogram {macd_hist:.6f} negative on BUY"
                if direction == 'BEAR' and macd_hist > 0.0001:
                    self._track_filter_rejection('macd_alignment')
                    return False, f"MACD histogram {macd_hist:.6f} positive on SELL"

        # Filter 5: EMA Stack Alignment (with backtest override support)
        # Check override first, then fall back to database config
        if 'scalp_require_ema_stack_alignment' in override:
            ema_stack_filter_enabled = override['scalp_require_ema_stack_alignment']
        else:
            ema_stack_filter_enabled = self._db_config.get_pair_scalp_require_ema_stack_alignment(epic)

        ema_stack = signal.get('ema_stack_order', 'mixed')
        ema_stack_misaligned = (
            (direction == 'BULL' and ema_stack != 'bullish') or
            (direction == 'BEAR' and ema_stack != 'bearish')
        )

        if ema_stack_misaligned:
            if ema_stack_filter_enabled:
                self._track_filter_rejection('ema_stack_alignment')
                return False, f"EMA stack {ema_stack} not aligned for {direction}"
            else:
                # v2.35.5: MONITORING mode - log would-reject without blocking
                self.logger.info(f"📊 [MONITOR] EMA stack {ema_stack} misaligned for {direction} (would reject if enabled)")
                self._track_filter_rejection('ema_stack_alignment_monitor')

        # Filter 5b: Breakout Regime Block (v2.35.5 - with backtest override support)
        # Codex analysis: 0% win rate in breakout regime
        scalp_block_breakout = override.get('scalp_block_breakout_regime', False)
        market_regime = signal.get('market_regime_detected', 'unknown')
        if market_regime == 'breakout':
            if scalp_block_breakout:
                self._track_filter_rejection('breakout_regime')
                return False, f"Breakout regime blocked (market_regime={market_regime})"
            else:
                # MONITORING mode - log would-reject without blocking
                self.logger.info(f"📊 [MONITOR] Breakout regime detected (would reject if scalp_block_breakout_regime enabled)")
                self._track_filter_rejection('breakout_regime_monitor')

        # =========================================================================
        # v2.32.0: USDCAD-specific filters based on Jan 2026 trade analysis
        # These filters address the 0% win rate in ranging markets and
        # 20% win rate in trending+low volatility conditions
        # =========================================================================

        # Filter 6: Block Ranging Markets (0% win rate on USDCAD)
        if self._db_config.get_pair_scalp_block_ranging_market(epic):
            regime = signal.get('market_regime_detected', 'unknown')
            if regime == 'ranging':
                self._track_filter_rejection('block_ranging_market')
                return False, f"market_regime={regime} (ranging blocked)"

        # Filter 7: Block Low Volatility + Trending (20% win rate on USDCAD)
        if self._db_config.get_pair_scalp_block_low_volatility_trending(epic):
            regime = signal.get('market_regime_detected', 'unknown')
            volatility = signal.get('volatility_state', 'unknown')
            if regime == 'trending' and volatility == 'low':
                self._track_filter_rejection('block_low_volatility_trending')
                return False, f"trending+low_volatility blocked (regime={regime}, vol={volatility})"

        # Filter 8: Minimum ADX (90% of USDCAD losses had ADX < 20)
        min_adx = self._db_config.get_pair_scalp_min_adx(epic)
        if min_adx is not None:
            current_adx = signal.get('adx_value', 0.0)
            if current_adx is not None and current_adx < min_adx:
                self._track_filter_rejection('min_adx')
                return False, f"ADX {current_adx:.1f} < {min_adx:.1f}"

        # =========================================================================
        # v2.35.5: Entry Quality/Momentum Gates (Codex recommendation)
        # These filters are BACKTEST-ONLY by default via override system
        # Codex analysis: entry_quality < 0.30 = 1 win in 12 trades, avg -$59.06
        #                 entry_momentum < 0.30 = 3 wins in 16 trades, avg -$39.20
        # =========================================================================

        # Filter 9: Minimum Entry Quality Score (backtest override only)
        min_entry_quality = override.get('scalp_min_entry_quality_score', None)
        if min_entry_quality is not None:
            entry_quality = signal.get('entry_quality_score')
            if entry_quality is not None and entry_quality < min_entry_quality:
                self._track_filter_rejection('entry_quality')
                return False, f"Entry quality {entry_quality:.2f} < {min_entry_quality:.2f}"

        # Filter 10: Minimum Entry Candle Momentum (backtest override only)
        min_entry_momentum = override.get('scalp_min_entry_candle_momentum', None)
        if min_entry_momentum is not None:
            entry_momentum = signal.get('entry_candle_momentum')
            if entry_momentum is not None and entry_momentum < min_entry_momentum:
                self._track_filter_rejection('entry_momentum')
                return False, f"Entry momentum {entry_momentum:.2f} < {min_entry_momentum:.2f}"

        return True, ""

    def _get_current_spread(self, epic: str, df: pd.DataFrame) -> float:
        """Get current spread estimate based on session and pair.

        For scalping, spread is critical since a 1 pip spread on 5 pip TP
        eats 20% of profit. This method estimates spread based on:
        - Time of day (session)
        - Currency pair liquidity

        Args:
            epic: IG Markets epic code
            df: DataFrame with timestamp index

        Returns:
            Estimated spread in pips
        """
        # Try to get spread from dataframe if available
        if 'spread' in df.columns and len(df) > 0:
            spread_val = df['spread'].iloc[-1]
            if spread_val is not None and spread_val > 0:
                return float(spread_val)

        # Estimate based on session and pair
        try:
            last_timestamp = df.index[-1]
            if hasattr(last_timestamp, 'hour'):
                hour_utc = last_timestamp.hour
            else:
                hour_utc = 12  # Default to overlap
        except (IndexError, AttributeError):
            hour_utc = 12

        # Extract pair from epic (e.g., 'CS.D.EURUSD.CEEM.IP' -> 'EURUSD')
        pair = ''
        if '.' in epic:
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2]

        # Typical spreads by session (in pips) - conservative estimates
        session_spreads = {
            'asian': {'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.2, 'AUDUSD': 1.5, 'default': 2.0},
            'london': {'EURUSD': 0.8, 'GBPUSD': 1.0, 'USDJPY': 0.8, 'AUDUSD': 1.0, 'default': 1.2},
            'overlap': {'EURUSD': 0.6, 'GBPUSD': 0.8, 'USDJPY': 0.7, 'AUDUSD': 0.8, 'default': 1.0},
            'ny': {'EURUSD': 0.8, 'GBPUSD': 1.0, 'USDJPY': 0.9, 'AUDUSD': 1.0, 'default': 1.2},
        }

        # Determine session
        if 7 <= hour_utc < 12:
            session = 'london'
        elif 12 <= hour_utc < 17:
            session = 'overlap'
        elif 17 <= hour_utc < 21:
            session = 'ny'
        else:
            session = 'asian'

        return session_spreads.get(session, {}).get(pair, session_spreads[session]['default'])

    def _check_scalp_spread_filter(self, epic: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if current spread is acceptable for scalping.

        This is a critical filter for scalp mode. With 5 pip TP targets,
        spread must be tight to maintain profitability.

        Args:
            epic: IG Markets epic code
            df: DataFrame with price data

        Returns:
            Tuple of (passed, reason)
        """
        if not self.scalp_mode_enabled or not self.scalp_require_tight_spread:
            return True, "Spread filter not active"

        current_spread = self._get_current_spread(epic, df)

        if current_spread > self.scalp_max_spread_pips:
            return False, f"Spread {current_spread:.1f} pips > max {self.scalp_max_spread_pips} pips"

        return True, f"Spread OK: {current_spread:.1f} pips"

    def _normalize_reversal_list(self, raw_value, default_list: List[str]) -> List[str]:
        """Normalize reversal filter list settings from config."""
        if raw_value is None:
            return default_list
        if isinstance(raw_value, list):
            return raw_value
        if isinstance(raw_value, tuple):
            return list(raw_value)
        if isinstance(raw_value, str):
            parts = [part.strip() for part in raw_value.split(',') if part.strip()]
            return parts or default_list
        return default_list

    def _evaluate_reversal_scalp_override(
        self,
        df_entry: Optional[pd.DataFrame],
        df_trigger: pd.DataFrame,
        df_4h: pd.DataFrame,
        direction: str,
        pullback_depth: float,
        market_price: float,
        swing_level: float,
        pip_value: float,
        rsi_value: Optional[float],
        epic: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate counter-trend scalp override when HTF alignment fails."""
        details: Dict[str, Any] = {}
        if not getattr(self, 'scalp_reversal_enabled', False):
            return False, "Reversal override disabled", details

        market_regime = 'unknown'
        volatility_state = 'unknown'
        entry_candle_momentum = None
        try:
            from forex_scanner.core.strategies.helpers.smc_performance_metrics import (
                get_performance_metrics_calculator
            )
            calculator = get_performance_metrics_calculator(self.logger)
            metrics = calculator.calculate_metrics(
                df_5m=df_entry,
                df_15m=df_trigger,
                df_4h=df_4h,
                signal_data={'signal_type': direction, 'pullback_depth': pullback_depth},
                epic=epic
            )
            market_regime = metrics.market_regime
            volatility_state = metrics.volatility_state
            entry_candle_momentum = metrics.entry_candle_momentum
        except Exception as e:
            self.logger.debug(f"Reversal metrics calculation failed: {e}")

        rsi_zone = None
        if rsi_value is not None:
            if rsi_value > 70:
                rsi_zone = 'overbought'
            elif rsi_value < 30:
                rsi_zone = 'oversold'
            else:
                rsi_zone = 'neutral'

        runway_pips = None
        if market_price is not None and swing_level is not None and pip_value > 0:
            if direction == 'BULL':
                runway_pips = (swing_level - market_price) / pip_value
            else:
                runway_pips = (market_price - swing_level) / pip_value
            runway_pips = max(runway_pips, 0.0)

        details.update({
            'market_regime_detected': market_regime,
            'volatility_state': volatility_state,
            'entry_candle_momentum': entry_candle_momentum,
            'rsi_zone': rsi_zone,
            'runway_pips': runway_pips,
        })

        block_regimes = self._normalize_reversal_list(
            getattr(self, 'scalp_reversal_block_regimes', None),
            ['breakout']
        )
        block_volatility = self._normalize_reversal_list(
            getattr(self, 'scalp_reversal_block_volatility_states', None),
            ['high']
        )

        if market_regime in block_regimes:
            return False, f"Reversal blocked: regime={market_regime}", details
        if volatility_state in block_volatility:
            return False, f"Reversal blocked: volatility={volatility_state}", details

        min_runway = getattr(self, 'scalp_reversal_min_runway_pips', 15.0)
        if runway_pips is None or runway_pips < min_runway:
            runway_str = "N/A" if runway_pips is None else f"{runway_pips:.1f}"
            return False, f"Reversal blocked: runway {runway_str} < {min_runway}", details

        min_momentum = getattr(self, 'scalp_reversal_min_entry_momentum', 0.60)
        momentum_ok = entry_candle_momentum is not None and entry_candle_momentum >= min_momentum
        rsi_ok = getattr(self, 'scalp_reversal_allow_rsi_extremes', True) and rsi_zone in ('overbought', 'oversold')
        if not (momentum_ok or rsi_ok):
            return False, "Reversal blocked: no momentum/RSI confirmation", details

        runway_display = f"{runway_pips:.1f}" if runway_pips is not None else "N/A"
        momentum_display = f"{entry_candle_momentum:.2f}" if entry_candle_momentum is not None else "N/A"
        reason = (
            f"Reversal override: regime={market_regime}, vol={volatility_state}, "
            f"runway={runway_display}, momentum={momentum_display}, rsi={rsi_zone}"
        )
        return True, reason, details

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
        # ================================================================
        # SCALP MODE: Apply per-pair configuration before signal detection
        # ================================================================
        # v2.25.0: Reset rejection candle confirmation flag
        self._scalp_rejection_confirmed = False
        # v2.25.1: Reset entry candle alignment confirmation flag
        self._scalp_entry_alignment_confirmed = False

        if self.scalp_mode_enabled:
            pair_scalp_config = self._apply_pair_scalp_config(epic)
            self.logger.debug(f"🎯 Scalp config applied for {pair}: EMA={pair_scalp_config['ema_period']}, "
                             f"Swing={pair_scalp_config['swing_lookback_bars']}")

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🔍 SMC SIMPLE Strategy v{self.strategy_version} - Signal Detection")
        self.logger.info(f"   Pair: {pair} ({epic})")
        self.logger.info(f"   Timeframes: {self.htf_timeframe} (bias) → {self.trigger_tf} (trigger) → {self.entry_tf} (entry)")
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
        reversal_override_applied = False
        reversal_override_reason = None
        reversal_override_details: Dict[str, Any] = {}

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
            # PRE-FILTER: Scalp Spread Check (only in scalp mode)
            # ================================================================
            if self.scalp_mode_enabled:
                spread_valid, spread_reason = self._check_scalp_spread_filter(epic, df_trigger)
                if not spread_valid:
                    self.logger.info(f"\n💰 SCALP SPREAD FILTER: {spread_reason}")
                    self._track_rejection(
                        stage='SCALP_SPREAD',
                        reason=spread_reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value)
                    )
                    return None
                self.logger.info(f"\n💰 SCALP SPREAD: {spread_reason}")

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
            self.logger.info(f"\n📊 TIER 1: Checking {self.htf_timeframe.upper()} {self.ema_period} EMA Bias")
            slope_status = "ON" if self.ema_slope_validation_enabled else "OFF"
            self.logger.info(f"   ⚙️  Settings: buffer={self.ema_buffer_pips}p, min_dist={self.min_distance_from_ema}p, slope={slope_status}")

            ema_result = self._check_ema_bias(df_4h, pip_value, epic=epic)

            if not ema_result['valid']:
                self.logger.info(f"   ❌ {ema_result['reason']}")
                # v2.16.0: Track EMA slope rejections separately from EMA position rejections
                rejection_stage = ema_result.get('rejection_type', 'TIER1_EMA')
                # Build context with EMA slope data if available
                context = self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result)
                if 'ema_slope_atr' in ema_result:
                    context['ema_slope_atr'] = ema_result['ema_slope_atr']
                # Track rejection with appropriate stage
                self._track_rejection(
                    stage=rejection_stage,
                    reason=ema_result['reason'],
                    epic=epic,
                    pair=pair,
                    candle_timestamp=candle_timestamp,
                    direction=ema_result.get('attempted_direction'),
                    context=context
                )
                return None

            direction = ema_result['direction']  # 'BULL' or 'BEAR'
            ema_value = ema_result['ema_value']
            ema_distance = ema_result['distance_pips']
            tier1_macd_aligned = ema_result.get('macd_aligned')  # v2.23.0

            self.logger.info(f"   ✅ Direction: {direction}")
            self.logger.info(f"   ✅ 50 EMA: {ema_value:.5f}")
            self.logger.info(f"   ✅ Distance: {ema_distance:.1f} pips from EMA")

            # ================================================================
            # Extract 4H candle direction for analytics
            # Note: iloc[-1] is forming candle, iloc[-2] is last CLOSED candle
            # ================================================================
            htf_candle_direction = self._get_candle_direction(
                df_4h['open'].iloc[-2], df_4h['close'].iloc[-2]
            )
            htf_candle_direction_prev = self._get_candle_direction(
                df_4h['open'].iloc[-3], df_4h['close'].iloc[-3]
            )
            self.logger.debug(f"   📊 HTF Candle: {htf_candle_direction} (prev: {htf_candle_direction_prev})")

            # ================================================================
            # v2.35.4: HTF CANDLE ALIGNMENT FILTER (scalp mode)
            # ================================================================
            # Reject signals where HTF candle direction contradicts trade direction.
            # Analysis of Jan 29, 2026 trades showed 73% of losing trades had
            # HTF candle direction OPPOSITE to trade direction.
            # - BULL signal + BEARISH HTF candle = likely reversal, reject
            # - BEAR signal + BULLISH HTF candle = likely reversal, reject
            # ================================================================
            if self.scalp_mode_enabled and getattr(self, 'scalp_require_htf_alignment', False):
                htf_aligned = (
                    (direction == 'BULL' and htf_candle_direction == 'BULLISH') or
                    (direction == 'BEAR' and htf_candle_direction == 'BEARISH') or
                    (htf_candle_direction == 'DOJI')  # Allow DOJI (neutral)
                )

                if not htf_aligned:
                    expected_htf = 'BULLISH' if direction == 'BULL' else 'BEARISH'
                    rejection_reason = (
                        f"HTF candle not aligned: {htf_candle_direction} candle for {direction} signal "
                        f"(need {expected_htf})"
                    )
                    self.logger.info(f"   ❌ {rejection_reason}")

                    # Track rejection with full market context
                    self._track_rejection(
                        stage='TIER1_HTF_CANDLE',
                        reason=rejection_reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        direction=direction,
                        context={
                            'htf_candle_direction': htf_candle_direction,
                            'htf_candle_direction_prev': htf_candle_direction_prev,
                            'expected_htf_candle': expected_htf,
                            'ema_direction': direction,
                            'ema_value': ema_value,
                            'ema_distance_pips': ema_distance,
                            'filter': 'htf_candle_alignment'
                        }
                    )
                    return None
                else:
                    self.logger.info(f"   ✅ HTF candle aligned: {htf_candle_direction} matches {direction}")

            # ================================================================
            # TIER 2: Swing Break Confirmation (15m or 1H based on config)
            # ================================================================
            self.logger.info(f"\n📈 TIER 2: Checking {self.trigger_tf} Swing Break")
            dynamic_lb = "ON" if getattr(self, 'dynamic_swing_lookback_enabled', False) else "OFF"
            self.logger.info(f"   ⚙️  Settings: lookback={self.swing_lookback}, strength={self.swing_strength}, dynamic={dynamic_lb}")

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

            self.logger.info(f"   ✅ Swing {'Low' if direction == 'BEAR' else 'High'} Break: {swing_level:.5f}")
            self.logger.info(f"   ✅ Opposite Swing: {opposite_swing:.5f}")  # v1.6.0
            self.logger.info(f"   ✅ Body Close Confirmed: Yes")
            if self.volume_enabled:
                self.logger.info(f"   {'✅' if volume_confirmed else '⚠️ '} Volume Spike: {'Yes' if volume_confirmed else 'No (optional)'}")

            # ================================================================
            # TIER 3: Pullback Entry Zone (5m or 15m based on config)
            # ================================================================
            self.logger.info(f"\n🎯 TIER 3: Checking {self.entry_tf} Pullback Zone")
            self.logger.info(f"   ⚙️  Settings: fib={self.fib_min*100:.0f}%-{self.fib_max*100:.0f}%, confirm_bars={self.pullback_confirm_bars}")

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

            # v2.24.0: Initialize pattern/divergence data for alternative entry detection
            pattern_data = None
            rsi_divergence_data = None
            alternative_entry_type = None  # Will be set if pattern/divergence triggers entry

            # v2.24.0: Detect patterns and divergence BEFORE pullback validation
            # This allows them to serve as alternative TIER 3 entries
            # v2.25.0: Also detect patterns when scalp_require_rejection_candle is enabled
            should_detect_patterns = (
                getattr(self, 'pattern_confirmation_enabled', False) or
                getattr(self, 'pattern_as_entry_enabled', False) or
                (self.scalp_mode_enabled and getattr(self, 'scalp_require_rejection_candle', False))
            )
            if should_detect_patterns:
                try:
                    from forex_scanner.core.strategies.helpers.smc_candlestick_patterns import SMCCandlestickPatterns
                    pattern_detector = SMCCandlestickPatterns(self.logger)
                    min_strength = getattr(self, 'pattern_min_strength', 0.70)

                    pattern = pattern_detector.detect_rejection_pattern(
                        entry_df[-10:] if len(entry_df) >= 10 else entry_df,
                        direction,
                        min_strength=min_strength
                    )

                    if pattern:
                        pattern_data = pattern
                        pattern_mode = getattr(self, 'pattern_confirmation_mode', 'MONITORING')
                        self.logger.info(f"   🎯 PATTERN [{pattern_mode}]: {pattern.get('pattern_type', 'unknown')} "
                                       f"(strength: {pattern.get('strength', 0)*100:.0f}%)")
                except ImportError as e:
                    self.logger.debug(f"   ⚠️ Pattern detection unavailable: {e}")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Pattern detection error: {e}")

            if getattr(self, 'rsi_divergence_enabled', False) or getattr(self, 'divergence_as_entry_enabled', False):
                try:
                    divergence = self._check_rsi_divergence(entry_df, direction)
                    if divergence.get('detected'):
                        rsi_divergence_data = divergence
                        div_mode = getattr(self, 'rsi_divergence_mode', 'MONITORING')
                        self.logger.info(f"   📊 DIVERGENCE [{div_mode}]: {divergence.get('type', 'unknown')} "
                                       f"(strength: {divergence.get('strength', 0)*100:.0f}%)")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ RSI divergence detection error: {e}")

            if not pullback_result['valid']:
                self.logger.info(f"   ❌ {pullback_result['reason']}")

                # v2.24.0: Check for alternative TIER 3 entries (pattern or divergence)
                pattern_entry_threshold = getattr(self, 'pattern_entry_min_strength', 0.80)
                divergence_entry_threshold = getattr(self, 'divergence_entry_min_strength', 0.50)
                pattern_as_entry = getattr(self, 'pattern_as_entry_enabled', False)
                divergence_as_entry = getattr(self, 'divergence_as_entry_enabled', False)

                # Check if pattern can serve as alternative entry
                if pattern_as_entry and pattern_data:
                    pattern_strength = pattern_data.get('strength', 0)
                    if pattern_strength >= pattern_entry_threshold:
                        alternative_entry_type = 'PATTERN'
                        self.logger.info(f"   ✅ ALTERNATIVE ENTRY: Pattern ({pattern_data.get('pattern_type', 'unknown')}) "
                                       f"strength {pattern_strength*100:.0f}% >= {pattern_entry_threshold*100:.0f}% threshold")

                # Check if divergence can serve as alternative entry
                if not alternative_entry_type and getattr(self, 'divergence_as_entry_enabled', False) and rsi_divergence_data:
                    divergence_strength = rsi_divergence_data.get('strength', 0)
                    if divergence_strength >= divergence_entry_threshold:
                        alternative_entry_type = 'DIVERGENCE'
                        self.logger.info(f"   ✅ ALTERNATIVE ENTRY: RSI Divergence ({rsi_divergence_data.get('type', 'unknown')}) "
                                       f"strength {divergence_strength*100:.0f}% >= {divergence_entry_threshold*100:.0f}% threshold")

                # If no alternative entry found, reject the signal
                if not alternative_entry_type:
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

            # v2.24.0: Set entry variables based on pullback or alternative entry
            if alternative_entry_type:
                # Alternative entry: use current market price
                market_price = entry_df['close'].iloc[-1]
                pullback_depth = 0.0  # Not applicable for pattern/divergence entries
                in_optimal_zone = False  # Not applicable

                if alternative_entry_type == 'PATTERN':
                    entry_type = 'PATTERN'
                    trigger_type = f"pattern_{pattern_data.get('pattern_type', 'unknown')}"
                else:  # DIVERGENCE
                    entry_type = 'DIVERGENCE'
                    trigger_type = f"divergence_{rsi_divergence_data.get('type', 'unknown')}"
            else:
                # Normal pullback entry
                market_price = pullback_result['entry_price']  # Current close (market price)
                pullback_depth = pullback_result['pullback_depth']
                in_optimal_zone = pullback_result['in_optimal_zone']
                entry_type = pullback_result.get('entry_type', 'PULLBACK')  # v1.8.0
                trigger_type = pullback_result.get('trigger_type', TRIGGER_SWING_PULLBACK)  # v2.23.0

            # Build trigger details
            trigger_details = {
                'base_trigger': trigger_type,
                'pullback_depth': pullback_depth,
                'entry_type': entry_type,
                'macd_aligned': tier1_macd_aligned,  # From TIER 1 EMA check
                'alternative_entry': alternative_entry_type,
            }

            # v2.23.0: Enhance trigger type with pattern/divergence suffixes (for pullback entries)
            if pattern_data and not alternative_entry_type:
                trigger_details['pattern'] = pattern_data
                trigger_details['pattern_confirmed'] = True

                pattern_type = pattern_data.get('pattern_type', '')
                if 'pin_bar' in pattern_type:
                    pattern_suffix = PATTERN_SUFFIX_PIN
                elif 'engulfing' in pattern_type:
                    pattern_suffix = PATTERN_SUFFIX_ENG
                elif 'inside' in pattern_type:
                    pattern_suffix = PATTERN_SUFFIX_INS
                elif 'hammer' in pattern_type or 'shooting' in pattern_type:
                    pattern_suffix = PATTERN_SUFFIX_PIN
                else:
                    pattern_suffix = ''

                min_strength = getattr(self, 'pattern_min_strength', 0.70)
                if pattern_suffix and pattern_data.get('strength', 0) >= min_strength:
                    trigger_type = f"{trigger_type}{pattern_suffix}"
                    trigger_details['enhanced_trigger'] = trigger_type

            if rsi_divergence_data and not alternative_entry_type:
                trigger_details['rsi_divergence'] = rsi_divergence_data
                if rsi_divergence_data.get('strength', 0) >= getattr(self, 'rsi_divergence_min_strength', 0.30):
                    trigger_type = f"{trigger_type}{PATTERN_SUFFIX_DIV}"
                    trigger_details['enhanced_trigger'] = trigger_type

            # v1.8.0: Log entry type (v2.24.0: added PATTERN and DIVERGENCE types)
            if entry_type == 'MOMENTUM':
                self.logger.info(f"   ✅ Entry Type: MOMENTUM (continuation)")
                self.logger.info(f"   ✅ Beyond Break: {pullback_depth*100:.1f}%")
            elif entry_type == 'PATTERN':
                self.logger.info(f"   ✅ Entry Type: PATTERN (alternative - {pattern_data.get('pattern_type', 'unknown')})")
                self.logger.info(f"   ✅ Pattern Strength: {pattern_data.get('strength', 0)*100:.0f}%")
            elif entry_type == 'DIVERGENCE':
                self.logger.info(f"   ✅ Entry Type: DIVERGENCE (alternative - {rsi_divergence_data.get('type', 'unknown')})")
                self.logger.info(f"   ✅ Divergence Strength: {rsi_divergence_data.get('strength', 0)*100:.0f}%")
            else:
                self.logger.info(f"   ✅ Entry Type: PULLBACK (retracement)")
                self.logger.info(f"   ✅ Pullback Depth: {pullback_depth*100:.1f}%")
            self.logger.info(f"   {'✅' if in_optimal_zone else '⚠️ '} Optimal Zone: {'Yes' if in_optimal_zone else 'No'}")

            # ================================================================
            # v2.22.0: SCALP ENTRY FILTERS (based on Jan 2026 trade analysis)
            # These filters block non-winning patterns in scalp mode:
            # - Pullback entries had 0% win rate (26 trades, 0 winners)
            # - Only HTF-aligned + momentum entries were profitable (40% win rate)
            # v2.24.0: PATTERN and DIVERGENCE entries are allowed (alternative entries)
            # ================================================================
            if self.scalp_mode_enabled:
                # Filter 1: Momentum-only (block pullback entries, allow PATTERN/DIVERGENCE)
                allowed_entry_types = ['MOMENTUM', 'PATTERN', 'DIVERGENCE']
                if self.scalp_momentum_only_filter and entry_type not in allowed_entry_types:
                    rejection_reason = f"Scalp filter: Non-momentum entry ({entry_type}) blocked"
                    self.logger.info(f"   ❌ {rejection_reason}")
                    # Collect full market context including prices for outcome analysis
                    context = self._collect_market_context(df_trigger, df_4h, entry_df, pip_value,
                                                          ema_result=ema_result, swing_result=swing_result,
                                                          pullback_result=pullback_result, direction=direction)
                    context.update({'entry_type': entry_type, 'filter': 'momentum_only'})
                    self._track_rejection(
                        stage='SCALP_ENTRY_FILTER',
                        reason=rejection_reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        direction=direction,
                        context=context
                    )
                    return None

                # Precompute RSI value for scalp filters and reversal override
                rsi_value = None
                if 'rsi' in entry_df.columns and len(entry_df) > 0:
                    rsi_value = entry_df['rsi'].iloc[-1]
                elif 'rsi' in df_trigger.columns and len(df_trigger) > 0:
                    rsi_value = df_trigger['rsi'].iloc[-1]

                # Filter 2: HTF Bias Score System (v2.35.0 - replaces binary HTF alignment)
                # Professional approach: continuous bias measurement instead of binary filter
                # See htf_bias_calculator.py for score interpretation:
                #   0.0-0.3: Strong counter-trend, 0.3-0.5: Weak counter-trend
                #   0.5-0.7: Neutral, 0.7-0.9: Aligned, 0.9-1.0: Strong alignment
                htf_bias_score = 0.5  # Default neutral
                htf_bias_details = {}
                htf_bias_mode = 'disabled'  # Per-pair mode

                if self.htf_bias_enabled and df_4h is not None and len(df_4h) >= 5:
                    # Get per-pair HTF bias mode
                    if hasattr(self, '_db_config') and self._db_config is not None:
                        htf_bias_mode = self._db_config.get_pair_htf_bias_mode(epic)
                        htf_bias_threshold = self._db_config.get_pair_htf_bias_threshold(epic)
                    else:
                        htf_bias_mode = 'active'  # Default to active if no config
                        htf_bias_threshold = self.htf_bias_min_threshold

                    if htf_bias_mode != 'disabled':
                        # Calculate HTF bias score
                        htf_bias_score, htf_bias_details = self._htf_bias_calculator.calculate_bias_score(
                            df_4h=df_4h,
                            direction=direction,
                            epic=epic
                        )

                        # Store in trigger details for alert history
                        trigger_details['htf_bias_score'] = htf_bias_score
                        trigger_details['htf_bias_mode'] = htf_bias_mode
                        trigger_details['htf_bias_details'] = htf_bias_details

                        # Apply confidence multiplier if enabled
                        if self.htf_bias_confidence_multiplier_enabled:
                            confidence_multiplier = self._htf_bias_calculator.get_confidence_multiplier(htf_bias_score)
                            trigger_details['htf_bias_confidence_multiplier'] = confidence_multiplier

                        # Check if should filter (active mode only)
                        should_reject, filter_reason = self._htf_bias_calculator.should_filter(
                            bias_score=htf_bias_score,
                            threshold=htf_bias_threshold,
                            mode=htf_bias_mode
                        )

                        interpretation = htf_bias_details.get('interpretation', 'UNKNOWN')
                        self.logger.info(
                            f"   📊 HTF Bias: {htf_bias_score:.3f} ({interpretation}) "
                            f"[mode={htf_bias_mode}, threshold={htf_bias_threshold}]"
                        )

                        if should_reject:
                            rejection_reason = f"Scalp filter: HTF bias score {htf_bias_score:.3f} < {htf_bias_threshold} ({interpretation})"
                            self.logger.info(f"   ❌ {rejection_reason}")
                            # Collect full market context including prices for outcome analysis
                            context = self._collect_market_context(df_trigger, df_4h, entry_df, pip_value,
                                                                  ema_result=ema_result, swing_result=swing_result,
                                                                  pullback_result=pullback_result, direction=direction)
                            context.update({
                                'htf_candle_direction': htf_candle_direction,
                                'htf_candle_direction_prev': htf_candle_direction_prev,
                                'htf_bias_score': htf_bias_score,
                                'htf_bias_mode': htf_bias_mode,
                                'htf_bias_details': htf_bias_details,
                                'rejection_details': {
                                    'filter': 'htf_bias_score',
                                    'htf_bias_score': htf_bias_score,
                                    'htf_bias_threshold': htf_bias_threshold,
                                    'htf_bias_mode': htf_bias_mode,
                                    'interpretation': interpretation,
                                    'signal_direction': direction
                                }
                            })
                            self._track_rejection(
                                stage='SCALP_ENTRY_FILTER',
                                reason=rejection_reason,
                                epic=epic,
                                pair=pair,
                                candle_timestamp=candle_timestamp,
                                direction=direction,
                                context=context
                            )
                            return None
                else:
                    # HTF bias disabled or insufficient data
                    trigger_details['htf_bias_mode'] = 'disabled'
                    trigger_details['htf_bias_score'] = None

                # Filter 3: RSI zone filter (avoid overbought buys, oversold sells)
                if rsi_value is not None:
                    if direction == 'BULL' and rsi_value > self.scalp_entry_rsi_buy_max:
                        rejection_reason = f"Scalp filter: RSI {rsi_value:.1f} > {self.scalp_entry_rsi_buy_max} (overbought BUY)"
                        self.logger.info(f"   ❌ {rejection_reason}")
                        # Collect full market context including prices for outcome analysis
                        context = self._collect_market_context(df_trigger, df_4h, entry_df, pip_value,
                                                              ema_result=ema_result, swing_result=swing_result,
                                                              pullback_result=pullback_result, direction=direction)
                        context.update({'rsi': rsi_value, 'filter': 'rsi_zone'})
                        self._track_rejection(
                            stage='SCALP_ENTRY_FILTER',
                            reason=rejection_reason,
                            epic=epic,
                            pair=pair,
                            candle_timestamp=candle_timestamp,
                            direction=direction,
                            context=context
                        )
                        return None
                    if direction == 'BEAR' and rsi_value < self.scalp_entry_rsi_sell_min:
                        rejection_reason = f"Scalp filter: RSI {rsi_value:.1f} < {self.scalp_entry_rsi_sell_min} (oversold SELL)"
                        self.logger.info(f"   ❌ {rejection_reason}")
                        # Collect full market context including prices for outcome analysis
                        context = self._collect_market_context(df_trigger, df_4h, entry_df, pip_value,
                                                              ema_result=ema_result, swing_result=swing_result,
                                                              pullback_result=pullback_result, direction=direction)
                        context.update({'rsi': rsi_value, 'filter': 'rsi_zone'})
                        self._track_rejection(
                            stage='SCALP_ENTRY_FILTER',
                            reason=rejection_reason,
                            epic=epic,
                            pair=pair,
                            candle_timestamp=candle_timestamp,
                            direction=direction,
                            context=context
                        )
                        return None

                # Filter 4: Minimum EMA distance
                if self.scalp_min_ema_distance_pips > 0 and ema_distance < self.scalp_min_ema_distance_pips:
                    rejection_reason = f"Scalp filter: EMA distance {ema_distance:.1f} < {self.scalp_min_ema_distance_pips} pips"
                    self.logger.info(f"   ❌ {rejection_reason}")
                    # Collect full market context including prices for outcome analysis
                    context = self._collect_market_context(df_trigger, df_4h, entry_df, pip_value,
                                                          ema_result=ema_result, swing_result=swing_result,
                                                          pullback_result=pullback_result, direction=direction)
                    context.update({'ema_distance': ema_distance, 'filter': 'ema_distance'})
                    self._track_rejection(
                        stage='SCALP_ENTRY_FILTER',
                        reason=rejection_reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        direction=direction,
                        context=context
                    )
                    return None

                # ================================================================
                # Filter 5: REJECTION CANDLE CONFIRMATION (v2.25.0)
                # Require entry-TF rejection candle before scalp entry
                # Based on Jan 2026 analysis: MAE=0 means no reversal confirmation
                # Per-pair override: helps USDJPY/EURUSD/AUDJPY, hurts GBPUSD
                # ================================================================
                # Check per-pair override first, then fall back to global
                scalp_require_rejection = getattr(self, 'scalp_require_rejection_candle', False)
                if hasattr(self, '_db_config') and self._db_config:
                    pair_override = self._db_config.get_pair_scalp_require_rejection_candle(epic)
                    if pair_override is not None:
                        scalp_require_rejection = pair_override

                if scalp_require_rejection:
                    rejection_min_strength = getattr(self, 'scalp_rejection_min_strength', 0.70)

                    # Check if we have a valid rejection candle (pattern_data from earlier detection)
                    has_rejection_candle = False
                    rejection_pattern_type = None
                    rejection_strength = 0.0

                    if pattern_data and pattern_data.get('strength', 0) >= rejection_min_strength:
                        has_rejection_candle = True
                        rejection_pattern_type = pattern_data.get('pattern_type', 'unknown')
                        rejection_strength = pattern_data.get('strength', 0)

                    if not has_rejection_candle:
                        rejection_reason = f"Scalp filter: No rejection candle (min strength {rejection_min_strength*100:.0f}%)"
                        self.logger.info(f"   ❌ {rejection_reason}")
                        # Collect full market context including prices for outcome analysis
                        context = self._collect_market_context(df_trigger, df_4h, entry_df, pip_value,
                                                              ema_result=ema_result, swing_result=swing_result,
                                                              pullback_result=pullback_result, direction=direction)
                        context.update({
                            'filter': 'rejection_candle',
                            'has_pattern': pattern_data is not None,
                            'pattern_strength': pattern_data.get('strength', 0) if pattern_data else 0,
                            'required_strength': rejection_min_strength
                        })
                        self._track_rejection(
                            stage='SCALP_ENTRY_FILTER',
                            reason=rejection_reason,
                            epic=epic,
                            pair=pair,
                            candle_timestamp=candle_timestamp,
                            direction=direction,
                            context=context
                        )
                        return None

                    # Rejection candle confirmed - log it
                    self.logger.info(f"   ✅ Rejection candle: {rejection_pattern_type} (strength: {rejection_strength*100:.0f}%)")

                    # Set flag for market order if scalp_use_market_on_rejection is enabled
                    use_market_on_rejection = getattr(self, 'scalp_use_market_on_rejection', True)
                    if use_market_on_rejection:
                        self._scalp_rejection_confirmed = True
                        self.logger.info(f"   📍 Using MARKET order (rejection candle confirmed)")

            # ================================================================
            # Filter 6: ENTRY CANDLE ALIGNMENT (v2.25.1)
            # Simple alternative to rejection candle - requires entry candle
            # color to match trade direction (green for BUY, red for SELL)
            #
            # NOTE (v2.35.4, Jan 29 2026): This filter is now DISABLED by default.
            # The HTF candle alignment filter (TIER1_HTF_CANDLE) is the critical one
            # for trend direction validation. This 1m filter is too noisy - candles
            # flip rapidly and over-filter valid trades. Analysis showed HTF filter
            # would have prevented 73% of losses; this filter was enabled but didn't
            # help. Keep only HTF alignment to avoid over-filtering.
            # ================================================================
            scalp_require_entry_alignment = getattr(self, 'scalp_require_entry_candle_alignment', False)

            # Check for per-pair override
            if hasattr(self, '_db_config') and self._db_config:
                pair_override = self._db_config.get_pair_scalp_require_entry_candle_alignment(epic)
                if pair_override is not None:
                    scalp_require_entry_alignment = pair_override

            if scalp_require_entry_alignment:
                self.logger.info(f"\n🕯️ Filter 6: Entry Candle Alignment Check")

                # Get entry candle OHLC
                entry_candle_open = entry_df['open'].iloc[-1]
                entry_candle_close = entry_df['close'].iloc[-1]

                is_bullish_candle = entry_candle_close > entry_candle_open
                is_bearish_candle = entry_candle_close < entry_candle_open
                is_doji = entry_candle_close == entry_candle_open

                candle_color = 'GREEN' if is_bullish_candle else ('RED' if is_bearish_candle else 'DOJI')

                # Check alignment: BUY needs green candle, SELL needs red candle
                candle_aligned = (
                    (direction == 'BULL' and is_bullish_candle) or
                    (direction == 'BEAR' and is_bearish_candle)
                )

                if not candle_aligned:
                    expected_color = 'GREEN' if direction == 'BULL' else 'RED'
                    alignment_reason = f"Entry candle not aligned: {candle_color} candle for {direction} signal (need {expected_color})"
                    self.logger.info(f"   ❌ {alignment_reason}")
                    # Collect full market context including prices for outcome analysis
                    context = self._collect_market_context(df_trigger, df_4h, entry_df, pip_value,
                                                          ema_result=ema_result, swing_result=swing_result,
                                                          pullback_result=pullback_result, direction=direction)
                    context.update({
                        'rejection_details': {
                            'filter': 'entry_candle_alignment',
                            'candle_color': candle_color,
                            'expected_color': expected_color,
                            'signal_direction': direction,
                            'entry_candle_open': float(entry_candle_open),
                            'entry_candle_close': float(entry_candle_close),
                            'is_doji': is_doji
                        }
                    })
                    self._track_rejection(
                        stage='SCALP_ENTRY_FILTER',
                        reason=alignment_reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        direction=direction,
                        context=context
                    )
                    return None

                self.logger.info(f"   ✅ Entry candle aligned: {candle_color} candle for {direction} signal")

                # Set flag for market order if scalp_use_market_on_entry_alignment is enabled
                use_market_on_alignment = getattr(self, 'scalp_use_market_on_entry_alignment', True)
                if use_market_on_alignment:
                    self._scalp_entry_alignment_confirmed = True
                    self.logger.info(f"   📍 Using MARKET order (entry candle aligned)")

            # ================================================================
            # TIER 4: Swing Proximity Validation (v2.15.0)
            # Prevents entries too close to opposing swing levels
            # Based on trade log analysis: 65% of losing trades were at wrong swing levels
            # ================================================================
            if self.swing_proximity_enabled and self.swing_proximity_validator:
                self.logger.info(f"\n🛡️ TIER 4: Checking Swing Proximity")

                # For BUY: check distance to nearest swing HIGH (resistance) = swing_level (broken high)
                # For SELL: check distance to nearest swing LOW (support) = swing_level (broken low)
                # After pullback entry, we're entering NEAR the broken swing level:
                # - BULL: swing_level = broken swing HIGH (resistance we just broke)
                # - BEAR: swing_level = broken swing LOW (support we just broke)
                # The pullback brings us back TOWARD these levels - that's the proximity concern!
                swing_signal_data = {
                    'nearest_resistance': swing_level if direction == 'BULL' else opposite_swing,
                    'nearest_support': opposite_swing if direction == 'BULL' else swing_level
                }
                proximity_result = self.swing_proximity_validator.validate_entry_proximity(
                    df=df_trigger,
                    current_price=market_price,
                    direction=direction,
                    epic=epic,
                    timeframe=self.trigger_tf,
                    signal=swing_signal_data  # Pass swing data from TIER 2
                )

                if not proximity_result['valid']:
                    self.logger.info(f"   ❌ {proximity_result.get('rejection_reason', 'Too close to swing level')}")
                    self._track_rejection(
                        stage='TIER4_PROXIMITY',
                        reason=proximity_result.get('rejection_reason', 'Swing proximity validation failed'),
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        direction=direction,
                        context={
                            **self._collect_market_context(df_trigger, df_4h, df_entry, pip_value,
                                                          ema_result=ema_result, swing_result=swing_result,
                                                          pullback_result=pullback_result, direction=direction),
                            'swing_proximity_distance': proximity_result.get('distance_to_swing'),
                            'nearest_swing_price': proximity_result.get('nearest_swing_price'),
                            'swing_type': proximity_result.get('swing_type')
                        }
                    )
                    return None

                dist_to_swing = proximity_result.get('distance_to_swing')
                if dist_to_swing is not None:
                    self.logger.info(f"   ✅ Distance to swing: {dist_to_swing:.1f} pips")
                else:
                    self.logger.info(f"   ✅ No nearby opposing swing detected")

            # ================================================================
            # v2.0.0: Calculate Limit Entry Price with Offset
            # v2.17.0: Pass epic for per-pair offset override
            # ================================================================
            self.logger.info(f"\n📊 LIMIT ORDER: Calculating Entry Offset")
            if self.limit_order_enabled:
                entry_price, limit_offset_pips = self._calculate_limit_entry(
                    market_price, direction, entry_type, pip_value, entry_df, epic
                )
            else:
                entry_price = market_price
                limit_offset_pips = 0.0
                self._current_api_order_type = 'MARKET'

            # ================================================================
            # v2.17.0: Preliminary Order Type (will be refined after confidence)
            # Store EMA slope for later confidence-based routing decision
            # ================================================================
            ema_slope_strength = abs(ema_result.get('ema_slope_atr', 0))

            # Load routing thresholds from database config
            market_order_min_confidence = 0.65  # Default
            market_order_min_ema_slope = 1.0    # Default
            low_confidence_extra_offset = 2.0   # Default

            if hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
                # _db_config is already the SMCSimpleConfig object (not the service)
                market_order_min_confidence = getattr(self._db_config, 'market_order_min_confidence', 0.65)
                market_order_min_ema_slope = getattr(self._db_config, 'market_order_min_ema_slope', 1.0)
                low_confidence_extra_offset = getattr(self._db_config, 'low_confidence_extra_offset', 2.0)

            # Preliminary order type - will be finalized after confidence calculation
            order_type = 'limit' if self.limit_order_enabled and limit_offset_pips > 0 else 'market'
            self.logger.info(f"   📋 Preliminary Order Type: {order_type.upper()} (offset={limit_offset_pips:.1f} pips)")

            # ================================================================
            # STEP 4: Calculate Stop Loss and Take Profit
            # ================================================================
            self.logger.info(f"\n🛑 STEP 4: Calculating SL/TP")

            # v2.15.0: Check for fixed SL/TP override (per-pair or global)
            # Priority: backtest override > per-pair DB setting > global DB setting > structural calculation
            fixed_sl_pips = None
            fixed_tp_pips = None
            using_fixed_sl_tp = False

            if self.fixed_sl_tp_override_enabled:
                # Priority: scalp mode > backtest override > per-pair DB setting > global DB setting
                # Check if backtest override was applied (indicated by _config_override having SL/TP)
                has_backtest_sl_override = self._config_override and 'fixed_stop_loss_pips' in self._config_override
                has_backtest_tp_override = self._config_override and 'fixed_take_profit_pips' in self._config_override

                if self.scalp_mode_enabled:
                    # SCALP MODE: Check for per-pair optimized SL/TP first (highest priority)
                    # v2.29.0: Per-pair scalp SL/TP based on ATR analysis
                    if self._using_database_config and self._db_config:
                        pair_scalp_sl = self._db_config.get_pair_scalp_sl(epic)
                        pair_scalp_tp = self._db_config.get_pair_scalp_tp(epic)

                        if pair_scalp_sl is not None and pair_scalp_tp is not None:
                            fixed_sl_pips = pair_scalp_sl
                            fixed_tp_pips = pair_scalp_tp
                            self.logger.info(f"   🎯 Using PER-PAIR SCALP SL/TP: SL={fixed_sl_pips:.1f}, TP={fixed_tp_pips:.1f} (ATR-optimized)")
                        else:
                            # Fall back to global scalp values
                            fixed_sl_pips = self.scalp_sl_pips
                            fixed_tp_pips = self.scalp_tp_pips
                            self.logger.info(f"   🎯 Using GLOBAL SCALP SL/TP: SL={fixed_sl_pips:.1f}, TP={fixed_tp_pips:.1f}")
                    else:
                        # Use global scalp values (file config mode)
                        fixed_sl_pips = self.scalp_sl_pips
                        fixed_tp_pips = self.scalp_tp_pips
                        self.logger.info(f"   🎯 Using SCALP MODE SL/TP")
                elif has_backtest_sl_override or has_backtest_tp_override:
                    # Use backtest overrides (highest priority after scalp mode)
                    fixed_sl_pips = self.fixed_stop_loss_pips
                    fixed_tp_pips = self.fixed_take_profit_pips
                    self.logger.info(f"   🧪 Using BACKTEST OVERRIDE SL/TP")
                elif self._using_database_config and self._db_config:
                    # Use per-pair override from database
                    fixed_sl_pips = self._db_config.get_pair_fixed_stop_loss(epic)
                    fixed_tp_pips = self._db_config.get_pair_fixed_take_profit(epic)
                else:
                    # Use instance attributes (set from config)
                    fixed_sl_pips = self.fixed_stop_loss_pips
                    fixed_tp_pips = self.fixed_take_profit_pips

                if fixed_sl_pips is not None and fixed_sl_pips > 0:
                    using_fixed_sl_tp = True
                    self.logger.info(f"   📌 Using FIXED SL/TP: SL={fixed_sl_pips} pips, TP={fixed_tp_pips} pips")

            if using_fixed_sl_tp:
                # Use fixed SL/TP values
                if direction == 'BULL':
                    stop_loss = entry_price - (fixed_sl_pips * pip_value)
                    take_profit = entry_price + (fixed_tp_pips * pip_value) if fixed_tp_pips else None
                else:
                    stop_loss = entry_price + (fixed_sl_pips * pip_value)
                    take_profit = entry_price - (fixed_tp_pips * pip_value) if fixed_tp_pips else None

                risk_pips = fixed_sl_pips
                self.logger.info(f"   Stop Loss: {stop_loss:.5f} ({risk_pips:.1f} pips) [FIXED]")

            else:
                # Original structural SL calculation
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

            # Find take profit (next swing structure) or use fixed TP
            if using_fixed_sl_tp and fixed_tp_pips is not None and fixed_tp_pips > 0:
                # Use fixed TP from config/override
                reward_pips = fixed_tp_pips
                rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                self.logger.info(f"   Take Profit: {take_profit:.5f} ({reward_pips:.1f} pips) [FIXED]")
                self.logger.info(f"   R:R Ratio: {rr_ratio:.2f}")
            else:
                # Calculate TP from market structure
                tp_result = self._calculate_take_profit(
                    df_trigger, direction, entry_price, risk_pips, pip_value
                )

                take_profit = tp_result['take_profit']
                reward_pips = tp_result['reward_pips']
                rr_ratio = tp_result['rr_ratio']

                self.logger.info(f"   Stop Loss: {stop_loss:.5f} ({risk_pips:.1f} pips)")
                self.logger.info(f"   Take Profit: {take_profit:.5f} ({reward_pips:.1f} pips)")
                self.logger.info(f"   R:R Ratio: {rr_ratio:.2f}")

            # Validate R:R - TEMPORARILY DISABLED
            # TODO: Re-enable R:R validation after testing
            # if rr_ratio < self.min_rr_ratio:
            #     reason = f"R:R too low ({rr_ratio:.2f} < {self.min_rr_ratio})"
            #     self.logger.info(f"   ❌ {reason}")
            #     # Track rejection
            #     risk_result = {
            #         'entry_price': entry_price,
            #         'stop_loss': stop_loss,
            #         'take_profit': take_profit,
            #         'risk_pips': risk_pips,
            #         'reward_pips': reward_pips,
            #         'rr_ratio': rr_ratio,
            #     }
            #     self._track_rejection(
            #         stage='RISK_RR',
            #         reason=reason,
            #         epic=epic,
            #         pair=pair,
            #         candle_timestamp=candle_timestamp,
            #         direction=direction,
            #         context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, risk_result=risk_result, direction=direction)
            #     )
            #     return None

            self.logger.info(f"   ⚠️ R:R check DISABLED - ratio: {rr_ratio:.2f} (min was {self.min_rr_ratio})")

            # Validate minimum TP - TEMPORARILY DISABLED
            # TODO: Re-enable TP validation after testing
            # if reward_pips < self.min_tp_pips:
            #     reason = f"TP too small ({reward_pips:.1f} < {self.min_tp_pips} pips)"
            #     self.logger.info(f"   ❌ {reason}")
            #     # Track rejection
            #     risk_result = {
            #         'entry_price': entry_price,
            #         'stop_loss': stop_loss,
            #         'take_profit': take_profit,
            #         'risk_pips': risk_pips,
            #         'reward_pips': reward_pips,
            #         'rr_ratio': rr_ratio,
            #     }
            #     self._track_rejection(
            #         stage='RISK_TP',
            #         reason=reason,
            #         epic=epic,
            #         pair=pair,
            #         candle_timestamp=candle_timestamp,
            #         direction=direction,
            #         context=self._collect_market_context(df_trigger, df_4h, df_entry, pip_value, ema_result=ema_result, swing_result=swing_result, pullback_result=pullback_result, risk_result=risk_result, direction=direction)
            #     )
            #     return None

            self.logger.info(f"   ⚠️ TP check DISABLED - reward: {reward_pips:.1f} pips (min was {self.min_tp_pips})")

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
                # v2.13.0: Use PREVIOUS complete candle for volume ratio, not current incomplete candle
                # The current 15m candle may only have 1-14 of 15 expected 1m candles,
                # making its volume artificially low compared to complete candles in SMA.
                # This was causing ~90% of signals to be rejected with "Volume ratio too low (0.09 < 0.25)"
                if len(df_trigger) >= self.volume_sma_period + 1:
                    # Use previous (complete) candle for current volume comparison
                    current_vol = df_trigger[vol_col].iloc[-2]
                    # SMA uses candles before the current one (all complete)
                    vol_sma = df_trigger[vol_col].iloc[-(self.volume_sma_period + 1):-1].mean()
                else:
                    # Fallback for insufficient data
                    current_vol = df_trigger[vol_col].iloc[-1]
                    vol_sma = df_trigger[vol_col].iloc[-self.volume_sma_period:].mean()
                if vol_sma > 0:
                    volume_ratio = current_vol / vol_sma
                    self.logger.debug(f"   📊 Volume ratio: {volume_ratio:.2f} (prev_candle={current_vol:.0f} / sma20={vol_sma:.0f})")

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

            # v2.35.0: Apply HTF bias confidence multiplier
            # Only applies if htf_bias_confidence_multiplier_enabled=True and we have a valid score
            htf_bias_multiplier = trigger_details.get('htf_bias_confidence_multiplier')
            if htf_bias_multiplier is not None and htf_bias_multiplier != 1.0:
                original_confidence = confidence
                confidence = confidence * htf_bias_multiplier
                confidence = min(confidence, 1.0)  # Cap at 100%
                htf_bias_score = trigger_details.get('htf_bias_score', 0.5)
                self.logger.info(
                    f"   📊 HTF bias multiplier: {htf_bias_multiplier:.2f}x "
                    f"(score={htf_bias_score:.3f}) → {original_confidence*100:.0f}% → {confidence*100:.0f}%"
                )

            # v2.11.0: Dynamic confidence threshold based on EMA distance, ATR, and Volume
            # Priority: Scalp mode > Backtest override > EMA distance > ATR > Volume > Pair-specific > Default
            # EMA distance is the strongest predictor of CONFIDENCE rejection outcomes
            # v2.12.0: Added direction-aware confidence threshold support
            # v2.20.0: Scalp mode always uses scalp_min_confidence (highest priority)
            # v2.31.2: Allow per-pair scalp_min_confidence override (NZDUSD needs 55-65% band)
            # v2.35.2: Use max(scalp_min, pair_min) to never bypass per-pair thresholds
            if self.scalp_mode_enabled:
                # SCALP MODE: Use max of scalp threshold and per-pair threshold
                # This ensures scalp mode never bypasses per-pair optimized thresholds
                pair_scalp_min = self._get_pair_param(epic, 'scalp_min_confidence', None)
                pair_general_min = self._db_config.get_pair_min_confidence(epic) if self._db_config else None

                # Determine scalp threshold (per-pair scalp or global)
                scalp_threshold = pair_scalp_min if pair_scalp_min is not None else self.scalp_min_confidence

                # v2.35.2: Take max of scalp threshold and per-pair min_confidence
                # This prevents scalp mode from taking trades below per-pair minimums
                if pair_general_min is not None and pair_general_min > scalp_threshold:
                    pair_min_confidence = pair_general_min
                    adjustment_type = "scalp-mode-capped"
                    self.logger.info(f"   🎯 Scalp threshold capped by pair min: {pair_general_min*100:.0f}% (scalp was {scalp_threshold*100:.0f}%)")
                elif pair_scalp_min is not None:
                    pair_min_confidence = pair_scalp_min
                    adjustment_type = "scalp-mode-pair"
                    self.logger.info(f"   🎯 Per-pair scalp threshold: {pair_scalp_min*100:.0f}%")
                else:
                    pair_min_confidence = self.scalp_min_confidence
                    adjustment_type = "scalp-mode"
                    self.logger.info(f"   🎯 Scalp mode threshold: {self.scalp_min_confidence*100:.0f}%")
            elif self._backtest_mode and self._config_override and 'min_confidence' in self._config_override:
                # Backtest mode with min_confidence override - use the overridden instance variable
                pair_min_confidence = self.min_confidence
                adjustment_type = "backtest-override"
            elif hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
                pair_min_confidence = self._db_config.get_min_confidence_directional(epic, direction)
                adjustment_type = "pair-specific"
            else:
                pair_min_confidence = self.pair_min_confidence.get(pair, self.pair_min_confidence.get(epic, self.min_confidence))
                adjustment_type = "pair-specific"

            # v2.11.0: EMA distance-adjusted confidence threshold (HIGHEST PRIORITY)
            # Near EMA (trend-aligned) = lower threshold, Far from EMA (extended) = higher threshold
            # SKIP in scalp mode - scalp uses fixed confidence threshold
            if adjustment_type != "scalp-mode" and self.ema_distance_adjusted_confidence_enabled and ema_distance is not None:
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
            # SKIP in scalp mode - scalp uses fixed confidence threshold
            if adjustment_type not in ("scalp-mode", "backtest-override") and self.atr_adjusted_confidence_enabled and atr is not None:
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
                # Check for backtest override first, then pair-specific
                high_vol_confidence = None
                if hasattr(self, '_config_override') and self._config_override and 'high_volume_confidence' in self._config_override:
                    # Use overridden global value for backtest testing
                    high_vol_confidence = self.high_volume_confidence
                else:
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
            # v2.29.0: Skip confidence cap in scalp mode (high confidence = more momentum confirmation)
            # v2.31.2: Allow per-pair scalp max confidence override (NZDUSD high conf = 5% WR!)
            if self._using_database_config and self._db_config:
                pair_max_confidence = self._db_config.get_pair_max_confidence(epic)
            else:
                pair_max_confidence = self._get_pair_param(epic, 'MAX_CONFIDENCE_THRESHOLD', self.max_confidence)

            # v2.31.2: Check per-pair scalp max confidence if configured
            apply_max_confidence_check = not self.scalp_mode_enabled
            if self.scalp_mode_enabled:
                scalp_max_conf = self._get_pair_param(epic, 'scalp_max_confidence', None)
                if scalp_max_conf is not None:
                    pair_max_confidence = scalp_max_conf
                    apply_max_confidence_check = True
                    self.logger.info(f"   ⚠️ Per-pair scalp max confidence: {scalp_max_conf*100:.0f}%")

            if apply_max_confidence_check and round(confidence, 4) > pair_max_confidence:
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
                # Priority: backtest overrides -> direction-specific -> pair parameter_overrides -> global
                if hasattr(self, '_config_override') and self._config_override and 'min_volume_ratio' in self._config_override:
                    # Use the overridden instance variable directly
                    pair_min_volume = self.min_volume_ratio
                elif hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
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
                # CRITICAL FIX (Jan 2026): Backtest overrides should take precedence over database
                # Priority: backtest override -> database config -> instance attribute
                if self._backtest_mode and self._config_override and 'macd_filter_enabled' in self._config_override:
                    # Use the overridden instance variable directly (set by _apply_config_overrides)
                    macd_filter_enabled = self.macd_alignment_filter_enabled
                elif self._config_service and hasattr(self._config_service, 'is_macd_filter_enabled'):
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
            # v2.17.0: FINALIZE ORDER TYPE BASED ON CONFIDENCE
            # Now that confidence is calculated, finalize the order routing decision
            # High confidence + strong trend = market order (immediate entry)
            # Lower confidence = stop order (momentum confirmation needed)
            # ================================================================
            if self.limit_order_enabled:
                if confidence >= market_order_min_confidence and ema_slope_strength >= market_order_min_ema_slope:
                    # High confidence + strong trend = immediate market entry
                    order_type = 'market'
                    entry_price = market_price  # Use current market price
                    limit_offset_pips = 0
                    # Recalculate risk/reward with new entry price
                    if direction == 'BULL':
                        risk_pips = (entry_price - stop_loss) / pip_value
                        reward_pips = (take_profit - entry_price) / pip_value if take_profit else 0
                    else:
                        risk_pips = (stop_loss - entry_price) / pip_value
                        reward_pips = (entry_price - take_profit) / pip_value if take_profit else 0
                    rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                    self.logger.info(f"\n📋 FINAL Order Type: MARKET (high quality: conf={confidence:.1%}, slope={ema_slope_strength:.2f}x ATR)")
                elif order_type == 'limit':
                    # v2.17.0: Add extra offset for low confidence signals (0.50-0.54)
                    # SCALP MODE: Skip extra offset - use fixed 1 pip offset for all signals
                    skip_extra_offset = getattr(self, 'scalp_disable_low_confidence_offset', False)
                    if confidence < 0.55 and low_confidence_extra_offset > 0 and not skip_extra_offset:
                        limit_offset_pips += low_confidence_extra_offset
                        # Recalculate entry price with extra offset
                        extra_offset = low_confidence_extra_offset * pip_value
                        if direction == 'BULL':
                            entry_price += extra_offset
                        else:
                            entry_price -= extra_offset
                        # Recalculate risk/reward with new entry price
                        if direction == 'BULL':
                            risk_pips = (entry_price - stop_loss) / pip_value
                            reward_pips = (take_profit - entry_price) / pip_value if take_profit else 0
                        else:
                            risk_pips = (stop_loss - entry_price) / pip_value
                            reward_pips = (entry_price - take_profit) / pip_value if take_profit else 0
                        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                        self.logger.info(f"\n📋 FINAL Order Type: STOP (low conf: +{low_confidence_extra_offset:.0f} pips extra, total={limit_offset_pips:.1f} pips)")
                    else:
                        self.logger.info(f"\n📋 FINAL Order Type: STOP (conf={confidence:.1%}, offset={limit_offset_pips:.1f} pips)")

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
                'timeframe': self.trigger_tf,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,

                # v2.35.1: Enable S/R validation for scalp mode with per-pair tolerance
                # Data showed trades <6 pips from opposing S/R had 27% win rate vs 46% for 10+ pips
                'skip_sr_validation': False,
                # v2.35.1: Per-pair S/R tolerance (5-8 pips based on volatility)
                'sr_tolerance_pips': self._db_config.get_pair_scalp_sr_tolerance(epic) if self.scalp_mode_enabled else None,
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

                # v2.23.0: Trigger type tracking for signal analytics
                'trigger_type': trigger_type,  # e.g., SWING_PULLBACK, SWING_OPTIMAL+PIN
                'trigger_details': trigger_details,  # Detailed breakdown of entry mechanism
                'pattern_type': pattern_data.get('pattern_type') if pattern_data else None,
                'pattern_strength': pattern_data.get('strength') if pattern_data else None,
                'rsi_divergence_detected': rsi_divergence_data is not None and rsi_divergence_data.get('detected', False),
                'rsi_divergence_type': rsi_divergence_data.get('type') if rsi_divergence_data else None,
                'macd_aligned': tier1_macd_aligned,  # v2.23.0: TIER 1 MACD alignment status

                # v2.17.0: 4H candle direction for HTF momentum analysis
                'htf_candle_direction': htf_candle_direction,
                'htf_candle_direction_prev': htf_candle_direction_prev,

                # v2.35.0: HTF Bias Score System (replaces binary HTF alignment)
                'htf_bias_score': trigger_details.get('htf_bias_score'),
                'htf_bias_mode': trigger_details.get('htf_bias_mode', 'disabled'),
                'htf_bias_details': trigger_details.get('htf_bias_details'),

                # v2.0.0: Limit order fields
                # v3.3.0: Added api_order_type (LIMIT vs STOP) and signal_price for slippage tracking
                'order_type': order_type,  # 'limit' or 'market'
                'market_price': market_price,  # Current market price (before offset)
                'signal_price': market_price,  # v3.3.0: Original signal price for slippage analysis
                'limit_offset_pips': round(limit_offset_pips, 1),  # Offset from market price
                'limit_expiry_minutes': self.limit_expiry_minutes if order_type == 'limit' else None,
                'api_order_type': getattr(self, '_current_api_order_type', 'STOP'),  # v3.3.0: LIMIT or STOP

                # Technical indicators for analytics (alert_history compatibility)
                'atr': round(atr, 6) if atr else 0.0,
                'volume_ratio': round(volume_ratio, 2),
                'macd_line': round(macd_line, 6),
                'macd_signal': round(macd_signal, 6),
                'macd_histogram': round(macd_histogram, 6),

                # v2.26.0: Filter metadata for performance analysis
                'filter_metadata': {
                    'volume_filter_enabled': self.volume_filter_enabled,
                    'min_volume_ratio_threshold': pair_min_volume if self.volume_filter_enabled else None,
                    'macd_filter_enabled': self._db_config.is_macd_filter_enabled(epic) if hasattr(self, '_db_config') and self._db_config else False,
                    'entry_candle_alignment_required': scalp_require_entry_alignment if 'scalp_require_entry_alignment' in locals() else False,
                    'entry_candle_alignment_confirmed': getattr(self, '_scalp_entry_alignment_confirmed', False),
                    'rejection_candle_required': self.scalp_require_rejection_candle if self.scalp_mode_enabled else False,
                    'rejection_candle_confirmed': getattr(self, '_scalp_rejection_confirmed', False),
                    'market_order_reason': 'entry_alignment' if getattr(self, '_scalp_entry_alignment_confirmed', False) else ('rejection_candle' if getattr(self, '_scalp_rejection_confirmed', False) else 'default'),
                    'reversal_override_applied': reversal_override_applied,
                    'reversal_override_reason': reversal_override_reason,
                },

                # Strategy indicators (for alert_history compatibility)
                'strategy_indicators': {
                    'tier1_ema': {
                        'timeframe': self.htf_timeframe,
                        'ema_period': self.ema_period,
                        'ema_value': ema_value,
                        'distance_pips': ema_distance,
                        'direction': direction,
                        'htf_candle_direction': htf_candle_direction,
                        'htf_candle_direction_prev': htf_candle_direction_prev
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
                        'ema_alignment': min(ema_distance / 30, 1.0) * 0.20,
                        'volume_bonus': 0.10 if volume_confirmed else 0.04,
                        'pullback_quality': self._calc_pullback_score_for_log(pullback_depth, in_optimal_zone),
                        'rr_quality': min(rr_ratio / 3.0, 1.0) * 0.20,
                        'fib_accuracy': 1.0 - min(abs(pullback_depth - 0.382) / 0.382, 1.0),
                        'deep_pullback_penalty': 'yes' if pullback_depth > 0.60 else 'no'
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
                    'atr_pips': round(atr / pip_value, 1) if atr else 0.0,
                    # v2.23.0: Trigger type tracking
                    'trigger_type': trigger_type,
                    'trigger_details': trigger_details,
                    'pattern_confirmation': pattern_data,
                    'rsi_divergence': rsi_divergence_data,
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

            # Cooldown handling differs between live and backtest modes:
            # - BACKTEST: Update in-memory cooldown immediately (no external validation pipeline)
            # - LIVE: Cooldowns read directly from alert_history database (single source of truth)
            #   No in-memory update needed - database is updated when alert is saved
            if self._backtest_mode:
                self._update_cooldown_backtest(pair, candle_dt)

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
            # Skip in backtest mode to improve performance (unless chart_mode enabled)
            # ================================================================
            if not self._backtest_mode or getattr(self, '_chart_mode_enabled', False):
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

            # ================================================================
            # PAIR SCALP FILTERS (v2.31.0)
            # Per-pair configurable filters for scalp trades based on
            # Jan 2026 NZDUSD trade analysis. Checks efficiency ratio,
            # regime, session hours, MACD alignment, EMA stack.
            # Only active for pairs with filters configured in database.
            # ================================================================
            # v2.31.1: Track signal detected (before filters) for backtest stats
            self._track_signal_detected()

            if self.scalp_mode_enabled:
                filter_passed, filter_reason = self._apply_pair_scalp_filters(
                    epic=epic,
                    signal=signal,
                    candle_timestamp=candle_timestamp
                )
                if not filter_passed:
                    self.logger.info(f"\n⚠️ PAIR_SCALP_FILTER rejected: {filter_reason}")
                    self._track_rejection(
                        stage='PAIR_SCALP_FILTER',
                        reason=filter_reason,
                        epic=epic,
                        pair=pair,
                        candle_timestamp=candle_timestamp,
                        direction=direction,
                        context={
                            'efficiency_ratio': signal.get('efficiency_ratio'),
                            'market_regime_detected': signal.get('market_regime_detected'),
                            'macd_histogram': signal.get('macd_histogram'),
                            'ema_stack_order': signal.get('ema_stack_order'),
                            **self._collect_market_context(df_trigger, df_4h, entry_df, pip_value, direction=direction)
                        }
                    )
                    return None

            # ================================================================
            # SCALP SIGNAL QUALIFICATION (v2.21.0)
            # Run momentum confirmation filters on scalp signals
            # Mode: MONITORING (logs only) or ACTIVE (blocks signals)
            # v2.24.1: Per-pair qualification mode support
            # ================================================================
            if self.scalp_mode_enabled and self._signal_qualifier and self._signal_qualifier.enabled:
                try:
                    # v2.24.1: Get per-pair qualification mode (JPY pairs use ACTIVE, others use global)
                    effective_mode = self._signal_qualifier.mode  # Default to global mode
                    if hasattr(self, '_db_config') and self._db_config:
                        pair_mode = self._db_config.get_pair_scalp_qualification_mode(epic)
                        if pair_mode:
                            effective_mode = pair_mode
                            self._signal_qualifier.mode = pair_mode  # Temporarily set for this signal

                    qual_passed, qual_score, qual_results = self._signal_qualifier.qualify_signal(
                        signal=signal,
                        df_entry=entry_df,
                        df_trigger=df_trigger
                    )

                    # Add qualification data to signal for logging/analysis
                    signal['qualification_score'] = qual_score
                    signal['qualification_results'] = qual_results
                    signal['qualification_mode'] = effective_mode

                    if not qual_passed:
                        self.logger.info(f"\n⚠️ SIGNAL BLOCKED BY QUALIFICATION")
                        self.logger.info(f"   Score: {qual_score:.0%} (min: {self._signal_qualifier.min_score:.0%})")
                        self._track_rejection(
                            stage='SCALP_QUALIFICATION',
                            reason=f'Score {qual_score:.0%} below threshold {self._signal_qualifier.min_score:.0%}',
                            epic=epic,
                            pair=pair,
                            candle_timestamp=candle_timestamp,
                            direction=direction,
                            context={
                                'qualification_score': qual_score,
                                'qualification_results': qual_results,
                                **self._collect_market_context(df_trigger, df_4h, entry_df, pip_value, direction=direction)
                            }
                        )
                        return None
                    else:
                        self.logger.info(f"\n✅ QUALIFICATION: {qual_score:.0%} ({self._signal_qualifier.mode} mode)")

                except Exception as qual_error:
                    # Don't fail the signal if qualification fails
                    self.logger.warning(f"⚠️ Signal qualification failed: {qual_error}")

            # v2.31.1: Track signal passed all filters for backtest stats
            self._track_signal_passed()
            return signal

        except Exception as e:
            self.logger.error(f"❌ Error detecting SMC Simple signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_candle_direction(self, open_price: float, close_price: float) -> str:
        """
        Determine candle direction based on open/close prices.

        Returns:
            'BULLISH' if close > open
            'BEARISH' if close < open
            'NEUTRAL' if close == open (doji)
        """
        if close_price > open_price:
            return 'BULLISH'
        elif close_price < open_price:
            return 'BEARISH'
        return 'NEUTRAL'

    def _check_ema_bias(self, df_4h: pd.DataFrame, pip_value: float, epic: str = None) -> Dict:
        """
        TIER 1: Check 4H 50 EMA directional bias

        v2.16.0: Added EMA slope validation to prevent counter-trend trades.
        Price above a FALLING EMA = bearish retest (not bullish continuation).
        Price below a RISING EMA = bullish retest (not bearish continuation).

        Returns:
            Dict with: valid, direction, ema_value, distance_pips, ema_slope_atr, reason
        """
        min_bars_needed = self.ema_period + self.ema_slope_lookback_bars + 1
        if len(df_4h) < min_bars_needed:
            return {
                'valid': False,
                'reason': f"Insufficient 4H data ({len(df_4h)} < {min_bars_needed} bars)"
            }

        # Calculate 50 EMA
        ema = df_4h['close'].ewm(span=self.ema_period, adjust=False).mean()
        ema_value = ema.iloc[-1]
        current_close = df_4h['close'].iloc[-1]

        # Calculate distance in pips
        distance_pips = abs(current_close - ema_value) / pip_value

        # Check if price is beyond EMA buffer zone
        # v2.31.2: Use per-pair EMA buffer override if configured (e.g., NZDUSD needs smaller buffer)
        effective_ema_buffer = self.ema_buffer_pips
        if epic and self.scalp_mode_enabled and hasattr(self, '_config_service') and self._config_service:
            pair_buffer = self._config_service.get_pair_scalp_ema_buffer_pips(epic)
            if pair_buffer is not None:
                effective_ema_buffer = pair_buffer
                self.logger.info(f"   📏 Per-pair EMA buffer: {effective_ema_buffer} pips for {epic}")

        buffer = effective_ema_buffer * pip_value

        if current_close > ema_value + buffer:
            direction = 'BULL'
        elif current_close < ema_value - buffer:
            direction = 'BEAR'
        else:
            return {
                'valid': False,
                'reason': f"Price in EMA buffer zone ({distance_pips:.1f} pips, need >{effective_ema_buffer})"
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

        # ================================================================
        # v2.16.0: EMA SLOPE VALIDATION (prevents counter-trend trades)
        # ================================================================
        # This check ensures the EMA itself is trending in the direction of the trade.
        # Without this, we take losing trades during bearish retests of falling EMAs.
        ema_slope_atr = 0.0
        if self.ema_slope_validation_enabled:
            # Calculate ATR for the 4H timeframe
            atr_4h = self._calculate_atr(df_4h)
            if atr_4h is None or atr_4h <= 0:
                # Fallback: use average range if ATR calculation fails
                atr_4h = (df_4h['high'] - df_4h['low']).tail(14).mean()

            # Calculate EMA slope over lookback period
            ema_current = ema.iloc[-1]
            lookback = int(self.ema_slope_lookback_bars)  # Ensure integer for iloc indexing
            ema_previous = ema.iloc[-lookback - 1]
            ema_slope = ema_current - ema_previous

            # Express slope as multiple of ATR
            ema_slope_atr = ema_slope / atr_4h if atr_4h > 0 else 0.0

            # Always log slope for debugging
            self.logger.info(f"   📐 EMA slope: {ema_slope_atr:.3f}x ATR, direction: {direction}")

            # v2.16.0 Option 2: Asymmetric validation
            # BULL requires EMA to be rising (slope >= 0) or flat
            # BEAR requires EMA to be falling (slope <= 0) or flat
            if direction == 'BULL' and ema_slope_atr < 0:
                # Price is above EMA but EMA is falling = bearish retest, NOT bullish
                return {
                    'valid': False,
                    'rejection_type': 'EMA_SLOPE',  # v2.16.0: Track slope rejections separately
                    'ema_slope_atr': ema_slope_atr,
                    'attempted_direction': direction,
                    'reason': f"BULL rejected: EMA is FALLING (slope: {ema_slope_atr:.3f}x ATR) - need rising EMA for BULL"
                }
            elif direction == 'BEAR' and ema_slope_atr > 0:
                # Price is below EMA but EMA is rising = bullish retest, NOT bearish
                return {
                    'valid': False,
                    'rejection_type': 'EMA_SLOPE',  # v2.16.0: Track slope rejections separately
                    'ema_slope_atr': ema_slope_atr,
                    'attempted_direction': direction,
                    'reason': f"BEAR rejected: EMA is RISING (slope: {ema_slope_atr:.3f}x ATR) - need falling EMA for BEAR"
                }

        # ================================================================
        # v2.23.0: OPTIONAL MACD ALIGNMENT CHECK (TIER 1 Enhancement)
        # ================================================================
        # When enabled, checks if MACD histogram aligns with trade direction
        # BULL: histogram > 0 and rising
        # BEAR: histogram < 0 and falling
        macd_aligned = None
        macd_alignment_boost = 0.0
        if getattr(self, 'macd_alignment_enabled', False) and 'macd_histogram' in df_4h.columns:
            try:
                macd_hist = df_4h['macd_histogram'].iloc[-1]
                macd_hist_prev = df_4h['macd_histogram'].iloc[-2]

                if direction == 'BULL':
                    # BULL alignment: histogram > 0 and rising (momentum increasing)
                    macd_aligned = macd_hist > 0 and macd_hist > macd_hist_prev
                else:
                    # BEAR alignment: histogram < 0 and falling (momentum decreasing)
                    macd_aligned = macd_hist < 0 and macd_hist < macd_hist_prev

                self.logger.info(f"   📉 MACD alignment: {'✅ aligned' if macd_aligned else '⚠️ not aligned'} "
                               f"(hist={macd_hist:.6f}, prev={macd_hist_prev:.6f})")

                # In strict mode (macd_alignment_required=True), reject if not aligned
                if getattr(self, 'macd_alignment_required', False) and not macd_aligned:
                    return {
                        'valid': False,
                        'rejection_type': 'MACD_ALIGNMENT',
                        'attempted_direction': direction,
                        'macd_histogram': macd_hist,
                        'macd_aligned': macd_aligned,
                        'reason': f"{direction} rejected: MACD not aligned (hist={macd_hist:.6f})"
                    }

                # Add confidence boost if aligned
                if macd_aligned:
                    macd_alignment_boost = getattr(self, 'macd_alignment_confidence_boost', 0.05)

            except Exception as e:
                self.logger.debug(f"   ⚠️ MACD alignment check failed: {e}")
                macd_aligned = None

        return {
            'valid': True,
            'direction': direction,
            'ema_value': ema_value,
            'distance_pips': distance_pips,
            'ema_slope_atr': ema_slope_atr,
            'macd_aligned': macd_aligned,
            'macd_alignment_boost': macd_alignment_boost,
            'reason': f"{direction} bias confirmed (EMA slope: {ema_slope_atr:.2f}x ATR)"
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

            # Scalp mode: allow near-breaks with tolerance (0.5 pips)
            scalp_tolerance = getattr(self, 'scalp_swing_break_tolerance_pips', 0) * pip_value if self.scalp_mode_enabled else 0

            for swing_idx, swing_level in recent_highs:
                # Check all candles AFTER this swing for a break
                # In scalp mode, allow breaks within tolerance (near-breaks)
                break_threshold = swing_level - scalp_tolerance
                for check_idx in range(swing_idx + 1, current_idx + 1):
                    if highs[check_idx] > break_threshold:
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
                # In scalp mode, also check with tolerance
                if self.scalp_mode_enabled and gap_pips <= getattr(self, 'scalp_swing_break_tolerance_pips', 0):
                    break_found = True
                    best_swing_level = swing_level
                    best_swing_idx = highest_swing[0]
                    best_break_idx = current_idx
                else:
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

            # Scalp mode: allow near-breaks with tolerance (0.5 pips)
            scalp_tolerance = getattr(self, 'scalp_swing_break_tolerance_pips', 0) * pip_value if self.scalp_mode_enabled else 0

            for swing_idx, swing_level in recent_lows:
                # Check all candles AFTER this swing for a break
                # In scalp mode, allow breaks within tolerance (near-breaks)
                break_threshold = swing_level + scalp_tolerance
                for check_idx in range(swing_idx + 1, current_idx + 1):
                    if lows[check_idx] < break_threshold:
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
                # In scalp mode, also check with tolerance
                if self.scalp_mode_enabled and gap_pips <= getattr(self, 'scalp_swing_break_tolerance_pips', 0):
                    break_found = True
                    best_swing_level = swing_level
                    best_swing_idx = lowest_swing[0]
                    best_break_idx = current_idx
                else:
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
        # v2.14.0: Fix for incomplete candle volume issue (same fix as volume_ratio in v2.13.0)
        # If the break candle is the current (incomplete) candle, use the previous complete candle instead
        # to avoid artificially low volume readings from partial candles
        volume_confirmed = False
        if self.volume_enabled and volumes is not None:
            current_idx = len(df) - 1

            # Determine which candle to use for volume confirmation
            if break_candle_idx == current_idx:
                # Break just happened on current incomplete candle - use previous complete candle
                vol_check_idx = break_candle_idx - 1 if break_candle_idx > 0 else break_candle_idx
            else:
                # Break happened on a previous (complete) candle - use it directly
                vol_check_idx = break_candle_idx

            # Calculate SMA from candles before the volume check candle (all complete)
            vol_sma = np.mean(volumes[max(0, vol_check_idx-self.volume_sma_period):vol_check_idx])
            check_vol = volumes[vol_check_idx]

            if vol_sma > 0:
                volume_confirmed = check_vol > vol_sma * self.volume_multiplier

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

        # v2.19.0: SCALP MODE - Use micro-pullback on entry timeframe (1m)
        # Instead of 5m swing levels, detect recent high/low on 1m data
        if self.scalp_mode_enabled and self.fib_min > 0:
            # Get micro-swing lookback (configurable, default 10 bars = 10 minutes on 1m)
            micro_lookback = getattr(self, 'scalp_micro_pullback_lookback', 10)
            lookback_bars = min(micro_lookback, len(df) - 1)

            if lookback_bars < 3:
                return {
                    'valid': False,
                    'reason': f"Insufficient 1m data for micro-pullback ({lookback_bars} bars)"
                }

            recent_df = df.iloc[-lookback_bars:]
            micro_high = recent_df['high'].max()
            micro_low = recent_df['low'].min()
            micro_range = micro_high - micro_low

            if micro_range <= 0:
                return {
                    'valid': False,
                    'reason': "No price movement in micro-range"
                }

            # Calculate micro-pullback depth
            if direction == 'BULL':
                # For BUY: want price to have pulled back from recent high
                # 0% = at micro_high (chasing), 100% = at micro_low
                micro_pullback = (micro_high - current_close) / micro_range
            else:  # BEAR
                # For SELL: want price to have pulled back from recent low
                # 0% = at micro_low (chasing), 100% = at micro_high
                micro_pullback = (current_close - micro_low) / micro_range

            micro_range_pips = micro_range / pip_value

            # v2.24.1: Get per-pair scalp fib thresholds (with fallback to global)
            pair_fib_min = self.fib_min  # Default to global
            pair_fib_max = self.fib_max
            if epic and hasattr(self, '_db_config') and self._db_config:
                pair_override = self._db_config.get_pair_scalp_fib_pullback_min(epic)
                if pair_override is not None:
                    pair_fib_min = pair_override

            if self.debug_logging:
                self.logger.debug(f"   Micro-pullback ({lookback_bars} bars): high={micro_high:.5f}, low={micro_low:.5f}")
                self.logger.debug(f"   Current: {current_close:.5f}, micro_depth={micro_pullback*100:.1f}%")

            # Validate micro-pullback
            if micro_pullback < pair_fib_min:
                return {
                    'valid': False,
                    'reason': f"Micro-pullback too shallow ({micro_pullback*100:.1f}% < {pair_fib_min*100:.1f}% on 1m)"
                }
            if micro_pullback > pair_fib_max:
                return {
                    'valid': False,
                    'reason': f"Micro-pullback too deep ({micro_pullback*100:.1f}% > {pair_fib_max*100:.1f}% on 1m)"
                }

            # Micro-pullback is valid
            # v2.23.0: Add trigger type for tracking
            trigger_type = TRIGGER_SWING_PULLBACK  # Micro-pullback is a form of pullback
            return {
                'valid': True,
                'entry_price': entry_price,
                'pullback_depth': micro_pullback,
                'in_optimal_zone': 0.10 <= micro_pullback <= 0.20,  # Optimal micro zone
                'entry_type': 'MICRO_PULLBACK',
                'trigger_type': trigger_type,
                'swing_range_pips': micro_range_pips,
                'reason': f"Micro-pullback valid ({micro_pullback*100:.1f}% on 1m)"
            }

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
        # Priority: scalp mode override -> direction-specific -> pair parameter_overrides -> global
        # FIX: Scalp mode sets self.momentum_min_depth to scalp_momentum_min_depth, so use it as floor
        if hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
            db_momentum_min = self._db_config.get_momentum_min_depth(epic, direction)
            # In scalp mode, use the more permissive value (lower = allows more extension)
            # self.momentum_min_depth is set to scalp_momentum_min_depth (-0.30) in scalp mode
            pair_momentum_min = min(db_momentum_min, self.momentum_min_depth)
        else:
            pair_momentum_min = self._get_pair_param(epic, 'MOMENTUM_MIN_DEPTH', self.momentum_min_depth)

        # Check for momentum continuation entry (price beyond break point)
        if pullback_depth < 0:
            if self.momentum_mode_enabled:
                # v1.8.0: Allow momentum entries within configured range
                # v2.12.0: Use direction-aware momentum_min_depth
                if pair_momentum_min <= pullback_depth <= self.momentum_max_depth:

                    # v2.18.0: ATR-based extension filter (prevents chasing extended moves)
                    if self.max_extension_atr_enabled:
                        atr = self._calculate_atr(df)
                        if atr > 0 and swing_range > 0:
                            # Calculate extension in ATR units
                            extension_distance = abs(pullback_depth) * swing_range
                            extension_atr = extension_distance / atr

                            if extension_atr > self.max_extension_atr:
                                extension_pips = extension_distance / pip_value
                                max_pips = (self.max_extension_atr * atr) / pip_value
                                return {
                                    'valid': False,
                                    'reason': f"Extension too far ({extension_atr:.2f} ATR / {extension_pips:.1f} pips > {self.max_extension_atr} ATR / {max_pips:.1f} pips max)"
                                }

                    # v2.18.0: Momentum staleness filter (rejects old swing breaks)
                    if self.momentum_staleness_enabled and break_candle:
                        bars_since_break = break_candle.get('bars_ago', 0)
                        if bars_since_break > self.max_momentum_staleness_bars:
                            return {
                                'valid': False,
                                'reason': f"Momentum entry too stale ({bars_since_break} bars since break > {self.max_momentum_staleness_bars} max)"
                            }

                    entry_type = 'MOMENTUM'
                    # v2.23.0: Add trigger type for tracking
                    trigger_type = TRIGGER_SWING_MOMENTUM
                    # Momentum entries are not in "optimal" Fib zone
                    return {
                        'valid': True,
                        'entry_price': entry_price,
                        'pullback_depth': pullback_depth,
                        'in_optimal_zone': False,
                        'entry_type': entry_type,
                        'trigger_type': trigger_type,
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
        # Priority: scalp mode -> backtest overrides -> direction-specific -> pair parameter_overrides -> global
        # Scalp mode uses relaxed fib settings (0-100%) - skip database lookup
        if self.scalp_mode_enabled:
            # SCALP MODE: Use scalp-configured fib settings (highest priority)
            pair_fib_min = self.fib_min  # Set to 0.0 in _configure_scalp_mode
            pair_fib_max = self.fib_max  # Set to 1.0 in _configure_scalp_mode
        elif hasattr(self, '_config_override') and self._config_override and \
           ('fib_pullback_min' in self._config_override or 'fib_min' in self._config_override or
            'fib_pullback_max' in self._config_override or 'fib_max' in self._config_override):
            # Use the overridden instance variables directly
            pair_fib_min = self.fib_min
            pair_fib_max = self.fib_max
        elif hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
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

        # v2.23.0: Determine trigger type based on Fib zone
        if in_optimal:
            trigger_type = TRIGGER_SWING_OPTIMAL  # Golden zone (38.2%-61.8%)
        else:
            trigger_type = TRIGGER_SWING_PULLBACK  # Standard Fib zone

        return {
            'valid': True,
            'entry_price': entry_price,
            'pullback_depth': pullback_depth,
            'in_optimal_zone': in_optimal,
            'entry_type': entry_type,
            'trigger_type': trigger_type,
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
        # v2.35.3: Added progressive penalty for deep pullbacks (>60%)
        # Perfect: In optimal zone (38.2-61.8%) AND close to 38.2%
        # Good: In optimal zone OR close to 38.2%
        # Acceptable: In valid zone (23.6-70%)
        # Poor: Edge of zone or deep pullback (>60%)

        fib_accuracy = 1.0 - min(abs(pullback_depth - 0.382) / 0.382, 1.0)

        if in_optimal_zone and fib_accuracy > 0.7:
            # Perfect: optimal zone + close to golden ratio
            pullback_score = 1.0 * 0.20
        elif in_optimal_zone:
            # Good: in optimal zone
            pullback_score = 0.8 * 0.20
        elif self.fib_min <= pullback_depth <= self.fib_max:
            # Acceptable: in valid zone, scale by Fib accuracy
            base_score = (0.4 + fib_accuracy * 0.3) * 0.20

            # v2.35.3: Progressive penalty for deep pullbacks
            # 60-65% = 20% penalty, 65-70% = 40% penalty, 70-75% = 60% penalty
            if pullback_depth > 0.70:
                # Very deep: likely failed breakout returning to origin
                pullback_score = base_score * 0.40  # 60% penalty
            elif pullback_depth > 0.65:
                # Deep: weak momentum, risky entry
                pullback_score = base_score * 0.60  # 40% penalty
            elif pullback_depth > 0.60:
                # Moderately deep: some concern
                pullback_score = base_score * 0.80  # 20% penalty
            else:
                pullback_score = base_score
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

    def _calc_pullback_score_for_log(self, pullback_depth: float, in_optimal_zone: bool) -> float:
        """
        Calculate pullback score for logging (mirrors actual confidence calculation)
        v2.35.3: Added to accurately log pullback quality with deep pullback penalties
        """
        fib_accuracy = 1.0 - min(abs(pullback_depth - 0.382) / 0.382, 1.0)

        if in_optimal_zone and fib_accuracy > 0.7:
            return 1.0 * 0.20
        elif in_optimal_zone:
            return 0.8 * 0.20
        elif self.fib_min <= pullback_depth <= self.fib_max:
            base_score = (0.4 + fib_accuracy * 0.3) * 0.20
            # Apply deep pullback penalty
            if pullback_depth > 0.70:
                return base_score * 0.40
            elif pullback_depth > 0.65:
                return base_score * 0.60
            elif pullback_depth > 0.60:
                return base_score * 0.80
            return base_score
        return 0.2 * 0.20

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

    def _check_rsi_divergence(
        self,
        df: pd.DataFrame,
        direction: str,
        lookback: int = None
    ) -> Dict:
        """
        Detect RSI divergence for momentum confirmation (v2.23.0).

        Divergence occurs when price and RSI move in opposite directions:
        - Bullish Divergence: Price makes lower low, RSI makes higher low
        - Bearish Divergence: Price makes higher high, RSI makes lower high

        Args:
            df: OHLCV DataFrame with 'rsi' column
            direction: Expected trade direction ('BULL' or 'BEAR')
            lookback: Number of bars to look back for divergence (default from config)

        Returns:
            Dict with:
                - detected: bool - Whether divergence was found
                - type: str - 'bullish_divergence' or 'bearish_divergence' or None
                - strength: float - Divergence strength (0-1)
                - confidence_boost: float - Suggested confidence boost
                - reason: str - Description of finding
        """
        if lookback is None:
            lookback = getattr(self, 'rsi_divergence_lookback', 20)

        # Check for RSI column
        if 'rsi' not in df.columns or len(df) < lookback:
            return {
                'detected': False,
                'type': None,
                'strength': 0.0,
                'confidence_boost': 0.0,
                'reason': 'Insufficient data for RSI divergence detection'
            }

        # Get RSI and price data for lookback period
        rsi = df['rsi'].iloc[-lookback:].values
        highs = df['high'].iloc[-lookback:].values
        lows = df['low'].iloc[-lookback:].values

        # Find swing points in both price and RSI
        # Simple approach: find local minima/maxima
        min_swing_strength = 2  # Bars on each side to qualify as swing

        if direction == 'BULL':
            # Look for bullish divergence: price lower low + RSI higher low
            price_lows = self._find_swing_points_simple(lows, 'low', min_swing_strength)
            rsi_lows = self._find_swing_points_simple(rsi, 'low', min_swing_strength)

            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                # Compare last two swing lows
                price_ll = price_lows[-1][1] < price_lows[-2][1]  # Price made lower low
                rsi_hl = rsi_lows[-1][1] > rsi_lows[-2][1]  # RSI made higher low

                if price_ll and rsi_hl:
                    # Calculate divergence strength based on RSI difference
                    rsi_diff = rsi_lows[-1][1] - rsi_lows[-2][1]
                    strength = min(1.0, abs(rsi_diff) / 15.0)  # Normalize to 0-1
                    min_strength = getattr(self, 'rsi_divergence_min_strength', 0.30)

                    if strength >= min_strength:
                        return {
                            'detected': True,
                            'type': 'bullish_divergence',
                            'strength': round(strength, 3),
                            'confidence_boost': getattr(self, 'rsi_divergence_confidence_boost', 0.08),
                            'reason': f'Bullish RSI divergence: price lower low, RSI higher low (+{rsi_diff:.1f})'
                        }

        elif direction == 'BEAR':
            # Look for bearish divergence: price higher high + RSI lower high
            price_highs = self._find_swing_points_simple(highs, 'high', min_swing_strength)
            rsi_highs = self._find_swing_points_simple(rsi, 'high', min_swing_strength)

            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                # Compare last two swing highs
                price_hh = price_highs[-1][1] > price_highs[-2][1]  # Price made higher high
                rsi_lh = rsi_highs[-1][1] < rsi_highs[-2][1]  # RSI made lower high

                if price_hh and rsi_lh:
                    # Calculate divergence strength based on RSI difference
                    rsi_diff = rsi_highs[-2][1] - rsi_highs[-1][1]
                    strength = min(1.0, abs(rsi_diff) / 15.0)  # Normalize to 0-1
                    min_strength = getattr(self, 'rsi_divergence_min_strength', 0.30)

                    if strength >= min_strength:
                        return {
                            'detected': True,
                            'type': 'bearish_divergence',
                            'strength': round(strength, 3),
                            'confidence_boost': getattr(self, 'rsi_divergence_confidence_boost', 0.08),
                            'reason': f'Bearish RSI divergence: price higher high, RSI lower high (-{rsi_diff:.1f})'
                        }

        return {
            'detected': False,
            'type': None,
            'strength': 0.0,
            'confidence_boost': 0.0,
            'reason': 'No divergence detected'
        }

    def _find_swing_points_simple(
        self,
        data: np.ndarray,
        swing_type: str,
        strength: int = 2
    ) -> List[Tuple[int, float]]:
        """
        Find swing points (local extrema) in a data array.

        Args:
            data: 1D numpy array of values
            swing_type: 'high' for swing highs, 'low' for swing lows
            strength: Number of bars on each side to confirm swing

        Returns:
            List of (index, value) tuples for detected swing points
        """
        swings = []
        n = len(data)

        for i in range(strength, n - strength):
            if swing_type == 'high':
                # Check if this is a swing high
                is_swing = all(data[i] >= data[i-j] for j in range(1, strength+1)) and \
                          all(data[i] >= data[i+j] for j in range(1, strength+1))
            else:
                # Check if this is a swing low
                is_swing = all(data[i] <= data[i-j] for j in range(1, strength+1)) and \
                          all(data[i] <= data[i+j] for j in range(1, strength+1))

            if is_swing:
                swings.append((i, float(data[i])))

        return swings

    def _calculate_limit_entry(
        self,
        current_close: float,
        direction: str,
        entry_type: str,
        pip_value: float,
        df: pd.DataFrame,
        epic: str = None
    ) -> Tuple[float, float]:
        """
        Calculate limit entry price with offset for momentum confirmation.

        v2.2.0: Stop-entry style - confirm price is moving in intended direction:
        - BUY orders placed ABOVE current price (enter when price breaks up)
        - SELL orders placed BELOW current price (enter when price breaks down)

        v2.17.0: Per-pair stop offset support - choppy pairs (AUDUSD, NZDUSD) use
        wider offset (5 pips) while trending pairs use standard offset (3 pips).

        Args:
            current_close: Current market price
            direction: Trade direction ('BULL' or 'BEAR')
            entry_type: Entry type ('PULLBACK' or 'MOMENTUM')
            pip_value: Pip value for the pair (e.g., 0.0001 for EURUSD)
            df: DataFrame for ATR calculation
            epic: Trading pair epic for per-pair offset override (v2.17.0)

        Returns:
            Tuple of (limit_entry_price, offset_pips)
        """
        if not self.limit_order_enabled:
            # Limit orders disabled - return current price (market order behavior)
            return current_close, 0.0

        # v2.25.0: Check for rejection candle confirmed - use market order
        # When scalp_require_rejection_candle is enabled and a rejection candle is detected,
        # the candle itself provides momentum confirmation, so we use market order instead of STOP
        if self.scalp_mode_enabled and getattr(self, '_scalp_rejection_confirmed', False):
            self.logger.info(f"   📍 Market price: {current_close:.5f}")
            self.logger.info(f"   📍 MARKET order (rejection candle confirmation)")
            self._current_api_order_type = 'MARKET'
            return current_close, 0.0

        # v2.25.1: Check for entry candle alignment confirmed - use market order
        # When scalp_require_entry_candle_alignment is enabled and candle color matches direction,
        # the aligned candle provides momentum confirmation, so we use market order instead of STOP
        if self.scalp_mode_enabled and getattr(self, '_scalp_entry_alignment_confirmed', False):
            self.logger.info(f"   📍 Market price: {current_close:.5f}")
            self.logger.info(f"   📍 MARKET order (entry candle alignment confirmation)")
            self._current_api_order_type = 'MARKET'
            return current_close, 0.0

        # v2.17.0: Check for per-pair stop offset override first
        # v2.22.0: Scalp mode now supports per-pair scalp_limit_offset_pips from database
        pair_offset = None
        if self.scalp_mode_enabled:
            # Scalp mode: check for per-pair scalp_limit_offset_pips override
            if epic and hasattr(self, '_db_config') and self._db_config:
                scalp_pair_offset = self._db_config.get_pair_scalp_limit_offset(epic)
                if scalp_pair_offset is not None:
                    pair_offset = scalp_pair_offset
                    offset_pips = pair_offset
                    self.logger.info(f"   📉 Scalp limit offset: {offset_pips:.1f} pips (per-pair override for {epic})")
                else:
                    self.logger.debug(f"   📉 Scalp mode: using global scalp offset (no per-pair override)")
            else:
                self.logger.debug(f"   📉 Scalp mode: using global scalp offset")
        elif epic and hasattr(self, '_using_database_config') and self._using_database_config and self._db_config:
            pair_offset = self._db_config.get_pair_stop_offset(epic, entry_type)
            if pair_offset != self.momentum_offset_pips and pair_offset != self.pullback_offset_max_pips:
                # Per-pair override is set - use it directly
                offset_pips = pair_offset
                self.logger.info(f"   📉 Limit offset ({entry_type}): {offset_pips:.1f} pips (per-pair override)")
            else:
                pair_offset = None  # No override, use normal logic

        if pair_offset is None:
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

        # v3.3.0: Check if scalp mode should use LIMIT orders (better price entry) instead of STOP orders
        # LIMIT orders: Enter at better price when price pulls back to entry level
        # STOP orders: Enter when price breaks through entry level (momentum confirmation)
        use_limit_entry = False
        if self.scalp_mode_enabled:
            # Check for scalp_use_limit_orders config option (defaults to False for backward compatibility)
            if hasattr(self, '_db_config') and self._db_config:
                use_limit_entry = getattr(self._db_config, 'scalp_use_limit_orders', False)
            else:
                use_limit_entry = getattr(self, 'scalp_use_limit_orders', False)

        if use_limit_entry:
            # LIMIT ORDER STYLE: Get better entry price by placing order on pullback side
            # This is the OPPOSITE direction from STOP orders
            if direction == 'BULL':
                # BUY LIMIT: Place order BELOW current price (enter when price dips to level)
                limit_entry = current_close - offset
            else:
                # SELL LIMIT: Place order ABOVE current price (enter when price rises to level)
                limit_entry = current_close + offset

            self.logger.info(f"   📍 Market price: {current_close:.5f}")
            self.logger.info(f"   📍 LIMIT entry: {limit_entry:.5f} ({offset_pips:.1f} pips better price)")
            self._current_api_order_type = 'LIMIT'  # v3.3.0: Track for signal dict
        else:
            # STOP ORDER STYLE (original): Confirm price is moving in intended direction
            if direction == 'BULL':
                # BUY STOP: Place order ABOVE current price (enter when price breaks up)
                limit_entry = current_close + offset
            else:
                # SELL STOP: Place order BELOW current price (enter when price breaks down)
                limit_entry = current_close - offset

            self.logger.info(f"   📍 Market price: {current_close:.5f}")
            self.logger.info(f"   📍 STOP entry: {limit_entry:.5f} ({offset_pips:.1f} pips momentum confirmation)")
            self._current_api_order_type = 'STOP'  # v3.3.0: Track for signal dict

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
        """Check if pair is in cooldown period using status-based or adaptive logic.

        v2.26.0: Scalp mode uses shorter status-based cooldowns (15 min even for filled orders)

        v3.3.0: Status-based cooldowns for live trading:
            STANDARD MODE:
                - filled: 4h cooldown (real trade opened)
                - placed: 30min (order working, wait for outcome)
                - pending: 30min (just generated)
                - expired: 30min (didn't fill, prevents expiry spam)
                - rejected: 15min (brief pause before retry)

            SCALP MODE (v2.26.0 FIX):
                - filled: 15min (fast scalp cycles, 10-20 signals/day target)
                - placed: 30min (order working, wait for outcome)
                - pending: 30min (just generated)
                - expired: 30min (didn't fill, prevents expiry spam)
                - rejected: 15min (brief pause before retry)

        v3.2.0: Now queries alert_history database directly for live trading (single source of truth).
        Backtest mode still uses in-memory tracking for performance.

        v3.0.0: Uses adaptive cooldown based on trade outcomes, win rates,
        and market context when ADAPTIVE_COOLDOWN_ENABLED is True.

        Args:
            pair: Currency pair name
            current_time: Current candle timestamp (for backtest mode). If None, uses datetime.now()
        """
        # Use provided time for backtest, or real time for live trading
        check_time = current_time if current_time is not None else datetime.now()

        # Get last signal time - different sources for backtest vs live
        if self._backtest_mode:
            # Backtest: use in-memory tracking (no database writes in backtest)
            if pair not in self.pair_cooldowns:
                return True, "No cooldown (first signal)"
            last_signal = self.pair_cooldowns[pair]
            order_status = None  # No status tracking in backtest
        else:
            # Live: query alert_history database directly (single source of truth)
            epic = self._get_epic_for_pair(pair)
            result = self._get_last_alert_time_from_db(epic)
            if result is None:
                return True, "No cooldown (no recent alerts)"
            last_signal, order_status = result

        # Handle timezone-aware vs naive datetime comparison
        if hasattr(last_signal, 'tzinfo') and last_signal.tzinfo is not None:
            if hasattr(check_time, 'tzinfo') and check_time.tzinfo is None:
                last_signal = last_signal.replace(tzinfo=None)
        elif hasattr(check_time, 'tzinfo') and check_time.tzinfo is not None:
            check_time = check_time.replace(tzinfo=None)

        hours_since = (check_time - last_signal).total_seconds() / 3600

        # v3.3.0: Status-based cooldown for live trading
        # v2.26.0 FIX: Scalp mode uses shorter cooldowns even for filled orders
        if not self._backtest_mode and order_status:
            # Status-based cooldown periods - different for scalp vs standard mode
            if self.scalp_mode_enabled:
                # SCALP MODE: Fast cycles (10-20 signals/day target)
                COOLDOWN_BY_STATUS = {
                    'filled': 0.25,     # 15 min - fast scalp cycles
                    'placed': 0.5,      # 30 min - order working, wait for outcome
                    'pending': 0.5,     # 30 min - just generated, brief pause
                    'expired': 0.5,     # 30 min - didn't fill, prevents expiry spam
                    'rejected': 0.25,   # 15 min - order failed, brief pause before retry
                }
            else:
                # STANDARD MODE: Slower cycles (1-2 signals/day target)
                COOLDOWN_BY_STATUS = {
                    'filled': 4.0,      # 4 hours - full cooldown after real trade
                    'placed': 0.5,      # 30 min - order working, wait for outcome
                    'pending': 0.5,     # 30 min - just generated, brief pause
                    'expired': 0.5,     # 30 min - didn't fill, prevents expiry spam
                    'rejected': 0.25,   # 15 min - order failed, brief pause before retry
                }

            required_cooldown = COOLDOWN_BY_STATUS.get(order_status, 0.5)

            # Check for consecutive expiries (spam prevention)
            if order_status == 'expired':
                consecutive_expiries = self._count_consecutive_expiries(epic)
                if consecutive_expiries >= 3:
                    return False, f"Blocked: {consecutive_expiries} consecutive expiries on {pair}"

            if hours_since < required_cooldown:
                return False, f"In status-based cooldown ({hours_since:.2f}h < {required_cooldown:.1f}h, status={order_status})"

            return True, f"Cooldown OK ({hours_since:.1f}h, status={order_status})"

        # Backtest mode or fallback: use adaptive or static cooldown
        # v2.31.0: Scalp mode in backtest should use scalp cooldown, not adaptive cooldown
        # This matches live behavior where scalp uses status-based 0.25-0.5h cooldowns
        if self.scalp_mode_enabled and self._backtest_mode:
            effective_cooldown = self.scalp_cooldown_minutes / 60.0  # Convert minutes to hours
            cooldown_breakdown = f"scalp={self.scalp_cooldown_minutes}min"
            cooldown_type = "scalp"
        elif self.adaptive_cooldown_enabled:
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

    def _get_last_alert_time_from_db(self, epic: str) -> Optional[Tuple[datetime, str]]:
        """Query alert_history for most recent alert with order status.

        v3.3.0: Returns tuple of (timestamp, order_status) for status-based cooldowns.
        v3.2.0: Database is the single source of truth for cooldowns in live mode.
        This eliminates state sync issues between in-memory tracking and database.

        Args:
            epic: IG Markets epic code (e.g., 'CS.D.EURUSD.CEEM.IP')

        Returns:
            Tuple of (datetime, order_status) or None if no recent alerts found.
            order_status values: 'pending', 'placed', 'filled', 'expired', 'rejected'
        """
        if self._db_manager is None:
            return None

        try:
            conn = self._db_manager.get_connection()
            cursor = conn.cursor()

            # Query for most recent alert within the max cooldown window
            max_lookback_hours = getattr(self, 'max_cooldown_hours', 12.0)

            query = """
                SELECT alert_timestamp, COALESCE(order_status, 'pending') as order_status
                FROM alert_history
                WHERE epic = %s
                  AND alert_timestamp >= NOW() - INTERVAL '%s hours'
                  AND strategy LIKE '%%SMC%%'
                ORDER BY alert_timestamp DESC
                LIMIT 1
            """
            cursor.execute(query, (epic, max_lookback_hours))
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row and row[0]:
                return (row[0], row[1])
            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to query last alert time for {epic}: {e}")
            return None

    def _count_consecutive_expiries(self, epic: str) -> int:
        """Count consecutive expired orders for an epic (spam prevention).

        v3.3.0: If 3+ consecutive expiries, block for 1 hour to prevent spam.
        This protects against scenarios where limit orders repeatedly expire
        without filling, wasting broker API calls.

        Args:
            epic: IG Markets epic code (e.g., 'CS.D.EURUSD.CEEM.IP')

        Returns:
            Number of consecutive 'expired' status alerts (0 if none or error)
        """
        if self._db_manager is None:
            return 0

        try:
            conn = self._db_manager.get_connection()
            cursor = conn.cursor()

            # Get recent alerts and count consecutive expiries from most recent
            query = """
                SELECT order_status
                FROM alert_history
                WHERE epic = %s
                  AND alert_timestamp >= NOW() - INTERVAL '4 hours'
                  AND strategy LIKE '%%SMC%%'
                ORDER BY alert_timestamp DESC
                LIMIT 10
            """
            cursor.execute(query, (epic,))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            # Count consecutive expiries from most recent
            consecutive = 0
            for row in rows:
                status = row[0] if row[0] else 'pending'
                if status == 'expired':
                    consecutive += 1
                else:
                    break  # Stop at first non-expired status

            return consecutive

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to count consecutive expiries for {epic}: {e}")
            return 0

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

    def _update_cooldown_backtest(self, pair: str, signal_time: datetime = None):
        """Update in-memory cooldown for backtest mode only.

        v3.2.0: This is ONLY used in backtest mode. Live mode reads directly from
        alert_history database (single source of truth).

        Args:
            pair: Currency pair name
            signal_time: Signal timestamp. If None, uses datetime.now()
        """
        if not self._backtest_mode:
            # Live mode: cooldowns are read from database, no in-memory update needed
            return

        effective_time = signal_time if signal_time is not None else datetime.now()
        self.pair_cooldowns[pair] = effective_time

        # Track current session for session change detection
        if self.adaptive_cooldown_enabled:
            self.pair_last_session[pair] = self._get_current_session(effective_time)

    def adjust_cooldown_for_unfilled_order(self, pair: str, signal_time: datetime):
        """Adjust cooldown when a limit order expires without filling.

        v3.4.0: In live trading, expired limit orders only trigger 30min cooldown
        instead of full 4h. This method replicates that behavior in backtest mode
        by advancing the cooldown time to allow sooner re-entry.

        The reduced cooldown prevents "wasting" a full cooldown period when
        no actual trade occurred.

        Args:
            pair: Currency pair name (e.g., 'EURUSD')
            signal_time: Original signal timestamp
        """
        if not self._backtest_mode:
            return

        # Live mode uses 0.5h (30min) cooldown for expired orders
        # Advance the cooldown timestamp so only 30min effective cooldown remains
        reduced_cooldown_hours = 0.5

        # Get current full cooldown (adaptive or static)
        if self.adaptive_cooldown_enabled:
            full_cooldown, _ = self._calculate_adaptive_cooldown(pair, signal_time)
        else:
            full_cooldown = self.cooldown_hours

        # Calculate how much to advance the cooldown timestamp
        # If full cooldown is 1.5h and reduced is 0.5h, advance by 1h
        cooldown_reduction = full_cooldown - reduced_cooldown_hours

        if cooldown_reduction > 0 and pair in self.pair_cooldowns:
            # Subtract the reduction from the original signal time
            # This makes it appear the signal happened earlier, so cooldown expires sooner
            original_time = self.pair_cooldowns[pair]
            adjusted_time = original_time - timedelta(hours=cooldown_reduction)
            self.pair_cooldowns[pair] = adjusted_time

            self.logger.debug(
                f"🔄 {pair}: Cooldown adjusted for unfilled limit order "
                f"(reduced from {full_cooldown:.1f}h to {reduced_cooldown_hours}h effective)"
            )

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
        """Reset all cooldowns - call this at the start of each backtest to ensure fresh state.

        v3.2.0: Only resets in-memory state used by backtest mode. Live mode reads
        cooldowns directly from alert_history database.
        """
        self.pair_cooldowns = {}
        self.pair_consecutive_losses = {}
        self.pair_last_session = {}
        self._trade_outcome_cache = {}
        if self.logger:
            self.logger.info("🔄 SMC Simple strategy cooldowns reset for new backtest (v3.2.0 in-memory state cleared)")

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
        # v2.31.1: Always track to in-memory stats for backtest summary
        self._track_filter_rejection(stage)

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

            # v2.35.5: MONITORING - Log entry quality/momentum below recommended thresholds
            eq_score = metrics.entry_quality_score
            momentum = metrics.entry_candle_momentum
            if eq_score is not None and eq_score < 0.30:
                self.logger.info(f"📊 [MONITOR] Entry quality {eq_score:.2f} < 0.30 threshold (would reject if gate enabled)")
                self._track_filter_rejection('entry_quality_monitor')
            if momentum is not None and momentum < 0.30:
                self.logger.info(f"📊 [MONITOR] Entry momentum {momentum:.2f} < 0.30 threshold (would reject if gate enabled)")
                self._track_filter_rejection('entry_momentum_monitor')

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
                'order_block_volume_factor': 1.2,  # Reduced from 1.3 - IG ltv data is sparse
                'order_block_min_confidence': 0.3,  # Lowered from 0.4 default for more OB detection
                'bos_threshold': pip_value * 5,
                'max_order_blocks': 5  # Increased from 3 for better visualization
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

    # =========================================================================
    # FILTER STATS TRACKING (v2.31.1 - Jan 2026)
    # In-memory tracking for backtest summary display
    # =========================================================================

    def _track_filter_rejection(self, filter_name: str) -> None:
        """
        Track a filter rejection in the in-memory stats (for backtest summary).

        Args:
            filter_name: Name of the filter that rejected (e.g., 'efficiency_ratio', 'session_hours')
        """
        if filter_name not in self._filter_stats['filter_rejections']:
            self._filter_stats['filter_rejections'][filter_name] = 0
        self._filter_stats['filter_rejections'][filter_name] += 1

    def _track_signal_detected(self) -> None:
        """Track that a signal was detected (before filters)."""
        self._filter_stats['signals_detected'] += 1

    def _track_signal_passed(self) -> None:
        """Track that a signal passed all filters."""
        self._filter_stats['signals_passed'] += 1

    def get_filter_stats(self) -> Dict:
        """
        Get filter statistics for backtest summary display.

        Returns:
            Dict containing:
            - signals_detected: Total signals detected before filters
            - signals_passed: Signals that passed all filters
            - signals_filtered: Signals rejected by filters
            - filter_rejections: Dict of {filter_name: rejection_count}
            - filter_rate: Percentage of signals filtered out
        """
        stats = self._filter_stats.copy()
        stats['signals_filtered'] = stats['signals_detected'] - stats['signals_passed']
        if stats['signals_detected'] > 0:
            stats['filter_rate'] = (stats['signals_filtered'] / stats['signals_detected']) * 100
        else:
            stats['filter_rate'] = 0.0
        return stats

    def reset_filter_stats(self) -> None:
        """Reset filter stats (call at start of new backtest)."""
        self._filter_stats = {
            'signals_detected': 0,
            'signals_passed': 0,
            'filter_rejections': {}
        }


def create_smc_simple_strategy(config, logger=None, db_manager=None, config_override: dict = None):
    """Factory function to create SMC Simple Strategy instance.

    Args:
        config: Main config module
        logger: Logger instance (optional)
        db_manager: DatabaseManager for rejection tracking (optional)
        config_override: Dict of parameter overrides for backtesting (optional)
                        When None = LIVE TRADING (unchanged behavior)
                        When dict = BACKTEST MODE (overrides applied)

    Returns:
        SMCSimpleStrategy instance
    """
    return SMCSimpleStrategy(config, logger, db_manager, config_override=config_override)
