"""
SMC Simple Strategy Configuration Tab Component

Renders configuration management UI for SMC Simple strategy with:
- Global Settings: All configuration parameters with expandable sections
- Per-Pair Overrides: Pair-specific parameter overrides
- Effective Config: Merged view of global + pair settings
- Audit Trail: Change history and tracking
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

from services.smc_simple_config_service import (
    get_global_config,
    get_pair_overrides,
    get_pair_override,
    save_global_config,
    save_pair_override,
    delete_pair_override,
    get_audit_history,
    get_enabled_pairs,
    get_effective_config_for_pair,
    fetch_optimizer_recommendations,
    apply_optimizer_recommendations,
)


def _values_equal(new_val, old_val, tolerance: float = 1e-9) -> bool:
    """Compare values with tolerance for floating point precision issues."""
    from decimal import Decimal
    if new_val is None and old_val is None:
        return True
    if new_val is None or old_val is None:
        return False
    # Handle numeric types including Decimal from database
    if isinstance(new_val, (int, float, Decimal)) and isinstance(old_val, (int, float, Decimal)):
        return abs(float(new_val) - float(old_val)) < tolerance
    return new_val == old_val


def render_smc_config_tab():
    """Main entry point for SMC Config tab"""
    st.header("SMC Simple Strategy Configuration")
    st.markdown("*Database-driven configuration management for SMC Simple strategy*")

    # Load current config
    config = get_global_config()

    if not config:
        st.error("Failed to load SMC Simple configuration from database.")
        st.info("Make sure the strategy_config database is set up and seeded.")
        return

    # Display version info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Version", config.get('version', 'N/A'))
    with col2:
        st.metric("Status", "Active" if config.get('is_active') else "Inactive")
    with col3:
        updated_at = config.get('updated_at')
        if updated_at:
            st.metric("Last Updated", updated_at.strftime("%Y-%m-%d %H:%M"))
    with col4:
        st.metric("Pairs Enabled", len(config.get('enabled_pairs', [])))

    st.divider()

    # Sub-tabs
    sub_tabs = st.tabs([
        "Global Settings",
        "Per-Pair Overrides",
        "Effective Config",
        "Parameter Optimizer",
        "Audit Trail"
    ])

    with sub_tabs[0]:
        render_global_settings(config)

    with sub_tabs[1]:
        render_pair_overrides(config)

    with sub_tabs[2]:
        render_effective_config(config)

    with sub_tabs[3]:
        render_parameter_optimizer(config)

    with sub_tabs[4]:
        render_audit_trail()


def render_global_settings(config: Dict[str, Any]):
    """Render global settings editor with expandable sections"""
    st.subheader("Global Configuration Settings")

    # Get current user identifier
    updated_by = st.text_input(
        "Your Name (for audit trail)",
        value=st.session_state.get('smc_config_user', 'streamlit_user'),
        key="smc_global_user"
    )
    st.session_state['smc_config_user'] = updated_by

    # Track changes
    if 'smc_pending_changes' not in st.session_state:
        st.session_state.smc_pending_changes = {}

    # TIER 1: 4H Directional Bias
    with st.expander("TIER 1: 4H Directional Bias", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            new_ema = st.number_input(
                "EMA Period",
                value=config.get('ema_period', 50),
                min_value=5, max_value=200,
                help="EMA period for directional bias (default: 50)",
                key="ema_period"
            )
            if not _values_equal(new_ema, config.get('ema_period')):
                st.session_state.smc_pending_changes['ema_period'] = new_ema

            new_ema_buffer = st.number_input(
                "EMA Buffer (pips)",
                value=float(config.get('ema_buffer_pips', 2.5)),
                min_value=0.0, max_value=20.0, step=0.5,
                help="Buffer from EMA for trade direction confirmation",
                key="ema_buffer_pips"
            )
            if not _values_equal(new_ema_buffer, config.get('ema_buffer_pips')):
                st.session_state.smc_pending_changes['ema_buffer_pips'] = new_ema_buffer

        with col2:
            new_min_dist = st.number_input(
                "Min Distance from EMA (pips)",
                value=float(config.get('min_distance_from_ema_pips', 3)),
                min_value=0.0, max_value=50.0, step=0.5,
                help="Minimum price distance from EMA to consider valid",
                key="min_distance_ema"
            )
            if not _values_equal(new_min_dist, config.get('min_distance_from_ema_pips')):
                st.session_state.smc_pending_changes['min_distance_from_ema_pips'] = new_min_dist

            new_require_close = st.checkbox(
                "Require Close Beyond EMA",
                value=config.get('require_close_beyond_ema', True),
                help="Candle must close beyond EMA, not just wick",
                key="require_close_beyond"
            )
            if not _values_equal(new_require_close, config.get('require_close_beyond_ema')):
                st.session_state.smc_pending_changes['require_close_beyond_ema'] = new_require_close

    # TIER 2: 15m Entry Trigger
    with st.expander("TIER 2: 15m Entry Trigger", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            new_swing_lookback = st.number_input(
                "Swing Lookback Bars",
                value=config.get('swing_lookback_bars', 20),
                min_value=5, max_value=100,
                help="Bars to look back for swing detection",
                key="swing_lookback"
            )
            if not _values_equal(new_swing_lookback, config.get('swing_lookback_bars')):
                st.session_state.smc_pending_changes['swing_lookback_bars'] = new_swing_lookback

            new_swing_strength = st.number_input(
                "Swing Strength Bars",
                value=config.get('swing_strength_bars', 2),
                min_value=1, max_value=10,
                help="Bars on each side to confirm swing",
                key="swing_strength"
            )
            if not _values_equal(new_swing_strength, config.get('swing_strength_bars')):
                st.session_state.smc_pending_changes['swing_strength_bars'] = new_swing_strength

        with col2:
            new_dynamic_swing = st.checkbox(
                "Use Dynamic Swing Lookback",
                value=config.get('use_dynamic_swing_lookback', True),
                help="Adapt lookback to ATR volatility",
                key="dynamic_swing"
            )
            if not _values_equal(new_dynamic_swing, config.get('use_dynamic_swing_lookback')):
                st.session_state.smc_pending_changes['use_dynamic_swing_lookback'] = new_dynamic_swing

            new_volume_confirm = st.checkbox(
                "Volume Confirmation Enabled",
                value=config.get('volume_confirmation_enabled', True),
                help="Require volume spike for confirmation",
                key="volume_confirm"
            )
            if not _values_equal(new_volume_confirm, config.get('volume_confirmation_enabled')):
                st.session_state.smc_pending_changes['volume_confirmation_enabled'] = new_volume_confirm

        col3, col4 = st.columns(2)
        with col3:
            new_vol_spike = st.number_input(
                "Volume Spike Multiplier",
                value=float(config.get('volume_spike_multiplier', 1.2)),
                min_value=1.0, max_value=3.0, step=0.1,
                help="Multiplier of SMA volume for spike detection",
                key="vol_spike"
            )
            if not _values_equal(new_vol_spike, config.get('volume_spike_multiplier')):
                st.session_state.smc_pending_changes['volume_spike_multiplier'] = new_vol_spike

    # TIER 3: 5m Execution
    with st.expander("TIER 3: 5m Execution", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            new_pullback = st.checkbox(
                "Pullback Enabled",
                value=config.get('pullback_enabled', True),
                help="Wait for pullback before entry",
                key="pullback_enabled"
            )
            if not _values_equal(new_pullback, config.get('pullback_enabled')):
                st.session_state.smc_pending_changes['pullback_enabled'] = new_pullback

            new_fib_min = st.number_input(
                "Fib Pullback Min",
                value=float(config.get('fib_pullback_min', 0.236)),
                min_value=0.0, max_value=1.0, step=0.01,
                help="Minimum Fibonacci retracement level",
                key="fib_min"
            )
            if not _values_equal(new_fib_min, config.get('fib_pullback_min')):
                st.session_state.smc_pending_changes['fib_pullback_min'] = new_fib_min

        with col2:
            new_fib_max = st.number_input(
                "Fib Pullback Max",
                value=float(config.get('fib_pullback_max', 0.70)),
                min_value=0.0, max_value=1.0, step=0.01,
                help="Maximum Fibonacci retracement level",
                key="fib_max"
            )
            if not _values_equal(new_fib_max, config.get('fib_pullback_max')):
                st.session_state.smc_pending_changes['fib_pullback_max'] = new_fib_max

            new_pullback_wait = st.number_input(
                "Max Pullback Wait Bars",
                value=config.get('max_pullback_wait_bars', 12),
                min_value=1, max_value=50,
                help="Maximum bars to wait for pullback",
                key="pullback_wait"
            )
            if not _values_equal(new_pullback_wait, config.get('max_pullback_wait_bars')):
                st.session_state.smc_pending_changes['max_pullback_wait_bars'] = new_pullback_wait

        # Momentum Mode
        st.markdown("**Momentum Continuation Mode**")
        col3, col4 = st.columns(2)
        with col3:
            new_momentum = st.checkbox(
                "Momentum Mode Enabled",
                value=config.get('momentum_mode_enabled', True),
                help="Allow entries without pullback in strong trends",
                key="momentum_enabled"
            )
            if not _values_equal(new_momentum, config.get('momentum_mode_enabled')):
                st.session_state.smc_pending_changes['momentum_mode_enabled'] = new_momentum

            new_momentum_depth = st.number_input(
                "Momentum Min Depth",
                value=float(config.get('momentum_min_depth', -0.50)),
                min_value=-1.0, max_value=0.0, step=0.05,
                help="Minimum depth for momentum entry",
                key="momentum_depth"
            )
            if not _values_equal(new_momentum_depth, config.get('momentum_min_depth')):
                st.session_state.smc_pending_changes['momentum_min_depth'] = new_momentum_depth

        with col4:
            new_momentum_penalty = st.number_input(
                "Momentum Confidence Penalty",
                value=float(config.get('momentum_confidence_penalty', 0.05)),
                min_value=0.0, max_value=0.5, step=0.01,
                help="Reduce confidence for momentum entries",
                key="momentum_penalty"
            )
            if not _values_equal(new_momentum_penalty, config.get('momentum_confidence_penalty')):
                st.session_state.smc_pending_changes['momentum_confidence_penalty'] = new_momentum_penalty

    # Risk Management
    with st.expander("Risk Management", expanded=False):
        # Fixed SL/TP Override Section
        st.markdown("**Fixed SL/TP Override**")
        st.caption("When enabled, uses fixed SL/TP values instead of strategy-calculated ones. Can be overridden per-pair.")
        col_fixed1, col_fixed2, col_fixed3 = st.columns(3)
        with col_fixed1:
            new_fixed_enabled = st.checkbox(
                "Fixed SL/TP Override Enabled",
                value=bool(config.get('fixed_sl_tp_override_enabled', True)),
                help="Master switch: use fixed SL/TP values",
                key="fixed_sl_tp_override_enabled"
            )
            if not _values_equal(new_fixed_enabled, config.get('fixed_sl_tp_override_enabled')):
                st.session_state.smc_pending_changes['fixed_sl_tp_override_enabled'] = new_fixed_enabled

        with col_fixed2:
            new_fixed_sl = st.number_input(
                "Fixed Stop Loss (pips)",
                value=float(config.get('fixed_stop_loss_pips', 9.0)),
                min_value=1.0, max_value=50.0, step=0.5,
                help="Default fixed stop loss (can be overridden per-pair)",
                key="fixed_stop_loss_pips"
            )
            if not _values_equal(new_fixed_sl, config.get('fixed_stop_loss_pips')):
                st.session_state.smc_pending_changes['fixed_stop_loss_pips'] = new_fixed_sl

        with col_fixed3:
            new_fixed_tp = st.number_input(
                "Fixed Take Profit (pips)",
                value=float(config.get('fixed_take_profit_pips', 15.0)),
                min_value=1.0, max_value=100.0, step=0.5,
                help="Default fixed take profit (can be overridden per-pair)",
                key="fixed_take_profit_pips"
            )
            if not _values_equal(new_fixed_tp, config.get('fixed_take_profit_pips')):
                st.session_state.smc_pending_changes['fixed_take_profit_pips'] = new_fixed_tp

        st.divider()
        st.markdown("**Dynamic SL Settings**")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_sl_atr = st.number_input(
                "SL ATR Multiplier",
                value=float(config.get('sl_atr_multiplier', 1.0)),
                min_value=0.5, max_value=3.0, step=0.1,
                help="ATR multiplier for stop loss",
                key="sl_atr"
            )
            if not _values_equal(new_sl_atr, config.get('sl_atr_multiplier')):
                st.session_state.smc_pending_changes['sl_atr_multiplier'] = new_sl_atr

            new_sl_buffer = st.number_input(
                "SL Buffer (pips)",
                value=float(config.get('sl_buffer_pips', 6)),
                min_value=0.0, max_value=20.0, step=0.5,
                help="Default stop loss buffer in pips",
                key="sl_buffer"
            )
            if not _values_equal(new_sl_buffer, config.get('sl_buffer_pips')):
                st.session_state.smc_pending_changes['sl_buffer_pips'] = new_sl_buffer

        with col2:
            new_max_atr = st.number_input(
                "Max SL ATR Multiplier",
                value=float(config.get('max_sl_atr_multiplier', 3.0)),
                min_value=1.0, max_value=10.0, step=0.5,
                help="Maximum ATR multiplier for dynamic cap",
                key="max_atr"
            )
            if not _values_equal(new_max_atr, config.get('max_sl_atr_multiplier')):
                st.session_state.smc_pending_changes['max_sl_atr_multiplier'] = new_max_atr

            new_max_sl = st.number_input(
                "Max SL (pips)",
                value=float(config.get('max_sl_absolute_pips', 30)),
                min_value=10.0, max_value=100.0, step=5.0,
                help="Absolute maximum stop loss in pips",
                key="max_sl"
            )
            if not _values_equal(new_max_sl, config.get('max_sl_absolute_pips')):
                st.session_state.smc_pending_changes['max_sl_absolute_pips'] = new_max_sl

        with col3:
            new_optimal_rr = st.number_input(
                "Optimal R:R Ratio",
                value=float(config.get('optimal_rr_ratio', 2.5)),
                min_value=1.0, max_value=5.0, step=0.1,
                help="Optimal risk-reward ratio target",
                key="optimal_rr_ratio"
            )
            if not _values_equal(new_optimal_rr, config.get('optimal_rr_ratio')):
                st.session_state.smc_pending_changes['optimal_rr_ratio'] = new_optimal_rr

            new_min_rr = st.number_input(
                "Min R:R Ratio",
                value=float(config.get('min_rr_ratio', 1.5)),
                min_value=0.5, max_value=3.0, step=0.1,
                help="Minimum risk-reward ratio",
                key="min_rr"
            )
            if not _values_equal(new_min_rr, config.get('min_rr_ratio')):
                st.session_state.smc_pending_changes['min_rr_ratio'] = new_min_rr

    # Session Filter
    with st.expander("Session Filter", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            new_session_filter = st.checkbox(
                "Session Filter Enabled",
                value=config.get('session_filter_enabled', True),
                help="Enable trading session filtering",
                key="session_filter"
            )
            if not _values_equal(new_session_filter, config.get('session_filter_enabled')):
                st.session_state.smc_pending_changes['session_filter_enabled'] = new_session_filter

            allowed_sessions = config.get('allowed_sessions', ['london', 'new_york', 'overlap'])
            session_options = ['london', 'new_york', 'asian', 'sydney', 'overlap']
            # Filter default to only include valid options (handles DB migration inconsistencies)
            filtered_defaults = [s for s in allowed_sessions if s in session_options]
            new_sessions = st.multiselect(
                "Allowed Sessions",
                options=session_options,
                default=filtered_defaults if filtered_defaults else ['london', 'new_york'],
                help="Trading sessions to allow",
                key="allowed_sessions"
            )
            if set(new_sessions) != set(allowed_sessions):
                st.session_state.smc_pending_changes['allowed_sessions'] = new_sessions

        with col2:
            new_block_asian = st.checkbox(
                "Block Asian Session (Default)",
                value=config.get('block_asian_session', True),
                help="Default to blocking Asian session",
                key="block_asian_session"
            )
            if not _values_equal(new_block_asian, config.get('block_asian_session')):
                st.session_state.smc_pending_changes['block_asian_session'] = new_block_asian

    # Confidence Scoring
    with st.expander("Confidence Scoring", expanded=False):
        st.markdown("**Confidence Thresholds**")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_min_conf = st.number_input(
                "Min Confidence Threshold",
                value=float(config.get('min_confidence_threshold', 0.48)),
                min_value=0.0, max_value=1.0, step=0.01,
                help="Minimum confidence score for trade entry",
                key="min_confidence_threshold"
            )
            if not _values_equal(new_min_conf, config.get('min_confidence_threshold')):
                st.session_state.smc_pending_changes['min_confidence_threshold'] = new_min_conf

        with col2:
            new_max_conf = st.number_input(
                "Max Confidence Cap",
                value=float(config.get('max_confidence_threshold', 0.75)),
                min_value=0.0, max_value=1.0, step=0.01,
                help="Maximum confidence cap - signals above this are rejected (paradox: high confidence = worse outcomes)",
                key="max_confidence_threshold"
            )
            if not _values_equal(new_max_conf, config.get('max_confidence_threshold')):
                st.session_state.smc_pending_changes['max_confidence_threshold'] = new_max_conf

        with col3:
            new_high_conf = st.number_input(
                "High Confidence Threshold",
                value=float(config.get('high_confidence_threshold', 0.75)),
                min_value=0.0, max_value=1.0, step=0.01,
                help="Threshold for marking signals as 'high confidence' (informational)",
                key="high_confidence_threshold"
            )
            if not _values_equal(new_high_conf, config.get('high_confidence_threshold')):
                st.session_state.smc_pending_changes['high_confidence_threshold'] = new_high_conf

        st.caption("⚠️ Analysis of 85 trades (Dec 2025) showed confidence > 75% had only 42% win rate. The Max Confidence Cap prevents overconfident signals.")

        st.markdown("**Confidence Weights**")
        conf_weights = config.get('confidence_weights', {})
        if isinstance(conf_weights, str):
            conf_weights = json.loads(conf_weights)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Current weights:", conf_weights)
        with col2:
            weight_json = st.text_area(
                "Edit Weights (JSON)",
                value=json.dumps(conf_weights, indent=2),
                height=150,
                key="conf_weights_json"
            )
            try:
                new_weights = json.loads(weight_json)
                if new_weights != conf_weights:
                    st.session_state.smc_pending_changes['confidence_weights'] = new_weights
            except json.JSONDecodeError:
                st.warning("Invalid JSON format for weights")

    # MACD Alignment
    with st.expander("MACD Alignment Filter", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            new_macd_enabled = st.checkbox(
                "MACD Alignment Filter Enabled",
                value=config.get('macd_alignment_filter_enabled', True),
                help="Block trades against MACD direction",
                key="macd_alignment_filter_enabled"
            )
            if not _values_equal(new_macd_enabled, config.get('macd_alignment_filter_enabled')):
                st.session_state.smc_pending_changes['macd_alignment_filter_enabled'] = new_macd_enabled

            macd_mode_options = ['momentum', 'trend', 'crossover']
            current_mode = config.get('macd_alignment_mode', 'momentum')
            mode_index = macd_mode_options.index(current_mode) if current_mode in macd_mode_options else 0
            new_macd_mode = st.selectbox(
                "MACD Alignment Mode",
                options=macd_mode_options,
                index=mode_index,
                help="Mode for MACD alignment filter",
                key="macd_alignment_mode"
            )
            if not _values_equal(new_macd_mode, config.get('macd_alignment_mode')):
                st.session_state.smc_pending_changes['macd_alignment_mode'] = new_macd_mode

        with col2:
            new_macd_strength = st.number_input(
                "MACD Min Strength",
                value=float(config.get('macd_min_strength', 0.0)),
                min_value=0.0, max_value=1.0, step=0.00000001,
                format="%.8f",
                help="Minimum MACD strength for alignment",
                key="macd_min_strength"
            )
            if not _values_equal(new_macd_strength, config.get('macd_min_strength')):
                st.session_state.smc_pending_changes['macd_min_strength'] = new_macd_strength

    # Swing Proximity Validation (v2.15.0)
    with st.expander("Swing Proximity Validation (TIER 4)", expanded=False):
        st.markdown("**Prevents entries too close to opposing swing levels**")
        st.caption("Based on trade analysis: 65% of losing trades were counter-trend entries at wrong swing levels. "
                   "BUY near resistance = 25% WR, SELL near support = 0% WR. Trades with 15+ pips clearance = 50%+ WR.")

        col1, col2 = st.columns(2)
        with col1:
            new_swing_prox_enabled = st.checkbox(
                "Swing Proximity Validation Enabled",
                value=config.get('swing_proximity_enabled', True),
                help="Enable swing proximity validation to prevent entries near swing levels",
                key="swing_proximity_enabled"
            )
            if not _values_equal(new_swing_prox_enabled, config.get('swing_proximity_enabled')):
                st.session_state.smc_pending_changes['swing_proximity_enabled'] = new_swing_prox_enabled

            new_swing_prox_min_dist = st.number_input(
                "Min Distance from Swing (pips)",
                value=config.get('swing_proximity_min_distance_pips', 12),
                min_value=1, max_value=50,
                help="Minimum distance in pips from opposing swing level (BUY from resistance, SELL from support). 10-15 pips optimal based on analysis.",
                key="swing_proximity_min_distance_pips"
            )
            if not _values_equal(new_swing_prox_min_dist, config.get('swing_proximity_min_distance_pips')):
                st.session_state.smc_pending_changes['swing_proximity_min_distance_pips'] = new_swing_prox_min_dist

            new_swing_prox_strict = st.checkbox(
                "Strict Mode (Reject vs Penalty)",
                value=config.get('swing_proximity_strict_mode', True),
                help="If enabled, reject signals near swings. If disabled, apply confidence penalty instead.",
                key="swing_proximity_strict_mode"
            )
            if not _values_equal(new_swing_prox_strict, config.get('swing_proximity_strict_mode')):
                st.session_state.smc_pending_changes['swing_proximity_strict_mode'] = new_swing_prox_strict

        with col2:
            new_swing_prox_res_buffer = st.number_input(
                "Resistance Buffer Multiplier",
                value=float(config.get('swing_proximity_resistance_buffer', 1.0)),
                min_value=0.5, max_value=3.0, step=0.1,
                help="Multiplier for resistance distance requirement (BUY signals)",
                key="swing_proximity_resistance_buffer"
            )
            if not _values_equal(new_swing_prox_res_buffer, config.get('swing_proximity_resistance_buffer')):
                st.session_state.smc_pending_changes['swing_proximity_resistance_buffer'] = new_swing_prox_res_buffer

            new_swing_prox_sup_buffer = st.number_input(
                "Support Buffer Multiplier",
                value=float(config.get('swing_proximity_support_buffer', 1.0)),
                min_value=0.5, max_value=3.0, step=0.1,
                help="Multiplier for support distance requirement (SELL signals)",
                key="swing_proximity_support_buffer"
            )
            if not _values_equal(new_swing_prox_sup_buffer, config.get('swing_proximity_support_buffer')):
                st.session_state.smc_pending_changes['swing_proximity_support_buffer'] = new_swing_prox_sup_buffer

            new_swing_prox_lookback = st.number_input(
                "Swing Lookback Count",
                value=config.get('swing_proximity_lookback_swings', 5),
                min_value=1, max_value=20,
                help="Number of recent swings to check for proximity",
                key="swing_proximity_lookback_swings"
            )
            if not _values_equal(new_swing_prox_lookback, config.get('swing_proximity_lookback_swings')):
                st.session_state.smc_pending_changes['swing_proximity_lookback_swings'] = new_swing_prox_lookback

    # Adaptive Cooldown
    with st.expander("Adaptive Cooldown", expanded=False):
        col1, col2 = st.columns(2)

        # Get cooldown config JSONB
        cooldown_config = config.get('adaptive_cooldown_config', {})
        if isinstance(cooldown_config, str):
            cooldown_config = json.loads(cooldown_config)

        with col1:
            new_signal_cooldown = st.number_input(
                "Signal Cooldown (hours)",
                value=config.get('signal_cooldown_hours', 3),
                min_value=1, max_value=24,
                help="Base cooldown between signals for same pair",
                key="signal_cooldown_hours"
            )
            if not _values_equal(new_signal_cooldown, config.get('signal_cooldown_hours')):
                st.session_state.smc_pending_changes['signal_cooldown_hours'] = new_signal_cooldown

            # Display enabled status from JSONB
            is_adaptive_enabled = cooldown_config.get('enabled', True)
            st.info(f"Adaptive mode: {'Enabled' if is_adaptive_enabled else 'Disabled'}")

        with col2:
            st.write("**Adaptive Cooldown Config (JSONB)**")
            cooldown_json = st.text_area(
                "Edit Cooldown Config",
                value=json.dumps(cooldown_config, indent=2),
                height=250,
                help="Full adaptive cooldown configuration including enabled flag, thresholds, multipliers",
                key="cooldown_json"
            )
            try:
                new_cooldown_cfg = json.loads(cooldown_json)
                if new_cooldown_cfg != cooldown_config:
                    st.session_state.smc_pending_changes['adaptive_cooldown_config'] = new_cooldown_cfg
            except json.JSONDecodeError:
                st.warning("Invalid JSON format for cooldown config")

    # Scalp Mode Configuration
    with st.expander("🎯 Scalp Mode (High-Frequency Trading)", expanded=False):
        st.markdown("""
        **Scalp Mode** enables high-frequency trading with:
        - 5 pip TP / 5 pip SL (1:1 R:R)
        - Faster timeframes: 1H → 5m → 1m
        - Relaxed filters for more signals
        - Spread filter (only trade when spread < 1 pip)
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Master toggle
            new_scalp_enabled = st.checkbox(
                "Enable Scalp Mode",
                value=config.get('scalp_mode_enabled', False),
                help="Master toggle for high-frequency scalping mode",
                key="scalp_mode_enabled"
            )
            if not _values_equal(new_scalp_enabled, config.get('scalp_mode_enabled')):
                st.session_state.smc_pending_changes['scalp_mode_enabled'] = new_scalp_enabled

            if new_scalp_enabled:
                st.success("⚡ Scalp Mode ACTIVE")
            else:
                st.info("📊 Swing Mode (default)")

            st.markdown("---")
            st.markdown("**Risk Settings**")

            new_scalp_tp = st.number_input(
                "Scalp Take Profit (pips)",
                value=float(config.get('scalp_tp_pips', 5.0)),
                min_value=1.0, max_value=20.0, step=0.5,
                help="Take profit target for scalp trades",
                key="scalp_tp_pips"
            )
            if not _values_equal(new_scalp_tp, config.get('scalp_tp_pips')):
                st.session_state.smc_pending_changes['scalp_tp_pips'] = new_scalp_tp

            new_scalp_sl = st.number_input(
                "Scalp Stop Loss (pips)",
                value=float(config.get('scalp_sl_pips', 5.0)),
                min_value=1.0, max_value=20.0, step=0.5,
                help="Stop loss for scalp trades",
                key="scalp_sl_pips"
            )
            if not _values_equal(new_scalp_sl, config.get('scalp_sl_pips')):
                st.session_state.smc_pending_changes['scalp_sl_pips'] = new_scalp_sl

            new_scalp_spread = st.number_input(
                "Max Spread Filter (pips)",
                value=float(config.get('scalp_max_spread_pips', 1.0)),
                min_value=0.1, max_value=5.0, step=0.1,
                help="Maximum spread allowed for scalp entries",
                key="scalp_max_spread_pips"
            )
            if not _values_equal(new_scalp_spread, config.get('scalp_max_spread_pips')):
                st.session_state.smc_pending_changes['scalp_max_spread_pips'] = new_scalp_spread

            new_scalp_confidence = st.number_input(
                "Min Confidence",
                value=float(config.get('scalp_min_confidence', 0.30)),
                min_value=0.0, max_value=1.0, step=0.05,
                help="Lower threshold for more scalp entries",
                key="scalp_min_confidence"
            )
            if not _values_equal(new_scalp_confidence, config.get('scalp_min_confidence')):
                st.session_state.smc_pending_changes['scalp_min_confidence'] = new_scalp_confidence

        with col2:
            st.markdown("**Timeframe Settings**")

            new_htf_tf = st.selectbox(
                "HTF Timeframe",
                options=["1h", "2h", "4h"],
                index=["1h", "2h", "4h"].index(config.get('scalp_htf_timeframe', '1h')),
                help="Higher timeframe for scalp mode bias",
                key="scalp_htf_timeframe"
            )
            if new_htf_tf != config.get('scalp_htf_timeframe', '1h'):
                st.session_state.smc_pending_changes['scalp_htf_timeframe'] = new_htf_tf

            new_trigger_tf = st.selectbox(
                "Trigger Timeframe",
                options=["1m", "5m", "15m"],
                index=["1m", "5m", "15m"].index(config.get('scalp_trigger_timeframe', '5m')),
                help="Trigger timeframe for scalp mode",
                key="scalp_trigger_timeframe"
            )
            if new_trigger_tf != config.get('scalp_trigger_timeframe', '5m'):
                st.session_state.smc_pending_changes['scalp_trigger_timeframe'] = new_trigger_tf

            new_entry_tf = st.selectbox(
                "Entry Timeframe",
                options=["1m", "5m"],
                index=["1m", "5m"].index(config.get('scalp_entry_timeframe', '1m')),
                help="Entry timeframe for scalp mode",
                key="scalp_entry_timeframe"
            )
            if new_entry_tf != config.get('scalp_entry_timeframe', '1m'):
                st.session_state.smc_pending_changes['scalp_entry_timeframe'] = new_entry_tf

            st.markdown("---")
            st.markdown("**Timing Settings**")

            new_scalp_ema = st.number_input(
                "EMA Period",
                value=config.get('scalp_ema_period', 20),
                min_value=5, max_value=50,
                help="Faster EMA period for scalp mode",
                key="scalp_ema_period"
            )
            if not _values_equal(new_scalp_ema, config.get('scalp_ema_period')):
                st.session_state.smc_pending_changes['scalp_ema_period'] = new_scalp_ema

            new_scalp_cooldown = st.number_input(
                "Cooldown (minutes)",
                value=config.get('scalp_cooldown_minutes', 15),
                min_value=1, max_value=60,
                help="Cooldown between scalp trades",
                key="scalp_cooldown_minutes"
            )
            if not _values_equal(new_scalp_cooldown, config.get('scalp_cooldown_minutes')):
                st.session_state.smc_pending_changes['scalp_cooldown_minutes'] = new_scalp_cooldown

        # Filter toggles
        st.markdown("---")
        st.markdown("**Filter Overrides** (Scalp mode relaxes filters for more signals)")

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            new_disable_ema_slope = st.checkbox(
                "Disable EMA Slope",
                value=config.get('scalp_disable_ema_slope_validation', True),
                help="Skip EMA slope validation in scalp mode",
                key="scalp_disable_ema_slope_validation"
            )
            if not _values_equal(new_disable_ema_slope, config.get('scalp_disable_ema_slope_validation')):
                st.session_state.smc_pending_changes['scalp_disable_ema_slope_validation'] = new_disable_ema_slope

        with filter_col2:
            new_disable_swing_prox = st.checkbox(
                "Disable Swing Proximity",
                value=config.get('scalp_disable_swing_proximity', True),
                help="Skip swing proximity check in scalp mode",
                key="scalp_disable_swing_proximity"
            )
            if not _values_equal(new_disable_swing_prox, config.get('scalp_disable_swing_proximity')):
                st.session_state.smc_pending_changes['scalp_disable_swing_proximity'] = new_disable_swing_prox

        with filter_col3:
            new_disable_volume = st.checkbox(
                "Disable Volume Filter",
                value=config.get('scalp_disable_volume_filter', True),
                help="Skip volume confirmation in scalp mode",
                key="scalp_disable_volume_filter"
            )
            if not _values_equal(new_disable_volume, config.get('scalp_disable_volume_filter')):
                st.session_state.smc_pending_changes['scalp_disable_volume_filter'] = new_disable_volume

        with filter_col4:
            new_disable_macd = st.checkbox(
                "Disable MACD Filter",
                value=config.get('scalp_disable_macd_filter', True),
                help="Skip MACD alignment in scalp mode",
                key="scalp_disable_macd_filter"
            )
            if not _values_equal(new_disable_macd, config.get('scalp_disable_macd_filter')):
                st.session_state.smc_pending_changes['scalp_disable_macd_filter'] = new_disable_macd

        # Spread filter toggle
        new_require_spread = st.checkbox(
            "Require Tight Spread (24/7 trading with spread gate)",
            value=config.get('scalp_require_tight_spread', True),
            help="Only enter trades when spread is below max threshold",
            key="scalp_require_tight_spread"
        )
        if not _values_equal(new_require_spread, config.get('scalp_require_tight_spread')):
            st.session_state.smc_pending_changes['scalp_require_tight_spread'] = new_require_spread

        # Warning message
        if new_scalp_enabled:
            st.warning("""
            ⚠️ **Scalp Mode Risk Warnings:**
            - 5 pip TP with 1 pip spread = 20% profit reduction
            - Requires 65%+ win rate for profitability
            - Monitor drawdown closely during high-frequency trading
            """)

    # Enabled Pairs
    with st.expander("Enabled Trading Pairs", expanded=False):
        enabled_pairs = config.get('enabled_pairs', [])
        # Known IG forex pairs - EURUSD is CEEM only, others are MINI
        known_pairs = [
            "CS.D.EURUSD.CEEM.IP",  # EURUSD only available as CEEM
            "CS.D.GBPUSD.MINI.IP",
            "CS.D.USDJPY.MINI.IP",
            "CS.D.USDCHF.MINI.IP",
            "CS.D.AUDUSD.MINI.IP",
            "CS.D.USDCAD.MINI.IP",
            "CS.D.NZDUSD.MINI.IP",
            "CS.D.EURJPY.MINI.IP",
            "CS.D.GBPJPY.MINI.IP",
            "CS.D.AUDJPY.MINI.IP",
            "CS.D.EURGBP.MINI.IP",
            "CS.D.EURCHF.MINI.IP",
        ]
        # Merge known pairs with any in database to ensure all defaults are valid options
        all_pairs = sorted(set(known_pairs) | set(enabled_pairs))

        new_pairs = st.multiselect(
            "Select Enabled Pairs",
            options=all_pairs,
            default=enabled_pairs,
            help="Trading pairs enabled for the strategy",
            key="enabled_pairs"
        )
        if set(new_pairs) != set(enabled_pairs):
            st.session_state.smc_pending_changes['enabled_pairs'] = new_pairs

    # Save button
    st.divider()
    if st.session_state.smc_pending_changes:
        st.warning(f"You have {len(st.session_state.smc_pending_changes)} pending changes")
        st.json(st.session_state.smc_pending_changes)

        change_reason = st.text_input(
            "Change Reason (required)",
            placeholder="Describe why you're making this change...",
            key="change_reason"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes", type="primary", disabled=not change_reason):
                success = save_global_config(
                    config['id'],
                    st.session_state.smc_pending_changes,
                    updated_by,
                    change_reason
                )
                if success:
                    st.success("Configuration saved successfully!")
                    st.session_state.smc_pending_changes = {}
                    # Clear the cache so fresh data is loaded
                    get_global_config.clear()
                    get_pair_overrides.clear()
                    st.rerun()
                else:
                    st.error("Failed to save configuration")

        with col2:
            if st.button("Discard Changes"):
                st.session_state.smc_pending_changes = {}
                st.rerun()
    else:
        st.info("No pending changes")


def render_pair_overrides(config: Dict[str, Any]):
    """Render per-pair override management"""
    st.subheader("Per-Pair Configuration Overrides")

    config_id = config.get('id')
    if not config_id:
        st.error("No active configuration found")
        return

    # Load existing overrides
    overrides = get_pair_overrides(config_id)
    enabled_pairs = config.get('enabled_pairs', [])

    # Pair selector
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_pair = st.selectbox(
            "Select Pair to Configure",
            options=enabled_pairs,
            key="pair_override_select"
        )
    with col2:
        if st.button("Add New Override", key="add_override_btn"):
            st.session_state['creating_new_override'] = True

    if not selected_pair:
        st.info("Select a pair to view or edit its override settings")
        return

    # Check if override exists
    existing = next((o for o in overrides if o['epic'] == selected_pair), None)

    st.divider()

    # User for audit
    updated_by = st.session_state.get('smc_config_user', 'streamlit_user')

    if existing:
        st.markdown(f"**Existing Override for {selected_pair}**")

        # Display and edit override
        col1, col2 = st.columns(2)

        override_changes = {}

        with col1:
            # Override Enabled - this is always pair-specific (no global to inherit)
            new_enabled = st.checkbox(
                "Override Enabled",
                value=existing.get('is_enabled', True),
                help="Enable/disable this pair override",
                key=f"override_enabled_{selected_pair}"
            )
            if not _values_equal(new_enabled, existing.get('is_enabled')):
                override_changes['is_enabled'] = new_enabled

            # Allow Asian Session - inherit from global block_asian_session (inverted logic)
            asian_override_value = existing.get('allow_asian_session')
            global_block_asian = config.get('block_asian_session', True)
            asian_effective = asian_override_value if asian_override_value is not None else (not global_block_asian)
            new_allow_asian = st.checkbox(
                "Allow Asian Session",
                value=asian_effective,
                help=f"{'Inherited from global (block_asian={global_block_asian})' if asian_override_value is None else 'Explicit override for this pair'}",
                key=f"override_asian_{selected_pair}"
            )
            # Compare against effective value to avoid false positives when inheriting
            if not _values_equal(new_allow_asian, asian_effective):
                override_changes['allow_asian_session'] = new_allow_asian

            # SL Buffer - inherit from global
            sl_override_value = existing.get('sl_buffer_pips')
            global_sl_buffer = config.get('sl_buffer_pips', 6)
            sl_effective = float(sl_override_value) if sl_override_value is not None else float(global_sl_buffer)
            new_sl_buffer = st.number_input(
                "SL Buffer (pips)",
                value=sl_effective,
                min_value=0.0, max_value=30.0, step=0.5,
                help=f"{'Inherited from global (' + str(global_sl_buffer) + ')' if sl_override_value is None else 'Explicit override for this pair'}",
                key=f"override_sl_{selected_pair}"
            )
            # Compare against effective value to avoid false positives when inheriting
            if not _values_equal(new_sl_buffer, sl_effective):
                override_changes['sl_buffer_pips'] = new_sl_buffer

        with col2:
            # Fixed Stop Loss - inherit from global
            fixed_sl_override_value = existing.get('fixed_stop_loss_pips')
            global_fixed_sl = config.get('fixed_stop_loss_pips', 9.0)
            fixed_sl_effective = float(fixed_sl_override_value) if fixed_sl_override_value is not None else float(global_fixed_sl)
            new_fixed_sl = st.number_input(
                "Fixed Stop Loss (pips)",
                value=fixed_sl_effective,
                min_value=1.0, max_value=50.0, step=0.5,
                help=f"{'Inherited from global (' + str(global_fixed_sl) + ')' if fixed_sl_override_value is None else 'Explicit override for this pair'}",
                key=f"override_fixed_sl_{selected_pair}"
            )
            if not _values_equal(new_fixed_sl, fixed_sl_effective):
                override_changes['fixed_stop_loss_pips'] = new_fixed_sl

            # Fixed Take Profit - inherit from global
            fixed_tp_override_value = existing.get('fixed_take_profit_pips')
            global_fixed_tp = config.get('fixed_take_profit_pips', 15.0)
            fixed_tp_effective = float(fixed_tp_override_value) if fixed_tp_override_value is not None else float(global_fixed_tp)
            new_fixed_tp = st.number_input(
                "Fixed Take Profit (pips)",
                value=fixed_tp_effective,
                min_value=1.0, max_value=100.0, step=0.5,
                help=f"{'Inherited from global (' + str(global_fixed_tp) + ')' if fixed_tp_override_value is None else 'Explicit override for this pair'}",
                key=f"override_fixed_tp_{selected_pair}"
            )
            if not _values_equal(new_fixed_tp, fixed_tp_effective):
                override_changes['fixed_take_profit_pips'] = new_fixed_tp

        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            # Min Confidence - inherit from global
            conf_override_value = existing.get('min_confidence')
            global_min_conf = config.get('min_confidence_threshold', 0.48)
            conf_effective = float(conf_override_value) if conf_override_value is not None else float(global_min_conf)
            new_min_conf = st.number_input(
                "Min Confidence",
                value=conf_effective,
                min_value=0.0, max_value=1.0, step=0.01,
                help=f"{'Inherited from global (' + str(global_min_conf) + ')' if conf_override_value is None else 'Explicit override for this pair'}",
                key=f"override_conf_{selected_pair}"
            )
            # Compare against effective value to avoid false positives when inheriting
            if not _values_equal(new_min_conf, conf_effective):
                override_changes['min_confidence'] = new_min_conf

            # Max Confidence Cap - inherit from global
            max_conf_override_value = existing.get('max_confidence')
            global_max_conf = config.get('max_confidence_threshold', 0.75)
            max_conf_effective = float(max_conf_override_value) if max_conf_override_value is not None else float(global_max_conf)
            new_max_conf = st.number_input(
                "Max Confidence Cap",
                value=max_conf_effective,
                min_value=0.0, max_value=1.0, step=0.01,
                help=f"{'Inherited from global (' + str(global_max_conf) + ')' if max_conf_override_value is None else 'Explicit override for this pair'} - Signals above this are rejected",
                key=f"override_max_conf_{selected_pair}"
            )
            # Compare against effective value to avoid false positives when inheriting
            if not _values_equal(new_max_conf, max_conf_effective):
                override_changes['max_confidence'] = new_max_conf

        with col_conf2:
            # MACD Filter - inherit from global
            macd_override_value = existing.get('macd_filter_enabled')
            global_macd = config.get('macd_alignment_filter_enabled', True)
            macd_effective = macd_override_value if macd_override_value is not None else global_macd
            new_macd = st.checkbox(
                "MACD Filter Enabled",
                value=macd_effective,
                help=f"{'Inherited from global (' + str(global_macd) + ')' if macd_override_value is None else 'Explicit override for this pair'}",
                key=f"override_macd_{selected_pair}"
            )
            # Compare against effective value to avoid false positives when inheriting
            if not _values_equal(new_macd, macd_effective):
                override_changes['macd_filter_enabled'] = new_macd

        # Swing & Filter Settings - new per-pair parameters for optimizer recommendations
        st.markdown("**Swing & Filter Settings**")
        st.caption("These settings control TIER2 swing validation. Lower values = more lenient swing detection.")
        col_swing1, col_swing2 = st.columns(2)
        with col_swing1:
            # Min Swing ATR Multiplier - inherit from global
            swing_atr_value = existing.get('min_swing_atr_multiplier')
            global_swing_atr = config.get('min_swing_atr_multiplier', 0.25)
            swing_atr_effective = float(swing_atr_value) if swing_atr_value is not None else float(global_swing_atr)
            new_swing_atr = st.number_input(
                "Min Swing ATR Multiplier",
                value=swing_atr_effective,
                min_value=0.0, max_value=1.0, step=0.05,
                format="%.2f",
                help=f"{'Inherited from global (' + str(global_swing_atr) + ')' if swing_atr_value is None else 'Explicit override for this pair'}. Lower = more lenient swing validation.",
                key=f"override_swing_atr_{selected_pair}"
            )
            if not _values_equal(new_swing_atr, swing_atr_effective):
                override_changes['min_swing_atr_multiplier'] = new_swing_atr

        with col_swing2:
            # Swing Lookback Bars - inherit from global
            swing_lookback_value = existing.get('swing_lookback_bars')
            global_swing_lookback = config.get('swing_lookback_bars', 20)
            swing_lookback_effective = int(swing_lookback_value) if swing_lookback_value is not None else int(global_swing_lookback)
            new_swing_lookback = st.number_input(
                "Swing Lookback Bars",
                value=swing_lookback_effective,
                min_value=5, max_value=100, step=5,
                help=f"{'Inherited from global (' + str(global_swing_lookback) + ')' if swing_lookback_value is None else 'Explicit override for this pair'}. Bars to look back for swing detection.",
                key=f"override_swing_lookback_{selected_pair}"
            )
            if not _values_equal(new_swing_lookback, swing_lookback_effective):
                override_changes['swing_lookback_bars'] = new_swing_lookback

        # SMC Conflict Tolerance - how many conflicts to allow before rejecting
        st.markdown("**SMC Conflict Filter**")
        col_smc1, col_smc2 = st.columns(2)
        with col_smc1:
            conflict_tol_value = existing.get('smc_conflict_tolerance')
            global_conflict_tol = config.get('smc_conflict_tolerance', 0)
            conflict_tol_effective = int(conflict_tol_value) if conflict_tol_value is not None else int(global_conflict_tol)
            new_conflict_tol = st.number_input(
                "SMC Conflict Tolerance",
                value=conflict_tol_effective,
                min_value=0, max_value=4, step=1,
                help=f"{'Inherited from global (' + str(global_conflict_tol) + ')' if conflict_tol_value is None else 'Explicit override for this pair'}. Number of SMC conflicts to allow (0 = strict, reject on any conflict; 1+ = allow minor conflicts).",
                key=f"override_conflict_tol_{selected_pair}"
            )
            if not _values_equal(new_conflict_tol, conflict_tol_effective):
                override_changes['smc_conflict_tolerance'] = new_conflict_tol
        with col_smc2:
            st.caption("💡 Tolerance controls how strict the SMC conflict filter is. 0 = reject on ANY conflict (order flow, ranging, consensus). 1 = allow 1 minor conflict to pass.")

        # Min Volume Ratio - pair-specific threshold for volume filter
        st.markdown("**Volume Filter Settings**")
        col_vol1, col_vol2 = st.columns(2)
        with col_vol1:
            vol_ratio_value = existing.get('min_volume_ratio')
            global_vol_ratio = config.get('min_volume_ratio', 0.25)
            vol_ratio_effective = float(vol_ratio_value) if vol_ratio_value is not None else float(global_vol_ratio)
            new_vol_ratio = st.number_input(
                "Min Volume Ratio",
                value=vol_ratio_effective,
                min_value=0.0, max_value=2.0, step=0.05,
                format="%.2f",
                help=f"{'Inherited from global (' + str(global_vol_ratio) + ')' if vol_ratio_value is None else 'Explicit override for this pair'}. Signals with volume ratio below this are rejected. Set to 0 to disable volume filter for this pair.",
                key=f"override_vol_ratio_{selected_pair}"
            )
            if not _values_equal(new_vol_ratio, vol_ratio_effective):
                override_changes['min_volume_ratio'] = new_vol_ratio
        with col_vol2:
            st.caption("💡 Volume ratio = current candle volume / 20-period SMA. Values 0.75-1.00 historically have best win rate.")

        # Swing Proximity Settings (v2.15.1) - per-pair overrides
        st.markdown("**Swing Proximity Validation (TIER 4)**")
        st.caption("Per-pair overrides for swing proximity validation. NULL = inherits from global settings.")
        col_swing_prox1, col_swing_prox2 = st.columns(2)
        with col_swing_prox1:
            # Swing Proximity Enabled - inherit from global
            swing_prox_enabled_value = existing.get('swing_proximity_enabled')
            global_swing_prox_enabled = config.get('swing_proximity_enabled', True)
            swing_prox_enabled_effective = swing_prox_enabled_value if swing_prox_enabled_value is not None else global_swing_prox_enabled
            new_swing_prox_enabled = st.checkbox(
                "Swing Proximity Enabled",
                value=swing_prox_enabled_effective,
                help=f"{'Inherited from global (' + str(global_swing_prox_enabled) + ')' if swing_prox_enabled_value is None else 'Explicit override for this pair'}. Enable swing proximity validation.",
                key=f"override_swing_prox_enabled_{selected_pair}"
            )
            if not _values_equal(new_swing_prox_enabled, swing_prox_enabled_effective):
                override_changes['swing_proximity_enabled'] = new_swing_prox_enabled

            # Swing Proximity Min Distance - inherit from global
            swing_prox_dist_value = existing.get('swing_proximity_min_distance_pips')
            global_swing_prox_dist = config.get('swing_proximity_min_distance_pips', 12)
            swing_prox_dist_effective = int(swing_prox_dist_value) if swing_prox_dist_value is not None else int(global_swing_prox_dist)
            new_swing_prox_dist = st.number_input(
                "Min Distance from Swing (pips)",
                value=swing_prox_dist_effective,
                min_value=1, max_value=50, step=1,
                help=f"{'Inherited from global (' + str(global_swing_prox_dist) + ')' if swing_prox_dist_value is None else 'Explicit override for this pair'}. Min distance from opposing swing level.",
                key=f"override_swing_prox_dist_{selected_pair}"
            )
            if not _values_equal(new_swing_prox_dist, swing_prox_dist_effective):
                override_changes['swing_proximity_min_distance_pips'] = new_swing_prox_dist

        with col_swing_prox2:
            # Swing Proximity Strict Mode - inherit from global
            swing_prox_strict_value = existing.get('swing_proximity_strict_mode')
            global_swing_prox_strict = config.get('swing_proximity_strict_mode', True)
            swing_prox_strict_effective = swing_prox_strict_value if swing_prox_strict_value is not None else global_swing_prox_strict
            new_swing_prox_strict = st.checkbox(
                "Strict Mode (Reject vs Penalty)",
                value=swing_prox_strict_effective,
                help=f"{'Inherited from global (' + str(global_swing_prox_strict) + ')' if swing_prox_strict_value is None else 'Explicit override for this pair'}. If enabled, reject signals; if disabled, apply confidence penalty.",
                key=f"override_swing_prox_strict_{selected_pair}"
            )
            if not _values_equal(new_swing_prox_strict, swing_prox_strict_effective):
                override_changes['swing_proximity_strict_mode'] = new_swing_prox_strict

            st.caption("💡 Based on trade analysis: BUY near resistance = 25% WR, SELL near support = 0% WR. Trades with 15+ pips clearance = 50%+ WR.")

        # EMA Overrides (v2.16.0) - per-pair EMA settings for optimization
        st.markdown("**EMA Settings (TIER 1)**")
        st.caption("Per-pair overrides for EMA trend validation. NULL = inherits from global settings.")
        col_ema1, col_ema2 = st.columns(2)
        with col_ema1:
            # EMA Period - inherit from global
            ema_period_value = existing.get('ema_period')
            global_ema_period = config.get('ema_period', 50)
            ema_period_effective = int(ema_period_value) if ema_period_value is not None else int(global_ema_period)
            new_ema_period = st.number_input(
                "EMA Period",
                value=ema_period_effective,
                min_value=10, max_value=200, step=5,
                help=f"{'Inherited from global (' + str(global_ema_period) + ')' if ema_period_value is None else 'Explicit override for this pair'}. Period for 4H EMA trend filter.",
                key=f"override_ema_period_{selected_pair}"
            )
            if not _values_equal(new_ema_period, ema_period_effective):
                override_changes['ema_period'] = new_ema_period

        with col_ema2:
            # EMA Slope Validation Enabled - inherit from global
            ema_slope_value = existing.get('ema_slope_validation_enabled')
            global_ema_slope = config.get('ema_slope_validation_enabled', True)
            ema_slope_effective = ema_slope_value if ema_slope_value is not None else global_ema_slope
            new_ema_slope = st.checkbox(
                "EMA Slope Validation",
                value=ema_slope_effective,
                help=f"{'Inherited from global (' + str(global_ema_slope) + ')' if ema_slope_value is None else 'Explicit override for this pair'}. Reject counter-trend trades (BULL when EMA falling, BEAR when EMA rising).",
                key=f"override_ema_slope_{selected_pair}"
            )
            if not _values_equal(new_ema_slope, ema_slope_effective):
                override_changes['ema_slope_validation_enabled'] = new_ema_slope

            st.caption("💡 EMA slope validation prevents counter-trend trades. Higher EMA period = smoother, slower trend detection.")

        # Dynamic confidence thresholds (absolute values, not adjustments)
        st.markdown("**Dynamic Confidence Thresholds**")
        st.caption("These are absolute confidence thresholds that override the global minimum when conditions are met. NULL = uses global min_confidence.")
        col3, col4 = st.columns(2)
        with col3:
            # High Volume Confidence - pair-specific (default: 0.45)
            vol_conf_value = existing.get('high_volume_confidence')
            vol_conf_default = 0.45
            vol_conf_effective = float(vol_conf_value) if vol_conf_value is not None else vol_conf_default
            new_vol_conf = st.number_input(
                "High Volume Confidence",
                value=vol_conf_effective,
                min_value=0.0, max_value=1.0, step=0.01,
                help=f"{'Using default (' + str(vol_conf_default) + ')' if vol_conf_value is None else 'Explicit value for this pair'}",
                key=f"override_vol_conf_{selected_pair}"
            )
            # Compare against effective value to avoid false positives
            if not _values_equal(new_vol_conf, vol_conf_effective):
                override_changes['high_volume_confidence'] = new_vol_conf

            # Low ATR Confidence - pair-specific (default: 0.44)
            low_atr_value = existing.get('low_atr_confidence')
            low_atr_default = 0.44
            low_atr_effective = float(low_atr_value) if low_atr_value is not None else low_atr_default
            new_low_atr = st.number_input(
                "Low ATR Confidence",
                value=low_atr_effective,
                min_value=0.0, max_value=1.0, step=0.01,
                help=f"{'Using default (' + str(low_atr_default) + ')' if low_atr_value is None else 'Explicit value for this pair'}",
                key=f"override_low_atr_{selected_pair}"
            )
            # Compare against effective value to avoid false positives
            if not _values_equal(new_low_atr, low_atr_effective):
                override_changes['low_atr_confidence'] = new_low_atr

        with col4:
            # High ATR Confidence - pair-specific (default: 0.52)
            high_atr_value = existing.get('high_atr_confidence')
            high_atr_default = 0.52
            high_atr_effective = float(high_atr_value) if high_atr_value is not None else high_atr_default
            new_high_atr = st.number_input(
                "High ATR Confidence",
                value=high_atr_effective,
                min_value=0.0, max_value=1.0, step=0.01,
                help=f"{'Using default (' + str(high_atr_default) + ')' if high_atr_value is None else 'Explicit value for this pair'}",
                key=f"override_high_atr_{selected_pair}"
            )
            # Compare against effective value to avoid false positives
            if not _values_equal(new_high_atr, high_atr_effective):
                override_changes['high_atr_confidence'] = new_high_atr

            # Near EMA Confidence - pair-specific (default: 0.44)
            near_ema_value = existing.get('near_ema_confidence')
            near_ema_default = 0.44
            near_ema_effective = float(near_ema_value) if near_ema_value is not None else near_ema_default
            new_near_ema = st.number_input(
                "Near EMA Confidence",
                value=near_ema_effective,
                min_value=0.0, max_value=1.0, step=0.01,
                help=f"{'Using default (' + str(near_ema_default) + ')' if near_ema_value is None else 'Explicit value for this pair'}",
                key=f"override_near_ema_{selected_pair}"
            )
            # Compare against effective value to avoid false positives
            if not _values_equal(new_near_ema, near_ema_effective):
                override_changes['near_ema_confidence'] = new_near_ema

        # Direction-aware overrides (v2.12.0)
        st.markdown("**Direction-Aware Overrides**")
        st.caption("Enable to set different thresholds for BULL vs BEAR trades. Useful when one direction performs significantly better.")

        direction_enabled = existing.get('direction_overrides_enabled', False)
        new_direction_enabled = st.checkbox(
            "Enable Direction-Specific Overrides",
            value=direction_enabled,
            help="When enabled, BULL and BEAR trades will use their own thresholds instead of a single value",
            key=f"direction_enabled_{selected_pair}"
        )
        if not _values_equal(new_direction_enabled, direction_enabled):
            override_changes['direction_overrides_enabled'] = new_direction_enabled

        if new_direction_enabled:
            st.markdown("##### Pullback Thresholds (Fib %)")
            col_bull, col_bear = st.columns(2)

            with col_bull:
                st.markdown("**BULL Settings**")
                # Fib Min Bull
                fib_min_bull_value = existing.get('fib_pullback_min_bull')
                global_fib_min = config.get('fib_pullback_min', 0.236)
                fib_min_bull_effective = float(fib_min_bull_value) if fib_min_bull_value is not None else float(global_fib_min)
                new_fib_min_bull = st.number_input(
                    "Min Pullback (BULL)",
                    value=fib_min_bull_effective,
                    min_value=0.0, max_value=0.5, step=0.01,
                    format="%.3f",
                    help=f"Min Fib pullback for BULL trades. Global: {global_fib_min:.1%}",
                    key=f"fib_min_bull_{selected_pair}"
                )
                if not _values_equal(new_fib_min_bull, fib_min_bull_effective):
                    override_changes['fib_pullback_min_bull'] = new_fib_min_bull

                # Fib Max Bull
                fib_max_bull_value = existing.get('fib_pullback_max_bull')
                global_fib_max = config.get('fib_pullback_max', 0.70)
                fib_max_bull_effective = float(fib_max_bull_value) if fib_max_bull_value is not None else float(global_fib_max)
                new_fib_max_bull = st.number_input(
                    "Max Pullback (BULL)",
                    value=fib_max_bull_effective,
                    min_value=0.3, max_value=1.5, step=0.01,
                    format="%.3f",
                    help=f"Max Fib pullback for BULL trades. Global: {global_fib_max:.1%}",
                    key=f"fib_max_bull_{selected_pair}"
                )
                if not _values_equal(new_fib_max_bull, fib_max_bull_effective):
                    override_changes['fib_pullback_max_bull'] = new_fib_max_bull

                # Momentum Min Bull
                mom_min_bull_value = existing.get('momentum_min_depth_bull')
                global_mom_min = config.get('momentum_min_depth', -0.50)
                mom_min_bull_effective = float(mom_min_bull_value) if mom_min_bull_value is not None else float(global_mom_min)
                new_mom_min_bull = st.number_input(
                    "Momentum Min Depth (BULL)",
                    value=mom_min_bull_effective,
                    min_value=-2.0, max_value=0.0, step=0.05,
                    format="%.2f",
                    help=f"How far beyond break for momentum. Global: {global_mom_min:.0%}",
                    key=f"mom_min_bull_{selected_pair}"
                )
                if not _values_equal(new_mom_min_bull, mom_min_bull_effective):
                    override_changes['momentum_min_depth_bull'] = new_mom_min_bull

                # Volume Ratio Bull
                vol_bull_value = existing.get('min_volume_ratio_bull')
                global_vol = existing.get('min_volume_ratio') or config.get('min_volume_ratio', 0.50)
                vol_bull_effective = float(vol_bull_value) if vol_bull_value is not None else float(global_vol)
                new_vol_bull = st.number_input(
                    "Min Volume Ratio (BULL)",
                    value=vol_bull_effective,
                    min_value=0.0, max_value=2.0, step=0.05,
                    format="%.2f",
                    help=f"Min volume ratio for BULL. Pair default: {global_vol:.2f}",
                    key=f"vol_bull_{selected_pair}"
                )
                if not _values_equal(new_vol_bull, vol_bull_effective):
                    override_changes['min_volume_ratio_bull'] = new_vol_bull

                # Confidence Bull
                conf_bull_value = existing.get('min_confidence_bull')
                global_conf = existing.get('min_confidence') or config.get('min_confidence_threshold', 0.44)
                conf_bull_effective = float(conf_bull_value) if conf_bull_value is not None else float(global_conf)
                new_conf_bull = st.number_input(
                    "Min Confidence (BULL)",
                    value=conf_bull_effective,
                    min_value=0.0, max_value=1.0, step=0.01,
                    format="%.2f",
                    help=f"Min confidence for BULL. Pair default: {global_conf:.0%}",
                    key=f"conf_bull_{selected_pair}"
                )
                if not _values_equal(new_conf_bull, conf_bull_effective):
                    override_changes['min_confidence_bull'] = new_conf_bull

            with col_bear:
                st.markdown("**BEAR Settings**")
                # Fib Min Bear
                fib_min_bear_value = existing.get('fib_pullback_min_bear')
                fib_min_bear_effective = float(fib_min_bear_value) if fib_min_bear_value is not None else float(global_fib_min)
                new_fib_min_bear = st.number_input(
                    "Min Pullback (BEAR)",
                    value=fib_min_bear_effective,
                    min_value=0.0, max_value=0.5, step=0.01,
                    format="%.3f",
                    help=f"Min Fib pullback for BEAR trades. Global: {global_fib_min:.1%}",
                    key=f"fib_min_bear_{selected_pair}"
                )
                if not _values_equal(new_fib_min_bear, fib_min_bear_effective):
                    override_changes['fib_pullback_min_bear'] = new_fib_min_bear

                # Fib Max Bear
                fib_max_bear_value = existing.get('fib_pullback_max_bear')
                fib_max_bear_effective = float(fib_max_bear_value) if fib_max_bear_value is not None else float(global_fib_max)
                new_fib_max_bear = st.number_input(
                    "Max Pullback (BEAR)",
                    value=fib_max_bear_effective,
                    min_value=0.3, max_value=1.5, step=0.01,
                    format="%.3f",
                    help=f"Max Fib pullback for BEAR trades. Global: {global_fib_max:.1%}",
                    key=f"fib_max_bear_{selected_pair}"
                )
                if not _values_equal(new_fib_max_bear, fib_max_bear_effective):
                    override_changes['fib_pullback_max_bear'] = new_fib_max_bear

                # Momentum Min Bear
                mom_min_bear_value = existing.get('momentum_min_depth_bear')
                mom_min_bear_effective = float(mom_min_bear_value) if mom_min_bear_value is not None else float(global_mom_min)
                new_mom_min_bear = st.number_input(
                    "Momentum Min Depth (BEAR)",
                    value=mom_min_bear_effective,
                    min_value=-2.0, max_value=0.0, step=0.05,
                    format="%.2f",
                    help=f"How far beyond break for momentum. Global: {global_mom_min:.0%}",
                    key=f"mom_min_bear_{selected_pair}"
                )
                if not _values_equal(new_mom_min_bear, mom_min_bear_effective):
                    override_changes['momentum_min_depth_bear'] = new_mom_min_bear

                # Volume Ratio Bear
                vol_bear_value = existing.get('min_volume_ratio_bear')
                vol_bear_effective = float(vol_bear_value) if vol_bear_value is not None else float(global_vol)
                new_vol_bear = st.number_input(
                    "Min Volume Ratio (BEAR)",
                    value=vol_bear_effective,
                    min_value=0.0, max_value=2.0, step=0.05,
                    format="%.2f",
                    help=f"Min volume ratio for BEAR. Pair default: {global_vol:.2f}",
                    key=f"vol_bear_{selected_pair}"
                )
                if not _values_equal(new_vol_bear, vol_bear_effective):
                    override_changes['min_volume_ratio_bear'] = new_vol_bear

                # Confidence Bear
                conf_bear_value = existing.get('min_confidence_bear')
                conf_bear_effective = float(conf_bear_value) if conf_bear_value is not None else float(global_conf)
                new_conf_bear = st.number_input(
                    "Min Confidence (BEAR)",
                    value=conf_bear_effective,
                    min_value=0.0, max_value=1.0, step=0.01,
                    format="%.2f",
                    help=f"Min confidence for BEAR. Pair default: {global_conf:.0%}",
                    key=f"conf_bear_{selected_pair}"
                )
                if not _values_equal(new_conf_bear, conf_bear_effective):
                    override_changes['min_confidence_bear'] = new_conf_bear

        # Advanced parameter overrides (JSON)
        st.markdown("**Advanced Parameter Overrides (JSON)**")
        param_overrides = existing.get('parameter_overrides', {})
        if isinstance(param_overrides, str):
            param_overrides = json.loads(param_overrides) if param_overrides else {}

        param_json = st.text_area(
            "Parameter Overrides",
            value=json.dumps(param_overrides, indent=2),
            height=150,
            help="Override any global parameter for this pair",
            key=f"param_overrides_{selected_pair}"
        )
        try:
            new_params = json.loads(param_json) if param_json.strip() else {}
            if new_params != param_overrides:
                override_changes['parameter_overrides'] = new_params
        except json.JSONDecodeError:
            st.warning("Invalid JSON format")

        # Description
        new_desc = st.text_input(
            "Description",
            value=existing.get('description', ''),
            placeholder="Why this pair has special settings...",
            key=f"override_desc_{selected_pair}"
        )
        if not _values_equal(new_desc, existing.get('description', '')):
            override_changes['description'] = new_desc

        # Actions
        st.divider()
        col_save, col_delete = st.columns(2)

        with col_save:
            if override_changes:
                st.warning(f"{len(override_changes)} pending changes")
                reason = st.text_input("Change Reason", key=f"override_reason_{selected_pair}")
                if st.button("Save Override", type="primary", disabled=not reason):
                    success = save_pair_override(
                        config_id, selected_pair, override_changes, updated_by, reason
                    )
                    if success:
                        st.success("Override saved!")
                        get_global_config.clear()
                        get_pair_overrides.clear()
                        st.rerun()

        with col_delete:
            if st.button("Delete Override", type="secondary"):
                if st.checkbox("Confirm delete", key=f"confirm_delete_{selected_pair}"):
                    success = delete_pair_override(
                        existing['id'], updated_by, "Deleted via Streamlit UI"
                    )
                    if success:
                        st.success("Override deleted!")
                        get_global_config.clear()
                        get_pair_overrides.clear()
                        st.rerun()

    else:
        # Create new override
        st.markdown(f"**Create Override for {selected_pair}**")
        st.info("This pair has no override - using global defaults. Create one to customize.")

        new_override = {
            'is_enabled': True,
            'allow_asian_session': False,
            'sl_buffer_pips': config.get('sl_buffer_pips', 6),
            'fixed_stop_loss_pips': config.get('fixed_stop_loss_pips', 9.0),
            'fixed_take_profit_pips': config.get('fixed_take_profit_pips', 15.0),
            'min_confidence': config.get('min_confidence_threshold', 0.48),
            'max_confidence': config.get('max_confidence_threshold', 0.75),
            'macd_filter_enabled': True,
        }

        col1, col2 = st.columns(2)
        with col1:
            new_override['allow_asian_session'] = st.checkbox(
                "Allow Asian Session",
                value=False,
                key=f"new_asian_{selected_pair}"
            )
            new_override['sl_buffer_pips'] = st.number_input(
                "SL Buffer (pips)",
                value=float(config.get('sl_buffer_pips', 6)),
                min_value=0.0, max_value=30.0, step=0.5,
                key=f"new_sl_{selected_pair}"
            )
            new_override['macd_filter_enabled'] = st.checkbox(
                "MACD Filter Enabled",
                value=True,
                key=f"new_macd_{selected_pair}"
            )

        with col2:
            new_override['fixed_stop_loss_pips'] = st.number_input(
                "Fixed Stop Loss (pips)",
                value=float(config.get('fixed_stop_loss_pips', 9.0)),
                min_value=1.0, max_value=50.0, step=0.5,
                help="Fixed stop loss override for this pair",
                key=f"new_fixed_sl_{selected_pair}"
            )
            new_override['fixed_take_profit_pips'] = st.number_input(
                "Fixed Take Profit (pips)",
                value=float(config.get('fixed_take_profit_pips', 15.0)),
                min_value=1.0, max_value=100.0, step=0.5,
                help="Fixed take profit override for this pair",
                key=f"new_fixed_tp_{selected_pair}"
            )

        col3, col4 = st.columns(2)
        with col3:
            new_override['min_confidence'] = st.number_input(
                "Min Confidence",
                value=float(config.get('min_confidence_threshold', 0.48)),
                min_value=0.0, max_value=1.0, step=0.01,
                key=f"new_conf_{selected_pair}"
            )
        with col4:
            new_override['max_confidence'] = st.number_input(
                "Max Confidence Cap",
                value=float(config.get('max_confidence_threshold', 0.75)),
                min_value=0.0, max_value=1.0, step=0.01,
                help="Signals above this are rejected (paradox: high confidence = worse outcomes)",
                key=f"new_max_conf_{selected_pair}"
            )

        new_override['description'] = st.text_input(
            "Description",
            placeholder="Reason for this pair's special configuration...",
            key=f"new_desc_{selected_pair}"
        )

        reason = st.text_input("Creation Reason", key=f"new_reason_{selected_pair}")
        if st.button("Create Override", type="primary", disabled=not reason):
            success = save_pair_override(
                config_id, selected_pair, new_override, updated_by, reason
            )
            if success:
                st.success("Override created!")
                get_global_config.clear()
                get_pair_overrides.clear()
                st.rerun()

    # Show all existing overrides
    st.divider()
    st.markdown("**All Existing Overrides**")
    if overrides:
        df = pd.DataFrame(overrides)
        display_cols = ['epic', 'is_enabled', 'fixed_stop_loss_pips', 'fixed_take_profit_pips',
                       'sl_buffer_pips', 'min_confidence', 'max_confidence',
                       'allow_asian_session', 'macd_filter_enabled',
                       'min_swing_atr_multiplier', 'smc_conflict_tolerance',
                       'swing_proximity_enabled', 'swing_proximity_min_distance_pips',
                       'swing_proximity_strict_mode', 'description']
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)
    else:
        st.info("No pair overrides configured yet")


def render_effective_config(config: Dict[str, Any]):
    """Render effective configuration viewer (merged global + pair overrides)"""
    st.subheader("Effective Configuration Viewer")
    st.markdown("*View the merged configuration for any pair (global defaults + pair overrides)*")

    enabled_pairs = config.get('enabled_pairs', [])

    selected_pair = st.selectbox(
        "Select Pair",
        options=enabled_pairs,
        key="effective_pair_select"
    )

    if selected_pair:
        effective = get_effective_config_for_pair(selected_pair)

        if effective:
            # Display as formatted JSON
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Effective Configuration for {selected_pair}**")
                st.json(effective)

            with col2:
                st.markdown("**Quick Reference**")
                st.metric("Version", effective.get('version', 'N/A'))
                # Check pair override first, then fall back to global setting
                min_conf = effective.get('min_confidence') if effective.get('min_confidence') is not None else effective.get('min_confidence_threshold', 'N/A')
                st.metric("Min Confidence", min_conf)
                st.metric("SL Buffer (pips)", effective.get('sl_buffer_pips', 'N/A'))
                # Check pair override first, then fall back to global setting
                macd_enabled = effective.get('macd_filter_enabled') if effective.get('macd_filter_enabled') is not None else effective.get('macd_alignment_filter_enabled', False)
                st.metric("MACD Filter", "On" if macd_enabled else "Off")

                # Check if has overrides
                override = get_pair_override(config['id'], selected_pair)
                if override:
                    st.success("Has pair-specific overrides")
                else:
                    st.info("Using global defaults only")

            # Download button
            json_str = json.dumps(effective, indent=2, default=str)
            st.download_button(
                "Download as JSON",
                data=json_str,
                file_name=f"smc_config_{selected_pair.replace('.', '_')}.json",
                mime="application/json"
            )
        else:
            st.warning("Could not load effective configuration")

    # Compare multiple pairs
    st.divider()
    st.markdown("**Compare Multiple Pairs**")

    compare_pairs = st.multiselect(
        "Select pairs to compare",
        options=enabled_pairs,
        default=enabled_pairs[:3] if len(enabled_pairs) >= 3 else enabled_pairs,
        key="compare_pairs"
    )

    if compare_pairs:
        compare_data = []
        for pair in compare_pairs:
            eff = get_effective_config_for_pair(pair)
            if eff:
                # Check pair override first, then fall back to global setting
                macd_val = eff.get('macd_filter_enabled') if eff.get('macd_filter_enabled') is not None else eff.get('macd_alignment_filter_enabled')
                min_conf_val = eff.get('min_confidence') if eff.get('min_confidence') is not None else eff.get('min_confidence_threshold')
                compare_data.append({
                    'Pair': pair.split('.')[2] if '.' in pair else pair,
                    'Min Confidence': min_conf_val,
                    'SL Buffer': eff.get('sl_buffer_pips'),
                    'Asian Session': eff.get('allow_asian_session', 'Default'),
                    'MACD Filter': macd_val,
                    'Has Override': get_pair_override(config['id'], pair) is not None
                })

        if compare_data:
            df = pd.DataFrame(compare_data)
            st.dataframe(df, use_container_width=True)


def render_audit_trail():
    """Render audit trail and change history"""
    st.subheader("Configuration Audit Trail")
    st.markdown("*Track all configuration changes with full history*")

    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input(
            "Show last N changes",
            value=50, min_value=10, max_value=500,
            key="audit_limit"
        )
    with col2:
        # Filter by pair
        enabled_pairs = get_enabled_pairs()
        filter_pair = st.selectbox(
            "Filter by Pair (optional)",
            options=['All'] + enabled_pairs,
            key="audit_pair_filter"
        )

    epic_filter = filter_pair if filter_pair != 'All' else None
    audit_records = get_audit_history(limit=limit, epic=epic_filter)

    if audit_records:
        # Convert to DataFrame for display
        df = pd.DataFrame(audit_records)

        # Format columns
        if 'changed_at' in df.columns:
            df['changed_at'] = pd.to_datetime(df['changed_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

        display_cols = ['changed_at', 'change_type', 'changed_by', 'epic', 'change_reason']
        available_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(df[available_cols], use_container_width=True)

        # Expandable details
        st.markdown("**Change Details**")
        for i, record in enumerate(audit_records[:10]):
            with st.expander(f"{record.get('changed_at')} - {record.get('change_type')} by {record.get('changed_by')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Previous Values**")
                    prev = record.get('previous_values')
                    if prev:
                        if isinstance(prev, str):
                            prev = json.loads(prev)
                        st.json(prev)
                    else:
                        st.info("N/A")

                with col2:
                    st.markdown("**New Values**")
                    new = record.get('new_values')
                    if new:
                        if isinstance(new, str):
                            new = json.loads(new)
                        st.json(new)
                    else:
                        st.info("N/A")

                st.markdown(f"**Reason:** {record.get('change_reason', 'Not specified')}")
    else:
        st.info("No audit records found")


def render_parameter_optimizer(config: Dict[str, Any]):
    """Render parameter optimizer with recommendations and apply button"""
    st.subheader("Parameter Optimizer")
    st.markdown("*AI-powered parameter recommendations based on rejection outcome analysis*")

    # Settings
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        days = st.number_input(
            "Analysis Period (days)",
            min_value=7, max_value=90, value=30,
            help="Number of days of rejection outcomes to analyze",
            key="optimizer_days"
        )
    with col2:
        min_confidence = st.slider(
            "Min Confidence",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="Only show recommendations with this confidence or higher",
            key="optimizer_min_conf"
        )
    with col3:
        st.info("Recommendations are generated by analyzing rejection outcomes to identify filters that are too aggressive (rejecting winners) or working well (filtering losers).")

    # Fetch recommendations button
    if st.button("Fetch Recommendations", type="primary", key="fetch_recommendations"):
        with st.spinner("Analyzing rejection outcomes..."):
            st.session_state['optimizer_data'] = fetch_optimizer_recommendations(days)

    # Display results
    if 'optimizer_data' in st.session_state:
        data = st.session_state['optimizer_data']

        if data.get('error'):
            st.error(f"Error fetching recommendations: {data['error']}")
            st.info("Make sure the rejection_outcome_analyzer has been run to populate outcome data.")
            return

        recommendations = data.get('recommendations', [])
        direction_recommendations = data.get('direction_recommendations', [])
        pairs_needing_direction = data.get('pairs_needing_direction_config', [])

        # Filter by confidence
        filtered_recs = [r for r in recommendations if r.get('confidence', 0) >= min_confidence]
        filtered_dir_recs = [r for r in direction_recommendations if r.get('confidence', 0) >= min_confidence]

        # Check if we have any recommendations
        has_global_recs = len(filtered_recs) > 0
        has_dir_recs = len(filtered_dir_recs) > 0

        if not has_global_recs and not has_dir_recs:
            st.success("No recommendations at this time. Your filters appear to be well-balanced!")

            # Show stage metrics anyway
            stage_metrics = data.get('stage_metrics', [])
            if stage_metrics:
                st.markdown("### Stage Win Rates")
                stage_df = pd.DataFrame(stage_metrics)
                if not stage_df.empty and 'rejection_stage' in stage_df.columns:
                    display_cols = ['rejection_stage', 'would_be_win_rate', 'total_analyzed',
                                   'missed_profit_pips', 'avoided_loss_pips']
                    available = [c for c in display_cols if c in stage_df.columns]
                    st.dataframe(stage_df[available], use_container_width=True)
            return

        # Separate relax vs tighten for global/pair recommendations
        relax_recs = [r for r in filtered_recs if r.get('action') == 'relax']
        tighten_recs = [r for r in filtered_recs if r.get('action') == 'tighten']

        # Separate direction recommendations
        dir_relax_recs = [r for r in filtered_dir_recs if r.get('action') == 'relax']
        dir_strict_recs = [r for r in filtered_dir_recs if r.get('action') == 'keep_strict']

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Global/Pair Recs", len(filtered_recs))
        with col2:
            st.metric("Direction Recs", len(filtered_dir_recs))
        with col3:
            st.metric("Relax Filters", len(relax_recs) + len(dir_relax_recs))
        with col4:
            st.metric("Keep Strict", len(tighten_recs) + len(dir_strict_recs))
        with col5:
            total_impact = sum(r.get('impact_pips', 0) for r in filtered_recs + filtered_dir_recs)
            st.metric("Potential Impact", f"{total_impact:.0f} pips")

        st.divider()

        # Direction-aware summary if pairs need direction config
        if pairs_needing_direction:
            st.markdown("### 🎯 Pairs Needing Direction-Aware Config")
            st.caption("These pairs show >20% win rate difference between BULL and BEAR trades")

            for pair_info in pairs_needing_direction:
                pair = pair_info.get('pair', '')
                bull_wr = pair_info.get('bull_win_rate', 0)
                bear_wr = pair_info.get('bear_win_rate', 0)
                diff = pair_info.get('difference', 0)
                action = pair_info.get('recommended_action', '')

                if bull_wr > bear_wr:
                    st.warning(f"**{pair}**: BULL {bull_wr:.0f}% vs BEAR {bear_wr:.0f}% ({diff:.0f}% diff) - {action}")
                else:
                    st.warning(f"**{pair}**: BEAR {bear_wr:.0f}% vs BULL {bull_wr:.0f}% ({diff:.0f}% diff) - {action}")

            st.divider()

        # Direction-aware recommendations section
        if dir_relax_recs:
            st.markdown("### 🔵 Direction-Aware: Relax Filters")
            st.caption("Per-pair, per-direction filter relaxations based on win rate analysis")

            total_dir_missed = sum(r.get('impact_pips', 0) for r in dir_relax_recs)
            st.caption(f"Potential recovery: {total_dir_missed:.0f} pips")

            for i, rec in enumerate(dir_relax_recs):
                title = _format_direction_recommendation_title(rec)
                with st.expander(f"**{title}**", expanded=i == 0):
                    _render_direction_recommendation_detail(rec)

        if dir_strict_recs:
            st.markdown("### 🔒 Direction-Aware: Keep Strict")
            st.caption("These direction+pair combinations are correctly filtering losers")

            for i, rec in enumerate(dir_strict_recs[:5]):  # Show max 5
                title = _format_direction_recommendation_title(rec)
                with st.expander(f"**{title}**", expanded=False):
                    _render_direction_recommendation_detail(rec)

        st.divider()

        # Global/Pair recommendations
        if relax_recs:
            st.markdown("### 🟢 Global/Pair: Relax Filters (Capture More Winners)")
            total_missed = sum(r.get('impact_pips', 0) for r in relax_recs)
            st.caption(f"Potential recovery: {total_missed:.0f} pips")

            for i, rec in enumerate(relax_recs):
                with st.expander(f"**{_format_recommendation_title(rec)}**", expanded=i == 0):
                    _render_recommendation_detail(rec)

        if tighten_recs:
            st.markdown("### 🔴 Global/Pair: Tighten Filters (Avoid More Losers)")
            total_avoided = sum(r.get('impact_pips', 0) for r in tighten_recs)
            st.caption(f"Potential savings: {total_avoided:.0f} pips")

            for i, rec in enumerate(tighten_recs):
                with st.expander(f"**{_format_recommendation_title(rec)}**", expanded=False):
                    _render_recommendation_detail(rec)

        st.divider()

        # Apply button
        st.markdown("### Apply Recommendations")

        # User name for audit
        updated_by = st.text_input(
            "Your Name (for audit trail)",
            value=st.session_state.get('smc_config_user', 'streamlit_optimizer'),
            key="optimizer_user"
        )

        # Combine all recommendations for selection
        all_recs = filtered_recs + [r for r in filtered_dir_recs if r.get('action') == 'relax']

        # Selection for which to apply
        apply_options = ["All Recommendations", "Global/Pair Only", "Direction-Aware Only", "Relax Only", "Select Individual"]
        apply_mode = st.radio(
            "What to apply",
            options=apply_options,
            horizontal=True,
            key="apply_mode"
        )

        recs_to_apply = []
        if apply_mode == "All Recommendations":
            recs_to_apply = all_recs
        elif apply_mode == "Global/Pair Only":
            recs_to_apply = filtered_recs
        elif apply_mode == "Direction-Aware Only":
            recs_to_apply = [r for r in filtered_dir_recs if r.get('action') == 'relax']
        elif apply_mode == "Relax Only":
            recs_to_apply = relax_recs + dir_relax_recs
        elif apply_mode == "Select Individual":
            st.markdown("**Select recommendations to apply:**")
            for i, rec in enumerate(all_recs):
                if rec.get('scope') == 'direction':
                    title = _format_direction_recommendation_title(rec)
                else:
                    title = _format_recommendation_title(rec)
                if st.checkbox(title, key=f"select_rec_{i}"):
                    recs_to_apply.append(rec)

        if recs_to_apply:
            st.info(f"Ready to apply {len(recs_to_apply)} recommendation(s)")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply Selected", type="primary", key="apply_recommendations"):
                    with st.spinner("Applying recommendations..."):
                        result = apply_optimizer_recommendations(recs_to_apply, updated_by)

                    if result['success']:
                        st.success(f"Successfully applied {result['applied_count']} recommendation(s)!")
                        st.balloons()

                        # Show details
                        for detail in result['details']:
                            if detail.get('success'):
                                if detail['type'] == 'global':
                                    st.write(f"✅ `{detail['param']}`: {detail['old']} → {detail['new']}")
                                elif detail['type'] == 'direction':
                                    st.write(f"✅ `{detail['epic']}` {detail['direction']}: {', '.join(detail.get('params', []))}")
                                else:
                                    st.write(f"✅ `{detail['epic']}.{detail['param']}`: {detail['old']} → {detail['new']}")

                        # Clear cached data
                        if 'optimizer_data' in st.session_state:
                            del st.session_state['optimizer_data']

                        st.info("Configuration updated. The scanner will use new settings on next run.")
                    else:
                        st.error(f"Failed to apply some recommendations. {result['failed_count']} failed.")
                        for detail in result['details']:
                            if not detail.get('success'):
                                st.write(f"❌ {detail}")

            with col2:
                if st.button("Clear Recommendations", key="clear_recommendations"):
                    if 'optimizer_data' in st.session_state:
                        del st.session_state['optimizer_data']
                    st.rerun()
        else:
            st.warning("No recommendations selected to apply")

        # Show raw data
        with st.expander("View Raw Analysis Data"):
            st.markdown("**Stage Metrics**")
            stage_metrics = data.get('stage_metrics', [])
            if stage_metrics:
                st.dataframe(pd.DataFrame(stage_metrics), use_container_width=True)

            st.markdown("**Pair Metrics**")
            pair_metrics = data.get('pair_metrics', [])
            if pair_metrics:
                st.dataframe(pd.DataFrame(pair_metrics), use_container_width=True)

            st.markdown("**Pair+Direction Metrics**")
            pair_dir_metrics = data.get('pair_direction_metrics', [])
            if pair_dir_metrics:
                st.dataframe(pd.DataFrame(pair_dir_metrics), use_container_width=True)

            st.markdown("**Direction Summary**")
            dir_summary = data.get('direction_summary', {})
            if dir_summary:
                st.json(dir_summary)

            st.markdown("**Parameter Suggestions from API**")
            suggestions = data.get('suggestions', {})
            if suggestions:
                st.json(suggestions)


def _format_recommendation_title(rec: Dict[str, Any]) -> str:
    """Format a recommendation as a title string."""
    if rec['scope'] == 'global':
        return f"{rec['target']}: {rec['current_value']} → {rec['recommended_value']}"
    else:
        epic, param = rec['target']
        pair_name = rec.get('pair_name', epic.split('.')[2] if '.' in epic else epic)
        return f"{pair_name}.{param}: {rec['current_value']} → {rec['recommended_value']}"


def _render_recommendation_detail(rec: Dict[str, Any]):
    """Render detailed view of a recommendation."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Action:** {'Relax' if rec['action'] == 'relax' else 'Tighten'}")
        st.markdown(f"**Stage:** {rec.get('stage', 'N/A')}")
        st.markdown(f"**Confidence:** {rec.get('confidence', 0):.0%}")

    with col2:
        st.markdown(f"**Current:** {rec['current_value']}")
        st.markdown(f"**Recommended:** {rec['recommended_value']}")
        st.markdown(f"**Impact:** {rec.get('impact_pips', 0):.0f} pips")

    st.markdown(f"**Reason:** {rec.get('reason', 'N/A')}")


def _format_direction_recommendation_title(rec: Dict[str, Any]) -> str:
    """Format a direction-aware recommendation as a title string."""
    pair_name = rec.get('pair_name', 'UNKNOWN')
    direction = rec.get('direction', '')
    action = rec.get('action', '')
    win_rate = rec.get('win_rate', 0)
    impact = rec.get('impact_pips', 0)

    icon = "+" if direction == 'BULL' else "-" if direction == 'BEAR' else "o"

    if action == 'relax':
        return f"{icon} {pair_name} {direction}: WR {win_rate:.0f}% - Relax filters ({impact:.0f} pips)"
    else:
        return f"{icon} {pair_name} {direction}: WR {win_rate:.0f}% - Keep strict ({impact:.0f} pips avoided)"


def _render_direction_recommendation_detail(rec: Dict[str, Any]):
    """Render detailed view of a direction-aware recommendation."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Pair:** {rec.get('pair_name', 'N/A')}")
        st.markdown(f"**Direction:** {rec.get('direction', 'N/A')}")
        st.markdown(f"**Action:** {'Relax Filters' if rec.get('action') == 'relax' else 'Keep Strict'}")
        st.markdown(f"**Win Rate:** {rec.get('win_rate', 0):.0f}%")

    with col2:
        st.markdown(f"**Total Analyzed:** {rec.get('total_analyzed', 0)}")
        st.markdown(f"**Confidence:** {rec.get('confidence', 0):.0%}")
        st.markdown(f"**Impact:** {rec.get('impact_pips', 0):.0f} pips")

    st.markdown(f"**Reason:** {rec.get('reason', 'N/A')}")

    # Show suggested parameter changes if available
    suggested_params = rec.get('suggested_params')
    if suggested_params:
        st.markdown("**Suggested Parameter Changes:**")
        for param, values in suggested_params.items():
            current = values.get('current', 'default')
            suggested = values.get('suggested', 'N/A')
            st.write(f"  - `{param}`: {current} → {suggested}")
