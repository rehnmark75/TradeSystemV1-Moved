"""
Backtest Configuration Tab Component

Provides a GUI for configuring and triggering backtests.
Features:
- Core settings (epic, days, strategy, timeframe)
- Execution options (parallel, workers, chart)
- Parameter overrides (~25 parameters)
- Non-blocking submission with status checking
- Recent backtests list
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from services.backtest_runner_service import BacktestRunnerService


def render_backtest_config_tab():
    """Render Backtest Configuration tab"""
    service = BacktestRunnerService()

    # Header
    st.header("Backtest Configuration")
    st.markdown("Configure and run backtests via FastAPI (direct execution)")

    # Check FastAPI health
    health = service.check_fastapi_health()
    if health.get('status') == 'healthy':
        st.success("FastAPI backtest service: Connected")
    elif health.get('status') == 'unavailable':
        st.error("FastAPI backtest service: Cannot connect. Is fastapi-dev container running?")
    else:
        st.warning(f"FastAPI backtest service: {health.get('status', 'unknown')}")

    # Initialize session state
    if 'last_job_id' not in st.session_state:
        st.session_state.last_job_id = None
    if 'last_backtest_command' not in st.session_state:
        st.session_state.last_backtest_command = None

    # Quick Start Presets
    _render_quick_start_section()

    st.divider()

    # Main configuration sections
    col_left, col_right = st.columns([1, 1])

    with col_left:
        config = _render_core_settings(service)
        config.update(_render_execution_options())

    with col_right:
        _render_status_section(service)

    # Parameter Overrides (full width, collapsible)
    overrides = _render_parameter_overrides(service)
    config['overrides'] = overrides

    # Snapshot Management
    snapshot = _render_snapshot_section(service)
    config['snapshot'] = snapshot

    # Parameter Variation Testing
    variation_config = _render_param_variation_section()
    config['variation'] = variation_config

    st.divider()

    # Run & Status Section
    _render_run_section(service, config)

    # Recent Backtests
    _render_recent_backtests(service)


def _render_quick_start_section():
    """Render quick start preset buttons"""
    st.subheader("Quick Start Presets")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Quick Test", help="7 days, no chart", use_container_width=True):
            st.session_state.preset = 'quick'
            st.rerun()

    with col2:
        if st.button("Standard", help="14 days, with chart", use_container_width=True):
            st.session_state.preset = 'standard'
            st.rerun()

    with col3:
        if st.button("Comprehensive", help="30 days, parallel", use_container_width=True):
            st.session_state.preset = 'comprehensive'
            st.rerun()

    with col4:
        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state.preset = None
            st.rerun()


def _get_preset_values() -> Dict[str, Any]:
    """Get values based on selected preset"""
    preset = st.session_state.get('preset')

    if preset == 'quick':
        return {
            'days': 7,
            'chart': False,
            'parallel': False,
            'pipeline': False
        }
    elif preset == 'standard':
        return {
            'days': 14,
            'chart': True,
            'parallel': False,
            'pipeline': False
        }
    elif preset == 'comprehensive':
        return {
            'days': 30,
            'chart': True,
            'parallel': True,
            'pipeline': True,
            'workers': 4
        }

    # Default values
    return {
        'days': 14,
        'chart': True,
        'parallel': False,
        'pipeline': False
    }


def _render_core_settings(service: BacktestRunnerService) -> Dict[str, Any]:
    """Render core settings section and return config dict"""
    st.subheader("Core Settings")

    preset = _get_preset_values()

    # Epic selection
    epic_names = [e[0] for e in service.EPIC_OPTIONS]
    selected_epic = st.selectbox(
        "Currency Pair",
        epic_names,
        index=0,
        key="bt_epic"
    )

    # Time period
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input(
            "Days to Test",
            value=preset.get('days', 14),
            min_value=1,
            max_value=365,
            key="bt_days"
        )

    with col2:
        use_date_range = st.checkbox("Use specific date range", key="bt_use_dates")

    # Date range (if enabled)
    start_date = None
    end_date = None
    if use_date_range:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=14),
                key="bt_start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                key="bt_end_date"
            )

    # Strategy (only SMC_SIMPLE active)
    strategy = st.selectbox(
        "Strategy",
        service.STRATEGY_OPTIONS,
        index=0,
        key="bt_strategy"
    )

    # Timeframe
    timeframe = st.selectbox(
        "Timeframe",
        service.TIMEFRAME_OPTIONS,
        index=1,  # Default to 15m
        key="bt_timeframe"
    )

    config = {
        'epic': selected_epic,
        'days': days,
        'strategy': strategy,
        'timeframe': timeframe,
    }

    if use_date_range and start_date and end_date:
        config['start_date'] = start_date.strftime('%Y-%m-%d')
        config['end_date'] = end_date.strftime('%Y-%m-%d')

    return config


def _render_execution_options() -> Dict[str, Any]:
    """Render execution options section"""
    st.subheader("Execution Options")

    preset = _get_preset_values()

    # Parallel execution
    parallel = st.checkbox(
        "Enable Parallel Execution",
        value=preset.get('parallel', False),
        help="Split backtest into chunks for faster execution",
        key="bt_parallel"
    )

    workers = 4
    chunk_days = 7
    if parallel:
        col1, col2 = st.columns(2)
        with col1:
            workers = st.slider(
                "Worker Count",
                min_value=2,
                max_value=8,
                value=preset.get('workers', 4),
                key="bt_workers"
            )
        with col2:
            chunk_days = st.number_input(
                "Chunk Days",
                value=7,
                min_value=1,
                max_value=30,
                key="bt_chunk_days"
            )

    # Pipeline mode
    pipeline = st.checkbox(
        "Full Pipeline Mode",
        value=preset.get('pipeline', False),
        help="Enable full validation (slower but production-accurate)",
        key="bt_pipeline"
    )

    # Chart generation
    chart = st.checkbox(
        "Generate Chart",
        value=preset.get('chart', True),
        help="Generate and store chart in MinIO",
        key="bt_chart"
    )

    # Historical intelligence
    use_historical_intelligence = st.checkbox(
        "Use Historical Intelligence",
        value=False,
        help="Replay stored market intelligence from database (default: OFF). "
             "Enable to match exact live trading conditions.",
        key="bt_historical_intel"
    )

    return {
        'parallel': parallel,
        'workers': workers if parallel else None,
        'chunk_days': chunk_days if parallel else None,
        'pipeline': pipeline,
        'chart': chart,
        'use_historical_intelligence': use_historical_intelligence,
    }


def _render_status_section(service: BacktestRunnerService):
    """Render current backtest status section"""
    st.subheader("Current Status")

    if st.session_state.last_job_id:
        job_id = st.session_state.last_job_id

        # Check Status button
        if st.button("Check Status", key="bt_check_status", use_container_width=True):
            status = service.get_job_status(job_id)
            _render_job_status_display(status)

        st.caption(f"Last submitted: {job_id}")

        # Show command if available
        if st.session_state.last_backtest_command:
            with st.expander("Show Command", expanded=False):
                st.code(st.session_state.last_backtest_command, language="bash")
    else:
        st.info("No backtest submitted yet")


def _render_job_status_display(status: dict):
    """Render job status with useful information (FastAPI format)"""
    if 'error' in status:
        st.error(status['error'])
        return

    state = status.get('status', 'unknown')
    config = status.get('config', {})
    progress = status.get('progress', {})

    # Status indicator
    if state == 'running':
        # Calculate live elapsed time from started_at
        started_at = status.get('started_at')
        if started_at:
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(started_at)
                elapsed = (datetime.now() - start_dt).total_seconds()
            except Exception:
                elapsed = progress.get('elapsed_seconds', 0)
        else:
            elapsed = progress.get('elapsed_seconds', 0)

        phase = progress.get('phase', 'initializing')
        last_activity = progress.get('last_activity', '')

        # Phase display names
        phase_names = {
            'loading_data': '📥 Loading market data...',
            'processing': '⚙️ Processing...',
            'running_variations': '🔬 Running parameter variations...',
            'analyzing_signals': '📊 Analyzing signals...',
            'generating_chart': '📈 Generating chart...',
            'completing': '✅ Completing...',
            'initializing': '🚀 Initializing...'
        }
        phase_display = phase_names.get(phase, f'⏳ {phase}')

        st.info(phase_display)

        # Show elapsed time (always show, calculated live)
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if minutes > 0:
            st.caption(f"⏱️ Elapsed: {minutes}m {seconds}s")
        else:
            st.caption(f"⏱️ Elapsed: {seconds}s")

        # Show last activity if available
        if last_activity:
            st.caption(f"📝 {last_activity[:80]}")

        # Show variation progress if available
        if progress.get('current') and progress.get('total'):
            current = progress['current']
            total = progress['total']
            pct = (current / total) * 100
            st.progress(pct / 100, text=f"Variation {current}/{total} ({pct:.0f}%)")

        # Show config info
        if config:
            st.caption(f"Epic: {config.get('epic')} | Days: {config.get('days')} | Strategy: {config.get('strategy')}")

        # Show recent output (last few lines)
        recent_output = status.get('recent_output', [])
        if recent_output:
            with st.expander("Recent Output", expanded=False):
                for line in recent_output[-5:]:
                    # Clean up the line display
                    if line.strip():
                        st.text(line[:100])  # Truncate long lines

        if status.get('started_at'):
            st.caption(f"Started: {status['started_at']}")

    elif state == 'completed':
        st.success("Backtest completed!")

        # Show backtest results
        result = status.get('result', {})
        if result:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Signals", result.get('signal_count', 0))
            with col2:
                win_rate = result.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("Total Pips", f"{result.get('total_pips', 0):.1f}")
            with col4:
                duration = result.get('duration_seconds', 0) or 0
                st.metric("Duration", f"{duration:.0f}s")

            if result.get('chart_url'):
                st.info(f"Chart available: {result['chart_url']}")

            if result.get('execution_id'):
                st.caption(f"Execution ID: {result['execution_id']}")

        if status.get('completed_at'):
            st.caption(f"Completed: {status['completed_at']}")

    elif state == 'failed':
        st.error("Backtest failed")
        result = status.get('result', {})
        if result.get('error'):
            st.code(result['error'])

    else:
        st.warning(f"Status: {state}")


def _render_parameter_overrides(service: BacktestRunnerService) -> Dict[str, Any]:
    """Render parameter overrides section (~25 parameters)"""
    defaults = service.get_default_overrides()
    overrides = {}

    with st.expander("Strategy Parameter Overrides", expanded=False):
        st.markdown("*Override default strategy parameters for this backtest*")

        # Risk Management (6 params)
        st.markdown("##### Risk Management")
        col1, col2, col3 = st.columns(3)
        with col1:
            fixed_sl = st.number_input(
                "Stop Loss (pips)",
                value=defaults['fixed_stop_loss_pips'],
                step=0.5,
                key="ov_sl"
            )
            if fixed_sl != defaults['fixed_stop_loss_pips']:
                overrides['fixed_stop_loss_pips'] = fixed_sl

            sl_buffer = st.number_input(
                "SL Buffer (pips)",
                value=defaults['sl_buffer_pips'],
                step=0.5,
                key="ov_sl_buf"
            )
            if sl_buffer != defaults['sl_buffer_pips']:
                overrides['sl_buffer_pips'] = sl_buffer

        with col2:
            fixed_tp = st.number_input(
                "Take Profit (pips)",
                value=defaults['fixed_take_profit_pips'],
                step=0.5,
                key="ov_tp"
            )
            if fixed_tp != defaults['fixed_take_profit_pips']:
                overrides['fixed_take_profit_pips'] = fixed_tp

            min_rr = st.number_input(
                "Min Risk:Reward",
                value=defaults['min_risk_reward'],
                step=0.1,
                key="ov_rr"
            )
            if min_rr != defaults['min_risk_reward']:
                overrides['min_risk_reward'] = min_rr

        with col3:
            max_pos = st.number_input(
                "Max Position Size",
                value=defaults['max_position_size'],
                step=0.1,
                key="ov_pos"
            )
            if max_pos != defaults['max_position_size']:
                overrides['max_position_size'] = max_pos

            use_atr = st.checkbox(
                "Use ATR-based SL",
                value=defaults['use_atr_stop_loss'],
                key="ov_atr_sl"
            )
            if use_atr != defaults['use_atr_stop_loss']:
                overrides['use_atr_stop_loss'] = use_atr

        # Entry Filters (8 params)
        st.markdown("##### Entry Filters")
        col1, col2 = st.columns(2)
        with col1:
            min_conf = st.slider(
                "Min Confidence",
                min_value=0.30,
                max_value=0.80,
                value=defaults['min_confidence'],
                step=0.02,
                key="ov_conf"
            )
            if abs(min_conf - defaults['min_confidence']) > 0.01:
                overrides['min_confidence'] = min_conf

            max_conf = st.slider(
                "Max Confidence",
                min_value=0.50,
                max_value=1.00,
                value=defaults['max_confidence'],
                step=0.02,
                key="ov_max_conf"
            )
            if abs(max_conf - defaults['max_confidence']) > 0.01:
                overrides['max_confidence'] = max_conf

            ema_period = st.number_input(
                "EMA Period",
                value=defaults['ema_period'],
                min_value=10,
                max_value=200,
                key="ov_ema"
            )
            if ema_period != defaults['ema_period']:
                overrides['ema_period'] = ema_period

            swing_lookback = st.number_input(
                "Swing Lookback",
                value=defaults['swing_lookback_bars'],
                min_value=5,
                max_value=100,
                key="ov_swing"
            )
            if swing_lookback != defaults['swing_lookback_bars']:
                overrides['swing_lookback_bars'] = swing_lookback

        with col2:
            macd_filter = st.checkbox(
                "MACD Filter Enabled",
                value=defaults['macd_filter_enabled'],
                key="ov_macd"
            )
            if macd_filter != defaults['macd_filter_enabled']:
                overrides['macd_filter_enabled'] = macd_filter

            volume_filter = st.checkbox(
                "Volume Filter Enabled",
                value=defaults['volume_filter_enabled'],
                key="ov_vol"
            )
            if volume_filter != defaults['volume_filter_enabled']:
                overrides['volume_filter_enabled'] = volume_filter

            ema_align = st.checkbox(
                "EMA Alignment Required",
                value=defaults['require_ema_alignment'],
                key="ov_ema_align"
            )
            if ema_align != defaults['require_ema_alignment']:
                overrides['require_ema_alignment'] = ema_align

            trend_filter = st.checkbox(
                "Trend Filter Enabled",
                value=defaults['trend_filter_enabled'],
                key="ov_trend"
            )
            if trend_filter != defaults['trend_filter_enabled']:
                overrides['trend_filter_enabled'] = trend_filter

        # Volume Settings (6 params)
        st.markdown("##### Volume Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            volume_enabled = st.checkbox(
                "Volume Confirmation",
                value=defaults.get('volume_enabled', True),
                help="Require volume confirmation on swing break",
                key="ov_vol_enabled"
            )
            if volume_enabled != defaults.get('volume_enabled', True):
                overrides['volume_enabled'] = volume_enabled

            min_vol_ratio = st.number_input(
                "Min Volume Ratio",
                value=defaults.get('min_volume_ratio', 0.50),
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                help="Minimum volume vs SMA ratio (filter)",
                key="ov_min_vol"
            )
            if abs(min_vol_ratio - defaults.get('min_volume_ratio', 0.50)) > 0.01:
                overrides['min_volume_ratio'] = min_vol_ratio

        with col2:
            vol_sma = st.number_input(
                "Volume SMA Period",
                value=defaults.get('volume_sma_period', 20),
                min_value=5,
                max_value=50,
                help="Period for volume moving average",
                key="ov_vol_sma"
            )
            if vol_sma != defaults.get('volume_sma_period', 20):
                overrides['volume_sma_period'] = vol_sma

            vol_spike = st.number_input(
                "Volume Spike Multiplier",
                value=defaults.get('volume_spike_multiplier', 1.3),
                min_value=1.0,
                max_value=3.0,
                step=0.1,
                help="Multiplier to confirm volume spike",
                key="ov_vol_spike"
            )
            if abs(vol_spike - defaults.get('volume_spike_multiplier', 1.3)) > 0.05:
                overrides['volume_spike_multiplier'] = vol_spike

        with col3:
            high_vol_thresh = st.number_input(
                "High Volume Threshold",
                value=defaults.get('high_volume_threshold', 0.70),
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                help="Threshold for high-volume confidence boost",
                key="ov_high_vol"
            )
            if abs(high_vol_thresh - defaults.get('high_volume_threshold', 0.70)) > 0.01:
                overrides['high_volume_threshold'] = high_vol_thresh

            allow_no_vol = st.checkbox(
                "Allow No Volume Data",
                value=defaults.get('allow_no_volume_data', True),
                help="Allow signals without volume data",
                key="ov_allow_no_vol"
            )
            if allow_no_vol != defaults.get('allow_no_volume_data', True):
                overrides['allow_no_volume_data'] = allow_no_vol

        # Session Filters (6 params)
        st.markdown("##### Session Filters")
        col1, col2 = st.columns(2)
        with col1:
            asian_block = st.checkbox(
                "Block Asian Session",
                value=defaults['block_asian_session'],
                key="ov_asian"
            )
            if asian_block != defaults['block_asian_session']:
                overrides['block_asian_session'] = asian_block

            london_open = st.number_input(
                "London Open Hour (UTC)",
                value=defaults['london_open_hour'],
                min_value=0,
                max_value=23,
                key="ov_london"
            )
            if london_open != defaults['london_open_hour']:
                overrides['london_open_hour'] = london_open

            ny_open = st.number_input(
                "NY Open Hour (UTC)",
                value=defaults['ny_open_hour'],
                min_value=0,
                max_value=23,
                key="ov_ny"
            )
            if ny_open != defaults['ny_open_hour']:
                overrides['ny_open_hour'] = ny_open

        with col2:
            weekend_filter = st.checkbox(
                "Weekend Filter",
                value=defaults['weekend_filter_enabled'],
                key="ov_weekend"
            )
            if weekend_filter != defaults['weekend_filter_enabled']:
                overrides['weekend_filter_enabled'] = weekend_filter

            session_end_buf = st.number_input(
                "Session End Buffer (mins)",
                value=defaults['session_end_buffer_minutes'],
                min_value=0,
                max_value=120,
                key="ov_end_buf"
            )
            if session_end_buf != defaults['session_end_buffer_minutes']:
                overrides['session_end_buffer_minutes'] = session_end_buf

            news_filter = st.checkbox(
                "Filter High Impact News",
                value=defaults['high_impact_news_filter'],
                key="ov_news"
            )
            if news_filter != defaults['high_impact_news_filter']:
                overrides['high_impact_news_filter'] = news_filter

        # Advanced (5 params)
        st.markdown("##### Advanced")
        col1, col2 = st.columns(2)
        with col1:
            cooldown = st.number_input(
                "Signal Cooldown (mins)",
                value=defaults['signal_cooldown_minutes'],
                min_value=0,
                max_value=360,
                key="ov_cool"
            )
            if cooldown != defaults['signal_cooldown_minutes']:
                overrides['signal_cooldown_minutes'] = cooldown

            max_daily = st.number_input(
                "Max Daily Signals",
                value=defaults['max_daily_signals'],
                min_value=1,
                max_value=20,
                key="ov_daily"
            )
            if max_daily != defaults['max_daily_signals']:
                overrides['max_daily_signals'] = max_daily

            require_sweep = st.checkbox(
                "Require Liquidity Sweep",
                value=defaults['require_liquidity_sweep'],
                key="ov_sweep"
            )
            if require_sweep != defaults['require_liquidity_sweep']:
                overrides['require_liquidity_sweep'] = require_sweep

        with col2:
            fvg_size = st.number_input(
                "Min FVG Size (pips)",
                value=defaults['fvg_minimum_size_pips'],
                step=0.5,
                key="ov_fvg"
            )
            if fvg_size != defaults['fvg_minimum_size_pips']:
                overrides['fvg_minimum_size_pips'] = fvg_size

            disp_mult = st.number_input(
                "Displacement ATR Mult",
                value=defaults['displacement_atr_multiplier'],
                step=0.1,
                key="ov_disp"
            )
            if disp_mult != defaults['displacement_atr_multiplier']:
                overrides['displacement_atr_multiplier'] = disp_mult

        # Swing & ATR Settings (Per-pair parameters)
        st.markdown("##### Swing & ATR Settings")
        col1, col2 = st.columns(2)
        with col1:
            min_swing_atr = st.number_input(
                "Min Swing ATR Multiplier",
                value=defaults.get('min_swing_atr_multiplier', 0.25),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="Lower = more lenient swing validation",
                key="ov_swing_atr"
            )
            if abs(min_swing_atr - defaults.get('min_swing_atr_multiplier', 0.25)) > 0.01:
                overrides['min_swing_atr_multiplier'] = min_swing_atr

            swing_prox_enabled = st.checkbox(
                "Swing Proximity Enabled",
                value=defaults.get('swing_proximity_enabled', True),
                help="Enable TIER 4 swing proximity validation",
                key="ov_swing_prox_enabled"
            )
            if swing_prox_enabled != defaults.get('swing_proximity_enabled', True):
                overrides['swing_proximity_enabled'] = swing_prox_enabled

        with col2:
            swing_prox_dist = st.number_input(
                "Swing Proximity Min Distance (pips)",
                value=defaults.get('swing_proximity_min_distance_pips', 12),
                min_value=1,
                max_value=50,
                help="Min distance from opposing swing level",
                key="ov_swing_prox_dist"
            )
            if swing_prox_dist != defaults.get('swing_proximity_min_distance_pips', 12):
                overrides['swing_proximity_min_distance_pips'] = swing_prox_dist

            swing_prox_strict = st.checkbox(
                "Swing Proximity Strict Mode",
                value=defaults.get('swing_proximity_strict_mode', True),
                help="Strict = reject; Non-strict = confidence penalty",
                key="ov_swing_prox_strict"
            )
            if swing_prox_strict != defaults.get('swing_proximity_strict_mode', True):
                overrides['swing_proximity_strict_mode'] = swing_prox_strict

        # Fibonacci Pullback Settings
        st.markdown("##### Fibonacci Pullback Settings")
        col1, col2 = st.columns(2)
        with col1:
            fib_min = st.number_input(
                "Fib Pullback Min",
                value=defaults.get('fib_pullback_min', 0.20),
                min_value=0.0,
                max_value=0.5,
                step=0.05,
                help="Minimum pullback depth (0.20 = 20%)",
                key="ov_fib_min"
            )
            if abs(fib_min - defaults.get('fib_pullback_min', 0.20)) > 0.01:
                overrides['fib_pullback_min'] = fib_min

        with col2:
            fib_max = st.number_input(
                "Fib Pullback Max",
                value=defaults.get('fib_pullback_max', 0.70),
                min_value=0.3,
                max_value=1.0,
                step=0.05,
                help="Maximum pullback depth (0.70 = 70%)",
                key="ov_fib_max"
            )
            if abs(fib_max - defaults.get('fib_pullback_max', 0.70)) > 0.01:
                overrides['fib_pullback_max'] = fib_max

        # High Volume Confidence
        st.markdown("##### Dynamic Confidence Settings")
        high_vol_conf = st.number_input(
            "High Volume Confidence",
            value=defaults.get('high_volume_confidence', 0.45),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Confidence threshold when high volume detected",
            key="ov_high_vol_conf"
        )
        if abs(high_vol_conf - defaults.get('high_volume_confidence', 0.45)) > 0.01:
            overrides['high_volume_confidence'] = high_vol_conf

        # Show active overrides
        if overrides:
            st.markdown("---")
            st.markdown(f"**Active Overrides:** {len(overrides)}")
            override_text = ", ".join([f"{k}={v}" for k, v in overrides.items()])
            st.caption(override_text)

    return overrides


def _render_snapshot_section(service: BacktestRunnerService) -> Optional[str]:
    """Render snapshot management section"""
    selected_snapshot = None

    with st.expander("Configuration Snapshots", expanded=False):
        snapshots = service.get_available_snapshots()

        if snapshots:
            selected_snapshot = st.selectbox(
                "Load Snapshot",
                ["None"] + snapshots,
                key="bt_snapshot"
            )
            if selected_snapshot == "None":
                selected_snapshot = None
        else:
            st.info("No snapshots available. Save current configuration as a snapshot using the CLI.")
            st.code("docker exec task-worker python /app/forex_scanner/snapshot_cli.py create <name>")

    return selected_snapshot


def _render_param_variation_section() -> Dict[str, Any]:
    """Render parameter variation testing section"""
    variation_config = {
        'enabled': False,
        'param_grid': {},
        'workers': 4,
        'rank_by': 'composite_score',
        'top_n': 10
    }

    with st.expander("Parameter Variation Testing", expanded=False):
        st.markdown("*Test multiple parameter combinations in parallel*")

        # Enable variation testing
        enable_variation = st.checkbox(
            "Enable Parameter Variation",
            key="bt_enable_variation",
            help="Run multiple backtests with different parameter values to find optimal settings"
        )
        variation_config['enabled'] = enable_variation

        if enable_variation:
            st.markdown("---")

            # Stop Loss variation
            col1, col2 = st.columns(2)

            with col1:
                vary_sl = st.checkbox("Vary Stop Loss", key="bt_vary_sl")
                if vary_sl:
                    sl_min = st.number_input("SL Min (pips)", value=8.0, step=1.0, key="bt_sl_min")
                    sl_max = st.number_input("SL Max (pips)", value=12.0, step=1.0, key="bt_sl_max")
                    sl_step = st.number_input("SL Step", value=2.0, step=0.5, key="bt_sl_step")

                    if sl_step > 0 and sl_max >= sl_min:
                        sl_values = []
                        v = sl_min
                        while v <= sl_max + 0.001:
                            sl_values.append(round(v, 1))
                            v += sl_step
                        variation_config['param_grid']['fixed_stop_loss_pips'] = sl_values
                        st.caption(f"Values: {sl_values}")

            with col2:
                vary_tp = st.checkbox("Vary Take Profit", key="bt_vary_tp")
                if vary_tp:
                    tp_min = st.number_input("TP Min (pips)", value=15.0, step=1.0, key="bt_tp_min")
                    tp_max = st.number_input("TP Max (pips)", value=25.0, step=1.0, key="bt_tp_max")
                    tp_step = st.number_input("TP Step", value=5.0, step=1.0, key="bt_tp_step")

                    if tp_step > 0 and tp_max >= tp_min:
                        tp_values = []
                        v = tp_min
                        while v <= tp_max + 0.001:
                            tp_values.append(round(v, 1))
                            v += tp_step
                        variation_config['param_grid']['fixed_take_profit_pips'] = tp_values
                        st.caption(f"Values: {tp_values}")

            # Confidence variation
            vary_conf = st.checkbox("Vary Min Confidence", key="bt_vary_conf")
            if vary_conf:
                col1, col2, col3 = st.columns(3)
                with col1:
                    conf_min = st.number_input("Conf Min", value=0.45, step=0.05, key="bt_conf_min")
                with col2:
                    conf_max = st.number_input("Conf Max", value=0.55, step=0.05, key="bt_conf_max")
                with col3:
                    conf_step = st.number_input("Conf Step", value=0.05, step=0.01, key="bt_conf_step")

                if conf_step > 0 and conf_max >= conf_min:
                    conf_values = []
                    v = conf_min
                    while v <= conf_max + 0.001:
                        conf_values.append(round(v, 2))
                        v += conf_step
                    variation_config['param_grid']['min_confidence'] = conf_values
                    st.caption(f"Values: {conf_values}")

            st.markdown("---")
            st.markdown("**Swing & Pullback Parameters**")

            # Swing Proximity variation
            col1, col2 = st.columns(2)

            with col1:
                vary_swing_prox = st.checkbox("Vary Swing Proximity Distance", key="bt_vary_swing_prox")
                if vary_swing_prox:
                    swing_min = st.number_input("Min Distance (pips)", value=8, step=1, key="bt_swing_prox_min")
                    swing_max = st.number_input("Max Distance (pips)", value=15, step=1, key="bt_swing_prox_max")
                    swing_step = st.number_input("Step", value=2, step=1, key="bt_swing_prox_step")

                    if swing_step > 0 and swing_max >= swing_min:
                        swing_values = []
                        v = swing_min
                        while v <= swing_max:
                            swing_values.append(int(v))
                            v += swing_step
                        variation_config['param_grid']['swing_proximity_min_distance_pips'] = swing_values
                        st.caption(f"Values: {swing_values}")

            with col2:
                vary_swing_lookback = st.checkbox("Vary Swing Lookback Bars", key="bt_vary_swing_lookback")
                if vary_swing_lookback:
                    lookback_min = st.number_input("Min Bars", value=15, step=5, key="bt_swing_lookback_min")
                    lookback_max = st.number_input("Max Bars", value=30, step=5, key="bt_swing_lookback_max")
                    lookback_step = st.number_input("Step", value=5, step=1, key="bt_swing_lookback_step")

                    if lookback_step > 0 and lookback_max >= lookback_min:
                        lookback_values = []
                        v = lookback_min
                        while v <= lookback_max:
                            lookback_values.append(int(v))
                            v += lookback_step
                        variation_config['param_grid']['swing_lookback_bars'] = lookback_values
                        st.caption(f"Values: {lookback_values}")

            # Pullback Fibonacci variations
            vary_pullback = st.checkbox("Vary Pullback Fibonacci Levels", key="bt_vary_pullback")
            if vary_pullback:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("*Fib Pullback Min*")
                    pb_min_start = st.number_input("Start", value=0.20, step=0.05, key="bt_pb_min_start")
                    pb_min_end = st.number_input("End", value=0.35, step=0.05, key="bt_pb_min_end")
                    pb_min_step = st.number_input("Step", value=0.05, step=0.01, key="bt_pb_min_step")

                    if pb_min_step > 0 and pb_min_end >= pb_min_start:
                        pb_min_values = []
                        v = pb_min_start
                        while v <= pb_min_end + 0.001:
                            pb_min_values.append(round(v, 3))
                            v += pb_min_step
                        variation_config['param_grid']['fib_pullback_min'] = pb_min_values
                        st.caption(f"Values: {pb_min_values}")

                with col2:
                    st.markdown("*Fib Pullback Max*")
                    pb_max_start = st.number_input("Start", value=0.60, step=0.05, key="bt_pb_max_start")
                    pb_max_end = st.number_input("End", value=0.75, step=0.05, key="bt_pb_max_end")
                    pb_max_step = st.number_input("Step", value=0.05, step=0.01, key="bt_pb_max_step")

                    if pb_max_step > 0 and pb_max_end >= pb_max_start:
                        pb_max_values = []
                        v = pb_max_start
                        while v <= pb_max_end + 0.001:
                            pb_max_values.append(round(v, 3))
                            v += pb_max_step
                        variation_config['param_grid']['fib_pullback_max'] = pb_max_values
                        st.caption(f"Values: {pb_max_values}")

            st.markdown("---")
            st.markdown("**Volume Parameters**")

            # Volume parameter variations
            col1, col2 = st.columns(2)

            with col1:
                vary_min_vol = st.checkbox("Vary Min Volume Ratio", key="bt_vary_min_vol")
                if vary_min_vol:
                    vol_min_start = st.number_input("Min Ratio Start", value=0.30, step=0.05, key="bt_vol_min_start")
                    vol_min_end = st.number_input("Min Ratio End", value=0.60, step=0.05, key="bt_vol_min_end")
                    vol_min_step = st.number_input("Step", value=0.10, step=0.05, key="bt_vol_min_step")

                    if vol_min_step > 0 and vol_min_end >= vol_min_start:
                        vol_min_values = []
                        v = vol_min_start
                        while v <= vol_min_end + 0.001:
                            vol_min_values.append(round(v, 2))
                            v += vol_min_step
                        variation_config['param_grid']['min_volume_ratio'] = vol_min_values
                        st.caption(f"Values: {vol_min_values}")

            with col2:
                vary_vol_spike = st.checkbox("Vary Volume Spike Multiplier", key="bt_vary_vol_spike")
                if vary_vol_spike:
                    spike_start = st.number_input("Spike Start", value=1.1, step=0.1, key="bt_spike_start")
                    spike_end = st.number_input("Spike End", value=1.5, step=0.1, key="bt_spike_end")
                    spike_step = st.number_input("Step", value=0.1, step=0.05, key="bt_spike_step")

                    if spike_step > 0 and spike_end >= spike_start:
                        spike_values = []
                        v = spike_start
                        while v <= spike_end + 0.001:
                            spike_values.append(round(v, 2))
                            v += spike_step
                        variation_config['param_grid']['volume_spike_multiplier'] = spike_values
                        st.caption(f"Values: {spike_values}")

            # High volume threshold variation
            vary_high_vol = st.checkbox("Vary High Volume Threshold", key="bt_vary_high_vol")
            if vary_high_vol:
                col1, col2, col3 = st.columns(3)
                with col1:
                    high_vol_start = st.number_input("Start", value=0.50, step=0.05, key="bt_high_vol_start")
                with col2:
                    high_vol_end = st.number_input("End", value=0.90, step=0.05, key="bt_high_vol_end")
                with col3:
                    high_vol_step = st.number_input("Step", value=0.10, step=0.05, key="bt_high_vol_step")

                if high_vol_step > 0 and high_vol_end >= high_vol_start:
                    high_vol_values = []
                    v = high_vol_start
                    while v <= high_vol_end + 0.001:
                        high_vol_values.append(round(v, 2))
                        v += high_vol_step
                    variation_config['param_grid']['high_volume_threshold'] = high_vol_values
                    st.caption(f"Values: {high_vol_values}")

            st.markdown("---")
            st.markdown("**ATR & Confidence Parameters**")

            col1, col2 = st.columns(2)

            with col1:
                vary_swing_atr = st.checkbox("Vary Swing ATR Multiplier", key="bt_vary_swing_atr")
                if vary_swing_atr:
                    atr_start = st.number_input("ATR Start", value=0.15, step=0.05, key="bt_atr_start")
                    atr_end = st.number_input("ATR End", value=0.35, step=0.05, key="bt_atr_end")
                    atr_step = st.number_input("Step", value=0.05, step=0.01, key="bt_atr_step")

                    if atr_step > 0 and atr_end >= atr_start:
                        atr_values = []
                        v = atr_start
                        while v <= atr_end + 0.001:
                            atr_values.append(round(v, 3))
                            v += atr_step
                        variation_config['param_grid']['min_swing_atr_multiplier'] = atr_values
                        st.caption(f"Values: {atr_values}")

            with col2:
                vary_high_vol_conf = st.checkbox("Vary High Volume Confidence", key="bt_vary_high_vol_conf")
                if vary_high_vol_conf:
                    hvc_start = st.number_input("HVC Start", value=0.35, step=0.05, key="bt_hvc_start")
                    hvc_end = st.number_input("HVC End", value=0.55, step=0.05, key="bt_hvc_end")
                    hvc_step = st.number_input("Step", value=0.05, step=0.01, key="bt_hvc_step")

                    if hvc_step > 0 and hvc_end >= hvc_start:
                        hvc_values = []
                        v = hvc_start
                        while v <= hvc_end + 0.001:
                            hvc_values.append(round(v, 2))
                            v += hvc_step
                        variation_config['param_grid']['high_volume_confidence'] = hvc_values
                        st.caption(f"Values: {hvc_values}")

            st.markdown("---")

            # Execution settings
            col1, col2, col3 = st.columns(3)

            with col1:
                variation_config['workers'] = st.slider(
                    "Parallel Workers",
                    min_value=2,
                    max_value=8,
                    value=4,
                    key="bt_vary_workers"
                )

            with col2:
                rank_options = {
                    'Composite Score': 'composite_score',
                    'Win Rate': 'win_rate',
                    'Total Pips': 'total_pips',
                    'Profit Factor': 'profit_factor'
                }
                rank_display = st.selectbox(
                    "Rank By",
                    list(rank_options.keys()),
                    key="bt_rank_by"
                )
                variation_config['rank_by'] = rank_options[rank_display]

            with col3:
                variation_config['top_n'] = st.number_input(
                    "Show Top N",
                    value=10,
                    min_value=1,
                    max_value=50,
                    key="bt_top_n"
                )

            # Calculate and show total combinations
            total_combos = 1
            for values in variation_config['param_grid'].values():
                total_combos *= len(values)

            if variation_config['param_grid']:
                if total_combos > 50:
                    st.warning(f"Will test {total_combos} combinations - this may take a while!")
                else:
                    st.info(f"Will test {total_combos} parameter combinations")
            else:
                st.info("Select at least one parameter to vary")

    return variation_config


def _render_run_section(service: BacktestRunnerService, config: Dict[str, Any]):
    """Render run and status section"""
    st.subheader("Run Backtest")

    # Action buttons row
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        run_button = st.button(
            "Start Backtest",
            type="primary",
            use_container_width=True,
            key="bt_run"
        )

    with col2:
        check_button = st.button(
            "Check Status",
            use_container_width=True,
            key="bt_check_main"
        )

    # Show command preview
    with col3:
        with st.expander("Preview Command", expanded=False):
            cmd = service.build_command(config)
            st.code(' '.join(cmd), language="bash")

    # Handle Run button
    if run_button:
        with st.spinner("Starting backtest via FastAPI..."):
            result = service.submit_backtest(config, async_mode=True)

        if result.get('job_id'):
            st.session_state.last_job_id = result['job_id']
            st.session_state.last_backtest_command = result.get('command')
            st.success(f"Backtest started! Job ID: {result['job_id']}")
            st.info("Click 'Check Status' to monitor progress.")
        elif result.get('success'):
            # Direct result from sync mode
            res = result.get('result', {})
            _display_backtest_result(res)
        else:
            st.error(f"Failed to start: {result.get('error', 'Unknown error')}")

    # Handle Check Status button
    if check_button:
        job_id = st.session_state.last_job_id
        if job_id:
            with st.spinner("Checking status..."):
                status = service.get_job_status(job_id)
            _render_job_status_display(status)
        else:
            st.warning("No backtest submitted yet. Start a backtest first.")


def _display_backtest_result(result: Dict[str, Any]):
    """Display backtest result from sync execution"""
    if result.get('success'):
        st.success("Backtest completed successfully!")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Signals", result.get('signal_count', 0))
        with col2:
            st.metric("Win Rate", f"{result.get('win_rate', 0):.1f}%")
        with col3:
            st.metric("Total Pips", f"{result.get('total_pips', 0):.1f}")
        with col4:
            st.metric("Duration", f"{result.get('duration_seconds', 0):.0f}s")

        if result.get('chart_url'):
            st.info(f"Chart available: {result['chart_url']}")

        if result.get('execution_id'):
            st.caption(f"Execution ID: {result['execution_id']}")
    else:
        st.error(f"Backtest failed: {result.get('error', 'Unknown error')}")


def _render_recent_backtests(service: BacktestRunnerService):
    """Render recent backtests section"""
    st.markdown("---")
    st.subheader("Recent Backtests")

    recent = service.get_recent_backtests(limit=5)

    if recent.empty:
        st.info("No recent backtests found")
        return

    for _, row in recent.iterrows():
        _render_recent_backtest_row(row, service)


def _render_recent_backtest_row(row: pd.Series, service: BacktestRunnerService):
    """Render a single recent backtest row"""
    exec_id = row.get('id', 0)
    status = row.get('status', 'unknown')
    strategy = row.get('strategy_name', 'N/A')

    # Status icon
    if status == 'completed':
        status_icon = "✅"
    elif status == 'running':
        status_icon = "🔄"
    elif status == 'failed':
        status_icon = "❌"
    else:
        status_icon = "⚪"

    # Format timestamp
    start_time = row.get('start_time')
    if isinstance(start_time, pd.Timestamp):
        timestamp_str = start_time.strftime('%Y-%m-%d %H:%M')
    else:
        timestamp_str = str(start_time)[:16] if start_time else 'N/A'

    # Format epics
    epics = row.get('epics_tested', [])
    if isinstance(epics, list) and epics:
        epic_str = epics[0].split('.')[2][:6] if '.' in str(epics[0]) else str(epics[0])[:6]
    else:
        epic_str = str(epics)[:10] if epics else 'N/A'

    # Duration
    duration = row.get('execution_duration_seconds', 0) or 0

    # Create clickable row
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
    with col1:
        st.write(f"{status_icon} #{exec_id}")
    with col2:
        st.write(timestamp_str)
    with col3:
        st.write(f"{epic_str} | {strategy}")
    with col4:
        st.write(f"{duration:.0f}s" if duration else "-")
    with col5:
        if st.button("View", key=f"view_{exec_id}"):
            st.session_state.last_backtest_id = exec_id
            st.rerun()
