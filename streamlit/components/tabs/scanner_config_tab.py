"""
Scanner Global Configuration Tab Component

Renders configuration management UI for global scanner settings with:
- Core Settings: Scan interval, confidence, timeframes
- Duplicate Detection: Cooldowns, thresholds, presets
- Risk Management: Position sizing, SL/TP, limits
- Trading Hours: Session controls, cutoff times
- Safety Filters: EMA200 filter, circuit breaker, presets
- ADX Filter: Trend strength filter settings
- SMC Conflict Filter: Smart Money Concepts signal filtering
- Claude Validation: AI-powered trade validation settings
- Audit Trail: Change history and tracking
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from decimal import Decimal

from services.scanner_config_service import (
    get_global_config,
    get_config_by_category,
    save_global_config,
    apply_preset,
    get_audit_history,
    get_config_version_info,
    get_field_metadata,
    values_equal,
)


def render_scanner_config_tab():
    """Main entry point for Scanner Config tab"""
    st.header("Scanner Global Configuration")
    st.markdown("*Database-driven configuration for Forex Scanner*")

    # Load current config
    config = get_global_config()

    if not config:
        st.error("Failed to load Scanner configuration from database.")
        st.info("Make sure the strategy_config database has the scanner_global_config table.")
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
            if isinstance(updated_at, datetime):
                st.metric("Last Updated", updated_at.strftime("%Y-%m-%d %H:%M"))
            else:
                st.metric("Last Updated", str(updated_at)[:16])
    with col4:
        st.metric("Scan Interval", f"{config.get('scan_interval', 120)}s")

    st.divider()

    # Sub-tabs for different categories
    sub_tabs = st.tabs([
        "Core Settings",
        "Indicators",
        "Data Quality",
        "Trading Control",
        "Duplicate Detection",
        "Risk Management",
        "Trading Hours",
        "Safety Filters",
        "ADX Filter",
        "SMC Conflict",
        "Claude AI",
        "Audit Trail"
    ])

    with sub_tabs[0]:
        render_core_settings(config)

    with sub_tabs[1]:
        render_indicator_settings(config)

    with sub_tabs[2]:
        render_data_quality_settings(config)

    with sub_tabs[3]:
        render_trading_control_settings(config)

    with sub_tabs[4]:
        render_dedup_settings(config)

    with sub_tabs[5]:
        render_risk_settings(config)

    with sub_tabs[6]:
        render_trading_hours_settings(config)

    with sub_tabs[7]:
        render_safety_settings(config)

    with sub_tabs[8]:
        render_adx_settings(config)

    with sub_tabs[9]:
        render_smc_conflict_settings(config)

    with sub_tabs[10]:
        render_claude_validation_settings(config)

    with sub_tabs[11]:
        render_audit_trail()


def _get_float(config: Dict, key: str, default: float = 0.0) -> float:
    """Safely get float value from config."""
    val = config.get(key, default)
    if isinstance(val, Decimal):
        return float(val)
    return float(val) if val is not None else default


def _get_int(config: Dict, key: str, default: int = 0) -> int:
    """Safely get int value from config."""
    val = config.get(key, default)
    return int(val) if val is not None else default


def render_core_settings(config: Dict[str, Any]):
    """Render core scanner settings"""
    st.subheader("Core Scanner Settings")
    st.markdown("Basic scanner operation parameters")

    # User identification
    updated_by = st.text_input(
        "Your Name (for audit trail)",
        value=st.session_state.get('scanner_config_user', 'streamlit_user'),
        key="scanner_core_user"
    )
    st.session_state['scanner_config_user'] = updated_by

    # Initialize pending changes
    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    col1, col2 = st.columns(2)

    with col1:
        new_scan_interval = st.number_input(
            "Scan Interval (seconds)",
            value=_get_int(config, 'scan_interval', 120),
            min_value=30, max_value=600, step=10,
            help="Seconds between scanner runs",
            key="scan_interval"
        )
        if not values_equal(new_scan_interval, config.get('scan_interval')):
            st.session_state.scanner_pending_changes['scan_interval'] = new_scan_interval

        new_min_confidence = st.number_input(
            "Minimum Confidence",
            value=_get_float(config, 'min_confidence', 0.40),
            min_value=0.0, max_value=1.0, step=0.01,
            format="%.2f",
            help="Minimum signal confidence threshold (0.0-1.0)",
            key="min_confidence"
        )
        if not values_equal(new_min_confidence, config.get('min_confidence')):
            st.session_state.scanner_pending_changes['min_confidence'] = new_min_confidence

        new_timeframe = st.selectbox(
            "Default Timeframe",
            options=['5m', '15m', '1h', '4h'],
            index=['5m', '15m', '1h', '4h'].index(config.get('default_timeframe', '15m')),
            help="Default signal timeframe",
            key="default_timeframe"
        )
        if new_timeframe != config.get('default_timeframe'):
            st.session_state.scanner_pending_changes['default_timeframe'] = new_timeframe

    with col2:
        new_1m_synthesis = st.checkbox(
            "Use 1M Base Synthesis",
            value=config.get('use_1m_base_synthesis', True),
            help="Synthesize candles from 1m base for better gap resilience",
            key="use_1m_base_synthesis"
        )
        if not values_equal(new_1m_synthesis, config.get('use_1m_base_synthesis')):
            st.session_state.scanner_pending_changes['use_1m_base_synthesis'] = new_1m_synthesis

        new_align_boundaries = st.checkbox(
            "Align to Boundaries",
            value=config.get('scan_align_to_boundaries', True),
            help="Align scans to 15m candle close times (:00, :15, :30, :45)",
            key="scan_align_to_boundaries"
        )
        if not values_equal(new_align_boundaries, config.get('scan_align_to_boundaries')):
            st.session_state.scanner_pending_changes['scan_align_to_boundaries'] = new_align_boundaries

        new_boundary_offset = st.number_input(
            "Boundary Offset (seconds)",
            value=_get_int(config, 'scan_boundary_offset_seconds', 60),
            min_value=0, max_value=120, step=10,
            help="Seconds after boundary to scan (allows data to settle)",
            key="scan_boundary_offset_seconds"
        )
        if not values_equal(new_boundary_offset, config.get('scan_boundary_offset_seconds')):
            st.session_state.scanner_pending_changes['scan_boundary_offset_seconds'] = new_boundary_offset

    st.divider()

    # Multi-Timeframe Analysis Section
    st.markdown("**Multi-Timeframe Analysis**")
    col1, col2 = st.columns(2)

    with col1:
        new_enable_mtf = st.checkbox(
            "Enable Multi-Timeframe Analysis",
            value=config.get('enable_multi_timeframe_analysis', False),
            help="Analyze signals across multiple timeframes for confluence",
            key="enable_multi_timeframe_analysis"
        )
        if not values_equal(new_enable_mtf, config.get('enable_multi_timeframe_analysis')):
            st.session_state.scanner_pending_changes['enable_multi_timeframe_analysis'] = new_enable_mtf

    with col2:
        new_min_confluence = st.slider(
            "Min Confluence Score",
            min_value=0.0, max_value=1.0, step=0.05,
            value=_get_float(config, 'min_confluence_score', 0.30),
            help="Minimum confluence score required across timeframes",
            key="min_confluence_score"
        )
        if not values_equal(new_min_confluence, config.get('min_confluence_score')):
            st.session_state.scanner_pending_changes['min_confluence_score'] = new_min_confluence

    # MTF Enhanced Confidence (only shown if MTF is enabled)
    if new_enable_mtf:
        new_mtf_confidence = st.slider(
            "MTF Enhanced Min Confidence",
            min_value=0.0, max_value=1.0, step=0.05,
            value=_get_float(config, 'mtf_enhanced_min_confidence', 0.60),
            help="Lower confidence threshold for MTF-validated signals (allows lower base confidence when MTF confirms)",
            key="mtf_enhanced_min_confidence"
        )
        if not values_equal(new_mtf_confidence, config.get('mtf_enhanced_min_confidence')):
            st.session_state.scanner_pending_changes['mtf_enhanced_min_confidence'] = new_mtf_confidence

    # Save section
    render_save_section(config, 'core', updated_by)


def render_indicator_settings(config: Dict[str, Any]):
    """Render technical indicator settings (MACD, KAMA, BB, Zero Lag, Two-Pole)"""
    st.subheader("Technical Indicator Settings")
    st.markdown("Configure parameters for technical indicators used in signal analysis and performance collection")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    # MACD Settings
    st.markdown("### MACD Indicator")
    col1, col2, col3 = st.columns(3)

    with col1:
        new_macd_fast = st.number_input(
            "MACD Fast Period",
            value=_get_int(config, 'macd_fast_period', 12),
            min_value=2, max_value=50, step=1,
            help="Fast EMA period for MACD calculation (default: 12)",
            key="macd_fast_period"
        )
        if not values_equal(new_macd_fast, config.get('macd_fast_period')):
            st.session_state.scanner_pending_changes['macd_fast_period'] = new_macd_fast

    with col2:
        new_macd_slow = st.number_input(
            "MACD Slow Period",
            value=_get_int(config, 'macd_slow_period', 26),
            min_value=5, max_value=100, step=1,
            help="Slow EMA period for MACD calculation (default: 26)",
            key="macd_slow_period"
        )
        if not values_equal(new_macd_slow, config.get('macd_slow_period')):
            st.session_state.scanner_pending_changes['macd_slow_period'] = new_macd_slow

    with col3:
        new_macd_signal = st.number_input(
            "MACD Signal Period",
            value=_get_int(config, 'macd_signal_period', 9),
            min_value=2, max_value=50, step=1,
            help="Signal line EMA period (default: 9)",
            key="macd_signal_period"
        )
        if not values_equal(new_macd_signal, config.get('macd_signal_period')):
            st.session_state.scanner_pending_changes['macd_signal_period'] = new_macd_signal

    st.divider()

    # KAMA Settings
    st.markdown("### KAMA (Kaufman Adaptive Moving Average)")
    col1, col2, col3 = st.columns(3)

    with col1:
        new_kama_period = st.number_input(
            "KAMA Period",
            value=_get_int(config, 'kama_period', 10),
            min_value=2, max_value=50, step=1,
            help="Efficiency ratio period (default: 10)",
            key="kama_period"
        )
        if not values_equal(new_kama_period, config.get('kama_period')):
            st.session_state.scanner_pending_changes['kama_period'] = new_kama_period

    with col2:
        new_kama_fast = st.number_input(
            "KAMA Fast SC",
            value=_get_int(config, 'kama_fast', 2),
            min_value=1, max_value=20, step=1,
            help="Fast smoothing constant (default: 2)",
            key="kama_fast"
        )
        if not values_equal(new_kama_fast, config.get('kama_fast')):
            st.session_state.scanner_pending_changes['kama_fast'] = new_kama_fast

    with col3:
        new_kama_slow = st.number_input(
            "KAMA Slow SC",
            value=_get_int(config, 'kama_slow', 30),
            min_value=10, max_value=100, step=1,
            help="Slow smoothing constant (default: 30)",
            key="kama_slow"
        )
        if not values_equal(new_kama_slow, config.get('kama_slow')):
            st.session_state.scanner_pending_changes['kama_slow'] = new_kama_slow

    st.divider()

    # Bollinger Bands & Supertrend Settings
    st.markdown("### Bollinger Bands & Supertrend")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bollinger Bands**")
        new_bb_period = st.number_input(
            "BB Period",
            value=_get_int(config, 'bb_period', 20),
            min_value=5, max_value=100, step=1,
            help="Bollinger Bands period (default: 20)",
            key="bb_period"
        )
        if not values_equal(new_bb_period, config.get('bb_period')):
            st.session_state.scanner_pending_changes['bb_period'] = new_bb_period

        new_bb_std = st.number_input(
            "BB Std Dev",
            value=_get_float(config, 'bb_std_dev', 2.0),
            min_value=0.5, max_value=5.0, step=0.1,
            format="%.1f",
            help="Standard deviation multiplier (default: 2.0)",
            key="bb_std_dev"
        )
        if not values_equal(new_bb_std, config.get('bb_std_dev')):
            st.session_state.scanner_pending_changes['bb_std_dev'] = new_bb_std

    with col2:
        st.markdown("**Supertrend**")
        new_st_period = st.number_input(
            "Supertrend Period",
            value=_get_int(config, 'supertrend_period', 10),
            min_value=5, max_value=50, step=1,
            help="ATR period for Supertrend (default: 10)",
            key="supertrend_period"
        )
        if not values_equal(new_st_period, config.get('supertrend_period')):
            st.session_state.scanner_pending_changes['supertrend_period'] = new_st_period

        new_st_mult = st.number_input(
            "Supertrend Multiplier",
            value=_get_float(config, 'supertrend_multiplier', 3.0),
            min_value=1.0, max_value=10.0, step=0.1,
            format="%.1f",
            help="ATR multiplier (default: 3.0)",
            key="supertrend_multiplier"
        )
        if not values_equal(new_st_mult, config.get('supertrend_multiplier')):
            st.session_state.scanner_pending_changes['supertrend_multiplier'] = new_st_mult

    st.divider()

    # Zero Lag Settings
    st.markdown("### Zero Lag EMA")
    col1, col2 = st.columns(2)

    with col1:
        new_zl_length = st.number_input(
            "Zero Lag Length",
            value=_get_int(config, 'zero_lag_length', 21),
            min_value=5, max_value=100, step=1,
            help="Zero lag EMA period (default: 21)",
            key="zero_lag_length"
        )
        if not values_equal(new_zl_length, config.get('zero_lag_length')):
            st.session_state.scanner_pending_changes['zero_lag_length'] = new_zl_length

    with col2:
        new_zl_band = st.number_input(
            "Zero Lag Band Multiplier",
            value=_get_float(config, 'zero_lag_band_mult', 1.5),
            min_value=0.5, max_value=5.0, step=0.1,
            format="%.1f",
            help="Band multiplier for volatility bands (default: 1.5)",
            key="zero_lag_band_mult"
        )
        if not values_equal(new_zl_band, config.get('zero_lag_band_mult')):
            st.session_state.scanner_pending_changes['zero_lag_band_mult'] = new_zl_band

    st.divider()

    # Two-Pole Filter Settings
    st.markdown("### Two-Pole Butterworth Filter")
    col1, col2, col3 = st.columns(3)

    with col1:
        new_tp_filter = st.number_input(
            "Filter Length",
            value=_get_int(config, 'two_pole_filter_length', 20),
            min_value=5, max_value=100, step=1,
            help="Two-pole filter length (default: 20)",
            key="two_pole_filter_length"
        )
        if not values_equal(new_tp_filter, config.get('two_pole_filter_length')):
            st.session_state.scanner_pending_changes['two_pole_filter_length'] = new_tp_filter

    with col2:
        new_tp_sma = st.number_input(
            "SMA Length",
            value=_get_int(config, 'two_pole_sma_length', 25),
            min_value=5, max_value=100, step=1,
            help="SMA length for smoothing (default: 25)",
            key="two_pole_sma_length"
        )
        if not values_equal(new_tp_sma, config.get('two_pole_sma_length')):
            st.session_state.scanner_pending_changes['two_pole_sma_length'] = new_tp_sma

    with col3:
        new_tp_delay = st.number_input(
            "Signal Delay",
            value=_get_int(config, 'two_pole_signal_delay', 4),
            min_value=1, max_value=20, step=1,
            help="Signal delay periods (default: 4)",
            key="two_pole_signal_delay"
        )
        if not values_equal(new_tp_delay, config.get('two_pole_signal_delay')):
            st.session_state.scanner_pending_changes['two_pole_signal_delay'] = new_tp_delay

    # Info box
    with st.expander("About Indicator Settings", expanded=False):
        st.markdown("""
        These settings control technical indicator calculations used for:

        - **Signal Analysis**: Indicators are calculated for all signals regardless of strategy
        - **Performance Collection**: Historical indicator values are stored for analysis
        - **MACD Filter**: Used by SMC Simple strategy for momentum confirmation

        **Note**: Changes take effect on the next scan cycle. The scanner must be restarted
        for changes to fully propagate through the caching system.

        **Default Values**:
        - MACD: 12/26/9 (standard)
        - KAMA: 10/2/30 (standard Kaufman settings)
        - Bollinger Bands: 20 period, 2.0 std dev
        - Supertrend: 10 period, 3.0 multiplier
        - Zero Lag: 21 period, 1.5 band multiplier
        - Two-Pole: 20 filter, 25 SMA, 4 delay
        """)

    # Save section
    render_save_section(config, 'indicators', updated_by)


def render_data_quality_settings(config: Dict[str, Any]):
    """Render data quality settings for trading safety"""
    st.subheader("Data Quality Settings")
    st.markdown("Configure data quality filtering, performance, and data fetching controls")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    # Performance / Data Fetching Settings (CRITICAL)
    st.markdown("### ⚠️ Data Fetching Settings (Critical)")
    st.caption("These settings control how much data is fetched. Incorrect values can cause 'Insufficient data' errors.")

    col1, col2 = st.columns(2)

    with col1:
        new_batch_size = st.number_input(
            "Data Batch Size",
            value=_get_int(config, 'data_batch_size', 25000),
            min_value=5000, max_value=100000, step=5000,
            help="Max rows fetched for 1m synthesis. For 4H data with 1m synthesis, need ~14,400+ rows (60 bars × 240). Set to 25000+ to account for weekend gaps.",
            key="data_batch_size"
        )
        if not values_equal(new_batch_size, config.get('data_batch_size')):
            st.session_state.scanner_pending_changes['data_batch_size'] = new_batch_size

        new_reduced_lookback = st.checkbox(
            "Enable Reduced Lookback",
            value=config.get('reduced_lookback_hours', True),
            help="Reduce lookback period by the reduction factor below for faster scans",
            key="reduced_lookback_hours"
        )
        if not values_equal(new_reduced_lookback, config.get('reduced_lookback_hours')):
            st.session_state.scanner_pending_changes['reduced_lookback_hours'] = new_reduced_lookback

    with col2:
        new_enable_cache = st.checkbox(
            "Enable Data Cache",
            value=config.get('enable_data_cache', False),
            help="Cache fetched data to reduce database load (may use stale data)",
            key="enable_data_cache"
        )
        if not values_equal(new_enable_cache, config.get('enable_data_cache')):
            st.session_state.scanner_pending_changes['enable_data_cache'] = new_enable_cache

        new_lazy_loading = st.checkbox(
            "Lazy Indicator Loading",
            value=config.get('lazy_indicator_loading', True),
            help="Only calculate required indicators (faster)",
            key="lazy_indicator_loading"
        )
        if not values_equal(new_lazy_loading, config.get('lazy_indicator_loading')):
            st.session_state.scanner_pending_changes['lazy_indicator_loading'] = new_lazy_loading

    # Show calculated requirements
    batch_size = _get_int(config, 'data_batch_size', 25000)
    if batch_size < 14400:
        st.error(f"⚠️ Batch size ({batch_size}) is below minimum required for 4H strategy (14,400). This WILL cause 'Insufficient data' errors!")
    elif batch_size < 20000:
        st.warning(f"⚠️ Batch size ({batch_size}) may be too low during weekends/gaps. Recommend 25,000+")
    else:
        st.success(f"✅ Batch size ({batch_size}) is sufficient for 4H strategy with weekend gaps")

    st.divider()

    # Lookback Reduction
    st.markdown("### Lookback Configuration")
    col1, col2 = st.columns(2)

    with col1:
        new_lookback_factor = st.slider(
            "Lookback Reduction Factor",
            min_value=0.1,
            max_value=1.0,
            value=_get_float(config, 'lookback_reduction_factor', 0.7),
            step=0.05,
            format="%.2f",
            help="Reduces lookback period for data fetching. Lower = less data, faster scans. (default: 0.7)",
            key="lookback_reduction_factor"
        )
        if not values_equal(new_lookback_factor, config.get('lookback_reduction_factor')):
            st.session_state.scanner_pending_changes['lookback_reduction_factor'] = new_lookback_factor

    with col2:
        st.info(
            f"Current setting: {_get_float(config, 'lookback_reduction_factor', 0.7):.0%} of calculated lookback period"
        )

    st.divider()

    # Data Quality Filtering
    st.markdown("### Data Quality Filtering")

    new_enable_filtering = st.checkbox(
        "Enable Data Quality Filtering",
        value=config.get('enable_data_quality_filtering', False),
        help="Filter out data points flagged as unsafe or low quality before analysis",
        key="enable_data_quality_filtering"
    )
    if not values_equal(new_enable_filtering, config.get('enable_data_quality_filtering')):
        st.session_state.scanner_pending_changes['enable_data_quality_filtering'] = new_enable_filtering

    if new_enable_filtering:
        col1, col2 = st.columns(2)

        with col1:
            new_block_trading = st.checkbox(
                "Block Trading on Data Issues",
                value=config.get('block_trading_on_data_issues', True),
                help="Prevent trading when data quality issues are detected",
                key="block_trading_on_data_issues"
            )
            if not values_equal(new_block_trading, config.get('block_trading_on_data_issues')):
                st.session_state.scanner_pending_changes['block_trading_on_data_issues'] = new_block_trading

        with col2:
            new_min_quality = st.slider(
                "Minimum Quality Score",
                min_value=0.0,
                max_value=1.0,
                value=_get_float(config, 'min_quality_score_for_trading', 0.5),
                step=0.05,
                format="%.2f",
                help="Minimum data quality score required for trading (0.0-1.0). Default: 0.5",
                key="min_quality_score_for_trading"
            )
            if not values_equal(new_min_quality, config.get('min_quality_score_for_trading')):
                st.session_state.scanner_pending_changes['min_quality_score_for_trading'] = new_min_quality
    else:
        st.caption("Data quality filtering is disabled. All data points will be used regardless of quality.")

    # Info box
    with st.expander("About Data Quality Settings", expanded=False):
        st.markdown("""
        These settings control how the scanner handles data fetching and quality:

        **⚠️ Data Batch Size (CRITICAL)**
        - Maximum rows fetched from database for 1m candle synthesis
        - For 4H strategy: need 60 bars × 240 candles = 14,400 minimum
        - Set to 25,000+ to account for weekend gaps and holidays
        - **Too low = "Insufficient 4h data" errors!**

        **Enable Reduced Lookback**
        - When enabled, applies the reduction factor to lookback calculations
        - Reduces data fetching for faster scans

        **Enable Data Cache**
        - Caches fetched data in memory to reduce database load
        - May use slightly stale data between refreshes

        **Lazy Indicator Loading**
        - Only calculates indicators that are actually needed
        - Significantly improves performance

        **Lookback Reduction Factor**
        - Controls how much historical data is fetched
        - 0.7 = fetch 70% of the calculated optimal lookback
        - Lower values = faster scans but less historical context
        - Higher values = more data but slower performance

        **Data Quality Filtering**
        - When enabled, filters out data points marked as unsafe
        - Checks for price anomalies, gaps, and data quality scores
        - Prevents trading on poor quality data

        **Block Trading on Data Issues**
        - When enabled, completely blocks trading if data quality issues are detected
        - When disabled, logs warnings but continues trading

        **Minimum Quality Score**
        - Data points below this threshold are filtered out
        - 0.5 = moderate quality requirement (default)
        - 0.7+ = stricter quality requirement
        - 0.3 = permissive, allows lower quality data
        """)

    # Save section
    render_save_section(config, 'data_quality', updated_by)


def render_trading_control_settings(config: Dict[str, Any]):
    """Render trading control settings for enabling/disabling live trading"""
    st.subheader("Trading Control Settings")
    st.markdown("⚠️ **CRITICAL**: Control live trading execution. Changes take effect immediately.")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    # Warning banner
    if config.get('auto_trading_enabled', False) or config.get('enable_order_execution', False):
        st.error("🔴 **LIVE TRADING IS ENABLED** - Real orders will be executed!")
    else:
        st.success("🟢 **LIVE TRADING IS DISABLED** - Signals will be generated but no orders executed")

    st.divider()

    # Main trading controls
    st.markdown("### Master Trading Controls")

    col1, col2 = st.columns(2)

    with col1:
        new_auto_trading = st.checkbox(
            "Auto Trading Enabled",
            value=config.get('auto_trading_enabled', False),
            help="Master switch for automated trading. When enabled, the system will automatically execute trades based on signals.",
            key="auto_trading_enabled"
        )
        if not values_equal(new_auto_trading, config.get('auto_trading_enabled')):
            st.session_state.scanner_pending_changes['auto_trading_enabled'] = new_auto_trading

        if new_auto_trading:
            st.warning("⚠️ Auto trading will execute trades automatically without manual confirmation!")

    with col2:
        new_order_execution = st.checkbox(
            "Enable Order Execution",
            value=config.get('enable_order_execution', False),
            help="Secondary control for order execution. Both this AND auto_trading_enabled must be True for orders to execute.",
            key="enable_order_execution"
        )
        if not values_equal(new_order_execution, config.get('enable_order_execution')):
            st.session_state.scanner_pending_changes['enable_order_execution'] = new_order_execution

        if new_order_execution:
            st.warning("⚠️ Order execution is enabled!")

    # Current trading status
    st.divider()
    st.markdown("### Current Trading Status")

    auto_enabled = config.get('auto_trading_enabled', False)
    order_enabled = config.get('enable_order_execution', False)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Auto Trading",
            "ENABLED" if auto_enabled else "DISABLED",
            delta=None
        )

    with col2:
        st.metric(
            "Order Execution",
            "ENABLED" if order_enabled else "DISABLED",
            delta=None
        )

    with col3:
        trading_active = auto_enabled or order_enabled
        st.metric(
            "Live Trading",
            "ACTIVE" if trading_active else "INACTIVE",
            delta=None
        )

    # Info box
    with st.expander("About Trading Control Settings", expanded=False):
        st.markdown("""
        These settings control whether the trading system executes real orders:

        **Auto Trading Enabled**
        - Master switch for automated trading
        - When enabled, signals can trigger automatic order execution
        - When disabled, signals are generated but no orders are placed

        **Enable Order Execution**
        - Secondary safety control for order execution
        - Provides additional layer of protection
        - Either flag being True enables trading (OR logic)

        **Safety Notes**
        - Both flags default to `False` for safety
        - Changes take effect on the next scan cycle
        - Always verify your settings before enabling live trading
        - The scanner will fail to start if database configuration is unavailable

        **NO FALLBACK**
        - These settings are stored ONLY in the database
        - If the database is unavailable, the scanner will NOT start
        - This prevents accidental trading with stale or default settings
        """)

    # Save section
    render_save_section(config, 'trading_control', updated_by)


def render_dedup_settings(config: Dict[str, Any]):
    """Render duplicate detection settings"""
    st.subheader("Duplicate Detection Settings")
    st.markdown("Control how duplicate signals are detected and filtered")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    # Preset selector
    col1, col2 = st.columns([2, 1])
    with col1:
        current_preset = config.get('deduplication_preset', 'standard')
        new_preset = st.selectbox(
            "Deduplication Preset",
            options=['strict', 'standard', 'relaxed'],
            index=['strict', 'standard', 'relaxed'].index(current_preset),
            help="Pre-configured deduplication settings",
            key="dedup_preset"
        )
        if new_preset != current_preset:
            st.session_state.scanner_pending_changes['deduplication_preset'] = new_preset

    with col2:
        if st.button("Apply Preset", key="apply_dedup_preset"):
            if apply_preset(config['id'], 'deduplication', new_preset, updated_by):
                st.success(f"Applied {new_preset} preset")
                st.rerun()

    st.divider()

    # Main dedup toggle
    new_enable_dedup = st.checkbox(
        "Enable Duplicate Check",
        value=config.get('enable_duplicate_check', True),
        help="Master switch for duplicate detection",
        key="enable_duplicate_check"
    )
    if not values_equal(new_enable_dedup, config.get('enable_duplicate_check')):
        st.session_state.scanner_pending_changes['enable_duplicate_check'] = new_enable_dedup

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Cooldown Settings**")

        new_signal_cooldown = st.number_input(
            "Signal Cooldown (minutes)",
            value=_get_int(config, 'signal_cooldown_minutes', 15),
            min_value=1, max_value=120, step=1,
            help="Cooldown between signals for same epic",
            key="signal_cooldown_minutes"
        )
        if not values_equal(new_signal_cooldown, config.get('signal_cooldown_minutes')):
            st.session_state.scanner_pending_changes['signal_cooldown_minutes'] = new_signal_cooldown

        new_alert_cooldown = st.number_input(
            "Alert Cooldown (minutes)",
            value=_get_int(config, 'alert_cooldown_minutes', 5),
            min_value=1, max_value=60, step=1,
            help="Cooldown between same epic+signal alerts",
            key="alert_cooldown_minutes"
        )
        if not values_equal(new_alert_cooldown, config.get('alert_cooldown_minutes')):
            st.session_state.scanner_pending_changes['alert_cooldown_minutes'] = new_alert_cooldown

        new_strategy_cooldown = st.number_input(
            "Strategy Cooldown (minutes)",
            value=_get_int(config, 'strategy_cooldown_minutes', 3),
            min_value=1, max_value=30, step=1,
            help="Strategy-specific cooldown",
            key="strategy_cooldown_minutes"
        )
        if not values_equal(new_strategy_cooldown, config.get('strategy_cooldown_minutes')):
            st.session_state.scanner_pending_changes['strategy_cooldown_minutes'] = new_strategy_cooldown

        new_global_cooldown = st.number_input(
            "Global Cooldown (seconds)",
            value=_get_int(config, 'global_cooldown_seconds', 30),
            min_value=0, max_value=120, step=5,
            help="Global cooldown between any alerts",
            key="global_cooldown_seconds"
        )
        if not values_equal(new_global_cooldown, config.get('global_cooldown_seconds')):
            st.session_state.scanner_pending_changes['global_cooldown_seconds'] = new_global_cooldown

    with col2:
        st.markdown("**Rate Limits**")

        new_max_alerts_hour = st.number_input(
            "Max Alerts Per Hour",
            value=_get_int(config, 'max_alerts_per_hour', 50),
            min_value=1, max_value=200, step=5,
            help="Maximum alerts across all pairs per hour",
            key="max_alerts_per_hour"
        )
        if not values_equal(new_max_alerts_hour, config.get('max_alerts_per_hour')):
            st.session_state.scanner_pending_changes['max_alerts_per_hour'] = new_max_alerts_hour

        new_max_alerts_epic = st.number_input(
            "Max Alerts Per Epic/Hour",
            value=_get_int(config, 'max_alerts_per_epic_hour', 6),
            min_value=1, max_value=50, step=1,
            help="Maximum alerts per pair per hour",
            key="max_alerts_per_epic_hour"
        )
        if not values_equal(new_max_alerts_epic, config.get('max_alerts_per_epic_hour')):
            st.session_state.scanner_pending_changes['max_alerts_per_epic_hour'] = new_max_alerts_epic

        new_sensitivity = st.selectbox(
            "Duplicate Sensitivity",
            options=['strict', 'smart', 'loose'],
            index=['strict', 'smart', 'loose'].index(config.get('duplicate_sensitivity', 'smart')),
            help="How strictly to detect duplicate signals",
            key="duplicate_sensitivity"
        )
        if new_sensitivity != config.get('duplicate_sensitivity'):
            st.session_state.scanner_pending_changes['duplicate_sensitivity'] = new_sensitivity

    # Advanced settings in expander
    with st.expander("Advanced Deduplication Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            new_price_threshold = st.number_input(
                "Price Similarity Threshold",
                value=_get_float(config, 'price_similarity_threshold', 0.0002),
                min_value=0.0001, max_value=0.01, step=0.0001,
                format="%.4f",
                help="Price similarity threshold (pips-equivalent)",
                key="price_similarity_threshold"
            )
            if not values_equal(new_price_threshold, config.get('price_similarity_threshold')):
                st.session_state.scanner_pending_changes['price_similarity_threshold'] = new_price_threshold

            new_conf_threshold = st.number_input(
                "Confidence Similarity Threshold",
                value=_get_float(config, 'confidence_similarity_threshold', 0.05),
                min_value=0.01, max_value=0.20, step=0.01,
                format="%.2f",
                help="Confidence similarity threshold",
                key="confidence_similarity_threshold"
            )
            if not values_equal(new_conf_threshold, config.get('confidence_similarity_threshold')):
                st.session_state.scanner_pending_changes['confidence_similarity_threshold'] = new_conf_threshold

        with col2:
            new_db_dedup = st.checkbox(
                "Use Database Dedup Check",
                value=config.get('use_database_dedup_check', True),
                help="Check database for recent duplicates",
                key="use_database_dedup_check"
            )
            if not values_equal(new_db_dedup, config.get('use_database_dedup_check')):
                st.session_state.scanner_pending_changes['use_database_dedup_check'] = new_db_dedup

            new_dedup_window = st.number_input(
                "Database Dedup Window (minutes)",
                value=_get_int(config, 'database_dedup_window_minutes', 15),
                min_value=5, max_value=60, step=5,
                help="Database check window",
                key="database_dedup_window_minutes"
            )
            if not values_equal(new_dedup_window, config.get('database_dedup_window_minutes')):
                st.session_state.scanner_pending_changes['database_dedup_window_minutes'] = new_dedup_window

            new_debug = st.checkbox(
                "Deduplication Debug Mode",
                value=config.get('deduplication_debug_mode', False),
                help="Enable verbose deduplication logging",
                key="deduplication_debug_mode"
            )
            if not values_equal(new_debug, config.get('deduplication_debug_mode')):
                st.session_state.scanner_pending_changes['deduplication_debug_mode'] = new_debug

    # Signal Hash Cache Settings
    with st.expander("Signal Hash Cache Settings", expanded=False):
        st.markdown("Configure in-memory signal hash caching for deduplication")

        new_enable_alert_dedup = st.checkbox(
            "Enable Alert Deduplication",
            value=config.get('enable_alert_deduplication', True),
            help="Master switch for alert deduplication system",
            key="enable_alert_deduplication"
        )
        if not values_equal(new_enable_alert_dedup, config.get('enable_alert_deduplication')):
            st.session_state.scanner_pending_changes['enable_alert_deduplication'] = new_enable_alert_dedup

        col1, col2 = st.columns(2)

        with col1:
            new_hash_cache_expiry = st.number_input(
                "Hash Cache Expiry (minutes)",
                value=_get_int(config, 'signal_hash_cache_expiry_minutes', 15),
                min_value=5, max_value=60, step=5,
                help="Time before cached signal hashes expire",
                key="signal_hash_cache_expiry_minutes"
            )
            if not values_equal(new_hash_cache_expiry, config.get('signal_hash_cache_expiry_minutes')):
                st.session_state.scanner_pending_changes['signal_hash_cache_expiry_minutes'] = new_hash_cache_expiry

            new_max_cache_size = st.number_input(
                "Max Cache Size",
                value=_get_int(config, 'max_signal_hash_cache_size', 1000),
                min_value=100, max_value=5000, step=100,
                help="Maximum in-memory signal hash cache entries",
                key="max_signal_hash_cache_size"
            )
            if not values_equal(new_max_cache_size, config.get('max_signal_hash_cache_size')):
                st.session_state.scanner_pending_changes['max_signal_hash_cache_size'] = new_max_cache_size

        with col2:
            new_enable_hash_check = st.checkbox(
                "Enable Signal Hash Check",
                value=config.get('enable_signal_hash_check', False),
                help="Check signal hash for exact duplicates (disabled: cooldown sufficient)",
                key="enable_signal_hash_check"
            )
            if not values_equal(new_enable_hash_check, config.get('enable_signal_hash_check')):
                st.session_state.scanner_pending_changes['enable_signal_hash_check'] = new_enable_hash_check

            new_time_hash = st.checkbox(
                "Enable Time-Based Hash Components",
                value=config.get('enable_time_based_hash_components', False),
                help="Include time bucket in hash (can be too strict)",
                key="enable_time_based_hash_components"
            )
            if not values_equal(new_time_hash, config.get('enable_time_based_hash_components')):
                st.session_state.scanner_pending_changes['enable_time_based_hash_components'] = new_time_hash

    render_save_section(config, 'dedup', updated_by)


def render_risk_settings(config: Dict[str, Any]):
    """Render risk management settings"""
    st.subheader("Risk Management Settings")
    st.markdown("Position sizing, stop loss, take profit, and trade limits")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Position Sizing**")

        new_position_size = st.number_input(
            "Position Size (%)",
            value=_get_float(config, 'position_size_percent', 1.0),
            min_value=0.1, max_value=10.0, step=0.1,
            format="%.1f",
            help="Position size as % of account",
            key="position_size_percent"
        )
        if not values_equal(new_position_size, config.get('position_size_percent')):
            st.session_state.scanner_pending_changes['position_size_percent'] = new_position_size

        new_min_size = st.number_input(
            "Min Position Size",
            value=_get_float(config, 'min_position_size', 0.01),
            min_value=0.01, max_value=1.0, step=0.01,
            format="%.2f",
            help="Minimum position size",
            key="min_position_size"
        )
        if not values_equal(new_min_size, config.get('min_position_size')):
            st.session_state.scanner_pending_changes['min_position_size'] = new_min_size

        new_max_size = st.number_input(
            "Max Position Size",
            value=_get_float(config, 'max_position_size', 1.0),
            min_value=0.1, max_value=10.0, step=0.1,
            format="%.1f",
            help="Maximum position size",
            key="max_position_size"
        )
        if not values_equal(new_max_size, config.get('max_position_size')):
            st.session_state.scanner_pending_changes['max_position_size'] = new_max_size

        new_risk_percent = st.number_input(
            "Risk Per Trade (%)",
            value=_get_float(config, 'risk_per_trade_percent', 0.02) * 100,
            min_value=0.1, max_value=5.0, step=0.1,
            format="%.1f",
            help="Risk per trade as % of account",
            key="risk_per_trade_percent"
        )
        if not values_equal(new_risk_percent / 100, config.get('risk_per_trade_percent')):
            st.session_state.scanner_pending_changes['risk_per_trade_percent'] = new_risk_percent / 100

    with col2:
        st.markdown("**SL/TP and Limits**")

        new_sl_pips = st.number_input(
            "Stop Loss (pips)",
            value=_get_int(config, 'stop_loss_pips', 5),
            min_value=1, max_value=100, step=1,
            help="Default stop loss in pips",
            key="stop_loss_pips"
        )
        if not values_equal(new_sl_pips, config.get('stop_loss_pips')):
            st.session_state.scanner_pending_changes['stop_loss_pips'] = new_sl_pips

        new_tp_pips = st.number_input(
            "Take Profit (pips)",
            value=_get_int(config, 'take_profit_pips', 15),
            min_value=1, max_value=200, step=1,
            help="Default take profit in pips",
            key="take_profit_pips"
        )
        if not values_equal(new_tp_pips, config.get('take_profit_pips')):
            st.session_state.scanner_pending_changes['take_profit_pips'] = new_tp_pips

        new_rr = st.number_input(
            "Default Risk:Reward",
            value=_get_float(config, 'default_risk_reward', 2.0),
            min_value=0.5, max_value=5.0, step=0.1,
            format="%.1f",
            help="Default risk:reward ratio",
            key="default_risk_reward"
        )
        if not values_equal(new_rr, config.get('default_risk_reward')):
            st.session_state.scanner_pending_changes['default_risk_reward'] = new_rr

        new_max_positions = st.number_input(
            "Max Open Positions",
            value=_get_int(config, 'max_open_positions', 3),
            min_value=1, max_value=20, step=1,
            help="Maximum concurrent open positions",
            key="max_open_positions"
        )
        if not values_equal(new_max_positions, config.get('max_open_positions')):
            st.session_state.scanner_pending_changes['max_open_positions'] = new_max_positions

        new_max_daily = st.number_input(
            "Max Daily Trades",
            value=_get_int(config, 'max_daily_trades', 10),
            min_value=1, max_value=100, step=1,
            help="Maximum trades per day",
            key="max_daily_trades"
        )
        if not values_equal(new_max_daily, config.get('max_daily_trades')):
            st.session_state.scanner_pending_changes['max_daily_trades'] = new_max_daily

        new_max_risk = st.number_input(
            "Max Risk Per Trade ($)",
            value=_get_int(config, 'max_risk_per_trade', 30),
            min_value=1, max_value=500, step=5,
            help="Maximum dollar risk per trade",
            key="max_risk_per_trade"
        )
        if not values_equal(new_max_risk, config.get('max_risk_per_trade')):
            st.session_state.scanner_pending_changes['max_risk_per_trade'] = new_max_risk

    render_save_section(config, 'risk', updated_by)


def render_trading_hours_settings(config: Dict[str, Any]):
    """Render trading hours settings"""
    st.subheader("Trading Hours Settings")
    st.markdown("Session controls, market hours, and time-based restrictions")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trading Hours**")

        new_start_hour = st.number_input(
            "Trading Start Hour (Local)",
            value=_get_int(config, 'trading_start_hour', 0),
            min_value=0, max_value=23, step=1,
            help="Trading start hour in local time",
            key="trading_start_hour"
        )
        if not values_equal(new_start_hour, config.get('trading_start_hour')):
            st.session_state.scanner_pending_changes['trading_start_hour'] = new_start_hour

        new_end_hour = st.number_input(
            "Trading End Hour (Local)",
            value=_get_int(config, 'trading_end_hour', 23),
            min_value=0, max_value=23, step=1,
            help="Trading end hour in local time",
            key="trading_end_hour"
        )
        if not values_equal(new_end_hour, config.get('trading_end_hour')):
            st.session_state.scanner_pending_changes['trading_end_hour'] = new_end_hour

        new_cutoff = st.number_input(
            "Trading Cutoff Hour (UTC)",
            value=_get_int(config, 'trading_cutoff_time_utc', 20),
            min_value=0, max_value=23, step=1,
            help="No new trades after this hour UTC (Friday protection)",
            key="trading_cutoff_time_utc"
        )
        if not values_equal(new_cutoff, config.get('trading_cutoff_time_utc')):
            st.session_state.scanner_pending_changes['trading_cutoff_time_utc'] = new_cutoff

        new_timezone = st.text_input(
            "User Timezone",
            value=config.get('user_timezone', 'Europe/Stockholm'),
            help="Your local timezone",
            key="user_timezone"
        )
        if new_timezone != config.get('user_timezone'):
            st.session_state.scanner_pending_changes['user_timezone'] = new_timezone

    with col2:
        st.markdown("**Controls**")

        new_respect_hours = st.checkbox(
            "Respect Market Hours",
            value=config.get('respect_market_hours', False),
            help="Enforce trading hours restrictions",
            key="respect_market_hours"
        )
        if not values_equal(new_respect_hours, config.get('respect_market_hours')):
            st.session_state.scanner_pending_changes['respect_market_hours'] = new_respect_hours

        new_weekend = st.checkbox(
            "Weekend Scanning",
            value=config.get('weekend_scanning', False),
            help="Allow scanning during weekends",
            key="weekend_scanning"
        )
        if not values_equal(new_weekend, config.get('weekend_scanning')):
            st.session_state.scanner_pending_changes['weekend_scanning'] = new_weekend

        new_time_controls = st.checkbox(
            "Enable Trading Time Controls",
            value=config.get('enable_trading_time_controls', True),
            help="Enable time-based trading controls",
            key="enable_trading_time_controls"
        )
        if not values_equal(new_time_controls, config.get('enable_trading_time_controls')):
            st.session_state.scanner_pending_changes['enable_trading_time_controls'] = new_time_controls

        new_trade_cooldown = st.checkbox(
            "Trade Cooldown Enabled",
            value=config.get('trade_cooldown_enabled', True),
            help="Enable cooldown after trades",
            key="trade_cooldown_enabled"
        )
        if not values_equal(new_trade_cooldown, config.get('trade_cooldown_enabled')):
            st.session_state.scanner_pending_changes['trade_cooldown_enabled'] = new_trade_cooldown

        new_cooldown_mins = st.number_input(
            "Trade Cooldown (minutes)",
            value=_get_int(config, 'trade_cooldown_minutes', 30),
            min_value=5, max_value=120, step=5,
            help="Cooldown after trade open/close",
            key="trade_cooldown_minutes"
        )
        if not values_equal(new_cooldown_mins, config.get('trade_cooldown_minutes')):
            st.session_state.scanner_pending_changes['trade_cooldown_minutes'] = new_cooldown_mins

    render_save_section(config, 'trading_hours', updated_by)


def render_safety_settings(config: Dict[str, Any]):
    """Render safety filter settings"""
    st.subheader("Safety Filter Settings")
    st.markdown("Critical safety filters to prevent invalid signals")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    # Preset selector
    col1, col2 = st.columns([2, 1])
    with col1:
        current_preset = config.get('active_safety_preset', 'balanced')
        new_preset = st.selectbox(
            "Safety Filter Preset",
            options=['strict', 'balanced', 'permissive', 'emergency'],
            index=['strict', 'balanced', 'permissive', 'emergency'].index(current_preset),
            help="Pre-configured safety settings",
            key="safety_preset"
        )
        if new_preset != current_preset:
            st.session_state.scanner_pending_changes['active_safety_preset'] = new_preset

    with col2:
        if st.button("Apply Preset", key="apply_safety_preset"):
            if apply_preset(config['id'], 'safety_filter', new_preset, updated_by):
                st.success(f"Applied {new_preset} preset")
                st.rerun()

    st.divider()

    # Master switch
    new_enable_safety = st.checkbox(
        "Enable Critical Safety Filters",
        value=config.get('enable_critical_safety_filters', True),
        help="Master switch for all safety filters",
        key="enable_critical_safety_filters"
    )
    if not values_equal(new_enable_safety, config.get('enable_critical_safety_filters')):
        st.session_state.scanner_pending_changes['enable_critical_safety_filters'] = new_enable_safety

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**EMA Filters**")

        new_ema200_filter = st.checkbox(
            "EMA200 Contradiction Filter",
            value=config.get('enable_ema200_contradiction_filter', True),
            help="Reject signals contradicting EMA200 trend",
            key="enable_ema200_contradiction_filter"
        )
        if not values_equal(new_ema200_filter, config.get('enable_ema200_contradiction_filter')):
            st.session_state.scanner_pending_changes['enable_ema200_contradiction_filter'] = new_ema200_filter

        new_ema_stack_filter = st.checkbox(
            "EMA Stack Contradiction Filter",
            value=config.get('enable_ema_stack_contradiction_filter', True),
            help="Reject signals with perfect stack contradiction",
            key="enable_ema_stack_contradiction_filter"
        )
        if not values_equal(new_ema_stack_filter, config.get('enable_ema_stack_contradiction_filter')):
            st.session_state.scanner_pending_changes['enable_ema_stack_contradiction_filter'] = new_ema_stack_filter

        new_ema_margin = st.number_input(
            "EMA200 Minimum Margin",
            value=_get_float(config, 'ema200_minimum_margin', 0.002),
            min_value=0.0005, max_value=0.01, step=0.0005,
            format="%.4f",
            help="Minimum margin for contra-trend signals",
            key="ema200_minimum_margin"
        )
        if not values_equal(new_ema_margin, config.get('ema200_minimum_margin')):
            st.session_state.scanner_pending_changes['ema200_minimum_margin'] = new_ema_margin

    with col2:
        st.markdown("**Consensus & Circuit Breaker**")

        new_require_consensus = st.checkbox(
            "Require Indicator Consensus",
            value=config.get('require_indicator_consensus', True),
            help="Require multiple indicators to confirm",
            key="require_indicator_consensus"
        )
        if not values_equal(new_require_consensus, config.get('require_indicator_consensus')):
            st.session_state.scanner_pending_changes['require_indicator_consensus'] = new_require_consensus

        new_min_indicators = st.number_input(
            "Min Confirming Indicators",
            value=_get_int(config, 'min_confirming_indicators', 1),
            min_value=0, max_value=5, step=1,
            help="Minimum indicators that must confirm",
            key="min_confirming_indicators"
        )
        if not values_equal(new_min_indicators, config.get('min_confirming_indicators')):
            st.session_state.scanner_pending_changes['min_confirming_indicators'] = new_min_indicators

        new_circuit_breaker = st.checkbox(
            "Emergency Circuit Breaker",
            value=config.get('enable_emergency_circuit_breaker', True),
            help="Enable emergency rejection for too many contradictions",
            key="enable_emergency_circuit_breaker"
        )
        if not values_equal(new_circuit_breaker, config.get('enable_emergency_circuit_breaker')):
            st.session_state.scanner_pending_changes['enable_emergency_circuit_breaker'] = new_circuit_breaker

        new_max_contradictions = st.number_input(
            "Max Contradictions Allowed",
            value=_get_int(config, 'max_contradictions_allowed', 5),
            min_value=0, max_value=10, step=1,
            help="Maximum contradictions before rejection",
            key="max_contradictions_allowed"
        )
        if not values_equal(new_max_contradictions, config.get('max_contradictions_allowed')):
            st.session_state.scanner_pending_changes['max_contradictions_allowed'] = new_max_contradictions

    # Large candle filter in expander
    with st.expander("Large Candle Filter Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            new_large_candle = st.checkbox(
                "Enable Large Candle Filter",
                value=config.get('enable_large_candle_filter', True),
                help="Filter out signals after large price movements",
                key="enable_large_candle_filter"
            )
            if not values_equal(new_large_candle, config.get('enable_large_candle_filter')):
                st.session_state.scanner_pending_changes['enable_large_candle_filter'] = new_large_candle

            new_atr_mult = st.number_input(
                "Large Candle ATR Multiplier",
                value=_get_float(config, 'large_candle_atr_multiplier', 2.5),
                min_value=1.0, max_value=5.0, step=0.1,
                format="%.1f",
                help="Candles larger than X * ATR are 'large'",
                key="large_candle_atr_multiplier"
            )
            if not values_equal(new_atr_mult, config.get('large_candle_atr_multiplier')):
                st.session_state.scanner_pending_changes['large_candle_atr_multiplier'] = new_atr_mult

        with col2:
            new_consecutive = st.number_input(
                "Consecutive Large Candles Threshold",
                value=_get_int(config, 'consecutive_large_candles_threshold', 2),
                min_value=1, max_value=5, step=1,
                help="Block if X+ large candles recently",
                key="consecutive_large_candles_threshold"
            )
            if not values_equal(new_consecutive, config.get('consecutive_large_candles_threshold')):
                st.session_state.scanner_pending_changes['consecutive_large_candles_threshold'] = new_consecutive

            new_cooldown = st.number_input(
                "Large Candle Filter Cooldown",
                value=_get_int(config, 'large_candle_filter_cooldown', 3),
                min_value=1, max_value=10, step=1,
                help="Periods to wait after large candle",
                key="large_candle_filter_cooldown"
            )
            if not values_equal(new_cooldown, config.get('large_candle_filter_cooldown')):
                st.session_state.scanner_pending_changes['large_candle_filter_cooldown'] = new_cooldown

    render_save_section(config, 'safety', updated_by)


def render_adx_settings(config: Dict[str, Any]):
    """Render ADX filter settings"""
    st.subheader("ADX Trend Strength Filter")
    st.markdown("Filter signals based on trend strength using ADX indicator")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    # Main toggle
    new_adx_enabled = st.checkbox(
        "Enable ADX Filter",
        value=config.get('adx_filter_enabled', False),
        help="Filter signals based on ADX trend strength",
        key="adx_filter_enabled"
    )
    if not values_equal(new_adx_enabled, config.get('adx_filter_enabled')):
        st.session_state.scanner_pending_changes['adx_filter_enabled'] = new_adx_enabled

    col1, col2 = st.columns(2)

    with col1:
        new_adx_mode = st.selectbox(
            "ADX Filter Mode",
            options=['strict', 'moderate', 'permissive', 'disabled'],
            index=['strict', 'moderate', 'permissive', 'disabled'].index(
                config.get('adx_filter_mode', 'moderate')
            ),
            help="How strictly to filter by ADX",
            key="adx_filter_mode"
        )
        if new_adx_mode != config.get('adx_filter_mode'):
            st.session_state.scanner_pending_changes['adx_filter_mode'] = new_adx_mode

        new_adx_period = st.number_input(
            "ADX Period",
            value=_get_int(config, 'adx_period', 14),
            min_value=5, max_value=50, step=1,
            help="ADX calculation period (standard is 14)",
            key="adx_period"
        )
        if not values_equal(new_adx_period, config.get('adx_period')):
            st.session_state.scanner_pending_changes['adx_period'] = new_adx_period

    with col2:
        new_grace_period = st.number_input(
            "ADX Grace Period (bars)",
            value=_get_int(config, 'adx_grace_period_bars', 2),
            min_value=0, max_value=10, step=1,
            help="Allow X bars of weak ADX if previous trend was strong",
            key="adx_grace_period_bars"
        )
        if not values_equal(new_grace_period, config.get('adx_grace_period_bars')):
            st.session_state.scanner_pending_changes['adx_grace_period_bars'] = new_grace_period

    # ADX Thresholds in expander
    with st.expander("ADX Thresholds", expanded=False):
        thresholds = config.get('adx_thresholds', {})
        if isinstance(thresholds, str):
            thresholds = json.loads(thresholds)

        st.markdown("**Trend Strength Thresholds**")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Strong Trend", f"> {thresholds.get('STRONG_TREND', 25.0)}")
            st.metric("Moderate Trend", f"> {thresholds.get('MODERATE_TREND', 22.0)}")

        with col2:
            st.metric("Weak Trend", f"> {thresholds.get('WEAK_TREND', 15.0)}")
            st.metric("Very Weak", f"< {thresholds.get('VERY_WEAK', 10.0)}")

    # Pair multipliers in expander
    with st.expander("ADX Pair Multipliers", expanded=False):
        multipliers = config.get('adx_pair_multipliers', {})
        if isinstance(multipliers, str):
            multipliers = json.loads(multipliers)

        if multipliers:
            df = pd.DataFrame([
                {'Pair': k, 'Multiplier': v}
                for k, v in multipliers.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No pair-specific multipliers configured")

    render_save_section(config, 'adx', updated_by)


def render_smc_conflict_settings(config: Dict[str, Any]):
    """Render SMC Conflict Filter settings"""
    st.subheader("SMC Conflict Filter")
    st.markdown("Smart Money Concepts analysis and signal filtering based on order flow and structure")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    # Main toggles
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Smart Money Analysis**")

        new_smc_readonly = st.checkbox(
            "Enable Smart Money Read-Only Analysis",
            value=config.get('smart_money_readonly_enabled', True),
            help="Enable smart money analysis for signals (read-only, no trading decisions)",
            key="smart_money_readonly_enabled"
        )
        if not values_equal(new_smc_readonly, config.get('smart_money_readonly_enabled')):
            st.session_state.scanner_pending_changes['smart_money_readonly_enabled'] = new_smc_readonly

        new_analysis_timeout = st.number_input(
            "Analysis Timeout (seconds)",
            value=_get_float(config, 'smart_money_analysis_timeout', 5.0),
            min_value=1.0, max_value=30.0, step=0.5,
            format="%.1f",
            help="Timeout for smart money analysis",
            key="smart_money_analysis_timeout"
        )
        if not values_equal(new_analysis_timeout, config.get('smart_money_analysis_timeout')):
            st.session_state.scanner_pending_changes['smart_money_analysis_timeout'] = new_analysis_timeout

    with col2:
        st.markdown("**Conflict Filter**")

        new_conflict_enabled = st.checkbox(
            "Enable SMC Conflict Filter",
            value=config.get('smc_conflict_filter_enabled', True),
            help="Reject signals when SMC data conflicts with signal direction",
            key="smc_conflict_filter_enabled"
        )
        if not values_equal(new_conflict_enabled, config.get('smc_conflict_filter_enabled')):
            st.session_state.scanner_pending_changes['smc_conflict_filter_enabled'] = new_conflict_enabled

        new_order_flow_reject = st.checkbox(
            "Reject Order Flow Conflicts",
            value=config.get('smc_reject_order_flow_conflict', True),
            help="Reject signals when order flow opposes signal direction",
            key="smc_reject_order_flow_conflict"
        )
        if not values_equal(new_order_flow_reject, config.get('smc_reject_order_flow_conflict')):
            st.session_state.scanner_pending_changes['smc_reject_order_flow_conflict'] = new_order_flow_reject

        new_ranging_reject = st.checkbox(
            "Reject Ranging Structure",
            value=config.get('smc_reject_ranging_structure', True),
            help="Reject signals when market structure is RANGING",
            key="smc_reject_ranging_structure"
        )
        if not values_equal(new_ranging_reject, config.get('smc_reject_ranging_structure')):
            st.session_state.scanner_pending_changes['smc_reject_ranging_structure'] = new_ranging_reject

    st.divider()

    # Thresholds
    st.markdown("**Consensus Thresholds**")
    col1, col2 = st.columns(2)

    with col1:
        new_directional_consensus = st.slider(
            "Min Directional Consensus",
            min_value=0.0, max_value=1.0, step=0.05,
            value=_get_float(config, 'smc_min_directional_consensus', 0.3),
            help="Minimum directional consensus score (0.0 = none, 1.0 = full agreement)",
            key="smc_min_directional_consensus"
        )
        if not values_equal(new_directional_consensus, config.get('smc_min_directional_consensus')):
            st.session_state.scanner_pending_changes['smc_min_directional_consensus'] = new_directional_consensus

    with col2:
        new_structure_score = st.slider(
            "Min Structure Score",
            min_value=0.0, max_value=1.0, step=0.05,
            value=_get_float(config, 'smc_min_structure_score', 0.5),
            help="Minimum market structure score required for signals",
            key="smc_min_structure_score"
        )
        if not values_equal(new_structure_score, config.get('smc_min_structure_score')):
            st.session_state.scanner_pending_changes['smc_min_structure_score'] = new_structure_score

    # Info box
    with st.expander("About SMC Conflict Filter", expanded=False):
        st.markdown("""
        The SMC Conflict Filter rejects signals when Smart Money Concepts data contradicts the signal direction:

        - **Order Flow Conflict**: Reject when institutional order flow opposes the signal
        - **Ranging Structure**: Reject when market is in a ranging/consolidation phase
        - **Directional Consensus**: Minimum agreement between SMC indicators (FVGs, OBs, liquidity)
        - **Structure Score**: Overall market structure quality score

        Lower thresholds = more permissive (more signals allowed)
        Higher thresholds = more strict (fewer, higher-quality signals)
        """)

    render_save_section(config, 'smc_conflict', updated_by)


def render_claude_validation_settings(config: Dict[str, Any]):
    """Render Claude AI Trade Validation settings"""
    st.subheader("Claude AI Trade Validation")
    st.markdown("AI-powered trade validation using Claude for signal quality assessment")

    updated_by = st.session_state.get('scanner_config_user', 'streamlit_user')

    if 'scanner_pending_changes' not in st.session_state:
        st.session_state.scanner_pending_changes = {}

    # Master switches
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Master Controls**")

        new_require_approval = st.checkbox(
            "Require Claude Approval",
            value=config.get('require_claude_approval', True),
            help="Trades must be approved by Claude before execution",
            key="require_claude_approval"
        )
        if not values_equal(new_require_approval, config.get('require_claude_approval')):
            st.session_state.scanner_pending_changes['require_claude_approval'] = new_require_approval

        new_fail_secure = st.checkbox(
            "Fail-Secure Mode",
            value=config.get('claude_fail_secure', True),
            help="Block trades on ANY Claude error (recommended for safety)",
            key="claude_fail_secure"
        )
        if not values_equal(new_fail_secure, config.get('claude_fail_secure')):
            st.session_state.scanner_pending_changes['claude_fail_secure'] = new_fail_secure

        new_save_rejections = st.checkbox(
            "Save Claude Rejections",
            value=config.get('save_claude_rejections', True),
            help="Log rejected signals to alert_history table for analysis",
            key="save_claude_rejections"
        )
        if not values_equal(new_save_rejections, config.get('save_claude_rejections')):
            st.session_state.scanner_pending_changes['save_claude_rejections'] = new_save_rejections

        new_validate_backtest = st.checkbox(
            "Validate in Backtest",
            value=config.get('claude_validate_in_backtest', False),
            help="Use Claude validation in backtest mode (uses API calls)",
            key="claude_validate_in_backtest"
        )
        if not values_equal(new_validate_backtest, config.get('claude_validate_in_backtest')):
            st.session_state.scanner_pending_changes['claude_validate_in_backtest'] = new_validate_backtest

    with col2:
        st.markdown("**Model & Quality**")

        model_options = ['haiku', 'sonnet', 'sonnet-old', 'opus']
        current_model = config.get('claude_model', 'sonnet')
        new_model = st.selectbox(
            "Claude Model",
            options=model_options,
            index=model_options.index(current_model) if current_model in model_options else 1,
            help="Model to use for trade validation",
            key="claude_model"
        )
        if new_model != current_model:
            st.session_state.scanner_pending_changes['claude_model'] = new_model

        new_min_score = st.slider(
            "Minimum Quality Score",
            min_value=1, max_value=10, step=1,
            value=_get_int(config, 'min_claude_quality_score', 6),
            help="Minimum Claude score (1-10) to approve trade",
            key="min_claude_quality_score"
        )
        if not values_equal(new_min_score, config.get('min_claude_quality_score')):
            st.session_state.scanner_pending_changes['min_claude_quality_score'] = new_min_score

    st.divider()

    # Vision settings
    st.markdown("**Vision Analysis (Chart-Based)**")
    col1, col2 = st.columns(2)

    with col1:
        new_include_chart = st.checkbox(
            "Include Chart in Analysis",
            value=config.get('claude_include_chart', True),
            help="Send chart image for visual analysis",
            key="claude_include_chart"
        )
        if not values_equal(new_include_chart, config.get('claude_include_chart')):
            st.session_state.scanner_pending_changes['claude_include_chart'] = new_include_chart

        new_vision_enabled = st.checkbox(
            "Enable Vision API",
            value=config.get('claude_vision_enabled', True),
            help="Enable Claude Vision API for chart analysis",
            key="claude_vision_enabled"
        )
        if not values_equal(new_vision_enabled, config.get('claude_vision_enabled')):
            st.session_state.scanner_pending_changes['claude_vision_enabled'] = new_vision_enabled

        new_save_artifacts = st.checkbox(
            "Save Vision Artifacts",
            value=config.get('claude_save_vision_artifacts', True),
            help="Save chart, prompt, and signal data to disk for analysis",
            key="claude_save_vision_artifacts"
        )
        if not values_equal(new_save_artifacts, config.get('claude_save_vision_artifacts')):
            st.session_state.scanner_pending_changes['claude_save_vision_artifacts'] = new_save_artifacts

        new_minio_enabled = st.checkbox(
            "Enable MinIO Storage",
            value=config.get('minio_enabled', True),
            help="Store charts and analysis artifacts in MinIO object storage",
            key="minio_enabled"
        )
        if not values_equal(new_minio_enabled, config.get('minio_enabled')):
            st.session_state.scanner_pending_changes['minio_enabled'] = new_minio_enabled

    with col2:
        # Display current timeframes
        current_timeframes = config.get('claude_chart_timeframes', ['4h', '1h', '15m'])
        if isinstance(current_timeframes, str):
            current_timeframes = json.loads(current_timeframes)
        st.markdown(f"**Chart Timeframes:** {', '.join(current_timeframes)}")

        # Display current vision strategies
        current_strategies = config.get('claude_vision_strategies', ['EMA_DOUBLE', 'SMC', 'SMC_STRUCTURE'])
        if isinstance(current_strategies, str):
            current_strategies = json.loads(current_strategies)
        st.markdown(f"**Vision Strategies:** {', '.join(current_strategies)}")

    # Info box
    with st.expander("About Claude Trade Validation", expanded=False):
        st.markdown("""
        Claude Trade Validation uses AI to assess trade quality before execution:

        - **Require Claude Approval**: When enabled, all trades must be approved by Claude
        - **Fail-Secure Mode**: Block trades if Claude errors occur (recommended)
        - **Quality Score**: Minimum score (1-10) required to approve a trade
        - **Vision Analysis**: Send chart images for visual pattern recognition

        **Model Selection:**
        - `haiku`: Fastest, cheapest, good for simple checks
        - `sonnet`: Balanced speed/quality (recommended)
        - `opus`: Highest quality, slower, more expensive

        **Note:** API calls consume tokens. Disable 'Validate in Backtest' to save costs.
        """)

    render_save_section(config, 'claude_validation', updated_by)


def render_audit_trail():
    """Render audit trail tab"""
    st.subheader("Configuration Audit Trail")
    st.markdown("History of all configuration changes")

    col1, col2 = st.columns([2, 1])
    with col1:
        limit = st.number_input("Show last N changes", value=50, min_value=10, max_value=500)
    with col2:
        category_filter = st.selectbox(
            "Filter by Category",
            options=['All', 'core', 'indicators', 'data_quality', 'trading_control', 'dedup', 'risk', 'trading_hours', 'safety', 'adx', 'smc_conflict', 'claude_validation'],
            index=0
        )

    category = None if category_filter == 'All' else category_filter
    audit_records = get_audit_history(limit=limit, category=category)

    if audit_records:
        # Format for display
        display_data = []
        for record in audit_records:
            changed_at = record.get('changed_at')
            if isinstance(changed_at, datetime):
                changed_at = changed_at.strftime('%Y-%m-%d %H:%M:%S')

            display_data.append({
                'Time': changed_at,
                'Type': record.get('change_type', 'N/A'),
                'User': record.get('changed_by', 'N/A'),
                'Category': record.get('category', 'N/A'),
                'Reason': record.get('change_reason', 'N/A')[:50] + '...' if record.get('change_reason') and len(record.get('change_reason', '')) > 50 else record.get('change_reason', 'N/A'),
            })

        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Expandable details
        with st.expander("View Change Details"):
            for i, record in enumerate(audit_records[:10]):
                with st.container():
                    st.markdown(f"**{record.get('change_type', 'N/A')}** by {record.get('changed_by', 'N/A')}")
                    st.caption(record.get('change_reason', 'No reason provided'))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("*Previous:*")
                        prev = record.get('previous_values')
                        if prev:
                            if isinstance(prev, str):
                                prev = json.loads(prev)
                            st.json(prev)
                    with col2:
                        st.markdown("*New:*")
                        new = record.get('new_values')
                        if new:
                            if isinstance(new, str):
                                new = json.loads(new)
                            st.json(new)

                    st.divider()
    else:
        st.info("No audit records found")


def render_save_section(config: Dict[str, Any], category: str, updated_by: str):
    """Render save button and pending changes display"""
    st.divider()

    pending = st.session_state.get('scanner_pending_changes', {})

    if pending:
        st.warning(f"You have {len(pending)} pending changes")

        with st.expander("View Pending Changes"):
            st.json(pending)

        change_reason = st.text_input(
            "Change Reason (required)",
            placeholder="Describe why you're making this change...",
            key=f"change_reason_{category}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes", type="primary", disabled=not change_reason, key=f"save_{category}"):
                success = save_global_config(
                    config['id'],
                    pending,
                    updated_by,
                    change_reason,
                    category=category
                )
                if success:
                    st.success("Configuration saved successfully!")
                    st.session_state.scanner_pending_changes = {}
                    st.rerun()
                else:
                    st.error("Failed to save configuration")

        with col2:
            if st.button("Discard Changes", key=f"discard_{category}"):
                st.session_state.scanner_pending_changes = {}
                st.rerun()
    else:
        st.info("No pending changes")
