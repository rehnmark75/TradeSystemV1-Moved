"""
Configuration System Documentation

Comprehensive reference for all trading system configuration:
- SMC Simple Strategy (database-driven)
- Trailing Stops (file-based)
- Infrastructure (environment variables)

This is a static documentation page with no live database queries.
"""

import streamlit as st
import pandas as pd

# Try to import streamlit-mermaid, fall back to code display if not available
try:
    import streamlit_mermaid as stmd
    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False


def render_mermaid(code: str, height: int = 500):
    """Render a Mermaid diagram using streamlit-mermaid or fallback to code display."""
    if MERMAID_AVAILABLE:
        stmd.st_mermaid(code, height=height)
    else:
        st.info("Install `streamlit-mermaid` for diagram rendering. Showing code:")
        st.code(code, language="mermaid")


# Page configuration
st.set_page_config(
    page_title="Configuration Guide",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for documentation styling
st.markdown("""
<style>
    .doc-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .section-header {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #0066cc;
        margin: 1.5rem 0 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .info-box {
        background: #e7f3ff;
        border: 1px solid #b3d9ff;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .code-file {
        background: #f4f4f4;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-family: monospace;
        font-size: 0.9em;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    th, td {
        border: 1px solid #dee2e6;
        padding: 0.75rem;
        text-align: left;
    }
    th {
        background: #f8f9fa;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar_navigation():
    """Render sidebar with section navigation."""
    st.sidebar.title("⚙️ Configuration Guide")
    st.sidebar.markdown("---")

    sections = [
        ("🎯 Overview", "overview"),
        ("📊 SMC Strategy Config", "smc_strategy"),
        ("🔧 Per-Pair Overrides", "pair_overrides"),
        ("📈 Trailing Stops", "trailing"),
        ("🏗️ Infrastructure", "infrastructure"),
        ("🐳 Container Ownership", "containers"),
        ("🗄️ Database Schema", "database"),
        ("📝 Common Operations", "operations"),
    ]

    selected = st.sidebar.radio(
        "Navigate to Section:",
        options=[s[0] for s in sections],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Links")
    st.sidebar.markdown("""
    - [Forex Documentation](/forex_documentation)
    - [System Status](/system_status)
    - [Infrastructure Status](/infrastructure_status)
    """)

    return selected


def render_header():
    """Render page header."""
    st.markdown("""
    <div class="doc-header">
        <h1>⚙️ Configuration System Guide</h1>
        <p>Complete reference for strategy, trailing stops, and infrastructure configuration</p>
    </div>
    """, unsafe_allow_html=True)


def render_overview_section():
    """Section 1: Configuration System Overview."""
    st.markdown('<div class="section-header"><h2>🎯 Configuration System Overview</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    The TradeSystemV1 uses a three-tier configuration system:

    1. **SMC Strategy Config** - Database-driven with ~80 parameters
    2. **Trailing Stops** - File-based, owned by fastapi-dev container
    3. **Infrastructure** - Environment variables and static mappings
    """)

    # Quick Reference Table
    st.markdown("### Quick Reference")

    df = pd.DataFrame({
        'Configuration Type': ['SMC Strategy', 'Trailing Stops (LIVE)', 'Trailing Stops (Backtest)', 'Infrastructure'],
        'Source of Truth': ['strategy_config DB', 'dev-app/config.py', 'config_trailing_stops.py', 'forex_scanner/config.py'],
        'Container Owner': ['task-worker', 'fastapi-dev', 'task-worker', 'task-worker'],
        'Hot Reload': ['Yes (120s TTL)', 'No (restart)', 'N/A', 'No (restart)']
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Architecture Diagram
    st.markdown("### Configuration Flow")
    render_mermaid("""
flowchart TB
    subgraph Sources["Configuration Sources"]
        DB[("strategy_config DB<br/>~80 parameters")]
        DEVCONF["dev-app/config.py<br/>Trailing Stops"]
        WRKCONF["forex_scanner/config.py<br/>Infrastructure"]
    end

    subgraph Services["Config Services"]
        SMCSVC["SMCSimpleConfigService<br/>120s TTL Cache"]
        GETTRAIL["get_trailing_config_for_epic()"]
    end

    subgraph Consumers["Consumers"]
        STRAT["SMCSimpleStrategy"]
        TRAIL["TrailingStopManager"]
        SCAN["Scanner/Orchestrator"]
    end

    DB --> SMCSVC
    SMCSVC --> STRAT
    DEVCONF --> GETTRAIL
    GETTRAIL --> TRAIL
    WRKCONF --> SCAN
    """, height=450)

    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Critical:</strong> Always verify which container owns a configuration before editing!
    Editing the wrong file will have no effect or cause unexpected behavior.
    </div>
    """, unsafe_allow_html=True)


def render_smc_strategy_section():
    """Section 2: SMC Simple Strategy Configuration."""
    st.markdown('<div class="section-header"><h2>📊 SMC Simple Strategy Configuration</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    All SMC Simple strategy parameters are stored in the `strategy_config` PostgreSQL database.
    The config service loads parameters with a 120-second cache TTL.
    """)

    st.markdown("### Config Service Usage")
    st.code("""
# Get singleton service instance
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config_service
service = get_smc_simple_config_service()

# Get current config (auto-refreshes every 120s)
config = service.get_config()

# Force refresh from database
config = service.get_config(force_refresh=True)

# Convenience function
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
config = get_smc_simple_config()
    """, language="python")

    # Parameter Categories
    st.markdown("### Parameter Categories (~80 parameters)")

    with st.expander("Tier 1: HTF Directional Bias (4H) - 5 parameters"):
        df = pd.DataFrame({
            'Parameter': ['htf_timeframe', 'ema_period', 'ema_buffer_pips', 'require_close_beyond_ema', 'min_distance_from_ema_pips'],
            'Type': ['str', 'int', 'float', 'bool', 'float'],
            'Default': ['"4h"', '50', '2.5', 'True', '3.0'],
            'Description': ['Higher timeframe for bias', 'EMA period for trend', 'Buffer around EMA', 'Require candle close beyond EMA', 'Minimum distance from EMA']
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Tier 2: Entry Trigger (15M) - 8 parameters"):
        df = pd.DataFrame({
            'Parameter': ['trigger_timeframe', 'swing_lookback_bars', 'swing_strength_bars', 'wick_tolerance_pips', 'volume_confirmation_enabled', 'volume_sma_period', 'volume_spike_multiplier', 'require_body_close_break'],
            'Type': ['str', 'int', 'int', 'float', 'bool', 'int', 'float', 'bool'],
            'Default': ['"15m"', '20', '2', '3.0', 'True', '20', '1.2', 'False'],
            'Description': ['Entry trigger timeframe', 'Bars for swing detection', 'Bars to confirm swing', 'Wick tolerance', 'Enable volume confirmation', 'SMA period for volume', 'Volume spike multiplier', 'Require body close break']
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Tier 3: Execution (5M) - 8 parameters"):
        df = pd.DataFrame({
            'Parameter': ['entry_timeframe', 'pullback_enabled', 'fib_pullback_min', 'fib_pullback_max', 'fib_optimal_zone_min', 'fib_optimal_zone_max', 'max_pullback_wait_bars', 'pullback_confirmation_bars'],
            'Type': ['str', 'bool', 'float', 'float', 'float', 'float', 'int', 'int'],
            'Default': ['"5m"', 'True', '0.236', '0.700', '0.382', '0.618', '12', '2'],
            'Description': ['Entry execution timeframe', 'Enable pullback entries', 'Min Fib retracement', 'Max Fib retracement', 'Optimal zone min', 'Optimal zone max', 'Max bars to wait', 'Confirmation bars']
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Risk Management - 13 parameters"):
        df = pd.DataFrame({
            'Parameter': ['min_rr_ratio', 'optimal_rr_ratio', 'max_rr_ratio', 'sl_buffer_pips', 'min_tp_pips', 'fixed_sl_tp_override_enabled', 'fixed_stop_loss_pips', 'fixed_take_profit_pips', 'risk_per_trade_pct', 'use_atr_stop', 'sl_atr_multiplier', 'use_swing_target', 'tp_structure_lookback'],
            'Type': ['float', 'float', 'float', 'int', 'int', 'bool', 'float', 'float', 'float', 'bool', 'float', 'bool', 'int'],
            'Default': ['1.5', '2.5', '5.0', '6', '8', 'True', '9.0', '15.0', '1.0', 'True', '1.0', 'True', '50'],
            'Description': ['Minimum R:R ratio', 'Optimal R:R ratio', 'Maximum R:R ratio', 'Stop loss buffer', 'Minimum take profit', 'Enable fixed SL/TP', 'Fixed SL (global)', 'Fixed TP (global)', 'Risk per trade %', 'Use ATR stops', 'SL ATR multiplier', 'Use swing target', 'TP structure lookback']
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Confidence Scoring - 7 parameters"):
        df = pd.DataFrame({
            'Parameter': ['min_confidence_threshold', 'max_confidence_threshold', 'high_confidence_threshold', 'volume_adjusted_confidence_enabled', 'atr_adjusted_confidence_enabled', 'ema_distance_adjusted_confidence_enabled', 'confidence_weights'],
            'Type': ['float', 'float', 'float', 'bool', 'bool', 'bool', 'dict'],
            'Default': ['0.48', '0.75', '0.75', 'True', 'True', 'True', '{ema: 0.20, ...}'],
            'Description': ['Min confidence to trade', 'Max confidence (paradox)', 'High confidence level', 'Adjust by volume', 'Adjust by ATR', 'Adjust by EMA distance', 'Component weights']
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Session Filter - 7 parameters"):
        df = pd.DataFrame({
            'Parameter': ['session_filter_enabled', 'london_session_start', 'london_session_end', 'ny_session_start', 'ny_session_end', 'allowed_sessions', 'block_asian_session'],
            'Type': ['bool', 'time', 'time', 'time', 'time', 'list', 'bool'],
            'Default': ['True', '07:00', '16:00', '12:00', '21:00', "['london', 'new_york', 'overlap']", 'True'],
            'Description': ['Enable session filter', 'London start (UTC)', 'London end (UTC)', 'NY start (UTC)', 'NY end (UTC)', 'Allowed sessions', 'Block Asian session']
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("MACD Alignment Filter - 3 parameters"):
        df = pd.DataFrame({
            'Parameter': ['macd_alignment_filter_enabled', 'macd_alignment_mode', 'macd_min_strength'],
            'Type': ['bool', 'str', 'float'],
            'Default': ['True', "'momentum'", '0.0'],
            'Description': ['Enable MACD filter', 'Mode: momentum or histogram', 'Minimum MACD strength']
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Adaptive Cooldown - 18 parameters"):
        st.markdown("""
        Adaptive cooldown adjusts the signal cooldown period based on:
        - Win/loss streaks
        - Overall win rate
        - Market volatility
        - Session changes

        Key parameters: `base_cooldown_hours` (2.0), `cooldown_after_win_multiplier` (0.5), `cooldown_after_loss_multiplier` (1.5)
        """)


def render_pair_overrides_section():
    """Section 3: Per-Pair Override System."""
    st.markdown('<div class="section-header"><h2>🔧 Per-Pair Override System</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    The system supports per-pair parameter overrides via the `smc_simple_pair_overrides` database table.
    This allows fine-tuning for specific currency pairs without changing global settings.
    """)

    st.markdown("### Override Resolution Order")
    render_mermaid("""
flowchart LR
    REQ["get_pair_min_confidence<br/>('EURUSD')"] --> CHECK1{"Pair override<br/>exists?"}
    CHECK1 -->|Yes| PAIR["Return pair value"]
    CHECK1 -->|No| CHECK2{"Global value<br/>set?"}
    CHECK2 -->|Yes| GLOBAL["Return global value"]
    CHECK2 -->|No| DEFAULT["Return dataclass default"]
    """, height=200)

    st.markdown("### Per-Pair Override Fields")

    df = pd.DataFrame({
        'Field': ['fixed_stop_loss_pips', 'fixed_take_profit_pips', 'min_confidence', 'max_confidence', 'sl_buffer_pips', 'min_volume_ratio', 'macd_filter_enabled', 'allow_asian_session', 'blocking_conditions'],
        'Type': ['NUMERIC(5,1)', 'NUMERIC(5,1)', 'DECIMAL(4,3)', 'DECIMAL(4,3)', 'INTEGER', 'DECIMAL(4,2)', 'BOOLEAN', 'BOOLEAN', 'JSONB'],
        'Description': ['Per-pair fixed stop loss', 'Per-pair fixed take profit', 'Per-pair min confidence', 'Per-pair max confidence (paradox)', 'Per-pair SL buffer', 'Per-pair min volume ratio', 'Toggle MACD filter for pair', 'Allow Asian session for pair', 'Complex blocking rules']
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Per-Pair API Methods")
    st.code("""
config = get_smc_simple_config()

# Get per-pair fixed SL (returns None if disabled)
sl = config.get_pair_fixed_stop_loss('CS.D.EURUSD.CEEM.IP')

# Get per-pair fixed TP
tp = config.get_pair_fixed_take_profit('CS.D.EURUSD.CEEM.IP')

# Get per-pair minimum confidence
min_conf = config.get_pair_min_confidence('CS.D.EURUSD.CEEM.IP')

# Get per-pair maximum confidence (paradox filter)
max_conf = config.get_pair_max_confidence('CS.D.EURUSD.CEEM.IP')

# Check if MACD filter enabled for pair
macd_on = config.is_macd_filter_enabled('CS.D.EURUSD.CEEM.IP')

# Check if Asian session allowed for pair
asian_ok = config.is_asian_session_allowed('CS.D.USDCHF.MINI.IP')

# Get dynamic confidence based on conditions
threshold = config.get_dynamic_confidence(
    epic='CS.D.EURUSD.CEEM.IP',
    volume_ratio=0.85,
    atr_value=0.0005,
    ema_distance_pips=25.5
)

# Check if signal should be blocked
blocked, reason = config.should_block_signal('CS.D.EURUSD.CEEM.IP', signal_data)
    """, language="python")

    st.markdown("### Blocking Conditions JSONB Structure")
    st.code("""
{
    "enabled": true,
    "blocking_logic": "any",  // or "all"
    "conditions": {
        "max_ema_distance_pips": 50.0,
        "require_volume_confirmation": true,
        "block_momentum_without_volume": true,
        "min_confidence_override": 0.55
    }
}
    """, language="json")


def render_trailing_section():
    """Section 4: Trailing Stop Configuration."""
    st.markdown('<div class="section-header"><h2>📈 Trailing Stop Configuration</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="danger-box">
    <strong>CRITICAL:</strong> Live trailing stops are ONLY configured in <code>dev-app/config.py</code> (fastapi-dev container).
    <br/><br/>
    The file <code>worker/app/forex_scanner/config_trailing_stops.py</code> is for <strong>BACKTESTING ONLY</strong>!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 4-Stage Progressive System")

    render_mermaid("""
stateDiagram-v2
    [*] --> Entry: Trade Opened
    Entry --> EarlyBE: +15-20 pips
    EarlyBE --> Stage1: +25-30 pips
    Stage1 --> Stage2: +38-45 pips
    Stage2 --> Stage3: +50-60 pips
    Stage3 --> [*]: Trail hits SL
    EarlyBE --> PartialClose: +20-25 pips
    PartialClose --> Stage1: Continue
    """, height=350)

    st.markdown("### Stage Details (v3.0.0)")

    df = pd.DataFrame({
        'Stage': ['Early BE', 'Partial Close', 'Stage 1', 'Stage 2', 'Stage 3'],
        'Trigger (Major)': ['+15 pips', '+20 pips', '+25 pips', '+38 pips', '+50 pips'],
        'Trigger (JPY)': ['+20 pips', '+25 pips', '+30 pips', '+45 pips', '+60 pips'],
        'Action': ['SL to entry+2', 'Close 40%', 'Lock +12 pips', 'Lock +20 pips', '2.0x ATR trailing'],
        'Parameters': ['early_breakeven_trigger_points, early_breakeven_buffer_points', 'partial_close_trigger_points, partial_close_size', 'stage1_trigger_points, stage1_lock_points', 'stage2_trigger_points, stage2_lock_points', 'stage3_trigger_points, stage3_atr_multiplier']
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Per-Pair Configuration Structure")
    st.code("""
# Example: EURUSD config (major pair - 15 pip early BE)
'CS.D.EURUSD.CEEM.IP': {
    'early_breakeven_trigger_points': 15,   # Trigger early BE at +15 pips
    'early_breakeven_buffer_points': 2,     # Move SL to entry+2
    'stage1_trigger_points': 25,            # Stage 1 at +25 pips
    'stage1_lock_points': 12,               # Lock +12 pips
    'stage2_trigger_points': 38,            # Stage 2 at +38 pips (Fibonacci)
    'stage2_lock_points': 20,               # Lock +20 pips
    'stage3_trigger_points': 50,            # Stage 3 at +50 pips
    'stage3_atr_multiplier': 2.0,           # Trail at 2.0x ATR
    'stage3_min_distance': 8,               # Min 8 pip distance
    'min_trail_distance': 10,               # Min trailing distance
    'break_even_trigger_points': 18,        # Legacy BE trigger
    'enable_partial_close': True,           # Enable partial close
    'partial_close_trigger_points': 20,     # Partial at +20 pips
    'partial_close_size': 0.4,              # Close 40%
}
    """, language="python")

    st.markdown("### Pair Categories (v3.0.0)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Major USD Pairs (15 pip early BE)**")
        st.markdown("""
        - EURUSD
        - AUDUSD
        - NZDUSD
        - USDCAD
        - USDCHF
        - GBPUSD
        """)

    with col2:
        st.markdown("**JPY Crosses (20 pip early BE)**")
        st.markdown("""
        - USDJPY
        - EURJPY
        - GBPJPY
        - AUDJPY
        - CADJPY
        - CHFJPY
        - NZDJPY
        """)


def render_infrastructure_section():
    """Section 5: Infrastructure Configuration."""
    st.markdown('<div class="section-header"><h2>🏗️ Infrastructure Configuration</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    Infrastructure configuration is located in `worker/app/forex_scanner/config.py`.
    After January 2026 cleanup, this file contains ONLY infrastructure settings (no strategy parameters).
    """)

    st.markdown("### Environment Variables")

    df = pd.DataFrame({
        'Variable': ['DATABASE_URL', 'STRATEGY_CONFIG_DATABASE_URL', 'CLAUDE_API_KEY', 'ORDER_API_URL', 'API_SUBSCRIPTION_KEY', 'USER_TIMEZONE', 'MINIO_ENDPOINT', 'MINIO_BUCKET_NAME'],
        'Purpose': ['Main forex database', 'Strategy config database', 'Claude AI integration', 'Order placement endpoint', 'API authentication key', 'User timezone', 'MinIO object storage', 'Chart storage bucket'],
        'Default': ['postgresql://...localhost/forex', 'postgresql://...postgres/strategy_config', 'None', 'http://fastapi-dev:8000/orders/place-order', '(default key)', 'Europe/Stockholm', 'minio:9000', 'claude-charts']
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Static Pair Mappings")

    with st.expander("PAIR_INFO - Pip multipliers"):
        st.code("""
PAIR_INFO = {
    'CS.D.EURUSD.CEEM.IP': {'pair': 'EURUSD', 'pip_multiplier': 10000},
    'CS.D.GBPUSD.MINI.IP': {'pair': 'GBPUSD', 'pip_multiplier': 10000},
    'CS.D.USDJPY.MINI.IP': {'pair': 'USDJPY', 'pip_multiplier': 100},
    # JPY pairs use 100, all others use 10000
    ...
}
        """, language="python")

    with st.expander("EPIC_LIST - Enabled trading pairs"):
        st.code("""
EPIC_LIST = [
    'CS.D.EURUSD.CEEM.IP',
    'CS.D.GBPUSD.MINI.IP',
    'CS.D.USDJPY.MINI.IP',
    'CS.D.AUDUSD.MINI.IP',
    'CS.D.USDCHF.MINI.IP',
    'CS.D.USDCAD.MINI.IP',
    'CS.D.NZDUSD.MINI.IP',
    'CS.D.EURJPY.MINI.IP',
    'CS.D.AUDJPY.MINI.IP'
]
        """, language="python")

    with st.expander("EPIC_MAP - Scanner to API epic mapping"):
        st.code("""
# Scanner epic -> Trading API epic
EPIC_MAP = {
    "CS.D.EURUSD.CEEM.IP": "EURUSD.1.MINI",
    "CS.D.GBPUSD.MINI.IP": "GBPUSD.1.MINI",
    "CS.D.USDJPY.MINI.IP": "USDJPY.100.MINI",
    ...
}

REVERSE_EPIC_MAP = {v: k for k, v in EPIC_MAP.items()}
        """, language="python")


def render_containers_section():
    """Section 6: Container Ownership Matrix."""
    st.markdown('<div class="section-header"><h2>🐳 Container Ownership Matrix</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    Each configuration type is owned by a specific container.
    Editing the wrong file will have no effect!
    """)

    # Ownership Diagram
    render_mermaid("""
flowchart TB
    subgraph FD["fastapi-dev Container"]
        FDC["dev-app/config.py"]
        TRAIL["Trailing Stops<br/>PAIR_TRAILING_CONFIGS"]
        TRADE["Trade Execution"]
    end

    subgraph TW["task-worker Container"]
        TWC["forex_scanner/config.py"]
        SMCSVC["SMC Config Service"]
        SCAN["Scanner/Strategy"]
    end

    subgraph ST["streamlit Container"]
        MOUNT["Read-only mount<br/>trailing_config.py"]
        DASH["Dashboard Display"]
    end

    FDC --> TRAIL
    TRAIL --> TRADE
    TWC --> SCAN
    SMCSVC --> SCAN
    FDC -.->|"docker mount"| MOUNT
    MOUNT --> DASH
    """, height=400)

    st.markdown("### Ownership Matrix")

    df = pd.DataFrame({
        'Setting Type': ['SMC Strategy Parameters', 'Trailing Stops (LIVE)', 'Trailing Stops (Backtest)', 'Epic Lists', 'API URLs', 'Environment Secrets'],
        'Container': ['task-worker', 'fastapi-dev', 'task-worker', 'task-worker', 'task-worker', 'All'],
        'File Path': ['Database (strategy_config)', 'dev-app/config.py', 'forex_scanner/config_trailing_stops.py', 'forex_scanner/config.py', 'forex_scanner/config.py', 'Docker environment'],
        'Hot Reload': ['Yes (120s TTL)', 'No', 'N/A', 'No', 'No', 'No'],
        'Restart Required': ['No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Container Access:</strong>
    <ul>
    <li><code>docker exec -it task-worker bash</code> - Strategy scanning, backtesting</li>
    <li><code>docker exec -it fastapi-dev bash</code> - Trade execution, trailing stops</li>
    <li><code>docker exec -it streamlit bash</code> - Dashboard (read-only configs)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def render_database_section():
    """Section 7: Database Schema Reference."""
    st.markdown('<div class="section-header"><h2>🗄️ Database Schema Reference</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    The `strategy_config` database contains all SMC Simple strategy configuration.
    """)

    st.markdown("### Tables")

    df = pd.DataFrame({
        'Table': ['smc_simple_global_config', 'smc_simple_pair_overrides', 'smc_simple_config_audit', 'smc_simple_parameter_metadata'],
        'Purpose': ['Main config with ~80 parameters', 'Per-pair parameter overrides', 'Audit trail for changes', 'UI metadata for parameters'],
        'Key Columns': ['version, is_active, fixed_stop_loss_pips, ...', 'epic, config_id, fixed_stop_loss_pips, min_confidence, ...', 'changed_at, field_name, old_value, new_value', 'parameter_name, display_name, category, min_value, max_value']
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Config Resolution Flow")
    render_mermaid("""
flowchart LR
    START["Request config<br/>for EURUSD"] --> SVC["SMCSimpleConfigService"]
    SVC --> CACHE{"Cache valid?<br/>(< 120s)"}
    CACHE -->|Yes| RET1["Return cached"]
    CACHE -->|No| LOAD["Load from DB"]
    LOAD --> BUILD["Build SMCSimpleConfig"]
    BUILD --> PAIR{"Per-pair<br/>override?"}
    PAIR -->|Yes| OVER["Apply override"]
    PAIR -->|No| GLOB["Use global"]
    OVER --> RET2["Return config"]
    GLOB --> RET2
    """, height=250)


def render_operations_section():
    """Section 8: Common Operations."""
    st.markdown('<div class="section-header"><h2>📝 Common Operations</h2></div>', unsafe_allow_html=True)

    st.markdown("### View Current SMC Config")
    st.code("""
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT version, strategy_status, min_confidence_threshold,
       fixed_stop_loss_pips, fixed_take_profit_pips
FROM smc_simple_global_config WHERE is_active = TRUE;"
    """, language="bash")

    st.markdown("### Update Global SL/TP")
    st.code("""
docker exec postgres psql -U postgres -d strategy_config -c "
UPDATE smc_simple_global_config
SET fixed_stop_loss_pips = 10, fixed_take_profit_pips = 18
WHERE is_active = TRUE;"
    """, language="bash")

    st.markdown("### Set Per-Pair Override")
    st.code("""
docker exec postgres psql -U postgres -d strategy_config -c "
UPDATE smc_simple_pair_overrides
SET fixed_stop_loss_pips = 12, fixed_take_profit_pips = 22
WHERE epic = 'CS.D.USDJPY.MINI.IP';"
    """, language="bash")

    st.markdown("### View Per-Pair Settings")
    st.code("""
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT epic, fixed_stop_loss_pips, fixed_take_profit_pips,
       min_confidence, max_confidence
FROM smc_simple_pair_overrides ORDER BY epic;"
    """, language="bash")

    st.markdown("### Invalidate Config Cache (Python)")
    st.code("""
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config_service
service = get_smc_simple_config_service()
service.invalidate_cache()
# Or just wait 120 seconds for auto-refresh
    """, language="python")

    st.markdown("### View Audit Trail")
    st.code("""
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT changed_at, field_name, old_value, new_value
FROM smc_simple_config_audit
ORDER BY changed_at DESC LIMIT 10;"
    """, language="bash")

    st.markdown("### Update Trailing Stops (LIVE)")
    st.code("""
# 1. Edit the source file
nano dev-app/config.py

# 2. Restart container to apply
docker restart fastapi-dev

# 3. Verify
docker exec fastapi-dev python3 -c "
from config import PAIR_TRAILING_CONFIGS
print(PAIR_TRAILING_CONFIGS['CS.D.EURUSD.CEEM.IP'])
"
    """, language="bash")

    st.markdown("""
    <div class="info-box">
    <strong>Troubleshooting:</strong>
    <ul>
    <li><strong>Config changes not taking effect:</strong> Wait 120s for cache TTL, or call <code>service.invalidate_cache()</code></li>
    <li><strong>Trailing stops not updating:</strong> Restart <code>fastapi-dev</code> container</li>
    <li><strong>Wrong container edited:</strong> Check the Container Ownership Matrix above</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main entry point."""
    render_header()

    selected = render_sidebar_navigation()

    # Map selection to section
    section_map = {
        "🎯 Overview": render_overview_section,
        "📊 SMC Strategy Config": render_smc_strategy_section,
        "🔧 Per-Pair Overrides": render_pair_overrides_section,
        "📈 Trailing Stops": render_trailing_section,
        "🏗️ Infrastructure": render_infrastructure_section,
        "🐳 Container Ownership": render_containers_section,
        "🗄️ Database Schema": render_database_section,
        "📝 Common Operations": render_operations_section,
    }

    # Render selected section
    if selected in section_map:
        section_map[selected]()


if __name__ == "__main__":
    main()
