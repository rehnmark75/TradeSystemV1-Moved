"""
Forex Scanner System Documentation

Comprehensive documentation of the TradeSystemV1 forex scanner architecture,
including system components, data flows, and configuration reference.

This page provides static reference documentation with Mermaid flowcharts.
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
    page_title="Forex System Documentation",
    page_icon="📚",
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
    st.sidebar.title("📚 Documentation")
    st.sidebar.markdown("---")

    sections = [
        ("🏗️ System Architecture", "architecture"),
        ("📊 Candle Data Flow", "candles"),
        ("🎯 Signal Detection", "signals"),
        ("💹 Trade Execution", "trades"),
        ("🐳 Container Details", "containers"),
        ("🗄️ Database Schema", "database"),
        ("📈 SMC Simple Strategy", "strategy"),
        ("⚙️ Configuration Guide", "config"),
        ("🔧 Parameter Optimizer", "optimizer"),
    ]

    selected = st.sidebar.radio(
        "Navigate to Section:",
        options=[s[0] for s in sections],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Links")
    st.sidebar.markdown("""
    - [CLAUDE.md](/) - Main reference
    - [Infrastructure Status](/infrastructure_status)
    - [Unified Analytics](/unified_analytics)
    """)

    return selected


def render_header():
    """Render page header."""
    st.markdown("""
    <div class="doc-header">
        <h1>📚 Forex Scanner System Documentation</h1>
        <p>Complete architecture reference for TradeSystemV1</p>
    </div>
    """, unsafe_allow_html=True)


def render_architecture_section():
    """Section 1: System Architecture Overview."""
    st.markdown('<div class="section-header"><h2>🏗️ System Architecture Overview</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    The TradeSystemV1 forex scanner is a microservices-based trading system running in Docker containers.
    All services communicate over the `lab-net` Docker network.
    """)

    # System Architecture Mermaid Diagram
    st.markdown("### High-Level Architecture")
    render_mermaid("""
graph TB
    subgraph External["External Services"]
        IG["IG Markets API<br/>(Lightstreamer)"]
        AKV["Azure Key Vault"]
    end

    subgraph Docker["Docker Network: lab-net"]
        subgraph Core["Core Trading"]
            FD["fastapi-dev<br/>:8001<br/>Live Trading"]
            FP["fastapi-prod<br/>:8002<br/>Production"]
            FS["fastapi-stream<br/>:8003<br/>WebSocket"]
            TW["task-worker<br/>Signal Scanner"]
        end

        subgraph Data["Data Layer"]
            PG[("PostgreSQL<br/>:5432")]
            VDB["vector-db<br/>:8090<br/>ChromaDB"]
        end

        subgraph UI["User Interface"]
            ST["streamlit<br/>:8501<br/>Dashboard"]
            TV["tradingview<br/>:8080<br/>Charts"]
        end

        subgraph Support["Support Services"]
            EC["economic-calendar<br/>:8091"]
            SM["system-monitor<br/>:8095"]
            NX["nginx<br/>:80/443"]
            BK["db-backup<br/>Scheduled"]
        end
    end

    IG --> FS
    IG --> TW
    AKV --> FD
    AKV --> TW

    TW --> PG
    TW --> VDB
    FD --> PG
    FP --> PG
    ST --> PG
    TV --> PG

    TW -->|"Order API"| FD
    NX --> FD
    NX --> FP
    NX --> ST

    SM --> PG
    EC --> PG
    BK --> PG
    """, height=600)

    st.markdown("### Key Architecture Principles")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Separation of Concerns:**
        - `task-worker`: Signal scanning & backtesting
        - `fastapi-dev`: Trade execution & monitoring
        - `streamlit`: Analytics & visualization
        - `postgres`: Central data store
        """)

    with col2:
        st.markdown("""
        **Configuration Ownership:**
        - Trailing stops: `dev-app/config.py` (fastapi-dev)
        - Strategy config: `strategy_config` database
        - Scanner settings: `worker/app/forex_scanner/config.py`
        """)

    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Critical:</strong> Never confuse container configurations!
    Trailing stops are ONLY configured in <code>dev-app/config.py</code> (fastapi-dev container).
    </div>
    """, unsafe_allow_html=True)


def render_candle_data_flow_section():
    """Section 2: Candle Data Flow."""
    st.markdown('<div class="section-header"><h2>📊 Candle Data Flow</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    Market data flows from IG Markets through Lightstreamer into the PostgreSQL database,
    where it's processed by the DataFetcher for strategy analysis.
    """)

    # Candle Flow Mermaid Diagram
    st.markdown("### Data Ingestion Pipeline")
    render_mermaid("""
flowchart LR
    subgraph Source["Data Source"]
        IG["IG Markets<br/>Lightstreamer API"]
    end

    subgraph Ingestion["Data Ingestion"]
        STREAM["ig_stream_to_postgres.py<br/>(streamlit/igstream/)"]
    end

    subgraph Storage["Database Storage"]
        DB[("ig_candles<br/>5-minute base<br/>PostgreSQL")]
    end

    subgraph Processing["Data Processing"]
        DF["DataFetcher<br/>(data_fetcher.py)"]
        RS["Resample Engine<br/>5m → 15m/1h/4h"]
        IND["Indicator Engine<br/>EMA/MACD/RSI/ATR"]
    end

    subgraph Analysis["Strategy Analysis"]
        STRAT["SMC Simple Strategy"]
    end

    IG -->|"Real-time ticks"| STREAM
    STREAM -->|"INSERT candles"| DB
    DB -->|"SELECT candles"| DF
    DF --> RS
    RS --> IND
    IND --> STRAT
    """, height=400)

    st.markdown("### Resampling Logic")
    st.markdown("""
    The system stores 5-minute candles as the base timeframe and resamples them on-demand:

    | Target Timeframe | Source Candles | Resampling Rule |
    |------------------|----------------|-----------------|
    | 15m | 3 x 5m candles | OHLC aggregation |
    | 1h | 12 x 5m candles | OHLC aggregation |
    | 4h | 48 x 5m candles | OHLC aggregation |

    **OHLC Aggregation:**
    - Open: First candle's open
    - High: Maximum high
    - Low: Minimum low
    - Close: Last candle's close
    - Volume: Sum of volumes
    """)

    st.markdown("### Key Files")
    st.markdown("""
    | File | Purpose | Location |
    |------|---------|----------|
    | `ig_stream_to_postgres.py` | Lightstreamer connection & candle storage | streamlit/igstream/ |
    | `data_fetcher.py` | Data retrieval & resampling (2400+ lines) | worker/app/forex_scanner/core/ |
    | `technical.py` | Technical indicator calculations | worker/app/forex_scanner/analysis/ |
    | `database.py` | Database connection utilities | worker/app/forex_scanner/core/ |
    """)

    st.markdown("### Indicator Calculations")
    with st.expander("View Available Indicators"):
        st.markdown("""
        The DataFetcher adds these indicators to the enhanced DataFrame:

        **Moving Averages:**
        - EMA (5, 13, 21, 50, 200 periods)
        - SMA (various periods)

        **Momentum Indicators:**
        - MACD (12/26/9 standard)
        - RSI (14 period)
        - Two-Pole Oscillator
        - Momentum Bias Index

        **Volatility Indicators:**
        - ATR (14 period)
        - Bollinger Bands

        **Volume Indicators:**
        - Volume SMA
        - Volume ratio
        """)


def render_signal_detection_section():
    """Section 3: Signal Detection Pipeline."""
    st.markdown('<div class="section-header"><h2>🎯 Signal Detection Pipeline</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    The signal detection pipeline is a 6-stage process coordinated by the TradingOrchestrator.
    Currently, only the **SMC Simple Strategy** is active.
    """)

    # Signal Detection Mermaid Diagram
    st.markdown("### Signal Flow")
    render_mermaid("""
flowchart TB
    subgraph Entry["Entry Point"]
        TS["trade_scan.py<br/>CLI Entry"]
    end

    subgraph Orchestration["Orchestration Layer"]
        TO["TradingOrchestrator<br/>(trading_orchestrator.py)"]
        IFS["IntelligentForexScanner<br/>(scanner.py)"]
    end

    subgraph Detection["Signal Detection"]
        SD["SignalDetector<br/>(signal_detector.py)"]
    end

    subgraph Strategy["SMC Simple Strategy"]
        SMC["SMCSimpleStrategy"]
        OB["Order Block<br/>Detection"]
        FVG["Fair Value Gap<br/>Analysis"]
        MS["Market Structure<br/>BOS/CHoCH"]
        LIQ["Liquidity<br/>Sweep Detection"]
    end

    subgraph Validation["Validation & Execution"]
        VAL["Signal Validation<br/>Confidence Check"]
        DEDUP["Deduplication<br/>Alert Cooldown"]
        ALERT["AlertHistoryManager<br/>(alert_history.py)"]
        EXEC["Order Execution<br/>(order_executor.py)"]
    end

    TS --> TO
    TO --> IFS
    IFS --> SD
    SD --> SMC
    SMC --> OB
    SMC --> FVG
    SMC --> MS
    SMC --> LIQ
    OB --> VAL
    FVG --> VAL
    MS --> VAL
    LIQ --> VAL
    VAL --> DEDUP
    DEDUP --> ALERT
    ALERT --> EXEC
    """, height=600)

    st.markdown("### Signal Detection Stages")

    stages = [
        ("1. Data Fetching", "TradingOrchestrator retrieves enhanced candle data via DataFetcher"),
        ("2. Strategy Analysis", "SMC Simple Strategy analyzes order blocks, FVGs, and market structure"),
        ("3. Signal Generation", "Strategy generates BUY/SELL signals with confidence scores"),
        ("4. Validation", "Signals are validated against minimum confidence thresholds"),
        ("5. Deduplication", "AlertHistoryManager prevents duplicate signals (cooldown period)"),
        ("6. Order Execution", "Valid signals are sent to fastapi-dev for order placement"),
    ]

    for stage, desc in stages:
        st.markdown(f"**{stage}:** {desc}")

    st.markdown("### Key Files")
    st.markdown("""
    | File | Purpose | Key Classes/Functions |
    |------|---------|----------------------|
    | `trade_scan.py` | CLI entry point | Main scan loop |
    | `trading_orchestrator.py` | Scan coordination | TradingOrchestrator |
    | `scanner.py` | Signal scanning | IntelligentForexScanner |
    | `signal_detector.py` | Strategy delegation | SignalDetector |
    | `smc_simple_strategy.py` | SMC strategy logic | SMCSimpleStrategy |
    | `alert_history.py` | Alert deduplication | AlertHistoryManager |
    | `order_executor.py` | Order placement | execute_order() |
    """)


def render_trade_execution_section():
    """Section 4: Trade Execution & Monitoring."""
    st.markdown('<div class="section-header"><h2>💹 Trade Execution & Monitoring</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    Once a signal passes validation, it's sent to the `fastapi-dev` container for order execution.
    The trade is then monitored with a progressive trailing stop system.
    """)

    # Trade Lifecycle Mermaid Diagram
    st.markdown("### Trade Lifecycle")
    render_mermaid("""
stateDiagram-v2
    [*] --> Signal: Signal Generated
    Signal --> Validation: Validate Signal
    Validation --> Rejected: Failed Validation
    Rejected --> [*]
    Validation --> OrderPlaced: Passed Validation
    OrderPlaced --> Open: IG API Confirms
    Open --> BreakEven: Profit +6-8 pips
    BreakEven --> PartialClose: Profit +13 pips
    PartialClose --> Stage2: Profit +15 pips
    Stage2 --> Stage3: Profit +20 pips
    Stage3 --> Closed: Trailing Stop Hit
    Open --> Closed: Stop Loss Hit
    BreakEven --> Closed: Stop Loss Hit
    PartialClose --> Closed: Trailing Stop Hit
    Stage2 --> Closed: Trailing Stop Hit
    Closed --> [*]
    """, height=500)

    st.markdown("### Progressive Trailing Stop System")
    render_mermaid("""
flowchart LR
    ENTRY["Entry<br/>Position Open"] --> BE["Break-Even<br/>+6-8 pips<br/>SL → Entry+1"]
    BE --> PC["Partial Close<br/>+13 pips<br/>Close 50%"]
    PC --> S1["Stage 1<br/>+10 pips<br/>Lock 5 pips"]
    S1 --> S2["Stage 2<br/>+15 pips<br/>Lock 10 pips"]
    S2 --> S3["Stage 3<br/>+20 pips<br/>% Trailing"]
    """, height=200)

    st.markdown("### Trailing Stop Configuration by Pair")
    st.markdown("""
    Configuration is in `dev-app/config.py` → `PAIR_TRAILING_CONFIGS`:

    | Pair Type | BE Trigger | Partial Close | Stage 3 Start |
    |-----------|------------|---------------|---------------|
    | Major (EURUSD, USDJPY) | 6 pips | 13 pips | 20 pips |
    | GBPUSD | 8 pips | 15 pips | 20 pips |
    | Cross pairs (GBPJPY) | 8 pips | 18 pips | 25 pips |
    """)

    st.markdown("### Key Files")
    st.markdown("""
    | File | Container | Purpose |
    |------|-----------|---------|
    | `trailing_class.py` | fastapi-dev | Main trailing stop logic |
    | `config.py` | fastapi-dev | PAIR_TRAILING_CONFIGS (source of truth) |
    | `adjust_stop_service.py` | fastapi-dev | IG API stop level adjustments |
    | `ig_orders.py` | fastapi-dev | Order execution & partial close |
    | `trade_monitor.py` | fastapi-dev | Monitoring loop |
    """)

    st.markdown("""
    <div class="info-box">
    <strong>📝 Note:</strong> Stop adjustments use <strong>absolute price levels</strong>, not offsets.
    The <code>new_stop_level</code> parameter passes the exact price to IG Markets API.
    </div>
    """, unsafe_allow_html=True)


def render_container_details_section():
    """Section 5: Container Details."""
    st.markdown('<div class="section-header"><h2>🐳 Container Details</h2></div>', unsafe_allow_html=True)

    st.markdown("### All Docker Containers")

    containers_data = [
        {"Container": "fastapi-dev", "Port": "8001", "Purpose": "Live trade execution, trailing stops, position monitoring", "Entry Point": "main.py", "Config": "config.py"},
        {"Container": "fastapi-prod", "Port": "8002", "Purpose": "Production API (mirror of dev)", "Entry Point": "main.py", "Config": "config.py"},
        {"Container": "fastapi-stream", "Port": "8003", "Purpose": "WebSocket streaming for real-time data", "Entry Point": "main.py", "Config": "config.py"},
        {"Container": "task-worker", "Port": "-", "Purpose": "Signal scanning, backtesting, strategy execution", "Entry Point": "trade_scan.py", "Config": "forex_scanner/config.py"},
        {"Container": "streamlit", "Port": "8501", "Purpose": "Analytics dashboard, visualization, configuration UI", "Entry Point": "streamlit_app.py", "Config": "config.py"},
        {"Container": "tradingview", "Port": "8080", "Purpose": "TradingView integration, chart API", "Entry Point": "api server", "Config": "-"},
        {"Container": "vector-db", "Port": "8090", "Purpose": "ChromaDB for embeddings and semantic search", "Entry Point": "vector_db_service.py", "Config": "-"},
        {"Container": "economic-calendar", "Port": "8091", "Purpose": "Economic calendar events service", "Entry Point": "main.py", "Config": "-"},
        {"Container": "system-monitor", "Port": "8095", "Purpose": "Container health monitoring, Telegram alerts", "Entry Point": "main.py", "Config": "-"},
        {"Container": "postgres", "Port": "5432", "Purpose": "PostgreSQL database (forex, forex_config, strategy_config, stocks)", "Entry Point": "-", "Config": "-"},
        {"Container": "nginx", "Port": "80/443", "Purpose": "Reverse proxy, SSL termination", "Entry Point": "-", "Config": "nginx.conf"},
        {"Container": "pgadmin", "Port": "4445", "Purpose": "Database administration UI", "Entry Point": "-", "Config": "-"},
        {"Container": "db-backup", "Port": "-", "Purpose": "Scheduled database backups", "Entry Point": "enhanced_backup.sh", "Config": "-"},
        {"Container": "stock-scheduler", "Port": "-", "Purpose": "Stock scanner scheduler", "Entry Point": "scheduler.py", "Config": "-"},
    ]

    df = pd.DataFrame(containers_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Container Communication")
    render_mermaid("""
flowchart LR
    subgraph External
        USER["User/Browser"]
        IGAPI["IG Markets API"]
    end

    subgraph Proxy
        NX["nginx<br/>:80/443"]
    end

    subgraph API
        FD["fastapi-dev<br/>:8001"]
        FS["fastapi-stream<br/>:8003"]
    end

    subgraph Worker
        TW["task-worker"]
    end

    subgraph UI
        ST["streamlit<br/>:8501"]
    end

    subgraph Data
        PG[("PostgreSQL<br/>:5432")]
    end

    USER --> NX
    NX --> FD
    NX --> ST
    TW -->|"ORDER_API_URL"| FD
    FD --> IGAPI
    FS --> IGAPI
    TW --> PG
    FD --> PG
    ST --> PG
    """, height=450)

    st.markdown("### Environment Variables")
    with st.expander("View Common Environment Variables"):
        st.markdown("""
        | Variable | Description | Used By |
        |----------|-------------|---------|
        | `DATABASE_URL` | Main forex database connection | All containers |
        | `CONFIG_DATABASE_URL` | forex_config database | fastapi-*, streamlit |
        | `STRATEGY_CONFIG_DATABASE_URL` | strategy_config database | task-worker, streamlit |
        | `ORDER_API_URL` | FastAPI order endpoint | task-worker |
        | `IG_API_KEY` | IG Markets API key | fastapi-*, task-worker |
        | `IG_PWD` | IG Markets password | fastapi-*, task-worker |
        | `VECTOR_DB_URL` | ChromaDB endpoint | task-worker |
        | `TRADINGVIEW_API_URL` | TradingView API endpoint | task-worker, streamlit |
        """)

    st.markdown("### Scheduled Tasks (Cron Jobs)")
    st.markdown("""
    The following cron jobs run on the **host machine** (not inside containers):
    """)

    cron_data = [
        {"Schedule": "0 2 * * * (2 AM daily)", "Task": "Stock Fundamentals Update", "Command": "docker exec task-worker overnight_fundamentals_update.sh", "Log": "logs/fundamentals_cron.log"},
        {"Schedule": "0 3 * * * (3 AM daily)", "Task": "SMC Rejection Outcome Analyzer", "Command": "docker exec task-worker rejection_outcome_analyzer.py", "Log": "logs/rejection_outcome_cron.log"},
    ]

    cron_df = pd.DataFrame(cron_data)
    st.dataframe(cron_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Container-based scheduled tasks:**
    - `db-backup` container runs `enhanced_backup.sh` every 24 hours (configured in docker-compose.yml)
    - Backups are stored in `/media/hr/TradeSystemBackup/TradeSystemV1-Backups/postgresbackup`
    """)

    with st.expander("View Cron Job Management"):
        st.code("""
# View current cron jobs
crontab -l

# Edit cron jobs
crontab -e

# View cron logs
tail -f /home/hr/Projects/TradeSystemV1/logs/fundamentals_cron.log
tail -f /home/hr/Projects/TradeSystemV1/logs/rejection_outcome_cron.log

# Manually trigger a cron job
docker exec task-worker /bin/bash /app/stock_scanner/scripts/overnight_fundamentals_update.sh
docker exec task-worker python /app/forex_scanner/monitoring/rejection_outcome_analyzer.py
        """, language="bash")

    st.markdown("### Backup Scripts")
    st.markdown("""
    Located in `scripts/` directory:

    | Script | Purpose |
    |--------|---------|
    | `enhanced_backup.sh` | Full database backup with retention policies |
    | `backup_database.sh` | Basic database backup |
    | `restore_database.sh` | Database restoration from backup |
    | `backup_health.sh` | Health check for backup system |
    """)


def render_database_section():
    """Section 6: Database Schema."""
    st.markdown('<div class="section-header"><h2>🗄️ Database Schema</h2></div>', unsafe_allow_html=True)

    st.markdown("### Database Overview")
    render_mermaid("""
erDiagram
    FOREX_DB {
        ig_candles PK "Market candle data"
        trade_log PK "Trade history"
        alert_history PK "Signal alerts"
        preferred_forex_prices PK "Price preferences"
    }

    FOREX_CONFIG_DB {
        system_config PK "System settings"
    }

    STRATEGY_CONFIG_DB {
        smc_simple_global_config PK "SMC global settings"
        smc_simple_pair_overrides PK "Per-pair overrides"
        smc_simple_config_audit PK "Audit trail"
        smc_simple_parameter_metadata PK "UI metadata"
    }

    STOCKS_DB {
        stock_candles PK "Stock market data"
        stock_signals PK "Stock signals"
    }
    """, height=400)

    st.markdown("### forex Database")

    st.markdown("#### ig_candles Table")
    st.markdown("""
    Primary market data storage for forex candles.

    | Column | Type | Description |
    |--------|------|-------------|
    | start_time | TIMESTAMP | Candle start time (PK) |
    | epic | VARCHAR(50) | Trading instrument (PK) |
    | timeframe | INTEGER | Timeframe in minutes (PK) |
    | open_price_mid | DECIMAL | Open price |
    | high_price_mid | DECIMAL | High price |
    | low_price_mid | DECIMAL | Low price |
    | close_price_mid | DECIMAL | Close price |
    | volume | INTEGER | Volume |
    | ltv | DECIMAL | Last traded volume |

    **Composite Primary Key:** (start_time, epic, timeframe)
    """)

    st.markdown("#### trade_log Table")
    st.markdown("""
    | Column | Type | Description |
    |--------|------|-------------|
    | id | SERIAL | Primary key |
    | epic | VARCHAR | Trading instrument |
    | direction | VARCHAR | BUY/SELL |
    | size | DECIMAL | Position size |
    | entry_price | DECIMAL | Entry price |
    | exit_price | DECIMAL | Exit price |
    | stop_loss | DECIMAL | Stop loss level |
    | take_profit | DECIMAL | Take profit level |
    | confidence | DECIMAL | Signal confidence (0-1) |
    | strategy | VARCHAR | Strategy name |
    | profit_loss_pips | DECIMAL | P&L in pips |
    | status | VARCHAR | OPEN/CLOSED/PENDING |
    | moved_to_breakeven | BOOLEAN | BE triggered flag |
    | partial_close_executed | BOOLEAN | Partial close flag |
    | timestamp | TIMESTAMP | Creation time |
    """)

    st.markdown("### strategy_config Database")
    st.markdown("""
    Database-driven configuration for SMC Simple strategy.

    #### smc_simple_global_config Table
    ~80 columns containing global strategy parameters:
    - EMA periods, swing lookback settings
    - Confidence thresholds
    - Order block settings
    - FVG detection parameters
    - Session filters (Asian block, etc.)
    - MACD alignment filters

    #### smc_simple_pair_overrides Table
    Per-pair override settings that inherit from global config:
    - Pair-specific confidence thresholds
    - Custom SL buffer settings
    - Session override flags
    - ATR-based adjustments

    #### smc_simple_config_audit Table
    Audit trail for all configuration changes:
    - Timestamp, user, change reason
    - Old/new values for each change
    """)

    st.markdown("### Database Queries")
    with st.expander("View Common Queries"):
        st.code("""
# Check recent candles
docker exec postgres psql -U postgres -d forex -c "
SELECT epic, timeframe, COUNT(*) as candle_count,
       MAX(start_time) as latest
FROM ig_candles
GROUP BY epic, timeframe
ORDER BY epic, timeframe;"

# Check open trades
docker exec postgres psql -U postgres -d forex -c "
SELECT id, epic, direction, entry_price, sl_price, status
FROM trade_log WHERE status = 'OPEN';"

# Check SMC global config
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT ema_period, min_confidence_threshold, swing_lookback_bars
FROM smc_simple_global_config WHERE is_active = TRUE;"

# Check pair overrides
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT epic, min_confidence_threshold, sl_buffer_pips
FROM smc_simple_pair_overrides WHERE config_id = 1;"
        """, language="bash")


def render_strategy_section():
    """Section 7: SMC Simple Strategy Reference."""
    st.markdown('<div class="section-header"><h2>📈 SMC Simple Strategy Reference</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    The **SMC Simple Strategy** is the only active trading strategy. It uses Smart Money Concepts
    (institutional trading patterns) with database-driven configuration.
    """)

    st.markdown("### Strategy Components")
    render_mermaid("""
flowchart TB
    subgraph Input["Market Data Input"]
        CANDLES["Enhanced Candle Data<br/>with Indicators"]
    end

    subgraph Analysis["SMC Analysis"]
        MS["Market Structure<br/>Swing Highs/Lows"]
        BOS["Break of Structure<br/>(BOS)"]
        CHOCH["Change of Character<br/>(CHoCH)"]
        OB["Order Block<br/>Detection"]
        FVG["Fair Value Gap<br/>Identification"]
        LIQ["Liquidity Sweep<br/>Detection"]
    end

    subgraph Signal["Signal Generation"]
        CONF["Confidence Scoring"]
        VALID["Validation Filters"]
        SIG["BUY/SELL Signal"]
    end

    CANDLES --> MS
    MS --> BOS
    MS --> CHOCH
    BOS --> OB
    CHOCH --> OB
    OB --> FVG
    FVG --> LIQ
    LIQ --> CONF
    CONF --> VALID
    VALID --> SIG
    """, height=500)

    st.markdown("### SMC Concepts Explained")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Order Blocks (OB):**
        - Last bearish candle before bullish move (bullish OB)
        - Last bullish candle before bearish move (bearish OB)
        - Represents institutional accumulation/distribution

        **Fair Value Gaps (FVG):**
        - Price imbalance between 3 consecutive candles
        - Gap between candle 1 high and candle 3 low (bullish)
        - Gap between candle 1 low and candle 3 high (bearish)
        """)

    with col2:
        st.markdown("""
        **Market Structure:**
        - **BOS (Break of Structure):** Price breaks previous swing high/low
        - **CHoCH (Change of Character):** Trend reversal signal
        - **Swing Points:** Local highs and lows for structure analysis

        **Liquidity:**
        - Stop loss clusters (above highs, below lows)
        - Smart money targets these levels before reversing
        """)

    st.markdown("### Configuration Architecture")
    render_mermaid("""
flowchart LR
    subgraph Database["strategy_config DB"]
        GLOBAL["smc_simple_global_config<br/>~80 parameters"]
        PAIR["smc_simple_pair_overrides<br/>Per-pair settings"]
        AUDIT["smc_simple_config_audit<br/>Change history"]
    end

    subgraph Service["Config Service Layer"]
        WS["Worker Service<br/>(task-worker)"]
        SS["Streamlit Service<br/>(streamlit)"]
    end

    subgraph Consumer["Consumers"]
        STRAT["SMCSimpleStrategy"]
        UI["Streamlit Config Tab"]
    end

    GLOBAL --> WS
    PAIR --> WS
    WS --> STRAT
    GLOBAL --> SS
    PAIR --> SS
    SS --> UI
    UI -->|"Updates"| SS
    SS --> GLOBAL
    SS --> AUDIT
    """, height=400)

    st.markdown("### Key Configuration Parameters")
    with st.expander("View Main Parameters"):
        st.markdown("""
        | Parameter | Default | Description |
        |-----------|---------|-------------|
        | `ema_period` | 21 | EMA period for trend filter |
        | `swing_lookback_bars` | 10 | Bars to look back for swing detection |
        | `swing_strength_bars` | 3 | Bars required to confirm swing |
        | `min_confidence_threshold` | 0.45 | Minimum signal confidence |
        | `sl_buffer_pips` | 2.0 | Stop loss buffer in pips |
        | `atr_period` | 14 | ATR period for volatility |
        | `block_asian_session` | True | Block signals during Asian session |
        | `macd_alignment_filter_enabled` | True | Require MACD alignment |
        | `volume_sma_period` | 20 | Volume SMA period |
        | `max_pullback_wait_bars` | 5 | Max bars to wait for pullback |
        """)

    st.markdown("### Key Files")
    st.markdown("""
    | File | Location | Purpose |
    |------|----------|---------|
    | `smc_simple_strategy.py` | worker/app/forex_scanner/core/strategies/ | Strategy implementation |
    | `smc_simple_config_service.py` | worker/app/forex_scanner/services/ | Config service (worker) |
    | `smc_simple_config_service.py` | streamlit/services/ | Config service (streamlit) |
    | `smc_config_tab.py` | streamlit/components/tabs/ | Configuration UI |
    | `create_strategy_config_db.sql` | worker/app/forex_scanner/migrations/ | Database schema |
    """)


def render_config_section():
    """Section 8: Configuration Guide."""
    st.markdown('<div class="section-header"><h2>⚙️ Configuration Guide</h2></div>', unsafe_allow_html=True)

    st.markdown("### Configuration Locations")
    st.markdown("""
    | Setting Type | Location | Container | Notes |
    |--------------|----------|-----------|-------|
    | Trading pairs | `forex_scanner/config.py` | task-worker | `EPIC_LIST` variable |
    | SMC strategy params | `strategy_config` database | task-worker | Via config service |
    | Trailing stops | `dev-app/config.py` | fastapi-dev | `PAIR_TRAILING_CONFIGS` |
    | Scan interval | `forex_scanner/config.py` | task-worker | `SCAN_INTERVAL_SECONDS` |
    | Confidence threshold | `strategy_config` database | task-worker | Per-strategy setting |
    """)

    st.markdown("### Modifying SMC Simple Configuration")
    st.markdown("""
    **Option 1: Streamlit UI** (Recommended)
    1. Navigate to Unified Analytics → Settings → SMC Config
    2. Modify global parameters or pair overrides
    3. Changes take effect within 2 minutes (cache TTL)

    **Option 2: Direct Database**
    ```bash
    docker exec postgres psql -U postgres -d strategy_config -c "
    UPDATE smc_simple_global_config
    SET min_confidence_threshold = 0.50
    WHERE is_active = TRUE;"
    ```
    """)

    st.markdown("### Modifying Trailing Stop Configuration")
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Source of Truth:</strong> <code>dev-app/config.py</code> is the ONLY location for trailing stop configuration.
    Do NOT edit <code>worker/app/forex_scanner/config_trailing_stops.py</code> for live trading.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ```python
    # dev-app/config.py
    PAIR_TRAILING_CONFIGS = {
        'CS.D.EURUSD.MINI.IP': {
            'early_breakeven_trigger_points': 6,
            'early_breakeven_buffer_points': 1,
            'stage1_trigger_points': 10,
            'stage1_lock_points': 5,
            'stage2_trigger_points': 15,
            'stage2_lock_points': 10,
            'stage3_trigger_points': 20,
            'partial_close_trigger_points': 13,
            'partial_close_size': 0.5,
            'enable_partial_close': True,
        },
        # ... more pairs
    }
    ```

    After editing, restart the container:
    ```bash
    docker restart fastapi-dev
    ```
    """)

    st.markdown("### Common Operations")

    with st.expander("View CLI Commands"):
        st.code("""
# Run single scan
docker exec -it task-worker python /app/trade_scan.py scan

# Run live trading mode
docker exec -it task-worker python /app/trade_scan.py live 120

# Check system status
docker exec -it task-worker python /app/trade_scan.py status

# Run backtest
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 7

# Check container logs
docker logs task-worker --tail 100
docker logs fastapi-dev --tail 100

# Check database
docker exec postgres psql -U postgres -d forex -c "SELECT * FROM trade_log ORDER BY timestamp DESC LIMIT 10;"
        """, language="bash")


def render_optimizer_section():
    """Section 9: Parameter Optimizer."""
    st.markdown('<div class="section-header"><h2>🔧 Parameter Optimizer</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    The **Parameter Optimizer** automatically analyzes rejection outcomes and generates
    recommendations to adjust SMC Simple strategy parameters. It helps optimize filter
    thresholds based on actual trade performance data.
    """)

    st.markdown("### How It Works")
    render_mermaid("""
flowchart TB
    subgraph Data["Data Collection"]
        REJ["smc_rejection_outcomes<br/>Table"]
        OUT["Outcome Analysis<br/>(WIN/LOSS/PENDING)"]
    end

    subgraph Analysis["Performance Analysis"]
        STAGE["Stage-Level Analysis<br/>Win rates per rejection stage"]
        PAIR["Pair-Level Analysis<br/>Win rates per currency pair"]
    end

    subgraph Recommendations["Recommendation Engine"]
        RELAX["RELAX Filter<br/>Win rate > 55%<br/>= Lower threshold"]
        TIGHTEN["TIGHTEN Filter<br/>Win rate < 45%<br/>= Raise threshold"]
        SKIP["NO CHANGE<br/>45-55% win rate<br/>or at bounds"]
    end

    subgraph Apply["Apply Changes"]
        DB["Update strategy_config DB"]
        AUDIT["Create audit record"]
    end

    REJ --> OUT
    OUT --> STAGE
    OUT --> PAIR
    STAGE --> RELAX
    STAGE --> TIGHTEN
    STAGE --> SKIP
    PAIR --> RELAX
    PAIR --> TIGHTEN
    RELAX --> DB
    TIGHTEN --> DB
    DB --> AUDIT
    """, height=500)

    st.markdown("### Analyzable Rejection Stages")
    st.markdown("""
    The optimizer analyzes rejections that have an `attempted_direction` field,
    allowing outcome correlation with market movement:

    | Stage | Parameter Affected | Relax Delta | Tighten Delta |
    |-------|-------------------|-------------|---------------|
    | CONFIDENCE | `min_confidence_threshold` | -0.02 | +0.02 |
    | CONFIDENCE_CAP | `min_confidence_threshold` | -0.02 | +0.02 |
    | TIER2_SWING | `min_body_percentage` | -0.05 | +0.05 |
    | TIER3_PULLBACK | `fib_pullback_min` | -0.02 | +0.02 |
    | VOLUME_LOW | `min_volume_ratio` | -0.05 | +0.05 |
    | MACD_MISALIGNED | `macd_alignment_filter_enabled` | False | True |
    | RISK_LIMIT | `min_rr_ratio` | -0.1 | +0.1 |

    **Non-analyzable stages** (no direction data): SESSION, COOLDOWN, TIER1_EMA
    """)

    st.markdown("### Parameter Bounds")
    st.markdown("""
    To prevent over-optimization (infinite adjustment loops), each parameter has
    minimum and maximum bounds:

    | Parameter | Min | Max |
    |-----------|-----|-----|
    | `min_confidence_threshold` | 0.40 | 0.60 |
    | `min_body_percentage` | 0.10 | 0.50 |
    | `fib_pullback_min` | 0.15 | 0.40 |
    | `min_volume_ratio` | 0.30 | 0.80 |
    | `min_rr_ratio` | 1.0 | 2.5 |

    When a parameter reaches its bound, the optimizer skips further adjustments
    in that direction and notes "at minimum/maximum bound" in the skip reason.
    """)

    st.markdown("### Using the Optimizer")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Streamlit UI** (Recommended)
        1. Navigate to **Unified Analytics**
        2. Go to **Settings** → **SMC Config**
        3. Select **Parameter Optimizer** tab
        4. Set analysis period (days)
        5. Click **Fetch Recommendations**
        6. Review recommendations
        7. Click **Apply Recommendations**
        """)

    with col2:
        st.markdown("""
        **CLI Usage**
        ```bash
        # Dry run (view recommendations)
        docker exec task-worker python \\
          /app/forex_scanner/monitoring/parameter_optimizer.py \\
          --days 30

        # Apply recommendations
        docker exec task-worker python \\
          /app/forex_scanner/monitoring/parameter_optimizer.py \\
          --days 30 --apply

        # Export SQL only
        docker exec task-worker python \\
          /app/forex_scanner/monitoring/parameter_optimizer.py \\
          --days 30 --export-sql
        ```
        """)

    st.markdown("### Recommendation Logic")
    st.markdown("""
    The optimizer uses these thresholds to determine actions:

    | Win Rate | Action | Reasoning |
    |----------|--------|-----------|
    | > 55% | **RELAX** filter | High win rate suggests filter is too strict |
    | < 45% | **TIGHTEN** filter | Low win rate suggests filter is too loose |
    | 45-55% | **NO CHANGE** | Win rate is acceptable |
    | At bound | **SKIP** | Parameter already at min/max limit |
    | < 10 samples | **SKIP** | Insufficient data for reliable recommendation |
    """)

    st.markdown("### Output Example")
    with st.expander("View Sample Optimizer Output"):
        st.code("""
=== SMC Simple Parameter Optimizer ===
Analysis Period: 30 days
Mode: DRY-RUN (preview only)

--- Stage-Level Analysis ---
Stage: CONFIDENCE
  Total: 45, Won: 28, Lost: 17
  Win Rate: 62.22%
  Recommendation: RELAX - Lower min_confidence_threshold by 0.02
  Current: 0.46, Proposed: 0.44

Stage: VOLUME_LOW
  Total: 23, Won: 8, Lost: 15
  Win Rate: 34.78%
  Recommendation: TIGHTEN - Raise min_volume_ratio by 0.05
  Current: 0.50, Proposed: 0.55

--- Pair-Level Analysis ---
Epic: CS.D.EURUSD.MINI.IP
  Total: 67, Won: 41, Lost: 26
  Win Rate: 61.19%
  Recommendation: RELAX - Lower min_confidence_threshold by 0.02
  Current: 0.44, Proposed: 0.42
  Note: Already at minimum bound (0.40), skipping

--- Summary ---
Recommendations generated: 2
Skipped (at bounds): 1
To apply changes, run with --apply flag
        """, language="text")

    st.markdown("### Key Files")
    st.markdown("""
    | File | Location | Purpose |
    |------|----------|---------|
    | `parameter_optimizer.py` | worker/app/forex_scanner/monitoring/ | CLI optimizer tool |
    | `smc_simple_config_service.py` | streamlit/services/ | Service functions with bounds |
    | `smc_config_tab.py` | streamlit/components/tabs/ | Parameter Optimizer UI tab |
    | `rejection_outcome_analyzer.py` | worker/app/forex_scanner/monitoring/ | Outcome data collector |
    """)

    st.markdown("### Scheduled Optimization")
    st.markdown("""
    The rejection outcome analyzer runs daily at 3 AM via cron job:

    ```bash
    0 3 * * * docker exec task-worker python /app/forex_scanner/monitoring/rejection_outcome_analyzer.py
    ```

    This populates the `smc_rejection_outcomes` table that the optimizer uses for analysis.
    You can run the optimizer manually after the analyzer to get fresh recommendations.
    """)

    st.markdown("""
    <div class="info-box">
    <strong>💡 Best Practice:</strong> Review recommendations before applying. The optimizer
    provides data-driven suggestions, but market conditions change. Consider running in
    dry-run mode first and reviewing the win rate statistics before making changes.
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    render_header()

    # Get selected section from sidebar
    selected_section = render_sidebar_navigation()

    # Render the selected section
    if selected_section == "🏗️ System Architecture":
        render_architecture_section()
    elif selected_section == "📊 Candle Data Flow":
        render_candle_data_flow_section()
    elif selected_section == "🎯 Signal Detection":
        render_signal_detection_section()
    elif selected_section == "💹 Trade Execution":
        render_trade_execution_section()
    elif selected_section == "🐳 Container Details":
        render_container_details_section()
    elif selected_section == "🗄️ Database Schema":
        render_database_section()
    elif selected_section == "📈 SMC Simple Strategy":
        render_strategy_section()
    elif selected_section == "⚙️ Configuration Guide":
        render_config_section()
    elif selected_section == "🔧 Parameter Optimizer":
        render_optimizer_section()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        📚 Forex Scanner System Documentation | TradeSystemV1<br/>
        Last Updated: December 2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
