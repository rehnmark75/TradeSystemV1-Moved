# System Architecture Overview

This document provides comprehensive architectural guidance for understanding TradeSystemV1's microservices design, database models, and system components.

## Service Architecture

The system follows a microservices architecture with Docker containers, designed for scalability, maintainability, and real-time trading operations.

### 1. FastAPI Services (dev-app, prod-app, stream-app)

**Core Responsibilities:**
- REST APIs for trading operations and system management
- WebSocket streaming for real-time market data and trade updates
- Integration with IG Markets API for order execution
- Real-time data processing and forwarding

**Key Features:**
- **Database Integration**: PostgreSQL with two schemas (`forex`, `forex_config`)
- **Authentication**: Azure Key Vault integration for secure credential management
- **Real-time Streaming**: WebSocket endpoints for live data feeds
- **Trading Operations**: Order placement, position management, P&L tracking

**Configuration:**
- **Development API**: Port 8001, hot-reload enabled
- **Production API**: Port 8000, optimized for performance
- **Streaming API**: Dedicated WebSocket service for real-time data

### 2. Task Worker (worker/app)

**Core Responsibilities:**
- Forex scanner with multiple trading strategies
- Signal detection and validation pipeline
- Backtesting engine for strategy performance analysis
- Alert system with intelligent deduplication

**Trading Strategies:**
- **EMA Strategy**: Advanced 5-layer validation system
- **MACD Strategy**: Timeframe-aware with database optimization
- **Zero-Lag Strategy**: Fast-reacting EMA-based signals
- **Smart Money Concepts**: Institutional flow analysis

**Key Components:**
```
worker/app/forex_scanner/
├── core/
│   ├── strategies/          # Strategy implementations
│   ├── detection/           # Signal detection pipeline
│   └── data_fetcher.py     # Data fetching and caching
├── optimization/           # Dynamic parameter system
├── backtests/             # Strategy validation tools
└── configdata/            # Modular configuration system
```

### 3. Streamlit Dashboard (streamlit/)

**Core Responsibilities:**
- Real-time trading visualization and monitoring
- Performance analytics and trade history analysis
- System log monitoring and debugging tools
- Interactive market intelligence and regime analysis

**Key Features:**
- **TradingView Integration**: Professional charting with custom overlays
- **Market Intelligence**: Real-time regime detection and session analysis
- **Trade Context Analysis**: AI-powered trade quality assessment
- **Performance Metrics**: P&L tracking, win rates, and strategy comparison

### 4. Supporting Services

**PostgreSQL Database:**
- **Primary Schema (`forex`)**: Market data, trades, optimization results
- **Config Schema (`forex_config`)**: System configuration and user settings
- **High Performance**: Optimized indexes for time-series data queries

**Nginx Reverse Proxy:**
- SSL/TLS termination with Let's Encrypt certificates
- Load balancing and request routing
- Static asset serving and caching

**PgAdmin:**
- Web-based database administration
- Query optimization and performance monitoring
- Database backup and maintenance tools

## Database Schema & Models

### Core Trading Models

#### TradeLog (dev-app/services/models.py)
```python
class TradeLog:
    id: int                    # Primary key
    epic: str                  # Trading instrument
    direction: str             # BUY/SELL
    size: float               # Position size
    entry_price: float        # Entry price
    exit_price: float         # Exit price (if closed)
    stop_loss: float          # Stop loss level
    take_profit: float        # Take profit level
    confidence: float         # Signal confidence (0.0-1.0)
    strategy: str             # Strategy name
    profit_loss_pips: float   # P&L in pips
    profit_loss_currency: float # P&L in account currency
    status: str               # OPEN/CLOSED/PENDING
    timestamp: datetime       # Creation timestamp
    closed_timestamp: datetime # Close timestamp
```

#### IGCandle (Market Data)
```python
class IGCandle:
    start_time: datetime      # Candle start time
    epic: str                 # Trading instrument
    timeframe: int            # Timeframe (5, 15, 60 minutes)
    open_price_mid: float     # Open price
    high_price_mid: float     # High price  
    low_price_mid: float      # Low price
    close_price_mid: float    # Close price
    volume: int               # Volume
    ltv: float               # Last traded volume
    
    # Composite primary key: (start_time, epic, timeframe)
```

### Optimization Tables

#### EMA Optimization Schema
```sql
-- Parameter optimization tracking
ema_optimization_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(100),
    description TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_combinations INTEGER,
    status VARCHAR(20)
);

-- Individual test results (14,406+ combinations per epic)
ema_optimization_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES ema_optimization_runs(id),
    epic VARCHAR(50),
    ema_config VARCHAR(20),
    confidence_threshold DECIMAL(5,4),
    timeframe VARCHAR(10),
    smart_money_enabled BOOLEAN,
    stop_loss_pips DECIMAL(6,2),
    take_profit_pips DECIMAL(6,2),
    risk_reward_ratio DECIMAL(6,3),
    total_signals INTEGER,
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(8,4),
    net_pips DECIMAL(10,2),
    composite_score DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Best parameters per epic
ema_best_parameters (
    epic VARCHAR(50) PRIMARY KEY,
    best_ema_config VARCHAR(20),
    best_confidence_threshold DECIMAL(5,4),
    best_timeframe VARCHAR(10),
    optimal_stop_loss_pips DECIMAL(6,2),
    optimal_take_profit_pips DECIMAL(6,2),
    best_win_rate DECIMAL(5,4),
    best_profit_factor DECIMAL(8,4),
    best_net_pips DECIMAL(10,2),
    last_updated TIMESTAMP DEFAULT NOW()
);
```

#### MACD Optimization Schema
```sql
-- Similar structure to EMA but with MACD-specific parameters
macd_best_parameters (
    epic VARCHAR(50) PRIMARY KEY,
    best_fast_ema INTEGER,
    best_slow_ema INTEGER,
    best_signal_ema INTEGER,
    best_confidence_threshold DECIMAL(5,4),
    best_timeframe VARCHAR(10),
    best_histogram_threshold DECIMAL(8,5),
    best_zero_line_filter BOOLEAN,
    best_rsi_filter_enabled BOOLEAN,
    best_momentum_confirmation BOOLEAN,
    best_mtf_enabled BOOLEAN,
    best_smart_money_enabled BOOLEAN,
    optimal_stop_loss_pips DECIMAL(6,2),
    optimal_take_profit_pips DECIMAL(6,2),
    best_win_rate DECIMAL(5,4),
    best_composite_score DECIMAL(10,6),
    last_updated TIMESTAMP DEFAULT NOW()
);
```

## Trading System Components

### Signal Detection Pipeline

The system implements a sophisticated 6-stage signal detection pipeline:

```
1. Data Fetching
   ├── Database queries (ig_candles)
   ├── Real-time API data (IG Markets)
   └── Data quality validation

2. Technical Indicator Calculation
   ├── EMAs (5, 13, 21, 50, 200)
   ├── MACD (12/26/9 or optimized)
   ├── Two-Pole Oscillator (momentum)
   ├── Momentum Bias Index
   └── Smart Money indicators

3. Multi-timeframe Analysis
   ├── Primary timeframe (5m/15m)
   ├── Confirmation timeframe (1h/4h)
   └── Trend timeframe (daily)

4. Signal Validation (5 Layers)
   ├── Core strategy logic
   ├── Momentum confirmation
   ├── Trend alignment
   ├── Multi-timeframe validation
   └── Risk management filters

5. Alert Generation
   ├── Signal confidence scoring
   ├── Deduplication logic
   ├── Position sizing calculation
   └── Risk parameter optimization

6. Order Execution
   ├── IG Markets API integration
   ├── Position management
   ├── Real-time monitoring
   └── P&L tracking
```

### EMA Strategy - Advanced 5-Layer Validation

The EMA strategy implements a sophisticated multi-layer validation system for high-quality signal generation:

**Signal Requirements:**
- **BULL**: Price crosses above EMA(21), with EMA(21) > EMA(50) > EMA(200)
- **BEAR**: Price crosses below EMA(21), with EMA(21) < EMA(50) < EMA(200)

**5-Layer Validation System:**
1. **EMA Crossover Detection**: Price crossing short EMA with proper trend alignment
2. **15m Two-Pole Oscillator**: Momentum validation (GREEN for BUY, PURPLE for SELL)
3. **1H Two-Pole Oscillator**: Higher timeframe momentum confirmation
4. **Momentum Bias Index**: Final momentum validation (AlgoAlpha's indicator)
5. **EMA 200 Trend Filter**: Major trend direction validation

**Configuration Example:**
```python
# Two-Pole Oscillator
TWO_POLE_OSCILLATOR_ENABLED = True
TWO_POLE_MTF_VALIDATION = True        # 1H timeframe validation

# Momentum Bias Index  
MOMENTUM_BIAS_ENABLED = True
MOMENTUM_BIAS_REQUIRE_ABOVE_BOUNDARY = False  # Optional stricter filtering

# EMA 200 Trend Filter
EMA_200_TREND_FILTER_ENABLED = True
```

**Performance Characteristics:**
- Very low signal frequency due to strict validation
- High win rates (often 70-90% in testing)
- Excellent risk management through multi-layer filtering
- Prevents trading against momentum or trend direction

### Key Configuration Files

| File | Purpose | Key Settings |
|------|---------|--------------|
| `worker/app/forex_scanner/config.py` | System-level configuration | Scanner settings, trading pairs, thresholds |
| `worker/app/forex_scanner/configdata/strategies/` | Strategy-specific configs | Modular strategy configurations |
| `dev-app/config.py` | API configuration | Database connections, endpoints |
| `docker-compose.yml` | Service orchestration | Container networking, volume mounts |

## Architectural Patterns

### Service Communication

**Internal Communication:**
- **Docker Network**: `lab-net` for inter-service communication
- **Database**: PostgreSQL as central data store and message queue
- **Caching**: Redis and file-based caching for performance optimization
- **Event-driven**: WebSocket connections for real-time updates

**External Integration:**
- **IG Markets API**: REST and streaming APIs for market data and trading
- **Azure Key Vault**: Secure credential and secret management
- **CloudFlare**: DNS and DDoS protection for web interfaces

### Error Handling & Resilience

**Logging Strategy:**
```python
# Comprehensive logging with RotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/trading.log', maxBytes=10MB, backupCount=5),
        StreamHandler()
    ]
)
```

**Retry Mechanisms:**
- **API Calls**: Exponential backoff with circuit breaker pattern
- **Database Operations**: Connection pooling with automatic retry
- **Signal Processing**: Queue-based processing with dead letter handling

**Transaction Management:**
```python
# Database transactions with proper rollback
with db_manager.get_connection() as conn:
    try:
        # Multiple operations
        conn.execute("INSERT INTO trades ...")
        conn.execute("UPDATE positions ...")
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction failed: {e}")
        raise
```

### Security Considerations

**Credential Management:**
- **Azure Key Vault**: All API keys and database credentials
- **Environment Variables**: Runtime configuration without hardcoded secrets
- **SSL/TLS**: End-to-end encryption via Nginx and Certbot
- **Access Control**: Role-based permissions for database and API access

**Network Security:**
```yaml
# Docker network isolation
networks:
  lab-net:
    driver: bridge
    internal: false
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Testing & Validation Approach

The system uses **backtesting** as the primary validation method rather than traditional unit tests:

**Historical Data Analysis:**
- Configurable test periods (7 days to 1 year)
- Multiple timeframe validation
- Performance metrics calculation (win rate, profit factor, Sharpe ratio)

**Strategy Comparison:**
- A/B testing of parameter configurations
- Performance benchmarking across different market conditions
- Statistical significance testing for strategy improvements

**Real-time Validation:**
- Production monitoring with detailed logging
- Alert systems for anomaly detection
- Performance tracking and automatic optimization triggering

**Quality Metrics:**
```python
# Key performance indicators
metrics = {
    'total_signals': int,
    'win_rate': float,           # Percentage of profitable trades
    'profit_factor': float,      # Gross profit / Gross loss
    'net_pips': float,           # Total pips gained/lost
    'max_drawdown': float,       # Largest peak-to-valley decline
    'sharpe_ratio': float,       # Risk-adjusted returns
    'composite_score': float     # Overall performance score
}
```

For detailed command usage, see [Commands & CLI](claude-commands.md).
For strategy development patterns, see [Strategy Development](claude-strategies.md).
For optimization system details, see [Dynamic Parameter System](claude-optimization.md).