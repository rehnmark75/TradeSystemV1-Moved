# Forex Scanner Architecture

This skill provides comprehensive knowledge about the forex_scanner system architecture, entry points, and component relationships.

---

## Entry Points

### 1. Live Scanner: `trade_scan.py`
**Location**: `worker/app/trade_scan.py`
**Purpose**: Main entry point for live trading and Docker deployment

```bash
# Docker default (continuous scanning)
docker exec -it task-worker python /app/trade_scan.py

# Single scan
docker exec -it task-worker python /app/trade_scan.py scan

# Live trading mode
docker exec -it task-worker python /app/trade_scan.py live [interval_seconds]

# Test Claude integration
docker exec -it task-worker python /app/trade_scan.py test-claude

# Show system status
docker exec -it task-worker python /app/trade_scan.py status
```

**Flow**:
```
trade_scan.py
  └── TradingSystem (wrapper class)
        └── TradingOrchestrator (core/trading/trading_orchestrator.py)
              ├── IntelligentForexScanner (core/scanner.py)
              │     └── SignalDetector (core/signal_detector.py)
              │           └── Strategies (core/strategies/*.py)
              ├── TradeValidator
              ├── RiskManager
              ├── OrderManager
              ├── IntegrationManager (Claude AI)
              └── AlertHistoryManager (alerts/alert_history.py)
```

### 2. Backtesting: `bt.py` / `backtest_cli.py`
**Location**: `worker/app/forex_scanner/bt.py` (wrapper) and `worker/app/forex_scanner/backtest_cli.py` (full CLI)

```bash
# Quick backtest (bt.py wrapper)
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 7
docker exec -it task-worker python /app/forex_scanner/bt.py GBPUSD 14 MACD --show-signals

# Full CLI
docker exec -it task-worker python /app/forex_scanner/backtest_cli.py --epic CS.D.EURUSD.CEEM.IP --days 7 --strategy EMA
```

**Strategy shortcuts in bt.py**:
- `EMA`, `MACD`, `BB` (Bollinger+Supertrend), `SMC`, `SMC_STRUCTURE`
- `MOMENTUM`, `ICHIMOKU`, `KAMA`, `ZEROLAG`, `MEANREV`, `RANGING`, `SCALPING`, `VP`

**Pair shortcuts**: `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `USDCHF`, `USDCAD`, `NZDUSD`, `EURJPY`, `AUDJPY`, `GBPJPY`

---

## Core Components

### 1. TradingOrchestrator
**File**: `core/trading/trading_orchestrator.py`
**Role**: High-level coordination of all trading components

**Responsibilities**:
- Intelligence filtering and configuration
- Risk management and validation
- Claude AI analysis via IntegrationManager
- Database operations via AlertHistoryManager
- Trade execution coordination
- Performance tracking and session management

**NOT responsible for** (handled by Scanner):
- Raw signal detection
- Basic confidence filtering
- Deduplication

### 2. IntelligentForexScanner
**File**: `core/scanner.py`
**Class**: `IntelligentForexScanner`
**Role**: Signal detection and deduplication

**Key attributes**:
```python
scanner = IntelligentForexScanner(
    db_manager=db_manager,
    epic_list=['CS.D.EURUSD.CEEM.IP', ...],
    min_confidence=0.7,
    scan_interval=60,
    user_timezone='Europe/Stockholm',
    intelligence_mode='backtest_consistent'
)
```

**Flow**:
```
scan_once() / start_continuous_scan()
  └── SignalDetector.detect_combined_signals()
        └── Strategy.check_for_signals()
```

### 3. SignalDetector
**File**: `core/signal_detector.py`
**Role**: Lightweight coordinator that delegates to specialized strategy modules

**Initializes strategies**:
```python
self.ema_strategy = EMAStrategy(data_fetcher=self.data_fetcher)
self.macd_strategy = MACDStrategy(...)  # Created per-epic with optimized params
self.scalping_strategy = ScalpingStrategy(...)  # If enabled
self.kama_strategy = KAMAStrategy(...)  # If enabled
# ... etc
```

**Key method**: `detect_combined_signals(epic, pair)` - runs enabled strategies and returns signals

### 4. DataFetcher
**File**: `core/data_fetcher.py`
**Role**: Fetches and enhances candle data with technical indicators

**Key methods**:
- `get_enhanced_data(epic, pair, timeframe, lookback_hours)` - Main entry point
- `_fetch_candle_data_optimized()` - Database queries
- `_resample_to_15m_optimized()` / `_resample_to_60m_optimized()` / `_resample_to_4h_optimized()` - Timeframe conversion
- `_enhance_with_analysis_optimized()` - Add technical indicators

### 5. Strategies
**Location**: `core/strategies/`

| Strategy | File | Description |
|----------|------|-------------|
| EMA | `ema_strategy.py` | 5-layer EMA validation system |
| MACD | `macd_strategy.py` | MACD with database-optimized params |
| SMC Structure | `smc_structure_strategy.py` | Smart Money Concepts (price action) |
| SMC Fast | `smc_strategy_fast.py` | Faster SMC variant |
| SMC Simple | `smc_simple_strategy.py` | Simplified SMC |
| Bollinger+Supertrend | `bb_supertrend_strategy.py` | Combined BB and Supertrend |
| Scalping | `scalping_strategy.py` | Multiple scalping modes |
| Momentum | `momentum_strategy.py` | Momentum-based signals |
| KAMA | `kama_strategy.py` | Kaufman Adaptive MA |
| Zero Lag | `zero_lag_strategy.py` | Zero-lag EMA strategy |
| Mean Reversion | `mean_reversion_strategy.py` | Mean reversion signals |
| Ranging Market | `ranging_market_strategy.py` | Range-bound market strategy |
| Ichimoku | `ichimoku_strategy.py` | Ichimoku Cloud strategy |
| Volume Profile | `volume_profile_strategy.py` | Volume-based analysis |
| Base | `base_strategy.py` | Abstract base class |

---

## Signal Flow (Live Trading)

```
1. trade_scan.py starts TradingSystem
   └── Creates TradingOrchestrator

2. TradingOrchestrator.scan_once() or start_continuous_scan()
   └── Calls IntelligentForexScanner.scan_once()

3. Scanner iterates over epic_list
   └── For each epic: SignalDetector.detect_combined_signals(epic, pair)

4. SignalDetector runs enabled strategies
   └── Each strategy: strategy.check_for_signals(df, epic, pair)
   └── Returns signal dict or None

5. Scanner collects and deduplicates signals
   └── AlertDeduplicationManager checks for recent duplicates

6. Orchestrator processes validated signals
   ├── TradeValidator validates trade parameters
   ├── RiskManager checks risk limits
   ├── IntegrationManager runs Claude analysis (if enabled)
   └── AlertHistoryManager saves to database

7. If trading enabled:
   └── OrderManager executes trade via IG Markets API
```

---

## Signal Flow (Backtesting)

```
1. bt.py wraps backtest_cli.py
   └── Parses shortcuts (EURUSD → CS.D.EURUSD.CEEM.IP)

2. backtest_cli.py creates BacktestScanner
   └── Uses BacktestDataFetcher (core/backtest_data_fetcher.py)

3. BacktestScanner iterates over historical data
   └── Simulates candle-by-candle progression

4. For each timestamp:
   └── SignalDetector.detect_combined_signals() with historical data

5. Signals tracked and evaluated
   └── Win/loss calculated based on SL/TP levels

6. Results aggregated
   └── Win rate, profit factor, total pips, etc.
```

---

## Key Configuration

### Main Config
**File**: `config.py` (65K+ lines)
- `EPIC_LIST` - Currency pairs to scan
- `MIN_CONFIDENCE` - Minimum signal confidence
- `SCAN_INTERVAL` - Seconds between scans
- `AUTO_TRADING_ENABLED` - Enable order execution
- `ENABLE_CLAUDE_ANALYSIS` - Enable Claude AI validation
- `INTELLIGENCE_PRESET` - Intelligence filtering level

### Strategy Configs
**Location**: `configdata/strategies/`
- `config_ema_strategy.py`
- `config_macd_strategy.py`
- `config_smc_strategy.py`
- etc.

### Trailing Stops
**File**: `config_trailing_stops.py`

---

## Directory Structure

```
worker/app/forex_scanner/
├── config.py                    # Main configuration
├── config_trailing_stops.py     # Trailing stop settings
├── bt.py                        # Quick backtest wrapper
├── backtest_cli.py              # Full backtest CLI
│
├── core/
│   ├── scanner.py               # IntelligentForexScanner
│   ├── signal_detector.py       # SignalDetector coordinator
│   ├── data_fetcher.py          # DataFetcher (candles + indicators)
│   ├── backtest_data_fetcher.py # Backtest-specific data fetcher
│   ├── backtest_scanner.py      # Backtest scanner
│   ├── database.py              # DatabaseManager
│   ├── memory_cache.py          # InMemoryForexCache
│   │
│   ├── strategies/              # Trading strategies
│   │   ├── base_strategy.py
│   │   ├── ema_strategy.py
│   │   ├── macd_strategy.py
│   │   ├── smc_structure_strategy.py
│   │   └── ... (12+ strategies)
│   │
│   ├── trading/                 # Trading orchestration
│   │   ├── trading_orchestrator.py
│   │   ├── order_manager.py
│   │   ├── risk_manager.py
│   │   ├── trade_validator.py
│   │   ├── session_manager.py
│   │   ├── integration_manager.py  # Claude AI
│   │   └── performance_tracker.py
│   │
│   ├── detection/               # Signal detection helpers
│   │   ├── price_adjuster.py
│   │   ├── large_candle_filter.py
│   │   └── enhanced_signal_validator.py
│   │
│   ├── intelligence/            # Market intelligence
│   │   ├── market_intelligence.py
│   │   └── intelligence_config.py
│   │
│   └── backtest/                # Backtest engine
│       ├── backtest_engine.py
│       ├── performance_analyzer.py
│       └── signal_analyzer.py
│
├── alerts/                      # Alert system
│   ├── alert_history.py         # AlertHistoryManager (database)
│   ├── order_executor.py        # IG Markets order execution
│   ├── claude_api.py            # Claude AI integration
│   ├── notifications.py         # Alert notifications
│   └── forex_chart_generator.py # Chart generation
│
├── configdata/                  # Modular configuration
│   ├── strategies/              # Per-strategy configs
│   └── dynamic_config_loader.py
│
├── optimization/                # Parameter optimization
│   ├── optimize_ema_parameters.py
│   ├── optimize_macd_parameters.py
│   └── optimal_parameter_service.py
│
└── commands/                    # CLI command modules (legacy)
    ├── scanner_commands.py
    ├── backtest_commands.py
    └── ...
```

---

## Common Tasks

### Add a New Strategy
1. Create `core/strategies/my_strategy.py` extending `BaseStrategy`
2. Add config in `configdata/strategies/config_my_strategy.py`
3. Register in `core/signal_detector.py`
4. Enable in `config.py`: `MY_STRATEGY_ENABLED = True`

### Debug Signal Detection
```bash
# Run single scan with verbose output
docker exec -it task-worker python /app/trade_scan.py scan

# Debug specific epic
docker exec -it task-worker python /app/trade_scan.py debug-data CS.D.EURUSD.CEEM.IP
```

### Check System Status
```bash
docker exec -it task-worker python /app/trade_scan.py status
```

### Force Scan (Ignore Market Hours)
```bash
docker exec -it task-worker python /app/trade_scan.py force-scan
```

---

## Key Classes Summary

| Class | File | Purpose |
|-------|------|---------|
| `TradingSystem` | `trade_scan.py` | Docker entry point wrapper |
| `TradingOrchestrator` | `core/trading/trading_orchestrator.py` | Coordinates all components |
| `IntelligentForexScanner` | `core/scanner.py` | Signal detection + deduplication |
| `SignalDetector` | `core/signal_detector.py` | Delegates to strategies |
| `DataFetcher` | `core/data_fetcher.py` | Candle data + indicators |
| `BacktestDataFetcher` | `core/backtest_data_fetcher.py` | Historical data for backtests |
| `AlertHistoryManager` | `alerts/alert_history.py` | Database operations |
| `OrderManager` | `core/trading/order_manager.py` | IG Markets order execution |
| `TradeValidator` | `core/trading/trade_validator.py` | Trade parameter validation |
| `RiskManager` | `core/trading/risk_manager.py` | Risk limit checking |
| `IntegrationManager` | `core/trading/integration_manager.py` | Claude AI analysis |
