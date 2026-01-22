# GEMINI.md

This file provides guidance to the Gemini CLI agent when working with code in this repository.

## 🐳 CRITICAL: Docker Environment

**ALL commands must run inside Docker containers. Never run Python/SQL directly on host.**

```bash
# Python scripts
docker exec -it task-worker python /app/forex_scanner/script.py

# Database queries
docker exec postgres psql -U postgres -d forex -c "SELECT ..."    # Forex candles/trades
docker exec postgres psql -U postgres -d stocks -c "SELECT ..."   # Stock candles

# Interactive shell
docker exec -it task-worker bash
```

**Path mapping**: `worker/app/` → `/app/` inside container

### Docker Compose Commands (CRITICAL)

**ALWAYS use `docker compose` (v2) NOT `docker-compose` (v1):**

```bash
# ✅ CORRECT - Use docker compose (v2, space-separated)
docker compose up -d stock-scheduler
docker compose restart task-worker
docker compose logs -f fastapi-dev

# ❌ WRONG - Never use docker-compose (v1, hyphenated)
docker-compose up -d  # OLD VERSION - causes ContainerConfig errors
```

**Safe container operations (avoid disrupting other services):**

```bash
# Restart a single container (safest)
docker restart stock-scheduler

# Recreate single container with new config (use --no-deps!)
docker compose up -d --no-deps --force-recreate stock-scheduler

# View logs
docker compose logs -f --tail 100 stock-scheduler

# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}"
```

**⚠️ NEVER run these without `--no-deps` flag:**
```bash
# DANGEROUS - may recreate dependent containers like postgres!
docker compose up -d stock-scheduler  # Without --no-deps

# SAFE - only affects the specified container
docker compose up -d --no-deps stock-scheduler
```

---

## 🚀 Entry Points

### Live Scanner
```bash
docker exec -it task-worker python /app/trade_scan.py              # Docker mode (default)
docker exec -it task-worker python /app/trade_scan.py scan         # Single scan
docker exec -it task-worker python /app/trade_scan.py live 120     # Live trading
docker exec -it task-worker python /app/trade_scan.py status       # System status
```

### Backtesting
```bash
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 7              # 7 days
docker exec -it task-worker python /app/forex_scanner/bt.py GBPUSD 14 SMC --show-signals
```
**Pair shortcuts**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD, EURJPY, AUDJPY, GBPJPY
**Strategy shortcuts**: SMC, SMC_SIMPLE (only active strategy after January 2026 cleanup)

### CRITICAL: Backtest --timeframe vs Strategy Timeframes (Jan 2026 Discovery)

The `--timeframe` parameter controls **scan interval** (how often backtest evaluates), NOT strategy timeframes:

```bash
# Scan every 5 minutes (recommended for live comparison)
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 7 --scalp --timeframe 5m

# Scan every 15 minutes (default - misses mid-candle signals)
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 7 --scalp --timeframe 15m
```

**Strategy timeframes are ALWAYS (regardless of --timeframe setting):**
| Tier | Timeframe | Purpose |
|------|-----------|---------|
| TIER 1 HTF | 15m | EMA bias/direction |
| TIER 2 Trigger | 5m | Swing break detection |
| TIER 3 Entry | 1m | Pullback entry |

**Why this matters:** Live scanner runs every 2-5 minutes. Using `--timeframe 15m` only evaluates at 15m boundaries, missing signals that occur mid-candle. Jan 15 comparison showed:
- `--timeframe 15m`: 20 signals
- `--timeframe 5m`: 60 signals
- Live (2 min interval): 56 signals

**Recommendation:** Use `--timeframe 5m` for accurate live vs backtest comparison.

### Backtest Parameter Isolation (January 2026)
Test strategy parameters during backtesting WITHOUT affecting live trading:

```bash
# Phase 1: In-memory parameter overrides
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 14 \
    --override fixed_stop_loss_pips=10 --override min_confidence=0.55

# Phase 2: Persistent config snapshots
docker exec -it task-worker python /app/forex_scanner/snapshot_cli.py create tight_sl \
    --set fixed_stop_loss_pips=8 --set min_confidence=0.6 --desc "Tighter SL test"
docker exec -it task-worker python /app/forex_scanner/snapshot_cli.py list
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 14 --snapshot tight_sl

# Phase 3: Historical intelligence replay (enabled by default)
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 14 --no-historical-intelligence
```

See **Backtest Parameter Isolation System** section below for full documentation.

---

## 🏗️ Core Architecture

### Signal Flow (Live Trading)
```
trade_scan.py (entry point)
  └── TradingOrchestrator (core/trading/trading_orchestrator.py)
        ├── IntelligentForexScanner (core/scanner.py) - signal detection + dedup
        │     └── SignalDetector (core/signal_detector.py) - delegates to strategies
        │           └── StrategyRegistry → SMCSimpleStrategy (only active)
        ├── DataFetcher (core/data_fetcher.py) - candles + indicators
        ├── TradeValidator, RiskManager, OrderManager
        ├── IntegrationManager - Claude AI analysis
        └── AlertHistoryManager (alerts/alert_history.py) - database
```

**Note:** After January 2026 cleanup, only SMC Simple strategy is active. Legacy strategies (EMA, MACD, etc.) are archived in `forex_scanner/archive/disabled_strategies/`.

### Key Files Quick Reference

| Purpose | File | Container |
|---------|------|-----------|
| **Live scanner entry** | `worker/app/trade_scan.py` | task-worker |
| **Backtest entry** | `worker/app/forex_scanner/bt.py` | task-worker |
| **Infrastructure config** | `worker/app/forex_scanner/config.py` | task-worker |
| **SMC Config Service** | `worker/app/forex_scanner/services/smc_simple_config_service.py` | task-worker |
| **Strategy Registry** | `worker/app/forex_scanner/core/strategies/strategy_registry.py` | task-worker |
| **Orchestrator** | `worker/app/forex_scanner/core/trading/trading_orchestrator.py` | task-worker |
| **Scanner** | `worker/app/forex_scanner/core/scanner.py` | task-worker |
| **Signal detector** | `worker/app/forex_scanner/core/signal_detector.py` | task-worker |
| **SMC Simple Strategy** | `worker/app/forex_scanner/core/strategies/smc_simple_strategy.py` | task-worker |
| **Strategy templates** | `worker/app/forex_scanner/core/strategies/templates/` | task-worker |
| **Adding new strategies** | `worker/app/forex_scanner/docs/adding_new_strategy.md` | task-worker |
| **Order executor** | `worker/app/forex_scanner/alerts/order_executor.py` | task-worker |
| **TRAILING STOPS (LIVE)** | `dev-app/config.py` → `PAIR_TRAILING_CONFIGS` | **fastapi-dev** |
| **Trailing stops (backtest)** | `worker/app/forex_scanner/config_trailing_stops.py` | task-worker |
| **Trade monitoring** | `dev-app/trailing_class.py` | fastapi-dev |

### Strategy System (January 2026 Cleanup)

**Active Strategy:**
| Strategy | File | Enable |
|----------|------|--------|
| SMC Simple | `smc_simple_strategy.py` | `SMC_SIMPLE_STRATEGY = True` (default) |

**Adding New Strategies:**
1. Copy template from `core/strategies/templates/strategy_template.py`
2. Implement strategy logic following `StrategyInterface`
3. Create database migration using `migrations/templates/strategy_config_template.sql`
4. Enable in database or config.py
5. See `docs/adding_new_strategy.md` for detailed guide

**Archived Strategies** (in `forex_scanner/archive/disabled_strategies/`):
- EMA, MACD, SMC Structure, Bollinger+Supertrend, Scalping
- Momentum, KAMA, Zero Lag, Ichimoku, Mean Reversion
- Volume Profile, Silver Bullet, etc. (16 total)

---

## 📊 Candle Data Flow

```
IG Markets API (Lightstreamer) → ig_candles table (5m base)
                                       ↓
                               DataFetcher resamples to 15m/1h/4h
                                       ↓
                               Adds technical indicators (EMA, MACD, RSI, etc.)
                                       ↓
                               Strategy analyzes enhanced DataFrame
```

**Database tables**: `ig_candles`, `preferred_forex_prices`
**Resampling**: 5m → 15m (3 candles), 5m → 1h (12 candles), 5m → 4h (48 candles)

---

## 📋 Extended Documentation

For detailed information, see these docs (read with Read tool when needed):

| Topic | File |
|-------|------|
| Commands & CLI | `claude-commands.md` |
| Full Architecture | `claude-architecture.md` |
| Strategy Development | `claude-strategies.md` |
| Parameter Optimization | `claude-optimization.md` |
| Market Intelligence | `claude-intelligence.md` |
| Trailing Stop System | `claude-trailing-system.md` |
| Virtual Stop Loss (Scalping) | `claude-vsl-system.md` |
| Development Best Practices | `claude-development.md` |
| Configuration System | `claude-configuration.md` |

---

## ⚠️ Important Notes

- **Docker Required**: All forex scanner commands must be run inside Docker containers
- **Database-Driven**: The system uses dynamic, optimized parameters from PostgreSQL
- **Modular Design**: New strategies follow lightweight configuration patterns
- **Real-time Intelligence**: Market analysis and trade context evaluation available

---

## 🗄️ Database-Driven Configuration (SMC Simple Strategy)

**Source of Truth**: `strategy_config` database (NOT config files!)

The SMC Simple strategy reads ALL configuration from the database, including per-pair overrides.

### Database Tables (in `strategy_config` database):
| Table | Purpose |
|-------|---------|
| `smc_simple_global_config` | Global strategy parameters (~80 settings) |
| `smc_simple_pair_overrides` | Per-pair parameter overrides (SL/TP, confidence, etc.) |
| `smc_simple_config_audit` | Change history for auditing |
| `smc_simple_parameter_metadata` | UI metadata for parameter display |

### Key Per-Pair Settings:
- `fixed_stop_loss_pips` - Per-pair stop loss override
- `fixed_take_profit_pips` - Per-pair take profit override
- `min_confidence` - Per-pair minimum confidence threshold
- `max_confidence` - Per-pair maximum confidence cap
- `sl_buffer_pips` - Per-pair SL buffer
- `macd_filter_enabled` - Per-pair MACD filter toggle

### Config Service:
```python
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config

config = get_smc_simple_config()  # Loads from database with caching
sl = config.get_pair_fixed_stop_loss('CS.D.EURUSD.CEEM.IP')  # Per-pair or global fallback
tp = config.get_pair_fixed_take_profit('CS.D.EURUSD.CEEM.IP')
```

### Updating Configuration:
```bash
# Set global SL/TP defaults
docker exec postgres psql -U postgres -d strategy_config -c "
UPDATE smc_simple_global_config
SET fixed_stop_loss_pips = 9, fixed_take_profit_pips = 15
WHERE is_active = TRUE;"

# Set per-pair override
docker exec postgres psql -U postgres -d strategy_config -c "
UPDATE smc_simple_pair_overrides
SET fixed_stop_loss_pips = 12, fixed_take_profit_pips = 20
WHERE epic = 'CS.D.USDJPY.MINI.IP';"

# View current per-pair settings
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT epic, fixed_stop_loss_pips, fixed_take_profit_pips, min_confidence
FROM smc_simple_pair_overrides ORDER BY epic;"
```

### Migrations:
Located in `worker/app/forex_scanner/migrations/`:
- `create_strategy_config_db.sql` - Initial schema
- `add_fixed_sl_tp_columns.sql` - Per-pair SL/TP support
- `add_max_confidence_to_pair_overrides.sql` - Confidence cap

**DO NOT** edit `worker/app/forex_scanner/config.py` or `configdata/strategies/config_smc_simple.py` for SMC Simple settings - these are LEGACY fallbacks only!

---

## ⚠️ Scanner Config Service (CRITICAL)

**File**: `worker/app/forex_scanner/services/scanner_config_service.py`

The `ScannerConfigService` loads scanner settings from the `scanner_global_config` table.

### IMPORTANT: Adding New Database Fields

When adding new fields to `scanner_global_config` table, you MUST also add them to the `direct_fields` list in `_build_config_from_row()` method (~line 565), otherwise:
- The field will NOT be loaded from the database
- The dataclass default value will be used instead
- **This caused a major bug in Jan 2026** where `data_batch_size=10000` (default) was used instead of the database value `25000`, causing "Insufficient 4h data" errors

### Checklist for New Scanner Config Fields:
1. ✅ Add column to `scanner_global_config` table
2. ✅ Add field to `ScannerConfig` dataclass (with default)
3. ✅ **Add field name to `direct_fields` list** (easy to forget!)
4. ✅ If integer, add to `int_fields` set
5. ✅ If float, add to `float_fields` set
6. ✅ Restart container: `docker restart task-worker` (NOT `docker compose up`)

### Key Performance Fields (in `scanner_global_config`):
| Field | Purpose | Default |
|-------|---------|---------|
| `data_batch_size` | Max rows fetched for 1m synthesis | 25000 |
| `reduced_lookback_hours` | Enable lookback reduction | true |
| `lookback_reduction_factor` | Reduction multiplier | 0.7 |
| `use_1m_base_synthesis` | Use 1m candles for resampling | true |

**Note**: For 4H data with 1m synthesis, need ~14,400+ 1m candles (60 bars × 240). Set `data_batch_size >= 25000` to account for weekend gaps.

---

## 🚨 CRITICAL: Container & Config Ownership

**NEVER confuse these containers - they have DIFFERENT config files!**

| Container | Purpose | Config Location | Owns |
|-----------|---------|-----------------|------|
| **fastapi-dev** | Live trade execution, trailing stops, breakeven | `dev-app/config.py` | **TRAILING STOPS (source of truth)** |
| **task-worker** | Strategy scanning, backtesting, signal generation | `worker/app/forex_scanner/config.py` | Strategies, backtesting |
| **streamlit** | Analytics dashboard, breakeven optimizer | Reads from fastapi-dev via mount | Display only |

### Trailing Stop Configuration

**Source of Truth**: `dev-app/config.py` → `PAIR_TRAILING_CONFIGS`

This file controls:
- `break_even_trigger_points` - When to move SL to breakeven
- `early_breakeven_trigger_points` - Early BE trigger
- `stage1/2/3_trigger_points` - Progressive trailing stages
- `min_trail_distance` - Minimum trailing distance

**DO NOT** edit `worker/app/forex_scanner/config_trailing_stops.py` for live trading changes - that file is for backtesting only!

### When updating trailing stops:
1. Edit `dev-app/config.py` (the ONLY source of truth)
2. Restart `fastapi-dev`: `docker restart fastapi-dev` (NOT `docker compose up`)
3. Verify: `docker exec fastapi-dev python3 -c "from config import PAIR_TRAILING_CONFIGS; print(PAIR_TRAILING_CONFIGS['CS.D.EURUSD.CEEM.IP'])"`

### Streamlit reads trailing config via docker-compose mount:
```yaml
- ./dev-app/config.py:/app/trailing_config.py:ro
```

## 🤖 Agent Configuration

**Automatic Agent Usage:**

- **trading-strategy-analyst**: Automatically use this agent for any task involving:
  - Strategy performance analysis, backtest result evaluation, trading strategy optimization
  - Win rate, profit/loss analysis, strategy parameter tuning, market regime performance assessment
  - Keywords: strategy, backtest, performance, win rate, profit, loss, optimization, trading, analysis, momentum, RSI, MACD, EMA, SMA, bollinger, stochastic, parameters, signals, entry, exit, stop loss, take profit
  - File patterns: */strategies/*, */backtests/*, *backtest*.py, *strategy*.py, *config_*.py

- **devops-engineer**: Automatically use this agent for any task involving:
  - Docker orchestration, CI/CD pipelines, infrastructure automation, production deployment
  - System monitoring, alerting, performance optimization, infrastructure scaling
  - Keywords: docker, kubernetes, deployment, infrastructure, monitoring, CI/CD, pipeline, container, scaling, production, devops, automation
  - File patterns: docker-compose*.yml, Dockerfile*, k8s/*, .github/workflows/*, .gitlab-ci.yml

- **quantitative-researcher**: Automatically use this agent for any task involving:
  - Mathematical modeling, statistical analysis, hypothesis testing, risk modeling
  - Research methodology, alternative data analysis, factor modeling, portfolio optimization
  - Keywords: statistical, mathematical, model, research, hypothesis, risk, factor, correlation, regression, optimization, quantitative, econometric, simulation
  - File patterns: */research/*, */models/*, *analysis*.py, *research*.py, *model*.py

- **real-time-systems-engineer**: Automatically use this agent for any task involving:
  - Ultra-low latency optimization, high-frequency processing, concurrent programming
  - Real-time data processing, performance optimization, memory management
  - Keywords: latency, real-time, concurrent, performance, optimization, threading, memory, high-frequency, microsecond, lock-free
  - File patterns: */real_time/*, */concurrent/*, */performance/*, *latency*.py, *concurrent*.py

- **financial-data-engineer**: Automatically use this agent for any task involving:
  - Market data feeds, tick data processing, financial data normalization, order book management
  - Financial database optimization, time series data, market microstructure
  - Keywords: market data, tick data, financial, forex, currency, price, quote, trade, order book, time series, OHLCV
  - File patterns: */market_data/*, */feeds/*, */financial/*, *market*.py, *price*.py, *forex*.py

## 📈 System Status

**January 2026 Cleanup Complete:**
- ✅ Archived 16 disabled strategies (preserved in `archive/disabled_strategies/`)
- ✅ Archived 67 unused helper modules (preserved in `archive/disabled_helpers/`)
- ✅ Cleaned signal_detector.py (3,605 → 630 lines, 83% reduction)
- ✅ Cleaned config.py (1,413 → 733 lines, 48% reduction)
- ✅ Implemented Strategy Registry pattern for easy extensibility
- ✅ Created strategy and migration templates
- ✅ Only SMC Simple strategy is active (database-driven configuration)

**System Features:**
- ✅ Dynamic parameter optimization system (database-driven)
- ✅ Market intelligence with regime detection
- ✅ Progressive trailing stop system with 4 stages
- ✅ Claude AI trade analysis integration
- ✅ Smart Money Concepts (SMC) analysis
- ✅ Virtual Stop Loss for scalping mode (bypasses IG min SL restrictions)

For detailed setup and usage instructions, start with the [Overview & Navigation](claude-overview.md).
