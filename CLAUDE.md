# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🐳 CRITICAL: Docker Environment

**ALL commands must run inside Docker containers. Never run Python/SQL directly on host.**

```bash
# Python scripts
docker exec -it task-worker python /app/forex_scanner/script.py

# Database queries
docker exec -it postgres psql -U postgres -d trading -c "SELECT ..."

# Interactive shell
docker exec -it task-worker bash
```

**Path mapping**: `worker/app/` → `/app/` inside container

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
docker exec -it task-worker python /app/forex_scanner/bt.py GBPUSD 14 MACD --show-signals
```
**Pair shortcuts**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD, EURJPY, AUDJPY, GBPJPY
**Strategy shortcuts**: EMA, MACD, BB, SMC, SMC_STRUCTURE, MOMENTUM, ICHIMOKU, KAMA, ZEROLAG

---

## 🏗️ Core Architecture

### Signal Flow (Live Trading)
```
trade_scan.py (entry point)
  └── TradingOrchestrator (core/trading/trading_orchestrator.py)
        ├── IntelligentForexScanner (core/scanner.py) - signal detection + dedup
        │     └── SignalDetector (core/signal_detector.py) - delegates to strategies
        │           └── Strategies (core/strategies/*.py)
        ├── DataFetcher (core/data_fetcher.py) - candles + indicators
        ├── TradeValidator, RiskManager, OrderManager
        ├── IntegrationManager - Claude AI analysis
        └── AlertHistoryManager (alerts/alert_history.py) - database
```

### Key Files Quick Reference

| Purpose | File | Container |
|---------|------|-----------|
| **Live scanner entry** | `worker/app/trade_scan.py` | task-worker |
| **Backtest entry** | `worker/app/forex_scanner/bt.py` | task-worker |
| **Strategy config** | `worker/app/forex_scanner/config.py` | task-worker |
| **Orchestrator** | `worker/app/forex_scanner/core/trading/trading_orchestrator.py` | task-worker |
| **Scanner** | `worker/app/forex_scanner/core/scanner.py` | task-worker |
| **Signal detector** | `worker/app/forex_scanner/core/signal_detector.py` | task-worker |
| **Strategies** | `worker/app/forex_scanner/core/strategies/*.py` | task-worker |
| **Order executor** | `worker/app/forex_scanner/alerts/order_executor.py` | task-worker |
| **TRAILING STOPS (LIVE)** | `dev-app/config.py` → `PAIR_TRAILING_CONFIGS` | **fastapi-dev** |
| **Trailing stops (backtest)** | `worker/app/forex_scanner/config_trailing_stops.py` | task-worker |
| **Trade monitoring** | `dev-app/trailing_class.py` | fastapi-dev |

### Strategies Available
| Strategy | File | Enable in config.py |
|----------|------|---------------------|
| EMA | `ema_strategy.py` | `EMA_STRATEGY_ENABLED` |
| MACD | `macd_strategy.py` | `MACD_STRATEGY_ENABLED` |
| SMC Structure | `smc_structure_strategy.py` | `SMC_STRUCTURE_STRATEGY_ENABLED` |
| Bollinger+Supertrend | `bb_supertrend_strategy.py` | `BB_SUPERTREND_STRATEGY_ENABLED` |
| Scalping | `scalping_strategy.py` | `SCALPING_STRATEGY_ENABLED` |
| Momentum | `momentum_strategy.py` | `MOMENTUM_STRATEGY_ENABLED` |
| KAMA | `kama_strategy.py` | `KAMA_STRATEGY` |
| Zero Lag | `zero_lag_strategy.py` | `ZERO_LAG_STRATEGY_ENABLED` |
| Ichimoku | `ichimoku_strategy.py` | `ICHIMOKU_STRATEGY_ENABLED` |

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
| Development Best Practices | `claude-development.md` |

---

## ⚠️ Important Notes

- **Docker Required**: All forex scanner commands must be run inside Docker containers
- **Database-Driven**: The system uses dynamic, optimized parameters from PostgreSQL
- **Modular Design**: New strategies follow lightweight configuration patterns
- **Real-time Intelligence**: Market analysis and trade context evaluation available

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
2. Restart `fastapi-dev`: `docker restart fastapi-dev`
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

**Recent Enhancements:**
- ✅ Dynamic parameter optimization system (database-driven)
- ✅ Market intelligence with regime detection
- ✅ Modular strategy configuration architecture
- ✅ Comprehensive documentation split
- ✅ Progressive trailing stop system with 4 stages
- ✅ Partial close at 13 pips (configurable per pair)
- ✅ Absolute stop level updates (not offset-based)

For detailed setup and usage instructions, start with the [Overview & Navigation](claude-overview.md).