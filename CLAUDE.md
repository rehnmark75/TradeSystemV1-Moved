# CLAUDE.md

Guidance for Claude Code working in this repository. Deep detail lives in
`claude-deep-reference.md` and the topic docs (see **Extended Documentation**);
read those on demand instead of bloating this always-loaded file.

## 🐳 CRITICAL: Docker Environment

**ALL commands run inside Docker containers. Never run Python/SQL directly on host.**

```bash
docker exec -it task-worker python /app/forex_scanner/script.py     # Python
docker exec postgres psql -U postgres -d forex -c "SELECT ..."      # Forex candles/trades
docker exec postgres psql -U postgres -d stocks -c "SELECT ..."     # Stock candles
docker exec postgres psql -U postgres -d strategy_config -c "..."   # Strategy/trailing config
```
**Path mapping**: `worker/app/` → `/app/` inside container.

**ALWAYS use `docker compose` (v2, space) NOT `docker-compose` (v1, hyphen — causes ContainerConfig errors).**
- Safest restart: `docker restart <container>`
- Recreate one container: `docker compose up -d --no-deps --force-recreate <container>`
- **⚠️ NEVER `docker compose up -d <container>` without `--no-deps`** — may recreate dependents like postgres.

---

## 🚀 Entry Points

```bash
# Live scanner (task-worker)
docker exec -it task-worker python /app/trade_scan.py [scan|live 120|status]

# Backtest (task-worker)
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 7
docker exec -it task-worker python /app/forex_scanner/bt.py GBPUSD 14 SMC --show-signals
```
**Pairs**: EURUSD GBPUSD USDJPY AUDUSD USDCHF USDCAD NZDUSD EURJPY AUDJPY GBPJPY
**Strategies**: SMC, SMC_SIMPLE (active), SMC_MOMENTUM, XAU_GOLD

⚠️ **`--timeframe` = scan interval, NOT strategy timeframe.** Use `--timeframe 5m` for live-parity comparison (default 15m misses mid-candle signals). Param isolation via `--override` / `--snapshot`. Full detail: `claude-deep-reference.md`.

---

## 🏗️ Core Architecture

```
trade_scan.py → TradingOrchestrator (core/trading/trading_orchestrator.py)
  ├── IntelligentForexScanner (core/scanner.py) — detection + dedup
  │     └── SignalDetector (core/signal_detector.py) → StrategyRegistry → strategies
  ├── DataFetcher (core/data_fetcher.py) — candles + indicators
  ├── TradeValidator, RiskManager, OrderManager
  ├── IntegrationManager — Claude AI analysis
  └── AlertHistoryManager (alerts/alert_history.py) — database
```
Candle flow: IG Lightstreamer → `ig_candles` (5m base) → DataFetcher resamples 15m/1h/4h + indicators → strategy. Long-lookback backtests use `ig_candles_backtest` (Dukascopy). See `claude-deep-reference.md`.

### Key Files

| Purpose | File (under `worker/app/` unless noted) | Container |
|---------|------|-----------|
| Live scanner entry | `trade_scan.py` | task-worker |
| Backtest entry | `forex_scanner/bt.py` | task-worker |
| Infrastructure config | `forex_scanner/config.py` | task-worker |
| SMC Config Service | `forex_scanner/services/smc_simple_config_service.py` | task-worker |
| Strategy Registry | `forex_scanner/core/strategies/strategy_registry.py` | task-worker |
| Orchestrator | `forex_scanner/core/trading/trading_orchestrator.py` | task-worker |
| Scanner / Signal detector | `forex_scanner/core/scanner.py` / `core/signal_detector.py` | task-worker |
| SMC Simple Strategy | `forex_scanner/core/strategies/smc_simple_strategy.py` | task-worker |
| Strategy templates / guide | `forex_scanner/core/strategies/templates/` · `docs/adding_new_strategy.md` | task-worker |
| Order executor | `forex_scanner/alerts/order_executor.py` | task-worker |
| **Trailing stops (LIVE)** | `trailing_pair_config` table (`strategy_config` DB) via `dev-app/services/trailing_config_service.py` | **fastapi-dev** |
| Trailing stops (backtest) | `forex_scanner/config_trailing_stops.py` | task-worker |
| Trade monitoring | `dev-app/trailing_class.py` | fastapi-dev |

### Active Strategies
| Strategy | File | Instrument | Enable |
|----------|------|------------|--------|
| SMC Simple | `smc_simple_strategy.py` | FX majors/crosses | `SMC_SIMPLE_STRATEGY = True` (default) |
| SMC Momentum | `smc_momentum_strategy.py` | FX majors/crosses | `smc_momentum_pair_overrides.is_enabled` |
| XAU Gold | `xau_gold_strategy.py` | Gold (`CS.D.CFEGOLD.CEE.IP`) | `xau_gold_pair_overrides.is_enabled` |

Legacy strategies (EMA, MACD, etc.) archived in `forex_scanner/archive/disabled_strategies/`.
**Adding a strategy:** copy `templates/strategy_template.py`, follow `StrategyInterface`, add a migration from `migrations/templates/strategy_config_template.sql`, enable in DB. Guide: `docs/adding_new_strategy.md`.

Strategy/instrument deep detail (XAU Gold 3-tier logic, regime/session filters, scalp trailing): **`claude-deep-reference.md`**.

---

## 🗄️ Database-Driven Configuration

**Source of Truth for SMC Simple config is the `strategy_config` database — NOT config files.**

⛔ **NEVER read `configdata/strategies/config_smc_simple.py` for current values — it is COMPLETELY DEPRECATED and contains STALE/WRONG data** (e.g. `ENABLED_PAIRS`). Always query the DB or the trading-ui Settings page.

```bash
# Enabled pairs (actual source of truth)
docker exec postgres psql -U postgres -d strategy_config -c "SELECT epic, is_enabled FROM smc_simple_pair_overrides WHERE is_enabled = TRUE;"
```
Per-pair settings: `fixed_stop_loss_pips`, `fixed_take_profit_pips`, `min_confidence`, `max_confidence`, `sl_buffer_pips`, `macd_filter_enabled`. Load via `get_smc_simple_config()`. Full tables, update examples, migrations: `claude-deep-reference.md`.

### ⚠️ Scanner Config Service gotcha
When adding a field to `scanner_global_config`, you MUST add the field name to the `direct_fields` list in `scanner_config_service.py::_build_config_from_row()` — else the DB value is silently ignored and the dataclass default is used (caused the Jan 2026 `data_batch_size` "Insufficient 4h data" bug). Full checklist: `claude-deep-reference.md`.

---

## 🚨 CRITICAL: Container & Config Ownership

**NEVER confuse these containers — they own DIFFERENT config files:**

| Container | Purpose | Config |
|-----------|---------|--------|
| **fastapi-dev** | Live execution, trailing stops, breakeven | `dev-app/config.py` (flags) + `trailing_pair_config` DB (trailing VALUES) |
| **task-worker** | Strategy scanning, backtesting, signal generation | `worker/app/forex_scanner/config.py` |
| **streamlit** | Analytics dashboard (display only) | reads from fastapi-dev via mount |

**Trailing stops (source of truth, DB-backed since Apr 2026)**: `trailing_pair_config` table in `strategy_config` DB. The `dev-app/config.py` `PAIR_TRAILING_CONFIGS`/`SCALP_TRAILING_CONFIGS` dicts are **legacy/seed only — the live runtime no longer reads them.** **DO NOT** edit `worker/app/forex_scanner/config_trailing_stops.py` for live changes (backtest only). Rows scoped by `config_set`+`is_scalp`+`strategy`+`epic`; update the DB row then `docker restart fastapi-dev`. Full columns & examples: `claude-deep-reference.md`.

---

## 🤖 Agent Configuration

**MANDATORY: delegate to the matching specialist agent BEFORE responding** — for questions, analysis, AND implementation. Do not answer inline then suggest an agent. Spawn multiple in parallel if several apply.

**Agents that write SQL must `Read` `.claude/agents/db-expert.md` first** — single source of truth for table schemas, columns, domain constants, pair config. Don't duplicate schema knowledge inline.

| Agent | Use for |
|-------|---------|
| **db-expert** | ad-hoc queries, schema, data exploration/debugging (alert_history, trade_log, *_pair_overrides, loss_prevention_rules, trailing_pair_config) |
| **trading-strategy-analyst** | strategy performance, backtest evaluation, parameter tuning, win/PF analysis, regime |
| **backtest-ablation-engineer** | running backtests, ablation/gate isolation, permissive baselines, live-vs-backtest, signals/month |
| **devops-engineer** | docker, CI/CD, infra automation, deployment, monitoring, scaling |
| **quantitative-researcher** | statistical/mathematical modeling, hypothesis testing, risk/factor models |
| **financial-data-engineer** | market/tick data feeds, normalization, order book, time series, OHLCV |
| **lpf-engineer** | Loss Prevention Filter rules — add/tune/audit, why a trade was blocked, coverage |
| **trailing-stop-engineer** | trailing stages, breakeven, lock points, partial close, live-vs-backtest parity |
| **strategy-lifecycle-manager** | promote monitor-only → demo → live, promotion gates, DB wiring, pair enablement |
| **rejection-signal-analyst** | over-blocking analysis across *_rejections / loss_prevention_decisions tables |
| **trade-outcome-analyst** | trade outcomes, win/loss patterns, alert↔trade correlation |

---

## ⚠️ Monitor-Only Pairs

Signals logged but NOT traded (check `parameter_overrides->>'monitor_only'`):
- **USDCHF** — breakeven (0.99 PF), filters degrade.
- **AUDUSD** — disabled on live, monitor-only on demo; under-filtered config floods ~445 BUY/quarter at PF 0.50. Do NOT flip to traded without re-tuning to peer gates + OOS validation.

The only actively-traded SMC_SIMPLE pair on demo is currently **EURUSD**. Full status + SQL: `claude-deep-reference.md`.

---

## 📋 Extended Documentation

Read with the Read tool when needed:

| Topic | File |
|-------|------|
| **Deep reference (relocated detail)** | `claude-deep-reference.md` |
| Overview & navigation | `claude-overview.md` |
| Commands & CLI | `claude-commands.md` |
| Full architecture | `claude-architecture.md` |
| Strategy development | `claude-strategies.md` |
| Parameter optimization | `claude-optimization.md` |
| Market intelligence | `claude-intelligence.md` |
| Trailing stop system | `claude-trailing-system.md` |
| Configuration system | `claude-configuration.md` |
| Development best practices | `claude-development.md` |

> The system is Docker-required, database-driven (PostgreSQL), and modular. `claude-vsl-system.md` is DEPRECATED (Jan 2026 — replaced by scalp trailing configs).
