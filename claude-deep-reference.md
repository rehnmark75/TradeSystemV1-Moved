# Claude Deep Reference

Detail relocated out of `CLAUDE.md` to keep the always-loaded file lean.
Read this on demand when working on the relevant subsystem.

---

## Backtest: `--timeframe` vs Strategy Timeframes (Jan 2026)

The `--timeframe` parameter controls **scan interval** (how often the backtest evaluates), NOT strategy timeframes.

```bash
# Scan every 5 minutes (recommended for live comparison)
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 7 --scalp --timeframe 5m

# Scan every 15 minutes (default - misses mid-candle signals)
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 7 --scalp --timeframe 15m
```

**Scalp mode strategy timeframes (regardless of `--timeframe`):**
| Tier | Timeframe | Purpose |
|------|-----------|---------|
| TIER 1 HTF | 1h (default) | EMA bias/direction |
| TIER 2 Trigger | 5m | Swing break detection |
| TIER 3 Entry | 1m | Pullback entry |

GBPUSD and NZDUSD override HTF to 15m — see `smc_simple_pair_overrides.scalp_htf_timeframe`.

**Why it matters:** live scanner runs every 2-5 min. `--timeframe 15m` only evaluates at 15m boundaries, missing mid-candle signals. Jan 15 comparison: 15m→20 signals, 5m→60 signals, live (2 min)→56 signals. **Use `--timeframe 5m` for accurate live vs backtest comparison.**

---

## Backtest Parameter Isolation (Jan 2026)

Test strategy parameters during backtesting WITHOUT affecting live trading.

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

---

## Candle Data Flow

```
IG Markets API (Lightstreamer) → ig_candles table (5m base)
                                       ↓
                               DataFetcher resamples to 15m/1h/4h
                                       ↓
                               Adds indicators (EMA, MACD, RSI, etc.)
                                       ↓
                               Strategy analyzes enhanced DataFrame
```

**Tables**: `ig_candles`, `preferred_forex_prices`
**Resampling**: 5m → 15m (3 candles), 5m → 1h (12), 5m → 4h (48)

### Historical Backfill: `ig_candles_backtest` (Dukascopy, excluded from backups)

Multi-year backtests use **Dukascopy Bank's** public feed to populate `ig_candles_backtest` with 1m bars from 2020 onward, then self-resample within the same table to 5m/15m/1h/4h. Dukascopy ≠ IG pricing (small spread delta on FX, meaningful on metals — gold lands as `CS.D.CFEGOLD.DUKAS.IP` to avoid polluting the live IG series).

**Key choice**: `ig_candles_backtest` is **excluded from Azure backups** (`--exclude-table-data=public.ig_candles_backtest` in `scripts/enhanced_backup.sh` + `scripts/azure_backup.sh`). Schema is still dumped (restore recreates the empty table + indexes); only data is skipped. Saves ~8-10 GB per backup.

**Restore after DR:**
```bash
# 1. Restore forex DB from backup (schema only for ig_candles_backtest)
# 2. Re-populate from Dukascopy:
python3 -m venv ~/.venvs/dukas
~/.venvs/dukas/bin/pip install dukascopy-python pandas
~/.venvs/dukas/bin/python scripts/dukascopy_download.py \
    --start 2020-01-01 --end 2025-09-17 --output-dir /tmp/dukas/
./scripts/dukascopy_push_local.sh /tmp/dukas/
```
Runtime: ~3-5 h download + ~15 min load/resample. See `scripts/dukascopy_download.py --list-epics`.

---

## XAU Gold Strategy (XAU_GOLD) — Apr 2026

Gold-specific 3-tier SMC strategy for `CS.D.CFEGOLD.CEE.IP`. Runs alongside SMC_SIMPLE (FX pairs → SMC_SIMPLE; gold routed automatically via `_is_gold_epic()` in `signal_detector.py`).

**3-Tier Signal Logic** (fires only when all 3 align; scanner evaluates every 5 min):
| Tier | Timeframe | Role |
|------|-----------|------|
| TIER 1 HTF | 4H | EMA(50/200) bias + structure (HH/HL vs LH/LL) |
| TIER 2 Trigger | 1H | BOS / CHOCH + MACD confirmation |
| TIER 3 Entry | 15m | Pullback to OB / FVG / 50% fib |

**Regime Filter:**
- `TRENDING` (ADX > 25, ATR pct > 40): trend-follow entries — primary edge
- `RANGING` (ADX < 20): blocked (gold ranges are whippy)
- `EXPANSION` (ATR pct > 80): blocked (news spikes, wide spreads)

**Session Filter (UTC):** London 07-10 + NY 13-20 primary; Asian 23-06 continuations only; Rollover 21-22 blocked.

**Key Files:**
| File | Purpose |
|------|---------|
| `core/strategies/xau_gold_strategy.py` | Strategy implementation |
| `services/xau_gold_config_service.py` | DB-backed config (5-min TTL cache) |
| `migrations/create_xau_gold_config.sql` | DB schema |

**Configuration (`strategy_config` DB):**
```bash
docker exec postgres psql -U postgres -d strategy_config -c "SELECT epic, is_enabled, is_traded, monitor_only FROM xau_gold_pair_overrides;"
docker exec postgres psql -U postgres -d strategy_config -c "SELECT * FROM xau_gold_global_config WHERE is_active = true;"
# Enable / disable trading:
UPDATE xau_gold_pair_overrides SET monitor_only = false, is_traded = true WHERE epic = 'CS.D.CFEGOLD.CEE.IP';
UPDATE xau_gold_pair_overrides SET monitor_only = true WHERE epic = 'CS.D.CFEGOLD.CEE.IP';
```

**Instrument Notes:**
- **Pip size**: 0.1 (1 point = $0.1). `get_point_value()` returns `0.1` for CFEGOLD.
- **Order size**: 0.1 lots (`dev-app/config.py` → `EPIC_ORDER_SIZES`)
- **Worker epic map**: `CS.D.CFEGOLD.CEE.IP` → `CFEGOLD.1.CEE` → full epic in dev-app
- **Min deal size**: 0.1 (confirmed via IG demo API, status `TRADEABLE`)

**Trailing Stops (gold-specific, `strategy_config.trailing_pair_config`, all pips, 1 pip = $0.1):**
| Stage | Trigger | Action |
|-------|---------|--------|
| Break-even | +30 pips | Move SL to entry |
| Stage 1 | +50 pips | Lock +25 pips |
| Stage 2 | +80 pips | Lock +50 pips |
| Stage 3 | +110 pips | ATR trail (1.5×), min 30 pip distance |

**Backtest:**
```bash
docker exec -it task-worker python /app/forex_scanner/bt.py CS.D.CFEGOLD.CEE.IP 90 XAU_GOLD --timeframe 5m --show-signals
```
75-day baseline (Apr 2026): 164 signals, PF 15.43 (avg win 160 pips, avg loss 20). High PF reflects fixed SL/TP defaults (80/160 pips); expect lower with tighter ATR-based stops.

**Claude Vision:** `XAU_GOLD` is in `scanner_global_config.claude_vision_strategies`. Chart TFs `["4h","1h","15m"]` align with the 3 tiers.

---

## Scalp Mode Trailing System (Jan 2026)

**Source of truth**: `trailing_pair_config` DB rows where `is_scalp=true` (in `strategy_config` DB, read via `dev-app/services/trailing_config_service.py`). The `SCALP_TRAILING_CONFIGS` dict in `dev-app/config.py` is **legacy/seed-only — the live runtime no longer reads it** (DB-backed since Apr 2026).

**Background:** the Virtual Stop Loss (VSL) system was deprecated Jan 2026 after analysis showed 67% premature stops (5-6 pips too tight, $2,506 loss over 2 days), early profit locks (83% locked at BE, 46% capture), and that optimal stops (12-20 pips) sit ABOVE IG's 10-pip minimum — making VSL unnecessary. Scalp trades now use **IG native stops** with progressive trailing, keyed by the `is_scalp_trade` flag.

**Key Files:**
- `trailing_pair_config` DB table (`is_scalp=true`) — scalp stops (12-20 pips, data-backed)
- `dev-app/services/trailing_config_service.py` — loads trailing rows from DB (5-min cache)
- `dev-app/enhanced_trade_processor.py` — dynamic config selection via `get_config_for_trade()`
- `dev-app/config_virtual_stop.py` — VSL disabled (`VIRTUAL_STOP_LOSS_ENABLED = False`)

**Per-Pair Optimal Stops (2-day analysis):**
| Pair | Initial Stop | BE Trigger | Stage1 | Stage2 | Partial Close | Success |
|------|-------------|-----------|---------|---------|---------------|---------|
| **USDCAD** | 12 pips | +6 | +10 → lock +5 | +12 → lock +8 | 50% @ +10 | 100% |
| **Majors** | 15 pips | +8 | +12 → lock +6 | +15 → lock +10 | 50% @ +12 | 67% |
| **JPY Pairs** | 20 pips | +10 | +15 → lock +8 | +20 → lock +12 | 50% @ +15 | 50% |

**Progressive stages:** Early BE (+6-10 → lock +1-1.5) → Stage 1 (+10-15 → lock +5-8) → Stage 2 (+12-20 → lock +8-12) → Stage 3 (+15-25 → ATR trail 1.5×).

**Flow:** scalp flag set in `orders_router.py` (`is_scalp = body.is_scalp_trade or (sl_limit and sl_limit <= 8)`) → `get_config_for_trade()` loads scalp vs regular config by flag → `trade_monitor.py` → `enhanced_trade_processor.py` applies per-trade.

**Verify:**
```bash
docker logs -f fastapi-dev | grep -E "(⚡|SCALP CONFIG|is_scalp_trade)"
```

---

## Database-Driven Config (SMC Simple) — full detail

**Source of Truth**: `strategy_config` database (NOT config files). SMC Simple reads ALL config from the DB, including per-pair overrides.

**Tables (`strategy_config` DB):**
| Table | Purpose |
|-------|---------|
| `smc_simple_global_config` | Global params (~80 settings) |
| `smc_simple_pair_overrides` | Per-pair overrides (SL/TP, confidence, etc.) |
| `smc_simple_config_audit` | Change history |
| `smc_simple_parameter_metadata` | UI metadata |

**Key per-pair settings:** `fixed_stop_loss_pips`, `fixed_take_profit_pips`, `min_confidence`, `max_confidence`, `sl_buffer_pips`, `macd_filter_enabled`.

**Config service:**
```python
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
config = get_smc_simple_config()  # DB-backed with caching
sl = config.get_pair_fixed_stop_loss('CS.D.EURUSD.CEEM.IP')  # per-pair or global fallback
tp = config.get_pair_fixed_take_profit('CS.D.EURUSD.CEEM.IP')
```

**Updating:**
```bash
# Global SL/TP defaults
docker exec postgres psql -U postgres -d strategy_config -c "
UPDATE smc_simple_global_config SET fixed_stop_loss_pips = 9, fixed_take_profit_pips = 15 WHERE is_active = TRUE;"
# Per-pair override
docker exec postgres psql -U postgres -d strategy_config -c "
UPDATE smc_simple_pair_overrides SET fixed_stop_loss_pips = 12, fixed_take_profit_pips = 20 WHERE epic = 'CS.D.USDJPY.MINI.IP';"
# View
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT epic, fixed_stop_loss_pips, fixed_take_profit_pips, min_confidence FROM smc_simple_pair_overrides ORDER BY epic;"
```

**Migrations** (`worker/app/forex_scanner/migrations/`): `create_strategy_config_db.sql`, `add_fixed_sl_tp_columns.sql`, `add_max_confidence_to_pair_overrides.sql`.

---

## Scanner Config Service (CRITICAL)

**File**: `worker/app/forex_scanner/services/scanner_config_service.py`. Loads scanner settings from `scanner_global_config`.

**Adding new DB fields — you MUST add the field name to the `direct_fields` list in `_build_config_from_row()` (~line 565)**, otherwise the field is NOT loaded and the dataclass default is used instead. This caused a Jan 2026 bug where `data_batch_size=10000` (default) was used instead of DB value `25000`, causing "Insufficient 4h data" errors.

**Checklist:** (1) add column to table; (2) add field to `ScannerConfig` dataclass with default; (3) **add field name to `direct_fields`**; (4) if int → `int_fields`; (5) if float → `float_fields`; (6) `docker restart task-worker`.

**Key performance fields:**
| Field | Purpose | Default |
|-------|---------|---------|
| `data_batch_size` | Max rows for 1m synthesis | 25000 |
| `reduced_lookback_hours` | Enable lookback reduction | true |
| `lookback_reduction_factor` | Reduction multiplier | 0.7 |
| `use_1m_base_synthesis` | Use 1m candles for resampling | true |

For 4H data with 1m synthesis, need ~14,400+ 1m candles (60 bars × 240). Set `data_batch_size >= 25000` for weekend gaps.

---

## Trailing Stop Configuration — full detail

**Source of Truth (DB-backed since Apr 2026)**: `trailing_pair_config` table in `strategy_config` DB. The `dev-app/config.py` `PAIR_TRAILING_CONFIGS` / `SCALP_TRAILING_CONFIGS` dicts are **legacy/seed only — the live runtime no longer reads them.** `get_trailing_config_for_epic()` delegates to `dev-app/services/trailing_config_service.py` (DB, 5-min cache).

Rows scoped by `config_set` ('live'/'demo'), `is_scalp` (true/false), `strategy` (`SMC_SIMPLE`/`XAU_GOLD`/`DEFAULT`), `epic`. Lookup falls back: pair+strategy → `DEFAULT` → empty. Key columns: `break_even_trigger_points`/`early_breakeven_trigger_points`, `stage1/2/3_trigger_points` + matching `*_lock_points`, `stage3_atr_multiplier`/`stage3_min_distance`, `min_trail_distance`, `enable_partial_close`/`partial_close_trigger_points`/`partial_close_size`.

**DO NOT** edit `worker/app/forex_scanner/config_trailing_stops.py` for live changes — backtest only.

**Updating:**
```bash
docker exec postgres psql -U postgres -d strategy_config -c "
UPDATE trailing_pair_config SET break_even_trigger_points = 12
WHERE config_set='demo' AND epic='CS.D.EURUSD.CEEM.IP' AND is_scalp=true AND strategy='SMC_SIMPLE';"
```
Cache (5-min TTL) is invalidated on write via `trailing_config_router`; to force immediately `docker restart fastapi-dev`. Verify:
```bash
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT config_set, epic, is_scalp, strategy, break_even_trigger_points, stage1_trigger_points
FROM trailing_pair_config WHERE epic='CS.D.EURUSD.CEEM.IP' ORDER BY config_set, is_scalp;"
```

> ⚠️ The streamlit `./dev-app/config.py:/app/trailing_config.py:ro` mount predates the DB migration — for trailing it reflects only the legacy seed dicts. Read trailing values from the DB.

---

## Monitor-Only Pairs — full detail (Feb–Jun 2026)

Signals logged but NOT traded. Check via `parameter_overrides->>'monitor_only'`.

- **USDCHF**: breakeven (0.99 PF), all filters degrade.
- **AUDUSD** (corrected Jun 1, 2026): **disabled on live** (`config_id=2`, `is_enabled=f`) and **monitor-only on demo** (`config_id=3`, `is_enabled=t`, `monitor_only=true`) — not traded anywhere. (Prior "re-enabled Mar 14" claim is stale per the DB.) Its scalp config is deliberately stripped of the entry-quality gates peers carry (loose `scalp_entry_rsi_buy_max=70` vs 44–55, no `scalp_min_adx` floor, no impulse/MFI filters), so in a trend it fires a distinct BUY every ~15 min (cooldown floor) → ~445 signals/quarter at PF 0.50. Harmless while monitor-only (only spams `alert_history`), but do NOT flip to traded without re-tuning to peer gates AND OOS-validating. Overlaying EURUSD's gate set collapses it 445→70 at PF 1.10 (n=75, marginal). The only actively-traded SMC_SIMPLE pair on demo is currently **EURUSD**.

```sql
-- Check monitor-only status
SELECT epic, parameter_overrides->>'monitor_only' FROM smc_simple_pair_overrides
WHERE parameter_overrides->>'monitor_only' = 'true';
-- Re-enable trading (remove flag)
UPDATE smc_simple_pair_overrides SET parameter_overrides = parameter_overrides - 'monitor_only'
WHERE epic = 'CS.D.AUDUSD.MINI.IP';
```

---

## System Status / Cleanup History (Jan 2026)

Archived 16 disabled strategies (`archive/disabled_strategies/`) + 67 unused helper modules (`archive/disabled_helpers/`). Cleaned `signal_detector.py` (3,605 → 630 lines) and `config.py` (1,413 → 733). Implemented Strategy Registry pattern + strategy/migration templates. SMC Simple + SMC Momentum active (DB-driven). VSL deprecated, replaced by scalp-specific trailing configs.
