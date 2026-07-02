# Edge-Map Router — Integration Notes (FOUNDATION layer)

This layer is **data + analyzer only**. It ships tagging, a learned edge-map
table, and a nightly analyzer. It does **not** wire into emission, the scheduler,
or `route.ts` — those three integration points are described below so the owning
agents can add the calls without re-deriving anything. **Nothing here changes
live or demo execution until step (c) is deliberately taken (and even then,
shadow-mode first).**

Files shipped by this layer (all NEW, no existing file edited):

| File | Role |
|------|------|
| `migrations/043_scanner_cell_edge_router.sql` | cell columns on `stock_scanner_signals` + `scanner_cell_edge` table |
| `core/routing/cell_tagger.py` | `classify_cell(...)` (pure) + `tag_signal_row(...)` (as-of DB) |
| `analysis/scanner_cell_edge_analyzer.py` | nightly edge-map compute + upsert; `run()` importable |
| `scripts/backfill_signal_cells.py` | one-shot/repeatable causal backfill of cell columns |

Character cell = `trend_state` (ADX) × `vol_regime` (ATR%) × `liquidity_tier`
(relative volume), with `cell_market_regime` snapshotted as-of from
`market_context`. Thresholds are the named constants in `cell_tagger.py` and are
mirrored in the migration + backfill SQL (the backfill self-verifies parity via
`--verify`).

> **Naming note:** `stock_scanner_signals` already had a scanner-written
> `market_regime` column, so migration 043 does **not** touch it. The router's
> own causal regime snapshot lives in the new `cell_market_regime` column. Don't
> confuse the two.

> **NaN gotcha (already handled):** `stock_screening_metrics.adx` / `atr_percent`
> can be Postgres numeric `NaN`. Postgres sorts `NaN` **greater than every real
> number**, so a naive `adx >= 25` bins a missing metric as `trend`. The backfill
> coerces `NULLIF(x,'NaN')` and the Python classifier maps `NaN -> None`; both
> agree (verified 0/500 mismatches). Any new SQL that bins these metrics must do
> the same.

---

## (a) Emission: tag each signal at save time  — **DO NOT let me edit this file**

**Call site:** `scanners/base_scanner.py :: BaseScanner.save_signals()` (the
`INSERT INTO stock_scanner_signals ...` around line 515), which is driven by
`scanners/scanner_manager.py :: ScannerManager._save_all_signals()` (~line 724).
Both are asyncpg (`$1` placeholders, `self.db.fetch`).

Two ways to populate the four cell columns; pick one:

**Option A (recommended, lowest-risk, reuses validated SQL).** Leave the INSERT
untouched and run a scoped, set-based causal tag immediately **after**
`_save_all_signals()` returns, tagging only today's just-saved rows:

```python
# after: total_saved = await self._save_all_signals(signals)
from stock_scanner.scripts.backfill_signal_cells import backfill
# only-missing keeps it idempotent; scope by running right after the daily save.
backfill(only_missing=True)          # sync psycopg2; run via asyncio.to_thread(...)
```

Because `backfill(only_missing=True)` only touches rows where `trend_state IS
NULL`, it is cheap and safe to call every pipeline run. Run it in a threadpool
(`await asyncio.to_thread(backfill, only_missing=True)`) so the sync psycopg2
call doesn't block the event loop.

**Option B (inline, if you want the values on the INSERT row itself).** Add
`trend_state, vol_regime, liquidity_tier, cell_market_regime` to the INSERT
column list and compute them per signal. Since the DB layer is asyncpg but
`tag_signal_row` is sync psycopg2, either (i) call `tag_signal_row` via
`asyncio.to_thread` with a short-lived psycopg2 connection, or (ii) port its two
`SELECT ... calculation_date <= signal_date ... LIMIT 1` lookups to asyncpg. The
cell dict keys map to columns as: `market_regime -> cell_market_regime`, the
other three keys map 1:1.

Either way: tagging is **failure-isolated** (`tag_signal_row` never raises;
`backfill` is a single UPDATE). A missing metric yields NULL cell fields and must
never block a save.

---

## (b) Scheduler: run the analyzer nightly, **after** outcome tracking

**Call site:** `scheduler.py :: run_pipeline()` — right after
`status_updates = await self.performance_tracker.update_signal_statuses()`
(~line 720), which is what closes signals and writes `realized_pnl_pct`. The
analyzer must run **after** outcomes are written **and after** the separate
contamination-flagging agent's step (so `status IN ('invalid','data_error')` and
any blow-up rows are already marked). Add:

```python
from stock_scanner.analysis.scanner_cell_edge_analyzer import run as run_cell_edge
# after update_signal_statuses() and after outcome-contamination flagging:
summary = await asyncio.to_thread(run_cell_edge, window_days=120)
logger.info("cell-edge map refreshed: %s", summary)
```

`run()` opens its own psycopg2 connection (or accepts `conn=`), computes both the
2-axis and 3-axis grids, upserts `scanner_cell_edge`, and prunes cells that fell
out of the rolling window. It is idempotent — safe to re-run. Standalone form for
cron/manual: `python -m stock_scanner.analysis.scanner_cell_edge_analyzer
--window-days 120`.

The analyzer's clean-outcome filter is self-contained and does **not** depend on
the flagging agent existing yet: it always excludes `ABS(realized_pnl_pct) > 100`
and guards `status NOT IN ('invalid','data_error')` whether or not those statuses
have been introduced.

---

## (c) route.ts: consume the edge-map to gate tradability — **SHADOW MODE FIRST**

**Call site:** `trading-ui/app/api/signals/top/route.ts` (the same file that
holds `MONITOR_ONLY_SCANNERS` / `DEFAULT_ENABLED_SCANNERS`). **Do not implement
this yet.** When the map has matured, the consumer:

1. For each candidate signal, read its stored cell (`trend_state`, `vol_regime`,
   and optionally `liquidity_tier`) directly off the signal row — no recompute.
2. Look up `scanner_cell_edge` for `(scanner_name, trend_state, vol_regime[,
   liquidity_tier])`. Start on the **2-axis** grid (`liquidity_tier IS NULL`);
   only move to the 3-axis grid once it shows real added dispersion with enough
   `n` per sub-cell.
3. Map `verdict`: `block` → would-drop, `monitor` → would-hold-out-of-pool,
   `trade` → allowed, `insufficient`/missing → **fail-open** (treat as allowed;
   never let an unknown cell silently kill a scanner).
4. **Shadow mode:** for the first forward window, only **log** the intended
   routing decision alongside the current behavior (`intended_action`,
   `cell_key`, `verdict`, `pf`, `n`) and change nothing about what actually
   trades. Compare logged intent vs realized outcomes before flipping any cell to
   an enforced drop.

SQL the consumer would run (2-axis, fail-open):

```sql
SELECT verdict, pf, n
FROM scanner_cell_edge
WHERE scanner_name = $1 AND trend_state = $2 AND vol_regime = $3
  AND liquidity_tier IS NULL AND market_regime IS NULL;
-- no row OR verdict IN ('insufficient') -> ALLOW (fail-open)
```

Guardrails already baked into the analyzer that the router relies on: a cell is
`insufficient` (never actionable) unless `n >= 30` **and** `calendar_days >= 21`,
which kills single-regime few-week artifacts (e.g. `pocket_pivot` at n=198 but
cd=14 stays `insufficient`).

---

## Verification performed at build time

- Migration applied cleanly (idempotent).
- Backfill tagged **10,162** rows; **9,981** taggable on 2-axis and 3-axis;
  `cell_market_regime` on 10,013. Python-vs-SQL parity: **0/500** mismatches
  after the NaN fix.
- Analyzer (120d) produced 111 two-axis + 249 three-axis cells; clean outcomes
  7,065, excluded (dirty) 145.
- **Clean-outcome filter proven:** `zlma_trend / trend / high` has raw
  outcomes=320 with 87 blow-ups (`|pnl%|>100`); stored `scanner_cell_edge.n=233`
  = exactly the clean count. The contaminated rows are excluded from `n`, `pf`,
  `win_rate`, and `avg_pnl_pct`.
