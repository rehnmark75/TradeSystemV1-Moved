# Auto-Pause Layer

A per-`(strategy, pair)` safety layer that automatically flips a cell to
**monitor-only** when its rolling live performance decays, and (later phases)
auto-resumes it when shadow performance recovers — without affecting any other
strategy or pair.

> **Framing:** this is infrastructure to catch the **next decay nobody is
> watching**, not a fix for any one strategy. A strategy already under manual
> review gains ~nothing from auto-pause; the payoff is the proven strategy that
> quietly decays while attention is elsewhere.

## Why a pause layer (and why it's narrow)

Validation across the strategies that actually decayed established:

- The trip rule works for **medium/high-frequency strategies with a real,
  established edge**.
- On a **marginal/no-edge** strategy (PF ≈ 1.0) it fires ~constantly (~86%
  false positives) — "decay" is undefined without a baseline edge.
- On **low-frequency** strategies (< 8 trades/month) the rolling-PF window never
  fills in time, and auto-resume could take months (easy-to-pause /
  impossible-to-resume = a dead strategy).

So the layer is deliberately gated to **eligible cells only** (see Eligibility).

## Eligibility — the frozen baseline (critical)

A cell gets protection only if it has an `eligible = TRUE` row in
`auto_pause_eligibility` (strategy_config DB). The `baseline_pf / baseline_n /
baseline_as_of / monthly_trade_rate` columns record the **established
promotion-time edge** (target: PF > 1.2, n ≥ 50, > 8 trades/month).

**These baselines are FROZEN.** They are set by a human at promotion time and
are never recomputed from recent performance. If eligibility were based on a
trailing-recent window, a decaying strategy would drop below the edge bar and
**lose protection exactly when it needs it** — defeating the entire purpose.

Add an eligible cell:

```sql
INSERT INTO auto_pause_eligibility
  (strategy, epic, config_set, eligible, baseline_pf, baseline_n, baseline_as_of, monthly_trade_rate, notes)
VALUES
  ('RANGE_FADE', 'CS.D.EURUSD.CEEM.IP', 'demo', TRUE, 1.31, 85, '2026-05-31', 23, 'forward-validated Apr+May cohort')
ON CONFLICT (strategy, epic, config_set) DO UPDATE
  SET eligible = EXCLUDED.eligible, baseline_pf = EXCLUDED.baseline_pf,
      baseline_n = EXCLUDED.baseline_n, baseline_as_of = EXCLUDED.baseline_as_of,
      monthly_trade_rate = EXCLUDED.monthly_trade_rate, notes = EXCLUDED.notes,
      updated_at = now();
```

## Trip rule (fixed, first-principles — do not tune)

Pause a cell when **either**:

- rolling last-`N` closed-trade **PF < 0.8** AND **n ≥ 10**, OR
- **≥ 5 consecutive losses** (universal safeguard — the only trigger that fires
  in a useful timeframe for borderline-frequency cells).

Defaults live in `auto_pause/config.py` (`AutoPauseParams`) and may be overridden
via env vars (`AUTO_PAUSE_TRIP_PF`, `AUTO_PAUSE_MIN_TRADES`,
`AUTO_PAUSE_MAX_CONSEC_LOSSES`, …). They are validated for *generalization*; do
not re-tune them to maximise SEK on history — that reintroduces single-regime
overfit.

PF is computed from a **single P&L field per evaluation** — never a per-row
`coalesce` — so it can't mix pips and SEK into one ratio. `profit_loss` is used
in practice (`pips_gained` is NULL); pips are used only if no row carries
`profit_loss`. Rows missing the chosen field are skipped and logged.

## Per-strategy adapter (storage is not uniform)

The `monitor_only` flag is stored differently per strategy, so `auto_pause/
adapters.py` hides the differences behind a registry:

| Aspect | SMC_SIMPLE | IMPULSE_FADE | RANGE_FADE | others (active) | DONCHIAN_TURTLE |
|--------|-----------|--------------|-----------|------------------|-----------------|
| monitor_only | JSONB key | column | column | column | column |
| scope key | `config_id` | `config_id` | `config_set`+`profile_name` | `config_set` | epic only |

Environment → scope value: `config_id` 3=demo / 2=live; `config_set` =
`'demo'`/`'live'`. JSONB resume **removes** the `monitor_only` key (project
convention) rather than writing `false`. Table names come only from the trusted
registry, never user input.

## Where it runs (Phase 2 — implemented)

The check runs **inside the worker** in
`trading_orchestrator._run_periodic_maintenance()`, as
`_run_auto_pause_check_if_due()` mirroring `_run_outcome_analysis_if_due()`
(hourly, timestamp-tracked, exception-isolated so it can never disrupt
scanning). In-worker placement lets it call the strategy's
`refresh()` / `invalidate_cache()` **in-process** so a pause takes effect
immediately rather than waiting up to 5 min for the TTL (best-effort —
`refresh_config_cache()` falls back to the TTL on any error).

Each cycle: load eligible cells for **this orchestrator's `config_set`**
(`TRADING_CONFIG_SET`), and for each cell that is not already paused, evaluate
the trip rule and — only on the active→paused transition — flip `monitor_only`
via the adapter, refresh the cache, and log loudly. A trip whose flip matches
**0 rows** is logged at ERROR (eligibility exists but no pair_overrides row →
the pause silently did nothing).

- **Phase 2 is PAUSE-ONLY.** Resume is Phase 3; until then a paused cell stays
  paused until a human clears `monitor_only` (or Phase 3 ships).
- **Kill-switch:** `AUTO_PAUSE_ENABLED=false` disables the hook entirely
  (default on).
- **Activation:** the running scanner loads the hook on the next
  `docker restart task-worker` (and `task-worker-live`). It is inert until at
  least one `auto_pause_eligibility` row exists, so enabling it changes nothing
  until a cell is opted in.

**Coexistence with `adaptive_bucket_gate`:** the bucket gate is the *tactical*
intra-cell layer (in-memory, per-signal, epic+direction). Auto-pause is the
*strategic* whole-cell layer (persistent `monitor_only`). They never write to
the same place; when a cell is paused the signal short-circuits before the
bucket gate, so auto-pause dominates while paused and the bucket gate resumes
its normal role when the cell is active.

## Resume (Phase 3 — Phase A implemented, propose-only)

Paused cells keep logging monitor-only signals to `alert_history`, but those
rows are unreliable for outcomes (price = 0 ~60% of the time; no SL/TP stamped).
So shadow P&L is **reconstructed** (`auto_pause/shadow.py`): `alert_timestamp +
epic → entry from ig_candles (1m) → apply the cell's fixed SL/TP → walk forward
to HIT_TP / HIT_SL` (a single candle spanning both resolves conservatively to
the stop). Pause lifecycle (when paused, eval/proposal tracking) lives in
`auto_pause_state`.

Resume rule R1 (`auto_pause/resume.py`): ≥ 15 fresh reconstructed outcomes,
shadow PF > 1.1 (hysteresis gap above the 0.8 trip), ≥ 10-day cooldown.

**Rollout is per-cell**, controlled by the `auto_resume` flag on the cell's
`auto_pause_eligibility` row:

1. **Propose-only** (`auto_resume = FALSE`, default): a met R1 logs
   `🔔 RESUME PROPOSED` and records it in `auto_pause_state`, but does NOT
   re-enable — a human clears `monitor_only`.
2. **Fully-auto** (`auto_resume = TRUE`): a met R1 flips `monitor_only` back off
   itself, refreshes the cache, and records `state='resumed'` + `resumed_at`
   (logs `✅ AUTO-RESUMED`). Enable per-cell once its proposals are trusted —
   **start on demo.** (The one-click-confirm middle step was skipped per request.)

```sql
-- graduate a cell to fully-auto resume
UPDATE auto_pause_eligibility SET auto_resume = TRUE
WHERE strategy='RANGE_FADE' AND epic='CS.D.EURUSD.CEEM.IP' AND config_set='demo';
```

Auto-pause moves *toward* safety (monitor-only, no capital at risk) → fully-auto
from day one. Resume moves *toward* risk → `auto_resume` is opt-in per cell.
**Even with `auto_resume=TRUE`, R1's reconstruction-fidelity caveats below still
apply** — the metric is conservative/approximate, so treat demo auto-resume as a
validation of the loop, not yet a trusted live mechanism.

### Reconstruction fidelity — known limitations (must close before Phase C)

The shadow metric is a deliberate **approximation**, fine for propose-only
observation but NOT yet trustworthy enough to auto-resume on:

- **Counts signals, not trade-equivalents.** A paused cell logs every signal,
  including ones live cooldown / LPF / validation would drop (divergence
  2–13×). Shadow counts therefore run optimistic on *volume*.
- **Fixed SL/TP, not live trailing/BE.** Live exits trail and break-even; the
  reconstruction uses fixed pips. So small live "wins" (trailing exits below TP)
  reconstruct as losses → shadow PF is *conservative* (e.g. RANGE_FADE EURUSD
  over its decay reconstructs ~0 PF, correctly declining resume). Good as an
  edge *proxy*, but not calibrated to live P&L.
- **ATR-based strategies are skipped** (no fixed SL/TP to read) — they log
  nothing and never auto-propose.

Phase B/C must validate that proposals correlate with genuine recovery before
resume is trusted.

## Trip Rule B — shadow ref-grid series (Jul 2026)

`trade_log` proved far too sparse for Rule A on most cells (≤3 closed trades
per cell / 30d), so a second, per-cell-selectable trip source was added:

- **Data**: `monitor_only_outcomes.ref_pnl_pips` (forex DB) — the fixed
  reference-grid outcome (FX 10/15 pips, gold 80/160) that
  `monitor_outcome_analyzer` computes for **every** logged signal (~40–170 per
  cell / 30d). Since Jul 2026 the analyzer also evaluates EXECUTED signals and
  stamps `was_executed`, so the series is the complete signal population.
- **Rule** (`shadow_source.py::decide_trip_shadow`, window = last 50 RESOLVED):
  trip when `shadow PF < 0.8` **AND** `shadow WR < baseline_shadow_wr − 12pp`
  (n ≥ 30), **or** ≥ 8 consecutive shadow losses. Both legs required because
  the ref grid is near-binary (PF ≈ 1.5·w/(1−w)); σ(WR) ≈ 7pp at n = 50.
- **Baselines**: `baseline_shadow_pf/wr/n` on `auto_pause_eligibility` — frozen
  at enrollment (same doctrine as `baseline_pf`). Enroll via
  `scripts/auto_pause_enroll.py` (computes from the OLDEST outcomes in the
  window, or accepts manual 90d-backtest values; warns when baseline WR is
  below the 40% ref-grid breakeven).
- **Selection**: `auto_pause_eligibility.trip_source` = `'trades' | 'shadow' |
  'both'`. Shadow-tripped cells also **resume** on the ref-grid series (post-
  pause outcomes only) so trip/resume hysteresis is on one metric; the resume
  adds a WR leg (within 5pp of baseline).
- **⚠️ The ref-grid PF is a decay PROXY, not live P&L** — a trailing-exit
  strategy can be live-profitable while its fixed-bracket proxy sits below 1.0.
  Cells with sub-breakeven baselines get weak (under-sensitive) protection:
  effectively only the consecutive-loss safeguard.
- **Dry-run**: `AUTO_PAUSE_DRY_RUN` (default **true**) — trips are logged +
  recorded to `auto_pause_events` but nothing is flipped. Flip to `false` on
  demo after ~a week of sane dry-run events.
- **Events + Telegram**: every decision → `auto_pause_events` (strategy_config
  DB, 24h dedupe for recurring dry-run trips / resume proposals); system-monitor
  polls unnotified rows → Telegram. Dashboard: trading-ui `/forex/decay-monitor`.
- **Replay acceptance** (`scripts/auto_pause_replay.py`): Rule B trips
  RANGE_FADE EURUSD on **2026-06-10** — three weeks before Rule A's Jul 2 trip;
  healthy cells (MEAN_REVERSION EURJPY PF 1.76, XAU_GOLD current PF 1.70)
  produce zero spurious trips. Do NOT grid-search thresholds against replays.

## Phase status

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Eligibility table + migration, adapter registry, PF/streak evaluator + trip rule, eligibility loader, tests, docs | **Done** |
| 2 | Orchestrator maintenance hook: evaluate eligible cells → flip `monitor_only` → in-process cache refresh (pause-only) | **Done** (needs `docker restart task-worker` to load) |
| 3 | Shadow-P&L reconstruction + resume proposals (propose-only) | **Done** |
| C | Fully-auto resume (per-cell `auto_resume` flag) | **Done** (one-click-confirm step skipped) |
| B (shadow) | Trip Rule B on `monitor_only_outcomes` ref-grid series + events + Telegram + decay dashboard; 9 cells enrolled dry-run | **Done** (Jul 2026, dry-run) |
| — | Reconstruction-fidelity validation (signals-vs-trades, fixed-vs-trailing SL/TP) | Deferred — required before trusting fully-auto beyond demo |

## Tests

```bash
docker exec task-worker python /app/forex_scanner/tests/test_auto_pause.py
# or: docker exec task-worker python -m pytest /app/forex_scanner/tests/test_auto_pause.py
```

Pure functions (scope clauses, monitor_only SQL, PF/streak, trip decision) are
unit-tested; the DB IO layer is validated by a rolled-back integration round-trip.
