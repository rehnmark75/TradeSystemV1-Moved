# SMC_SIMPLE Over-Filtering Review — May 6, 2026

## TL;DR

The strategy has **31 distinct filter points across 4 layers**. Three genuine over-filtering problems and three implementation bugs were found. The biggest structural issue is that EURUSD R:R (0.8:1) means no filter change will produce positive expectancy at 45% WR — that must be fixed first.

---

## Signal Funnel (90 days)

| Stage | Count | Monthly Rate |
|---|---|---|
| Evaluated by LPF | ~2,505 | ~835/month |
| Post-LPF (in alert_history) | 721 | **~240/month** |
| Vision analyzed | 394 (54.6%) | ~131/month |
| Vision approved | 117 (16.2%) | ~39/month |

**LPF block rate: 71%** (1,784 of ~2,505 signals blocked over 90 days).

---

## Confirmed Over-Filtering

### 1. Ranging Market Block — LIKELY OVER-FILTERING
**Evidence:** Ranging-regime signals that reached trade_log show **68.2% WR, avg +14.43 P&L** (n=22 closed trades). This is the highest avg P&L of any regime — higher than trending (60.3%, -4.17).

- Pairs with `scalp_block_ranging_market=true`: USDJPY, USDCAD (plus JSONB variants on others)
- Estimated suppression: ~17 ranging signals/month
- The block was added for MACD-based strategies, not necessarily SMC scalp mode

**Verification:** Run 90d BT on EURJPY + USDJPY with `--override scalp_block_ranging_market=false`. Accept if n≥30, WR≥58%, PF≥1.20.

---

### 2. smc_high_conf_block — CORRECT INTENT, BROKEN IMPLEMENTATION
**Evidence:** Confidence band 0.65–0.69 shows **44.4% WR, -25.90 avg P&L** (worst in dataset, n=54). The rule correctly targets this band (penalty=1.00). But boost rules (asian_session -0.10, trending_aligned -0.20, sweet_spot_conf -0.15 = **-0.45 total**) cancel the penalty, allowing 78% of qualifying signals through.

**This is not over-filtering — it's under-filtering for high-conf signals.** The rule is failing to block what it's supposed to block.

**Fix:** Make smc_high_conf_block an unconditional hard block (separate code path, not additive penalty), OR raise penalty to 2.00 so boosts cannot cancel it.

---

### 3. Vision Coverage Gap — 45% of Signals Bypass Quality Gate
**Evidence:** 327 of 721 post-LPF signals (45.4%) have null `claude_decision` — never analyzed by vision. Not a "block" but the quality gate isn't applied uniformly.

**Verification:** Check `claude_analyzer.py` — identify the code path where signals skip vision. Determine if these are auto-approved or auto-rejected.

---

## Implementation Bugs

### Bug 1: AUDUSD min_confidence > max_confidence
- `min_confidence=0.650`, `max_confidence=0.640` — impossible band
- Every signal gets clamped to 0.640 then rejected by the 0.650 floor
- **Bug also exists in JSONB `parameter_overrides`**, so fixing only the DB column won't work
- Currently masked because AUDUSD is monitor_only. Will break 100% if re-enabled.
- **Fix:** Set both DB column and JSONB to `min=0.55, max=0.75` (audit all pairs for same issue)

### Bug 2: Confidence Threshold Stacking (6 independent adjustors, no logging)
The final effective confidence floor combines: DB min → EMA-distance adj → ATR-volatility adj → volume adj → HTF-bias multiplier → rolling_perf delta. Combined effect can raise the floor 8–15pp above the DB value with no observable log line.

- **Fix:** Add one log line showing `effective_floor` vs `signal_confidence` before the gate check

### Bug 3: EURJPY/NZDUSD Backtest-Live Divergence (12x gap)
- EURJPY: 49 live signals (30d) vs 4 BT signals — **12x gap**
- NZDUSD: 17 live signals vs 1 BT signal
- Lowering `min_confidence` has zero effect on BT count → confidence is not the cause
- Likely cause: HTF 1h data availability in `ig_candles_backtest` for JPY crosses, or regime calibration difference
- **Impact:** Cannot validate EURJPY/NZDUSD using backtests — creating a blind spot in the validation framework

### Bug 4: Vision Fail-Secure Blocks Infrastructure Failures
`claude_fail_secure=True` blocks trades on ANY chart generation failure (MinIO error, chart-gen timeout) — not just Claude API errors. Valid signals blocked by infra failures are silent losses.

---

## Structural Issues (Not Over-Filtering, But Must Fix First)

### EURUSD R:R Math Failure
- SL=8 pips, TP=10 pips = **0.8:1 R:R**
- Break-even WR at 0.8:1 requires **55.6%**. Actual WR is **45%**.
- No amount of filter tuning produces positive expectancy with this setup.
- **Fix:** Raise TP to 14–16 pips (1.75–2.0:1 R:R requires only 36–40% WR to break even). Backtest EURUSD 90d with `--override fixed_take_profit_pips=14` before applying.

### GBPUSD Self-Selection Into Anti-Predictive Confidence Band
- Avg confidence: 0.693 — highest of all pairs, sitting squarely in the 0.65–0.69 inverse zone
- WR: 37.5% (30d BT), PF: 0.72
- MACD filter removal makes it worse (+9 signals, WR drops to 36.4%, PF 0.62)
- Recommendation: move to monitor_only until confidence band issue resolved

### Regime Triple-Counting
Same regime label is evaluated at three independent layers: (1) signal_detector.py downgrades trending→ranging via efficiency_ratio, (2) strategy applies `scalp_block_ranging_market`, (3) LPF has regime-conditional rules. No layer knows what the others did.

---

## Filter Rankings: Most Over-Filtering → Clearly Justified

| Rank | Filter | Verdict | Monthly Impact |
|---|---|---|---|
| 1 | Ranging market block | LIKELY OVER-FILTERING | ~17 suppressed, 68.2% WR in live |
| 2 | smc_high_conf_block (implementation) | BUG — under-enforced | 78% of high-conf signals leak through |
| 3 | Vision coverage gap | SYSTEMIC GAP | 327/721 signals untested (45%) |
| 4 | LPF stacking at threshold=0.60 | NEEDS VALIDATION | ~200/month at borderline |
| 5 | Confidence stacking (6 adjustors) | POSSIBLE HIDDEN FILTER | 8–15pp above configured floor |
| 6 | Cooldown system (4 independent timers) | POSSIBLE REDUNDANCY | Unquantified |
| 7 | Sweep protection + vision double-scoring | REDUNDANCY | Unquantified |
| 8 | Breakout regime block | CLEARLY JUSTIFIED | 40.9% WR, -23.07 avg P&L |
| 9 | smc_high_conf_block (intent) | CLEARLY JUSTIFIED | 44.4% WR, -25.90 avg P&L |
| 10 | sell_near_support (LPF #32) | CLEARLY JUSTIFIED | 15.8% WR, -$916 |
| 11 | MACD filter (GBPUSD) | JUSTIFIED | Removal makes results worse |

---

## Verification & Mitigation Plan

### Phase 1 — Fix Structural Issues Before Expanding Signals (Week 1)

| # | Action | File / Command | Risk |
|---|--------|----------------|------|
| 1.1 | Fix AUDUSD min/max conf inversion (both DB + JSONB) | DB update + JSONB patch | Low |
| 1.2 | Add `effective_floor` log line in confidence gate | `smc_simple_strategy.py` | Low |
| 1.3 | Fix smc_high_conf_block: raise penalty to 2.00 | `strategy_config.loss_prevention_rules` | Medium |
| 1.4 | Diagnose EURJPY/NZDUSD BT gap (run `--show-signals`, check HTF data depth) | `bt.py EURJPY 30 --scalp --show-signals` | Low |
| 1.5 | Move GBPUSD to monitor_only | DB update | Low |

### Phase 2 — Fix EURUSD R:R (Week 1-2)

| # | Action | Backtest Gate Before Applying |
|---|--------|-------------------------------|
| 2.1 | Backtest EURUSD 90d with `--override fixed_take_profit_pips=14` | n≥80, WR≥42%, PF≥1.30 |
| 2.2 | If gate passed: update DB `fixed_take_profit_pips=14` for EURUSD | — |

### Phase 3 — Test Ranging Block Removal (Week 2)

| # | Action | Backtest Gate |
|---|--------|---------------|
| 3.1 | BT EURJPY 90d `--override scalp_block_ranging_market=false --scalp --timeframe 5m` | n≥30, WR≥58%, PF≥1.20 |
| 3.2 | BT USDJPY 90d same override | n≥30, WR≥58%, PF≥1.20 |
| 3.3 | If both gates pass: remove flag via JSONB update on those 2 pairs | — |

### Phase 4 — Vision Coverage Audit (Week 2)

| # | Action |
|---|--------|
| 4.1 | Check `claude_analyzer.py` for path where signals exit without `claude_decision` being set |
| 4.2 | Determine if these 327 signals are auto-approved (good) or auto-rejected (bad) |
| 4.3 | If auto-approved: evaluate whether applying vision would have changed outcomes |

### Phase 5 — LPF Threshold Review (Week 3)

| # | Action |
|---|--------|
| 5.1 | Run 30d LPF in monitor mode with threshold=0.55 (lower), check how many borderline signals would be released |
| 5.2 | Query the 174 `bad_hours + low_mtf + moderate_exhaustion` borderline combo — check if any have outcome data from pairs where they leaked through |
| 5.3 | Consider requiring ≥2 rules with penalty≥0.20 to block (prevents 3 minor rules stacking to threshold) |

---

## Do Not Change (Clearly Justified Filters)

- Breakout regime block (40.9% WR confirmed)
- smc_high_conf_block intent (44.4% WR at 0.65–0.69 confirmed) — just fix enforcement
- sell_near_support LPF rule (15.8% WR confirmed)
- MACD filter for GBPUSD (removal makes things worse)
- Confidence threshold above 0.65 (inverse predictive confirmed across 90-day dataset)
