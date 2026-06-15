# RANGE_FADE Wider SL/TP — Fixed-Bracket Sweep (pre-registered)

**Date locked:** 2026-06-15
**Question:** Does the RANGE_FADE *entry* have a profitable edge at a wider SL/TP if we
**bypass the progressive trailing** and use pure fixed brackets?

This is an exit-only experiment on a FROZEN entry set. We do not touch the entry logic,
gates, or config. We re-resolve the existing OOS signals under fixed brackets.

## Entry set (frozen)
The 1,946 RANGE_FADE signals from the honest OOS window (Sep 18 2025 → Mar 31 2026),
parsed from `/tmp/rf_oos2/*.log` per-trade result tables. Each row gives:
market `TIMESTAMP` (UTC), pair, side (BUY/SELL), entry `PRICE`. Parse count MUST match
the per-pair signal totals (EURUSD 189, GBPUSD 105, USDJPY 153, USDCAD 414, USDCHF 397,
EURJPY 12, AUDJPY 40, AUDUSD 376, NZDUSD 260 = 1946). Integrity-check this before proceeding.

## Exit model
- **Pure fixed bracket, NO trailing.** First-passage on **1-minute** candles from
  `ig_candles` (timeframe=1) starting at the entry timestamp.
- Pip size: FX majors 0.0001; JPY pairs (USDJPY, EURJPY, AUDJPY) 0.01.
- **Entry cost 2.0 pips** worsening the fill (BUY fill = price + 2pip; SELL fill = price − 2pip),
  matching the engine (1.5 spread + 0.5 slip). SL/TP measured from the worsened fill.
- BUY: SL = fill − SL_pips, TP = fill + TP_pips (low ≤ SL → stop; high ≥ TP → target).
  SELL mirrored. If a single 1m candle straddles BOTH levels → count as **LOSS** (pessimistic).
- Horizon cap **72h**; if neither level hit, close at the 72h price (residual P&L counts).
  Report the timeout rate.

## Grid (30 cells)
- SL ∈ {8, 12, 16, 20, 25, 30} pips
- TP = R × SL, R ∈ {1.0, 1.5, 2.0, 2.5, 3.0}

## Metrics (pooled across 8 pairs AND per-pair)
Per cell: n, WR, **PF = Σ win pips / Σ loss pips (net of cost)**, expectancy pips/trade.

## Mechanism diagnostic (compute once, report regardless of verdict)
For all trades, MFE (max favorable excursion) and MAE (max adverse) in pips over the 72h horizon:
- Of trades that lose at the current SL=8: what fraction ever reached MFE ≥ 8 / ≥ 12 / ≥ 16
  favorable before the stop (i.e. "rescuable" by wider stop+target) vs die-on-arrival (MFE < 2)?
- Winner MFE left-on-table at the fixed TP. This tells us mechanically if wider can help.

## Robustness (anti-overfit, LOCKED)
Split OOS by date into H1 (Sep 18 – Dec 31 2025) and H2 (Jan 1 – Mar 31 2026).
A cell is a **PASS** only if pooled PF ≥ 1.2 AND n ≥ 100 in **BOTH** H1 and H2.

## DECISION (LOCKED before looking)
- **GO** (wider SL/TP rescues RANGE_FADE): a **contiguous region of ≥2 adjacent grid cells**
  PASSES the robustness rule above. Report the region; then (separate step) confirm under live
  trailing widened to match before any config change.
- **NO-GO:** no robust contiguous region clears the bar — OR the diagnostic shows losers die on
  arrival. Verdict: "entry edge is not realizable at any tested width." Do NOT cherry-pick a
  single lucky cell. Report and stop.

## Constraints
- Do NOT modify any DB row. This is a read-only analytic sweep over candles + parsed logs.
- Produce a self-contained script at `worker/app/range_fade_width_sweep.py`; run in task-worker.
- Bulk-load 1m candles per pair ONCE over the window; slice per trade in memory (no 1946 queries).

---

# RESULTS (run 2026-06-15, independently re-run & verified; no DB changes)

Scripts: `worker/app/range_fade_width_sweep.py` (pass 1), `worker/app/range_fade_diag3.py` (pass 2).
Integrity PASSED (1946, all per-pair exact). 1m candles confirmed (~199K/pair). Sanity anchor
SL=8/R=2 → PF 0.77 (≈ the 0.70 trailing run). All numbers below reproduced by my own clean runs.

## Mechanism (72h unconditional MFE of the 1404 current SL=8 losers)
- Die-on-arrival (MFE < 2 pips): **6.1%** — losers are NOT dead on arrival; they DO revert.
- Reach ≥16 / ≥25 / ≥50 / ≥62 / ≥75 pips favorable: 70% / 58% / 36% / 28% / 21%.
- MFE percentiles: P50 **32.8**, P75 66.9, P90 111.6 pips. They revert but don't SUSTAIN.

## Pooled PF heatmap (SL × R, fixed bracket, MTM timeouts counted per spec)
```
SL\R   1.0   1.5   2.0   2.5   3.0
 8    0.67  0.74  0.77  0.84  0.87
12    0.85  0.95  0.94  1.00  1.06
16    0.99  1.03  1.08  1.13  1.15
20    1.04  1.07  1.15  1.16  1.19
25    1.09  1.18  1.20  1.23  1.24
30    1.05  1.17  1.17  1.18  1.17
```

## Robustness + the decisive timeout check
Under the LITERAL locked rule (timeouts = MTM), SL=25 × R=2.5 (PF 1.235; H1 1.217 / H2 1.254)
and R=3.0 (1.239; H1 1.233 / H2 1.244) form a 2-cell contiguous region that PASSES H1/H2.
BUT those cells carry 12–17% timeouts, and excluding them (real resolved trades only):

| Cell | PF incl. MTM timeouts | PF clean-resolved | clean WR | needed WR |
|------|----:|----:|----:|----:|
| SL=25 R=2.0 (TP50) | 1.167 | **1.092** | 35.3% | 37.5% |
| SL=25 R=2.5 (TP62) | 1.198 | **1.039** | 29.4% | 32.4% |
| SL=25 R=3.0 (TP75) | 1.214 | **0.934** | 23.7% | 28.6% |

The apparent pass is a 72h mark-to-market artifact: ~12–17% of trades never reach the wide TP,
sit slightly green at the arbitrary 72h cutoff, and that residual is what lifts pooled PF over 1.2.
On genuinely TP/SL-resolved trades the PF is 0.93–1.09. Also: SL=25/TP=62 is a multi-day swing,
not the intraday 5m fade RANGE_FADE is — widening to it = a different, unvalidated strategy.

## VERDICT: **NO-GO**
Wider SL/TP does not give RANGE_FADE a deployable edge. The only cells clearing 1.2 do so via a
horizon-cutoff artifact and collapse on clean resolution. Bracket width is the WRONG lever:
the exit is not the constraint — the entry reverts but doesn't sustain far enough to reach a wide
target. Consistent with the OOS KILL ([[project_range_fade_oos_kill_jun14]]) and the June
regime-decay finding. Config unchanged.
