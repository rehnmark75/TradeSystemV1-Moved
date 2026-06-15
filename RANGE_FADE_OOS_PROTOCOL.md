# RANGE_FADE Out-of-Sample Validation — Pre-Registered Protocol

**Date locked:** 2026-06-14
**Goal:** Determine whether RANGE_FADE clears the user's bar:
**portfolio ~1+ trade/day, pooled OOS PF ≥ 1.2 at n ≥ 100, net of costs, under live trailing exits.**

This is pre-registered. The PASS *and* KILL conditions are fixed before running.
The point is to escape the overfitting treadmill — so a FAIL is an acceptable,
publishable result, not a trigger to add gates until it passes.

## Fixed facts (verified 2026-06-14)
- IG `ig_candles` 5m history: **Sep 18 2025 → Jun 12 2026** (~9 mo) for all 8 FX pairs.
- Backtest already models **1.5 spread + 0.5 slippage** at fill (entry-side, no commission).
- RANGE_FADE backtest uses **progressive trailing exits** (live config), seeded by SL 8 / TP 12.
  → backtest PF is comparable to live/forward, NOT a clean fixed bracket.
- Current DB config was tuned on **Apr–Jun 2026** → that window is in-sample for today's config.

## Locked pair set (do NOT drop pairs after seeing results)
All 8 FX demo pairs: EURUSD, GBPUSD, USDJPY, USDCAD, USDCHF, EURJPY, AUDJPY, AUDUSD, NZDUSD.
(NZDUSD listed = 8 majors/crosses RANGE_FADE is enabled on in demo.)

## Windows
- **PRIMARY OOS (honest):** Sep 18 2025 → Mar 31 2026. Current config was never fit to this. **This is the verdict window.**
- **Recent (contaminated, secondary):** Apr 1 → Jun 12 2026. Report for context only; lower trust.

## Method
1. **Freeze current DB config.** Do NOT edit any RANGE_FADE DB rows during this study.
2. Run `backtest_cli.py` per pair over the PRIMARY OOS window, `--scalp --timeframe 5m`,
   with `--no-historical-intelligence` (avoid adaptive contamination).
3. Pool results across all 8 pairs: total n (trades), pooled PF (Σ win / Σ loss), WR, signals/day.
4. Report **per-pair as diagnostic only** — judgment is on the **pooled** number.

## Cost-sensitivity (a result, not a checkbox)
Re-run PRIMARY OOS with realistic per-pair spreads via `--override spread_pips=`:
EURUSD 1.2, GBPUSD 1.8, AUDUSD 1.5, NZDUSD 2.0, USDCAD 1.8, USDCHF 1.8,
USDJPY 1.8, EURJPY 2.2, AUDJPY 2.5. Report pooled PF degradation vs the 1.5 default.

## Decision rule (LOCKED)
- **PASS:** pooled PRIMARY-OOS PF ≥ 1.2 AND n ≥ 100 (default costs) AND still ≥ 1.2 under stressed spreads.
  → proceed to a clean forward-demo gate (frozen config, fixed n target) before any live flip.
- **MARGINAL:** 1.0 ≤ PF < 1.2, or passes at default cost but fails stressed.
  → report honestly as "edge too thin to clear the bar after costs." No rescue tuning.
- **KILL:** pooled PF < 1.0. → verdict: "no stable portfolio edge at this bar."
  Do NOT add/loosen/tune gates to rescue it — that is the treadmill. Report and stop.

## What we explicitly will NOT do
- No dropping the worst pair to lift the pool.
- No new regime/ADX/session gate added *after* seeing OOS results.
- No anchoring to the forward 1.31 (different, contaminated window).

---

# RESULTS (run 2026-06-14, clean batch /tmp/rf_oos2, no DB changes)

PRIMARY OOS window Sep 18 2025 → Mar 31 2026, frozen current config, default costs
(1.5 spread + 0.5 slippage), live progressive trailing exits. All 9 rc=0.

| pair    |   n | WR%  |  PF  | net pips |
|---------|----:|-----:|-----:|---------:|
| EURUSD  | 189 | 51.9 | 0.81 |    -138  |
| GBPUSD  | 105 | 36.2 | 0.64 |    -193  |
| USDJPY  | 153 | 43.8 | 0.87 |    -112  |
| USDCAD  | 414 | 30.9 | 0.56 |   -1007  |
| USDCHF  | 397 | 39.5 | 0.64 |    -692  |
| EURJPY  |  12 | 66.7 | 2.08 |     +43  |
| AUDJPY  |  40 | 55.0 | 1.03 |      +5  |
| AUDUSD  | 376 | 47.3 | 0.62 |    -602  |
| NZDUSD  | 260 | 53.8 | 0.97 |     -29  |
| **POOL**| **1946** | — | **0.70** | **-2724** |

Pooled PF **0.70**, expectancy **-1.40 pips/trade**, ~304 signals/mo (~10/day portfolio).

## VERDICT: **KILL**
Pooled OOS PF 0.70 < 1.0 → "no stable portfolio edge at this bar."
- Only 2 of 9 pairs positive: EURJPY (n=12, too small) and AUDJPY (PF 1.03, breakeven n=40).
- EURUSD (the one live pair) = 0.81 OOS, despite the forward demo 1.31 → confirms the
  forward number was window-contaminated / small-sample, not a durable edge.
- Frequency was never the constraint (~10/day, 10× the target). **Edge is.**
- Cost-stress NOT run: it already fails at default cost; stress only lowers PF further.

Per the locked rule: report and stop. Do NOT add gates to rescue. Config unchanged.
