# EURJPY Model Search (pre-registered)

**Date locked:** 2026-06-15
**Goal:** Find a EURJPY entry model with a real, out-of-sample edge — good entries + PF — by
searching a broad battery IN-SAMPLE and confirming survivors OUT-OF-SAMPLE.

**Overfitting is the enemy here** (broad search, single pair). Defense = strict search/confirm split:
generate & rank on in-sample only; confirm the few best ONCE on a held-out slice; require sub-period
consistency. Most variants will die OOS — that is expected and fine. We report what survives.

## Data
- `ig_candles_backtest`, EURJPY (`CS.D.EURJPY.MINI.IP`), timeframes 1H (60) and 4H (240).
- Pip size 0.01. Cost: 2.0 pips round-trip per trade (stress 3.0). Dukascopy source.
- **IN-SAMPLE (search/rank): 2020-01-01 → 2023-12-31** (4y).
- **OUT-OF-SAMPLE (confirm, never used to select): 2024-01-01 → 2026-06-12** (~2.5y).

## Battery (entries)
Per timeframe ∈ {1H, 4H}:
1. **Trend:** EMA(20/50) cross, EMA(50/200) cross, Donchian(20) & (55) breakout, TSMOM(k=30,90).
2. **Mean-reversion:** Bollinger(20,2) fade (close re-enters band after breach), RSI(14) 30/70
   reversion, z-score(price vs SMA20) ≥|2| reversion.
3. **Pullback-in-trend:** EMA200 trend filter + RSI(14) pullback entry (buy dips up-trend / sell
   rallies down-trend).
4. **Session (UTC):** London-open (06–09) breakout of the Tokyo/Asian range (23–06);
   hour-of-day-conditioned momentum (restrict a base signal to its best in-sample session block).
5. **Volatility breakout:** opening-range breakout (first N bars of London), ATR-channel breakout.

## Exits (kept small to limit comparisons)
ATR(14)-based fixed brackets: SL = 1.5×ATR; TP ∈ {1.0, 1.5, 2.0}×ATR. First-passage on the same
timeframe's bars (intrabar: pessimistic straddle = loss). Horizon cap 10 bars×TF-in-hours? — use a
generous cap (e.g. 200 bars) and report timeout rate; exclude timeouts from the headline PF.

## Metrics (per model×exit)
n, WR, PF (Σ win / |Σ loss|, net cost), avg pips, annualized Sharpe, trades/day, and **entry quality**
= median MFE and the die-on-arrival fraction (MFE<0.25×ATR) — to judge *entries*, not just PF.

## Selection & decision (LOCKED)
1. IN-SAMPLE: keep only models with n ≥ 150 and PF ≥ 1.3 (intraday should clear n easily).
   Rank by PF. Take the **top 5** as candidates.
2. OUT-OF-SAMPLE confirm (the held-out 2024–26 slice): a candidate **PASSES** only if
   OOS PF ≥ 1.2 AND OOS n ≥ 60 AND OOS PF in BOTH halves of the OOS window > 1.0
   (no losing half) AND no catastrophic collapse vs in-sample (OOS PF ≥ 0.75 × in-sample PF).
3. **GO:** ≥1 candidate passes → that's the EURJPY model; document it, then (separate) wire monitor-only.
4. **NO-GO:** none pass → report honestly; the broad search found no OOS-robust EURJPY entry. Do NOT
   then hand-pick an in-sample winner and ship it — that's the overfit trap this whole protocol exists
   to avoid.

## Constraints
- Read-only; no DB writes. Standalone script; bulk-load candles once per timeframe.
- Heavy run via durable file output (not an agent stdout pipe).
- Honesty: report the FULL in-sample→OOS table for the top candidates, incl. the failures, so the
  degree of in-sample shrinkage is visible.

---

# RESULTS (run 2026-06-15, `worker/app/eurjpy_model_search.py`; no DB changes)

25 models × 3 ATR-TP exits = 75 cells. 1H=40,221 / 4H=10,383 bars.

## VERDICT: **NO-GO** — 0 of 75 cells passed even the IN-SAMPLE gate (n≥150 AND PF≥1.3).
Nothing reached the OOS confirm step. Whole mean-reversion family (BB_fade, RSI, z-score) and most
breakouts (Donchian, ORB, ATR-channel, TSMOM) are losing (PF<1.0) even in-sample.

## The one genuine bright spot (does NOT clear the bar — reported honestly)
**EMA_20/50 momentum crossover, 1H** has the best ENTRY quality in the entire battery:
- WR **66.4%** (TP1.0), die-on-arrival only **10.6%** (lowest of all 25 models — 89% of trades show
  favorable movement), medMFE ~20–30 pips, avg +0.4–0.6 pips net of cost.
- PF **stable ~1.10–1.13 across all three TP exits** (1.099/1.126/1.133) — stability = not a lucky cell;
  PF rises slightly as TP widens → winners have room to run (coherent momentum profile).
- Restricted to the **London open (07–09 UTC)** it jumps to PF 1.5–2.25 / WR 72% / DOA 7% — BUT
  n=57 over 4y (~14/yr): uninvestable, likely overfit. Still, a real hint EMA momentum concentrates
  in the London expansion window.

The gap is **exit/RR, not entry**: a good-timing momentum entry (66% WR, 89% favorable) capped by a
symmetric 1.5×ATR-SL / 1×ATR-TP bracket lands at PF ~1.1. The measured bottleneck is how winners are
monetized, not entry selection. (Contrast RANGE_FADE, whose losers genuinely died on arrival.)

## EMA-cross OOS closer (run as-is, zero new knobs — the decisive test)
| Window | TP1.0 | TP1.5 | TP2.0 | WR(TP1.0) | DOA |
|--------|------:|------:|------:|----------:|----:|
| IS 2020-23 (ref) | 1.099 | 1.126 | 1.133 | 66.4% | 10.6% |
| **OOS full 24-26** | **0.973** | **0.980** | **0.967** | 62.4% | 12.1% |
| OOS H1 2024 | 1.011 | 0.960 | 1.006 | 62.7% | — |
| OOS H2 2025+ | 0.949 | 0.994 | 0.942 | 62.2% | — |

The entry *timing* partially persisted (WR 66%→62%, DOA stayed ~12% — a faint real momentum-drift),
but PF shrank 1.10→**0.97**: breakeven-losing after costs, both halves. The good WR can't overcome the
1.5-SL/1-TP risk:reward plus 2-pip cost. NOTE: the "fix the exit to monetize runners" idea was NOT
pursued — that is the RANGE_FADE width-sweep trap (add exit DOF until PF crosses 1.2, gain is artifact).

## Forward read — FINAL
EURJPY price-action is closed. The best of 25 canonical models is breakeven-losing OOS. This is the 4th
independent FX-price-action negative (fade-kill, width-sweep, HTF-probe, this). Mining more chart
patterns won't change it. The remaining FX hope is **non-price-action** — carry / cross-sectional /
rate-differential (the HTF probe fingered the rate-driven JPY move as the only real FX money) — which
needs *different data*, not another entry battery. Decision handed to user.
