# Higher-Timeframe FX Edge Probe (pre-registered)

**Date locked:** 2026-06-15
**Question:** Before building any bespoke H4/Daily strategy, does a *capturable trend edge*
even exist on these pairs at this horizon? Test CANONICAL, well-documented signals — not a
custom-fitted design — measured cleanly out-of-sample after costs.

The point: avoid the 15-strategy treadmill. We do NOT design/tune a strategy until the raw
horizon proves it has something to capture. A clean negative here is a valid, valuable result.

## Data
- `ig_candles_backtest`, 4H (timeframe=240), resample to Daily (4×4H = 1 day) as needed.
- 8 pairs (drop GBPJPY — stale feed): EURUSD, GBPUSD, USDJPY, USDCAD, USDCHF, EURJPY, AUDJPY,
  AUDUSD, NZDUSD.
- Window: 2020-01-01 → 2026-06-12 (~6.5 years). Dukascopy source (HTF → pricing delta negligible).

## Costs
- 2.0 pips round-trip per trade, charged to each trade's return. Also report a 4.0-pip stress.
- Pip size: FX 0.0001; JPY pairs 0.01.

## Canonical signals (minimal params — this is NOT a tuning sweep)
Position/returns-based (the natural way to measure a trend signal; no bracket-design confound):
1. **TSMOM (time-series momentum)** — long if trailing K-bar return > 0, short if < 0; flip on
   sign change. Daily K ∈ {20, 60}; H4 K ∈ {30, 90}.
2. **Donchian breakout (turtle)** — enter on close beyond prior N-bar high/low; exit on opposite
   N/2-bar extreme. Daily N ∈ {20, 55}; H4 N ∈ {20, 55}.
3. **MA trend** — long if EMA50 > EMA200 and price > EMA50; flip on cross. Daily + H4.

Each "trade" = entry→exit; trade return = directional price change − 2-pip cost.

## Metrics (per signal×param×pair, and POOLED across pairs)
n trades, hit rate, **PF = Σ winning-trade returns / |Σ losing-trade returns|** (net cost),
avg trade return, annualized Sharpe of the equity curve, portfolio trades/day.

## Robustness (anti-overfit, LOCKED)
Split into 3 sub-periods: 2020–2021, 2022–2023, 2024–2026.
A signal **QUALIFIES** only if:
- Full-sample pooled PF ≥ 1.2 (net 2-pip cost), AND
- pooled PF > 1.0 in EACH of the 3 sub-periods (no losing regime), AND
- PF > 1.0 in ≥ 6 of the 8 pairs (broad support, not carried by one pair).

## DECISION (LOCKED before looking)
- **GO-to-build:** ≥ 1 signal family QUALIFIES. → build one clean strategy around that survivor,
  re-confirm OOS on a held-out slice, THEN consider monitor-only demo. No tuning to rescue a
  near-miss.
- **NO-GO:** no canonical trend signal qualifies → even documented HTF trend edges don't survive
  on these pairs after costs. Report honestly; reconsider the lane (instrument / edge-source).
  Do NOT start adding filters to a failing canonical signal — that's the treadmill.

## Constraints
- Read-only. No DB writes. Standalone probe script; bulk-load candles once per pair.
- Heavy run executed via durable file output (not an agent's stdout pipe — that has burned us).

---

# RESULTS (run 2026-06-15, `worker/app/htf_edge_probe.py`; no DB changes)

9 pairs, 2020-01-01→2026-06-12, ~10,380 4H bars + 1,675 daily bars each. 2-pip cost.

## VERDICT: **NO-GO** — no canonical trend signal qualifies.
Every variant's pooled PF sits between **0.83 and 1.06** (need ≥1.2); none robust across all
3 sub-periods; none with ≥6/9 pair support.

| Signal | Pooled PF (2p) | (4p) | 2020-21 / 2022-23 / 2024-26 | pairs>1.0 |
|--------|---:|---:|---|---:|
| TSMOM Daily K20  | 0.976 | 0.938 | 0.83 / 1.08 / 1.01 | 5/9 |
| TSMOM Daily K60  | **1.059** | 1.021 | 0.93 / 0.92 / 1.34 | 5/9 |
| TSMOM H4 K30     | 0.879 | 0.807 | 0.90 / 0.88 / 0.86 | 0/9 |
| TSMOM H4 K90     | 0.958 | 0.892 | 1.11 / 0.95 / 0.88 | 2/9 |
| Donchian D N20   | 0.988 | 0.962 | 0.95 / 1.07 / 0.94 | 4/9 |
| Donchian D N55   | 1.013 | 0.997 | 0.91 / 0.88 / 1.26 | 5/9 |
| Donchian H4 N20  | 0.866 | 0.814 | 1.02 / 0.78 / 0.86 | 1/9 |
| Donchian H4 N55  | 0.832 | 0.802 | 0.95 / 0.83 / 0.76 | 1/9 |
| MA Daily 50/200  | 0.923 | 0.892 | 0.94 / 0.98 / 0.85 | 2/9 |
| MA H4 50/200     | 0.848 | 0.784 | 0.91 / 0.76 / 0.90 | 0/9 |

## What the result actually says (informative, stated precisely)
1. **The HTF thesis was directionally right:** Daily > H4, longer lookback > shorter (K60>K20,
   N55>N20).
2. **Cost vs signal — be exact:** H4 variants are **cost-killed** (K30 0.879→0.807 at 4p, many
   trades). Daily variants **barely move on cost** (K60 1.059→1.021) — they are simply **thin-gross /
   weak signal**, not cost-decayed. So "costs killed it" holds only for H4; daily just has no edge.
3. **The only real money was the 2022 USD/JPY move:** USDJPY Donchian N55 PF 2.28, TSMOM K60 1.97;
   AUDJPY similar — one rate-divergence-driven mega-trend, NOT broad (USDCHF/NZDUSD/AUDUSD <1.0).

## Scope (do NOT overstate)
This tested **time-series trend, per-pair**. It did **not** test **cross-sectional** FX (rank pairs,
long top / short bottom) — where the documented FX premium actually sits — nor an actual **carry**
strategy (earning the rate differential). What's closed is *time-series price-action/trend on FX
majors, 5m→daily, after costs*.

## Forward read
Three independent negatives (5m fade OOS-kill, width sweep, this probe) close **time-series FX
price-action/trend**. Per the locked rule: do NOT add filters to rescue (treadmill).
**Hypothesis (untested, not a finding):** the biggest FX moves are rate-differential-driven, so IF a
systematic FX edge exists it more plausibly lives in rates/carry/cross-sectional than chart patterns —
that would need its own pre-registered probe. Pivot lanes (not equal): **equities** is better-grounded
(documented cross-sectional anomalies + existing pipeline + live account → monitor-first) ahead of FX
carry/cross-sectional (a hypothesis needing new data wiring).
