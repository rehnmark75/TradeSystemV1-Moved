# Non-Price-Action FX: Carry + Cross-Sectional — Scope

**Date:** 2026-06-15
**Why:** FX price-action is closed (4 independent negatives: fade-kill, width-sweep, HTF-probe,
EURJPY model search). This scopes the one remaining FX avenue — harvesting documented FX *premia*
rather than predicting price from charts.

## The edge (what we'd actually be harvesting)
Three related, documented effects — none is chart-pattern prediction:
1. **Carry** — long high-interest-rate currencies / short low-rate ones; earn the rate differential
   (manifests on an IG account as overnight **swap/rollover**). Documented (Lustig-Verdelhan,
   Menkhoff et al.). Real but **crash-prone**: unwinds violently in risk-off (the carry's fat left tail).
2. **Cross-sectional momentum** — rank the currency universe by past return; long the strongest,
   short the weakest. KEY: this is *relative* ranking, NOT the time-series/absolute trend we already
   tested and killed — cross-sectional momentum is documented to work in FX where TS-momentum doesn't.
3. **Carry + momentum combo** — the robust version; momentum diversifies carry's crash risk.

## Universe
**8 G10 currencies, all rankable: USD, EUR, GBP, AUD, NZD, JPY, CAD, CHF.** Build each currency's
USD-denominated return series from the existing pairs (USD itself = flat numeraire return; others via
the pair or its inversion). **CRITICAL construction fix:** USD must be IN the ranked set, not merely the
base — rank all 8 by signal, long top-k / short bottom-k, **equal-weight (auto currency-neutral).**
Otherwise we structurally cannot express "long USD" books — and in 2022–23 USD was itself a high-carry,
high-momentum currency (the dominant move). Ranking "7 vs USD" would discard the dollar factor entirely.
**Limitation (up front):** small universe. The documented carry premium is strongest with
**emerging-market** high-rate currencies, which this system does not trade. G10-only carry is materially
weaker, especially post-2008. We are testing the weak version of a real effect.

## Data
| Need | Status |
|------|--------|
| Spot prices (for returns + momentum) | **HAVE** — Dukascopy 4H/daily, 2020–2026, all pairs |
| Cross-sectional **momentum** signal | **Needs nothing new** — derived from spot ← cheap first probe |
| **Carry** signal (short-term rates per ccy, monthly) | **GAP** — must source ~8 rate series |

Rate-data options for carry: central-bank **policy rates** (coarse, step-changes — easy to compile)
or **3M interbank/OIS** (better carry proxy — sourceable from FRED etc.). ~8 monthly series 2020–2026.
Must be sourced carefully (accuracy matters); this is the only real data build.

### ⚠️ The carry killer-risk = IG swap markup (the carry analog of spread)
This is the same failure mode that killed all 4 price-action attempts, in disguise. The clean
interbank/policy-rate differential we'd backtest is **NOT** what an IG account earns — IG marks up
overnight financing (pays less positive swap, charges more negative) by a margin that, on a thin G10
differential, can eat the **entire** edge. So: model **realistic IG swap**, not theoretical rates. If
IG historical swap is unavailable, **haircut the interbank carry hard and treat the clean-rate PF/Sharpe
as a CEILING, not an estimate.** A clean-rate Sharpe 0.5 could be ~0 after IG's markup. Decide the
haircut before looking.

## Test design (pre-registered, same discipline as before)
- **Monthly rebalance** (test weekly as a variant). Long top-k / short bottom-k basket by the signal.
- IS 2020–2022 / OOS 2023–2026. **Caveat:** carry signal barely exists in 2020–2021 (global
  near-zero rates → no rate dispersion → no carry). The real carry test is the 2022+ divergence era,
  which shortens the effective sample.
- Metrics: annualized **Sharpe**, **max drawdown** (CRITICAL — carry's whole risk is the crash),
  PF, hit rate, turnover, net-of-cost equity curve.
- Costs: rebalance spread (~2 pip/leg/rebalance) + for carry, the **swap** P&L (the actual rate
  carry earned/paid).
- **Bar + burden of proof (calibrated to the PRIOR):** unlike price-action (no prior → we demanded
  bulletproof OOS), carry has a 40-year literature and a real economic mechanism. So the honest
  question is *"does the known premium survive THIS universe, THIS sizing, and IG's costs?"* — judged on
  **net Sharpe (~≥0.5), max-drawdown, and economic sanity** — NOT "discover an edge from scratch."
  Power caveat: monthly rebalance over 6.5y ≈ ~78 obs (dozens, not the thousands we had intraday), and
  the crash risk lives in rare tails we can't sample — so don't over-demand OOS (and reject a real-but-
  thin premium) nor under-demand it (and get fooled by 40 lucky months). Cross-sectional momentum has a
  weaker prior than carry → hold it to a stricter OOS standard.

## Honest limitations & expectations
- **Low frequency:** monthly rebalance, positions held weeks, only a handful open at once. This is
  NOT "1 trade/day" — it's a different animal. (Upside: low freq makes a **scheduled monthly
  rebalance** feasible WITHOUT the real-time scanner — simpler to run, not harder.)
- **Execution mismatch:** the whole current stack (2–5 min scan, scalp pip-trailing, fixed-pip
  brackets) is intraday. Carry/xsec is a hold-for-weeks, swap-aware portfolio rebalance — a separate
  execution model. Not a blocker, but a real build if it validates.
- **Realistic outcome:** a modest, crash-prone G10 edge. Plausible net Sharpe ~0.3–0.6 with ugly
  tails. It may clear the bar; it may not. We test honestly.

## Recommended sequencing (front-load the cheap test)
1. **Cross-sectional MOMENTUM probe FIRST** — free, no new data. Tests whether relative-ranking has
   edge where absolute trend failed. Decisive go/no-go in ~1 hour. **Temper expectations:** it's built
   from the *same* return series whose time-series momentum we already killed — relative ranking removes
   the common USD factor and *can* isolate a cleaner signal (worth the free hour), but it's a
   recombination of dead material, not fresh edge. G10-only FX momentum is documented as weak and
   cost-sensitive. Don't be surprised if it's also a NO-GO.
2. If promising → **source rate data + CARRY + carry-momentum combo** probe.
3. Only if something validates → discuss execution wiring (scheduled monthly rebalance).

## Constraints (unchanged)
Read-only probes, durable file output, pre-registered bar/window before looking, run heavy jobs
myself (not agent stdout pipes).

---

# RESULTS — Leg 1: Cross-sectional MOMENTUM (run 2026-06-15; `worker/app/fx_xsec_momentum_probe.py`)

Universe construction VERIFIED: dollar-neutral every rebalance; USD-returns economically sane
(JPY −32%, CHF +21% over 2020–26). 16 configs (K∈{1,3,6,12}mo × basket{2,3} × {monthly,weekly}),
2-pip turnover cost, IS 2020–22 / OOS 2023–26.

## VERDICT: **NO-GO** — 0/16 configs clear the bar (OOS Sharpe≥0.5, MaxDD≤25%, TotRet>0).
- Weekly rebalances all clearly negative (turnover cost drag): OOS Sharpe −0.43 to −1.28.
- Monthly better but still short: most negative; **best = K=12mo, basket=3, monthly** → OOS Sharpe
  0.458, OOS TotRet +6.7%, MaxDD 6.3%, HitRate 66% — *close* but below 0.5 AND its IS Sharpe was
  −0.09 (not IS-consistent), so not a real survivor.
- Read: G10-only cross-sectional momentum is weak/cost-sensitive, as expected (recombined from the
  same return series whose time-series momentum we already killed). Faint annual-horizon whiff only.

**Carry (Leg 2) is a SEPARATE, stronger-prior bet** (40-yr literature + economic mechanism); momentum
failing does NOT kill it. But Leg 2 requires the rate-data build + realistic IG-swap cost model, and is
gated on the user explicitly accepting the low-frequency product shape.
