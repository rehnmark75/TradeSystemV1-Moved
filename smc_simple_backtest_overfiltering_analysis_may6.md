# SMC_SIMPLE Backtest Over-Filtering Analysis — May 6, 2026

## Backtest Results Summary

All tests: 30 days, scalp mode, --timeframe 5m

| Test | Signals | WR% | PF | Expectancy | Avg Win | Avg Loss |
|------|---------|-----|-----|-----------|---------|----------|
| EURUSD baseline | 160 | 45.0% | 0.69 | **-1.5 pips** | 7.4 | 8.8 |
| EURUSD no MACD | 160 | 45.0% | 0.69 | **-1.5 pips** | 7.4 | 8.8 |
| EURUSD min_conf=0.50 | 160 | 45.0% | 0.69 | **-1.5 pips** | 7.4 | 8.8 |
| GBPUSD baseline | 24 | 37.5% | 0.72 | **-1.2 pips** | 8.4 | 7.0 |
| EURJPY baseline | 4 | 75.0% | 0.77 | **-0.9 pips** | 3.8 | 15.0 |

**All 5 variants are losing money. Every pair has negative expectancy.**

---

## Critical Finding 1: MACD and min_confidence Filters Have Zero Effect on EURUSD

Disabling MACD and lowering min_confidence from 0.45 to 0.50 both produced **identical results** (160 signals, 45% WR, PF 0.69). This means:

- **MACD is not active for EURUSD** at the backtest config level — zero MACD rejections in filter breakdown
- **Natural confidence distribution floor is already ≥ 0.50** for EURUSD — no signals exist below the current floor. Lowering it unlocks nothing.

Implication: The two most commonly proposed "over-filtering" fixes for EURUSD (disable MACD, lower confidence) are **inert**. They cannot be the cause of low signal quality or count.

Note: MACD IS active for GBPUSD (232 rejections = 3% of all GBPUSD attempts), confirming the filter is pair-specific.

---

## Critical Finding 2: EURJPY Session Filter Is the Primary Signal Killer

EURJPY filter breakdown (cumulative evaluation attempts):

| Filter | Rejections | % of total |
|--------|-----------|-----------|
| **Session** | **2,678** | **40%** |
| Tier2 Swing | 2,496 | 37% |
| Tier3 Pullback | 442 | 7% |
| Tier1 EMA | 321 | 5% |
| Cooldown | 296 | 4% |
| Scalp Entry Filter | 251 | 4% |
| Pair Scalp Filter | 78 | 1% |

The Session filter blocks **40% of all evaluation attempts** for EURJPY, producing only **4 signals in 30 days** (49 were seen live over the same period — a 12x gap). The session window configured in EURJPY's pair overrides is extremely narrow, far more restrictive than other pairs.

**This is the primary over-filtering candidate for EURJPY.** The 12x live-vs-backtest gap is unexplained — the same session filter should apply to both. This needs investigation: either the backtest applies session gates more aggressively than live, or live signals are generated outside the session window through a different code path.

---

## Critical Finding 3: EURUSD Cooldown is the #2 Rejection Source

EURUSD filter breakdown:

| Filter | Rejections | % of total |
|--------|-----------|-----------|
| **Tier3 Pullback** | **2,386** | **35%** |
| **Cooldown** | **1,939** | **28%** |
| Tier2 Swing | 956 | 14% |
| Scalp Entry Filter | 342 | 5% |
| Risk TP | 341 | 5% |
| Tier1 EMA | 190 | 3% |
| Pair Scalp Filter | 148 | 2% |
| Min ADX | 94 | 1% |
| MFI Filter | 56 | 1% |
| Block Low Volatility Trending | 54 | 1% |
| Regime Breakout | 50 | 1% |
| Impulse Quality | 48 | 1% |

**Cooldown accounts for 28% of all EURUSD rejections.** With 160 signals from 366 detected (54.1% filtered), cooldown alone is suppressing ~74 signals per month for EURUSD. No outcome data exists for cooldown-suppressed signals — we don't know if they would have won or lost.

**Risk TP: 341 rejections (5%)** — The TP value is failing a risk check 341 times. With EURUSD SL=8/TP=10 this should always pass a 1.0 R:R minimum. The dynamic TP adjustment (volatility-scaled) is sometimes producing values that fail validation. This is likely discarding valid signals unnecessarily.

---

## Critical Finding 4: EURUSD R:R Math Is the Root Cause of Negative Expectancy

- SL = 8 pips, TP = 10 pips → **R:R = 1.25**
- Breakeven WR at 1.25 R:R = **44.4%**
- Actual WR = **45.0%** — barely above breakeven *in theory*
- But realized avg loss (8.8 pips) > avg win (7.4 pips) → **realized R:R = 0.84**
- Breakeven WR at realized 0.84 R:R = **54.3%**
- Actual WR 45% vs needed 54.3% = structural loss

The gap between configured R:R (1.25) and realized R:R (0.84) is caused by trailing stops exiting winners early. 24 of 72 winners exited via trailing stop at reduced profits rather than hitting the 10-pip TP target. The trailing stop is systematically converting 10-pip wins into ~4–6 pip wins, destroying the R:R math.

**This is not an over-filtering problem. More signals at 45% WR and -1.5 pips expectancy means more losses.**

---

## Critical Finding 5: GBPUSD Has 100% BUY Directional Bias

All 24 GBPUSD signals in 30 days are BUY. Zero SELL signals. The HTF bias filter correctly identified GBPUSD as bullish during April-May 2026 (cable bull run), but the SELL side is completely blocked. This is correct strategy behavior (trend-follow), not over-filtering. However:

- All 24 BUYs at 37.5% WR = structural underperformance
- GBPUSD BUY signals are firing into a trending bull market and losing at 37.5%
- The pair scalp filter (453 rejections = 6%) is very active for GBPUSD
- GBPUSD filter rate: 88.9% filtered (623 detected, only 69 passed, then dedup'd to 24)

The extremely high filter rate (88.9%) with poor outcomes suggests GBPUSD's signal conditions are fundamentally weak in the current regime, not that filtering is too aggressive.

---

## Critical Finding 6: EURJPY Trailing Stop Destroys Positive WR

EURJPY exit breakdown:
- **0 trades hit profit target** (TP = 24 pips)
- **3 trades exited via trailing stop** (avg win = 3.8 pips)
- **1 trade hit stop loss** (loss = 15 pips)

75% WR but avg winner = 3.8 pips vs avg loser = 15.0 pips → R:R = **0.25**. The trailing stop locks in ~4 pips of profit on winners and lets the single loser run to full 15-pip SL. With a 24-pip TP target that never gets reached, the trailing system is:
1. Triggering too early (locking in 3.8 pips on 24-pip targets)
2. Not protecting enough on the losing side

This is a trailing stop configuration issue, not a signal quality or filter problem.

---

## Over-Filtering Verdict by Filter

| Filter | Verdict | Evidence |
|--------|---------|----------|
| **EURJPY Session** | ✅ LIKELY OVER-FILTERING | 40% of attempts blocked, 12x live/BT gap unexplained. Test: expand session window, check if more signals + positive expectancy. |
| **EURUSD Cooldown (28%)** | ⚠️ NEEDS VALIDATION | 1,939 rejections/month with no outcome data. Could be suppressing valid setups OR preventing over-trading same setup. |
| **Risk TP (5% EURUSD)** | ⚠️ LIKELY BUG | 341 rejections on a 1.25 R:R config that should always pass. Dynamic TP adjustment producing out-of-range values. |
| **MACD (EURUSD)** | ✅ NOT ACTIVE | Zero effect — already disabled for EURUSD. |
| **MACD (GBPUSD)** | ✅ JUSTIFIED | 232 rejections (3%) but removal makes GBPUSD worse (confirmed in prior test). |
| **min_confidence ≥ 0.50** | ✅ NOT CONSTRAINING | No signals exist below current floor. Lowering it is inert for EURUSD. |
| **Tier3 Pullback (35%)** | ❓ UNKNOWN | Largest single filter (2,386 rejections). No outcome data for rejected signals. |
| **Tier2 Swing (14-44%)** | ❓ UNKNOWN | Core strategy filter — relaxing would change strategy character entirely. |
| **Regime Breakout (1%)** | ✅ JUSTIFIED | 40.9% WR confirmed in live data. |
| **Ranging Market Block** | ✅ INVESTIGATE | Only 23 EURJPY rejections from this — small but live data showed 68.2% WR for ranging signals. Low priority given tiny count. |

---

## Root Cause Summary: Not Over-Filtering, But Three Structural Problems

The backtests reveal the main issues are **not** from over-filtering but from three structural problems:

### Problem 1: EURUSD Trailing Stop Eating R:R (Most Urgent)
- Configured R:R: 1.25 | Realized R:R: 0.84 (trailing stops exit winners early)
- Fix: Review trailing stop trigger levels for EURUSD. Either disable early trailing on scalp mode OR raise TP to 14–16 pips so trailing stop exits are still above breakeven math.

### Problem 2: EURJPY Session Window Too Narrow
- Session filter blocks 40% of all attempts → 4 signals in 30 days
- Fix: Check EURJPY session config in pair overrides. Run 30d backtest with broader session window (e.g., 06-22 UTC instead of current narrow window). Gate: n≥20, PF≥1.20.

### Problem 3: Risk TP Validation Incorrectly Rejecting EURUSD Signals (5%)
- 341 rejections that should pass a 1.25 R:R check
- Fix: Add logging to the Risk TP rejection path to identify what value is being computed when it fails. Likely a dynamic volatility adjustment producing a TP < SL scenario.

---

## Verification Plan

### Immediate (this week)

1. **EURJPY session audit**
   ```bash
   docker exec postgres psql -U postgres -d strategy_config -c "
   SELECT parameter_overrides->>'scalp_allowed_sessions', parameter_overrides->>'scalp_session_start', parameter_overrides->>'scalp_session_end' 
   FROM smc_simple_pair_overrides WHERE epic LIKE '%EURJPY%';"
   ```
   Then run: `bt.py EURJPY 30 --scalp --timeframe 5m --override scalp_session_start=6 --override scalp_session_end=22`

2. **EURUSD Risk TP bug**
   Add `logger.warning(f"Risk TP rejected: sl={sl_pips}, tp={tp_pips}, rr={rr}")` in the Risk TP validation path and re-run backtest to capture rejected values.

3. **EURUSD trailing stop impact**
   Run: `bt.py EURUSD 30 --scalp --timeframe 5m --override fixed_take_profit_pips=14`
   Gate: n≥80, WR≥42%, PF≥1.10

### Next week

4. **EURUSD 90d backtest** before any live config change (30d window = April only, may be anomalous)
5. **GBPUSD monitor-only** — 37.5% WR, 0 SELLs, 88.9% filter rate. Move to monitor_only until regime changes.

---

## Do Not Change

- MACD filter for GBPUSD (removal confirmed worse)
- min_confidence floors (inert for EURUSD; lowering does nothing)
- Breakout regime block (justified by live data)
- Ranging market block (only 23 EURJPY rejections — negligible impact, investigate session first)
- Tier3 Pullback and Tier2 Swing (core strategy mechanics — relaxing changes strategy character)
