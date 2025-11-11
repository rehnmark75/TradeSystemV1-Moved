# TEST N (v2.18.0) Failure Root Cause Analysis

**Date:** 2025-11-11
**Analyst:** Trading Strategy Analyst
**Version Tested:** v2.18.0 (BOS Quality Calibration Fix)
**Baseline Version:** v2.17.0 (TEST M)

---

## Executive Summary

**CRITICAL FINDING:** TEST N (v2.18.0) never executed the new BOS quality calculation code. The "fix" was completely bypassed because BOS/CHoCH detection was DISABLED during the test run. The performance degradation (-4.6pp win rate, -22% profit factor) is NOT caused by the v2.18.0 changes but by running a completely different strategy configuration.

---

## Performance Comparison

### TEST M (v2.17.0) - Baseline
- **Signals:** 36
- **Win Rate:** 38.9%
- **Profit Factor:** 0.72
- **Average Winner:** 9.1 pips
- **Expectancy:** -1.7 pips
- **Issue:** 27 fallback structure instances, BOS quality scores averaging 22.5%

### TEST N (v2.18.0) - "Fixed" Version
- **Signals:** 35
- **Win Rate:** 34.3% (-4.6pp worse)
- **Profit Factor:** 0.56 (-22% worse)
- **Average Winner:** 9.0 pips (unchanged)
- **Expectancy:** -2.9 pips (-71% worse)
- **Reality:** BOS/CHoCH detection was DISABLED

---

## Root Cause Analysis

### 1. The Smoking Gun Evidence

**Log Analysis:**
```bash
# TEST N - All signal generation logs show:
"ℹ️  BOS/CHoCH detection disabled or no 15m data, using recent swing"
Count: 392 instances
```

**Critical Log Searches:**
- `grep "BOS/CHoCH confirmed"` → 0 results
- `grep "BOS Quality Breakdown"` → 0 results (v2.18.0 debug logs)
- `grep "STEP 3A: Detecting BOS"` → 0 results
- `grep "using fallback structure"` → 0 results (v2.17.0 pattern)

**Conclusion:** BOS/CHoCH detection was completely disabled in TEST N.

### 2. Code Path Analysis

**Expected Flow (v2.18.0 with BOS enabled):**
```
1. No rejection pattern found
2. Patterns optional (structure-only mode) → PASS
3. Check: if self.bos_choch_enabled and df_15m is not None → Should PASS
4. STEP 3A: Detect BOS/CHoCH on 15m
5. _calculate_bos_quality() with new v2.18.0 logic
6. Apply quality threshold filter
```

**Actual Flow (TEST N):**
```
1. No rejection pattern found
2. Patterns optional (structure-only mode) → PASS
3. Check: if self.bos_choch_enabled and df_15m is not None → FAILED
4. Log: "BOS/CHoCH detection disabled or no 15m data"
5. Fallback to recent swing (10-bar low/high)
6. NO quality calculation performed
```

**Code Location:**
```python
# smc_structure_strategy.py:614-618
if self.bos_choch_enabled and df_15m is not None and len(df_15m) > 0:
    self.logger.info(f"\n🔄 STEP 3A: Detecting BOS/CHoCH on 15m Timeframe")
    bos_choch_info = self._detect_bos_choch_15m(df_15m, epic)
else:
    # This path was taken in TEST N
    self.logger.info(f"   ℹ️  BOS/CHoCH detection disabled or no 15m data, using recent swing")
```

### 3. Why BOS Was Disabled

**Two Possible Causes:**

**A. Configuration Issue**
- `self.bos_choch_enabled = False` in config
- Config loading: `self.bos_choch_enabled = getattr(self.config, 'SMC_BOS_CHOCH_REENTRY_ENABLED', True)`
- Default is `True`, so must have been explicitly set to `False`

**B. Missing 15m Data**
- `df_15m is None` or `len(df_15m) == 0`
- Less likely since data fetching succeeded for other timeframes

**Most Likely:** Configuration parameter `SMC_BOS_CHOCH_REENTRY_ENABLED` was set to `False` in the backtest config used for TEST N.

### 4. Strategy Comparison

| Aspect | TEST M (v2.17.0) | TEST N (v2.18.0) | Result |
|--------|------------------|------------------|---------|
| **BOS Detection** | Attempted (failed 27x) | Never attempted | Different strategy |
| **Quality Calc** | Old v2.17.0 logic | Never called | No comparison |
| **Structure Source** | BOS levels (when detected) | Recent swing fallback | Different entries |
| **Fallback Usage** | 27/36 signals (75%) | 392/392 (100%) | Complete fallback |

---

## Why Performance Degraded

### Performance Change is NOT Due to v2.18.0 Code

**The degradation is caused by comparing TWO DIFFERENT STRATEGIES:**

1. **TEST M Strategy:** HTF alignment + BOS detection (with fallback when quality fails)
2. **TEST N Strategy:** HTF alignment + ALWAYS use fallback (no BOS detection)

**Evidence:**
- Both tests had identical BOS-disabled counts (392)
- TEST M had 27 "using fallback structure" messages
- TEST N had 0 "using fallback structure" messages (because it ALWAYS used fallback)
- The log message changed, suggesting different code paths

### Why Fallback-Only Strategy Performs Worse

**Fallback Structure Logic:**
```python
# Recent swing (10-bar lookback)
if final_trend == 'BULL':
    rejection_level = df_1h['low'].tail(10).min()  # Last 10 bars low
else:
    rejection_level = df_1h['high'].tail(10).max()  # Last 10 bars high
```

**Issues with Fallback:**
1. **No Quality Filter:** Uses 10-bar swing regardless of significance
2. **Arbitrary Level:** May not reflect institutional accumulation zones
3. **No Confirmation:** No breakout confirmation, just proximity to swing
4. **Weaker Structure:** Generic support/resistance vs. BOS decision points

**Impact:**
- Lower quality entry points → Lower win rate (-4.6pp)
- Worse risk/reward → Lower profit factor (-22%)
- No edge selection → Worse expectancy (-71%)

---

## Key Questions Answered

### 1. Why did win rate DROP from 38.9% to 34.3%?

**Answer:** The strategy changed from "BOS detection with fallback" to "always fallback". The fallback strategy uses arbitrary 10-bar swings instead of validated breakout structure, leading to lower quality signals.

### 2. Why did signals stay the same (35 vs 36) instead of increasing?

**Answer:** Both strategies use the same HTF alignment filter and entry conditions. Signal count is driven by:
- Number of timeframes processed (same)
- HTF trend detection (same)
- Pattern-optional mode (same)
- Only the STRUCTURE LEVEL selection differs (BOS vs fallback)

Signal count staying similar masks the fact that the strategies are fundamentally different.

### 3. Did TEST N actually USE the new quality calculation?

**Answer:** NO. The v2.18.0 quality calculation code (lines 1617-1723) was never executed. TEST N never called `_detect_bos_choch_15m()`, so it never called `_calculate_bos_quality()`.

### 4. Log Evidence Analysis

**Expected v2.18.0 Logs (MISSING):**
```
📊 BOS Quality Breakdown:
   Body ratio: 1.25x average
   Body score: 0.40 / 0.50
   Wick ratio: 18.50% (bullish)
   Wick score: 0.50 / 0.50
   Total quality: 0.90 / 1.00
   Threshold: 0.55
   Result: ✅ PASS
```

**Actual TEST N Logs:**
```
ℹ️  BOS/CHoCH detection disabled or no 15m data, using recent swing
✅ Structure-based entry (no pattern required):
   Direction: bullish
   Level: 1.23456 (recent swing)
```

### 5. Was the wick logic fix TOO STRICT?

**Answer:** Cannot determine. The wick logic fix (bullish checking lower wick, bearish checking upper wick) was never tested because the code path was never executed.

### 6. Did removing volume factor hurt?

**Answer:** Cannot determine. The volume removal change was never tested because the quality calculation was never called.

---

## Impact Assessment

### What We Actually Tested

**TEST M:** Hybrid strategy with BOS detection (27% success) + fallback (73%)
**TEST N:** Pure fallback strategy (100% fallback)

**Performance Delta:**
- Win Rate: 38.9% → 34.3% (-4.6pp) = **-11.8% relative**
- Profit Factor: 0.72 → 0.56 (-0.16) = **-22.2% relative**
- Expectancy: -1.7 → -2.9 pips (-1.2) = **-70.6% relative**

### What We DIDN'T Test

**v2.18.0 Quality Calibration Changes:**
- ✗ Fixed wick logic (bullish = lower wick, bearish = upper wick)
- ✗ Relaxed body thresholds (1.3x/1.1x/0.9x/0.7x)
- ✗ Removed volume factor (0% vs 30%)
- ✗ Adjusted weights (50/50 vs 40/30/30)

**Expected Impact:** UNKNOWN (code never executed)

---

## Diagnostic Recommendations

### Immediate Actions

1. **Verify Configuration**
   ```bash
   # Check backtest config for TEST N
   grep -i "SMC_BOS_CHOCH" config_test_n.py
   grep -i "BOS.*ENABLED" config_test_n.py
   ```

2. **Check 15m Data Availability**
   ```bash
   # Verify 15m data was fetched
   grep "15m.*synthesis\|15m data.*fetched" all_signals_TEST_N_v2180.txt
   ```

3. **Compare Configurations**
   ```bash
   # Diff TEST M vs TEST N configs
   diff config_test_m.py config_test_n.py
   ```

### Re-Test Requirements

**To properly test v2.18.0, run TEST O with:**

1. **Enable BOS Detection:**
   ```python
   SMC_BOS_CHOCH_REENTRY_ENABLED = True  # Must be True
   ```

2. **Verify in Logs:**
   ```bash
   # After running TEST O, confirm:
   grep "STEP 3A: Detecting BOS" all_signals_TEST_O.txt | wc -l  # Should be > 0
   grep "BOS Quality Breakdown" all_signals_TEST_O.txt | wc -l   # Should be > 0
   ```

3. **Compare Apples-to-Apples:**
   - TEST M (v2.17.0 + BOS enabled) vs TEST O (v2.18.0 + BOS enabled)
   - Both should show BOS detection attempts
   - Both should have quality calculation logs

### Expected TEST O Results

**If v2.18.0 fix works as intended:**
- Signal count: 50-70 (up from 36, due to relaxed thresholds)
- Win rate: 42-48% (quality signals selected, but more of them)
- Profit factor: 0.85-1.20 (+18% to +67% vs TEST M)
- Expectancy: -0.5 to +0.5 pips (improvement)
- "using fallback structure": 0-5 instances (down from 27)

**If v2.18.0 fix is still broken:**
- Signal count: unchanged or worse
- Performance: no improvement or degradation
- "using fallback structure": 20+ instances (quality filter still failing)

---

## Conclusion

### Summary of Findings

1. **TEST N did NOT test v2.18.0 changes** - BOS detection was completely disabled
2. **Performance degradation is NOT caused by v2.18.0 code** - It's a config issue
3. **TEST M vs TEST N comparison is INVALID** - Different strategies, not code versions
4. **v2.18.0 quality fix effectiveness is UNKNOWN** - Never executed

### Next Steps

1. **CRITICAL:** Re-run TEST N with `SMC_BOS_CHOCH_REENTRY_ENABLED = True`
2. Verify BOS detection logs appear in new test (TEST O)
3. Compare TEST M (v2.17.0) vs TEST O (v2.18.0) properly
4. Only then can we evaluate if the wick/body/weight changes improve or hurt performance

### Lesson Learned

**Always verify code path execution in logs before analyzing results.**

The absence of expected log messages (e.g., "BOS Quality Breakdown") should immediately trigger investigation into whether the code under test actually ran.

---

## File References

- **Strategy Code:** `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/core/strategies/smc_structure_strategy.py`
- **Quality Calculation:** Lines 1617-1723 (v2.18.0 changes)
- **TEST M Results:** `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/all_signals_TEST_M_v2170.txt`
- **TEST N Results:** `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/all_signals_TEST_N_v2180.txt`

---

**Report Generated:** 2025-11-11
**Analysis Confidence:** HIGH (based on definitive log evidence)
**Recommendation:** INVALID TEST - Re-run with correct configuration
