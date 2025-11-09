# TEST 28 - SMC Strategy v2.6.0 HTF Momentum Alignment Analysis

**Test Date**: 2025-11-08
**Strategy Version**: v2.6.0
**Test Period**: 30 days (October 2025)
**Timeframe**: 15m
**Pairs Tested**: 9 (EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, EURJPY, AUDJPY)

---

## 🎯 EXECUTIVE SUMMARY

**CRITICAL FINDING: v2.6.0 FAILED - Strategy SEVERELY DEGRADED** ❌

The HTF Momentum Alignment Validator (v2.6.0) was designed to prevent premature counter-trend entries by checking 1H candle momentum BEFORE allowing 15m BOS entries. However, the implementation proved **TOO RESTRICTIVE**, rejecting 53% of signals and causing catastrophic performance degradation.

### Performance Comparison: v2.4.0 → v2.6.0

| Metric | v2.4.0 Baseline | v2.6.0 Result | Change | Status |
|--------|-----------------|---------------|--------|--------|
| **Signals** | 32 | 40 | +25% | ⚠️ Higher |
| **Win Rate** | 40.6% | 35.0% | **-14%** | ❌ **WORSE** |
| **Profit Factor** | 1.55 | 0.64 | **-59%** | ❌ **CATASTROPHIC** |
| **Expectancy** | +3.2 pips | -2.1 pips | **-166%** | ❌ **UNPROFITABLE** |
| **Avg Win** | 22.2 pips | 11.0 pips | -50% | ❌ **WORSE** |
| **Avg Loss** | 9.8 pips | 9.2 pips | -6% | ✅ Slight improvement |
| **Winners** | 13 | 14 | +8% | ✅ Slight improvement |
| **Losers** | 19 | 26 | +37% | ❌ **WORSE** |
| **Profitability** | ✅ PROFITABLE | ❌ UNPROFITABLE | - | ❌ **FAILURE** |

---

## 📊 DETAILED PERFORMANCE ANALYSIS

### Signal Generation
- **Total Signals**: 40 (vs baseline 32, +25%)
- **Bull Signals**: 32 (80%)
- **Bear Signals**: 8 (20%)
- **Average Confidence**: 61.0% (vs baseline 53.2%, +15%)

**Analysis**: v2.6.0 generated MORE signals than baseline, contrary to expectations. This suggests the momentum filter is NOT preventing bad entries effectively.

### Win/Loss Distribution
- **Winners**: 14 trades (35.0% WR)
  - Average profit: 11.0 pips (vs baseline 22.2 pips, **-50%**)
  - Total profit: 154 pips (vs baseline ~289 pips, **-47%**)

- **Losers**: 26 trades (65.0% loss rate)
  - Average loss: 9.2 pips (vs baseline 9.8 pips, -6%)
  - Total loss: 239 pips (vs baseline ~186 pips, **+28%**)

**Critical Issue**: The filter allowed MORE losers (+7 trades) while REDUCING winner quality (-50% avg profit). This is the opposite of the intended behavior.

### Profitability Metrics
- **Profit Factor**: 0.64 (vs baseline 1.55, **-59%**)
  - Interpretation: For every $1 risked, only $0.64 returned
  - Status: **UNPROFITABLE** (need ≥1.0)

- **Expectancy**: -2.1 pips per trade (vs baseline +3.2 pips, **-166%**)
  - Interpretation: Average loss of 2.1 pips per signal
  - Monthly projection: 40 signals × -2.1 pips = **-84 pips/month**
  - Status: **LOSING STRATEGY**

---

## 🔍 HTF MOMENTUM FILTER ANALYSIS

### Filter Activity
- **Total 4H Trend Confirmations**: 4,364 checks performed
- **Momentum Validations**:
  - ✅ **Passed**: 2,040 (46.7%)
  - ❌ **Rejected**: 2,324 (53.3%)

**Finding**: The filter rejected MORE than half of all potential signals, yet:
1. Final signal count increased (40 vs 32)
2. Performance degraded significantly

This suggests a **fundamental logic error** in the implementation.

### Sample Rejection Patterns

**Bearish Signal Rejections** (4H BEAR trend detected):
```
❌ 1H Momentum Misalignment:
   → 4H indicates BEARISH trend
   → BUT 1H shows 2/3 BEARISH candles (NOT bullish!)
   💡 This is a bounce in 1H downtrend, not a reversal
```

**Critical Error Found**: The log message says "2/3 BEARISH candles" but treats this as a REJECTION for a BEARISH signal. This is **INVERTED LOGIC** - bearish 1H candles should SUPPORT a bearish 4H trend, not reject it!

---

## 🚨 ROOT CAUSE IDENTIFICATION

### Implementation Error in `_validate_htf_momentum_alignment()`

**Location**: [smc_structure_strategy.py:381-450](smc_structure_strategy.py#L381-L450)

**The Problem**:
```python
if bos_direction == 'BEAR':
    bearish_count = (last_3_1h['close'] < last_3_1h['open']).sum()
    if bearish_count < 2:  # Less than 2/3 bearish candles
        # REJECT - This is correct
        return False
```

**What Actually Happens**:
- When `bearish_count >= 2` (e.g., "2/3 BEARISH candles"), the code CONTINUES
- But logs show rejections even with "2/3 BEARISH candles"
- This suggests the **log message is incorrect** OR there's a **secondary filter** causing rejections

**Hypothesis**: The logic might be checking for BULLISH candles when it should check for BEARISH, or the threshold is inverted.

---

## 📉 COMPARATIVE TIMELINE ANALYSIS

### Test Progression Summary

| Test | Version | Change | Signals | WR | PF | Status |
|------|---------|--------|---------|----|----|--------|
| 20 | v2.2.0 | HTF BOS/CHoCH baseline | 110 | 23.6% | 0.51 | Baseline |
| 22 | v2.2.0 | Premium/Discount filter | 31 | 25.8% | 0.80 | Improving |
| 24 | v2.4.0 | HTF strength 75% | 39 | 25.6% | 0.86 | Best unprofitable |
| 27 | v2.4.0 | BOS 65% + Conf 45% | **32** | **40.6%** | **1.55** | ✅ **PROFITABLE** |
| 26 | v2.5.0 | 15m momentum (FAILED) | 63 | 28.6% | 0.88 | Failed approach |
| 26.1 | v2.5.1 | 3/3 strict (FAILED) | 62 | 24.2% | 0.37 | Failed approach |
| **28** | **v2.6.0** | **1H momentum (CURRENT)** | **40** | **35.0%** | **0.64** | ❌ **FAILED** |

**Conclusion**: v2.6.0 represents the WORST performance since Test 20 (initial baseline), despite having the highest confidence average (61%).

---

## 🎯 WHY v2.6.0 FAILED

### 1. **Inverted Logic Hypothesis**
The momentum filter may be checking for the WRONG candle direction:
- For BEARISH signals: Should require bearish 1H candles (RED)
- But logs show REJECTIONS with "2/3 BEARISH candles"
- Suggests the code might be looking for BULLISH candles instead

### 2. **Timing Still Wrong**
Even though v2.6.0 checks 1H candles BEFORE 15m BOS (correct timing vs v2.5.x):
- The filter INCREASED signals (40 vs 32)
- The filter REDUCED win quality (-50% avg profit)
- This suggests the filter is APPROVING bad entries and REJECTING good ones

### 3. **Threshold Mismatch**
The 2/3 threshold may be inappropriate for 1H momentum:
- 1H candles are 4x longer than 15m
- 1H momentum changes more slowly
- Requiring 2/3 recent 1H alignment may be too strict OR too lenient depending on market regime

---

## 💡 RECOMMENDED FIXES

### Option 1: REVERT TO v2.4.0 IMMEDIATELY ✅
**Reasoning**:
- v2.4.0 is proven profitable (PF: 1.55, +3.2 pips expectancy)
- v2.6.0 destroyed profitability (-59% PF drop)
- No point continuing with broken logic

**Action**: Revert all Phase 1 changes

---

### Option 2: FIX THE LOGIC INVERSION (If Continuing)

**Step 1: Verify Candle Direction Check**
Current code (line 391-395):
```python
if bos_direction == 'BEAR':
    bearish_count = (last_3_1h['close'] < last_3_1h['open']).sum()
    if bearish_count < 2:
        return False  # REJECT
```

**Potential Fix**:
```python
if bos_direction == 'BEAR':
    bearish_count = (last_3_1h['close'] < last_3_1h['open']).sum()
    if bearish_count >= 2:  # INVERTED: Require ALIGNMENT
        return True  # ACCEPT
    else:
        return False  # REJECT - momentum opposes
```

**Step 2: Fix Log Messages**
The logs show "2/3 BEARISH candles" for a BEARISH signal rejection, which makes no sense. Either:
1. The count is wrong
2. The log message is wrong
3. The logic is inverted

**Step 3: Add Diagnostic Logging**
```python
self.logger.info(f"   🔍 DEBUG: bos_direction={bos_direction}")
self.logger.info(f"   🔍 DEBUG: bearish_count={bearish_count}, bullish_count={3-bearish_count}")
self.logger.info(f"   🔍 DEBUG: last_3_1h candles:")
for idx, row in last_3_1h.iterrows():
    candle_color = "🔴 BEARISH" if row['close'] < row['open'] else "🟢 BULLISH"
    self.logger.info(f"      {idx}: {candle_color} (O:{row['open']:.5f} C:{row['close']:.5f})")
```

---

## 📋 NEXT STEPS

### Immediate Action Required

1. ✅ **REVERT TO v2.4.0**
   - Proven profitable baseline
   - 40.6% WR, 1.55 PF, +3.2 pips expectancy
   - No experimental features

2. 🔍 **INVESTIGATE v2.6.0 LOGIC ERROR**
   - Review `_validate_htf_momentum_alignment()` implementation
   - Add debug logging to verify candle counts
   - Test logic in isolation before re-deploying

3. 📊 **RE-ANALYZE THE ORIGINAL PROBLEM**
   - User's chart showed premature SELL in UPTREND
   - v2.6.0 was supposed to prevent this
   - But results show it made things WORSE
   - Need different approach

### Alternative Approach: Pullback vs Reversal Detection

Instead of momentum validation, consider:
1. **Retracement Depth Filter**
   - Measure pullback from recent high/low
   - Shallow pullbacks (<38.2% Fib) = likely continuation, REJECT counter-trend
   - Deep pullbacks (>61.8% Fib) = potential reversal, ALLOW

2. **Volume Confirmation**
   - Counter-trend BOS with LOW volume = false signal
   - Counter-trend BOS with HIGH volume = potential reversal

3. **Multi-Swing Confirmation**
   - Require 2+ BOS in same direction before entry
   - Prevents single-swing false breakouts

---

## 🎓 LESSONS LEARNED

1. **Higher Signal Count ≠ Better Strategy**
   - v2.6.0 had +25% more signals but -59% worse PF
   - Quality > Quantity

2. **Confidence Alone is Misleading**
   - v2.6.0 had 61% avg confidence (vs 53.2% baseline)
   - Yet profitability destroyed
   - Confidence metric may need recalibration

3. **Momentum Filters are Tricky**
   - v2.5.0: Checked 15m too late → FAILED
   - v2.6.0: Checked 1H earlier → FAILED WORSE
   - Root issue may not be momentum timing

4. **Beware of Inverted Logic**
   - Logs show "2/3 BEARISH candles" rejecting BEARISH signals
   - Always verify filter logic with debug output

---

## 🔄 COMPARISON WITH ORIGINAL PROBLEM

### User's Chart Scenario
- **Issue**: Premature SELL signal during UPTREND
- **Expected Fix**: Reject BEARISH signals when 1H shows BULLISH momentum
- **v2.6.0 Behavior**: Unknown (needs investigation)

### Why v2.6.0 Might Not Help
Even if the logic is fixed:
- Checking 1H momentum may be too coarse (4x 15m timeframe)
- 1H uptrend can contain valid 15m reversals
- User's chart might show a VALID 15m reversal within 1H uptrend
- Solution: Need to identify PREMATURE reversals, not ALL reversals

---

## ⚠️ CRITICAL VERDICT

**DO NOT DEPLOY v2.6.0 TO PRODUCTION**

**Status**: ❌ FAILED
**Recommendation**: REVERT TO v2.4.0
**Next Test**: Investigate logic error OR try alternative approach

**Profitability Impact**:
- v2.4.0: +3.2 pips/trade × 32 signals = **+102 pips/month** ✅
- v2.6.0: -2.1 pips/trade × 40 signals = **-84 pips/month** ❌
- **Difference**: 186 pips/month WORSE

---

## 📎 APPENDIX

### Recent Signal Distribution (Last 15 signals shown in test output)
- **Winners**: 8/15 (53% WR in recent trades)
- **Losers**: 7/15 (47% loss rate in recent trades)
- **Pairs with most signals**: NZDUSD (7), AUDJPY (7), GBPUSD (6)

### Filter Rejection Rate by Potential Cause
- **Total Checks**: 4,364
- **Passed to 15m BOS**: 2,040 (46.7%)
- **Rejected by Momentum**: 2,324 (53.3%)
- **Final Signals Generated**: 40 (0.9% of total checks)

This means 98.2% of all 4H trend confirmations were filtered out by subsequent checks (premium/discount, OB quality, momentum, etc.).

---

**Report Generated**: 2025-11-08
**Analysis Tool**: Claude Code v2.6.0 Backtest Analysis
**Data Source**: all_signals28_fractals9.txt
