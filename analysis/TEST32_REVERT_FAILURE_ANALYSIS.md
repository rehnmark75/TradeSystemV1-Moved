# TEST 32 - Revert to v2.4.0 FAILED ❌

**Test Date**: 2025-11-09
**Strategy Version**: v2.4.0 (after revert from v2.6.0)
**Test Period**: 30 days (October 2025)
**Expected**: Restore v2.4.0 baseline performance
**Result**: **CATASTROPHIC FAILURE - WORSE THAN MACD FILTER**

---

## 🚨 CRITICAL ISSUE

**Expected Performance** (Test 27, v2.4.0 baseline):
- 32 signals, 40.6% WR, 1.55 PF, +3.2 pips exp, 22.2 pips avg win

**Actual Performance** (Test 32, after revert):
- **68 signals** (+113% ❌)
- **29.4% WR** (-28% ❌)
- **0.41 PF** (-74% ❌)
- **-4.3 pips exp** (-234% ❌)
- **9.9 pips avg win** (-55% ❌)

**This is WORSE than the MACD filter (PF: 0.42 vs 0.41)**

---

## 📊 PERFORMANCE COMPARISON

| Metric | Test 27 (v2.4.0) | Test 32 (Revert) | Change | Status |
|--------|------------------|------------------|--------|--------|
| **Signals** | 32 | **68** | **+113%** | ❌ **MASSIVE INCREASE** |
| **Win Rate** | 40.6% | **29.4%** | **-28%** | ❌ **CATASTROPHIC** |
| **Profit Factor** | 1.55 | **0.41** | **-74%** | ❌ **WORST EVER** |
| **Expectancy** | +3.2 pips | **-4.3 pips** | **-234%** | ❌ **WORST EVER** |
| **Avg Win** | 22.2 pips | **9.9 pips** | **-55%** | ❌ **DESTROYED** |
| **Avg Loss** | 9.8 pips | 10.2 pips | +4% | ❌ Slightly worse |
| **Winners** | 13 | 20 | +54% | More winners |
| **Losers** | 19 | **48** | **+153%** | ❌ **MASSIVE INCREASE** |
| **Bull/Bear** | 78%/22% | 51%/49% | More balanced | Good |
| **Avg Confidence** | 53.2% | 59.9% | +13% | Higher but worse |
| **Monthly P/L** | **+102 pips** | **-292 pips** | **-394 pips** | ❌ **CATASTROPHIC** |

---

## ❌ ROOT CAUSE ANALYSIS

### Problem 1: Signal Count DOUBLED (32 → 68)

**This means a critical filter is DISABLED or BROKEN**

Possible causes:
1. **MIN_BOS_QUALITY not being applied** (should filter 65% of signals)
2. **MIN_CONFIDENCE not being applied** (should filter 45% floor)
3. **Premium/Discount filter disabled**
4. **HTF alignment broken**
5. **Order Block re-entry logic broken**

### Problem 2: Win Rate COLLAPSED (40.6% → 29.4%)

**With MORE signals AND lower win rate = quality filter failure**

### Problem 3: Profit Factor WORST EVER (0.41)

**Even worse than MACD filter (0.42)**
- This is the WORST performance in all 6 tests
- Something is fundamentally broken

### Problem 4: Average Win DESTROYED (22.2 → 9.9 pips)

**Same pattern as all failed filters** (-55% winner quality)
- But NO filter should be active after revert
- This suggests code issue, not configuration

---

## 🔍 DIAGNOSTIC QUESTIONS

### Question 1: Are Quality Filters Being Applied?

**Expected behavior** (v2.4.0):
```python
MIN_BOS_QUALITY = 0.65  # Should reject 35% of BOS signals
MIN_CONFIDENCE = 0.45   # Should reject low-confidence entries
```

**Test 32 has 68 signals vs Test 27's 32** = filters NOT working

### Question 2: Is Order Block Re-entry Working?

**Expected**: Wait for pullback to OB, enter at rejection
**Test 32 avg win: 9.9 pips** (same as MACD filter) = late entries OR no OB logic

### Question 3: Did Revert Actually Complete?

**Need to verify**:
1. No MACD filter code executing
2. No EMA filter code executing
3. Quality filters (0.65, 0.45) executing
4. OB re-entry logic executing

### Question 4: Is There a Code Path Issue?

**Hypothesis**: Code may have a fallback path that bypasses filters when config is missing

---

## 🔬 REQUIRED INVESTIGATION

### Step 1: Check Filter Execution Logs

```bash
grep -c "MIN_CONFIDENCE\|MIN_BOS_QUALITY" all_signals32_fractals13.txt
```

Expected: Should see rejections like:
- "Signal confidence too low: XX% < 45%"
- "BOS quality too low: XX% < 65%"

### Step 2: Check MACD/EMA Filter Execution

```bash
grep -c "MACD\|EMA.*Filter" all_signals32_fractals13.txt
```

Expected: 0 matches (filters removed)

### Step 3: Check Order Block Logic

```bash
grep -c "Order Block\|OB re-entry" all_signals32_fractals13.txt
```

Expected: Should see OB detection and re-entry logic

### Step 4: Compare Code to Test 27

**Need to identify**: What changed between Test 27 (working) and Test 32 (broken)?

---

## 💡 HYPOTHESES

### Hypothesis 1: Filters Not Loading from Config ⚠️

**Evidence**:
- 68 signals vs 32 baseline = filters not working
- Config shows correct values (0.65, 0.45)
- Code may not be reading config correctly

**Test**: Add logging to verify filter thresholds loaded

### Hypothesis 2: Filter Code Path Broken ⚠️

**Evidence**:
- Revert removed MACD filter code
- May have accidentally broken quality filter execution
- Need to review detect_signal() flow

### Hypothesis 3: Different Test Data 🤔

**Evidence**:
- All tests claim "October 2025, 30 days"
- But signal patterns very different
- May be testing different date range?

**Test**: Verify exact date range of Test 32 vs Test 27

### Hypothesis 4: Order Block Logic Changed 🤔

**Evidence**:
- Avg win 9.9 pips (same as MACD filter)
- This is "late entry" signature
- OB re-entry may be broken

---

## 🔧 IMMEDIATE ACTIONS NEEDED

### 1. Verify Config Loading
```bash
grep "MIN_BOS_QUALITY\|MIN_CONFIDENCE" logs
```

### 2. Check Filter Execution
```bash
grep "confidence too low\|quality too low" all_signals32_fractals13.txt
```

### 3. Compare Code to Test 27 Git State
```bash
git diff <test27_commit> HEAD -- worker/app/forex_scanner/core/strategies/smc_structure_strategy.py
```

### 4. Review Recent Code Changes
Look for changes to:
- Quality filter logic
- Confidence calculation
- BOS quality score calculation
- Order Block re-entry logic

---

## ⚠️ CRITICAL FINDINGS

### Finding 1: Performance WORSE than All Filters

| Test | Filter | PF | Expectancy | Avg Win |
|------|--------|----|-----------|----|
| 27 | None (baseline) | **1.55** | **+3.2** | **22.2** |
| 28 | 1H Momentum | 0.64 | -2.1 | 11.0 |
| 29 | EMA 50 | 0.64 | -2.0 | 8.8 |
| 30 | EMA 20 | 0.68 | -1.9 | 10.6 |
| 31 | MACD | 0.42 | -4.1 | 10.4 |
| **32** | **None (revert)** | **0.41** | **-4.3** | **9.9** |

**Test 32 is DEAD LAST** - worse than even MACD filter!

### Finding 2: Signal Count Explosion

**Signal count by test**:
- Test 27 (baseline): 32 signals ✅
- Test 28-31 (filters): 27-49 signals
- **Test 32 (revert): 68 signals** ❌ (+113%)

**This proves quality filters are NOT working**

### Finding 3: Winner Quality Same as MACD Filter

**Average win**:
- Test 27 (baseline): 22.2 pips ✅
- Test 31 (MACD): 10.4 pips ❌
- **Test 32 (revert): 9.9 pips** ❌

**Nearly identical to MACD filter** - suggests similar late entry issue

---

## 📋 NEXT STEPS

### Priority 1: URGENT - Find What Broke
1. Check if quality filters executing
2. Check if OB re-entry executing
3. Compare code to Test 27 state
4. Review all changes since Test 27

### Priority 2: Fix Quality Filters
1. Verify MIN_BOS_QUALITY = 0.65 applied
2. Verify MIN_CONFIDENCE = 0.45 applied
3. Add logging to confirm filters executing

### Priority 3: Verify OB Re-entry Logic
1. Check OB detection working
2. Check OB rejection signals
3. Verify entry at OB zone, not random

### Priority 4: Revert to Known Good State
If can't fix quickly:
- Git revert to Test 27 commit
- Restore exact code state that produced 1.55 PF

---

## 🚨 CRITICAL VERDICT

**Status**: ❌ **REVERT FAILED - CRITICAL BUG INTRODUCED**

**Performance**: 0.41 PF, -4.3 pips exp, -292 pips/month
**vs Baseline**: -74% PF, -234% expectancy, -394 pips/month worse

**This is the WORST performance across ALL 6 tests.**

**Immediate action required**: Find and fix what broke during revert.

---

**Report Generated**: 2025-11-09
**Status**: CRITICAL - SYSTEM BROKEN
**Action**: URGENT INVESTIGATION REQUIRED
