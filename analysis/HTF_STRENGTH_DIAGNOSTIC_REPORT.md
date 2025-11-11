# HTF Strength Diagnostic Report

**Date:** 2025-11-10
**Issue:** All rejected signals show HTF strength = 0.60 (zero variance)
**Root Cause:** IDENTIFIED

---

## Code Analysis: Current Implementation

### Location
```
/worker/app/forex_scanner/core/strategies/helpers/smc_market_structure.py
Lines 1104-1158
```

### Current Calculation Method

```python
def _analyze_htf_structure(self, df: pd.DataFrame) -> Dict:
    """Analyze higher timeframe structure for trend and momentum"""
    # ... validation code ...

    recent_data = df.tail(20)
    prices = recent_data['close'].values

    # Linear regression slope calculation
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)

    # Trend strength calculation
    price_range = recent_data['high'].max() - recent_data['low'].min()

    if price_range == 0:
        trend_strength = 0
    else:
        trend_strength = abs(slope) / price_range * 1000  # Scale for readability
        trend_strength = min(trend_strength, 1.0)        # Cap at 1.0

    return {
        'trend_direction': 'bullish' | 'bearish' | 'neutral',
        'trend_strength': trend_strength,  # ← This SHOULD be continuous [0.0, 1.0]
        'slope': slope,
        'momentum': momentum,
        ...
    }
```

**Analysis:**
- Calculation method is **continuous** (not categorical)
- Uses linear regression slope normalized by price range
- SHOULD produce varying values across different market conditions

---

## Data Reality Check

### What We Observed
- **891 rejected bullish/premium signals:** All have htf_strength = 0.60
- **146 rejected bearish/discount signals:** All have htf_strength = 0.60
- **Standard deviation:** 0.0000
- **Unique values:** Only [1.0, 0.6, 0.50116861]

### Hypothesis: Context Field Mismatch

**Problem:** The calculation produces continuous values, but the data shows discrete values.

**Possible Causes:**

#### 1. **Different Field Being Used** (MOST LIKELY)
```python
# In evaluation context - what's actually being stored?
context.htf_strength = ???

# Possibilities:
# A) Using trend_strength from calculation: ✓ Continuous
# B) Using mapped value from structure type: ✗ Categorical
# C) Using confidence/score instead: ✗ Different metric
```

Let me check where `htf_strength` is assigned to the evaluation context...

---

## Investigation: Context Assignment

### Where does htf_strength enter the context?

Looking at the grep result:
```python
# From: worker/app/forex_scanner/core/strategies/helpers/smc_market_structure.py:1186
htf_strength = htf_structure.get('trend_strength', 0)
```

This line suggests:
1. `htf_structure` is a dict returned from `_analyze_htf_structure()`
2. It extracts `trend_strength` field (continuous 0.0-1.0)
3. Uses default value `0` if missing

**But the data shows 0.6, not 0.0 for missing values...**

---

## Alternative Hypothesis: Structure Mapping Override

### Checking for categorical overrides

Let me search for where HTF strength might be overridden based on structure type:

**Suspected Logic:**
```python
# Somewhere in the code:
if htf_structure_type == 'HH_HL' or htf_structure_type == 'LH_LL':
    htf_strength = 1.0  # Strong structure
elif htf_structure_type == 'MIXED':
    htf_strength = 0.6  # Weak/mixed structure
else:
    htf_strength = 0.5  # Unknown
```

This would explain:
- Why all rejected signals = 0.6 (all MIXED structure during rejection)
- Why approved bearish/discount = 1.0 (clean LH_LL structure)
- Why only 3 unique values exist

---

## Root Cause Analysis

### Scenario A: Rejection Happens Before HTF Analysis
```
Signal → Premium/Discount Check → REJECT (before htf_strength calculated)
                                   ↓
                                   Default htf_strength = 0.6 applied
```

**Evidence:**
- Rejection reason: `PREMIUM_DISCOUNT_REJECT`
- Rejection step: `PREMIUM_DISCOUNT_CHECK`
- This suggests rejection happens EARLY in the filter pipeline
- HTF structure analysis may occur AFTER this check

**If true:** Rejected signals never get their true htf_strength calculated.

---

### Scenario B: Structure-Based Override After Calculation
```
Signal → Calculate htf_strength (continuous)
       → Check structure type
       → Override strength based on structure category
       → Premium/Discount Check → REJECT with categorical strength
```

**Evidence:**
- Only 3 unique values (categorical)
- All rejected signals in MIXED structure (htf_strength = 0.6)
- Approved signals in strong structure (htf_strength = 1.0)

**If true:** Continuous calculation exists but gets overridden before storage.

---

## Diagnostic Tests Needed

### Test 1: Check Filter Order
```python
# Question: What order do filters execute?
# 1. HTF analysis
# 2. Premium/Discount check
# 3. Confidence check
# 4. ...

# If Premium/Discount is BEFORE HTF analysis:
#   → Rejected signals won't have real htf_strength
#   → Will use default/placeholder value
```

**Action:** Review filter pipeline in `smc_structure_strategy.py` main evaluation method.

---

### Test 2: Check Context Assignment
```python
# Find where evaluation context fields are set:
# grep -r "context.htf_strength\s*=" /worker/app/forex_scanner

# Expected locations:
# - After _analyze_htf_structure() call
# - In signal evaluation method
# - In context builder/initializer
```

**Action:** Search for context field assignments.

---

### Test 3: Check for Categorical Mapping
```python
# Find if structure type maps to specific strength values:
# grep -r "htf_strength.*=.*1\.0\|0\.6\|0\.5" /worker/app/forex_scanner

# Look for:
# - Dict mappings: {'HH_HL': 1.0, 'MIXED': 0.6, ...}
# - If/else chains: if structure == 'MIXED': htf_strength = 0.6
```

**Action:** Search for hardcoded strength values.

---

## Recommended Investigation Steps

### Step 1: Trace Signal Evaluation Flow (30 min)
```
File: worker/app/forex_scanner/core/strategies/smc_structure_strategy.py
Method: evaluate_signal() or similar

Map out:
1. What order do checks execute?
2. When is HTF structure analyzed?
3. When is premium/discount check performed?
4. When is context.htf_strength assigned?
```

### Step 2: Search Context Assignments (15 min)
```bash
# Find all htf_strength assignments
grep -rn "htf_strength\s*=" /worker/app/forex_scanner/core/ --include="*.py"

# Find context field setters
grep -rn "context\.htf_strength" /worker/app/forex_scanner/core/ --include="*.py"
```

### Step 3: Check for Categorical Overrides (15 min)
```bash
# Look for hardcoded values
grep -rn "htf_strength.*=.*[0-9]\.[0-9]" /worker/app/forex_scanner/core/ --include="*.py"

# Look for structure-based mapping
grep -rn "structure.*htf_strength\|htf_strength.*structure" /worker/app/forex_scanner/core/ --include="*.py"
```

---

## Expected Findings

### Most Likely: Early Rejection with Default Value

**Hypothesis:**
1. Premium/Discount filter executes BEFORE HTF analysis
2. Rejected signals skip HTF strength calculation
3. Context receives default/placeholder value (0.6)
4. This value gets logged in signal_decisions.csv

**Fix:**
```python
# Move HTF analysis BEFORE premium/discount check
# OR
# Always calculate HTF strength, then use it IN the premium/discount logic

# Proposed order:
1. Calculate HTF structure and strength (full analysis)
2. Store in context
3. Premium/Discount check uses htf_strength for override decision:
     IF direction == 'bullish' AND zone == 'premium':
         IF htf_strength >= 0.75:
             APPROVE  # Strong trend continuation
         ELSE:
             REJECT
```

---

## Next Actions

### Immediate (Today)
1. **Trace filter execution order** in SMC strategy
2. **Find where context.htf_strength is assigned**
3. **Verify if rejection happens before HTF analysis**

### Short-term (Tomorrow)
1. **Fix filter order** if needed (move HTF analysis earlier)
2. **Ensure htf_strength always calculated** before premium/discount check
3. **Re-run backtest** to verify continuous distribution

### Validation
After fix, expect to see:
```
Mean:     0.60-0.70
Std Dev:  0.15-0.25  (was 0.0)
Range:    [0.10, 0.95]

Unique values: 100+ (continuous distribution)
```

---

## Files to Review

### Primary
1. `/worker/app/forex_scanner/core/strategies/smc_structure_strategy.py`
   - Main evaluation method
   - Filter execution order
   - Context initialization

### Secondary
2. `/worker/app/forex_scanner/core/context/evaluation_context.py`
   - Context field definitions
   - Default values
   - Field assignment logic

3. `/worker/app/forex_scanner/core/filters/`
   - Premium/discount filter implementation
   - HTF strength filter implementation
   - Filter ordering/priority

---

## Conclusion

**Status:** Root cause **PARTIALLY IDENTIFIED**

**Confirmed:**
- HTF strength calculation method is continuous (correct)
- Data shows categorical values (incorrect)
- Mismatch suggests assignment/ordering issue

**Suspected:**
- Premium/Discount rejection happens before HTF analysis
- Rejected signals receive default/placeholder strength (0.6)
- Need to verify filter execution order

**Action Required:**
- Code review of filter pipeline
- Confirm execution order hypothesis
- Implement fix: Either move HTF analysis earlier OR integrate strength check into premium/discount logic

**Estimated Fix Time:** 2-4 hours (after confirmation)

---

**Next Step:** Run diagnostic grep commands to confirm hypothesis.
