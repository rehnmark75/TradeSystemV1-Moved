# TEST J Failure Analysis - v2.14.0 R:R Fix Did NOT Work

**Date**: 2025-11-10
**Version**: v2.14.0
**Test Period**: Oct 6 - Nov 5, 2025 (30 days)
**Status**: ❌ FAILED - Unprofitable

---

## Executive Summary

TEST J (v2.14.0) implemented a 2.0R minimum R:R ratio to address unprofitability caused by partial TP math. **The fix FAILED to achieve profitability** despite correctly calculating 2.0R+ targets.

### Performance Results

| Metric | TEST I (v2.13.0) | TEST J (v2.14.0) | Change | Target | Status |
|--------|------------------|------------------|---------|--------|--------|
| **Signals** | 42 | 38 | -9.5% | 25-35 | ✅ |
| **Win Rate** | 42.9% | 44.7% | +1.8pp | 40-45% | ✅ |
| **Profit Factor** | 0.84 | **0.91** | +8.3% | **1.2-1.8** | ❌ |
| **Avg Winner** | 10.9 pips | **10.9 pips** | **0.0%** | **15-20 pips** | ❌ |
| **Avg Loser** | 9.7 pips | 9.7 pips | 0.0% | - | - |
| **Expectancy** | -0.9 pips | **-0.5 pips** | +44% | **Positive** | ❌ |

### Key Finding

**CRITICAL**: Average winner stayed at 10.9 pips despite 2.0R minimum requirement. The R:R calculation is correct (targets set at 2.0R+) but **actual trade execution never reaches those targets**.

---

## Root Cause Analysis

### Expected vs Actual Behavior

**Expected** (with 2.0R minimum):
- Stop Loss: ~10 pips (1.5× ATR)
- Take Profit: ~20 pips (2.0R)
- Partial TP: 50% at 10 pips (1.0R)
- Remaining 50%: At 20 pips (2.0R)
- **Average Winner**: (0.5 × 10) + (0.5 × 20) = 15 pips

**Actual** (TEST J results):
- Average Winner: **10.9 pips** (same as TEST I)
- This indicates winners are exiting around 1.0-1.1R
- **Conclusion**: Second 50% position NEVER reaches 2.0R target

### Why Winners Are Small

Three possible causes:

#### 1. **Trailing Stop Premature Exit** (Most Likely)
- Breakeven stop triggered at 1.5R (default in [trailing_stop_simulator.py:83-85](../worker/app/forex_scanner/core/trading/trailing_stop_simulator.py#L83-L85))
- After 50% partial at 1.0R, remaining 50% moved to breakeven
- Price reverses before reaching 2.0R target
- Trade exits at breakeven = 0.5R total profit
- Math: (50% × 1.0R) + (50% × 0R) = 0.5R = ~5 pips
- Many trades exit around 10 pips suggests partial TP + small gains on remainder

#### 2. **Market Doesn't Reach 2.0R Targets**
- 2.0R targets may be too aggressive for 15m entries
- Price moves 1.0-1.5R then reverses
- Trailing stop or BE stop catches trades before 2.0R

#### 3. **Simulation Bug** (Less Likely)
- TP targets calculated correctly (confirmed in logs)
- But simulation may not be applying full TP correctly
- However, trailing_stop_simulator.py unchanged since baseline

---

## Evidence

### R:R Ratios Calculated Correctly

From TEST J logs:
```
R:R Ratio: 2.00 ✅
R:R Ratio: 3.99 ✅
R:R Ratio: 2.00 ✅
```

All signals meet the 2.0R minimum requirement during signal generation.

### Average Winner Unchanged

| Test | Min R:R | Avg Winner | Expected Winner |
|------|---------|------------|-----------------|
| I | 1.2 | 10.9 pips | 10-12 pips |
| J | 2.0 | 10.9 pips | 15-20 pips |

**Gap**: 4-9 pips missing per winner (45-82% shortfall)

### Partial TP Configuration

```python
# config_smc_structure.py
SMC_PARTIAL_PROFIT_ENABLED = True
SMC_PARTIAL_PROFIT_PERCENT = 50
SMC_PARTIAL_PROFIT_RR = 1.0
SMC_MOVE_SL_TO_BE_AFTER_PARTIAL = True  # ← CULPRIT?
```

**Hypothesis**: After partial TP at 1.0R, stop moves to breakeven. Price then reverses before reaching 2.0R, exiting at BE for 0R on second half.

---

## Mathematical Proof of Unprofitability

### Current Reality (TEST J)

```
Win Rate: 44.7%
Average Winner: 10.9 pips
Average Loser: 9.7 pips

Expectancy = (0.447 × 10.9) - (0.553 × 9.7)
           = 4.87 - 5.36
           = -0.49 pips per trade ❌
```

### Required for Profitability

At 44.7% win rate, breakeven requires:
```
(0.447 × W) = (0.553 × 9.7)
W = 5.36 / 0.447
W = 11.99 pips

Required average winner: 12.0 pips (currently: 10.9)
Shortfall: 1.1 pips (9%)
```

### With 2.0R Average Winners

If second 50% actually reached 2.0R:
```
Average Winner = (0.5 × 10) + (0.5 × 20) = 15 pips

Expectancy = (0.447 × 15) - (0.553 × 9.7)
           = 6.71 - 5.36
           = +1.35 pips per trade ✅ PROFITABLE
```

---

## Why The Fix Failed

### The 2.0R Fix Changed:
1. ✅ Signal filtering (fewer signals: 42 → 38)
2. ✅ Win rate improved (+1.8pp)
3. ✅ R:R calculations (all targets ≥2.0R)

### The 2.0R Fix Did NOT Change:
1. ❌ Trailing stop logic (still 1.5R breakeven trigger)
2. ❌ Partial TP behavior (still 50% at 1.0R + BE move)
3. ❌ Actual profit capture (still 10.9 pips average)

### The Problem

**Increasing the R:R target from 1.2 to 2.0 had ZERO effect on profit capture** because:
- Trades are exiting via trailing stops/BE stops BEFORE reaching targets
- The target is set correctly, but price never gets there
- Moving the goalpost further away doesn't help if you can't reach the original one

---

## Diagnostic Evidence

### Trailing Stop Configuration

From [trailing_stop_simulator.py:83-85](../worker/app/forex_scanner/core/trading/trailing_stop_simulator.py#L83-L85):
```python
breakeven_trigger = initial_stop_pips * 1.5  # Move to BE at 1.5R
```

**Timeline of Typical Trade**:
1. Entry at price X
2. Stop at X - 10 pips (1R)
3. Price moves +10 pips → Partial TP triggered (50% closed at 1.0R)
4. Stop moved to breakeven (X) per config
5. Price moves to +15 pips (1.5R) → BE trigger activated
6. Price reverses to +11 pips
7. **Trade exits at +11 pips total** (0.5R partial + 0.1R remainder)
8. **Result**: 11 pips profit (vs 15 pips if 1.5R captured, 20 pips if 2.0R captured)

This matches the observed 10.9 pip average.

### Comparison to Baseline

**Baseline (v2.4.0)** had 22.2 pips average winner with:
- Min R:R: 1.2
- Same partial TP settings
- Same trailing stop logic

**Question**: Why did baseline capture 2x the profit (22.2 vs 10.9)?

**Hypothesis**: Baseline had DIFFERENT entry conditions that led to stronger momentum moves reaching targets. Current version's signals may be weaker quality despite higher win rate.

---

## Conclusions

### Primary Root Cause

**Trailing Stop + Breakeven Management is incompatible with 2.0R targets.**

The current system:
1. Takes 50% profit at 1.0R ✅
2. Moves stop to breakeven ✅
3. Sets target at 2.0R ✅
4. **Price never reaches 2.0R before reversing** ❌

### Secondary Issues

1. **15m timeframe insufficient for 2.0R moves**: May need 1H entries for larger moves
2. **Signal quality degraded**: Despite 44.7% WR, signals lack momentum to reach targets
3. **Partial TP timing wrong**: Taking 50% at 1.0R may be too early

---

## Next Steps - Three Options

### Option 1: Disable Partial TP (Simplest)

**Change**:
```python
SMC_PARTIAL_PROFIT_ENABLED = False
SMC_MIN_RR_RATIO = 2.0  # Keep 2.0R requirement
```

**Expected**:
- Winners: All-or-nothing at 2.0R target
- Some trades will hit 2.0R (full 20 pips)
- Some will hit trailing stop (15 pips at 1.5R BE trigger)
- **Expected Avg Winner**: 15-18 pips (vs 10.9 current)
- **Trade-off**: Lower win rate (40% → 35-38%)

**Profitability Check**:
```
At 37% WR, 17 pips avg winner:
(0.37 × 17) - (0.63 × 9.7) = 6.29 - 6.11 = +0.18 pips ✅ Barely profitable
```

### Option 2: Adjust Breakeven Trigger (Moderate)

**Change**:
```python
# In trailing_stop_simulator.py
breakeven_trigger = initial_stop_pips * 2.5  # Move to BE at 2.5R (was 1.5R)
```

**Expected**:
- Allows price to reach 2.0R before BE protection
- Partial TP still at 1.0R
- Remaining 50% can reach 2.0R without premature BE exit
- **Expected Avg Winner**: 13-15 pips
- **Trade-off**: More risk (later BE protection)

**Profitability Check**:
```
At 44% WR, 14 pips avg winner:
(0.44 × 14) - (0.56 × 9.7) = 6.16 - 5.43 = +0.73 pips ✅ Profitable
```

### Option 3: Revert to Baseline Logic (Nuclear)

**Change**:
- Revert ALL strategy changes back to v2.4.0 (Test 27 baseline)
- Abandon improvement attempts

**Expected**:
- Restore 22.2 pip winners
- Restore 1.55 profit factor
- Restore +3.2 pips expectancy
- **Trade-off**: Give up on understanding/improving strategy

---

## Recommendation

**OPTION 2: Adjust Breakeven Trigger to 2.5R**

### Rationale

1. **Least invasive**: Single parameter change
2. **Preserves wins**: Keeps partial TP for risk management
3. **Allows targets**: Gives price room to reach 2.0R
4. **Testable**: Can validate if BE timing is the blocker

### Implementation

[trailing_stop_simulator.py:83](../worker/app/forex_scanner/core/trading/trailing_stop_simulator.py#L83):
```python
# Before:
breakeven_trigger = initial_stop_pips * 1.5

# After:
breakeven_trigger = initial_stop_pips * 2.5  # Allow 2.0R targets to be reached
```

### Expected TEST K Results

```
Signals: 35-40 (similar to TEST J)
Win Rate: 42-45% (maintain quality)
Avg Winner: 13-15 pips (+20-38% from TEST J)
Avg Loser: 9.7 pips (unchanged)
Profit Factor: 1.1-1.3 (PROFITABLE)
Expectancy: +0.5 to +1.5 pips (POSITIVE)
```

---

## Alternative: If Option 2 Fails

If adjusting BE trigger still doesn't achieve profitability, the issue is **signal quality**, not trade management.

**Next Investigation**:
1. Compare TEST J signals to baseline v2.4.0 signals
2. Identify what changed in signal generation
3. Root cause why current signals lack momentum despite higher WR
4. May need to revert to baseline and start over with smaller changes

---

**Test**: TEST J (v2.14.0)
**Result**: FAILED - PF 0.91, Expectancy -0.5 pips
**Root Cause**: Trailing stop + BE management prevents reaching 2.0R targets
**Next Test**: TEST K - Adjust BE trigger to 2.5R
**Expected**: Profitable (PF 1.1-1.3, Expectancy +0.5 to +1.5)
