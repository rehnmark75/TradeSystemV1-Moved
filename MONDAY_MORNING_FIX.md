# Monday Morning Data Gap Fix

**Date**: January 26, 2026
**Issue**: Scanner fails on Monday mornings with "insufficient 1h data" errors
**Status**: ✅ FIXED with dynamic lookback calculation

---

## Problem

Every Monday morning after the weekend, the scanner would fail for all 9 pairs:

```
⚠️ Insufficient 1h data for CS.D.EURUSD.CEEM.IP (got 11 bars, need 40 for 30 EMA)
⚠️ Insufficient 1h data for CS.D.GBPUSD.MINI.IP (got 11 bars, need 25 for 15 EMA)
```

### Root Cause

1. Markets close Friday ~10pm UTC and reopen Monday ~12am UTC (~60 hours gap)
2. Monday morning at 6:45am CET = only ~6 hours of trading data available
3. Scanner was hardcoded to fetch **48 hours lookback** for 1h timeframe
4. 48 hours lookback only gets Monday's 6 hours (not enough for EMA calculations)
5. Need 20-40 1h bars depending on pair's EMA configuration

---

## Solution: Dynamic Lookback Calculation

### Previous Approach (BROKEN)
```python
# Hardcoded lookback - doesn't account for weekends
htf_lookback = 48  # Only 2 days
```

### New Approach (FIXED)
```python
# Calculate based on actual EMA requirements
min_htf_bars = ema_period + 10  # e.g., 30 EMA → 40 bars
required_hours = min_htf_bars * 1  # 40 bars × 1h = 40 hours trading time

# Add 2.5x multiplier for weekend gaps (~60% market uptime)
htf_lookback = max(int(required_hours * 2.5), 168)  # Minimum 7 days
```

### How It Works

| Pair   | EMA | Min Bars | Required | Multiplier | Result | Days |
|--------|-----|----------|----------|------------|--------|------|
| EURJPY | 30  | 40 bars  | 40h      | × 2.5      | 168h   | 7.0  |
| GBPUSD | 15  | 25 bars  | 25h      | × 2.5      | 168h   | 7.0  |
| USDCAD | 10  | 20 bars  | 20h      | × 2.5      | 168h   | 7.0  |

**All pairs use minimum 168 hours (7 days) to always cover weekends.**

---

## Why This Prevents Future Issues

✅ **Dynamic**: Automatically adjusts if pair EMA configurations change
✅ **Weekend-proof**: 7-day minimum always covers Fri → Mon gap
✅ **Future-proof**: Works for any EMA period (10-50)
✅ **No manual updates**: No more hardcoded values to remember

### Example Scenarios

**Monday 6am after weekend:**
- Lookback: 168h (7 days) → goes back to previous Tuesday
- Includes all of last week's trading data
- ✅ Has enough data for all EMA calculations

**Friday evening:**
- Lookback: 168h (7 days) → goes back to previous Friday
- Includes full week of data
- ✅ Has enough data for all EMA calculations

**If someone adds 50 EMA pair in future:**
- Calculation: 60 bars × 1h × 2.5 = 150h
- Uses: max(150, 168) = 168h (7-day minimum)
- ✅ Still covers weekends automatically

---

## Verification

```bash
# Check for errors (should return 0)
docker logs task-worker 2>&1 | grep "insufficient.*1h" | wc -l

# Verify all 9 pairs scanning (should show all pairs)
docker logs task-worker 2>&1 | tail -100 | grep "Pair:" | tail -9
```

**Expected Result**: All 9 pairs scan successfully with 0 errors.

---

## Files Changed

- `worker/app/forex_scanner/core/signal_detector.py` (lines 282-321)
  - Replaced hardcoded 48h/120h with dynamic calculation
  - Added weekend buffer multiplier (2.5x)
  - Enforced 168h (7-day) minimum
  - Added debug logging for transparency

## Commit

```
commit 4fb659a
fix(scanner): dynamic lookback prevents Monday morning data gaps
```

---

## History

- **Jan 19, 2026**: First occurrence, fixed with hardcoded 120h
- **Jan 26, 2026**: Issue recurred, fixed with dynamic calculation
- **Future**: Should never happen again (automated solution)
