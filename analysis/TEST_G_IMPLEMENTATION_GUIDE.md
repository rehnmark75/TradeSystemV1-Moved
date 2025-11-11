# TEST G (v2.11.0) - Implementation Guide

**Test**: Lower HTF Strength Threshold from 0.75 to 0.55
**Rationale**: 58.9% of baseline profitable signals occurred at 0.6 strength
**Expected Outcome**: Restore continuation trade signals, PF 1.2-1.6

---

## Code Changes Required

### File: `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/core/strategies/smc_structure_strategy.py`

**Change 1: Line 926 (Bearish/Discount threshold)**

```python
# BEFORE (v2.10.0):
if final_trend == 'BEAR' and final_strength >= 0.75:

# AFTER (v2.11.0):
if final_trend == 'BEAR' and final_strength >= 0.55:
```

**Change 2: Line 953 (Bullish/Premium threshold)**

```python
# BEFORE (v2.10.0):
if final_trend == 'BULL' and final_strength >= 0.75:

# AFTER (v2.11.0):
if final_trend == 'BULL' and final_strength >= 0.55:
```

### File: `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/configdata/strategies/config_smc_structure.py`

**Change 3: Version update**

```python
# BEFORE:
STRATEGY_VERSION = "2.10.0"

# AFTER:
STRATEGY_VERSION = "2.11.0"
```

---

## Exact Implementation Commands

```bash
# 1. Modify strategy file - bearish/discount threshold
sed -i 's/if final_trend == '\''BEAR'\'' and final_strength >= 0\.75:/if final_trend == '\''BEAR'\'' and final_strength >= 0.55:/' \
  /home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/core/strategies/smc_structure_strategy.py

# 2. Modify strategy file - bullish/premium threshold
sed -i 's/if final_trend == '\''BULL'\'' and final_strength >= 0\.75:/if final_trend == '\''BULL'\'' and final_strength >= 0.55:/' \
  /home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/core/strategies/smc_structure_strategy.py

# 3. Update config version
sed -i 's/STRATEGY_VERSION = "2.10.0"/STRATEGY_VERSION = "2.11.0"/' \
  /home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/configdata/strategies/config_smc_structure.py

# 4. Verify changes
echo "=== Verify Threshold Changes ==="
grep -n "final_strength >= 0\." /home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/core/strategies/smc_structure_strategy.py | grep "BEAR\|BULL"

echo "=== Verify Version Update ==="
grep "STRATEGY_VERSION" /home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/configdata/strategies/config_smc_structure.py
```

---

## Test Execution

```bash
# Run TEST G backtest with decision logging
docker exec task-worker bash -c "cd /app/forex_scanner && python -m run_backtest \
  --pairs EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY,EURJPY,AUDJPY \
  --timeframe 15m \
  --strategy SMC_STRUCTURE \
  --start-date 2025-10-09 \
  --end-date 2025-11-08 \
  --log-decisions \
  --label 'TEST_G_v2110_THRESHOLD_055' \
  > /app/forex_scanner/all_signals_TEST_G_v2110.txt 2>&1"

# Check results
docker exec task-worker bash -c "cd /app/forex_scanner && tail -100 all_signals_TEST_G_v2110.txt"
```

---

## Success Criteria

| Metric | Current (TEST F) | Target (TEST G) | Status |
|--------|-----------------|-----------------|---------|
| **Signal Count** | 25 | 40-50 | More signals from continuation trades |
| **Win Rate** | 32.0% | ≥38.0% | Improved quality |
| **Profit Factor** | 0.44 | ≥1.2 | **PROFITABLE** |
| **Expectancy** | -3.6 pips | ≥+2.0 pips | Positive per-trade profit |
| **P/D Rejections** | 339 | 200-250 | Fewer rejections from 0.6 strength |

### Decision Criteria

**If TEST G succeeds (PF ≥1.2)**:
- ✅ Declare v2.11.0 as new production baseline
- Document: "HTF strength threshold optimized to 0.55 based on empirical data"
- Move forward: Focus on Order Block quality filtering (next enhancement)

**If TEST G partially succeeds (0.8 ≤ PF < 1.2)**:
- ⚠️ Test intermediate thresholds (0.50, 0.52, 0.57)
- Or test Option 2 (structure-aware thresholds)
- Do NOT declare production-ready until PF ≥1.2

**If TEST G fails (PF < 0.8)**:
- ❌ Execute Option 5 (full revert to baseline commit e6b49cd)
- Abandon threshold tuning approach
- Investigate why baseline code worked (different logic, not just params)

---

## Analysis After Test Completion

### 1. Signal Distribution Analysis

```bash
# Count approved signals by strength
docker exec task-worker bash -c "awk -F',' '\$5 == \"APPROVED\" {print \$11}' \
  /app/forex_scanner/logs/backtest_signals/execution_*/signal_decisions.csv | \
  sort | uniq -c | sort -rn"

# Check how many continuation signals were allowed
docker exec task-worker bash -c "grep -c 'continuation signal' \
  /app/forex_scanner/all_signals_TEST_G_v2110.txt"
```

### 2. Compare to Baseline

```bash
# Side-by-side comparison
echo "=== Baseline (execution_1775) ==="
grep "APPROVED" /home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/logs/backtest_signals/execution_1775/signal_decisions.csv | wc -l

echo "=== TEST G (execution_XXXX) ==="
grep "APPROVED" /home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/logs/backtest_signals/execution_*/signal_decisions.csv | tail -1 | cut -d: -f2 | wc -l
```

### 3. Identify Specific Continuation Signals

```bash
# Find bearish/discount signals that were APPROVED in TEST G
docker exec task-worker bash -c "awk -F',' '\
  \$5 == \"APPROVED\" && \$4 == \"bearish\" && \$27 == \"discount\" \
  {print \$1,\$3,\$10,\$11,\$27}' \
  /app/forex_scanner/logs/backtest_signals/execution_*/signal_decisions.csv | tail -20"

# Find bullish/premium signals that were APPROVED in TEST G
docker exec task-worker bash -c "awk -F',' '\
  \$5 == \"APPROVED\" && \$4 == \"bullish\" && \$27 == \"premium\" \
  {print \$1,\$3,\$10,\$11,\$27}' \
  /app/forex_scanner/logs/backtest_signals/execution_*/signal_decisions.csv | tail -20"
```

---

## Expected Log Output Changes

### BEFORE (TEST F - Rejection at 0.6 strength):

```
💎 STEP 3D: Premium/Discount Zone Entry Timing Validation
   📊 Range Analysis (15m - last 50 bars): 120.5 pips
   📍 Current Zone: DISCOUNT
   🎯 HTF Trend Context: BEAR (strength: 60%)
   ❌ BEARISH entry in DISCOUNT zone - poor timing
   💡 Wait for rally to premium zone (supply)
   📊 HTF: BEAR 60% (need ≥75% for discount entry)
   [SIGNAL REJECTED - PREMIUM_DISCOUNT_REJECT]
```

### AFTER (TEST G - Approved at 0.6 strength):

```
💎 STEP 3D: Premium/Discount Zone Entry Timing Validation
   📊 Range Analysis (15m - last 50 bars): 120.5 pips
   📍 Current Zone: DISCOUNT
   🎯 HTF Trend Context: BEAR (strength: 60%)
   ✅ BEARISH pullback in strong DOWNTREND - continuation signal
   📊 HTF: BEAR 60% - discount = retracement zone
   🎯 Trend strength ≥55% allows discount entry (continuation > textbook)
   [SIGNAL APPROVED - Continuation trade logic]
```

---

## Rollback Plan (If TEST G Fails)

### Option A: Test Lower Thresholds

```bash
# Try 0.50 threshold
sed -i 's/final_strength >= 0\.55/final_strength >= 0.50/' \
  /home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/core/strategies/smc_structure_strategy.py

# Run TEST G2
docker exec task-worker bash -c "cd /app/forex_scanner && python -m run_backtest \
  --pairs EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY,EURJPY,AUDJPY \
  --timeframe 15m --strategy SMC_STRUCTURE \
  --start-date 2025-10-09 --end-date 2025-11-08 \
  --log-decisions --label 'TEST_G2_v2111_THRESHOLD_050'"
```

### Option B: Full Revert to Baseline

```bash
# Revert to baseline code (commit e6b49cd)
cd /home/hr/Projects/TradeSystemV1
git checkout e6b49cd worker/app/forex_scanner/core/strategies/smc_structure_strategy.py
git checkout e6b49cd worker/app/forex_scanner/configdata/strategies/config_smc_structure.py

# Update version label
sed -i 's/STRATEGY_VERSION = .*/STRATEGY_VERSION = "2.4.0-BASELINE-RESTORE"/' \
  worker/app/forex_scanner/configdata/strategies/config_smc_structure.py

# Run TEST H (baseline restoration verification)
docker exec task-worker bash -c "cd /app/forex_scanner && python -m run_backtest \
  --pairs EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY,EURJPY,AUDJPY \
  --timeframe 15m --strategy SMC_STRUCTURE \
  --start-date 2025-10-09 --end-date 2025-11-08 \
  --log-decisions --label 'TEST_H_BASELINE_RESTORE'"
```

---

## Key Implementation Notes

### 1. Threshold Rationale

The 0.55 threshold is chosen because:
- **Data-backed**: 58.9% of baseline signals at 0.6 strength
- **Safety margin**: 0.55 captures 0.6 signals with buffer
- **Quality filter**: Still rejects very weak trends (<55%)

### 2. Why Not 0.60 Exactly?

Using 0.55 instead of 0.60 provides:
- **Inclusive**: Catches signals with slight variations (0.58, 0.59)
- **Conservative**: If baseline had 0.6, we want to be slightly more permissive
- **Testing range**: Can raise to 0.57 or 0.60 if too permissive

### 3. Alternative Thresholds to Consider

| Threshold | Behavior | Risk |
|-----------|----------|------|
| 0.50 | Very permissive | May allow too many weak trends |
| 0.55 | **Recommended** | Balanced - captures 0.6 with margin |
| 0.57 | Conservative | May still miss some 0.6 signals |
| 0.60 | Exact match | Risky - no margin for error |
| 0.75 | Current (failed) | Too restrictive - zero impact |

---

## Post-Test Documentation

After TEST G completes, document:

1. **Actual signal count**: How many more signals approved vs TEST F?
2. **Continuation signal count**: How many "wrong zone" signals allowed?
3. **Strength distribution**: What was the actual strength of approved signals?
4. **Win rate by strength**: Do 0.55-0.60 signals perform differently than 1.0 signals?
5. **Decision**: Threshold tuning success, needs adjustment, or revert?

---

## Timeline

- **Implementation**: 5 minutes (code changes + verification)
- **Test execution**: 5-10 minutes (backtest run)
- **Analysis**: 15-20 minutes (compare results, verify logic)
- **Decision**: Immediate (based on PF and signal count)

**Total time**: ~30-40 minutes to know if this approach works.

---

**Implementation Guide Created**: 2025-11-10
**Target Version**: v2.11.0
**Test Label**: TEST_G_v2110_THRESHOLD_055
**Confidence**: HIGH (data-backed threshold selection)
