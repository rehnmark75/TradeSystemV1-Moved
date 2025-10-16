# Directional Bias Fix - Validation Checklist

## Quick Reference for Post-Fix Validation

---

## Pre-Fix Baseline (Known Values)

```
Period: 7 days (15m timeframe, 9 pairs)
Total signals: 13
BULL signals: 13 (100%)
BEAR signals: 0 (0%)
Signal rate: 0.215%
Statistical significance: p = 0.000122 (EXTREME BIAS)
```

---

## Validation Tests

### ✅ Test 1: Historical Backtest (Same Period)

**Command**:
```bash
docker exec task-worker bash -c "cd /app/forex_scanner && \
  python bt.py --all 7 EMA --pipeline --timeframe 15m"
```

**Expected Results**:
```
Total signals: 60-90 (vs 13 pre-fix)
BULL signals: 30-50 (vs 13 pre-fix)
BEAR signals: 25-45 (vs 0 pre-fix)
BULL/BEAR ratio: 0.7-1.4 (vs ∞ pre-fix)
Signal rate: 1.0-1.5% (vs 0.215% pre-fix)
```

**Pass Criteria**:
- ✅ Total signals > 50
- ✅ BEAR signals > 20
- ✅ BULL/BEAR ratio between 0.5 and 1.5
- ✅ All 9 pairs show BOTH directions

---

### ✅ Test 2: Binomial Statistical Test

**Calculate**:
```python
from scipy.stats import binomial_test

bull_count = # Count from backtest
total_count = # Total signals
p_value = binomial_test(bull_count, total_count, p=0.5, alternative='two-sided')
```

**Pass Criteria**:
- ✅ p_value > 0.05 (no statistically significant bias)

**Interpretation**:
- p > 0.10: Excellent (no bias)
- 0.05 < p ≤ 0.10: Good (marginal)
- p ≤ 0.05: FAIL (bias still exists)

---

### ✅ Test 3: Chi-Square Goodness of Fit

**Calculate**:
```python
expected_bull = total_count / 2
expected_bear = total_count / 2
chi_square = (bull_count - expected_bull)**2 / expected_bull + \
             (bear_count - expected_bear)**2 / expected_bear
```

**Pass Criteria**:
- ✅ χ² < 3.841 (critical value at α=0.05, df=1)

**Interpretation**:
- χ² < 2.71: Excellent fit (p > 0.10)
- 2.71 ≤ χ² < 3.841: Good fit (0.05 < p ≤ 0.10)
- χ² ≥ 3.841: FAIL (significant deviation)

---

### ✅ Test 4: Per-Pair Distribution

**Check each pair**:
```python
for pair in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD',
             'USDCAD', 'USDCHF', 'EURJPY', 'GBPJPY']:
    pair_signals = df[df['pair'] == pair]
    bull = (pair_signals['direction'] == 'BULL').sum()
    bear = (pair_signals['direction'] == 'BEAR').sum()
    ratio = bull / (bull + bear) if (bull + bear) > 0 else 0
    print(f"{pair}: BULL={bull}, BEAR={bear}, Ratio={ratio:.2f}")
```

**Pass Criteria**:
- ✅ All pairs have BOTH bull and bear signals
- ✅ No pair has ratio > 0.85 or < 0.15
- ✅ Standard deviation of ratios < 0.15

---

### ✅ Test 5: Performance Metric Decorrelation

**Check independence**:
```python
correlation = df['st_bull_performance'].corr(df['st_bear_performance'])
```

**Pass Criteria**:
- ✅ correlation < 0.70 (metrics are sufficiently independent)

**Interpretation**:
- corr < 0.50: Excellent (highly independent)
- 0.50 ≤ corr < 0.70: Good (moderately independent)
- corr ≥ 0.70: FAIL (still too correlated)

---

### ✅ Test 6: Z-Score Analysis

**Calculate**:
```python
import numpy as np
from scipy import stats

expected = total_count / 2
std_dev = np.sqrt(total_count * 0.5 * 0.5)
z_score = (bull_count - expected) / std_dev
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
```

**Pass Criteria**:
- ✅ |z_score| < 1.96 (within 95% confidence interval)
- ✅ p_value > 0.05

**Interpretation**:
- |Z| < 1.00: Excellent (within 1σ)
- 1.00 ≤ |Z| < 1.96: Good (within 2σ)
- 1.96 ≤ |Z| < 2.58: Marginal (within 2.5σ)
- |Z| ≥ 2.58: FAIL (beyond 2.5σ)

---

### ✅ Test 7: 95% Confidence Interval

**Calculate expected range**:
```python
from scipy.stats import binom

lower_bound = binom.ppf(0.025, total_count, 0.5)
upper_bound = binom.ppf(0.975, total_count, 0.5)
```

**Pass Criteria**:
- ✅ bull_count is within [lower_bound, upper_bound]

**Example** (for 75 total signals):
```
Expected: 37.5 BULL signals
95% CI: [30, 45]
Observed must be in [30, 45]
```

---

## Overall Pass Criteria

**Fix is validated if 6 out of 7 tests pass** (85% pass rate)

### Critical Tests (Must Pass):
1. ✅ Test 2: Binomial test (p > 0.05)
2. ✅ Test 4: Per-pair distribution (all pairs show both directions)

### Important Tests (Strongly Recommended):
3. ✅ Test 3: Chi-square (χ² < 3.841)
4. ✅ Test 6: Z-score (|Z| < 1.96)

### Supporting Tests:
5. ✅ Test 1: Signal count increase
6. ✅ Test 5: Performance decorrelation
7. ✅ Test 7: Confidence interval

---

## Quick Validation Script

```python
#!/usr/bin/env python3
"""Quick validation script for bias fix"""

import pandas as pd
from scipy import stats
import numpy as np

def validate_fix(backtest_results_csv):
    """
    Run all validation tests on backtest results

    Args:
        backtest_results_csv: Path to backtest results CSV

    Returns:
        Dictionary with test results
    """
    # Load results
    df = pd.read_csv(backtest_results_csv)

    # Count signals
    bull_count = (df['direction'] == 'BULL').sum()
    bear_count = (df['direction'] == 'BEAR').sum()
    total_count = bull_count + bear_count
    bull_ratio = bull_count / total_count if total_count > 0 else 0

    results = {
        'total_signals': total_count,
        'bull_signals': bull_count,
        'bear_signals': bear_count,
        'bull_ratio': bull_ratio,
        'tests_passed': 0,
        'tests_failed': 0
    }

    # Test 1: Signal count
    print("Test 1: Signal Count")
    if total_count >= 50:
        print(f"  ✅ PASS: {total_count} signals (≥50)")
        results['tests_passed'] += 1
    else:
        print(f"  ❌ FAIL: {total_count} signals (<50)")
        results['tests_failed'] += 1

    # Test 2: Binomial test
    print("\nTest 2: Binomial Test")
    p_value_binom = stats.binom_test(bull_count, total_count, p=0.5,
                                     alternative='two-sided')
    if p_value_binom > 0.05:
        print(f"  ✅ PASS: p={p_value_binom:.4f} (>0.05)")
        results['tests_passed'] += 1
    else:
        print(f"  ❌ FAIL: p={p_value_binom:.4f} (≤0.05)")
        results['tests_failed'] += 1
    results['binomial_pvalue'] = p_value_binom

    # Test 3: Chi-square
    print("\nTest 3: Chi-Square Goodness of Fit")
    expected = total_count / 2
    chi_square = (bull_count - expected)**2 / expected + \
                 (bear_count - expected)**2 / expected
    if chi_square < 3.841:
        print(f"  ✅ PASS: χ²={chi_square:.3f} (<3.841)")
        results['tests_passed'] += 1
    else:
        print(f"  ❌ FAIL: χ²={chi_square:.3f} (≥3.841)")
        results['tests_failed'] += 1
    results['chi_square'] = chi_square

    # Test 4: Per-pair distribution
    print("\nTest 4: Per-Pair Distribution")
    pairs = df['pair'].unique()
    pair_test_pass = True
    for pair in pairs:
        pair_df = df[df['pair'] == pair]
        pair_bull = (pair_df['direction'] == 'BULL').sum()
        pair_bear = (pair_df['direction'] == 'BEAR').sum()
        pair_total = pair_bull + pair_bear
        pair_ratio = pair_bull / pair_total if pair_total > 0 else 0

        if pair_total > 0 and 0.15 <= pair_ratio <= 0.85:
            print(f"  ✅ {pair}: {pair_bull}B/{pair_bear}B (ratio={pair_ratio:.2f})")
        else:
            print(f"  ❌ {pair}: {pair_bull}B/{pair_bear}B (ratio={pair_ratio:.2f})")
            pair_test_pass = False

    if pair_test_pass:
        results['tests_passed'] += 1
    else:
        results['tests_failed'] += 1

    # Test 5: Performance decorrelation (if available)
    print("\nTest 5: Performance Metric Decorrelation")
    if 'st_bull_performance' in df.columns and 'st_bear_performance' in df.columns:
        corr = df['st_bull_performance'].corr(df['st_bear_performance'])
        if corr < 0.70:
            print(f"  ✅ PASS: correlation={corr:.3f} (<0.70)")
            results['tests_passed'] += 1
        else:
            print(f"  ❌ FAIL: correlation={corr:.3f} (≥0.70)")
            results['tests_failed'] += 1
        results['performance_correlation'] = corr
    else:
        print("  ⚠️  SKIP: Performance columns not available")

    # Test 6: Z-score
    print("\nTest 6: Z-Score Analysis")
    expected = total_count / 2
    std_dev = np.sqrt(total_count * 0.5 * 0.5)
    z_score = (bull_count - expected) / std_dev
    p_value_z = 2 * (1 - stats.norm.cdf(abs(z_score)))
    if abs(z_score) < 1.96:
        print(f"  ✅ PASS: Z={z_score:.3f} (|Z|<1.96)")
        results['tests_passed'] += 1
    else:
        print(f"  ❌ FAIL: Z={z_score:.3f} (|Z|≥1.96)")
        results['tests_failed'] += 1
    results['z_score'] = z_score

    # Test 7: Confidence interval
    print("\nTest 7: 95% Confidence Interval")
    ci_lower = stats.binom.ppf(0.025, total_count, 0.5)
    ci_upper = stats.binom.ppf(0.975, total_count, 0.5)
    if ci_lower <= bull_count <= ci_upper:
        print(f"  ✅ PASS: {bull_count} in [{ci_lower:.0f}, {ci_upper:.0f}]")
        results['tests_passed'] += 1
    else:
        print(f"  ❌ FAIL: {bull_count} not in [{ci_lower:.0f}, {ci_upper:.0f}]")
        results['tests_failed'] += 1
    results['ci_lower'] = ci_lower
    results['ci_upper'] = ci_upper

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    total_tests = results['tests_passed'] + results['tests_failed']
    pass_rate = results['tests_passed'] / total_tests * 100 if total_tests > 0 else 0
    print(f"Tests passed: {results['tests_passed']}/{total_tests} ({pass_rate:.1f}%)")
    print(f"Signal distribution: {bull_count}B/{bear_count}B ({bull_ratio:.1%} BULL)")
    print(f"Statistical significance: p={p_value_binom:.4f}")

    if results['tests_passed'] >= 6:
        print("\n✅ VALIDATION PASSED - Fix is working correctly")
        results['validation_status'] = 'PASSED'
    else:
        print("\n❌ VALIDATION FAILED - Fix needs review")
        results['validation_status'] = 'FAILED'
    print("="*80)

    return results

# Example usage:
# results = validate_fix('backtest_results.csv')
```

---

## Monitoring After Deployment

### Daily Checks (First Week)

```bash
# Get last 7 days of signals
docker exec task-worker bash -c "cd /app && python3 -c \"
from forex_scanner.database import get_signals_last_n_days
signals = get_signals_last_n_days(7)
bull = len([s for s in signals if s.direction == 'BULL'])
bear = len([s for s in signals if s.direction == 'BEAR'])
total = bull + bear
ratio = bull / total if total > 0 else 0
print(f'Last 7 days: {bull}B/{bear}B (ratio={ratio:.2%})')
if ratio > 0.75 or ratio < 0.25:
    print('⚠️  WARNING: Directional imbalance detected')
else:
    print('✅ Balance looks good')
\""
```

### Alert Thresholds

- 🚨 **Critical**: BULL ratio > 85% or < 15%
- ⚠️  **Warning**: BULL ratio > 75% or < 25%
- ✅ **Normal**: BULL ratio between 25% and 75%

---

## Expected Timeline

| Day | Action | Expected Outcome |
|-----|--------|------------------|
| **Day 0** | Run 7-day backtest | 60-90 signals, balanced distribution |
| **Day 0** | Validate with 7 tests | 6/7 tests pass |
| **Day 0** | Deploy if validated | Live trading with fix |
| **Day 1-7** | Monitor daily | Balanced signals continue |
| **Day 7** | Full validation | Statistical test on live data |
| **Day 30** | Performance review | Strategy performance analysis |

---

## Success Metrics

### Immediate (Day 0)
- ✅ Backtest shows balanced signals (0.5-1.5 ratio)
- ✅ 6/7 validation tests pass
- ✅ All pairs show both directions

### Short-term (Week 1)
- ✅ Live signals remain balanced
- ✅ No alerts triggered
- ✅ Signal quality maintained

### Medium-term (Month 1)
- ✅ Risk-adjusted returns improve (Sharpe +15-20%)
- ✅ Portfolio volatility decreases (-15-20%)
- ✅ Win rate maintains or improves

---

## Troubleshooting

### If Validation Fails

**Scenario 1: Still too many BULL signals (ratio > 0.75)**
- Check if performance filter is still using global metric
- Verify separate bull/bear performance columns exist
- Check PERFORMANCE_THRESHOLD value

**Scenario 2: Too few total signals (< 50)**
- This is expected (filters are restrictive by design)
- Consider reducing STABILITY_BARS from 8 to 6
- Consider reducing TREND_STRENGTH from 0.15% to 0.10%

**Scenario 3: High correlation between bull/bear performance**
- Verify separate EMA smoothing is applied
- Check that filtering uses correct performance metric
- Inspect DataFrame columns: st_bull_performance, st_bear_performance

---

## Contact & Support

**Questions or Issues?**
- Review: `/home/hr/Projects/TradeSystemV1/QUANTITATIVE_BIAS_ANALYSIS.md`
- Check: `/home/hr/Projects/TradeSystemV1/DIRECTIONAL_BIAS_FIX.md`
- Code: `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/core/strategies/helpers/ema_signal_calculator.py` (lines 391-399)

---

**Last Updated**: 2025-10-16
**Version**: 1.0
**Status**: Ready for validation testing
