# Mathematical Validation of Context-Aware Premium/Discount Filter

**Research Date:** 2025-11-10
**Analysis Type:** Statistical validation and hypothesis testing
**Data Source:** execution_1775 backtest (1,831 signal evaluations)
**Researcher:** Quantitative Research Team

---

## Executive Summary

### Critical Finding: Zero Variance in HTF Strength

**The analysis reveals a fundamental data limitation that invalidates the proposed 0.75 threshold approach.**

**Key Discovery:**
- **ALL 891 rejected bullish/premium signals have HTF strength = 0.60 (exactly)**
- **ALL 146 rejected bearish/discount signals have HTF strength = 0.60 (exactly)**
- **Standard deviation = 0.0000** (no variance whatsoever)

**Mathematical Implication:**
```
P(HTF_strength ≥ 0.75 | rejected_signal) = 0
```

**Conclusion:** The proposed threshold of 0.75 would recover **ZERO signals** because no rejected signals exceed this threshold. The entire premise of the enhancement requires re-evaluation.

---

## Research Questions & Findings

### 1. HTF Strength Distribution Analysis

#### Rejected Signals (Bullish in Premium)
- **Count:** 891 signals
- **Mean:** 0.6000
- **Median:** 0.6000
- **Std Dev:** 0.0000
- **Range:** [0.6000, 0.6000]

**Distribution at Different Thresholds:**
| Threshold | Signals Passing | Percentage |
|-----------|-----------------|------------|
| ≥ 0.60    | 752            | 100.00%    |
| ≥ 0.65    | 0              | 0.00%      |
| ≥ 0.70    | 0              | 0.00%      |
| ≥ 0.75    | 0              | 0.00%      |
| ≥ 0.80    | 0              | 0.00%      |
| ≥ 0.90    | 0              | 0.00%      |

**Statistical Interpretation:**
- Zero variance indicates that HTF strength is **not being calculated dynamically**
- All signals receive a constant value (0.6)
- No statistical distribution exists to analyze
- Cannot determine percentile positions (all values identical)

---

### 2. Approved Signal Characteristics

#### Bullish Signals (BULL trend)
- **Total:** 21 approved signals
- **Premium zone:** 0 signals (strategy correctly rejects these)
- **Discount zone:** 14 signals
  - HTF Strength: 0.6000 ± 0.0000
  - R:R Ratio: 5.5173 ± 4.5981
  - Confidence: 0.5899 ± 0.1372

#### Bearish Signals (BEAR trend)
- **Total:** 18 approved signals
- **Premium zone:** 8 signals
  - HTF Strength: 0.5876 ± 0.0349
  - R:R Ratio: 3.0925 ± 0.5837
  - Confidence: 0.6045 ± 0.0628
- **Discount zone:** 8 signals (**"wrong zone" entries already approved**)
  - HTF Strength: 1.0000 ± 0.0000 ← **Only high-strength signals!**
  - R:R Ratio: 1.2000 ± 0.0000
  - Confidence: 0.6188 ± 0.0718

**Key Insight:** Bearish entries in discount zones (8 signals) have HTF strength = 1.0, suggesting the strategy **already implements this logic for bearish signals**.

---

### 3. Optimal Threshold Selection

#### Statistical Methods Applied

**Method 1: Percentile-Based Threshold**
- 25th percentile of approved signals: 0.6000
- 75th percentile of rejected signals: 0.6000
- **Result:** No statistical separation exists

**Method 2: Effect Size Analysis**
- Cohen's d = 0.0000 (no effect)
- **Interpretation:** Rejected and approved signals are statistically identical with respect to HTF strength

**Method 3: Signal Recovery Analysis**
- At threshold 0.75: **0 signals recovered**
- At threshold 0.60: **752 signals recovered** (100%)
- **Binary outcome:** Either recover none (>0.6) or all (≤0.6)

**Method 4: Risk-Adjusted Thresholds**
```
Conservative (75th percentile): 0.60 → 100% recovery
Moderate (50th percentile):     0.60 → 100% recovery
Aggressive (25th percentile):   0.60 → 100% recovery
```

**All risk profiles converge to 0.60 due to zero variance.**

**Method 5: Statistical Hypothesis Testing**
- Cannot perform Mann-Whitney U test at threshold 0.75
- Reason: Empty sample (no signals ≥ 0.75)

---

### 4. Expected Value & Risk/Reward Modeling

#### Data Limitations

**Critical Missing Data:**
1. Actual trade outcomes (win/loss)
2. Realized profit/loss amounts
3. Historical win rates by zone
4. Drawdown statistics by entry type

**Proxy Analysis (Insufficient):**
- Can only analyze signal characteristics (R:R ratio, confidence)
- Cannot calculate E[premium_entry] without outcome data
- Cannot compute P(success | premium, strong_trend)

#### Baseline vs Proposed Comparison

**Current Baseline (Discount Entries):**
- Count: 14 bullish signals
- Avg HTF Strength: 0.6000
- Avg R:R Ratio: 5.5173
- Avg Confidence: 0.5899

**Proposed (Premium Strong Trend):**
- Count: **0 signals** (at threshold 0.75)
- Cannot compute statistics on empty set

**Signal Volume Impact:**
```
Current:  14 signals
Proposed: 14 + 0 = 14 signals
Increase: +0.00%
```

---

### 5. Multi-Factor Correlation Analysis

#### Correlation Matrix (39 approved signals with complete data)

**HTF Strength Correlations:**
| Feature              | Correlation | Interpretation        |
|----------------------|-------------|-----------------------|
| htf_score            | +0.6854     | Strong positive       |
| confidence           | +0.0884     | Weak positive         |
| htf_pullback_depth   | -0.1125     | Weak negative         |
| rr_ratio             | -0.3245     | Moderate negative     |
| pattern_score        | -0.4157     | Moderate negative     |
| zone_position_pct    | -0.4708     | Moderate negative     |
| entry_quality        | -0.4723     | Moderate negative     |
| sr_score             | -0.5362     | Moderate negative     |

#### Feature Independence Assessment

**Dependent Features (|r| ≥ 0.3):**
- `htf_strength` ↔ `htf_score` (r = +0.685)
- `rr_ratio` ↔ `entry_quality` (r = +0.629)
- `confidence` ↔ `htf_score` (r = +0.552)
- `pattern_score` ↔ `htf_score` (r = -0.432)

**Conclusion:** HTF strength is NOT an independent feature. It is moderately-to-strongly correlated with htf_score, suggesting composite scoring rather than single-threshold filtering.

#### Unique HTF Strength Values in Dataset
```python
[1.0, 0.6, 0.50116861]
```

**Only 3 discrete values exist**, confirming lack of continuous distribution.

---

## Mathematical Conclusions

### Invalid Assumptions

The research proposal assumed:
1. HTF strength varies continuously across rejected signals ❌
2. A threshold (e.g., 0.75) could separate "strong" from "weak" trends ❌
3. Distribution percentiles could guide threshold selection ❌

**Reality:** HTF strength is either 0.6 (weak/mixed), 1.0 (strong), or 0.5 (very weak), with no intermediate values.

### Revised Approach Required

#### Option A: Binary Strong/Weak Classification
```python
IF htf_strength == 1.0:  # Strong trend (perfect structure)
    APPROVE premium continuation
ELSE:
    REJECT  # Weak trend (0.6) or very weak (0.5)
```

**Impact:** Would recover **0 bullish/premium signals** (none have HTF strength = 1.0).

#### Option B: Lower Threshold to Current Minimum
```python
IF htf_strength >= 0.6:  # Minimum trend strength
    APPROVE premium continuation
```

**Impact:** Would approve **ALL 891 rejected signals** (100% recovery).

**Risk:** No selectivity—defeats purpose of quality filtering.

#### Option C: Composite Scoring (Recommended)
```python
strong_trend_score = (
    htf_strength * 0.4 +
    htf_score * 0.3 +
    momentum_strength * 0.3  # New metric needed
)

IF strong_trend_score >= 0.70:
    APPROVE premium continuation
```

**Requires:** Implementation of continuous momentum/strength metrics.

---

## Sensitivity Analysis

### Threshold Robustness

| Threshold | Signals | % Recovery | Δ from 0.75 |
|-----------|---------|------------|-------------|
| 0.65      | 0       | 0.00%      | 0           |
| 0.70      | 0       | 0.00%      | 0           |
| **0.75**  | **0**   | **0.00%**  | **0** (BASE)|
| 0.80      | 0       | 0.00%      | 0           |
| 0.85      | 0       | 0.00%      | 0           |

**Coefficient of Variation:** 0.0000 (perfectly stable, but uninformative)

**Interpretation:** Threshold is "stable" only because no data exists in the range [0.65, 0.90]. This is a data quality issue, not a validation of robustness.

---

## Recommendations

### 1. Investigate HTF Strength Calculation

**Urgent Action Required:**
- Review `/worker/app/forex_scanner/core/strategies/smc_structure_strategy.py`
- Check `calculate_htf_strength()` method
- Verify if strength is:
  - Categorical (0.5, 0.6, 1.0) or continuous (0.5-1.0)
  - Based on structure quality or momentum
  - Updated dynamically or set at initialization

### 2. Enhance HTF Strength Metric

**Proposed Improvements:**
```python
def calculate_htf_strength_v2(self, htf_data):
    """
    Calculate continuous trend strength (0.0 - 1.0)

    Components:
    - Structure quality (HH/HL vs LH/LL consistency)
    - Momentum (EMA separation, angle)
    - Volume/volatility confirmation
    - Pullback behavior (shallow vs deep retracements)
    """
    structure_score = self._structure_consistency()  # 0.0-0.4
    momentum_score = self._momentum_strength()       # 0.0-0.3
    volume_score = self._volume_confirmation()       # 0.0-0.2
    pullback_score = self._pullback_quality()        # 0.0-0.1

    return structure_score + momentum_score + volume_score + pullback_score
```

**Expected Distribution:** Continuous values in [0.0, 1.0] with mean ~0.5-0.7.

### 3. Alternative Filter Criteria

Until HTF strength is enhanced, consider:

**A. Momentum-Based Filter**
```python
IF direction == 'bullish' AND zone == 'premium':
    ema_separation = (EMA20 - EMA50) / ATR
    IF ema_separation > 2.0:  # Strong momentum
        APPROVE
```

**B. Pullback Depth Filter**
```python
IF direction == 'bullish' AND zone == 'premium':
    IF htf_pullback_depth < 0.382:  # Shallow pullback
        APPROVE  # Strong trend, quick resumption
```

**C. HTF Structure + Pattern Confirmation**
```python
IF direction == 'bullish' AND zone == 'premium':
    IF htf_structure == 'HH_HL' AND pattern_strength >= 0.8:
        APPROVE  # Clear structure + strong reversal pattern
```

### 4. Data Collection for Future Analysis

**Minimum Viable Dataset:**
- **Duration:** 60 trading days
- **Signals:** Minimum 100 premium continuation entries
- **Metrics to collect:**
  - Entry price, SL, TP
  - Exit price, exit reason (TP hit, SL hit, manual)
  - Holding time
  - Realized R:R ratio
  - Market regime (trend strength, volatility)

**Analysis After Collection:**
- Win rate by HTF strength quartile
- Expectancy calculation: E = (P(win) × Avg_win) - (P(loss) × Avg_loss)
- Optimal threshold via ROC curve maximization
- Comparison: Premium entries vs Discount entries (same HTF strength)

---

## Statistical Confidence Assessment

### Current Analysis: LOW Confidence

**Invalidating Factors:**
- ❌ Zero variance in primary variable (HTF strength)
- ❌ No continuous distribution to analyze
- ❌ Binary outcome (0 or 752 signals recovered)
- ❌ No trade outcome data for validation
- ❌ Small approved signal sample (n=39 with complete data)

**Valid Findings:**
- ✓ Large total sample size (1,831 evaluations)
- ✓ Clear data quality issue identified
- ✓ Correlation structure analyzed (39 signals)
- ✓ Strategy asymmetry discovered (bearish/discount already approved)

### Revised Analysis: Post-Enhancement

After implementing continuous HTF strength calculation:

**Expected Confidence: MODERATE-HIGH**
- ✓ Continuous distribution (0.0-1.0)
- ✓ Percentile-based threshold selection
- ✓ Statistical hypothesis testing (t-test, Mann-Whitney U)
- ✓ Effect size calculation (Cohen's d)
- ⚠ Still missing trade outcome data

---

## Implementation Roadmap

### Phase 1: Fix HTF Strength Calculation (1-2 days)
**Priority: CRITICAL**

Tasks:
1. Audit current `calculate_htf_strength()` logic
2. Implement continuous strength metric
3. Backtest with new metric on execution_1775 data
4. Verify distribution: mean ~0.6-0.7, std ~0.15-0.25

**Success Criteria:**
- HTF strength values span [0.0, 1.0]
- At least 20 unique values in dataset
- Standard deviation > 0.10

---

### Phase 2: Re-run Mathematical Validation (1 day)
**Priority: HIGH**

Tasks:
1. Execute `premium_discount_mathematical_validation.py` with new data
2. Confirm distribution characteristics:
   - Mean: 0.60-0.70
   - Std: 0.15-0.25
   - Skewness: -0.5 to +0.5 (roughly symmetric)
3. Calculate optimal threshold using percentile method
4. Perform statistical hypothesis testing
5. Generate ROC curve (if outcome data available)

**Success Criteria:**
- Recommended threshold in range [0.65, 0.85]
- At least 10-30% of rejected signals recovered
- Cohen's d ≥ 0.3 (medium effect size)

---

### Phase 3: Conservative Implementation (1-2 weeks)
**Priority: MEDIUM**

**A/B Test Design:**
```
Control:   Current strategy (reject all wrong-zone signals)
Treatment: Enable HTF strength >= [optimized_threshold] override
Split:     50/50 (randomized by signal ID)
Duration:  20 trading days or 50 signals (whichever first)
```

**Metrics:**
| Primary Metrics        | Secondary Metrics       |
|------------------------|-------------------------|
| Win rate               | Max drawdown            |
| Profit factor          | Average holding time    |
| Expectancy (R-multiple)| Sharpe ratio            |
| Risk-adjusted return   | Volatility              |

**Statistical Test:**
```python
from scipy.stats import ttest_ind, mannwhitneyu

# Win rate comparison
z_stat, p_value = proportions_ztest(
    [wins_treatment, wins_control],
    [total_treatment, total_control]
)

# Profit distribution comparison
u_stat, p_value = mannwhitneyu(
    profits_treatment,
    profits_control,
    alternative='two-sided'
)

# Significance level: α = 0.05
# Minimum detectable effect: 10% relative difference
```

**Success Criteria:**
- Treatment win rate ≥ Control - 5%
- Treatment expectancy > 0 (profitable)
- No significant drawdown increase (p > 0.05)
- Statistical power ≥ 0.80

---

### Phase 4: Full Deployment (If Phase 3 successful)
**Priority: LOW**

**Gradual Rollout:**
1. **Week 1-2:** Bullish/premium only (original use case)
2. **Week 3-4:** Add bearish/discount (if not already enabled)
3. **Week 5+:** Full deployment with monitoring

**Adaptive Threshold:**
```python
class AdaptiveThresholdManager:
    def __init__(self):
        self.base_threshold = 0.75  # From Phase 2 analysis
        self.performance_window = 30  # Last 30 signals

    def adjust_threshold(self, recent_results):
        win_rate = recent_results['wins'] / len(recent_results)

        if win_rate < 0.40:  # Below acceptable
            self.base_threshold += 0.05  # Stricter
        elif win_rate > 0.60:  # Excellent
            self.base_threshold -= 0.05  # More aggressive

        # Bounds: [0.60, 0.90]
        self.base_threshold = np.clip(self.base_threshold, 0.60, 0.90)

        return self.base_threshold
```

**Monitoring Dashboard:**
- Win rate by HTF strength bucket
- Performance by premium/discount zone
- Threshold adjustment history
- Signal volume trends

---

## Appendix A: Code Audit Checklist

### Files to Review
```
worker/app/forex_scanner/core/strategies/smc_structure_strategy.py
worker/app/forex_scanner/core/analyzers/htf_analyzer.py
worker/app/forex_scanner/core/context/evaluation_context.py
```

### Questions to Answer
1. **Where is `htf_strength` calculated?**
   - Function: `calculate_htf_strength()` or similar
   - Input: HTF candle data, structure type, momentum indicators
   - Output: Float in [0.0, 1.0] or discrete {0.5, 0.6, 1.0}

2. **What determines the strength value?**
   - Structure type mapping:
     ```python
     if structure == 'HH_HL': strength = 1.0  # Strong bullish
     elif structure == 'MIXED': strength = 0.6  # Weak/consolidation
     elif structure == 'LH_LL': strength = 1.0  # Strong bearish
     else: strength = 0.5  # Unknown
     ```
   - Or momentum-based:
     ```python
     ema_separation = (EMA20 - EMA50) / ATR
     strength = sigmoid(ema_separation)  # Continuous 0-1
     ```

3. **Is strength updated dynamically?**
   - Once per signal evaluation? ✓
   - Cached for cooldown period? ⚠
   - Fixed at strategy initialization? ❌

4. **Why do all rejected signals have strength = 0.6?**
   - Hypothesis A: `if structure == 'MIXED': strength = 0.6`
   - Hypothesis B: Premium/discount rejection happens before strength calculation
   - Hypothesis C: Default value when calculation fails

---

## Appendix B: Statistical Formulas Used

### Effect Size (Cohen's d)
```python
pooled_std = sqrt((std1^2 + std2^2) / 2)
d = (mean1 - mean2) / pooled_std

Interpretation:
  |d| < 0.2: Small effect
  |d| < 0.5: Medium effect
  |d| ≥ 0.5: Large effect
```

### Coefficient of Variation
```python
CV = std / mean

Interpretation:
  CV < 0.1: Low variability (stable)
  CV < 0.3: Moderate variability
  CV ≥ 0.3: High variability (unstable)
```

### Mann-Whitney U Test (Non-parametric)
```python
# Null hypothesis: Two samples from same distribution
# Alternative: Distributions differ
U, p_value = mannwhitneyu(sample1, sample2, alternative='two-sided')

Reject H0 if p_value < 0.05
```

### Two-Proportion Z-Test
```python
p1 = wins1 / n1
p2 = wins2 / n2
p_pooled = (wins1 + wins2) / (n1 + n2)

SE = sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
z = (p1 - p2) / SE

p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed
```

---

## Appendix C: Expected Distributions (Post-Fix)

### HTF Strength (Target Distribution)
```
Mean:     0.65
Median:   0.67
Std Dev:  0.18
Range:    [0.15, 0.95]

Percentiles:
  25th: 0.52
  50th: 0.67
  75th: 0.78
  90th: 0.87
```

**Interpretation:**
- Strong trends (≥0.75): ~30% of signals
- Moderate trends (0.55-0.75): ~45% of signals
- Weak trends (<0.55): ~25% of signals

### Signal Recovery (Projected)
At threshold 0.75:
- Bullish/premium rejected: 891 signals
- Meeting threshold (≥0.75): ~260 signals (30%)
- Recovery rate: 29.2%

**Signal Volume Impact:**
```
Current bullish approved: 14
New premium strong:       260
Total:                    274
Increase:                 +1,857% (19.6x)
```

**Note:** Massive increase suggests threshold may need adjustment upward (0.80-0.85) to maintain selectivity.

---

## References

1. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed.
2. Mann, H. B.; Whitney, D. R. (1947). "On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other". *Annals of Mathematical Statistics*.
3. Hanley, J. A.; McNeil, B. J. (1982). "The meaning and use of the area under a receiver operating characteristic (ROC) curve". *Radiology*.
4. Sullivan, G. M.; Feinn, R. (2012). "Using Effect Size—or Why the P Value Is Not Enough". *Journal of Graduate Medical Education*.

---

## Contact

**Questions or feedback on this analysis?**
- Mathematical methodology inquiries
- Statistical assumptions clarification
- Implementation strategy discussion
- Data collection planning

---

**Document Version:** 1.0
**Last Updated:** 2025-11-10
**Next Review:** After HTF strength calculation enhancement (Phase 1 completion)
