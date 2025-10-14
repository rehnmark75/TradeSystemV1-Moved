# MACD Strategy Rejection Analysis

**Analysis Date:** 2025-10-14
**Log Files Analyzed:** forex_scanner logs from Oct 7-14, 2025
**Total MACD Rejections Found:** 355

---

## Summary

The analysis reveals that **MACD signals are primarily blocked by ADX threshold requirements**, accounting for the vast majority of rejections. Swing proximity validation appears to be working well and is NOT a significant source of rejection.

---

## Rejection Breakdown by Reason

| Reason | Count | Percentage | Notes |
|--------|-------|------------|-------|
| **ADX Below Threshold** | **305** | **85.9%** | Dominant rejection reason |
| MACD Line Position | 50 | 14.0% | Secondary filter catching premature reversals |
| EMA Filter | 87 | 24.5% | Overlaps with other rejections |
| Histogram Too Small | 0 | 0% | Not triggering rejections |
| RSI Extreme | 0 | 0% | Not triggering rejections |
| Swing Proximity | 0 | 0% | **Not blocking signals** |

---

## Key Findings

### 1. ADX Threshold (85.9% of rejections)

**The primary blocker for MACD signals is the ADX threshold requirement (currently 25).**

- 305 out of 355 rejections are due to ADX being below 25
- Many rejections have ADX values close to threshold (18-24 range)
- This suggests the market is often in weak-to-moderate trending conditions

**Sample rejections:**
```
❌ [CS.D.GBPUSD.MINI.IP] BEAR rejected: ADX 22.0 < 25
❌ [CS.D.USDJPY.MINI.IP] BULL rejected: ADX 18.8 < 25
❌ [CS.D.AUDJPY.MINI.IP] BEAR rejected: ADX 20.7 < 25
```

**Recommendation:**
- The ADX threshold of 25 is quite strict and filters out a lot of potential signals
- Consider whether this threshold is optimal for your trading style
- You might want to analyze if signals with ADX 20-25 would have been profitable
- The system already has pair-specific ADX thresholds - verify they are properly configured

### 2. MACD Line Position (14.0% of rejections)

**The second most common rejection is the MACD line position filter.**

- 50 rejections due to MACD line being too far in opposite territory
- This is a **smart filter** that prevents taking BEAR signals when MACD line is strongly positive (>0.05) and vice versa
- Prevents trading against strong momentum

**Pattern observed:**
- BEAR signals rejected when MACD line > 0.05 (still in bullish territory)
- BULL signals rejected when MACD line < -0.05 (still in bearish territory)

**Sample rejections:**
```
❌ BEAR rejected: MACD line too positive 0.202452 (still in bullish territory)
❌ BULL rejected: MACD line too negative -0.181136 (still in bearish territory)
```

**Recommendation:**
- This filter is working as designed and catching premature reversals
- Keep this filter enabled - it's preventing poor quality signals

### 3. Swing Proximity Validation (0% of rejections)

**Swing proximity validation is NOT blocking any MACD signals.**

This is interesting and could mean:
1. The swing validator thresholds are too lenient
2. MACD signals are not frequently occurring near swing points
3. The ADX filter is already preventing most trades near swing extremes

**Recommendation:**
- If you specifically want swing control to be more active, you may need to:
  - Review the swing proximity configuration in `config_macd_strategy.py`
  - Check the `MACD_SWING_VALIDATION` settings
  - Verify swing points are being detected properly

### 4. Signal Direction Distribution

| Direction | Count | Percentage |
|-----------|-------|------------|
| BEAR | 193 | 54.3% |
| BULL | 162 | 45.6% |
| ADX BULL | 0 | 0% |
| ADX BEAR | 0 | 0% |

- Fairly balanced between BULL and BEAR rejections
- ADX crossover signals are not appearing (or not being rejected)

---

## Recommendations

### High Priority

1. **Review ADX Threshold Configuration**
   - 85.9% of rejections are due to ADX < 25
   - Consider if this threshold should be adjusted based on:
     - Backtesting results
     - Market conditions
     - Currency pair characteristics
   - The system supports pair-specific ADX thresholds - ensure they are properly configured

2. **Verify ADX Calculation**
   - Ensure ADX is being calculated correctly (using Wilder's smoothing)
   - Check if ADX values match TradingView or other platforms
   - Verify the ADX catch-up window feature is working as expected

### Medium Priority

3. **Evaluate MACD Line Filter Effectiveness**
   - The MACD line position filter (14% of rejections) appears to be working well
   - Monitor if these filtered signals would have been profitable
   - This is likely preventing many losing trades

4. **Analyze Swing Proximity Configuration**
   - Swing proximity is blocking 0% of signals - investigate if this is desired
   - Review `MACD_SWING_VALIDATION` configuration
   - Consider if swing control should be more restrictive

### Low Priority

5. **Monitor Other Filters**
   - Histogram threshold: Not currently blocking any signals
   - RSI extreme: Not currently blocking any signals
   - These filters may be redundant or need threshold adjustments

---

## Next Steps

1. **Backtest Analysis**
   - Run backtests with different ADX thresholds (20, 22, 25, 28)
   - Compare win rates and profitability
   - Determine optimal ADX threshold per currency pair

2. **Signal Quality Review**
   - Review trades that were rejected due to ADX 20-24
   - Determine if lowering threshold would improve signal volume without sacrificing quality

3. **Swing Proximity Tuning**
   - If swing control is desired, review configuration
   - Test more aggressive swing proximity thresholds
   - Ensure swing point detection is working properly

---

## Configuration Files to Review

1. **`worker/app/forex_scanner/configdata/strategies/config_macd_strategy.py`**
   - `MACD_MIN_ADX` or pair-specific ADX thresholds
   - `MACD_SWING_VALIDATION` settings
   - `MACD_ADX_CATCHUP_ENABLED` and related settings

2. **`worker/app/forex_scanner/core/strategies/macd_strategy.py`**
   - Line 62-65: ADX threshold loading
   - Line 580-684: Signal validation logic
   - Line 634-650: MACD line position filter

---

## Conclusion

**The ADX threshold is the dominant filter preventing MACD signals from triggering.** This is by design - the strategy requires strong trends (ADX >= 25) before taking trades. The MACD line position filter is also working well to prevent premature reversals. Swing proximity validation is not currently blocking any signals, which may warrant investigation if swing control is a desired feature.

Consider analyzing whether the current ADX threshold of 25 is optimal for your trading objectives, as lowering it could significantly increase signal volume (though quality may decrease).
