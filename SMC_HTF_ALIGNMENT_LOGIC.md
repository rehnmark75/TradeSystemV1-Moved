# SMC Structure Strategy - HTF Alignment Logic Documentation

**CRITICAL**: This document explains the Higher Timeframe (HTF) alignment logic which is a **HARD REQUIREMENT** for the SMC Structure strategy. This is NOT part of confidence scoring - signals without HTF alignment are immediately rejected.

## Overview

The SMC Structure strategy requires multi-timeframe confirmation to ensure trades are taken in the direction of the higher timeframe trend. This prevents low-quality counter-trend entries that have historically resulted in immediate stop loss hits.

## Configuration

### Location
`/worker/app/forex_scanner/configdata/strategies/config_smc_structure.py`

### Settings

```python
# Lines 327-335
SMC_REQUIRE_1H_ALIGNMENT = True   # 1H must align with BOS/CHoCH direction
SMC_REQUIRE_4H_ALIGNMENT = True   # 4H must align with BOS/CHoCH direction
```

**IMPORTANT**: Both settings are currently enabled and should remain enabled. Disabling either creates low-quality signals.

## How It Works

### Signal Flow

1. **15m BOS/CHoCH Detection**
   - Strategy detects Break of Structure (BOS) or Change of Character (CHoCH) on 15m timeframe
   - Determines direction: 'bullish' or 'bearish'
   - Identifies structure level (price where break occurred)

2. **HTF Alignment Validation** (MANDATORY)
   - Calls `_validate_htf_alignment()` method ([smc_structure_strategy.py:832-885](worker/app/forex_scanner/core/strategies/smc_structure_strategy.py#L832-L885))
   - Checks if 1H trend matches BOS/CHoCH direction
   - Checks if 4H trend matches BOS/CHoCH direction
   - **Returns False if ANY mismatch** → Signal immediately rejected

3. **Entry Pattern Validation**
   - Only proceeds if HTF alignment confirmed
   - Looks for rejection pattern at structure level
   - Validates volume, ADX, wick size (Phase 2.1 filters)

### HTF Alignment Logic Details

```python
def _validate_htf_alignment(self, bos_direction: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame, epic: str) -> bool:
    """
    Validate that 1H and 4H timeframes align with BOS/CHoCH direction.

    Args:
        bos_direction: 'bullish' or 'bearish' from 15m BOS/CHoCH
        df_1h: 1H DataFrame
        df_4h: 4H DataFrame
        epic: Currency pair

    Returns:
        True if HTF alignment confirmed, False otherwise
    """
```

#### 1H Alignment Check

```python
# If enabled (SMC_REQUIRE_1H_ALIGNMENT = True)
trend_1h = self.trend_analyzer.analyze_trend(df=df_1h, epic=epic, lookback=self.htf_alignment_lookback)

expected_trend_1h = 'BULL' if bos_direction == 'bullish' else 'BEAR'

if trend_1h['trend'] != expected_trend_1h:
    # MISMATCH - Signal rejected immediately
    return False
```

**Example**:
- 15m shows bullish BOS (price breaks above previous high)
- 1H trend analyzer determines trend = 'BEAR'
- **REJECTED**: Cannot take bullish entry in bearish 1H trend

#### 4H Alignment Check

```python
# If enabled (SMC_REQUIRE_4H_ALIGNMENT = True)
trend_4h = self.trend_analyzer.analyze_trend(df=df_4h, epic=epic, lookback=self.htf_alignment_lookback)

expected_trend_4h = 'BULL' if bos_direction == 'bullish' else 'BEAR'

if trend_4h['trend'] != expected_trend_4h:
    # MISMATCH - Signal rejected immediately
    return False
```

**Example**:
- 15m shows bullish BOS
- 1H trend = 'BULL' ✅
- 4H trend = 'BEAR' ❌
- **REJECTED**: Cannot take bullish entry when 4H is bearish

### Success Criteria

A signal is only accepted if:
- 15m BOS/CHoCH detected (bullish or bearish)
- 1H trend matches direction (BULL for bullish, BEAR for bearish)
- 4H trend matches direction (BULL for bullish, BEAR for bearish)
- All other filters pass (ADX, volume, wick size, etc.)

##  Why HTF Alignment is Critical

### Historical Data

**Phase 2.2 Analysis** revealed that 93.75% of rejected signals had HTF score = 0.00, meaning they lacked higher timeframe confirmation.

| Signal Quality | HTF Score | Confidence | Win Rate |
|----------------|-----------|------------|----------|
| **Rejected (weak)** | 0.00 (no HTF) | 27-35% | Unknown (never traded) |
| **Accepted (strong)** | 0.40+ (HTF confirmed) | 60-91% | 60% (small sample) |

### Prevents Counter-Trend Entries

Without HTF alignment:
- 15m bullish BOS in 1H/4H bearish trend → **Fades quickly**
- Entry against institutional flow → **High stop loss hit rate**
- Small timeframe noise mistaken for trend change → **Poor win rate**

With HTF alignment:
- 15m entry confirms higher timeframe direction → **Trend continuation**
- Institutional money on same side → **Better probability**
- Structure break represents real market shift → **Higher win rate**

## Configuration Options

### Current Configuration (RECOMMENDED)

```python
SMC_REQUIRE_1H_ALIGNMENT = True   # ✅ ENABLED
SMC_REQUIRE_4H_ALIGNMENT = True   # ✅ ENABLED
SMC_HTF_ALIGNMENT_LOOKBACK = 50   # Bars to analyze for trend
```

**Expected Results**:
- Moderate signal reduction (filters out counter-trend setups)
- Higher win rate (entries with trend)
- Lower stop loss hit rate (institutional flow aligned)

### Alternative Configurations (NOT RECOMMENDED)

#### Option A: 1H Only
```python
SMC_REQUIRE_1H_ALIGNMENT = True
SMC_REQUIRE_4H_ALIGNMENT = False  # ⚠️ Allows 4H misalignment
```
**Impact**: More signals, but some may fade when 4H reverses

#### Option B: 4H Only
```python
SMC_REQUIRE_1H_ALIGNMENT = False  # ⚠️ Allows 1H misalignment
SMC_REQUIRE_4H_ALIGNMENT = True
```
**Impact**: May miss early entries when 4H still forming

#### Option C: Disabled (DANGEROUS)
```python
SMC_REQUIRE_1H_ALIGNMENT = False  # ❌ DO NOT USE
SMC_REQUIRE_4H_ALIGNMENT = False  # ❌ DO NOT USE
```
**Impact**:
- Massive signal increase
- Many counter-trend entries
- High stop loss hit rate
- Low win rate
- **NOT VIABLE FOR PRODUCTION**

## Logging Output

When HTF alignment is validated, the logs show:

```
🔍 Validating HTF Alignment (bullish BOS/CHoCH)
   1H Trend: BULL (85%)
   ✅ 1H aligned with bullish direction
   4H Trend: BULL (72%)
   ✅ 4H aligned with bullish direction
   ✅ HTF Alignment Confirmed
```

When HTF alignment fails:

```
🔍 Validating HTF Alignment (bullish BOS/CHoCH)
   1H Trend: BEAR (68%)
   ❌ 1H trend mismatch (expected BULL) - SIGNAL REJECTED
```

## Code References

### Implementation
- **Method**: `_validate_htf_alignment()` at [smc_structure_strategy.py:832-885](worker/app/forex_scanner/core/strategies/smc_structure_strategy.py#L832-L885)
- **Called from**: `detect_signal()` at [smc_structure_strategy.py:458](worker/app/forex_scanner/core/strategies/smc_structure_strategy.py#L458)
- **Configuration**: [config_smc_structure.py:331,335](worker/app/forex_scanner/configdata/strategies/config_smc_structure.py#L331)

### Related Components
- **Trend Analyzer**: `TrendAnalyzer.analyze_trend()` - Determines BULL/BEAR/RANGING
- **BOS/CHoCH Detector**: `_detect_bos_choch_15m()` - Finds 15m structure breaks
- **Market Structure**: `MarketStructureDetector` - Identifies swing highs/lows

## Troubleshooting

### Problem: Too Few Signals

**Symptom**: Only 5-10 signals per month across all pairs

**Possible Causes**:
1. Both 1H and 4H alignment required AND other strict filters
2. Market in consolidation (no clear trend)
3. HTF timeframes frequently conflicting

**Solution**:
- Review HTF lookback period (increase from 50 to 100 bars for smoother trend)
- Consider 1H-only alignment temporarily
- Check if ADX/volume filters are too aggressive

### Problem: High Stop Loss Hit Rate

**Symptom**: >40% of losses hit stop immediately

**Diagnosis**: HTF alignment may not be working correctly

**Debug Steps**:
1. Check logs for "HTF Alignment Confirmed" messages
2. Verify `SMC_REQUIRE_1H_ALIGNMENT` and `SMC_REQUIRE_4H_ALIGNMENT` are True
3. Check if trend analyzer is returning RANGING instead of BULL/BEAR
4. Increase minimum trend strength threshold

### Problem: Signals with Low Confidence Still Passing

**Symptom**: Signals with 30-40% confidence being generated

**Root Cause**: HTF score component is 0.00 (not being calculated)

**Solution**: HTF alignment is a REJECTION filter, not a confidence component. Low confidence signals that pass HTF alignment are valid but weak. Consider adding minimum confidence filter if quality is poor.

## Best Practices

### DO:
✅ Keep both 1H and 4H alignment enabled
✅ Review logs to confirm HTF validation is working
✅ Use HTF alignment as first filter (before other validations)
✅ Document any configuration changes with rationale

### DON'T:
❌ Disable HTF alignment to increase signal count
❌ Use HTF alignment as confidence component instead of hard filter
❌ Mix HTF alignment with counter-trend strategies
❌ Ignore HTF mismatches hoping for "reversals"

## Summary

Higher Timeframe (HTF) alignment is the **PRIMARY quality filter** for the SMC Structure strategy. It ensures that:

1. **15m entries align with 1H trend** - Prevents immediate reversals
2. **15m entries align with 4H trend** - Confirms institutional direction
3. **Trades with the trend** - Higher probability of success
4. **Filters counter-trend noise** - Reduces stop loss hits

**This is a HARD REQUIREMENT, not a confidence modifier.** Signals without HTF alignment should NEVER be generated.

---

**Last Updated**: 2025-11-03
**Configuration Version**: Phase 2.1 (Post-confidence filter removal)
**Status**: ✅ PRODUCTION READY
