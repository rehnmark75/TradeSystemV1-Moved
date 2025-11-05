# Phase 2.1 Retest - Action Plan

**Created**: 2025-11-03
**Priority**: HIGH
**Estimated Time**: 4-6 hours

---

## Problem Statement

Phase 2.1 backtest was executed with wrong configuration:
- Used 1H timeframe instead of 15m
- Applied ADX 25 instead of 12
- Enabled MACD/EMA hybrid instead of pure SMC
- Result: Only 4 signals (statistically insignificant)

---

## Step 1: Fix Configuration Management (2 hours)

### 1.1 Add Backtest Configuration Validator

Create file: `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/backtest_config_validator.py`

```python
"""Backtest configuration validation"""
import logging

logger = logging.getLogger(__name__)

def validate_backtest_config(strategy, timeframe, config, data_fetcher):
    """
    Validate backtest configuration before execution

    Raises:
        ValueError: If configuration is invalid
    """
    errors = []
    warnings = []

    # Log configuration
    logger.info("=" * 60)
    logger.info("BACKTEST CONFIGURATION VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Requested Timeframe: {timeframe}")

    # Validate timeframe match
    if hasattr(data_fetcher, 'actual_timeframe'):
        actual_tf = data_fetcher.actual_timeframe
        if actual_tf != timeframe:
            errors.append(
                f"Timeframe mismatch: requested={timeframe}, "
                f"actual={actual_tf}"
            )

    # Strategy-specific validation
    if strategy == 'SMC_STRUCTURE':
        logger.info("SMC Structure Strategy Configuration:")

        # Check for strategy overlays
        if hasattr(config, 'macd_enabled') and config.macd_enabled:
            warnings.append("MACD enabled - should be pure SMC for Phase 2.1")

        if hasattr(config, 'ema_enabled') and config.ema_enabled:
            warnings.append("EMA enabled - should be pure SMC for Phase 2.1")

        # Validate SMC parameters
        adx_threshold = getattr(config, 'adx_threshold', None)
        if adx_threshold is not None:
            logger.info(f"  ADX Threshold: {adx_threshold}")
            if adx_threshold != 12:
                errors.append(
                    f"Wrong ADX threshold: {adx_threshold} (expected 12)"
                )

        volume_mult = getattr(config, 'volume_multiplier', None)
        if volume_mult is not None:
            logger.info(f"  Volume Filter: {volume_mult}x")
            if volume_mult != 0.9:
                errors.append(
                    f"Wrong volume filter: {volume_mult}x (expected 0.9x)"
                )

        min_conf = getattr(config, 'min_confidence', None)
        if min_conf is not None:
            logger.info(f"  Min Confidence: {min_conf}%")

    # Print warnings
    if warnings:
        logger.warning("Configuration Warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Fail on errors
    if errors:
        logger.error("Configuration Errors:")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError(
            f"Backtest configuration validation failed:\n"
            + "\n".join(errors)
        )

    logger.info("=" * 60)
    logger.info("CONFIGURATION VALIDATION PASSED")
    logger.info("=" * 60)

    return True
```

### 1.2 Integrate Validator into Backtest CLI

Edit: `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/backtest_cli.py`

Add after imports:
```python
from forex_scanner.backtest_config_validator import validate_backtest_config
```

Add before backtest execution:
```python
# Validate configuration
validate_backtest_config(
    strategy=args.strategy,
    timeframe=args.timeframe,
    config=config_object,
    data_fetcher=data_fetcher
)
```

### 1.3 Add Data Fetcher Timeframe Tracking

Edit: `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/core/data/backtest_data_fetcher.py`

Add property to track actual timeframe:
```python
@property
def actual_timeframe(self):
    """Return the actual timeframe being used (not requested)"""
    if hasattr(self, '_resampled_timeframe'):
        return self._resampled_timeframe
    return self._requested_timeframe

def resample_data(self, df, target_timeframe):
    """Resample data and track actual timeframe"""
    self._resampled_timeframe = target_timeframe
    # ... rest of resampling logic
```

---

## Step 2: Update SMC Configuration (30 minutes)

### 2.1 Verify Phase 2.1 Parameters

Edit: `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/configdata/strategies/config_smc_structure.py`

Confirm these values:
```python
# Phase 2.1 parameters (already set correctly in config)
SMC_ENTRY_TIMEFRAME = '15m'  # Line 306
SMC_MIN_PATTERN_STRENGTH = 0.55  # Line 100
SMC_MIN_RR_RATIO = 2.0  # Line 134
SMC_SR_PROXIMITY_PIPS = 15  # Line 86
```

### 2.2 Add Phase 2.1 Filter Parameters

Add new section to config file (after line 227):
```python
# ============================================================================
# PHASE 2.1 FILTER PARAMETERS
# ============================================================================

# ADX threshold for trend filtering
# Phase 2.1: Reduced from Phase 1's 20 to 12 for 15m timeframe
# Rationale: 15m naturally has lower ADX than 1H/4H
SMC_ADX_THRESHOLD = 12

# Volume filter multiplier
# Phase 2.1: Reduced from Phase 1's 1.2x to 0.9x
# Rationale: SMC rejections occur on normal volume, not elevated
SMC_VOLUME_MULTIPLIER = 0.9

# Minimum signal confidence percentage
# Phase 2.1: Set to 60% (upgraded from initial 50% based on analysis)
# Rationale: Signals <60% had 0% win rate in testing
SMC_MIN_CONFIDENCE = 60
```

---

## Step 3: Disable Strategy Overlays (15 minutes)

### 3.1 Create Pure SMC Mode

Edit: `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/strategies/smc_structure.py`

Add initialization check:
```python
def __init__(self, config):
    # Verify pure SMC mode for Phase 2.1
    if hasattr(config, 'macd_enabled') and config.macd_enabled:
        logger.warning(
            "MACD enabled but running SMC_STRUCTURE - "
            "may cause hybrid behavior"
        )

    if hasattr(config, 'ema_enabled') and config.ema_enabled:
        logger.warning(
            "EMA enabled but running SMC_STRUCTURE - "
            "may cause hybrid behavior"
        )

    # Initialize pure SMC
    self.pure_mode = not (
        getattr(config, 'macd_enabled', False) or
        getattr(config, 'ema_enabled', False)
    )

    logger.info(f"SMC Structure Strategy Mode: {'PURE' if self.pure_mode else 'HYBRID'}")
```

### 3.2 Update Main Config

Edit: `/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/configdata/config.py`

For Phase 2.1 test only, set:
```python
# Disable strategy overlays for pure SMC test
MACD_ENABLED = False
EMA_ENABLED = False
```

---

## Step 4: Run Phase 2.1 Retest (1 hour)

### 4.1 Prepare Test Environment

```bash
# Enter Docker container
cd /home/hr/Projects/TradeSystemV1
docker-compose exec worker bash

# Verify configuration
python -c "
from forex_scanner.configdata.strategies import config_smc_structure as smc
print(f'Entry Timeframe: {smc.SMC_ENTRY_TIMEFRAME}')
print(f'ADX Threshold: {smc.SMC_ADX_THRESHOLD}')
print(f'Volume Multiplier: {smc.SMC_VOLUME_MULTIPLIER}')
print(f'Min Confidence: {smc.SMC_MIN_CONFIDENCE}')
"
```

### 4.2 Execute Backtest

```bash
# Phase 2.1 retest - 30 days, all pairs, 15m timeframe
python /app/forex_scanner/backtest_cli.py \
    --days 30 \
    --strategy SMC_STRUCTURE \
    --timeframe 15m \
    --show-signals \
    --max-signals 200 \
    > /tmp/phase_21_retest.txt 2>&1
```

### 4.3 Monitor Execution

Check for validation messages:
```bash
tail -f /tmp/phase_21_retest.txt | grep -E "CONFIGURATION|VALIDATION|Timeframe|ADX|Volume"
```

Expected output:
```
BACKTEST CONFIGURATION VALIDATION
Strategy: SMC_STRUCTURE
Requested Timeframe: 15m
SMC Structure Strategy Configuration:
  ADX Threshold: 12
  Volume Filter: 0.9x
  Min Confidence: 60%
CONFIGURATION VALIDATION PASSED
```

---

## Step 5: Analyze Results (1-2 hours)

### 5.1 Success Criteria

Phase 2.1 retest will be considered SUCCESSFUL if:

- **Signal Flow**: 8-15 signals/month (240-450/year)
- **Win Rate**: 60-65%
- **Stop Loss Hit Rate**: <40% of losses
- **Sample Size**: Minimum 50 signals for statistical validity
- **Confidence**: All signals ≥60%

### 5.2 Analysis Checklist

Run analysis agent:
```bash
# Provide the retest results to the trading-strategy-analyst agent
# Include comparison to:
# - Baseline (18 signals/month, 55.6% win rate)
# - Phase 1 (0.67 signals/month, 66.7% win rate)
# - Phase 2.1 targets (8-15 signals/month, 60-65% win rate)
```

Key questions to answer:
1. How many signals generated? (compare to 0.13/month from invalid test)
2. What was the actual win rate?
3. How many losses hit stop loss vs trailing stop?
4. What was the confidence distribution?
5. Which pairs performed best/worst?
6. Were there any common loss patterns?

---

## Step 6: Decision Tree

### IF Phase 2.1 Retest SUCCEEDS (8-15 signals/month, 60-65% win rate):
→ **Proceed to production testing**
→ Document filter parameters as Phase 2.1 production baseline
→ Set up forward testing on demo account

### IF Phase 2.1 Retest PARTIAL (signals restored but win rate <60%):
→ **Analyze loss patterns**
→ Identify which filter needs adjustment
→ Plan Phase 2.2 with targeted improvements

### IF Phase 2.1 Retest FAILS (<8 signals/month):
→ **Fundamental strategy incompatibility**
→ Consider Phase 2.2 alternatives:
  - Remove ADX filter entirely
  - Enable BOS/CHoCH re-entry detection
  - Switch to structure strength score
  - Test on 30m timeframe instead of 15m

---

## Timeline

| Task | Duration | Responsible | Status |
|------|----------|-------------|--------|
| Step 1: Config Validator | 2h | Dev Team | PENDING |
| Step 2: Update SMC Config | 30m | Dev Team | PENDING |
| Step 3: Disable Overlays | 15m | Dev Team | PENDING |
| Step 4: Run Retest | 1h | Automated | PENDING |
| Step 5: Analyze Results | 1-2h | Analyst | PENDING |
| Step 6: Decision | 1h | Team | PENDING |
| **TOTAL** | **4-6h** | | |

---

## Success Metrics

| Metric | Current (Invalid) | Target | Threshold |
|--------|-------------------|--------|-----------|
| Signals/Month | 0.13 | 8-15 | ≥8 |
| Win Rate | 50% | 60-65% | ≥60% |
| SL Hit Rate | 100% | <40% | <50% |
| Sample Size | 4 | 50+ | ≥50 |

---

## Risk Mitigation

**Risk 1**: Retest still yields <8 signals/month
- **Mitigation**: Have Phase 2.2 alternative strategies ready
- **Fallback**: Consider 30m timeframe or alternative entry logic

**Risk 2**: Configuration validator breaks existing tests
- **Mitigation**: Make validator optional via flag (--validate-config)
- **Fallback**: Can be disabled for legacy tests

**Risk 3**: Data fetcher changes affect other strategies
- **Mitigation**: Only add tracking properties, don't change behavior
- **Testing**: Verify MACD/EMA strategies still work correctly

---

## Deliverables

1. Configuration validator module
2. Updated backtest CLI with validation
3. Phase 2.1 retest results (50+ signals minimum)
4. Performance analysis report
5. Go/no-go decision for production testing

---

## Next Review

**Date**: After Phase 2.1 retest completion
**Attendees**: Dev team, trading analyst
**Agenda**: Review results, make production testing decision
