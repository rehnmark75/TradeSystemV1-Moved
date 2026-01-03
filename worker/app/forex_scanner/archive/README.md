# Archived Code - Task Worker Cleanup (January 2026)

This directory contains code that was archived during the task-worker container cleanup.
All code here was disabled and unused as of the archival date.

## Why Archived?

After 6 months of development, the forex scanner accumulated 16 disabled strategies
while only SMC Simple remained active. This cleanup:
- Reduces codebase from ~50,000 to ~10,000 lines
- Simplifies signal_detector.py from 3,605 to ~400 lines
- Makes it easier to add new strategies

## Directory Structure

- `disabled_strategies/` - Strategy files that were disabled (EMA, MACD, Ichimoku, etc.)
- `legacy_configs/` - Configuration files for disabled strategies
- `disabled_helpers/` - Helper modules only used by disabled strategies

## Restoring Code

If you need to restore any of this code:

1. **From archive folder:**
   ```bash
   cp archive/disabled_strategies/ema_strategy.py core/strategies/
   ```

2. **From git history:**
   ```bash
   git checkout pre-refactor-backup -- worker/app/forex_scanner/core/strategies/ema_strategy.py
   ```

## Active Code (NOT archived)

The following remain active:
- `core/strategies/smc_simple_strategy.py` - Only active strategy
- `core/strategies/base_strategy.py` - Base class for all strategies
- `core/strategies/helpers/smc_fair_value_gaps.py` - Used by SMC Simple
- `core/strategies/helpers/smc_order_blocks.py` - Used by SMC Simple
- `core/strategies/helpers/smc_performance_metrics.py` - Used by SMC Simple

## Git Reference

Pre-refactor state preserved at git tag: `pre-refactor-backup`
