-- ============================================================================
-- Phase 1: Database Parameter Tuning for SMC_SIMPLE Strategy
-- Date: 2026-01-13
-- Status: APPLIED
-- ============================================================================
--
-- PROBLEM: Too few signals generated (~1 signal per 25 hours)
-- Analysis showed 33 swing proximity rejections in 24h, many at borderline distances
--
-- CHANGES APPLIED:
-- ============================================================================

-- 1.1: Confidence Threshold
-- Before: 0.41 | After: 0.38
-- Rationale: Rejected signals showed 70% WR with +582 pips - leaving money on table
-- Expected: +20-25% more signals
UPDATE smc_simple_global_config SET min_confidence_threshold = 0.38 WHERE is_active = TRUE;

-- 1.2: Swing Proximity Filter
-- Before: 12 pips strict | After: 8 pips penalty mode
-- Rationale: 33 rejections in 24h, 21% at 9-12 pips (borderline)
-- Expected: +20-30% more signals (confidence penalty instead of hard rejection)
UPDATE smc_simple_global_config SET
    swing_proximity_min_distance_pips = 8,
    swing_proximity_strict_mode = FALSE
WHERE is_active = TRUE;

-- 1.3: Momentum Staleness Window
-- Before: 8 bars (2h) | After: 12 bars (3h on 15m)
-- Rationale: Allow slightly older swing breaks for momentum entries
-- Expected: +10-15% more momentum entries
UPDATE smc_simple_global_config SET max_momentum_staleness_bars = 12 WHERE is_active = TRUE;

-- 1.4: ATR Extension Filter
-- Before: 0.50 ATR | After: 0.70 ATR
-- Rationale: Allow momentum entries further from swing break
-- Expected: +15-20% more momentum entries
UPDATE smc_simple_global_config SET max_extension_atr = 0.70 WHERE is_active = TRUE;

-- 1.5: Breakout ATR Ratio
-- Before: 0.50 | After: 0.35
-- Rationale: Allow weaker breakouts (still validated by other filters)
-- Expected: +10-15% more signals
UPDATE smc_simple_global_config SET min_breakout_atr_ratio = 0.35 WHERE is_active = TRUE;

-- 1.6: EMA Slope Multiplier
-- Before: 0.5 ATR | After: 0.2 ATR
-- Rationale: Only reject strong counter-slopes, allow mild retests
-- Expected: Fewer false rejections during consolidation
UPDATE smc_simple_global_config SET ema_slope_min_atr_multiplier = 0.2 WHERE is_active = TRUE;

-- 1.7: Pullback Minimum Threshold
-- Before: 0.20 (20%) | After: 0.15 (15%)
-- Rationale: 42 rejections in 24h, 8 at 15-20% (borderline good entries)
-- Expected: +8 more signals/day
UPDATE smc_simple_global_config SET fib_pullback_min = 0.15 WHERE is_active = TRUE;

-- 1.8: Pullback Maximum Threshold
-- Before: 0.70 (70%) | After: 0.80 (80%)
-- Rationale: 13 rejections in 24h, 5 at 70-80% (borderline valid deep pullbacks)
-- Risk: Deeper pullbacks have higher failure rate - MONITOR CLOSELY
-- Expected: +5 more signals/day
UPDATE smc_simple_global_config SET fib_pullback_max = 0.80 WHERE is_active = TRUE;

-- ============================================================================
-- ROLLBACK (if needed):
-- ============================================================================
/*
UPDATE smc_simple_global_config SET
    min_confidence_threshold = 0.41,
    swing_proximity_min_distance_pips = 12,
    swing_proximity_strict_mode = TRUE,
    max_momentum_staleness_bars = 8,
    max_extension_atr = 0.50,
    min_breakout_atr_ratio = 0.50,
    ema_slope_min_atr_multiplier = 0.5,
    fib_pullback_min = 0.20,
    fib_pullback_max = 0.70
WHERE is_active = TRUE;
*/

-- ============================================================================
-- EXPECTED OUTCOMES:
-- ============================================================================
-- | Metric              | Before    | After     |
-- |---------------------|-----------|-----------|
-- | Signals per day     | ~1-2      | ~4-8      |
-- | Signal pass rate    | ~1%       | ~3-5%     |
-- | Swing rejections    | 33/day    | ~10/day   |
-- ============================================================================
