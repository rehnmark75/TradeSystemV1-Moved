-- ============================================================================
-- TIER2_SWING Optimization - Reduce Swing Detection Strictness
-- Date: 2026-01-13
-- Status: APPLIED
-- ============================================================================
--
-- PROBLEM: TIER2_SWING responsible for 41.3% of all rejections (515/1,246)
-- Analysis showed 71.3% of TIER2_SWING rejections were "No recent swing highs/lows"
-- Root cause: swing_strength_bars=3 is too strict (institutional-grade) and
-- ATR thresholds (8-15 pips) miscalibrated for current market (3-6 pips ATR)
--
-- CHANGES APPLIED:
-- ============================================================================

-- 2.1: Swing Strength Bars
-- Before: 3 | After: 2
-- Rationale: 3 bars requires 7-bar (1h 45m) perfect pivot formation
--            2 bars is industry standard, reduces to 5-bar window
-- Expected: 60-65% reduction in "No recent swings" rejections
UPDATE smc_simple_global_config SET swing_strength_bars = 2 WHERE is_active = TRUE;

-- 2.2: Dynamic Swing Lookback ATR Low Threshold
-- Before: 8 pips | After: 5 pips
-- Rationale: Current market ATR is 3-6 pips, old threshold forced minimum
--            lookback constantly (90% of the time)
-- Expected: Better ATR-based scaling for major pairs
UPDATE smc_simple_global_config SET swing_lookback_atr_low = 5 WHERE is_active = TRUE;

-- 2.3: Dynamic Swing Lookback ATR High Threshold
-- Before: 15 pips | After: 10 pips
-- Rationale: Enables dynamic scaling to work in mid-volatility conditions
-- Expected: 10-15% more swings detected in mid-volatility pairs
UPDATE smc_simple_global_config SET swing_lookback_atr_high = 10 WHERE is_active = TRUE;

-- 2.4: Minimum Swing Lookback Window
-- Before: 15 bars | After: 20 bars
-- Rationale: With swing_strength=2 (5-bar pivot), 15 bars leaves only 10
--            eligible for detection. 20 bars gives 15 eligible (50% more)
-- Expected: 15-20% reduction in "No recent swings" rejections
UPDATE smc_simple_global_config SET swing_lookback_min = 20 WHERE is_active = TRUE;

-- ============================================================================
-- ROLLBACK (if needed):
-- ============================================================================
/*
UPDATE smc_simple_global_config SET
    swing_strength_bars = 3,
    swing_lookback_atr_low = 8,
    swing_lookback_atr_high = 15,
    swing_lookback_min = 15
WHERE is_active = TRUE;
*/

-- ============================================================================
-- EXPECTED OUTCOMES:
-- ============================================================================
-- | Metric                        | Before    | After     |
-- |-------------------------------|-----------|-----------|
-- | TIER2_SWING rejection rate    | 41.3%     | 25-30%    |
-- | "No recent swings" rejections | 1,656     | 400-600   |
-- | Signal increase               | baseline  | +25-35%   |
-- | Win rate impact               | baseline  | +/- 2%    |
-- ============================================================================
--
-- PHASE 2 (Future - if Phase 1 succeeds after 7-10 days):
-- - Reduce min_body_percentage from 0.20 to 0.18
-- - Expected: +5-8% additional signals
-- ============================================================================
