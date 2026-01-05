-- Migration: Add direction-aware pair overrides
-- Date: 2026-01-05
-- Description: Adds direction-specific (BULL/BEAR) override columns to smc_simple_pair_overrides
--              This allows per-pair configuration that differs based on trade direction
--
-- Background: Analysis showed EURUSD BEAR signals have 73.7% win rate on TIER3_PULLBACK rejections
--             while BULL signals only have 14.8%. Direction-specific thresholds allow relaxing
--             filters for profitable directions while keeping them strict for unprofitable ones.

-- Connect to strategy_config database
\c strategy_config

-- ============================================================================
-- Add direction-specific pullback/momentum columns
-- ============================================================================

-- FIB_PULLBACK_MIN: Minimum pullback required before entry (default 23.6%)
-- Setting this lower allows "insufficient pullback" signals to pass
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS fib_pullback_min_bull NUMERIC(5,4) DEFAULT NULL;

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS fib_pullback_min_bear NUMERIC(5,4) DEFAULT NULL;

COMMENT ON COLUMN smc_simple_pair_overrides.fib_pullback_min_bull IS
'Min pullback depth for BULL trades (0.0-1.0). NULL = use global. Lower = more momentum entries allowed.';

COMMENT ON COLUMN smc_simple_pair_overrides.fib_pullback_min_bear IS
'Min pullback depth for BEAR trades (0.0-1.0). NULL = use global. Lower = more momentum entries allowed.';

-- FIB_PULLBACK_MAX: Maximum pullback allowed (default 70%)
-- Setting this higher allows "pullback too deep" signals to pass
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS fib_pullback_max_bull NUMERIC(5,4) DEFAULT NULL;

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS fib_pullback_max_bear NUMERIC(5,4) DEFAULT NULL;

COMMENT ON COLUMN smc_simple_pair_overrides.fib_pullback_max_bull IS
'Max pullback depth for BULL trades (0.0-1.5). NULL = use global. Higher = deeper pullbacks allowed.';

COMMENT ON COLUMN smc_simple_pair_overrides.fib_pullback_max_bear IS
'Max pullback depth for BEAR trades (0.0-1.5). NULL = use global. Higher = deeper pullbacks allowed.';

-- MOMENTUM_MIN_DEPTH: Minimum depth for momentum continuation entries (default -45%)
-- More negative = allows entries further beyond the break point
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS momentum_min_depth_bull NUMERIC(5,4) DEFAULT NULL;

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS momentum_min_depth_bear NUMERIC(5,4) DEFAULT NULL;

COMMENT ON COLUMN smc_simple_pair_overrides.momentum_min_depth_bull IS
'Min momentum depth for BULL trades (-2.0 to 0.0). NULL = use global. More negative = further beyond break allowed.';

COMMENT ON COLUMN smc_simple_pair_overrides.momentum_min_depth_bear IS
'Min momentum depth for BEAR trades (-2.0 to 0.0). NULL = use global. More negative = further beyond break allowed.';

-- MIN_VOLUME_RATIO: Minimum volume ratio required for entry (default 0.50)
-- Lower = allows entries with less volume confirmation
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS min_volume_ratio_bull NUMERIC(5,4) DEFAULT NULL;

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS min_volume_ratio_bear NUMERIC(5,4) DEFAULT NULL;

COMMENT ON COLUMN smc_simple_pair_overrides.min_volume_ratio_bull IS
'Min volume ratio for BULL trades (0.0-2.0). NULL = use pair/global default. Lower = less volume required.';

COMMENT ON COLUMN smc_simple_pair_overrides.min_volume_ratio_bear IS
'Min volume ratio for BEAR trades (0.0-2.0). NULL = use pair/global default. Lower = less volume required.';

-- MIN_CONFIDENCE: Minimum confidence score required
-- Already exists as min_confidence, add direction-specific versions
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS min_confidence_bull NUMERIC(5,4) DEFAULT NULL;

ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS min_confidence_bear NUMERIC(5,4) DEFAULT NULL;

COMMENT ON COLUMN smc_simple_pair_overrides.min_confidence_bull IS
'Min confidence for BULL trades (0.0-1.0). NULL = use pair/global default.';

COMMENT ON COLUMN smc_simple_pair_overrides.min_confidence_bear IS
'Min confidence for BEAR trades (0.0-1.0). NULL = use pair/global default.';

-- ============================================================================
-- Add metadata columns for UI and audit
-- ============================================================================

-- Track which direction overrides are enabled
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS direction_overrides_enabled BOOLEAN DEFAULT FALSE;

COMMENT ON COLUMN smc_simple_pair_overrides.direction_overrides_enabled IS
'When TRUE, direction-specific columns (*_bull/*_bear) are used. When FALSE, only non-directional values apply.';

-- ============================================================================
-- Update audit table to track direction-specific changes
-- ============================================================================

-- The existing audit table uses JSONB for new_values, so it will automatically
-- capture the new columns without schema changes.

-- ============================================================================
-- Example: Set up EURUSD with recommended direction-aware settings
-- Based on rejection analysis showing BEAR has 73.7% WR, BULL has 14.8% WR
-- ============================================================================

-- UPDATE smc_simple_pair_overrides
-- SET
--     direction_overrides_enabled = TRUE,
--     -- BEAR direction: relax filters (73.7% win rate on rejected signals)
--     fib_pullback_min_bear = 0.05,      -- Was 0.236, allow more momentum entries
--     fib_pullback_max_bear = 0.90,      -- Was 0.70, allow deeper pullbacks
--     momentum_min_depth_bear = -1.00,   -- Was -0.45, allow further beyond break
--     min_volume_ratio_bear = 0.10,      -- Was 0.40, allow lower volume
--     -- BULL direction: keep strict (14.8% win rate)
--     fib_pullback_min_bull = NULL,      -- Use global default
--     fib_pullback_max_bull = NULL,      -- Use global default
--     momentum_min_depth_bull = NULL,    -- Use global default
--     min_volume_ratio_bull = NULL,      -- Use global default
--     updated_by = 'migration',
--     change_reason = 'Direction-aware config based on rejection outcome analysis'
-- WHERE epic = 'CS.D.EURUSD.CEEM.IP';

-- ============================================================================
-- Verification query
-- ============================================================================

SELECT
    epic,
    direction_overrides_enabled,
    fib_pullback_min_bull,
    fib_pullback_min_bear,
    momentum_min_depth_bull,
    momentum_min_depth_bear,
    min_volume_ratio_bull,
    min_volume_ratio_bear
FROM smc_simple_pair_overrides
ORDER BY epic;
