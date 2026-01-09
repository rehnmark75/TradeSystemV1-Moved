-- Migration: Add Swing Proximity Validation Configuration
-- Version: 2.15.0
-- Date: 2026-01-09
-- Purpose: Adds swing proximity validation parameters to prevent entries too close to swing levels
--
-- Based on trade log analysis:
--   - 65% of trades (13/20) were counter-trend entries at wrong swing levels
--   - SELL at support: 0% win rate (catastrophic)
--   - BUY at resistance: 25% win rate (poor)
--   - Trades with 15+ pips clearance: 50%+ win rate (good)
--
-- This migration adds columns to control swing proximity validation:
--   - swing_proximity_enabled: Enable/disable the filter
--   - swing_proximity_min_distance_pips: Minimum distance from opposing swing
--   - swing_proximity_strict_mode: Reject (true) vs confidence penalty (false)

-- Add columns to smc_simple_global_config
ALTER TABLE smc_simple_global_config
ADD COLUMN IF NOT EXISTS swing_proximity_enabled BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS swing_proximity_min_distance_pips INTEGER DEFAULT 12,
ADD COLUMN IF NOT EXISTS swing_proximity_strict_mode BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS swing_proximity_resistance_buffer NUMERIC(3,2) DEFAULT 1.0,
ADD COLUMN IF NOT EXISTS swing_proximity_support_buffer NUMERIC(3,2) DEFAULT 1.0,
ADD COLUMN IF NOT EXISTS swing_proximity_lookback_swings INTEGER DEFAULT 5;

-- Add comment describing the columns
COMMENT ON COLUMN smc_simple_global_config.swing_proximity_enabled IS 'v2.15.0: Enable swing proximity validation to prevent entries near swing levels';
COMMENT ON COLUMN smc_simple_global_config.swing_proximity_min_distance_pips IS 'v2.15.0: Minimum distance in pips from opposing swing level (BUY from resistance, SELL from support)';
COMMENT ON COLUMN smc_simple_global_config.swing_proximity_strict_mode IS 'v2.15.0: If true, reject signals; if false, apply confidence penalty';
COMMENT ON COLUMN smc_simple_global_config.swing_proximity_resistance_buffer IS 'v2.15.0: Multiplier for resistance distance requirement';
COMMENT ON COLUMN smc_simple_global_config.swing_proximity_support_buffer IS 'v2.15.0: Multiplier for support distance requirement';
COMMENT ON COLUMN smc_simple_global_config.swing_proximity_lookback_swings IS 'v2.15.0: Number of recent swings to check for proximity';

-- Update the active configuration row with recommended values
UPDATE smc_simple_global_config
SET
    swing_proximity_enabled = TRUE,
    swing_proximity_min_distance_pips = 12,  -- Based on trade analysis: 10-15 pips optimal
    swing_proximity_strict_mode = TRUE,       -- Reject signals (not just penalize)
    swing_proximity_resistance_buffer = 1.0,
    swing_proximity_support_buffer = 1.0,
    swing_proximity_lookback_swings = 5
WHERE is_active = TRUE;

-- Add swing_proximity_validated column to alert_history for tracking
-- This marks signals that passed the new proximity validation
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS swing_proximity_validated BOOLEAN DEFAULT FALSE;

COMMENT ON COLUMN alert_history.swing_proximity_validated IS 'v2.15.0: TRUE if signal passed swing proximity validation (post-fix), FALSE for legacy signals';

-- ============================================================================
-- Per-Pair Overrides for Swing Proximity (v2.15.1)
-- Allows different swing proximity settings per currency pair
-- ============================================================================

-- Add swing proximity columns to smc_simple_pair_overrides
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS swing_proximity_enabled BOOLEAN DEFAULT NULL,
ADD COLUMN IF NOT EXISTS swing_proximity_min_distance_pips INTEGER DEFAULT NULL,
ADD COLUMN IF NOT EXISTS swing_proximity_strict_mode BOOLEAN DEFAULT NULL;

-- Add comments for pair override columns
COMMENT ON COLUMN smc_simple_pair_overrides.swing_proximity_enabled IS 'v2.15.1: Per-pair override for swing proximity validation (NULL = use global)';
COMMENT ON COLUMN smc_simple_pair_overrides.swing_proximity_min_distance_pips IS 'v2.15.1: Per-pair minimum distance in pips from opposing swing (NULL = use global)';
COMMENT ON COLUMN smc_simple_pair_overrides.swing_proximity_strict_mode IS 'v2.15.1: Per-pair strict mode (NULL = use global)';

-- Log the migration
DO $$
BEGIN
    RAISE NOTICE 'Migration completed: Swing Proximity Validation (v2.15.0 + v2.15.1)';
    RAISE NOTICE '  - Added swing_proximity_* columns to smc_simple_global_config';
    RAISE NOTICE '  - Added swing_proximity_validated to alert_history';
    RAISE NOTICE '  - Added swing_proximity_* columns to smc_simple_pair_overrides (per-pair)';
    RAISE NOTICE '  - Default: 12 pips minimum distance from opposing swings';
END $$;
