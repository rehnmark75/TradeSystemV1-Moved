-- Migration: Add S/R Path Blocking Fields
-- Purpose: Track S/R path-to-target blocking rejections for analytics
-- Date: 2025-12-25

-- Add new columns to smc_simple_rejections table for path blocking analysis
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS sr_blocking_level NUMERIC;
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS sr_blocking_type VARCHAR(20);
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS sr_blocking_distance_pips NUMERIC;
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS sr_path_blocked_pct NUMERIC;
ALTER TABLE smc_simple_rejections ADD COLUMN IF NOT EXISTS target_distance_pips NUMERIC;

-- Add comments for clarity
COMMENT ON COLUMN smc_simple_rejections.sr_blocking_level IS 'Price level of the blocking S/R';
COMMENT ON COLUMN smc_simple_rejections.sr_blocking_type IS 'Type: support or resistance';
COMMENT ON COLUMN smc_simple_rejections.sr_blocking_distance_pips IS 'Distance from entry to blocking S/R in pips';
COMMENT ON COLUMN smc_simple_rejections.sr_path_blocked_pct IS 'Percentage of path to TP that is blocked by S/R';
COMMENT ON COLUMN smc_simple_rejections.target_distance_pips IS 'Total distance from entry to TP in pips';

-- Add index for filtering by rejection stage
CREATE INDEX IF NOT EXISTS idx_smc_rejections_sr_path_blocked
ON smc_simple_rejections(rejection_stage)
WHERE rejection_stage = 'SR_PATH_BLOCKED';

-- Add index for path blocking percentage queries
CREATE INDEX IF NOT EXISTS idx_smc_rejections_path_blocked_pct
ON smc_simple_rejections(sr_path_blocked_pct)
WHERE sr_path_blocked_pct IS NOT NULL;

-- Add validation_details column to alert_history if not exists
-- This stores the full S/R validation context for approved signals
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS sr_path_blocking_details JSON;

COMMENT ON COLUMN alert_history.sr_path_blocking_details IS 'JSON with S/R path blocking analysis for approved signals';

-- Verify migration
DO $$
BEGIN
    RAISE NOTICE 'S/R Path Blocking migration completed successfully';
    RAISE NOTICE 'New columns added to smc_simple_rejections: sr_blocking_level, sr_blocking_type, sr_blocking_distance_pips, sr_path_blocked_pct, target_distance_pips';
    RAISE NOTICE 'New column added to alert_history: sr_path_blocking_details';
END
$$;
