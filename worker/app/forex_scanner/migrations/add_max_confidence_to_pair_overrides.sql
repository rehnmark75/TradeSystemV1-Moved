-- Migration: Add max_confidence to smc_simple_pair_overrides
-- Date: 2026-01-02
-- Purpose: Allow per-pair override of max confidence threshold (confidence cap)
--
-- The max_confidence_threshold caps the maximum allowed confidence score.
-- Analysis of 85 trades (Dec 2025) showed confidence > 75% had only 42% win rate.
-- This "paradox" means overconfident signals tend to perform worse.
-- Per-pair control allows tuning this for pairs with different characteristics.

-- Add max_confidence column to pair overrides table
ALTER TABLE smc_simple_pair_overrides
ADD COLUMN IF NOT EXISTS max_confidence DECIMAL(4,3);

-- Add comment explaining the column
COMMENT ON COLUMN smc_simple_pair_overrides.max_confidence IS
    'Maximum confidence cap for this pair. Signals above this threshold are rejected. NULL = use global max_confidence_threshold.';

-- Verify the column was added
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'smc_simple_pair_overrides'
  AND column_name IN ('min_confidence', 'max_confidence')
ORDER BY column_name;
