-- Migration: Add unique constraint to prevent duplicate rejections
-- Date: 2026-01-02
-- Purpose: Prevent the same rejection from being logged multiple times per scan cycle
--
-- The combination of (scan_timestamp, epic, rejection_stage, attempted_direction)
-- should be unique - we only want to record one rejection per candle/pair/stage/direction

-- Step 1: Remove existing duplicates (keep only the first occurrence)
-- This CTE identifies duplicates and deletes all but the earliest created_at
WITH duplicates AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY scan_timestamp, epic, rejection_stage, attempted_direction
               ORDER BY created_at ASC
           ) as rn
    FROM smc_simple_rejections
)
DELETE FROM smc_simple_rejections
WHERE id IN (
    SELECT id FROM duplicates WHERE rn > 1
);

-- Step 2: Add unique constraint
-- Using a unique index instead of constraint for better control and performance
CREATE UNIQUE INDEX IF NOT EXISTS idx_smc_rej_unique_candle_rejection
ON smc_simple_rejections (scan_timestamp, epic, rejection_stage, attempted_direction)
WHERE attempted_direction IS NOT NULL;

-- Also handle cases where attempted_direction might be NULL
CREATE UNIQUE INDEX IF NOT EXISTS idx_smc_rej_unique_candle_rejection_null_dir
ON smc_simple_rejections (scan_timestamp, epic, rejection_stage)
WHERE attempted_direction IS NULL;

-- Verify: Show how many records remain after dedup
SELECT
    'Records after deduplication' as status,
    COUNT(*) as total_records,
    COUNT(DISTINCT (scan_timestamp, epic, rejection_stage, attempted_direction)) as unique_combinations
FROM smc_simple_rejections;
