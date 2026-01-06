-- Migration: Remove ADX Filter Columns
-- Date: January 2026
-- Reason: ADX filter was never used by active strategies. The ADX indicator data
--         is still calculated and available, but the filter configuration that
--         would block signals based on ADX thresholds was never implemented.
--
-- NOTE: This migration drops columns that are no longer used. The application code
-- has been updated to not reference these columns. Run this after deploying the
-- updated application code.
--
-- To apply this migration, run:
-- docker exec <postgres-container> psql -U postgres -d forex -f /path/to/this/file.sql

-- ============================================================================
-- SAFETY: Check if columns exist before dropping (idempotent migration)
-- ============================================================================

DO $$
BEGIN
    -- Drop ADX filter columns from scanner_global_config table
    -- These columns are no longer read by the application

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'adx_filter_enabled') THEN
        ALTER TABLE scanner_global_config DROP COLUMN adx_filter_enabled;
        RAISE NOTICE 'Dropped column: adx_filter_enabled';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'adx_filter_mode') THEN
        ALTER TABLE scanner_global_config DROP COLUMN adx_filter_mode;
        RAISE NOTICE 'Dropped column: adx_filter_mode';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'adx_period') THEN
        ALTER TABLE scanner_global_config DROP COLUMN adx_period;
        RAISE NOTICE 'Dropped column: adx_period';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'adx_grace_period_bars') THEN
        ALTER TABLE scanner_global_config DROP COLUMN adx_grace_period_bars;
        RAISE NOTICE 'Dropped column: adx_grace_period_bars';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'adx_thresholds') THEN
        ALTER TABLE scanner_global_config DROP COLUMN adx_thresholds;
        RAISE NOTICE 'Dropped column: adx_thresholds';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'adx_pair_multipliers') THEN
        ALTER TABLE scanner_global_config DROP COLUMN adx_pair_multipliers;
        RAISE NOTICE 'Dropped column: adx_pair_multipliers';
    END IF;

    RAISE NOTICE 'ADX filter columns migration completed successfully';

END $$;

-- ============================================================================
-- VERIFICATION: Show remaining columns (optional, for debugging)
-- ============================================================================

-- Uncomment to verify the columns were dropped:
-- SELECT column_name, data_type
-- FROM information_schema.columns
-- WHERE table_name = 'scanner_global_config'
-- AND column_name LIKE 'adx%'
-- ORDER BY column_name;
