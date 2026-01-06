-- Migration: Remove Safety Filter Columns
-- Date: January 2026
-- Reason: EMA200/consensus filters were redundant with SMC Simple strategy's
--         built-in 4H 50 EMA bias check. The strategy handles trend alignment internally.
--
-- NOTE: This migration drops columns that are no longer used. The application code
-- has been updated to not reference these columns. Run this after deploying the
-- updated application code.
--
-- To apply this migration, run:
-- docker exec postgres psql -U postgres -d forex -f /path/to/this/file.sql

-- ============================================================================
-- SAFETY: Check if columns exist before dropping (idempotent migration)
-- ============================================================================

DO $$
BEGIN
    -- Drop safety filter columns from scanner_global_config table
    -- These columns are no longer read by the application

    -- EMA200 related columns
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'enable_critical_safety_filters') THEN
        ALTER TABLE scanner_global_config DROP COLUMN enable_critical_safety_filters;
        RAISE NOTICE 'Dropped column: enable_critical_safety_filters';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'enable_ema200_contradiction_filter') THEN
        ALTER TABLE scanner_global_config DROP COLUMN enable_ema200_contradiction_filter;
        RAISE NOTICE 'Dropped column: enable_ema200_contradiction_filter';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'enable_ema_stack_contradiction_filter') THEN
        ALTER TABLE scanner_global_config DROP COLUMN enable_ema_stack_contradiction_filter;
        RAISE NOTICE 'Dropped column: enable_ema_stack_contradiction_filter';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'ema200_minimum_margin') THEN
        ALTER TABLE scanner_global_config DROP COLUMN ema200_minimum_margin;
        RAISE NOTICE 'Dropped column: ema200_minimum_margin';
    END IF;

    -- Consensus/circuit breaker columns
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'require_indicator_consensus') THEN
        ALTER TABLE scanner_global_config DROP COLUMN require_indicator_consensus;
        RAISE NOTICE 'Dropped column: require_indicator_consensus';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'min_confirming_indicators') THEN
        ALTER TABLE scanner_global_config DROP COLUMN min_confirming_indicators;
        RAISE NOTICE 'Dropped column: min_confirming_indicators';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'enable_emergency_circuit_breaker') THEN
        ALTER TABLE scanner_global_config DROP COLUMN enable_emergency_circuit_breaker;
        RAISE NOTICE 'Dropped column: enable_emergency_circuit_breaker';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'max_contradictions_allowed') THEN
        ALTER TABLE scanner_global_config DROP COLUMN max_contradictions_allowed;
        RAISE NOTICE 'Dropped column: max_contradictions_allowed';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'active_safety_preset') THEN
        ALTER TABLE scanner_global_config DROP COLUMN active_safety_preset;
        RAISE NOTICE 'Dropped column: active_safety_preset';
    END IF;

    -- Large candle filter columns
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'enable_large_candle_filter') THEN
        ALTER TABLE scanner_global_config DROP COLUMN enable_large_candle_filter;
        RAISE NOTICE 'Dropped column: enable_large_candle_filter';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'large_candle_atr_multiplier') THEN
        ALTER TABLE scanner_global_config DROP COLUMN large_candle_atr_multiplier;
        RAISE NOTICE 'Dropped column: large_candle_atr_multiplier';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'consecutive_large_candles_threshold') THEN
        ALTER TABLE scanner_global_config DROP COLUMN consecutive_large_candles_threshold;
        RAISE NOTICE 'Dropped column: consecutive_large_candles_threshold';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'movement_lookback_periods') THEN
        ALTER TABLE scanner_global_config DROP COLUMN movement_lookback_periods;
        RAISE NOTICE 'Dropped column: movement_lookback_periods';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'large_candle_filter_cooldown') THEN
        ALTER TABLE scanner_global_config DROP COLUMN large_candle_filter_cooldown;
        RAISE NOTICE 'Dropped column: large_candle_filter_cooldown';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'excessive_movement_threshold_pips') THEN
        ALTER TABLE scanner_global_config DROP COLUMN excessive_movement_threshold_pips;
        RAISE NOTICE 'Dropped column: excessive_movement_threshold_pips';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'safety_filter_log_level') THEN
        ALTER TABLE scanner_global_config DROP COLUMN safety_filter_log_level;
        RAISE NOTICE 'Dropped column: safety_filter_log_level';
    END IF;

    -- JSONB preset columns
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'safety_filter_presets') THEN
        ALTER TABLE scanner_global_config DROP COLUMN safety_filter_presets;
        RAISE NOTICE 'Dropped column: safety_filter_presets';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'scanner_global_config' AND column_name = 'large_candle_filter_presets') THEN
        ALTER TABLE scanner_global_config DROP COLUMN large_candle_filter_presets;
        RAISE NOTICE 'Dropped column: large_candle_filter_presets';
    END IF;

    RAISE NOTICE '✅ Safety filter columns migration completed successfully';

END $$;

-- ============================================================================
-- VERIFICATION: Show remaining columns (optional, for debugging)
-- ============================================================================

-- Uncomment to verify the columns were dropped:
-- SELECT column_name, data_type
-- FROM information_schema.columns
-- WHERE table_name = 'scanner_global_config'
-- ORDER BY ordinal_position;
