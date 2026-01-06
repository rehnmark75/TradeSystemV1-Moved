-- Migration: Add Session Hours and Order Executor Thresholds
-- Date: January 2026
-- Reason: Make hardcoded session hours and order executor thresholds database-configurable
--
-- These values were previously hardcoded in:
-- - smc_simple_strategy.py (session hours: 7, 12, 16, 21 UTC)
-- - order_executor.py (confidence thresholds: 0.7, 0.8; SL/TP sanity: 100/200 pips)
--
-- To apply this migration, run:
-- docker exec <postgres-container> psql -U postgres -d strategy_config -f /path/to/this/file.sql

-- ============================================================================
-- SESSION HOURS CONFIGURATION
-- ============================================================================

DO $$
BEGIN
    -- Asian session start hour (UTC) - trading blocked during Asian session by default
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'session_asian_start_hour') THEN
        ALTER TABLE scanner_global_config ADD COLUMN session_asian_start_hour INTEGER DEFAULT 21;
        RAISE NOTICE 'Added column: session_asian_start_hour (default: 21)';
    END IF;

    -- Asian session end hour (UTC)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'session_asian_end_hour') THEN
        ALTER TABLE scanner_global_config ADD COLUMN session_asian_end_hour INTEGER DEFAULT 7;
        RAISE NOTICE 'Added column: session_asian_end_hour (default: 7)';
    END IF;

    -- London session start hour (UTC)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'session_london_start_hour') THEN
        ALTER TABLE scanner_global_config ADD COLUMN session_london_start_hour INTEGER DEFAULT 7;
        RAISE NOTICE 'Added column: session_london_start_hour (default: 7)';
    END IF;

    -- London session end hour (UTC)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'session_london_end_hour') THEN
        ALTER TABLE scanner_global_config ADD COLUMN session_london_end_hour INTEGER DEFAULT 16;
        RAISE NOTICE 'Added column: session_london_end_hour (default: 16)';
    END IF;

    -- New York session start hour (UTC)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'session_newyork_start_hour') THEN
        ALTER TABLE scanner_global_config ADD COLUMN session_newyork_start_hour INTEGER DEFAULT 12;
        RAISE NOTICE 'Added column: session_newyork_start_hour (default: 12)';
    END IF;

    -- New York session end hour (UTC)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'session_newyork_end_hour') THEN
        ALTER TABLE scanner_global_config ADD COLUMN session_newyork_end_hour INTEGER DEFAULT 21;
        RAISE NOTICE 'Added column: session_newyork_end_hour (default: 21)';
    END IF;

    -- London/NY overlap start hour (UTC) - best trading window
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'session_overlap_start_hour') THEN
        ALTER TABLE scanner_global_config ADD COLUMN session_overlap_start_hour INTEGER DEFAULT 12;
        RAISE NOTICE 'Added column: session_overlap_start_hour (default: 12)';
    END IF;

    -- London/NY overlap end hour (UTC)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'session_overlap_end_hour') THEN
        ALTER TABLE scanner_global_config ADD COLUMN session_overlap_end_hour INTEGER DEFAULT 16;
        RAISE NOTICE 'Added column: session_overlap_end_hour (default: 16)';
    END IF;

    -- Block trading during Asian session
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'block_asian_session') THEN
        ALTER TABLE scanner_global_config ADD COLUMN block_asian_session BOOLEAN DEFAULT TRUE;
        RAISE NOTICE 'Added column: block_asian_session (default: TRUE)';
    END IF;

    RAISE NOTICE 'Session hours columns migration completed';
END $$;

-- ============================================================================
-- ORDER EXECUTOR THRESHOLDS
-- ============================================================================

DO $$
BEGIN
    -- High confidence threshold for position sizing (was hardcoded 0.8)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'executor_high_confidence_threshold') THEN
        ALTER TABLE scanner_global_config ADD COLUMN executor_high_confidence_threshold NUMERIC(4,2) DEFAULT 0.80;
        RAISE NOTICE 'Added column: executor_high_confidence_threshold (default: 0.80)';
    END IF;

    -- Medium confidence threshold for position sizing (was hardcoded 0.7)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'executor_medium_confidence_threshold') THEN
        ALTER TABLE scanner_global_config ADD COLUMN executor_medium_confidence_threshold NUMERIC(4,2) DEFAULT 0.70;
        RAISE NOTICE 'Added column: executor_medium_confidence_threshold (default: 0.70)';
    END IF;

    -- Maximum reasonable stop loss in pips (sanity check, was hardcoded 100)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'executor_max_stop_loss_pips') THEN
        ALTER TABLE scanner_global_config ADD COLUMN executor_max_stop_loss_pips INTEGER DEFAULT 100;
        RAISE NOTICE 'Added column: executor_max_stop_loss_pips (default: 100)';
    END IF;

    -- Maximum reasonable take profit in pips (sanity check, was hardcoded 200)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'executor_max_take_profit_pips') THEN
        ALTER TABLE scanner_global_config ADD COLUMN executor_max_take_profit_pips INTEGER DEFAULT 200;
        RAISE NOTICE 'Added column: executor_max_take_profit_pips (default: 200)';
    END IF;

    -- Stop distance multiplier for high confidence trades (was hardcoded 0.8)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'executor_high_conf_stop_multiplier') THEN
        ALTER TABLE scanner_global_config ADD COLUMN executor_high_conf_stop_multiplier NUMERIC(4,2) DEFAULT 0.80;
        RAISE NOTICE 'Added column: executor_high_conf_stop_multiplier (default: 0.80)';
    END IF;

    -- Stop distance multiplier for low confidence trades (was hardcoded 1.2)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'scanner_global_config' AND column_name = 'executor_low_conf_stop_multiplier') THEN
        ALTER TABLE scanner_global_config ADD COLUMN executor_low_conf_stop_multiplier NUMERIC(4,2) DEFAULT 1.20;
        RAISE NOTICE 'Added column: executor_low_conf_stop_multiplier (default: 1.20)';
    END IF;

    RAISE NOTICE 'Order executor thresholds columns migration completed';
END $$;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Verify columns were added
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'scanner_global_config'
  AND column_name IN (
    'session_asian_start_hour', 'session_asian_end_hour',
    'session_london_start_hour', 'session_london_end_hour',
    'session_newyork_start_hour', 'session_newyork_end_hour',
    'session_overlap_start_hour', 'session_overlap_end_hour',
    'block_asian_session',
    'executor_high_confidence_threshold', 'executor_medium_confidence_threshold',
    'executor_max_stop_loss_pips', 'executor_max_take_profit_pips',
    'executor_high_conf_stop_multiplier', 'executor_low_conf_stop_multiplier'
  )
ORDER BY column_name;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Migration completed: add_session_hours_and_executor_thresholds.sql';
    RAISE NOTICE 'Session hours and order executor thresholds are now database-configurable';
END $$;
