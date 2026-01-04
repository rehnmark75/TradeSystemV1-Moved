-- ============================================================================
-- ADD INTELLIGENCE MANAGER COLUMNS TO SCANNER_GLOBAL_CONFIG
-- ============================================================================
-- Database: strategy_config
-- Purpose: Add intelligence_preset and intelligence_debug_mode columns
--          for database-driven IntelligenceManager configuration
--
-- Usage:
--   docker exec postgres psql -U postgres -d strategy_config -f /app/forex_scanner/migrations/add_intelligence_manager_columns.sql
--
-- Related Migration: intelligence_manager.py now uses database-only configuration
-- ============================================================================

\c strategy_config;

-- ============================================================================
-- ADD INTELLIGENCE MANAGER COLUMNS
-- ============================================================================

-- intelligence_preset: Controls which intelligence filtering preset to use
-- Options: 'disabled', 'minimal', 'balanced', 'conservative', 'collect_only', 'testing'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config'
        AND column_name = 'intelligence_preset'
    ) THEN
        ALTER TABLE scanner_global_config
        ADD COLUMN intelligence_preset VARCHAR(30) NOT NULL DEFAULT 'minimal';

        COMMENT ON COLUMN scanner_global_config.intelligence_preset IS
            'Intelligence filtering preset: disabled, minimal, balanced, conservative, collect_only, testing';

        RAISE NOTICE 'Added column: intelligence_preset';
    ELSE
        RAISE NOTICE 'Column intelligence_preset already exists - skipping';
    END IF;
END
$$;

-- intelligence_debug_mode: Enable detailed intelligence scoring logs
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config'
        AND column_name = 'intelligence_debug_mode'
    ) THEN
        ALTER TABLE scanner_global_config
        ADD COLUMN intelligence_debug_mode BOOLEAN NOT NULL DEFAULT FALSE;

        COMMENT ON COLUMN scanner_global_config.intelligence_debug_mode IS
            'Enable detailed debug logging for intelligence score calculations';

        RAISE NOTICE 'Added column: intelligence_debug_mode';
    ELSE
        RAISE NOTICE 'Column intelligence_debug_mode already exists - skipping';
    END IF;
END
$$;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
DO $$
DECLARE
    preset_exists BOOLEAN;
    debug_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config'
        AND column_name = 'intelligence_preset'
    ) INTO preset_exists;

    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config'
        AND column_name = 'intelligence_debug_mode'
    ) INTO debug_exists;

    IF preset_exists AND debug_exists THEN
        RAISE NOTICE 'SUCCESS: All intelligence manager columns added';
    ELSE
        RAISE WARNING 'Some columns may be missing - check table structure';
    END IF;
END
$$;

-- Show current intelligence-related settings
SELECT
    intelligence_preset,
    intelligence_debug_mode,
    enable_market_intelligence_capture,
    enable_market_intelligence_filtering,
    market_intelligence_min_confidence,
    enable_multi_timeframe_analysis,
    min_confluence_score
FROM scanner_global_config
WHERE is_active = TRUE;
