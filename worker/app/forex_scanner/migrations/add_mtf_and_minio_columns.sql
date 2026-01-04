-- ============================================================================
-- ADD MTF ENHANCED CONFIDENCE AND MINIO COLUMNS TO SCANNER_GLOBAL_CONFIG
-- ============================================================================
-- Database: strategy_config
-- Purpose: Add mtf_enhanced_min_confidence and minio_enabled columns
--          for database-driven configuration (no config.py fallback)
--
-- Usage:
--   docker exec postgres psql -U postgres -d strategy_config -f /app/forex_scanner/migrations/add_mtf_and_minio_columns.sql
--
-- Related Files:
--   - signal_processor.py uses mtf_enhanced_min_confidence
--   - claude_analyzer.py uses minio_enabled
-- ============================================================================

\c strategy_config;

-- ============================================================================
-- ADD MTF ENHANCED MIN CONFIDENCE COLUMN
-- ============================================================================

-- mtf_enhanced_min_confidence: Lower confidence threshold for MTF-validated signals
-- When multi-timeframe analysis confirms a signal, we can accept lower base confidence
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config'
        AND column_name = 'mtf_enhanced_min_confidence'
    ) THEN
        ALTER TABLE scanner_global_config
        ADD COLUMN mtf_enhanced_min_confidence DECIMAL(4,3) NOT NULL DEFAULT 0.60;

        COMMENT ON COLUMN scanner_global_config.mtf_enhanced_min_confidence IS
            'Lower confidence threshold for MTF-validated signals (allows lower base confidence when MTF confirms)';

        RAISE NOTICE 'Added column: mtf_enhanced_min_confidence';
    ELSE
        RAISE NOTICE 'Column mtf_enhanced_min_confidence already exists - skipping';
    END IF;
END
$$;

-- ============================================================================
-- ADD MINIO ENABLED COLUMN
-- ============================================================================

-- minio_enabled: Toggle MinIO object storage for charts and analysis artifacts
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config'
        AND column_name = 'minio_enabled'
    ) THEN
        ALTER TABLE scanner_global_config
        ADD COLUMN minio_enabled BOOLEAN NOT NULL DEFAULT TRUE;

        COMMENT ON COLUMN scanner_global_config.minio_enabled IS
            'Enable MinIO object storage for charts and analysis artifacts';

        RAISE NOTICE 'Added column: minio_enabled';
    ELSE
        RAISE NOTICE 'Column minio_enabled already exists - skipping';
    END IF;
END
$$;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
DO $$
DECLARE
    mtf_exists BOOLEAN;
    minio_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config'
        AND column_name = 'mtf_enhanced_min_confidence'
    ) INTO mtf_exists;

    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scanner_global_config'
        AND column_name = 'minio_enabled'
    ) INTO minio_exists;

    IF mtf_exists AND minio_exists THEN
        RAISE NOTICE 'SUCCESS: All columns added successfully';
    ELSE
        RAISE WARNING 'Some columns may be missing - check table structure';
    END IF;
END
$$;

-- Show current MTF and Claude-related settings
SELECT
    enable_multi_timeframe_analysis,
    min_confluence_score,
    mtf_enhanced_min_confidence,
    minio_enabled,
    claude_include_chart,
    claude_save_vision_artifacts
FROM scanner_global_config
WHERE is_active = TRUE;
