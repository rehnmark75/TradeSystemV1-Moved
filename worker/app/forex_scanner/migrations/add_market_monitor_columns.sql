-- ============================================================================
-- ADD MARKET MONITOR COLUMNS TO SCANNER_GLOBAL_CONFIG
-- ============================================================================
-- Migration: add_market_monitor_columns.sql
-- Purpose: Add market monitor configuration settings to database
-- Date: 2026-01-04
--
-- Settings migrated from config.py:
--   - Volatility thresholds (LOW, NORMAL, HIGH, EXTREME)
--   - Spread thresholds (TIGHT, NORMAL, WIDE)
--   - Market session definitions
--   - Cache duration settings
--
-- Usage:
--   docker exec postgres psql -U postgres -d strategy_config -f /app/forex_scanner/migrations/add_market_monitor_columns.sql
-- ============================================================================

\c strategy_config;

-- ============================================================================
-- ADD VOLATILITY THRESHOLD COLUMNS
-- ============================================================================

-- Low volatility threshold (score at or below this = LOW volatility)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS low_volatility_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.5;

-- Normal volatility threshold (score at or below this = NORMAL volatility)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS normal_volatility_threshold DOUBLE PRECISION NOT NULL DEFAULT 1.0;

-- High volatility threshold (score at or above this = HIGH volatility)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS high_volatility_threshold DOUBLE PRECISION NOT NULL DEFAULT 2.0;

-- Extreme volatility threshold (score at or above this = EXTREME volatility)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS extreme_volatility_threshold DOUBLE PRECISION NOT NULL DEFAULT 3.0;

-- ============================================================================
-- ADD SPREAD THRESHOLD COLUMNS
-- ============================================================================

-- Tight spread threshold (spread at or below this = TIGHT/excellent conditions)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS tight_spread_threshold DOUBLE PRECISION NOT NULL DEFAULT 2.0;

-- Normal spread threshold (spread at or below this = NORMAL/good conditions)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS normal_spread_threshold DOUBLE PRECISION NOT NULL DEFAULT 3.0;

-- Wide spread threshold (spread at or below this = WIDE/reduced position recommended)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS wide_spread_threshold DOUBLE PRECISION NOT NULL DEFAULT 5.0;

-- ============================================================================
-- ADD MARKET SESSION DEFINITIONS (JSONB)
-- ============================================================================

-- Market sessions with open/close hours in UTC
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS market_sessions JSONB NOT NULL DEFAULT '{
    "SYDNEY": {"open": 21, "close": 6},
    "TOKYO": {"open": 0, "close": 9},
    "LONDON": {"open": 8, "close": 17},
    "NEW_YORK": {"open": 13, "close": 22}
}'::jsonb;

-- ============================================================================
-- ADD CACHE DURATION SETTINGS
-- ============================================================================

-- Market condition cache duration in minutes
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS market_condition_cache_minutes INTEGER NOT NULL DEFAULT 5;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN scanner_global_config.low_volatility_threshold IS 'Volatility score threshold for LOW level classification';
COMMENT ON COLUMN scanner_global_config.normal_volatility_threshold IS 'Volatility score threshold for NORMAL level classification';
COMMENT ON COLUMN scanner_global_config.high_volatility_threshold IS 'Volatility score threshold for HIGH level classification';
COMMENT ON COLUMN scanner_global_config.extreme_volatility_threshold IS 'Volatility score threshold for EXTREME level classification';
COMMENT ON COLUMN scanner_global_config.tight_spread_threshold IS 'Spread threshold in pips for TIGHT/excellent conditions';
COMMENT ON COLUMN scanner_global_config.normal_spread_threshold IS 'Spread threshold in pips for NORMAL/good conditions';
COMMENT ON COLUMN scanner_global_config.wide_spread_threshold IS 'Spread threshold in pips for WIDE/reduced position recommended';
COMMENT ON COLUMN scanner_global_config.market_sessions IS 'JSON object defining market session open/close hours in UTC';
COMMENT ON COLUMN scanner_global_config.market_condition_cache_minutes IS 'Duration in minutes to cache market condition assessments';

-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
DECLARE
    col_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_name = 'scanner_global_config'
    AND column_name IN (
        'low_volatility_threshold',
        'normal_volatility_threshold',
        'high_volatility_threshold',
        'extreme_volatility_threshold',
        'tight_spread_threshold',
        'normal_spread_threshold',
        'wide_spread_threshold',
        'market_sessions',
        'market_condition_cache_minutes'
    );

    IF col_count = 9 THEN
        RAISE NOTICE 'SUCCESS: All 9 market monitor columns added to scanner_global_config';
    ELSE
        RAISE WARNING 'Only % of 9 expected market monitor columns found', col_count;
    END IF;
END
$$;

-- Show current market monitor settings
SELECT
    id,
    low_volatility_threshold,
    normal_volatility_threshold,
    high_volatility_threshold,
    extreme_volatility_threshold,
    tight_spread_threshold,
    normal_spread_threshold,
    wide_spread_threshold,
    market_sessions,
    market_condition_cache_minutes
FROM scanner_global_config
WHERE is_active = TRUE;
