-- ============================================================================
-- ADD NEWS FILTER COLUMNS TO SCANNER_GLOBAL_CONFIG
-- ============================================================================
-- Migration: add_news_filter_columns.sql
-- Purpose: Add complete news filtering configuration to database
-- Date: 2026-01-04
--
-- Usage:
--   docker exec postgres psql -U postgres -d strategy_config -f /app/forex_scanner/migrations/add_news_filter_columns.sql
-- ============================================================================

\c strategy_config;

-- ============================================================================
-- ADD NEWS FILTER COLUMNS
-- ============================================================================

-- Economic calendar service URL
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS economic_calendar_url VARCHAR(255) NOT NULL DEFAULT 'http://economic-calendar:8091';

-- Buffer times (minutes before news events to block trading)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS news_high_impact_buffer_minutes INTEGER NOT NULL DEFAULT 30;

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS news_medium_impact_buffer_minutes INTEGER NOT NULL DEFAULT 15;

-- Lookahead window (hours to look ahead for news events)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS news_lookahead_hours INTEGER NOT NULL DEFAULT 4;

-- Block trade controls
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS block_trades_before_high_impact_news BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS block_trades_before_medium_impact_news BOOLEAN NOT NULL DEFAULT FALSE;

-- Critical economic events (JSONB array of keywords)
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS critical_economic_events JSONB NOT NULL DEFAULT '["Non-Farm Employment Change", "NFP", "FOMC", "Federal Funds Rate", "ECB Press Conference", "Interest Rate Decision", "CPI", "Core CPI", "GDP", "Employment", "Unemployment"]'::jsonb;

-- Cache and timeout settings
ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS news_cache_duration_minutes INTEGER NOT NULL DEFAULT 5;

ALTER TABLE scanner_global_config
ADD COLUMN IF NOT EXISTS news_service_timeout_seconds INTEGER NOT NULL DEFAULT 5;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON COLUMN scanner_global_config.economic_calendar_url IS 'URL of the economic calendar service for news data';
COMMENT ON COLUMN scanner_global_config.news_high_impact_buffer_minutes IS 'Minutes before high impact news to block trading';
COMMENT ON COLUMN scanner_global_config.news_medium_impact_buffer_minutes IS 'Minutes before medium impact news to block trading';
COMMENT ON COLUMN scanner_global_config.news_lookahead_hours IS 'Hours to look ahead for upcoming news events';
COMMENT ON COLUMN scanner_global_config.block_trades_before_high_impact_news IS 'Block trades before high impact news events';
COMMENT ON COLUMN scanner_global_config.block_trades_before_medium_impact_news IS 'Block trades before medium impact news events';
COMMENT ON COLUMN scanner_global_config.critical_economic_events IS 'Array of critical event keywords that always trigger high risk';
COMMENT ON COLUMN scanner_global_config.news_cache_duration_minutes IS 'Minutes to cache news data before refreshing';
COMMENT ON COLUMN scanner_global_config.news_service_timeout_seconds IS 'Timeout in seconds for news service requests';

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
        'economic_calendar_url',
        'news_high_impact_buffer_minutes',
        'news_medium_impact_buffer_minutes',
        'news_lookahead_hours',
        'block_trades_before_high_impact_news',
        'block_trades_before_medium_impact_news',
        'critical_economic_events',
        'news_cache_duration_minutes',
        'news_service_timeout_seconds'
    );

    IF col_count = 9 THEN
        RAISE NOTICE 'SUCCESS: All 9 news filter columns added to scanner_global_config';
    ELSE
        RAISE WARNING 'Only % of 9 expected news filter columns found', col_count;
    END IF;
END
$$;

-- Show current news filter settings
SELECT
    id,
    enable_news_filtering,
    reduce_confidence_near_news,
    news_filter_fail_secure,
    economic_calendar_url,
    news_high_impact_buffer_minutes,
    news_medium_impact_buffer_minutes,
    news_lookahead_hours,
    block_trades_before_high_impact_news,
    block_trades_before_medium_impact_news,
    news_cache_duration_minutes,
    news_service_timeout_seconds
FROM scanner_global_config
WHERE is_active = TRUE;
