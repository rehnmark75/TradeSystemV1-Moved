-- Migration: Add HTF (4H) Candle Direction Columns
-- Date: 2026-01-12
-- Purpose: Track 4H candle direction at signal time for correlation with trade outcomes
--
-- Run with: docker exec postgres psql -U postgres -d forex -f /app/forex_scanner/migrations/add_htf_candle_direction_columns.sql

BEGIN;

-- ============================================================================
-- 4H Candle Direction Columns for alert_history
-- ============================================================================
-- htf_candle_direction: Direction of the last CLOSED 4H candle at signal time
-- Values: 'BULLISH' (close > open), 'BEARISH' (close < open), 'NEUTRAL' (close == open)
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS htf_candle_direction VARCHAR(10);

-- htf_candle_direction_prev: Direction of the previous 4H candle (before the last closed)
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS htf_candle_direction_prev VARCHAR(10);

-- ============================================================================
-- Create Index for Analysis Queries
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_alert_htf_direction
ON alert_history(htf_candle_direction) WHERE htf_candle_direction IS NOT NULL;

COMMIT;

-- ============================================================================
-- Verification Query
-- ============================================================================
SELECT 'HTF candle direction columns added:' as info;
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'alert_history'
AND column_name IN ('htf_candle_direction', 'htf_candle_direction_prev')
ORDER BY column_name;
