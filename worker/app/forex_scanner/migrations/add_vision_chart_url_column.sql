-- Migration: Add vision_chart_url column to alert_history table
-- Purpose: Store MinIO URL for Claude vision analysis charts
-- Date: 2025-01-02

-- Add vision_chart_url column to store MinIO object URLs
ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS vision_chart_url VARCHAR(512);

-- Create partial index for efficient lookups (only index non-null values)
CREATE INDEX IF NOT EXISTS idx_alert_history_vision_chart_url
ON alert_history(vision_chart_url)
WHERE vision_chart_url IS NOT NULL;

-- Add column comment for documentation
COMMENT ON COLUMN alert_history.vision_chart_url IS 'MinIO URL for Claude vision analysis chart PNG (30-day retention)';

-- Verify the column was added
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'alert_history'
        AND column_name = 'vision_chart_url'
    ) THEN
        RAISE NOTICE 'Column vision_chart_url successfully added to alert_history table';
    ELSE
        RAISE EXCEPTION 'Failed to add vision_chart_url column';
    END IF;
END $$;
