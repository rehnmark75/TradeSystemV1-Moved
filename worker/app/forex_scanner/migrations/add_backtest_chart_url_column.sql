-- Migration: Add chart_url column to backtest_executions table
-- Purpose: Store MinIO URL for backtest result charts
-- Date: 2026-01-09

-- Add chart_url column to store MinIO public URL
ALTER TABLE backtest_executions
ADD COLUMN IF NOT EXISTS chart_url VARCHAR(512);

-- Add chart_object_name to store MinIO object name (for deletion if needed)
ALTER TABLE backtest_executions
ADD COLUMN IF NOT EXISTS chart_object_name VARCHAR(256);

-- Create partial index for efficient lookups of executions with charts
CREATE INDEX IF NOT EXISTS idx_backtest_executions_chart_url
ON backtest_executions (chart_url)
WHERE chart_url IS NOT NULL;

-- Add comment
COMMENT ON COLUMN backtest_executions.chart_url IS 'MinIO public URL for backtest result chart';
COMMENT ON COLUMN backtest_executions.chart_object_name IS 'MinIO object name for chart (for cleanup)';
