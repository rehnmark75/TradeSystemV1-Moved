-- Migration: 014_avg_daily_movement.sql
-- Date: 2024-12-30
-- Description: Add average daily movement (5-day) metric to screening metrics

-- Add avg_daily_change_5d to stock_screening_metrics
ALTER TABLE stock_screening_metrics
ADD COLUMN IF NOT EXISTS avg_daily_change_5d DECIMAL(6,2);

-- Add to stock_watchlist for display
ALTER TABLE stock_watchlist
ADD COLUMN IF NOT EXISTS avg_daily_change_5d DECIMAL(6,2);

-- Comments
COMMENT ON COLUMN stock_screening_metrics.avg_daily_change_5d IS
    'Average absolute daily % change over 5 days: mean(|close[i] - close[i-1]| / close[i-1] * 100)';

COMMENT ON COLUMN stock_watchlist.avg_daily_change_5d IS
    'Average absolute daily % change over 5 days (snapshot from screening metrics)';
