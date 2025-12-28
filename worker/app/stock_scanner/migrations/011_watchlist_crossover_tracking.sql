-- Migration 011: Add crossover tracking to watchlist results
-- Created: 2025-12-28
-- Purpose: Track when crossovers first occurred and enable expiry of stale entries

-- Add crossover_date column (date when crossover first happened)
ALTER TABLE stock_watchlist_results
ADD COLUMN IF NOT EXISTS crossover_date DATE;

-- Add status column (active/expired)
ALTER TABLE stock_watchlist_results
ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active';

-- Backfill existing entries: set crossover_date = scan_date for existing data
UPDATE stock_watchlist_results
SET crossover_date = scan_date
WHERE crossover_date IS NULL;

-- Drop old unique constraint (watchlist_name, ticker, scan_date)
ALTER TABLE stock_watchlist_results
DROP CONSTRAINT IF EXISTS stock_watchlist_results_watchlist_name_ticker_scan_date_key;

-- Create partial unique index for active entries only
-- This allows only ONE active entry per (watchlist_name, ticker) at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_active_entry
ON stock_watchlist_results(watchlist_name, ticker) WHERE status = 'active';

-- Index for querying active entries efficiently
CREATE INDEX IF NOT EXISTS idx_watchlist_status
ON stock_watchlist_results(watchlist_name, status) WHERE status = 'active';

-- Index for cleanup queries (finding old entries)
CREATE INDEX IF NOT EXISTS idx_watchlist_crossover_date
ON stock_watchlist_results(crossover_date);

-- Add constraint for valid status values
ALTER TABLE stock_watchlist_results
ADD CONSTRAINT IF NOT EXISTS check_status
CHECK (status IN ('active', 'expired'));

-- Comment on new columns
COMMENT ON COLUMN stock_watchlist_results.crossover_date IS 'Date when the crossover/event first occurred';
COMMENT ON COLUMN stock_watchlist_results.status IS 'active = currently valid, expired = condition no longer met';
