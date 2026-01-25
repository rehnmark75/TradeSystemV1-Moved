-- Migration 020: Optimize watchlist queries with composite indexes
-- Created: 2026-01-24
-- Purpose: Speed up initial load of watchlist tab by optimizing sorting and lateral joins

-- 1. Index for get_watchlist_results main query (filtering by watchlist/status, sorting by volume)
-- This allows retrieving top N stocks by volume without a full sort
CREATE INDEX IF NOT EXISTS idx_watchlist_results_volume_sort
ON stock_watchlist_results(watchlist_name, status, volume DESC)
WHERE status = 'active';

-- 2. Index for the LATERAL JOIN with stock_scanner_signals
-- Used to find the "latest active signal" for each stock efficiently
CREATE INDEX IF NOT EXISTS idx_scanner_signals_ticker_status_score
ON stock_scanner_signals(ticker, status, composite_score DESC);

-- 3. Index for stock_screening_metrics join (fetching max date)
-- Although we typically filter by calculation_date, having an index on ticker+date helps
CREATE INDEX IF NOT EXISTS idx_screening_metrics_ticker_date
ON stock_screening_metrics(ticker, calculation_date DESC);

-- 4. Index for stock_instruments active check
CREATE INDEX IF NOT EXISTS idx_instruments_active
ON stock_instruments(is_active) WHERE is_active = true;
