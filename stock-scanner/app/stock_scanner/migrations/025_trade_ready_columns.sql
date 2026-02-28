-- Migration 025: Add trade-ready filter and structure-based SL/TP columns
-- These replace the fixed 3%/5%/10% SL/TP with market-structure-aware levels

ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS trade_ready BOOLEAN DEFAULT FALSE;
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS trade_ready_score INTEGER;
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS trade_ready_reasons TEXT;
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS structure_stop_loss DECIMAL(18,4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS structure_target_1 DECIMAL(18,4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS structure_target_2 DECIMAL(18,4);
ALTER TABLE stock_watchlist_results ADD COLUMN IF NOT EXISTS structure_rr_ratio DECIMAL(5,2);

-- Index for filtering trade-ready stocks
CREATE INDEX IF NOT EXISTS idx_watchlist_trade_ready
    ON stock_watchlist_results (trade_ready, trade_ready_score DESC NULLS LAST)
    WHERE status = 'active';
