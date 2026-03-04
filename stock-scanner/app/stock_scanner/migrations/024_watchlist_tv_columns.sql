-- Migration 024: Add TradingView technical summary columns to watchlist results
-- These columns store the TV indicator consensus data used in trade-ready scoring

ALTER TABLE stock_watchlist_results
ADD COLUMN IF NOT EXISTS tv_overall_score DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS tv_overall_signal VARCHAR(15),
ADD COLUMN IF NOT EXISTS tv_ma_buy INTEGER,
ADD COLUMN IF NOT EXISTS tv_ma_sell INTEGER,
ADD COLUMN IF NOT EXISTS tv_osc_buy INTEGER,
ADD COLUMN IF NOT EXISTS tv_osc_sell INTEGER;
