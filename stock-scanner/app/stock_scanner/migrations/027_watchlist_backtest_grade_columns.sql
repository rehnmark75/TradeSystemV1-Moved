-- Migration 027: Add backtest grading columns to stock_watchlist_results
-- These columns store computed grades that account for sample size,
-- win rate, profit factor, and avg PnL to provide a visual quality signal.

ALTER TABLE stock_watchlist_results
  ADD COLUMN IF NOT EXISTS bt_ema50_90d_score NUMERIC(5,1),
  ADD COLUMN IF NOT EXISTS bt_ema50_90d_grade VARCHAR(3),
  ADD COLUMN IF NOT EXISTS bt_ema50_90d_confidence VARCHAR(20),
  ADD COLUMN IF NOT EXISTS bt_ema50_90d_supports_signal VARCHAR(20);

COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_score IS 'Composite backtest score 0-100, sample-size-adjusted';
COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_grade IS 'Letter grade: A+, A, B, C, D, F';
COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_confidence IS 'Statistical confidence: none, low, medium, high';
COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_supports_signal IS 'Signal support verdict: supports, neutral, contradicts, insufficient_data';
