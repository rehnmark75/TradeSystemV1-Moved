-- Migration 028: Add EMA50 crossover signal validation + stop method columns

ALTER TABLE stock_watchlist_results
  ADD COLUMN IF NOT EXISTS signal_validated BOOLEAN,
  ADD COLUMN IF NOT EXISTS signal_validation_reasons TEXT,
  ADD COLUMN IF NOT EXISTS bt_stop_method VARCHAR(10);

COMMENT ON COLUMN stock_watchlist_results.signal_validated IS 'True if the EMA50 signal passes all visual-quality validation filters (EMA slope, EMA stack, volume, close distance)';
COMMENT ON COLUMN stock_watchlist_results.signal_validation_reasons IS 'Validation result details: "validated" or "failed:<reason1>,<reason2>" for UI display';
COMMENT ON COLUMN stock_watchlist_results.bt_stop_method IS 'Backtest stop method used: atr (ATR-based 1.5x/2.25x) or pct (percentage from suggested levels)';
