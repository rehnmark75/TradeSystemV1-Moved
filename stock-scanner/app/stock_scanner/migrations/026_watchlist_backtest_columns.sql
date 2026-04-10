-- Add backtest summary columns for EMA50 crossover watchlist

ALTER TABLE stock_watchlist_results
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_signals INTEGER,
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_wins INTEGER,
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_losses INTEGER,
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_win_rate DECIMAL(5,2),
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_avg_pnl DECIMAL(10,4),
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_total_pnl DECIMAL(10,4),
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_profit_factor DECIMAL(10,4),
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_avg_hold_days DECIMAL(6,2),
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_start_date DATE,
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_end_date DATE,
    ADD COLUMN IF NOT EXISTS bt_ema50_90d_last_run TIMESTAMPTZ;

COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_signals IS 'EMA50 crossover backtest: total signals over last 90 days';
COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_win_rate IS 'EMA50 crossover backtest: win rate (%) over last 90 days';
COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_avg_pnl IS 'EMA50 crossover backtest: average PnL % over last 90 days';
COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_total_pnl IS 'EMA50 crossover backtest: total PnL % over last 90 days';
COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_profit_factor IS 'EMA50 crossover backtest: profit factor over last 90 days';
COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_avg_hold_days IS 'EMA50 crossover backtest: average holding days over last 90 days';
COMMENT ON COLUMN stock_watchlist_results.bt_ema50_90d_last_run IS 'Last run timestamp for EMA50 crossover backtest';
