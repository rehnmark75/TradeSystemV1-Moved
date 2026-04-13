-- Migration 029: Add full-setup backtest columns (EMA50 + RS + DAQ filters, 180d lookback)
-- These replace the narrow bt_ema50_90d_* columns over time.
-- Old columns are kept for one migration cycle and can be dropped in 030+.

ALTER TABLE stock_watchlist_results
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_signals      integer,
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_wins         integer,
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_losses       integer,
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_win_rate     numeric(6,2),
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_avg_pnl      numeric(10,4),
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_total_pnl    numeric(10,4),
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_profit_factor numeric(8,4),
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_avg_hold_days numeric(6,2),
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_start_date   date,
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_end_date     date,
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_last_run     timestamp,
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_score        numeric(5,1),
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_grade        varchar(2),
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_confidence   varchar(10),
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_supports_signal varchar(20),
  -- How many historical setups passed the RS+DAQ quality filters
  ADD COLUMN IF NOT EXISTS bt_fullsetup_180d_filtered_count integer,
  -- Minimum RS/DAQ thresholds used in this backtest run
  ADD COLUMN IF NOT EXISTS bt_fullsetup_min_rs            integer,
  ADD COLUMN IF NOT EXISTS bt_fullsetup_min_daq           integer;

COMMENT ON COLUMN stock_watchlist_results.bt_fullsetup_180d_signals IS
  'Number of historical EMA50 crossovers that passed RS + DAQ quality filters in 180d window';
COMMENT ON COLUMN stock_watchlist_results.bt_fullsetup_180d_supports_signal IS
  'supports | neutral | contradicts | insufficient_data — verdict from full-setup backtest';
COMMENT ON COLUMN stock_watchlist_results.bt_fullsetup_180d_grade IS
  'Letter grade A+/A/B/C/D/F — sample-size-adjusted composite score';
