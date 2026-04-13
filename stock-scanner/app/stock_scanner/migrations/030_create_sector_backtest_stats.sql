-- Migration 030: Sector-wide full-setup backtest aggregates
-- When per-ticker sample is too small, the UI can show sector-wide stats as fallback.

CREATE TABLE IF NOT EXISTS stock_backtest_sector_stats (
    id              serial PRIMARY KEY,
    sector          varchar(100)  NOT NULL,
    watchlist_name  varchar(50)   NOT NULL DEFAULT 'ema_50_crossover',
    stat_date       date          NOT NULL DEFAULT CURRENT_DATE,
    lookback_days   integer       NOT NULL DEFAULT 180,
    min_rs          integer       NOT NULL DEFAULT 60,
    min_daq         integer       NOT NULL DEFAULT 50,
    -- Aggregated across all tickers in sector that had qualifying setups
    total_tickers   integer,
    total_signals   integer,
    total_wins      integer,
    total_losses    integer,
    win_rate        numeric(6,2),
    avg_pnl         numeric(10,4),
    profit_factor   numeric(8,4),
    avg_hold_days   numeric(6,2),
    created_at      timestamp     NOT NULL DEFAULT NOW(),
    UNIQUE (sector, watchlist_name, stat_date, lookback_days, min_rs, min_daq)
);

CREATE INDEX IF NOT EXISTS idx_sector_bt_stats_sector
    ON stock_backtest_sector_stats (sector, watchlist_name, stat_date DESC);
