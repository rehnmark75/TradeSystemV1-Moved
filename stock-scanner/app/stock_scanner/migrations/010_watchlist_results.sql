-- Migration 010: Add watchlist results table for predefined technical screens
-- Created: 2025-12-28

-- Table to store daily watchlist scan results
CREATE TABLE IF NOT EXISTS stock_watchlist_results (
    id SERIAL PRIMARY KEY,
    watchlist_name VARCHAR(50) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    scan_date DATE NOT NULL,

    -- Price data
    price DECIMAL(10,2),
    volume BIGINT,
    avg_volume BIGINT,

    -- Moving averages
    ema_20 DECIMAL(10,2),
    ema_50 DECIMAL(10,2),
    ema_200 DECIMAL(10,2),

    -- Indicators
    rsi_14 DECIMAL(5,2),
    macd DECIMAL(10,4),
    macd_signal DECIMAL(10,4),
    macd_histogram DECIMAL(10,4),

    -- Gap/Change data
    gap_pct DECIMAL(5,2),
    price_change_1d DECIMAL(5,2),

    -- VWAP (for gap up continuation)
    vwap DECIMAL(10,2),

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),

    -- Ensure unique entries per watchlist/ticker/date
    UNIQUE(watchlist_name, ticker, scan_date)
);

-- Index for efficient queries by date and watchlist
CREATE INDEX IF NOT EXISTS idx_watchlist_results_date
ON stock_watchlist_results(scan_date DESC, watchlist_name);

-- Index for ticker lookups
CREATE INDEX IF NOT EXISTS idx_watchlist_results_ticker
ON stock_watchlist_results(ticker, scan_date DESC);

-- Comment on table
COMMENT ON TABLE stock_watchlist_results IS 'Daily scan results for predefined technical watchlists (EMA crossovers, MACD, Gap Up, RSI)';

-- Add enum-like check for valid watchlist names
ALTER TABLE stock_watchlist_results
ADD CONSTRAINT check_watchlist_name
CHECK (watchlist_name IN (
    'ema_50_crossover',
    'ema_20_crossover',
    'macd_bullish_cross',
    'gap_up_continuation',
    'rsi_oversold_bounce'
));
