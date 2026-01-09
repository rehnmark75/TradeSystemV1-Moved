-- Migration: Create backtest candles table with pre-computed 5m, 15m, and 4h timeframes
-- Purpose: Fast backtesting without runtime resampling from 1m candles
-- Created: January 2026

-- Create the backtest candles table (same structure as ig_candles)
CREATE TABLE IF NOT EXISTS ig_candles_backtest (
    start_time       TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    epic             VARCHAR NOT NULL,
    timeframe        INTEGER NOT NULL,  -- 5, 15, 240 (minutes)
    open             DOUBLE PRECISION NOT NULL,
    high             DOUBLE PRECISION NOT NULL,
    low              DOUBLE PRECISION NOT NULL,
    close            DOUBLE PRECISION NOT NULL,
    volume           INTEGER NOT NULL,
    ltv              INTEGER,
    resampled_from   INTEGER DEFAULT 1,  -- Source timeframe (1m)
    created_at       TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (start_time, epic, timeframe)
);

-- Create indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_backtest_candles_epic
ON ig_candles_backtest (epic);

CREATE INDEX IF NOT EXISTS idx_backtest_candles_epic_tf_time
ON ig_candles_backtest (epic, timeframe, start_time DESC);

-- Add comment explaining the table
COMMENT ON TABLE ig_candles_backtest IS
'Pre-computed candles for fast backtesting. Contains 5m, 15m, and 4h candles resampled from 1m data.
Live trading uses ig_candles (1m), backtesting uses this table for instant data access.';

COMMENT ON COLUMN ig_candles_backtest.resampled_from IS
'Source timeframe in minutes that this candle was resampled from (typically 1)';
