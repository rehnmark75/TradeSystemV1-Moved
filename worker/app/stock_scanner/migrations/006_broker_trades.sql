-- Migration: 006_broker_trades.sql
-- Description: Create tables for storing broker trade data from RoboMarkets
-- Date: 2024-12-25

-- Table for storing open and closed positions/deals from broker
CREATE TABLE IF NOT EXISTS broker_trades (
    id SERIAL PRIMARY KEY,
    deal_id VARCHAR(50) UNIQUE NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'long' or 'short'
    quantity DECIMAL(18, 8) NOT NULL,
    open_price DECIMAL(18, 8) NOT NULL,
    close_price DECIMAL(18, 8),
    current_price DECIMAL(18, 8),
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8),
    profit DECIMAL(18, 4),
    profit_pct DECIMAL(10, 4),
    status VARCHAR(20) NOT NULL DEFAULT 'open',  -- 'open', 'closed'
    open_time TIMESTAMP WITH TIME ZONE NOT NULL,
    close_time TIMESTAMP WITH TIME ZONE,
    duration_hours DECIMAL(10, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_broker_trades_ticker ON broker_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_broker_trades_status ON broker_trades(status);
CREATE INDEX IF NOT EXISTS idx_broker_trades_open_time ON broker_trades(open_time);
CREATE INDEX IF NOT EXISTS idx_broker_trades_close_time ON broker_trades(close_time);
CREATE INDEX IF NOT EXISTS idx_broker_trades_side ON broker_trades(side);

-- Table for daily performance snapshots
CREATE TABLE IF NOT EXISTS broker_daily_stats (
    id SERIAL PRIMARY KEY,
    stat_date DATE UNIQUE NOT NULL,
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    total_profit DECIMAL(18, 4) DEFAULT 0,
    total_loss DECIMAL(18, 4) DEFAULT 0,
    net_profit DECIMAL(18, 4) DEFAULT 0,
    win_rate DECIMAL(6, 2) DEFAULT 0,
    profit_factor DECIMAL(10, 4) DEFAULT 0,
    avg_win DECIMAL(18, 4) DEFAULT 0,
    avg_loss DECIMAL(18, 4) DEFAULT 0,
    largest_win DECIMAL(18, 4) DEFAULT 0,
    largest_loss DECIMAL(18, 4) DEFAULT 0,
    open_positions INT DEFAULT 0,
    open_unrealized_pnl DECIMAL(18, 4) DEFAULT 0,
    long_trades INT DEFAULT 0,
    short_trades INT DEFAULT 0,
    long_profit DECIMAL(18, 4) DEFAULT 0,
    short_profit DECIMAL(18, 4) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_broker_daily_stats_date ON broker_daily_stats(stat_date);

-- Table for sync metadata
CREATE TABLE IF NOT EXISTS broker_sync_log (
    id SERIAL PRIMARY KEY,
    sync_type VARCHAR(50) NOT NULL,  -- 'positions', 'history', 'full'
    records_fetched INT DEFAULT 0,
    records_inserted INT DEFAULT 0,
    records_updated INT DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'running',  -- 'running', 'completed', 'failed'
    error_message TEXT
);
