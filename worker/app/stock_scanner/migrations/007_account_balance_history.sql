-- Migration: 007_account_balance_history.sql
-- Description: Create table for tracking account balance history over time
-- Date: 2024-12-25

-- Table for storing account balance snapshots
CREATE TABLE IF NOT EXISTS broker_account_balance (
    id SERIAL PRIMARY KEY,
    total_value DECIMAL(18, 4) NOT NULL,      -- my_portfolio - total account value
    invested DECIMAL(18, 4) NOT NULL,          -- investments - value in positions
    available DECIMAL(18, 4) NOT NULL,         -- available_to_invest - free cash
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for efficient time-based queries
CREATE INDEX IF NOT EXISTS idx_broker_account_balance_recorded_at ON broker_account_balance(recorded_at);

-- Index for getting latest balance quickly
CREATE INDEX IF NOT EXISTS idx_broker_account_balance_id_desc ON broker_account_balance(id DESC);
