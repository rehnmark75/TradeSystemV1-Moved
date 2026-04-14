-- Migration: 008_broker_trades_signal_id.sql
-- Description: Persist originating stock_scanner_signals.id on broker_trades
-- Date: 2026-04-14

ALTER TABLE broker_trades
ADD COLUMN IF NOT EXISTS signal_id BIGINT;

CREATE INDEX IF NOT EXISTS idx_broker_trades_signal_id
    ON broker_trades(signal_id);

COMMENT ON COLUMN broker_trades.signal_id IS
'Originating stock_scanner_signals.id resolved during broker sync';
