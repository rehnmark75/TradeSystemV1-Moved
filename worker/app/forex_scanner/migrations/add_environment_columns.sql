-- Migration: Add environment column to alert_history and trade_log
-- Purpose: Distinguish live vs demo trades in the shared database
-- Date: 2026-04-11
-- Database: forex (NOT strategy_config)

BEGIN;

-- alert_history: add environment column
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS environment VARCHAR(10) DEFAULT 'demo';
CREATE INDEX IF NOT EXISTS idx_alert_history_environment ON alert_history(environment);

-- trade_log: add environment column
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS environment VARCHAR(10) DEFAULT 'demo';
CREATE INDEX IF NOT EXISTS idx_trade_log_environment ON trade_log(environment);

-- Backfill: all existing data was generated on the demo account
UPDATE alert_history SET environment = 'demo' WHERE environment IS NULL;
UPDATE trade_log SET environment = 'demo' WHERE environment IS NULL;

COMMIT;
