-- Migration: Add MAE (Maximum Adverse Excursion) tracking columns to trade_log
-- Date: 2026-01-19
-- Purpose: Track worst drawdown during trades in real-time via Lightstreamer tick data

-- Add MAE tracking columns
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS vsl_mae_pips FLOAT DEFAULT 0.0;
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS vsl_mae_price FLOAT;
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS vsl_mae_timestamp TIMESTAMP;

-- Add comment for documentation
COMMENT ON COLUMN trade_log.vsl_mae_pips IS 'Maximum Adverse Excursion in pips (worst drawdown from entry)';
COMMENT ON COLUMN trade_log.vsl_mae_price IS 'Price at which MAE was recorded';
COMMENT ON COLUMN trade_log.vsl_mae_timestamp IS 'Timestamp when MAE was recorded';
