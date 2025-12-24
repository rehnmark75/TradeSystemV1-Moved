-- Migration for early breakeven tracking (v2.8.0)
-- Adds columns to track early breakeven execution before partial close

ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS early_be_executed BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS early_be_time TIMESTAMP;
