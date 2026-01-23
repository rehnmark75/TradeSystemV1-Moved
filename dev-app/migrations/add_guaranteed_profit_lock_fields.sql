-- Add Guaranteed Profit Lock tracking fields to trade_log table
-- Created: 2026-01-22
-- Purpose: Track when trades trigger the +10 pip profit protection

-- Guaranteed profit lock fields
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS guaranteed_profit_lock_applied BOOLEAN DEFAULT FALSE;
ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS guaranteed_profit_lock_timestamp TIMESTAMP;

-- Add comments for documentation
COMMENT ON COLUMN trade_log.guaranteed_profit_lock_applied IS 'True when trade reached +10 pips and SL was moved to guaranteed +1 pip minimum';
COMMENT ON COLUMN trade_log.guaranteed_profit_lock_timestamp IS 'Timestamp when guaranteed profit lock was applied';

-- Add index for analytics queries
CREATE INDEX IF NOT EXISTS idx_trade_log_profit_lock_applied
    ON trade_log(guaranteed_profit_lock_applied)
    WHERE guaranteed_profit_lock_applied = TRUE;
